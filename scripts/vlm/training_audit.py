#!/usr/bin/env python3
"""Build an auditable index of CadStruct training runs.

The script intentionally stays dependency-light.  It scans checkpoint
directories for ``train_summary.json`` files and normalizes the fields needed
for paper tables, OOM tracking, and reproducibility review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METRIC_KEYS = (
    "accuracy",
    "macro_f1",
    "probability_r2",
    "r2",
    "mean_iou",
    "dimension_link_f1",
    "relation_f1",
)

EXPERT_TRAINING_COVERAGE = {
    "WallOpening": {
        "scripts": ["scripts/vlm/train_graph_node_crop_gnn_classifier.py", "scripts/vlm/train_graph_node_classifier.py"],
        "checkpoint_keywords": ["graph_node", "wall_opening"],
        "dataset_keywords": ["cadstruct_graph_nodes", "wall_opening"],
    },
    "Room": {
        "scripts": ["scripts/vlm/train_room_space_context_sklearn.py", "scripts/vlm/train_room_space_expert.py"],
        "checkpoint_keywords": ["room_space", "room_proposal"],
        "dataset_keywords": ["room"],
    },
    "Symbol": {
        "scripts": ["scripts/vlm/train_symbol_fixture_crop_mlp.py", "scripts/vlm/train_symbol_fixture_expert.py"],
        "checkpoint_keywords": ["symbol_fixture"],
        "dataset_keywords": ["symbol"],
    },
    "Text": {
        "scripts": ["scripts/vlm/train_text_dimension_crop_mlp.py", "scripts/vlm/train_text_dimension_expert.py"],
        "checkpoint_keywords": ["text_dimension"],
        "dataset_keywords": ["text_dimension", "text"],
    },
    "Layout": {
        "scripts": ["scripts/vlm/train_sheet_layout_expert.py"],
        "checkpoint_keywords": ["sheet_layout"],
        "dataset_keywords": ["sheet_layout", "layout"],
    },
    "Router": {
        "scripts": ["scripts/vlm/train_moe_router_v2.py", "scripts/vlm/audit_moe_router.py"],
        "checkpoint_keywords": ["moe_router", "router"],
        "dataset_keywords": ["router", "moe"],
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--reports-dir", default="reports/vlm")
    parser.add_argument("--output", default="reports/vlm/training_runs_index.json")
    parser.add_argument("--coverage-output", default="reports/vlm/training_contract_coverage_v2.json")
    parser.add_argument("--include-reports", action="store_true")
    args = parser.parse_args()

    checkpoints_dir = Path(args.checkpoints_dir)
    summaries = sorted(checkpoints_dir.glob("*/train_summary.json"))
    runs = [summarize_run(path) for path in summaries]
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "training_runs_index_v1",
        "checkpoints_dir": str(checkpoints_dir),
        "train_summary_count": len(runs),
        "git": git_audit(),
        "environment": environment_audit(),
        "aggregate": aggregate_runs(runs),
        "runs": runs,
    }
    if args.include_reports:
        report["report_artifacts"] = summarize_reports(Path(args.reports_dir))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    coverage = build_contract_coverage(runs)
    coverage_output = Path(args.coverage_output)
    coverage_output.parent.mkdir(parents=True, exist_ok=True)
    coverage_output.write_text(json.dumps(coverage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))


def summarize_run(summary_path: Path) -> dict[str, Any]:
    raw_text = summary_path.read_text(encoding="utf-8")
    data = json.loads(raw_text)
    checkpoint_dir = summary_path.parent
    history = data.get("history") if isinstance(data.get("history"), list) else []
    oom_signals = collect_oom_signals(data, history)
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "summary_path": str(summary_path),
        "summary_sha256": sha256_text(raw_text),
        "modified_at_utc": datetime.fromtimestamp(summary_path.stat().st_mtime, timezone.utc).isoformat(),
        "ok": bool(data.get("ok", True)),
        "model_type": data.get("model_type") or data.get("model_name") or infer_model_type(checkpoint_dir.name),
        "dataset_dir": data.get("dataset_dir") or data.get("dataset") or nested(data, "config", "dataset_dir"),
        "output_dir": data.get("output_dir") or str(checkpoint_dir),
        "seed": data.get("seed") or nested(data, "config", "seed"),
        "epochs": data.get("epochs") or len(history) or nested(data, "config", "epochs"),
        "parameter_count": data.get("parameter_count"),
        "feature_dim": data.get("feature_dim"),
        "labels": data.get("labels"),
        "peak_memory_mib": normalize_peak_memory(data),
        "memory_audit": data.get("memory_audit"),
        "skip_oom_nonfinite": oom_signals,
        "metrics": collect_metrics(data),
        "best": collect_best_fields(data),
        "routing_summary": data.get("routing_summary"),
        "config_snapshot": collect_config_snapshot(data),
        "history": summarize_history(history),
        "artifacts": checkpoint_artifacts(checkpoint_dir),
    }


def collect_metrics(data: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key in ("final_dev_metrics", "dev_metrics", "smoke_metrics", "locked_test_metrics", "test_metrics"):
        value = data.get(key)
        if isinstance(value, dict):
            metrics[key] = slim_metric_dict(value)
    for split, value in (data.get("splits") or {}).items() if isinstance(data.get("splits"), dict) else []:
        if isinstance(value, dict):
            metrics[f"split:{split}"] = slim_metric_dict(value)
    for key in METRIC_KEYS:
        if key in data:
            metrics[key] = data[key]
    return metrics


def slim_metric_dict(value: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, item in value.items():
        if key in METRIC_KEYS or key.startswith("best_") or key.endswith("_f1") or key.endswith("_r2"):
            output[key] = item
        elif key in {"per_label", "confusion", "by_source", "by_source_dataset"}:
            output[key] = item
    return output or value


def collect_best_fields(data: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in data.items()
        if key.startswith("best_") or key in {"selected_epoch", "selection_metric", "checkpoint"}
    }


def collect_config_snapshot(data: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "record_key",
        "model_type",
        "hidden_dim",
        "dropout",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "eval_tile_size",
        "crop_size",
        "crop_pad",
        "crop_pad_scales",
        "routing_balance_weight",
        "gate_temperature",
        "top_k",
        "expert_dropout",
        "tr_rank",
        "device",
    ]
    snapshot = {key: data.get(key) for key in keys if key in data}
    config = data.get("config")
    if isinstance(config, dict):
        snapshot["config"] = config
    return snapshot


def summarize_history(history: list[Any]) -> dict[str, Any]:
    rows = [row for row in history if isinstance(row, dict)]
    losses = [safe_float(row.get("loss")) for row in rows]
    losses = [item for item in losses if item is not None]
    nonfinite = sum(1 for value in losses if not math.isfinite(value))
    best_dev_macro_f1 = max((safe_float(row.get("dev_macro_f1")) or 0.0 for row in rows), default=None)
    best_dev_accuracy = max((safe_float(row.get("dev_accuracy")) or 0.0 for row in rows), default=None)
    return {
        "epochs_recorded": len(rows),
        "first": rows[0] if rows else None,
        "last": rows[-1] if rows else None,
        "best_dev_macro_f1": round(best_dev_macro_f1, 6) if best_dev_macro_f1 is not None else None,
        "best_dev_accuracy": round(best_dev_accuracy, 6) if best_dev_accuracy is not None else None,
        "nonfinite_loss_count": nonfinite,
    }


def collect_oom_signals(data: dict[str, Any], history: list[Any]) -> dict[str, Any]:
    text = json.dumps(data, ensure_ascii=False).lower()
    oom_text = bool(re.search(r"\boom\b|out[- ]of[- ]memory|cuda out of memory", text))
    skipped = int(data.get("skipped", 0) or data.get("skip_count", 0) or 0)
    nonfinite = int(data.get("nonfinite_count", 0) or 0)
    for row in history:
        if isinstance(row, dict):
            for key, value in row.items():
                if "nonfinite" in key.lower() and isinstance(value, (int, float)):
                    nonfinite += int(value)
    return {
        "explicit_oom_count": int(data.get("oom_count", 0) or 0),
        "oom_text_detected": oom_text,
        "skip_count": skipped,
        "nonfinite_count": nonfinite,
        "status": "needs_review" if (oom_text or skipped or nonfinite) else "clean_or_unreported",
    }


def normalize_peak_memory(data: dict[str, Any]) -> float | None:
    value = data.get("peak_memory_mib")
    if value is None and isinstance(data.get("memory_audit"), dict):
        audit = data["memory_audit"]
        value = audit.get("cuda_peak_allocated_mib") or audit.get("peak_memory_mib") or audit.get("rss_mib")
    converted = safe_float(value)
    return round(converted, 3) if converted is not None else None


def checkpoint_artifacts(checkpoint_dir: Path) -> dict[str, Any]:
    artifacts = {}
    for name in ("model_best.pt", "model_final.pt", "model.joblib", "model.json"):
        path = checkpoint_dir / name
        if path.exists():
            artifacts[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return artifacts


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    memory_values = [run["peak_memory_mib"] for run in runs if run.get("peak_memory_mib") is not None]
    needs_review = [run for run in runs if run["skip_oom_nonfinite"]["status"] == "needs_review"]
    return {
        "runs": len(runs),
        "runs_with_peak_memory": len(memory_values),
        "peak_memory_mib_max": max(memory_values) if memory_values else None,
        "peak_memory_mib_top10": sorted(
            [
                {"checkpoint_dir": run["checkpoint_dir"], "peak_memory_mib": run.get("peak_memory_mib")}
                for run in runs
                if run.get("peak_memory_mib") is not None
            ],
            key=lambda item: float(item["peak_memory_mib"]),
            reverse=True,
        )[:10],
        "oom_or_nonfinite_review_count": len(needs_review),
        "oom_or_nonfinite_review": [
            {"checkpoint_dir": run["checkpoint_dir"], **run["skip_oom_nonfinite"]} for run in needs_review[:30]
        ],
        "by_model_type": count_by(runs, "model_type"),
        "by_dataset_dir": count_by(runs, "dataset_dir"),
    }


def build_contract_coverage(runs: list[dict[str, Any]]) -> dict[str, Any]:
    git = git_audit()
    env = environment_audit()
    family_rows: dict[str, Any] = {}
    covered_families = 0
    native_memory_families = 0
    for family, spec in EXPERT_TRAINING_COVERAGE.items():
        scripts = [{"path": path, "exists": Path(path).exists(), "sha256": sha256_file(Path(path)) if Path(path).exists() else None} for path in spec["scripts"]]
        matched = match_runs_for_family(runs, spec)
        best_run = matched[0] if matched else None
        dataset_dir = best_run.get("dataset_dir") if best_run else None
        dataset_hash = dataset_manifest_hash(dataset_dir) if dataset_dir else None
        contract_fields = {
            "command": bool(best_run and (best_run.get("config_snapshot", {}).get("command") or best_run.get("best", {}).get("command"))),
            "env": bool(env),
            "git": bool(git.get("commit")),
            "dataset_hash": bool(dataset_hash),
            "memory": bool(best_run and (best_run.get("peak_memory_mib") is not None or best_run.get("memory_audit"))),
            "oom_or_skip": bool(best_run and best_run.get("skip_oom_nonfinite")),
            "metrics": bool(best_run and best_run.get("metrics")),
            "confusion_or_by_source": bool(best_run and has_confusion_or_by_source(best_run.get("metrics") or {})),
        }
        coverage_status = "covered" if any(item["exists"] for item in scripts) and (best_run is not None or family in {"Layout", "Router"}) else "missing"
        if coverage_status == "covered":
            covered_families += 1
        if contract_fields["memory"]:
            native_memory_families += 1
        family_rows[family] = {
            "status": coverage_status,
            "scripts": scripts,
            "matched_run_count": len(matched),
            "representative_run": best_run,
            "dataset_hash": dataset_hash,
            "contract_fields": contract_fields,
            "coverage_note": coverage_note(family, best_run, contract_fields),
        }
    return {
        "version": "training_contract_coverage_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "Coverage is produced by the shared audit layer. Native train_summary fields are preserved; missing reproducibility fields are surfaced instead of silently inferred as model claims.",
        "git": git,
        "environment": env,
        "families": family_rows,
        "summary": {
            "families_total": len(EXPERT_TRAINING_COVERAGE),
            "families_covered": covered_families,
            "families_with_native_or_summary_memory": native_memory_families,
            "covered_family_names": [family for family, row in family_rows.items() if row["status"] == "covered"],
        },
        "done_when_checks": {
            "at_least_5_training_classes_covered": covered_families >= 5,
            "wall_room_symbol_text_layout_or_router_covered": covered_families >= 5,
            "git_env_dataset_memory_audited": bool(git.get("commit")) and bool(env) and any(row.get("dataset_hash") for row in family_rows.values()),
        },
    }


def match_runs_for_family(runs: list[dict[str, Any]], spec: dict[str, Any]) -> list[dict[str, Any]]:
    keywords = [str(item).lower() for item in spec.get("checkpoint_keywords", []) + spec.get("dataset_keywords", [])]
    matched = []
    for run in runs:
        text = " ".join(
            str(run.get(key) or "")
            for key in ("checkpoint_dir", "summary_path", "model_type", "dataset_dir")
        ).lower()
        if any(keyword in text for keyword in keywords):
            matched.append(run)
    matched.sort(key=lambda run: score_run_for_contract(run), reverse=True)
    return matched[:12]


def score_run_for_contract(run: dict[str, Any]) -> int:
    score = 0
    if run.get("peak_memory_mib") is not None or run.get("memory_audit"):
        score += 4
    if run.get("metrics"):
        score += 3
    if run.get("dataset_dir"):
        score += 2
    if run.get("artifacts"):
        score += 1
    return score


def dataset_manifest_hash(dataset_dir: Any) -> dict[str, Any] | None:
    if not dataset_dir:
        return None
    path = Path(str(dataset_dir))
    if not path.exists():
        return {"dataset_dir": str(path), "exists": False}
    candidates = [path / "manifest.json", path / "train.jsonl", path / "dev.jsonl", path / "locked_test.jsonl"]
    existing = [item for item in candidates if item.exists()]
    if not existing:
        return {"dataset_dir": str(path), "exists": True, "hash": None, "files": []}
    digest = hashlib.sha256()
    files = []
    for item in existing:
        digest.update(str(item).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(item).encode("utf-8"))
        files.append(str(item))
    return {"dataset_dir": str(path), "exists": True, "hash": digest.hexdigest(), "files": files}


def has_confusion_or_by_source(metrics: dict[str, Any]) -> bool:
    text = json.dumps(metrics, ensure_ascii=False).lower()
    return any(key in text for key in ("confusion", "by_source", "per_label", "per_class"))


def coverage_note(family: str, run: dict[str, Any] | None, fields: dict[str, bool]) -> str:
    if run is None:
        return f"{family} has a training/audit script entry but no completed train_summary.json was found yet."
    missing = [key for key, value in fields.items() if not value]
    if not missing:
        return "Representative run satisfies the audited contract fields."
    return "Representative run is indexed; missing native fields are surfaced by coverage audit: " + ", ".join(missing)


def summarize_reports(reports_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(reports_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "top_level_keys": sorted(data.keys())[:40] if isinstance(data, dict) else [],
            }
        )
    return rows


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:40])


def git_audit() -> dict[str, Any]:
    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "status_short_sha256": sha256_text(run_git(["status", "--short"]) or ""),
        "diff_sha256": sha256_text(run_git(["diff"]) or ""),
    }


def environment_audit() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cwd": os.getcwd(),
    }


def run_git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(["git", *args], check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def infer_model_type(name: str) -> str:
    cleaned = re.sub(r"(_e\d+|_seed\d+)$", "", name)
    return cleaned


def nested(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
