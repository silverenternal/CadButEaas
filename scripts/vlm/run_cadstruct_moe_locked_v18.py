#!/usr/bin/env python3
"""One-command locked-run manifest for the raster-only CadStruct MoE v18 path."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "reports/vlm/cadstruct_moe_locked_manifest_v18.json"
QUALITY_FLOOR = 0.98
KEY_REPORTS = {
    "v17_image_moe": ROOT / "reports/vlm/image_only_moe_v17_eval.json",
    "v17_runtime_contract": ROOT / "reports/vlm/moe_expert_runtime_contract_v17.json",
    "v18_relation_baseline": ROOT / "reports/vlm/relation_graph_reconstruction_v18_eval.json",
    "v18_relation_compressed_diagnostic": ROOT / "reports/vlm/relation_graph_from_scored_cache_contains_assignment_compressed_v18_eval.json",
    "v18_symbol_type": ROOT / "reports/vlm/symbol_type_classifier_v18_eval.json",
    "v18_symbol_visual_evidence": ROOT / "reports/vlm/symbol_visual_evidence_v8_to_v18_audit.json",
}


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def git_status_summary() -> dict[str, Any]:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return {"returncode": result.returncode, "changed_path_count": len(lines), "sample": lines[:40]}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"load_error": str(exc)}


def artifact(path: str, role: str) -> dict[str, Any]:
    p = ROOT / path
    return {
        "path": path,
        "role": role,
        "exists": p.exists(),
        "size_bytes": p.stat().st_size if p.exists() else 0,
    }


def add_metric(metrics: dict[str, float], key: str, value: Any) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        metrics[key] = round(float(value), 6)


def collect_metrics() -> dict[str, float]:
    metrics: dict[str, float] = {}
    v17 = load_json(KEY_REPORTS["v17_image_moe"])
    add_metric(metrics, "image_moe_candidate_mean_f1", v17.get("candidate_mean_f1"))
    add_metric(metrics, "image_moe_final_mean_f1", v17.get("final_mean_f1"))
    for family, report in (v17.get("expert_metrics") or {}).items():
        if isinstance(report, dict):
            for field in ("precision", "recall", "f1", "label_precision", "label_recall", "label_f1"):
                add_metric(metrics, f"image_moe_{family}_{field}", report.get(field))

    relation = load_json(KEY_REPORTS["v18_relation_baseline"])
    for rel, report in (relation.get("relation_metrics") or {}).items():
        if isinstance(report, dict):
            for field in ("precision", "recall", "f1"):
                add_metric(metrics, f"relation_baseline_{rel}_{field}", report.get(field))

    compressed = load_json(KEY_REPORTS["v18_relation_compressed_diagnostic"])
    for rel, report in (compressed.get("relation_metrics") or {}).items():
        if isinstance(report, dict):
            for field in ("precision", "recall", "f1"):
                add_metric(metrics, f"relation_compressed_{rel}_{field}", report.get(field))

    symbol_type = load_json(KEY_REPORTS["v18_symbol_type"])
    locked_symbol = symbol_type.get("locked") or {}
    add_metric(metrics, "symbol_type_bbox_center_recall", locked_symbol.get("symbol_bbox_center_recall"))
    typed = locked_symbol.get("typed_label") or {}
    for field in ("precision", "recall", "f1"):
        add_metric(metrics, f"symbol_type_raw_typed_label_{field}", typed.get(field))
    safe_locked_symbol = symbol_type.get("locked_safe_export") or {}
    add_metric(metrics, "symbol_type_safe_bbox_center_recall", safe_locked_symbol.get("symbol_bbox_center_recall"))
    add_metric(metrics, "symbol_type_safe_typed_prediction_coverage", safe_locked_symbol.get("typed_prediction_coverage"))
    safe_typed = safe_locked_symbol.get("typed_label") or {}
    for field in ("precision", "recall", "f1"):
        add_metric(metrics, f"symbol_type_safe_typed_label_{field}", safe_typed.get(field))

    visual_evidence = load_json(KEY_REPORTS["v18_symbol_visual_evidence"])
    add_metric(metrics, "symbol_visual_evidence_reduction", visual_evidence.get("reduction"))
    add_metric(metrics, "symbol_visual_evidence_locked_recall_drop_abs", visual_evidence.get("locked_recall_drop_abs"))
    visual_after = visual_evidence.get("locked_recall_after") or {}
    add_metric(metrics, "symbol_visual_evidence_locked_recall_after", visual_after.get("center_or_iou_recall"))
    return metrics


def worst_metrics(metrics: dict[str, float], limit: int = 20) -> list[dict[str, Any]]:
    below = [
        {"metric": key, "value": value, "gap_to_0_98": round(QUALITY_FLOOR - value, 6)}
        for key, value in metrics.items()
        if value < QUALITY_FLOOR
    ]
    below.sort(key=lambda item: item["value"])
    return below[:limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--quality-floor", type=float, default=QUALITY_FLOOR)
    args = parser.parse_args()
    started_at = datetime.now(timezone.utc).isoformat()
    metrics = collect_metrics()
    failing = worst_metrics(metrics, limit=10000)
    worst = worst_metrics(metrics, limit=20)
    missing_reports = [
        str(path.relative_to(ROOT))
        for path in KEY_REPORTS.values()
        if not path.exists()
    ]
    runtime_contract = load_json(KEY_REPORTS["v17_runtime_contract"])
    contract_gate = runtime_contract.get("source_integrity_gate") or {}
    failure_reasons: list[str] = []
    if missing_reports:
        failure_reasons.append(f"Missing locked report artifacts: {', '.join(missing_reports)}")
    if contract_gate and not bool(contract_gate.get("passed")):
        failure_reasons.append("Runtime/source-integrity contract failed.")
    if failing:
        failure_reasons.append(f"{len(failing)} collected locked metrics are below {args.quality_floor}.")
    failure_reasons.append("No integrated locked v18 raster-to-graph runner exists yet; this is a locked gate aggregator over current artifacts.")

    manifest = {
        "schema_version": "cadstruct_artifact_manifest_v1",
        "run_id": "cadstruct_moe_locked_v18",
        "pipeline_id": "cadstruct_moe_locked_v18",
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "mode": "locked_gate_aggregator",
        "command_or_commands": [
            ".venv/bin/python scripts/vlm/run_cadstruct_moe_locked_v18.py",
            "Aggregates current locked eval/audit artifacts and applies absolute 0.98 gates.",
        ],
        "cwd": str(ROOT),
        "git_status_summary": git_status_summary(),
        "inputs": [
            artifact("struct.json", "architecture contract"),
            artifact("todo.json", "active execution plan"),
            artifact("datasets/cadstruct_cubicasa5k_moe_locked", "locked offline evaluation labels"),
        ],
        "checkpoints": [],
        "outputs": [artifact("reports/vlm/cadstruct_moe_locked_manifest_v18.json", "this manifest")],
        "eval_reports": [
            artifact("reports/vlm/image_only_moe_v17_eval.json", "image-only MoE v17 locked/dev evaluation"),
            artifact("reports/vlm/relation_graph_reconstruction_v18_eval.json", "relation graph v18 baseline evaluation"),
            artifact("reports/vlm/relation_graph_from_scored_cache_contains_assignment_compressed_v18_eval.json", "relation compressed diagnostic evaluation"),
            artifact("reports/vlm/symbol_type_classifier_v18_eval.json", "raster symbol type classifier evaluation"),
            artifact("reports/vlm/symbol_visual_evidence_v8_to_v18_audit.json", "raster-only symbol visual-evidence gate audit"),
        ],
        "audit_reports": [
            artifact("reports/vlm/moe_expert_runtime_contract_v17.json", "runtime/source-integrity contract"),
            artifact("reports/vlm/symbol_visual_evidence_v8_to_v18_audit.json", "symbol visual-evidence v8 to v18 adapter audit"),
        ],
        "review_packs": [],
        "metrics_summary": {
            "production_quality_target": args.quality_floor,
            "metric_count": len(metrics),
            "metrics_below_target": len(failing),
            "worst_metrics": worst,
            **metrics,
        },
        "gate_decision": "fail",
        "failure_reasons": failure_reasons,
        "source_integrity_summary": {
            "inference_input": "raster image only",
            "offline_labels_allowed_for_locked_eval": True,
            "svg_or_parser_geometry_allowed_at_runtime": False,
            "v17_contract_gate": contract_gate,
        },
    }
    write_json(Path(args.output), manifest)


if __name__ == "__main__":
    main()
