#!/usr/bin/env python3
"""Audit zero-shot VLM reports and source-level graph-model generalization."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


LABELS = ["hard_wall", "door", "window"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", default="reports/vlm")
    parser.add_argument("--output", default="reports/vlm/zero_shot_performance_audit.json")
    parser.add_argument("--graph-predictions", default="reports/vlm/graph_node_classifier_ensemble_weighted_smoke_predictions.jsonl")
    parser.add_argument("--object-predictions", default="reports/vlm/graph_object_oracle_classifier_v3_patch_smoke_predictions.jsonl")
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    report = {
        "date": "2026-04-30",
        "purpose": "Separate base/zero-shot VLM capability from trained CadStruct structural-module behavior.",
        "zero_shot_vlm_reports": collect_vlm_reports(reports_dir),
        "trained_structure_source_breakdown": {
            "primitive_node_weighted_ensemble": source_breakdown(Path(args.graph_predictions), "nodes"),
            "oracle_object_patch_classifier": source_breakdown(Path(args.object_predictions), "groups"),
        },
        "interpretation": [
            "Existing VLM zero-shot reports are too small for paper claims; several are 1-2 sample smoke checks.",
            "Base VLM zero-shot JSON/schema behavior can be measured with evaluate_backend.py, but dense primitive id assignment remains the key failure mode.",
            "For paper-grade zero-shot claims, run the same CadStruct smoke/dev split across Qwen3-VL, InternVL3.5, GLM-4.6V, and Kimi-VL without adapters, then report semantic_exact_f1 and geometry_consistency.",
            "For our trained structure path, source-dataset breakdown is the minimum guard against reporting a single aggregate number that hides cross-dataset weakness.",
        ],
        "recommended_zero_shot_protocol": {
            "datasets": [
                "datasets/cadstruct/smoke.jsonl for fast reproducibility",
                "datasets/cadstruct/dev.jsonl for paper-grade zero-shot model comparison",
            ],
            "models": [
                "Qwen/Qwen3-VL-8B-Instruct",
                "Qwen/Qwen3-VL-32B-Instruct",
                "OpenGVLab/InternVL3_5-14B-HF without LoRA",
                "GLM-4.6V and Kimi-VL where loaders are compatible",
            ],
            "metrics": [
                "json_success_rate",
                "semantic_hit_rate",
                "semantic_exact_f1_mean",
                "geometry_consistency_mean",
                "relation_f1_mean",
                "empty_semantic_rate",
                "latency_ms.mean",
            ],
            "paper_rule": "Do not cite reports with total < 30 as zero-shot evidence; mark them as smoke only.",
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def collect_vlm_reports(reports_dir: Path) -> list[dict[str, Any]]:
    candidates = [
        reports_dir / name
        for name in [
            "qwen3_vl_8b_smoke.json",
            "qwen3_vl_8b_smoke_4.json",
            "internvl3_5_14b_smoke_1.json",
            "cadstruct_14b_lora_smoke_1.json",
            "cadstruct_14b_lora_smoke_2.json",
            "cadstruct_14b_lora_structural_smoke_2_exact.json",
            "cadstruct_14b_lora_structural_smoke_2_exact.audit.json",
        ]
    ]
    candidates.extend(sorted((reports_dir / "zero_shot_runs").glob("*.json")))
    rows = []
    seen = set()
    for path in candidates:
        if path.name == "summary.json" or path in seen:
            continue
        seen.add(path)
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        rows.append(summarize_vlm_report(path, data))
    return rows


def summarize_vlm_report(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    rows = data.get("rows") if isinstance(data.get("rows"), list) else []
    total = int(data.get("total", len(rows)) or 0)
    model_kind = infer_model_kind(path.name)
    return {
        "report": str(path),
        "model_kind": model_kind,
        "evidence_level": "paper_candidate" if total >= 30 else "smoke_only",
        "total": total,
        "ok": data.get("ok"),
        "dataset": data.get("dataset"),
        "json_success_rate": data.get("json_success_rate"),
        "semantic_hit_rate": data.get("semantic_hit_rate"),
        "semantic_exact_f1_mean": data.get("semantic_exact_f1_mean"),
        "relation_f1_mean": data.get("relation_f1_mean"),
        "geometry_consistency_mean": data.get("geometry_consistency_mean"),
        "empty_semantic_rate": data.get("empty_semantic_rate"),
        "semantic_count_mean": data.get("semantic_count_mean"),
        "latency_ms_mean": (data.get("latency_ms") or {}).get("mean") if isinstance(data.get("latency_ms"), dict) else None,
        "warning_counts": data.get("warning_counts"),
    }


def infer_model_kind(name: str) -> str:
    if "qwen" in name:
        return "zero_shot_base_vlm"
    if "internvl3_5_14b" in name:
        return "zero_shot_base_vlm"
    if "lora" in name:
        return "adapted_lora"
    return "unknown"


def source_breakdown(path: Path, record_key: str) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(path):
        source = str(row.get("source_dataset") or "unknown")
        buckets[source].append(row)
    return {
        "path": str(path),
        "record_key": record_key,
        "overall": metrics_for_rows([row for rows in buckets.values() for row in rows], record_key),
        "by_source_dataset": {source: metrics_for_rows(rows, record_key) for source, rows in sorted(buckets.items())},
    }


def metrics_for_rows(rows: list[dict[str, Any]], record_key: str) -> dict[str, Any]:
    confusion = {target: {pred: 0 for pred in LABELS} for target in LABELS}
    confidences = []
    samples = len(rows)
    records = 0
    for row in rows:
        for item in row.get(record_key) or []:
            target = item.get("label")
            pred = item.get("prediction")
            if target not in LABELS or pred not in LABELS:
                continue
            confusion[target][pred] += 1
            records += 1
            if item.get("confidence") is not None:
                confidences.append(float(item["confidence"]))
    return {
        "samples": samples,
        "records": records,
        **metrics_from_confusion(confusion),
        "confidence_mean": round(statistics.mean(confidences), 6) if confidences else 0.0,
    }


def metrics_from_confusion(confusion: dict[str, dict[str, int]]) -> dict[str, Any]:
    total = sum(sum(row.values()) for row in confusion.values())
    correct = sum(confusion[label][label] for label in LABELS)
    per_label = {}
    f1s = []
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[target][label] for target in LABELS) - tp
        fn = sum(confusion[label].values()) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        per_label[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": sum(confusion[label].values()),
        }
    return {
        "accuracy": round(correct / total, 6) if total else 0.0,
        "macro_f1": round(sum(f1s) / len(f1s), 6),
        "per_label": per_label,
        "confusion": [[confusion[target][pred] for pred in LABELS] for target in LABELS],
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    main()
