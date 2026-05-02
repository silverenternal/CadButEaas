#!/usr/bin/env python3
"""Audit teacher-assisted TextDimension recovery without changing the main path."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from train_text_dimension_expert import evaluate_predictions, load_jsonl, predict_dimension_links, write_jsonl
except ImportError:  # pragma: no cover
    from scripts.vlm.train_text_dimension_expert import evaluate_predictions, load_jsonl, predict_dimension_links, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="datasets/text_dimension_expert_v2")
    parser.add_argument("--baseline-dir", default="checkpoints/text_dimension_expert_v2")
    parser.add_argument("--output", default="reports/vlm/text_dimension_teacher_ablation_v1.json")
    parser.add_argument("--predictions-output", default="reports/vlm/text_dimension_teacher_predictions_v1.jsonl")
    parser.add_argument("--confidence-threshold", type=float, default=0.05)
    args = parser.parse_args()

    started = time.perf_counter()
    dataset_dir = Path(args.dataset_dir)
    baseline_dir = Path(args.baseline_dir)

    splits: dict[str, Any] = {}
    teacher_rows_for_export: list[dict[str, Any]] = []
    for split in ("dev", "smoke"):
        baseline_path = baseline_dir / f"{split}_predictions.jsonl"
        data_path = dataset_dir / f"{split}.jsonl"
        if not baseline_path.exists() or not data_path.exists():
            continue
        dataset_rows = load_jsonl(data_path)
        baseline_rows = load_jsonl(baseline_path)
        ablated = build_teacher_rows(baseline_rows, dataset_rows, args.confidence_threshold)
        teacher_rows_for_export.extend(ablated["teacher_feature_rows"])
        splits[split] = {
            "baseline": evaluate_predictions(baseline_rows),
            "teacher_feature_low_conf_only": evaluate_predictions(ablated["teacher_feature_rows"]),
            "teacher_distillation_upper_bound": evaluate_predictions(ablated["teacher_distillation_rows"]),
            "low_confidence_threshold": args.confidence_threshold,
            "teacher_signal_audit": ablated["audit"],
        }

    report = {
        "version": "text_dimension_teacher_ablation_v1",
        "dataset_dir": str(dataset_dir),
        "baseline_dir": str(baseline_dir),
        "teacher_type": "annotation_backed_svg_ocr_upper_bound",
        "deployment_note": (
            "No external OCR/VLM teacher predictions were present in the workspace. "
            "This audit uses available SVG/vector annotation fields as an auditable teacher upper bound; "
            "it must not be reported as a deployed OCR/VLM result."
        ),
        "splits": splits,
        "latency_audit": {
            "external_teacher_calls": 0,
            "measured_wall_time_seconds": round(time.perf_counter() - started, 6),
            "expected_runtime_impact": "offline-only teacher hints; no inference-time cost if distilled",
        },
        "failure_recovery_summary": summarize_recovery(splits),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_jsonl(Path(args.predictions_output), teacher_rows_for_export)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_teacher_rows(
    baseline_rows: list[dict[str, Any]],
    dataset_rows: list[dict[str, Any]],
    confidence_threshold: float,
) -> dict[str, Any]:
    gold_by_row = {row_key(row): row for row in dataset_rows}
    feature_rows: list[dict[str, Any]] = []
    distill_rows: list[dict[str, Any]] = []
    audit: Counter[str] = Counter()
    recovered_pairs: Counter[str] = Counter()

    for row in baseline_rows:
        gold_row = gold_by_row.get(row_key(row), {})
        gold_items = {str(item.get("id")): item for item in gold_row.get("text_candidates") or []}
        feature_row = dict(row)
        distill_row = dict(row)
        feature_items = []
        distill_items = []
        for item in row.get("text_candidates") or []:
            item_id = str(item.get("id"))
            gold_item = gold_items.get(item_id, {})
            teacher_label = gold_item.get("text_type")
            raw_text = str(gold_item.get("raw_text") or gold_item.get("normalized_text") or "")
            has_ocr_signal = bool(raw_text.strip())
            feature_item = dict(item)
            distill_item = dict(item)
            low_conf = float(item.get("confidence") or 0.0) < confidence_threshold
            if has_ocr_signal:
                audit["items_with_nonempty_ocr_text"] += 1
            else:
                audit["items_without_ocr_text"] += 1
            if teacher_label and low_conf:
                audit["teacher_feature_candidates"] += 1
                if item.get("prediction") != teacher_label:
                    recovered_pairs[f"{item.get('gold')}->{item.get('prediction')}"] += 1
                    audit["teacher_feature_changed_errors"] += int(item.get("gold") == teacher_label)
                feature_item["prediction"] = teacher_label
                feature_item["confidence"] = max(float(item.get("confidence") or 0.0), 0.9)
                feature_item["teacher_hint"] = "low_conf_annotation_backed_svg_ocr"
            if teacher_label:
                distill_item["prediction"] = teacher_label
                distill_item["confidence"] = 1.0
                distill_item["teacher_hint"] = "annotation_backed_distillation_upper_bound"
            feature_items.append(feature_item)
            distill_items.append(distill_item)
            audit["items"] += 1
        feature_row["text_candidates"] = feature_items
        distill_row["text_candidates"] = distill_items
        feature_row["dimension_links_pred"] = predict_dimension_links(feature_items)
        distill_row["dimension_links_pred"] = predict_dimension_links(distill_items)
        feature_row["teacher_mode"] = "feature_low_conf_only"
        distill_row["teacher_mode"] = "distillation_upper_bound"
        feature_rows.append(feature_row)
        distill_rows.append(distill_row)

    audit["recovered_error_pairs"] = dict(recovered_pairs)
    return {"teacher_feature_rows": feature_rows, "teacher_distillation_rows": distill_rows, "audit": dict(audit)}


def summarize_recovery(splits: dict[str, Any]) -> dict[str, Any]:
    summary = {}
    for split, payload in splits.items():
        base = payload["baseline"]
        feature = payload["teacher_feature_low_conf_only"]
        distill = payload["teacher_distillation_upper_bound"]
        summary[split] = {
            "macro_f1_delta_feature": round(feature["macro_f1"] - base["macro_f1"], 6),
            "dimension_link_f1_delta_feature": round(feature["dimension_link"]["f1"] - base["dimension_link"]["f1"], 6),
            "macro_f1_oracle_gap": round(distill["macro_f1"] - base["macro_f1"], 6),
            "dimension_link_f1_oracle_gap": round(distill["dimension_link"]["f1"] - base["dimension_link"]["f1"], 6),
            "conclusion": "relation recovery follows text-type recovery when dimension links are recomputed; the remaining blocker is replacing annotation-backed teacher hints with deployable OCR/VLM or learned low-confidence recovery.",
        }
    return summary


def row_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("image")), str(row.get("annotation"))


if __name__ == "__main__":
    main()
