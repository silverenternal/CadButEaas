#!/usr/bin/env python3
"""Sweep gates over P0-68 added candidates to reduce precision loss/inflation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from apply_symbol_rtdetr_complement_policy_p065 import read_exported_golds
from sweep_symbol_rtdetr_complement_gate_p063 import pred_area_bucket, read_predictions, score
from train_symbol_tile_detector_v20 import bbox_iou, rel, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21"
DEFAULT_YOLO_DIR = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21_yolo_seg_rect_v27"
DEFAULT_V28 = ROOT / "reports/vlm/symbol_yolov8s_seg_rect_v28_smoke_v30_page_predictions_p062_refresh.jsonl"
DEFAULT_COMBINED = ROOT / "reports/vlm/symbol_combined_optional_policy_p068_smoke_v30_predictions.jsonl"
DEFAULT_JSON = ROOT / "reports/vlm/symbol_added_candidate_precision_gate_p069_smoke_v30.json"
DEFAULT_MD = ROOT / "reports/vlm/symbol_added_candidate_precision_gate_p069_smoke_v30.md"
DEFAULT_PRED = ROOT / "reports/vlm/symbol_added_candidate_precision_gate_p069_smoke_v30_predictions.jsonl"

LABEL_GROUPS = {
    "all": None,
    "sink": {"sink"},
    "sink_shower_equipment": {"sink", "shower", "equipment"},
    "sink_equipment": {"sink", "equipment"},
    "not_shower": {"sink", "equipment", "appliance", "column", "generic_symbol", "stair", "bathtub"},
}
AREA_GROUPS = {
    "all": None,
    "tiny": {"tiny_le_64"},
    "tiny_small": {"tiny_le_64", "small_le_256"},
    "nonlarge": {"tiny_le_64", "small_le_256", "medium_le_1024"},
}
SOURCE_GROUPS = {
    "all": None,
    "p065_only": {"p065_rtdetr_complement"},
    "p067_only": {"p067_center_only_box_repair"},
    "p065_p067": {"p065_rtdetr_complement", "p067_center_only_box_repair"},
}


def max_overlap(prediction: dict[str, Any], references: list[dict[str, Any]]) -> float:
    pred_bbox = [float(value) for value in prediction["bbox"]]
    return max((bbox_iou(pred_bbox, [float(value) for value in reference["bbox"]]) for reference in references), default=0.0)


def source_name(prediction: dict[str, Any]) -> str:
    return str(prediction.get("source_policy") or "v28")


def keep_added(prediction: dict[str, Any], v28_items: list[dict[str, Any]], config: dict[str, Any]) -> bool:
    labels = LABEL_GROUPS[str(config["labels"])]
    areas = AREA_GROUPS[str(config["areas"])]
    sources = SOURCE_GROUPS[str(config["sources"])]
    if labels is not None and str(prediction.get("label")) not in labels:
        return False
    if areas is not None and pred_area_bucket(prediction) not in areas:
        return False
    if sources is not None and source_name(prediction) not in sources:
        return False
    if float(prediction.get("score", 0.0)) < float(config["score_min"]):
        return False
    overlap = max_overlap(prediction, v28_items)
    if overlap < float(config["min_iou_with_v28"]):
        return False
    if overlap >= float(config["max_iou_with_v28"]):
        return False
    return True


def gated_predictions(v28_predictions: dict[str, list[dict[str, Any]]], combined_predictions: dict[str, list[dict[str, Any]]], config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    max_add_per_page = int(config["max_add_per_page"])
    for row_id in sorted(set(v28_predictions) | set(combined_predictions)):
        v28_items = list(v28_predictions.get(row_id, []))
        added = [item for item in combined_predictions.get(row_id, []) if source_name(item) != "v28"]
        kept = [item for item in added if keep_added(item, v28_items, config)]
        kept.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        output[row_id] = v28_items + kept[:max_add_per_page]
    return output


def rows_from_predictions(predictions: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [{"row_id": row_id, "predicted_symbols": items} for row_id, items in sorted(predictions.items())]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--yolo-dir", default=str(DEFAULT_YOLO_DIR))
    parser.add_argument("--split", default="smoke_v30")
    parser.add_argument("--v28-predictions", default=str(DEFAULT_V28))
    parser.add_argument("--combined-predictions", default=str(DEFAULT_COMBINED))
    parser.add_argument("--output-json", default=str(DEFAULT_JSON))
    parser.add_argument("--output-md", default=str(DEFAULT_MD))
    parser.add_argument("--output-predictions", default=str(DEFAULT_PRED))
    args = parser.parse_args()

    v28_predictions = read_predictions(Path(args.v28_predictions))
    combined_predictions = read_predictions(Path(args.combined_predictions))
    golds = read_exported_golds(Path(args.data), Path(args.yolo_dir), args.split, set(v28_predictions) | set(combined_predictions))
    baseline = score(golds, v28_predictions, {row_id: [] for row_id in v28_predictions}, {"labels": "all", "areas": "all", "score_min": 1.1, "max_iou_with_v28": 0.0, "max_add_per_page": 0})
    full_combined = score(golds, combined_predictions, {row_id: [] for row_id in combined_predictions}, {"labels": "all", "areas": "all", "score_min": 1.1, "max_iou_with_v28": 0.0, "max_add_per_page": 0})
    target_half_gain = baseline["iou_0_30_recall"] + 0.5 * (full_combined["iou_0_30_recall"] - baseline["iou_0_30_recall"])

    configs: list[dict[str, Any]] = [
        {"sources": "all", "labels": "all", "areas": "all", "score_min": 0.10, "min_iou_with_v28": 0.0, "max_iou_with_v28": 1.01, "max_add_per_page": 20},
        {"sources": "all", "labels": "all", "areas": "all", "score_min": 0.20, "min_iou_with_v28": 0.0, "max_iou_with_v28": 1.01, "max_add_per_page": 20},
        {"sources": "all", "labels": "all", "areas": "all", "score_min": 0.35, "min_iou_with_v28": 0.0, "max_iou_with_v28": 1.01, "max_add_per_page": 20},
        {"sources": "p067_only", "labels": "sink_shower_equipment", "areas": "tiny_small", "score_min": 0.10, "min_iou_with_v28": 0.0, "max_iou_with_v28": 1.01, "max_add_per_page": 20},
        {"sources": "p067_only", "labels": "sink_shower_equipment", "areas": "tiny_small", "score_min": 0.20, "min_iou_with_v28": 0.0, "max_iou_with_v28": 1.01, "max_add_per_page": 20},
        {"sources": "p067_only", "labels": "sink_shower_equipment", "areas": "tiny_small", "score_min": 0.35, "min_iou_with_v28": 0.0, "max_iou_with_v28": 1.01, "max_add_per_page": 20},
        {"sources": "p065_p067", "labels": "sink", "areas": "tiny", "score_min": 0.20, "min_iou_with_v28": 0.0, "max_iou_with_v28": 1.01, "max_add_per_page": 20},
        {"sources": "p065_p067", "labels": "sink", "areas": "tiny", "score_min": 0.35, "min_iou_with_v28": 0.0, "max_iou_with_v28": 1.01, "max_add_per_page": 20},
        {"sources": "p067_only", "labels": "sink_shower_equipment", "areas": "tiny_small", "score_min": 0.10, "min_iou_with_v28": 0.05, "max_iou_with_v28": 1.01, "max_add_per_page": 20},
        {"sources": "p067_only", "labels": "sink_shower_equipment", "areas": "tiny_small", "score_min": 0.10, "min_iou_with_v28": 0.0, "max_iou_with_v28": 0.60, "max_add_per_page": 20},
        {"sources": "p067_only", "labels": "sink", "areas": "tiny", "score_min": 0.10, "min_iou_with_v28": 0.0, "max_iou_with_v28": 1.01, "max_add_per_page": 20},
        {"sources": "all", "labels": "sink", "areas": "tiny", "score_min": 0.20, "min_iou_with_v28": 0.0, "max_iou_with_v28": 1.01, "max_add_per_page": 20},
    ]

    results = []
    for config in configs:
        candidate_predictions = gated_predictions(v28_predictions, combined_predictions, config)
        metrics = score(golds, candidate_predictions, {row_id: [] for row_id in candidate_predictions}, {"labels": "all", "areas": "all", "score_min": 1.1, "max_iou_with_v28": 0.0, "max_add_per_page": 0})
        metrics["config"] = config
        metrics["delta_vs_v28"] = {key: round(float(metrics[key]) - float(baseline[key]), 6) for key in ["iou_0_30_recall", "center_recall", "candidate_inflation", "precision"]}
        metrics["retained_iou_gain_fraction"] = round((metrics["iou_0_30_recall"] - baseline["iou_0_30_recall"]) / max(full_combined["iou_0_30_recall"] - baseline["iou_0_30_recall"], 1e-9), 6)
        metrics["tiny_iou_delta"] = round(metrics["per_area_iou_recall"].get("tiny_le_64", 0.0) - baseline["per_area_iou_recall"].get("tiny_le_64", 0.0), 6)
        metrics["sink_iou_delta"] = round(metrics["per_label_iou_recall"].get("sink", 0.0) - baseline["per_label_iou_recall"].get("sink", 0.0), 6)
        results.append(metrics)

    feasible = [item for item in results if item["iou_0_30_recall"] >= target_half_gain and item["candidate_inflation"] < full_combined["candidate_inflation"] and item["precision"] >= full_combined["precision"]]
    feasible.sort(key=lambda item: (item["precision"], -item["candidate_inflation"], item["iou_0_30_recall"], item["tiny_iou_delta"]), reverse=True)
    recall_sorted = sorted(results, key=lambda item: (item["iou_0_30_recall"], item["precision"], -item["candidate_inflation"]), reverse=True)[:20]
    best = feasible[0] if feasible else None
    if best:
        write_jsonl(Path(args.output_predictions), rows_from_predictions(gated_predictions(v28_predictions, combined_predictions, best["config"])))

    report = {
        "version": "symbol_added_candidate_precision_gate_p069_smoke_v30",
        "source_integrity": "offline gold is used only for smoke sweep; runtime gate uses raster-derived prediction fields only",
        "inputs": {"v28_predictions": rel(Path(args.v28_predictions)), "combined_predictions": rel(Path(args.combined_predictions)), "data": rel(Path(args.data)), "yolo_dir": rel(Path(args.yolo_dir))},
        "baseline_v28": baseline,
        "full_combined_p068": full_combined,
        "target_half_gain_iou_recall": round(target_half_gain, 6),
        "best_feasible": best,
        "top_feasible": feasible[:20],
        "top_recall": recall_sorted,
        "sweep_count": len(results),
        "decision": "positive_smoke_gate_validate_locked" if best else "negative_no_precision_gate_found",
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# P0-69 added-candidate precision gate smoke sweep",
        "",
        "## Summary",
        "",
        f"- baseline v28 IoU / inflation / precision: `{baseline['iou_0_30_recall']:.6f}` / `{baseline['candidate_inflation']:.6f}` / `{baseline['precision']:.6f}`",
        f"- full P0-68 IoU / inflation / precision: `{full_combined['iou_0_30_recall']:.6f}` / `{full_combined['candidate_inflation']:.6f}` / `{full_combined['precision']:.6f}`",
        f"- half-gain target IoU: `{target_half_gain:.6f}`",
        f"- decision: `{report['decision']}`",
    ]
    if best:
        delta = best["delta_vs_v28"]
        lines.extend([
            f"- best IoU / inflation / precision: `{best['iou_0_30_recall']:.6f}` / `{best['candidate_inflation']:.6f}` / `{best['precision']:.6f}`",
            f"- delta vs v28 IoU / inflation / precision: `{delta['iou_0_30_recall']:+.6f}` / `{delta['candidate_inflation']:+.6f}` / `{delta['precision']:+.6f}`",
            f"- retained P0-68 IoU gain fraction: `{best['retained_iou_gain_fraction']:.6f}`",
            f"- tiny / sink IoU delta: `{best['tiny_iou_delta']:+.6f}` / `{best['sink_iou_delta']:+.6f}`",
            f"- config: `{json.dumps(best['config'], ensure_ascii=False)}`",
        ])
    lines.extend(["", "## Artifacts", "", f"- `{rel(Path(args.output_json))}`", f"- `{rel(Path(args.output_md))}`", f"- `{rel(Path(args.output_predictions))}`", ""])
    Path(args.output_md).write_text("\n".join(lines))
    print(json.dumps({"decision": report["decision"], "baseline": {k: baseline[k] for k in ["iou_0_30_recall", "candidate_inflation", "precision"]}, "full_combined": {k: full_combined[k] for k in ["iou_0_30_recall", "candidate_inflation", "precision"]}, "best": None if best is None else {k: best[k] for k in ["iou_0_30_recall", "candidate_inflation", "precision", "retained_iou_gain_fraction", "tiny_iou_delta", "sink_iou_delta", "config"]}}, ensure_ascii=False, indent=2)[:6000])


if __name__ == "__main__":
    main()
