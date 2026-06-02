#!/usr/bin/env python3
"""Sweep simple geometric repairs for post-policy center-only symbol boxes."""

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
from train_symbol_tile_detector_v20 import rel, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21"
DEFAULT_YOLO_DIR = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21_yolo_seg_rect_v27"
DEFAULT_POLICY = ROOT / "reports/vlm/symbol_rtdetr_complement_policy_p065_smoke_v30_predictions.jsonl"
DEFAULT_JSON = ROOT / "reports/vlm/symbol_center_only_box_repair_p067_smoke_v30.json"
DEFAULT_MD = ROOT / "reports/vlm/symbol_center_only_box_repair_p067_smoke_v30.md"
DEFAULT_PRED = ROOT / "reports/vlm/symbol_center_only_box_repair_p067_smoke_v30_predictions.jsonl"

LABEL_GROUPS = {
    "sink": {"sink"},
    "shower": {"shower"},
    "equipment": {"equipment"},
    "sink_shower_equipment": {"sink", "shower", "equipment"},
    "sink_shower": {"sink", "shower"},
}
AREA_GROUPS = {
    "tiny": {"tiny_le_64"},
    "tiny_small": {"tiny_le_64", "small_le_256"},
    "small_medium": {"small_le_256", "medium_le_1024"},
    "all_nonlarge": {"tiny_le_64", "small_le_256", "medium_le_1024"},
    "all": {"tiny_le_64", "small_le_256", "medium_le_1024", "large_le_4096", "xlarge_gt_4096"},
}


def scale_bbox(bbox: list[float], scale_x: float, scale_y: float) -> list[float]:
    left, top, right, bottom = [float(value) for value in bbox]
    cx = (left + right) / 2.0
    cy = (top + bottom) / 2.0
    width = max(1.0, (right - left) * scale_x)
    height = max(1.0, (bottom - top) * scale_y)
    return [cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0]


def repaired_predictions(page_predictions: dict[str, list[dict[str, Any]]], config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    labels = LABEL_GROUPS[str(config["labels"])]
    areas = AREA_GROUPS[str(config["areas"])]
    score_min = float(config["score_min"])
    scale_x = float(config["scale_x"])
    scale_y = float(config["scale_y"])
    max_add_per_page = int(config["max_add_per_page"])
    output: dict[str, list[dict[str, Any]]] = {}
    for row_id, predictions in page_predictions.items():
        additions = []
        for prediction in predictions:
            if str(prediction.get("label")) not in labels:
                continue
            if pred_area_bucket(prediction) not in areas:
                continue
            if float(prediction.get("score", 0.0)) < score_min:
                continue
            repaired = dict(prediction)
            repaired["bbox"] = scale_bbox([float(value) for value in prediction["bbox"]], scale_x, scale_y)
            repaired["score"] = float(prediction.get("score", 0.0)) * 0.999
            repaired["source_policy"] = "p067_center_only_box_repair"
            repaired["repair_scale_x"] = scale_x
            repaired["repair_scale_y"] = scale_y
            additions.append(repaired)
        additions.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        output[row_id] = list(predictions) + additions[:max_add_per_page]
    return output


def rows_from_predictions(page_predictions: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [{"row_id": row_id, "predicted_symbols": predictions} for row_id, predictions in sorted(page_predictions.items())]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--yolo-dir", default=str(DEFAULT_YOLO_DIR))
    parser.add_argument("--split", default="smoke_v30")
    parser.add_argument("--policy-predictions", default=str(DEFAULT_POLICY))
    parser.add_argument("--output-json", default=str(DEFAULT_JSON))
    parser.add_argument("--output-md", default=str(DEFAULT_MD))
    parser.add_argument("--output-predictions", default=str(DEFAULT_PRED))
    args = parser.parse_args()

    policy_predictions = read_predictions(Path(args.policy_predictions))
    golds = read_exported_golds(Path(args.data), Path(args.yolo_dir), args.split, set(policy_predictions))
    baseline = score(golds, policy_predictions, {row_id: [] for row_id in policy_predictions}, {"labels": "all", "areas": "all", "score_min": 1.1, "max_iou_with_v28": 0.0, "max_add_per_page": 0})

    configs: list[dict[str, Any]] = []
    for labels in ["sink", "shower", "equipment", "sink_shower_equipment"]:
        for areas in ["tiny", "tiny_small", "all_nonlarge"]:
            for score_min in [0.10, 0.20, 0.35]:
                for scale_x, scale_y in [(0.85, 0.85), (1.15, 1.15), (1.35, 1.35), (0.75, 1.35), (1.35, 0.75)]:
                    for max_add_per_page in [5, 20]:
                        configs.append({
                            "labels": labels,
                            "areas": areas,
                            "score_min": score_min,
                            "scale_x": scale_x,
                            "scale_y": scale_y,
                            "max_add_per_page": max_add_per_page,
                        })

    results = []
    for config in configs:
        candidate_predictions = repaired_predictions(policy_predictions, config)
        metrics = score(golds, candidate_predictions, {row_id: [] for row_id in candidate_predictions}, {"labels": "all", "areas": "all", "score_min": 1.1, "max_iou_with_v28": 0.0, "max_add_per_page": 0})
        metrics["config"] = config
        metrics["delta_vs_policy"] = {
            key: round(float(metrics[key]) - float(baseline[key]), 6)
            for key in ["iou_0_30_recall", "center_recall", "candidate_inflation", "precision"]
        }
        metrics["tiny_delta"] = round(metrics["per_area_iou_recall"].get("tiny_le_64", 0.0) - baseline["per_area_iou_recall"].get("tiny_le_64", 0.0), 6)
        metrics["sink_delta"] = round(metrics["per_label_iou_recall"].get("sink", 0.0) - baseline["per_label_iou_recall"].get("sink", 0.0), 6)
        metrics["shower_delta"] = round(metrics["per_label_iou_recall"].get("shower", 0.0) - baseline["per_label_iou_recall"].get("shower", 0.0), 6)
        metrics["equipment_delta"] = round(metrics["per_label_iou_recall"].get("equipment", 0.0) - baseline["per_label_iou_recall"].get("equipment", 0.0), 6)
        results.append(metrics)

    feasible = [item for item in results if item["delta_vs_policy"]["iou_0_30_recall"] >= 0 and item["delta_vs_policy"]["candidate_inflation"] <= 1.0 and item["delta_vs_policy"]["precision"] >= -0.015]
    feasible.sort(key=lambda item: (item["delta_vs_policy"]["iou_0_30_recall"], item["tiny_delta"], item["sink_delta"], -item["delta_vs_policy"]["candidate_inflation"]), reverse=True)
    top_all = sorted(results, key=lambda item: (item["delta_vs_policy"]["iou_0_30_recall"], item["tiny_delta"], -item["delta_vs_policy"]["candidate_inflation"]), reverse=True)[:20]
    best = feasible[0] if feasible else None
    if best:
        write_jsonl(Path(args.output_predictions), rows_from_predictions(repaired_predictions(policy_predictions, best["config"])))

    report = {
        "version": "symbol_center_only_box_repair_p067_smoke_v30",
        "source_integrity": "offline gold is used only for smoke sweep; runtime repair uses detector boxes/scores/labels from raster-derived predictions only",
        "inputs": {"policy_predictions": rel(Path(args.policy_predictions)), "data": rel(Path(args.data)), "yolo_dir": rel(Path(args.yolo_dir))},
        "baseline_policy": baseline,
        "best_feasible": best,
        "top_feasible": feasible[:20],
        "top_unconstrained": top_all,
        "sweep_count": len(results),
        "decision": "positive_smoke_candidate_validate_locked" if best and best["delta_vs_policy"]["iou_0_30_recall"] > 0 else "negative_do_not_apply_geometric_repair",
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# P0-67 center-only box repair smoke sweep",
        "",
        "## Summary",
        "",
        f"- baseline policy IoU / center / inflation / precision: `{baseline['iou_0_30_recall']:.6f}` / `{baseline['center_recall']:.6f}` / `{baseline['candidate_inflation']:.6f}` / `{baseline['precision']:.6f}`",
        f"- decision: `{report['decision']}`",
    ]
    if best:
        delta = best["delta_vs_policy"]
        lines.extend([
            f"- best IoU / center / inflation / precision: `{best['iou_0_30_recall']:.6f}` / `{best['center_recall']:.6f}` / `{best['candidate_inflation']:.6f}` / `{best['precision']:.6f}`",
            f"- delta IoU / center / inflation / precision: `{delta['iou_0_30_recall']:+.6f}` / `{delta['center_recall']:+.6f}` / `{delta['candidate_inflation']:+.6f}` / `{delta['precision']:+.6f}`",
            f"- tiny / sink / shower / equipment delta: `{best['tiny_delta']:+.6f}` / `{best['sink_delta']:+.6f}` / `{best['shower_delta']:+.6f}` / `{best['equipment_delta']:+.6f}`",
            f"- config: `{json.dumps(best['config'], ensure_ascii=False)}`",
        ])
    lines.extend(["", "## Artifacts", "", f"- `{rel(Path(args.output_json))}`", f"- `{rel(Path(args.output_md))}`", f"- `{rel(Path(args.output_predictions))}`", ""])
    Path(args.output_md).write_text("\n".join(lines))
    print(json.dumps({"decision": report["decision"], "baseline": {k: baseline[k] for k in ["iou_0_30_recall", "center_recall", "candidate_inflation", "precision"]}, "best": None if best is None else {k: best[k] for k in ["iou_0_30_recall", "center_recall", "candidate_inflation", "precision", "delta_vs_policy", "tiny_delta", "sink_delta", "shower_delta", "equipment_delta", "config"]}}, ensure_ascii=False, indent=2)[:6000])


if __name__ == "__main__":
    main()
