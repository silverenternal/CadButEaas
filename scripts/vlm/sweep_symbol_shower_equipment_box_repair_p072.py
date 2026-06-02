#!/usr/bin/env python3
"""Sweep shower/equipment-focused geometric box repairs after P0-70."""

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
from sweep_symbol_center_only_box_repair_p067 import scale_bbox, rows_from_predictions
from sweep_symbol_rtdetr_complement_gate_p063 import pred_area_bucket, read_predictions, score
from train_symbol_tile_detector_v20 import rel, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21"
DEFAULT_YOLO_DIR = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21_yolo_seg_rect_v27"
DEFAULT_POLICY = ROOT / "reports/vlm/symbol_precision_gated_policy_p070_smoke_v30_predictions.jsonl"
DEFAULT_JSON = ROOT / "reports/vlm/symbol_shower_equipment_box_repair_p072_smoke_v30.json"
DEFAULT_MD = ROOT / "reports/vlm/symbol_shower_equipment_box_repair_p072_smoke_v30.md"
DEFAULT_PRED = ROOT / "reports/vlm/symbol_shower_equipment_box_repair_p072_smoke_v30_predictions.jsonl"

LABEL_GROUPS = {
    "shower": {"shower"},
    "equipment": {"equipment"},
    "shower_equipment": {"shower", "equipment"},
}
AREA_GROUPS = {
    "tiny_small": {"tiny_le_64", "small_le_256"},
    "small_medium": {"small_le_256", "medium_le_1024"},
    "nonlarge": {"tiny_le_64", "small_le_256", "medium_le_1024"},
    "all": {"tiny_le_64", "small_le_256", "medium_le_1024", "large_le_4096", "xlarge_gt_4096"},
}


def repaired_predictions(page_predictions: dict[str, list[dict[str, Any]]], config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    labels = LABEL_GROUPS[str(config["labels"])]
    areas = AREA_GROUPS[str(config["areas"])]
    score_min = float(config["score_min"])
    scale_x = float(config["scale_x"])
    scale_y = float(config["scale_y"])
    max_add_per_page = int(config["max_add_per_page"])
    output = {}
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
            repaired["score"] = float(prediction.get("score", 0.0)) * 0.998
            repaired["source_policy"] = "p072_shower_equipment_box_repair"
            repaired["repair_scale_x"] = scale_x
            repaired["repair_scale_y"] = scale_y
            additions.append(repaired)
        additions.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        output[row_id] = list(predictions) + additions[:max_add_per_page]
    return output


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
    configs = [
        {"labels":"shower","areas":"all","score_min":0.10,"scale_x":0.8,"scale_y":0.8,"max_add_per_page":10},
        {"labels":"shower","areas":"all","score_min":0.20,"scale_x":0.8,"scale_y":0.8,"max_add_per_page":10},
        {"labels":"shower","areas":"all","score_min":0.10,"scale_x":0.65,"scale_y":0.65,"max_add_per_page":10},
        {"labels":"shower","areas":"nonlarge","score_min":0.10,"scale_x":0.8,"scale_y":0.8,"max_add_per_page":10},
        {"labels":"shower","areas":"small_medium","score_min":0.10,"scale_x":0.8,"scale_y":0.8,"max_add_per_page":10},
        {"labels":"equipment","areas":"all","score_min":0.10,"scale_x":0.8,"scale_y":0.8,"max_add_per_page":10},
        {"labels":"equipment","areas":"nonlarge","score_min":0.10,"scale_x":0.8,"scale_y":0.8,"max_add_per_page":10},
        {"labels":"equipment","areas":"tiny_small","score_min":0.10,"scale_x":0.8,"scale_y":0.8,"max_add_per_page":10},
        {"labels":"shower_equipment","areas":"all","score_min":0.10,"scale_x":0.8,"scale_y":0.8,"max_add_per_page":10},
        {"labels":"shower_equipment","areas":"nonlarge","score_min":0.10,"scale_x":0.8,"scale_y":0.8,"max_add_per_page":10},
        {"labels":"shower_equipment","areas":"all","score_min":0.10,"scale_x":1.1,"scale_y":1.1,"max_add_per_page":10},
        {"labels":"shower_equipment","areas":"all","score_min":0.10,"scale_x":0.75,"scale_y":1.25,"max_add_per_page":10},
        {"labels":"shower_equipment","areas":"all","score_min":0.10,"scale_x":1.25,"scale_y":0.75,"max_add_per_page":10},
    ]
    results=[]
    for config in configs:
        candidate = repaired_predictions(policy_predictions, config)
        metrics = score(golds, candidate, {row_id: [] for row_id in candidate}, {"labels":"all","areas":"all","score_min":1.1,"max_iou_with_v28":0.0,"max_add_per_page":0})
        metrics["config"] = config
        metrics["delta_vs_p070"] = {k: round(metrics[k]-baseline[k], 6) for k in ["iou_0_30_recall","center_recall","candidate_inflation","precision"]}
        metrics["shower_delta"] = round(metrics["per_label_iou_recall"].get("shower",0.0)-baseline["per_label_iou_recall"].get("shower",0.0),6)
        metrics["equipment_delta"] = round(metrics["per_label_iou_recall"].get("equipment",0.0)-baseline["per_label_iou_recall"].get("equipment",0.0),6)
        results.append(metrics)
    feasible=[m for m in results if m["delta_vs_p070"]["iou_0_30_recall"]>=0 and m["delta_vs_p070"]["candidate_inflation"]<=0.75 and m["delta_vs_p070"]["precision"]>=-0.01 and (m["shower_delta"]>0 or m["equipment_delta"]>0)]
    feasible.sort(key=lambda m:(m["shower_delta"]+m["equipment_delta"], m["delta_vs_p070"]["iou_0_30_recall"], -m["delta_vs_p070"]["candidate_inflation"]), reverse=True)
    best=feasible[0] if feasible else None
    if best:
        write_jsonl(Path(args.output_predictions), rows_from_predictions(repaired_predictions(policy_predictions, best["config"])))
    report={
        "version":"symbol_shower_equipment_box_repair_p072_smoke_v30",
        "source_integrity":"offline gold is used only for smoke sweep; runtime repair uses raster-derived prediction fields only",
        "inputs":{"policy_predictions":rel(Path(args.policy_predictions)),"data":rel(Path(args.data)),"yolo_dir":rel(Path(args.yolo_dir))},
        "baseline_p070":baseline,
        "best_feasible":best,
        "top_feasible":feasible[:20],
        "top_unconstrained":sorted(results,key=lambda m:(m["delta_vs_p070"]["iou_0_30_recall"],m["shower_delta"]+m["equipment_delta"]),reverse=True)[:20],
        "sweep_count":len(results),
        "decision":"positive_smoke_candidate_validate_locked" if best and best["delta_vs_p070"]["iou_0_30_recall"]>0 else "negative_no_safe_shower_equipment_repair",
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    lines=["# P0-72 shower/equipment box repair smoke sweep","","## Summary","",f"- baseline P0-70 IoU / inflation / precision: `{baseline['iou_0_30_recall']:.6f}` / `{baseline['candidate_inflation']:.6f}` / `{baseline['precision']:.6f}`",f"- decision: `{report['decision']}`"]
    if best:
        d=best["delta_vs_p070"]
        lines += [f"- best IoU / inflation / precision: `{best['iou_0_30_recall']:.6f}` / `{best['candidate_inflation']:.6f}` / `{best['precision']:.6f}`",f"- delta IoU / inflation / precision: `{d['iou_0_30_recall']:+.6f}` / `{d['candidate_inflation']:+.6f}` / `{d['precision']:+.6f}`",f"- shower / equipment delta: `{best['shower_delta']:+.6f}` / `{best['equipment_delta']:+.6f}`",f"- config: `{json.dumps(best['config'],ensure_ascii=False)}`"]
    lines += ["","## Artifacts","",f"- `{rel(Path(args.output_json))}`",f"- `{rel(Path(args.output_md))}`",f"- `{rel(Path(args.output_predictions))}`",""]
    Path(args.output_md).write_text("\n".join(lines))
    print(json.dumps({"decision":report["decision"],"baseline":{k:baseline[k] for k in ["iou_0_30_recall","candidate_inflation","precision"]},"best":None if best is None else {k:best[k] for k in ["iou_0_30_recall","candidate_inflation","precision","delta_vs_p070","shower_delta","equipment_delta","config"]}},ensure_ascii=False,indent=2)[:6000])

if __name__ == "__main__":
    main()
