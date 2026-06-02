#!/usr/bin/env python3
"""Probe RTDETR shower/stair no-center recovery on top of P0-70."""

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
DEFAULT_BASE = ROOT / "reports/vlm/symbol_precision_gated_policy_p070_smoke_v30_predictions.jsonl"
DEFAULT_RTDETR = ROOT / "reports/vlm/symbol_rtdetr_l_bbox_p061_smoke4000_v3_smoke_v30_page_predictions.jsonl"
DEFAULT_JSON = ROOT / "reports/vlm/symbol_shower_stair_no_center_recovery_p073_smoke_v30.json"
DEFAULT_MD = ROOT / "reports/vlm/symbol_shower_stair_no_center_recovery_p073_smoke_v30.md"
DEFAULT_PRED = ROOT / "reports/vlm/symbol_shower_stair_no_center_recovery_p073_smoke_v30_predictions.jsonl"

LABEL_GROUPS = {
    "shower": {"shower"},
    "stair": {"stair"},
    "shower_stair": {"shower", "stair"},
}
AREA_GROUPS = {
    "small": {"small_le_256"},
    "tiny_small": {"tiny_le_64", "small_le_256"},
    "small_medium": {"small_le_256", "medium_le_1024"},
    "nonlarge": {"tiny_le_64", "small_le_256", "medium_le_1024"},
    "all": {"tiny_le_64", "small_le_256", "medium_le_1024", "large_le_4096", "xlarge_gt_4096"},
}


def max_overlap(prediction: dict[str, Any], refs: list[dict[str, Any]]) -> float:
    box = [float(v) for v in prediction["bbox"]]
    return max((bbox_iou(box, [float(x) for x in ref["bbox"]]) for ref in refs), default=0.0)


def add_candidates(base: dict[str, list[dict[str, Any]]], rtdetr: dict[str, list[dict[str, Any]]], config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    labels = LABEL_GROUPS[str(config["labels"])]
    areas = AREA_GROUPS[str(config["areas"])]
    score_min = float(config["score_min"])
    min_iou = float(config["min_iou_with_base"])
    max_iou = float(config["max_iou_with_base"])
    max_add = int(config["max_add_per_page"])
    output = {}
    for row_id in sorted(set(base) | set(rtdetr)):
        base_items = list(base.get(row_id, []))
        additions = []
        for pred in rtdetr.get(row_id, []):
            if str(pred.get("label")) not in labels:
                continue
            if pred_area_bucket(pred) not in areas:
                continue
            if float(pred.get("score", 0.0)) < score_min:
                continue
            overlap = max_overlap(pred, base_items)
            if overlap < min_iou or overlap >= max_iou:
                continue
            item = dict(pred)
            item["source_policy"] = "p073_rtdetr_shower_stair_recovery"
            additions.append(item)
        additions.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        output[row_id] = base_items + additions[:max_add]
    return output


def rows_from_predictions(preds: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [{"row_id": row_id, "predicted_symbols": items} for row_id, items in sorted(preds.items())]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--yolo-dir", default=str(DEFAULT_YOLO_DIR))
    parser.add_argument("--split", default="smoke_v30")
    parser.add_argument("--base-predictions", default=str(DEFAULT_BASE))
    parser.add_argument("--rtdetr-predictions", default=str(DEFAULT_RTDETR))
    parser.add_argument("--output-json", default=str(DEFAULT_JSON))
    parser.add_argument("--output-md", default=str(DEFAULT_MD))
    parser.add_argument("--output-predictions", default=str(DEFAULT_PRED))
    args = parser.parse_args()

    base = read_predictions(Path(args.base_predictions))
    rtdetr = read_predictions(Path(args.rtdetr_predictions))
    golds = read_exported_golds(Path(args.data), Path(args.yolo_dir), args.split, set(base) | set(rtdetr))
    baseline = score(golds, base, {row_id: [] for row_id in base}, {"labels":"all","areas":"all","score_min":1.1,"max_iou_with_v28":0.0,"max_add_per_page":0})
    configs = [
        {"labels":"shower","areas":"small_medium","score_min":0.10,"min_iou_with_base":0.0,"max_iou_with_base":1.01,"max_add_per_page":3},
        {"labels":"shower","areas":"small_medium","score_min":0.20,"min_iou_with_base":0.0,"max_iou_with_base":1.01,"max_add_per_page":3},
        {"labels":"shower","areas":"nonlarge","score_min":0.10,"min_iou_with_base":0.0,"max_iou_with_base":1.01,"max_add_per_page":3},
        {"labels":"stair","areas":"small_medium","score_min":0.10,"min_iou_with_base":0.0,"max_iou_with_base":1.01,"max_add_per_page":3},
        {"labels":"stair","areas":"small_medium","score_min":0.20,"min_iou_with_base":0.0,"max_iou_with_base":1.01,"max_add_per_page":3},
        {"labels":"stair","areas":"nonlarge","score_min":0.10,"min_iou_with_base":0.0,"max_iou_with_base":1.01,"max_add_per_page":3},
        {"labels":"shower_stair","areas":"small_medium","score_min":0.10,"min_iou_with_base":0.0,"max_iou_with_base":1.01,"max_add_per_page":3},
        {"labels":"shower_stair","areas":"nonlarge","score_min":0.10,"min_iou_with_base":0.0,"max_iou_with_base":1.01,"max_add_per_page":3},
        {"labels":"shower_stair","areas":"small_medium","score_min":0.10,"min_iou_with_base":0.01,"max_iou_with_base":1.01,"max_add_per_page":3},
        {"labels":"shower_stair","areas":"small_medium","score_min":0.10,"min_iou_with_base":0.0,"max_iou_with_base":0.50,"max_add_per_page":3},
        {"labels":"shower_stair","areas":"all","score_min":0.20,"min_iou_with_base":0.0,"max_iou_with_base":1.01,"max_add_per_page":1},
    ]
    results = []
    for config in configs:
        candidate = add_candidates(base, rtdetr, config)
        metrics = score(golds, candidate, {row_id: [] for row_id in candidate}, {"labels":"all","areas":"all","score_min":1.1,"max_iou_with_v28":0.0,"max_add_per_page":0})
        metrics["config"] = config
        metrics["delta_vs_p070"] = {k: round(metrics[k]-baseline[k], 6) for k in ["iou_0_30_recall","center_recall","candidate_inflation","precision"]}
        metrics["shower_delta"] = round(metrics["per_label_iou_recall"].get("shower",0.0)-baseline["per_label_iou_recall"].get("shower",0.0),6)
        metrics["stair_delta"] = round(metrics["per_label_iou_recall"].get("stair",0.0)-baseline["per_label_iou_recall"].get("stair",0.0),6)
        results.append(metrics)
    feasible = [m for m in results if m["delta_vs_p070"]["iou_0_30_recall"] > 0 and m["delta_vs_p070"]["candidate_inflation"] <= 0.5 and m["delta_vs_p070"]["precision"] >= -0.003 and (m["shower_delta"] > 0 or m["stair_delta"] > 0)]
    feasible.sort(key=lambda m:(m["delta_vs_p070"]["iou_0_30_recall"], m["shower_delta"]+m["stair_delta"], -m["delta_vs_p070"]["candidate_inflation"]), reverse=True)
    best = feasible[0] if feasible else None
    if best:
        write_jsonl(Path(args.output_predictions), rows_from_predictions(add_candidates(base, rtdetr, best["config"])))
    report = {
        "version":"symbol_shower_stair_no_center_recovery_p073_smoke_v30",
        "source_integrity":"offline gold is used only for smoke probe; runtime additions use raster-derived RTDETR prediction fields only",
        "inputs":{"base_predictions":rel(Path(args.base_predictions)),"rtdetr_predictions":rel(Path(args.rtdetr_predictions)),"data":rel(Path(args.data)),"yolo_dir":rel(Path(args.yolo_dir))},
        "baseline_p070":baseline,
        "best_feasible":best,
        "top_feasible":feasible[:20],
        "top_unconstrained":sorted(results,key=lambda m:(m["delta_vs_p070"]["iou_0_30_recall"],m["shower_delta"]+m["stair_delta"]),reverse=True)[:20],
        "sweep_count":len(results),
        "decision":"positive_smoke_candidate_validate_locked" if best else "negative_no_safe_no_center_recovery",
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    lines=["# P0-73 shower/stair no-center recovery smoke probe","","## Summary","",f"- baseline P0-70 IoU / inflation / precision: `{baseline['iou_0_30_recall']:.6f}` / `{baseline['candidate_inflation']:.6f}` / `{baseline['precision']:.6f}`",f"- decision: `{report['decision']}`"]
    if best:
        d=best["delta_vs_p070"]
        lines += [f"- best IoU / inflation / precision: `{best['iou_0_30_recall']:.6f}` / `{best['candidate_inflation']:.6f}` / `{best['precision']:.6f}`",f"- delta IoU / inflation / precision: `{d['iou_0_30_recall']:+.6f}` / `{d['candidate_inflation']:+.6f}` / `{d['precision']:+.6f}`",f"- shower / stair delta: `{best['shower_delta']:+.6f}` / `{best['stair_delta']:+.6f}`",f"- config: `{json.dumps(best['config'],ensure_ascii=False)}`"]
    lines += ["","## Artifacts","",f"- `{rel(Path(args.output_json))}`",f"- `{rel(Path(args.output_md))}`",f"- `{rel(Path(args.output_predictions))}`",""]
    Path(args.output_md).write_text("\n".join(lines))
    print(json.dumps({"decision":report["decision"],"baseline":{k:baseline[k] for k in ["iou_0_30_recall","candidate_inflation","precision"]},"best":None if best is None else {k:best[k] for k in ["iou_0_30_recall","candidate_inflation","precision","delta_vs_p070","shower_delta","stair_delta","config"]}},ensure_ascii=False,indent=2)[:6000])

if __name__ == "__main__":
    main()
