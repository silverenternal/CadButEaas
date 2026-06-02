#!/usr/bin/env python3
"""Run YOLO symbol detector on the P101/P182 overlay page set only.

This keeps the GPU evaluation focused on the 74 official overlay rows used by
P165-P182/P101, then emits page-level detector predictions that can be fused by
fuse_symbol_detector_with_p182_p186.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ultralytics import RTDETR, YOLO

from eval_symbol_yolo_tile_detector_v22 import (
    collect_yolo_predictions,
    score_predictions,
    selection_key,
)
from train_symbol_tile_detector_v20 import FORBIDDEN_RUNTIME_FIELDS, load_jsonl, rel, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21"
DEFAULT_YOLO_DIR = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21_yolo_seg_rect_v27"
DEFAULT_BASE = ROOT / "reports/vlm/symbol_policy_moe_overlay_p182_best.jsonl"


def base_row_ids(path: Path) -> set[str]:
    rows = load_jsonl(path)
    return {str(row.get("row_id") or row.get("id")) for row in rows}


def p101_rows(data: Path, split: str, base_overlay: Path) -> list[dict[str, Any]]:
    wanted = base_row_ids(base_overlay)
    rows = load_jsonl(data / f"{split}.jsonl")
    out = [row for row in rows if str(row.get("row_id") or row.get("id")) in wanted]
    if not out:
        raise RuntimeError(f"no {split} tiles overlap {base_overlay}")
    return out


def compression_selection_key(row: dict[str, Any], target: float) -> tuple[float, float, float, float, float]:
    metrics = row["metrics"]
    iou = metrics["symbol_bbox_iou_0_30"]
    inflation = float(metrics["candidate_inflation"])
    return (
        1.0 if inflation <= target else 0.0,
        float(iou["f1"]),
        float(iou["precision"]),
        float(iou["recall"]),
        -inflation,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--yolo-dir", default=str(DEFAULT_YOLO_DIR))
    parser.add_argument("--base-overlay", default=str(DEFAULT_BASE))
    parser.add_argument("--weights", required=True)
    parser.add_argument("--split", default="locked", choices=["locked", "dev"])
    parser.add_argument("--eval-output", required=True)
    parser.add_argument("--predictions-output", required=True)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--decode-conf", type=float, default=0.001)
    parser.add_argument("--decode-iou", type=float, default=0.7)
    parser.add_argument("--max-det-per-tile", type=int, default=300)
    parser.add_argument("--predict-batch", type=int, default=16)
    parser.add_argument("--score-threshold-grid", default="0.001,0.003,0.005,0.01,0.02,0.05,0.10,0.15,0.20,0.30,0.45,0.60")
    parser.add_argument("--nms-threshold-grid", default="0.35,0.45,0.55,0.65,0.75")
    parser.add_argument("--max-per-page-grid", default="24,32,48,64,96,128,192,256")
    parser.add_argument("--selection-mode", choices=["balanced_f1", "compression"], default="balanced_f1")
    parser.add_argument("--candidate-inflation-target", type=float, default=1.0)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    data = Path(args.data)
    yolo_dir = Path(args.yolo_dir)
    base_overlay = Path(args.base_overlay)
    rows = p101_rows(data, args.split, base_overlay)
    model = RTDETR(args.weights) if "rtdetr" in str(args.weights).lower() else YOLO(args.weights)
    page_preds, page_golds = collect_yolo_predictions(model, rows, args.split, yolo_dir, args)

    grid_reports: list[dict[str, Any]] = []
    for score_threshold in [float(x) for x in args.score_threshold_grid.split(",") if x.strip()]:
        for nms_threshold in [float(x) for x in args.nms_threshold_grid.split(",") if x.strip()]:
            for max_per_page in [int(x) for x in args.max_per_page_grid.split(",") if x.strip()]:
                metrics, _ = score_predictions(page_preds, page_golds, score_threshold, nms_threshold, max_per_page, len(rows))
                grid_reports.append({"score_threshold": score_threshold, "nms_threshold": nms_threshold, "max_per_page": max_per_page, "metrics": metrics})
    if args.selection_mode == "compression":
        grid_reports.sort(key=lambda row: compression_selection_key(row, args.candidate_inflation_target), reverse=True)
    else:
        grid_reports.sort(key=lambda row: selection_key(row, "balanced_f1"), reverse=True)
    selected = grid_reports[0]
    metrics, predictions = score_predictions(page_preds, page_golds, float(selected["score_threshold"]), float(selected["nms_threshold"]), int(selected["max_per_page"]), len(rows))
    report = {
        "version": "symbol_yolo_p101_tiles_p186",
        "claim_boundary": "P101-only detector inference over raster tiles. Gold labels are used only for offline threshold selection/evaluation; output must be confirmed via P186/P101 overlay before paper use.",
        "source_integrity": {
            "model_input": "raster_tile_pixels_only",
            "raw_label_or_semantic_type_used_as_runtime_feature": False,
            "forbidden_runtime_features": FORBIDDEN_RUNTIME_FIELDS,
            "locked_gold_use": "evaluation_only",
        },
        "dataset": rel(data),
        "yolo_dir": rel(yolo_dir),
        "base_overlay": rel(base_overlay),
        "weights": rel(Path(args.weights)),
        "config": vars(args),
        "p101_tile_count": len(rows),
        "p101_page_count": len(page_golds),
        "threshold_grid": grid_reports,
        "selection_mode": args.selection_mode,
        "selected_thresholds": {"score_threshold": selected["score_threshold"], "nms_threshold": selected["nms_threshold"], "max_per_page": selected["max_per_page"]},
        args.split: metrics,
    }
    write_json(Path(args.eval_output), report)
    write_jsonl(Path(args.predictions_output), predictions)
    print(json.dumps({"selected": report["selected_thresholds"], args.split: metrics, "predictions": args.predictions_output}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
