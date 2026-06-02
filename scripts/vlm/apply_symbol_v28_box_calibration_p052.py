#!/usr/bin/env python3
"""P0-52: runtime-safe frozen v28 box calibration smoke grid."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from eval_symbol_yolo_tile_detector_v22 import score_predictions
from train_symbol_tile_detector_v20 import load_jsonl, rel, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[2]


def load_predictions(path: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        out[str(row["row_id"])] = list(row.get("predicted_symbols") or [])
    return out


def load_page_golds(tile_jsonl: Path, row_ids: set[str]) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for row in load_jsonl(tile_jsonl):
        row_id = str(row.get("row_id"))
        if row_id not in row_ids:
            continue
        gold_map = out.setdefault(row_id, {})
        for gold in ((row.get("targets") or {}).get("boxes") or []):
            target_id = str(gold.get("target_id") or f"{row_id}_{len(gold_map)}")
            gold_map[target_id] = {
                "target_id": target_id,
                "bbox": [float(v) for v in gold.get("page_bbox") or gold.get("bbox")],
                "label": str(gold.get("label") or "generic_symbol"),
            }
    return out


def calibrate_box(box: list[float], scale: float, min_size: float) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = max(min_size, (x2 - x1) * scale)
    h = max(min_size, (y2 - y1) * scale)
    return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]


def calibrate_predictions(preds: dict[str, list[dict[str, Any]]], scale_map: dict[str, float], default_scale: float, min_size: float, score_floor: float) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row_id, items in preds.items():
        calibrated = []
        for pred in items:
            item = deepcopy(pred)
            if float(item.get("score", 0.0)) >= score_floor:
                scale = scale_map.get(str(item.get("label")), default_scale)
                item["bbox"] = calibrate_box([float(v) for v in item["bbox"]], scale, min_size)
                item["calibration_scale"] = scale
            calibrated.append(item)
        out[row_id] = calibrated
    return out


def parse_scale_map(raw: str) -> dict[str, float]:
    result: dict[str, float] = {}
    if not raw:
        return result
    for item in raw.split(","):
        if not item.strip():
            continue
        label, value = item.split(":", 1)
        result[label.strip()] = float(value)
    return result


def selection_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    metrics = row["metrics"]
    iou = metrics["symbol_bbox_iou_0_30"]
    return (
        float(iou["recall"]),
        float(metrics["symbol_bbox_center_recall"]),
        -abs(float(metrics["candidate_inflation"]) - 7.112284),
        float(iou["precision"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles", default="datasets/symbol_tile_detector_tiny_sahi_v21/smoke_v30.jsonl")
    parser.add_argument("--predictions", default="reports/vlm/symbol_yolov8s_seg_rect_v28_smoke_v30_page_predictions.jsonl")
    parser.add_argument("--output", default="reports/vlm/symbol_v28_box_calibration_p052_eval.json")
    parser.add_argument("--predictions-output", default="reports/vlm/symbol_v28_box_calibration_p052_predictions.jsonl")
    parser.add_argument("--default-scales", default="0.80,0.85,0.90,0.95,1.00,1.05")
    parser.add_argument("--focus-scale-overrides", default="sink:0.85,shower:0.85,equipment:0.90,stair:0.95")
    parser.add_argument("--score-floors", default="0.0,0.05,0.10")
    parser.add_argument("--nms-threshold", type=float, default=0.65)
    parser.add_argument("--score-threshold", type=float, default=0.005)
    parser.add_argument("--max-per-page", type=int, default=500)
    parser.add_argument("--min-size", type=float, default=4.0)
    parser.add_argument("--tile-count", type=int, default=200)
    args = parser.parse_args()

    raw_preds = load_predictions(Path(args.predictions))
    golds = load_page_golds(Path(args.tiles), set(raw_preds))
    scale_overrides = parse_scale_map(args.focus_scale_overrides)
    grid = []
    best_predictions: list[dict[str, Any]] = []
    for default_scale in [float(x) for x in args.default_scales.split(",") if x.strip()]:
        for score_floor in [float(x) for x in args.score_floors.split(",") if x.strip()]:
            calibrated = calibrate_predictions(raw_preds, scale_overrides, default_scale, args.min_size, score_floor)
            metrics, predictions = score_predictions(calibrated, golds, args.score_threshold, args.nms_threshold, args.max_per_page, tile_count=args.tile_count)
            row = {"default_scale": default_scale, "score_floor": score_floor, "scale_overrides": scale_overrides, "metrics": metrics}
            grid.append(row)
            if not best_predictions or selection_key(row) > selection_key(max(grid[:-1], key=selection_key)):
                best_predictions = predictions
    best = max(grid, key=selection_key)
    report = {
        "version": "symbol_v28_box_calibration_p052",
        "claim_boundary": "Runtime-safe frozen v28 box calibration over detector predictions; smoke_v30 only.",
        "source_integrity": {
            "runtime_inputs": ["v28 detector predictions", "static calibration config"],
            "gold_use": "offline grid evaluation only",
            "uses_svg_or_cad_geometry_at_runtime": False,
        },
        "inputs": {"tiles": rel(Path(args.tiles)), "predictions": rel(Path(args.predictions))},
        "reference_v28_smoke_v30": {
            "center_recall": 0.882917,
            "iou_0_30_recall": 0.71977,
            "precision": 0.101201,
            "candidate_inflation": 7.112284,
        },
        "config": vars(args),
        "grid": grid,
        "selected": {k: best[k] for k in ["default_scale", "score_floor", "scale_overrides"]},
        "smoke_v30": best["metrics"],
        "decision_gate": {
            "beats_v28_iou_recall": best["metrics"]["symbol_bbox_iou_0_30"]["recall"] > 0.71977,
            "keeps_center_recall": best["metrics"]["symbol_bbox_center_recall"] >= 0.882917,
            "inflation_lte_v28_plus_0_25": best["metrics"]["candidate_inflation"] <= 7.362284,
        },
    }
    report["decision_gate"]["passed"] = all(report["decision_gate"].values())
    write_json(Path(args.output), report)
    write_jsonl(Path(args.predictions_output), best_predictions)
    print(json.dumps({"selected": report["selected"], "smoke_v30": report["smoke_v30"], "gate": report["decision_gate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
