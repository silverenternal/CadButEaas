#!/usr/bin/env python3
"""Evaluate SAHI-style page slicing for the raster symbol body detector."""

from __future__ import annotations

import argparse
import json
import resource
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image
from ultralytics import YOLO

from eval_symbol_yolo_tile_detector_v22 import (
    BASELINE_CENTER_RECALL,
    BASELINE_CANDIDATE_INFLATION,
    BASELINE_TINY_IOU_RECALL,
    filter_rows_with_exported_images,
    memory_audit,
    sample_tiles_area_aware,
    score_predictions,
    selection_key,
)
from train_symbol_tile_detector_v20 import (
    FORBIDDEN_RUNTIME_FIELDS,
    ID_TO_LABEL,
    load_jsonl,
    rel,
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21"
YOLO_DIR = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21_yolo_v22"
WEIGHTS = ROOT / "runs/detect/runs/vlm/symbol_yolov8n_pretrained_v22_dedup_hi640_probe/weights/best.pt"
REPORT = ROOT / "reports/vlm/symbol_yolo_sliced_hi640_v24_page_eval.json"
PREDICTIONS = ROOT / "reports/vlm/symbol_yolo_sliced_hi640_v24_page_predictions.jsonl"


def center_in_box(box: list[float], container: list[float]) -> bool:
    cx = (float(box[0]) + float(box[2])) / 2.0
    cy = (float(box[1]) + float(box[3])) / 2.0
    return float(container[0]) <= cx <= float(container[2]) and float(container[1]) <= cy <= float(container[3])


def covered_by_sampled_tiles(box: list[float], coverage: list[list[float]]) -> bool:
    return any(center_in_box(box, tile_box) for tile_box in coverage)


def slice_origins(width: int, height: int, slice_size: int, stride: int) -> list[tuple[int, int, int, int]]:
    xs = list(range(0, max(width - slice_size, 0) + 1, stride))
    ys = list(range(0, max(height - slice_size, 0) + 1, stride))
    if not xs or xs[-1] + slice_size < width:
        xs.append(max(width - slice_size, 0))
    if not ys or ys[-1] + slice_size < height:
        ys.append(max(height - slice_size, 0))
    boxes: list[tuple[int, int, int, int]] = []
    for y in ys:
        for x in xs:
            boxes.append((x, y, min(x + slice_size, width), min(y + slice_size, height)))
    return boxes


def build_eval_contract(rows: list[dict[str, Any]]) -> tuple[dict[str, Path], dict[str, list[list[float]]], dict[str, dict[str, dict[str, Any]]]]:
    page_images: dict[str, Path] = {}
    coverage: dict[str, list[list[float]]] = defaultdict(list)
    page_golds: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        row_id = str(row.get("row_id"))
        page_images[row_id] = Path(str(row.get("image")))
        coverage[row_id].append([float(v) for v in (row.get("tile") or {}).get("bbox")])
        for gold in ((row.get("targets") or {}).get("boxes") or []):
            target_id = str(gold.get("target_id") or f"{row_id}_{len(page_golds[row_id])}")
            page_golds[row_id][target_id] = {
                "target_id": target_id,
                "bbox": [float(v) for v in gold.get("page_bbox") or gold.get("bbox")],
                "label": str(gold.get("label") or "generic_symbol"),
            }
    return page_images, coverage, page_golds


def collect_sliced_predictions(model: YOLO, page_images: dict[str, Path], coverage: dict[str, list[list[float]]], args: argparse.Namespace) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    page_preds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    audit = {
        "slice_size": int(args.slice_size),
        "slice_overlap": float(args.slice_overlap),
        "slice_stride": None,
        "total_slices": 0,
        "rows": len(page_images),
    }
    stride = max(1, int(round(args.slice_size * (1.0 - args.slice_overlap))))
    audit["slice_stride"] = stride
    batch_images: list[Image.Image] = []
    batch_meta: list[tuple[str, int, int, str]] = []

    def flush() -> None:
        nonlocal batch_images, batch_meta
        if not batch_images:
            return
        results = model.predict(
            source=batch_images,
            imgsz=args.imgsz,
            conf=args.decode_conf,
            iou=args.decode_iou,
            max_det=args.max_det_per_slice,
            device=args.device,
            batch=max(1, len(batch_images)),
            stream=False,
            verbose=False,
        )
        for (row_id, left, top, slice_id), result in zip(batch_meta, results, strict=True):
            if result.boxes is None:
                continue
            xyxy = result.boxes.xyxy.detach().cpu().tolist()
            confs = result.boxes.conf.detach().cpu().tolist()
            classes = result.boxes.cls.detach().cpu().tolist()
            for box, conf, cls in zip(xyxy, confs, classes, strict=True):
                label_id = int(cls) + 1
                if label_id not in ID_TO_LABEL:
                    continue
                page_box = [float(box[0] + left), float(box[1] + top), float(box[2] + left), float(box[3] + top)]
                if not covered_by_sampled_tiles(page_box, coverage[row_id]):
                    continue
                page_preds[row_id].append(
                    {
                        "bbox": page_box,
                        "label_id": label_id,
                        "label": ID_TO_LABEL[label_id],
                        "score": float(conf),
                        "tile_id": slice_id,
                    }
                )
        batch_images = []
        batch_meta = []

    for row_id, image_path in sorted(page_images.items()):
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            for index, (left, top, right, bottom) in enumerate(slice_origins(width, height, args.slice_size, stride)):
                if args.skip_slices_outside_sampled_tiles and not any(
                    right >= tile[0] and left <= tile[2] and bottom >= tile[1] and top <= tile[3] for tile in coverage[row_id]
                ):
                    continue
                batch_images.append(image.crop((left, top, right, bottom)))
                batch_meta.append((row_id, left, top, f"{row_id}_slice_{index}_{left}_{top}_{right}_{bottom}"))
                audit["total_slices"] += 1
                if len(batch_images) >= args.predict_batch:
                    flush()
        flush()
    return page_preds, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA))
    parser.add_argument("--yolo-dir", default=str(YOLO_DIR))
    parser.add_argument("--weights", default=str(WEIGHTS))
    parser.add_argument("--split", default="locked", choices=["dev", "locked", "smoke_v30"])
    parser.add_argument("--eval-output", default=str(REPORT))
    parser.add_argument("--predictions-output", default=str(PREDICTIONS))
    parser.add_argument("--limit-tiles", type=int, default=2000)
    parser.add_argument("--positive-ratio", type=float, default=0.85)
    parser.add_argument("--small-positive-ratio", type=float, default=0.75)
    parser.add_argument("--slice-size", type=int, default=384)
    parser.add_argument("--slice-overlap", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--decode-conf", type=float, default=0.001)
    parser.add_argument("--decode-iou", type=float, default=0.7)
    parser.add_argument("--max-det-per-slice", type=int, default=300)
    parser.add_argument("--predict-batch", type=int, default=8)
    parser.add_argument("--score-threshold-grid", default="0.001,0.005,0.01,0.02,0.05")
    parser.add_argument("--nms-threshold-grid", default="0.45,0.55,0.65,0.75")
    parser.add_argument("--max-per-page", type=int, default=500)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument("--skip-slices-outside-sampled-tiles", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    split_rows = load_jsonl(Path(args.data) / f"{args.split}.jsonl")
    if args.split.startswith("smoke"):
        exported_rows = split_rows
    else:
        exported_rows = filter_rows_with_exported_images(split_rows, args.split, Path(args.yolo_dir))
    rows = sample_tiles_area_aware(
        exported_rows,
        args.limit_tiles,
        args.seed + (2 if args.split == "locked" else 1),
        args.positive_ratio,
        args.small_positive_ratio,
    )
    if not rows:
        raise RuntimeError(f"no exported YOLO images found for split={args.split} under {args.yolo_dir}")
    page_images, coverage, page_golds = build_eval_contract(rows)
    model = YOLO(args.weights)
    page_preds, slicing_audit = collect_sliced_predictions(model, page_images, coverage, args)
    score_grid = [float(item) for item in args.score_threshold_grid.split(",") if item.strip()]
    nms_grid = [float(item) for item in args.nms_threshold_grid.split(",") if item.strip()]
    grid_reports: list[dict[str, Any]] = []
    for score_threshold in score_grid:
        for nms_threshold in nms_grid:
            metrics, _ = score_predictions(page_preds, page_golds, score_threshold, nms_threshold, args.max_per_page, slicing_audit["total_slices"])
            grid_reports.append({"score_threshold": score_threshold, "nms_threshold": nms_threshold, "metrics": metrics})
    grid_reports.sort(key=selection_key, reverse=True)
    selected = grid_reports[0]
    locked_eval, predictions = score_predictions(
        page_preds,
        page_golds,
        float(selected["score_threshold"]),
        float(selected["nms_threshold"]),
        args.max_per_page,
        slicing_audit["total_slices"],
    )
    report = {
        "version": "symbol_yolo_sliced_page_detector_v24",
        "claim_boundary": "SAHI-style sliced inference over raster page pixels, evaluated against the same page-level symbol metrics and sampled-tile coverage as the fixed-tile baseline.",
        "source_integrity": {
            "model_input": "raster_page_pixels_and_slice_coordinates_only",
            "raw_label_or_semantic_type_used_as_runtime_feature": False,
            "forbidden_runtime_features": FORBIDDEN_RUNTIME_FIELDS,
            "locked_gold_use": "evaluation_only",
        },
        "baseline_to_beat": {
            "adopted_v22_center_recall": 0.911595,
            "adopted_v22_iou_0_30_recall": 0.719572,
            "adopted_v22_tiny_iou_recall": 0.449555,
            "adopted_v22_missed_no_center_rate": 0.243421,
            "legacy_scaled_faster_rcnn_center_recall": BASELINE_CENTER_RECALL,
            "legacy_scaled_faster_rcnn_tiny_iou_recall": BASELINE_TINY_IOU_RECALL,
            "legacy_scaled_faster_rcnn_candidate_inflation": BASELINE_CANDIDATE_INFLATION,
        },
        "dataset": rel(Path(args.data)),
        "weights": rel(Path(args.weights)),
        "config": vars(args),
        "slicing_audit": slicing_audit,
        "threshold_grid": grid_reports,
        "selected_thresholds": {"score_threshold": float(selected["score_threshold"]), "nms_threshold": float(selected["nms_threshold"])},
        args.split: locked_eval,
        "gate": {
            "beats_adopted_v22_center_0_911595": locked_eval["symbol_bbox_center_recall"] > 0.911595,
            "beats_adopted_v22_iou_0_719572": locked_eval["symbol_bbox_iou_0_30"]["recall"] > 0.719572,
            "beats_adopted_v22_tiny_iou_0_449555": locked_eval["area_iou_recall"].get("tiny_le_64", 0.0) > 0.449555,
            "candidate_inflation_lte_15": locked_eval["candidate_inflation"] <= 15.0,
        },
        "memory_audit": memory_audit(),
    }
    report["gate"]["passed"] = all(bool(value) for value in report["gate"].values())
    write_json(Path(args.eval_output), report)
    write_jsonl(Path(args.predictions_output), predictions)
    print(json.dumps({args.split: locked_eval, "gate": report["gate"], "selected": report["selected_thresholds"], "slicing_audit": slicing_audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
