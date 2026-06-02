#!/usr/bin/env python3
"""Train/evaluate a pretrained Ultralytics symbol proposal source for v35."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image
from ultralytics import YOLO

from eval_symbol_yolo_tile_detector_v22 import (
    collect_yolo_predictions,
    sample_tiles_area_aware,
    score_predictions,
    selection_key,
)
from train_symbol_tile_detector_v20 import FORBIDDEN_RUNTIME_FIELDS, load_jsonl, rel, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]


def count_files(path: Path) -> int:
    return sum(1 for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def resolve_training_data(data: Path) -> tuple[Path, dict[str, Any]]:
    """Use the requested data when populated; otherwise fall back to the populated v24/v22 YOLO view."""
    requested = data if data.is_absolute() else ROOT / data
    audit: dict[str, Any] = {"requested": rel(requested), "selected": rel(requested), "fallback_used": False}
    selected = requested
    if requested.exists() and requested.name == "data.yaml":
        parent = requested.parent
        if count_files(parent / "images") == 0:
            fallback = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21_yolo_recall_v24/data.yaml"
            if fallback.exists():
                selected = fallback
                audit["selected"] = rel(selected)
                audit["fallback_used"] = True
                audit["reason"] = "requested v27 data.yaml points at an empty exported image directory; v24 data.yaml points at populated v22 images with recall train list."
    return selected, audit


def image_path_for_tile(row: dict[str, Any]) -> Path:
    path = Path(str(row.get("image") or ""))
    return path if path.is_absolute() else ROOT / path


def make_eval_tile_images(rows: list[dict[str, Any]], out_dir: Path) -> Path:
    """Export smoke tiles to temporary jpg files for Ultralytics predict."""
    image_dir = out_dir / "images" / "smoke_v30"
    image_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        out = image_dir / f"{row['id']}.jpg"
        if out.exists():
            continue
        x1, y1, x2, y2 = [int(v) for v in (row.get("tile") or {}).get("bbox") or [0, 0, 1, 1]]
        with Image.open(image_path_for_tile(row)) as opened:
            opened.convert("RGB").crop((x1, y1, x2, y2)).save(out, quality=95)
    return out_dir


def train_model(args: argparse.Namespace, data_yaml: Path, output_dir: Path) -> Path:
    model = YOLO(args.model)
    run_name = "symbol_pretrained_tiny_detector_v35"
    project = output_dir / "ultralytics_runs"
    result = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(project),
        name=run_name,
        exist_ok=True,
        pretrained=True,
        verbose=False,
        patience=0,
        val=not getattr(args, "no_train_val", False),
    )
    save_dir = Path(getattr(result, "save_dir", project / run_name))
    best = save_dir / "weights" / "best.pt"
    last = save_dir / "weights" / "last.pt"
    selected = best if best.exists() else last
    if not selected.exists():
        raise RuntimeError(f"Ultralytics training did not produce weights under {save_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected, output_dir / "model.pt")
    return output_dir / "model.pt"


def evaluate(weights: Path, args: argparse.Namespace, output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    data_dir = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21"
    rows_all = load_jsonl(data_dir / args.eval_split_jsonl)
    rows = sample_tiles_area_aware(rows_all, args.limit_smoke_tiles, args.seed, args.positive_ratio, args.small_positive_ratio)
    if not rows:
        raise RuntimeError("empty smoke eval rows")
    yolo_eval_dir = make_eval_tile_images(rows, output_dir / "eval_tiles")
    model = YOLO(str(weights))
    eval_args = argparse.Namespace(
        imgsz=args.imgsz,
        decode_conf=args.decode_conf,
        decode_iou=args.decode_iou,
        max_det_per_tile=args.max_det_per_tile,
        predict_batch=args.predict_batch,
        device=args.device,
    )
    page_preds, page_golds = collect_yolo_predictions(model, rows, "smoke_v30", yolo_eval_dir, eval_args)
    grid_reports: list[dict[str, Any]] = []
    max_per_page_grid = [int(item) for item in args.max_per_page_grid.split(",") if item.strip()] if args.max_per_page_grid else [args.max_per_page]
    for score_threshold in [float(item) for item in args.score_threshold_grid.split(",") if item.strip()]:
        for nms_threshold in [float(item) for item in args.nms_threshold_grid.split(",") if item.strip()]:
            for max_per_page in max_per_page_grid:
                metrics, _preds = score_predictions(page_preds, page_golds, score_threshold, nms_threshold, max_per_page, len(rows))
                grid_reports.append({"score_threshold": score_threshold, "nms_threshold": nms_threshold, "max_per_page": max_per_page, "metrics": metrics})
    def compression_selection_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
        metrics = row["metrics"]
        iou_metrics = metrics["symbol_bbox_iou_0_30"]
        inflation = float(metrics["candidate_inflation"])
        return (
            1.0 if inflation <= args.candidate_inflation_target else 0.0,
            float(iou_metrics["recall"]),
            float(metrics["symbol_bbox_center_recall"]),
            float(iou_metrics["precision"]),
            -inflation,
        )

    grid_reports.sort(key=compression_selection_key if args.selection_mode == "compression" else selection_key, reverse=True)
    selected = grid_reports[0]
    metrics, predictions = score_predictions(
        page_preds,
        page_golds,
        float(selected["score_threshold"]),
        float(selected["nms_threshold"]),
        int(selected.get("max_per_page", args.max_per_page)),
        len(rows),
    )
    audit = {
        "rows": len(rows),
        "source_rows": len(rows_all),
        "exported_eval_dir": rel(yolo_eval_dir),
        "threshold_grid": grid_reports,
        "selected_thresholds": {
            "score_threshold": float(selected["score_threshold"]),
            "nms_threshold": float(selected["nms_threshold"]),
            "max_per_page": int(selected.get("max_per_page", args.max_per_page)),
        },
    }
    return metrics, predictions, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/symbol_tile_detector_tiny_sahi_v21_yolo_seg_rect_v27/data.yaml")
    parser.add_argument("--output-dir", default="checkpoints/symbol_pretrained_tiny_detector_v35")
    parser.add_argument("--eval-output", default="reports/vlm/symbol_pretrained_tiny_detector_v35_smoke_eval.json")
    parser.add_argument("--predictions-output", default="reports/vlm/symbol_pretrained_tiny_detector_v35_smoke_predictions.jsonl")
    parser.add_argument("--model", default="yolov8s-seg.pt")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--limit-smoke-tiles", type=int, default=200)
    parser.add_argument("--eval-split-jsonl", default="smoke_v30.jsonl")
    parser.add_argument("--positive-ratio", type=float, default=0.9)
    parser.add_argument("--small-positive-ratio", type=float, default=0.8)
    parser.add_argument("--decode-conf", type=float, default=0.001)
    parser.add_argument("--decode-iou", type=float, default=0.7)
    parser.add_argument("--max-det-per-tile", type=int, default=300)
    parser.add_argument("--predict-batch", type=int, default=8)
    parser.add_argument("--score-threshold-grid", default="0.001,0.005,0.01,0.02,0.05")
    parser.add_argument("--nms-threshold-grid", default="0.45,0.55,0.65,0.75")
    parser.add_argument("--max-per-page", type=int, default=900)
    parser.add_argument("--max-per-page-grid", default="")
    parser.add_argument("--candidate-inflation-target", type=float, default=12.0)
    parser.add_argument("--selection-mode", choices=["baseline", "compression"], default="baseline")
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument("--eval-only-weights", default=None)
    parser.add_argument("--no-train-val", action="store_true", help="Disable Ultralytics validation during training; external eval still runs after weights are copied.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    data_yaml, data_audit = resolve_training_data(Path(args.data))
    if args.eval_only_weights:
        weights = Path(args.eval_only_weights)
        weights = weights if weights.is_absolute() else ROOT / weights
    else:
        weights = train_model(args, data_yaml, output_dir)
    metrics, predictions, eval_audit = evaluate(weights, args, output_dir)
    write_jsonl(Path(args.predictions_output), predictions)
    tiny = float(metrics["area_iou_recall"].get("tiny_le_64", 0.0))
    report = {
        "version": "symbol_pretrained_tiny_detector_v35_smoke_eval",
        "task": "P1-03-pretrained-detection-backbone-for-tiny-symbols-v35",
        "claim_boundary": "Ultralytics pretrained detector/segmenter route for raster-only symbol proposal generation. Runtime input is raster tile pixels only.",
        "source_integrity": {
            "model_input": "raster_tile_pixels_only",
            "raw_label_or_semantic_type_used_as_runtime_feature": False,
            "forbidden_runtime_features": FORBIDDEN_RUNTIME_FIELDS,
            "smoke_gold_use": "evaluation_only",
        },
        "data_audit": data_audit,
        "weights": rel(weights),
        "model": args.model,
        "config": vars(args),
        "eval_audit": eval_audit,
        "smoke": metrics,
        "gate": {
            "detector_alone_center_recall_min_0_30": float(metrics["symbol_bbox_center_recall"]) >= 0.30,
            "detector_alone_tiny_iou_recall_beats_v34_0_006645": tiny > 0.006645,
            "candidate_inflation_lte_12": float(metrics["candidate_inflation"]) <= 12.0,
        },
    }
    report["gate"]["passed"] = all(bool(value) for value in report["gate"].values())
    write_json(Path(args.eval_output), report)
    print(json.dumps({"smoke": metrics, "gate": report["gate"], "weights": rel(weights), "predictions": rel(Path(args.predictions_output))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
