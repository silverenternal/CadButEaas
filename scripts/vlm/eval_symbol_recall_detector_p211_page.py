#!/usr/bin/env python3
"""Evaluate P211 YOLO crop detector as page-level symbol proposals.

P211 training data writes cropped tile images with a sampled suffix, e.g.
``cubicasa5k_locked_00000_s384_tile_0_576_384_960_000017.jpg``.  This
script strips the suffix, joins back to the v21 tile jsonl contract, translates
YOLO crop predictions into page coordinates, and scores them with the same
page-level proposal metrics used by the earlier symbol audits.
"""
from __future__ import annotations

import argparse
import json
import resource
from collections import defaultdict
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from eval_symbol_yolo_tile_detector_v22 import score_predictions, selection_key
from train_symbol_tile_detector_v20 import (
    FORBIDDEN_RUNTIME_FIELDS,
    ID_TO_LABEL,
    load_jsonl,
    rel,
    write_json,
    write_jsonl,
)

ROOT = Path(__file__).resolve().parents[2]
V21 = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21"
YOLO_DIR = ROOT / "datasets/symbol_recall_detector_p211_yolo_20k_server"
WEIGHTS = ROOT / "checkpoints/symbol_recall_detector_p211_20k_yolov8s/model.pt"
REPORT = ROOT / "reports/vlm/symbol_recall_detector_p211_20k_yolov8s_page_eval.json"
PREDICTIONS = ROOT / "reports/vlm/symbol_recall_detector_p211_20k_yolov8s_page_predictions.jsonl"


def strip_sample_suffix(path: str) -> str:
    stem = Path(path).stem
    head, sep, tail = stem.rpartition("_")
    if sep and tail.isdigit() and len(tail) == 6:
        return head
    return stem


def load_rows_by_id(v21_dir: Path, split: str) -> dict[str, dict[str, Any]]:
    return {str(row.get("id")): row for row in load_jsonl(v21_dir / f"{split}.jsonl")}


def load_eval_items(yolo_dir: Path, v21_dir: Path, split: str, limit: int | None) -> list[tuple[Path, dict[str, Any]]]:
    list_path = yolo_dir / f"{split}.txt"
    if not list_path.exists():
        raise FileNotFoundError(list_path)
    rows_by_id = load_rows_by_id(v21_dir, split)
    items: list[tuple[Path, dict[str, Any]]] = []
    missing = 0
    for line in list_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        image_path = Path(line.strip())
        tile_id = strip_sample_suffix(str(image_path))
        row = rows_by_id.get(tile_id)
        if row is None:
            missing += 1
            continue
        items.append((image_path, row))
        if limit and len(items) >= limit:
            break
    if not items:
        raise RuntimeError(f"no P211 eval items found for split={split}; missing={missing}")
    return items


def collect_golds(items: list[tuple[Path, dict[str, Any]]]) -> dict[str, dict[str, dict[str, Any]]]:
    page_golds: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for _image_path, row in items:
        row_id = str(row.get("row_id"))
        for gold in ((row.get("targets") or {}).get("boxes") or []):
            target_id = str(gold.get("target_id") or f"{row_id}_{len(page_golds[row_id])}")
            page_golds[row_id][target_id] = {
                "target_id": target_id,
                "bbox": [float(v) for v in gold.get("page_bbox") or gold.get("bbox")],
                "label": str(gold.get("label") or "generic_symbol"),
            }
    return page_golds


def collect_predictions(model: YOLO, items: list[tuple[Path, dict[str, Any]]], args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    page_preds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    batch_size = max(1, int(args.predict_batch))
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        paths = [str(path) for path, _row in chunk]
        results = model.predict(
            source=paths,
            imgsz=args.imgsz,
            conf=args.decode_conf,
            iou=args.decode_iou,
            max_det=args.max_det_per_tile,
            device=args.device,
            batch=batch_size,
            stream=False,
            verbose=False,
        )
        for (_path, row), result in zip(chunk, results, strict=True):
            left, top, _right, _bottom = [float(v) for v in (row.get("tile") or {}).get("bbox")]
            row_id = str(row.get("row_id"))
            if result.boxes is None:
                continue
            xyxy = result.boxes.xyxy.detach().cpu().tolist()
            confs = result.boxes.conf.detach().cpu().tolist()
            classes = result.boxes.cls.detach().cpu().tolist()
            for box, conf, cls in zip(xyxy, confs, classes, strict=True):
                label_id = int(cls) + 1
                if label_id not in ID_TO_LABEL:
                    continue
                page_preds[row_id].append(
                    {
                        "bbox": [float(box[0] + left), float(box[1] + top), float(box[2] + left), float(box[3] + top)],
                        "label_id": label_id,
                        "label": ID_TO_LABEL[label_id],
                        "score": float(conf),
                        "tile_id": row.get("id"),
                    }
                )
    return page_preds


def memory_audit() -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {"max_rss_kb": int(usage.ru_maxrss)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v21-dir", default=str(V21))
    parser.add_argument("--yolo-dir", default=str(YOLO_DIR))
    parser.add_argument("--weights", default=str(WEIGHTS))
    parser.add_argument("--split", default="locked", choices=["dev", "locked"])
    parser.add_argument("--eval-output", default=str(REPORT))
    parser.add_argument("--predictions-output", default=str(PREDICTIONS))
    parser.add_argument("--limit-images", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--decode-conf", type=float, default=0.001)
    parser.add_argument("--decode-iou", type=float, default=0.7)
    parser.add_argument("--max-det-per-tile", type=int, default=300)
    parser.add_argument("--predict-batch", type=int, default=32)
    parser.add_argument("--score-threshold-grid", default="0.001,0.003,0.005,0.01,0.02,0.05")
    parser.add_argument("--nms-threshold-grid", default="0.45,0.55,0.65,0.75")
    parser.add_argument("--max-per-page", type=int, default=700)
    parser.add_argument("--selection-mode", default="balanced_f1", choices=["balanced_f1", "recall_gate", "precision_at_recall60", "low_inflation_at_recall60"])
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    items = load_eval_items(Path(args.yolo_dir), Path(args.v21_dir), args.split, args.limit_images or None)
    page_golds = collect_golds(items)
    model = YOLO(args.weights)
    page_preds = collect_predictions(model, items, args)
    score_grid = [float(item) for item in args.score_threshold_grid.split(",") if item.strip()]
    nms_grid = [float(item) for item in args.nms_threshold_grid.split(",") if item.strip()]
    grid_reports: list[dict[str, Any]] = []
    for score_threshold in score_grid:
        for nms_threshold in nms_grid:
            metrics, _ = score_predictions(page_preds, page_golds, score_threshold, nms_threshold, args.max_per_page, len(items))
            grid_reports.append({"score_threshold": score_threshold, "nms_threshold": nms_threshold, "metrics": metrics})
    grid_reports.sort(key=lambda row: selection_key(row, args.selection_mode), reverse=True)
    selected = grid_reports[0]
    eval_metrics, predictions = score_predictions(
        page_preds,
        page_golds,
        float(selected["score_threshold"]),
        float(selected["nms_threshold"]),
        args.max_per_page,
        len(items),
    )
    report = {
        "version": "symbol_recall_detector_p211_page_eval",
        "claim_boundary": "P211 YOLO crop detector restored to page coordinates; gold labels are used only for evaluation.",
        "source_integrity": {
            "model_input": "raster_tile_pixels_only",
            "raw_label_or_semantic_type_used_as_runtime_feature": False,
            "forbidden_runtime_features": FORBIDDEN_RUNTIME_FIELDS,
            "locked_gold_use": "evaluation_only",
        },
        "dataset": rel(Path(args.yolo_dir)),
        "v21_dir": rel(Path(args.v21_dir)),
        "weights": rel(Path(args.weights)),
        "config": vars(args),
        "eval_items": len(items),
        "rows": len(page_golds),
        "threshold_grid": grid_reports,
        "selection_mode": args.selection_mode,
        "selected_thresholds": {"score_threshold": float(selected["score_threshold"]), "nms_threshold": float(selected["nms_threshold"])},
        args.split: eval_metrics,
        "memory_audit": memory_audit(),
    }
    write_json(Path(args.eval_output), report)
    write_jsonl(Path(args.predictions_output), predictions)
    print(json.dumps({args.split: eval_metrics, "selected": report["selected_thresholds"], "eval_items": len(items)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
