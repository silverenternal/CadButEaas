#!/usr/bin/env python3
"""Run P224 YOLO detector as sliced inference on P222/P101 pages."""
from __future__ import annotations

import argparse
import json
import resource
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image
from ultralytics import YOLO

from fuse_symbol_p206g_with_p211_p212 import LABEL_TO_ID, load_p206g, rel, score_predictions, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
P206G = ROOT / "reports/vlm/symbol_p222_p221a_frozen_overlay.jsonl"
WEIGHTS = ROOT / "checkpoints/symbol_p224_detector_yolo/model.pt"
PREDICTIONS = ROOT / "reports/vlm/symbol_p224_detector_pages_sliced_predictions.jsonl"
REPORT = ROOT / "reports/vlm/symbol_p224_detector_pages_sliced_eval.json"


def slice_boxes(width: int, height: int, size: int, overlap: float) -> list[tuple[int, int, int, int]]:
    stride = max(1, int(round(size * (1.0 - overlap))))
    xs = list(range(0, max(width - size + 1, 1), stride))
    ys = list(range(0, max(height - size + 1, 1), stride))
    if not xs or xs[-1] != max(width - size, 0):
        xs.append(max(width - size, 0))
    if not ys or ys[-1] != max(height - size, 0):
        ys.append(max(height - size, 0))
    return [(x, y, min(x + size, width), min(y + size, height)) for y in ys for x in xs]


def collect_sliced(model: YOLO, rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    page_preds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    audit = {"rows": len(rows), "slice_size": args.slice_size, "slice_overlap": args.slice_overlap, "total_slices": 0}
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
            batch=len(batch_images),
            stream=False,
            verbose=False,
        )
        for (row_id, left, top, slice_id), result in zip(batch_meta, results, strict=True):
            if result.boxes is None:
                continue
            xyxy = result.boxes.xyxy.detach().cpu().tolist()
            confs = result.boxes.conf.detach().cpu().tolist()
            classes = result.boxes.cls.detach().cpu().tolist()
            class_labels = [item.strip() for item in str(getattr(args, "class_labels", "")).split(",") if item.strip()]
            for box, conf, cls in zip(xyxy, confs, classes, strict=True):
                class_index = int(cls)
                if class_labels:
                    label = class_labels[class_index] if class_index < len(class_labels) else class_labels[-1]
                    label_id = LABEL_TO_ID.get(label, LABEL_TO_ID["generic_symbol"])
                else:
                    label_id = class_index + 1
                    label = next((name for name, idx in LABEL_TO_ID.items() if idx == label_id), "generic_symbol")
                page_preds[row_id].append({
                    "bbox": [float(box[0] + left), float(box[1] + top), float(box[2] + left), float(box[3] + top)],
                    "label_id": label_id,
                    "label": label,
                    "score": float(conf),
                    "tile_id": slice_id,
                    "source": "p224_sliced_p222_page",
                })
        batch_images = []
        batch_meta = []

    for row in rows:
        row_id = str(row.get("id") or row.get("row_id"))
        image_path = Path(str(row.get("image_path") or row.get("image")))
        if not image_path.is_absolute():
            image_path = ROOT / image_path
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            for idx, (left, top, right, bottom) in enumerate(slice_boxes(width, height, args.slice_size, args.slice_overlap)):
                batch_images.append(image.crop((left, top, right, bottom)))
                batch_meta.append((row_id, left, top, f"{row_id}_slice_{idx}_{left}_{top}_{right}_{bottom}"))
                audit["total_slices"] += 1
                if len(batch_images) >= args.predict_batch:
                    flush()
        flush()
    return page_preds, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p206g", default=str(P206G))
    parser.add_argument("--weights", default=str(WEIGHTS))
    parser.add_argument("--predictions-output", default=str(PREDICTIONS))
    parser.add_argument("--eval-output", default=str(REPORT))
    parser.add_argument("--slice-size", type=int, default=640)
    parser.add_argument("--slice-overlap", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--decode-conf", type=float, default=0.001)
    parser.add_argument("--decode-iou", type=float, default=0.7)
    parser.add_argument("--max-det-per-slice", type=int, default=300)
    parser.add_argument("--predict-batch", type=int, default=16)
    parser.add_argument("--score-threshold-grid", default="0.001,0.003,0.005,0.01,0.02,0.05,0.1,0.2,0.3")
    parser.add_argument("--nms-threshold-grid", default="0.25,0.35,0.45,0.55,0.65")
    parser.add_argument("--max-per-page", type=int, default=900)
    parser.add_argument("--device", default="0")
    parser.add_argument("--class-labels", default="", help="Optional comma-separated model class labels, e.g. stair for one-class specialists")
    args = parser.parse_args()

    rows, _core, golds = load_p206g(Path(args.p206g))
    model = YOLO(args.weights)
    page_preds, audit = collect_sliced(model, rows, args)
    prediction_rows = [{"row_id": row_id, "predicted_symbols": preds, "gold_symbol_count": len(golds.get(row_id, {}))} for row_id, preds in sorted(page_preds.items())]
    write_jsonl(Path(args.predictions_output), prediction_rows)
    grid=[]
    for score in [float(x) for x in args.score_threshold_grid.split(',') if x.strip()]:
        for nms in [float(x) for x in args.nms_threshold_grid.split(',') if x.strip()]:
            metrics, _ = score_predictions(page_preds, golds, score, nms, args.max_per_page, audit["total_slices"])
            grid.append({"score_threshold": score, "nms_threshold": nms, "metrics": metrics})
    grid.sort(key=lambda row: (row["metrics"]["symbol_bbox_iou_0_30"]["f1"], row["metrics"]["symbol_bbox_iou_0_30"]["recall"]), reverse=True)
    report = {
        "id":"P224_detector_sliced_on_p222_pages",
        "claim_boundary":"P224 detector run as raster sliced inference on exact P222/P101 pages; selected thresholds are planning evidence only until fusion/bootstrap.",
        "weights": rel(Path(args.weights)),
        "p206g": rel(Path(args.p206g)),
        "config": vars(args),
        "audit": audit,
        "selected": grid[0],
        "threshold_grid": grid,
        "outputs": {"predictions": rel(Path(args.predictions_output)), "eval": rel(Path(args.eval_output))},
        "memory_audit": {"max_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)},
    }
    write_json(Path(args.eval_output), report)
    print(json.dumps({"selected": grid[0], "audit": audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
