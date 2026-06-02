#!/usr/bin/env python3
"""Run P221b stair-specialist YOLO as page-level sliced raster inference."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image
from ultralytics import YOLO

from fuse_symbol_p206g_with_p211_p212 import simple_nms

ROOT = Path(__file__).resolve().parents[2]
BASE_OVERLAY = ROOT / "reports/vlm/symbol_p222_p221a_frozen_overlay.jsonl"
WEIGHTS = ROOT / "checkpoints/symbol_p221b_stair_specialist_yolo/model.pt"
OUTPUT = ROOT / "reports/vlm/symbol_p221b_stair_specialist_page_predictions.jsonl"
REPORT = ROOT / "reports/vlm/symbol_p221b_stair_specialist_page_infer_report.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("row_id"))


def image_path(row: dict[str, Any]) -> Path:
    raw = row.get("image_path") or row.get("image")
    if not raw:
        raise ValueError(f"missing image path for row {row_id(row)}")
    path = Path(str(raw))
    return path if path.is_absolute() else ROOT / path


def slice_boxes(width: int, height: int, size: int, stride: int) -> list[tuple[int, int, int, int]]:
    xs = list(range(0, max(width - size, 0) + 1, stride))
    ys = list(range(0, max(height - size, 0) + 1, stride))
    if not xs or xs[-1] + size < width:
        xs.append(max(width - size, 0))
    if not ys or ys[-1] + size < height:
        ys.append(max(height - size, 0))
    return [(x, y, min(x + size, width), min(y + size, height)) for y in ys for x in xs]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-overlay", default=str(BASE_OVERLAY))
    parser.add_argument("--weights", default=str(WEIGHTS))
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--report", default=str(REPORT))
    parser.add_argument("--slice-size", type=int, default=192)
    parser.add_argument("--slice-overlap", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--decode-conf", type=float, default=0.01)
    parser.add_argument("--decode-iou", type=float, default=0.7)
    parser.add_argument("--max-det-per-slice", type=int, default=80)
    parser.add_argument("--nms-iou", type=float, default=0.55)
    parser.add_argument("--keep-top-per-row", type=int, default=300)
    parser.add_argument("--predict-batch", type=int, default=64)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    rows = read_jsonl(Path(args.base_overlay))
    model = YOLO(args.weights)
    stride = max(1, int(round(args.slice_size * (1.0 - args.slice_overlap))))
    by_row: dict[str, list[dict[str, Any]]] = defaultdict(list)
    batch_images: list[Image.Image] = []
    batch_meta: list[tuple[str, int, int, str]] = []
    audit = {
        "id": "P221b_stair_specialist_page_inference",
        "base_overlay": str(Path(args.base_overlay).relative_to(ROOT) if Path(args.base_overlay).is_relative_to(ROOT) else args.base_overlay),
        "weights": str(Path(args.weights).relative_to(ROOT) if Path(args.weights).is_relative_to(ROOT) else args.weights),
        "rows": len(rows),
        "slice_size": args.slice_size,
        "slice_overlap": args.slice_overlap,
        "slice_stride": stride,
        "imgsz": args.imgsz,
        "decode_conf": args.decode_conf,
        "total_slices": 0,
        "raw_predictions": 0,
        "kept_predictions": 0,
        "claim_boundary": "Runtime page inference from raster pixels and frozen model weights only; no gold/parser features used.",
    }

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
        for (rid, left, top, tile_id), result in zip(batch_meta, results, strict=True):
            if result.boxes is None:
                continue
            boxes = result.boxes.xyxy.detach().cpu().tolist()
            confs = result.boxes.conf.detach().cpu().tolist()
            for box, conf in zip(boxes, confs, strict=True):
                audit["raw_predictions"] += 1
                by_row[rid].append({
                    "bbox": [float(box[0] + left), float(box[1] + top), float(box[2] + left), float(box[3] + top)],
                    "label": "stair",
                    "label_id": 8,
                    "score": float(conf),
                    "tile_id": tile_id,
                    "source": "p221b_stair_specialist",
                })
        batch_images = []
        batch_meta = []

    for row in rows:
        rid = row_id(row)
        with Image.open(image_path(row)) as image:
            image = image.convert("RGB")
            width, height = image.size
            for index, (left, top, right, bottom) in enumerate(slice_boxes(width, height, args.slice_size, stride)):
                batch_images.append(image.crop((left, top, right, bottom)))
                batch_meta.append((rid, left, top, f"{rid}_p221b_slice_{index}_{left}_{top}_{right}_{bottom}"))
                audit["total_slices"] += 1
                if len(batch_images) >= args.predict_batch:
                    flush()
        flush()

    output_rows = []
    for row in rows:
        rid = row_id(row)
        preds = simple_nms(by_row.get(rid, []), args.nms_iou)[: args.keep_top_per_row]
        audit["kept_predictions"] += len(preds)
        output_rows.append({"row_id": rid, "predicted_symbols": preds})
    write_jsonl(Path(args.output), output_rows)
    audit["output"] = str(Path(args.output).relative_to(ROOT) if Path(args.output).is_relative_to(ROOT) else args.output)
    write_json(Path(args.report), audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
