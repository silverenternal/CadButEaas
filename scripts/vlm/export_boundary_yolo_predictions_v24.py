#!/usr/bin/env python3
"""Export boundary YOLO tile predictions in the COCO-like format used by v24."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "datasets/boundary_expert_public_raster_v19"
DEFAULT_YOLO_DIR = ROOT / "datasets/boundary_public_raster_v24_yolo_probe"
DEFAULT_WEIGHTS = ROOT / "runs/detect/runs/detect/runs/vlm/boundary_public_raster_v24_yolo_probe/weights/best.pt"
DEFAULT_OUTPUT = ROOT / "reports/vlm/boundary_public_raster_v24_yolo_probe_dev_predictions.json"


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def collect_tile_paths(yolo_dir: Path, split: str, row_ids: set[str]) -> tuple[list[Path], dict[str, Any]]:
    yolo_split = "val" if split == "dev" else split
    image_dir = yolo_dir / "images" / yolo_split
    paths: list[Path] = []
    tile_row_ids: set[str] = set()
    for path in sorted(image_dir.glob("*.jpg")):
        stem = path.stem
        row_id = stem.split("_t", 1)[0]
        if row_id in row_ids:
            tile_row_ids.add(row_id)
            paths.append(path)
    missing_row_ids = sorted(row_ids - tile_row_ids)
    audit = {
        "requested_rows": len(row_ids),
        "tile_rows": len(tile_row_ids),
        "missing_rows": len(missing_row_ids),
        "row_coverage": round(len(tile_row_ids) / max(1, len(row_ids)), 6),
        "tiles": len(paths),
        "missing_row_id_sample": missing_row_ids[:50],
        "image_dir": str(image_dir),
    }
    return paths, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--yolo-dir", default=str(DEFAULT_YOLO_DIR))
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--split", default="dev")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--min-row-coverage", type=float, default=0.0)
    parser.add_argument("--audit-output", default="")
    args = parser.parse_args()

    rows = load_jsonl(Path(args.dataset) / f"{args.split}.jsonl", args.limit or None)
    row_ids = {str(row.get("id")) for row in rows}
    tile_paths, coverage_audit = collect_tile_paths(Path(args.yolo_dir), args.split, row_ids)
    if args.audit_output:
        audit_output = Path(args.audit_output)
        audit_output.parent.mkdir(parents=True, exist_ok=True)
        audit_output.write_text(json.dumps(coverage_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if coverage_audit["row_coverage"] < args.min_row_coverage:
        print(
            json.dumps(
                {
                    "error": "row_coverage_below_threshold",
                    "min_row_coverage": args.min_row_coverage,
                    **coverage_audit,
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        sys.exit(2)
    model = YOLO(str(args.weights))
    predictions: list[dict[str, Any]] = []
    for start in range(0, len(tile_paths), max(1, args.batch)):
        chunk = tile_paths[start : start + max(1, args.batch)]
        results = model.predict(
            source=[str(path) for path in chunk],
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
            batch=max(1, args.batch),
            stream=False,
            verbose=False,
        )
        for path, result in zip(chunk, results, strict=True):
            if result.boxes is None:
                continue
            xyxy = result.boxes.xyxy.detach().cpu().tolist()
            confs = result.boxes.conf.detach().cpu().tolist()
            classes = result.boxes.cls.detach().cpu().tolist()
            for box, score, cls in zip(xyxy, confs, classes, strict=True):
                predictions.append(
                    {
                        "image_id": path.stem,
                        "file_name": path.name,
                        "category_id": int(cls) + 1,
                        "bbox": [
                            round(float(box[0]), 3),
                            round(float(box[1]), 3),
                            round(float(box[2] - box[0]), 3),
                            round(float(box[3] - box[1]), 3),
                        ],
                        "score": round(float(score), 6),
                    }
                )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(predictions, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"rows": len(rows), **coverage_audit, "predictions": len(predictions), "output": str(output)},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
