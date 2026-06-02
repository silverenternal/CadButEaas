#!/usr/bin/env python3
"""Export boundary_public_raster_v19 as YOLO tile supervision for v24."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "datasets/boundary_expert_public_raster_v19"
OUT = ROOT / "datasets/boundary_public_raster_v24_yolo"
REPORT = ROOT / "reports/vlm/boundary_public_raster_v24_yolo_dataset_audit.json"
LABELS = ["wall", "opening", "window"]
FORBIDDEN_RUNTIME_FIELDS = ["raw_label", "semantic_type", "expected_json", "annotation_path", "svg_geometry"]


def load_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def tile_origins(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    origins = list(range(0, max(1, length - tile_size + 1), stride))
    last = length - tile_size
    if origins[-1] != last:
        origins.append(last)
    return origins


def clip_box(box: list[float], tile: list[int], min_visible_ratio: float) -> list[float] | None:
    x1, y1, x2, y2 = [float(v) for v in box]
    tx1, ty1, tx2, ty2 = [float(v) for v in tile]
    ix1, iy1 = max(x1, tx1), max(y1, ty1)
    ix2, iy2 = min(x2, tx2), min(y2, ty2)
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    area = max(1.0, (x2 - x1) * (y2 - y1))
    if ((ix2 - ix1) * (iy2 - iy1)) / area < min_visible_ratio:
        return None
    return [ix1 - tx1, iy1 - ty1, ix2 - tx1, iy2 - ty1]


def yolo_line(label: str, box: list[float], width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2.0) / max(width, 1)
    cy = ((y1 + y2) / 2.0) / max(height, 1)
    bw = (x2 - x1) / max(width, 1)
    bh = (y2 - y1) / max(height, 1)
    cls = LABELS.index(label)
    return f"{cls} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}"


def export_split(source_dir: Path, out_dir: Path, split: str, args: argparse.Namespace) -> dict[str, Any]:
    source_split = "val" if split == "dev" else split
    rows = load_jsonl(source_dir / f"{split}.jsonl", int(getattr(args, f"limit_{split}") or 0))
    image_dir = out_dir / "images" / source_split
    label_dir = out_dir / "labels" / source_split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    label_counts = Counter()
    stress_counts = Counter()
    missed_targets = Counter()
    for row in rows:
        image_path = ROOT / str(row.get("image") or "")
        if not image_path.exists():
            stats["missing_images"] += 1
            continue
        targets = []
        for target in ((row.get("targets") or {}).get("boxes") or []):
            label = str(target.get("label") or "")
            box = target.get("bbox")
            if label not in LABELS or not isinstance(box, list) or len(box) != 4:
                continue
            targets.append({"id": str(target.get("target_id")), "label": label, "bbox": [float(v) for v in box]})
        coverage = {target["id"]: 0 for target in targets}
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            width, height = image.size
            for top in tile_origins(height, args.tile_size, args.stride):
                for left in tile_origins(width, args.tile_size, args.stride):
                    tile = [left, top, min(width, left + args.tile_size), min(height, top + args.tile_size)]
                    tile_w, tile_h = tile[2] - tile[0], tile[3] - tile[1]
                    lines = []
                    for target in targets:
                        clipped = clip_box(target["bbox"], tile, args.min_visible_ratio)
                        if clipped is None:
                            continue
                        coverage[target["id"]] += 1
                        lines.append(yolo_line(target["label"], clipped, tile_w, tile_h))
                        label_counts[target["label"]] += 1
                    if not lines and stats[f"{row['id']}::empty"] >= args.max_empty_per_page:
                        continue
                    tile_id = f"{row['id']}_t{args.tile_size}_{left}_{top}_{tile[2]}_{tile[3]}"
                    image.crop(tuple(tile)).save(image_dir / f"{tile_id}.jpg", quality=90)
                    (label_dir / f"{tile_id}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                    stats["tiles"] += 1
                    stats["positive_tiles" if lines else "empty_tiles"] += 1
                    if not lines:
                        stats[f"{row['id']}::empty"] += 1
        for target in targets:
            if coverage[target["id"]] == 0:
                missed_targets[target["label"]] += 1
        for bucket in row.get("stress_buckets") or []:
            stress_counts[str(bucket)] += 1
        stats["rows"] += 1
        stats["targets"] += len(targets)
    for key in list(stats):
        if "::empty" in key:
            del stats[key]
    return {
        "rows": int(stats["rows"]),
        "tiles": int(stats["tiles"]),
        "positive_tiles": int(stats["positive_tiles"]),
        "empty_tiles": int(stats["empty_tiles"]),
        "targets": int(stats["targets"]),
        "label_instances": dict(label_counts),
        "missed_targets": dict(missed_targets),
        "stress_buckets": dict(stress_counts),
        "missing_images": int(stats["missing_images"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(SOURCE))
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--report-output", default=str(REPORT))
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--stride", type=int, default=480)
    parser.add_argument("--min-visible-ratio", type=float, default=0.25)
    parser.add_argument("--max-empty-per-page", type=int, default=2)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-dev", type=int, default=0)
    parser.add_argument("--limit-locked", type=int, default=0)
    parser.add_argument("--limit-smoke", type=int, default=0)
    args = parser.parse_args()

    source_dir = ROOT / args.source if not Path(args.source).is_absolute() else Path(args.source)
    out_dir = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_stats = {}
    for split in ["train", "dev", "locked", "smoke"]:
        split_stats[split] = export_split(source_dir, out_dir, split, args)
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {out_dir}",
                "train: images/train",
                "val: images/val",
                "test: images/locked",
                "names:",
                *[f"  {idx}: {label}" for idx, label in enumerate(LABELS)],
                "",
            ]
        ),
        encoding="utf-8",
    )
    audit = {
        "version": "boundary_public_raster_v24_yolo_dataset",
        "source": rel(source_dir),
        "output": rel(out_dir),
        "data_yaml": rel(data_yaml),
        "config": vars(args),
        "splits": split_stats,
        "source_integrity": {
            "model_input": "raster_tile_pixels_only",
            "raw_label_or_semantic_type_used_as_runtime_feature": False,
            "forbidden_runtime_features": FORBIDDEN_RUNTIME_FIELDS,
            "offline_gold_used_for": ["supervised_training", "dev_selection", "locked_evaluation", "audit"],
        },
    }
    write_json(Path(args.report_output), audit)
    print(json.dumps({"data_yaml": audit["data_yaml"], "splits": split_stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
