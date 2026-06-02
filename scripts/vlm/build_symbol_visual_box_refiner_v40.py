#!/usr/bin/env python3
"""Build visual crop data for the v40 symbol box refiner."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from train_symbol_support_suppression_v35 import load_jsonl
from train_symbol_tile_detector_v20 import rel, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]


def source_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def tile_map(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["id"]): row for row in load_jsonl(path)}


def candidate_tile_map(cache_path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for page in load_jsonl(cache_path):
        page_id = str(page["row_id"])
        for index, pred in enumerate(page.get("predicted_symbols") or []):
            tile_id = str(pred.get("tile_id") or "")
            if tile_id:
                out[f"{page_id}:{index}"] = tile_id
    return out


def expand_box(box: list[float], image_size: list[int], scale: float, pad: float) -> list[int]:
    width, height = int(image_size[0]), int(image_size[1])
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    nw, nh = bw * scale + 2 * pad, bh * scale + 2 * pad
    return [
        max(0, int(math.floor(cx - nw * 0.5))),
        max(0, int(math.floor(cy - nh * 0.5))),
        min(width, int(math.ceil(cx + nw * 0.5))),
        min(height, int(math.ceil(cy + nh * 0.5))),
    ]


def box_in_crop(box: list[float], crop: list[int], crop_size: int) -> list[float]:
    sx = crop_size / max(crop[2] - crop[0], 1)
    sy = crop_size / max(crop[3] - crop[1], 1)
    return [(box[0] - crop[0]) * sx, (box[1] - crop[1]) * sy, (box[2] - crop[0]) * sx, (box[3] - crop[1]) * sy]


def choose_rows(rows: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if not limit or len(rows) <= limit:
        return rows
    focus = [r for r in rows if (r.get("labels") or {}).get("target_area_bucket") in {"tiny_le_64", "small_le_256"} or r.get("label") in {"sink", "shower", "equipment"}]
    focus_ids = {id(r) for r in focus}
    other = [r for r in rows if id(r) not in focus_ids]
    rng = random.Random(seed)
    rng.shuffle(focus)
    rng.shuffle(other)
    keep_focus = min(len(focus), int(limit * 0.8))
    selected = focus[:keep_focus] + other[: max(0, limit - keep_focus)]
    rng.shuffle(selected)
    return selected[:limit]


def build_split(
    split: str,
    rows_path: Path,
    cache_path: Path,
    tiles_path: Path,
    out_dir: Path,
    crop_size: int,
    limit: int,
    seed: int,
    context_scale: float,
    context_pad: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = choose_rows(load_jsonl(rows_path), limit, seed)
    cand_to_tile = candidate_tile_map(cache_path)
    tiles = tile_map(tiles_path)
    records: list[dict[str, Any]] = []
    counts = Counter()
    image_cache: dict[str, Image.Image] = {}
    for row in rows:
        cid = str(row["candidate_id"])
        tile_id = cand_to_tile.get(cid)
        tile = tiles.get(tile_id or "")
        if not tile:
            counts["missing_tile"] += 1
            continue
        image_path = str(tile.get("image") or "")
        if image_path not in image_cache:
            image_cache[image_path] = Image.open(source_path(image_path)).convert("RGB")
        image = image_cache[image_path]
        image_size = [int(v) for v in tile.get("image_size") or list(image.size)]
        bbox = [float(v) for v in row["bbox"]]
        target_bbox = [float(v) for v in (row.get("labels") or {})["target_bbox"]]
        crop_box = expand_box(bbox, image_size, context_scale, context_pad)
        crop_path = out_dir / "crops" / split / f"{cid.replace(':', '_')}.jpg"
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        crop = image.crop(tuple(crop_box)).convert("RGB")
        crop = ImageOps.autocontrast(crop).resize((crop_size, crop_size), Image.Resampling.BICUBIC)
        crop.save(crop_path, quality=92)
        rec = {
            "id": cid,
            "page_id": row["page_id"],
            "split": split,
            "crop": {"path": rel(crop_path), "crop_box": crop_box, "size": [crop_size, crop_size]},
            "image_size": image_size,
            "proposal": {"bbox": bbox, "bbox_in_crop": box_in_crop(bbox, crop_box, crop_size), "label": row["label"], "score": row.get("score")},
            "target": {"bbox": target_bbox, "bbox_in_crop": box_in_crop(target_bbox, crop_box, crop_size), "offset": [(target_bbox[0]-bbox[0])/max(bbox[2]-bbox[0],1e-6), (target_bbox[1]-bbox[1])/max(bbox[3]-bbox[1],1e-6), (target_bbox[2]-bbox[2])/max(bbox[2]-bbox[0],1e-6), (target_bbox[3]-bbox[3])/max(bbox[3]-bbox[1],1e-6)], "area_bucket": (row.get("labels") or {}).get("target_area_bucket"), "label": (row.get("labels") or {}).get("target_label")},
            "source_integrity": {"gold_used_for_inference": False, "runtime_uses_svg_or_cad_geometry": False},
        }
        records.append(rec)
        counts["records"] += 1
        counts[f"label:{row['label']}"] += 1
        counts[f"area:{rec['target']['area_bucket']}"] += 1
    write_jsonl(out_dir / f"{split}.jsonl", records)
    return records, dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v36-dir", default="datasets/symbol_support_suppression_v36")
    parser.add_argument("--v38-dir", default="datasets/symbol_box_refiner_v38")
    parser.add_argument("--tile-dir", default="datasets/symbol_tile_detector_tiny_sahi_v21")
    parser.add_argument("--output-dir", default="datasets/symbol_visual_box_refiner_v40")
    parser.add_argument("--crop-size", type=int, default=96)
    parser.add_argument("--train-limit", type=int, default=20000)
    parser.add_argument("--dev-limit", type=int, default=4000)
    parser.add_argument("--locked-limit", type=int, default=4000)
    parser.add_argument("--context-scale", type=float, default=2.0)
    parser.add_argument("--context-pad", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=20260512)
    args = parser.parse_args()
    v36 = source_path(args.v36_dir)
    v38 = source_path(args.v38_dir)
    tile_dir = source_path(args.tile_dir)
    out_dir = source_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_records: list[dict[str, Any]] = []
    counts = {}
    for split, limit in [("train", args.train_limit), ("dev", args.dev_limit), ("locked", args.locked_limit)]:
        records, split_counts = build_split(
            split,
            v38 / f"{split}.jsonl",
            v36 / f"{split}_cache.jsonl",
            tile_dir / f"{split}.jsonl",
            out_dir,
            args.crop_size,
            limit,
            args.seed + len(all_records),
            args.context_scale,
            args.context_pad,
        )
        all_records.extend(records)
        counts[split] = split_counts
    write_jsonl(out_dir / "rows.jsonl", all_records)
    manifest = {
        "version": "symbol_visual_box_refiner_v40",
        "task": "P1-11-visual-crop-box-refiner-v40",
        "outputs": {"rows": rel(out_dir / "rows.jsonl"), "train": rel(out_dir / "train.jsonl"), "dev": rel(out_dir / "dev.jsonl"), "locked": rel(out_dir / "locked.jsonl")},
        "counts": counts,
        "source_integrity": {"runtime_input_allowed": ["raster crop pixels", "candidate bbox/score/type"], "offline_labels_used_for": ["training", "dev_evaluation", "locked_evaluation"], "gold_used_for_inference": False},
    }
    write_json(out_dir / "manifest.json", manifest)
    print(json.dumps({"manifest": rel(out_dir / "manifest.json"), "counts": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
