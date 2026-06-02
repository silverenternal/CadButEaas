#!/usr/bin/env python3
"""Build hard-case targeted visual crop data for symbol bbox refinement."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from train_symbol_support_suppression_v35 import load_jsonl
from train_symbol_tile_detector_v20 import area_bucket, rel, write_json, write_jsonl


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
    return [
        (box[0] - crop[0]) * sx,
        (box[1] - crop[1]) * sy,
        (box[2] - crop[0]) * sx,
        (box[3] - crop[1]) * sy,
    ]


def offset(target_bbox: list[float], bbox: list[float]) -> list[float]:
    width = max(bbox[2] - bbox[0], 1e-6)
    height = max(bbox[3] - bbox[1], 1e-6)
    return [
        (target_bbox[0] - bbox[0]) / width,
        (target_bbox[1] - bbox[1]) / height,
        (target_bbox[2] - bbox[2]) / width,
        (target_bbox[3] - bbox[3]) / height,
    ]


def hardcase_buckets(row: dict[str, Any]) -> list[str]:
    labels = row.get("labels") or {}
    feats = row.get("features") or {}
    label = str(row.get("label") or "")
    target_label = str(labels.get("target_label") or "")
    target_area = str(labels.get("target_area_bucket") or "")
    best_iou = float(labels.get("best_iou") or feats.get("best_iou_train_label") or 0.0)
    buckets: list[str] = []
    if bool(labels.get("center_only_no_iou") or feats.get("is_center_only_no_iou")):
        buckets.append("center_covered_iou_lt_030")
    if 0.10 <= best_iou < 0.30:
        buckets.append("best_iou_010_030")
    if best_iou < 0.10:
        buckets.append("best_iou_lt_010")
    if label in {"sink", "stair"} or target_label in {"sink", "stair"}:
        buckets.append("runtime_or_target_sink_stair")
    if target_area in {"large_le_4096", "xlarge_gt_4096"}:
        buckets.append("target_large_xlarge")
    if target_area in {"tiny_le_64", "small_le_256"}:
        buckets.append("target_tiny_small")
    if best_iou >= 0.30:
        buckets.append("positive_anchor_iou_ge_030")
    return buckets


def priority(row: dict[str, Any]) -> tuple[int, float]:
    labels = row.get("labels") or {}
    best_iou = float(labels.get("best_iou") or 0.0)
    buckets = set(hardcase_buckets(row))
    score = 0
    if "center_covered_iou_lt_030" in buckets:
        score += 700
    if "best_iou_010_030" in buckets:
        score += 600
    if "runtime_or_target_sink_stair" in buckets:
        score += 250
    if "target_large_xlarge" in buckets:
        score += 200
    if "target_tiny_small" in buckets:
        score += 125
    if "positive_anchor_iou_ge_030" in buckets:
        score += 50
    return score, best_iou


def is_fulltarget_candidate(row: dict[str, Any]) -> bool:
    labels = row.get("labels") or {}
    best_iou = float(labels.get("best_iou") or 0.0)
    runtime_label = str(row.get("label") or "")
    target_label = str(labels.get("target_label") or "")
    target_area = str(labels.get("target_area_bucket") or "")
    return best_iou < 0.30 and (
        runtime_label in {"sink", "stair"}
        or target_label in {"sink", "stair"}
        or target_area in {"large_le_4096", "xlarge_gt_4096"}
        or bool(labels.get("center_only_no_iou"))
    )


def choose_rows(
    rows: list[dict[str, Any]],
    limit: int,
    seed: int,
    positive_fraction: float,
    selection_mode: str,
) -> list[dict[str, Any]]:
    if not limit or len(rows) <= limit:
        return rows
    rng = random.Random(seed)
    candidates = [row for row in rows if isinstance((row.get("labels") or {}).get("target_bbox"), list)]
    if selection_mode == "fulltarget":
        hard = [row for row in candidates if is_fulltarget_candidate(row)]
    else:
        hard = [row for row in candidates if float((row.get("labels") or {}).get("best_iou") or 0.0) < 0.30]
    positive = [row for row in candidates if float((row.get("labels") or {}).get("best_iou") or 0.0) >= 0.30]
    rng.shuffle(hard)
    rng.shuffle(positive)
    hard.sort(key=priority, reverse=True)
    positive.sort(key=priority, reverse=True)
    positive_n = min(len(positive), int(limit * positive_fraction))
    hard_n = min(len(hard), limit - positive_n)
    selected = hard[:hard_n] + positive[:positive_n]
    if len(selected) < limit:
        selected.extend(hard[hard_n : hard_n + (limit - len(selected))])
    if len(selected) < limit:
        selected.extend(positive[positive_n : positive_n + (limit - len(selected))])
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
    positive_fraction: float,
    selection_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_rows = load_jsonl(rows_path)
    rows = choose_rows(source_rows, limit, seed, positive_fraction, selection_mode)
    cand_to_tile = candidate_tile_map(cache_path)
    tiles = tile_map(tiles_path)
    records: list[dict[str, Any]] = []
    counts = Counter()
    bucket_counts = Counter()
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
        labels = row.get("labels") or {}
        target_bbox = [float(v) for v in labels["target_bbox"]]
        crop_box = expand_box(bbox, image_size, context_scale, context_pad)
        crop_path = out_dir / "crops" / split / f"{cid.replace(':', '_')}.jpg"
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        crop = image.crop(tuple(crop_box)).convert("RGB")
        crop = ImageOps.autocontrast(crop).resize((crop_size, crop_size), Image.Resampling.BICUBIC)
        crop.save(crop_path, quality=92)
        buckets = hardcase_buckets(row)
        rec = {
            "id": cid,
            "page_id": row["page_id"],
            "split": split,
            "crop": {"path": rel(crop_path), "crop_box": crop_box, "size": [crop_size, crop_size]},
            "image_size": image_size,
            "proposal": {
                "bbox": bbox,
                "bbox_in_crop": box_in_crop(bbox, crop_box, crop_size),
                "label": row["label"],
                "score": row.get("score"),
            },
            "target": {
                "bbox": target_bbox,
                "bbox_in_crop": box_in_crop(target_bbox, crop_box, crop_size),
                "offset": offset(target_bbox, bbox),
                "area_bucket": labels.get("target_area_bucket") or area_bucket(target_bbox),
                "label": labels.get("target_label"),
            },
            "hardcase": {
                "buckets": buckets,
                "best_iou": float(labels.get("best_iou") or 0.0),
                "center_only_no_iou": bool(labels.get("center_only_no_iou")),
            },
            "source_integrity": {"gold_used_for_inference": False, "runtime_uses_svg_or_cad_geometry": False},
        }
        records.append(rec)
        counts["records"] += 1
        counts[f"runtime_label:{row['label']}"] += 1
        counts[f"target_label:{rec['target']['label']}"] += 1
        counts[f"target_area:{rec['target']['area_bucket']}"] += 1
        counts[f"proposal_area:{area_bucket(bbox)}"] += 1
        counts["join_candidate_to_tile"] += 1
        if rec["hardcase"]["best_iou"] < 0.30:
            counts["best_iou_lt_030"] += 1
        else:
            counts["best_iou_ge_030"] += 1
        for bucket in buckets:
            bucket_counts[bucket] += 1
    for image in image_cache.values():
        image.close()
    write_jsonl(out_dir / f"{split}.jsonl", records)
    return records, {
        "source_rows": len(source_rows),
        "selected_rows": len(rows),
        "candidate_to_tile_join_rate": round(counts["join_candidate_to_tile"] / max(len(rows), 1), 6),
        "counts": dict(counts),
        "hardcase_buckets": dict(sorted(bucket_counts.items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v36-dir", default="datasets/symbol_support_suppression_v36")
    parser.add_argument("--v38-dir", default="datasets/symbol_box_refiner_v38")
    parser.add_argument("--tile-dir", default="datasets/symbol_tile_detector_tiny_sahi_v21")
    parser.add_argument("--output-dir", default="datasets/symbol_visual_box_refiner_v43_hardcases")
    parser.add_argument("--crop-size", type=int, default=96)
    parser.add_argument("--train-limit", type=int, default=20000)
    parser.add_argument("--dev-limit", type=int, default=4000)
    parser.add_argument("--locked-limit", type=int, default=4000)
    parser.add_argument("--positive-fraction", type=float, default=0.20)
    parser.add_argument("--selection-mode", choices=["hardcase", "fulltarget"], default="hardcase")
    parser.add_argument("--context-scale", type=float, default=2.25)
    parser.add_argument("--context-pad", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    v36 = source_path(args.v36_dir)
    v38 = source_path(args.v38_dir)
    tile_dir = source_path(args.tile_dir)
    out_dir = source_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_records: list[dict[str, Any]] = []
    split_reports: dict[str, Any] = {}
    limits = {"train": args.train_limit, "dev": args.dev_limit, "locked": args.locked_limit}
    for split, limit in limits.items():
        records, report = build_split(
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
            args.positive_fraction,
            args.selection_mode,
        )
        all_records.extend(records)
        split_reports[split] = report
    write_jsonl(out_dir / "rows.jsonl", all_records)
    manifest = {
        "version": "symbol_visual_box_refiner_v43_hardcases",
        "task": "P1-15-hardcase-targeted-visual-refiner-v43",
        "strategy": {
            "selection_mode": args.selection_mode,
            "primary_hardcases": [
                "center_covered_iou_lt_030",
                "best_iou_010_030",
                "runtime_or_target_sink_stair",
                "target_large_xlarge",
            ],
            "positive_fraction": args.positive_fraction,
            "note": "Gold labels are used only for offline supervised crop export/training/evaluation. Runtime inference uses crop pixels and proposal fields only.",
        },
        "outputs": {
            "rows": rel(out_dir / "rows.jsonl"),
            "train": rel(out_dir / "train.jsonl"),
            "dev": rel(out_dir / "dev.jsonl"),
            "locked": rel(out_dir / "locked.jsonl"),
        },
        "splits": split_reports,
        "source_integrity": {
            "runtime_input_allowed": ["raster crop pixels", "candidate bbox/score/type"],
            "offline_labels_used_for": ["hardcase selection", "training", "dev_evaluation", "locked_evaluation"],
            "gold_used_for_inference": False,
            "runtime_uses_svg_or_cad_geometry": False,
        },
    }
    write_json(out_dir / "manifest.json", manifest)
    summary = {
        "manifest": rel(out_dir / "manifest.json"),
        "records": {split: report["counts"].get("records", 0) for split, report in split_reports.items()},
        "join_rates": {split: report["candidate_to_tile_join_rate"] for split, report in split_reports.items()},
        "hardcase_buckets": {split: report["hardcase_buckets"] for split, report in split_reports.items()},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
