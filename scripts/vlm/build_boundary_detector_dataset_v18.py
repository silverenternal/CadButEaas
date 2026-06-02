#!/usr/bin/env python3
"""Build v18 raster boundary detector supervision from offline structured gold."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageStat

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "datasets/image_only_structured_targets_v16"
OUT = ROOT / "datasets/image_only_boundary_detector_v18"

SPLITS = ("train", "dev", "locked", "smoke")
BOUNDARY_LABELS = ("wall", "opening", "window")
LABEL_TO_ID = {"wall": 1, "opening": 2, "window": 3}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def norm_bbox(value: Any, width: int, height: int) -> list[int] | None:
    if not isinstance(value, list) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(item) for item in value[:4]]
    except (TypeError, ValueError):
        return None
    left = max(0, min(width - 1, int(math.floor(min(x1, x2)))))
    top = max(0, min(height - 1, int(math.floor(min(y1, y2)))))
    right = max(0, min(width - 1, int(math.ceil(max(x1, x2)))))
    bottom = max(0, min(height - 1, int(math.ceil(max(y1, y2)))))
    if right < left or bottom < top:
        return None
    return [left, top, right, bottom]


def norm_point(value: Any, width: int, height: int) -> tuple[int, int] | None:
    if not isinstance(value, list) or len(value) < 2:
        return None
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    return (
        max(0, min(width - 1, int(round(x)))),
        max(0, min(height - 1, int(round(y)))),
    )


def image_stats(path: Path) -> dict[str, Any]:
    try:
        with Image.open(path) as img:
            gray = img.convert("L")
            stat = ImageStat.Stat(gray)
            return {
                "mean": round(float(stat.mean[0]), 4),
                "stddev": round(float(stat.stddev[0]), 4),
            }
    except Exception as exc:
        return {"mean": None, "stddev": None, "error": f"{type(exc).__name__}: {exc}"}


def edge_record(edge: dict[str, Any], width: int, height: int) -> dict[str, Any] | None:
    label = str(edge.get("class") or edge.get("semantic_type") or "").strip()
    if label not in BOUNDARY_LABELS:
        return None
    bbox = norm_bbox(edge.get("bbox"), width, height)
    if bbox is None:
        return None
    p1 = norm_point(edge.get("p1"), width, height)
    p2 = norm_point(edge.get("p2"), width, height)
    x1, y1, x2, y2 = bbox
    return {
        "target_id": str(edge.get("id") or ""),
        "label": label,
        "label_id": LABEL_TO_ID[label],
        "bbox": bbox,
        "centerline": [list(p1), list(p2)] if p1 and p2 else None,
        "length": round(math.hypot((p2[0] - p1[0]), (p2[1] - p1[1])), 4) if p1 and p2 else None,
        "thin_extent": min(max(1, x2 - x1 + 1), max(1, y2 - y1 + 1)),
        "label_source": "offline_svg_structured_gold",
    }


def stress_buckets(targets: list[dict[str, Any]], stats: dict[str, Any], rooms: int) -> list[str]:
    buckets: set[str] = set()
    counts = Counter(item["label"] for item in targets)
    if any(int(item.get("thin_extent") or 99) <= 2 for item in targets):
        buckets.add("thin_lines")
    if len(targets) >= 300:
        buckets.add("dense_boundary_graph")
    if counts.get("opening", 0) >= 60:
        buckets.add("many_openings")
    if counts.get("window", 0) >= 25:
        buckets.add("many_windows")
    if rooms >= 12:
        buckets.add("dense_rooms")
    stddev = stats.get("stddev")
    if isinstance(stddev, (int, float)) and stddev < 35.0:
        buckets.add("low_contrast")
    if not buckets:
        buckets.add("standard")
    return sorted(buckets)


def draw_masks(row_id: str, width: int, height: int, targets: list[dict[str, Any]], split: str) -> dict[str, str]:
    mask_dir = OUT / "masks" / split
    center_dir = OUT / "centerlines" / split
    mask_dir.mkdir(parents=True, exist_ok=True)
    center_dir.mkdir(parents=True, exist_ok=True)
    mask = Image.new("L", (width, height), 0)
    center = Image.new("L", (width, height), 0)
    mask_draw = ImageDraw.Draw(mask)
    center_draw = ImageDraw.Draw(center)
    for target in targets:
        value = int(target["label_id"])
        x1, y1, x2, y2 = target["bbox"]
        mask_draw.rectangle([x1, y1, x2, y2], fill=value)
        line = target.get("centerline")
        if line:
            p1, p2 = tuple(line[0]), tuple(line[1])
            center_draw.line([p1, p2], fill=value, width=1)
    mask_path = mask_dir / f"{row_id}_boundary_mask.png"
    center_path = center_dir / f"{row_id}_centerline.png"
    mask.save(mask_path)
    center.save(center_path)
    return {
        "boundary_mask": str(mask_path.relative_to(ROOT)),
        "centerline_mask": str(center_path.relative_to(ROOT)),
    }


def convert_row(row: dict[str, Any], split: str, write_masks_enabled: bool) -> dict[str, Any]:
    width, height = [int(v) for v in row.get("image_size") or [512, 512]]
    structured = row.get("structured") if isinstance(row.get("structured"), dict) else {}
    targets = []
    for edge in structured.get("edges") or []:
        converted = edge_record(edge, width, height)
        if converted is not None:
            targets.append(converted)
    label_counts = Counter(item["label"] for item in targets)
    image = str(row.get("image") or "")
    stats = image_stats(ROOT / image) if image else {}
    masks = draw_masks(str(row.get("id")), width, height, targets, split) if write_masks_enabled else {}
    return {
        "id": row.get("id"),
        "source_key": row.get("source_key"),
        "split": split,
        "image": image,
        "image_size": [width, height],
        "targets": {
            "boxes": targets,
            "boundary_mask": masks.get("boundary_mask"),
            "centerline_mask": masks.get("centerline_mask"),
        },
        "target_counts": {
            "total": len(targets),
            **{label: label_counts.get(label, 0) for label in BOUNDARY_LABELS},
        },
        "stress_buckets": stress_buckets(targets, stats, len(structured.get("rooms") or [])),
        "image_stats": stats,
        "source_integrity": {
            "source_mode": "offline_training_gold",
            "model_input": "raster_image_only",
            "label_use": "training_or_locked_evaluation_only",
            "for_model_credit_inference": False,
        },
    }


def split_sanity(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    keys_by_split = {
        split: {str(row.get("source_key") or row.get("id")) for row in rows}
        for split, rows in rows_by_split.items()
    }
    overlaps = {}
    split_names = list(keys_by_split)
    for i, left in enumerate(split_names):
        for right in split_names[i + 1:]:
            overlaps[f"{left}_{right}"] = len(keys_by_split[left] & keys_by_split[right])
    formal_overlap_keys = [
        key for key in overlaps
        if not key.endswith("_smoke") and not key.startswith("smoke_")
    ]
    return {
        "source_key_counts": {split: len(keys) for split, keys in keys_by_split.items()},
        "overlaps": overlaps,
        "formal_split_overlap_keys": formal_overlap_keys,
        "formal_split_sanity_passed": all(overlaps[key] == 0 for key in formal_overlap_keys),
        "smoke_is_subset_allowed": True,
        "split_sanity_passed": all(overlaps[key] == 0 for key in formal_overlap_keys),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    rows_by_split = {split: load_jsonl(SOURCE / f"{split}.jsonl") for split in SPLITS}
    converted_by_split: dict[str, list[dict[str, Any]]] = {}
    label_totals: Counter[str] = Counter()
    stress_totals: Counter[str] = Counter()

    for split, rows in rows_by_split.items():
        converted = [convert_row(row, split, not args.no_masks) for row in rows]
        converted_by_split[split] = converted
        write_jsonl(OUT / f"{split}.jsonl", converted)
        for item in converted:
            for label in BOUNDARY_LABELS:
                label_totals[label] += int((item.get("target_counts") or {}).get(label) or 0)
            stress_totals.update(item.get("stress_buckets") or [])

    manifest = {
        "version": "image_only_boundary_detector_v18",
        "task": "IMG-MOE-V18-P0-002",
        "created": "2026-05-08",
        "dataset": str(OUT.relative_to(ROOT)),
        "source_dataset": str(SOURCE.relative_to(ROOT)),
        "splits": {split: len(rows) for split, rows in converted_by_split.items()},
        "split_sanity": split_sanity(rows_by_split),
        "label_map": LABEL_TO_ID,
        "target_schema": {
            "boxes": ["target_id", "label", "label_id", "bbox", "centerline", "length", "thin_extent"],
            "masks": ["boundary_mask", "centerline_mask"],
            "stress_buckets": ["thin_lines", "dense_boundary_graph", "many_openings", "many_windows", "dense_rooms", "low_contrast", "standard"],
        },
        "aggregate_counts": {
            "labels": dict(label_totals),
            "stress_buckets": dict(sorted(stress_totals.items())),
            "rows": sum(len(rows) for rows in converted_by_split.values()),
            "targets": sum(label_totals.values()),
        },
        "source_integrity_policy": {
            "offline_gold_used_for": ["training_targets", "locked_evaluation"],
            "offline_gold_forbidden_for": ["model_credit_inference"],
            "inference_contract": "raster image only",
        },
        "validation": {
            "manifest_exists": True,
            "all_splits_nonempty": all(len(rows) > 0 for rows in converted_by_split.values()),
            "all_rows_have_images": all(bool(row.get("image")) for rows in converted_by_split.values() for row in rows),
            "all_rows_have_mask_paths": all(
                bool(((row.get("targets") or {}).get("boundary_mask")))
                and bool(((row.get("targets") or {}).get("centerline_mask")))
                for rows in converted_by_split.values()
                for row in rows
            ) if not args.no_masks else None,
            "formal_split_sanity_passed": split_sanity(rows_by_split)["formal_split_sanity_passed"],
            "split_sanity_passed": split_sanity(rows_by_split)["split_sanity_passed"],
        },
    }
    write_json(OUT / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-masks", action="store_true")
    args = parser.parse_args()
    manifest = build(args)
    print(json.dumps({
        "task": manifest["task"],
        "dataset": manifest["dataset"],
        "splits": manifest["splits"],
        "targets": manifest["aggregate_counts"]["targets"],
        "split_sanity_passed": manifest["split_sanity"]["split_sanity_passed"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
