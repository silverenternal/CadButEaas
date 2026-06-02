#!/usr/bin/env python3
"""Build a fixed 1/N smoke split for symbol proposal evaluation.

The smoke split is for fast development checks only. It is sampled from an
existing evaluation split and writes symlinks for exported YOLO images/labels so
the normal page-level evaluator can run with --split smoke_v30.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def row_has_symbols(row: dict[str, Any]) -> bool:
    return int((row.get("target_counts") or {}).get("symbols") or 0) > 0


def row_has_tiny_or_small(row: dict[str, Any]) -> bool:
    counts = (row.get("target_counts") or {}).get("area_buckets") or {}
    return int(counts.get("tiny_le_64") or 0) > 0 or int(counts.get("small_le_256") or 0) > 0


def sample_area_aware(rows: list[dict[str, Any]], target_count: int, seed: int, positive_ratio: float, small_positive_ratio: float) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    positives = [row for row in rows if row_has_symbols(row)]
    empties = [row for row in rows if not row_has_symbols(row)]
    small = [row for row in positives if row_has_tiny_or_small(row)]
    small_ids = {id(row) for row in small}
    other_pos = [row for row in positives if id(row) not in small_ids]
    for group in (small, other_pos, empties):
        rng.shuffle(group)

    pos_target = min(len(positives), int(round(target_count * positive_ratio)))
    small_target = min(len(small), int(round(pos_target * small_positive_ratio)))
    selected = small[:small_target] + other_pos[: max(0, pos_target - small_target)]
    if len(selected) < pos_target:
        selected.extend(small[small_target : small_target + (pos_target - len(selected))])
    selected.extend(empties[: max(0, target_count - len(selected))])
    if len(selected) < target_count:
        used = {id(row) for row in selected}
        leftovers = [row for row in rows if id(row) not in used]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: target_count - len(selected)])
    rng.shuffle(selected)
    return selected[:target_count]


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def replace_symlink(dst: Path, src: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def link_exported_assets(yolo_dir: Path, source_split: str, smoke_split: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    linked = Counter()
    missing: list[str] = []
    source_image_dir = yolo_dir / "images" / ("val" if source_split == "dev" else source_split)
    smoke_image_dir = yolo_dir / "images" / smoke_split
    source_label_dir = yolo_dir / "labels" / ("val" if source_split == "dev" else source_split)
    smoke_label_dir = yolo_dir / "labels" / smoke_split
    for row in rows:
        row_id = str(row["id"])
        src_image = source_image_dir / f"{row_id}.jpg"
        dst_image = smoke_image_dir / f"{row_id}.jpg"
        if not src_image.exists():
            missing.append(str(src_image))
            continue
        replace_symlink(dst_image, src_image)
        linked["images"] += 1
        src_label = source_label_dir / f"{row_id}.txt"
        if src_label.exists():
            replace_symlink(smoke_label_dir / f"{row_id}.txt", src_label)
            linked["labels"] += 1
    return {"linked": dict(linked), "missing_images": missing[:25], "missing_image_count": len(missing)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="datasets/symbol_tile_detector_tiny_sahi_v21")
    parser.add_argument("--yolo-dir", default="datasets/symbol_tile_detector_tiny_sahi_v21_yolo_seg_rect_v27")
    parser.add_argument("--source-split", default="locked", choices=["dev", "locked"])
    parser.add_argument("--smoke-split", default="smoke_v30")
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--positive-ratio", type=float, default=0.85)
    parser.add_argument("--small-positive-ratio", type=float, default=0.75)
    parser.add_argument("--manifest-output", default="datasets/symbol_tile_detector_tiny_sahi_v21/smoke_v30_manifest.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    yolo_dir = Path(args.yolo_dir)
    source_path = data_dir / f"{args.source_split}.jsonl"
    rows = load_jsonl(source_path)
    exported = [row for row in rows if (yolo_dir / "images" / args.source_split / f"{row['id']}.jpg").exists()]
    if not exported:
        raise RuntimeError(f"no exported images found under {yolo_dir}/images/{args.source_split}")
    target_count = max(1, int(round(len(exported) * args.fraction)))
    selected = sample_area_aware(exported, target_count, args.seed, args.positive_ratio, args.small_positive_ratio)
    smoke_rows = []
    for row in selected:
        copied = dict(row)
        copied["split"] = args.smoke_split
        copied["smoke_source_split"] = args.source_split
        copied["smoke_sampling"] = {
            "version": "symbol_smoke_split_v30",
            "fraction": args.fraction,
            "seed": args.seed,
            "positive_ratio": args.positive_ratio,
            "small_positive_ratio": args.small_positive_ratio,
        }
        smoke_rows.append(copied)

    smoke_path = data_dir / f"{args.smoke_split}.jsonl"
    write_jsonl(smoke_path, smoke_rows)
    link_report = link_exported_assets(yolo_dir, args.source_split, args.smoke_split, selected)
    symbol_rows = sum(1 for row in selected if row_has_symbols(row))
    tiny_small_rows = sum(1 for row in selected if row_has_tiny_or_small(row))
    manifest = {
        "version": "symbol_smoke_split_v30",
        "claim_boundary": "Fixed smoke split for fast development only; final claims must use locked/full evaluation.",
        "data_dir": rel(data_dir),
        "yolo_dir": rel(yolo_dir),
        "source_split": args.source_split,
        "smoke_split": args.smoke_split,
        "source_exported_rows": len(exported),
        "smoke_rows": len(smoke_rows),
        "fraction": args.fraction,
        "seed": args.seed,
        "symbol_rows": symbol_rows,
        "tiny_or_small_symbol_rows": tiny_small_rows,
        "smoke_jsonl": rel(smoke_path),
        "asset_links": link_report,
        "source_integrity": {
            "model_input": "raster_tile_pixels_only",
            "locked_gold_use": "evaluation_subset_only",
            "runtime_forbidden": ["SVG/parser geometry", "CAD vector primitives", "expected_json", "gold labels"],
        },
    }
    write_json(Path(args.manifest_output), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
