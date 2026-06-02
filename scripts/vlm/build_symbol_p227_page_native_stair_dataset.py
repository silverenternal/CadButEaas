#!/usr/bin/env python3
"""Build P227 page-native stair YOLO dataset from full-page overlays and inference-like slices."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from fuse_symbol_p206g_with_p211_p212 import bbox_iou, load_p206g, write_json

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports/vlm/symbol_p224a_column_frozen_overlay.jsonl"
OUT = ROOT / "datasets/symbol_p227_page_native_stair_yolo"
REPORT = OUT / "build_report.json"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("row_id"))


def slice_boxes(width: int, height: int, size: int, overlap: float) -> list[tuple[int, int, int, int]]:
    stride = max(1, int(round(size * (1.0 - overlap))))
    xs = list(range(0, max(width - size + 1, 1), stride))
    ys = list(range(0, max(height - size + 1, 1), stride))
    if not xs or xs[-1] != max(width - size, 0):
        xs.append(max(width - size, 0))
    if not ys or ys[-1] != max(height - size, 0):
        ys.append(max(height - size, 0))
    return [(x, y, min(x + size, width), min(y + size, height)) for y in ys for x in xs]


def area_bucket(box: list[float]) -> str:
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    if area <= 64:
        return "tiny"
    if area <= 256:
        return "small"
    if area <= 1024:
        return "medium"
    if area <= 4096:
        return "large"
    return "xlarge"


def clip_box_to_slice(box: list[float], sl: tuple[int, int, int, int]) -> list[float] | None:
    left, top, right, bottom = sl
    ix1 = max(float(box[0]), left); iy1 = max(float(box[1]), top)
    ix2 = min(float(box[2]), right); iy2 = min(float(box[3]), bottom)
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return [ix1 - left, iy1 - top, ix2 - left, iy2 - top]


def yolo_line(local: list[float], width: int, height: int) -> str | None:
    x1, y1, x2, y2 = local
    bw = max(0.0, x2 - x1); bh = max(0.0, y2 - y1)
    if bw <= 0.25 or bh <= 0.25:
        return None
    cx = (x1 + x2) / 2.0 / max(width, 1)
    cy = (y1 + y2) / 2.0 / max(height, 1)
    return f"0 {cx:.8f} {cy:.8f} {bw / max(width, 1):.8f} {bh / max(height, 1):.8f}"


def is_base_matched(gold: dict[str, Any], preds: list[dict[str, Any]], iou_threshold: float) -> bool:
    gbox = [float(v) for v in gold["bbox"]]
    return any(str(pred.get("label")) == "stair" and bbox_iou([float(v) for v in pred["bbox"]], gbox) >= iou_threshold for pred in preds)


def choose_split(row_index: int, split: str) -> str:
    if split in {"train", "dev", "locked"}:
        return split
    return "train"


def split_for_row(row: dict[str, Any], index: int, train_ratio: float, dev_ratio: float, ignore_explicit_split: bool) -> str:
    explicit = str(row.get("split") or "").lower()
    if not ignore_explicit_split and explicit in {"train", "dev", "val", "valid", "locked", "test"}:
        if explicit in {"val", "valid"}:
            return "dev"
        if explicit == "test":
            return "locked"
        return explicit
    value = int(hashlib.sha1(row_id(row).encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if value < train_ratio:
        return "train"
    if value < train_ratio + dev_ratio:
        return "dev"
    return "locked"


def sample_negative_slices(negative: list[dict[str, Any]], target: int, seed: int) -> list[dict[str, Any]]:
    if len(negative) <= target:
        return negative
    rng = random.Random(seed)
    return rng.sample(negative, target)


def build(args: argparse.Namespace) -> dict[str, Any]:
    rows, base_preds, _golds = load_p206g(Path(args.base))
    out = Path(args.out)
    if args.rebuild and out.exists():
        shutil.rmtree(out)
    for split in ["train", "dev", "locked"]:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    selected_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    negative_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    stats = {"raw_rows": len(rows), "raw_positive_slices": Counter(), "raw_negative_slices": Counter(), "gold_bucket_counts": Counter(), "base_missed_bucket_counts": Counter()}
    for index, row in enumerate(rows):
        rid = row_id(row)
        split = split_for_row(row, index, args.train_ratio, args.dev_ratio, args.ignore_explicit_split)
        image_path = Path(str(row.get("image_path") or row.get("image")))
        if not image_path.is_absolute():
            image_path = ROOT / image_path
        stair_golds = [gold for gold in (row.get("targets") or {}).get("symbol", []) if str(gold.get("semantic_type")) == "stair"]
        if not stair_golds:
            continue
        for gold in stair_golds:
            bucket = area_bucket([float(v) for v in gold["bbox"]])
            stats["gold_bucket_counts"][bucket] += 1
            if not is_base_matched({"bbox": gold["bbox"]}, base_preds.get(rid, []), args.match_iou):
                stats["base_missed_bucket_counts"][bucket] += 1
        with Image.open(image_path) as image:
            width, height = image.size
        slices = slice_boxes(width, height, args.slice_size, args.slice_overlap)
        for slice_index, sl in enumerate(slices):
            labels = []
            buckets = Counter()
            has_base_missed = False
            for gold in stair_golds:
                gbox = [float(v) for v in gold["bbox"]]
                local = clip_box_to_slice(gbox, sl)
                if local is None:
                    continue
                clipped_area = max(0.0, local[2] - local[0]) * max(0.0, local[3] - local[1])
                original_area = max(1e-6, (gbox[2] - gbox[0]) * (gbox[3] - gbox[1]))
                if clipped_area / original_area < args.min_visible_frac:
                    continue
                line = yolo_line(local, sl[2] - sl[0], sl[3] - sl[1])
                if line is None:
                    continue
                labels.append(line)
                bucket = area_bucket(gbox)
                buckets[bucket] += 1
                if not is_base_matched({"bbox": gbox}, base_preds.get(rid, []), args.match_iou):
                    has_base_missed = True
            item = {"row_id": rid, "image_path": image_path, "slice": sl, "slice_index": slice_index, "labels": labels, "buckets": dict(buckets), "has_base_missed": has_base_missed, "split": split}
            if labels:
                weight = 1 + 8 * buckets.get("tiny", 0) + 6 * buckets.get("small", 0) + 3 * int(has_base_missed) + 2 * buckets.get("xlarge", 0)
                copies = min(args.max_positive_copies, max(1, weight)) if split == "train" else 1
                for _ in range(copies):
                    selected_by_split[split].append(item)
                stats["raw_positive_slices"][split] += 1
            else:
                negative_by_split[split].append(item)
                stats["raw_negative_slices"][split] += 1
    rng = random.Random(args.seed)
    final_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in ["train", "dev", "locked"]:
        positives = selected_by_split[split]
        rng.shuffle(positives)
        pos_cap = {"train": args.train_positive_cap, "dev": args.dev_positive_cap, "locked": args.locked_positive_cap}[split]
        positives = positives[:pos_cap]
        neg_target = int(len(positives) * args.negative_ratio)
        negatives = sample_negative_slices(negative_by_split[split], neg_target, args.seed + len(split))
        final = positives + negatives
        rng.shuffle(final)
        final_by_split[split] = final
    selected_stats = {}
    for split, items in final_by_split.items():
        lines = []
        bucket_counts = Counter()
        positive_images = 0
        target_count = 0
        for item_index, item in enumerate(items):
            image_path: Path = item["image_path"]
            sl = item["slice"]
            left, top, right, bottom = sl
            name = f"{item['row_id']}_s{item['slice_index']:04d}_{left}_{top}_{right}_{bottom}_{item_index:06d}.png"
            out_image = out / "images" / split / name
            out_label = out / "labels" / split / f"{out_image.stem}.txt"
            with Image.open(image_path) as image:
                crop = image.convert("RGB").crop((left, top, right, bottom))
                crop.save(out_image)
            label_text = "\n".join(item["labels"])
            out_label.write_text(label_text + ("\n" if label_text else ""), encoding="utf-8")
            lines.append(str(out_image))
            positive_images += int(bool(item["labels"]))
            target_count += len(item["labels"])
            bucket_counts.update(item["buckets"])
        list_path = out / f"{split}.txt"
        list_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        selected_stats[split] = {"list": rel(list_path), "images": len(items), "positive_images": positive_images, "negative_images": len(items) - positive_images, "stair_targets": target_count, "bucket_counts": dict(bucket_counts.most_common())}
    (out / "data.yaml").write_text(f"path: {out}\ntrain: train.txt\nval: dev.txt\ntest: locked.txt\nnames:\n  0: stair\n", encoding="utf-8")
    report = {
        "id": "P227_page_native_stair_yolo_dataset",
        "strategy": "Full-page overlay rows are sliced with the exact page-inference geometry; labels are stair-only and positives are oversampled for tiny/small/base-missed stair.",
        "base": rel(Path(args.base)),
        "config": vars(args),
        "raw": {k: dict(v) if isinstance(v, Counter) else v for k, v in stats.items()},
        "selected": selected_stats,
        "data_yaml": rel(out / "data.yaml"),
        "claim_boundary": "Offline supervised dataset construction from raster rows and annotations; runtime detector uses raster pixels only.",
    }
    write_json(out / "build_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=str(BASE))
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--slice-size", type=int, default=384)
    parser.add_argument("--slice-overlap", type=float, default=0.5)
    parser.add_argument("--min-visible-frac", type=float, default=0.45)
    parser.add_argument("--match-iou", type=float, default=0.30)
    parser.add_argument("--negative-ratio", type=float, default=0.60)
    parser.add_argument("--max-positive-copies", type=int, default=12)
    parser.add_argument("--train-positive-cap", type=int, default=16000)
    parser.add_argument("--dev-positive-cap", type=int, default=2500)
    parser.add_argument("--locked-positive-cap", type=int, default=2500)
    parser.add_argument("--train-ratio", type=float, default=0.72)
    parser.add_argument("--dev-ratio", type=float, default=0.14)
    parser.add_argument("--seed", type=int, default=227)
    parser.add_argument("--ignore-explicit-split", action="store_true", default=True, help="Use row-hash split even when all source rows are marked locked; internal probe only")
    parser.add_argument("--use-explicit-split", dest="ignore_explicit_split", action="store_false")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    report = build(args)
    print(json.dumps({"data_yaml": report["data_yaml"], "raw": report["raw"], "selected": report["selected"]}, ensure_ascii=False, indent=2)[:10000])


if __name__ == "__main__":
    main()
