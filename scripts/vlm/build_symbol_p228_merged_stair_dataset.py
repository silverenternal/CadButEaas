#!/usr/bin/env python3
"""Build P228 merged-stair object dataset using connected stair annotation groups."""
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

from audit_symbol_p228_stair_line_groups import components, is_sentinel, union_box
from fuse_symbol_p206g_with_p211_p212 import load_p206g, write_json

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports/vlm/symbol_p224a_column_frozen_overlay.jsonl"
OUT = ROOT / "datasets/symbol_p228_merged_stair_yolo"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("row_id"))


def box(target: dict[str, Any]) -> list[float]:
    return [float(v) for v in target["bbox"]]


def slice_boxes(width: int, height: int, size: int, overlap: float) -> list[tuple[int, int, int, int]]:
    stride = max(1, int(round(size * (1.0 - overlap))))
    xs = list(range(0, max(width - size + 1, 1), stride))
    ys = list(range(0, max(height - size + 1, 1), stride))
    if not xs or xs[-1] != max(width - size, 0):
        xs.append(max(width - size, 0))
    if not ys or ys[-1] != max(height - size, 0):
        ys.append(max(height - size, 0))
    return [(x, y, min(x + size, width), min(y + size, height)) for y in ys for x in xs]


def split_for_row(rid: str, train_ratio: float, dev_ratio: float) -> str:
    value = int(hashlib.sha1(rid.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if value < train_ratio:
        return "train"
    if value < train_ratio + dev_ratio:
        return "dev"
    return "locked"


def clip_box_to_slice(b: list[float], sl: tuple[int, int, int, int]) -> tuple[list[float] | None, float]:
    left, top, right, bottom = sl
    ix1 = max(float(b[0]), left); iy1 = max(float(b[1]), top)
    ix2 = min(float(b[2]), right); iy2 = min(float(b[3]), bottom)
    if ix2 <= ix1 or iy2 <= iy1:
        return None, 0.0
    original = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    clipped = (ix2 - ix1) * (iy2 - iy1)
    return [ix1 - left, iy1 - top, ix2 - left, iy2 - top], clipped / original


def yolo_line(local: list[float], width: int, height: int) -> str | None:
    x1, y1, x2, y2 = local
    bw = max(0.0, x2 - x1); bh = max(0.0, y2 - y1)
    if bw <= 2.0 or bh <= 2.0:
        return None
    cx = (x1 + x2) / 2.0 / max(width, 1)
    cy = (y1 + y2) / 2.0 / max(height, 1)
    return f"0 {cx:.8f} {cy:.8f} {bw / max(width, 1):.8f} {bh / max(height, 1):.8f}"


def area(b: list[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def bucket(b: list[float]) -> str:
    a = area(b)
    if a <= 1024:
        return "medium_or_smaller"
    if a <= 4096:
        return "large"
    if a <= 20000:
        return "xlarge"
    return "huge_merged"


def merged_groups(row: dict[str, Any], margin: float, center_gap: float, min_members: int) -> list[dict[str, Any]]:
    stair_targets = [target for target in (row.get("targets") or {}).get("symbol", []) if str(target.get("semantic_type")) == "stair"]
    real = [box(target) for target in stair_targets if not is_sentinel(box(target))]
    comps = components(real, margin, center_gap) if real else []
    groups = []
    for comp in comps:
        members = [real[i] for i in comp]
        merged = union_box(members)
        if len(members) < min_members and area(merged) < 1024:
            continue
        groups.append({"bbox": merged, "members": len(members), "member_area": sum(area(b) for b in members), "bucket": bucket(merged)})
    return groups


def build(args: argparse.Namespace) -> dict[str, Any]:
    rows, _base_preds, _golds = load_p206g(Path(args.base))
    out = Path(args.out)
    if args.rebuild and out.exists():
        shutil.rmtree(out)
    for split in ["train", "dev", "locked"]:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    positives: dict[str, list[dict[str, Any]]] = defaultdict(list)
    negatives: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_stats = {"rows": len(rows), "rows_with_groups": 0, "groups": 0, "group_buckets": Counter(), "positive_slices": Counter(), "negative_slices": Counter()}
    for row in rows:
        rid = row_id(row)
        groups = merged_groups(row, args.merge_margin, args.center_gap, args.min_members)
        if not groups:
            continue
        raw_stats["rows_with_groups"] += 1
        raw_stats["groups"] += len(groups)
        raw_stats["group_buckets"].update(group["bucket"] for group in groups)
        split = split_for_row(rid, args.train_ratio, args.dev_ratio)
        image_path = Path(str(row.get("image_path") or row.get("image")))
        if not image_path.is_absolute():
            image_path = ROOT / image_path
        with Image.open(image_path) as image:
            width, height = image.size
        for slice_index, sl in enumerate(slice_boxes(width, height, args.slice_size, args.slice_overlap)):
            labels = []
            slice_buckets = Counter()
            for group in groups:
                local, visible = clip_box_to_slice(group["bbox"], sl)
                if local is None or visible < args.min_visible_frac:
                    continue
                line = yolo_line(local, sl[2] - sl[0], sl[3] - sl[1])
                if line is None:
                    continue
                labels.append(line)
                slice_buckets[group["bucket"]] += 1
            item = {"row_id": rid, "image_path": image_path, "slice": sl, "slice_index": slice_index, "labels": labels, "buckets": dict(slice_buckets), "split": split}
            if labels:
                weight = 1 + 5 * slice_buckets.get("huge_merged", 0) + 3 * slice_buckets.get("xlarge", 0)
                copies = min(args.max_positive_copies, max(1, weight)) if split == "train" else 1
                for _ in range(copies):
                    positives[split].append(item)
                raw_stats["positive_slices"][split] += 1
            else:
                negatives[split].append(item)
                raw_stats["negative_slices"][split] += 1
    rng = random.Random(args.seed)
    final: dict[str, list[dict[str, Any]]] = {}
    for split in ["train", "dev", "locked"]:
        pos = positives[split]
        rng.shuffle(pos)
        cap = {"train": args.train_positive_cap, "dev": args.dev_positive_cap, "locked": args.locked_positive_cap}[split]
        pos = pos[:cap]
        neg_target = int(len(pos) * args.negative_ratio)
        neg = negatives[split]
        if len(neg) > neg_target:
            neg = rng.sample(neg, neg_target)
        final[split] = pos + neg
        rng.shuffle(final[split])
    selected = {}
    for split, items in final.items():
        list_lines = []
        bucket_counts = Counter()
        target_count = 0
        positive_count = 0
        for index, item in enumerate(items):
            left, top, right, bottom = item["slice"]
            name = f"{item['row_id']}_s{item['slice_index']:04d}_{left}_{top}_{right}_{bottom}_{index:06d}.png"
            out_image = out / "images" / split / name
            out_label = out / "labels" / split / f"{out_image.stem}.txt"
            with Image.open(item["image_path"]) as image:
                crop = image.convert("RGB").crop((left, top, right, bottom))
                crop.save(out_image)
            text = "\n".join(item["labels"])
            out_label.write_text(text + ("\n" if text else ""), encoding="utf-8")
            list_lines.append(str(out_image))
            positive_count += int(bool(item["labels"]))
            target_count += len(item["labels"])
            bucket_counts.update(item["buckets"])
        list_path = out / f"{split}.txt"
        list_path.write_text("\n".join(list_lines) + ("\n" if list_lines else ""), encoding="utf-8")
        selected[split] = {"list": rel(list_path), "images": len(items), "positive_images": positive_count, "negative_images": len(items) - positive_count, "targets": target_count, "bucket_counts": dict(bucket_counts.most_common())}
    (out / "data.yaml").write_text(f"path: {out}\ntrain: train.txt\nval: dev.txt\ntest: locked.txt\nnames:\n  0: stair\n", encoding="utf-8")
    report = {
        "id": "P228_merged_stair_yolo_dataset",
        "strategy": "Drop sentinel stair artifacts, group connected real stair components, and train object-level merged stair boxes on page-native slices.",
        "base": rel(Path(args.base)),
        "config": vars(args),
        "raw": {key: dict(value) if isinstance(value, Counter) else value for key, value in raw_stats.items()},
        "selected": selected,
        "data_yaml": rel(out / "data.yaml"),
        "claim_boundary": "Offline label-representation experiment; original evaluation still uses official stair targets unless explicitly mapped.",
    }
    write_json(out / "build_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=str(BASE))
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--slice-size", type=int, default=384)
    parser.add_argument("--slice-overlap", type=float, default=0.5)
    parser.add_argument("--min-visible-frac", type=float, default=0.35)
    parser.add_argument("--merge-margin", type=float, default=18.0)
    parser.add_argument("--center-gap", type=float, default=160.0)
    parser.add_argument("--min-members", type=int, default=1)
    parser.add_argument("--negative-ratio", type=float, default=0.60)
    parser.add_argument("--max-positive-copies", type=int, default=8)
    parser.add_argument("--train-positive-cap", type=int, default=12000)
    parser.add_argument("--dev-positive-cap", type=int, default=2500)
    parser.add_argument("--locked-positive-cap", type=int, default=2500)
    parser.add_argument("--train-ratio", type=float, default=0.72)
    parser.add_argument("--dev-ratio", type=float, default=0.14)
    parser.add_argument("--seed", type=int, default=228)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    report = build(args)
    print(json.dumps({"data_yaml": report["data_yaml"], "raw": report["raw"], "selected": report["selected"]}, ensure_ascii=False, indent=2)[:10000])


if __name__ == "__main__":
    main()
