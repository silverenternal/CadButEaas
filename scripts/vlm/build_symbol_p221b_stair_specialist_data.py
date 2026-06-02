#!/usr/bin/env python3
"""Build stair-only P221b residual specialist YOLO data from P222 FNs."""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from fuse_symbol_p206g_with_p211_p212 import area_bucket, load_p206g

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports/vlm/symbol_p222_p221a_frozen_overlay.jsonl"
CASES = ROOT / "reports/vlm/symbol_p221b_stair_equipment_residual_cases.jsonl"
OUT = ROOT / "datasets/symbol_p221b_stair_specialist_yolo"


def load_cases(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def crop_around(box: list[float], image_size: tuple[int, int], crop_size: int, jitter: int, rng: random.Random) -> tuple[int, int, int, int]:
    width, height = image_size
    cx = (box[0] + box[2]) / 2.0 + rng.randint(-jitter, jitter)
    cy = (box[1] + box[3]) / 2.0 + rng.randint(-jitter, jitter)
    left = max(0, min(int(round(cx - crop_size / 2)), max(width - crop_size, 0)))
    top = max(0, min(int(round(cy - crop_size / 2)), max(height - crop_size, 0)))
    return left, top, min(left + crop_size, width), min(top + crop_size, height)


def stair_yolo_lines(stair_targets: list[dict[str, Any]], crop: tuple[int, int, int, int]) -> list[str]:
    left, top, right, bottom = crop
    crop_width = max(1, right - left)
    crop_height = max(1, bottom - top)
    lines: list[str] = []
    seen: set[str] = set()
    for target in stair_targets:
        x1, y1, x2, y2 = [float(value) for value in target["bbox"]]
        ix1, iy1 = max(x1, left), max(y1, top)
        ix2, iy2 = min(x2, right), min(y2, bottom)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        visible = (ix2 - ix1) * (iy2 - iy1) / max((x2 - x1) * (y2 - y1), 1e-9)
        if visible < 0.50:
            continue
        values = [
            0,
            ((ix1 + ix2) / 2.0 - left) / crop_width,
            ((iy1 + iy2) / 2.0 - top) / crop_height,
            (ix2 - ix1) / crop_width,
            (iy2 - iy1) / crop_height,
        ]
        line = " ".join([str(values[0]), *[f"{value:.8f}" for value in values[1:]]])
        if line not in seen:
            seen.add(line)
            lines.append(line)
    return lines


def extract_stair_targets(row: dict[str, Any]) -> list[dict[str, Any]]:
    targets = []
    for target in (row.get("targets") or {}).get("symbol") or []:
        label = str(target.get("semantic_type") or target.get("label") or "")
        if label != "stair":
            continue
        box = [float(value) for value in target.get("bbox")]
        targets.append({"target_id": target.get("target_id"), "label": "stair", "bbox": box, "bucket": area_bucket(box)})
    return targets


def negative_boxes(case: dict[str, Any], row: dict[str, Any]) -> list[list[float]]:
    boxes: list[list[float]] = []
    nearest = case.get("nearest_any") or {}
    if nearest.get("label") != "stair" and nearest.get("bbox"):
        boxes.append([float(value) for value in nearest["bbox"]])
    for pred in row.get("symbol_candidates") or []:
        label = str(pred.get("label", pred.get("symbol_type", "unknown")))
        if label == "stair":
            continue
        bbox = pred.get("bbox")
        if bbox:
            boxes.append([float(value) for value in bbox])
            if len(boxes) >= 2:
                break
    return boxes


def export_split(name: str, items: list[dict[str, Any]], rows_by_id: dict[str, dict[str, Any]], out: Path, crop_size: int, pos_aug: int, neg_aug: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    image_dir = out / "images" / name
    label_dir = out / "labels" / name
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    buckets = Counter()
    rows = Counter()
    list_lines: list[str] = []
    image_cache: dict[Path, Image.Image] = {}

    def get_image(row: dict[str, Any]) -> tuple[Image.Image, Path]:
        image_path = Path(str(row.get("image_path") or row.get("image")))
        if not image_path.is_absolute():
            image_path = ROOT / image_path
        if image_path not in image_cache:
            image_cache[image_path] = Image.open(image_path).convert("RGB")
        return image_cache[image_path], image_path

    def write_crop(row: dict[str, Any], center_box: list[float], stem: str, aug_index: int, aug_total: int, positive: bool) -> None:
        image, _image_path = get_image(row)
        crop = crop_around(center_box, image.size, crop_size, max(4, crop_size // 8), rng)
        out_img = image_dir / f"{stem}_{aug_index:02d}.jpg"
        out_lbl = label_dir / f"{stem}_{aug_index:02d}.txt"
        image.crop(crop).save(out_img, quality=95)
        lines = stair_yolo_lines(extract_stair_targets(row), crop) if positive else []
        out_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        list_lines.append(str(out_img.resolve()))
        stats["images"] += 1
        stats["targets"] += len(lines)
        if lines:
            stats["positive_images"] += 1
        else:
            stats["negative_images"] += 1

    for index, item in enumerate(items):
        row = rows_by_id[item["row_id"]]
        buckets[item["bucket"]] += 1
        rows[item["row_id"]] += 1
        for aug_index in range(pos_aug):
            stem = f"{item['row_id']}_{item.get('target_id')}_{index:05d}_pos"
            write_crop(row, [float(value) for value in item["bbox"]], stem, aug_index, pos_aug, True)
        negs = negative_boxes(item, row)
        for neg_index, box in enumerate(negs[:2]):
            for aug_index in range(neg_aug):
                stem = f"{item['row_id']}_{item.get('target_id')}_{index:05d}_neg{neg_index}"
                write_crop(row, box, stem, aug_index, neg_aug, False)

    for image in image_cache.values():
        image.close()
    list_path = out / f"{name}.txt"
    list_path.write_text("\n".join(list_lines) + ("\n" if list_lines else ""), encoding="utf-8")
    return {
        "items": len(items),
        "images": int(stats["images"]),
        "positive_images": int(stats["positive_images"]),
        "negative_images": int(stats["negative_images"]),
        "targets": int(stats["targets"]),
        "bucket_counts": dict(buckets),
        "row_counts_top10": dict(rows.most_common(10)),
        "list": str(list_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=str(BASE))
    parser.add_argument("--cases", default=str(CASES))
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--crop-size", type=int, default=192)
    parser.add_argument("--pos-aug", type=int, default=14)
    parser.add_argument("--neg-aug", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2215)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    if args.overwrite and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    rows, _preds, _golds = load_p206g(Path(args.base))
    rows_by_id = {str(row.get("id") or row.get("row_id")): row for row in rows}
    cases = [case for case in load_cases(Path(args.cases)) if case.get("label") == "stair"]
    rng = random.Random(args.seed)
    rng.shuffle(cases)
    n = len(cases)
    train = cases[: int(n * 0.70)]
    dev = cases[int(n * 0.70): int(n * 0.85)]
    locked = cases[int(n * 0.85):]
    report = {
        "id": "P221b_stair_specialist_yolo_data",
        "source_base": str(Path(args.base)),
        "source_cases": str(Path(args.cases)),
        "claim_boundary": "Offline gold/residual labels are used only to build supervised stair proposal training data. Runtime model uses raster pixels only.",
        "class_names": {0: "stair"},
        "crop_size": args.crop_size,
        "pos_aug": args.pos_aug,
        "neg_aug": args.neg_aug,
        "total_stair_items": n,
        "splits": {
            "train": export_split("train", train, rows_by_id, out, args.crop_size, args.pos_aug, args.neg_aug, args.seed + 1),
            "dev": export_split("dev", dev, rows_by_id, out, args.crop_size, args.pos_aug, args.neg_aug, args.seed + 2),
            "locked": export_split("locked", locked, rows_by_id, out, args.crop_size, args.pos_aug, args.neg_aug, args.seed + 3),
        },
    }
    (out / "data.yaml").write_text(
        "\n".join([
            f"path: {out.resolve()}",
            f"train: {(out / 'train.txt').resolve()}",
            f"val: {(out / 'dev.txt').resolve()}",
            f"test: {(out / 'locked.txt').resolve()}",
            "names:",
            "  0: stair",
            "",
        ]),
        encoding="utf-8",
    )
    (out / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
