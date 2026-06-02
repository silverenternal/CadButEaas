#!/usr/bin/env python3
"""Build a YOLO segmentation view for symbol proposal training.

The current symbol frontend is bbox-only.  This converter creates a first
instance-segmentation training view by turning each bbox target into a tight
rectangle polygon.  It is not a final mask annotation, but it lets us train a
pretrained segmentation backbone and evaluate mask-derived tight boxes under
the same locked page metric.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21_yolo_v22"
TRAIN_TXT = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21_yolo_recall_v24/train_recall_v24.txt"
OUT = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21_yolo_seg_rect_v27"


def yolo_box_to_polygon(line: str) -> str | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls, cx, cy, bw, bh = parts[:5]
    cx_f = float(cx)
    cy_f = float(cy)
    bw_f = float(bw)
    bh_f = float(bh)
    x1 = max(0.0, min(1.0, cx_f - bw_f / 2.0))
    y1 = max(0.0, min(1.0, cy_f - bh_f / 2.0))
    x2 = max(0.0, min(1.0, cx_f + bw_f / 2.0))
    y2 = max(0.0, min(1.0, cy_f + bh_f / 2.0))
    if x2 <= x1 or y2 <= y1:
        return None
    values = [cls, x1, y1, x2, y1, x2, y2, x1, y2]
    return " ".join([values[0], *[f"{float(v):.8f}" for v in values[1:]]])


def convert_label(src: Path, dst: Path) -> tuple[int, int]:
    converted: list[str] = []
    invalid = 0
    if src.exists():
        for line in src.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = yolo_box_to_polygon(line)
            if item is None:
                invalid += 1
            else:
                converted.append(item)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(converted) + ("\n" if converted else ""), encoding="utf-8")
    return len(converted), invalid


def image_to_label(path: Path, root: Path) -> Path:
    rel = path.relative_to(root / "images")
    return root / "labels" / rel.with_suffix(".txt")


def copy_images_and_labels(split: str, counts: dict[str, int]) -> None:
    image_dir = SRC / "images" / split
    out_image_dir = OUT / "images" / split
    out_label_dir = OUT / "labels" / split
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)
    for image in sorted(image_dir.glob("*.jpg")):
        dst_image = out_image_dir / image.name
        if not dst_image.exists():
            try:
                dst_image.symlink_to(image.resolve())
            except OSError:
                shutil.copy2(image, dst_image)
        label_count, invalid = convert_label(image_to_label(image, SRC), out_label_dir / image.with_suffix(".txt").name)
        counts[f"{split}_images"] += 1
        counts[f"{split}_labels"] += label_count
        counts[f"{split}_invalid"] += invalid


def build_train_txt(counts: dict[str, int]) -> None:
    out_txt = OUT / "train_recall_v27.txt"
    rows: list[str] = []
    for line in TRAIN_TXT.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        src_image = Path(line.strip())
        rel = src_image.relative_to(SRC / "images")
        out_image = OUT / "images" / rel
        if out_image.exists():
            rows.append(str(out_image.absolute()))
    out_txt.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    counts["train_recall_rows"] = len(rows)


def write_yaml() -> None:
    names = {
        0: "appliance",
        1: "bathtub",
        2: "column",
        3: "equipment",
        4: "generic_symbol",
        5: "shower",
        6: "sink",
        7: "stair",
        8: "table",
    }
    text = [
        f"path: {OUT.resolve()}",
        f"train: {str((OUT / 'train_recall_v27.txt').resolve())}",
        "val: images/val",
        "test: images/locked",
        "names:",
    ]
    text.extend(f"  {idx}: {name}" for idx, name in names.items())
    (OUT / "data.yaml").write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUT))
    args = parser.parse_args()
    OUT = Path(args.output)
    counts: dict[str, int] = defaultdict(int)
    for split in ("train", "val", "locked"):
        copy_images_and_labels(split, counts)
    build_train_txt(counts)
    write_yaml()
    manifest = {
        "dataset": str(OUT.relative_to(ROOT) if OUT.is_relative_to(ROOT) else OUT),
        "source": str(SRC.relative_to(ROOT)),
        "train_source": str(TRAIN_TXT.relative_to(ROOT)),
        "label_mode": "bbox_to_rectangle_polygon_segmentation",
        "claim_boundary": "Rectangular pseudo-masks from offline bbox supervision. Runtime model input remains raster pixels only.",
        "counts": counts,
        "data_yaml": str((OUT / "data.yaml").relative_to(ROOT) if (OUT / "data.yaml").is_relative_to(ROOT) else OUT / "data.yaml"),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
