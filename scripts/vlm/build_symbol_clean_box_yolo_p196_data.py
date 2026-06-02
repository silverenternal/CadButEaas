#!/usr/bin/env python3
"""Build clean box-label YOLO data lists by mapping P191/P195 lists onto v22 box labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_SEG = ROOT / "datasets/symbol_tiny_small_specialist_p191_yolo"
BOX = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21_yolo_v22"
OUT = ROOT / "datasets/symbol_clean_box_yolo_p196"


def read_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def map_to_box_path(image: str) -> str | None:
    p = Path(image)
    split = p.parent.name
    name = p.name
    candidate = BOX / "images" / split / name
    label = BOX / "labels" / split / p.with_suffix(".txt").name
    if candidate.exists() and label.exists() and label.read_text().strip():
        return str(candidate.resolve())
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_src = read_list(SRC_SEG / "train_p191.txt")
    val_src = read_list(SRC_SEG / "val_p191.txt")
    train = []
    missing_train = 0
    for image in train_src:
        mapped = map_to_box_path(image)
        if mapped:
            train.append(mapped)
        else:
            missing_train += 1
    val = []
    missing_val = 0
    seen_val = set()
    for image in val_src:
        mapped = map_to_box_path(image)
        if mapped and mapped not in seen_val:
            val.append(mapped); seen_val.add(mapped)
        else:
            missing_val += 1
    (out / "train_box_clean.txt").write_text("\n".join(train) + "\n")
    (out / "val_box_clean.txt").write_text("\n".join(val) + "\n")
    data_yaml = out / "data.yaml"
    data_yaml.write_text(
        "path: " + str(BOX.resolve()) + "\n"
        "train: " + str((out / "train_box_clean.txt").resolve()) + "\n"
        "val: " + str((out / "val_box_clean.txt").resolve()) + "\n"
        "test: images/locked\n"
        "names:\n"
        "  0: appliance\n  1: bathtub\n  2: column\n  3: equipment\n  4: generic_symbol\n  5: shower\n  6: sink\n  7: stair\n  8: table\n"
    )
    report = {
        "id": "P196_clean_box_yolo_data",
        "source_seg_list": str(SRC_SEG),
        "box_dataset": str(BOX),
        "outputs": {"data_yaml": str(data_yaml), "train": str(out / "train_box_clean.txt"), "val": str(out / "val_box_clean.txt")},
        "counts": {"train_in": len(train_src), "train_out": len(train), "train_missing_or_empty": missing_train, "val_in": len(val_src), "val_out": len(val), "val_missing_or_empty_or_duplicate": missing_val},
        "claim_boundary": "Uses box-label YOLO view to avoid current Ultralytics segmentation validation loss bug; still raster supervised training only.",
    }
    (out / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
