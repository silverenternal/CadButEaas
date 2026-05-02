#!/usr/bin/env python3
"""Create train-only FloorPlanCAD door morphology augmentations."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variants", default="wide,tall,large")
    parser.add_argument("--max-augmented-samples", type=int, default=240)
    parser.add_argument("--canvas-size", type=float, default=1000.0)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    train = load_jsonl(input_dir / "train.jsonl")
    augmented = build_augmented_train(train, variants, args.max_augmented_samples, args.canvas_size)
    write_jsonl(output_dir / "train.jsonl", augmented)
    for split in ("dev", "smoke"):
        (output_dir / f"{split}.jsonl").write_text((input_dir / f"{split}.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "variants": variants,
        "policy": "Only train split is augmented. Dev and smoke are copied unchanged.",
        "original_train_samples": len(train),
        "augmented_train_samples": len(augmented),
        "added_samples": len(augmented) - len(train),
    }
    (output_dir / "augmentation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_augmented_train(
    samples: list[dict[str, Any]],
    variants: list[str],
    max_augmented_samples: int,
    canvas_size: float,
) -> list[dict[str, Any]]:
    output = list(samples)
    added = 0
    for sample in samples:
        if added >= max_augmented_samples:
            break
        nodes = sample.get("nodes") or []
        door_indices = [index for index, node in enumerate(nodes) if str(node.get("label")) == "door"]
        if not door_indices:
            continue
        for variant in variants:
            if added >= max_augmented_samples:
                break
            augmented = copy.deepcopy(sample)
            augmented["augmentation"] = {"type": "door_morphology", "variant": variant}
            augmented["image"] = sample.get("image")
            changed = 0
            for index in door_indices:
                node = augmented["nodes"][index]
                if augment_door_node(node, variant, canvas_size):
                    changed += 1
            if changed:
                output.append(augmented)
                added += 1
    return output


def augment_door_node(node: dict[str, Any], variant: str, canvas_size: float) -> bool:
    features = node.get("features")
    if not isinstance(features, dict):
        return False
    bbox = features.get("bbox")
    if not isinstance(bbox, list) or len(bbox) < 4:
        return False
    x1, y1, x2, y2 = [float(value or 0.0) for value in bbox[:4]]
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    if variant == "wide":
        new_width, new_height = width * 1.8, height * 1.05
    elif variant == "tall":
        new_width, new_height = width * 1.05, height * 1.8
    elif variant == "large":
        new_width, new_height = width * 1.45, height * 1.45
    else:
        raise ValueError(f"Unknown variant: {variant}")
    new_width = min(max(new_width, 1.0), canvas_size)
    new_height = min(max(new_height, 1.0), canvas_size)
    nx1 = clamp(cx - new_width * 0.5, 0.0, canvas_size)
    ny1 = clamp(cy - new_height * 0.5, 0.0, canvas_size)
    nx2 = clamp(cx + new_width * 0.5, 0.0, canvas_size)
    ny2 = clamp(cy + new_height * 0.5, 0.0, canvas_size)
    if nx2 - nx1 < 1.0 or ny2 - ny1 < 1.0:
        return False
    features["bbox"] = [round(nx1, 3), round(ny1, 3), round(nx2, 3), round(ny2, 3)]
    features["centroid"] = [round((nx1 + nx2) * 0.5, 3), round((ny1 + ny2) * 0.5, 3)]
    width = nx2 - nx1
    height = ny2 - ny1
    features["length"] = max(width, height)
    features["orientation"] = orientation_for(width, height)
    area_frac = (width * height) / max(canvas_size * canvas_size, 1.0)
    features["log_area_frac"] = math.log1p(area_frac)
    features["log_length_frac"] = math.log1p(max(width, height) / max(canvas_size, 1.0))
    features["aspect_log"] = math.log(max(width, 1e-6) / max(height, 1e-6))
    features["se2_width"] = width / max(canvas_size, 1.0)
    features["se2_height"] = height / max(canvas_size, 1.0)
    features["se2_area"] = area_frac
    features["door_morph_augmented"] = 1.0
    return True


def orientation_for(width: float, height: float) -> str:
    ratio = width / max(height, 1e-6)
    if ratio >= 3.0:
        return "horizontal"
    if ratio <= 1.0 / 3.0:
        return "vertical"
    return "rectangular"


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
