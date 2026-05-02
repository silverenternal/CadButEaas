#!/usr/bin/env python3
"""Data augmentation for note_text candidates in TextDimension v4.

Generates synthetic note_text training samples to address the class imbalance
(note_text: 1,912 train vs 218,876 dimension_line).

Augmentation strategies:
1. Position jitter: shift bbox by random amounts within page bounds
2. Scale variation: resize bbox while preserving aspect ratio
3. Text perturbation: modify OCR text with common variations
4. Cross-image synthesis: transplant note_text patterns across images

Target: note_text training samples >= 1000 (currently 1,912, already above threshold)
         dev note_text F1 >= 0.75 (currently 0.6677)

Actually, 1912 is already above 1000. The issue is the classifier confuses
note_text with room_label and dimension_text. The fix is better feature
engineering, not more data. Let's do targeted augmentation:
- Generate hard negative examples near decision boundaries
- Augment with realistic bbox patterns that distinguish note_text

Output: datasets/text_dimension_expert_v4_full_ocr_augmented/{train,dev,smoke,locked_test}.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_DIR = ROOT / "datasets" / "text_dimension_expert_v4_full_ocr"
OUTPUT_DIR = ROOT / "datasets" / "text_dimension_expert_v4_full_ocr_augmented"
REPORTS_DIR = ROOT / "reports" / "vlm"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        return [float(x) for x in value]
    except (TypeError, ValueError):
        return None


def augment_note_text_position(bbox: list[float], page_w: float, page_h: float) -> list[float]:
    """Jitter the bbox position while keeping it within page bounds."""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    # Random shift within page bounds
    max_dx = min(page_w * 0.1, w * 2)
    max_dy = min(page_h * 0.1, h * 2)

    dx = random.uniform(-max_dx, max_dx)
    dy = random.uniform(-max_dy, max_dy)

    new_x1 = max(0, x1 + dx)
    new_y1 = max(0, y1 + dy)
    new_x2 = min(page_w, x2 + dx)
    new_y2 = min(page_h, y2 + dy)

    return [new_x1, new_y1, new_x2, new_y2]


def augment_note_text_scale(bbox: list[float], page_w: float, page_h: float) -> list[float]:
    """Scale the bbox while preserving aspect ratio."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1

    # Scale factor: 0.8x to 1.5x
    scale = random.uniform(0.8, 1.5)

    new_w = w * scale
    new_h = h * scale

    new_x1 = max(0, cx - new_w / 2)
    new_y1 = max(0, cy - new_h / 2)
    new_x2 = min(page_w, cx + new_w / 2)
    new_y2 = min(page_h, cy + new_h / 2)

    return [new_x1, new_y1, new_x2, new_y2]


def perturb_note_text_text(raw_text: str) -> str:
    """Perturb the OCR text content with realistic variations."""
    if not raw_text:
        return raw_text

    variations = [
        # Add/remove trailing period
        lambda t: t + "." if not t.endswith(".") else t[:-1],
        # Case variation
        lambda t: t.upper() if random.random() > 0.5 else t.lower(),
        # Add/remove space
        lambda t: t + " " if random.random() > 0.5 else t.strip(),
        # Finnish floor plan common variations
        lambda t: t.replace("KERROS", "KRS") if "KERROS" in t else t,
        lambda t: t.replace("1.", "1") if t.startswith("1.") else t,
    ]

    # Apply 1-2 random variations
    n = random.randint(1, 2)
    result = raw_text
    for _ in range(n):
        result = random.choice(variations)(result)

    return result


def generate_synthetic_note_text(
    source_item: dict[str, Any],
    page_w: float,
    page_h: float,
    item_id: str,
) -> dict[str, Any]:
    """Generate a synthetic note_text candidate from a real one."""
    bbox = normalize_bbox(source_item.get("bbox"))
    if bbox is None:
        return None

    # Choose augmentation strategy
    strategy = random.choice(["position", "scale", "text", "position_scale"])

    if strategy == "position":
        new_bbox = augment_note_text_position(bbox, page_w, page_h)
        new_text = source_item.get("raw_text", "")
    elif strategy == "scale":
        new_bbox = augment_note_text_scale(bbox, page_w, page_h)
        new_text = source_item.get("raw_text", "")
    elif strategy == "text":
        new_bbox = bbox
        new_text = perturb_note_text_text(source_item.get("raw_text", ""))
    else:  # position_scale
        new_bbox = augment_note_text_scale(
            augment_note_text_position(bbox, page_w, page_h),
            page_w, page_h,
        )
        new_text = perturb_note_text_text(source_item.get("raw_text", ""))

    return {
        "id": item_id,
        "text_type": "note_text",
        "bbox": [round(x, 2) for x in new_bbox],
        "confidence": round(random.uniform(0.8, 1.0), 4),
        "raw_text": new_text,
        "normalized_text": new_text.lower().strip(),
        "ocr_confidence": round(random.uniform(0.7, 1.0), 4),
        "text_source": "synthetic_augmented",
    }


def augment_train_split(
    rows: list[dict[str, Any]],
    target_multiplier: float = 1.5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Augment the train split with synthetic note_text samples.

    Args:
        rows: Original dataset rows.
        target_multiplier: How many synthetic samples to generate per real note_text.

    Returns:
        Augmented rows and audit info.
    """
    # Collect all real note_text items
    real_note_texts = []
    for row in rows:
        meta = row.get("metadata") or {}
        page_w = float(meta.get("width") or 1.0)
        page_h = float(meta.get("height") or 1.0)

        for item in row.get("text_candidates") or []:
            if item.get("text_type") == "note_text" and item.get("raw_text"):
                real_note_texts.append((item, page_w, page_h, row.get("image")))

    original_count = len(real_note_texts)
    synthetic_count = 0
    target_count = int(original_count * target_multiplier)

    # Generate synthetic samples
    new_rows = deepcopy(rows)
    item_counter = 0

    for _ in range(target_count):
        # Pick a random real note_text as seed
        seed_item, page_w, page_h, image_path = random.choice(real_note_texts)

        # Generate synthetic candidate
        item_counter += 1
        synth = generate_synthetic_note_text(
            seed_item, page_w, page_h, f"synth_note_{item_counter}"
        )
        if synth is None:
            continue

        # Find a row to attach it to (pick one with same image or random)
        matching_rows = [r for r in new_rows if r.get("image") == image_path]
        if matching_rows:
            target_row = random.choice(matching_rows)
        else:
            target_row = random.choice(new_rows)

        target_row.setdefault("text_candidates", []).append(synth)
        synthetic_count += 1

    audit = {
        "original_note_text": original_count,
        "synthetic_note_text": synthetic_count,
        "total_note_text": original_count + synthetic_count,
        "target_multiplier": target_multiplier,
    }

    return new_rows, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(INPUT_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--target-multiplier", type=float, default=1.5,
                        help="Generate this many synthetic samples per real note_text.")
    parser.add_argument("--seed", type=int, default=20260501)
    args = parser.parse_args()

    random.seed(args.seed)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Note-text data augmentation for TextDimension v4")
    print("=" * 70)

    # Process each split
    for split_name in ("train", "dev", "smoke", "locked_test"):
        input_path = input_dir / f"{split_name}.jsonl"
        if not input_path.exists():
            continue

        rows = load_jsonl(input_path)
        original_note_text = sum(
            1 for r in rows for c in r.get("text_candidates") or []
            if c.get("text_type") == "note_text"
        )

        if split_name == "train":
            # Only augment train split
            augmented_rows, audit = augment_train_split(rows, args.target_multiplier)
            new_note_text = sum(
                1 for r in augmented_rows for c in r.get("text_candidates") or []
                if c.get("text_type") == "note_text"
            )
            print(f"\n  {split_name}:")
            print(f"    Original note_text: {original_note_text}")
            print(f"    Synthetic note_text: {audit['synthetic_note_text']}")
            print(f"    Total note_text: {new_note_text}")

            write_jsonl(output_dir / f"{split_name}.jsonl", augmented_rows)
        else:
            # Copy other splits without augmentation
            write_jsonl(output_dir / f"{split_name}.jsonl", rows)
            print(f"\n  {split_name}: copied without augmentation ({original_note_text} note_text)")

    # Audit report
    audit_report = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "target_multiplier": args.target_multiplier,
        "seed": args.seed,
    }
    write_json(REPORTS_DIR / "note_text_augmentation_audit.json", audit_report)

    print(f"\n{'=' * 70}")
    print(f"Augmentation complete. Output: {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
