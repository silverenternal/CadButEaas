#!/usr/bin/env python3
"""OCR teacher v3: Extract text directly from SVG <text> elements.

The dataset text_candidates have synthetic bboxes with empty raw_text.
The SVG files contain actual <text> elements with real text content.

Strategy:
1. Parse SVG <text> elements to get (text_content, svg_bbox)
2. For each dataset text_candidate, find nearest SVG text element
3. Copy the SVG text into raw_text

This bypasses the OCR matching problem entirely - the text is already in the SVG.

Output:
  datasets/text_dimension_expert_v3_ocr_augmented/{train,dev,smoke}.jsonl
  reports/vlm/ocr_teacher_v3_audit.json
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = ROOT / "datasets" / "cadstruct_text_dimensions_v1"
OUTPUT_DIR = ROOT / "datasets" / "text_dimension_expert_v3_ocr_augmented"
REPORTS_DIR = ROOT / "reports" / "vlm"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_svg_texts(svg_path: Path) -> list[dict[str, Any]]:
    """Parse <text> elements from SVG file.
    
    Returns list of {text, x, y, svg_bbox} where svg_bbox is [x1, y1, x2, y2].
    """
    try:
        svg_text = svg_path.read_text(encoding="utf-8")
    except Exception:
        return []
    
    results = []
    # Match <text> elements with x, y attributes
    pattern = r'<text[^>]*?(?:\sx="([^"]*)")?[^>]*?(?:\sy="([^"]*)")?[^>]*>(.*?)</text>'
    for match in re.finditer(pattern, svg_text, re.DOTALL):
        x_str, y_str, content = match.groups()
        content = content.strip()
        if not content:
            continue
        
        # Parse x, y (can be space-separated lists)
        try:
            x = float(x_str.split()[0]) if x_str else 0.0
            y = float(y_str.split()[0]) if y_str else 0.0
        except (ValueError, IndexError):
            x, y = 0.0, 0.0
        
        # Estimate bbox size from text content (rough approximation)
        char_width = 5.0  # approximate character width in SVG coords
        text_len = len(content)
        w = text_len * char_width
        h = 10.0  # approximate text height
        
        results.append({
            "text": content,
            "x": x,
            "y": y,
            "svg_bbox": [x, y, x + w, y + h],
        })
    
    return results


def find_svg_path(image_path: Path) -> Path | None:
    """Find corresponding SVG file for an image."""
    parent = image_path.parent
    svg_candidate = parent / "model.svg"
    if svg_candidate.exists():
        return svg_candidate
    for p in image_path.parents:
        svg_candidate = p / "model.svg"
        if svg_candidate.exists():
            return svg_candidate
    return None


def center_distance(bbox1: list[float], bbox2: list[float]) -> float:
    """Compute center-to-center distance between two bboxes."""
    cx1 = (bbox1[0] + bbox1[2]) / 2.0
    cy1 = (bbox1[1] + bbox1[3]) / 2.0
    cx2 = (bbox2[0] + bbox2[2]) / 2.0
    cy2 = (bbox2[1] + bbox2[3]) / 2.0
    return math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)


def resolve_image_path(image_path_str: str) -> Path | None:
    """Resolve image path."""
    p = Path(image_path_str)
    if p.is_absolute() and p.exists():
        return p
    candidate = ROOT / p
    if candidate.exists():
        return candidate
    return None


def normalize_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def main() -> int:
    print("=" * 70)
    print("OCR Teacher v3: Extract text from SVG <text> elements")
    print("=" * 70)

    # Load dataset splits
    print("\n1. Loading dataset splits...")
    splits = {}
    for split_name in ("train", "dev", "smoke"):
        path = DATASET_DIR / f"{split_name}.jsonl"
        if path.exists():
            splits[split_name] = load_jsonl(path)
            print(f"   {split_name}: {len(splits[split_name])} records")

    # Process each split
    total_images = sum(len(rows) for rows in splits.values())
    print(f"\n2. Processing {total_images} images...")
    t0 = time.time()

    audit = {
        "total_records": total_images,
        "total_text_candidates": 0,
        "svg_texts_found": 0,
        "svg_not_found": 0,
        "raw_text_populated": 0,
        "raw_text_still_empty": 0,
        "per_split": {},
    }

    for split_name in ("smoke", "dev", "train"):
        if split_name not in splits:
            continue
        rows = splits[split_name]
        split_audit = {
            "records": len(rows),
            "text_candidates": 0,
            "raw_text_populated": 0,
            "raw_text_still_empty": 0,
            "svg_texts_extracted": 0,
            "svg_not_found": 0,
        }

        for idx, row in enumerate(rows):
            image_path_str = row.get("image") or row.get("image_path") or ""
            if not image_path_str:
                continue

            resolved = resolve_image_path(image_path_str)
            if resolved is None:
                continue

            # Find and parse SVG file
            svg_path = find_svg_path(resolved)
            svg_texts = []
            if svg_path:
                svg_texts = parse_svg_texts(svg_path)
                split_audit["svg_texts_extracted"] += len(svg_texts)
                audit["svg_texts_found"] += len(svg_texts)
            else:
                split_audit["svg_not_found"] += 1
                audit["svg_not_found"] += 1

            # Match text candidates to SVG text elements
            text_candidates = row.get("text_candidates") or []
            split_audit["text_candidates"] += len(text_candidates)

            for tc in text_candidates:
                tc_bbox = normalize_bbox(tc.get("bbox"))
                if tc_bbox is None:
                    split_audit["raw_text_still_empty"] += 1
                    audit["raw_text_still_empty"] += 1
                    continue

                # Find nearest SVG text element by center distance
                best_svg = None
                best_dist = float("inf")
                for svg_item in svg_texts:
                    dist = center_distance(tc_bbox, svg_item["svg_bbox"])
                    if dist < best_dist:
                        best_dist = dist
                        best_svg = svg_item

                # Accept if within reasonable distance (50 SVG units)
                if best_svg and best_dist < 50.0:
                    tc["raw_text"] = best_svg["text"]
                    tc["normalized_text"] = best_svg["text"].lower().strip()
                    tc["svg_match_distance"] = round(best_dist, 2)
                    split_audit["raw_text_populated"] += 1
                    audit["raw_text_populated"] += 1
                else:
                    split_audit["raw_text_still_empty"] += 1
                    audit["raw_text_still_empty"] += 1

            if (idx + 1) % 100 == 0:
                elapsed = time.time() - t0
                print(f"   [{split_name}] {idx+1}/{len(rows)} ({elapsed:.0f}s) - populated {split_audit['raw_text_populated']} so far")

        audit["per_split"][split_name] = split_audit
        pct = split_audit["raw_text_populated"] / max(split_audit["text_candidates"], 1) * 100
        print(f"   [{split_name}] Done: {split_audit['raw_text_populated']}/{split_audit['text_candidates']} raw_text populated ({pct:.1f}%)")

    # Write augmented datasets
    print(f"\n3. Writing augmented datasets to {OUTPUT_DIR}...")
    for split_name, rows in splits.items():
        write_jsonl(OUTPUT_DIR / f"{split_name}.jsonl", rows)

    # Write audit
    audit["elapsed_seconds"] = round(time.time() - t0, 1)
    write_json(REPORTS_DIR / "ocr_teacher_v3_audit.json", audit)
    
    print(f"\n{'=' * 70}")
    print(f"OCR Teacher v3 complete: {audit['raw_text_populated']}/{audit['total_text_candidates']} raw_text populated")
    print(f"SVG texts extracted: {audit['svg_texts_found']}")
    print(f"{'=' * 70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
