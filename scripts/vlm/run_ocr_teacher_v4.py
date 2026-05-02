#!/usr/bin/env python3
"""OCR teacher v4: Simple center-distance matching in pixel space.

Strategy:
1. Run EasyOCR on PNG image → get (text, pixel_bbox) pairs
2. For each dataset text_candidate with synthetic bbox:
   - Convert to pixel coords if needed (detect if bbox is in SVG or pixel space)
   - Find nearest OCR detection by center distance
   - Copy OCR text to raw_text

This avoids the SVG coordinate complexity entirely.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

try:
    import easyocr
except ImportError:
    print("ERROR: easyocr is required. pip install easyocr")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow is required. pip install Pillow")
    sys.exit(1)

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


def resolve_image_path(image_path_str: str) -> Path | None:
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


def detect_bbox_space(bbox: list[float], img_w: int, img_h: int) -> str:
    """Detect if bbox is in pixel coords or some other space (SVG/normalized).
    
    Returns 'pixel', 'normalized', or 'other'.
    """
    x1, y1, x2, y2 = bbox
    # Check if values are all 0-1 range
    if all(0 <= v <= 1 for v in bbox):
        return 'normalized'
    # Check if values exceed image dimensions significantly
    if x2 > img_w * 2 or y2 > img_h * 2 or x1 < -img_w or y1 < -img_h:
        return 'other'  # Likely SVG coords
    # Likely pixel coords
    return 'pixel'


def convert_to_pixel_bbox(bbox: list[float], space: str, img_w: int, img_h: int) -> list[float]:
    """Convert bbox to pixel coordinates."""
    if space == 'pixel':
        return list(bbox)
    elif space == 'normalized':
        return [bbox[0] * img_w, bbox[1] * img_h, bbox[2] * img_w, bbox[3] * img_h]
    else:
        # For 'other' (SVG) - we can't reliably convert without viewBox
        # Return as-is and let distance matching handle it
        return list(bbox)


def center_distance(bbox1: list[float], bbox2: list[float]) -> float:
    cx1 = (bbox1[0] + bbox1[2]) / 2.0
    cy1 = (bbox1[1] + bbox1[3]) / 2.0
    cx2 = (bbox2[0] + bbox2[2]) / 2.0
    cy2 = (bbox2[1] + bbox2[3]) / 2.0
    return math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)


def main() -> int:
    print("=" * 70)
    print("OCR Teacher v4: EasyOCR + center-distance matching")
    print("=" * 70)

    print("\n1. Initializing EasyOCR...")
    reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    print("   EasyOCR loaded.")

    print("\n2. Loading dataset splits...")
    splits = {}
    for split_name in ("train", "dev", "smoke"):
        path = DATASET_DIR / f"{split_name}.jsonl"
        if path.exists():
            splits[split_name] = load_jsonl(path)
            print(f"   {split_name}: {len(splits[split_name])} records")

    total_images = sum(len(rows) for rows in splits.values())
    print(f"\n3. Processing {total_images} images...")
    t0 = time.time()

    audit = {
        "total_records": total_images,
        "total_text_candidates": 0,
        "raw_text_populated": 0,
        "raw_text_still_empty": 0,
        "ocr_errors": 0,
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
            "ocr_detections": 0,
        }

        for idx, row in enumerate(rows):
            image_path_str = row.get("image") or row.get("image_path") or ""
            if not image_path_str:
                continue

            resolved = resolve_image_path(image_path_str)
            if resolved is None:
                continue

            try:
                img = Image.open(resolved).convert("L")
                img_w, img_h = img.size
            except Exception:
                audit["ocr_errors"] += 1
                continue

            text_candidates = row.get("text_candidates") or []
            split_audit["text_candidates"] += len(text_candidates)

            # Run EasyOCR
            try:
                ocr_results = reader.readtext(str(resolved), detail=1)
            except Exception:
                audit["ocr_errors"] += 1
                continue

            split_audit["ocr_detections"] += len(ocr_results)

            # Convert OCR results to pixel bboxes
            ocr_items = []
            for ocr_bbox, ocr_text, ocr_conf in ocr_results:
                if not ocr_text or not ocr_text.strip():
                    continue
                # EasyOCR returns 4 corners: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                xs = [p[0] for p in ocr_bbox]
                ys = [p[1] for p in ocr_bbox]
                pixel_bbox = [min(xs), min(ys), max(xs), max(ys)]
                ocr_items.append({
                    "text": ocr_text.strip(),
                    "confidence": ocr_conf,
                    "bbox": pixel_bbox,
                })

            # Match each text candidate to nearest OCR detection
            for tc in text_candidates:
                tc_bbox = normalize_bbox(tc.get("bbox"))
                if tc_bbox is None:
                    split_audit["raw_text_still_empty"] += 1
                    audit["raw_text_still_empty"] += 1
                    continue

                # Detect coordinate space and convert to pixels
                space = detect_bbox_space(tc_bbox, img_w, img_h)
                pixel_tc_bbox = convert_to_pixel_bbox(tc_bbox, space, img_w, img_h)

                # Find nearest OCR detection
                best_ocr = None
                best_dist = float("inf")
                for ocr_item in ocr_items:
                    dist = center_distance(pixel_tc_bbox, ocr_item["bbox"])
                    if dist < best_dist:
                        best_dist = dist
                        best_ocr = ocr_item

                # Accept if within threshold (100 pixels)
                if best_ocr and best_dist < 100.0:
                    tc["raw_text"] = best_ocr["text"]
                    tc["normalized_text"] = best_ocr["text"].lower().strip()
                    tc["ocr_confidence"] = best_ocr["confidence"]
                    tc["ocr_center_dist"] = round(best_dist, 2)
                    split_audit["raw_text_populated"] += 1
                    audit["raw_text_populated"] += 1
                else:
                    split_audit["raw_text_still_empty"] += 1
                    audit["raw_text_still_empty"] += 1

            if (idx + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(f"   [{split_name}] {idx+1}/{len(rows)} ({elapsed:.0f}s) - populated {split_audit['raw_text_populated']}/{split_audit['text_candidates']}")

        audit["per_split"][split_name] = split_audit
        pct = split_audit["raw_text_populated"] / max(split_audit["text_candidates"], 1) * 100
        print(f"   [{split_name}] Done: {pct:.1f}% populated")

    # Write augmented datasets
    print(f"\n4. Writing to {OUTPUT_DIR}...")
    for split_name, rows in splits.items():
        write_jsonl(OUTPUT_DIR / f"{split_name}.jsonl", rows)

    # Write audit
    audit["elapsed_seconds"] = round(time.time() - t0, 1)
    write_json(REPORTS_DIR / "ocr_teacher_v4_audit.json", audit)
    
    print(f"\n{'=' * 70}")
    print(f"OCR Teacher v4: {audit['raw_text_populated']}/{audit['total_text_candidates']} populated")
    print(f"{'=' * 70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
