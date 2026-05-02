#!/usr/bin/env python3
"""OCR teacher v5: Use source dataset text field + EasyOCR for empty cases.

The source cadstruct_cubicasa5k_moe dataset has:
- Pixel-space bboxes for text candidates  
- A 'text' field that's sometimes populated (28%), sometimes empty (72%)

Strategy:
1. For text candidates with non-empty 'text' in source: use it directly
2. For text candidates with empty 'text': run EasyOCR and match by bbox overlap

This populates raw_text for the TextDimension v3 training dataset.

Output:
  datasets/text_dimension_expert_v3_ocr_augmented/{train,dev,smoke}.jsonl
  reports/vlm/ocr_teacher_v5_audit.json
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
SOURCE_DIR = ROOT / "datasets" / "cadstruct_cubicasa5k_moe"
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


def normalize_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def bbox_iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / max(union, 1e-6)


def resolve_image_path(image_path_str: str) -> Path | None:
    p = Path(image_path_str)
    if p.is_absolute() and p.exists():
        return p
    candidate = ROOT / p
    if candidate.exists():
        return candidate
    return None


def build_source_text_index(split_name: str) -> dict[str, dict[str, Any]]:
    """Build an index of text candidates from source dataset keyed by image path.
    
    Returns {image_path: {candidate_id: {bbox, text, text_type, ...}}}
    """
    source_path = SOURCE_DIR / f"{split_name}.jsonl"
    if not source_path.exists():
        return {}
    
    index = {}
    with source_path.open() as f:
        for line in f:
            row = json.loads(line)
            img = row.get("image_path") or row.get("image") or ""
            if not img:
                continue
            
            expected = row.get("expected_json", {})
            for tc in expected.get("text_candidates", []):
                tc_id = str(tc.get("id", ""))
                if not tc_id:
                    continue
                
                if img not in index:
                    index[img] = {}
                index[img][tc_id] = {
                    "bbox": tc.get("bbox"),
                    "text": tc.get("text", ""),
                    "text_type": tc.get("text_type"),
                }
    
    return index


def main() -> int:
    print("=" * 70)
    print("OCR Teacher v5: Source text field + EasyOCR for empty cases")
    print("=" * 70)

    # Initialize EasyOCR for empty text cases
    print("\n1. Initializing EasyOCR...")
    reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    print("   EasyOCR loaded.")

    # Load dataset splits
    print("\n2. Loading dataset splits...")
    splits = {}
    for split_name in ("train", "dev", "smoke"):
        path = DATASET_DIR / f"{split_name}.jsonl"
        if path.exists():
            splits[split_name] = load_jsonl(path)
            print(f"   {split_name}: {len(splits[split_name])} records")

    # Build source text index
    print("\n3. Building source text index...")
    source_index = {}
    for split_name in splits:
        idx = build_source_text_index(split_name)
        source_index[split_name] = idx
        print(f"   {split_name}: {len(idx)} images indexed")

    # Process each split
    total_images = sum(len(rows) for rows in splits.values())
    print(f"\n4. Processing {total_images} images...")
    t0 = time.time()

    audit = {
        "total_records": total_images,
        "total_text_candidates": 0,
        "from_source_text": 0,
        "from_ocr": 0,
        "still_empty": 0,
        "ocr_errors": 0,
        "per_split": {},
    }

    for split_name in ("smoke", "dev", "train"):
        if split_name not in splits:
            continue
        rows = splits[split_name]
        split_idx = source_index.get(split_name, {})
        split_audit = {
            "records": len(rows),
            "text_candidates": 0,
            "from_source_text": 0,
            "from_ocr": 0,
            "still_empty": 0,
            "ocr_detections": 0,
        }

        for idx, row in enumerate(rows):
            image_path_str = row.get("image") or row.get("image_path") or ""
            if not image_path_str:
                continue

            text_candidates = row.get("text_candidates") or []
            split_audit["text_candidates"] += len(text_candidates)

            # Get source text for this image
            img_source = split_idx.get(image_path_str, {})

            # First pass: use source text where available
            for tc in text_candidates:
                tc_id = str(tc.get("id", ""))
                source_entry = img_source.get(tc_id)
                
                if source_entry and source_entry["text"] and source_entry["text"].strip():
                    tc["raw_text"] = source_entry["text"]
                    tc["normalized_text"] = source_entry["text"].lower().strip()
                    tc["text_source"] = "dataset"
                    split_audit["from_source_text"] += 1
                    audit["from_source_text"] += 1

            # Second pass: run EasyOCR for remaining empty text
            empty_candidates = [
                tc for tc in text_candidates 
                if not tc.get("raw_text") or not tc["raw_text"].strip()
            ]
            
            if not empty_candidates:
                continue

            # Run EasyOCR
            resolved = resolve_image_path(image_path_str)
            if resolved is None:
                for tc in empty_candidates:
                    split_audit["still_empty"] += 1
                    audit["still_empty"] += 1
                continue

            try:
                ocr_results = reader.readtext(str(resolved), detail=1)
            except Exception:
                audit["ocr_errors"] += 1
                for tc in empty_candidates:
                    split_audit["still_empty"] += 1
                    audit["still_empty"] += 1
                continue

            split_audit["ocr_detections"] += len(ocr_results)

            # Convert OCR results to pixel bboxes
            ocr_items = []
            for ocr_bbox, ocr_text, ocr_conf in ocr_results:
                if not ocr_text or not ocr_text.strip():
                    continue
                xs = [p[0] for p in ocr_bbox]
                ys = [p[1] for p in ocr_bbox]
                pixel_bbox = [min(xs), min(ys), max(xs), max(ys)]
                ocr_items.append({
                    "text": ocr_text.strip(),
                    "confidence": ocr_conf,
                    "bbox": pixel_bbox,
                })

            # Match empty candidates to OCR by IoU
            for tc in empty_candidates:
                tc_bbox = normalize_bbox(tc.get("bbox"))
                if tc_bbox is None:
                    split_audit["still_empty"] += 1
                    audit["still_empty"] += 1
                    continue

                best_ocr = None
                best_iou = 0.0
                for ocr_item in ocr_items:
                    iou = bbox_iou(tc_bbox, ocr_item["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_ocr = ocr_item

                if best_ocr and best_iou > 0.05:  # Low threshold for small text boxes
                    tc["raw_text"] = best_ocr["text"]
                    tc["normalized_text"] = best_ocr["text"].lower().strip()
                    tc["ocr_confidence"] = best_ocr["confidence"]
                    tc["ocr_iou"] = round(best_iou, 4)
                    tc["text_source"] = "ocr"
                    split_audit["from_ocr"] += 1
                    audit["from_ocr"] += 1
                else:
                    split_audit["still_empty"] += 1
                    audit["still_empty"] += 1

            if (idx + 1) % 50 == 0:
                elapsed = time.time() - t0
                total_pop = split_audit["from_source_text"] + split_audit["from_ocr"]
                print(f"   [{split_name}] {idx+1}/{len(rows)} ({elapsed:.0f}s) - populated {total_pop}/{split_audit['text_candidates']}")

        audit["per_split"][split_name] = split_audit
        total_pop = split_audit["from_source_text"] + split_audit["from_ocr"]
        pct = total_pop / max(split_audit["text_candidates"], 1) * 100
        print(f"   [{split_name}] Done: {pct:.1f}% populated (source={split_audit['from_source_text']}, ocr={split_audit['from_ocr']})")

    # Write augmented datasets
    print(f"\n5. Writing to {OUTPUT_DIR}...")
    for split_name, rows in splits.items():
        write_jsonl(OUTPUT_DIR / f"{split_name}.jsonl", rows)

    # Write audit
    audit["elapsed_seconds"] = round(time.time() - t0, 1)
    write_json(REPORTS_DIR / "ocr_teacher_v5_audit.json", audit)
    
    print(f"\n{'=' * 70}")
    total_pop = audit["from_source_text"] + audit["from_ocr"]
    print(f"OCR Teacher v5: {total_pop}/{audit['total_text_candidates']} populated")
    print(f"  From source text: {audit['from_source_text']}")
    print(f"  From OCR: {audit['from_ocr']}")
    print(f"  Still empty: {audit['still_empty']}")
    print(f"{'=' * 70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
