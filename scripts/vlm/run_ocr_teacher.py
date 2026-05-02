#!/usr/bin/env python3
"""OCR teacher: run EasyOCR on floorplan images and populate raw_text fields.

For each text candidate in the dataset, find the best-matching OCR detection
by IoU overlap and copy the OCR text into the candidate's raw_text field.

This bridges the gap where the source dataset has empty raw_text fields.

Output:
  datasets/text_dimension_expert_v2_ocr_augmented/train.jsonl (with raw_text populated)
  datasets/text_dimension_expert_v2_ocr_augmented/dev.jsonl
  datasets/text_dimension_expert_v2_ocr_augmented/smoke.jsonl
  reports/vlm/ocr_teacher_audit.json
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter
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
OUTPUT_DIR = ROOT / "datasets" / "text_dimension_expert_v2_ocr_augmented"
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


def convert_ocr_bbox(ocr_bbox, img_width: float, img_height: float) -> list[float]:
    """Convert EasyOCR bbox (4 corners) to [x1, y1, x2, y2] in image coords."""
    # EasyOCR returns [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    pts = ocr_bbox
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def main() -> int:
    print("=" * 70)
    print("OCR Teacher: Populate raw_text via EasyOCR spatial matching")
    print("=" * 70)

    # 1. Initialize OCR
    print("\n1. Initializing EasyOCR...")
    reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    print("   EasyOCR loaded.")

    # 2. Load dataset splits
    print("\n2. Loading dataset splits...")
    splits = {}
    for split_name in ("train", "dev", "smoke"):
        path = DATASET_DIR / f"{split_name}.jsonl"
        if path.exists():
            splits[split_name] = load_jsonl(path)
            print(f"   {split_name}: {len(splits[split_name])} records")

    # 3. Process each split
    total_images = sum(len(rows) for rows in splits.values())
    print(f"\n3. Processing {total_images} images with EasyOCR...")
    t0 = time.time()

    audit = {
        "total_records": total_images,
        "total_text_candidates": 0,
        "ocr_matched": 0,
        "ocr_no_detection": 0,
        "ocr_error": 0,
        "ocr_empty_text": 0,
        "raw_text_populated": 0,
        "per_split": {},
    }

    # Process only smoke first for quick validation
    for split_name in ("smoke", "dev", "train"):
        if split_name not in splits:
            continue
        rows = splits[split_name]
        split_audit = {
            "records": len(rows),
            "text_candidates_before": 0,
            "text_candidates_after": 0,
            "raw_text_populated": 0,
            "raw_text_empty": 0,
            "ocr_detections_total": 0,
            "iou_threshold": 0.3,
        }

        for idx, row in enumerate(rows):
            image_path = row.get("image_path") or row.get("image") or ""
            if isinstance(image_path, str):
                # Resolve path
                candidates = [image_path]
                if image_path.startswith("datasets/"):
                    candidates.append(str(ROOT / image_path))
                resolved = None
                for c in candidates:
                    if Path(c).exists():
                        resolved = c
                        break
                if resolved is None:
                    continue
            else:
                continue

            try:
                img = Image.open(resolved).convert("L")
                img_w, img_h = img.size
            except Exception:
                audit["ocr_error"] += 1
                continue

            text_candidates = row.get("text_candidates") or []
            split_audit["text_candidates_before"] += len(text_candidates)

            # Run OCR
            try:
                ocr_results = reader.readtext(resolved, detail=1)
            except Exception:
                audit["ocr_error"] += 1
                continue

            split_audit["ocr_detections_total"] += len(ocr_results)

            # Convert OCR results to normalized bboxes
            ocr_items = []
            for ocr_bbox, ocr_text, ocr_conf in ocr_results:
                abs_bbox = convert_ocr_bbox(ocr_bbox, img_w, img_h)
                norm_bbox = [
                    abs_bbox[0] / max(img_w, 1),
                    abs_bbox[1] / max(img_h, 1),
                    abs_bbox[2] / max(img_w, 1),
                    abs_bbox[3] / max(img_h, 1),
                ]
                ocr_items.append({
                    "text": ocr_text.strip(),
                    "confidence": ocr_conf,
                    "bbox_abs": abs_bbox,
                    "bbox_norm": norm_bbox,
                })

            # Match each text candidate to best OCR detection by center distance
            for tc in text_candidates:
                tc_bbox = normalize_bbox(tc.get("bbox"))
                if tc_bbox is None:
                    # Handle already-normalized bboxes (0-1 range)
                    raw = tc.get("bbox", [0, 0, 0, 0])
                    # If values are all < 1, assume normalized
                    if all(v < 1.0 for v in raw):
                        tc_bbox = list(raw)
                    else:
                        tc_bbox = [
                            raw[0] / max(img_w, 1),
                            raw[1] / max(img_h, 1),
                            raw[2] / max(img_w, 1),
                            raw[3] / max(img_h, 1),
                        ]

                tc_cx = (tc_bbox[0] + tc_bbox[2]) / 2.0
                tc_cy = (tc_bbox[1] + tc_bbox[3]) / 2.0
                tc_w = tc_bbox[2] - tc_bbox[0]
                tc_h = tc_bbox[3] - tc_bbox[1]
                tc_scale = max(tc_w, tc_h, 0.01)  # avoid div by zero

                best_ocr = None
                best_score = float("inf")
                for ocr_item in ocr_items:
                    ob = ocr_item["bbox_norm"]
                    ocr_cx = (ob[0] + ob[2]) / 2.0
                    ocr_cy = (ob[1] + ob[3]) / 2.0
                    # Normalized center distance
                    dist = math.sqrt((tc_cx - ocr_cx) ** 2 + (tc_cy - ocr_cy) ** 2)
                    # Score: distance normalized by candidate size
                    score = dist / tc_scale
                    # Also check IoU as a secondary signal
                    iou = bbox_iou(tc_bbox, ob)
                    # Accept if center is close (score < 2.0) OR IoU > 0.1
                    if score < best_score and (score < 2.0 or iou > 0.1):
                        best_score = score
                        best_ocr = ocr_item

                if best_ocr and best_ocr["text"] and best_score < 3.0:
                    tc["raw_text"] = best_ocr["text"]
                    tc["normalized_text"] = best_ocr["text"].lower().strip()
                    tc["ocr_confidence"] = best_ocr["confidence"]
                    tc["ocr_match_score"] = round(best_score, 4)
                    split_audit["raw_text_populated"] += 1
                    audit["raw_text_populated"] += 1
                else:
                    split_audit["raw_text_empty"] += 1

            audit["total_text_candidates"] += len(text_candidates)
            split_audit["text_candidates_after"] += len(text_candidates)

            if (idx + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"   [{split_name}] {idx+1}/{len(rows)} ({elapsed:.0f}s) - populated {split_audit['raw_text_populated']} so far")

        audit["per_split"][split_name] = split_audit
        print(f"   [{split_name}] Done: {split_audit['raw_text_populated']}/{split_audit['text_candidates_after']} raw_text populated")

    # 4. Write augmented datasets
    print(f"\n4. Writing augmented datasets to {OUTPUT_DIR}...")
    for split_name, rows in splits.items():
        write_jsonl(OUTPUT_DIR / f"{split_name}.jsonl", rows)
        print(f"   Wrote {OUTPUT_DIR / f'{split_name}.jsonl'}")

    # 5. Write audit
    audit["elapsed_seconds"] = round(time.time() - t0, 1)
    write_json(REPORTS_DIR / "ocr_teacher_audit.json", audit)
    print(f"\n   Audit: {json.dumps(audit, indent=2)}")

    print(f"\n{'=' * 70}")
    print(f"OCR Teacher complete: {audit['raw_text_populated']}/{audit['total_text_candidates']} raw_text populated")
    print(f"{'=' * 70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
