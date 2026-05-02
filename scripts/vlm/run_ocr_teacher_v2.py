#!/usr/bin/env python3
"""OCR teacher v2: fix SVG→pixel coordinate transform for text candidate matching.

The dataset text candidates have bboxes in SVG coordinate space (viewBox coords).
EasyOCR detects text in pixel space (PNG image coords).

This script:
1. Parses SVG viewBox to get SVG coordinate dimensions
2. Reads PNG image dimensions  
3. Computes SVG→pixel transform (proportional scaling)
4. Converts text candidate bboxes from SVG to pixel space
5. Matches against EasyOCR detections by IoU
6. Populates raw_text fields

Output:
  datasets/text_dimension_expert_v3_ocr_augmented/{train,dev,smoke}.jsonl
  reports/vlm/ocr_teacher_v2_audit.json
"""

from __future__ import annotations

import json
import math
import re
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


def get_svg_viewbox(svg_path: Path) -> tuple[float, float] | None:
    """Extract viewBox dimensions from SVG file. Returns (width, height) or None."""
    try:
        svg_text = svg_path.read_text(encoding="utf-8")
        match = re.search(r'viewBox="([^"]+)"', svg_text)
        if match:
            parts = match.group(1).split()
            if len(parts) == 4:
                return float(parts[2]), float(parts[3])
        # Fallback: try width/height attributes
        w_match = re.search(r'width="([^"]+)"', svg_text)
        h_match = re.search(r'height="([^"]+)"', svg_text)
        if w_match and h_match:
            return float(w_match.group(1)), float(h_match.group(1))
    except Exception:
        pass
    return None


def find_svg_path(image_path: Path) -> Path | None:
    """Find corresponding SVG file for a PNG image."""
    # For CubiCasa5K: .../6363/F1_original.png → .../6363/model.svg
    parent = image_path.parent
    svg_candidate = parent / "model.svg"
    if svg_candidate.exists():
        return svg_candidate
    # Search parent directories
    for p in image_path.parents:
        svg_candidate = p / "model.svg"
        if svg_candidate.exists():
            return svg_candidate
    return None


def svg_to_pixel_bbox(svg_bbox: list[float], svg_size: tuple[float, float],
                      png_size: tuple[int, int]) -> list[float]:
    """Convert bbox from SVG viewBox coords to PNG pixel coords.
    
    Uses proportional scaling. CubiCasa5K renders SVG→PNG with proportional
    fitting, so we compute scale factors and apply them.
    """
    svg_w, svg_h = svg_size
    png_w, png_h = png_size
    
    # Compute scale factors
    scale_x = png_w / max(svg_w, 1)
    scale_y = png_h / max(svg_h, 1)
    
    # Use minimum scale (proportional fitting preserves aspect ratio)
    scale = min(scale_x, scale_y)
    
    x1, y1, x2, y2 = svg_bbox
    return [
        x1 * scale,
        y1 * scale,
        x2 * scale,
        y2 * scale,
    ]


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


def convert_ocr_bbox(ocr_bbox: list) -> list[float]:
    """Convert EasyOCR bbox (4 corners) to [x1, y1, x2, y2] in pixel coords."""
    pts = ocr_bbox
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def resolve_image_path(image_path_str: str) -> Path | None:
    """Resolve image path, trying relative and absolute."""
    p = Path(image_path_str)
    if p.is_absolute() and p.exists():
        return p
    # Try relative to ROOT
    candidate = ROOT / p
    if candidate.exists():
        return candidate
    return None


def main() -> int:
    print("=" * 70)
    print("OCR Teacher v2: SVG→Pixel Coordinate Transform + EasyOCR Matching")
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
        "svg_coord_transforms": 0,
        "svg_not_found": 0,
        "per_split": {},
    }

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
            "iou_threshold": 0.1,
            "svg_transforms_used": 0,
            "svg_not_found": 0,
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
                audit["ocr_error"] += 1
                continue

            # Find SVG file for coordinate transform
            svg_path = find_svg_path(resolved)
            svg_size = None
            if svg_path:
                svg_size = get_svg_viewbox(svg_path)
                if svg_size:
                    audit["svg_coord_transforms"] += 1
                    split_audit["svg_transforms_used"] += 1
                else:
                    audit["svg_not_found"] += 1
                    split_audit["svg_not_found"] += 1
            else:
                audit["svg_not_found"] += 1
                split_audit["svg_not_found"] += 1

            text_candidates = row.get("text_candidates") or []
            split_audit["text_candidates_before"] += len(text_candidates)

            # Run OCR
            try:
                ocr_results = reader.readtext(str(resolved), detail=1)
            except Exception:
                audit["ocr_error"] += 1
                continue

            split_audit["ocr_detections_total"] += len(ocr_results)

            # Convert OCR results to pixel bboxes [x1, y1, x2, y2]
            ocr_items = []
            for ocr_bbox, ocr_text, ocr_conf in ocr_results:
                if not ocr_text or not ocr_text.strip():
                    continue
                abs_bbox = convert_ocr_bbox(ocr_bbox)
                ocr_items.append({
                    "text": ocr_text.strip(),
                    "confidence": ocr_conf,
                    "bbox": abs_bbox,
                })

            # Match each text candidate to best OCR detection
            for tc in text_candidates:
                tc_bbox_raw = tc.get("bbox")
                if tc_bbox_raw is None:
                    continue

                tc_bbox = normalize_bbox(tc_bbox_raw)
                if tc_bbox is None:
                    continue

                # Transform from SVG coords to pixel coords if we have SVG size
                if svg_size:
                    tc_bbox = svg_to_pixel_bbox(tc_bbox, svg_size, (img_w, img_h))

                # Compute IoU with each OCR detection
                best_ocr = None
                best_iou = 0.0
                for ocr_item in ocr_items:
                    iou = bbox_iou(tc_bbox, ocr_item["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_ocr = ocr_item

                if best_ocr and best_iou > 0.1:
                    tc["raw_text"] = best_ocr["text"]
                    tc["normalized_text"] = best_ocr["text"].lower().strip()
                    tc["ocr_confidence"] = best_ocr["confidence"]
                    tc["ocr_iou"] = round(best_iou, 4)
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
    write_json(REPORTS_DIR / "ocr_teacher_v2_audit.json", audit)
    print(f"\nAudit summary: {json.dumps(audit, indent=2)}")

    print(f"\n{'=' * 70}")
    print(f"OCR Teacher v2 complete: {audit['raw_text_populated']}/{audit['total_text_candidates']} raw_text populated")
    print(f"SVG coordinate transforms: {audit['svg_coord_transforms']}")
    print(f"{'=' * 70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
