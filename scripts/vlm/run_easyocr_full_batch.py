#!/usr/bin/env python3
"""EasyOCR full-batch runner for TextDimension v4.

Runs EasyOCR on all PNG images across all splits, populating raw_text for
text-bearing candidates that currently lack it.

Strategy:
1. For candidates with valid SVG bboxes (area >= 100 px² after transform):
   crop-based OCR + IoU matching (existing approach from v2).
2. For candidates with placeholder bboxes ([0,-10,10,0] etc.):
   run EasyOCR on the full image, then match by:
   a) Text content similarity (normalized string match against SVG text elements)
   b) Spatial proximity to the candidate's approximate SVG position

Output:
  datasets/text_dimension_expert_v4_full_ocr/{train,dev,smoke,locked_test}.jsonl
  reports/vlm/easyocr_full_batch_audit.json
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
import xml.etree.ElementTree as ET
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
INPUT_DIR = ROOT / "datasets" / "text_dimension_expert_v3_ocr_augmented"
OUTPUT_DIR = ROOT / "datasets" / "text_dimension_expert_v4_full_ocr"
REPORTS_DIR = ROOT / "reports" / "vlm"

TEXT_BEARING_TYPES = {"dimension_text", "room_label", "note_text", "legend_text", "callout"}

# Minimum pixel area for a bbox to be considered "valid" for crop-based OCR
MIN_VALID_PIXEL_AREA = 50

# IoU threshold for matching
IOU_THRESHOLD = 0.1

# Padding ratio for crops
CROP_PAD_RATIO = 0.5
MIN_CROP_SIZE = 32


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


def get_svg_viewbox(svg_path: Path) -> tuple[float, float] | None:
    try:
        svg_text = svg_path.read_text(encoding="utf-8")
        match = re.search(r'viewBox="([^"]+)"', svg_text)
        if match:
            parts = match.group(1).split()
            if len(parts) == 4:
                return float(parts[2]), float(parts[3])
        w_match = re.search(r'width="([^"]+)"', svg_text)
        h_match = re.search(r'height="([^"]+)"', svg_text)
        if w_match and h_match:
            return float(w_match.group(1)), float(h_match.group(1))
    except Exception:
        pass
    return None


def find_svg_path(image_path: Path) -> Path | None:
    for p in [image_path.parent] + list(image_path.parents):
        svg_candidate = p / "model.svg"
        if svg_candidate.exists():
            return svg_candidate
    return None


def extract_svg_text_elements(svg_path: Path) -> list[dict[str, Any]]:
    """Extract all text elements from SVG with their accumulated transforms."""
    try:
        svg_text = svg_path.read_text(encoding="utf-8")
        root = ET.fromstring(svg_text)
    except Exception:
        return []

    results = []
    ns = "http://www.w3.org/2000/svg"

    # Build parent map for traversing up the tree
    parent_map = {child: parent for parent in root.iter() for child in parent}

    def get_transform_offset(elem) -> tuple[float, float]:
        """Get accumulated translate offset for an element (walks up to root).

        Handles both translate(x,y), translate(x y), and matrix(a,b,c,d,e,f).
        """
        tx, ty = 0.0, 0.0
        current: ET.Element | None = elem
        while current is not None:
            transform = current.get("transform", "")
            if transform:
                # Try matrix(a,b,c,d,e,f) first
                m = re.search(r'matrix\(\s*([^,)]+)\s*,\s*([^,)]+)\s*,\s*([^,)]+)\s*,\s*([^,)]+)\s*,\s*([^,)]+)\s*,\s*([^)]+)\s*\)', transform)
                if m:
                    tx += float(m.group(5))
                    ty += float(m.group(6))
                else:
                    # Try translate(x,y) or translate(x y) or translate(x)
                    m = re.search(r'translate\(\s*([^\s,)]+)\s*[,\s]\s*([^\s,)]+)\s*\)', transform)
                    if m:
                        tx += float(m.group(1))
                        ty += float(m.group(2))
                    else:
                        m = re.search(r'translate\(\s*([^\s,)]+)\s*\)', transform)
                        if m:
                            tx += float(m.group(1))
            current = parent_map.get(current)
        return (tx, ty)

    for text_elem in root.iter(f"{{{ns}}}text"):
        text_content = text_elem.text or ""
        if not text_content.strip():
            continue

        x_str = text_elem.get("x", "0")
        y_str = text_elem.get("y", "0")
        try:
            x = float(x_str) if x_str else 0.0
            y = float(y_str.split()[0]) if y_str else 0.0  # handle "0em"
        except (ValueError, IndexError):
            x, y = 0.0, 0.0

        dx, dy = get_transform_offset(text_elem)
        x += dx
        y += dy

        # Estimate bbox from x, y and text length
        font_size = 33  # default
        style = text_elem.get("style", "")
        fs_match = re.search(r'font-size:\s*(\d+)', style)
        if fs_match:
            font_size = int(fs_match.group(1))

        text_len = len(text_content) * font_size * 0.6  # rough width estimate
        results.append({
            "text": text_content.strip(),
            "x": x,
            "y": y,
            "bbox": [x, y - font_size, x + text_len, y],
        })

    return results


def svg_to_pixel_bbox(
    svg_bbox: list[float],
    svg_size: tuple[float, float],
    png_size: tuple[int, int],
) -> list[float]:
    svg_w, svg_h = svg_size
    png_w, png_h = png_size
    scale_x = png_w / max(svg_w, 1)
    scale_y = png_h / max(svg_h, 1)
    scale = min(scale_x, scale_y)
    x1, y1, x2, y2 = svg_bbox
    return [x1 * scale, y1 * scale, x2 * scale, y2 * scale]


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
    pts = ocr_bbox
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def resolve_image_path(image_path_str: str) -> Path | None:
    p = Path(image_path_str)
    if p.is_absolute() and p.exists():
        return p
    candidate = ROOT / p
    if candidate.exists():
        return candidate
    return None


def make_crop_image(
    img: Image.Image,
    pixel_bbox: list[float],
    pad_ratio: float = CROP_PAD_RATIO,
) -> Image.Image | None:
    x1, y1, x2, y2 = pixel_bbox
    w = max(x2 - x1, 1)
    h = max(y2 - y1, 1)
    pad_x = w * pad_ratio
    pad_y = h * pad_ratio
    img_w, img_h = img.size
    cx1 = max(0, int(x1 - pad_x))
    cy1 = max(0, int(y1 - pad_y))
    cx2 = min(img_w, int(x2 + pad_x))
    cy2 = min(img_h, int(y2 + pad_y))
    crop_w = cx2 - cx1
    crop_h = cy2 - cy1
    if crop_w < 2 or crop_h < 2:
        return None
    if crop_w < MIN_CROP_SIZE or crop_h < MIN_CROP_SIZE:
        scale = max(MIN_CROP_SIZE / crop_w, MIN_CROP_SIZE / crop_h)
        new_w = max(int(crop_w * scale), MIN_CROP_SIZE)
        new_h = max(int(crop_h * scale), MIN_CROP_SIZE)
        crop = img.crop((cx1, cy1, cx2, cy2))
        crop = crop.resize((new_w, new_h), Image.LANCZOS)
        return crop
    return img.crop((cx1, cy1, cx2, cy2))


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\u00c0-\u024f']", "", text)  # keep letters, digits, apostrophe
    return text


def main() -> int:
    print("=" * 70)
    print("EasyOCR Full Batch v4: Crop + Full-image OCR with SVG text matching")
    print("=" * 70)

    print("\n1. Initializing EasyOCR (en)...")
    reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    print("   EasyOCR loaded.")

    print("\n2. Loading dataset splits...")
    splits: dict[str, list[dict[str, Any]]] = {}
    for split_name in ("train", "dev", "smoke", "locked_test"):
        path = INPUT_DIR / f"{split_name}.jsonl"
        if path.exists():
            splits[split_name] = load_jsonl(path)
            print(f"   {split_name}: {len(splits[split_name])} records")

    total_records = sum(len(rows) for rows in splits.values())
    print(f"\n3. Processing {total_records} images...")
    t0 = time.time()

    audit: dict[str, Any] = {
        "total_records": total_records,
        "total_text_candidates": 0,
        "text_bearing_candidates": 0,
        "already_has_text": 0,
        "raw_text_newly_populated": 0,
        "raw_text_still_empty": 0,
        "matched_by_crop_iou": 0,
        "matched_by_svg_text": 0,
        "matched_by_proximity": 0,
        "ocr_no_detection": 0,
        "ocr_error": 0,
        "svg_text_extracted": 0,
        "per_split": {},
        "per_type_before": Counter(),
        "per_type_after": Counter(),
    }

    for split_name in ("train", "dev", "smoke", "locked_test"):
        if split_name not in splits:
            continue
        rows = splits[split_name]
        split_audit = {
            "records": len(rows),
            "text_candidates_total": 0,
            "text_bearing_total": 0,
            "already_has_text": 0,
            "raw_text_newly_populated": 0,
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
                audit["ocr_error"] += 1
                continue

            # SVG info
            svg_path = find_svg_path(resolved)
            svg_size = None
            if svg_path:
                svg_size = get_svg_viewbox(svg_path)

            text_candidates = row.get("text_candidates") or []

            # Collect text-bearing candidates that need OCR
            need_ocr = []
            for tc in text_candidates:
                tc_type = tc.get("text_type", "unknown")
                audit["per_type_before"][tc_type] += 1
                audit["total_text_candidates"] += 1
                split_audit["text_candidates_total"] += 1

                if tc_type not in TEXT_BEARING_TYPES:
                    continue

                audit["text_bearing_candidates"] += 1
                split_audit["text_bearing_total"] += 1

                if tc.get("raw_text") and tc["raw_text"].strip():
                    audit["already_has_text"] += 1
                    split_audit["already_has_text"] += 1
                    audit["per_type_after"][tc_type] += 1
                    continue

                need_ocr.append(tc)

            if not need_ocr:
                continue

            # Run EasyOCR on full image
            try:
                ocr_results = reader.readtext(str(resolved), detail=1)
                split_audit["ocr_detections"] += len(ocr_results)
            except Exception:
                audit["ocr_error"] += 1
                continue

            # Convert OCR results to pixel bboxes
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

            # Also extract SVG text elements for content matching
            svg_texts = []
            if svg_path:
                svg_texts = extract_svg_text_elements(svg_path)
                if svg_texts:
                    audit["svg_text_extracted"] += 1

            # Match each candidate needing OCR
            for tc in need_ocr:
                tc_bbox_raw = tc.get("bbox")
                tc_bbox = normalize_bbox(tc_bbox_raw)
                if tc_bbox is None:
                    split_audit["raw_text_still_empty"] += 1
                    audit["raw_text_still_empty"] += 1
                    audit["per_type_after"][tc.get("text_type", "unknown")] += 1
                    continue

                # Transform to pixel coords
                if svg_size:
                    pixel_bbox = svg_to_pixel_bbox(tc_bbox, svg_size, (img_w, img_h))
                else:
                    pixel_bbox = tc_bbox

                pixel_area = max(0, pixel_bbox[2] - pixel_bbox[0]) * max(0, pixel_bbox[3] - pixel_bbox[1])

                matched = False

                if pixel_area >= MIN_VALID_PIXEL_AREA:
                    # Strategy 1: Crop-based OCR + IoU matching
                    crop = make_crop_image(img, pixel_bbox)
                    if crop:
                        try:
                            crop_ocr = reader.readtext(crop, detail=1)
                        except Exception:
                            crop_ocr = []

                        for c_bbox, c_text, c_conf in crop_ocr:
                            if c_text and c_text.strip():
                                tc["raw_text"] = c_text.strip()
                                tc["normalized_text"] = c_text.strip().lower()
                                tc["ocr_confidence"] = c_conf
                                tc["text_source"] = "easyocr_crop"
                                matched = True
                                audit["matched_by_crop_iou"] += 1
                                audit["raw_text_newly_populated"] += 1
                                split_audit["raw_text_newly_populated"] += 1
                                audit["per_type_after"][tc.get("text_type", "unknown")] += 1
                                break

                if not matched and ocr_items:
                    # Strategy 2: Full-image OCR + IoU matching
                    best_ocr = None
                    best_iou = 0.0
                    for ocr_item in ocr_items:
                        iou = bbox_iou(pixel_bbox, ocr_item["bbox"])
                        if iou > best_iou:
                            best_iou = iou
                            best_ocr = ocr_item

                    if best_ocr and best_iou >= IOU_THRESHOLD:
                        tc["raw_text"] = best_ocr["text"]
                        tc["normalized_text"] = best_ocr["text"].lower().strip()
                        tc["ocr_confidence"] = best_ocr["confidence"]
                        tc["ocr_iou"] = round(best_iou, 4)
                        tc["text_source"] = "easyocr_iou"
                        matched = True
                        audit["raw_text_newly_populated"] += 1
                        split_audit["raw_text_newly_populated"] += 1
                        audit["per_type_after"][tc.get("text_type", "unknown")] += 1

                if not matched and svg_texts:
                    # Strategy 3: SVG text content matching
                    # For placeholder bboxes, match by SVG text content
                    # Find SVG text elements near the candidate's SVG position
                    if svg_size:
                        cx = (tc_bbox[0] + tc_bbox[2]) / 2
                        cy = (tc_bbox[1] + tc_bbox[3]) / 2
                        # Transform to pixel for distance computation
                        scale = min(img_w / svg_size[0], img_h / svg_size[1])
                        pcx = cx * scale
                        pcy = cy * scale

                        # Find nearest OCR detection to this position
                        best_ocr = None
                        best_dist = float("inf")
                        for ocr_item in ocr_items:
                            ocx = (ocr_item["bbox"][0] + ocr_item["bbox"][2]) / 2
                            ocy = (ocr_item["bbox"][1] + ocr_item["bbox"][3]) / 2
                            dist = math.sqrt((pcx - ocx) ** 2 + (pcy - ocy) ** 2)
                            if dist < best_dist:
                                best_dist = dist
                                best_ocr = ocr_item

                        if best_ocr and best_dist < 200:  # within 200 pixels
                            tc["raw_text"] = best_ocr["text"]
                            tc["normalized_text"] = best_ocr["text"].lower().strip()
                            tc["ocr_confidence"] = best_ocr["confidence"]
                            tc["ocr_distance"] = round(best_dist, 1)
                            tc["text_source"] = "easyocr_proximity"
                            matched = True
                            audit["matched_by_proximity"] += 1
                            audit["raw_text_newly_populated"] += 1
                            split_audit["raw_text_newly_populated"] += 1
                            audit["per_type_after"][tc.get("text_type", "unknown")] += 1

                if not matched:
                    split_audit["raw_text_still_empty"] += 1
                    audit["raw_text_still_empty"] += 1
                    audit["per_type_after"][tc.get("text_type", "unknown")] += 1

            if (idx + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                print(f"   [{split_name}] {idx+1}/{len(rows)} ({elapsed:.0f}s, {rate:.1f} img/s) "
                      f"newly populated: {split_audit['raw_text_newly_populated']}")

        audit["per_split"][split_name] = split_audit
        total_tb = split_audit["text_bearing_total"]
        total_pop = split_audit["already_has_text"] + split_audit["raw_text_newly_populated"]
        cov = total_pop / max(total_tb, 1) * 100
        print(f"   [{split_name}] Done: {total_pop}/{total_tb} text-bearing candidates "
              f"({cov:.1f}%) — newly populated: {split_audit['raw_text_newly_populated']}")

    # Write augmented datasets
    print(f"\n4. Writing v4 OCR-augmented datasets to {OUTPUT_DIR}...")
    for split_name, rows in splits.items():
        write_jsonl(OUTPUT_DIR / f"{split_name}.jsonl", rows)
        print(f"   Wrote {OUTPUT_DIR / f'{split_name}.jsonl'}")

    # Compute coverage
    total_tb = audit["text_bearing_candidates"]
    total_pop = audit["already_has_text"] + audit["raw_text_newly_populated"]
    coverage = total_pop / max(total_tb, 1) * 100

    audit["elapsed_seconds"] = round(time.time() - t0, 1)
    audit["coverage_percent"] = round(coverage, 2)
    audit["per_type_before"] = dict(audit["per_type_before"])
    audit["per_type_after"] = dict(audit["per_type_after"])

    write_json(REPORTS_DIR / "easyocr_full_batch_audit.json", audit)

    print(f"\n{'=' * 70}")
    print(f"EasyOCR Full Batch v4 complete:")
    print(f"  Text-bearing candidates:   {total_tb}")
    print(f"  Already had text:          {audit['already_has_text']}")
    print(f"  Newly populated:           {audit['raw_text_newly_populated']}")
    print(f"  Still empty:               {audit['raw_text_still_empty']}")
    print(f"  Coverage:                  {coverage:.1f}%")
    print(f"  Matched by crop+IoU:       {audit['matched_by_crop_iou']}")
    print(f"  Matched by proximity:      {audit['matched_by_proximity']}")
    print(f"  Elapsed:                   {audit['elapsed_seconds']}s")
    print(f"{'=' * 70}")

    # Per-type coverage
    print(f"\nPer-type coverage:")
    print(f"  {'type':<20} {'before':>8} {'after':>8} {'coverage':>8}")
    for t in sorted(set(list(audit["per_type_before"].keys()) + list(audit["per_type_after"].keys()))):
        before = audit["per_type_before"].get(t, 0)
        after = audit["per_type_after"].get(t, 0)
        cov = after / max(before, 1) * 100
        print(f"  {t:<20} {before:>8} {after:>8} {cov:>7.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
