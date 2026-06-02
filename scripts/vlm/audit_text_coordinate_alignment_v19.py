#!/usr/bin/env python3
"""Audit whether cadstruct text labels are usable as raster-pixel localizer supervision."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets/cadstruct_text_dimensions_v1"
REPORT = ROOT / "reports/vlm"


def load_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def abs_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def normalize_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(item) for item in value[:4]]
    except (TypeError, ValueError):
        return None
    left, top, right, bottom = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
    if right <= left or bottom <= top:
        return None
    return [left, top, right, bottom]


def image_size(path_value: str) -> tuple[int, int] | None:
    path = abs_path(path_value)
    if not path.exists():
        return None
    try:
        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except Exception:
        return None


def svg_viewbox(path_value: str) -> tuple[float, float] | None:
    path = abs_path(path_value)
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    match = re.search(r'viewBox="([^"]+)"', text)
    if match:
        parts = match.group(1).split()
        if len(parts) == 4:
            try:
                return float(parts[2]), float(parts[3])
            except ValueError:
                return None
    width = re.search(r'width="([^"]+)"', text)
    height = re.search(r'height="([^"]+)"', text)
    if width and height:
        try:
            return float(width.group(1)), float(height.group(1))
        except ValueError:
            return None
    return None


def in_frame(bbox: list[float], size: tuple[int, int]) -> bool:
    width, height = size
    return bbox[0] >= 0 and bbox[1] >= 0 and bbox[2] <= width and bbox[3] <= height


def scale_bbox(bbox: list[float], svg_size: tuple[float, float], png_size: tuple[int, int]) -> list[float]:
    sx = png_size[0] / max(svg_size[0], 1.0)
    sy = png_size[1] / max(svg_size[1], 1.0)
    scale = min(sx, sy)
    return [bbox[0] * scale, bbox[1] * scale, bbox[2] * scale, bbox[3] * scale]


def audit_split(split: str, limit: int) -> dict[str, Any]:
    rows = load_jsonl(DATA / f"{split}.jsonl", limit)
    totals = Counter()
    labels = Counter()
    direct_labels = Counter()
    scaled_labels = Counter()
    placeholder_labels = Counter()
    examples = {"placeholder": [], "direct_pixel": [], "scaled_pixel": [], "missing_image_or_svg": []}
    for row in rows:
        totals["rows"] += 1
        png_size = image_size(str(row.get("image") or ""))
        svg_size = svg_viewbox(str(row.get("annotation") or ""))
        if png_size is None:
            totals["missing_image"] += 1
        if svg_size is None:
            totals["missing_svg_viewbox"] += 1
        for item in row.get("text_candidates") or []:
            label = str(item.get("text_type") or "unknown")
            labels[label] += 1
            totals["targets"] += 1
            bbox = normalize_bbox(item.get("bbox"))
            if bbox is None:
                totals["invalid_bbox"] += 1
                continue
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if bbox[0] <= 0 <= bbox[2] and bbox[1] <= 0 <= bbox[3]:
                totals["origin_anchored_or_placeholder"] += 1
                placeholder_labels[label] += 1
                if len(examples["placeholder"]) < 8:
                    examples["placeholder"].append({"image": row.get("image"), "label": label, "bbox": bbox, "raw_text": item.get("raw_text") or item.get("text")})
            if png_size and in_frame(bbox, png_size):
                totals["direct_pixel_in_frame"] += 1
                direct_labels[label] += 1
                if len(examples["direct_pixel"]) < 8:
                    examples["direct_pixel"].append({"image": row.get("image"), "label": label, "bbox": bbox})
            if png_size and svg_size:
                scaled = scale_bbox(bbox, svg_size, png_size)
                if in_frame(scaled, png_size):
                    totals["svg_scaled_in_frame"] += 1
                    scaled_labels[label] += 1
                    if len(examples["scaled_pixel"]) < 8:
                        examples["scaled_pixel"].append({"image": row.get("image"), "label": label, "bbox": bbox, "scaled_bbox": [round(v, 3) for v in scaled]})
            elif len(examples["missing_image_or_svg"]) < 8:
                examples["missing_image_or_svg"].append({"image": row.get("image"), "annotation": row.get("annotation"), "png_size": png_size, "svg_size": svg_size})
            if area <= 25:
                totals["tiny_area_le_25"] += 1
    targets = max(totals["targets"], 1)
    return {
        "split": split,
        "rows": len(rows),
        "counts": dict(totals),
        "rates": {
            "direct_pixel_in_frame": round(totals["direct_pixel_in_frame"] / targets, 6),
            "svg_scaled_in_frame": round(totals["svg_scaled_in_frame"] / targets, 6),
            "origin_anchored_or_placeholder": round(totals["origin_anchored_or_placeholder"] / targets, 6),
            "tiny_area_le_25": round(totals["tiny_area_le_25"] / targets, 6),
        },
        "labels": dict(labels),
        "direct_pixel_labels": dict(direct_labels),
        "svg_scaled_labels": dict(scaled_labels),
        "placeholder_labels": dict(placeholder_labels),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    splits = {split: audit_split(split, args.limit) for split in ("train", "dev", "smoke")}
    report = {
        "schema_version": "text_coordinate_alignment_v19",
        "dataset": str(DATA.relative_to(ROOT)),
        "purpose": "Decide whether cadstruct_text_dimensions_v1 can train raster text localization or should remain OCR/semantic pretraining data.",
        "splits": splits,
        "decision": {
            "can_use_as_primary_raster_localizer_supervision": False,
            "reason": "Most text bboxes are SVG-local or origin-anchored placeholders; even scaled boxes need visual/teacher validation before training a raster localizer.",
            "allowed_use_now": [
                "OCR transcript pretraining",
                "text semantic type classification",
                "dimension/leader/link semantic training",
            ],
            "blocked_use_until_fixed": [
                "direct raster text localizer bbox training",
                "locked metric claims for raster text detection",
            ],
        },
    }
    write_json(REPORT / "text_coordinate_alignment_v19_audit.json", report)
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
