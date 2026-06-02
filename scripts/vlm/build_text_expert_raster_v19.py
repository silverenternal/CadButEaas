#!/usr/bin/env python3
"""Build the v19 raster text expert dataset contract.

This does not pretend that every offline label source is directly usable as
pixel-level supervision. It separates raster-aligned v18 labels from larger
CAD/SVG-derived text labels that must pass coordinate alignment before they can
train a localization head.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "datasets/text_expert_raster_v19"
REPORT = ROOT / "reports/vlm"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def image_size(_path_value: str, fallback: Any = None) -> list[int]:
    if fallback and len(fallback) >= 2:
        return [int(fallback[0]), int(fallback[1])]
    return [0, 0]


def normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("ö", "o").replace("ä", "a").replace("å", "a")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clamp_bbox(value: Any, width: int, height: int) -> tuple[list[int] | None, dict[str, Any]]:
    info = {"bbox_parse_ok": False, "in_frame_before_clamp": False, "clipped": False, "area": 0}
    if not isinstance(value, list) or len(value) < 4:
        return None, info
    try:
        x1, y1, x2, y2 = [float(item) for item in value[:4]]
    except (TypeError, ValueError):
        return None, info
    info["bbox_parse_ok"] = True
    left_f, top_f = min(x1, x2), min(y1, y2)
    right_f, bottom_f = max(x1, x2), max(y1, y2)
    if width <= 0 or height <= 0:
        if right_f <= left_f or bottom_f <= top_f:
            return None, info
        info["area"] = int(max(0.0, right_f - left_f) * max(0.0, bottom_f - top_f))
        return [int(math.floor(left_f)), int(math.floor(top_f)), int(math.ceil(right_f)), int(math.ceil(bottom_f))], info
    info["in_frame_before_clamp"] = left_f >= 0 and top_f >= 0 and right_f <= width and bottom_f <= height
    left = max(0, min(max(width - 1, 0), int(math.floor(left_f))))
    top = max(0, min(max(height - 1, 0), int(math.floor(top_f))))
    right = max(0, min(width, int(math.ceil(right_f))))
    bottom = max(0, min(height, int(math.ceil(bottom_f))))
    info["clipped"] = [left, top, right, bottom] != [int(math.floor(left_f)), int(math.floor(top_f)), int(math.ceil(right_f)), int(math.ceil(bottom_f))]
    if right <= left or bottom <= top:
        return None, info
    info["area"] = int((right - left) * (bottom - top))
    return [left, top, right, bottom], info


def convert_v18_row(row: dict[str, Any], split: str) -> dict[str, Any]:
    size = image_size(str(row.get("image") or ""), row.get("image_size") or [512, 512])
    width, height = size
    targets = []
    counts = Counter()
    for index, item in enumerate(((row.get("targets") or {}).get("texts") or [])):
        bbox, bbox_info = clamp_bbox(item.get("bbox"), width, height)
        if bbox is None:
            counts["dropped_invalid_bbox"] += 1
            continue
        semantic = str(item.get("semantic_type") or "unknown_text")
        raw = str(item.get("raw_text") or "")
        normalized = normalize_text(item.get("normalized_text") or raw)
        targets.append(
            {
                "target_id": str(item.get("target_id") or f"{row.get('id')}_text_{index}"),
                "bbox": bbox,
                "semantic_type": semantic,
                "raw_text": raw,
                "normalized_text": normalized,
                "room_type_hint": item.get("room_type_hint"),
                "linked_room_id": item.get("linked_room_id"),
                "has_digits": bool(item.get("has_digits")),
                "label_source": "offline_svg_structured_gold",
                "coordinate_space": "raster_pixel_aligned",
                "can_train_localizer": True,
                "can_train_ocr": bool(normalized),
                "bbox_audit": bbox_info,
            }
        )
        counts[semantic] += 1
    return {
        "id": f"v18_{row.get('id')}",
        "source_row_id": row.get("id"),
        "source_dataset": "image_only_text_ocr_v18",
        "split": split,
        "image": row.get("image"),
        "image_size": size,
        "text_targets": targets,
        "mask": (row.get("targets") or {}).get("text_mask"),
        "target_counts": dict(counts),
        "source_integrity": source_integrity("raster_pixel_aligned", "train_or_eval_by_split"),
    }


def convert_cadstruct_row(row: dict[str, Any], split: str) -> dict[str, Any]:
    meta = row.get("metadata") or {}
    fallback = [meta.get("width"), meta.get("height")] if meta.get("width") and meta.get("height") else None
    size = image_size(str(row.get("image") or ""), fallback)
    width, height = size
    targets = []
    counts = Counter()
    usable = Counter()
    for index, item in enumerate(row.get("text_candidates") or []):
        bbox, bbox_info = clamp_bbox(item.get("bbox"), width, height)
        semantic = str(item.get("text_type") or item.get("semantic_type") or "unknown_text")
        raw = str(item.get("raw_text") or item.get("text") or "")
        normalized = normalize_text(raw)
        if bbox is None:
            counts["dropped_invalid_bbox"] += 1
            continue
        can_train_localizer = bool(bbox_info["in_frame_before_clamp"] and bbox_info["area"] > 0)
        can_train_ocr = bool(normalized)
        targets.append(
            {
                "target_id": str(item.get("id") or f"{row.get('image')}_text_{index}"),
                "bbox": bbox,
                "semantic_type": semantic,
                "raw_text": raw,
                "normalized_text": normalized,
                "room_type_hint": None,
                "linked_room_id": None,
                "has_digits": bool(re.search(r"\d", raw)),
                "label_source": "offline_cad_svg_converted_label",
                "coordinate_space": "needs_alignment_audit",
                "can_train_localizer": can_train_localizer,
                "can_train_ocr": can_train_ocr,
                "bbox_audit": bbox_info,
            }
        )
        counts[semantic] += 1
        if can_train_localizer:
            usable[semantic] += 1
    return {
        "id": f"cadstruct_text_{split}_{stable_id(str(row.get('image') or row.get('annotation') or ''))}",
        "source_row_id": row.get("image"),
        "source_dataset": "cadstruct_text_dimensions_v1",
        "split": split,
        "image": row.get("image"),
        "annotation": row.get("annotation"),
        "image_size": size,
        "text_targets": targets,
        "target_counts": dict(counts),
        "usable_localizer_counts": dict(usable),
        "dimension_links": row.get("dimension_links") or [],
        "source_integrity": source_integrity("needs_alignment_audit", "training_after_alignment_only"),
    }


def stable_id(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def source_integrity(coordinate_space: str, label_use: str) -> dict[str, Any]:
    return {
        "model_input": "raster_image_only",
        "offline_gold_used_for": ["dataset_conversion", "training_targets", "evaluation"],
        "offline_gold_forbidden_for": ["runtime_features", "model_credit_inference"],
        "coordinate_space": coordinate_space,
        "label_use": label_use,
        "runtime_uses_svg_or_cad_geometry": False,
    }


def summarize(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    split_summary: dict[str, Any] = {}
    total = Counter()
    per_source: dict[str, Counter[str]] = {}
    data_gaps = []
    for split, rows in rows_by_split.items():
        counts = Counter()
        labels = Counter()
        localizer_labels = Counter()
        ocr_labels = Counter()
        for row in rows:
            counts["rows"] += 1
            counts[f"source.{row['source_dataset']}"] += 1
            source_counter = per_source.setdefault(row["source_dataset"], Counter())
            source_counter["rows"] += 1
            for target in row.get("text_targets") or []:
                counts["targets"] += 1
                source_counter["targets"] += 1
                labels[target["semantic_type"]] += 1
                if target.get("can_train_localizer"):
                    counts["localizer_targets"] += 1
                    source_counter["localizer_targets"] += 1
                    localizer_labels[target["semantic_type"]] += 1
                if target.get("can_train_ocr"):
                    counts["ocr_targets"] += 1
                    source_counter["ocr_targets"] += 1
                    ocr_labels[target["semantic_type"]] += 1
        total.update(counts)
        split_summary[split] = {
            **{key: int(value) for key, value in sorted(counts.items())},
            "semantic_types": dict(sorted(labels.items())),
            "localizer_semantic_types": dict(sorted(localizer_labels.items())),
            "ocr_semantic_types": dict(sorted(ocr_labels.items())),
        }
    for label in ("dimension_text", "note_text", "dimension_line", "room_label"):
        if split_summary.get("locked", {}).get("semantic_types", {}).get(label, 0) < 50:
            data_gaps.append(
                {
                    "class": label,
                    "locked_count": int(split_summary.get("locked", {}).get("semantic_types", {}).get(label, 0)),
                    "risk": "locked domain has too few examples for stable per-class reporting",
                    "mitigation": "report macro and per-source metrics; use cadstruct_text_dimensions_v1 only after alignment audit",
                }
            )
    return {
        "splits": split_summary,
        "total_counts": {key: int(value) for key, value in sorted(total.items())},
        "per_source": {key: dict(value) for key, value in sorted(per_source.items())},
        "data_gaps": data_gaps,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "locked": [], "smoke": []}
    for split in ("train", "dev", "locked", "smoke"):
        for row in load_jsonl(ROOT / "datasets/image_only_text_ocr_v18" / f"{split}.jsonl"):
            rows_by_split[split].append(convert_v18_row(row, split))
    for split in ("train", "dev", "smoke"):
        rows = load_jsonl(ROOT / "datasets/cadstruct_text_dimensions_v1" / f"{split}.jsonl")
        if args.max_cadstruct_rows and split == "train":
            rows = rows[: args.max_cadstruct_rows]
        for row in rows:
            rows_by_split[split].append(convert_cadstruct_row(row, split))

    for split, rows in rows_by_split.items():
        write_jsonl(OUT / f"{split}.jsonl", rows)

    summary = summarize(rows_by_split)
    manifest = {
        "version": "text_expert_raster_v19",
        "created": "2026-05-10",
        "dataset": str(OUT.relative_to(ROOT)),
        "goal": "Dedicated raster text detection/OCR/semantic expert dataset for P0-TEXT-001.",
        "source_datasets": [
            "datasets/image_only_text_ocr_v18",
            "datasets/cadstruct_text_dimensions_v1",
        ],
        "contract": {
            "runtime_input": "raster_image_only",
            "offline_label_use": "training/evaluation/audit only",
            "v18_locked_use": "locked evaluation only",
            "cadstruct_text_dimensions_v1_use": "large semantic/OCR pretraining source; localization use requires coordinate alignment audit",
        },
        "target_schema": {
            "text_targets": [
                "bbox",
                "semantic_type",
                "raw_text",
                "normalized_text",
                "can_train_localizer",
                "can_train_ocr",
                "coordinate_space",
            ],
            "heads": ["text_region_localizer", "ocr_recognizer", "semantic_type_classifier"],
        },
        **summary,
    }
    write_json(OUT / "manifest.json", manifest)
    write_json(REPORT / "text_expert_raster_v19_dataset_audit.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-cadstruct-rows",
        type=int,
        default=0,
        help="Optional cap for quick smoke builds. Default 0 uses all rows.",
    )
    args = parser.parse_args()
    manifest = build(args)
    print(
        json.dumps(
            {
                "dataset": manifest["dataset"],
                "splits": {key: value["rows"] for key, value in manifest["splits"].items()},
                "total_targets": manifest["total_counts"].get("targets", 0),
                "localizer_targets": manifest["total_counts"].get("localizer_targets", 0),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
