#!/usr/bin/env python3
"""Build v18 raster text/OCR supervision from offline structured gold."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "datasets/image_only_structured_targets_v16"
OUT = ROOT / "datasets/image_only_text_ocr_v18"
SPLITS = ("train", "dev", "locked", "smoke")

ROOM_HINTS = {
    "balcony": ["balcony", "terrace", "terassi", "parveke", "patio", "kuisti"],
    "bathroom": ["bath", "bathroom", "pesuh", "pesu", "kph", "ph", "kh", "sh", "suihku", "sauna"],
    "bedroom": ["bed", "bedroom", "mh", "makuuhuone"],
    "closet": ["closet", "wardrobe", "vh", "vaatehuone", "pukuh", "pkh"],
    "corridor": ["hall", "corridor", "entry", "entrance", "et", "tk", "aula", "eteinen", "kaytava"],
    "kitchen": ["kit", "kitchen", "keittio", "keit", "kk", "tupak", "tupakeittio", "apuk", "k"],
    "living_room": ["living", "lounge", "oh", "olohuone", "rt", "ruok", "rh"],
    "office": ["office", "study", "tyoh", "th", "toimisto"],
    "storage": ["storage", "store", "utility", "laundry", "khh", "var", "tekn", "at"],
    "toilet": ["wc", "toilet"],
}


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


def norm_bbox(value: Any, width: int, height: int) -> list[int] | None:
    if not isinstance(value, list) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(item) for item in value[:4]]
    except (TypeError, ValueError):
        return None
    left = max(0, min(width - 1, int(math.floor(min(x1, x2)))))
    top = max(0, min(height - 1, int(math.floor(min(y1, y2)))))
    right = max(0, min(width - 1, int(math.ceil(max(x1, x2)))))
    bottom = max(0, min(height - 1, int(math.ceil(max(y1, y2)))))
    if right <= left or bottom <= top:
        return None
    return [left, top, right, bottom]


def normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("ö", "o").replace("ä", "a").replace("å", "a")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def room_hint(normalized: str) -> str | None:
    if not normalized:
        return None
    tokens = set(normalized.split())
    for label, keywords in ROOM_HINTS.items():
        for keyword in keywords:
            key = normalize_text(keyword)
            if len(key) <= 3:
                if key in tokens:
                    return label
            elif key in normalized:
                return label
    return None


def bbox_center(bbox: list[int]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def point_in_polygon(x: float, y: float, polygon: list[list[float]]) -> bool:
    inside = False
    count = len(polygon)
    if count < 3:
        return False
    j = count - 1
    for i in range(count):
        xi, yi = float(polygon[i][0]), float(polygon[i][1])
        xj, yj = float(polygon[j][0]), float(polygon[j][1])
        if (yi > y) != (yj > y):
            x_at_y = (xj - xi) * (y - yi) / max(yj - yi, 1e-9) + xi
            if x < x_at_y:
                inside = not inside
        j = i
    return inside


def linked_room_id(text_bbox: list[int], rooms: list[dict[str, Any]]) -> str | None:
    cx, cy = bbox_center(text_bbox)
    for room in rooms:
        polygon = room.get("polygon") or []
        if point_in_polygon(cx, cy, polygon):
            return str(room.get("id") or "")
    for room in rooms:
        bbox = room.get("bbox")
        if isinstance(bbox, list) and len(bbox) >= 4 and bbox[0] <= cx <= bbox[2] and bbox[1] <= cy <= bbox[3]:
            return str(room.get("id") or "")
    return None


def text_record(text: dict[str, Any], rooms: list[dict[str, Any]], width: int, height: int) -> dict[str, Any] | None:
    bbox = norm_bbox(text.get("bbox"), width, height)
    if bbox is None:
        return None
    raw = str(text.get("text") or "")
    normalized = normalize_text(raw)
    semantic_type = str(text.get("semantic_type") or text.get("class") or "text")
    return {
        "target_id": str(text.get("id") or ""),
        "class": "text",
        "semantic_type": semantic_type,
        "bbox": bbox,
        "raw_text": raw,
        "normalized_text": normalized,
        "room_type_hint": room_hint(normalized),
        "linked_room_id": linked_room_id(bbox, rooms),
        "has_digits": bool(re.search(r"\d", raw)),
        "label_source": "offline_svg_structured_gold",
    }


def draw_mask(row_id: str, width: int, height: int, texts: list[dict[str, Any]], split: str) -> str:
    mask_dir = OUT / "masks" / split
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for target in texts:
        draw.rectangle(target["bbox"], fill=1)
    mask_path = mask_dir / f"{row_id}_text_mask.png"
    mask.save(mask_path)
    return str(mask_path.relative_to(ROOT))


def convert_row(row: dict[str, Any], split: str, write_masks_enabled: bool) -> dict[str, Any]:
    width, height = [int(value) for value in row.get("image_size") or [512, 512]]
    structured = row.get("structured") if isinstance(row.get("structured"), dict) else {}
    rooms = list(structured.get("rooms") or [])
    targets = [
        converted for text in structured.get("texts") or []
        if (converted := text_record(text, rooms, width, height)) is not None
    ]
    type_counts = Counter(item["semantic_type"] for item in targets)
    hint_counts = Counter(item["room_type_hint"] for item in targets if item.get("room_type_hint"))
    linked = sum(1 for item in targets if item.get("linked_room_id"))
    mask_path = draw_mask(str(row.get("id")), width, height, targets, split) if write_masks_enabled else None
    return {
        "id": row.get("id"),
        "source_key": row.get("source_key"),
        "split": split,
        "image": str(row.get("image") or ""),
        "image_size": [width, height],
        "targets": {
            "texts": targets,
            "text_mask": mask_path,
        },
        "target_counts": {
            "texts": len(targets),
            "semantic_types": dict(sorted(type_counts.items())),
            "room_type_hints": dict(sorted(hint_counts.items())),
            "linked_to_room": linked,
            "with_digits": sum(1 for item in targets if item.get("has_digits")),
        },
        "source_integrity": {
            "source_mode": "offline_training_gold",
            "model_input": "raster_image_only",
            "label_use": "training_or_locked_evaluation_only",
            "for_model_credit_inference": False,
        },
    }


def split_sanity(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    keys_by_split = {
        split: {str(row.get("source_key") or row.get("id")) for row in rows}
        for split, rows in rows_by_split.items()
    }
    overlaps = {}
    split_names = list(keys_by_split)
    for idx, left in enumerate(split_names):
        for right in split_names[idx + 1:]:
            overlaps[f"{left}_{right}"] = len(keys_by_split[left] & keys_by_split[right])
    formal_overlap_keys = [
        key for key in overlaps
        if not key.endswith("_smoke") and not key.startswith("smoke_")
    ]
    return {
        "source_key_counts": {split: len(keys) for split, keys in keys_by_split.items()},
        "overlaps": overlaps,
        "formal_split_overlap_keys": formal_overlap_keys,
        "formal_split_sanity_passed": all(overlaps[key] == 0 for key in formal_overlap_keys),
        "smoke_is_subset_allowed": True,
        "split_sanity_passed": all(overlaps[key] == 0 for key in formal_overlap_keys),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    rows_by_split = {split: load_jsonl(SOURCE / f"{split}.jsonl") for split in SPLITS}
    converted_by_split: dict[str, list[dict[str, Any]]] = {}
    semantic_totals: Counter[str] = Counter()
    hint_totals: Counter[str] = Counter()
    aggregate = Counter()

    for split, rows in rows_by_split.items():
        converted = [convert_row(row, split, not args.no_masks) for row in rows]
        converted_by_split[split] = converted
        write_jsonl(OUT / f"{split}.jsonl", converted)
        for item in converted:
            counts = item.get("target_counts") or {}
            semantic_totals.update(counts.get("semantic_types") or {})
            hint_totals.update(counts.get("room_type_hints") or {})
            aggregate.update({
                "texts": int(counts.get("texts") or 0),
                "linked_to_room": int(counts.get("linked_to_room") or 0),
                "with_digits": int(counts.get("with_digits") or 0),
            })

    sanity = split_sanity(rows_by_split)
    manifest = {
        "version": "image_only_text_ocr_v18",
        "task": "IMG-MOE-V18-P0-006",
        "created": "2026-05-08",
        "dataset": str(OUT.relative_to(ROOT)),
        "source_dataset": str(SOURCE.relative_to(ROOT)),
        "splits": {split: len(rows) for split, rows in converted_by_split.items()},
        "split_sanity": sanity,
        "target_schema": {
            "texts": ["target_id", "semantic_type", "bbox", "raw_text", "normalized_text", "room_type_hint", "linked_room_id"],
            "masks": ["text_mask"],
            "metrics_enabled": ["text_bbox_recall", "ocr_exact_or_normalized_accuracy", "room_label_link_recall"],
        },
        "aggregate_counts": {
            "rows": sum(len(rows) for rows in converted_by_split.values()),
            "texts": aggregate["texts"],
            "linked_to_room": aggregate["linked_to_room"],
            "with_digits": aggregate["with_digits"],
            "semantic_types": dict(sorted(semantic_totals.items())),
            "room_type_hints": dict(sorted(hint_totals.items())),
        },
        "source_integrity_policy": {
            "offline_gold_used_for": ["training_targets", "locked_evaluation", "ocr_transcript_supervision"],
            "offline_gold_forbidden_for": ["model_credit_inference"],
            "inference_contract": "raster image only",
        },
        "validation": {
            "manifest_exists": True,
            "all_splits_nonempty": all(len(rows) > 0 for rows in converted_by_split.values()),
            "all_rows_have_images": all(bool(row.get("image")) for rows in converted_by_split.values() for row in rows),
            "all_rows_have_mask_paths": all(
                bool(((row.get("targets") or {}).get("text_mask")))
                for rows in converted_by_split.values()
                for row in rows
            ) if not args.no_masks else None,
            "formal_split_sanity_passed": sanity["formal_split_sanity_passed"],
            "split_sanity_passed": sanity["split_sanity_passed"],
            "ocr_transcript_accuracy_measurable": True,
            "room_label_linking_measurable": True,
        },
    }
    write_json(OUT / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-masks", action="store_true")
    args = parser.parse_args()
    manifest = build(args)
    print(json.dumps({
        "task": manifest["task"],
        "dataset": manifest["dataset"],
        "splits": manifest["splits"],
        "texts": manifest["aggregate_counts"]["texts"],
        "linked_to_room": manifest["aggregate_counts"]["linked_to_room"],
        "split_sanity_passed": manifest["split_sanity"]["split_sanity_passed"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
