#!/usr/bin/env python3
"""Build v18 raster symbol detector supervision from offline structured gold."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "datasets/image_only_structured_targets_v16"
OUT = ROOT / "datasets/image_only_symbol_detector_v18"
SPLITS = ("train", "dev", "locked", "smoke")


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


def symbol_record(symbol: dict[str, Any], index: int, width: int, height: int) -> dict[str, Any] | None:
    bbox = norm_bbox(symbol.get("bbox"), width, height)
    if bbox is None:
        return None
    label = str(symbol.get("symbol_type") or symbol.get("semantic_type") or symbol.get("class") or "generic_symbol")
    area = max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])
    return {
        "target_id": str(symbol.get("id") or f"symbol_{index}"),
        "class": "symbol",
        "symbol_type": label,
        "semantic_type": label,
        "bbox": bbox,
        "area": area,
        "aspect_ratio": round((bbox[2] - bbox[0]) / max(bbox[3] - bbox[1], 1), 6),
        "rotation": float(symbol.get("rotation") or 0.0),
        "raw_label": symbol.get("raw_label") or symbol.get("class") or label,
        "label_source": "offline_svg_structured_gold",
    }


def draw_mask(row_id: str, width: int, height: int, symbols: list[dict[str, Any]], split: str) -> str:
    mask_dir = OUT / "masks" / split
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for target in symbols:
        draw.rectangle(target["bbox"], fill=1)
    mask_path = mask_dir / f"{row_id}_symbol_mask.png"
    mask.save(mask_path)
    return str(mask_path.relative_to(ROOT))


def convert_row(row: dict[str, Any], split: str, write_masks_enabled: bool) -> dict[str, Any]:
    width, height = [int(value) for value in row.get("image_size") or [512, 512]]
    structured = row.get("structured") if isinstance(row.get("structured"), dict) else {}
    targets = [
        converted for index, symbol in enumerate(structured.get("symbols") or [])
        if (converted := symbol_record(symbol, index, width, height)) is not None
    ]
    type_counts = Counter(item["symbol_type"] for item in targets)
    tiny = sum(1 for item in targets if item["area"] <= 12)
    mask_path = draw_mask(str(row.get("id")), width, height, targets, split) if write_masks_enabled else None
    return {
        "id": row.get("id"),
        "source_key": row.get("source_key"),
        "split": split,
        "image": str(row.get("image") or ""),
        "image_size": [width, height],
        "targets": {
            "symbols": targets,
            "symbol_mask": mask_path,
        },
        "target_counts": {
            "symbols": len(targets),
            "symbol_types": dict(sorted(type_counts.items())),
            "tiny_symbols_area_le_12": tiny,
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
    overlaps: dict[str, int] = {}
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
        "smoke_is_locked_subset": keys_by_split.get("smoke", set()).issubset(keys_by_split.get("locked", set())),
    }


def build(write_masks_enabled: bool = True) -> dict[str, Any]:
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    aggregate_types: Counter[str] = Counter()
    aggregate = Counter()
    for split in SPLITS:
        rows = [convert_row(row, split, write_masks_enabled) for row in load_jsonl(SOURCE / f"{split}.jsonl")]
        rows_by_split[split] = rows
        write_jsonl(OUT / f"{split}.jsonl", rows)
        for row in rows:
            aggregate["rows"] += 1
            aggregate["symbols"] += row["target_counts"]["symbols"]
            aggregate["tiny_symbols_area_le_12"] += row["target_counts"]["tiny_symbols_area_le_12"]
            aggregate_types.update(row["target_counts"]["symbol_types"])

    manifest = {
        "task": "IMG-MOE-V18-P1-007",
        "dataset": str(OUT.relative_to(ROOT)),
        "source_dataset": str(SOURCE.relative_to(ROOT)),
        "splits": {split: len(rows) for split, rows in rows_by_split.items()},
        "aggregate": {
            **dict(aggregate),
            "symbol_types": dict(sorted(aggregate_types.items())),
        },
        "schema": {
            "input": "raster image path plus image_size",
            "targets": "symbols with bbox, symbol_type, rotation, raw_label, area, aspect_ratio, and symbol_mask",
            "source_integrity": "offline gold labels are training/evaluation only and forbidden for model-credit inference",
        },
        "split_sanity": split_sanity(rows_by_split),
    }
    write_json(OUT / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-masks", action="store_true")
    args = parser.parse_args()
    manifest = build(write_masks_enabled=not args.no_masks)
    print("task", manifest["task"])
    print("dataset", manifest["dataset"])
    print("splits", "/".join(f"{k} {v}" for k, v in manifest["splits"].items()))
    print("symbols", manifest["aggregate"]["symbols"])
    print("tiny_symbols_area_le_12", manifest["aggregate"]["tiny_symbols_area_le_12"])
    print("split_sanity_passed", manifest["split_sanity"]["formal_split_sanity_passed"])


if __name__ == "__main__":
    main()
