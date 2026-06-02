#!/usr/bin/env python3
"""Build symbol crop classification records for v18.

The crop records point to raster images and offline symbol boxes. Labels are
for training/evaluation only; inference scripts must classify detector crops
without reading these labels.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "datasets/image_only_symbol_detector_v18"
OUT = ROOT / "datasets/image_only_symbol_crops_v18"
SPLITS = ("train", "dev", "locked", "smoke")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def crop_rows_for_split(split: str, limit_pages: int | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    source_rows = load_jsonl(SOURCE / f"{split}.jsonl")
    if limit_pages:
        source_rows = source_rows[:limit_pages]
    for row in source_rows:
        for index, symbol in enumerate((row.get("targets") or {}).get("symbols") or []):
            bbox = symbol.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            label = str(symbol.get("symbol_type") or symbol.get("semantic_type") or "generic_symbol")
            out.append({
                "id": f"{row['id']}_symbol_crop_{index}",
                "row_id": row["id"],
                "split": split,
                "image": row["image"],
                "bbox": [int(v) for v in bbox],
                "symbol_type": label,
                "semantic_type": label,
                "area": symbol.get("area"),
                "aspect_ratio": symbol.get("aspect_ratio"),
                "crop_source": "offline_gold_symbol_bbox",
                "source_integrity": {
                    "model_input": "raster_image_crop",
                    "label_use": "training_or_locked_evaluation_only",
                    "for_model_credit_inference": False,
                },
            })
    return out


def build(smoke: bool = False) -> dict[str, Any]:
    counts = Counter()
    type_counts = Counter()
    split_counts: dict[str, int] = {}
    for split in SPLITS:
        limit = 5 if smoke else None
        rows = crop_rows_for_split(split, limit_pages=limit)
        write_jsonl(OUT / f"{split}.jsonl", rows)
        split_counts[split] = len(rows)
        counts["rows"] += len(rows)
        for row in rows:
            type_counts[row["symbol_type"]] += 1
            if max(0, row["bbox"][2] - row["bbox"][0]) * max(0, row["bbox"][3] - row["bbox"][1]) <= 12:
                counts["tiny_area_le_12"] += 1
    manifest = {
        "task": "IMG-MOE-V18-P1-007",
        "dataset": str(OUT.relative_to(ROOT)),
        "source_dataset": str(SOURCE.relative_to(ROOT)),
        "smoke": smoke,
        "splits": split_counts,
        "aggregate": {
            "rows": int(counts["rows"]),
            "tiny_area_le_12": int(counts["tiny_area_le_12"]),
            "symbol_types": dict(sorted(type_counts.items())),
        },
        "schema": {
            "input": "raster image crop from bbox",
            "label": "symbol_type",
            "source_integrity": "offline labels are forbidden for model-credit inference",
        },
    }
    write_json(OUT / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    manifest = build(smoke=args.smoke)
    print("task", manifest["task"])
    print("dataset", manifest["dataset"])
    print("splits", manifest["splits"])
    print("rows", manifest["aggregate"]["rows"])
    print("tiny_area_le_12", manifest["aggregate"]["tiny_area_le_12"])


if __name__ == "__main__":
    main()
