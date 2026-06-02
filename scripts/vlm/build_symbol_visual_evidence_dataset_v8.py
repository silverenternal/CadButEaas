#!/usr/bin/env python3
"""Build crop-feature dataset for symbol visual evidence / empty-symbol decisions."""

from __future__ import annotations

import json
from typing import Any

from PIL import Image, ImageStat

from v8_raster_e2e_utils import DATASET_DIR, ROOT, clamp_bbox_to_image, load_jsonl, normalize_bbox, write_json, write_jsonl


def main() -> None:
    outputs: dict[str, list[dict[str, Any]]] = {}
    for split in ["train", "dev", "locked"]:
        rows = load_jsonl(DATASET_DIR / f"{split}.jsonl")
        cases: list[dict[str, Any]] = []
        row_limit = {"train": 650, "dev": 180, "locked": len(rows)}[split]
        for row in rows[:row_limit]:
            image_path = ROOT / str(row.get("image") or "")
            if not image_path.exists():
                continue
            with Image.open(image_path).convert("L") as image:
                symbol_items = [item for item in row.get("gold_items") or [] if item.get("family") == "symbol"]
                if split in {"train", "dev"}:
                    symbol_items = symbol_items[:35]
                for item in symbol_items:
                    bbox = normalize_bbox(item.get("bbox"))
                    if not bbox:
                        continue
                    features = crop_features(image, bbox)
                    label = weak_label(features)
                    cases.append(
                        {
                            "sample_id": row.get("sample_id"),
                            "split": split,
                            "image": row.get("image"),
                            "bbox": bbox,
                            "semantic_type": item.get("semantic_type"),
                            "label": label,
                            "label_source": "weak_visual_ink_evidence_from_training_crop_not_manual_truth",
                            "features": features,
                        }
                    )
                    if split != "locked":
                        neg_bbox = shifted_negative_bbox(bbox, row.get("image_size") or {})
                        neg_features = crop_features(image, neg_bbox)
                        cases.append(
                            {
                                "sample_id": row.get("sample_id"),
                                "split": split,
                                "image": row.get("image"),
                                "bbox": neg_bbox,
                                "semantic_type": "background_negative",
                                "label": "empty_or_review",
                                "label_source": "synthetic_background_negative_near_symbol",
                                "features": neg_features,
                            }
                        )
        outputs[split] = cases
        write_jsonl(f"datasets/symbol_visual_evidence_v8/{split}.jsonl", cases)
    audit = {
        "version": "symbol_visual_evidence_dataset_v8",
        "splits": {split: len(rows) for split, rows in outputs.items()},
        "positive_counts": {split: sum(1 for row in rows if row["label"] == "keep") for split, rows in outputs.items()},
        "negative_counts": {split: sum(1 for row in rows if row["label"] != "keep") for split, rows in outputs.items()},
        "sampling": {
            "train_rows_scanned": min(650, len(load_jsonl(DATASET_DIR / "train.jsonl"))),
            "dev_rows_scanned": min(180, len(load_jsonl(DATASET_DIR / "dev.jsonl"))),
            "locked_rows_scanned": len(load_jsonl(DATASET_DIR / "locked.jsonl")),
            "train_dev_symbol_cap_per_image": 35,
            "locked": "full locked manifest",
        },
        "claim_boundary": "Labels are visual-evidence weak labels and synthetic negatives; adoption requires locked precision guard.",
    }
    write_json("reports/vlm/symbol_visual_evidence_dataset_v8_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def crop_features(image: Image.Image, bbox: list[float]) -> dict[str, float]:
    x1, y1, x2, y2 = clamp_bbox_to_image(bbox, image.size)
    if x2 <= x1 or y2 <= y1:
        return {"dark_ratio": 0.0, "very_dark_ratio": 0.0, "mean": 1.0, "std": 0.0, "area": 0.0, "width": 0.0, "height": 0.0, "aspect": 0.0}
    crop = image.crop((x1, y1, x2, y2))
    stat = ImageStat.Stat(crop)
    pixels = list(crop.getdata())
    total = max(len(pixels), 1)
    dark = sum(1 for value in pixels if value < 210)
    very_dark = sum(1 for value in pixels if value < 80)
    w, h = crop.size
    return {
        "dark_ratio": dark / total,
        "very_dark_ratio": very_dark / total,
        "mean": stat.mean[0] / 255.0,
        "std": stat.stddev[0] / 255.0,
        "area": float(w * h),
        "width": float(w),
        "height": float(h),
        "aspect": max(w, h) / max(min(w, h), 1),
    }


def weak_label(features: dict[str, float]) -> str:
    if features.get("dark_ratio", 0.0) < 0.006 or features.get("very_dark_ratio", 0.0) < 0.001:
        return "empty_or_review"
    return "keep"


def shifted_negative_bbox(bbox: list[float], image_size: dict[str, Any]) -> list[float]:
    width = float(image_size.get("width") or 0.0)
    height = float(image_size.get("height") or 0.0)
    bw = max(1.0, bbox[2] - bbox[0])
    bh = max(1.0, bbox[3] - bbox[1])
    x1 = min(max(0.0, bbox[0] + bw * 2.5), max(0.0, width - bw))
    y1 = min(max(0.0, bbox[1] + bh * 2.5), max(0.0, height - bh))
    return [x1, y1, x1 + bw, y1 + bh]


if __name__ == "__main__":
    main()
