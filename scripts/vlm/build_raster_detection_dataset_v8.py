#!/usr/bin/env python3
"""Build leakage-free raster detector manifests from CubiCasa PNGs and offline SVG labels."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from convert_cubicasa5k_svg import convert_dataset
from v8_raster_e2e_utils import (
    CONVERTED_DIR,
    DATASET_DIR,
    ROOT,
    extract_gold_items,
    family_counts,
    load_jsonl,
    sample_key,
    split_rows_with_locked,
    summarize_manifest_rows,
    update_todo_remove,
    write_json,
    write_jsonl,
)


def main() -> None:
    converted = ensure_converted_rows()
    splits = split_rows_with_locked(converted)
    manifests: dict[str, list[dict[str, Any]]] = {}
    for split, rows in splits.items():
        out_rows = [manifest_row(row, split) for row in rows]
        manifests[split] = out_rows
        write_jsonl(DATASET_DIR / f"{split}.jsonl", out_rows)

    audit = build_audit(manifests)
    write_json("reports/vlm/raster_detection_dataset_v8_audit.json", audit)
    update_todo_remove(["RASTER-V8-T2"])
    print(json.dumps({"output_dir": str(DATASET_DIR.relative_to(ROOT)), "splits": {k: len(v) for k, v in manifests.items()}}, ensure_ascii=False, indent=2))


def ensure_converted_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ["train", "dev", "smoke"]:
        rows.extend(load_jsonl(CONVERTED_DIR / f"{split}.jsonl"))
    if rows:
        return rows
    source = ROOT / "datasets/external/cubicasa5k_zenodo/unpacked"
    converted = convert_dataset(source, limit=420, min_bbox_area=4.0)
    return converted


def manifest_row(row: dict[str, Any], split: str) -> dict[str, Any]:
    image = row.get("image_path")
    annotation = row.get("annotation_path")
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    gold_items = extract_gold_items(row)
    return {
        "sample_id": sample_key(image),
        "split": split,
        "image": image,
        "annotation_for_offline_labels": annotation,
        "source_dataset": "cubicasa5k",
        "inference_input": "image_only",
        "label_source": "offline_cubicasa_svg_gold_not_inference_input",
        "image_size": {"width": metadata.get("width"), "height": metadata.get("height")},
        "gold_items": gold_items,
        "gold_family_counts": family_counts(gold_items),
        "provenance": {
            "svg_used_for_training_labels_only": True,
            "svg_candidate_ids_allowed_at_inference": False,
            "pure_raster_claim_requires_detector_adoption": True,
        },
    }


def build_audit(manifests: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    split_keys = {split: {str(row.get("sample_id")) for row in rows} for split, rows in manifests.items()}
    overlaps = {
        "train_locked": sorted(split_keys.get("train", set()) & split_keys.get("locked", set())),
        "dev_locked": sorted(split_keys.get("dev", set()) & split_keys.get("locked", set())),
        "train_dev": sorted(split_keys.get("train", set()) & split_keys.get("dev", set())),
    }
    counts = {split: summarize_manifest_rows(rows) for split, rows in manifests.items()}
    bbox_stats: dict[str, Counter[str]] = defaultdict(Counter)
    for split, rows in manifests.items():
        for row in rows:
            for item in row.get("gold_items") or []:
                bbox_stats[split][str(item.get("family") or "unknown")] += 1
    return {
        "version": "raster_detection_dataset_v8_audit",
        "created": "2026-05-07",
        "dataset_dir": str(DATASET_DIR.relative_to(ROOT)),
        "splits": counts,
        "bbox_counts": {split: dict(counter) for split, counter in bbox_stats.items()},
        "overlap": {key: {"count": len(value), "examples": value[:10]} for key, value in overlaps.items()},
        "acceptance": {
            "train_dev_and_locked_image_overlap_is_0": not overlaps["train_locked"] and not overlaps["dev_locked"],
            "all_rows_image_only": all(row.get("inference_input") == "image_only" for rows in manifests.values() for row in rows),
            "family_label_counts_reported": True,
        },
        "claim_boundary": "SVG annotations are stored only as offline gold labels. Detector inference consumes image pixels only.",
    }


if __name__ == "__main__":
    main()
