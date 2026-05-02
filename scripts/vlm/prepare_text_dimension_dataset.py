#!/usr/bin/env python3
"""Prepare TextDimensionExpert weak-supervision records from CadStruct MoE records."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="datasets/cadstruct_cubicasa5k_moe")
    parser.add_argument("--output-dir", default="datasets/cadstruct_text_dimensions_v1")
    parser.add_argument("--min-text-area", type=float, default=1.0)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {"source": str(input_dir), "splits": {}, "labels": Counter()}
    for split in ("train", "dev", "smoke"):
        input_path = input_dir / f"{split}.jsonl"
        if not input_path.exists():
            continue
        rows = [row for row in (to_text_sample(record, args.min_text_area) for record in load_jsonl(input_path)) if row]
        write_jsonl(output_dir / f"{split}.jsonl", rows)
        label_counts = Counter(item["text_type"] for row in rows for item in row["text_candidates"])
        dimension_links = sum(len(row.get("dimension_links") or []) for row in rows)
        candidate_counts = [len(row["text_candidates"]) for row in rows]
        manifest["splits"][split] = {
            "rows": len(rows),
            "text_candidates": sum(candidate_counts),
            "dimension_links": dimension_links,
            "label_counts": dict(label_counts),
            "candidate_audit": {
                "max_text_candidates_per_record": max(candidate_counts) if candidate_counts else 0,
                "mean_text_candidates_per_record": sum(candidate_counts) / max(len(candidate_counts), 1),
                "dimension_link_coverage": dimension_links / max(sum(candidate_counts), 1),
            },
        }
        manifest["labels"].update(label_counts)

    manifest["labels"] = dict(manifest["labels"])
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def to_text_sample(record: dict[str, Any], min_text_area: float) -> dict[str, Any] | None:
    expected = record.get("expected_json") or {}
    text_candidates = []
    for item in expected.get("text_candidates") or []:
        if not isinstance(item, dict):
            continue
        bbox = normalize_bbox(item.get("bbox"))
        if bbox is None or bbox_area(bbox) < min_text_area:
            continue
        text_candidates.append(
            {
                "id": str(item.get("id") or f"text_{len(text_candidates)}"),
                "text_type": str(item.get("text_type") or "note_text"),
                "bbox": bbox,
                "confidence": float(item.get("confidence") or 1.0),
            }
        )
    if not text_candidates:
        return None
    return {
        "image": record.get("image_path"),
        "annotation": record.get("annotation_path"),
        "source_dataset": record.get("source_dataset"),
        "text_candidates": text_candidates,
        "dimension_links": weak_dimension_links(text_candidates),
        "metadata": {
            "width": (record.get("metadata") or {}).get("width"),
            "height": (record.get("metadata") or {}).get("height"),
            "text_candidate_count": len(text_candidates),
        },
    }


def weak_dimension_links(text_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dimension_texts = [item for item in text_candidates if item["text_type"] == "dimension_text"]
    dimension_lines = [item for item in text_candidates if item["text_type"] == "dimension_line"]
    links = []
    for text in dimension_texts:
        nearest = nearest_candidate(text["bbox"], dimension_lines)
        if nearest is None:
            continue
        links.append(
            {
                "source": text["id"],
                "target": nearest["id"],
                "relation": "dimension_of",
                "evidence": "nearest_dimension_line",
                "distance": round(bbox_distance(text["bbox"], nearest["bbox"]), 3),
            }
        )
    return links


def nearest_candidate(bbox: list[float], candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    return min(candidates, key=lambda item: bbox_distance(bbox, item["bbox"]))


def normalize_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def bbox_area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def bbox_distance(left: list[float], right: list[float]) -> float:
    dx = max(left[0] - right[2], right[0] - left[2], 0.0)
    dy = max(left[1] - right[3], right[1] - left[3], 0.0)
    return (dx * dx + dy * dy) ** 0.5


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
