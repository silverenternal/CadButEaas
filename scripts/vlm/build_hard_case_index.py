#!/usr/bin/env python3
"""Merge current expert errors into a unified hard-case index."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_INPUTS = [
    ("room_space", "dev", "reports/vlm/moe/room_space_context_predicted_upstream_dev_predictions.jsonl", "rooms"),
    ("room_space", "smoke", "reports/vlm/moe/room_space_context_predicted_upstream_smoke_predictions.jsonl", "rooms"),
    ("symbol_fixture", "dev", "checkpoints/cadstruct_moe_symbol_fixture_crop_mlp/dev_predictions.jsonl", "symbols"),
    ("symbol_fixture", "smoke", "checkpoints/cadstruct_moe_symbol_fixture_crop_mlp/smoke_predictions.jsonl", "symbols"),
    ("text_dimension", "dev", "checkpoints/cadstruct_moe_text_dimension_crop_mlp/dev_predictions.jsonl", "text_candidates"),
    ("text_dimension", "smoke", "checkpoints/cadstruct_moe_text_dimension_crop_mlp/smoke_predictions.jsonl", "text_candidates"),
    ("wall_opening", "locked_test", "reports/vlm/paper_v2_h512_two_stage_router_locked_test_predictions.jsonl", "nodes"),
    ("wall_opening", "dev", "reports/vlm/paper_v2_h512_two_stage_router_dev_predictions.jsonl", "nodes"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/vlm/hard_case_index_v1.jsonl")
    parser.add_argument("--summary-output", default="reports/vlm/hard_case_index_summary_v1.json")
    parser.add_argument("--max-per-expert", type=int, default=500)
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    missing_inputs = []
    for expert, split, path_text, field in DEFAULT_INPUTS:
        path = Path(path_text)
        if not path.exists():
            missing_inputs.append(path_text)
            continue
        records.extend(collect_errors(path, expert, split, field, args.max_per_expert))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = summarize(records, missing_inputs)
    summary_output = Path(args.summary_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "records": len(records), "missing_inputs": len(missing_inputs)}, ensure_ascii=False))


def collect_errors(path: Path, expert: str, split: str, field: str, max_per_expert: int) -> list[dict[str, Any]]:
    out = []
    seen = 0
    for row_idx, row in enumerate(load_jsonl(path)):
        items = extract_items(row, field)
        sample_id = sample_id_from_row(row, row_idx)
        for item in items:
            gold = item.get("gold") or item.get("target") or item.get("label") or item.get("expected")
            pred = item.get("prediction") or item.get("pred") or item.get("predicted") or item.get("semantic_type")
            if gold is None or pred is None or str(gold) == str(pred):
                continue
            record = {
                "expert": expert,
                "split": split,
                "sample_id": sample_id,
                "source": row.get("source_dataset") or row.get("source") or row.get("source_bucket"),
                "image": row.get("image") or row.get("image_path"),
                "annotation": row.get("annotation") or row.get("annotation_path"),
                "item_id": item.get("id") or item.get("node_id"),
                "class": str(gold),
                "prediction": str(pred),
                "target": str(gold),
                "confidence": item.get("confidence") or item.get("probability") or item.get("score"),
                "bbox": item.get("bbox"),
                "iou": item.get("iou"),
                "error_tags": error_tags(expert, item, str(gold), str(pred)),
            }
            out.append(record)
            seen += 1
            if seen >= max_per_expert:
                return out
    return out


def extract_items(row: dict[str, Any], field: str) -> list[dict[str, Any]]:
    value = row.get(field)
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if field == "nodes":
        for alt in ("predictions", "items", "records"):
            value = row.get(alt)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        if any(key in row for key in ("gold", "prediction", "target", "pred")):
            return [row]
    return []


def error_tags(expert: str, item: dict[str, Any], gold: str, pred: str) -> list[str]:
    tags = [f"{gold}_to_{pred}"]
    confidence = safe_float(item.get("confidence") or item.get("probability") or item.get("score"))
    if confidence is not None:
        if confidence >= 0.8:
            tags.append("high_confidence_error")
        elif confidence <= 0.35:
            tags.append("low_confidence_error")
    if expert == "room_space" and item.get("iou") == 1.0:
        tags.append("type_error_given_gold_geometry")
    if expert == "symbol_fixture" and (gold == "generic_symbol" or pred == "generic_symbol"):
        tags.append("generic_symbol_confusion")
    if expert == "text_dimension" and "dimension" in (gold + pred):
        tags.append("dimension_text_or_line_confusion")
    if expert == "wall_opening" and ({gold, pred} & {"hard_wall", "door", "window"}):
        tags.append("wall_opening_boundary_confusion")
    return tags


def summarize(records: list[dict[str, Any]], missing_inputs: list[str]) -> dict[str, Any]:
    by_expert = Counter(record["expert"] for record in records)
    by_error = Counter((record["expert"], record["class"], record["prediction"]) for record in records)
    by_tag = Counter(tag for record in records for tag in record["error_tags"])
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "hard_case_index_v1",
        "records": len(records),
        "by_expert": dict(by_expert),
        "top_error_pairs": [
            {"expert": expert, "target": target, "prediction": pred, "count": count}
            for (expert, target, pred), count in by_error.most_common(30)
        ],
        "top_error_tags": dict(by_tag.most_common(30)),
        "missing_inputs": missing_inputs,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def sample_id_from_row(row: dict[str, Any], row_idx: int) -> str:
    return str(row.get("sample_id") or row.get("annotation") or row.get("annotation_path") or row.get("image") or row.get("image_path") or f"row_{row_idx}")


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
