#!/usr/bin/env python3
"""Audit TextDimension errors by OCR/type/parse/relation family."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="checkpoints/cadstruct_moe_text_dimension_crop_mlp/dev_predictions.jsonl")
    parser.add_argument("--output", default="reports/vlm/text_dimension_error_audit_v2.json")
    parser.add_argument("--cases-output", default="reports/vlm/text_dimension_error_cases_v2.jsonl")
    parser.add_argument("--max-cases", type=int, default=5000)
    args = parser.parse_args()

    cases = []
    family_counts = Counter()
    pair_counts = Counter()
    for row_index, row in enumerate(load_jsonl(Path(args.predictions))):
        for item in row.get("text_candidates") or []:
            gold = str(item.get("gold"))
            pred = str(item.get("prediction"))
            if gold == pred:
                continue
            family = classify_family(gold, pred)
            family_counts[family] += 1
            pair_counts[(gold, pred)] += 1
            if len(cases) < args.max_cases:
                cases.append(
                    {
                        "sample_id": row.get("annotation") or row.get("image") or f"row_{row_index}",
                        "source": row.get("source_dataset"),
                        "text_id": item.get("id"),
                        "target": gold,
                        "prediction": pred,
                        "confidence": item.get("confidence"),
                        "bbox": item.get("bbox"),
                        "error_family": family,
                    }
                )
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "text_dimension_error_audit_v2",
        "predictions": args.predictions,
        "errors": sum(family_counts.values()),
        "error_family_counts": dict(family_counts),
        "top_error_pairs": [{"target": a, "prediction": b, "count": c} for (a, b), c in pair_counts.most_common(30)],
        "cases_output": args.cases_output,
        "finding": "Current audit can separate type confusion and dimension-text/room-label confusion; OCR exact requires raw OCR text fields in v2 data.",
    }
    write_json(Path(args.output), report)
    write_jsonl(Path(args.cases_output), cases)
    print(json.dumps({"output": args.output, "errors": report["errors"]}, ensure_ascii=False))


def classify_family(gold: str, pred: str) -> str:
    pair = {gold, pred}
    if "dimension_text" in pair and "room_label" in pair:
        return "dimension_text_vs_room_label_type_error"
    if "dimension_line" in pair or "leader_line" in pair:
        return "line_geometry_type_error"
    if "note_text" in pair:
        return "note_text_type_error"
    if "dimension_text" in pair:
        return "dimension_text_type_error"
    return "text_type_error"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
