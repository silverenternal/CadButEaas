#!/usr/bin/env python3
"""Build SymbolFixture hard-case pack from v2 predictions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="checkpoints/symbol_fixture_expert_v2/dev_predictions.jsonl")
    parser.add_argument("--output-dir", default="datasets/symbol_fixture_hard_cases_round_1")
    parser.add_argument("--report", default="reports/vlm/symbol_fixture_hard_case_round_1.jsonl")
    parser.add_argument("--max-cases", type=int, default=3000)
    args = parser.parse_args()

    cases = []
    for row_index, row in enumerate(load_jsonl(Path(args.predictions))):
        for symbol in row.get("symbols") or []:
            gold = str(symbol.get("gold"))
            pred = str(symbol.get("prediction"))
            if gold == pred:
                continue
            cases.append(
                {
                    "sample_id": row.get("annotation") or row.get("image") or f"row_{row_index}",
                    "source": row.get("source_dataset"),
                    "image": row.get("image"),
                    "annotation": row.get("annotation"),
                    "symbol_id": symbol.get("id"),
                    "target": gold,
                    "prediction": pred,
                    "confidence": symbol.get("confidence"),
                    "bbox": symbol.get("bbox"),
                    "room_type": symbol.get("room_type"),
                    "symbol_type_raw": symbol.get("symbol_type_raw"),
                    "error_tags": error_tags(symbol, gold, pred),
                }
            )
    cases.sort(key=lambda item: (-(item.get("confidence") or 0.0), item["target"], item["prediction"]))
    selected = cases[: args.max_cases]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = output_dir / "symbol_fixture_v2_hard_cases.jsonl"
    write_jsonl(cases_path, selected)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(report_path, selected)

    pair_counts = Counter((item["target"], item["prediction"]) for item in selected)
    tag_counts = Counter(tag for item in selected for tag in item["error_tags"])
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "symbol_fixture_hard_cases_round_1",
        "source_predictions": args.predictions,
        "cases": len(selected),
        "total_errors_seen": len(cases),
        "cases_path": str(cases_path),
        "report_path": str(report_path),
        "top_error_pairs": [{"target": a, "prediction": b, "count": c} for (a, b), c in pair_counts.most_common(30)],
        "top_error_tags": dict(tag_counts.most_common(30)),
        "next_training_focus": [
            "replace bbox/context prototype with actual raster crop encoder once torch/PIL are available",
            "add raw subtype supervision for sink/shower/bathtub inside sanitary_fixture",
            "add hard-negative branch for stair/column and equipment/appliance confusion",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(output_dir / "manifest.json"), "cases": len(selected)}, ensure_ascii=False))


def error_tags(symbol: dict[str, Any], gold: str, pred: str) -> list[str]:
    tags = [f"{gold}_to_{pred}"]
    raw = str(symbol.get("symbol_type_raw") or "")
    if raw:
        tags.append(f"raw_{raw}")
    if (symbol.get("confidence") or 0.0) >= 0.7:
        tags.append("high_confidence_error")
    if {gold, pred} & {"stair", "column"}:
        tags.append("stair_column_confusion")
    if {gold, pred} & {"equipment", "appliance"}:
        tags.append("equipment_appliance_confusion")
    if gold == "sanitary_fixture" or pred == "sanitary_fixture":
        tags.append("sanitary_fixture_confusion")
    if gold == "generic_symbol" or pred == "generic_symbol":
        tags.append("generic_symbol_confusion")
    return tags


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
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
