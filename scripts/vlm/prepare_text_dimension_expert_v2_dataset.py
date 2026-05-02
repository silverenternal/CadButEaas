#!/usr/bin/env python3
"""Upgrade TextDimension v1 records to the v2 auditable contract."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="datasets/cadstruct_text_dimensions_v1")
    parser.add_argument("--output-dir", default="datasets/text_dimension_expert_v2")
    parser.add_argument("--audit-output", default="reports/vlm/text_dimension_dataset_v2_audit.json")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "version": "text_dimension_dataset_v2", "splits": {}}
    for split in ("train", "dev", "smoke"):
        path = input_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        rows = [upgrade_row(row) for row in load_jsonl(path)]
        write_jsonl(output_dir / f"{split}.jsonl", rows)
        audit["splits"][split] = summarize(rows)
    if (output_dir / "smoke.jsonl").exists():
        (output_dir / "locked_test.jsonl").write_text((output_dir / "smoke.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
        audit["splits"]["locked_test"] = audit["splits"].get("smoke", {})
    (output_dir / "manifest.json").write_text(json.dumps({"dataset_dir": str(output_dir), **audit}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_json(Path(args.audit_output), audit)
    print(json.dumps({"output_dir": str(output_dir), "splits": list(audit["splits"])}, ensure_ascii=False))


def upgrade_row(row: dict[str, Any]) -> dict[str, Any]:
    candidates = []
    for item in row.get("text_candidates") or []:
        raw_text = str(item.get("raw_text") or item.get("text") or "")
        candidates.append(
            {
                **item,
                "raw_text": raw_text,
                "normalized_text": normalize_text(raw_text),
                "candidate_relations": [],
            }
        )
    upgraded = dict(row)
    upgraded["text_candidates"] = candidates
    upgraded["relation_targets"] = row.get("dimension_links") or []
    upgraded["contract_version"] = "text_dimension_expert_v2"
    return upgraded


def normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().replace(",", ".").split())


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter()
    relations = 0
    raw_text_present = 0
    for row in rows:
        relations += len(row.get("relation_targets") or [])
        for item in row.get("text_candidates") or []:
            labels[str(item.get("text_type") or "note_text")] += 1
            raw_text_present += int(bool(item.get("raw_text")))
    return {"rows": len(rows), "text_candidates": sum(labels.values()), "label_counts": dict(labels), "relations": relations, "raw_text_present": raw_text_present}


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
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
