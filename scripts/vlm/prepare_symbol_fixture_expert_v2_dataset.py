#!/usr/bin/env python3
"""Normalize SymbolFixture v2 dataset with locked-test aliases and audit."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="datasets/symbol_fixture_expert_v2")
    parser.add_argument("--output-dir", default="datasets/symbol_fixture_expert_v2")
    parser.add_argument("--locked-source", default="datasets/cadstruct_real_world_benchmark_v1/symbol_fixture/cubicasa5k_symbol_smoke_locked.jsonl")
    parser.add_argument("--audit-output", default="reports/vlm/symbol_fixture_dataset_v2_audit.json")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_dir != output_dir:
        for split in ("train", "dev", "smoke"):
            src = input_dir / f"{split}.jsonl"
            if src.exists():
                shutil.copyfile(src, output_dir / f"{split}.jsonl")
    if (output_dir / "smoke.jsonl").exists():
        shutil.copyfile(output_dir / "smoke.jsonl", output_dir / "locked_test.jsonl")
    else:
        locked_source = Path(args.locked_source)
        if locked_source.exists():
            shutil.copyfile(locked_source, output_dir / "locked_test.jsonl")

    audit = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "symbol_fixture_dataset_v2_audit",
        "dataset_dir": str(output_dir),
        "splits": {},
        "status": "ok",
    }
    for split in ("train", "dev", "locked_test", "smoke"):
        path = output_dir / f"{split}.jsonl"
        if not path.exists():
            audit["status"] = "needs_review"
            audit["splits"][split] = {"missing": True}
            continue
        rows = load_jsonl(path)
        audit["splits"][split] = summarize(rows)
    (output_dir / "manifest.json").write_text(json.dumps({"version": "symbol_fixture_expert_v2", **audit}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output = Path(args.audit_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": audit["status"]}, ensure_ascii=False))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter()
    raw = Counter()
    room_context = Counter()
    missing_crop_ref = 0
    for row in rows:
        for symbol in row.get("symbols") or []:
            labels[str(symbol.get("symbol_type") or "generic_symbol")] += 1
            raw[str(symbol.get("symbol_type_raw") or symbol.get("symbol_type") or "generic_symbol")] += 1
            room_context[str(symbol.get("room_type") or "unknown_room")] += 1
            if not row.get("image"):
                missing_crop_ref += 1
    return {
        "rows": len(rows),
        "symbols": sum(labels.values()),
        "label_counts": dict(labels),
        "raw_label_counts": dict(raw),
        "room_context_counts": dict(room_context),
        "missing_image_ref_symbols": missing_crop_ref,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    main()
