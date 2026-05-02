#!/usr/bin/env python3
"""Create grouped train/dev/locked-test splits for CubiCasa MoE records."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="datasets/cadstruct_cubicasa5k_moe")
    parser.add_argument("--output-dir", default="datasets/cadstruct_cubicasa5k_moe_locked")
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--locked-ratio", type=float, default=0.1)
    parser.add_argument("--smoke", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260430)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    records = []
    for split in ("train", "dev", "smoke"):
        records.extend(load_jsonl(input_dir / f"{split}.jsonl"))
    records = dedupe_records(records)
    rng = random.Random(args.seed)
    rng.shuffle(records)

    smoke = records[: args.smoke]
    rest = records[args.smoke :]
    locked_count = int(len(rest) * args.locked_ratio)
    dev_count = int(len(rest) * args.dev_ratio)
    locked_test = rest[:locked_count]
    dev = rest[locked_count : locked_count + dev_count]
    train = rest[locked_count + dev_count :]
    splits = {"train": train, "dev": dev, "locked_test": locked_test, "smoke": smoke}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        write_jsonl(output_dir / f"{name}.jsonl", rows)

    manifest = {
        "source_dir": str(input_dir),
        "seed": args.seed,
        "split_policy": "annotation_path grouped by record; deterministic shuffle; smoke carved first, then locked/dev/train",
        "total": len(records),
        "splits": {name: split_audit(rows) for name, rows in splits.items()},
        "leakage_audit": leakage_audit(splits),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def split_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    room_labels: Counter[str] = Counter()
    text_labels: Counter[str] = Counter()
    text_with_content = 0
    room_count = 0
    for row in rows:
        expected = row.get("expected_json") or {}
        for room in expected.get("room_candidates") or []:
            room_labels[str(room.get("room_type") or "room")] += 1
            room_count += 1
        for text in expected.get("text_candidates") or []:
            text_labels[str(text.get("text_type") or "note_text")] += 1
            if str(text.get("text") or "").strip():
                text_with_content += 1
    return {
        "records": len(rows),
        "rooms": room_count,
        "room_label_counts": dict(room_labels),
        "text_label_counts": dict(text_labels),
        "text_candidates_with_content": text_with_content,
    }


def leakage_audit(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    groups = {
        name: {group_key(row) for row in rows}
        for name, rows in splits.items()
    }
    overlaps = {}
    names = list(groups)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlaps[f"{left}__{right}"] = len(groups[left] & groups[right])
    return overlaps


def group_key(row: dict[str, Any]) -> str:
    return str(row.get("annotation_path") or row.get("image_path") or "")


def dedupe_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []
    for row in rows:
        key = group_key(row)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
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
