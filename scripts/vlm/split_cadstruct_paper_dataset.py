#!/usr/bin/env python3
"""Build a leakage-aware paper split from CadStruct JSONL records."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ANGLE_SUFFIX = re.compile(r"_(0|45|90|135|180|225|270|315)$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="datasets/cadstruct")
    parser.add_argument("--output-dir", default="datasets/cadstruct_paper_split")
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--dev-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.15)
    args = parser.parse_args()

    records = load_records(Path(args.input_dir))
    groups = group_records(records)
    split_groups = split_group_keys(groups, args.dev_frac, args.test_frac, args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "seed": args.seed,
        "dev_frac": args.dev_frac,
        "test_frac": args.test_frac,
        "split_policy": "source-stratified group split; CVC-FP rotation suffixes share one group",
        "splits": {},
    }
    for split, keys in split_groups.items():
        rows = [record for key in keys for record in groups[key]]
        write_jsonl(output_dir / f"{split}.jsonl", rows)
        manifest["splits"][split] = summarize(rows, keys)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def load_records(input_dir: Path) -> list[dict[str, Any]]:
    records = []
    seen = set()
    for split in ["train", "dev", "smoke"]:
        path = input_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                key = record_key(record)
                if key in seen:
                    continue
                seen.add(key)
                record["_original_split"] = split
                records.append(record)
    return records


def record_key(record: dict[str, Any]) -> str:
    image = str(record.get("image_path") or "")
    source = str(record.get("source_dataset") or "unknown")
    return f"{source}:{image}"


def group_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[group_key(record)].append(record)
    return dict(groups)


def group_key(record: dict[str, Any]) -> str:
    source = str(record.get("source_dataset") or "unknown")
    image = Path(str(record.get("image_path") or "unknown"))
    stem = image.stem
    if source == "cvc_fp":
        stem = ANGLE_SUFFIX.sub("", stem)
    return f"{source}:{image.parent}:{stem}"


def split_group_keys(
    groups: dict[str, list[dict[str, Any]]], dev_frac: float, test_frac: float, seed: int
) -> dict[str, list[str]]:
    by_source: dict[str, list[str]] = defaultdict(list)
    for key, records in groups.items():
        source = str(records[0].get("source_dataset") or "unknown")
        by_source[source].append(key)
    rng = random.Random(seed)
    output = {"train": [], "dev": [], "smoke": []}
    for source, keys in sorted(by_source.items()):
        keys = list(keys)
        rng.shuffle(keys)
        total = len(keys)
        test_count = max(1, round(total * test_frac)) if total >= 3 else max(0, total - 1)
        dev_count = max(1, round(total * dev_frac)) if total - test_count >= 3 else max(0, total - test_count - 1)
        output["smoke"].extend(keys[:test_count])
        output["dev"].extend(keys[test_count : test_count + dev_count])
        output["train"].extend(keys[test_count + dev_count :])
    for keys in output.values():
        keys.sort()
    return output


def summarize(rows: list[dict[str, Any]], group_keys: list[str]) -> dict[str, Any]:
    source_counts: dict[str, int] = defaultdict(int)
    original_counts: dict[str, int] = defaultdict(int)
    for record in rows:
        source_counts[str(record.get("source_dataset") or "unknown")] += 1
        original_counts[str(record.get("_original_split") or "unknown")] += 1
    return {
        "rows": len(rows),
        "groups": len(group_keys),
        "source_counts": dict(sorted(source_counts.items())),
        "original_split_counts": dict(sorted(original_counts.items())),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            row = dict(row)
            row.pop("_original_split", None)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
