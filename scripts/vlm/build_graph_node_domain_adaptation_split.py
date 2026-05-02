#!/usr/bin/env python3
"""Build few-shot target-domain adaptation splits for graph-node experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset", required=True, help="Source source_dataset value, e.g. cvc_fp.")
    parser.add_argument("--target-dataset", required=True, help="Target source_dataset value, e.g. floorplancad.")
    parser.add_argument("--input-dir", default="datasets/cadstruct_graph_nodes_paper_v2_source_raster")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-adapt-frac", type=float, default=0.5)
    parser.add_argument("--seed", default="20260430")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_train = filter_source(load_jsonl(input_dir / "train.jsonl"), args.source_dataset)
    target_dev_all = filter_source(load_jsonl(input_dir / "dev.jsonl"), args.target_dataset)
    target_adapt, target_select = split_target_dev(target_dev_all, args.seed, args.target_adapt_frac)
    target_locked = filter_source(load_jsonl(input_dir / "smoke.jsonl"), args.target_dataset)

    splits = {
        "train": source_train + target_adapt,
        "dev": target_select,
        "smoke": target_locked,
    }
    for split, rows in splits.items():
        write_jsonl(output_dir / f"{split}.jsonl", rows)

    manifest = {
        "policy": "few-shot target-domain adaptation; source train plus a deterministic target-dev subset for training, held-out target dev for model selection, target smoke locked for final evaluation",
        "source_dataset": args.source_dataset,
        "target_dataset": args.target_dataset,
        "input_dir": str(input_dir),
        "target_adapt_frac": args.target_adapt_frac,
        "seed": args.seed,
        "splits": {split: summarize(rows) for split, rows in splits.items()},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def split_target_dev(rows: list[dict[str, Any]], seed: str, frac: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scored = []
    for row in rows:
        key = f"{seed}:{row.get('image')}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        scored.append((int(digest[:16], 16), row))
    scored.sort(key=lambda item: item[0])
    cut = max(1, min(len(scored) - 1, round(len(scored) * frac)))
    adapt = [row for _, row in scored[:cut]]
    select = [row for _, row in scored[cut:]]
    return adapt, select


def filter_source(rows: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("source_dataset") == source]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    label_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    nodes = 0
    for row in rows:
        source = str(row.get("source_dataset") or "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1
        for node in row.get("nodes") or []:
            nodes += 1
            label = str(node.get("label"))
            label_counts[label] = label_counts.get(label, 0) + 1
    return {
        "rows": len(rows),
        "nodes": nodes,
        "source_counts": source_counts,
        "label_counts": label_counts,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
