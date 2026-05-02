#!/usr/bin/env python3
"""Add supervised keep/suppress labels to graph object proposals."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


OPENING_LABELS = {"door", "window"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="datasets/cadstruct_graph_objects_topology_singleton_proposals")
    parser.add_argument("--output-dir", default="datasets/cadstruct_graph_object_selection")
    parser.add_argument("--min-keep-purity", type=float, default=0.98)
    parser.add_argument("--opening-singletons-only", action="store_true", default=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": str(input_dir),
        "min_keep_purity": args.min_keep_purity,
        "opening_singletons_only": args.opening_singletons_only,
        "splits": {},
        "note": "proposal keep/suppress labels for learned proposal selection; labels use ground truth for supervision only",
    }
    for split in ["train", "dev", "smoke"]:
        input_path = input_dir / f"{split}.jsonl"
        if not input_path.exists():
            continue
        output_path = output_dir / f"{split}.jsonl"
        manifest["splits"][split] = convert_split(input_path, output_path, args.min_keep_purity, args.opening_singletons_only)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def convert_split(input_path: Path, output_path: Path, min_keep_purity: float, opening_singletons_only: bool) -> dict[str, Any]:
    rows = 0
    groups = 0
    keep_counts = Counter()
    keep_by_label = defaultdict(Counter)
    with input_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            sample = json.loads(line)
            groups_with_labels = label_groups(sample.get("groups") or [], min_keep_purity, opening_singletons_only)
            if not groups_with_labels:
                continue
            sample["groups"] = groups_with_labels
            target.write(json.dumps(sample, ensure_ascii=False) + "\n")
            rows += 1
            groups += len(groups_with_labels)
            for group in groups_with_labels:
                keep_label = int(group["keep_label"])
                label = str(group.get("label"))
                keep_counts[str(keep_label)] += 1
                keep_by_label[label][str(keep_label)] += 1
    return {
        "rows": rows,
        "groups": groups,
        "keep_counts": dict(keep_counts),
        "keep_by_label": {label: dict(counts) for label, counts in sorted(keep_by_label.items())},
    }


def label_groups(groups: list[dict[str, Any]], min_keep_purity: float, opening_singletons_only: bool) -> list[dict[str, Any]]:
    output = []
    preferred_wall_nodes = set()
    for group in groups:
        member_ids = [int(node_id) for node_id in group.get("member_ids") or []]
        label = str(group.get("label"))
        purity = float(group.get("label_purity", 1.0) or 0.0)
        if label == "hard_wall" and len(member_ids) > 1 and purity >= min_keep_purity:
            preferred_wall_nodes.update(member_ids)

    for group in groups:
        member_ids = [int(node_id) for node_id in group.get("member_ids") or []]
        label = str(group.get("label"))
        purity = float(group.get("label_purity", 1.0) or 0.0)
        singleton = len(member_ids) == 1
        if label in OPENING_LABELS:
            keep = singleton if opening_singletons_only else purity >= min_keep_purity
        elif label == "hard_wall":
            keep = purity >= min_keep_purity and (not singleton or not member_ids or member_ids[0] not in preferred_wall_nodes)
        else:
            keep = False
        item = dict(group)
        item["semantic_label"] = label
        item["keep_label"] = 1 if keep else 0
        item["selection_label"] = "keep" if keep else "suppress"
        item["label"] = item["selection_label"]
        output.append(item)
    return output


if __name__ == "__main__":
    main()
