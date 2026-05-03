#!/usr/bin/env python3
"""Create matched graph-node dataset with Lie/SE(2) features removed."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from graph_node_model import LIE_NUMERIC_FEATURES

ROOT = Path(__file__).resolve().parent.parent.parent
STRIP_FEATURES = {"angle_degrees", *LIE_NUMERIC_FEATURES}


def strip_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    nodes = []
    for node in row.get("nodes") or []:
        copied = dict(node)
        features = dict(copied.get("features") or {})
        for name in STRIP_FEATURES:
            features.pop(name, None)
        copied["features"] = features
        nodes.append(copied)
    out["nodes"] = nodes
    return out


def convert_jsonl(src: Path, dst: Path) -> dict[str, Any]:
    rows = []
    stripped_nodes = 0
    total_nodes = 0
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for node in row.get("nodes") or []:
            total_nodes += 1
            if any(name in (node.get("features") or {}) for name in STRIP_FEATURES):
                stripped_nodes += 1
        rows.append(strip_row(row))
    dst.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return {"rows": len(rows), "nodes": total_nodes, "nodes_with_removed_lie_features": stripped_nodes}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="datasets/cadstruct_graph_nodes_paper_v2_floor_target_halfdev")
    parser.add_argument("--output-dir", default="datasets/cadstruct_graph_nodes_paper_v2_floor_target_halfdev_no_lie_v1")
    args = parser.parse_args()
    src_dir = ROOT / args.input_dir
    dst_dir = ROOT / args.output_dir
    dst_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "version": "graph_node_no_lie_dataset_v1",
        "source": args.input_dir,
        "removed_features": sorted(STRIP_FEATURES),
        "splits": {},
    }
    for split in ["train", "dev", "smoke"]:
        summary["splits"][split] = convert_jsonl(src_dir / f"{split}.jsonl", dst_dir / f"{split}.jsonl")
    manifest = src_dir / "manifest.json"
    if manifest.exists():
        copied = json.loads(manifest.read_text(encoding="utf-8"))
        copied["derived_no_lie_v1"] = summary
        (dst_dir / "manifest.json").write_text(json.dumps(copied, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for extra in src_dir.iterdir():
        if extra.name in {"train.jsonl", "dev.jsonl", "smoke.jsonl", "manifest.json"}:
            continue
        if extra.is_file():
            shutil.copy2(extra, dst_dir / extra.name)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
