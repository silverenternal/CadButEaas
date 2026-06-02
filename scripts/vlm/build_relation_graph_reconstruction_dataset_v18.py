#!/usr/bin/env python3
"""Build the relation-graph reconstruction dataset for v18."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from relation_graph_reconstruction_v18 import DEFAULT_ADAPTER, DEFAULT_DATASET, DEFAULT_INPUT, DEFAULT_AUDIT, build_dataset_rows, load_by_id, load_pages, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--output", default=str(DEFAULT_DATASET))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    pages = load_pages(Path(args.input), smoke=args.smoke, limit=args.limit)
    adapter_by_id = load_by_id(Path(args.adapter), limit=args.limit)
    rows, audit = build_dataset_rows(pages, adapter_by_id, smoke=args.smoke, limit=args.limit)

    write_jsonl(Path(args.output), rows)
    write_json(Path(args.audit), audit)
    print(json.dumps({"rows": len(rows), "positive_edges": audit["positive_edges"], "output": args.output}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
