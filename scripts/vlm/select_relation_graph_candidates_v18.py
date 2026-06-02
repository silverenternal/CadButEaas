#!/usr/bin/env python3
"""Apply the relation-graph reconstruction policy to candidate topology pages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from relation_graph_reconstruction_v18 import DEFAULT_ADAPTER, DEFAULT_INPUT, DEFAULT_MODEL, DEFAULT_AUDIT, DEFAULT_EVAL, evaluate_selection, load_by_id, load_jsonl, render_pages, write_json, write_jsonl


def load_model(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--output", default="reports/vlm/relation_graph_reconstruction_v18_candidates.jsonl")
    parser.add_argument("--features-output", default="reports/vlm/relation_graph_reconstruction_v18_features.jsonl")
    parser.add_argument("--eval-output", default=str(DEFAULT_EVAL))
    parser.add_argument("--audit-output", default=str(DEFAULT_AUDIT))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    limit = 5 if args.smoke else args.limit
    pages = load_jsonl(Path(args.input), limit=limit)
    adapter_rows = load_jsonl(Path(args.adapter), limit=limit)
    adapter_by_id = {str(row.get("id")): row for row in adapter_rows}
    model = load_model(Path(args.model))
    policy = model.get("policy") or {}
    page_rows, feature_rows, audit = render_pages(pages, adapter_by_id, model, policy)
    write_jsonl(Path(args.output), page_rows)
    write_jsonl(Path(args.features_output), feature_rows)
    write_json(Path(args.audit_output), audit)
    eval_report = evaluate_selection(page_rows, adapter_rows)
    eval_report["policy"] = policy
    eval_report["rows"] = len(page_rows)
    eval_report["features"] = len(feature_rows)
    write_json(Path(args.eval_output), eval_report)
    print(json.dumps({"rows": len(page_rows), "features": len(feature_rows), "output": args.output}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
