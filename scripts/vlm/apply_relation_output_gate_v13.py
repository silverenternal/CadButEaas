#!/usr/bin/env python3
"""Gate model_v13 relation outputs for reviewed-gold visual evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v5_pipeline_utils import load_jsonl, update_todo_remove, write_json, write_jsonl


MODES = {
    "contains_only_for_reviewed_gold": {"contains"},
    "full_schema_for_visualization": None,
    "full_schema_for_internal_audit": None,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="reports/vlm/real_upstream_model_predictions_model_v13_real_infer.jsonl")
    parser.add_argument("--output", default="reports/vlm/real_upstream_model_predictions_model_v13_real_infer_rel_contains.jsonl")
    parser.add_argument("--audit", default="reports/vlm/model_v13_relation_gate_audit.json")
    parser.add_argument("--mode", choices=sorted(MODES), default="contains_only_for_reviewed_gold")
    parser.add_argument("--update-todo", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    allowed = MODES[args.mode]
    out = []
    relation_before = Counter()
    relation_after = Counter()
    dropped = Counter()
    for row in rows:
        item = json.loads(json.dumps(row, ensure_ascii=False))
        graph = item.setdefault("scene_graph", {})
        kept_edges = []
        for edge in graph.get("edges") or []:
            relation = str(edge.get("relation") or "")
            relation_before[relation] += 1
            if allowed is None or relation in allowed:
                kept_edges.append(edge)
                relation_after[relation] += 1
            else:
                dropped[relation] += 1
        graph["edges"] = kept_edges
        item.setdefault("route_trace", {})["model_v13_relation_gate"] = {
            "mode": args.mode,
            "input_edges": sum(relation_before.values()),
            "output_edges_for_this_record": len(kept_edges),
            "claim_boundary": "Reviewed locked visual gold currently evaluates contains relations; other relation families are preserved only in full-schema modes.",
        }
        out.append(item)

    audit = {
        "version": "model_v13_relation_gate_audit",
        "input": args.input,
        "output": args.output,
        "mode": args.mode,
        "allowed_relations": sorted(allowed) if allowed is not None else "all",
        "relation_before": dict(relation_before),
        "relation_after": dict(relation_after),
        "dropped": dict(dropped),
        "rows": len(out),
    }
    write_jsonl(args.output, out)
    write_json(args.audit, audit)
    if args.update_todo:
        update_todo_remove(["V13-E2E-P0-003"])
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
