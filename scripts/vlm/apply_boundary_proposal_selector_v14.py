#!/usr/bin/env python3
"""Apply boundary v14 semantic selection to a model_v13 visual stream."""

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

from v5_pipeline_utils import load_json, load_jsonl, update_todo_remove, write_json, write_jsonl


BOUNDARY_LABELS = {"door", "hard_wall", "opening", "partition_wall", "window"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="reports/vlm/real_upstream_model_predictions_model_v13_real_infer_rel_contains.jsonl")
    parser.add_argument("--output", default="reports/vlm/real_upstream_model_predictions_model_v13_real_infer_rel_contains_boundary_v14.jsonl")
    parser.add_argument("--checkpoint", default="checkpoints/boundary_proposal_selector_v14/policy.json")
    parser.add_argument("--audit", default="reports/vlm/boundary_proposal_selector_v14_apply_audit.json")
    parser.add_argument("--update-todo", action="store_true")
    args = parser.parse_args()

    policy = load_json(args.checkpoint, {})
    policy_name = str(policy.get("policy") or "raw_label")
    rows = []
    counts = Counter()
    changes = Counter()
    for row in load_jsonl(args.input):
        item = json.loads(json.dumps(row, ensure_ascii=False))
        for node in ((item.get("scene_graph") or {}).get("nodes") or []):
            if not isinstance(node, dict) or node.get("family") != "boundary":
                continue
            metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
            old = str(node.get("semantic_type") or "")
            raw = normalize(metadata.get("raw_label") or metadata.get("base_raw_label"))
            new = select_label(policy_name, old, raw)
            if new and new != old:
                node["semantic_type"] = new
                metadata["boundary_v14_previous_label"] = old
                metadata["model_label"] = new
                metadata["model_source"] = "boundary_proposal_selector_v14"
                metadata["boundary_proposal_selector_v14_policy"] = policy_name
                node["source_expert"] = "boundary_proposal_selector_v14"
                node.setdefault("audit_trace", {})["boundary_v14"] = {"old_label": old, "new_label": new, "policy": policy_name}
                changes[f"{old}->{new}"] += 1
            counts["boundary_seen"] += 1
        item.setdefault("route_trace", {})["boundary_proposal_selector_v14"] = policy
        rows.append(item)
    audit = {"version": "boundary_proposal_selector_v14_apply_audit", "input": args.input, "output": args.output, "policy": policy, "counts": dict(counts), "changes": dict(changes)}
    write_jsonl(args.output, rows)
    write_json(args.audit, audit)
    if args.update_todo:
        update_todo_remove(["V13-E2E-P1-004"])
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def normalize(value: Any) -> str:
    label = str(value or "")
    return label if label in BOUNDARY_LABELS else ""


def select_label(policy: str, current: str, raw: str) -> str:
    if policy == "model_label":
        return current
    if policy == "raw_when_disagree_else_model":
        return raw if raw and raw != current else current
    return raw or current


if __name__ == "__main__":
    main()
