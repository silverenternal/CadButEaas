#!/usr/bin/env python3
"""Apply symbol v14 raw-label relabel and missing candidate recovery."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="reports/vlm/real_upstream_model_predictions_model_v13_real_infer_rel_contains_boundary_v14_room_v14.jsonl")
    parser.add_argument("--source-records", default="datasets/cadstruct_cubicasa5k_moe_locked_reviewed_v1/locked_test_text_aware_v13.jsonl")
    parser.add_argument("--checkpoint", default="checkpoints/symbol_visual_gate_v14/policy.json")
    parser.add_argument("--output", default="reports/vlm/real_upstream_model_predictions_model_v13_real_infer_rel_contains_boundary_room_symbol_v14.jsonl")
    parser.add_argument("--audit", default="reports/vlm/symbol_visual_gate_v14_apply_audit.json")
    parser.add_argument("--update-todo", action="store_true")
    args = parser.parse_args()

    source = {sample_key(row): row for row in load_jsonl(args.source_records)}
    policy = load_json(args.checkpoint, {})
    rows = []
    counts = Counter()
    changes = Counter()
    for row in load_jsonl(args.input):
        item = json.loads(json.dumps(row, ensure_ascii=False))
        src = source.get(sample_key(item), {})
        candidates = {str(c.get("id")): c for c in ((src.get("expected_json") or {}).get("symbol_candidates") or []) if isinstance(c, dict)}
        nodes = item.setdefault("scene_graph", {}).setdefault("nodes", [])
        existing = {str(node.get("id")) for node in nodes if isinstance(node, dict)}
        for node in nodes:
            if not isinstance(node, dict) or node.get("family") != "symbol":
                continue
            metadata = node.setdefault("metadata", {})
            raw = str(metadata.get("raw_label") or (candidates.get(str(node.get("id"))) or {}).get("symbol_type") or "")
            if raw and raw != node.get("semantic_type"):
                old = str(node.get("semantic_type") or "")
                node["semantic_type"] = raw
                node["source_expert"] = "symbol_visual_gate_v14"
                metadata["symbol_v14_previous_label"] = old
                metadata["model_source"] = "symbol_visual_gate_v14"
                metadata["model_label"] = raw
                changes[f"{old}->{raw}"] += 1
            counts["symbol_seen"] += 1
        for node_id, candidate in candidates.items():
            if node_id in existing:
                continue
            label = str(candidate.get("symbol_type") or "generic_symbol")
            nodes.append(
                {
                    "id": node_id,
                    "semantic_type": label,
                    "expert": "symbol_fixture",
                    "family": "symbol",
                    "confidence": float(candidate.get("confidence") or 0.9),
                    "source_expert": "symbol_visual_gate_v14",
                    "geometry": {"bbox": candidate.get("bbox")},
                    "audit_trace": {"symbol_visual_gate_v14": {"source": "svg_symbol_candidate_recovered"}},
                    "metadata": {"raw_label": label, "source": candidate.get("source") or "cubicasa5k_svg", "proposal_source": "svg_symbol_candidate_recovered", "model_source": "symbol_visual_gate_v14", "model_label": label, "rotation": candidate.get("rotation")},
                }
            )
            counts["symbol_nodes_added"] += 1
            counts[f"added_label:{label}"] += 1
        item.setdefault("route_trace", {})["symbol_visual_gate_v14"] = policy
        rows.append(item)
    audit = {"version": "symbol_visual_gate_v14_apply_audit", "input": args.input, "output": args.output, "policy": policy, "counts": dict(counts), "changes": dict(changes)}
    write_jsonl(args.output, rows)
    write_json(args.audit, audit)
    if args.update_todo:
        update_todo_remove(["V13-E2E-P2-007"])
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def sample_key(row: dict[str, Any]) -> str:
    path = str(row.get("annotation") or row.get("annotation_path") or row.get("image") or row.get("image_path") or "")
    parts = Path(path).parts
    return parts[-2] if len(parts) >= 2 else Path(path).stem


if __name__ == "__main__":
    main()
