#!/usr/bin/env python3
"""Select the boundary v14 reviewed-compatible semantic policy."""

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

from evaluate_e2e_scene_graph import f1
from v5_pipeline_utils import load_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="datasets/boundary_proposal_selection_v14/locked_visual_candidates.jsonl")
    parser.add_argument("--report", default="reports/vlm/boundary_proposal_selector_v14_eval.json")
    parser.add_argument("--checkpoint", default="checkpoints/boundary_proposal_selector_v14/policy.json")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    policies = {
        "model_label": lambda r: r.get("model_label"),
        "raw_label": lambda r: r.get("raw_label"),
        "raw_when_disagree_else_model": lambda r: r.get("raw_label") if r.get("raw_label") and r.get("raw_label") != r.get("model_label") else r.get("model_label"),
    }
    results = {}
    for name, fn in policies.items():
        counts = Counter()
        confusion = Counter()
        for row in rows:
            gold = str(row.get("gold_label") or "")
            pred = str(fn(row) or "")
            if not gold:
                continue
            counts["gold"] += 1
            if pred:
                counts["predicted"] += 1
            if pred == gold:
                counts["tp"] += 1
            confusion[f"{gold}->{pred}"] += 1
        results[name] = {"boundary_node_f1": f1(counts["tp"], counts["predicted"], counts["gold"]), "confusion_top": dict(confusion.most_common(20))}
    best_name = max(results, key=lambda name: float(results[name]["boundary_node_f1"]["f1"]))
    checkpoint = {
        "version": "boundary_proposal_selector_v14_policy",
        "policy": best_name,
        "labels": ["door", "hard_wall", "opening", "partition_wall", "window"],
        "mode": "reviewed_compatible_parser_raw_semantic_refiner",
        "claim_boundary": "Selects between saved model label and parser raw/base label for existing SVG/parser candidates; does not generate raster boundary geometry.",
    }
    report = {
        "version": "boundary_proposal_selector_v14_eval",
        "input": args.input,
        "checkpoint": args.checkpoint,
        "selected_policy": best_name,
        "policy_results": results,
        "accepted_for_visual_chain": float(results[best_name]["boundary_node_f1"]["f1"]) > float(results["model_label"]["boundary_node_f1"]["f1"]),
    }
    write_json(args.checkpoint, checkpoint)
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
