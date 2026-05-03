#!/usr/bin/env python3
"""Freeze-one-expert contribution audit for real-upstream MoE fusion.

The audit reuses the real-upstream dev predictions and evaluates the fused scene
graph after removing one expert family at a time. It is intentionally inference
only: no model training is rerun.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts" / "vlm") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "vlm"))

from fuse_real_upstream import (  # noqa: E402
    DEV_SPLIT,
    PREDICTIONS,
    compute_invalid_graph_rate,
    evaluate_nodes,
    evaluate_relations,
    extract_gold,
    fuse_predictions_with_gold_id_space,
    load_jsonl,
)

EXPERT_TO_FAMILY = {
    "wall_opening": "boundary",
    "room_space": "space",
    "symbol_fixture": "symbol",
    "text_dimension": "text",
    "sheet_layout": "sheet",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default=str(PREDICTIONS))
    parser.add_argument("--dev-split", default=str(DEV_SPLIT))
    parser.add_argument("--output", default="reports/vlm/freeze_one_expert_contribution_v1.json")
    args = parser.parse_args()

    predictions = load_jsonl(Path(args.predictions))
    records = load_jsonl(Path(args.dev_split))
    gold_nodes, gold_edges = extract_gold(records)

    baseline = evaluate(predictions, records, gold_nodes, gold_edges)
    freezes: dict[str, dict[str, Any]] = {}
    for expert, family in EXPERT_TO_FAMILY.items():
        kept = [pred for pred in predictions if str(pred.get("expert")) != expert and str(pred.get("family")) != family]
        metrics = evaluate(kept, records, gold_nodes, gold_edges)
        freezes[expert] = {
            **metrics,
            "removed_family": family,
            "removed_predictions": len(predictions) - len(kept),
            "node_macro_f1_drop": round(baseline["node_macro_f1"] - metrics["node_macro_f1"], 6),
            "relation_f1_drop": round(baseline["relation_f1"] - metrics["relation_f1"], 6),
            "invalid_rate_delta": round(metrics["invalid_graph_rate"] - baseline["invalid_graph_rate"], 6),
        }

    report = {
        "version": "freeze_one_expert_contribution_v1",
        "predictions": args.predictions,
        "dev_split": args.dev_split,
        "baseline": baseline,
        "freezes": freezes,
        "contribution_measurements": len(freezes),
        "acceptance": {
            "target": "5 expert contribution measurements",
            "passed": len(freezes) == 5,
        },
    }

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def evaluate(
    predictions: list[dict[str, Any]],
    records: list[dict[str, Any]],
    gold_nodes: list[dict[str, Any]],
    gold_edges: list[dict[str, Any]],
) -> dict[str, Any]:
    nodes, edges = fuse_predictions_with_gold_id_space(predictions, records)
    node_metrics = evaluate_nodes(nodes, gold_nodes)
    relation_metrics = evaluate_relations(edges, gold_edges)
    return {
        "predictions": len(predictions),
        "nodes": len(nodes),
        "edges": len(edges),
        "node_macro_f1": node_metrics["macro_f1"],
        "relation_f1": relation_metrics["f1"],
        "invalid_graph_rate": round(compute_invalid_graph_rate(nodes, edges), 6),
    }


if __name__ == "__main__":
    main()
