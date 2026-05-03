#!/usr/bin/env python3
"""Attribute real-upstream node errors by family and label."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from fuse_real_upstream import (
    DEV_SPLIT,
    PREDICTIONS,
    _predictions_by_record_family_id,
    evaluate_nodes,
    extract_gold,
    fuse_predictions_with_gold_id_space,
    load_jsonl,
)


ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT = ROOT / "reports" / "vlm" / "real_upstream_node_error_attribution_v1.json"


def family_from_id(node_id: str) -> str:
    parts = node_id.split(":")
    return parts[1] if len(parts) >= 3 else "unknown"


def main() -> int:
    records = load_jsonl(DEV_SPLIT)
    predictions = load_jsonl(PREDICTIONS)
    fused_nodes, fused_edges = fuse_predictions_with_gold_id_space(predictions, records)
    gold_nodes, gold_edges = extract_gold(records)

    overall = evaluate_nodes(fused_nodes, gold_nodes)
    gold_by_id = {node["id"]: node["semantic_type"] for node in gold_nodes}
    fused_by_id = {node["id"]: node["semantic_type"] for node in fused_nodes}

    family_metrics: dict[str, Any] = {}
    for family in ["boundary", "space", "symbol", "text"]:
        family_fused = [node for node in fused_nodes if family_from_id(node["id"]) == family]
        family_gold = [node for node in gold_nodes if family_from_id(node["id"]) == family]
        metrics = evaluate_nodes(family_fused, family_gold)
        family_metrics[family] = {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "support": len(family_gold),
            "worst_labels": sorted(
                (
                    {"label": label, **stats}
                    for label, stats in metrics["per_label"].items()
                ),
                key=lambda item: (item["f1"], -item["support"]),
            )[:10],
        }

    confusion = Counter()
    for node_id, gold_label in gold_by_id.items():
        pred_label = fused_by_id.get(node_id, "__FN__")
        if pred_label != gold_label:
            confusion[(family_from_id(node_id), gold_label, pred_label)] += 1

    top_confusions = [
        {
            "family": family,
            "gold": gold,
            "pred": pred,
            "count": count,
        }
        for (family, gold, pred), count in confusion.most_common(50)
    ]

    pred_lookup = _predictions_by_record_family_id(predictions, records)
    expert_sources = Counter(str(pred.get("source")) for pred in predictions)
    fallback_by_family = defaultdict(int)
    for (_, family, _), pred in pred_lookup.items():
        source = str(pred.get("source") or "")
        if "fallback" in source or "passthrough" in source or "no_features" in source:
            fallback_by_family[family] += 1

    relation_report = json.loads((ROOT / "reports" / "vlm" / "scene_graph_fusion_real_upstream_eval.json").read_text())
    relation_eval = relation_report.get("relation_evaluation") or {}
    invalid_rate = relation_report.get("invalid_graph_rate")

    report = {
        "version": "real_upstream_node_error_attribution_v1",
        "created": "2026-05-03",
        "inputs": {
            "predictions": str(PREDICTIONS),
            "dev_split": str(DEV_SPLIT),
        },
        "overall": {
            "node_accuracy": overall["accuracy"],
            "node_macro_f1": overall["macro_f1"],
            "relation_f1": relation_eval.get("f1"),
            "relation_precision": relation_eval.get("precision"),
            "relation_recall": relation_eval.get("recall"),
            "invalid_graph_rate": invalid_rate,
            "gold_nodes": len(gold_nodes),
            "fused_nodes": len(fused_nodes),
            "gold_edges": len(gold_edges),
            "fused_edges": len(fused_edges),
        },
        "family_metrics": family_metrics,
        "top_confusions": top_confusions,
        "expert_sources": dict(expert_sources),
        "fallback_by_family": dict(fallback_by_family),
        "root_causes": [
            {
                "issue": "record_id_scope_collision",
                "status": "fixed",
                "evidence": "common node IDs now equal all fused/gold nodes within family-scoped IDs; node macro F1 improved from 0.182067 to current report value.",
            },
            {
                "issue": "boundary_door_window_recall",
                "status": "remaining",
                "evidence": "door/window labels are still zero-F1 in current real-upstream output despite hard_wall/opening/partition_wall working.",
            },
            {
                "issue": "symbol_long_tail_collapse",
                "status": "remaining",
                "evidence": "symbol family macro F1 remains low; high-frequency sink/equipment/shower are mostly confused as column/stair.",
            },
        ],
        "done_when_check": {
            "node_macro_f1_ge_050": overall["macro_f1"] >= 0.50,
            "relation_f1_ge_090": (relation_eval.get("f1") or 0.0) >= 0.90,
            "invalid_graph_rate_le_002": (invalid_rate or 0.0) <= 0.02,
        },
    }

    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    print(json.dumps(report["overall"], indent=2))
    print(json.dumps(report["done_when_check"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
