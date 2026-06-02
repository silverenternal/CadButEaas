#!/usr/bin/env python3
"""Apply documented visual postprocess v5 as a separate stream."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

from v5_pipeline_utils import load_json, load_jsonl, model_probabilities, sample_id, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="reports/vlm/real_upstream_model_predictions_model_v5.jsonl")
    parser.add_argument("--output", default="reports/vlm/real_upstream_model_postprocessed_predictions_v5.jsonl")
    parser.add_argument("--ledger", default="reports/vlm/real_model_error_ledger_v5.json")
    parser.add_argument("--ablation", default="reports/vlm/postprocess_v5_ablation.json")
    parser.add_argument("--max-equipment-confidence", type=float, default=0.60)
    parser.add_argument("--max-margin", type=float, default=0.25)
    args = parser.parse_args()

    rows = []
    events: list[dict[str, Any]] = []
    for row in load_jsonl(args.predictions):
        row = json.loads(json.dumps(row, ensure_ascii=False))
        sid = sample_id(row)
        for node in ((row.get("scene_graph") or {}).get("nodes") or []):
            event = maybe_symbol_fix(sid, node, args.max_equipment_confidence, args.max_margin)
            if event:
                events.append(event)
            mark_versions(node)
        row.setdefault("route_trace", {})["postprocess_v5"] = {
            "postprocess_version": "postprocess_v5",
            "event_count": sum(1 for item in events if item["sample_id"] == sid),
            "thresholds": {"max_equipment_confidence": args.max_equipment_confidence, "max_margin": args.max_margin},
            "claim_boundary": "Presentation/fusion cleanup after model_v5, reported separately from expert retraining.",
        }
        rows.append(row)
    write_jsonl(args.output, rows)

    ledger = load_json(args.ledger, {})
    ablation = {
        "version": "postprocess_v5_ablation",
        "inputs": {"model_v5": args.predictions, "ledger": args.ledger},
        "output": args.output,
        "thresholds": {"max_equipment_confidence": args.max_equipment_confidence, "max_margin": args.max_margin},
        "events": events,
        "summary": {
            "event_count": len(events),
            "decision_counts": dict(Counter(item["decision"] for item in events).most_common()),
            "raw_ledger_counts": ledger.get("raw_counts") or {},
            "postprocess_reference_counts": ledger.get("postprocess_adjusted_counts") or {},
        },
        "claim_boundary": "Suppressed or corrected items remain visible in this trace; these are not counted as model-retraining gains.",
    }
    write_json(args.ablation, ablation)
    print(ablation["summary"])


def mark_versions(node: Any) -> None:
    if not isinstance(node, dict):
        return
    metadata = node.setdefault("metadata", {})
    if isinstance(metadata, dict):
        metadata.setdefault("model_version", "model_v5")
        metadata["postprocess_version"] = "postprocess_v5"


def maybe_symbol_fix(sid: str, node: Any, max_equipment_conf: float, max_margin: float) -> dict[str, Any] | None:
    if not isinstance(node, dict) or str(node.get("family")) != "symbol" or str(node.get("semantic_type")) != "equipment":
        return None
    metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
    raw_label = str(metadata.get("raw_label") or metadata.get("base_raw_label") or "").lower()
    if raw_label != "appliance":
        return None
    probs = model_probabilities(node)
    equipment_prob = float(probs.get("equipment", node.get("confidence") or 1.0))
    appliance_prob = float(probs.get("appliance", 0.0))
    margin = equipment_prob - appliance_prob
    if equipment_prob > max_equipment_conf or margin > max_margin:
        return None
    old = str(node.get("semantic_type") or "")
    node["semantic_type"] = "appliance"
    node["confidence"] = max(appliance_prob, 1.0 - max(0.0, margin))
    metadata["postprocess_v5_event"] = {
        "old_label": old,
        "new_label": "appliance",
        "equipment_prob": round(equipment_prob, 6),
        "appliance_prob": round(appliance_prob, 6),
        "margin": round(margin, 6),
        "rule": "raw_label_appliance_low_margin_equipment",
    }
    node["metadata"] = metadata
    return {
        "sample_id": sid,
        "node_id": str(node.get("id") or ""),
        "decision": "appliance_over_low_margin_equipment",
        "old_label": old,
        "new_label": "appliance",
        "equipment_prob": round(equipment_prob, 6),
        "appliance_prob": round(appliance_prob, 6),
        "margin": round(margin, 6),
    }


if __name__ == "__main__":
    main()
