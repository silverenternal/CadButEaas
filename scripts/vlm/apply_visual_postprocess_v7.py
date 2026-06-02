#!/usr/bin/env python3
"""Apply visual postprocess v7 as a separate, auditable stream."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

from v5_pipeline_utils import load_jsonl, model_probabilities, sample_id, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="reports/vlm/real_upstream_model_predictions_model_v7.jsonl")
    parser.add_argument("--output", default="reports/vlm/real_upstream_model_postprocessed_predictions_v7.jsonl")
    parser.add_argument("--ablation", default="reports/vlm/postprocess_v7_ablation.json")
    parser.add_argument("--max-equipment-confidence", type=float, default=0.60)
    parser.add_argument("--max-margin", type=float, default=0.25)
    args = parser.parse_args()

    rows = []
    events: list[dict[str, Any]] = []
    for row in load_jsonl(args.predictions):
        out = json.loads(json.dumps(row, ensure_ascii=False))
        sid = sample_id(out)
        sample_events = []
        for node in ((out.get("scene_graph") or {}).get("nodes") or []):
            event = maybe_symbol_fix(sid, node, args.max_equipment_confidence, args.max_margin)
            if event:
                events.append(event)
                sample_events.append(event)
            mark_postprocess(node)
        out.setdefault("route_trace", {})["postprocess_v7"] = {
            "postprocess_version": "postprocess_v7",
            "event_count": len(sample_events),
            "thresholds": {"max_equipment_confidence": args.max_equipment_confidence, "max_margin": args.max_margin},
            "claim_boundary": "Presentation/fusion cleanup after model_v7. These events are reported separately from model retraining.",
        }
        rows.append(out)
    write_jsonl(args.output, rows)
    ablation = {
        "version": "postprocess_v7_ablation",
        "input": args.predictions,
        "output": args.output,
        "thresholds": {"max_equipment_confidence": args.max_equipment_confidence, "max_margin": args.max_margin},
        "events": events,
        "summary": {
            "event_count": len(events),
            "decision_counts": dict(Counter(item["decision"] for item in events).most_common()),
            "by_sample": dict(Counter(item["sample_id"] for item in events).most_common()),
        },
        "claim_boundary": "Postprocess is not counted as model recognition success; every correction remains traceable in events.",
    }
    write_json(args.ablation, ablation)
    print(json.dumps(ablation["summary"], ensure_ascii=False, indent=2))


def mark_postprocess(node: Any) -> None:
    if not isinstance(node, dict):
        return
    metadata = node.setdefault("metadata", {})
    if isinstance(metadata, dict):
        metadata.setdefault("model_version", "model_v7")
        metadata["postprocess_version"] = "postprocess_v7"


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
    metadata["postprocess_v7_event"] = {
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
