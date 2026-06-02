#!/usr/bin/env python3
"""Apply conservative appliance/equipment symbol arbitration for visual v4."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="reports/vlm/real_upstream_model_postprocessed_predictions_boundaryfix_v4.jsonl")
    parser.add_argument("--output", default="reports/vlm/real_upstream_model_postprocessed_predictions_symbolfix_v4.jsonl")
    parser.add_argument("--report", default="reports/vlm/symbol_appliance_equipment_residual_v4.json")
    parser.add_argument("--eval", default="reports/vlm/symbol_fixture_v12_eval.json")
    parser.add_argument("--max-equipment-confidence", type=float, default=0.60)
    parser.add_argument("--max-margin", type=float, default=0.25)
    args = parser.parse_args()

    rows = []
    events: list[dict[str, Any]] = []
    for row in load_jsonl(Path(args.predictions)):
        row = json.loads(json.dumps(row, ensure_ascii=False))
        sid = sample_id(row)
        for node in ((row.get("scene_graph") or {}).get("nodes") or []):
            event = maybe_fix_symbol(sid, node, args.max_equipment_confidence, args.max_margin)
            if event:
                events.append(event)
                row.setdefault("warnings", []).append(f"symbol_semantic_fix_v4:{event['node_id']}:{event['decision']}")
        row.setdefault("route_trace", {})["symbol_semantic_fix_v4"] = {
            "events": sum(1 for event in events if event["sample_id"] == sid),
            "claim_boundary": "Raw-label-aware postprocess over saved symbol expert probabilities; no SymbolFixture v12 retrain adopted in this run.",
        }
        rows.append(row)

    write_jsonl(Path(args.output), rows)
    report = {
        "version": "symbol_appliance_equipment_residual_v4",
        "inputs": {"predictions": args.predictions},
        "output": args.output,
        "thresholds": {"max_equipment_confidence": args.max_equipment_confidence, "max_margin": args.max_margin},
        "summary": {
            "event_count": len(events),
            "decision_counts": dict(Counter(event["decision"] for event in events).most_common()),
            "affected_nodes": [event["node_id"] for event in events],
        },
        "events": events,
    }
    write_json(Path(args.report), report)
    eval_report = {
        "version": "symbol_fixture_v12_eval",
        "adopted_model": "none",
        "adopted_postprocess": "symbol_semantic_fix_v4_raw_label_threshold",
        "locked_macro_f1": None,
        "reason": "Only one visual residual was found and fixed by low-confidence raw-label-aware arbitration; no retraining was necessary for this execution step.",
        "visual_event_count": len(events),
    }
    write_json(Path(args.eval), eval_report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


def maybe_fix_symbol(sid: str, node: dict[str, Any], max_equipment_conf: float, max_margin: float) -> dict[str, Any] | None:
    if not isinstance(node, dict) or str(node.get("family")) != "symbol" or str(node.get("semantic_type")) != "equipment":
        return None
    metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
    raw_label = str(metadata.get("raw_label") or metadata.get("base_raw_label") or "").lower()
    if raw_label != "appliance":
        return None
    probs = model_probabilities(node)
    equipment_prob = probs.get("equipment", safe_float(node.get("confidence"), 1.0))
    appliance_prob = probs.get("appliance", 0.0)
    margin = equipment_prob - appliance_prob
    if equipment_prob > max_equipment_conf or margin > max_margin:
        return None
    old_label = str(node.get("semantic_type") or "")
    node["semantic_type"] = "appliance"
    node["confidence"] = max(float(appliance_prob), 1.0 - max(0.0, margin))
    metadata["symbol_semantic_fix_v4"] = {
        "old_label": old_label,
        "new_label": "appliance",
        "equipment_prob": round(float(equipment_prob), 6),
        "appliance_prob": round(float(appliance_prob), 6),
        "margin": round(float(margin), 6),
        "rule": "raw_label_appliance_low_confidence_equipment",
    }
    metadata["model_label_before_symbol_semantic_fix_v4"] = old_label
    node["metadata"] = metadata
    flags = node.setdefault("quality_flags", [])
    if isinstance(flags, list):
        flags.append("symbol_semantic_fix_v4:raw_label_appliance")
    return {
        "sample_id": sid,
        "node_id": str(node.get("id") or ""),
        "old_label": old_label,
        "new_label": "appliance",
        "raw_label": raw_label,
        "equipment_prob": round(float(equipment_prob), 6),
        "appliance_prob": round(float(appliance_prob), 6),
        "margin": round(float(margin), 6),
        "decision": "appliance_over_low_margin_equipment",
    }


def model_probabilities(node: dict[str, Any]) -> dict[str, float]:
    metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
    upstream = metadata.get("upstream_metadata") if isinstance(metadata.get("upstream_metadata"), dict) else {}
    for key in ["symbol_long_tail_model_v1_probs", "arbitration_v2_probs", "model_probs", "probabilities"]:
        probs = upstream.get(key) or metadata.get(key)
        if isinstance(probs, dict):
            return {str(label): float(value) for label, value in probs.items() if is_number(value)}
    return {}


def sample_id(row: dict[str, Any]) -> str:
    for key in ["sample_id", "image", "image_path"]:
        value = str(row.get(key) or "")
        if value:
            parts = Path(value).parts
            if len(parts) >= 2 and parts[-1].lower().endswith(".png"):
                return parts[-2]
            return Path(value).stem
    return ""


def safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
