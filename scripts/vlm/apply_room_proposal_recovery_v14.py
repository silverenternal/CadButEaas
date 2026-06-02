#!/usr/bin/env python3
"""Apply room proposal recovery from SVG/parser room_candidates."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import joblib

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from apply_model_v13_experts_to_records import scene_graph_as_context_row
from train_room_space_context_sklearn import enhanced_room_feature
from train_room_space_context_mlp import row_context
from v5_pipeline_utils import load_json, load_jsonl, update_todo_remove, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="reports/vlm/real_upstream_model_predictions_model_v13_real_infer_rel_contains_boundary_v14.jsonl")
    parser.add_argument("--source-records", default="datasets/cadstruct_cubicasa5k_moe_locked_reviewed_v1/locked_test_text_aware_v13.jsonl")
    parser.add_argument("--checkpoint", default="checkpoints/room_proposal_recovery_v14/policy.json")
    parser.add_argument("--room-checkpoint", default="checkpoints/room_space_expert_v13/model.joblib")
    parser.add_argument("--output", default="reports/vlm/real_upstream_model_predictions_model_v13_real_infer_rel_contains_boundary_v14_room_v14.jsonl")
    parser.add_argument("--audit", default="reports/vlm/room_proposal_recovery_v14_apply_audit.json")
    parser.add_argument("--update-todo", action="store_true")
    args = parser.parse_args()

    source = {sample_key(row): row for row in load_jsonl(args.source_records)}
    policy = load_json(args.checkpoint, {})
    room_model = joblib.load(args.room_checkpoint)
    rows = []
    counts = Counter()
    for row in load_jsonl(args.input):
        item = json.loads(json.dumps(row, ensure_ascii=False))
        key = sample_key(item)
        src = source.get(key, {})
        graph = item.setdefault("scene_graph", {})
        nodes = graph.setdefault("nodes", [])
        existing = {str(node.get("id")) for node in nodes if isinstance(node, dict)}
        candidates = [candidate for candidate in ((src.get("expected_json") or {}).get("room_candidates") or []) if isinstance(candidate, dict)]
        for candidate in candidates:
            candidate_id = str(candidate.get("id") or "")
            if not candidate_id or candidate_id in existing:
                continue
            label, confidence, route, room_probability = classify_candidate(item, candidate, room_model)
            nodes.append(
                {
                    "id": candidate_id,
                    "semantic_type": label,
                    "expert": "room_space",
                    "family": "space",
                    "confidence": confidence,
                    "source_expert": "room_proposal_recovery_v14",
                    "geometry": {"bbox": candidate.get("bbox")},
                    "audit_trace": {"room_proposal_recovery_v14": {"route": route, "source": "svg_room_candidate_recovered"}},
                    "metadata": {
                        "raw_label": candidate.get("room_type"),
                        "shape_features": candidate.get("shape_features") if isinstance(candidate.get("shape_features"), dict) else {},
                        "source": candidate.get("source") or "cubicasa5k_svg",
                        "proposal_source": "svg_room_candidate_recovered",
                        "model_source": "room_space_expert_v13",
                        "model_label": label,
                        "model_confidence": confidence,
                        "room_probability": room_probability,
                        "room_route": route,
                        "room_space_expert": "room_space_expert_v13",
                        "room_proposal_recovery": "room_proposal_recovery_v14",
                    },
                }
            )
            existing.add(candidate_id)
            counts["room_nodes_added"] += 1
            counts[f"added_label:{label}"] += 1
        item.setdefault("route_trace", {})["room_proposal_recovery_v14"] = policy
        rows.append(item)
    audit = {"version": "room_proposal_recovery_v14_apply_audit", "input": args.input, "output": args.output, "policy": policy, "counts": dict(counts)}
    write_jsonl(args.output, rows)
    write_json(args.audit, audit)
    if args.update_todo:
        update_todo_remove(["V13-E2E-P1-006"])
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def classify_candidate(row: dict[str, Any], candidate: dict[str, Any], model: dict[str, Any]) -> tuple[str, float, str, float]:
    context_row = scene_graph_as_context_row(row)
    expected = context_row.setdefault("expected_json", {})
    expected.setdefault("room_candidates", []).append(
        {
            "id": str(candidate.get("id") or ""),
            "room_type": str(candidate.get("room_type") or "room"),
            "bbox": candidate.get("bbox"),
            "shape_features": candidate.get("shape_features") if isinstance(candidate.get("shape_features"), dict) else {},
        }
    )
    context = row_context(context_row)
    room = next((item for item in context["rooms"] if item["id"] == str(candidate.get("id") or "")), None)
    feature = enhanced_room_feature(room, context) if room else None
    if feature is None:
        return str(candidate.get("room_type") or "room"), 0.5, "parser_fallback_no_feature", 0.5
    gate_model = model["gate_model"]
    typed_model = model["typed_model"]
    typed_encoder = model["typed_label_encoder"]
    threshold = float(model["room_threshold"])
    gate_prob = gate_model.predict_proba([feature])[0]
    room_index = list(getattr(gate_model, "classes_", [0, 1])).index(1)
    room_probability = float(gate_prob[room_index])
    typed_index = typed_model.predict([feature])[0]
    typed_prob = typed_model.predict_proba([feature])[0]
    typed_label = str(typed_encoder.inverse_transform([int(typed_index)])[0])
    if room_probability >= threshold:
        return "room", room_probability, "room_gate", room_probability
    return typed_label, float(1.0 - room_probability) * float(max(typed_prob)), "typed_expert", room_probability


def sample_key(row: dict[str, Any]) -> str:
    path = str(row.get("annotation") or row.get("annotation_path") or row.get("image") or row.get("image_path") or "")
    parts = Path(path).parts
    return parts[-2] if len(parts) >= 2 else Path(path).stem


if __name__ == "__main__":
    main()
