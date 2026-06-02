#!/usr/bin/env python3
"""Apply available model_v13 specialist checkpoints to saved visual records."""

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

from train_room_space_context_sklearn import enhanced_room_feature
from train_room_space_context_mlp import page_size, row_context
from v5_pipeline_utils import load_json, load_jsonl, update_todo_remove, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="reports/vlm/real_upstream_model_predictions_model_v13.jsonl")
    parser.add_argument("--output", default="reports/vlm/real_upstream_model_predictions_model_v13_real_infer.jsonl")
    parser.add_argument("--audit", default="reports/vlm/model_v13_real_infer_component_audit.json")
    parser.add_argument("--room-checkpoint", default="checkpoints/room_space_expert_v13/model.joblib")
    parser.add_argument("--decisions", default="reports/vlm/model_v13_adoption_decisions.json")
    parser.add_argument("--update-todo", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    decisions = load_json(args.decisions, {})
    room_model = load_room_model(args.room_checkpoint)
    out: list[dict[str, Any]] = []
    audit = {
        "version": "model_v13_real_infer_component_audit",
        "input": args.input,
        "output": args.output,
        "decisions": args.decisions,
        "components": {
            "room_space": {
                "checkpoint": args.room_checkpoint,
                "mode": "checkpoint_rerun" if room_model else "unavailable_inherited",
            },
            "boundary": component_mode(decisions, "boundary"),
            "symbol_fixture": component_mode(decisions, "symbol_fixture"),
            "text_dimension": component_mode(decisions, "text_dimension"),
        },
        "counts": Counter(),
        "label_changes": Counter(),
        "warnings": [],
    }

    for row in rows:
        item = json.loads(json.dumps(row, ensure_ascii=False))
        if room_model:
            apply_room_space_v13(item, room_model, audit)
        mark_inherited_components(item, audit)
        item.setdefault("route_trace", {})["model_v13_real_infer"] = {
            "input_stream": args.input,
            "room_space": audit["components"]["room_space"],
            "boundary": audit["components"]["boundary"],
            "symbol_fixture": audit["components"]["symbol_fixture"],
            "text_dimension": audit["components"]["text_dimension"],
            "claim_boundary": "Available v13 checkpoints are rerun on saved visual candidates; components without checkpoints remain inherited from the upstream saved stream.",
        }
        out.append(item)

    audit["counts"] = dict(audit["counts"])
    audit["label_changes"] = dict(audit["label_changes"])
    audit["stream_rows"] = len(out)
    audit["component_map"] = decisions.get("component_map") or {}
    write_jsonl(args.output, out)
    write_json(args.audit, audit)
    if args.update_todo:
        update_todo_remove(["V13-E2E-P0-001"])
    print(json.dumps({"output": args.output, "audit": args.audit, "counts": audit["counts"], "label_changes": audit["label_changes"]}, ensure_ascii=False, indent=2))


def load_room_model(path: str) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    model = joblib.load(p)
    required = {"gate_model", "typed_model", "typed_label_encoder", "room_threshold"}
    if not isinstance(model, dict) or not required.issubset(model):
        raise SystemExit(f"invalid room checkpoint contract: {path}")
    return model


def component_mode(decisions: dict[str, Any], key: str) -> dict[str, Any]:
    component = ((decisions.get("component_map") or {}).get(key) or {}) if isinstance(decisions, dict) else {}
    checkpoint = component.get("checkpoint")
    return {
        "name": component.get("name"),
        "adopted": bool(component.get("adopted")),
        "checkpoint": checkpoint,
        "mode": "checkpoint_available_not_implemented" if checkpoint else "report_only_inherited",
    }


def apply_room_space_v13(row: dict[str, Any], model: dict[str, Any], audit: dict[str, Any]) -> None:
    context_row = scene_graph_as_context_row(row)
    context = row_context(context_row)
    room_by_id = {room["id"]: room for room in context["rooms"]}
    features = []
    node_room_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for node in ((row.get("scene_graph") or {}).get("nodes") or []):
        if not isinstance(node, dict) or node.get("family") != "space":
            continue
        room = room_by_id.get(str(node.get("id") or ""))
        if not room:
            audit["counts"]["room_v13_missing_context"] += 1
            continue
        feature = enhanced_room_feature(room, context)
        if feature is None:
            audit["counts"]["room_v13_feature_skipped"] += 1
            continue
        features.append(feature)
        node_room_pairs.append((node, room))

    if not features:
        return

    gate_model = model["gate_model"]
    typed_model = model["typed_model"]
    typed_encoder = model["typed_label_encoder"]
    threshold = float(model["room_threshold"])
    gate_probs = gate_model.predict_proba(features)
    typed_indices = typed_model.predict(features)
    typed_probs = typed_model.predict_proba(features)
    typed_labels = typed_encoder.inverse_transform(typed_indices)
    gate_classes = list(getattr(gate_model, "classes_", [0, 1]))
    room_index = gate_classes.index(1) if 1 in gate_classes else len(gate_classes) - 1

    for (node, _room), gate_prob, typed_label, typed_prob in zip(node_room_pairs, gate_probs, typed_labels, typed_probs):
        old_label = str(node.get("semantic_type") or "")
        room_probability = float(gate_prob[room_index])
        if room_probability >= threshold:
            new_label = "room"
            confidence = room_probability
            route = "room_gate"
        else:
            new_label = str(typed_label)
            confidence = float(1.0 - room_probability) * float(max(typed_prob))
            route = "typed_expert"
        node["semantic_type"] = new_label
        node["confidence"] = confidence
        node["expert"] = "room_space"
        node["source_expert"] = "room_space_expert_v13"
        metadata = node.setdefault("metadata", {})
        metadata["model_version"] = "model_v13"
        metadata["room_space_expert"] = "room_space_expert_v13"
        metadata["room_space_checkpoint"] = "checkpoints/room_space_expert_v13/model.joblib"
        metadata["model_source"] = "room_space_expert_v13"
        metadata["model_label"] = new_label
        metadata["model_confidence"] = confidence
        metadata["room_probability"] = room_probability
        metadata["room_route"] = route
        audit_trace = node.setdefault("audit_trace", {})
        audit_trace["model_v13_real_infer"] = {"component": "room_space_expert_v13", "route": route}
        audit["counts"]["room_v13_rerun"] += 1
        if old_label != new_label:
            audit["label_changes"][f"{old_label}->{new_label}"] += 1


def scene_graph_as_context_row(row: dict[str, Any]) -> dict[str, Any]:
    nodes = [node for node in ((row.get("scene_graph") or {}).get("nodes") or []) if isinstance(node, dict)]
    expected = {"room_candidates": [], "symbol_candidates": [], "text_candidates": []}
    primitive_nodes = []
    for node in nodes:
        bbox = ((node.get("geometry") or {}).get("bbox") or node.get("bbox"))
        family = node.get("family")
        semantic = str(node.get("semantic_type") or "")
        metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
        if family == "space":
            expected["room_candidates"].append(
                {
                    "id": str(node.get("id") or ""),
                    "room_type": str(metadata.get("raw_label") or metadata.get("base_raw_label") or semantic or "room"),
                    "bbox": bbox,
                    "shape_features": metadata.get("shape_features") if isinstance(metadata.get("shape_features"), dict) else {},
                }
            )
        elif family == "symbol":
            expected["symbol_candidates"].append(
                {
                    "id": str(node.get("id") or ""),
                    "symbol_type": semantic or str(metadata.get("raw_label") or "generic_symbol"),
                    "bbox": bbox,
                }
            )
        elif family == "text":
            expected["text_candidates"].append(
                {
                    "id": str(node.get("id") or ""),
                    "text_type": semantic or str(metadata.get("raw_text_type") or metadata.get("raw_label") or "note_text"),
                    "text": str(metadata.get("text") or ""),
                    "font_size": metadata.get("font_size"),
                    "bbox": bbox,
                }
            )
        elif family == "boundary":
            primitive_nodes.append({"semantic_type": semantic, "bbox": bbox})

    width, height = infer_page_size(row)
    return {
        "image_path": row.get("image"),
        "annotation_path": row.get("annotation"),
        "source_dataset": row.get("source_dataset"),
        "metadata": {"width": width, "height": height},
        "expected_json": expected,
        "request_hints": {"primitive_graph": {"nodes": primitive_nodes}},
    }


def infer_page_size(row: dict[str, Any]) -> tuple[float | None, float | None]:
    for node in ((row.get("scene_graph") or {}).get("nodes") or []):
        metadata = node.get("metadata") if isinstance(node, dict) and isinstance(node.get("metadata"), dict) else {}
        canvas = metadata.get("source_canvas_bbox")
        if isinstance(canvas, list) and len(canvas) >= 4:
            try:
                return float(canvas[2]) - float(canvas[0]), float(canvas[3]) - float(canvas[1])
            except (TypeError, ValueError):
                pass
    try:
        return page_size({"image_path": row.get("image"), "metadata": {}})
    except Exception:
        return None, None


def mark_inherited_components(row: dict[str, Any], audit: dict[str, Any]) -> None:
    for node in ((row.get("scene_graph") or {}).get("nodes") or []):
        if not isinstance(node, dict):
            continue
        family = str(node.get("family") or "")
        if family == "boundary":
            audit["counts"]["boundary_inherited"] += 1
        elif family == "symbol":
            audit["counts"]["symbol_inherited"] += 1
        elif family == "text":
            audit["counts"]["text_inherited"] += 1
        node.setdefault("metadata", {})["model_version"] = "model_v13"


if __name__ == "__main__":
    main()
