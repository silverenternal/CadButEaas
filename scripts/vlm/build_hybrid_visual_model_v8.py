#!/usr/bin/env python3
"""Build hybrid v8 stream with explicit SVG-candidate plus raster-evidence source labels."""

from __future__ import annotations

import json
from typing import Any

import joblib

from v8_raster_e2e_utils import ROOT, image_ink_features, load_json, load_jsonl, normalize_bbox, update_todo_remove, write_json, write_jsonl


def main() -> None:
    base_rows = load_jsonl("reports/vlm/real_upstream_model_predictions_model_v7.jsonl")
    symbol_eval = load_json("reports/vlm/symbol_visual_evidence_v8_eval.json", {})
    adopted_symbol = bool(symbol_eval.get("adopted"))
    model_bundle = None
    if adopted_symbol and (ROOT / "checkpoints/symbol_visual_evidence_v8/model.joblib").exists():
        model_bundle = joblib.load(ROOT / "checkpoints/symbol_visual_evidence_v8/model.joblib")
    out_rows = []
    evidence_events = []
    for row in base_rows:
        item = json.loads(json.dumps(row, ensure_ascii=False))
        item.setdefault("route_trace", {})["source_mode"] = "hybrid_svg_candidates_plus_raster_evidence"
        item["route_trace"]["candidate_geometry_source"] = "svg_candidate_geometry"
        item["route_trace"]["model_v8_hybrid"] = {
            "model_version": "model_v8_hybrid",
            "base_stream": "reports/vlm/real_upstream_model_predictions_model_v7.jsonl",
            "source_mode": "hybrid_svg_candidates_plus_raster_evidence",
            "pure_raster_e2e": False,
            "adopted_components": ["symbol_visual_evidence_v8"] if adopted_symbol else [],
            "rejected_components": ["raster_candidate_detector_v8"],
            "claim_boundary": "Geometry remains SVG/parser candidates; raster visual evidence is crop-based model evidence only.",
        }
        if model_bundle:
            apply_symbol_visual_evidence(item, model_bundle, evidence_events)
        out_rows.append(item)
    write_jsonl("reports/vlm/hybrid_visual_model_v8_predictions.jsonl", out_rows)
    decisions = {
        "version": "hybrid_visual_model_v8_adoption_decisions",
        "output": "reports/vlm/hybrid_visual_model_v8_predictions.jsonl",
        "rows": len(out_rows),
        "adopted_components": ["symbol_visual_evidence_v8"] if adopted_symbol else [],
        "rejected_components": ["raster_candidate_detector_v8"],
        "symbol_visual_evidence_v8": symbol_eval,
        "raster_candidate_detector_v8": load_json("reports/vlm/raster_candidate_detector_v8_eval.json", {}),
        "evidence_event_count": len(evidence_events),
        "evidence_events_preview": evidence_events[:25],
        "claim_boundary": "Hybrid v8 is not pure raster E2E; it is v7 SVG-candidate geometry plus adopted raster crop evidence.",
    }
    write_json("reports/vlm/hybrid_visual_model_v8_adoption_decisions.json", decisions)
    update_todo_remove(["RASTER-V8-T6"])
    print(json.dumps({"rows": len(out_rows), "adopted_components": decisions["adopted_components"], "events": len(evidence_events)}, ensure_ascii=False, indent=2))


def apply_symbol_visual_evidence(row: dict[str, Any], bundle: dict[str, Any], events: list[dict[str, Any]]) -> None:
    model = bundle["model"]
    features = bundle["features"]
    threshold = float(bundle.get("threshold") or 0.99)
    image = row.get("image") or row.get("image_path")
    if not image:
        return
    nodes = (row.get("scene_graph") or {}).get("nodes") or []
    for node in nodes:
        if not isinstance(node, dict) or node.get("family") != "symbol":
            continue
        bbox = normalize_bbox((node.get("geometry") or {}).get("bbox") or node.get("bbox"))
        if not bbox:
            continue
        feat_dict = image_ink_features(image, bbox)
        vector = [[float(feat_dict.get(name) or 0.0) for name in features]]
        prob_empty = float(model.predict_proba(vector)[0][1])
        metadata = node.setdefault("metadata", {})
        metadata["symbol_visual_evidence_v8"] = {"prob_empty_or_review": round(prob_empty, 6), "threshold": threshold}
        if prob_empty >= threshold:
            node["needs_review"] = True
            node["confidence"] = min(float(node.get("confidence") or 0.5), 0.35)
            events.append({"image": image, "node_id": node.get("id"), "prob_empty_or_review": round(prob_empty, 6), "action": "mark_needs_review_low_visual_evidence"})
            row.setdefault("warnings", []).append(f"symbol_visual_evidence_v8:{node.get('id')}:review_low_visual_evidence")


if __name__ == "__main__":
    main()
