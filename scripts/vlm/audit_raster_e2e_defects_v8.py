#!/usr/bin/env python3
"""Audit v8 raster E2E, hybrid, and visual defect attribution."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from v8_raster_e2e_utils import FAMILIES, f1, load_json, load_jsonl, update_todo_remove, write_json, write_jsonl


def main() -> None:
    detector = load_json("reports/vlm/raster_candidate_detector_v8_eval.json", {})
    raster_decision = load_json("reports/vlm/raster_e2e_model_v8_adoption_decisions.json", {})
    hybrid_decision = load_json("reports/vlm/hybrid_visual_model_v8_adoption_decisions.json", {})
    v7_ablation = load_json("reports/vlm/visual_defect_ablation_v7.json", {})
    error_cases = load_jsonl("reports/vlm/raster_candidate_detector_v8_error_cases.jsonl")
    locked_eval = detector.get("locked_eval") or {}
    report = {
        "version": "raster_e2e_defect_audit_v8",
        "raster_detector": {
            "adopted": detector.get("adopted"),
            "macro_f1": detector.get("macro_f1"),
            "per_family": locked_eval.get("per_family", {}),
            "candidate_inflation": locked_eval.get("candidate_inflation", {}),
        },
        "raster_e2e_stream": {
            "adopted": raster_decision.get("adopted"),
            "rows": raster_decision.get("rows"),
            "failure_mode": (raster_decision.get("no_adoption") or {}).get("reason"),
        },
        "hybrid_stream": {
            "rows": hybrid_decision.get("rows"),
            "adopted_components": hybrid_decision.get("adopted_components"),
            "evidence_event_count": hybrid_decision.get("evidence_event_count"),
            "source_mode": "hybrid_svg_candidates_plus_raster_evidence",
        },
        "attribution": {
            "detector_false_positive": count_cases(error_cases, "detector_false_positive"),
            "detector_miss": count_cases(error_cases, "detector_miss"),
            "expert_label_error": "not evaluated for raster_e2e because detector rejected; v7 SVG-candidate expert metrics remain baseline",
            "renderer_error": "not implicated by raster detector rejection; renderer source-mode labels are audited separately",
            "postprocess_effect": "empty_symbol cleanup remains postprocess in v7; v8 has model-side symbol visual evidence only as review flags in hybrid stream",
        },
        "v7_baseline_visual_defects": (v7_ablation.get("summaries") or {}).get("model_v7", {}),
        "claim_boundary": "Detector errors are separated from expert classification and renderer/postprocess effects.",
    }
    write_json("reports/vlm/raster_e2e_defect_audit_v8.json", report)
    write_jsonl("reports/vlm/raster_e2e_detection_error_cases_v8.jsonl", error_cases[:1000])
    write_json(
        "reports/vlm/visual_defect_ablation_v8.json",
        {
            "version": "visual_defect_ablation_v8",
            "v7_baseline": v7_ablation.get("summaries", {}),
            "v8_raster_e2e": {"adopted": raster_decision.get("adopted"), "defect_status": "rejected_detector_no_visual_success_claim"},
            "v8_hybrid": {"source_mode": "hybrid_svg_candidates_plus_raster_evidence", "symbol_visual_evidence_events": hybrid_decision.get("evidence_event_count")},
            "claim_boundary": "v8_raster_e2e is rejected; hybrid is not pure raster.",
        },
    )
    write_json(
        "reports/vlm/real_model_locked_eval_v8.json",
        {
            "version": "real_model_locked_eval_v8",
            "raster_candidate_detector_v8": detector,
            "symbol_visual_evidence_v8": load_json("reports/vlm/symbol_visual_evidence_v8_eval.json", {}),
            "raster_e2e_decision": raster_decision,
            "hybrid_decision": hybrid_decision,
            "claim_boundary": "Pure raster E2E is not adopted. Hybrid v8 may use adopted symbol visual-evidence review flags over SVG candidate geometry.",
        },
    )
    update_todo_remove(["RASTER-V8-T8"])
    print(json.dumps({"detector_adopted": detector.get("adopted"), "error_cases": len(error_cases)}, ensure_ascii=False, indent=2))


def count_cases(rows: list[dict[str, Any]], error_type: str) -> dict[str, int]:
    counter = Counter(str(row.get("family") or "unknown") for row in rows if row.get("error_type") == error_type)
    return dict(counter)


if __name__ == "__main__":
    main()
