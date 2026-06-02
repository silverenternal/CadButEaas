#!/usr/bin/env python3
"""Build raster_e2e v8 prediction stream or explicit no-adoption report."""

from __future__ import annotations

import json
from typing import Any

from v8_raster_e2e_utils import load_json, load_jsonl, update_todo_remove, write_json, write_jsonl


def main() -> None:
    detector = load_json("reports/vlm/raster_candidate_detector_v8_eval.json", {})
    detector_predictions = load_jsonl("reports/vlm/raster_candidate_detector_v8_locked_predictions.jsonl")
    adopted = bool(detector.get("adopted"))
    rows: list[dict[str, Any]] = []
    if adopted:
        for row in detector_predictions:
            nodes = []
            for item in row.get("proposals") or []:
                nodes.append(
                    {
                        "id": item.get("id"),
                        "semantic_type": item.get("semantic_type") or item.get("family"),
                        "family": item.get("family"),
                        "confidence": item.get("confidence", 0.5),
                        "geometry": {"bbox": item.get("bbox")},
                        "source_expert": "raster_candidate_detector_v8",
                        "audit_trace": {"origin": "raster_image_only_component"},
                        "metadata": {"proposal_source": "raster_candidate_detector_v8", "model_version": "model_v8_raster"},
                    }
                )
            rows.append(
                {
                    "image": row.get("image"),
                    "source_dataset": row.get("source_dataset"),
                    "split": "locked",
                    "scene_graph": {"nodes": nodes, "edges": []},
                    "warnings": ["raster_e2e_v8:no_relations_baseline"],
                    "quality_report": {"candidate_counts": count_families(nodes), "model_output_contract": "raster image-only proposals"},
                    "route_trace": {
                        "source_mode": "raster_e2e",
                        "proposal_source": "raster_candidate_detector_v8",
                        "svg_candidate_ids_used": False,
                        "model_v8_raster": {"adopted": True, "components": ["raster_candidate_detector_v8"]},
                    },
                    "gold_source": "offline_svg_labels_for_locked_eval_only",
                }
            )
    write_jsonl("reports/vlm/raster_e2e_model_v8_predictions.jsonl", rows)
    decisions = {
        "version": "raster_e2e_model_v8_adoption_decisions",
        "adopted": adopted,
        "output_stream": "reports/vlm/raster_e2e_model_v8_predictions.jsonl",
        "rows": len(rows),
        "detector": detector,
        "no_adoption": None
        if adopted
        else {
            "reason": "raster_candidate_detector_v8 failed locked adoption guard",
            "stream_status": "explicitly_absent_for_claims_empty_predictions_file_written_for_pipeline_stability",
            "claim_boundary": "Do not present v8 raster_e2e as recognized floorplan output; use it only as failure evidence.",
        },
        "claim_boundary": "This stream is pure raster only if detector adopted=true. SVG candidate geometry is never used here.",
    }
    write_json("reports/vlm/raster_e2e_model_v8_adoption_decisions.json", decisions)
    update_todo_remove(["RASTER-V8-T5"])
    print(json.dumps({"adopted": adopted, "rows": len(rows)}, ensure_ascii=False, indent=2))


def count_families(nodes: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for node in nodes:
        family = str(node.get("family") or "unknown")
        out[family] = out.get(family, 0) + 1
    return out


if __name__ == "__main__":
    main()
