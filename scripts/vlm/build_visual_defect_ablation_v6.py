#!/usr/bin/env python3
"""Collect v6 visual-model and postprocess ablation metrics."""

from __future__ import annotations

import argparse

from v5_pipeline_utils import count_defects, load_json, load_jsonl, summarize_rows, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v5-model-summary", default="reports/vlm/visual_demo/model_defect_summary_model_v5.json")
    parser.add_argument("--v5-post-summary", default="reports/vlm/visual_demo/model_defect_summary_postprocessed_v5.json")
    parser.add_argument("--v6-model-summary", default="reports/vlm/visual_demo/model_defect_summary_model_v6.json")
    parser.add_argument("--v6-post-summary", default="reports/vlm/visual_demo/model_defect_summary_postprocessed_v6.json")
    parser.add_argument("--boundary-eval", default="reports/vlm/boundary_geometry_refiner_v6_eval.json")
    parser.add_argument("--model-stream", default="reports/vlm/real_upstream_model_predictions_model_v6.jsonl")
    parser.add_argument("--post-stream", default="reports/vlm/real_upstream_model_postprocessed_predictions_v6.jsonl")
    parser.add_argument("--output", default="reports/vlm/visual_defect_ablation_v6.json")
    parser.add_argument("--locked-output", default="reports/vlm/real_model_locked_eval_v6.json")
    args = parser.parse_args()

    summaries = {
        "model_v5": count_defects(args.v5_model_summary),
        "postprocess_v5": count_defects(args.v5_post_summary),
        "model_v6": count_defects(args.v6_model_summary),
        "postprocess_v6": count_defects(args.v6_post_summary),
    }
    keys = sorted({key for counts in summaries.values() for key in counts} | {"unsupported_wall", "empty_symbol", "missing_visible_text", "room_without_label", "label_without_room", "extra_room"})
    deltas = {
        key: {
            "model_v5": summaries["model_v5"].get(key, 0),
            "postprocess_v5": summaries["postprocess_v5"].get(key, 0),
            "model_v6": summaries["model_v6"].get(key, 0),
            "postprocess_v6": summaries["postprocess_v6"].get(key, 0),
            "model_v6_delta_from_model_v5": summaries["model_v6"].get(key, 0) - summaries["model_v5"].get(key, 0),
            "postprocess_v6_delta_from_model_v6": summaries["postprocess_v6"].get(key, 0) - summaries["model_v6"].get(key, 0),
        }
        for key in keys
    }
    boundary = load_json(args.boundary_eval, {})
    locked = {
        "version": "real_model_locked_eval_v6",
        "adopted_experts": ["boundary_geometry_refiner_v6"] if boundary.get("adopted") else [],
        "boundary_geometry_refiner_v6": boundary,
        "visual_demo_metrics": deltas,
        "model_stream_summary": summarize_rows(load_jsonl(args.model_stream)),
        "postprocess_stream_summary": summarize_rows(load_jsonl(args.post_stream)),
        "claim_boundary": "Boundary v6 is a trained geometry-output refiner adopted into the model stream. Symbol appliance/equipment cleanup remains postprocess.",
    }
    report = {
        "version": "visual_defect_ablation_v6",
        "summaries": {
            "model_v5": args.v5_model_summary,
            "postprocess_v5": args.v5_post_summary,
            "model_v6": args.v6_model_summary,
            "postprocess_v6": args.v6_post_summary,
        },
        "deltas": deltas,
        "adopted_model_components": locked["adopted_experts"],
        "claim_boundary": locked["claim_boundary"],
    }
    write_json(args.output, report)
    write_json(args.locked_output, locked)
    print({"output": args.output, "locked": args.locked_output, "adopted": locked["adopted_experts"]})


if __name__ == "__main__":
    main()
