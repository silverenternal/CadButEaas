#!/usr/bin/env python3
"""Collect v7 model/postprocess visual defect ablation metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from v5_pipeline_utils import count_defects, load_json, load_jsonl, summarize_rows, write_json


TRACKED = {"unsupported_wall", "empty_symbol", "missing_visible_text", "room_without_label", "label_without_room", "extra_room"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v5-model-summary", default="reports/vlm/visual_demo/model_defect_summary_model_v5.json")
    parser.add_argument("--v5-post-summary", default="reports/vlm/visual_demo/model_defect_summary_postprocessed_v5.json")
    parser.add_argument("--v6-model-summary", default="reports/vlm/visual_demo/model_defect_summary_model_v6.json")
    parser.add_argument("--v6-post-summary", default="reports/vlm/visual_demo/model_defect_summary_postprocessed_v6.json")
    parser.add_argument("--v7-model-summary", default="reports/vlm/visual_demo/model_defect_summary_model_v7.json")
    parser.add_argument("--v7-post-summary", default="reports/vlm/visual_demo/model_defect_summary_postprocessed_v7.json")
    parser.add_argument("--model-stream", default="reports/vlm/real_upstream_model_predictions_model_v7.jsonl")
    parser.add_argument("--post-stream", default="reports/vlm/real_upstream_model_postprocessed_predictions_v7.jsonl")
    parser.add_argument("--output", default="reports/vlm/visual_defect_ablation_v7.json")
    parser.add_argument("--locked-output", default="reports/vlm/real_model_locked_eval_v7.json")
    args = parser.parse_args()

    summaries = {
        "model_v5": count_defects(args.v5_model_summary),
        "postprocess_v5": count_defects(args.v5_post_summary),
        "model_v6": count_defects(args.v6_model_summary),
        "postprocess_v6": count_defects(args.v6_post_summary),
        "model_v7": count_defects(args.v7_model_summary),
        "postprocess_v7": count_defects(args.v7_post_summary),
    }
    keys = sorted({k for counts in summaries.values() for k in counts} | TRACKED)
    deltas = {
        key: {
            **{name: summaries[name].get(key, 0) for name in summaries},
            "model_v7_delta_from_model_v5": summaries["model_v7"].get(key, 0) - summaries["model_v5"].get(key, 0),
            "postprocess_v7_delta_from_model_v7": summaries["postprocess_v7"].get(key, 0) - summaries["model_v7"].get(key, 0),
        }
        for key in keys
    }
    boundary = load_json("reports/vlm/boundary_geometry_refiner_v7_eval.json", {})
    symbol = load_json("reports/vlm/symbol_fixture_expert_v13_eval.json", {})
    decisions = load_json("reports/vlm/model_v7_adoption_decisions.json", {})
    post = load_json("reports/vlm/postprocess_v7_ablation.json", {})
    locked = {
        "version": "real_model_locked_eval_v7",
        "adopted_experts": decisions.get("adopted_experts") or [],
        "boundary_geometry_refiner_v7": boundary,
        "symbol_fixture_expert_v13": symbol,
        "postprocess_v7": post,
        "visual_demo_metrics": deltas,
        "model_stream_summary": summarize_rows(load_jsonl(args.model_stream)),
        "postprocess_stream_summary": summarize_rows(load_jsonl(args.post_stream)),
        "claim_boundary": "Boundary v7 is a full locked adopted model-side geometry refiner. SymbolFixture v13 was trained but rejected; appliance/equipment cleanup is postprocess_v7.",
    }
    report = {
        "version": "visual_defect_ablation_v7",
        "summaries": {
            "model_v5": args.v5_model_summary,
            "postprocess_v5": args.v5_post_summary,
            "model_v6": args.v6_model_summary,
            "postprocess_v6": args.v6_post_summary,
            "model_v7": args.v7_model_summary,
            "postprocess_v7": args.v7_post_summary,
        },
        "deltas": deltas,
        "adopted_model_components": locked["adopted_experts"],
        "rejected_model_components": ["symbol_fixture_expert_v13"] if symbol and not symbol.get("adopted") else [],
        "claim_boundary": locked["claim_boundary"],
    }
    write_json(args.output, report)
    write_json(args.locked_output, locked)
    print(json.dumps({"output": args.output, "locked": args.locked_output, "adopted": locked["adopted_experts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
