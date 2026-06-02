#!/usr/bin/env python3
"""Collect v5 locked/visual metrics into final ablation reports."""

from __future__ import annotations

import argparse

from v5_pipeline_utils import count_defects, load_json, load_jsonl, summarize_rows, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-summary", default="reports/vlm/visual_demo/model_defect_summary_raw_for_v5.json")
    parser.add_argument("--model-summary", default="reports/vlm/visual_demo/model_defect_summary_model_v5.json")
    parser.add_argument("--post-summary", default="reports/vlm/visual_demo/model_defect_summary_postprocessed_v5.json")
    parser.add_argument("--model-stream", default="reports/vlm/real_upstream_model_predictions_model_v5.jsonl")
    parser.add_argument("--post-stream", default="reports/vlm/real_upstream_model_postprocessed_predictions_v5.jsonl")
    parser.add_argument("--adoption", default="reports/vlm/model_v5_adoption_decisions.json")
    parser.add_argument("--locked-output", default="reports/vlm/real_model_locked_eval_v5.json")
    parser.add_argument("--output", default="reports/vlm/visual_defect_ablation_v5.json")
    args = parser.parse_args()

    raw = count_defects(args.raw_summary)
    model = count_defects(args.model_summary)
    post = count_defects(args.post_summary)
    tracked = sorted(set(raw) | set(model) | set(post) | {"missing_visible_text", "unsupported_wall", "empty_symbol", "room_without_label", "label_without_room", "extra_room"})
    deltas = {
        key: {
            "raw": int(raw.get(key, 0)),
            "model_v5": int(model.get(key, 0)),
            "postprocess_v5": int(post.get(key, 0)),
            "model_delta_from_raw": int(model.get(key, 0)) - int(raw.get(key, 0)),
            "post_delta_from_model": int(post.get(key, 0)) - int(model.get(key, 0)),
        }
        for key in tracked
    }
    adoption = load_json(args.adoption, {})
    model_rows = load_jsonl(args.model_stream)
    post_rows = load_jsonl(args.post_stream)
    locked = {
        "version": "real_model_locked_eval_v5",
        "adopted_experts": adoption.get("adopted_experts") or [],
        "expert_decisions": adoption.get("expert_decisions") or {},
        "visual_demo_metrics": deltas,
        "model_stream_summary": summarize_rows(model_rows),
        "postprocess_stream_summary": summarize_rows(post_rows),
        "scene_invalid_graph_rate": summarize_rows(post_rows)["invalid_graph_rate"],
        "claim_boundary": "Raw model, model_v5, and postprocess_v5 metrics are separated. No rejected retrained expert is counted as adopted.",
    }
    ablation = {
        "version": "visual_defect_ablation_v5",
        "summaries": {"raw": args.raw_summary, "model_v5": args.model_summary, "postprocess_v5": args.post_summary},
        "deltas": deltas,
        "adoption_decisions": args.adoption,
        "claim_boundary": locked["claim_boundary"],
    }
    write_json(args.locked_output, locked)
    write_json(args.output, ablation)
    print({"locked": args.locked_output, "ablation": args.output, "adopted": locked["adopted_experts"]})


if __name__ == "__main__":
    main()
