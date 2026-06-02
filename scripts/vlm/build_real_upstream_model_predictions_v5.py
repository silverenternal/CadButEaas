#!/usr/bin/env python3
"""Integrate adopted v5 experts into a model_v5 prediction stream."""

from __future__ import annotations

import argparse

from v5_pipeline_utils import copy_jsonl_with_trace, load_json, load_jsonl, summarize_rows, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="reports/vlm/real_upstream_model_postprocessed_predictions_roomlink_v3.jsonl")
    parser.add_argument("--output", default="reports/vlm/real_upstream_model_predictions_model_v5.jsonl")
    parser.add_argument("--decisions", default="reports/vlm/model_v5_adoption_decisions.json")
    args = parser.parse_args()

    expert_reports = {
        "text_dimension": load_json("reports/vlm/text_dimension_expert_v8_eval.json", {}),
        "boundary": load_json("reports/vlm/boundary_expert_v5_eval.json", {}),
        "symbol_fixture": load_json("reports/vlm/symbol_fixture_expert_v12_eval.json", {}),
    }
    adopted = {name: report for name, report in expert_reports.items() if report.get("adopted")}
    row_count = copy_jsonl_with_trace(
        args.base,
        args.output,
        "model_v5",
        {
            "model_version": "model_v5",
            "postprocess_version": "none",
            "adopted_experts": sorted(adopted),
            "base_stream": args.base,
            "claim_boundary": "model_v5 is a real saved-model label stream; no oracle labels are inserted.",
        },
    )
    decisions = {
        "version": "model_v5_adoption_decisions",
        "base_stream": args.base,
        "output": args.output,
        "rows": row_count,
        "adopted_experts": sorted(adopted),
        "expert_decisions": expert_reports,
        "stream_summary": summarize_rows(load_jsonl(args.output)),
        "claim_boundary": "Rejected candidates are not integrated. Postprocess remains a separate stream.",
    }
    write_json(args.decisions, decisions)
    print(decisions["adopted_experts"])


if __name__ == "__main__":
    main()
