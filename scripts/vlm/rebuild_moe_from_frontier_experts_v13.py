#!/usr/bin/env python3
"""Rebuild MoE routing audits only after specialist gains are verified."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from v5_pipeline_utils import load_json, write_json


def locked_macro_f1(report: dict) -> float | None:
    for key in ["locked_metrics", "locked_symbol_metrics", "locked_boundary_metrics"]:
        value = report.get(key)
        if isinstance(value, dict) and value.get("macro_f1") is not None:
            return float(value["macro_f1"])
    if report.get("locked_macro_f1") is not None:
        return float(report["locked_macro_f1"])
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundary", default="reports/vlm/boundary_expert_v13_eval.json")
    parser.add_argument("--room", default="reports/vlm/room_space_expert_v13_eval.json")
    parser.add_argument("--symbol", default="reports/vlm/symbol_fixture_expert_v13_eval.json")
    parser.add_argument("--text", default="reports/vlm/text_dimension_expert_v13_eval.json")
    parser.add_argument("--output", default="reports/vlm/moe_frontier_reintegration_v13.json")
    parser.add_argument("--matrix", default="reports/vlm/expert_contribution_matrix_v13.json")
    args = parser.parse_args()

    report_paths = {"boundary": args.boundary, "room": args.room, "symbol": args.symbol, "text": args.text}
    reports = {name: load_json(path, {}) for name, path in report_paths.items()}
    adopted = {name: bool(report.get("adopted")) for name, report in reports.items()}
    improved = sum(adopted.values()) >= 2
    adopted_experts = {
        name: {
            "adopted_model": report.get("adopted_model"),
            "checkpoint": report.get("checkpoint"),
            "report": report_paths[name],
            "locked_macro_f1": locked_macro_f1(report),
            "claim_boundary": report.get("claim_boundary"),
        }
        for name, report in reports.items()
        if report.get("adopted")
    }
    rejected_experts = {
        name: {
            "adopted_model": report.get("adopted_model"),
            "candidate_report": report_paths[name],
            "locked_macro_f1": locked_macro_f1(report),
            "reason": report.get("reason") or "candidate did not pass its adoption gate",
        }
        for name, report in reports.items()
        if not report.get("adopted")
    }
    matrix = {
        "version": "expert_contribution_matrix_v13",
        "adopted": adopted,
        "adopted_experts": adopted_experts,
        "rejected_experts": rejected_experts,
        "specialists_improved": improved,
        "contribution_policy": "only refresh routing after at least two specialists improve on locked evaluation",
        "claims": {
            "boundary": reports["boundary"].get("claim_boundary"),
            "room": reports["room"].get("claim_boundary"),
            "symbol": reports["symbol"].get("claim_boundary"),
            "text": reports["text"].get("claim_boundary"),
        },
    }
    out = {
        "version": "moe_frontier_reintegration_v13",
        "specialists": adopted,
        "adopted_experts": adopted_experts,
        "rejected_experts": rejected_experts,
        "specialists_improved": improved,
        "router_refresh_allowed": improved,
        "runtime_component_map": {
            "room_space": adopted_experts.get("room", {}).get("checkpoint") or "checkpoints/room_space_expert_v3/model.joblib",
            "boundary": adopted_experts.get("boundary", {}).get("checkpoint"),
            "symbol_fixture": adopted_experts.get("symbol", {}).get("checkpoint"),
            "text_dimension": adopted_experts.get("text", {}).get("checkpoint"),
        },
        "claim_boundary": "MoE refresh is an audit outcome only after specialists improve; it does not overwrite the protected v7/v8 baseline.",
    }
    write_json(args.output, out)
    write_json(args.matrix, matrix)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
