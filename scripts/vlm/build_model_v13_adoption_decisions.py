#!/usr/bin/env python3
"""Build model_v13 adoption decisions from accepted frontier specialists."""

from __future__ import annotations

import argparse
import json
from typing import Any

from v5_pipeline_utils import load_json, load_jsonl, summarize_rows, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-stream", default="reports/vlm/real_upstream_model_predictions_model_v7.jsonl")
    parser.add_argument("--output-stream", default="reports/vlm/real_upstream_model_predictions_model_v13.jsonl")
    parser.add_argument("--decisions", default="reports/vlm/model_v13_adoption_decisions.json")
    parser.add_argument("--locked-output", default="reports/vlm/real_model_locked_eval_v13.json")
    args = parser.parse_args()

    reports = {
        "room_space_expert_v13": load_json("reports/vlm/room_space_expert_v13_eval.json", {}),
        "boundary_geometry_refiner_v13": load_json("reports/vlm/boundary_expert_v13_eval.json", {}),
        "symbol_fixture_expert_v13": load_json("reports/vlm/symbol_fixture_expert_v13_eval.json", {}),
        "text_dimension_expert_v13": load_json("reports/vlm/text_dimension_expert_v13_eval.json", {}),
    }
    adopted = [name for name, report in reports.items() if report.get("adopted")]
    component_map = {
        "room_space": component("room_space_expert_v13", reports["room_space_expert_v13"]),
        "boundary": component("boundary_geometry_refiner_v13", reports["boundary_geometry_refiner_v13"]),
        "symbol_fixture": component("symbol_fixture_expert_v13", reports["symbol_fixture_expert_v13"]),
        "text_dimension": component("text_dimension_expert_v13", reports["text_dimension_expert_v13"]),
    }

    rows = load_jsonl(args.base_stream)
    out = [mark_model_v13(row, adopted, component_map, args.base_stream) for row in rows]
    write_jsonl(args.output_stream, out)

    decisions = {
        "version": "model_v13_adoption_decisions",
        "base_stream": args.base_stream,
        "output_stream": args.output_stream,
        "adopted_experts": adopted,
        "component_map": component_map,
        "expert_reports": reports,
        "stream_summary": summarize_rows(out),
        "claim_boundary": "model_v13 integrates accepted specialist checkpoints at the MoE decision layer. RoomSpace v13 replaces v3 for candidate-level room classification; this is still not pure raster end-to-end room polygon detection.",
    }
    locked = {
        "version": "real_model_locked_eval_v13",
        "adopted_experts": adopted,
        "component_map": component_map,
        "expert_decisions": reports,
        "model_stream_summary": summarize_rows(out),
        "claim_boundary": decisions["claim_boundary"],
    }
    write_json(args.decisions, decisions)
    write_json(args.locked_output, locked)
    print(json.dumps({"output": args.output_stream, "adopted": adopted, "rows": len(out)}, ensure_ascii=False, indent=2))


def component(name: str, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": report.get("adopted_model") or name,
        "adopted": bool(report.get("adopted")),
        "checkpoint": report.get("checkpoint"),
        "locked_macro_f1": locked_macro_f1(report),
        "report_version": report.get("version"),
    }


def mark_model_v13(row: dict[str, Any], adopted: list[str], component_map: dict[str, Any], base_stream: str) -> dict[str, Any]:
    item = json.loads(json.dumps(row, ensure_ascii=False))
    item.setdefault("route_trace", {})["model_v13"] = {
        "model_version": "model_v13",
        "base_stream": base_stream,
        "adopted_experts": adopted,
        "component_map": component_map,
        "claim_boundary": "Accepted v13 specialists are integrated at the MoE decision layer.",
    }
    for node in ((item.get("scene_graph") or {}).get("nodes") or []):
        if not isinstance(node, dict):
            continue
        metadata = node.setdefault("metadata", {})
        if not isinstance(metadata, dict):
            continue
        metadata["model_version"] = "model_v13"
        family = str(node.get("family") or "")
        if family == "space" and component_map["room_space"]["adopted"]:
            metadata["room_space_expert"] = component_map["room_space"]["name"]
            metadata["room_space_checkpoint"] = component_map["room_space"]["checkpoint"]
    return item


def locked_macro_f1(report: dict[str, Any]) -> float | None:
    for key in ["locked_metrics", "locked_symbol_metrics", "locked_boundary_metrics"]:
        value = report.get(key)
        if isinstance(value, dict) and value.get("macro_f1") is not None:
            return float(value["macro_f1"])
    if report.get("locked_macro_f1") is not None:
        return float(report["locked_macro_f1"])
    return None


if __name__ == "__main__":
    main()
