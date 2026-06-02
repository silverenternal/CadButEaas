#!/usr/bin/env python3
"""Build model_v7 stream from adopted retrained/calibrated components only."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from v5_pipeline_utils import load_json, load_jsonl, summarize_rows, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="reports/vlm/real_upstream_model_predictions_model_v5.jsonl")
    parser.add_argument("--boundary-stream", default="reports/vlm/real_upstream_model_predictions_model_v7_boundary.jsonl")
    parser.add_argument("--output", default="reports/vlm/real_upstream_model_predictions_model_v7.jsonl")
    parser.add_argument("--decisions", default="reports/vlm/model_v7_adoption_decisions.json")
    args = parser.parse_args()

    boundary = load_json("reports/vlm/boundary_geometry_refiner_v7_eval.json", {})
    symbol = load_json("reports/vlm/symbol_fixture_expert_v13_eval.json", {})
    adopted: list[str] = []
    source = args.base
    if boundary.get("adopted") and boundary.get("run_mode") == "full":
        source = args.boundary_stream
        adopted.append("boundary_geometry_refiner_v7")
    if symbol.get("adopted"):
        adopted.append("symbol_fixture_expert_v13")

    rows = load_jsonl(source)
    out: list[dict[str, Any]] = []
    for row in rows:
        item = json.loads(json.dumps(row, ensure_ascii=False))
        item.setdefault("route_trace", {})["model_v7"] = {
            "model_version": "model_v7",
            "base_stream": args.base,
            "source_stream": source,
            "adopted_experts": adopted,
            "rejected_candidates": [name for name, report in {"symbol_fixture_expert_v13": symbol}.items() if report and not report.get("adopted")],
            "claim_boundary": "Only full-locked adopted model components are integrated. Postprocess fixes are not part of this stream.",
        }
        for node in ((item.get("scene_graph") or {}).get("nodes") or []):
            if isinstance(node, dict):
                metadata = node.setdefault("metadata", {})
                if isinstance(metadata, dict):
                    metadata["model_version"] = "model_v7"
                    metadata.setdefault("postprocess_version", "none")
        out.append(item)
    write_jsonl(args.output, out)

    decisions = {
        "version": "model_v7_adoption_decisions",
        "base_stream": args.base,
        "output": args.output,
        "rows": len(out),
        "adopted_experts": adopted,
        "boundary_geometry_refiner_v7": boundary,
        "symbol_fixture_expert_v13": symbol,
        "stream_summary": summarize_rows(out),
        "claim_boundary": "Boundary v7 is adopted only when full locked eval passes; SymbolFixture v13 is rejected if it does not beat v11 guards.",
    }
    write_json(args.decisions, decisions)
    print(json.dumps({"output": args.output, "adopted": adopted, "rows": len(out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
