#!/usr/bin/env python3
"""Build the v5 raw/model/postprocess error ledger for CubiCasa visual demos."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from v5_pipeline_utils import (
    count_defects,
    find_node,
    load_json,
    load_jsonl,
    markdown_table,
    model_probabilities,
    sample_id,
    write_json,
)


OWNER_BY_DEFECT = {
    "missing_visible_text": "parser_candidate",
    "unsupported_wall": "renderer",
    "empty_symbol": "expert_model",
    "needs_review_symbol": "expert_model",
    "room_without_label": "fusion_relation",
    "label_without_room": "fusion_relation",
    "extra_room": "parser_candidate",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-predictions", default="reports/vlm/real_upstream_model_postprocessed_predictions_roomlink_v3.jsonl")
    parser.add_argument("--postprocessed-predictions", default="reports/vlm/real_upstream_model_postprocessed_predictions_v4.jsonl")
    parser.add_argument("--cases", default="reports/vlm/visual_demo/model_defect_cases_roomspace_v4.jsonl")
    parser.add_argument("--ablation", default="reports/vlm/visual_defect_ablation_v4.json")
    parser.add_argument("--raw-summary", default="reports/vlm/visual_demo/model_defect_summary_roomspace_v4.json")
    parser.add_argument("--post-summary", default="reports/vlm/visual_demo/model_defect_summary_v4.json")
    parser.add_argument("--output-json", default="reports/vlm/real_model_error_ledger_v5.json")
    parser.add_argument("--output-md", default="reports/vlm/real_model_error_ledger_v5.md")
    args = parser.parse_args()

    raw_rows = {sample_id(row): row for row in load_jsonl(args.raw_predictions)}
    post_rows = {sample_id(row): row for row in load_jsonl(args.postprocessed_predictions)}
    cases = load_jsonl(args.cases)
    ablation = load_json(args.ablation, {})
    raw_counts = count_defects(args.raw_summary)
    post_counts = count_defects(args.post_summary)

    ledger: list[dict[str, Any]] = []
    hidden_by_postprocess: Counter[str] = Counter()
    owners: Counter[str] = Counter()
    for case in cases:
        sid = str(case.get("sample_id") or sample_id(case))
        node_id = str(case.get("node_id") or "")
        defect = str(case.get("type") or "unknown")
        raw_node = find_node(raw_rows.get(sid, {}), node_id)
        post_node = find_node(post_rows.get(sid, {}), node_id)
        owner = owner_for(defect, raw_node, case)
        owners[owner] += 1
        hidden = bool(raw_node) and not bool(post_node)
        if hidden or post_counts.get(defect, 0) < raw_counts.get(defect, 0):
            hidden_by_postprocess[defect] += 1
        ledger.append(
            {
                "sample_id": sid,
                "node_id": node_id,
                "defect_type": defect,
                "family": case.get("family"),
                "raw_label": case.get("raw_label"),
                "model_label": raw_node.get("semantic_type") or case.get("semantic_type"),
                "confidence": raw_node.get("confidence") or case.get("confidence"),
                "bbox": case.get("bbox") or ((raw_node.get("geometry") or {}).get("bbox") if raw_node else None),
                "primary_owner": owner,
                "postprocess_hidden_or_fixed": hidden or post_counts.get(defect, 0) < raw_counts.get(defect, 0),
                "raw_node_exists": bool(raw_node),
                "postprocess_node_exists": bool(post_node),
                "model_probabilities": model_probabilities(raw_node),
                "claim_boundary": "This is an error-attribution ledger over saved model labels and parser/SVG candidates, not a pure raster detector benchmark.",
            }
        )

    report = {
        "version": "real_model_error_ledger_v5",
        "inputs": {
            "raw_predictions": args.raw_predictions,
            "postprocessed_predictions": args.postprocessed_predictions,
            "cases": args.cases,
            "ablation": args.ablation,
        },
        "raw_counts": raw_counts,
        "postprocess_adjusted_counts": post_counts,
        "owner_counts": dict(owners.most_common()),
        "postprocess_hidden_or_fixed_counts": dict(hidden_by_postprocess.most_common()),
        "v4_ablation_deltas": ablation.get("deltas") or {},
        "suppression_warning": "v4 zero-defect visual summary is a postprocessed presentation stream. The raw ledger still contains true upstream/parser/model issues.",
        "cases": ledger,
    }
    write_json(args.output_json, report)
    Path(args.output_md).write_text(render_markdown(report), encoding="utf-8")
    print({"cases": len(ledger), "owners": report["owner_counts"], "hidden": report["postprocess_hidden_or_fixed_counts"]})


def owner_for(defect: str, node: dict[str, Any], case: dict[str, Any]) -> str:
    if defect == "unsupported_wall":
        aspect = float(((case.get("evidence") or {}).get("aspect_ratio") if isinstance(case.get("evidence"), dict) else 0.0) or 0.0)
        geometry = node.get("geometry") if isinstance(node.get("geometry"), dict) else {}
        if geometry.get("source_geometry") or aspect > 50:
            return "renderer"
    return OWNER_BY_DEFECT.get(defect, "expert_model")


def render_markdown(report: dict[str, Any]) -> str:
    rows = [["Defect", "Raw", "Postprocess", "Delta"]]
    keys = sorted(set(report["raw_counts"]) | set(report["postprocess_adjusted_counts"]))
    for key in keys:
        raw = int(report["raw_counts"].get(key, 0))
        post = int(report["postprocess_adjusted_counts"].get(key, 0))
        rows.append([key, raw, post, post - raw])
    owner_rows = [["Owner", "Count"], *[[k, v] for k, v in report["owner_counts"].items()]]
    return "\n\n".join(
        [
            "# CadStruct-MoE Real Model Error Ledger v5",
            report["suppression_warning"],
            "## Raw vs Postprocess",
            markdown_table(rows),
            "## Primary Owners",
            markdown_table(owner_rows),
        ]
    ) + "\n"


if __name__ == "__main__":
    main()
