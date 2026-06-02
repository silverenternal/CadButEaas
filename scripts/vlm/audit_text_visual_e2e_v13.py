#!/usr/bin/env python3
"""Audit model_v13 visual text localization and numeric-text coverage."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_e2e_scene_graph import f1
from v5_pipeline_utils import load_jsonl, update_todo_remove, write_json, write_jsonl


NUMERIC_RE = re.compile(r"\d")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="reports/vlm/real_upstream_model_predictions_model_v13_real_infer_rel_contains_boundary_v14.jsonl")
    parser.add_argument("--gold", default="datasets/cadstruct_cubicasa5k_moe_locked_reviewed_v1/locked_test_text_aware_v13.jsonl")
    parser.add_argument("--output", default="reports/vlm/text_visual_e2e_v13_eval.json")
    parser.add_argument("--cases", default="reports/vlm/text_visual_e2e_v13_cases.jsonl")
    parser.add_argument("--update-todo", action="store_true")
    args = parser.parse_args()

    gold_rows = {sample_key(row): row for row in load_jsonl(args.gold)}
    totals = Counter()
    cases = []
    for pred in load_jsonl(args.predictions):
        key = sample_key(pred)
        gold = gold_rows.get(key, {})
        pred_text = text_nodes(pred)
        gold_text = text_nodes_from_gold(gold)
        pred_by_id = {str(node.get("id")): node for node in pred_text}
        gold_by_id = {str(node.get("id")): node for node in gold_text}
        matched = sorted(set(pred_by_id) & set(gold_by_id))
        totals["predicted"] += len(pred_by_id)
        totals["gold"] += len(gold_by_id)
        totals["tp"] += sum(1 for node_id in matched if pred_by_id[node_id].get("semantic_type") == gold_by_id[node_id].get("semantic_type"))
        for node in gold_by_id.values():
            if NUMERIC_RE.search(text_value(node)):
                totals["numeric_gold"] += 1
                pred_node = pred_by_id.get(str(node.get("id")))
                if pred_node and pred_node.get("semantic_type") == node.get("semantic_type"):
                    totals["numeric_matched"] += 1
        false_non_text = [
            {"id": node_id, "pred": pred_by_id[node_id].get("semantic_type"), "bbox": ((pred_by_id[node_id].get("geometry") or {}).get("bbox"))}
            for node_id in sorted(set(pred_by_id) - set(gold_by_id))[:80]
        ]
        missing_numeric = [
            {"id": node_id, "gold": gold_by_id[node_id].get("semantic_type"), "text": text_value(gold_by_id[node_id])}
            for node_id in sorted(set(gold_by_id) - set(pred_by_id))
            if NUMERIC_RE.search(text_value(gold_by_id[node_id]))
        ][:80]
        if false_non_text or missing_numeric:
            cases.append({"sample_id": key, "false_text_on_non_text": false_non_text, "missing_numeric_text": missing_numeric})
    report = {
        "version": "text_visual_e2e_v13_eval",
        "predictions": args.predictions,
        "gold": args.gold,
        "text_node_f1": f1(totals["tp"], totals["predicted"], totals["gold"]),
        "numeric_text_recall": round(totals["numeric_matched"] / max(totals["numeric_gold"], 1), 6),
        "counts": dict(totals),
        "case_count": len(cases),
        "claim_boundary": "Text E2E here evaluates SVG text candidate ids/types and numeric presence. It separates this from OCR transcription quality.",
    }
    write_json(args.output, report)
    write_jsonl(args.cases, cases)
    if args.update_todo:
        update_todo_remove(["V13-E2E-P1-005"])
    print(json.dumps(report, ensure_ascii=False, indent=2))


def sample_key(row: dict[str, Any]) -> str:
    path = str(row.get("annotation") or row.get("annotation_path") or row.get("image") or row.get("image_path") or "")
    parts = Path(path).parts
    return parts[-2] if len(parts) >= 2 else Path(path).stem


def text_nodes(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [node for node in ((row.get("scene_graph") or {}).get("nodes") or []) if isinstance(node, dict) and node.get("family") == "text"]


def text_nodes_from_gold(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [node for node in (((row.get("expected_json") or {}).get("scene_graph") or {}).get("nodes") or []) if isinstance(node, dict) and node.get("family") == "text"]


def text_value(node: dict[str, Any]) -> str:
    metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
    return str(metadata.get("text") or node.get("text") or "")


if __name__ == "__main__":
    main()
