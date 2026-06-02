#!/usr/bin/env python3
"""Build text-aware visual gold by adding reviewed text_candidates as scene nodes."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v5_pipeline_utils import load_jsonl, update_todo_remove, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="datasets/cadstruct_cubicasa5k_moe_locked_reviewed_v1/locked_test.jsonl")
    parser.add_argument("--predictions", default="reports/vlm/real_upstream_model_predictions_model_v13_real_infer_rel_contains_boundary_v14.jsonl")
    parser.add_argument("--output", default="datasets/cadstruct_cubicasa5k_moe_locked_reviewed_v1/locked_test_text_aware_v13.jsonl")
    parser.add_argument("--audit", default="reports/vlm/text_aware_visual_gold_v13_audit.json")
    args = parser.parse_args()

    wanted = {sample_key(row) for row in load_jsonl(args.predictions)}
    out = []
    counts = Counter()
    for row in load_jsonl(args.gold):
        item = json.loads(json.dumps(row, ensure_ascii=False))
        if wanted and sample_key(item) not in wanted:
            out.append(item)
            continue
        expected = item.setdefault("expected_json", {})
        scene = expected.setdefault("scene_graph", {})
        nodes = scene.setdefault("nodes", [])
        edges = scene.setdefault("edges", [])
        existing = {str(node.get("id")) for node in nodes if isinstance(node, dict)}
        for text in expected.get("text_candidates") or []:
            if not isinstance(text, dict):
                continue
            node_id = str(text.get("id") or "")
            if not node_id or node_id in existing:
                continue
            nodes.append(
                {
                    "id": node_id,
                    "semantic_type": str(text.get("text_type") or "note_text"),
                    "expert": "text_dimension",
                    "family": "text",
                    "confidence": float(text.get("confidence") or 1.0),
                    "geometry": {"bbox": text.get("bbox")},
                    "metadata": {"text": text.get("text") or "", "font_size": text.get("font_size"), "gold_adapter": "text_aware_visual_gold_v13"},
                }
            )
            counts["text_nodes_added"] += 1
            counts[f"text_type:{text.get('text_type') or 'note_text'}"] += 1
            if any(ch.isdigit() for ch in str(text.get("text") or "")):
                counts["numeric_text_nodes_added"] += 1
            existing.add(node_id)
        scene["edges"] = edges
        out.append(item)
    write_jsonl(args.output, out)
    write_json(
        args.audit,
        {
            "version": "text_aware_visual_gold_v13_audit",
            "gold": args.gold,
            "predictions_scope": args.predictions,
            "output": args.output,
            "counts": dict(counts),
            "claim_boundary": "Adds CubiCasa SVG text_candidates to reviewed visual gold for text localization/type evaluation; it does not add OCR content grading beyond numeric presence audits.",
        },
    )
    print(json.dumps({"output": args.output, "counts": dict(counts)}, ensure_ascii=False, indent=2))


def sample_key(row: dict[str, Any]) -> str:
    path = str(row.get("annotation") or row.get("annotation_path") or row.get("image") or row.get("image_path") or "")
    parts = Path(path).parts
    return parts[-2] if len(parts) >= 2 else Path(path).stem


if __name__ == "__main__":
    main()
