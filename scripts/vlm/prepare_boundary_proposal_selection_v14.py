#!/usr/bin/env python3
"""Prepare boundary proposal/semantic selection audit data for v14."""

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

from v5_pipeline_utils import load_jsonl, write_json, write_jsonl


BOUNDARY_LABELS = {"door", "hard_wall", "opening", "partition_wall", "window"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="reports/vlm/real_upstream_model_predictions_model_v13_real_infer_rel_contains.jsonl")
    parser.add_argument("--gold", default="datasets/cadstruct_cubicasa5k_moe_locked_reviewed_v1/locked_test.jsonl")
    parser.add_argument("--output", default="datasets/boundary_proposal_selection_v14/locked_visual_candidates.jsonl")
    parser.add_argument("--audit", default="reports/vlm/boundary_proposal_selection_v14_prepare_audit.json")
    args = parser.parse_args()

    gold_rows = {key(row): row for row in load_jsonl(args.gold)}
    rows = []
    counts = Counter()
    for pred in load_jsonl(args.predictions):
        gold = gold_rows.get(key(pred), {})
        gold_nodes = {
            str(node.get("id")): node
            for node in (((gold.get("expected_json") or {}).get("scene_graph") or {}).get("nodes") or [])
            if isinstance(node, dict) and node.get("family") == "boundary"
        }
        for node in ((pred.get("scene_graph") or {}).get("nodes") or []):
            if not isinstance(node, dict) or node.get("family") != "boundary":
                continue
            metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
            gold_node = gold_nodes.get(str(node.get("id") or ""))
            raw_label = normalize_label(metadata.get("raw_label") or metadata.get("base_raw_label"))
            model_label = normalize_label(node.get("semantic_type") or metadata.get("model_label"))
            gold_label = normalize_label((gold_node or {}).get("semantic_type"))
            bbox = ((node.get("geometry") or {}).get("bbox") or [])
            item = {
                "sample_id": key(pred)[0],
                "node_id": node.get("id"),
                "bbox": bbox,
                "raw_label": raw_label,
                "model_label": model_label,
                "gold_label": gold_label,
                "raw_correct": raw_label == gold_label,
                "model_correct": model_label == gold_label,
                "has_gold": bool(gold_label),
                "keep": bool(gold_label),
                "metadata": {
                    "model_confidence": metadata.get("model_confidence"),
                    "was_clipped_to_canvas": metadata.get("was_clipped_to_canvas"),
                    "source_canvas_bbox": metadata.get("source_canvas_bbox"),
                },
            }
            rows.append(item)
            counts["items"] += 1
            counts[f"raw_correct:{raw_label == gold_label}"] += 1
            counts[f"model_correct:{model_label == gold_label}"] += 1
            if raw_label != model_label:
                counts["raw_model_disagree"] += 1
    write_jsonl(args.output, rows)
    write_json(
        args.audit,
        {
            "version": "boundary_proposal_selection_v14_prepare_audit",
            "predictions": args.predictions,
            "gold": args.gold,
            "output": args.output,
            "counts": dict(counts),
            "claim_boundary": "This dataset audits candidate-level boundary keep/semantic selection on reviewed visual samples; it is not a raster wall detector dataset.",
        },
    )
    print(json.dumps({"output": args.output, "counts": dict(counts)}, ensure_ascii=False, indent=2))


def key(row: dict[str, Any]) -> tuple[str, str]:
    annotation = str(row.get("annotation") or row.get("annotation_path") or "")
    parts = Path(annotation).parts
    return (parts[-2] if len(parts) >= 2 else Path(annotation).stem, "/".join(parts[-3:]) if annotation else "")


def normalize_label(value: Any) -> str:
    label = str(value or "")
    return label if label in BOUNDARY_LABELS else ""


if __name__ == "__main__":
    main()
