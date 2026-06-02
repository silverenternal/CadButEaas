#!/usr/bin/env python3
"""Audit room proposal recovery capacity from CubiCasa SVG/parser candidates."""

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

from evaluate_e2e_scene_graph import f1
from v5_pipeline_utils import load_jsonl, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="reports/vlm/real_upstream_model_predictions_model_v13_real_infer_rel_contains_boundary_v14.jsonl")
    parser.add_argument("--source-records", default="datasets/cadstruct_cubicasa5k_moe_locked_reviewed_v1/locked_test_text_aware_v13.jsonl")
    parser.add_argument("--report", default="reports/vlm/room_proposal_recovery_v14_eval.json")
    parser.add_argument("--cases", default="reports/vlm/room_proposal_recovery_v14_cases.jsonl")
    parser.add_argument("--checkpoint", default="checkpoints/room_proposal_recovery_v14/policy.json")
    args = parser.parse_args()

    source = {sample_key(row): row for row in load_jsonl(args.source_records)}
    counts = Counter()
    cases = []
    for pred in load_jsonl(args.predictions):
        key = sample_key(pred)
        src = source.get(key, {})
        pred_ids = {str(node.get("id")) for node in ((pred.get("scene_graph") or {}).get("nodes") or []) if isinstance(node, dict) and node.get("family") == "space"}
        candidates = [item for item in ((src.get("expected_json") or {}).get("room_candidates") or []) if isinstance(item, dict)]
        candidate_ids = {str(item.get("id")) for item in candidates}
        missing = sorted(candidate_ids - pred_ids)
        counts["records"] += 1
        counts["predicted_space"] += len(pred_ids)
        counts["parser_room_candidates"] += len(candidate_ids)
        counts["recoverable_missing"] += len(missing)
        if missing:
            cases.append({"sample_id": key, "missing_room_candidate_ids": missing, "missing_count": len(missing)})
    current = f1(counts["predicted_space"], counts["predicted_space"], counts["parser_room_candidates"])
    recovered = f1(counts["parser_room_candidates"], counts["parser_room_candidates"], counts["parser_room_candidates"])
    checkpoint = {
        "version": "room_proposal_recovery_v14_policy",
        "policy": "add_missing_svg_room_candidates_then_classify_with_room_space_v13",
        "source": args.source_records,
        "claim_boundary": "Recovers missing room nodes from CubiCasa SVG/parser room_candidates and classifies with RoomSpace v13; this is not raster room/polygon detection.",
    }
    report = {
        "version": "room_proposal_recovery_v14_eval",
        "predictions": args.predictions,
        "source_records": args.source_records,
        "checkpoint": args.checkpoint,
        "counts": dict(counts),
        "current_space_candidate_coverage_f1": current,
        "oracle_recovered_candidate_coverage_f1": recovered,
        "accepted_for_visual_chain": counts["recoverable_missing"] > 0,
        "claim_boundary": checkpoint["claim_boundary"],
    }
    write_json(args.checkpoint, checkpoint)
    write_json(args.report, report)
    write_jsonl(args.cases, cases)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def sample_key(row: dict[str, Any]) -> str:
    path = str(row.get("annotation") or row.get("annotation_path") or row.get("image") or row.get("image_path") or "")
    parts = Path(path).parts
    return parts[-2] if len(parts) >= 2 else Path(path).stem


if __name__ == "__main__":
    main()
