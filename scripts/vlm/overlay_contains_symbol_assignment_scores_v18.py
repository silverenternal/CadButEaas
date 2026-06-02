#!/usr/bin/env python3
"""Overlay contains_symbol assignment scores onto an existing scored cache.

The output preserves the input cache row universe exactly: no rows are added,
removed, or re-keyed. Only assignment-specific fields are updated for matching
contains_symbol relation_ids so scored-cache replay can audit a new selector.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/vlm"
DEFAULT_CACHE = REPORT / "relation_graph_scored_rows_cache_v18.jsonl"
DEFAULT_ASSIGNMENT = REPORT / "contains_symbol_assignment_compressed_v18_dataset.jsonl"
DEFAULT_OUTPUT = REPORT / "relation_graph_scored_rows_cache_contains_assignment_compressed_v18.jsonl"
DEFAULT_AUDIT = REPORT / "relation_graph_scored_rows_cache_contains_assignment_compressed_v18_audit.json"


def load_assignment(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            relation_id = str(row.get("relation_id") or "")
            if relation_id:
                out[relation_id] = row
    return out


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def overlay(cache_path: Path, assignment_path: Path, output_path: Path) -> dict[str, Any]:
    assignment = load_assignment(assignment_path)
    counts = Counter()
    missing_examples: list[str] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            row = json.loads(line)
            counts["input_rows"] += 1
            relation = str(row.get("relation") or "")
            if relation == "contains_symbol":
                counts["input_contains_symbol_rows"] += 1
                relation_id = str(row.get("relation_id") or "")
                scored = assignment.get(relation_id)
                if scored:
                    counts["contains_symbol_rows_overlaid"] += 1
                    row["assignment_score"] = scored.get("assignment_score")
                    row["assignment_symbol_rank"] = scored.get("assignment_symbol_rank")
                    row["assignment_symbol_percentile"] = scored.get("assignment_symbol_percentile")
                    row["assignment_score_source"] = str(assignment_path)
                else:
                    counts["contains_symbol_rows_missing_assignment_score"] += 1
                    if len(missing_examples) < 50:
                        missing_examples.append(relation_id)
            dst.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            counts["output_rows"] += 1
    return {
        "task": "IMG-MOE-V18-REBUILD-006.step_existing_cache_assignment_score_overlay",
        "cache": str(cache_path),
        "assignment_scores": str(assignment_path),
        "output": str(output_path),
        "counts": dict(counts),
        "missing_assignment_examples": missing_examples,
        "row_universe_preserved": counts["input_rows"] == counts["output_rows"],
        "new_runtime_candidates_created": False,
        "new_relation_edges_created": False,
        "source_integrity": {
            "source_mode": "scored_cache_assignment_score_overlay",
            "model_input": "existing_scored_cache_rows_only",
            "svg_candidate_ids_used": False,
            "annotation_geometry_used_at_inference": False,
            "gold_used_for_inference": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--assignment", default=str(DEFAULT_ASSIGNMENT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--audit-output", default=str(DEFAULT_AUDIT))
    args = parser.parse_args()

    audit = overlay(Path(args.cache), Path(args.assignment), Path(args.output))
    write_json(Path(args.audit_output), audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
