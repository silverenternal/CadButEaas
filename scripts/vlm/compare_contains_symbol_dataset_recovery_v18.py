#!/usr/bin/env python3
"""Compare two contains_symbol bipartite datasets by canonical gold recovery."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from build_topology_relations_v18 import integrity, write_json
from nms_topology_relations_v18 import load_jsonl

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/vlm"

DEFAULT_BASELINE = REPORT / "contains_symbol_bipartite_dataset.jsonl"
DEFAULT_VARIANT = REPORT / "contains_symbol_bipartite_dataset_symbol_subcandidate_augmented.jsonl"
DEFAULT_OUTPUT = REPORT / "contains_symbol_subcandidate_recovery_delta.json"


def recovered_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        labels = row.get("labels") if isinstance(row.get("labels"), dict) else {}
        key = labels.get("gold_key")
        if key:
            out.setdefault(str(key), []).append(row)
    return out


def target_kind(row: dict[str, Any]) -> str:
    features = row.get("edge_features") if isinstance(row.get("edge_features"), dict) else {}
    if features.get("target_kind_symbol_subcandidate_connected_component"):
        return "symbol_subcandidate_connected_component"
    payload_kind = features.get("target_candidate_kind")
    if payload_kind:
        return str(payload_kind)
    target_id = str(row.get("target_candidate_id") or "")
    if "patch_heatmap_symbol_v" in target_id:
        return "patch_heatmap_symbol_v"
    if "targeted_symbol_family_proposal_v18" in target_id:
        return "targeted_symbol_family_proposal_v18"
    if "symbol_recall_expert_v18" in target_id:
        return "symbol_recall_expert_v18"
    if "_subcc_" in target_id:
        return "symbol_subcandidate_connected_component"
    if "dark_anchor" in target_id:
        return "dark_pixel_anchor"
    if "component" in target_id:
        return "dark_connected_component"
    return "unknown"


def summarize_positive(rows_by_key: dict[str, list[dict[str, Any]]], keys: set[str]) -> dict[str, Any]:
    edge_counts = Counter()
    target_kinds = Counter()
    examples: list[dict[str, Any]] = []
    for key in sorted(keys):
        rows = rows_by_key.get(key, [])
        edge_counts[len(rows)] += 1
        for row in rows:
            target_kinds[target_kind(row)] += 1
        if len(examples) < 50 and rows:
            first = rows[0]
            examples.append(
                {
                    "gold_key": key,
                    "row_id": first.get("row_id"),
                    "source_candidate_id": first.get("source_candidate_id"),
                    "target_candidate_id": first.get("target_candidate_id"),
                    "target_kind": target_kind(first),
                    "positive_edge_count_for_key": len(rows),
                }
            )
    return {
        "gold_key_count": len(keys),
        "positive_edge_count_histogram": dict(edge_counts),
        "positive_target_kind_counts": dict(target_kinds),
        "examples": examples,
    }


def compare(baseline_rows: list[dict[str, Any]], variant_rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_by_key = recovered_rows(baseline_rows)
    variant_by_key = recovered_rows(variant_rows)
    baseline_keys = set(baseline_by_key)
    variant_keys = set(variant_by_key)
    gained = variant_keys - baseline_keys
    lost = baseline_keys - variant_keys
    shared = baseline_keys & variant_keys
    baseline_positive_edges = sum(len(rows) for rows in baseline_by_key.values())
    variant_positive_edges = sum(len(rows) for rows in variant_by_key.values())
    return {
        "baseline_rows": len(baseline_rows),
        "variant_rows": len(variant_rows),
        "row_delta": len(variant_rows) - len(baseline_rows),
        "row_delta_ratio": round((len(variant_rows) - len(baseline_rows)) / max(len(baseline_rows), 1), 6),
        "baseline_recoverable_gold": len(baseline_keys),
        "variant_recoverable_gold": len(variant_keys),
        "recoverable_gold_delta": len(variant_keys) - len(baseline_keys),
        "gained_gold_keys": len(gained),
        "lost_gold_keys": len(lost),
        "shared_gold_keys": len(shared),
        "baseline_positive_edges": baseline_positive_edges,
        "variant_positive_edges": variant_positive_edges,
        "positive_edge_delta": variant_positive_edges - baseline_positive_edges,
        "added_rows_per_gained_gold": round((len(variant_rows) - len(baseline_rows)) / max(len(gained), 1), 6),
        "added_positive_edges_per_gained_gold": round((variant_positive_edges - baseline_positive_edges) / max(len(gained), 1), 6),
        "gained_summary": summarize_positive(variant_by_key, gained),
        "lost_summary": summarize_positive(baseline_by_key, lost),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    parser.add_argument("--variant", default=str(DEFAULT_VARIANT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    baseline_rows = load_jsonl(Path(args.baseline))
    variant_rows = load_jsonl(Path(args.variant))
    report = {
        "task": "IMG-MOE-V18-REBUILD-001.step_g_symbol_subcandidate_recovery_delta",
        "baseline": str(args.baseline),
        "variant": str(args.variant),
        "comparison": compare(baseline_rows, variant_rows),
        "source_integrity": integrity(),
        "gold_loaded_after_inference_for_diagnosis_only": True,
        "gold_used_for_inference": False,
    }
    write_json(Path(args.output), report)
    print(json.dumps(report["comparison"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
