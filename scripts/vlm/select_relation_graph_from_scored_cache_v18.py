#!/usr/bin/env python3
"""Replay relation-graph selection from an all-candidate scored-row cache."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from relation_graph_reconstruction_v18 import (
    DEFAULT_ADAPTER,
    DEFAULT_INPUT,
    DEFAULT_MODEL,
    evaluate_selection,
    integrity,
    load_jsonl,
    select_rows,
    write_json,
    write_jsonl,
)

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/vlm"
DEFAULT_CACHE = REPORT / "relation_graph_scored_rows_cache_v18.jsonl"
DEFAULT_OUTPUT = REPORT / "relation_graph_from_scored_cache_v18_candidates.jsonl"
DEFAULT_FEATURES = REPORT / "relation_graph_from_scored_cache_v18_features.jsonl"
DEFAULT_EVAL = REPORT / "relation_graph_from_scored_cache_v18_eval.json"
DEFAULT_AUDIT = REPORT / "relation_graph_from_scored_cache_v18_audit.json"


def load_model(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def group_by_row(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("row_id"))].append(row)
    return grouped


def replay_pages(
    pages: list[dict[str, Any]],
    adapter_by_id: dict[str, dict[str, Any]],
    scored_by_row: dict[str, list[dict[str, Any]]],
    policy: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    page_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    page_stats: list[dict[str, Any]] = []
    missing_cache_rows = 0
    missing_adapter_rows = 0

    for page in pages:
        row_id = str(page.get("id"))
        adapter = adapter_by_id.get(row_id)
        if not adapter:
            missing_adapter_rows += 1
            continue
        scored_rows = scored_by_row.get(row_id) or []
        if not scored_rows:
            missing_cache_rows += 1
            continue
        kept = select_rows(scored_rows, policy)
        kept_by_id = {str(row.get("relation_id")): row for row in kept}
        relations = list(((page.get("scene_graph") or {}).get("relations") or []))
        relations_out: list[dict[str, Any]] = []
        for rel in relations:
            rid = str(rel.get("relation_id"))
            row = kept_by_id.get(rid)
            if not row:
                continue
            out = dict(rel)
            evidence = dict(out.get("evidence") if isinstance(out.get("evidence"), dict) else {})
            evidence.update(
                {
                    "relation_graph_policy": "relation_graph_from_scored_cache_v18",
                    "relation_graph_score": row.get("relation_score"),
                    "relation_graph_component_id": row.get("component_id"),
                    "relation_graph_duplicate_count": row.get("duplicate_relation_count"),
                    "source_cluster_id": row.get("source_cluster_id"),
                    "target_cluster_id": row.get("target_cluster_id"),
                    "scored_cache_replay": True,
                }
            )
            out["evidence"] = evidence
            out["relation_graph_score"] = row.get("relation_score")
            out["relation_graph_component_id"] = row.get("component_id")
            out["relation_graph_duplicate_count"] = row.get("duplicate_relation_count")
            out["source_cluster_id"] = row.get("source_cluster_id")
            out["target_cluster_id"] = row.get("target_cluster_id")
            out["symbol_instance_cluster_id"] = row.get("symbol_instance_cluster_id")
            out["source_integrity"] = integrity()
            relations_out.append(out)
            feature_rows.append(
                {
                    "row_id": row_id,
                    "relation_id": rid,
                    "relation": rel.get("relation"),
                    "source_candidate_id": rel.get("source_candidate_id"),
                    "target_candidate_id": rel.get("target_candidate_id"),
                    "source_cluster_id": row.get("source_cluster_id"),
                    "target_cluster_id": row.get("target_cluster_id"),
                    "symbol_instance_cluster_id": row.get("symbol_instance_cluster_id"),
                    "component_id": row.get("component_id"),
                    "relation_score": row.get("relation_score"),
                    "assignment_score": row.get("assignment_score"),
                    "support_criticality_score": row.get("support_criticality_score"),
                    "selection_reason": row.get("assignment_selection_reason") or row.get("selection_reason"),
                    "edge_features": row.get("edge_features") if isinstance(row.get("edge_features"), dict) else {},
                    "listwise_features": row.get("listwise_features") if isinstance(row.get("listwise_features"), dict) else {},
                    "source_integrity": integrity(),
                }
            )
        page_stats.append(
            {
                "row_id": row_id,
                "cache_rows": len(scored_rows),
                "before_relations": len(relations),
                "after_relations": len(relations_out),
                "selected_reduction": round(1.0 - len(relations_out) / max(len(relations), 1), 6),
            }
        )
        page_rows.append(
            {
                "id": row_id,
                "image": page.get("image") or adapter.get("image"),
                "image_size": page.get("image_size") or adapter.get("image_size") or [512, 512],
                "source_integrity": integrity(),
                "route_trace": {
                    **integrity(),
                    "stage": "relation_graph_from_scored_cache_v18",
                    "gold_loaded_after_inference_for_evaluation_only": False,
                },
                "scene_graph": {
                    "nodes": [],
                    "relations": relations_out,
                    "candidate_counts": ((adapter.get("scene_graph") or {}).get("candidate_counts") or {}),
                    "relation_counts": dict(Counter(str(rel.get("relation")) for rel in relations_out)),
                },
            }
        )

    audit = {
        "task": "IMG-MOE-V18-REBUILD-005.scored_cache_replay",
        "rows": len(page_rows),
        "features": len(feature_rows),
        "missing_adapter_rows": missing_adapter_rows,
        "missing_cache_rows": missing_cache_rows,
        "page_stats": page_stats[:100],
        "source_integrity": integrity(),
        "gold_loaded_after_inference_for_evaluation_only": False,
        "gold_used_for_inference": False,
    }
    return page_rows, feature_rows, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--features-output", default=str(DEFAULT_FEATURES))
    parser.add_argument("--eval-output", default=str(DEFAULT_EVAL))
    parser.add_argument("--audit-output", default=str(DEFAULT_AUDIT))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    limit = 5 if args.smoke else args.limit
    pages = load_jsonl(Path(args.input), limit=limit)
    adapter_rows = load_jsonl(Path(args.adapter), limit=limit)
    scored_rows = load_jsonl(Path(args.cache))
    if limit is not None:
        wanted = {str(row.get("id")) for row in pages}
        scored_rows = [row for row in scored_rows if str(row.get("row_id")) in wanted]
    model = load_model(Path(args.model))
    policy = model.get("policy") or {}
    page_rows, feature_rows, audit = replay_pages(
        pages,
        {str(row.get("id")): row for row in adapter_rows},
        group_by_row(scored_rows),
        policy,
    )
    write_jsonl(Path(args.output), page_rows)
    write_jsonl(Path(args.features_output), feature_rows)
    write_json(Path(args.audit_output), audit)
    eval_report = evaluate_selection(page_rows, adapter_rows)
    eval_report["policy"] = policy
    eval_report["rows"] = len(page_rows)
    eval_report["features"] = len(feature_rows)
    eval_report["scored_cache"] = args.cache
    write_json(Path(args.eval_output), eval_report)
    print(
        json.dumps(
            {
                "rows": len(page_rows),
                "features": len(feature_rows),
                "output": args.output,
                "eval_output": args.eval_output,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
