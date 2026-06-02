#!/usr/bin/env python3
"""Build an all-candidate scored-row cache for v18 relation-graph policy work."""

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
    DEFAULT_SPLITTER_MODEL,
    connected_components,
    graph_node_key,
    integrity,
    load_jsonl,
    relation_features,
    row_candidate_map,
    score_rows,
    write_json,
    write_jsonl,
)
from nms_topology_relations_v18 import cluster_candidates
from train_relation_reranker_v18 import build_symbol_instance_ids, safe_float, topology_pair_key

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/vlm"
DEFAULT_OUTPUT = REPORT / "relation_graph_scored_rows_cache_v18.jsonl"
DEFAULT_AUDIT = REPORT / "relation_graph_scored_rows_cache_v18_audit.json"


def load_model(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def scored_rows_for_page(
    page: dict[str, Any],
    adapter: dict[str, Any],
    model: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row_id = str(page.get("id"))
    candidates = row_candidate_map(adapter)
    cluster_ids, cluster_summaries, cluster_warnings = cluster_candidates(row_id, candidates)
    symbol_instance_ids, split_audit = build_symbol_instance_ids(row_id, candidates, cluster_ids, DEFAULT_SPLITTER_MODEL)
    relations = list(((page.get("scene_graph") or {}).get("relations") or []))

    pair_buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for rel in relations:
        pair_buckets[topology_pair_key(rel, cluster_ids, symbol_instance_ids)].append(rel)

    component_input = [
        {
            **rel,
            "source_cluster_id": cluster_ids.get(str(rel.get("source_candidate_id")), f"missing:{rel.get('source_candidate_id')}"),
            "target_cluster_id": cluster_ids.get(str(rel.get("target_candidate_id")), f"missing:{rel.get('target_candidate_id')}"),
        }
        for rel in relations
    ]
    node_component, component_sizes, degree, component_edges = connected_components(component_input)

    rows: list[dict[str, Any]] = []
    for rel in relations:
        source_cluster_id = cluster_ids.get(str(rel.get("source_candidate_id")), f"missing:{rel.get('source_candidate_id')}")
        target_cluster_id = cluster_ids.get(str(rel.get("target_candidate_id")), f"missing:{rel.get('target_candidate_id')}")
        enriched_rel = {
            **rel,
            "source_cluster_id": source_cluster_id,
            "target_cluster_id": target_cluster_id,
        }
        pair_key = topology_pair_key(rel, cluster_ids, symbol_instance_ids)
        duplicate_count = max(1, len(pair_buckets[pair_key]))
        features = relation_features(
            enriched_rel,
            candidates,
            duplicate_count,
            node_component,
            component_sizes,
            degree,
            component_edges,
        )
        rows.append(
            {
                "row_id": row_id,
                "relation_id": rel.get("relation_id"),
                "relation": rel.get("relation"),
                "source_candidate_id": rel.get("source_candidate_id"),
                "target_candidate_id": rel.get("target_candidate_id"),
                "source_cluster_id": source_cluster_id,
                "target_cluster_id": target_cluster_id,
                "symbol_instance_cluster_id": symbol_instance_ids.get(str(rel.get("target_candidate_id")), target_cluster_id),
                "component_id": node_component.get(graph_node_key(enriched_rel, "source"))
                or node_component.get(graph_node_key(enriched_rel, "target"))
                or "relation_component_missing",
                "duplicate_relation_count": duplicate_count,
                "edge_features": features,
                "confidence": safe_float(rel.get("confidence")),
                "labels": {},
                "source_integrity": integrity(),
            }
        )

    scored = score_rows(rows, model)
    page_audit = {
        "row_id": row_id,
        "relations": len(relations),
        "scored_rows": len(scored),
        "cluster_count": len(cluster_summaries),
        "component_count": len(component_sizes),
        "split_audit": split_audit,
        "warning_counts": dict(Counter(item.get("warning") for item in cluster_warnings if item.get("warning"))),
    }
    return scored, page_audit


def build_cache(
    pages: list[dict[str, Any]],
    adapter_by_id: dict[str, dict[str, Any]],
    model: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    warning_counts: Counter[str] = Counter()
    page_stats: list[dict[str, Any]] = []
    missing_adapter_rows = 0

    for page in pages:
        row_id = str(page.get("id"))
        adapter = adapter_by_id.get(row_id)
        if not adapter:
            missing_adapter_rows += 1
            continue
        scored, page_audit = scored_rows_for_page(page, adapter, model)
        all_rows.extend(scored)
        page_stats.append(page_audit)
        warning_counts.update(page_audit.get("warning_counts") or {})

    relation_counts = Counter(str(row.get("relation")) for row in all_rows)
    scored_field_counts = {
        "relation_score": sum(1 for row in all_rows if row.get("relation_score") is not None),
        "listwise_features": sum(1 for row in all_rows if isinstance(row.get("listwise_features"), dict)),
        "support_criticality_score": sum(1 for row in all_rows if row.get("support_criticality_score") is not None),
        "assignment_score": sum(1 for row in all_rows if row.get("assignment_score") is not None),
    }
    audit = {
        "task": "IMG-MOE-V18-REBUILD-005.scored_rows_cache",
        "rows": len(page_stats),
        "scored_rows": len(all_rows),
        "relation_counts": dict(relation_counts),
        "scored_field_counts": scored_field_counts,
        "missing_adapter_rows": missing_adapter_rows,
        "warning_counts": {str(k): int(v) for k, v in warning_counts.items() if k},
        "page_stats": page_stats[:100],
        "source_integrity": integrity(),
        "gold_loaded_after_inference_for_evaluation_only": False,
        "gold_used_for_inference": False,
    }
    return all_rows, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--audit-output", default=str(DEFAULT_AUDIT))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    limit = 5 if args.smoke else args.limit
    pages = load_jsonl(Path(args.input), limit=limit)
    adapter_rows = load_jsonl(Path(args.adapter), limit=limit)
    adapter_by_id = {str(row.get("id")): row for row in adapter_rows}
    model = load_model(Path(args.model))
    scored_rows, audit = build_cache(pages, adapter_by_id, model)
    write_jsonl(Path(args.output), scored_rows)
    write_json(Path(args.audit_output), audit)
    print(
        json.dumps(
            {
                "rows": audit["rows"],
                "scored_rows": audit["scored_rows"],
                "relation_counts": audit["relation_counts"],
                "output": args.output,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
