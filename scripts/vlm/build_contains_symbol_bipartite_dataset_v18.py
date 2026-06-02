#!/usr/bin/env python3
"""Build an auditable contains_symbol bipartite edge dataset for v18.

Gold labels are loaded only after detector/topology candidate generation and are
written under labels/audit fields. The exported edge_features are restricted to
detector/topology features available at inference time.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

from build_topology_relations_v18 import integrity, load_gold, relation_label, write_json, write_jsonl
from nms_topology_relations_v18 import cluster_candidates, load_by_id, load_jsonl, row_candidate_map
from train_relation_reranker_v18 import (
    DEFAULT_SPLITTER_MODEL,
    build_symbol_instance_ids,
    feature_vector,
    safe_float,
    topology_pair_key,
)

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/vlm"

DEFAULT_INPUT = REPORT / "topology_relations_v18_symbol_boundary_fixed_candidates.jsonl"
DEFAULT_ADAPTER = REPORT / "detector_adapter_v18_symbol_boundary_fixed_candidates.jsonl"
DEFAULT_OUTPUT = REPORT / "contains_symbol_bipartite_dataset.jsonl"
DEFAULT_AUDIT = REPORT / "contains_symbol_bipartite_dataset_audit.json"


def candidate_cap_rank(cand: dict[str, Any]) -> float:
    payload = cand.get("payload") if isinstance(cand.get("payload"), dict) else {}
    for key in ("cap_rank", "rank", "cluster_member_rank"):
        if key in payload:
            return safe_float(payload.get(key))
    for nested in ("boundary_rerank_v18", "symbol_objectness_v18", "symbol_type_v18", "room_proposal_model_v18"):
        value = payload.get(nested)
        if isinstance(value, dict):
            for key in ("cap_rank", "rank", "cluster_member_rank"):
                if key in value:
                    return safe_float(value.get(key))
    return 0.0


def score_gaps(scores: list[float]) -> dict[str, float]:
    ordered = sorted(scores, reverse=True)
    if not ordered:
        return {"top1_score": 0.0, "top2_score_gap": 0.0, "top3_score_gap": 0.0}
    top1 = ordered[0]
    top2_gap = top1 - ordered[1] if len(ordered) >= 2 else 0.0
    top3_gap = top1 - ordered[2] if len(ordered) >= 3 else 0.0
    return {"top1_score": round(top1, 6), "top2_score_gap": round(top2_gap, 6), "top3_score_gap": round(top3_gap, 6)}


def rank_lookup(edges: list[dict[str, Any]], group_key: str) -> tuple[dict[str, int], dict[str, dict[str, float]]]:
    ranks: dict[str, int] = {}
    gaps: dict[str, dict[str, float]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        grouped[str(edge[group_key])].append(edge)
    for key, group in grouped.items():
        ordered = sorted(group, key=lambda item: safe_float(item["relation"].get("confidence")), reverse=True)
        group_gaps = score_gaps([safe_float(item["relation"].get("confidence")) for item in ordered])
        gaps[key] = group_gaps
        for index, edge in enumerate(ordered, start=1):
            ranks[str(edge["relation"].get("relation_id"))] = index
    return ranks, gaps


def assign_components(edges: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, int]]:
    graph: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        room_node = f"room:{edge['room_instance_cluster_id']}"
        symbol_node = f"symbol:{edge['symbol_instance_cluster_id']}"
        graph[room_node].add(symbol_node)
        graph[symbol_node].add(room_node)

    node_component: dict[str, str] = {}
    component_sizes: dict[str, int] = {}
    for node in sorted(graph):
        if node in node_component:
            continue
        component_id = f"component_{len(component_sizes):05d}"
        queue: deque[str] = deque([node])
        node_component[node] = component_id
        size = 0
        while queue:
            current = queue.popleft()
            size += 1
            for nxt in graph[current]:
                if nxt not in node_component:
                    node_component[nxt] = component_id
                    queue.append(nxt)
        component_sizes[component_id] = size

    relation_component = {}
    for edge in edges:
        rid = str(edge["relation"].get("relation_id"))
        relation_component[rid] = node_component.get(f"room:{edge['room_instance_cluster_id']}", "component_missing")
    return relation_component, component_sizes


def build_page_edges(page: dict[str, Any], adapter: dict[str, Any], gold: dict[str, dict[str, Any]] | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row_id = str(page.get("id"))
    candidates = row_candidate_map(adapter)
    cluster_ids, cluster_summaries, cluster_warnings = cluster_candidates(row_id, candidates)
    symbol_instance_ids, split_audit = build_symbol_instance_ids(row_id, candidates, cluster_ids, DEFAULT_SPLITTER_MODEL)
    contains = [
        rel
        for rel in ((page.get("scene_graph") or {}).get("relations") or [])
        if str(rel.get("relation")) == "contains_symbol"
    ]

    pair_buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    raw_edges: list[dict[str, Any]] = []
    missing_cluster_relations = 0
    for rel in contains:
        src = str(rel.get("source_candidate_id"))
        tgt = str(rel.get("target_candidate_id"))
        if src not in cluster_ids or tgt not in cluster_ids:
            missing_cluster_relations += 1
        pair_key = topology_pair_key(rel, cluster_ids, symbol_instance_ids)
        pair_buckets[pair_key].append(rel)
        raw_edges.append(
            {
                "relation": rel,
                "source_cluster_id": cluster_ids.get(src, f"missing:{src}"),
                "target_cluster_id": cluster_ids.get(tgt, f"missing:{tgt}"),
                "room_instance_cluster_id": cluster_ids.get(src, f"missing:{src}"),
                "symbol_instance_cluster_id": symbol_instance_ids.get(tgt, cluster_ids.get(tgt, f"missing:{tgt}")),
                "relation_pair_key": "|".join(pair_key),
            }
        )

    component_by_relation, component_sizes = assign_components(raw_edges)
    room_ranks, room_gaps = rank_lookup(raw_edges, "room_instance_cluster_id")
    symbol_ranks, symbol_gaps = rank_lookup(raw_edges, "symbol_instance_cluster_id")
    pair_gaps = {"|".join(key): score_gaps([safe_float(rel.get("confidence")) for rel in bucket]) for key, bucket in pair_buckets.items()}
    room_degrees = Counter(str(edge["room_instance_cluster_id"]) for edge in raw_edges)
    symbol_degrees = Counter(str(edge["symbol_instance_cluster_id"]) for edge in raw_edges)
    component_edges = Counter(component_by_relation.get(str(edge["relation"].get("relation_id")), "component_missing") for edge in raw_edges)

    rows: list[dict[str, Any]] = []
    positives_by_pair = Counter()
    for edge in raw_edges:
        rel = edge["relation"]
        rid = str(rel.get("relation_id"))
        src_id = str(rel.get("source_candidate_id"))
        tgt_id = str(rel.get("target_candidate_id"))
        source = candidates.get(src_id) or {}
        target = candidates.get(tgt_id) or {}
        if gold is None:
            left_gold, right_gold, ok = None, None, False
        else:
            left_gold, right_gold, ok = relation_label({**rel, "row_id": row_id}, candidates, gold)
        pair_key = str(edge["relation_pair_key"])
        if ok and left_gold and right_gold:
            positives_by_pair[pair_key] += 1
        base_features = feature_vector(rel, candidates, duplicate_count=len(pair_buckets[tuple(pair_key.split("|"))]))
        component_id = component_by_relation.get(rid, "component_missing")
        graph_features = {
            "room_graph_degree": float(room_degrees[str(edge["room_instance_cluster_id"])]),
            "symbol_graph_degree": float(symbol_degrees[str(edge["symbol_instance_cluster_id"])]),
            "component_node_count": float(component_sizes.get(component_id, 0)),
            "component_edge_count": float(component_edges.get(component_id, 0)),
            "edge_rank_within_room": float(room_ranks.get(rid, 0)),
            "edge_rank_within_symbol": float(symbol_ranks.get(rid, 0)),
            "room_top2_score_gap": room_gaps.get(str(edge["room_instance_cluster_id"]), {}).get("top2_score_gap", 0.0),
            "room_top3_score_gap": room_gaps.get(str(edge["room_instance_cluster_id"]), {}).get("top3_score_gap", 0.0),
            "symbol_top2_score_gap": symbol_gaps.get(str(edge["symbol_instance_cluster_id"]), {}).get("top2_score_gap", 0.0),
            "symbol_top3_score_gap": symbol_gaps.get(str(edge["symbol_instance_cluster_id"]), {}).get("top3_score_gap", 0.0),
            "pair_top2_score_gap": pair_gaps.get(pair_key, {}).get("top2_score_gap", 0.0),
            "pair_top3_score_gap": pair_gaps.get(pair_key, {}).get("top3_score_gap", 0.0),
            "source_cap_rank": candidate_cap_rank(source),
            "target_cap_rank": candidate_cap_rank(target),
            "source_has_provenance": 1.0 if source.get("provenance") else 0.0,
            "target_has_provenance": 1.0 if target.get("provenance") else 0.0,
            "source_has_audit_trace": 1.0 if source.get("audit_trace") else 0.0,
            "target_has_audit_trace": 1.0 if target.get("audit_trace") else 0.0,
        }
        rows.append(
            {
                "row_id": row_id,
                "edge_id": rid,
                "relation_id": rid,
                "relation": "contains_symbol",
                "source_candidate_id": src_id,
                "target_candidate_id": tgt_id,
                "source_cluster_id": edge["source_cluster_id"],
                "target_cluster_id": edge["target_cluster_id"],
                "room_instance_cluster_id": edge["room_instance_cluster_id"],
                "symbol_instance_cluster_id": edge["symbol_instance_cluster_id"],
                "component_id": component_id,
                "relation_pair_key": pair_key,
                "edge_features": {**base_features, **graph_features},
                "labels": {
                    "label_positive": bool(ok and left_gold and right_gold),
                    "gold_room_id": left_gold,
                    "gold_symbol_id": right_gold,
                    "gold_key": f"{row_id}|{left_gold}|{right_gold}" if ok and left_gold and right_gold else None,
                    "gold_loaded_after_inference_for_training_only": True,
                    "gold_used_for_inference": False,
                },
                "provenance": {
                    "source_provenance": source.get("provenance"),
                    "target_provenance": target.get("provenance"),
                    "source_audit_trace": source.get("audit_trace"),
                    "target_audit_trace": target.get("audit_trace"),
                },
                "source_integrity": integrity(),
            }
        )

    audit = {
        "row_id": row_id,
        "contains_symbol_edges": len(rows),
        "positive_edges": sum(1 for row in rows if row["labels"]["label_positive"]),
        "cluster_count": len(cluster_summaries),
        "component_count": len(component_sizes),
        "component_sizes": dict(Counter(component_sizes.values())),
        "missing_cluster_relations": missing_cluster_relations,
        "multiple_positive_edges_same_pair": sum(max(0, count - 1) for count in positives_by_pair.values()),
        "cluster_warnings": cluster_warnings[:25],
        "symbol_instance_splitter": split_audit,
    }
    return rows, audit


def build_dataset(relation_pages: list[dict[str, Any]], adapter_by_id: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gold = load_gold()
    rows: list[dict[str, Any]] = []
    page_audits: list[dict[str, Any]] = []
    missing_adapter_rows = 0
    for page in relation_pages:
        row_id = str(page.get("id"))
        adapter = adapter_by_id.get(row_id)
        if not adapter:
            missing_adapter_rows += 1
            continue
        page_rows, page_audit = build_page_edges(page, adapter, gold)
        rows.extend(page_rows)
        page_audits.append(page_audit)

    positives = sum(1 for row in rows if row["labels"]["label_positive"])
    components = Counter(row["component_id"] for row in rows)
    positive_components = Counter(row["component_id"] for row in rows if row["labels"]["label_positive"])
    pair_counts = Counter(row["relation_pair_key"] for row in rows)
    positive_pair_counts = Counter(row["relation_pair_key"] for row in rows if row["labels"]["label_positive"])
    audit = {
        "task": "IMG-MOE-V18-REBUILD-001.step_a_build_contains_symbol_bipartite_dataset",
        "rows": len({row["row_id"] for row in rows}),
        "edge_rows": len(rows),
        "positive_edges": positives,
        "negative_edges": len(rows) - positives,
        "positive_rate": round(positives / max(len(rows), 1), 6),
        "component_count": len(components),
        "component_edge_count_histogram": dict(Counter(components.values())),
        "positive_component_edge_count_histogram": dict(Counter(positive_components.values())),
        "relation_pair_count": len(pair_counts),
        "multi_edge_pair_count": sum(1 for count in pair_counts.values() if count > 1),
        "multi_positive_pair_count": sum(1 for count in positive_pair_counts.values() if count > 1),
        "multiple_positive_edges_same_pair": sum(max(0, count - 1) for count in positive_pair_counts.values()),
        "missing_adapter_rows": missing_adapter_rows,
        "page_audits_sample": page_audits[:100],
        "leakage_audit": {
            "gold_fields_in_edge_features": False,
            "gold_loaded_after_inference_for_training_only": True,
            "gold_used_for_inference": False,
            "edge_features_inference_available_only": True,
        },
        "source_integrity": integrity(),
    }
    return rows, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--locked", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    limit = 5 if args.smoke else args.limit
    relation_pages = load_jsonl(Path(args.input), limit=limit)
    adapter_by_id = load_by_id(Path(args.adapter), limit=limit)
    rows, audit = build_dataset(relation_pages, adapter_by_id)
    audit["locked"] = bool(args.locked)
    audit["smoke"] = bool(args.smoke)
    audit["input"] = str(args.input)
    audit["adapter"] = str(args.adapter)

    write_jsonl(Path(args.output), rows)
    write_json(Path(args.audit), audit)
    print(json.dumps({"edge_rows": len(rows), "positive_edges": audit["positive_edges"], "positive_rate": audit["positive_rate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
