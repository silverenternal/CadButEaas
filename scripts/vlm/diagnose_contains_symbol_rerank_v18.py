#!/usr/bin/env python3
"""Diagnose where contains_symbol recall is lost after relation compression."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_topology_relations_v18 import integrity, load_gold, relation_label
from nms_topology_relations_v18 import cluster_candidates, load_by_id, load_jsonl, relation_pair_key, row_candidate_map

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/vlm"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(REPORT / "topology_relations_v18_symbol_boundary_fixed_candidates.jsonl"))
    parser.add_argument("--kept", default=str(REPORT / "topology_relations_v18_relation_reranked_candidates.jsonl"))
    parser.add_argument("--adapter", default=str(REPORT / "detector_adapter_v18_symbol_boundary_fixed_candidates.jsonl"))
    parser.add_argument("--output", default=str(REPORT / "contains_symbol_rerank_loss_diagnostic.json"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    pages = load_jsonl(Path(args.input), limit=args.limit or None)
    kept_by_id = load_by_id(Path(args.kept), limit=args.limit or None)
    adapter_by_id = load_by_id(Path(args.adapter), limit=args.limit or None)
    gold = load_gold()

    summary = Counter()
    lost_by_reason = Counter()
    pair_positive_counts = Counter()
    cluster_positive_counts = Counter()
    pair_loss_counts = Counter()
    examples: list[dict[str, Any]] = []

    for page in pages:
        row_id = str(page.get("id"))
        adapter = adapter_by_id.get(row_id)
        kept_page = kept_by_id.get(row_id)
        if not adapter or not kept_page:
            continue
        candidates = row_candidate_map(adapter)
        cluster_ids, _clusters, _warnings = cluster_candidates(row_id, candidates)
        kept_positive_keys: set[tuple[str, str, str]] = set()
        kept_cluster_keys: set[tuple[str, str, str]] = set()
        for rel in ((kept_page.get("scene_graph") or {}).get("relations") or []):
            if rel.get("relation") != "contains_symbol":
                continue
            left, right, ok = relation_label({**rel, "row_id": row_id}, candidates, gold)
            if ok and left and right:
                kept_positive_keys.add((row_id, left, right))
                kept_cluster_keys.add(relation_pair_key(rel, cluster_ids))

        all_positive_by_cluster: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        all_positive_records: list[dict[str, Any]] = []
        for rel in ((page.get("scene_graph") or {}).get("relations") or []):
            if rel.get("relation") != "contains_symbol":
                continue
            left, right, ok = relation_label({**rel, "row_id": row_id}, candidates, gold)
            if not (ok and left and right):
                continue
            summary["pre_positive_relations"] += 1
            gold_key = (row_id, left, right)
            cluster_key = relation_pair_key(rel, cluster_ids)
            all_positive_by_cluster[cluster_key].append(rel)
            all_positive_records.append(
                {
                    "rel": rel,
                    "gold_key": gold_key,
                    "cluster_key": cluster_key,
                    "kept_positive": gold_key in kept_positive_keys,
                    "kept_cluster_pair": cluster_key in kept_cluster_keys,
                }
            )
            if gold_key in kept_positive_keys:
                summary["positive_gold_kept"] += 1
            else:
                summary["positive_gold_lost"] += 1

        for record in all_positive_records:
            rel = record["rel"]
            gold_key = record["gold_key"]
            cluster_key = record["cluster_key"]
            if gold_key in kept_positive_keys:
                continue
            if not record["kept_cluster_pair"]:
                lost_by_reason["cluster_pair_dropped_by_source_cap"] += 1
                reason = "source_cap_drop"
            elif len(all_positive_by_cluster[cluster_key]) > 1:
                lost_by_reason["joint_pair_bottleneck"] += 1
                reason = "joint_pair_bottleneck"
            else:
                lost_by_reason["wrong_representative_within_cluster_pair"] += 1
                reason = "wrong_representative_within_cluster_pair"
            if len(examples) < 50:
                examples.append(
                    {
                        "row_id": row_id,
                        "gold_key": list(gold_key),
                        "cluster_key": list(cluster_key),
                        "source_candidate_id": rel.get("source_candidate_id"),
                        "target_candidate_id": rel.get("target_candidate_id"),
                        "confidence": rel.get("confidence"),
                        "reason": reason,
                    }
                )

        for cluster_key, rels in all_positive_by_cluster.items():
            pair_positive_counts[str(len(rels))] += 1
            if len(rels) > 1:
                pair_loss_counts["multi_positive_cluster_pair"] += 1
            if cluster_key in kept_cluster_keys and len(rels) > 1:
                kept_keys_for_pair = sum(
                    1
                    for record in all_positive_records
                    if record["cluster_key"] == cluster_key and record["gold_key"] in kept_positive_keys
                )
                pair_loss_counts["multi_positive_kept_count_total"] += kept_keys_for_pair
            source_clusters = {cluster_ids.get(str(rel.get("source_candidate_id"))) for rel in rels}
            target_clusters = {cluster_ids.get(str(rel.get("target_candidate_id"))) for rel in rels}
            if len(source_clusters) > 1:
                cluster_positive_counts["multiple_source_clusters"] += 1
            if len(target_clusters) > 1:
                cluster_positive_counts["multiple_target_clusters"] += 1
            if len(rels) > 1:
                cluster_positive_counts["multiple_positive_relations_same_cluster_pair"] += 1

    report = {
        "task": "IMG-MOE-V18-NEXT-004-CONTAINS-SYMBOL-DIAGNOSTIC",
        "summary": dict(summary),
        "lost_by_reason": dict(lost_by_reason),
        "positive_relations_per_cluster_pair": dict(pair_positive_counts),
        "cluster_positive_counts": dict(cluster_positive_counts),
        "pair_loss_counts": dict(pair_loss_counts),
        "examples": examples,
        "source_integrity": integrity(),
        "gold_loaded_after_inference_for_diagnosis_only": True,
        "gold_used_for_inference": False,
    }
    write_json(Path(args.output), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
