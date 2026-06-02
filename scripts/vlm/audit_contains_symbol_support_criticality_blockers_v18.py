#!/usr/bin/env python3
"""Audit blockers for learned contains_symbol support compression.

This is an offline diagnostic. It reads the support-criticality dataset emitted
by train_contains_symbol_support_criticality_v18.py and explains why aggressive
room-symbol support policies fail the locked no-regression requirement.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from relation_graph_reconstruction_v18 import load_jsonl, safe_float, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/vlm"
CHECKPOINT = ROOT / "checkpoints/relation_graph_policy_v18"

DEFAULT_DATASET = REPORT / "contains_symbol_support_criticality_v18_dataset.jsonl"
DEFAULT_AUDIT = REPORT / "contains_symbol_support_criticality_blockers_v18_audit.json"
DEFAULT_EXAMPLES = REPORT / "contains_symbol_support_criticality_blockers_v18_examples.jsonl"
DEFAULT_MODEL = CHECKPOINT / "contains_symbol_support_criticality_v18.json"


def relation_id(row: dict[str, Any]) -> str:
    return str(row.get("relation_id") or f"{row.get('row_id')}|{row.get('source_candidate_id')}|{row.get('target_candidate_id')}")


def split_name(row: dict[str, Any]) -> str:
    return str(row.get("split") or "unknown")


def labels(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("labels") if isinstance(row.get("labels"), dict) else {}


def features(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("features") if isinstance(row.get("features"), dict) else {}


def gold_key(row: dict[str, Any]) -> str:
    return str(labels(row).get("gold_key") or "")


def relation_score(row: dict[str, Any]) -> float:
    return safe_float(features(row).get("relation_score"))


def support_score(row: dict[str, Any]) -> float:
    return safe_float(row.get("support_criticality_score"))


def rank_bucket(value: int) -> str:
    if value <= 0:
        return "missing"
    if value <= 2:
        return "01-02"
    if value <= 4:
        return "03-04"
    if value <= 8:
        return "05-08"
    if value <= 16:
        return "09-16"
    if value <= 32:
        return "17-32"
    return "33+"


def size_bucket(value: int) -> str:
    if value <= 1:
        return "001"
    if value <= 4:
        return "002-004"
    if value <= 8:
        return "005-008"
    if value <= 16:
        return "009-016"
    if value <= 32:
        return "017-032"
    return "033+"


def select_policy(rows: list[dict[str, Any]], policy: dict[str, Any], *, oracle_critical: bool = False) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if relation_score(row) >= safe_float(policy.get("threshold")):
            grouped[str(row.get("contains_support_set_key"))].append(row)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    top_k = int(policy.get("top_k") or 0)
    tail_slots = int(policy.get("tail_slots") or 0)
    overflow_slots = int(policy.get("critical_overflow_slots") or 0)
    threshold = safe_float(policy.get("critical_threshold"))

    def commit(row: dict[str, Any], reason: str) -> None:
        rid = relation_id(row)
        if rid in selected_ids:
            return
        item = dict(row)
        item["selection_reason"] = reason
        selected.append(item)
        selected_ids.add(rid)

    for key in sorted(grouped):
        bucket = grouped[key]
        for row in sorted(bucket, key=lambda item: (relation_score(item), support_score(item)), reverse=True):
            if int(row.get("contains_support_rank") or 999999) <= top_k:
                commit(row, "support_top_k")
        if tail_slots > 0:
            for row in sorted(bucket, key=lambda item: (int(item.get("contains_support_tail_rank") or 999999), -relation_score(item))):
                if int(row.get("contains_support_tail_rank") or 999999) <= tail_slots:
                    commit(row, "support_tail_slot")
        if oracle_critical:
            for row in bucket:
                if labels(row).get("support_critical"):
                    commit(row, "oracle_support_critical")
        elif overflow_slots > 0:
            overflowed = 0
            for row in sorted(bucket, key=lambda item: (support_score(item), relation_score(item)), reverse=True):
                if relation_id(row) in selected_ids or support_score(row) < threshold:
                    continue
                commit(row, "learned_critical_overflow")
                overflowed += 1
                if overflowed >= overflow_slots:
                    break
    return selected


def metrics(rows: list[dict[str, Any]], selected: list[dict[str, Any]]) -> dict[str, Any]:
    selected_ids = {relation_id(row) for row in selected}
    all_gold = {gold_key(row) for row in rows if gold_key(row)}
    matched_gold = {gold_key(row) for row in rows if gold_key(row) and relation_id(row) in selected_ids}
    critical_total = critical_kept = bridge_total = bridge_kept = 0
    duplicate_total = duplicate_kept = high_total = high_kept = 0
    split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        kept = relation_id(row) in selected_ids
        lab = labels(row)
        split_counts[split_name(row)]["input"] += 1
        split_counts[split_name(row)]["selected"] += int(kept)
        if lab.get("support_critical"):
            critical_total += 1
            critical_kept += int(kept)
            split_counts[split_name(row)]["critical"] += 1
            split_counts[split_name(row)]["critical_kept"] += int(kept)
        if lab.get("bridge_positive"):
            bridge_total += 1
            bridge_kept += int(kept)
        if lab.get("duplicate_support"):
            duplicate_total += 1
            duplicate_kept += int(kept)
        if lab.get("high_score_negative"):
            high_total += 1
            high_kept += int(kept)
    return {
        "input_edges": len(rows),
        "selected_edges": len(selected),
        "candidate_reduction": round(1.0 - len(selected) / max(len(rows), 1), 6),
        "gold_keys": len(all_gold),
        "gold_keys_matched": len(matched_gold),
        "gold_key_recall": round(len(matched_gold) / max(len(all_gold), 1), 6),
        "support_critical_recall": round(critical_kept / max(critical_total, 1), 6),
        "bridge_recall": round(bridge_kept / max(bridge_total, 1), 6),
        "duplicate_support_reduction": round(1.0 - duplicate_kept / max(duplicate_total, 1), 6),
        "high_score_negative_reduction": round(1.0 - high_kept / max(high_total, 1), 6),
        "split": {
            split: {
                "input_edges": int(counter["input"]),
                "selected_edges": int(counter["selected"]),
                "critical_edges": int(counter["critical"]),
                "critical_edges_kept": int(counter["critical_kept"]),
                "critical_recall": round(counter["critical_kept"] / max(counter["critical"], 1), 6),
            }
            for split, counter in sorted(split_counts.items())
        },
    }


def missed_critical_profile(rows: list[dict[str, Any]], selected: list[dict[str, Any]]) -> dict[str, Any]:
    selected_ids = {relation_id(row) for row in selected}
    missed = [row for row in rows if labels(row).get("support_critical") and relation_id(row) not in selected_ids]
    by_split = Counter(split_name(row) for row in missed)
    by_rank = Counter(rank_bucket(int(row.get("contains_support_rank") or 0)) for row in missed)
    by_tail = Counter(rank_bucket(int(row.get("contains_support_tail_rank") or 0)) for row in missed)
    by_size = Counter(size_bucket(int(row.get("contains_support_set_size") or 0)) for row in missed)
    by_label = Counter()
    scores = sorted(support_score(row) for row in missed)
    for row in missed:
        lab = labels(row)
        if lab.get("gold_representative"):
            by_label["gold_representative"] += 1
        if lab.get("bridge_positive"):
            by_label["bridge_positive"] += 1
    quantiles: dict[str, float] = {}
    if scores:
        for name, q in [("p00", 0.0), ("p25", 0.25), ("p50", 0.5), ("p75", 0.75), ("p90", 0.9), ("p100", 1.0)]:
            quantiles[name] = round(scores[min(len(scores) - 1, int(q * (len(scores) - 1)))], 6)
    return {
        "missed_critical_edges": len(missed),
        "by_split": dict(by_split),
        "by_support_rank": dict(by_rank),
        "by_support_tail_rank": dict(by_tail),
        "by_support_set_size": dict(by_size),
        "by_critical_label": dict(by_label),
        "support_criticality_score_quantiles": quantiles,
    }


def build_examples(rows: list[dict[str, Any]], selected: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    selected_ids = {relation_id(row) for row in selected}
    missed = [row for row in rows if labels(row).get("support_critical") and relation_id(row) not in selected_ids]
    missed.sort(
        key=lambda row: (
            split_name(row),
            int(row.get("contains_support_rank") or 999999),
            -support_score(row),
            relation_id(row),
        )
    )
    examples: list[dict[str, Any]] = []
    for row in missed[:limit]:
        examples.append(
            {
                "row_id": row.get("row_id"),
                "split": split_name(row),
                "relation_id": relation_id(row),
                "source_cluster_id": row.get("source_cluster_id"),
                "target_cluster_id": row.get("target_cluster_id"),
                "symbol_instance_cluster_id": row.get("symbol_instance_cluster_id"),
                "support_set_key": row.get("contains_support_set_key"),
                "support_set_size": row.get("contains_support_set_size"),
                "support_rank": row.get("contains_support_rank"),
                "support_tail_rank": row.get("contains_support_tail_rank"),
                "relation_score": relation_score(row),
                "support_criticality_score": support_score(row),
                "labels": labels(row),
            }
        )
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--output", default=str(DEFAULT_AUDIT))
    parser.add_argument("--examples-output", default=str(DEFAULT_EXAMPLES))
    parser.add_argument("--example-limit", type=int, default=200)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.dataset))
    model = json.loads(Path(args.model).read_text(encoding="utf-8")) if Path(args.model).exists() else {}
    locked_policy = model.get("policy") if isinstance(model.get("policy"), dict) else {
        "threshold": 0.01,
        "top_k": 4,
        "tail_slots": 4,
        "critical_overflow_slots": 2,
        "critical_threshold": 0.0,
    }
    policies = {
        "dev_best_aggressive": {
            "threshold": 0.01,
            "top_k": 2,
            "tail_slots": 2,
            "critical_overflow_slots": 2,
            "critical_threshold": 0.747445,
        },
        "locked_safe": locked_policy,
        "rank_tail_top8_tail8": {
            "threshold": 0.01,
            "top_k": 8,
            "tail_slots": 8,
            "critical_overflow_slots": 0,
            "critical_threshold": 1.0,
        },
        "oracle_top2_tail2_plus_all_critical": {
            "threshold": 0.01,
            "top_k": 2,
            "tail_slots": 2,
            "critical_overflow_slots": 0,
            "critical_threshold": 1.0,
            "oracle_critical": True,
        },
        "oracle_top4_plus_all_critical": {
            "threshold": 0.01,
            "top_k": 4,
            "tail_slots": 0,
            "critical_overflow_slots": 0,
            "critical_threshold": 1.0,
            "oracle_critical": True,
        },
    }
    policy_reports: dict[str, Any] = {}
    selected_by_policy: dict[str, list[dict[str, Any]]] = {}
    for name, policy in policies.items():
        selected = select_policy(rows, policy, oracle_critical=bool(policy.get("oracle_critical")))
        selected_by_policy[name] = selected
        policy_reports[name] = {
            "policy": policy,
            "metrics": metrics(rows, selected),
            "missed_critical_profile": missed_critical_profile(rows, selected),
        }

    report = {
        "task": "IMG-MOE-V18-REBUILD-005.contains_symbol_support_criticality_blockers",
        "dataset": args.dataset,
        "model": args.model,
        "rows": len(rows),
        "policy_reports": policy_reports,
        "diagnosis": (
            "dev_best_aggressive shows the compression target but misses full locked support-critical edges; "
            "oracle policies estimate the upper bound if support-critical representatives can be identified perfectly."
        ),
        "gold_loaded_after_inference_for_audit_only": True,
        "gold_used_for_inference": False,
    }
    write_json(Path(args.output), report)
    write_jsonl(Path(args.examples_output), build_examples(rows, selected_by_policy["dev_best_aggressive"], args.example_limit))
    print(
        json.dumps(
            {
                "rows": len(rows),
                "output": args.output,
                "dev_best": policy_reports["dev_best_aggressive"]["metrics"],
                "locked_safe": policy_reports["locked_safe"]["metrics"],
                "oracle_top2_tail2": policy_reports["oracle_top2_tail2_plus_all_critical"]["metrics"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
