#!/usr/bin/env python3
"""Audit support-set compression policies for contains_symbol.

The current relation-graph policy barely compresses contains_symbol because a
plain pair cap keeps almost every room-symbol duplicate. This script evaluates
room/symbol support-set policies offline, using labels only for reporting and
policy selection diagnostics.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from relation_graph_reconstruction_v18 import (
    DEFAULT_DATASET,
    apply_listwise_scores,
    load_jsonl,
    relation_model,
    safe_float,
    score_row,
    write_json,
    write_jsonl,
)

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/vlm"
CHECKPOINT = ROOT / "checkpoints/relation_graph_policy_v18"

DEFAULT_MODEL = CHECKPOINT / "listwise_model.json"
DEFAULT_HARD_CASES = REPORT / "relation_graph_hard_cases_v18.jsonl"
DEFAULT_OUTPUT = REPORT / "contains_symbol_support_set_policy_v18_audit.json"
DEFAULT_EXAMPLES = REPORT / "contains_symbol_support_set_policy_v18_examples.jsonl"


def load_model(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def relation_id(row: dict[str, Any]) -> str:
    return str(row.get("relation_id") or f"{row.get('row_id')}|{row.get('source_candidate_id')}|{row.get('target_candidate_id')}")


def label_positive(row: dict[str, Any]) -> bool:
    labels = row.get("labels") if isinstance(row.get("labels"), dict) else {}
    return bool(labels.get("label_positive"))


def gold_key(row: dict[str, Any]) -> str | None:
    labels = row.get("labels") if isinstance(row.get("labels"), dict) else {}
    value = labels.get("gold_key")
    return str(value) if value else None


def load_hard_case_index(path: Path | None) -> dict[str, set[str]]:
    if not path or not path.exists():
        return {}
    out: dict[str, set[str]] = defaultdict(set)
    for row in load_jsonl(path):
        if str(row.get("relation")) != "contains_symbol":
            continue
        case_type = str(row.get("case_type") or "")
        if case_type:
            out[relation_id(row)].add(case_type)
    return dict(out)


def score_contains_rows(dataset: list[dict[str, Any]], model: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in dataset if str(row.get("relation")) == "contains_symbol"]
    scored: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["relation_score"] = score_row(item, relation_model(model, "contains_symbol"))
        scored.append(item)
    return apply_listwise_scores(scored, model)


def support_key(row: dict[str, Any], variant: str) -> str:
    row_id = str(row.get("row_id"))
    source = str(row.get("source_cluster_id") or row.get("source_candidate_id"))
    target = str(row.get("target_cluster_id") or row.get("target_candidate_id"))
    symbol = str(row.get("symbol_instance_cluster_id") or target)
    component = str(row.get("component_id") or "component_missing")
    if variant == "room_symbol_instance":
        return "|".join([row_id, source, symbol])
    if variant == "room_target_cluster":
        return "|".join([row_id, source, target])
    if variant == "component_room_symbol_instance":
        return "|".join([row_id, component, source, symbol])
    if variant == "component_symbol_instance":
        return "|".join([row_id, component, symbol])
    raise ValueError(f"unknown support key variant: {variant}")


def annotate_support_ranks(rows: list[dict[str, Any]], variant: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[support_key(row, variant)].append(row)
    out: list[dict[str, Any]] = []
    for key, bucket in grouped.items():
        ordered = sorted(bucket, key=lambda item: (safe_float(item.get("relation_score")), safe_float(item.get("confidence"))), reverse=True)
        tail_ordered = list(reversed(ordered))
        rank_by_id = {relation_id(row): index for index, row in enumerate(ordered, start=1)}
        tail_by_id = {relation_id(row): index for index, row in enumerate(tail_ordered, start=1)}
        for row in bucket:
            item = dict(row)
            item["contains_support_set_key"] = key
            item["contains_support_set_size"] = len(bucket)
            item["contains_support_rank"] = rank_by_id[relation_id(row)]
            item["contains_support_tail_rank"] = tail_by_id[relation_id(row)]
            out.append(item)
    return out


def select_support_policy(rows: list[dict[str, Any]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    threshold = safe_float(policy.get("threshold"))
    top_k = int(policy.get("top_k") or 0)
    tail_slots = int(policy.get("tail_slots") or 0)
    max_per_room = int(policy.get("max_per_room") or 999999)
    max_per_symbol = int(policy.get("max_per_symbol") or 999999)
    if max_per_room >= 999999 and max_per_symbol >= 999999:
        selected_fast: list[dict[str, Any]] = []
        for row in rows:
            if safe_float(row.get("relation_score")) < threshold:
                continue
            rank = int(row.get("contains_support_rank") or 999999)
            tail_rank = int(row.get("contains_support_tail_rank") or 999999)
            if rank <= top_k:
                item = dict(row)
                item["support_set_selection_reason"] = "selected_support_top_k"
                selected_fast.append(item)
            elif tail_slots > 0 and tail_rank <= tail_slots:
                item = dict(row)
                item["support_set_selection_reason"] = "selected_support_tail_slot"
                selected_fast.append(item)
        return selected_fast
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    room_counts: Counter[str] = Counter()
    symbol_counts: Counter[str] = Counter()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if safe_float(row.get("relation_score")) >= threshold:
            grouped[str(row.get("contains_support_set_key"))].append(row)

    def can_select(row: dict[str, Any]) -> bool:
        rid = relation_id(row)
        if rid in selected_ids:
            return False
        room = str(row.get("source_cluster_id") or row.get("source_candidate_id"))
        symbol = str(row.get("symbol_instance_cluster_id") or row.get("target_cluster_id") or row.get("target_candidate_id"))
        return room_counts[room] < max_per_room and symbol_counts[symbol] < max_per_symbol

    def commit(row: dict[str, Any], reason: str) -> None:
        item = dict(row)
        item["support_set_selection_reason"] = reason
        selected.append(item)
        selected_ids.add(relation_id(row))
        room = str(row.get("source_cluster_id") or row.get("source_candidate_id"))
        symbol = str(row.get("symbol_instance_cluster_id") or row.get("target_cluster_id") or row.get("target_candidate_id"))
        room_counts[room] += 1
        symbol_counts[symbol] += 1

    for key in sorted(grouped):
        bucket = grouped[key]
        ordered = sorted(bucket, key=lambda item: (safe_float(item.get("relation_score")), safe_float(item.get("confidence"))), reverse=True)
        for row in ordered:
            if int(row.get("contains_support_rank") or 999999) <= top_k and can_select(row):
                commit(row, "selected_support_top_k")
        if tail_slots > 0:
            tail_ordered = sorted(bucket, key=lambda item: (int(item.get("contains_support_tail_rank") or 999999), -safe_float(item.get("relation_score")), relation_id(item)))
            for row in tail_ordered:
                if int(row.get("contains_support_tail_rank") or 999999) <= tail_slots and can_select(row):
                    commit(row, "selected_support_tail_slot")
    return selected


def metrics(selected: list[dict[str, Any]], all_rows: list[dict[str, Any]], hard_case_index: dict[str, set[str]]) -> dict[str, Any]:
    selected_ids = {relation_id(row) for row in selected}
    all_gold = {gold_key(row) for row in all_rows if gold_key(row)}
    selected_gold: set[str] = set()
    selected_positive_edges = 0
    duplicate_positive_selected = 0
    total_positive_edges = 0
    total_duplicate_positive = 0
    seen_all_positive: set[str] = set()
    reason_counts = Counter(row.get("support_set_selection_reason") for row in selected)
    hard_counts: Counter[str] = Counter()
    hard_selected: Counter[str] = Counter()
    hard_dropped_by_case_reason: dict[str, Counter[str]] = defaultdict(Counter)
    for row in all_rows:
        key = gold_key(row)
        rid = relation_id(row)
        selected_flag = rid in selected_ids
        if key:
            total_positive_edges += 1
            if key in seen_all_positive:
                total_duplicate_positive += 1
            seen_all_positive.add(key)
            if selected_flag:
                selected_positive_edges += 1
                if key in selected_gold:
                    duplicate_positive_selected += 1
                selected_gold.add(key)
        for case_type in hard_case_index.get(rid, set()):
            hard_counts[case_type] += 1
            if selected_flag:
                hard_selected[case_type] += 1
            else:
                rank = int(row.get("contains_support_rank") or 0)
                tail = int(row.get("contains_support_tail_rank") or 0)
                if safe_float(row.get("relation_score")) < 0.01:
                    reason = "dropped_below_threshold"
                elif rank > 0:
                    reason = "dropped_support_rank"
                elif tail > 0:
                    reason = "dropped_support_tail_rank"
                else:
                    reason = "dropped_other"
                hard_dropped_by_case_reason[case_type][reason] += 1
    selected_count = len(selected)
    return {
        "selected_edges": selected_count,
        "candidate_reduction": round(1.0 - selected_count / max(len(all_rows), 1), 6),
        "recoverable_gold_keys": len(all_gold),
        "selected_gold_keys": len(selected_gold),
        "gold_key_recall": round(len(selected_gold) / max(len(all_gold), 1), 6),
        "selected_positive_edges": selected_positive_edges,
        "positive_edge_recall": round(selected_positive_edges / max(total_positive_edges, 1), 6),
        "precision_against_selected_edges": round(len(selected_gold) / max(selected_count, 1), 6),
        "duplicate_positive_edges_before": total_duplicate_positive,
        "duplicate_positive_edges_selected": duplicate_positive_selected,
        "duplicate_positive_reduction": round(1.0 - duplicate_positive_selected / max(total_duplicate_positive, 1), 6),
        "selection_reason_counts": dict(reason_counts),
        "hard_case_counts": dict(hard_counts),
        "hard_case_selected": dict(hard_selected),
        "hard_case_recall": {
            case_type: round(hard_selected[case_type] / max(count, 1), 6)
            for case_type, count in sorted(hard_counts.items())
        },
        "hard_case_dropped_by_reason": {case_type: dict(counter) for case_type, counter in sorted(hard_dropped_by_case_reason.items())},
    }


def sweep(rows: list[dict[str, Any]], hard_case_index: dict[str, set[str]]) -> list[dict[str, Any]]:
    policies: list[dict[str, Any]] = []
    for variant in [
        "room_symbol_instance",
        "room_target_cluster",
        "component_room_symbol_instance",
        "component_symbol_instance",
    ]:
        annotated = annotate_support_ranks(rows, variant)
        for top_k in [1, 2, 3, 4, 6, 8, 12, 16]:
            for tail_slots in [0, 1, 2]:
                policy = {
                    "support_key": variant,
                    "threshold": 0.01,
                    "top_k": top_k,
                    "tail_slots": tail_slots,
                    "max_per_room": 999999,
                    "max_per_symbol": 999999,
                }
                selected = select_support_policy(annotated, policy)
                policies.append({"policy": policy, "metrics": metrics(selected, annotated, hard_case_index)})
    policies.sort(
        key=lambda item: (
            item["metrics"]["gold_key_recall"] >= 1.0,
            item["metrics"]["gold_key_recall"],
            item["metrics"]["candidate_reduction"],
            item["metrics"]["duplicate_positive_reduction"],
            item["metrics"]["precision_against_selected_edges"],
        ),
        reverse=True,
    )
    return policies


def build_examples(rows: list[dict[str, Any]], selected: list[dict[str, Any]], hard_case_index: dict[str, set[str]], limit: int = 200) -> list[dict[str, Any]]:
    selected_ids = {relation_id(row) for row in selected}
    examples: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: (str(item.get("row_id")), str(item.get("contains_support_set_key")), int(item.get("contains_support_rank") or 0))):
        rid = relation_id(row)
        interesting = bool(gold_key(row) and rid not in selected_ids) or bool(hard_case_index.get(rid)) or int(row.get("contains_support_set_size") or 0) >= 8
        if not interesting:
            continue
        examples.append(
            {
                "row_id": row.get("row_id"),
                "relation_id": rid,
                "selected": rid in selected_ids,
                "relation_score": row.get("relation_score"),
                "source_candidate_id": row.get("source_candidate_id"),
                "target_candidate_id": row.get("target_candidate_id"),
                "source_cluster_id": row.get("source_cluster_id"),
                "target_cluster_id": row.get("target_cluster_id"),
                "symbol_instance_cluster_id": row.get("symbol_instance_cluster_id"),
                "component_id": row.get("component_id"),
                "support_set_key": row.get("contains_support_set_key"),
                "support_set_size": row.get("contains_support_set_size"),
                "support_rank": row.get("contains_support_rank"),
                "support_tail_rank": row.get("contains_support_tail_rank"),
                "gold_key": gold_key(row),
                "label_positive": label_positive(row),
                "hard_case_types": sorted(hard_case_index.get(rid) or []),
            }
        )
        if len(examples) >= limit:
            break
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--hard-cases", default=str(DEFAULT_HARD_CASES))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--examples-output", default=str(DEFAULT_EXAMPLES))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    dataset = load_jsonl(Path(args.dataset), limit=args.limit)
    model = load_model(Path(args.model))
    hard_case_index = load_hard_case_index(Path(args.hard_cases) if args.hard_cases else None)
    rows = score_contains_rows(dataset, model)
    top = sweep(rows, hard_case_index)
    best = top[0] if top else {"policy": {}, "metrics": {}}
    best_rows = annotate_support_ranks(rows, str(best["policy"].get("support_key") or "room_symbol_instance"))
    best_selected = select_support_policy(best_rows, best["policy"])
    examples = build_examples(best_rows, best_selected, hard_case_index)
    report = {
        "task": "IMG-MOE-V18-REBUILD-005.contains_symbol_support_set_policy_audit",
        "dataset": args.dataset,
        "model": args.model,
        "hard_cases": args.hard_cases,
        "input_edges": len(rows),
        "positive_edges": sum(1 for row in rows if label_positive(row)),
        "recoverable_gold_keys": len({gold_key(row) for row in rows if gold_key(row)}),
        "hard_case_indexed_edges": len(hard_case_index),
        "best_policy": best,
        "top_policies": top[:25],
        "gold_loaded_after_inference_for_audit_only": True,
        "gold_used_for_inference": False,
    }
    write_json(Path(args.output), report)
    write_jsonl(Path(args.examples_output), examples)
    print(
        json.dumps(
            {
                "input_edges": report["input_edges"],
                "best_policy": best.get("policy"),
                "best_metrics": best.get("metrics"),
                "output": args.output,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
