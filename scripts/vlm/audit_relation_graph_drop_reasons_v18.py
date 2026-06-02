#!/usr/bin/env python3
"""Audit per-edge keep/drop reasons for the v18 relation-graph policy.

This is an evaluation-only tool. It loads offline labels from the reconstruction
dataset only to summarize which positives/bridge representatives survive each
inference-available pruning reason.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from relation_graph_reconstruction_v18 import (
    DEFAULT_DATASET,
    DEFAULT_MODEL,
    apply_listwise_scores,
    bounded_by_room_side,
    bounded_by_wall_segment_key,
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
DEFAULT_OUTPUT = REPORT / "relation_graph_drop_reason_audit_v18.json"
DEFAULT_EXAMPLES = REPORT / "relation_graph_drop_reason_examples_v18.jsonl"
DEFAULT_LISTWISE_MODEL = CHECKPOINT / "listwise_model.json"
DEFAULT_HARD_CASES = REPORT / "relation_graph_hard_cases_v18.jsonl"


def load_model(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def label_info(row: dict[str, Any]) -> dict[str, Any]:
    labels = row.get("labels") if isinstance(row.get("labels"), dict) else {}
    listwise_labels = row.get("listwise_labels") if isinstance(row.get("listwise_labels"), dict) else {}
    return {
        "label_positive": bool(labels.get("label_positive")),
        "gold_key": labels.get("gold_key"),
        "graph_role": row.get("graph_role") or labels.get("graph_role"),
        "is_gold_representative": bool(listwise_labels.get("is_gold_representative")),
        "is_bridge_positive": bool(listwise_labels.get("is_bridge_positive")),
        "is_duplicate_support": bool(listwise_labels.get("is_duplicate_support")),
    }


def score_dataset_rows(rows: list[dict[str, Any]], model: dict[str, Any]) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        relation = str(item.get("relation"))
        item["relation_score"] = score_row(item, relation_model(model, relation))
        scored.append(item)
    return apply_listwise_scores(scored, model)


def relation_id(row: dict[str, Any]) -> str:
    return str(row.get("relation_id") or f"{row.get('row_id')}|{row.get('source_candidate_id')}|{row.get('target_candidate_id')}")


def source_id(row: dict[str, Any]) -> str:
    return str(row.get("source_cluster_id") or row.get("source_candidate_id"))


def target_id(row: dict[str, Any]) -> str:
    return str(row.get("target_cluster_id") or row.get("target_candidate_id"))


def component_id(row: dict[str, Any]) -> str:
    return str(row.get("component_id") or "relation_component_missing")


def load_hard_case_index(path: Path | None) -> dict[str, set[str]]:
    if not path or not path.exists():
        return {}
    index: dict[str, set[str]] = defaultdict(set)
    for row in load_jsonl(path):
        rid = relation_id(row)
        case_type = str(row.get("case_type") or "")
        if rid and case_type:
            index[rid].add(case_type)
    return dict(index)


def audit_selection(rows: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return relation_id -> audit decision using the same ordering as select_rows."""
    decisions: dict[str, dict[str, Any]] = {}
    threshold_pass: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        relation = str(row.get("relation"))
        params = policy.get(relation) or {}
        threshold = safe_float(params.get("threshold"))
        rid = relation_id(row)
        if safe_float(row.get("relation_score")) < threshold:
            decisions[rid] = {
                "selected": False,
                "reason": "dropped_below_threshold",
                "selected_phase": None,
                "threshold": threshold,
            }
            continue
        threshold_pass[relation].append(row)

    for relation, bucket in threshold_pass.items():
        params = policy.get(relation) or {}
        max_per_source = int(params.get("max_per_source") or 999999)
        max_per_target = int(params.get("max_per_target") or 999999)
        max_per_component = int(params.get("max_per_component") or 999999)
        max_per_pair = int(params.get("max_per_pair") or 999999)
        max_pair_tail_slots = int(params.get("max_pair_tail_slots") or 0)
        max_pair_alignment_slots = int(params.get("max_pair_alignment_slots") or 0)
        max_per_source_side = int(params.get("max_per_source_side") or 999999)
        max_per_source_side_segment = int(params.get("max_per_source_side_segment") or 999999)
        max_wall_segment_tail_slots = int(params.get("max_wall_segment_tail_slots") or 0)
        max_wall_segment_support_overflow_slots = int(params.get("max_wall_segment_support_overflow_slots") or 0)
        wall_segment_support_score_rank_max = int(params.get("wall_segment_support_score_rank_max") or 0)
        wall_segment_support_tail_rank_max = int(params.get("wall_segment_support_tail_rank_max") or 0)
        wall_segment_bin_size = int(params.get("wall_segment_bin_size") or 16)

        source_counts: Counter[str] = Counter()
        target_counts: Counter[str] = Counter()
        component_counts: Counter[str] = Counter()
        pair_counts: Counter[tuple[str, str]] = Counter()
        pair_tail_counts: Counter[tuple[str, str]] = Counter()
        pair_alignment_counts: Counter[tuple[str, str]] = Counter()
        source_side_counts: Counter[tuple[str, str]] = Counter()
        source_side_segment_counts: Counter[tuple[str, str, str]] = Counter()
        source_side_segment_tail_counts: Counter[tuple[str, str, str]] = Counter()
        source_side_segment_support_counts: Counter[tuple[str, str, str]] = Counter()
        selected_ids: set[str] = set()

        ordered = sorted(bucket, key=lambda item: (safe_float(item.get("relation_score")), safe_float(item.get("confidence"))), reverse=True)

        def support_overflow_candidate(row: dict[str, Any]) -> bool:
            if relation != "bounded_by" or max_wall_segment_support_overflow_slots <= 0:
                return False
            listwise = row.get("listwise_features") if isinstance(row.get("listwise_features"), dict) else {}
            segment_score_rank = safe_float(listwise.get("bounded_wall_segment_score_rank"))
            segment_tail_rank = safe_float(listwise.get("bounded_wall_segment_tail_rank"))
            return (
                wall_segment_support_score_rank_max > 0
                and wall_segment_support_tail_rank_max > 0
                and 0 < segment_score_rank <= wall_segment_support_score_rank_max
                and 0 < segment_tail_rank <= wall_segment_support_tail_rank_max
            )

        def inspect(row: dict[str, Any], *, allow_support_overflow: bool = False) -> tuple[bool, str, dict[str, Any]]:
            rid = relation_id(row)
            src = source_id(row)
            tgt = target_id(row)
            comp = component_id(row)
            pair = (src, tgt)
            side = bounded_by_room_side(row) if relation == "bounded_by" else relation
            segment = bounded_by_wall_segment_key(row, wall_segment_bin_size) if relation == "bounded_by" else relation
            source_side = (src, side)
            source_side_segment = (src, side, segment)
            listwise = row.get("listwise_features") if isinstance(row.get("listwise_features"), dict) else {}
            pair_tail_rank = safe_float(listwise.get("cluster_pair_tail_rank"))
            pair_alignment_rank = safe_float(listwise.get("cluster_pair_alignment_rank"))
            wall_segment_tail_rank = safe_float(listwise.get("bounded_wall_segment_tail_rank"))
            use_tail_slot = max_pair_tail_slots > 0 and 0 < pair_tail_rank <= max_pair_tail_slots
            use_alignment_slot = max_pair_alignment_slots > 0 and 0 < pair_alignment_rank <= max_pair_alignment_slots
            use_wall_segment_tail_slot = relation == "bounded_by" and max_wall_segment_tail_slots > 0 and 0 < wall_segment_tail_rank <= max_wall_segment_tail_slots
            can_use_wall_segment_tail = use_wall_segment_tail_slot and source_side_segment_tail_counts[source_side_segment] < max_wall_segment_tail_slots
            can_use_support_overflow = (
                allow_support_overflow
                and support_overflow_candidate(row)
                and source_side_segment_support_counts[source_side_segment] < max_wall_segment_support_overflow_slots
            )
            can_use_tail = use_tail_slot and pair_tail_counts[pair] < max_pair_tail_slots
            can_use_alignment = use_alignment_slot and pair_alignment_counts[pair] < max_pair_alignment_slots
            meta = {
                "source_id": src,
                "target_id": tgt,
                "component_id": comp,
                "pair_id": "|".join(pair),
                "room_side": side,
                "wall_segment_key": segment,
                "pair_tail_rank": pair_tail_rank,
                "pair_alignment_rank": pair_alignment_rank,
                "wall_segment_score_rank": safe_float(listwise.get("bounded_wall_segment_score_rank")),
                "wall_segment_tail_rank": wall_segment_tail_rank,
                "source_count_before": source_counts[src],
                "target_count_before": target_counts[tgt],
                "component_count_before": component_counts[comp],
                "pair_count_before": pair_counts[pair],
                "source_side_count_before": source_side_counts[source_side],
                "source_side_segment_count_before": source_side_segment_counts[source_side_segment],
                "support_overflow_candidate": support_overflow_candidate(row),
                "can_use_support_overflow": can_use_support_overflow,
            }
            if rid in selected_ids:
                return False, "dropped_already_selected", meta
            if source_counts[src] >= max_per_source:
                return False, "dropped_source_cap", meta
            if target_counts[tgt] >= max_per_target:
                return False, "dropped_target_cap", meta
            if component_counts[comp] >= max_per_component:
                return False, "dropped_component_cap", meta
            if source_side_counts[source_side] >= max_per_source_side:
                return False, "dropped_source_side_cap", meta
            if source_side_segment_counts[source_side_segment] >= max_per_source_side_segment and not (can_use_wall_segment_tail or can_use_support_overflow):
                return False, "dropped_wall_segment_cap", meta
            if pair_counts[pair] >= max_per_pair and not (can_use_tail or can_use_alignment):
                return False, "dropped_pair_cap", meta
            return True, "selected_support_overflow" if can_use_support_overflow else "selected_normal", meta

        def commit(row: dict[str, Any], reason: str, meta: dict[str, Any]) -> None:
            rid = relation_id(row)
            src = source_id(row)
            tgt = target_id(row)
            comp = component_id(row)
            pair = (src, tgt)
            side = bounded_by_room_side(row) if relation == "bounded_by" else relation
            segment = bounded_by_wall_segment_key(row, wall_segment_bin_size) if relation == "bounded_by" else relation
            source_side = (src, side)
            source_side_segment = (src, side, segment)
            listwise = row.get("listwise_features") if isinstance(row.get("listwise_features"), dict) else {}
            pair_tail_rank = safe_float(listwise.get("cluster_pair_tail_rank"))
            pair_alignment_rank = safe_float(listwise.get("cluster_pair_alignment_rank"))
            wall_segment_tail_rank = safe_float(listwise.get("bounded_wall_segment_tail_rank"))
            use_tail_slot = max_pair_tail_slots > 0 and 0 < pair_tail_rank <= max_pair_tail_slots
            use_alignment_slot = max_pair_alignment_slots > 0 and 0 < pair_alignment_rank <= max_pair_alignment_slots
            use_wall_segment_tail_slot = relation == "bounded_by" and max_wall_segment_tail_slots > 0 and 0 < wall_segment_tail_rank <= max_wall_segment_tail_slots
            selected_ids.add(rid)
            source_counts[src] += 1
            target_counts[tgt] += 1
            component_counts[comp] += 1
            source_side_counts[source_side] += 1
            source_side_segment_counts[source_side_segment] += 1
            if use_wall_segment_tail_slot:
                source_side_segment_tail_counts[source_side_segment] += 1
            if reason == "selected_support_overflow":
                source_side_segment_support_counts[source_side_segment] += 1
            pair_counts[pair] += 1
            if use_tail_slot:
                pair_tail_counts[pair] += 1
            if use_alignment_slot:
                pair_alignment_counts[pair] += 1
            decisions[rid] = {
                "selected": True,
                "reason": reason,
                "selected_phase": "support_overflow" if reason == "selected_support_overflow" else "normal",
                **meta,
            }

        for row in ordered:
            ok, reason, meta = inspect(row)
            if ok:
                commit(row, reason, meta)
            else:
                decisions[relation_id(row)] = {"selected": False, "reason": reason, "selected_phase": None, **meta}

        if relation == "bounded_by" and max_wall_segment_support_overflow_slots > 0:
            support_ordered = [row for row in bucket if support_overflow_candidate(row)]
            support_ordered.sort(
                key=lambda item: (
                    safe_float((item.get("listwise_features") or {}).get("bounded_wall_segment_tail_rank")),
                    safe_float((item.get("listwise_features") or {}).get("bounded_wall_segment_score_rank")),
                    -safe_float(item.get("relation_score")),
                    relation_id(item),
                )
            )
            for row in support_ordered:
                if relation_id(row) in selected_ids:
                    continue
                ok, reason, meta = inspect(row, allow_support_overflow=True)
                if ok:
                    commit(row, reason, meta)
                elif decisions.get(relation_id(row), {}).get("reason") == "dropped_wall_segment_cap":
                    decisions[relation_id(row)] = {"selected": False, "reason": reason, "selected_phase": None, **meta}
    return decisions


def summarize(
    rows: list[dict[str, Any]],
    decisions: dict[str, dict[str, Any]],
    hard_case_index: dict[str, set[str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_relation: dict[str, dict[str, Any]] = {}
    examples: list[dict[str, Any]] = []
    for relation in sorted({str(row.get("relation")) for row in rows}):
        bucket = [row for row in rows if str(row.get("relation")) == relation]
        reason_counts: Counter[str] = Counter()
        selected_reason_counts: Counter[str] = Counter()
        positive_by_reason: Counter[str] = Counter()
        representative_by_reason: Counter[str] = Counter()
        bridge_by_reason: Counter[str] = Counter()
        duplicate_support_by_reason: Counter[str] = Counter()
        hard_case_counts: Counter[str] = Counter()
        selected_hard_case_counts: Counter[str] = Counter()
        hard_case_by_reason: dict[str, Counter[str]] = defaultdict(Counter)
        selected = 0
        positives = 0
        selected_positives = 0
        reps = 0
        selected_reps = 0
        bridges = 0
        selected_bridges = 0
        duplicate_support = 0
        selected_duplicate_support = 0
        for row in bucket:
            rid = relation_id(row)
            decision = decisions.get(rid) or {"selected": False, "reason": "missing_decision"}
            reason = str(decision.get("reason"))
            info = label_info(row)
            hard_cases = sorted(hard_case_index.get(rid) or [])
            reason_counts[reason] += 1
            if decision.get("selected"):
                selected += 1
                selected_reason_counts[reason] += 1
            for case_type in hard_cases:
                hard_case_counts[case_type] += 1
                hard_case_by_reason[case_type][reason] += 1
                if decision.get("selected"):
                    selected_hard_case_counts[case_type] += 1
            if info["label_positive"]:
                positives += 1
                positive_by_reason[reason] += 1
                if decision.get("selected"):
                    selected_positives += 1
            if info["is_gold_representative"]:
                reps += 1
                representative_by_reason[reason] += 1
                if decision.get("selected"):
                    selected_reps += 1
            if info["is_bridge_positive"]:
                bridges += 1
                bridge_by_reason[reason] += 1
                if decision.get("selected"):
                    selected_bridges += 1
            if info["is_duplicate_support"]:
                duplicate_support += 1
                duplicate_support_by_reason[reason] += 1
                if decision.get("selected"):
                    selected_duplicate_support += 1
            if len(examples) < 200 and (
                (info["label_positive"] and not decision.get("selected"))
                or hard_cases
                or reason in {"selected_support_overflow", "dropped_wall_segment_cap", "dropped_pair_cap"}
            ):
                examples.append(
                    {
                        "row_id": row.get("row_id"),
                        "relation_id": rid,
                        "relation": relation,
                        "selected": bool(decision.get("selected")),
                        "reason": reason,
                        "relation_score": row.get("relation_score"),
                        "source_candidate_id": row.get("source_candidate_id"),
                        "target_candidate_id": row.get("target_candidate_id"),
                        "source_cluster_id": row.get("source_cluster_id"),
                        "target_cluster_id": row.get("target_cluster_id"),
                        "component_id": row.get("component_id"),
                        "labels": info,
                        "hard_case_types": hard_cases,
                        "decision_meta": {
                            key: decision.get(key)
                            for key in [
                                "source_id",
                                "target_id",
                                "room_side",
                                "wall_segment_key",
                                "pair_tail_rank",
                                "wall_segment_score_rank",
                                "wall_segment_tail_rank",
                                "source_side_segment_count_before",
                                "support_overflow_candidate",
                                "can_use_support_overflow",
                            ]
                            if key in decision
                        },
                    }
                )
        by_relation[relation] = {
            "input_edges": len(bucket),
            "selected_edges": selected,
            "candidate_reduction": round(1.0 - selected / max(len(bucket), 1), 6),
            "reason_counts": dict(reason_counts),
            "selected_reason_counts": dict(selected_reason_counts),
            "positive_edges": positives,
            "selected_positive_edges": selected_positives,
            "positive_edge_recall": round(selected_positives / max(positives, 1), 6),
            "gold_representatives": reps,
            "selected_gold_representatives": selected_reps,
            "gold_representative_recall": round(selected_reps / max(reps, 1), 6),
            "bridge_positives": bridges,
            "selected_bridge_positives": selected_bridges,
            "bridge_positive_recall": round(selected_bridges / max(bridges, 1), 6),
            "duplicate_support_edges": duplicate_support,
            "selected_duplicate_support_edges": selected_duplicate_support,
            "duplicate_support_reduction": round(1.0 - selected_duplicate_support / max(duplicate_support, 1), 6),
            "hard_case_counts": dict(hard_case_counts),
            "selected_hard_case_counts": dict(selected_hard_case_counts),
            "hard_case_recall": {
                case_type: round(selected_hard_case_counts[case_type] / max(count, 1), 6)
                for case_type, count in sorted(hard_case_counts.items())
            },
            "hard_case_by_reason": {case_type: dict(counter) for case_type, counter in hard_case_by_reason.items()},
            "positive_by_reason": dict(positive_by_reason),
            "gold_representative_by_reason": dict(representative_by_reason),
            "bridge_positive_by_reason": dict(bridge_by_reason),
            "duplicate_support_by_reason": dict(duplicate_support_by_reason),
        }
    overall_edges = len(rows)
    overall_selected = sum(1 for row in rows if decisions.get(relation_id(row), {}).get("selected"))
    report = {
        "task": "IMG-MOE-V18-REBUILD-005",
        "audit_type": "relation_graph_per_edge_drop_reasons",
        "input_edges": overall_edges,
        "selected_edges": overall_selected,
        "candidate_reduction": round(1.0 - overall_selected / max(overall_edges, 1), 6),
        "by_relation": by_relation,
        "gold_loaded_after_inference_for_audit_only": True,
        "gold_used_for_inference": False,
    }
    return report, examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--model", default=str(DEFAULT_LISTWISE_MODEL))
    parser.add_argument("--hard-cases", default=str(DEFAULT_HARD_CASES))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--examples-output", default=str(DEFAULT_EXAMPLES))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    dataset = load_jsonl(Path(args.dataset), limit=args.limit)
    model = load_model(Path(args.model))
    hard_case_index = load_hard_case_index(Path(args.hard_cases) if args.hard_cases else None)
    policy = model.get("policy") or {}
    scored = score_dataset_rows(dataset, model)
    decisions = audit_selection(scored, policy)
    report, examples = summarize(scored, decisions, hard_case_index)
    report["dataset"] = args.dataset
    report["model"] = args.model
    report["hard_cases"] = args.hard_cases
    report["hard_case_indexed_edges"] = len(hard_case_index)
    report["policy"] = policy
    write_json(Path(args.output), report)
    write_jsonl(Path(args.examples_output), examples)
    print(
        json.dumps(
            {
                "input_edges": report["input_edges"],
                "selected_edges": report["selected_edges"],
                "candidate_reduction": report["candidate_reduction"],
                "output": args.output,
                "examples_output": args.examples_output,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
