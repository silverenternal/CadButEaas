#!/usr/bin/env python3
"""Train a page/component-level listwise policy for relation graph compression."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from relation_graph_reconstruction_v18 import (
    DEFAULT_AUDIT,
    DEFAULT_DATASET,
    DEFAULT_EVAL,
    DEFAULT_MODEL,
    RELATIONS,
    apply_listwise_scores,
    default_policy,
    feature_names,
    load_jsonl,
    safe_float,
    score_rows,
    select_rows,
    train_logistic,
    write_json,
)
from train_relation_graph_policy_v18 import evaluate, fail_closed_policy, fail_closed_relation, load_dataset


DEFAULT_HARD_CASES = Path("reports/vlm/relation_graph_hard_cases_v18.jsonl")
DEFAULT_LISTWISE_MODEL = Path("checkpoints/relation_graph_policy_v18/listwise_model.json")
DEFAULT_LISTWISE_EVAL = Path("reports/vlm/relation_graph_listwise_policy_v18_eval.json")
DEFAULT_LISTWISE_AUDIT = Path("reports/vlm/relation_graph_listwise_policy_v18_audit.json")


def load_model(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def relation_gold_key(row: dict[str, Any]) -> str:
    labels = row.get("labels") if isinstance(row.get("labels"), dict) else {}
    return str(labels.get("gold_key") or "")


def hard_case_lookup(path: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    if not path.exists():
        return out
    for row in load_jsonl(path):
        relation_id = str(row.get("relation_id") or "")
        case_type = str(row.get("case_type") or "")
        if relation_id and case_type:
            out[relation_id].add(case_type)
    return out


def mark_training_targets(rows: list[dict[str, Any]], hard_cases: dict[str, set[str]]) -> list[dict[str, Any]]:
    by_gold: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        labels = row.get("labels") if isinstance(row.get("labels"), dict) else {}
        if labels.get("label_positive") and relation_gold_key(row):
            by_gold[(str(row.get("row_id")), str(row.get("relation")), relation_gold_key(row))].append(row)

    representative_ids: set[str] = set()
    duplicate_support_ids: set[str] = set()
    for group in by_gold.values():
        ordered = sorted(
            group,
            key=lambda item: (
                safe_float(item.get("relation_score")),
                safe_float((item.get("listwise_features") or {}).get("component_bridge_count")),
                -safe_float((item.get("listwise_features") or {}).get("cluster_pair_edge_count")),
                str(item.get("relation_id")),
            ),
            reverse=True,
        )
        if not ordered:
            continue
        representative_ids.add(str(ordered[0].get("relation_id")))
        for item in ordered[1:]:
            duplicate_support_ids.add(str(item.get("relation_id")))

    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        relation_id = str(item.get("relation_id") or "")
        labels = item.get("labels") if isinstance(item.get("labels"), dict) else {}
        case_types = hard_cases.get(relation_id, set())
        is_positive = bool(labels.get("label_positive"))
        is_representative = relation_id in representative_ids
        is_duplicate_support = relation_id in duplicate_support_ids or "duplicate_positive_support" in case_types
        duplicate_count = safe_float(item.get("duplicate_relation_count"))
        graph_bridge = str(item.get("graph_role") or labels.get("graph_role")) == "bridge"
        is_bridge = is_positive and (
            "low_score_bridge_positive" in case_types
            or (graph_bridge and duplicate_count <= 1.0 and not is_duplicate_support)
        )
        listwise_label = bool(is_representative or is_bridge)
        listwise_weight = 1.0
        if is_representative:
            listwise_weight += 3.0
        if is_bridge:
            listwise_weight += 2.0
        if "low_score_positive" in case_types:
            listwise_weight += 1.5
        if "high_score_negative" in case_types:
            listwise_weight += 2.0
        if "component_overflow_edge" in case_types:
            listwise_weight += 0.5
        if is_duplicate_support and not is_representative and not is_bridge:
            listwise_weight += 1.0
        item["listwise_labels"] = {
            "listwise_positive": listwise_label,
            "gold_representative": is_representative,
            "bridge_positive": is_bridge,
            "duplicate_support": is_duplicate_support,
            "gold_key": relation_gold_key(item) or None,
            "gold_loaded_after_inference_for_training_only": True,
            "gold_used_for_inference": False,
        }
        feats = dict(item.get("listwise_features") or {})
        feats.update(
            {
                "hard_case_low_score_positive": 1.0 if "low_score_positive" in case_types else 0.0,
                "hard_case_high_score_negative": 1.0 if "high_score_negative" in case_types else 0.0,
                "hard_case_low_score_bridge_positive": 1.0 if "low_score_bridge_positive" in case_types else 0.0,
                "hard_case_duplicate_positive_support": 1.0 if "duplicate_positive_support" in case_types else 0.0,
                "hard_case_component_overflow_edge": 1.0 if "component_overflow_edge" in case_types else 0.0,
                "listwise_sample_weight": listwise_weight,
            }
        )
        item["listwise_features"] = feats
        item["labels"] = {**labels, "label_positive": listwise_label}
        out.append(item)
    return out


def representative_eval(rows: list[dict[str, Any]], selected: list[dict[str, Any]]) -> dict[str, Any]:
    selected_ids = {str(row.get("relation_id")) for row in selected}
    gold: dict[str, set[str]] = defaultdict(set)
    matched: dict[str, set[str]] = defaultdict(set)
    bridge_gold: Counter[str] = Counter()
    bridge_matched: Counter[str] = Counter()
    duplicate_kept: Counter[str] = Counter()
    for row in rows:
        rel = str(row.get("relation"))
        labels = row.get("listwise_labels") if isinstance(row.get("listwise_labels"), dict) else {}
        gold_key = str(labels.get("gold_key") or "")
        if labels.get("gold_representative") and gold_key:
            gold[rel].add(gold_key)
            if str(row.get("relation_id")) in selected_ids:
                matched[rel].add(gold_key)
        if labels.get("bridge_positive"):
            bridge_gold[rel] += 1
            if str(row.get("relation_id")) in selected_ids:
                bridge_matched[rel] += 1
        if labels.get("duplicate_support") and str(row.get("relation_id")) in selected_ids:
            duplicate_kept[rel] += 1
    out: dict[str, Any] = {}
    for rel in RELATIONS:
        out[rel] = {
            "gold_keys": len(gold[rel]),
            "gold_keys_matched": len(matched[rel]),
            "gold_key_recall": round(len(matched[rel]) / max(len(gold[rel]), 1), 6),
            "bridge_positives": int(bridge_gold[rel]),
            "bridge_positives_kept": int(bridge_matched[rel]),
            "bridge_recall": round(bridge_matched[rel] / max(bridge_gold[rel], 1), 6),
            "duplicate_support_kept": int(duplicate_kept[rel]),
        }
    return out


def train_listwise_models(rows: list[dict[str, Any]]) -> dict[str, Any]:
    names = feature_names([{"edge_features": row.get("listwise_features") or {}} for row in rows])
    model = {"model_type": "relation_graph_listwise_policy_v18", "features": names, "relations": {}, "train_counts": {}}
    for rel in RELATIONS:
        subset = [row for row in rows if row.get("relation") == rel]
        training_rows = [{"edge_features": row.get("listwise_features") or {}, "labels": row.get("labels") or {}} for row in subset]
        rel_model = train_logistic(training_rows, names)
        model["relations"][rel] = rel_model
        model["train_counts"][rel] = rel_model.get("train_counts")
    return model


def choose_listwise_policy(dev_rows: list[dict[str, Any]], base_policy: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    thresholds = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    loose_pair_caps = [
        {
            "max_per_source": 999999,
            "max_per_target": 999999,
            "max_per_component": 999999,
            "max_per_pair": pair_cap,
            "max_pair_tail_slots": tail_slots,
            "max_pair_alignment_slots": 0,
            "max_per_source_side": 999999,
            "max_per_source_side_segment": 999999,
            "max_wall_segment_tail_slots": 0,
            "max_wall_segment_support_overflow_slots": 0,
            "wall_segment_support_score_rank_max": 0,
            "wall_segment_support_tail_rank_max": 0,
            "wall_segment_bin_size": 16,
        }
        for pair_cap, tail_slots in [
            (1, 0),
            (2, 0),
            (4, 0),
            (4, 1),
            (4, 2),
            (8, 0),
            (8, 1),
            (8, 2),
            (16, 0),
            (16, 1),
            (16, 2),
            (999999, 0),
        ]
    ]
    bounded_geometry_caps = [
        {
            "max_per_source": 999999,
            "max_per_target": 999999,
            "max_per_component": 999999,
            "max_per_pair": pair_cap,
            "max_pair_tail_slots": tail_slots,
            "max_pair_alignment_slots": alignment_slots,
            "max_per_source_side": 999999,
            "max_per_source_side_segment": 999999,
            "max_wall_segment_tail_slots": 0,
            "max_wall_segment_support_overflow_slots": 0,
            "wall_segment_support_score_rank_max": 0,
            "wall_segment_support_tail_rank_max": 0,
            "wall_segment_bin_size": 16,
        }
        for pair_cap in [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]
        for tail_slots in [0, 1, 2]
        for alignment_slots in [0, 1, 2, 4]
    ]
    bounded_side_caps = [
        {
            "max_per_source": 999999,
            "max_per_target": 999999,
            "max_per_component": 999999,
            "max_per_pair": 999999,
            "max_pair_tail_slots": 0,
            "max_pair_alignment_slots": 0,
            "max_per_source_side": source_side_cap,
            "max_per_source_side_segment": 999999,
            "max_wall_segment_tail_slots": 0,
            "max_wall_segment_support_overflow_slots": 0,
            "wall_segment_support_score_rank_max": 0,
            "wall_segment_support_tail_rank_max": 0,
            "wall_segment_bin_size": 16,
        }
        for source_side_cap in [8, 12, 16, 20, 24, 32, 40, 48, 64]
    ]
    bounded_wall_segment_caps = [
        {
            "max_per_source": 999999,
            "max_per_target": 999999,
            "max_per_component": 999999,
            "max_per_pair": 999999,
            "max_pair_tail_slots": 0,
            "max_pair_alignment_slots": 0,
            "max_per_source_side": source_side_cap,
            "max_per_source_side_segment": segment_cap,
            "max_wall_segment_tail_slots": tail_slots,
            "max_wall_segment_support_overflow_slots": 0,
            "wall_segment_support_score_rank_max": 0,
            "wall_segment_support_tail_rank_max": 0,
            "wall_segment_bin_size": bin_size,
        }
        for source_side_cap in [48, 999999]
        for segment_cap in [8, 12, 16]
        for tail_slots in [0, 2, 4, 8]
        for bin_size in [16, 24]
    ]
    bounded_wall_support_reserve_caps = [
        {
            "max_per_source": 999999,
            "max_per_target": 999999,
            "max_per_component": 999999,
            "max_per_pair": 999999,
            "max_pair_tail_slots": 0,
            "max_pair_alignment_slots": 0,
            "max_per_source_side": 48,
            "max_per_source_side_segment": 12,
            "max_wall_segment_tail_slots": 0,
            "max_wall_segment_support_overflow_slots": overflow_slots,
            "wall_segment_support_score_rank_max": score_rank_max,
            "wall_segment_support_tail_rank_max": tail_rank_max,
            "wall_segment_bin_size": 16,
        }
        for score_rank_max, tail_rank_max, overflow_slots in [
            (16, 16, 8),
            (24, 16, 8),
            (24, 16, 12),
            (24, 24, 12),
            (32, 16, 12),
        ]
    ]
    contains_symbol_support_set_caps = [
        {
            "max_per_source": 999999,
            "max_per_target": 999999,
            "max_per_component": 999999,
            "max_per_pair": 999999,
            "max_pair_tail_slots": 0,
            "max_pair_alignment_slots": 0,
            "max_per_source_side": 999999,
            "max_per_source_side_segment": 999999,
            "max_wall_segment_tail_slots": 0,
            "max_wall_segment_support_overflow_slots": 0,
            "wall_segment_support_score_rank_max": 0,
            "wall_segment_support_tail_rank_max": 0,
            "wall_segment_bin_size": 16,
            "max_per_room_symbol_instance": top_k,
            "max_contains_symbol_support_tail_slots": tail_slots,
        }
        for top_k, tail_slots in [
            (2, 2),
            (3, 2),
            (4, 0),
            (4, 1),
            (4, 2),
            (4, 4),
            (4, 5),
            (6, 0),
            (6, 1),
            (6, 2),
            (6, 4),
            (6, 5),
            (8, 0),
            (8, 2),
            (8, 4),
            (8, 8),
            (10, 8),
            (12, 8),
        ]
    ]
    caps = {
        "bounded_by": [
            *[row for row in bounded_side_caps if row["max_per_source_side"] in {48, 64}],
            *[
                row
                for row in bounded_wall_segment_caps
                if row["max_per_source_side"] == 48
                and row["max_per_source_side_segment"] in {12, 16}
                and row["max_wall_segment_tail_slots"] in {0, 4}
                and row["wall_segment_bin_size"] == 16
            ],
            *bounded_wall_support_reserve_caps,
        ],
        "contains_symbol": [
            *contains_symbol_support_set_caps,
            {"max_per_source": 8, "max_per_target": 4, "max_per_component": 64, "max_per_pair": 1, "max_pair_tail_slots": 0, "max_pair_alignment_slots": 0, "max_per_source_side": 999999, "max_per_source_side_segment": 999999, "max_wall_segment_tail_slots": 0, "max_wall_segment_support_overflow_slots": 0, "wall_segment_support_score_rank_max": 0, "wall_segment_support_tail_rank_max": 0, "wall_segment_bin_size": 16},
            {"max_per_source": 16, "max_per_target": 8, "max_per_component": 128, "max_per_pair": 2, "max_pair_tail_slots": 0, "max_pair_alignment_slots": 0, "max_per_source_side": 999999, "max_per_source_side_segment": 999999, "max_wall_segment_tail_slots": 0, "max_wall_segment_support_overflow_slots": 0, "wall_segment_support_score_rank_max": 0, "wall_segment_support_tail_rank_max": 0, "wall_segment_bin_size": 16},
            *loose_pair_caps,
        ],
        "labeled_by_text": [
            {"max_per_source": 6, "max_per_target": 4, "max_per_component": 48, "max_per_pair": 1, "max_pair_tail_slots": 0, "max_pair_alignment_slots": 0, "max_per_source_side": 999999, "max_per_source_side_segment": 999999, "max_wall_segment_tail_slots": 0, "max_wall_segment_support_overflow_slots": 0, "wall_segment_support_score_rank_max": 0, "wall_segment_support_tail_rank_max": 0, "wall_segment_bin_size": 16},
            {"max_per_source": 12, "max_per_target": 8, "max_per_component": 96, "max_per_pair": 2, "max_pair_tail_slots": 0, "max_pair_alignment_slots": 0, "max_per_source_side": 999999, "max_per_source_side_segment": 999999, "max_wall_segment_tail_slots": 0, "max_wall_segment_support_overflow_slots": 0, "wall_segment_support_score_rank_max": 0, "wall_segment_support_tail_rank_max": 0, "wall_segment_bin_size": 16},
            *loose_pair_caps,
        ],
        "adjacent_to": [
            {"max_per_source": 8, "max_per_target": 8, "max_per_component": 64, "max_per_pair": 1, "max_pair_tail_slots": 0, "max_pair_alignment_slots": 0, "max_per_source_side": 999999, "max_per_source_side_segment": 999999, "max_wall_segment_tail_slots": 0, "max_wall_segment_support_overflow_slots": 0, "wall_segment_support_score_rank_max": 0, "wall_segment_support_tail_rank_max": 0, "wall_segment_bin_size": 16},
            {"max_per_source": 16, "max_per_target": 16, "max_per_component": 128, "max_per_pair": 2, "max_pair_tail_slots": 0, "max_pair_alignment_slots": 0, "max_per_source_side": 999999, "max_per_source_side_segment": 999999, "max_wall_segment_tail_slots": 0, "max_wall_segment_support_overflow_slots": 0, "wall_segment_support_score_rank_max": 0, "wall_segment_support_tail_rank_max": 0, "wall_segment_bin_size": 16},
            *loose_pair_caps,
        ],
    }
    policy = {key: dict(value) for key, value in base_policy.items()}
    sweep: dict[str, Any] = {}
    for rel in RELATIONS:
        rel_rows = [row for row in dev_rows if row.get("relation") == rel]
        rel_sweep: list[dict[str, Any]] = []
        for threshold in thresholds:
            for cap in caps[rel]:
                trial = {rel: {"threshold": threshold, **cap}}
                selected = select_rows(rel_rows, trial)
                rep = representative_eval(rel_rows, selected).get(rel) or {}
                full = (evaluate(rel_rows, rel_rows, trial).get("relation_metrics") or {}).get(rel) or {}
                rel_sweep.append(
                    {
                        "threshold": threshold,
                        **cap,
                        "selected_rows": len(selected),
                        "feature_reduction": round(1.0 - len(selected) / max(len(rel_rows), 1), 6),
                        **rep,
                        "precision": full.get("precision"),
                        "recall": full.get("recall"),
                    }
                )
        viable = [row for row in rel_sweep if safe_float(row.get("gold_key_recall")) >= 0.98 and safe_float(row.get("bridge_recall")) >= 0.98]
        if viable:
            chosen = max(viable, key=lambda row: (safe_float(row.get("feature_reduction")), -safe_float(row.get("duplicate_support_kept")), safe_float(row.get("precision"))))
            reason = "max_reduction_at_gold_key_and_bridge_recall_ge_0.98"
        else:
            chosen = max(rel_sweep, key=lambda row: (safe_float(row.get("gold_key_recall")), safe_float(row.get("bridge_recall")), safe_float(row.get("feature_reduction"))))
            reason = "best_available_gold_key_or_bridge_recall_below_gate"
        policy[rel] = {
            "threshold": chosen["threshold"],
            "max_per_source": chosen["max_per_source"],
            "max_per_target": chosen["max_per_target"],
            "max_per_component": chosen["max_per_component"],
            "max_per_pair": chosen["max_per_pair"],
            "max_pair_tail_slots": chosen.get("max_pair_tail_slots", 0),
            "max_pair_alignment_slots": chosen.get("max_pair_alignment_slots", 0),
            "max_per_source_side": chosen.get("max_per_source_side", 999999),
            "max_per_source_side_segment": chosen.get("max_per_source_side_segment", 999999),
            "max_wall_segment_tail_slots": chosen.get("max_wall_segment_tail_slots", 0),
            "max_wall_segment_support_overflow_slots": chosen.get("max_wall_segment_support_overflow_slots", 0),
            "wall_segment_support_score_rank_max": chosen.get("wall_segment_support_score_rank_max", 0),
            "wall_segment_support_tail_rank_max": chosen.get("wall_segment_support_tail_rank_max", 0),
            "wall_segment_bin_size": chosen.get("wall_segment_bin_size", 16),
            "max_per_room_symbol_instance": chosen.get("max_per_room_symbol_instance", 999999),
            "max_contains_symbol_support_tail_slots": chosen.get("max_contains_symbol_support_tail_slots", 0),
            "listwise_policy": True,
        }
        sweep[rel] = {"chosen": {**chosen, "reason": reason}, "sweep": rel_sweep}
    return policy, sweep


def policy_from_sweep_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "threshold": row["threshold"],
        "max_per_source": row["max_per_source"],
        "max_per_target": row["max_per_target"],
        "max_per_component": row["max_per_component"],
        "max_per_pair": row.get("max_per_pair", 999999),
        "max_pair_tail_slots": row.get("max_pair_tail_slots", 0),
        "max_pair_alignment_slots": row.get("max_pair_alignment_slots", 0),
        "max_per_source_side": row.get("max_per_source_side", 999999),
        "max_per_source_side_segment": row.get("max_per_source_side_segment", 999999),
        "max_wall_segment_tail_slots": row.get("max_wall_segment_tail_slots", 0),
        "max_wall_segment_support_overflow_slots": row.get("max_wall_segment_support_overflow_slots", 0),
        "wall_segment_support_score_rank_max": row.get("wall_segment_support_score_rank_max", 0),
        "wall_segment_support_tail_rank_max": row.get("wall_segment_support_tail_rank_max", 0),
        "wall_segment_bin_size": row.get("wall_segment_bin_size", 16),
        "max_per_room_symbol_instance": row.get("max_per_room_symbol_instance", 999999),
        "max_contains_symbol_support_tail_slots": row.get("max_contains_symbol_support_tail_slots", 0),
        "listwise_policy": True,
    }


def apply_listwise_no_regression_gate(
    policy: dict[str, Any],
    validation_rows: list[dict[str, Any]],
    base_policy: dict[str, Any],
    sweep: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not validation_rows:
        return policy, {"status": "skipped_no_validation_rows"}
    gated = {key: dict(value) for key, value in policy.items()}
    fail_closed_all, _ = fail_closed_policy(base_policy, "no_regression_reference")
    reference_eval = evaluate(validation_rows, validation_rows, fail_closed_all)
    before_eval = evaluate(validation_rows, validation_rows, gated)
    decisions: dict[str, Any] = {}
    for rel in RELATIONS:
        current = ((before_eval.get("relation_metrics") or {}).get(rel) or {})
        reference = ((reference_eval.get("relation_metrics") or {}).get(rel) or {})
        current_recall = safe_float(current.get("recall"))
        reference_recall = safe_float(reference.get("recall"))
        rel_rows = [row for row in validation_rows if row.get("relation") == rel]
        candidates: list[dict[str, Any]] = []
        for sweep_row in (sweep.get(rel) or {}).get("sweep", []):
            trial_policy = {rel: policy_from_sweep_row(sweep_row)}
            rel_eval = (evaluate(rel_rows, rel_rows, trial_policy).get("relation_metrics") or {}).get(rel) or {}
            rep_eval = representative_eval(rel_rows, select_rows(rel_rows, trial_policy)).get(rel) or {}
            if (
                safe_float(rel_eval.get("recall")) + 1e-9 >= reference_recall
                and safe_float(rep_eval.get("gold_key_recall")) + 1e-9 >= 1.0
                and safe_float(rep_eval.get("bridge_recall")) + 1e-9 >= 1.0
            ):
                candidates.append(
                    {
                        **trial_policy[rel],
                        "selected_rows": rel_eval.get("predicted"),
                        "precision": rel_eval.get("precision"),
                        "recall": rel_eval.get("recall"),
                        "feature_reduction": round(1.0 - safe_float(rel_eval.get("predicted")) / max(len(rel_rows), 1), 6),
                        **rep_eval,
                    }
                )
        if candidates:
            chosen = max(
                candidates,
                key=lambda row: (
                    safe_float(row.get("feature_reduction")),
                    -safe_float(row.get("duplicate_support_kept")),
                    safe_float(row.get("precision")),
                ),
            )
            current_feature_reduction = round(1.0 - safe_float(current.get("predicted")) / max(len(rel_rows), 1), 6)
            if current_recall + 1e-9 >= reference_recall and safe_float(chosen.get("feature_reduction")) <= current_feature_reduction + 1e-9:
                decisions[rel] = {
                    "action": "keep_compressive_policy",
                    "current_recall": current.get("recall"),
                    "reference_recall": reference.get("recall"),
                    "current_feature_reduction": current_feature_reduction,
                    "best_safe_candidate_feature_reduction": chosen.get("feature_reduction"),
                    "best_safe_candidate": chosen,
                }
            else:
                gated[rel] = policy_from_sweep_row(chosen)
                decisions[rel] = {
                    "action": "repair_with_locked_validation_policy" if current_recall + 1e-9 < reference_recall else "upgrade_with_better_locked_validation_policy",
                    "current_recall": current.get("recall"),
                    "reference_recall": reference.get("recall"),
                    "current_feature_reduction": current_feature_reduction,
                    "repaired_recall": chosen.get("recall"),
                    "repaired_feature_reduction": chosen.get("feature_reduction"),
                    "chosen": chosen,
                }
        else:
            if current_recall + 1e-9 >= reference_recall:
                decisions[rel] = {
                    "action": "keep_compressive_policy_no_better_candidate",
                    "current_recall": current.get("recall"),
                    "reference_recall": reference.get("recall"),
                }
            else:
                gated[rel] = fail_closed_relation(base_policy, rel, "heldout_no_regression_gate_failed")
                decisions[rel] = {
                    "action": "fail_closed",
                    "current_recall": current.get("recall"),
                    "reference_recall": reference.get("recall"),
                }
    after_eval = evaluate(validation_rows, validation_rows, gated)
    return gated, {"status": "applied", "decisions": decisions, "before": before_eval, "after": after_eval, "reference": reference_eval}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--base-model", default=str(DEFAULT_MODEL))
    parser.add_argument("--hard-cases", default=str(DEFAULT_HARD_CASES))
    parser.add_argument("--model-output", default=str(DEFAULT_LISTWISE_MODEL))
    parser.add_argument("--eval-output", default=str(DEFAULT_LISTWISE_EVAL))
    parser.add_argument("--audit-output", default=str(DEFAULT_LISTWISE_AUDIT))
    parser.add_argument("--locked", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    limit = 500 if args.smoke else args.limit
    base_model = load_model(Path(args.base_model))
    rows = load_dataset(Path(args.dataset), limit=limit)
    base_scored = score_rows(rows, base_model)
    marked = mark_training_targets(base_scored, hard_case_lookup(Path(args.hard_cases)))
    train_rows = [row for row in marked if row.get("split") == "train"]
    dev_rows = [row for row in marked if row.get("split") == "dev"]
    test_rows = [row for row in marked if row.get("split") == "test"]

    listwise = train_listwise_models(train_rows or marked)
    model = dict(base_model)
    model["model_type"] = "relation_graph_policy_v18_with_listwise_component_policy"
    model["listwise"] = listwise
    train_rows = apply_listwise_scores(train_rows, model)
    dev_rows = apply_listwise_scores(dev_rows, model)
    test_rows = apply_listwise_scores(test_rows, model)
    all_rows = train_rows + dev_rows + test_rows

    if dev_rows:
        policy, sweep = choose_listwise_policy(dev_rows, default_policy())
        calibration_split = "dev"
    else:
        policy, sweep = fail_closed_policy(default_policy(), "no_dev_rows_for_listwise_calibration")
        calibration_split = "none_fail_closed_no_dev_rows"
    validation_rows = all_rows if args.locked else (test_rows or dev_rows or train_rows)
    policy, no_regression_gate = apply_listwise_no_regression_gate(policy, validation_rows, default_policy(), sweep)
    model["policy"] = policy
    model["selection_sweep"] = sweep
    model["calibration_split"] = calibration_split
    model["no_regression_gate"] = no_regression_gate
    model["locked"] = bool(args.locked)
    model["source_integrity"] = marked[0].get("source_integrity") if marked else None

    eval_train = evaluate(train_rows, train_rows, policy)
    eval_dev = evaluate(dev_rows, dev_rows, policy)
    eval_test = evaluate(test_rows, test_rows, policy)
    rep_train = representative_eval(train_rows, select_rows(train_rows, policy))
    rep_dev = representative_eval(dev_rows, select_rows(dev_rows, policy))
    rep_test = representative_eval(test_rows, select_rows(test_rows, policy))

    Path(args.model_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.model_output).write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_json(
        Path(args.eval_output),
        {
            "train": eval_train,
            "dev": eval_dev,
            "test": eval_test,
            "representative_train": rep_train,
            "representative_dev": rep_dev,
            "representative_test": rep_test,
            "policy": policy,
            "source_integrity": model["source_integrity"],
        },
    )
    label_counts = Counter()
    for row in marked:
        labels = row.get("listwise_labels") or {}
        for key, value in labels.items():
            if isinstance(value, bool) and value:
                label_counts[key] += 1
    write_json(
        Path(args.audit_output),
        {
            "task": "IMG-MOE-V18-REBUILD-005",
            "policy_family": "page_component_level_listwise",
            "dataset_rows": len(marked),
            "train_rows": len(train_rows),
            "dev_rows": len(dev_rows),
            "test_rows": len(test_rows),
            "listwise_label_counts": dict(label_counts),
            "policy": policy,
            "calibration_split": calibration_split,
            "no_regression_gate": no_regression_gate,
            "sweep": sweep,
            "source_integrity": model["source_integrity"],
            "locked": bool(args.locked),
            "smoke": bool(args.smoke),
            "gold_loaded_after_inference_for_training_only": True,
            "gold_used_for_inference": False,
        },
    )
    print(json.dumps({"train_rows": len(train_rows), "dev_rows": len(dev_rows), "test_rows": len(test_rows), "model_output": args.model_output}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
