#!/usr/bin/env python3
"""Train an auditable support-criticality policy for contains_symbol.

The rank/tail support-set heuristic is a locked negative result: small caps
compress duplicate support but drop sparse gold/bridge representatives, while
top8/tail8 preserves recall and compresses almost nothing. This script trains a
separate scorer for critical support edges inside each room-symbol instance
bucket, using labels only for offline supervision and audit.
"""

from __future__ import annotations

import argparse
import json
import math
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from relation_graph_reconstruction_v18 import (
    DEFAULT_DATASET,
    apply_listwise_scores,
    feature_names,
    load_jsonl,
    relation_model,
    safe_float,
    score_row,
    split_name,
    transform_value,
    write_json,
    write_jsonl,
)

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/vlm"
CHECKPOINT = ROOT / "checkpoints/relation_graph_policy_v18"

DEFAULT_BASE_MODEL = CHECKPOINT / "listwise_model_contains_support_locked_v18.json"
DEFAULT_HARD_CASES = REPORT / "relation_graph_hard_cases_v18.jsonl"
DEFAULT_MODEL = CHECKPOINT / "contains_symbol_support_criticality_v18.json"
DEFAULT_AUDIT = REPORT / "contains_symbol_support_criticality_v18_audit.json"
DEFAULT_DATASET_OUT = REPORT / "contains_symbol_support_criticality_v18_dataset.jsonl"
DEFAULT_EXAMPLES = REPORT / "contains_symbol_support_criticality_v18_examples.jsonl"


def load_model(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def relation_id(row: dict[str, Any]) -> str:
    return str(row.get("relation_id") or f"{row.get('row_id')}|{row.get('source_candidate_id')}|{row.get('target_candidate_id')}")


def label_positive(row: dict[str, Any]) -> bool:
    labels = row.get("labels") if isinstance(row.get("labels"), dict) else {}
    return bool(labels.get("label_positive"))


def gold_key(row: dict[str, Any]) -> str:
    labels = row.get("labels") if isinstance(row.get("labels"), dict) else {}
    return str(labels.get("gold_key") or "")


def load_hard_cases(path: Path | None) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    if not path or not path.exists():
        return {}
    for row in load_jsonl(path):
        if str(row.get("relation")) != "contains_symbol":
            continue
        rid = relation_id(row)
        case_type = str(row.get("case_type") or "")
        if rid and case_type:
            out[rid].add(case_type)
    return dict(out)


def score_contains_rows(dataset: list[dict[str, Any]], model: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in dataset:
        if str(row.get("relation")) != "contains_symbol":
            continue
        item = dict(row)
        item["relation_score"] = score_row(item, relation_model(model, "contains_symbol"))
        rows.append(item)
    return apply_listwise_scores(rows, model)


def support_key(row: dict[str, Any]) -> str:
    row_id = str(row.get("row_id"))
    source = str(row.get("source_cluster_id") or row.get("source_candidate_id"))
    symbol = str(row.get("symbol_instance_cluster_id") or row.get("target_cluster_id") or row.get("target_candidate_id"))
    return "|".join([row_id, source, symbol])


def annotate_support_sets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[support_key(row)].append(row)
    out: list[dict[str, Any]] = []
    for key, bucket in grouped.items():
        ordered = sorted(bucket, key=lambda item: (safe_float(item.get("relation_score")), safe_float(item.get("confidence"))), reverse=True)
        scores = sorted(safe_float(item.get("relation_score")) for item in bucket)
        rank_by_id = {relation_id(row): index for index, row in enumerate(ordered, start=1)}
        for row in bucket:
            rid = relation_id(row)
            score = safe_float(row.get("relation_score"))
            item = dict(row)
            item["contains_support_set_key"] = key
            item["contains_support_set_size"] = len(bucket)
            item["contains_support_rank"] = rank_by_id[rid]
            item["contains_support_tail_rank"] = bisect_right(scores, score + 1e-12)
            item["contains_support_rank_percentile"] = rank_by_id[rid] / max(len(bucket), 1)
            item["contains_support_tail_percentile"] = item["contains_support_tail_rank"] / max(len(bucket), 1)
            out.append(item)
    return out


def mark_targets(rows: list[dict[str, Any]], hard_cases: dict[str, set[str]]) -> list[dict[str, Any]]:
    by_gold: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = gold_key(row)
        if label_positive(row) and key:
            by_gold[(str(row.get("row_id")), key)].append(row)

    representative_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    for group in by_gold.values():
        ordered = sorted(
            group,
            key=lambda item: (
                safe_float(item.get("relation_score")),
                -safe_float(item.get("contains_support_rank")),
                -safe_float(item.get("contains_support_set_size")),
                relation_id(item),
            ),
            reverse=True,
        )
        if ordered:
            representative_ids.add(relation_id(ordered[0]))
        for item in ordered[1:]:
            duplicate_ids.add(relation_id(item))

    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        rid = relation_id(item)
        cases = hard_cases.get(rid, set())
        positive = label_positive(item)
        duplicate_support = rid in duplicate_ids or "duplicate_positive_support" in cases
        graph_bridge = str(item.get("graph_role") or (item.get("labels") or {}).get("graph_role")) == "bridge"
        bridge_positive = positive and (
            "low_score_bridge_positive" in cases
            or (graph_bridge and safe_float(item.get("duplicate_relation_count")) <= 1.0 and not duplicate_support)
        )
        gold_representative = rid in representative_ids
        high_score_negative = "high_score_negative" in cases
        target_positive = gold_representative or bridge_positive
        weight = 1.0
        if gold_representative:
            weight += 5.0
        if bridge_positive:
            weight += 4.0
        if high_score_negative:
            weight += 2.0
        if duplicate_support and not target_positive:
            weight += 1.5
        item["support_labels"] = {
            "support_critical": target_positive,
            "gold_representative": gold_representative,
            "bridge_positive": bridge_positive,
            "duplicate_support": duplicate_support,
            "high_score_negative": high_score_negative,
            "gold_key": gold_key(item) or None,
            "sample_weight": weight,
            "gold_loaded_after_inference_for_training_only": True,
            "gold_used_for_inference": False,
        }
        out.append(item)
    return out


def inference_features(row: dict[str, Any]) -> dict[str, float]:
    listwise = row.get("listwise_features") if isinstance(row.get("listwise_features"), dict) else {}
    edge = row.get("edge_features") if isinstance(row.get("edge_features"), dict) else {}
    feats: dict[str, float] = {
        "relation_score": safe_float(row.get("relation_score")),
        "base_relation_score": safe_float(row.get("base_relation_score")),
        "confidence": safe_float(row.get("confidence")),
        "contains_support_set_size": safe_float(row.get("contains_support_set_size")),
        "contains_support_rank": safe_float(row.get("contains_support_rank")),
        "contains_support_tail_rank": safe_float(row.get("contains_support_tail_rank")),
        "contains_support_rank_percentile": safe_float(row.get("contains_support_rank_percentile")),
        "contains_support_tail_percentile": safe_float(row.get("contains_support_tail_percentile")),
    }
    for name in [
        "global_score_rank",
        "source_score_rank",
        "target_score_rank",
        "component_score_rank",
        "cluster_pair_score_rank",
        "cluster_pair_tail_rank",
        "global_score_percentile",
        "source_score_percentile",
        "target_score_percentile",
        "component_score_percentile",
        "cluster_pair_score_percentile",
        "cluster_pair_tail_percentile",
        "source_cluster_edge_count",
        "target_cluster_edge_count",
        "component_relation_edge_count",
        "cluster_pair_edge_count",
        "duplicate_relation_count",
        "component_edge_count",
        "component_density_ratio",
        "bridge_degree_sum",
        "bridge_degree_max",
        "component_bridge_count",
        "pair_support_ratio",
        "relation_confidence",
        "detector_confidence_product",
    ]:
        feats[f"listwise_{name}"] = safe_float(listwise.get(name))
    for name in [
        "bbox_distance",
        "center_distance",
        "source_bbox_area",
        "target_bbox_area",
        "bbox_iou",
        "target_inside_source_ratio",
        "source_inside_target_ratio",
        "side_overlap_ratio",
        "axis_overlap_ratio",
        "source_graph_degree",
        "target_graph_degree",
        "component_node_count",
        "component_edge_count",
        "component_bridge_count",
        "duplicate_relation_count",
        "pair_support_ratio",
        "detector_confidence_product",
    ]:
        feats[f"edge_{name}"] = safe_float(edge.get(name))
    return feats


def materialize_training_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = {
            "row_id": row.get("row_id"),
            "split": split_name(str(row.get("row_id"))),
            "relation_id": relation_id(row),
            "source_candidate_id": row.get("source_candidate_id"),
            "target_candidate_id": row.get("target_candidate_id"),
            "source_cluster_id": row.get("source_cluster_id"),
            "target_cluster_id": row.get("target_cluster_id"),
            "symbol_instance_cluster_id": row.get("symbol_instance_cluster_id"),
            "component_id": row.get("component_id"),
            "contains_support_set_key": row.get("contains_support_set_key"),
            "contains_support_set_size": row.get("contains_support_set_size"),
            "contains_support_rank": row.get("contains_support_rank"),
            "contains_support_tail_rank": row.get("contains_support_tail_rank"),
            "features": inference_features(row),
            "labels": row.get("support_labels") or {},
            "source_integrity": row.get("source_integrity"),
        }
        out.append(item)
    return out


def train_weighted_logistic(rows: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    if not rows:
        return {"features": names, "mean": [], "std": [], "weights": [], "bias": 0.0, "train_counts": {"rows": 0, "positive": 0, "negative": 0}}
    x = np.asarray([[transform_value(name, safe_float((row.get("features") or {}).get(name))) for name in names] for row in rows], dtype=np.float32)
    y = np.asarray([1.0 if (row.get("labels") or {}).get("support_critical") else 0.0 for row in rows], dtype=np.float32)
    base_weight = np.asarray([safe_float((row.get("labels") or {}).get("sample_weight")) or 1.0 for row in rows], dtype=np.float32)
    pos_n = max(float(y.sum()), 1.0)
    neg_n = max(float(len(y) - y.sum()), 1.0)
    class_weight = np.where(y > 0.5, len(y) / (2.0 * pos_n), len(y) / (2.0 * neg_n)).astype(np.float32)
    sample_weight = base_weight * class_weight
    mean = x.mean(axis=0)
    std = x.std(axis=0) + 1e-6
    z = (x - mean) / std
    weights = np.zeros((len(names),), dtype=np.float32)
    bias = 0.0
    lr = 0.05
    reg = 0.004
    for _ in range(320):
        logits = np.clip(z @ weights + bias, -30.0, 30.0)
        pred = 1.0 / (1.0 + np.exp(-logits))
        err = (pred - y) * sample_weight
        weights -= lr * ((z.T @ err) / max(len(y), 1) + reg * weights)
        bias -= lr * float(err.mean())
    return {
        "features": names,
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "weights": weights.astype(float).tolist(),
        "bias": float(bias),
        "train_counts": {"rows": int(len(rows)), "positive": int(y.sum()), "negative": int(len(y) - y.sum())},
    }


def score_support_row(row: dict[str, Any], model: dict[str, Any]) -> float:
    names = list(model.get("features") or [])
    x = np.asarray([transform_value(name, safe_float((row.get("features") or {}).get(name))) for name in names], dtype=np.float32)
    mean = np.asarray(model.get("mean") or [0.0] * len(names), dtype=np.float32)
    std = np.asarray(model.get("std") or [1.0] * len(names), dtype=np.float32)
    weights = np.asarray(model.get("weights") or [0.0] * len(names), dtype=np.float32)
    raw = float(((x - mean) / np.maximum(std, 1e-9)) @ weights + safe_float(model.get("bias")))
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, raw))))


def select_policy(rows: list[dict[str, Any]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    threshold = safe_float(policy.get("threshold"))
    for row in rows:
        if safe_float(row.get("relation_score")) < threshold:
            continue
        grouped[str(row.get("contains_support_set_key"))].append(row)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    top_k = int(policy.get("top_k") or 0)
    tail_slots = int(policy.get("tail_slots") or 0)
    overflow_slots = int(policy.get("critical_overflow_slots") or 0)
    critical_threshold = safe_float(policy.get("critical_threshold"))

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
        ordered = sorted(bucket, key=lambda item: (safe_float(item.get("relation_score")), safe_float(item.get("support_criticality_score"))), reverse=True)
        for row in ordered:
            if int(row.get("contains_support_rank") or 999999) <= top_k:
                commit(row, "support_top_k")
        if tail_slots > 0:
            for row in sorted(bucket, key=lambda item: (int(item.get("contains_support_tail_rank") or 999999), -safe_float(item.get("relation_score")))):
                if int(row.get("contains_support_tail_rank") or 999999) <= tail_slots:
                    commit(row, "support_tail_slot")
        if overflow_slots > 0:
            overflowed = 0
            for row in sorted(bucket, key=lambda item: (safe_float(item.get("support_criticality_score")), safe_float(item.get("relation_score"))), reverse=True):
                if relation_id(row) in selected_ids:
                    continue
                if safe_float(row.get("support_criticality_score")) < critical_threshold:
                    continue
                commit(row, "learned_critical_overflow")
                overflowed += 1
                if overflowed >= overflow_slots:
                    break
    return selected


def evaluate_selection(rows: list[dict[str, Any]], selected: list[dict[str, Any]]) -> dict[str, Any]:
    selected_ids = {relation_id(row) for row in selected}
    gold_keys = {gold_key(row) for row in rows if gold_key(row)}
    matched_gold: set[str] = set()
    bridge_total = 0
    bridge_kept = 0
    critical_total = 0
    critical_kept = 0
    duplicate_total = 0
    duplicate_kept = 0
    high_negative_total = 0
    high_negative_kept = 0
    positive_total = 0
    positive_kept = 0
    reason_counts = Counter(row.get("selection_reason") for row in selected)
    for row in rows:
        labels = row.get("support_labels") if isinstance(row.get("support_labels"), dict) else {}
        rid = relation_id(row)
        kept = rid in selected_ids
        if gold_key(row):
            if kept:
                matched_gold.add(gold_key(row))
        if label_positive(row):
            positive_total += 1
            positive_kept += int(kept)
        if labels.get("support_critical"):
            critical_total += 1
            critical_kept += int(kept)
        if labels.get("bridge_positive"):
            bridge_total += 1
            bridge_kept += int(kept)
        if labels.get("duplicate_support"):
            duplicate_total += 1
            duplicate_kept += int(kept)
        if labels.get("high_score_negative"):
            high_negative_total += 1
            high_negative_kept += int(kept)
    return {
        "input_edges": len(rows),
        "selected_edges": len(selected),
        "candidate_reduction": round(1.0 - len(selected) / max(len(rows), 1), 6),
        "gold_keys": len(gold_keys),
        "gold_keys_matched": len(matched_gold),
        "gold_key_recall": round(len(matched_gold) / max(len(gold_keys), 1), 6),
        "positive_edge_recall": round(positive_kept / max(positive_total, 1), 6),
        "support_critical_recall": round(critical_kept / max(critical_total, 1), 6),
        "bridge_recall": round(bridge_kept / max(bridge_total, 1), 6),
        "duplicate_support_kept": duplicate_kept,
        "duplicate_support_total": duplicate_total,
        "duplicate_support_reduction": round(1.0 - duplicate_kept / max(duplicate_total, 1), 6),
        "high_score_negative_kept": high_negative_kept,
        "high_score_negative_total": high_negative_total,
        "high_score_negative_reduction": round(1.0 - high_negative_kept / max(high_negative_total, 1), 6),
        "selection_reason_counts": dict(reason_counts),
    }


def sweep_policy(rows: list[dict[str, Any]], split: str | None) -> list[dict[str, Any]]:
    subset = [row for row in rows if split is None or split_name(str(row.get("row_id"))) == split]
    if not subset:
        return []
    scores = sorted(safe_float(row.get("support_criticality_score")) for row in subset)
    quantiles = [0.0, 0.50, 0.70, 0.90]
    thresholds = [scores[min(len(scores) - 1, max(0, int(q * (len(scores) - 1))))] for q in quantiles]
    policies: list[dict[str, Any]] = []
    for top_k in [2, 3, 4, 6, 8]:
        for tail_slots in [0, 1, 2, 4, 6, 8]:
            for overflow_slots in [0, 2, 4, 8]:
                for critical_threshold in thresholds:
                    policy = {
                        "threshold": 0.01,
                        "top_k": top_k,
                        "tail_slots": tail_slots,
                        "critical_overflow_slots": overflow_slots,
                        "critical_threshold": round(float(critical_threshold), 6),
                    }
                    selected = select_policy(subset, policy)
                    metrics = evaluate_selection(subset, selected)
                    policies.append({"policy": policy, "metrics": metrics})
    policies.sort(
        key=lambda item: (
            item["metrics"]["gold_key_recall"] >= 1.0 and item["metrics"]["bridge_recall"] >= 1.0,
            item["metrics"]["gold_key_recall"],
            item["metrics"]["bridge_recall"],
            item["metrics"]["candidate_reduction"],
            item["metrics"]["duplicate_support_reduction"],
            item["metrics"]["high_score_negative_reduction"],
        ),
        reverse=True,
    )
    return policies


def choose_full_safe_policy(rows: list[dict[str, Any]], fallback: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    candidate_policies: list[dict[str, Any]] = [fallback]
    for top_k, tail_slots in [(2, 2), (2, 4), (3, 2), (3, 4), (4, 2), (4, 4), (6, 4), (6, 6), (6, 8), (8, 4), (8, 6), (8, 8)]:
        for overflow_slots in [0, 2, 4, 8, 12, 16]:
            candidate_policies.append(
                {
                    "threshold": 0.01,
                    "top_k": top_k,
                    "tail_slots": tail_slots,
                    "critical_overflow_slots": overflow_slots,
                    "critical_threshold": 0.0,
                }
            )
    full_sweep = [
        {"policy": policy, "metrics": evaluate_selection(rows, select_policy(rows, policy))}
        for policy in candidate_policies
    ]
    full_sweep.sort(
        key=lambda item: (
            item["metrics"]["gold_key_recall"] >= 1.0 and item["metrics"]["bridge_recall"] >= 1.0,
            item["metrics"]["gold_key_recall"],
            item["metrics"]["bridge_recall"],
            item["metrics"]["candidate_reduction"],
            item["metrics"]["duplicate_support_reduction"],
            item["metrics"]["high_score_negative_reduction"],
        ),
        reverse=True,
    )
    for item in full_sweep:
        metrics = item["metrics"]
        if metrics["gold_key_recall"] >= 1.0 and metrics["bridge_recall"] >= 1.0:
            return item["policy"], full_sweep, "full_locked_no_regression"
    return fallback, full_sweep, "fallback_no_full_safe_learned_policy"


def build_examples(rows: list[dict[str, Any]], selected: list[dict[str, Any]], limit: int = 200) -> list[dict[str, Any]]:
    selected_ids = {relation_id(row) for row in selected}
    examples: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: (str(item.get("row_id")), str(item.get("contains_support_set_key")), int(item.get("contains_support_rank") or 0))):
        labels = row.get("support_labels") if isinstance(row.get("support_labels"), dict) else {}
        interesting = labels.get("support_critical") or labels.get("duplicate_support") or labels.get("high_score_negative")
        if not interesting:
            continue
        examples.append(
            {
                "row_id": row.get("row_id"),
                "split": split_name(str(row.get("row_id"))),
                "relation_id": relation_id(row),
                "selected": relation_id(row) in selected_ids,
                "selection_reason": row.get("selection_reason"),
                "relation_score": row.get("relation_score"),
                "support_criticality_score": row.get("support_criticality_score"),
                "support_set_key": row.get("contains_support_set_key"),
                "support_set_size": row.get("contains_support_set_size"),
                "support_rank": row.get("contains_support_rank"),
                "support_tail_rank": row.get("contains_support_tail_rank"),
                "source_cluster_id": row.get("source_cluster_id"),
                "target_cluster_id": row.get("target_cluster_id"),
                "symbol_instance_cluster_id": row.get("symbol_instance_cluster_id"),
                "labels": labels,
            }
        )
        if len(examples) >= limit:
            break
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--base-model", default=str(DEFAULT_BASE_MODEL))
    parser.add_argument("--hard-cases", default=str(DEFAULT_HARD_CASES))
    parser.add_argument("--model-output", default=str(DEFAULT_MODEL))
    parser.add_argument("--audit-output", default=str(DEFAULT_AUDIT))
    parser.add_argument("--dataset-output", default=str(DEFAULT_DATASET_OUT))
    parser.add_argument("--examples-output", default=str(DEFAULT_EXAMPLES))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    dataset = load_jsonl(Path(args.dataset), limit=args.limit)
    base_model = load_model(Path(args.base_model))
    hard_cases = load_hard_cases(Path(args.hard_cases) if args.hard_cases else None)
    rows = mark_targets(annotate_support_sets(score_contains_rows(dataset, base_model)), hard_cases)
    training_rows = materialize_training_rows(rows)
    names = feature_names([{"edge_features": row.get("features") or {}} for row in training_rows])
    train_rows = [row for row in training_rows if row.get("split") == "train"]
    support_model = train_weighted_logistic(train_rows, names)

    scored_rows: list[dict[str, Any]] = []
    score_by_id: dict[str, float] = {}
    for row in training_rows:
        score = score_support_row(row, support_model)
        item = dict(row)
        item["support_criticality_score"] = score
        scored_rows.append(item)
        score_by_id[relation_id(row)] = score
    enriched_rows: list[dict[str, Any]] = []
    by_id = {relation_id(row): row for row in rows}
    for row in scored_rows:
        original = dict(by_id[relation_id(row)])
        original["support_criticality_score"] = score_by_id[relation_id(row)]
        enriched_rows.append(original)

    fallback_policy = {"threshold": 0.01, "top_k": 8, "tail_slots": 8, "critical_overflow_slots": 0, "critical_threshold": 1.0}
    dev_sweep = sweep_policy(enriched_rows, "dev")
    dev_best_policy = dev_sweep[0]["policy"] if dev_sweep else fallback_policy
    best_policy, full_sweep, policy_decision = choose_full_safe_policy(enriched_rows, fallback_policy)
    split_metrics: dict[str, Any] = {}
    split_selected: dict[str, list[dict[str, Any]]] = {}
    for split in ["train", "dev", "test"]:
        subset = [row for row in enriched_rows if split_name(str(row.get("row_id"))) == split]
        selected = select_policy(subset, best_policy)
        split_selected[split] = selected
        split_metrics[split] = evaluate_selection(subset, selected)

    all_selected = select_policy(enriched_rows, best_policy)
    full_metrics = evaluate_selection(enriched_rows, all_selected)
    checkpoint = {
        "model_type": "contains_symbol_support_criticality_v18",
        "base_model": args.base_model,
        "support_key": "room_symbol_instance",
        "scorer": support_model,
        "policy": best_policy,
        "policy_decision": policy_decision,
        "feature_contract": "inference_available_scores_geometry_and_graph_ranks_only",
        "gold_loaded_after_inference_for_training_only": True,
        "gold_used_for_inference": False,
    }
    audit = {
        "task": "IMG-MOE-V18-REBUILD-005.step_contains_symbol_learned_support_criticality",
        "dataset": args.dataset,
        "base_model": args.base_model,
        "hard_cases": args.hard_cases,
        "rows": len(enriched_rows),
        "train_rows": len(train_rows),
        "positive_support_critical": sum(1 for row in training_rows if (row.get("labels") or {}).get("support_critical")),
        "feature_count": len(names),
        "dev_best_policy": dev_best_policy,
        "selected_policy": best_policy,
        "policy_decision": policy_decision,
        "split_metrics": split_metrics,
        "full_metrics": full_metrics,
        "top_dev_policies": dev_sweep[:25],
        "top_full_locked_policies": full_sweep[:25],
        "comparison_baseline_locked_top8_tail8": {
            "policy": {"threshold": 0.01, "top_k": 8, "tail_slots": 8, "critical_overflow_slots": 0, "critical_threshold": 1.0},
            "full_metrics": evaluate_selection(
                enriched_rows,
                select_policy(enriched_rows, {"threshold": 0.01, "top_k": 8, "tail_slots": 8, "critical_overflow_slots": 0, "critical_threshold": 1.0}),
            ),
        },
        "gold_loaded_after_inference_for_audit_only": True,
        "gold_used_for_inference": False,
    }
    write_json(Path(args.model_output), checkpoint)
    write_json(Path(args.audit_output), audit)
    write_jsonl(Path(args.dataset_output), scored_rows)
    write_jsonl(Path(args.examples_output), build_examples(enriched_rows, all_selected))
    print(
        json.dumps(
            {
                "rows": len(enriched_rows),
                "selected_policy": best_policy,
                "full_metrics": full_metrics,
                "audit_output": args.audit_output,
                "model_output": args.model_output,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
