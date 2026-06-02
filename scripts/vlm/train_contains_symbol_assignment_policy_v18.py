#!/usr/bin/env python3
"""Train a symbol-centric assignment policy for contains_symbol.

This is intentionally different from support-set cap tuning. The policy treats
contains_symbol as a sparse assignment problem: for each detected symbol
instance, choose a small number of candidate rooms, then keep a small number of
support edges inside each chosen room-symbol bucket. Labels are used only
offline for training, policy selection, and audit.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from relation_graph_reconstruction_v18 import load_jsonl, safe_float, transform_value, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/vlm"
CHECKPOINT = ROOT / "checkpoints/relation_graph_policy_v18"

DEFAULT_DATASET = REPORT / "contains_symbol_visual_support_features_v18_dataset.jsonl"
DEFAULT_MODEL = CHECKPOINT / "contains_symbol_assignment_policy_v18.json"
DEFAULT_AUDIT = REPORT / "contains_symbol_assignment_policy_v18_audit.json"
DEFAULT_SCORED = REPORT / "contains_symbol_assignment_policy_v18_dataset.jsonl"
DEFAULT_EXAMPLES = REPORT / "contains_symbol_assignment_policy_v18_examples.jsonl"


def relation_id(row: dict[str, Any]) -> str:
    return str(row.get("relation_id") or f"{row.get('row_id')}|{row.get('source_candidate_id')}|{row.get('target_candidate_id')}")


def labels(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("labels") if isinstance(row.get("labels"), dict) else {}


def features(row: dict[str, Any]) -> dict[str, Any]:
    feats = dict(row.get("features") if isinstance(row.get("features"), dict) else {})
    if "prior_support_criticality_score" not in feats and row.get("support_criticality_score") is not None:
        feats["prior_support_criticality_score"] = safe_float(row.get("support_criticality_score"))
    return feats


def split_name(row: dict[str, Any]) -> str:
    return str(row.get("split") or "unknown")


def gold_key(row: dict[str, Any]) -> str:
    return str(labels(row).get("gold_key") or "")


def is_positive(row: dict[str, Any]) -> bool:
    return bool(gold_key(row))


def is_critical(row: dict[str, Any]) -> bool:
    lab = labels(row)
    return bool(lab.get("support_critical") or lab.get("gold_representative") or lab.get("bridge_positive"))


def symbol_key(row: dict[str, Any]) -> str:
    return "|".join([str(row.get("row_id")), str(row.get("symbol_instance_cluster_id") or row.get("target_cluster_id") or row.get("target_candidate_id"))])


def room_key(row: dict[str, Any]) -> str:
    return str(row.get("source_cluster_id") or row.get("source_candidate_id"))


def room_symbol_key(row: dict[str, Any]) -> str:
    return "|".join([symbol_key(row), room_key(row)])


def feature_names(rows: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for row in rows:
        names.update(str(name) for name in features(row).keys())
    return sorted(names)


def vector(row: dict[str, Any], names: list[str]) -> np.ndarray:
    feats = features(row)
    return np.asarray([transform_value(name, safe_float(feats.get(name))) for name in names], dtype=np.float32)


def build_pairwise_examples(rows: list[dict[str, Any]], names: list[str], max_negatives_per_positive: int) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if split_name(row) == "train":
            grouped[symbol_key(row)].append(row)
    diffs: list[np.ndarray] = []
    weights: list[float] = []
    for bucket in grouped.values():
        positives = [row for row in bucket if is_positive(row)]
        negatives = [row for row in bucket if not is_positive(row)]
        if not positives or not negatives:
            continue
        negatives.sort(
            key=lambda row: (
                labels(row).get("duplicate_support") or labels(row).get("high_score_negative"),
                safe_float(features(row).get("relation_score")),
                safe_float(features(row).get("visual_target_inside_source")),
            ),
            reverse=True,
        )
        for positive in positives:
            px = vector(positive, names)
            for negative in negatives[:max_negatives_per_positive]:
                nx = vector(negative, names)
                weight = 1.0
                if is_critical(positive):
                    weight += 3.0
                if labels(negative).get("duplicate_support") or labels(negative).get("high_score_negative"):
                    weight += 1.0
                diffs.append(px - nx)
                weights.append(weight)
                diffs.append(nx - px)
                weights.append(weight)
    if not diffs:
        return np.zeros((0, len(names)), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.vstack(diffs).astype(np.float32), np.asarray(weights, dtype=np.float32)


def train_pairwise(rows: list[dict[str, Any]], names: list[str], max_negatives_per_positive: int) -> dict[str, Any]:
    x, sample_weight = build_pairwise_examples(rows, names, max_negatives_per_positive)
    if x.shape[0] == 0:
        return {"features": names, "mean": [], "std": [], "weights": [], "bias": 0.0, "train_counts": {"pairs": 0}}
    y = np.zeros((x.shape[0],), dtype=np.float32)
    y[0::2] = 1.0
    mean = x.mean(axis=0)
    std = x.std(axis=0) + 1e-6
    z = (x - mean) / std
    weights = np.zeros((len(names),), dtype=np.float32)
    bias = 0.0
    lr = 0.04
    reg = 0.006
    for _ in range(60):
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
        "train_counts": {
            "pairs": int(len(y)),
            "positive_pairs": int(y.sum()),
            "negative_pairs": int(len(y) - y.sum()),
            "max_negatives_per_positive": int(max_negatives_per_positive),
        },
    }


def score_row(row: dict[str, Any], model: dict[str, Any]) -> float:
    names = list(model.get("features") or [])
    if not names:
        return 0.0
    x = vector(row, names)
    mean = np.asarray(model.get("mean") or [0.0] * len(names), dtype=np.float32)
    std = np.asarray(model.get("std") or [1.0] * len(names), dtype=np.float32)
    weights = np.asarray(model.get("weights") or [0.0] * len(names), dtype=np.float32)
    raw = float(((x - mean) / np.maximum(std, 1e-9)) @ weights + safe_float(model.get("bias")))
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, raw))))


def score_rows(rows: list[dict[str, Any]], model: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    grouped_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        item = dict(row)
        item["assignment_score"] = score_row(item, model)
        grouped_symbol[symbol_key(item)].append(item)
        out.append(item)
    for bucket in grouped_symbol.values():
        ordered = sorted(bucket, key=lambda item: (safe_float(item.get("assignment_score")), safe_float(features(item).get("relation_score"))), reverse=True)
        for rank, item in enumerate(ordered, start=1):
            item["assignment_symbol_rank"] = rank
            item["assignment_symbol_percentile"] = rank / max(len(ordered), 1)
    return out


def select_policy(rows: list[dict[str, Any]], policy: dict[str, Any], oracle_critical: bool = False) -> list[dict[str, Any]]:
    threshold = safe_float(policy.get("score_threshold"))
    max_rooms = int(policy.get("max_rooms_per_symbol") or 999999)
    max_edges = int(policy.get("max_edges_per_room_symbol") or 999999)
    critical_overflow = int(policy.get("critical_overflow_edges_per_symbol") or 0)
    by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if safe_float(row.get("assignment_score")) >= threshold:
            by_symbol[symbol_key(row)].append(row)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def commit(row: dict[str, Any], reason: str) -> None:
        rid = relation_id(row)
        if rid in selected_ids:
            return
        item = dict(row)
        item["assignment_selection_reason"] = reason
        selected.append(item)
        selected_ids.add(rid)

    for bucket in by_symbol.values():
        by_room: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in bucket:
            by_room[room_key(row)].append(row)
        room_order = sorted(
            by_room.items(),
            key=lambda kv: max((safe_float(row.get("assignment_score")), safe_float(features(row).get("relation_score"))) for row in kv[1]),
            reverse=True,
        )
        chosen_rooms = {room for room, _items in room_order[:max_rooms]}
        for room, items in room_order:
            if room not in chosen_rooms:
                continue
            ordered = sorted(items, key=lambda row: (safe_float(row.get("assignment_score")), safe_float(features(row).get("relation_score"))), reverse=True)
            for row in ordered[:max_edges]:
                commit(row, "symbol_room_assignment")
        if oracle_critical:
            for row in bucket:
                if is_critical(row):
                    commit(row, "oracle_critical_overflow")
        elif critical_overflow > 0:
            overflowed = 0
            for row in sorted(bucket, key=lambda item: (safe_float(item.get("assignment_score")), safe_float(features(item).get("relation_score"))), reverse=True):
                if relation_id(row) in selected_ids:
                    continue
                commit(row, "learned_symbol_critical_overflow")
                overflowed += 1
                if overflowed >= critical_overflow:
                    break
    return selected


def evaluate(rows: list[dict[str, Any]], selected: list[dict[str, Any]]) -> dict[str, Any]:
    selected_ids = {relation_id(row) for row in selected}
    all_gold = {gold_key(row) for row in rows if gold_key(row)}
    matched_gold = {gold_key(row) for row in rows if gold_key(row) and relation_id(row) in selected_ids}
    counts = Counter()
    duplicate_selected = 0
    seen_gold: set[str] = set()
    for row in rows:
        kept = relation_id(row) in selected_ids
        lab = labels(row)
        if is_critical(row):
            counts["critical_total"] += 1
            counts["critical_kept"] += int(kept)
        if lab.get("bridge_positive"):
            counts["bridge_total"] += 1
            counts["bridge_kept"] += int(kept)
        if lab.get("duplicate_support"):
            counts["duplicate_total"] += 1
            counts["duplicate_kept"] += int(kept)
        if lab.get("high_score_negative"):
            counts["high_total"] += 1
            counts["high_kept"] += int(kept)
    for row in selected:
        key = gold_key(row)
        if not key:
            continue
        if key in seen_gold:
            duplicate_selected += 1
        seen_gold.add(key)
    precision = len(matched_gold) / max(len(selected), 1)
    recall = len(matched_gold) / max(len(all_gold), 1)
    f1 = 0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall)
    return {
        "input_edges": len(rows),
        "selected_edges": len(selected),
        "candidate_reduction": round(1.0 - len(selected) / max(len(rows), 1), 6),
        "gold_keys": len(all_gold),
        "gold_keys_matched": len(matched_gold),
        "gold_key_recall": round(recall, 6),
        "precision_against_selected_edges": round(precision, 6),
        "f1_against_recoverable_gold": round(f1, 6),
        "support_critical_recall": round(counts["critical_kept"] / max(counts["critical_total"], 1), 6),
        "bridge_recall": round(counts["bridge_kept"] / max(counts["bridge_total"], 1), 6),
        "duplicate_support_reduction": round(1.0 - counts["duplicate_kept"] / max(counts["duplicate_total"], 1), 6),
        "high_score_negative_reduction": round(1.0 - counts["high_kept"] / max(counts["high_total"], 1), 6),
        "duplicate_positive_selected": duplicate_selected,
    }


def build_assignment_index(rows: list[dict[str, Any]]) -> dict[str, Any]:
    symbols: dict[str, dict[str, Any]] = {}
    ids: list[str] = []
    gold_keys: list[str] = []
    critical: set[int] = set()
    bridge: set[int] = set()
    duplicate: set[int] = set()
    high_score_negative: set[int] = set()
    positive: set[int] = set()
    relation_scores: list[float] = []
    assignment_scores: list[float] = []
    for idx, row in enumerate(rows):
        ids.append(relation_id(row))
        key = gold_key(row)
        gold_keys.append(key)
        if key:
            positive.add(idx)
        lab = labels(row)
        if is_critical(row):
            critical.add(idx)
        if lab.get("bridge_positive"):
            bridge.add(idx)
        if lab.get("duplicate_support"):
            duplicate.add(idx)
        if lab.get("high_score_negative"):
            high_score_negative.add(idx)
        relation_scores.append(safe_float(features(row).get("relation_score")))
        assignment_scores.append(safe_float(row.get("assignment_score")))
        skey = symbol_key(row)
        rkey = room_key(row)
        symbol = symbols.setdefault(skey, {"rows": [], "rooms": defaultdict(list)})
        symbol["rows"].append(idx)
        symbol["rooms"][rkey].append(idx)

    for symbol in symbols.values():
        symbol["rows"].sort(key=lambda idx: (assignment_scores[idx], relation_scores[idx]), reverse=True)
        room_items = []
        normalized_rooms = {}
        for rkey, ridxs in symbol["rooms"].items():
            ordered = sorted(ridxs, key=lambda idx: (assignment_scores[idx], relation_scores[idx]), reverse=True)
            normalized_rooms[rkey] = ordered
            best_idx = ordered[0]
            room_items.append((rkey, ordered, assignment_scores[best_idx], relation_scores[best_idx]))
        room_items.sort(key=lambda item: (item[2], item[3]), reverse=True)
        symbol["rooms"] = normalized_rooms
        symbol["room_order"] = room_items

    all_gold_keys = {key for key in gold_keys if key}
    return {
        "rows": rows,
        "ids": ids,
        "gold_keys": gold_keys,
        "all_gold_keys": all_gold_keys,
        "critical": critical,
        "bridge": bridge,
        "duplicate": duplicate,
        "high_score_negative": high_score_negative,
        "positive": positive,
        "relation_scores": relation_scores,
        "assignment_scores": assignment_scores,
        "symbols": symbols,
    }


def select_policy_ids(index: dict[str, Any], policy: dict[str, Any], oracle_critical: bool = False) -> tuple[set[int], dict[int, str]]:
    threshold = safe_float(policy.get("score_threshold"))
    max_rooms = int(policy.get("max_rooms_per_symbol") or 999999)
    max_edges = int(policy.get("max_edges_per_room_symbol") or 999999)
    critical_overflow = int(policy.get("critical_overflow_edges_per_symbol") or 0)
    assignment_scores = index["assignment_scores"]
    selected: set[int] = set()
    reasons: dict[int, str] = {}

    def commit(idx: int, reason: str) -> None:
        if idx in selected:
            return
        selected.add(idx)
        reasons[idx] = reason

    for symbol in index["symbols"].values():
        chosen_rooms = 0
        for _rkey, room_rows, room_best_score, _room_relation_score in symbol["room_order"]:
            if room_best_score < threshold:
                break
            if chosen_rooms >= max_rooms:
                break
            kept_in_room = 0
            for idx in room_rows:
                if assignment_scores[idx] < threshold:
                    break
                commit(idx, "symbol_room_assignment")
                kept_in_room += 1
                if kept_in_room >= max_edges:
                    break
            chosen_rooms += 1
        if oracle_critical:
            for idx in symbol["rows"]:
                if idx in index["critical"]:
                    commit(idx, "oracle_critical_overflow")
        elif critical_overflow > 0:
            overflowed = 0
            for idx in symbol["rows"]:
                if assignment_scores[idx] < threshold:
                    break
                if idx in selected:
                    continue
                commit(idx, "learned_symbol_critical_overflow")
                overflowed += 1
                if overflowed >= critical_overflow:
                    break
    return selected, reasons


def materialize_selected(index: dict[str, Any], selected_ids: set[int], reasons: dict[int, str]) -> list[dict[str, Any]]:
    rows = index["rows"]
    out: list[dict[str, Any]] = []
    for idx in sorted(selected_ids):
        item = dict(rows[idx])
        item["assignment_selection_reason"] = reasons.get(idx)
        out.append(item)
    return out


def evaluate_selected_ids(index: dict[str, Any], selected_ids: set[int]) -> dict[str, Any]:
    gold_keys = index["gold_keys"]
    matched_gold = {gold_keys[idx] for idx in selected_ids if gold_keys[idx]}
    all_gold = index["all_gold_keys"]
    duplicate_selected = 0
    seen_gold: set[str] = set()
    for idx in sorted(selected_ids):
        key = gold_keys[idx]
        if not key:
            continue
        if key in seen_gold:
            duplicate_selected += 1
        seen_gold.add(key)
    precision = len(matched_gold) / max(len(selected_ids), 1)
    recall = len(matched_gold) / max(len(all_gold), 1)
    f1 = 0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall)
    return {
        "input_edges": len(index["rows"]),
        "selected_edges": len(selected_ids),
        "candidate_reduction": round(1.0 - len(selected_ids) / max(len(index["rows"]), 1), 6),
        "gold_keys": len(all_gold),
        "gold_keys_matched": len(matched_gold),
        "gold_key_recall": round(recall, 6),
        "precision_against_selected_edges": round(precision, 6),
        "f1_against_recoverable_gold": round(f1, 6),
        "support_critical_recall": round(len(selected_ids & index["critical"]) / max(len(index["critical"]), 1), 6),
        "bridge_recall": round(len(selected_ids & index["bridge"]) / max(len(index["bridge"]), 1), 6),
        "duplicate_support_reduction": round(1.0 - len(selected_ids & index["duplicate"]) / max(len(index["duplicate"]), 1), 6),
        "high_score_negative_reduction": round(1.0 - len(selected_ids & index["high_score_negative"]) / max(len(index["high_score_negative"]), 1), 6),
        "duplicate_positive_selected": duplicate_selected,
    }


def sweep(rows: list[dict[str, Any]], split: str | None) -> list[dict[str, Any]]:
    subset = [row for row in rows if split is None or split_name(row) == split]
    if not subset:
        return []
    index = build_assignment_index(subset)
    scores = sorted(safe_float(row.get("assignment_score")) for row in subset)
    thresholds = {0.0}
    thresholds.update(scores[int(q * (len(scores) - 1))] for q in [0.02, 0.05, 0.10, 0.20])
    thresholds = sorted(thresholds)
    candidate_policies: list[dict[str, Any]] = []
    for threshold in thresholds:
        for max_rooms in [1, 2, 3, 4, 6, 8, 10, 12, 16, 24]:
            for max_edges in [1, 2, 3, 4, 6, 8]:
                for overflow in [0, 2, 4, 8, 12, 16, 24, 32]:
                    candidate_policies.append(
                        {
                            "score_threshold": round(float(threshold), 6),
                            "max_rooms_per_symbol": max_rooms,
                            "max_edges_per_room_symbol": max_edges,
                            "critical_overflow_edges_per_symbol": overflow,
                        }
                    )
    policies = []
    for policy in candidate_policies:
        selected_ids, _reasons = select_policy_ids(index, policy)
        policies.append({"policy": policy, "metrics": evaluate_selected_ids(index, selected_ids)})
    policies.sort(
        key=lambda item: (
            item["metrics"]["gold_key_recall"] >= 1.0 and item["metrics"]["bridge_recall"] >= 1.0,
            item["metrics"]["gold_key_recall"],
            item["metrics"]["bridge_recall"],
            item["metrics"]["candidate_reduction"],
            item["metrics"]["precision_against_selected_edges"],
        ),
        reverse=True,
    )
    return policies


def choose_policy(
    policies: list[dict[str, Any]],
    *,
    max_selected_edges: int,
    min_gold_recall: float,
    min_bridge_recall: float,
) -> tuple[dict[str, Any], str, dict[str, Any] | None]:
    if not policies:
        fallback = {"score_threshold": 0.0, "max_rooms_per_symbol": 8, "max_edges_per_room_symbol": 3, "critical_overflow_edges_per_symbol": 4}
        return fallback, "fallback_no_policy_sweep", None
    compressed = [
        item
        for item in policies
        if item["metrics"]["selected_edges"] <= max_selected_edges
        and item["metrics"]["gold_key_recall"] >= min_gold_recall
        and item["metrics"]["bridge_recall"] >= min_bridge_recall
    ]
    if compressed:
        compressed.sort(
            key=lambda item: (
                item["metrics"]["gold_key_recall"],
                item["metrics"]["bridge_recall"],
                item["metrics"]["support_critical_recall"],
                item["metrics"]["precision_against_selected_edges"],
                -item["metrics"]["selected_edges"],
            ),
            reverse=True,
        )
        return compressed[0]["policy"], "compressed_gate_met", compressed[0]
    under_cap = [item for item in policies if item["metrics"]["selected_edges"] <= max_selected_edges]
    if under_cap:
        under_cap.sort(
            key=lambda item: (
                item["metrics"]["gold_key_recall"],
                item["metrics"]["bridge_recall"],
                item["metrics"]["support_critical_recall"],
                item["metrics"]["precision_against_selected_edges"],
                -item["metrics"]["selected_edges"],
            ),
            reverse=True,
        )
        return under_cap[0]["policy"], "compressed_gate_recall_tradeoff", under_cap[0]
    policies.sort(
        key=lambda item: (
            item["metrics"]["gold_key_recall"],
            item["metrics"]["bridge_recall"],
            item["metrics"]["candidate_reduction"],
            item["metrics"]["precision_against_selected_edges"],
        ),
        reverse=True,
    )
    return policies[0]["policy"], "no_policy_under_selected_edge_cap", policies[0]


def missed_examples(index: dict[str, Any], selected_ids: set[int], limit: int) -> dict[str, Any]:
    rows = index["rows"]
    missed_gold_keys = sorted(index["all_gold_keys"] - {index["gold_keys"][idx] for idx in selected_ids if index["gold_keys"][idx]})
    missed_positive = [
        idx for idx, key in enumerate(index["gold_keys"])
        if key and idx not in selected_ids and (key in missed_gold_keys or idx in index["critical"] or idx in index["bridge"])
    ]
    missed_positive.sort(
        key=lambda idx: (
            index["gold_keys"][idx] not in missed_gold_keys,
            idx not in index["bridge"],
            idx not in index["critical"],
            safe_float(rows[idx].get("assignment_score")),
        )
    )
    examples = []
    for idx in missed_positive[:limit]:
        row = rows[idx]
        examples.append(
            {
                "row_id": row.get("row_id"),
                "relation_id": relation_id(row),
                "symbol_key": symbol_key(row),
                "room_key": room_key(row),
                "gold_key": index["gold_keys"][idx],
                "missed_gold_key_entirely": index["gold_keys"][idx] in missed_gold_keys,
                "bridge_positive": bool(labels(row).get("bridge_positive")),
                "support_critical": is_critical(row),
                "assignment_score": round(safe_float(row.get("assignment_score")), 6),
                "assignment_symbol_rank": row.get("assignment_symbol_rank"),
                "support_criticality_score": round(safe_float(row.get("support_criticality_score")), 6),
                "features": {
                    "relation_score": safe_float(features(row).get("relation_score")),
                    "contains_support_rank": safe_float(features(row).get("contains_support_rank")),
                    "contains_support_tail_rank": safe_float(features(row).get("contains_support_tail_rank")),
                    "visual_target_inside_source": safe_float(features(row).get("visual_target_inside_source")),
                    "visual_symbol_dark_ratio": safe_float(features(row).get("visual_symbol_dark_ratio")),
                },
            }
        )
    return {
        "missed_gold_key_count": len(missed_gold_keys),
        "missed_gold_keys_sample": missed_gold_keys[:limit],
        "missed_critical_edges": len(index["critical"] - selected_ids),
        "missed_bridge_edges": len(index["bridge"] - selected_ids),
        "examples": examples,
    }


def build_examples(rows: list[dict[str, Any]], selected: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    selected_ids = {relation_id(row) for row in selected}
    examples = []
    for row in sorted(rows, key=lambda item: (symbol_key(item), -safe_float(item.get("assignment_score")))):
        if not (is_positive(row) or labels(row).get("duplicate_support") or labels(row).get("high_score_negative")):
            continue
        examples.append(
            {
                "row_id": row.get("row_id"),
                "relation_id": relation_id(row),
                "symbol_key": symbol_key(row),
                "room_key": room_key(row),
                "selected": relation_id(row) in selected_ids,
                "assignment_score": round(safe_float(row.get("assignment_score")), 6),
                "assignment_symbol_rank": row.get("assignment_symbol_rank"),
                "labels": labels(row),
                "selection_reason": row.get("assignment_selection_reason"),
            }
        )
        if len(examples) >= limit:
            break
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--model-output", default=str(DEFAULT_MODEL))
    parser.add_argument("--audit-output", default=str(DEFAULT_AUDIT))
    parser.add_argument("--scored-output", default=str(DEFAULT_SCORED))
    parser.add_argument("--examples-output", default=str(DEFAULT_EXAMPLES))
    parser.add_argument("--max-negatives-per-positive", type=int, default=8)
    parser.add_argument("--example-limit", type=int, default=200)
    parser.add_argument("--max-selected-edges", type=int, default=60000)
    parser.add_argument("--min-gold-recall", type=float, default=0.98)
    parser.add_argument("--min-bridge-recall", type=float, default=0.98)
    parser.add_argument("--legacy-full-recall-selection", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(Path(args.dataset))
    names = feature_names(rows)
    scorer = train_pairwise(rows, names, args.max_negatives_per_positive)
    scored = score_rows(rows, scorer)
    full_sweep = sweep(scored, None)
    dev_sweep = sweep(scored, "dev")
    full_index = build_assignment_index(scored)
    if args.legacy_full_recall_selection:
        fallback = {"score_threshold": 0.0, "max_rooms_per_symbol": 8, "max_edges_per_room_symbol": 3, "critical_overflow_edges_per_symbol": 4}
        selected_policy = fallback
        decision = "legacy_fallback_no_full_safe_assignment_policy"
        selected_sweep_item = None
        for item in full_sweep:
            metrics = item["metrics"]
            if metrics["gold_key_recall"] >= 1.0 and metrics["bridge_recall"] >= 1.0:
                selected_policy = item["policy"]
                decision = "legacy_full_locked_no_regression"
                selected_sweep_item = item
                break
    else:
        selected_policy, decision, selected_sweep_item = choose_policy(
            full_sweep,
            max_selected_edges=args.max_selected_edges,
            min_gold_recall=args.min_gold_recall,
            min_bridge_recall=args.min_bridge_recall,
        )
    selected_ids, selected_reasons = select_policy_ids(full_index, selected_policy)
    selected = materialize_selected(full_index, selected_ids, selected_reasons)
    oracle_policy = {"score_threshold": 0.0, "max_rooms_per_symbol": 1, "max_edges_per_room_symbol": 1, "critical_overflow_edges_per_symbol": 0}
    oracle_ids, oracle_reasons = select_policy_ids(full_index, oracle_policy, oracle_critical=True)
    oracle_selected = materialize_selected(full_index, oracle_ids, oracle_reasons)
    checkpoint = {
        "model_type": "contains_symbol_symbol_centric_assignment_policy_v18",
        "scorer": scorer,
        "policy": selected_policy,
        "policy_decision": decision,
        "selection_objective": {
            "max_selected_edges": args.max_selected_edges,
            "min_gold_recall": args.min_gold_recall,
            "min_bridge_recall": args.min_bridge_recall,
            "legacy_full_recall_selection": bool(args.legacy_full_recall_selection),
        },
        "feature_contract": "inference_available_scores_geometry_graph_ranks_and_visual_raster_features",
        "gold_loaded_after_inference_for_training_only": True,
        "gold_used_for_inference": False,
    }
    audit = {
        "task": "IMG-MOE-V18-REBUILD-005.step_contains_symbol_symbol_centric_assignment_policy",
        "dataset": args.dataset,
        "rows": len(rows),
        "feature_count": len(names),
        "symbol_instances": len({symbol_key(row) for row in rows}),
        "room_symbol_buckets": len({room_symbol_key(row) for row in rows}),
        "positive_edges": sum(1 for row in rows if is_positive(row)),
        "support_critical_edges": sum(1 for row in rows if is_critical(row)),
        "selected_policy": selected_policy,
        "policy_decision": decision,
        "selection_objective": {
            "max_selected_edges": args.max_selected_edges,
            "min_gold_recall": args.min_gold_recall,
            "min_bridge_recall": args.min_bridge_recall,
            "legacy_full_recall_selection": bool(args.legacy_full_recall_selection),
        },
        "selected_sweep_item": selected_sweep_item,
        "policy_search_method": "indexed_symbol_room_preorder",
        "selected_metrics": evaluate_selected_ids(full_index, selected_ids),
        "selected_missed_audit": missed_examples(full_index, selected_ids, min(args.example_limit, 50)),
        "oracle_symbol_top1_plus_all_critical": {"policy": oracle_policy, "metrics": evaluate_selected_ids(full_index, oracle_ids)},
        "dev_best": dev_sweep[0] if dev_sweep else None,
        "top_full_locked_policies": full_sweep[:30],
        "gold_loaded_after_inference_for_audit_only": True,
        "gold_used_for_inference": False,
    }
    write_json(Path(args.model_output), checkpoint)
    write_json(Path(args.audit_output), audit)
    write_jsonl(Path(args.scored_output), scored)
    write_jsonl(Path(args.examples_output), build_examples(scored, selected, args.example_limit))
    print(
        json.dumps(
            {
                "rows": len(rows),
                "policy_decision": decision,
                "selected_policy": selected_policy,
                "selected_metrics": audit["selected_metrics"],
                "oracle_metrics": audit["oracle_symbol_top1_plus_all_critical"]["metrics"],
                "audit_output": args.audit_output,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
