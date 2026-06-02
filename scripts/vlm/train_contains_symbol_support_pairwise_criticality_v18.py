#!/usr/bin/env python3
"""Train a pairwise support-criticality ranker for contains_symbol.

The previous support-criticality scorer is a pointwise classifier. It preserves
recall only with wide top/tail reservations because it is not optimized to rank
the sparse critical edge above duplicate support negatives inside the same
room-symbol support set. This script trains a linear pairwise ranker on
critical-vs-noncritical edges within each support set, then audits whether the
same locked selection contract can move closer to the oracle top2/tail2 policy.
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

DEFAULT_DATASET = REPORT / "contains_symbol_support_criticality_v18_dataset.jsonl"
DEFAULT_BASE_MODEL = CHECKPOINT / "listwise_model_contains_support_locked_v18.json"
DEFAULT_MODEL = CHECKPOINT / "contains_symbol_support_pairwise_criticality_v18.json"
DEFAULT_COMBINED_MODEL = CHECKPOINT / "listwise_model_contains_support_pairwise_criticality_locked_v18.json"
DEFAULT_AUDIT = REPORT / "contains_symbol_support_pairwise_criticality_v18_audit.json"
DEFAULT_SCORED = REPORT / "contains_symbol_support_pairwise_criticality_v18_dataset.jsonl"
DEFAULT_EXAMPLES = REPORT / "contains_symbol_support_pairwise_criticality_v18_examples.jsonl"


def relation_id(row: dict[str, Any]) -> str:
    return str(row.get("relation_id") or f"{row.get('row_id')}|{row.get('source_candidate_id')}|{row.get('target_candidate_id')}")


def labels(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("labels") if isinstance(row.get("labels"), dict) else {}


def features(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("features") if isinstance(row.get("features"), dict) else {}


def split_name(row: dict[str, Any]) -> str:
    return str(row.get("split") or "unknown")


def gold_key(row: dict[str, Any]) -> str:
    return str(labels(row).get("gold_key") or "")


def is_critical(row: dict[str, Any]) -> bool:
    return bool(labels(row).get("support_critical"))


def feature_names(rows: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for row in rows:
        names.update(str(name) for name in features(row).keys())
    return sorted(names)


def vectorize(row: dict[str, Any], names: list[str]) -> np.ndarray:
    feats = features(row)
    return np.asarray([transform_value(name, safe_float(feats.get(name))) for name in names], dtype=np.float32)


def build_pairwise_examples(rows: list[dict[str, Any]], names: list[str], *, max_negatives_per_positive: int) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if split_name(row) == "train":
            grouped[str(row.get("contains_support_set_key"))].append(row)

    diffs: list[np.ndarray] = []
    weights: list[float] = []
    for bucket in grouped.values():
        positives = [row for row in bucket if is_critical(row)]
        if not positives:
            continue
        negatives = [row for row in bucket if not is_critical(row)]
        if not negatives:
            continue
        negatives.sort(
            key=lambda row: (
                labels(row).get("duplicate_support") or labels(row).get("high_score_negative"),
                safe_float(row.get("support_criticality_score")),
                safe_float(features(row).get("relation_score")),
            ),
            reverse=True,
        )
        for positive in positives:
            pos_x = vectorize(positive, names)
            for negative in negatives[:max_negatives_per_positive]:
                neg_x = vectorize(negative, names)
                hard = bool(labels(negative).get("duplicate_support") or labels(negative).get("high_score_negative"))
                weight = 2.0 if hard else 1.0
                if labels(positive).get("bridge_positive"):
                    weight += 2.0
                if labels(positive).get("gold_representative"):
                    weight += 2.0
                diffs.append(pos_x - neg_x)
                weights.append(weight)
                diffs.append(neg_x - pos_x)
                weights.append(weight)
    if not diffs:
        return np.zeros((0, len(names)), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.vstack(diffs).astype(np.float32), np.asarray(weights, dtype=np.float32)


def train_pairwise_ranker(rows: list[dict[str, Any]], names: list[str], *, max_negatives_per_positive: int) -> dict[str, Any]:
    x, sample_weight = build_pairwise_examples(rows, names, max_negatives_per_positive=max_negatives_per_positive)
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
    for _ in range(40):
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
    x = np.asarray([transform_value(name, safe_float(features(row).get(name))) for name in names], dtype=np.float32)
    mean = np.asarray(model.get("mean") or [0.0] * len(names), dtype=np.float32)
    std = np.asarray(model.get("std") or [1.0] * len(names), dtype=np.float32)
    weights = np.asarray(model.get("weights") or [0.0] * len(names), dtype=np.float32)
    raw = float(((x - mean) / np.maximum(std, 1e-9)) @ weights + safe_float(model.get("bias")))
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, raw))))


def relation_score(row: dict[str, Any]) -> float:
    return safe_float(features(row).get("relation_score"))


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
        for row in sorted(bucket, key=lambda item: (relation_score(item), safe_float(item.get("support_criticality_score"))), reverse=True):
            if int(row.get("contains_support_rank") or 999999) <= top_k:
                commit(row, "support_top_k")
        if tail_slots > 0:
            for row in sorted(bucket, key=lambda item: (int(item.get("contains_support_tail_rank") or 999999), -relation_score(item))):
                if int(row.get("contains_support_tail_rank") or 999999) <= tail_slots:
                    commit(row, "support_tail_slot")
        if oracle_critical:
            for row in bucket:
                if is_critical(row):
                    commit(row, "oracle_support_critical")
        elif overflow_slots > 0:
            overflowed = 0
            for row in sorted(bucket, key=lambda item: (safe_float(item.get("support_criticality_score")), relation_score(item)), reverse=True):
                if relation_id(row) in selected_ids or safe_float(row.get("support_criticality_score")) < critical_threshold:
                    continue
                commit(row, "learned_critical_overflow")
                overflowed += 1
                if overflowed >= overflow_slots:
                    break
    return selected


def evaluate(rows: list[dict[str, Any]], selected: list[dict[str, Any]]) -> dict[str, Any]:
    selected_ids = {relation_id(row) for row in selected}
    all_gold = {gold_key(row) for row in rows if gold_key(row)}
    matched_gold = {gold_key(row) for row in rows if gold_key(row) and relation_id(row) in selected_ids}
    counts = Counter()
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
    return {
        "input_edges": len(rows),
        "selected_edges": len(selected),
        "candidate_reduction": round(1.0 - len(selected) / max(len(rows), 1), 6),
        "gold_keys": len(all_gold),
        "gold_keys_matched": len(matched_gold),
        "gold_key_recall": round(len(matched_gold) / max(len(all_gold), 1), 6),
        "support_critical_recall": round(counts["critical_kept"] / max(counts["critical_total"], 1), 6),
        "bridge_recall": round(counts["bridge_kept"] / max(counts["bridge_total"], 1), 6),
        "duplicate_support_reduction": round(1.0 - counts["duplicate_kept"] / max(counts["duplicate_total"], 1), 6),
        "high_score_negative_reduction": round(1.0 - counts["high_kept"] / max(counts["high_total"], 1), 6),
    }


def missed_profile(rows: list[dict[str, Any]], selected: list[dict[str, Any]]) -> dict[str, Any]:
    selected_ids = {relation_id(row) for row in selected}
    missed = [row for row in rows if is_critical(row) and relation_id(row) not in selected_ids]
    return {
        "missed_critical_edges": len(missed),
        "missed_gold_representatives": sum(1 for row in missed if labels(row).get("gold_representative")),
        "missed_bridge_positives": sum(1 for row in missed if labels(row).get("bridge_positive")),
        "by_split": dict(Counter(split_name(row) for row in missed)),
        "score_quantiles": quantiles([safe_float(row.get("support_criticality_score")) for row in missed]),
    }


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    out: dict[str, float] = {}
    for name, q in [("p00", 0.0), ("p25", 0.25), ("p50", 0.5), ("p75", 0.75), ("p90", 0.9), ("p100", 1.0)]:
        out[name] = round(ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1)))], 6)
    return out


def sweep(rows: list[dict[str, Any]], *, split: str | None) -> list[dict[str, Any]]:
    subset = [row for row in rows if split is None or split_name(row) == split]
    if not subset:
        return []
    scores = sorted(safe_float(row.get("support_criticality_score")) for row in subset)
    high_threshold = scores[int(0.90 * (len(scores) - 1))]
    candidate_policies = [
        {"threshold": 0.01, "top_k": 2, "tail_slots": 2, "critical_overflow_slots": 2, "critical_threshold": 0.0},
        {"threshold": 0.01, "top_k": 2, "tail_slots": 2, "critical_overflow_slots": 4, "critical_threshold": 0.0},
        {"threshold": 0.01, "top_k": 2, "tail_slots": 4, "critical_overflow_slots": 2, "critical_threshold": 0.0},
        {"threshold": 0.01, "top_k": 3, "tail_slots": 2, "critical_overflow_slots": 2, "critical_threshold": 0.0},
        {"threshold": 0.01, "top_k": 3, "tail_slots": 4, "critical_overflow_slots": 2, "critical_threshold": 0.0},
        {"threshold": 0.01, "top_k": 4, "tail_slots": 4, "critical_overflow_slots": 0, "critical_threshold": 1.0},
        {"threshold": 0.01, "top_k": 8, "tail_slots": 8, "critical_overflow_slots": 0, "critical_threshold": 1.0},
        {"threshold": 0.01, "top_k": 2, "tail_slots": 2, "critical_overflow_slots": 2, "critical_threshold": round(float(high_threshold), 6)},
    ]
    policies: list[dict[str, Any]] = []
    for policy in candidate_policies:
        selected = select_policy(subset, policy)
        policies.append({"policy": policy, "metrics": evaluate(subset, selected), "missed_profile": missed_profile(subset, selected)})
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


def build_examples(rows: list[dict[str, Any]], selected: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    selected_ids = {relation_id(row) for row in selected}
    examples: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: (split_name(item), str(item.get("contains_support_set_key")), -safe_float(item.get("support_criticality_score")))):
        lab = labels(row)
        if not (is_critical(row) or lab.get("duplicate_support") or lab.get("high_score_negative")):
            continue
        examples.append(
            {
                "row_id": row.get("row_id"),
                "split": split_name(row),
                "relation_id": relation_id(row),
                "selected": relation_id(row) in selected_ids,
                "selection_reason": row.get("selection_reason"),
                "support_set_key": row.get("contains_support_set_key"),
                "support_set_size": row.get("contains_support_set_size"),
                "support_rank": row.get("contains_support_rank"),
                "support_tail_rank": row.get("contains_support_tail_rank"),
                "relation_score": relation_score(row),
                "support_criticality_score": row.get("support_criticality_score"),
                "labels": lab,
            }
        )
        if len(examples) >= limit:
            break
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--base-model", default=str(DEFAULT_BASE_MODEL))
    parser.add_argument("--model-output", default=str(DEFAULT_MODEL))
    parser.add_argument("--combined-model-output", default=str(DEFAULT_COMBINED_MODEL))
    parser.add_argument("--audit-output", default=str(DEFAULT_AUDIT))
    parser.add_argument("--scored-output", default=str(DEFAULT_SCORED))
    parser.add_argument("--examples-output", default=str(DEFAULT_EXAMPLES))
    parser.add_argument("--max-negatives-per-positive", type=int, default=4)
    parser.add_argument("--example-limit", type=int, default=200)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.dataset))
    names = feature_names(rows)
    scorer = train_pairwise_ranker(rows, names, max_negatives_per_positive=args.max_negatives_per_positive)
    scored_rows = []
    for row in rows:
        item = dict(row)
        item["support_criticality_score"] = score_row(item, scorer)
        scored_rows.append(item)

    dev_sweep = sweep(scored_rows, split="dev")
    full_sweep = sweep(scored_rows, split=None)
    fallback_policy = {"threshold": 0.01, "top_k": 8, "tail_slots": 8, "critical_overflow_slots": 0, "critical_threshold": 1.0}
    selected_policy = fallback_policy
    policy_decision = "fallback_no_full_safe_pairwise_policy"
    for item in full_sweep:
        metrics = item["metrics"]
        if metrics["gold_key_recall"] >= 1.0 and metrics["bridge_recall"] >= 1.0:
            selected_policy = item["policy"]
            policy_decision = "full_locked_no_regression"
            break

    selected = select_policy(scored_rows, selected_policy)
    oracle_policy = {"threshold": 0.01, "top_k": 2, "tail_slots": 2, "critical_overflow_slots": 0, "critical_threshold": 1.0}
    oracle_selected = select_policy(scored_rows, oracle_policy, oracle_critical=True)
    checkpoint = {
        "model_type": "contains_symbol_support_pairwise_criticality_v18",
        "support_key": "room_symbol_instance",
        "scorer": scorer,
        "policy": selected_policy,
        "policy_decision": policy_decision,
        "feature_contract": "inference_available_scores_geometry_and_graph_ranks_only",
        "gold_loaded_after_inference_for_training_only": True,
        "gold_used_for_inference": False,
    }
    base_model = json.loads(Path(args.base_model).read_text(encoding="utf-8"))
    combined = dict(base_model)
    combined["contains_symbol_support_criticality"] = {
        "model_path": args.model_output,
        "support_key": "room_symbol_instance",
        "scorer": scorer,
        "policy_decision": policy_decision,
    }
    combined.setdefault("support_criticality_integration", {})["pairwise_v18"] = {
        "model_path": args.model_output,
        "audit": args.audit_output,
        "selected_policy": selected_policy,
        "policy_decision": policy_decision,
    }
    combined_policy = dict(combined.get("policy") or {})
    contains_policy = dict(combined_policy.get("contains_symbol") or {})
    contains_policy.update(
        {
            "threshold": selected_policy["threshold"],
            "max_per_pair": 999999,
            "max_per_room_symbol_instance": selected_policy["top_k"],
            "max_contains_symbol_support_tail_slots": selected_policy["tail_slots"],
            "max_contains_symbol_support_overflow_slots": selected_policy["critical_overflow_slots"],
            "contains_symbol_support_criticality_threshold": selected_policy["critical_threshold"],
        }
    )
    combined_policy["contains_symbol"] = contains_policy
    combined["policy"] = combined_policy

    audit = {
        "task": "IMG-MOE-V18-REBUILD-005.step_contains_symbol_pairwise_support_criticality",
        "dataset": args.dataset,
        "base_model": args.base_model,
        "rows": len(rows),
        "feature_count": len(names),
        "support_critical_positive": sum(1 for row in rows if is_critical(row)),
        "selected_policy": selected_policy,
        "policy_decision": policy_decision,
        "selected_metrics": evaluate(scored_rows, selected),
        "selected_missed_profile": missed_profile(scored_rows, selected),
        "oracle_top2_tail2_plus_all_critical": {
            "policy": oracle_policy,
            "metrics": evaluate(scored_rows, oracle_selected),
        },
        "dev_best": dev_sweep[0] if dev_sweep else None,
        "top_full_locked_policies": full_sweep[:25],
        "gold_loaded_after_inference_for_audit_only": True,
        "gold_used_for_inference": False,
    }
    write_json(Path(args.model_output), checkpoint)
    write_json(Path(args.combined_model_output), combined)
    write_json(Path(args.audit_output), audit)
    write_jsonl(Path(args.scored_output), scored_rows)
    write_jsonl(Path(args.examples_output), build_examples(scored_rows, selected, args.example_limit))
    print(
        json.dumps(
            {
                "rows": len(rows),
                "policy_decision": policy_decision,
                "selected_policy": selected_policy,
                "selected_metrics": audit["selected_metrics"],
                "oracle_metrics": audit["oracle_top2_tail2_plus_all_critical"]["metrics"],
                "model_output": args.model_output,
                "combined_model_output": args.combined_model_output,
                "audit_output": args.audit_output,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
