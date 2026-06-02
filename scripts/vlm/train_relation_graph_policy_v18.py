#!/usr/bin/env python3
"""Train an auditable relation-graph policy for v18."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from relation_graph_reconstruction_v18 import (
    CHECKPOINT,
    DEFAULT_AUDIT,
    DEFAULT_DATASET,
    DEFAULT_EVAL,
    DEFAULT_MODEL,
    RELATIONS,
    default_policy,
    feature_names,
    load_jsonl,
    safe_float,
    score_rows,
    select_rows,
    split_name,
    train_logistic,
    write_json,
)


def load_dataset(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = load_jsonl(path, limit=limit)
    row_ids = sorted({str(row.get("row_id") or "") for row in rows})
    train_cut = max(1, int(len(row_ids) * 0.70))
    dev_cut = min(len(row_ids), max(train_cut + 1, int(len(row_ids) * 0.80))) if len(row_ids) >= 3 else train_cut
    row_split: dict[str, str] = {}
    for index, row_id in enumerate(row_ids):
        if index < train_cut:
            row_split[row_id] = "train"
        elif index < dev_cut:
            row_split[row_id] = "dev"
        else:
            row_split[row_id] = "test"
    for row in rows:
        row["split"] = row_split.get(str(row.get("row_id") or ""), split_name(str(row.get("row_id") or "")))
    return rows


def evaluate(rows: list[dict[str, Any]], relation_rows: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    selected = select_rows(relation_rows, policy)
    selected_by_id = {str(row.get("relation_id")): row for row in selected}
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    matched: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    gold_keys: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for row in rows:
        if not row.get("labels", {}).get("label_positive"):
            continue
        rel_type = str(row.get("relation"))
        key = (str(row.get("row_id")), str(row.get("labels", {}).get("gold_room_id")), str(row.get("labels", {}).get("gold_target_id")))
        gold_keys[rel_type].add(key)
    for row in selected:
        rel_type = str(row.get("relation"))
        counters[rel_type]["predicted"] += 1
        if str(row.get("relation_id")) in selected_by_id and row.get("labels", {}).get("label_positive"):
            key = (str(row.get("row_id")), str(row.get("labels", {}).get("gold_room_id")), str(row.get("labels", {}).get("gold_target_id")))
            if key not in matched[rel_type]:
                matched[rel_type].add(key)
                counters[rel_type]["true_positive"] += 1
    metrics: dict[str, Any] = {}
    for rel_type in RELATIONS:
        tp = counters[rel_type]["true_positive"]
        pred = counters[rel_type]["predicted"]
        gold_total = len(gold_keys[rel_type])
        precision = tp / max(pred, 1)
        recall = tp / max(gold_total, 1)
        f1 = 0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall)
        metrics[rel_type] = {
            "true_positive": tp,
            "predicted": pred,
            "gold": gold_total,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }
    feature_reduction = 1.0 - len(selected) / max(len(rows), 1)
    return {
        "rows": len({row.get("row_id") for row in rows}),
        "edge_rows": len(rows),
        "selected_rows": len(selected),
        "feature_reduction": round(feature_reduction, 6),
        "relation_metrics": metrics,
        "policy": policy,
        "source_integrity": rows[0].get("source_integrity") if rows else None,
    }


CAP_PROFILES: dict[str, list[dict[str, int]]] = {
    "bounded_by": [
        {"max_per_source": 4, "max_per_target": 4, "max_per_component": 6},
        {"max_per_source": 8, "max_per_target": 8, "max_per_component": 16},
        {"max_per_source": 16, "max_per_target": 16, "max_per_component": 32},
        {"max_per_source": 999999, "max_per_target": 999999, "max_per_component": 999999},
    ],
    "contains_symbol": [
        {"max_per_source": 2, "max_per_target": 1, "max_per_component": 4},
        {"max_per_source": 8, "max_per_target": 2, "max_per_component": 24},
        {"max_per_source": 16, "max_per_target": 4, "max_per_component": 64},
        {"max_per_source": 999999, "max_per_target": 999999, "max_per_component": 999999},
    ],
    "labeled_by_text": [
        {"max_per_source": 2, "max_per_target": 1, "max_per_component": 3},
        {"max_per_source": 6, "max_per_target": 2, "max_per_component": 12},
        {"max_per_source": 12, "max_per_target": 4, "max_per_component": 32},
        {"max_per_source": 999999, "max_per_target": 999999, "max_per_component": 999999},
    ],
    "adjacent_to": [
        {"max_per_source": 3, "max_per_target": 3, "max_per_component": 8},
        {"max_per_source": 8, "max_per_target": 8, "max_per_component": 24},
        {"max_per_source": 16, "max_per_target": 16, "max_per_component": 64},
        {"max_per_source": 999999, "max_per_target": 999999, "max_per_component": 999999},
    ],
}


def relation_policy_sweep(rows: list[dict[str, Any]], base_policy: dict[str, Any], rel_type: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for threshold in [0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        for caps in CAP_PROFILES.get(rel_type, [dict(base_policy.get(rel_type) or {})]):
            policy = {key: dict(value) for key, value in base_policy.items()}
            policy[rel_type] = {**policy.get(rel_type, {}), "threshold": threshold, **caps}
            metrics = evaluate(rows, rows, policy)
            rel_metrics = (metrics.get("relation_metrics") or {}).get(rel_type) or {}
            out.append(
                {
                    "threshold": threshold,
                    **caps,
                    "selected_rows": metrics.get("selected_rows"),
                    "feature_reduction": metrics.get("feature_reduction"),
                    **rel_metrics,
                }
            )
    return out


def choose_policy(dev_rows: list[dict[str, Any]], base_policy: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    policy = {key: dict(value) for key, value in base_policy.items()}
    sweep: dict[str, Any] = {}
    for rel_type in RELATIONS:
        rel_sweep = relation_policy_sweep(dev_rows, policy, rel_type)
        viable = [row for row in rel_sweep if safe_float(row.get("recall")) >= 0.98]
        if viable:
            chosen = max(viable, key=lambda row: (safe_float(row.get("feature_reduction")), safe_float(row.get("precision"))))
            reason = "max_reduction_at_recall_ge_0.98"
        else:
            chosen = max(rel_sweep, key=lambda row: (safe_float(row.get("recall")), safe_float(row.get("precision")), safe_float(row.get("feature_reduction"))))
            reason = "best_available_recall_below_gate"
        policy[rel_type] = {
            **policy.get(rel_type, {}),
            "threshold": chosen["threshold"],
            "max_per_source": chosen["max_per_source"],
            "max_per_target": chosen["max_per_target"],
            "max_per_component": chosen["max_per_component"],
        }
        sweep[rel_type] = {"chosen": {**chosen, "reason": reason}, "sweep": rel_sweep}
    return policy, sweep


def fail_closed_policy(base_policy: dict[str, Any], reason: str) -> tuple[dict[str, Any], dict[str, Any]]:
    policy = {key: dict(value) for key, value in base_policy.items()}
    sweep: dict[str, Any] = {}
    for rel_type in RELATIONS:
        policy[rel_type] = {
            **policy.get(rel_type, {}),
            "threshold": 0.0,
            "max_per_source": 999999,
            "max_per_target": 999999,
            "max_per_component": 999999,
            "fail_closed": True,
            "fail_closed_reason": reason,
        }
        sweep[rel_type] = {"chosen": {"reason": reason, **policy[rel_type]}, "sweep": []}
    return policy, sweep


def fail_closed_relation(base_policy: dict[str, Any], rel_type: str, reason: str) -> dict[str, Any]:
    return {
        **dict(base_policy.get(rel_type) or {}),
        "threshold": 0.0,
        "max_per_source": 999999,
        "max_per_target": 999999,
        "max_per_component": 999999,
        "fail_closed": True,
        "fail_closed_reason": reason,
    }


def apply_no_regression_gate(policy: dict[str, Any], validation_rows: list[dict[str, Any]], base_policy: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not validation_rows:
        return policy, {"status": "skipped_no_validation_rows"}
    gated = {key: dict(value) for key, value in policy.items()}
    fail_closed_all, _ = fail_closed_policy(base_policy, "no_regression_reference")
    current_eval = evaluate(validation_rows, validation_rows, gated)
    reference_eval = evaluate(validation_rows, validation_rows, fail_closed_all)
    decisions: dict[str, Any] = {}
    for rel_type in RELATIONS:
        current = ((current_eval.get("relation_metrics") or {}).get(rel_type) or {})
        reference = ((reference_eval.get("relation_metrics") or {}).get(rel_type) or {})
        current_recall = safe_float(current.get("recall"))
        reference_recall = safe_float(reference.get("recall"))
        if current_recall + 1e-9 < reference_recall:
            gated[rel_type] = fail_closed_relation(base_policy, rel_type, "heldout_no_regression_gate_failed")
            decisions[rel_type] = {
                "action": "fail_closed",
                "current_recall": current.get("recall"),
                "reference_recall": reference.get("recall"),
            }
        else:
            decisions[rel_type] = {
                "action": "keep_compressive_policy",
                "current_recall": current.get("recall"),
                "reference_recall": reference.get("recall"),
            }
    gated_eval = evaluate(validation_rows, validation_rows, gated)
    return gated, {"status": "applied", "decisions": decisions, "before": current_eval, "after": gated_eval, "reference": reference_eval}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--model-output", default=str(DEFAULT_MODEL))
    parser.add_argument("--eval-output", default=str(DEFAULT_EVAL))
    parser.add_argument("--audit-output", default=str(DEFAULT_AUDIT))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--locked", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    limit = 500 if args.smoke else args.limit
    rows = load_dataset(Path(args.dataset), limit=limit)
    train_rows = [row for row in rows if row.get("split") == "train"]
    dev_rows = [row for row in rows if row.get("split") == "dev"]
    test_rows = [row for row in rows if row.get("split") == "test"]

    names = feature_names(train_rows or rows)
    model = {
        "model_type": "relation_graph_policy_v18",
        "features": names,
        "relations": {},
        "train_counts": {},
    }
    for rel_type in RELATIONS:
        subset = [row for row in train_rows if row.get("relation") == rel_type]
        rel_model = train_logistic(subset, names)
        model["relations"][rel_type] = rel_model
        model["train_counts"][rel_type] = rel_model["train_counts"]

    train_rows = score_rows(train_rows, model)
    dev_rows = score_rows(dev_rows, model)
    test_rows = score_rows(test_rows, model)
    if dev_rows:
        calibration_split = "dev"
        policy, sweep = choose_policy(dev_rows, default_policy())
    else:
        calibration_split = "none_fail_closed_no_dev_rows"
        policy, sweep = fail_closed_policy(default_policy(), "no_dev_rows_for_threshold_or_cap_calibration")
    validation_rows = rows if args.locked else (test_rows or dev_rows or train_rows)
    policy, no_regression_gate = apply_no_regression_gate(policy, validation_rows, default_policy())

    eval_train = evaluate(train_rows, train_rows, policy)
    eval_dev = evaluate(dev_rows, dev_rows, policy)
    eval_test = evaluate(test_rows, test_rows, policy)
    model["policy"] = policy
    model["selection_sweep"] = sweep
    model["calibration_split"] = calibration_split
    model["no_regression_gate"] = no_regression_gate
    model["locked"] = bool(args.locked)
    model["source_integrity"] = train_rows[0].get("source_integrity") if train_rows else None

    Path(args.model_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.model_output).write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_json(Path(args.eval_output), {"train": eval_train, "dev": eval_dev, "test": eval_test, "policy": policy, "source_integrity": model["source_integrity"]})
    write_json(
        Path(args.audit_output),
        {
            "task": "IMG-MOE-V18-REBUILD-005",
            "dataset_rows": len(rows),
            "train_rows": len(train_rows),
            "dev_rows": len(dev_rows),
            "test_rows": len(test_rows),
            "policy": policy,
            "calibration_split": calibration_split,
            "no_regression_gate": no_regression_gate,
            "sweep": sweep,
            "source_integrity": model["source_integrity"],
            "locked": bool(args.locked),
            "smoke": bool(args.smoke),
        },
    )
    print(json.dumps({"train_rows": len(train_rows), "dev_rows": len(dev_rows), "test_rows": len(test_rows), "model_output": args.model_output}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
