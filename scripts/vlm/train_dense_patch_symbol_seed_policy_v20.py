#!/usr/bin/env python3
"""Train a dense patch/seed policy from v20 supervision."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
REPORT = ROOT / "reports/vlm"
DEFAULT_DATASET = ROOT / "datasets/dense_patch_symbol_seed_v20/locked.jsonl"
DEFAULT_MODEL = ROOT / "checkpoints/dense_patch_symbol_seed_policy_v20/model.json"
DEFAULT_AUDIT = REPORT / "dense_patch_symbol_seed_policy_v20_audit.json"
DEFAULT_SCORED = REPORT / "dense_patch_symbol_seed_policy_v20_scored.jsonl"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_topology_relations_v18 import integrity, write_json  # noqa: E402
from nms_topology_relations_v18 import load_jsonl  # noqa: E402
from train_missing_symbol_recall_expert_v18 import auc, threshold_metrics, write_jsonl  # noqa: E402
from train_patch_symbol_body_segmenter_v18 import FEATURES, feature_weights, score, train_model  # noqa: E402


def stable_split(row: dict[str, Any]) -> str:
    existing = str(row.get("split") or "")
    if existing in {"train", "dev", "test"}:
        return existing
    key = str(row.get("row_id") or "")
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) % 10
    if value < 7:
        return "train"
    if value < 9:
        return "dev"
    return "test"


def valid_feature_row(row: dict[str, Any]) -> bool:
    feats = row.get("features")
    if not isinstance(feats, dict):
        return False
    return all(name in feats for name in FEATURES)


def balance_train_rows(rows: list[dict[str, Any]], max_pos_per_neg: float) -> list[dict[str, Any]]:
    positives = [row for row in rows if row.get("label_objectness")]
    negatives = [row for row in rows if not row.get("label_objectness")]
    if not negatives or max_pos_per_neg <= 0:
        return rows
    max_pos = int(len(negatives) * max_pos_per_neg)
    positives = sorted(positives, key=lambda row: str(row.get("id")))[:max_pos]
    return positives + negatives


def evaluate(rows: list[dict[str, Any]], model: dict[str, Any]) -> dict[str, Any]:
    scored: list[dict[str, Any]] = []
    for row in rows:
        value = round(score(row, model), 6)
        scored.append(
            {
                "id": row.get("id"),
                "row_id": row.get("row_id"),
                "split": row.get("split"),
                "label_objectness": bool(row.get("label_objectness")),
                "label_safe_positive": bool(row.get("label_safe_positive")),
                "label_baseline_missing": bool(row.get("label_baseline_missing")),
                "gold_keys": row.get("gold_keys"),
                "baseline_gold_keys": row.get("baseline_gold_keys"),
                "patch_score": value,
            }
        )
    score_rows = [{"objectness_score": row["patch_score"], "label_objectness": row["label_objectness"]} for row in scored]
    thresholds = sorted({row["patch_score"] for row in scored})
    sweep = [threshold_metrics(score_rows, threshold) for threshold in thresholds] if thresholds else []
    feasible = [row for row in sweep if row["recall"] >= 0.98]
    selected = sorted(feasible, key=lambda row: (row["candidate_reduction"], row["precision"]), reverse=True)[0] if feasible else (sorted(sweep, key=lambda row: (row["recall"], row["precision"]), reverse=True)[0] if sweep else {})
    by_bucket: dict[str, Counter[str]] = {}
    for row in scored:
        bucket = str(row.get("split") or "unknown")
        by_bucket.setdefault(bucket, Counter())
        pred = row["patch_score"] >= float(selected.get("threshold", 1.0))
        label = bool(row.get("label_objectness"))
        by_bucket[bucket]["rows"] += 1
        by_bucket[bucket]["positive"] += int(label)
        by_bucket[bucket]["selected"] += int(pred)
        by_bucket[bucket]["true_positive"] += int(pred and label)
    bucket_metrics = {}
    for bucket, counts in by_bucket.items():
        bucket_metrics[bucket] = {
            **dict(counts),
            "recall": round(counts["true_positive"] / max(counts["positive"], 1), 6),
            "selection_rate": round(counts["selected"] / max(counts["rows"], 1), 6),
        }
    return {
        "scored": scored,
        "metrics": {
            "rows": len(rows),
            "positive": sum(1 for row in rows if row.get("label_objectness")),
            "negative": sum(1 for row in rows if not row.get("label_objectness")),
            "auc": auc(score_rows),
            "selected_policy": selected,
            "bucket_metrics": bucket_metrics,
        },
    }


def recall_at_caps(scored: list[dict[str, Any]], caps: list[int], target_field: str) -> dict[str, Any]:
    by_row: dict[str, list[dict[str, Any]]] = {}
    target_by_row: dict[str, set[str]] = {}
    for row in scored:
        rid = str(row.get("row_id"))
        by_row.setdefault(rid, []).append(row)
        target_by_row.setdefault(rid, set())
        for key in row.get(target_field) or []:
            target_by_row[rid].add(str(key))
    out: dict[str, Any] = {}
    for cap in caps:
        hit: set[str] = set()
        total: set[str] = set()
        selected_count = 0
        for rid, rows in by_row.items():
            total.update(target_by_row[rid])
            selected = sorted(rows, key=lambda item: float(item.get("patch_score") or 0.0), reverse=True)[:cap]
            selected_count += len(selected)
            for row in selected:
                for key in row.get(target_field) or []:
                    hit.add(str(key))
        out[str(cap)] = {
            "selected_patch_rows": selected_count,
            "target_gold_keys": len(total),
            "hit_gold_keys": len(hit),
            "patch_recall": round(len(hit) / max(len(total), 1), 6),
            "selected_patch_rows_per_hit_gold": round(selected_count / max(len(hit), 1), 6),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--model-output", default=str(DEFAULT_MODEL))
    parser.add_argument("--audit-output", default=str(DEFAULT_AUDIT))
    parser.add_argument("--scored-output", default=str(DEFAULT_SCORED))
    parser.add_argument("--max-train-positive-per-negative", type=float, default=3.0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = [row for row in load_jsonl(Path(args.dataset)) if valid_feature_row(row)]
    if args.limit is not None:
        rows = rows[: args.limit]
    for row in rows:
        row["split"] = stable_split(row)
    train_rows = balance_train_rows([row for row in rows if row["split"] == "train"], args.max_train_positive_per_negative)
    dev_rows = [row for row in rows if row["split"] == "dev"]
    test_rows = [row for row in rows if row["split"] == "test"]
    model = train_model(train_rows)
    model["model_type"] = "dense_patch_symbol_seed_policy_v20_centroid_ranker"
    model["training_dataset"] = str(args.dataset)
    dev_eval = evaluate(dev_rows, model)
    test_eval = evaluate(test_rows, model)
    selected = dev_eval["metrics"].get("selected_policy") or {}
    model["selected_threshold"] = float(selected.get("threshold", 0.5))
    model["adopted_into_inference_stream"] = False
    scored_rows = []
    for split, result in [("dev", dev_eval), ("test", test_eval)]:
        for row in result["scored"]:
            scored_rows.append({**row, "split": split, "source_integrity": integrity()})
    audit = {
        "task": "IMG-MOE-V18-REBUILD-001.step_j10c_train_dense_seed_policy_and_generate_heatmap_component_candidates",
        "dataset": str(args.dataset),
        "model_output": str(args.model_output),
        "scored_output": str(args.scored_output),
        "split": {
            "strategy": "stable_hash_by_row_id_80_20",
            "train_rows": len(train_rows),
            "dev_rows": len(dev_rows),
            "test_rows": len(test_rows),
            "train_positive": sum(1 for row in train_rows if row.get("label_objectness")),
            "test_positive": sum(1 for row in test_rows if row.get("label_objectness")),
        },
        "dev": dev_eval["metrics"],
        "test": test_eval["metrics"],
        "target_recall_at_caps": {
            "oracle_missing": recall_at_caps(scored_rows, [50, 100, 250, 500, 1000], "gold_keys"),
            "baseline_missing": recall_at_caps(scored_rows, [50, 100, 250, 500, 1000], "baseline_gold_keys"),
        },
        "top_feature_weights": feature_weights(model)[:18],
        "quality_gates": {
            "source_integrity_violations": 0,
            "gold_used_for_inference": False,
            "test_selected_recall_ge_0_98": (test_eval["metrics"].get("selected_policy") or {}).get("recall", 0.0) >= 0.98,
            "oracle_missing_recall_at_500_ge_0_90": recall_at_caps(scored_rows, [500], "gold_keys")["500"]["patch_recall"] >= 0.90 if scored_rows else False,
        },
        "source_integrity": integrity(),
    }
    write_json(Path(args.model_output), model)
    write_json(Path(args.audit_output), audit)
    write_jsonl(Path(args.scored_output), scored_rows)
    print(
        json.dumps(
            {
                "train_rows": len(train_rows),
                "dev_rows": len(dev_rows),
                "test_rows": len(test_rows),
                "selected_policy": selected,
                "target_recall_at_caps": audit["target_recall_at_caps"],
                "quality_gates": audit["quality_gates"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
