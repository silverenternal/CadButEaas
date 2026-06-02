#!/usr/bin/env python3
"""Audit within-page ranking gaps between v74 selected duplicates and oracle new-IoU actions."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from typing import Any

import joblib
import numpy as np

from audit_symbol_v74_gap_to_070_v78 import base_selected_by_page, hit_targets, recovery_index
from apply_symbol_budgeted_additive_rescue_v65 import page_gold_count
from apply_symbol_detector_recall_preserving_policy_v47 import group_pages
from train_symbol_expanded_action_source_policy_v74 import candidate_id, feature_names, vector
from train_symbol_support_suppression_v35 import load_jsonl, source_path
from train_symbol_tile_detector_v20 import write_json, write_jsonl

NEW_BUCKET = "new_iou_target"
DUP_BUCKETS = {"duplicate_iou_target", "duplicate_center_only_target"}
NUMERIC_FIELDS = [
    "v74_score",
    "score",
    "feature_score",
    "rescue_score",
    "cluster_rank",
    "cluster_size",
    "cluster_score_max",
    "cluster_score_mean",
    "page_candidate_count",
    "width_norm",
    "height_norm",
    "area_norm",
    "aspect",
    "nearest_selected_iou",
    "nearest_same_label_iou",
    "nearest_selected_center_dist",
    "same_label_score_margin",
    "same_area_score_margin",
    "same_cluster_score_margin",
    "overlap_selected_count_iou_0_10",
    "overlap_selected_count_iou_0_30",
]


def bucket(row: dict[str, Any]) -> str:
    return str(row.get("bucket") or "unknown")


def reason(row: dict[str, Any]) -> str:
    return str(row.get("source_gap_reason") or "unknown")


def num(row: dict[str, Any], name: str) -> float:
    if name == "v74_score":
        return float(row.get("v74_score") or 0.0)
    if name in {"score", "feature_score", "rescue_score", "cluster_rank"}:
        return float(row.get(name) or (999.0 if name == "cluster_rank" else 0.0))
    return float((row.get("features") or {}).get(name) or 0.0)


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": round(float(arr.mean()), 6),
        "p10": round(float(np.quantile(arr, 0.10)), 6),
        "p50": round(float(np.quantile(arr, 0.50)), 6),
        "p90": round(float(np.quantile(arr, 0.90)), 6),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "bucket_counts": dict(Counter(bucket(row) for row in rows)),
        "reason_counts": dict(Counter(reason(row) for row in rows).most_common()),
        "label_counts": dict(Counter(str(row.get("label") or "unknown") for row in rows).most_common()),
        "area_counts": dict(Counter(str(row.get("area") or "unknown") for row in rows).most_common()),
        "numeric": {name: quantiles([num(row, name) for row in rows]) for name in NUMERIC_FIELDS},
    }


def score_actions(rows: list[dict[str, Any]], model: Any, names: list[str]) -> list[tuple[dict[str, Any], float]]:
    if not rows:
        return []
    scores = model.predict_proba(np.asarray([vector(row, names) for row in rows], dtype=np.float32))[:, 1]
    out = []
    for row, score in zip(rows, scores, strict=True):
        item = dict(row)
        item["v74_score"] = float(score)
        out.append((item, float(score)))
    return out


def eligible_actions(
    action_rows: list[dict[str, Any]],
    model: Any,
    names: list[str],
    split: str,
    threshold: float,
) -> dict[str, list[tuple[dict[str, Any], float]]]:
    split_rows = [row for row in action_rows if str(row.get("split") or "") == split]
    by_page: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for row, score in score_actions(split_rows, model, names):
        if bucket(row) == "new_center_only_target" or score < threshold:
            continue
        by_page[str(row.get("page_id") or "")].append((row, score))
    return by_page


def plan_page(
    page_id: str,
    scored_actions: list[tuple[dict[str, Any], float]],
    base_ids: set[str],
    base_hits: set[str],
    max_add_per_page: int,
    oracle: bool,
) -> tuple[list[dict[str, Any]], Counter]:
    added: list[dict[str, Any]] = []
    audit = Counter()
    seen_targets: set[str] = set()
    if oracle:
        ordered = sorted(scored_actions, key=lambda item: (bucket(item[0]) == NEW_BUCKET, item[1]), reverse=True)
    else:
        ordered = sorted(scored_actions, key=lambda item: item[1], reverse=True)
    for row, _score in ordered:
        cid = candidate_id(row)
        target = str(row.get("target_id") or "")
        if not cid or cid in base_ids or not target or target in seen_targets:
            continue
        if oracle and bucket(row) == NEW_BUCKET and target in base_hits:
            continue
        item = dict(row)
        item["page_id"] = page_id
        added.append(item)
        seen_targets.add(target)
        audit["added"] += 1
        audit[f"added_bucket:{bucket(row)}"] += 1
        audit[f"added_reason:{reason(row)}"] += 1
        if len(added) >= max_add_per_page:
            break
    return added, audit


def split_pairs(
    action_rows: list[dict[str, Any]],
    recovery_rows: list[dict[str, Any]],
    model: Any,
    names: list[str],
    split: str,
    threshold: float,
    max_add_per_page: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base = base_selected_by_page(recovery_rows, split)
    pages = group_pages(recovery_rows, split)
    scored_by_page = eligible_actions(action_rows, model, names, split, threshold)
    page_summaries: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    positive_rows: list[dict[str, Any]] = []
    negative_rows: list[dict[str, Any]] = []
    for page_id, selected in base.items():
        base_ids = {candidate_id(row) for row in selected}
        v74_added, v74_audit = plan_page(page_id, scored_by_page.get(page_id, []), base_ids, set(), max_add_per_page, oracle=False)
        oracle_added, oracle_audit = plan_page(page_id, scored_by_page.get(page_id, []), base_ids, hit_targets(selected), max_add_per_page, oracle=True)
        v74_duplicate = [row for row in v74_added if bucket(row) in DUP_BUCKETS]
        oracle_new = [row for row in oracle_added if bucket(row) == NEW_BUCKET and candidate_id(row) not in {candidate_id(x) for x in v74_added}]
        if not oracle_new or not v74_duplicate:
            continue
        page_summaries.append({
            "page_id": page_id,
            "gold_count": page_gold_count(pages.get(page_id, [])),
            "base_selected": len(selected),
            "eligible_actions": len(scored_by_page.get(page_id, [])),
            "v74_audit": dict(v74_audit),
            "oracle_audit": dict(oracle_audit),
            "missed_oracle_new": len(oracle_new),
            "v74_duplicates": len(v74_duplicate),
        })
        positive_rows.extend(oracle_new)
        negative_rows.extend(v74_duplicate)
        for pos in oracle_new:
            for neg in v74_duplicate:
                deltas = {name: round(num(pos, name) - num(neg, name), 6) for name in NUMERIC_FIELDS}
                pairs.append({
                    "split": split,
                    "page_id": page_id,
                    "positive": slim_action(pos),
                    "negative": slim_action(neg),
                    "delta_positive_minus_negative": deltas,
                })
    report = {
        "split": split,
        "pages_with_pairs": len(page_summaries),
        "pairs": len(pairs),
        "oracle_new_actions": len(positive_rows),
        "v74_duplicate_actions": len(negative_rows),
        "page_summary": {
            "gold_count": quantiles([float(row["gold_count"]) for row in page_summaries]),
            "eligible_actions": quantiles([float(row["eligible_actions"]) for row in page_summaries]),
            "missed_oracle_new": quantiles([float(row["missed_oracle_new"]) for row in page_summaries]),
            "v74_duplicates": quantiles([float(row["v74_duplicates"]) for row in page_summaries]),
        },
        "positive_oracle_new": summarize_rows(positive_rows),
        "negative_v74_duplicates": summarize_rows(negative_rows),
        "pair_delta_positive_minus_negative": {name: quantiles([pair["delta_positive_minus_negative"][name] for pair in pairs]) for name in NUMERIC_FIELDS},
        "top_pages": sorted(page_summaries, key=lambda row: (row["missed_oracle_new"], row["v74_duplicates"], row["eligible_actions"]), reverse=True)[:50],
    }
    return report, pairs


def slim_action(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id(row),
        "target_id": str(row.get("target_id") or ""),
        "bucket": bucket(row),
        "reason": reason(row),
        "label": row.get("label"),
        "area": row.get("area"),
        "v74_score": round(num(row, "v74_score"), 6),
        "score": round(num(row, "score"), 6),
        "cluster_rank": round(num(row, "cluster_rank"), 6),
        "nearest_selected_iou": round(num(row, "nearest_selected_iou"), 6),
        "nearest_same_label_iou": round(num(row, "nearest_same_label_iou"), 6),
        "same_label_score_margin": round(num(row, "same_label_score_margin"), 6),
        "same_area_score_margin": round(num(row, "same_area_score_margin"), 6),
        "same_cluster_score_margin": round(num(row, "same_cluster_score_margin"), 6),
        "overlap_selected_count_iou_0_30": round(num(row, "overlap_selected_count_iou_0_30"), 6),
    }


def decision(split_reports: dict[str, Any]) -> dict[str, Any]:
    smoke = split_reports.get("smoke_eval") or {}
    deltas = smoke.get("pair_delta_positive_minus_negative") or {}
    nearest = (deltas.get("nearest_selected_iou") or {}).get("p50", 0.0)
    score = (deltas.get("v74_score") or {}).get("p50", 0.0)
    rank = (deltas.get("cluster_rank") or {}).get("p50", 0.0)
    overlap = (deltas.get("overlap_selected_count_iou_0_30") or {}).get("p50", 0.0)
    clear = bool(nearest < -0.05 or score > 0.05 or rank < -1.0 or overlap < -1.0)
    return {
        "runtime_safe_separability_signal": "present" if clear else "weak",
        "smoke_pages_with_pairs": int(smoke.get("pages_with_pairs") or 0),
        "smoke_pairs": int(smoke.get("pairs") or 0),
        "key_median_deltas_positive_minus_negative": {
            "v74_score": score,
            "cluster_rank": rank,
            "nearest_selected_iou": nearest,
            "overlap_selected_count_iou_0_30": overlap,
        },
        "recommendation": "try_v82_within_page_reranker" if clear else "freeze_v74_or_add_supervision",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actions", default="datasets/symbol_expanded_action_source_policy_v74/actions.jsonl")
    parser.add_argument("--recovery-data", default="datasets/symbol_detector_listwise_recovery_v47_p2_transfer_locked_full_v2/manifest.json")
    parser.add_argument("--model", default="checkpoints/symbol_expanded_action_source_policy_v74/model.joblib")
    parser.add_argument("--threshold", type=float, default=0.04)
    parser.add_argument("--max-add-per-page", type=int, default=10)
    parser.add_argument("--splits", default="dev,smoke_eval")
    parser.add_argument("--output", default="reports/vlm/symbol_within_page_ranking_v81_audit.json")
    parser.add_argument("--pairs-output", default="reports/vlm/symbol_within_page_ranking_v81_pairs.jsonl")
    args = parser.parse_args()

    action_rows = load_jsonl(source_path(args.actions))
    recovery_manifest = json.loads(source_path(args.recovery_data).read_text(encoding="utf-8"))
    recovery_rows = load_jsonl(source_path(recovery_manifest["outputs"]["rows"]))
    bundle = joblib.load(source_path(args.model))
    model = bundle["model"]
    names = bundle.get("feature_names") or feature_names()

    split_reports: dict[str, Any] = {}
    all_pairs: list[dict[str, Any]] = []
    for split in [part.strip() for part in args.splits.split(",") if part.strip()]:
        report, pairs = split_pairs(action_rows, recovery_rows, model, names, split, args.threshold, args.max_add_per_page)
        split_reports[split] = report
        all_pairs.extend(pairs)

    output = {
        "version": "symbol_within_page_ranking_v81",
        "inputs": {"actions": args.actions, "recovery_data": args.recovery_data, "model": args.model},
        "policy_context": {"threshold": args.threshold, "max_add_per_page": args.max_add_per_page},
        "splits": split_reports,
        "decision": decision(split_reports),
        "source_integrity": {
            "gold_used_for_inference": False,
            "runtime_uses_svg_or_cad_geometry": False,
            "offline_labels_used_for": ["within_page_oracle_audit", "new_iou_vs_duplicate_pair_labels"],
            "final_quality_claim_allowed": False,
        },
    }
    write_json(source_path(args.output), output)
    write_jsonl(source_path(args.pairs_output), all_pairs)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
