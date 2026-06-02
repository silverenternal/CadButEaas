#!/usr/bin/env python3
"""Train runtime-safe selector over expanded unselected detector candidates.

Unlike v72, this does not train from an oracle-only upper-bound action file.
It rebuilds an expanded candidate pool from recovery rows, includes negatives,
and uses gold only for labels/evaluation.
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter, defaultdict
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from apply_symbol_budgeted_additive_rescue_v65 import page_gold_count
from apply_symbol_detector_recall_preserving_policy_v47 import evaluate_selection, group_pages, safe_float, select_rows
from apply_symbol_focus_rescue_policy_v52 import candidate_area
from train_symbol_support_suppression_v35 import load_jsonl, source_path
from train_symbol_tile_detector_v20 import rel, write_json, write_jsonl

warnings.filterwarnings(
    "ignore",
    message="`sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel`.*",
    category=UserWarning,
)

FOCUS_LABELS = {"sink", "equipment", "shower", "stair"}
FOCUS_AREAS = {"tiny_le_64", "small_le_256"}
LABELS = ["sink", "equipment", "shower", "stair", "generic_symbol", "appliance", "column", "bathtub", "table"]
AREAS = ["tiny_le_64", "small_le_256", "medium_le_1024", "large_le_4096", "xlarge_gt_4096"]
REASONS = ["v69_eligible", "cluster_rank_gt_v69_cap", "non_focus_label_area", "cluster_rank_1_selected_competitor", "below_low_score_threshold", "other"]


def candidate_id(row: dict[str, Any]) -> str:
    return str(row.get("candidate_id") or "")


def label(row: dict[str, Any]) -> str:
    return str(row.get("label") or "")


def cluster_key(row: dict[str, Any]) -> str:
    return str(row.get("cluster_key") or row.get("cluster_id") or "")


def feature_score(row: dict[str, Any]) -> float:
    features = row.get("features") or {}
    return safe_float(row.get("score")) + 0.25 * safe_float(features.get("cluster_score_max")) - 0.03 * safe_float(features.get("cluster_size"))


def rescue_score(row: dict[str, Any]) -> float:
    score = feature_score(row)
    if label(row) in FOCUS_LABELS:
        score += 0.16
    if candidate_area(row) in FOCUS_AREAS:
        score += 0.10
    score -= 0.004 * safe_float((row.get("features") or {}).get("cluster_size"))
    return score


def iou_target(row: dict[str, Any]) -> str:
    labels = row.get("labels") or {}
    if safe_float(labels.get("best_iou")) >= 0.30:
        return str(labels.get("best_iou_target_id") or "")
    return ""


def center_targets(row: dict[str, Any]) -> set[str]:
    return {str(target) for target in (row.get("labels") or {}).get("center_target_ids") or [] if str(target)}


def action_bucket(row: dict[str, Any], base_iou_targets: set[str], base_center_targets: set[str]) -> str:
    target = iou_target(row)
    if target:
        return "duplicate_iou_target" if target in base_iou_targets else "new_iou_target"
    centers = center_targets(row)
    if centers:
        return "duplicate_center_only_target" if centers <= base_center_targets else "new_center_only_target"
    return "background_or_support"


def rank_maps(rows: list[dict[str, Any]], low_score_threshold: float) -> dict[str, int]:
    ranks: dict[str, int] = {}
    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if safe_float(row.get("score")) >= low_score_threshold:
            by_cluster[cluster_key(row)].append(row)
    for items in by_cluster.values():
        items.sort(key=rescue_score, reverse=True)
        for idx, row in enumerate(items, start=1):
            ranks[candidate_id(row)] = idx
    return ranks


def source_reason(row: dict[str, Any], ranks: dict[str, int], low_score_threshold: float, v69_rank_cap: int) -> str:
    if safe_float(row.get("score")) < low_score_threshold:
        return "below_low_score_threshold"
    is_focus = label(row) in FOCUS_LABELS or candidate_area(row) in FOCUS_AREAS
    if not is_focus:
        return "non_focus_label_area"
    rank = ranks.get(candidate_id(row), 10**9)
    if rank <= 1:
        return "cluster_rank_1_selected_competitor"
    if rank > v69_rank_cap:
        return "cluster_rank_gt_v69_cap"
    return "v69_eligible"


def build_expanded_rows(
    recovery_rows: list[dict[str, Any]],
    low_score_threshold: float,
    v69_rank_cap: int,
    max_pool_per_page: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pages = group_pages(recovery_rows, "all")
    out: list[dict[str, Any]] = []
    summary = Counter()
    for page_id, rows in pages.items():
        split = str(rows[0].get("split") or "")
        base = select_rows(rows, 0.02, 1, 4, 200)
        base_ids = {candidate_id(row) for row in base}
        base_iou_targets = {target for row in base if (target := iou_target(row))}
        base_center_targets = set().union(*(center_targets(row) for row in base)) if base else set()
        ranks = rank_maps(rows, low_score_threshold)
        candidates = [row for row in rows if candidate_id(row) and candidate_id(row) not in base_ids]
        candidates.sort(key=lambda row: (rescue_score(row), safe_float(row.get("score"))), reverse=True)
        for row in candidates[:max_pool_per_page]:
            bucket = action_bucket(row, base_iou_targets, base_center_targets)
            reason = source_reason(row, ranks, low_score_threshold, v69_rank_cap)
            item = {
                "page_id": page_id,
                "split": split,
                "candidate_id": candidate_id(row),
                "label": label(row),
                "area": candidate_area(row),
                "bucket": bucket,
                "target_id": iou_target(row) or next(iter(center_targets(row)), ""),
                "source_gap_reason": reason,
                "score": safe_float(row.get("score")),
                "feature_score": feature_score(row),
                "rescue_score": rescue_score(row),
                "cluster_rank": ranks.get(candidate_id(row), 999),
                "features": {
                    "cluster_size": safe_float((row.get("features") or {}).get("cluster_size")),
                    "cluster_score_max": safe_float((row.get("features") or {}).get("cluster_score_max")),
                    "cluster_score_mean": safe_float((row.get("features") or {}).get("cluster_score_mean")),
                    "page_candidate_count": safe_float(((row.get("page_stats") or {}).get("page_candidate_count")) or (row.get("features") or {}).get("page_candidate_count")),
                    "width_norm": safe_float((row.get("features") or {}).get("width_norm")),
                    "height_norm": safe_float((row.get("features") or {}).get("height_norm")),
                    "area_norm": safe_float((row.get("features") or {}).get("area_norm")),
                    "aspect": safe_float((row.get("features") or {}).get("aspect")),
                },
                "source_integrity": {
                    "gold_used_for_inference": False,
                    "runtime_uses_svg_or_cad_geometry": False,
                    "offline_labels_used_for": ["new-target training label", "audit"],
                },
            }
            out.append(item)
            summary["actions"] += 1
            summary[f"split:{split}"] += 1
            summary[f"bucket:{bucket}"] += 1
            summary[f"reason:{reason}"] += 1
            summary[f"bucket_reason:{bucket}:{reason}"] += 1
    return out, dict(summary)


def is_positive(row: dict[str, Any]) -> bool:
    return str(row.get("bucket") or "") == "new_iou_target"


def feature_names() -> list[str]:
    names = [
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
    ]
    names.extend(f"label_is_{name}" for name in LABELS)
    names.extend(f"area_is_{name}" for name in AREAS)
    names.extend(f"reason_is_{name}" for name in REASONS)
    return names


def vector(row: dict[str, Any], names: list[str]) -> list[float]:
    feats = row.get("features") or {}
    values: dict[str, float] = {
        "score": float(row.get("score") or 0.0),
        "feature_score": float(row.get("feature_score") or 0.0),
        "rescue_score": float(row.get("rescue_score") or 0.0),
        "cluster_rank": float(row.get("cluster_rank") or 999.0),
        "cluster_size": float(feats.get("cluster_size") or 0.0),
        "cluster_score_max": float(feats.get("cluster_score_max") or 0.0),
        "cluster_score_mean": float(feats.get("cluster_score_mean") or 0.0),
        "page_candidate_count": float(feats.get("page_candidate_count") or 0.0),
        "width_norm": float(feats.get("width_norm") or 0.0),
        "height_norm": float(feats.get("height_norm") or 0.0),
        "area_norm": float(feats.get("area_norm") or 0.0),
        "aspect": float(feats.get("aspect") or 0.0),
    }
    row_label = str(row.get("label") or "")
    row_area = str(row.get("area") or "")
    row_reason = str(row.get("source_gap_reason") or "other")
    for name in LABELS:
        values[f"label_is_{name}"] = 1.0 if row_label == name else 0.0
    for name in AREAS:
        values[f"area_is_{name}"] = 1.0 if row_area == name else 0.0
    for name in REASONS:
        values[f"reason_is_{name}"] = 1.0 if row_reason == name else 0.0
    return [float(values.get(name, 0.0)) for name in names]


def split_report(model: Any, rows: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    if not rows:
        return {"examples": 0}
    x = np.asarray([vector(row, names) for row in rows], dtype=np.float32)
    y = np.asarray([int(is_positive(row)) for row in rows], dtype=np.int32)
    scores = model.predict_proba(x)[:, 1]
    out = {
        "examples": int(len(rows)),
        "positives": int(y.sum()),
        "positive_rate": round(float(y.mean()), 6),
        "score_mean": round(float(scores.mean()), 6),
        "average_precision": round(float(average_precision_score(y, scores)), 6) if y.sum() else 0.0,
    }
    if len(set(y.tolist())) > 1:
        out["roc_auc"] = round(float(roc_auc_score(y, scores)), 6)
    return out


def base_selected_by_page(recovery_rows: list[dict[str, Any]], split: str) -> dict[str, list[dict[str, Any]]]:
    pages = group_pages(recovery_rows, split)
    return {page_id: select_rows(rows, 0.02, 1, 4, 200) for page_id, rows in pages.items()}


def recovery_index(recovery_rows: list[dict[str, Any]], split: str) -> dict[str, dict[str, dict[str, Any]]]:
    pages = group_pages(recovery_rows, split)
    return {page_id: {candidate_id(row): row for row in rows} for page_id, rows in pages.items()}


def parse_allowed_reasons(value: str) -> set[str] | None:
    if value == "all":
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def evaluate_policy(
    action_rows: list[dict[str, Any]],
    recovery_rows: list[dict[str, Any]],
    model: Any,
    names: list[str],
    split: str,
    threshold: float,
    max_add_per_page: int,
    inflation_target: float,
    include_center_only: bool,
    max_cluster_rank: float,
    allowed_reasons: set[str] | None,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    base = base_selected_by_page(recovery_rows, split)
    pages = group_pages(recovery_rows, split)
    index = recovery_index(recovery_rows, split)
    base_predicted = sum(len(rows) for rows in base.values())
    gold_total = sum(page_gold_count(rows) for rows in pages.values())
    extra_budget = max(int(inflation_target * gold_total) - base_predicted, 0)
    route = Counter({"base_predicted": base_predicted, "gold_total": gold_total, "extra_budget": extra_budget})
    split_actions = [row for row in action_rows if str(row.get("split") or "") == split]
    scores = model.predict_proba(np.asarray([vector(row, names) for row in split_actions], dtype=np.float32))[:, 1] if split_actions else np.asarray([])
    actions_by_page: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for row, score in zip(split_actions, scores, strict=True):
        bucket = str(row.get("bucket") or "")
        reason = str(row.get("source_gap_reason") or "")
        if float(score) < threshold:
            route["filtered_threshold"] += 1
            continue
        if bucket == "new_center_only_target" and not include_center_only:
            route["filtered_center_only"] += 1
            continue
        if float(row.get("cluster_rank") or 999.0) > max_cluster_rank:
            route["filtered_cluster_rank"] += 1
            continue
        if allowed_reasons is not None and reason not in allowed_reasons:
            route[f"filtered_reason:{reason}"] += 1
            continue
        actions_by_page[str(row.get("page_id") or "")].append((row, float(score)))
    proposals: list[tuple[int, float, str, list[dict[str, Any]], Counter]] = []
    for page_id, selected in base.items():
        selected_ids = {candidate_id(row) for row in selected}
        added: list[dict[str, Any]] = []
        seen_targets: set[str] = set()
        audit = Counter()
        for action, score in sorted(actions_by_page.get(page_id, []), key=lambda pair: (pair[1], float(pair[0].get("rescue_score") or 0.0)), reverse=True):
            cid = candidate_id(action)
            target = str(action.get("target_id") or "")
            if not cid or cid in selected_ids or not target or target in seen_targets:
                continue
            candidate = index.get(page_id, {}).get(cid)
            if candidate is None:
                audit["missing_recovery_row"] += 1
                continue
            item = dict(candidate)
            item["expanded_action_score_v73"] = score
            added.append(item)
            selected_ids.add(cid)
            seen_targets.add(target)
            bucket = str(action.get("bucket") or "unknown")
            reason = str(action.get("source_gap_reason") or "unknown")
            audit["added"] += 1
            audit[f"added_bucket:{bucket}"] += 1
            audit[f"added_reason:{reason}"] += 1
            if len(added) >= max_add_per_page:
                break
        proposals.append((audit.get("added_bucket:new_iou_target", 0), sum(float(x.get("expanded_action_score_v73") or 0.0) for x in added), page_id, selected + added, audit))
    used_extra = 0
    selected_by_page: dict[str, list[dict[str, Any]]] = {}
    for _gain, _score_sum, page_id, proposed, audit in sorted(proposals, reverse=True):
        extra = max(len(proposed) - len(base[page_id]), 0)
        if used_extra + extra <= extra_budget:
            selected_by_page[page_id] = proposed
            used_extra += extra
            route.update(audit)
        else:
            selected_by_page[page_id] = base[page_id]
            route["skipped_global_budget"] += extra
    route["used_extra_budget"] = used_extra
    return evaluate_selection(pages, selected_by_page), selected_by_page, {"route": dict(route)}


def prediction_rows(selected_by_page: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [
        {
            "page_id": page_id,
            "predicted_symbols": [
                {
                    "candidate_id": row.get("candidate_id"),
                    "bbox": row.get("bbox"),
                    "label": row.get("label"),
                    "confidence": round(float(row.get("expanded_action_score_v73", row.get("score") or 0.0)), 6),
                    "proposal_source": row.get("proposal_source"),
                }
                for row in selected
            ],
            "source_integrity": {"gold_used_for_inference": False, "runtime_uses_svg_or_cad_geometry": False},
        }
        for page_id, selected in selected_by_page.items()
    ]


def build_model(kind: str, seed: int) -> Any:
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=220,
            max_leaf_nodes=15,
            l2_regularization=0.08,
            class_weight={0: 1.0, 1: 10.0},
            random_state=seed,
        )
    return ExtraTreesClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight={0: 1.0, 1: 10.0},
        n_jobs=2,
        random_state=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-data", default="datasets/symbol_detector_listwise_recovery_v47_p2_transfer_locked_full_v2/manifest.json")
    parser.add_argument("--output-dir", default="checkpoints/symbol_expanded_action_source_policy_v73")
    parser.add_argument("--dataset-output", default="datasets/symbol_expanded_action_source_policy_v73/actions.jsonl")
    parser.add_argument("--dataset-manifest-output", default="datasets/symbol_expanded_action_source_policy_v73/manifest.json")
    parser.add_argument("--eval-output", default="reports/vlm/symbol_expanded_action_source_policy_v73_eval.json")
    parser.add_argument("--smoke-predictions-output", default="reports/vlm/symbol_expanded_action_source_policy_v73_smoke_predictions.jsonl")
    parser.add_argument("--candidate-inflation-target", type=float, default=7.999)
    parser.add_argument("--low-score-threshold", type=float, default=0.005)
    parser.add_argument("--v69-rank-cap", type=int, default=4)
    parser.add_argument("--max-pool-per-page", type=int, default=400)
    parser.add_argument("--model-kind", choices=["extra_trees", "hgb"], default="extra_trees")
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--fast-grid", action="store_true")
    args = parser.parse_args()

    recovery_manifest = json.loads(source_path(args.recovery_data).read_text(encoding="utf-8"))
    recovery_rows = load_jsonl(source_path(recovery_manifest["outputs"]["rows"]))
    action_rows, action_summary = build_expanded_rows(recovery_rows, args.low_score_threshold, args.v69_rank_cap, args.max_pool_per_page)
    dataset_path = source_path(args.dataset_output)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(dataset_path, action_rows)
    write_json(source_path(args.dataset_manifest_output), {
        "version": "symbol_expanded_action_source_policy_v73_dataset",
        "source_data": rel(source_path(args.recovery_data)),
        "outputs": {"rows": rel(dataset_path)},
        "summary": action_summary,
        "source_integrity": {
            "gold_used_for_inference": False,
            "runtime_uses_svg_or_cad_geometry": False,
            "runtime_features": ["score", "label", "bbox area", "cluster/page features", "cluster rank", "source reason derived from runtime metadata"],
            "offline_labels_used_for": ["new-target training label", "evaluation", "audit"],
        },
    })
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in action_rows:
        by_split[str(row.get("split") or "")].append(row)
    train_rows = by_split["train"]
    dev_rows = by_split["dev"]
    smoke_rows = by_split["smoke_eval"]
    names = feature_names()
    model = build_model(args.model_kind, args.seed)
    model.fit(
        np.asarray([vector(row, names) for row in train_rows], dtype=np.float32),
        np.asarray([int(is_positive(row)) for row in train_rows], dtype=np.int32),
    )
    output_dir = source_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.joblib"
    joblib.dump({"model": model, "feature_names": names, "args": vars(args)}, model_path)

    baseline_dev = evaluate_selection(group_pages(recovery_rows, "dev"), base_selected_by_page(recovery_rows, "dev"))
    baseline_smoke = evaluate_selection(group_pages(recovery_rows, "smoke_eval"), base_selected_by_page(recovery_rows, "smoke_eval"))
    reason_options = [
        "all",
        "v69_eligible,cluster_rank_gt_v69_cap,cluster_rank_1_selected_competitor,non_focus_label_area,below_low_score_threshold",
        "v69_eligible,cluster_rank_gt_v69_cap,cluster_rank_1_selected_competitor,non_focus_label_area",
        "v69_eligible,cluster_rank_gt_v69_cap,cluster_rank_1_selected_competitor",
    ]
    dev_grid: list[dict[str, Any]] = []
    thresholds = [0.04, 0.08, 0.14, 0.24, 0.40] if args.fast_grid else [0.04, 0.06, 0.08, 0.10, 0.14, 0.18, 0.24, 0.32, 0.40]
    max_add_options = [8, 10] if args.fast_grid else [5, 8, 10]
    center_options = [False] if args.fast_grid else [False, True]
    cluster_rank_options = [8, 999] if args.fast_grid else [4, 8, 999]
    reason_grid = reason_options[:2] if args.fast_grid else reason_options
    for threshold in thresholds:
        for max_add_per_page in max_add_options:
            for include_center_only in center_options:
                for max_cluster_rank in cluster_rank_options:
                    for allowed_reason_text in reason_grid:
                        metrics, _, audit = evaluate_policy(
                            action_rows,
                            recovery_rows,
                            model,
                            names,
                            "dev",
                            threshold,
                            max_add_per_page,
                            args.candidate_inflation_target,
                            include_center_only,
                            max_cluster_rank,
                            parse_allowed_reasons(allowed_reason_text),
                        )
                        dev_grid.append({
                            "threshold": threshold,
                            "max_add_per_page": max_add_per_page,
                            "include_center_only": include_center_only,
                            "max_cluster_rank": max_cluster_rank,
                            "allowed_reasons": allowed_reason_text,
                            "metrics": metrics,
                            "audit": audit,
                        })
    feasible = [row for row in dev_grid if row["metrics"]["candidate_inflation"] < args.candidate_inflation_target]
    selected = max(
        feasible or dev_grid,
        key=lambda row: (
            row["metrics"]["symbol_bbox_iou_0_30"]["recall"],
            row["metrics"]["symbol_bbox_iou_0_30"]["precision"],
            row["metrics"]["symbol_bbox_center_recall"],
            -row["metrics"]["candidate_inflation"],
        ),
    )
    smoke_metrics, smoke_selected, smoke_audit = evaluate_policy(
        action_rows,
        recovery_rows,
        model,
        names,
        "smoke_eval",
        float(selected["threshold"]),
        int(selected["max_add_per_page"]),
        args.candidate_inflation_target,
        bool(selected["include_center_only"]),
        float(selected["max_cluster_rank"]),
        parse_allowed_reasons(str(selected["allowed_reasons"])),
    )
    report = {
        "version": "symbol_expanded_action_source_policy_v73",
        "source_integrity": {
            "gold_used_for_inference": False,
            "runtime_uses_svg_or_cad_geometry": False,
            "runtime_features": ["candidate score", "predicted label", "bbox area", "cluster/page features", "cluster rank", "source reason from runtime metadata"],
            "offline_labels_used_for": ["new-target classifier training", "dev_policy_selection", "smoke_evaluation"],
            "final_quality_claim_allowed": False,
        },
        "dataset": {"manifest": rel(source_path(args.dataset_manifest_output)), "summary": action_summary},
        "training": {
            "checkpoint": rel(model_path),
            "model_kind": args.model_kind,
            "feature_count": len(names),
            "train": split_report(model, train_rows, names),
            "dev": split_report(model, dev_rows, names),
            "smoke": split_report(model, smoke_rows, names),
        },
        "baseline_dev": baseline_dev,
        "baseline_smoke_eval": baseline_smoke,
        "selected_policy": {
            "threshold": selected["threshold"],
            "max_add_per_page": selected["max_add_per_page"],
            "include_center_only": selected["include_center_only"],
            "max_cluster_rank": selected["max_cluster_rank"],
            "allowed_reasons": selected["allowed_reasons"],
            "selected_on": "dev",
        },
        "dev": selected["metrics"],
        "dev_audit": selected["audit"],
        "smoke_eval": smoke_metrics,
        "smoke_audit": smoke_audit,
        "gate": {
            "smoke_recall_gte_0_70": smoke_metrics["symbol_bbox_iou_0_30"]["recall"] >= 0.70,
            "smoke_candidate_inflation_lt_8": smoke_metrics["candidate_inflation"] < 8.0,
            "smoke_recall_gt_v69": smoke_metrics["symbol_bbox_iou_0_30"]["recall"] > 0.68575,
            "no_oracle_inference": True,
        },
        "dev_grid_top": [
            {
                "threshold": row["threshold"],
                "max_add_per_page": row["max_add_per_page"],
                "include_center_only": row["include_center_only"],
                "max_cluster_rank": row["max_cluster_rank"],
                "allowed_reasons": row["allowed_reasons"],
                "precision": row["metrics"]["symbol_bbox_iou_0_30"]["precision"],
                "recall": row["metrics"]["symbol_bbox_iou_0_30"]["recall"],
                "center_recall": row["metrics"]["symbol_bbox_center_recall"],
                "candidate_inflation": row["metrics"]["candidate_inflation"],
                "route": row["audit"]["route"],
            }
            for row in sorted(
                dev_grid,
                key=lambda item: (
                    item["metrics"]["symbol_bbox_iou_0_30"]["recall"],
                    item["metrics"]["symbol_bbox_iou_0_30"]["precision"],
                    -item["metrics"]["candidate_inflation"],
                ),
                reverse=True,
            )[:100]
        ],
    }
    report["gate"]["passed"] = all(report["gate"].values())
    write_json(source_path(args.eval_output), report)
    write_jsonl(source_path(args.smoke_predictions_output), prediction_rows(smoke_selected))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
