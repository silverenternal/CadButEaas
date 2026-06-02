#!/usr/bin/env python3
"""Train v77 hard-negative reranker for expanded action source.

Goal: separate new_iou_target actions from duplicate_iou/duplicate_center/background
hard negatives in the v74 expanded action pool, then select under the same budget.
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
from apply_symbol_detector_recall_preserving_policy_v47 import evaluate_selection, group_pages, select_rows
from train_symbol_expanded_action_source_policy_v74 import candidate_id, feature_names, vector
from train_symbol_support_suppression_v35 import load_jsonl, source_path
from train_symbol_tile_detector_v20 import rel, write_json, write_jsonl

warnings.filterwarnings(
    "ignore",
    message="`sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel`.*",
    category=UserWarning,
)


def bucket(row: dict[str, Any]) -> str:
    return str(row.get("bucket") or "unknown")


def is_positive(row: dict[str, Any]) -> bool:
    return bucket(row) == "new_iou_target"


def is_hard_negative(row: dict[str, Any]) -> bool:
    return bucket(row) in {"duplicate_iou_target", "duplicate_center_only_target", "background_or_support"}


def base_selected_by_page(recovery_rows: list[dict[str, Any]], split: str) -> dict[str, list[dict[str, Any]]]:
    pages = group_pages(recovery_rows, split)
    return {page_id: select_rows(rows, 0.02, 1, 4, 200) for page_id, rows in pages.items()}


def recovery_index(recovery_rows: list[dict[str, Any]], split: str) -> dict[str, dict[str, dict[str, Any]]]:
    pages = group_pages(recovery_rows, split)
    return {page_id: {candidate_id(row): row for row in rows} for page_id, rows in pages.items()}


def split_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("split") or "") == split]


def load_v74_scores(rows: list[dict[str, Any]], model: Any, names: list[str]) -> np.ndarray:
    if not rows:
        return np.asarray([])
    return model.predict_proba(np.asarray([vector(row, names) for row in rows], dtype=np.float32))[:, 1]


def train_rows_for_hard_model(action_rows: list[dict[str, Any]], v74_model: Any, names: list[str], max_neg_per_page: int) -> list[dict[str, Any]]:
    train = split_rows(action_rows, "train")
    scores = load_v74_scores(train, v74_model, names)
    by_page: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for row, score in zip(train, scores, strict=True):
        by_page[str(row.get("page_id") or "")].append((row, float(score)))
    selected: list[dict[str, Any]] = []
    for rows in by_page.values():
        positives = [(row, score) for row, score in rows if is_positive(row)]
        hard_negs = [(row, score) for row, score in rows if is_hard_negative(row)]
        hard_negs.sort(key=lambda item: item[1], reverse=True)
        selected.extend(row for row, _ in positives)
        selected.extend(row for row, _ in hard_negs[: max_neg_per_page * max(1, len(positives))])
    return selected


def split_report(model: Any, rows: list[dict[str, Any]], names: list[str], v74_model: Any, alpha: float) -> dict[str, Any]:
    if not rows:
        return {"examples": 0}
    x = np.asarray([vector(row, names) for row in rows], dtype=np.float32)
    y = np.asarray([int(is_positive(row)) for row in rows], dtype=np.int32)
    hard_scores = model.predict_proba(x)[:, 1]
    v74_scores = v74_model.predict_proba(x)[:, 1]
    combined = alpha * v74_scores + (1.0 - alpha) * hard_scores
    out = {
        "examples": int(len(rows)),
        "positives": int(y.sum()),
        "positive_rate": round(float(y.mean()), 6),
        "hard_average_precision": round(float(average_precision_score(y, hard_scores)), 6) if y.sum() else 0.0,
        "combined_average_precision": round(float(average_precision_score(y, combined)), 6) if y.sum() else 0.0,
    }
    if len(set(y.tolist())) > 1:
        out["hard_roc_auc"] = round(float(roc_auc_score(y, hard_scores)), 6)
        out["combined_roc_auc"] = round(float(roc_auc_score(y, combined)), 6)
    return out


def parse_allowed_reasons(value: str) -> set[str] | None:
    if value == "all":
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def evaluate_policy(
    action_rows: list[dict[str, Any]],
    recovery_rows: list[dict[str, Any]],
    v74_model: Any,
    hard_model: Any,
    names: list[str],
    split: str,
    params: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    base = base_selected_by_page(recovery_rows, split)
    pages = group_pages(recovery_rows, split)
    index = recovery_index(recovery_rows, split)
    base_predicted = sum(len(rows) for rows in base.values())
    gold_total = sum(page_gold_count(rows) for rows in pages.values())
    extra_budget = max(int(float(params["candidate_inflation_target"]) * gold_total) - base_predicted, 0)
    route = Counter({"base_predicted": base_predicted, "gold_total": gold_total, "extra_budget": extra_budget})
    rows = split_rows(action_rows, split)
    x = np.asarray([vector(row, names) for row in rows], dtype=np.float32) if rows else np.asarray([])
    v74_scores = v74_model.predict_proba(x)[:, 1] if rows else np.asarray([])
    hard_scores = hard_model.predict_proba(x)[:, 1] if rows else np.asarray([])
    alpha = float(params["alpha_v74"])
    combined_scores = alpha * v74_scores + (1.0 - alpha) * hard_scores
    allowed_reasons = parse_allowed_reasons(str(params["allowed_reasons"]))
    actions_by_page: dict[str, list[tuple[dict[str, Any], float, float, float]]] = defaultdict(list)
    for row, v74_score, hard_score, combined in zip(rows, v74_scores, hard_scores, combined_scores, strict=True):
        reason = str(row.get("source_gap_reason") or "")
        if float(v74_score) < float(params["v74_threshold"]):
            route["filtered_v74_threshold"] += 1
            continue
        if float(combined) < float(params["combined_threshold"]):
            route["filtered_combined_threshold"] += 1
            continue
        if bucket(row) == "new_center_only_target" and not bool(params["include_center_only"]):
            route["filtered_center_only"] += 1
            continue
        if float(row.get("cluster_rank") or 999.0) > float(params["max_cluster_rank"]):
            route["filtered_cluster_rank"] += 1
            continue
        if allowed_reasons is not None and reason not in allowed_reasons:
            route[f"filtered_reason:{reason}"] += 1
            continue
        actions_by_page[str(row.get("page_id") or "")].append((row, float(v74_score), float(hard_score), float(combined)))

    proposals: list[tuple[int, float, str, list[dict[str, Any]], Counter]] = []
    for page_id, selected in base.items():
        selected_ids = {candidate_id(row) for row in selected}
        seen_targets: set[str] = set()
        added: list[dict[str, Any]] = []
        audit = Counter()
        ordered = sorted(actions_by_page.get(page_id, []), key=lambda item: (item[3], item[2], item[1]), reverse=True)
        for action, v74_score, hard_score, combined in ordered:
            cid = candidate_id(action)
            target = str(action.get("target_id") or "")
            if not cid or cid in selected_ids or not target or target in seen_targets:
                continue
            candidate = index.get(page_id, {}).get(cid)
            if candidate is None:
                audit["missing_recovery_row"] += 1
                continue
            item = dict(candidate)
            item["expanded_action_score_v77"] = combined
            item["expanded_action_hard_score_v77"] = hard_score
            item["expanded_action_v74_score_v77"] = v74_score
            added.append(item)
            selected_ids.add(cid)
            seen_targets.add(target)
            b = bucket(action)
            reason = str(action.get("source_gap_reason") or "unknown")
            audit["added"] += 1
            audit[f"added_bucket:{b}"] += 1
            audit[f"added_reason:{reason}"] += 1
            if len(added) >= int(params["max_add_per_page"]):
                break
        proposals.append((audit.get("added_bucket:new_iou_target", 0), sum(float(row.get("expanded_action_score_v77") or 0.0) for row in added), page_id, selected + added, audit))

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
                    "confidence": round(float(row.get("expanded_action_score_v77", row.get("score") or 0.0)), 6),
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
            learning_rate=0.035,
            max_iter=220,
            max_leaf_nodes=15,
            l2_regularization=0.08,
            class_weight={0: 1.0, 1: 8.0},
            random_state=seed,
        )
    return ExtraTreesClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight={0: 1.0, 1: 8.0},
        n_jobs=2,
        random_state=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actions", default="datasets/symbol_expanded_action_source_policy_v74/actions.jsonl")
    parser.add_argument("--recovery-data", default="datasets/symbol_detector_listwise_recovery_v47_p2_transfer_locked_full_v2/manifest.json")
    parser.add_argument("--v74-model", default="checkpoints/symbol_expanded_action_source_policy_v74/model.joblib")
    parser.add_argument("--output-dir", default="checkpoints/symbol_expanded_action_pairwise_policy_v77")
    parser.add_argument("--eval-output", default="reports/vlm/symbol_expanded_action_pairwise_policy_v77_eval.json")
    parser.add_argument("--smoke-predictions-output", default="reports/vlm/symbol_expanded_action_pairwise_policy_v77_smoke_predictions.jsonl")
    parser.add_argument("--candidate-inflation-target", type=float, default=7.999)
    parser.add_argument("--model-kind", choices=["extra_trees", "hgb"], default="hgb")
    parser.add_argument("--max-neg-per-positive", type=int, default=12)
    parser.add_argument("--fast-grid", action="store_true")
    parser.add_argument("--seed", type=int, default=20260513)
    args = parser.parse_args()

    action_rows = load_jsonl(source_path(args.actions))
    recovery_manifest = json.loads(source_path(args.recovery_data).read_text(encoding="utf-8"))
    recovery_rows = load_jsonl(source_path(recovery_manifest["outputs"]["rows"]))
    v74_bundle = joblib.load(source_path(args.v74_model))
    v74_model = v74_bundle["model"]
    names = v74_bundle.get("feature_names") or feature_names()
    hard_train = train_rows_for_hard_model(action_rows, v74_model, names, args.max_neg_per_positive)
    hard_model = build_model(args.model_kind, args.seed)
    hard_model.fit(
        np.asarray([vector(row, names) for row in hard_train], dtype=np.float32),
        np.asarray([int(is_positive(row)) for row in hard_train], dtype=np.int32),
    )
    output_dir = source_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.joblib"
    joblib.dump({"model": hard_model, "feature_names": names, "args": vars(args)}, model_path)

    baseline_dev = evaluate_selection(group_pages(recovery_rows, "dev"), base_selected_by_page(recovery_rows, "dev"))
    baseline_smoke = evaluate_selection(group_pages(recovery_rows, "smoke_eval"), base_selected_by_page(recovery_rows, "smoke_eval"))
    alpha_options = [0.0, 0.25, 0.5, 0.75] if args.fast_grid else [0.0, 0.2, 0.4, 0.6, 0.8]
    combined_thresholds = [0.02, 0.04, 0.08, 0.12] if args.fast_grid else [0.0, 0.02, 0.04, 0.08, 0.12, 0.18]
    v74_thresholds = [0.02, 0.04] if args.fast_grid else [0.0, 0.02, 0.04]
    max_add_options = [8, 10]
    dev_grid: list[dict[str, Any]] = []
    for alpha in alpha_options:
        for combined_threshold in combined_thresholds:
            for v74_threshold in v74_thresholds:
                for max_add_per_page in max_add_options:
                    params = {
                        "candidate_inflation_target": args.candidate_inflation_target,
                        "v74_threshold": v74_threshold,
                        "combined_threshold": combined_threshold,
                        "alpha_v74": alpha,
                        "max_add_per_page": max_add_per_page,
                        "include_center_only": False,
                        "max_cluster_rank": 999,
                        "allowed_reasons": "all",
                    }
                    metrics, _, audit = evaluate_policy(action_rows, recovery_rows, v74_model, hard_model, names, "dev", params)
                    route = audit["route"]
                    added_new = route.get("added_bucket:new_iou_target", 0)
                    added_dup = route.get("added_bucket:duplicate_iou_target", 0) + route.get("added_bucket:duplicate_center_only_target", 0) + route.get("added_bucket:background_or_support", 0)
                    dev_grid.append({"params": params, "metrics": metrics, "audit": audit, "added_new": added_new, "added_dup": added_dup})
    feasible = [row for row in dev_grid if row["metrics"]["candidate_inflation"] < args.candidate_inflation_target]
    selected = max(
        feasible or dev_grid,
        key=lambda row: (
            row["metrics"]["symbol_bbox_iou_0_30"]["recall"],
            row["added_new"] / max(row["added_dup"], 1),
            row["metrics"]["symbol_bbox_iou_0_30"]["precision"],
            -row["metrics"]["candidate_inflation"],
        ),
    )
    smoke_metrics, smoke_selected, smoke_audit = evaluate_policy(action_rows, recovery_rows, v74_model, hard_model, names, "smoke_eval", selected["params"])
    report = {
        "version": "symbol_expanded_action_pairwise_policy_v77",
        "source_integrity": {
            "gold_used_for_inference": False,
            "runtime_uses_svg_or_cad_geometry": False,
            "runtime_features": ["v74 runtime features", "hard-negative reranker score", "v74 score"],
            "offline_labels_used_for": ["hard-negative classifier training", "dev_policy_selection", "smoke_evaluation"],
            "final_quality_claim_allowed": False,
        },
        "inputs": {"actions": args.actions, "recovery_data": args.recovery_data, "v74_model": args.v74_model},
        "training": {
            "checkpoint": rel(model_path),
            "model_kind": args.model_kind,
            "hard_train_examples": len(hard_train),
            "hard_train_positives": sum(1 for row in hard_train if is_positive(row)),
            "train": split_report(hard_model, split_rows(action_rows, "train"), names, v74_model, 0.5),
            "dev": split_report(hard_model, split_rows(action_rows, "dev"), names, v74_model, 0.5),
            "smoke": split_report(hard_model, split_rows(action_rows, "smoke_eval"), names, v74_model, 0.5),
        },
        "baseline_dev": baseline_dev,
        "baseline_smoke_eval": baseline_smoke,
        "selected_policy": selected["params"],
        "dev": selected["metrics"],
        "dev_audit": selected["audit"],
        "smoke_eval": smoke_metrics,
        "smoke_audit": smoke_audit,
        "gate": {
            "smoke_recall_gte_0_70": smoke_metrics["symbol_bbox_iou_0_30"]["recall"] >= 0.70,
            "smoke_candidate_inflation_lt_8": smoke_metrics["candidate_inflation"] < 8.0,
            "smoke_recall_gt_v74": smoke_metrics["symbol_bbox_iou_0_30"]["recall"] > 0.699,
            "no_oracle_inference": True,
        },
        "dev_grid_top": [
            {
                "params": row["params"],
                "precision": row["metrics"]["symbol_bbox_iou_0_30"]["precision"],
                "recall": row["metrics"]["symbol_bbox_iou_0_30"]["recall"],
                "center_recall": row["metrics"]["symbol_bbox_center_recall"],
                "candidate_inflation": row["metrics"]["candidate_inflation"],
                "added_new": row["added_new"],
                "added_dup": row["added_dup"],
                "route": row["audit"]["route"],
            }
            for row in sorted(
                dev_grid,
                key=lambda item: (
                    item["metrics"]["symbol_bbox_iou_0_30"]["recall"],
                    item["added_new"] / max(item["added_dup"], 1),
                    item["metrics"]["symbol_bbox_iou_0_30"]["precision"],
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
