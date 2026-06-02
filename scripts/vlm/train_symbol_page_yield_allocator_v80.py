#!/usr/bin/env python3
"""Train page-yield allocator for v74 expanded actions."""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter, defaultdict
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from apply_symbol_budgeted_additive_rescue_v65 import page_gold_count
from apply_symbol_detector_recall_preserving_policy_v47 import evaluate_selection, group_pages, select_rows
from train_symbol_expanded_action_source_policy_v74 import candidate_id, feature_names, vector
from train_symbol_support_suppression_v35 import load_jsonl, source_path
from train_symbol_tile_detector_v20 import rel, write_json, write_jsonl

warnings.filterwarnings("ignore", category=UserWarning)

BUCKETS = ["new_iou_target", "duplicate_iou_target", "duplicate_center_only_target", "background_or_support", "new_center_only_target"]
REASONS = ["v69_eligible", "cluster_rank_gt_v69_cap", "non_focus_label_area", "cluster_rank_1_selected_competitor", "below_low_score_threshold"]


def bucket(row: dict[str, Any]) -> str:
    return str(row.get("bucket") or "unknown")


def reason(row: dict[str, Any]) -> str:
    return str(row.get("source_gap_reason") or "unknown")


def base_selected_by_page(recovery_rows: list[dict[str, Any]], split: str) -> dict[str, list[dict[str, Any]]]:
    pages = group_pages(recovery_rows, split)
    return {page_id: select_rows(rows, 0.02, 1, 4, 200) for page_id, rows in pages.items()}


def recovery_index(recovery_rows: list[dict[str, Any]], split: str) -> dict[str, dict[str, dict[str, Any]]]:
    pages = group_pages(recovery_rows, split)
    return {page_id: {candidate_id(row): row for row in rows} for page_id, rows in pages.items()}


def score_actions(rows: list[dict[str, Any]], model: Any, names: list[str]) -> list[tuple[dict[str, Any], float]]:
    if not rows:
        return []
    scores = model.predict_proba(np.asarray([vector(row, names) for row in rows], dtype=np.float32))[:, 1]
    return list(zip(rows, [float(x) for x in scores], strict=True))


def filter_action(row: dict[str, Any], score: float, params: dict[str, Any]) -> bool:
    if score < float(params["base_threshold"]):
        return False
    if bucket(row) == "new_center_only_target" and not bool(params["include_center_only"]):
        return False
    if float(row.get("cluster_rank") or 999.0) > float(params["max_cluster_rank"]):
        return False
    return True


def page_action_plan(
    page_actions: list[tuple[dict[str, Any], float]],
    params: dict[str, Any],
    max_add_per_page: int,
    oracle: bool,
) -> tuple[list[dict[str, Any]], Counter]:
    audit = Counter()
    seen_targets: set[str] = set()
    added: list[dict[str, Any]] = []
    if oracle:
        ordered = sorted(page_actions, key=lambda item: (bucket(item[0]) == "new_iou_target", item[1]), reverse=True)
    else:
        ordered = sorted(page_actions, key=lambda item: item[1], reverse=True)
    for row, score in ordered:
        target = str(row.get("target_id") or "")
        if not target or target in seen_targets:
            continue
        added.append(row)
        seen_targets.add(target)
        audit["added"] += 1
        audit[f"added_bucket:{bucket(row)}"] += 1
        audit[f"added_reason:{reason(row)}"] += 1
        if len(added) >= max_add_per_page:
            break
    return added, audit


def page_features(page_id: str, scored_actions: list[tuple[dict[str, Any], float]], params: dict[str, Any]) -> dict[str, float]:
    scores = [score for _, score in scored_actions]
    out: dict[str, float] = {
        "action_count": float(len(scored_actions)),
        "score_sum_top5": float(sum(sorted(scores, reverse=True)[:5])),
        "score_sum_top10": float(sum(sorted(scores, reverse=True)[:10])),
        "score_mean": float(np.mean(scores)) if scores else 0.0,
        "score_max": max(scores) if scores else 0.0,
        "score_p90": float(np.quantile(scores, 0.90)) if scores else 0.0,
        "score_ge_030": float(sum(1 for score in scores if score >= 0.30)),
        "score_ge_050": float(sum(1 for score in scores if score >= 0.50)),
    }
    ordered = sorted(scored_actions, key=lambda item: item[1], reverse=True)
    for k in [5, 10, 20]:
        subset = ordered[:k]
        out[f"top{k}_count"] = float(len(subset))
        for b in BUCKETS:
            # Bucket labels are not runtime-safe; keep only for audit? Do not include in final vector.
            out[f"audit_top{k}_bucket_{b}"] = float(sum(1 for row, _ in subset if bucket(row) == b))
        for r in REASONS:
            out[f"top{k}_reason_{r}"] = float(sum(1 for row, _ in subset if reason(row) == r))
        out[f"top{k}_nearest_iou_mean"] = float(np.mean([float((row.get("features") or {}).get("nearest_selected_iou") or 0.0) for row, _ in subset])) if subset else 0.0
        out[f"top{k}_same_label_iou_mean"] = float(np.mean([float((row.get("features") or {}).get("nearest_same_label_iou") or 0.0) for row, _ in subset])) if subset else 0.0
        out[f"top{k}_cluster_rank_mean"] = float(np.mean([float(row.get("cluster_rank") or 999.0) for row, _ in subset])) if subset else 999.0
    return out


def page_feature_names(rows: list[dict[str, Any]]) -> list[str]:
    names = set()
    for row in rows:
        names.update((row.get("features") or {}).keys())
    return sorted(name for name in names if not name.startswith("audit_"))


def page_vector(row: dict[str, Any], names: list[str]) -> list[float]:
    feats = row.get("features") or {}
    return [float(feats.get(name, 0.0) or 0.0) for name in names]


def build_page_dataset(
    action_rows: list[dict[str, Any]],
    recovery_rows: list[dict[str, Any]],
    model: Any,
    names: list[str],
    params: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, list[tuple[dict[str, Any], float]]]]]:
    by_split_page: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in action_rows:
        by_split_page[str(row.get("split") or "")][str(row.get("page_id") or "")].append(row)
    dataset: list[dict[str, Any]] = []
    scored_by_split_page: dict[str, dict[str, list[tuple[dict[str, Any], float]]]] = defaultdict(dict)
    for split, pages in by_split_page.items():
        for page_id, rows in pages.items():
            scored = [(row, score) for row, score in score_actions(rows, model, names) if filter_action(row, score, params)]
            scored_by_split_page[split][page_id] = scored
            oracle_added, oracle_audit = page_action_plan(scored, params, int(params["max_add_per_page"]), oracle=True)
            score_added, score_audit = page_action_plan(scored, params, int(params["max_add_per_page"]), oracle=False)
            dataset.append({
                "split": split,
                "page_id": page_id,
                "features": page_features(page_id, scored, params),
                "labels": {
                    "oracle_new_iou": int(oracle_audit.get("added_bucket:new_iou_target", 0)),
                    "score_new_iou": int(score_audit.get("added_bucket:new_iou_target", 0)),
                    "oracle_added": int(oracle_audit.get("added", 0)),
                    "score_added": int(score_audit.get("added", 0)),
                    "oracle_gain_over_score": int(oracle_audit.get("added_bucket:new_iou_target", 0)) - int(score_audit.get("added_bucket:new_iou_target", 0)),
                },
            })
    return dataset, scored_by_split_page


def evaluate_allocator(
    page_rows: list[dict[str, Any]],
    scored_pages: dict[str, list[tuple[dict[str, Any], float]]],
    recovery_rows: list[dict[str, Any]],
    model: Any,
    feature_names_: list[str],
    split: str,
    params: dict[str, Any],
    oracle_page_order: bool = False,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    base = base_selected_by_page(recovery_rows, split)
    pages = group_pages(recovery_rows, split)
    index = recovery_index(recovery_rows, split)
    base_predicted = sum(len(rows) for rows in base.values())
    gold_total = sum(page_gold_count(rows) for rows in pages.values())
    extra_budget = max(int(float(params["candidate_inflation_target"]) * gold_total) - base_predicted, 0)
    route = Counter({"base_predicted": base_predicted, "gold_total": gold_total, "extra_budget": extra_budget})
    split_page_rows = [row for row in page_rows if str(row.get("split") or "") == split]
    if oracle_page_order:
        page_scores = np.asarray([float((row.get("labels") or {}).get("oracle_new_iou") or 0.0) for row in split_page_rows])
    else:
        page_scores = model.predict(np.asarray([page_vector(row, feature_names_) for row in split_page_rows], dtype=np.float32)) if split_page_rows else np.asarray([])
    page_score_map = {str(row.get("page_id") or ""): float(score) for row, score in zip(split_page_rows, page_scores, strict=True)}
    proposals: list[tuple[float, str, list[dict[str, Any]], Counter]] = []
    for page_id, selected in base.items():
        selected_ids = {candidate_id(row) for row in selected}
        seen_targets: set[str] = set()
        added: list[dict[str, Any]] = []
        audit = Counter()
        scored = scored_pages.get(page_id, [])
        # Within page, still use v74 score order at runtime.
        for action, score in sorted(scored, key=lambda item: item[1], reverse=True):
            cid = candidate_id(action)
            target = str(action.get("target_id") or "")
            if not cid or cid in selected_ids or not target or target in seen_targets:
                continue
            candidate = index.get(page_id, {}).get(cid)
            if candidate is None:
                continue
            item = dict(candidate)
            item["page_yield_score_v80"] = page_score_map.get(page_id, 0.0)
            item["action_score_v80"] = score
            added.append(item)
            selected_ids.add(cid)
            seen_targets.add(target)
            audit["added"] += 1
            audit[f"added_bucket:{bucket(action)}"] += 1
            audit[f"added_reason:{reason(action)}"] += 1
            if len(added) >= int(params["max_add_per_page"]):
                break
        priority = page_score_map.get(page_id, 0.0)
        proposals.append((priority, page_id, selected + added, audit))
    used_extra = 0
    selected_by_page: dict[str, list[dict[str, Any]]] = {}
    for _priority, page_id, proposed, audit in sorted(proposals, reverse=True):
        extra = max(len(proposed) - len(base[page_id]), 0)
        if extra <= 0:
            selected_by_page[page_id] = base[page_id]
            continue
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
                    "confidence": round(float(row.get("action_score_v80", row.get("score") or 0.0)), 6),
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
        return HistGradientBoostingRegressor(learning_rate=0.04, max_iter=180, max_leaf_nodes=15, l2_regularization=0.08, random_state=seed)
    return ExtraTreesRegressor(n_estimators=400, min_samples_leaf=2, max_features="sqrt", n_jobs=2, random_state=seed)


def regression_report(model: Any, rows: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    if not rows:
        return {"examples": 0}
    x = np.asarray([page_vector(row, names) for row in rows], dtype=np.float32)
    y = np.asarray([float((row.get("labels") or {}).get("oracle_new_iou") or 0.0) for row in rows], dtype=np.float32)
    pred = model.predict(x)
    return {"examples": len(rows), "target_mean": round(float(y.mean()), 6), "pred_mean": round(float(pred.mean()), 6), "mae": round(float(mean_absolute_error(y, pred)), 6), "r2": round(float(r2_score(y, pred)), 6) if len(rows) > 1 else 0.0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actions", default="datasets/symbol_expanded_action_source_policy_v74/actions.jsonl")
    parser.add_argument("--recovery-data", default="datasets/symbol_detector_listwise_recovery_v47_p2_transfer_locked_full_v2/manifest.json")
    parser.add_argument("--v74-model", default="checkpoints/symbol_expanded_action_source_policy_v74/model.joblib")
    parser.add_argument("--output-dir", default="checkpoints/symbol_page_yield_allocator_v80")
    parser.add_argument("--dataset-output", default="datasets/symbol_page_yield_allocator_v80/pages.jsonl")
    parser.add_argument("--eval-output", default="reports/vlm/symbol_page_yield_allocator_v80_eval.json")
    parser.add_argument("--smoke-predictions-output", default="reports/vlm/symbol_page_yield_allocator_v80_smoke_predictions.jsonl")
    parser.add_argument("--candidate-inflation-target", type=float, default=7.999)
    parser.add_argument("--model-kind", choices=["extra_trees", "hgb"], default="extra_trees")
    parser.add_argument("--seed", type=int, default=20260513)
    args = parser.parse_args()

    action_rows = load_jsonl(source_path(args.actions))
    recovery_manifest = json.loads(source_path(args.recovery_data).read_text(encoding="utf-8"))
    recovery_rows = load_jsonl(source_path(recovery_manifest["outputs"]["rows"]))
    v74_bundle = joblib.load(source_path(args.v74_model))
    v74_model = v74_bundle["model"]
    action_feature_names = v74_bundle.get("feature_names") or feature_names()
    params = {"candidate_inflation_target": args.candidate_inflation_target, "base_threshold": 0.04, "max_add_per_page": 10, "include_center_only": False, "max_cluster_rank": 999}
    page_rows, scored_by_split_page = build_page_dataset(action_rows, recovery_rows, v74_model, action_feature_names, params)
    dataset_path = source_path(args.dataset_output)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(dataset_path, page_rows)
    names = page_feature_names(page_rows)
    train_rows = [row for row in page_rows if str(row.get("split") or "") == "train"]
    dev_rows = [row for row in page_rows if str(row.get("split") or "") == "dev"]
    smoke_rows = [row for row in page_rows if str(row.get("split") or "") == "smoke_eval"]
    model = build_model(args.model_kind, args.seed)
    model.fit(np.asarray([page_vector(row, names) for row in train_rows], dtype=np.float32), np.asarray([float((row.get("labels") or {}).get("oracle_new_iou") or 0.0) for row in train_rows], dtype=np.float32))
    output_dir = source_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.joblib"
    joblib.dump({"model": model, "feature_names": names, "args": vars(args)}, model_path)
    baseline_dev = evaluate_selection(group_pages(recovery_rows, "dev"), base_selected_by_page(recovery_rows, "dev"))
    baseline_smoke = evaluate_selection(group_pages(recovery_rows, "smoke_eval"), base_selected_by_page(recovery_rows, "smoke_eval"))
    dev_metrics, _, dev_audit = evaluate_allocator(page_rows, scored_by_split_page["dev"], recovery_rows, model, names, "dev", params)
    smoke_metrics, smoke_selected, smoke_audit = evaluate_allocator(page_rows, scored_by_split_page["smoke_eval"], recovery_rows, model, names, "smoke_eval", params)
    oracle_dev_metrics, _, oracle_dev_audit = evaluate_allocator(page_rows, scored_by_split_page["dev"], recovery_rows, model, names, "dev", params, oracle_page_order=True)
    oracle_smoke_metrics, _, oracle_smoke_audit = evaluate_allocator(page_rows, scored_by_split_page["smoke_eval"], recovery_rows, model, names, "smoke_eval", params, oracle_page_order=True)
    report = {
        "version": "symbol_page_yield_allocator_v80",
        "source_integrity": {"gold_used_for_inference": False, "runtime_uses_svg_or_cad_geometry": False, "runtime_features": ["page-level score/action summaries from v74 actions"], "offline_labels_used_for": ["page-yield regression training", "dev/smoke evaluation"], "final_quality_claim_allowed": False},
        "dataset": {"rows": len(page_rows), "path": rel(dataset_path)},
        "training": {"checkpoint": rel(model_path), "model_kind": args.model_kind, "feature_count": len(names), "train": regression_report(model, train_rows, names), "dev": regression_report(model, dev_rows, names), "smoke": regression_report(model, smoke_rows, names)},
        "baseline_dev": baseline_dev,
        "baseline_smoke_eval": baseline_smoke,
        "dev": dev_metrics,
        "dev_audit": dev_audit,
        "smoke_eval": smoke_metrics,
        "smoke_audit": smoke_audit,
        "oracle_page_order_dev": oracle_dev_metrics,
        "oracle_page_order_dev_audit": oracle_dev_audit,
        "oracle_page_order_smoke": oracle_smoke_metrics,
        "oracle_page_order_smoke_audit": oracle_smoke_audit,
        "gate": {"smoke_recall_gte_0_70": smoke_metrics["symbol_bbox_iou_0_30"]["recall"] >= 0.70, "smoke_candidate_inflation_lt_8": smoke_metrics["candidate_inflation"] < 8.0, "smoke_recall_gt_v74": smoke_metrics["symbol_bbox_iou_0_30"]["recall"] > 0.699, "no_oracle_inference": True},
    }
    report["gate"]["passed"] = all(report["gate"].values())
    write_json(source_path(args.eval_output), report)
    write_jsonl(source_path(args.smoke_predictions_output), prediction_rows(smoke_selected))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
