#!/usr/bin/env python3
"""Train/apply hard-negative pairwise reranker for uncovered-target rescue."""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter, defaultdict
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from apply_symbol_budgeted_additive_rescue_v65 import page_gold_count
from apply_symbol_detector_recall_preserving_policy_v47 import evaluate_selection, group_pages, select_rows
from train_symbol_support_suppression_v35 import load_jsonl, source_path
from train_symbol_tile_detector_v20 import rel, write_json, write_jsonl


warnings.filterwarnings(
    "ignore",
    message="`sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel`.*",
    category=UserWarning,
)


HARD_NEG_BUCKETS = {"background_or_support", "duplicate_iou_target", "duplicate_center_only_target"}


def feature_names(rows: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for row in rows:
        names.update((row.get("features") or {}).keys())
    return sorted(names)


def vector(row: dict[str, Any], names: list[str]) -> np.ndarray:
    feats = row.get("features") or {}
    return np.asarray([float(feats.get(name, 0.0) or 0.0) for name in names], dtype=np.float32)


def is_positive(row: dict[str, Any]) -> bool:
    return bool((row.get("labels") or {}).get("new_iou_target"))


def bucket(row: dict[str, Any]) -> str:
    return str((row.get("labels") or {}).get("bucket") or "unknown")


def group_action_pages(rows: list[dict[str, Any]], split: str | None = None) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if split is not None and str(row.get("split") or "") != split:
            continue
        out[str(row.get("page_id") or "")].append(row)
    return out


def build_pair_rows(rows: list[dict[str, Any]], names: list[str], max_neg_per_pos: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    x_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    summary = Counter()
    for page_rows in group_action_pages(rows).values():
        positives = [row for row in page_rows if is_positive(row)]
        negatives = [row for row in page_rows if bucket(row) in HARD_NEG_BUCKETS]
        negatives.sort(key=lambda row: (float((row.get("features") or {}).get("score") or 0.0), float((row.get("features") or {}).get("feature_score") or 0.0)), reverse=True)
        for pos in positives:
            pos_vec = vector(pos, names)
            for neg in negatives[:max_neg_per_pos]:
                neg_vec = vector(neg, names)
                x_rows.append(pos_vec - neg_vec)
                y_rows.append(1)
                x_rows.append(neg_vec - pos_vec)
                y_rows.append(0)
                summary["pairs"] += 1
                summary[f"negative_bucket:{bucket(neg)}"] += 1
    if not x_rows:
        return np.zeros((0, len(names)), dtype=np.float32), np.zeros((0,), dtype=np.int32), dict(summary)
    return np.vstack(x_rows).astype(np.float32), np.asarray(y_rows, dtype=np.int32), dict(summary)


def pairwise_score(model: Any, row: dict[str, Any], page_rows: list[dict[str, Any]], names: list[str], top_negatives: int) -> float:
    row_vec = vector(row, names)
    negatives = [item for item in page_rows if bucket(item) in HARD_NEG_BUCKETS and item is not row]
    if not negatives:
        return 1.0
    negatives.sort(key=lambda item: (float((item.get("features") or {}).get("score") or 0.0), float((item.get("features") or {}).get("feature_score") or 0.0)), reverse=True)
    diffs = np.vstack([row_vec - vector(neg, names) for neg in negatives[:top_negatives]]).astype(np.float32)
    probs = model.predict_proba(diffs)[:, 1]
    return float(np.mean(probs))


def score_split(model: Any, rows: list[dict[str, Any]], names: list[str], top_negatives: int) -> dict[str, Any]:
    if not rows:
        return {"examples": 0}
    scored = attach_pairwise_scores(model, rows, names, top_negatives, 24)
    scores = [float(row.get("pairwise_score") or 0.0) for row in scored]
    y = [int(is_positive(row)) for row in scored]
    y_arr = np.asarray(y, dtype=np.int32)
    score_arr = np.asarray(scores, dtype=np.float32)
    out = {
        "examples": int(len(rows)),
        "positives": int(y_arr.sum()),
        "positive_rate": round(float(y_arr.mean()), 6),
        "score_mean": round(float(score_arr.mean()), 6),
        "average_precision": round(float(average_precision_score(y_arr, score_arr)), 6) if y_arr.sum() else 0.0,
    }
    if len(set(y_arr.tolist())) > 1:
        out["roc_auc"] = round(float(roc_auc_score(y_arr, score_arr)), 6)
    return out


def base_rank_score(row: dict[str, Any]) -> float:
    feats = row.get("features") or {}
    return (
        float(feats.get("score") or 0.0)
        + 0.5 * float(feats.get("feature_score") or 0.0)
        + 0.25 * float(feats.get("label_area_prior") or 0.0)
        - 0.05 * float(feats.get("nearest_selected_iou") or 0.0)
        - 0.002 * float(feats.get("same_label_selected") or 0.0)
    )


def attach_pairwise_scores(model: Any, rows: list[dict[str, Any]], names: list[str], top_negatives: int, max_actions_per_page: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for page_rows in group_action_pages(rows).values():
        page_rows = sorted(page_rows, key=base_rank_score, reverse=True)[:max_actions_per_page]
        if not page_rows:
            continue
        vectors = {id(row): vector(row, names) for row in page_rows}
        hard_negatives = [row for row in page_rows if bucket(row) in HARD_NEG_BUCKETS]
        hard_negatives.sort(key=base_rank_score, reverse=True)
        hard_negatives = hard_negatives[:top_negatives]
        if not hard_negatives:
            for row in page_rows:
                item = dict(row)
                item["pairwise_score"] = 1.0
                out.append(item)
            continue
        neg_vectors = [vectors[id(row)] for row in hard_negatives]
        diffs: list[np.ndarray] = []
        spans: list[tuple[int, int]] = []
        for row in page_rows:
            start = len(diffs)
            row_vec = vectors[id(row)]
            for neg_vec in neg_vectors:
                diffs.append(row_vec - neg_vec)
            spans.append((start, len(diffs)))
        probs = model.predict_proba(np.vstack(diffs).astype(np.float32))[:, 1]
        for row, (start, end) in zip(page_rows, spans, strict=True):
            item = dict(row)
            item["pairwise_score"] = float(np.mean(probs[start:end]))
            out.append(item)
    return out


def candidate_id(row: dict[str, Any]) -> str:
    return str(row.get("candidate_id") or "")


def base_selected_by_page(recovery_rows: list[dict[str, Any]], split: str) -> dict[str, list[dict[str, Any]]]:
    pages = group_pages(recovery_rows, split)
    return {page_id: select_rows(rows, 0.02, 1, 4, 200) for page_id, rows in pages.items()}


def recovery_index(recovery_rows: list[dict[str, Any]], split: str) -> dict[str, dict[str, dict[str, Any]]]:
    pages = group_pages(recovery_rows, split)
    out: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for page_id, rows in pages.items():
        for row in rows:
            out[page_id][candidate_id(row)] = row
    return out


def evaluate_policy(
    action_rows: list[dict[str, Any]],
    recovery_rows: list[dict[str, Any]],
    split: str,
    threshold: float,
    max_add_per_page: int,
    inflation_target: float,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    base = base_selected_by_page(recovery_rows, split)
    pages = group_pages(recovery_rows, split)
    index = recovery_index(recovery_rows, split)
    base_predicted = sum(len(rows) for rows in base.values())
    gold_total = sum(page_gold_count(rows) for rows in pages.values())
    extra_budget = max(int(inflation_target * gold_total) - base_predicted, 0)
    route = Counter({"base_predicted": base_predicted, "gold_total": gold_total, "extra_budget": extra_budget})

    actions_by_page: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for page_id, page_rows in group_action_pages(action_rows, split).items():
        for row in page_rows:
            score = float(row.get("pairwise_score") or 0.0)
            if score >= threshold:
                actions_by_page[page_id].append((row, score))

    proposals: list[tuple[str, list[dict[str, Any]], Counter]] = []
    for page_id, selected in base.items():
        selected_ids = {candidate_id(row) for row in selected}
        added: list[dict[str, Any]] = []
        audit = Counter()
        for action, score in sorted(actions_by_page.get(page_id, []), key=lambda pair: pair[1], reverse=True):
            cid = candidate_id(action)
            if cid in selected_ids:
                continue
            candidate = index.get(page_id, {}).get(cid)
            if candidate is None:
                continue
            item = dict(candidate)
            item["uncovered_target_score"] = score
            added.append(item)
            selected_ids.add(cid)
            audit["added"] += 1
            audit[f"added_bucket:{bucket(action)}"] += 1
            if len(added) >= max_add_per_page:
                break
        proposals.append((page_id, selected + added, audit))

    used_extra = 0
    selected_by_page: dict[str, list[dict[str, Any]]] = {}
    for page_id, proposed, audit in sorted(proposals, key=lambda item: item[2].get("added_bucket:new_iou_target", 0), reverse=True):
        extra = max(len(proposed) - len(base[page_id]), 0)
        if used_extra + extra <= extra_budget:
            selected_by_page[page_id] = proposed
            used_extra += extra
            route.update(audit)
        else:
            selected_by_page[page_id] = base[page_id]
            route["skipped_global_budget"] += extra
    route["used_extra_budget"] = used_extra
    metrics = evaluate_selection(pages, selected_by_page)
    return metrics, selected_by_page, {"route": dict(route)}


def prediction_rows(selected_by_page: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [
        {
            "page_id": page_id,
            "predicted_symbols": [
                {
                    "candidate_id": row.get("candidate_id"),
                    "bbox": row.get("bbox"),
                    "label": row.get("label"),
                    "confidence": round(float(row.get("uncovered_target_score", row.get("score") or 0.0)), 6),
                    "proposal_source": row.get("proposal_source"),
                }
                for row in selected
            ],
            "source_integrity": {"gold_used_for_inference": False, "runtime_uses_svg_or_cad_geometry": False},
        }
        for page_id, selected in selected_by_page.items()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actions", default="datasets/symbol_uncovered_target_actions_v69/manifest.json")
    parser.add_argument("--recovery-data", default="datasets/symbol_detector_listwise_recovery_v47_p2_transfer_locked_full_v2/manifest.json")
    parser.add_argument("--output-dir", default="checkpoints/symbol_uncovered_target_pairwise_policy_v70")
    parser.add_argument("--eval-output", default="reports/vlm/symbol_uncovered_target_pairwise_policy_v70_eval.json")
    parser.add_argument("--smoke-predictions-output", default="reports/vlm/symbol_uncovered_target_pairwise_policy_v70_smoke_predictions.jsonl")
    parser.add_argument("--candidate-inflation-target", type=float, default=7.999)
    parser.add_argument("--max-neg-per-pos", type=int, default=24)
    parser.add_argument("--top-negatives", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260513)
    args = parser.parse_args()

    action_manifest = json.loads(source_path(args.actions).read_text(encoding="utf-8"))
    action_rows = load_jsonl(source_path(action_manifest["outputs"]["rows"]))
    recovery_manifest = json.loads(source_path(args.recovery_data).read_text(encoding="utf-8"))
    recovery_rows = load_jsonl(source_path(recovery_manifest["outputs"]["rows"]))
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in action_rows:
        by_split[str(row.get("split") or "")].append(row)
    train_rows = by_split["train"]
    dev_rows = by_split["dev"]
    smoke_rows = by_split["smoke_eval"]
    names = feature_names(train_rows)
    x_train, y_train, pair_summary = build_pair_rows(train_rows, names, args.max_neg_per_pos)
    model = ExtraTreesClassifier(
        n_estimators=160,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=2,
        random_state=args.seed,
    )
    model.fit(x_train, y_train)
    output_dir = source_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.joblib"
    joblib.dump({"model": model, "feature_names": names, "args": vars(args)}, model_path)

    baseline_smoke = evaluate_selection(group_pages(recovery_rows, "smoke_eval"), base_selected_by_page(recovery_rows, "smoke_eval"))
    baseline_dev = evaluate_selection(group_pages(recovery_rows, "dev"), base_selected_by_page(recovery_rows, "dev"))
    scored_by_top_negatives: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for top_negatives in [8]:
        scored_by_top_negatives[top_negatives] = {
            "dev": attach_pairwise_scores(model, dev_rows, names, top_negatives, 24),
            "smoke_eval": attach_pairwise_scores(model, smoke_rows, names, top_negatives, 24),
        }

    dev_grid: list[dict[str, Any]] = []
    for threshold in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        for max_add_per_page in [5, 8, 10]:
            for top_negatives in [8]:
                metrics, _, audit = evaluate_policy(
                    scored_by_top_negatives[top_negatives]["dev"],
                    recovery_rows,
                    "dev",
                    threshold,
                    max_add_per_page,
                    args.candidate_inflation_target,
                )
                dev_grid.append(
                    {
                        "threshold": threshold,
                        "max_add_per_page": max_add_per_page,
                        "top_negatives": top_negatives,
                        "metrics": metrics,
                        "audit": audit,
                    }
                )
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
        scored_by_top_negatives[int(selected["top_negatives"])]["smoke_eval"],
        recovery_rows,
        "smoke_eval",
        float(selected["threshold"]),
        int(selected["max_add_per_page"]),
        args.candidate_inflation_target,
    )
    report = {
        "version": "symbol_uncovered_target_pairwise_policy_v70",
        "source_integrity": {
            "gold_used_for_inference": False,
            "runtime_uses_svg_or_cad_geometry": False,
            "runtime_features": ["v69 runtime-safe action features", "pairwise differences against same-page hard negatives"],
            "offline_labels_used_for": ["pairwise ranking training", "dev_policy_selection", "smoke_evaluation"],
        },
        "training": {
            "checkpoint": rel(model_path),
            "feature_count": len(names),
            "pair_summary": pair_summary,
            "train_pairs": int(len(y_train)),
            "train_positive_pairs": int(y_train.sum()),
            "dev": score_split(model, dev_rows, names, args.top_negatives),
            "smoke": score_split(model, smoke_rows, names, args.top_negatives),
        },
        "baseline_dev": baseline_dev,
        "baseline_smoke_eval": baseline_smoke,
        "selected_policy": {
            "threshold": selected["threshold"],
            "max_add_per_page": selected["max_add_per_page"],
            "top_negatives": selected["top_negatives"],
            "selected_on": "dev",
        },
        "dev": selected["metrics"],
        "dev_audit": selected["audit"],
        "smoke_eval": smoke_metrics,
        "smoke_audit": smoke_audit,
        "gate": {
            "smoke_recall_gt_v69": smoke_metrics["symbol_bbox_iou_0_30"]["recall"] > 0.68575,
            "smoke_recall_gte_0_70": smoke_metrics["symbol_bbox_iou_0_30"]["recall"] >= 0.70,
            "smoke_candidate_inflation_lt_8": smoke_metrics["candidate_inflation"] < 8.0,
            "no_oracle_inference": True,
        },
        "dev_grid_top": [
            {
                "threshold": row["threshold"],
                "max_add_per_page": row["max_add_per_page"],
                "top_negatives": row["top_negatives"],
                "precision": row["metrics"]["symbol_bbox_iou_0_30"]["precision"],
                "recall": row["metrics"]["symbol_bbox_iou_0_30"]["recall"],
                "center_recall": row["metrics"]["symbol_bbox_center_recall"],
                "candidate_inflation": row["metrics"]["candidate_inflation"],
                "added": row["audit"]["route"].get("added", 0),
                "added_new_iou_target": row["audit"]["route"].get("added_bucket:new_iou_target", 0),
                "added_background_or_support": row["audit"]["route"].get("added_bucket:background_or_support", 0),
            }
            for row in sorted(
                dev_grid,
                key=lambda item: (
                    item["metrics"]["symbol_bbox_iou_0_30"]["recall"],
                    item["metrics"]["symbol_bbox_iou_0_30"]["precision"],
                    -item["metrics"]["candidate_inflation"],
                ),
                reverse=True,
            )[:80]
        ],
    }
    report["gate"]["passed"] = all(report["gate"].values())
    write_json(source_path(args.eval_output), report)
    write_jsonl(source_path(args.smoke_predictions_output), prediction_rows(smoke_selected))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
