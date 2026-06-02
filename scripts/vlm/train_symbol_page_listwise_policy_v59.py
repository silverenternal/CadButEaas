#!/usr/bin/env python3
"""Train and evaluate a page-level swap action policy from v58 action data."""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter, defaultdict
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from apply_symbol_focus_rescue_policy_v52 import candidate_area
from apply_symbol_sink_tiny_refiner_page_v49 import evaluate, load_gold, score_candidates, select_page
from train_symbol_support_suppression_v35 import load_jsonl, source_path
from train_symbol_tile_detector_v20 import rel, write_json, write_jsonl


warnings.filterwarnings(
    "ignore",
    message="`sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel`.*",
    category=UserWarning,
)


def feature_names(rows: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for row in rows:
        names.update((row.get("features") or {}).keys())
    return sorted(names)


def vector(row: dict[str, Any], names: list[str]) -> list[float]:
    feats = row.get("features") or {}
    return [float(feats.get(name, 0.0) or 0.0) for name in names]


def model_report(model: Any, rows: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    if not rows:
        return {"examples": 0}
    y = np.asarray([float((row.get("labels") or {}).get("reward") or 0.0) for row in rows], dtype=np.float32)
    pred = model.predict(np.asarray([vector(row, names) for row in rows], dtype=np.float32))
    return {
        "examples": int(len(rows)),
        "positive_reward": int((y > 0).sum()),
        "reward_mean": round(float(y.mean()), 6),
        "pred_mean": round(float(pred.mean()), 6),
        "mae": round(float(mean_absolute_error(y, pred)), 6),
        "r2": round(float(r2_score(y, pred)), 6) if len(set(y.tolist())) > 1 else 0.0,
    }


def group_pages(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row.get("page_id") or "")].append(row)
    return out


def load_candidate_index(scored_rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    by_page: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in scored_rows:
        by_page[str(row.get("page_id") or "")][str(row.get("candidate_id") or "")] = row
    return by_page


def apply_actions(
    action_rows: list[dict[str, Any]],
    action_scores: np.ndarray,
    scored_rows: list[dict[str, Any]],
    gold_all: dict[str, dict[str, dict[str, Any]]],
    threshold: float,
    max_actions_per_page: int,
    selection_threshold: float,
    cluster_topk: int,
    max_per_page: int,
) -> tuple[dict[str, Any], Counter, list[dict[str, Any]]]:
    pages: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        pages[str(row.get("page_id") or "")].append(row)
    selected_base = {
        page_id: select_page(page_rows, selection_threshold, cluster_topk, max_per_page)
        for page_id, page_rows in pages.items()
    }
    candidate_index = load_candidate_index(scored_rows)
    actions_by_page: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for row, score in zip(action_rows, action_scores, strict=True):
        if float(score) >= threshold:
            actions_by_page[str(row.get("page_id") or "")].append((row, float(score)))
    selected_by_page: dict[str, list[dict[str, Any]]] = {}
    audit = Counter()
    for page_id, selected in selected_base.items():
        selected_ids = {str(row.get("candidate_id") or "") for row in selected}
        used_add: set[str] = set()
        used_drop: set[str] = set()
        chosen: list[dict[str, Any]] = []
        for action, score in sorted(actions_by_page.get(page_id, []), key=lambda pair: pair[1], reverse=True):
            add_id = str(action.get("add_candidate_id") or "")
            drop_id = str(action.get("drop_candidate_id") or "")
            if add_id in used_add or drop_id in used_drop or add_id in selected_ids or drop_id not in selected_ids:
                continue
            chosen.append(action)
            used_add.add(add_id)
            used_drop.add(drop_id)
            if len(chosen) >= max_actions_per_page:
                break
        drop_ids = used_drop
        out = [row for row in selected if str(row.get("candidate_id") or "") not in drop_ids]
        for action in chosen:
            add_id = str(action.get("add_candidate_id") or "")
            add_row = candidate_index.get(page_id, {}).get(add_id)
            if add_row:
                item = dict(add_row)
                item["listwise_action_score"] = float(action_scores[action_rows.index(action)]) if action in action_rows else None
                out.append(item)
                audit["added"] += 1
                audit[f"added_label:{item.get('label')}"] += 1
                audit[f"added_area:{candidate_area(item)}"] += 1
            audit["dropped"] += 1
            labels = action.get("labels") or {}
            audit["offline_drop_iou_hit"] += int(bool(labels.get("drop_iou_hit")))
            audit["offline_drop_center_hit"] += int(bool(labels.get("drop_center_hit")))
            audit["offline_positive_reward_action"] += int(float(labels.get("reward") or 0.0) > 0)
        out.sort(key=lambda row: (float(row.get("policy_score") or 0.0), float(row.get("score") or 0.0)), reverse=True)
        selected_by_page[page_id] = out
    gold = {page_id: gold_all[page_id] for page_id in selected_by_page if page_id in gold_all}
    metrics = evaluate(selected_by_page, gold)
    predictions = [
        {
            "page_id": page_id,
            "predicted_symbols": [
                {
                    "candidate_id": row.get("candidate_id"),
                    "bbox": row.get("bbox"),
                    "label": row.get("label"),
                    "confidence": round(float(row.get("policy_score") or 0.0), 6),
                    "listwise_action_score": row.get("listwise_action_score"),
                }
                for row in selected
            ],
            "source_integrity": {"gold_used_for_inference": False, "runtime_uses_svg_or_cad_geometry": False},
        }
        for page_id, selected in selected_by_page.items()
    ]
    return metrics, audit, predictions


def evaluate_split(
    model: Any,
    names: list[str],
    action_rows: list[dict[str, Any]],
    scored_rows: list[dict[str, Any]],
    gold_all: dict[str, dict[str, dict[str, Any]]],
    threshold: float,
    max_actions_per_page: int,
    selection_threshold: float,
    cluster_topk: int,
    max_per_page: int,
) -> tuple[dict[str, Any], Counter, list[dict[str, Any]]]:
    scores = model.predict(np.asarray([vector(row, names) for row in action_rows], dtype=np.float32)) if action_rows else np.asarray([])
    return apply_actions(action_rows, scores, scored_rows, gold_all, threshold, max_actions_per_page, selection_threshold, cluster_topk, max_per_page)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actions", default="datasets/symbol_page_listwise_swap_v58/manifest.json")
    parser.add_argument("--recovery-data", default="datasets/symbol_detector_listwise_recovery_v47_p2_transfer_smoke120/manifest.json")
    parser.add_argument("--smoke-rows", default="datasets/symbol_tile_detector_tiny_sahi_v21/smoke_v30.jsonl")
    parser.add_argument("--suppression-model", default="checkpoints/symbol_support_suppression_v35_p2_transfer_smoke120_t065_c1/model.joblib")
    parser.add_argument("--output-dir", default="checkpoints/symbol_page_listwise_policy_v59")
    parser.add_argument("--eval-output", default="reports/vlm/symbol_page_listwise_policy_v59_eval.json")
    parser.add_argument("--smoke-predictions-output", default="reports/vlm/symbol_page_listwise_policy_v59_smoke_predictions.jsonl")
    parser.add_argument("--all-predictions-output", default="reports/vlm/symbol_page_listwise_policy_v59_all_cache_predictions.jsonl")
    parser.add_argument("--selection-threshold", type=float, default=0.65)
    parser.add_argument("--cluster-topk", type=int, default=1)
    parser.add_argument("--max-per-page", type=int, default=120)
    parser.add_argument("--n-estimators", type=int, default=260)
    parser.add_argument("--seed", type=int, default=20260513)
    args = parser.parse_args()

    actions_manifest = json.loads(source_path(args.actions).read_text(encoding="utf-8"))
    action_rows = load_jsonl(source_path(actions_manifest["outputs"]["rows"]))
    by_split = defaultdict(list)
    for row in action_rows:
        by_split[str(row.get("split") or "")].append(row)
    train_rows = by_split["train"]
    dev_rows = by_split["dev"]
    smoke_rows = by_split["smoke_eval"]
    names = feature_names(train_rows)
    model = ExtraTreesRegressor(
        n_estimators=args.n_estimators,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=2,
        random_state=args.seed,
    )
    model.fit(
        np.asarray([vector(row, names) for row in train_rows], dtype=np.float32),
        np.asarray([float((row.get("labels") or {}).get("reward") or 0.0) for row in train_rows], dtype=np.float32),
    )
    output_dir = source_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.joblib"
    joblib.dump({"model": model, "feature_names": names, "args": vars(args)}, model_path)

    recovery_manifest = json.loads(source_path(args.recovery_data).read_text(encoding="utf-8"))
    recovery_rows = load_jsonl(source_path(recovery_manifest["outputs"]["rows"]))
    scored_rows = score_candidates(recovery_rows, joblib.load(source_path(args.suppression_model)))
    scored_by_split = defaultdict(list)
    for row in scored_rows:
        scored_by_split[str(row.get("split") or "")].append(row)
    gold_all = load_gold(source_path(args.smoke_rows))

    dev_grid: list[dict[str, Any]] = []
    for threshold in [-0.25, 0.0, 0.25, 0.5, 0.75, 1.0]:
        for max_actions in [1, 2, 3, 5]:
            metrics, audit, _ = evaluate_split(
                model,
                names,
                dev_rows,
                scored_by_split["dev"],
                gold_all,
                threshold,
                max_actions,
                args.selection_threshold,
                args.cluster_topk,
                args.max_per_page,
            )
            dev_grid.append({"threshold": threshold, "max_actions_per_page": max_actions, "metrics": metrics, "route_audit": dict(audit)})
    selected_policy = max(
        dev_grid,
        key=lambda row: (
            row["metrics"]["symbol_bbox_iou_0_30"]["recall"],
            row["metrics"]["symbol_bbox_iou_0_30"]["precision"],
            row["metrics"]["symbol_bbox_center_recall"],
            -row["metrics"]["candidate_inflation"],
        ),
    )
    threshold = float(selected_policy["threshold"])
    max_actions = int(selected_policy["max_actions_per_page"])
    smoke_metrics, smoke_audit, smoke_predictions = evaluate_split(
        model,
        names,
        smoke_rows,
        scored_by_split["smoke_eval"],
        gold_all,
        threshold,
        max_actions,
        args.selection_threshold,
        args.cluster_topk,
        args.max_per_page,
    )
    all_metrics, all_audit, all_predictions = evaluate_split(
        model,
        names,
        action_rows,
        scored_rows,
        gold_all,
        threshold,
        max_actions,
        args.selection_threshold,
        args.cluster_topk,
        args.max_per_page,
    )
    report = {
        "version": "symbol_page_listwise_policy_v59",
        "source_integrity": {
            "gold_used_for_inference": False,
            "runtime_uses_svg_or_cad_geometry": False,
            "offline_labels_used_for": ["reward_training", "dev_policy_selection", "evaluation"],
            "final_quality_claim_allowed": False,
        },
        "training": {
            "checkpoint": rel(model_path),
            "feature_count": len(names),
            "train": model_report(model, train_rows, names),
            "dev": model_report(model, dev_rows, names),
            "smoke": model_report(model, smoke_rows, names),
        },
        "selected_policy": {"threshold": threshold, "max_actions_per_page": max_actions, "selected_on": "dev"},
        "smoke_eval": {"metrics": smoke_metrics, "route_audit": dict(smoke_audit)},
        "all_cache": {"metrics": all_metrics, "route_audit": dict(all_audit)},
        "dev_grid": [
            {
                "threshold": row["threshold"],
                "max_actions_per_page": row["max_actions_per_page"],
                "precision": row["metrics"]["symbol_bbox_iou_0_30"]["precision"],
                "recall": row["metrics"]["symbol_bbox_iou_0_30"]["recall"],
                "center_recall": row["metrics"]["symbol_bbox_center_recall"],
                "candidate_inflation": row["metrics"]["candidate_inflation"],
                "added": row["route_audit"].get("added", 0),
                "offline_drop_iou_hit": row["route_audit"].get("offline_drop_iou_hit", 0),
                "offline_drop_center_hit": row["route_audit"].get("offline_drop_center_hit", 0),
            }
            for row in dev_grid
        ],
    }
    write_json(source_path(args.eval_output), report)
    write_jsonl(source_path(args.smoke_predictions_output), smoke_predictions)
    write_jsonl(source_path(args.all_predictions_output), all_predictions)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
