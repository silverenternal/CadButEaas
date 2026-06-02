#!/usr/bin/env python3
"""Train/apply uncovered-target-aware add-only symbol rescue policy."""

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


def feature_names(rows: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for row in rows:
        names.update((row.get("features") or {}).keys())
    return sorted(names)


def vector(row: dict[str, Any], names: list[str]) -> list[float]:
    feats = row.get("features") or {}
    return [float(feats.get(name, 0.0) or 0.0) for name in names]


def is_positive(row: dict[str, Any]) -> bool:
    return bool((row.get("labels") or {}).get("new_iou_target"))


def split_report(model: Any, rows: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    if not rows:
        return {"examples": 0}
    y = np.asarray([int(is_positive(row)) for row in rows], dtype=np.int32)
    scores = model.predict_proba(np.asarray([vector(row, names) for row in rows], dtype=np.float32))[:, 1]
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


def candidate_id(row: dict[str, Any]) -> str:
    return str(row.get("candidate_id") or "")


def action_candidate_id(row: dict[str, Any]) -> str:
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
    model: Any,
    names: list[str],
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

    split_actions = [row for row in action_rows if str(row.get("split") or "") == split]
    if split_actions:
        scores = model.predict_proba(np.asarray([vector(row, names) for row in split_actions], dtype=np.float32))[:, 1]
    else:
        scores = np.asarray([])
    actions_by_page: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for row, score in zip(split_actions, scores, strict=True):
        if float(score) >= threshold:
            actions_by_page[str(row.get("page_id") or "")].append((row, float(score)))

    proposals: list[tuple[str, list[dict[str, Any]], Counter]] = []
    for page_id, selected in base.items():
        selected_ids = {candidate_id(row) for row in selected}
        added: list[dict[str, Any]] = []
        audit = Counter()
        for action, score in sorted(actions_by_page.get(page_id, []), key=lambda pair: pair[1], reverse=True):
            cid = action_candidate_id(action)
            if cid in selected_ids:
                continue
            candidate = index.get(page_id, {}).get(cid)
            if candidate is None:
                continue
            item = dict(candidate)
            item["uncovered_target_score"] = score
            added.append(item)
            selected_ids.add(cid)
            labels = action.get("labels") or {}
            bucket = str(labels.get("bucket") or "unknown")
            audit["added"] += 1
            audit[f"added_bucket:{bucket}"] += 1
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
    parser.add_argument("--actions", default="datasets/symbol_uncovered_target_actions_v67/manifest.json")
    parser.add_argument("--recovery-data", default="datasets/symbol_detector_listwise_recovery_v47_p2_transfer_locked_full_v2/manifest.json")
    parser.add_argument("--output-dir", default="checkpoints/symbol_uncovered_target_policy_v68")
    parser.add_argument("--eval-output", default="reports/vlm/symbol_uncovered_target_policy_v68_eval.json")
    parser.add_argument("--smoke-predictions-output", default="reports/vlm/symbol_uncovered_target_policy_v68_smoke_predictions.jsonl")
    parser.add_argument("--candidate-inflation-target", type=float, default=7.999)
    parser.add_argument("--n-estimators", type=int, default=300)
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
    model = ExtraTreesClassifier(
        n_estimators=args.n_estimators,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight={0: 1.0, 1: 20.0},
        n_jobs=2,
        random_state=args.seed,
    )
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
    dev_grid: list[dict[str, Any]] = []
    for threshold in [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5]:
        for max_add_per_page in [1, 2, 3, 5, 8, 10]:
            metrics, _, audit = evaluate_policy(
                action_rows,
                recovery_rows,
                model,
                names,
                "dev",
                threshold,
                max_add_per_page,
                args.candidate_inflation_target,
            )
            dev_grid.append({"threshold": threshold, "max_add_per_page": max_add_per_page, "metrics": metrics, "audit": audit})
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
    )
    report = {
        "version": "symbol_uncovered_target_policy_v68",
        "source_integrity": {
            "gold_used_for_inference": False,
            "runtime_uses_svg_or_cad_geometry": False,
            "runtime_features": ["candidate score", "predicted label", "bbox area", "cluster/page features", "base selected-set context counts"],
            "offline_labels_used_for": ["new-target classifier training", "dev_policy_selection", "smoke_evaluation"],
        },
        "training": {
            "checkpoint": rel(model_path),
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
            "selected_on": "dev",
        },
        "dev": selected["metrics"],
        "dev_audit": selected["audit"],
        "smoke_eval": smoke_metrics,
        "smoke_audit": smoke_audit,
        "gate": {
            "smoke_recall_gt_baseline": smoke_metrics["symbol_bbox_iou_0_30"]["recall"] > baseline_smoke["symbol_bbox_iou_0_30"]["recall"],
            "smoke_recall_gt_v63": smoke_metrics["symbol_bbox_iou_0_30"]["recall"] > 0.6845,
            "smoke_recall_gte_0_70": smoke_metrics["symbol_bbox_iou_0_30"]["recall"] >= 0.70,
            "smoke_candidate_inflation_lt_8": smoke_metrics["candidate_inflation"] < 8.0,
            "no_oracle_inference": True,
        },
        "dev_grid": [
            {
                "threshold": row["threshold"],
                "max_add_per_page": row["max_add_per_page"],
                "precision": row["metrics"]["symbol_bbox_iou_0_30"]["precision"],
                "recall": row["metrics"]["symbol_bbox_iou_0_30"]["recall"],
                "center_recall": row["metrics"]["symbol_bbox_center_recall"],
                "candidate_inflation": row["metrics"]["candidate_inflation"],
                "added": row["audit"]["route"].get("added", 0),
                "added_new_iou_target": row["audit"]["route"].get("added_bucket:new_iou_target", 0),
                "added_background_or_support": row["audit"]["route"].get("added_bucket:background_or_support", 0),
            }
            for row in dev_grid
        ],
    }
    report["gate"]["passed"] = all(report["gate"].values())
    write_json(source_path(args.eval_output), report)
    write_jsonl(source_path(args.smoke_predictions_output), prediction_rows(smoke_selected))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
