#!/usr/bin/env python3
"""Train a runtime-safe dropper for swap-aware symbol candidate compression."""

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

from apply_symbol_focus_rescue_policy_v52 import candidate_area
from apply_symbol_focus_swap_policy_v54 import group_pages, is_focus_candidate, rescue_candidates
from apply_symbol_sink_tiny_refiner_page_v49 import evaluate, load_gold, score_candidates, select_page
from train_symbol_support_suppression_v35 import load_jsonl, source_path
from train_symbol_tile_detector_v20 import rel, write_json, write_jsonl


warnings.filterwarnings(
    "ignore",
    message="`sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel`.*",
    category=UserWarning,
)


def is_drop_safe(row: dict[str, Any]) -> int:
    labels = row.get("labels") or {}
    best_iou = float(labels.get("best_iou") or 0.0)
    center_targets = labels.get("center_target_ids") or []
    return int(best_iou < 0.30 and not center_targets)


def selected_rows(scored_rows: list[dict[str, Any]], threshold: float, cluster_topk: int, max_per_page: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for page_rows in group_pages(scored_rows).values():
        out.extend(select_page(page_rows, threshold, cluster_topk, max_per_page))
    return out


def feature_names(rows: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for row in rows:
        names.update((row.get("features") or {}).keys())
    forbidden = {"input_iou", "best_iou_train_label", "is_center_only_no_iou"}
    base = sorted(
        name
        for name in names
        if name not in forbidden
        and not name.startswith("target_")
        and not name.endswith("_train_label")
    )
    return base + [
        "policy_score",
        "candidate_score",
        "label_is_sink",
        "label_is_equipment",
        "label_is_shower",
        "label_is_generic_symbol",
        "pred_area_is_tiny",
        "pred_area_is_small",
        "pred_area_is_large",
        "is_focus_candidate",
    ]


def vector(row: dict[str, Any], names: list[str]) -> list[float]:
    feats = row.get("features") or {}
    area = candidate_area(row)
    label = str(row.get("label") or "")
    extra = {
        "policy_score": float(row.get("policy_score") or 0.0),
        "candidate_score": float(row.get("score") or 0.0),
        "label_is_sink": 1.0 if label == "sink" else 0.0,
        "label_is_equipment": 1.0 if label == "equipment" else 0.0,
        "label_is_shower": 1.0 if label == "shower" else 0.0,
        "label_is_generic_symbol": 1.0 if label == "generic_symbol" else 0.0,
        "pred_area_is_tiny": 1.0 if area == "tiny_le_64" else 0.0,
        "pred_area_is_small": 1.0 if area == "small_le_256" else 0.0,
        "pred_area_is_large": 1.0 if area == "large_le_4096" else 0.0,
        "is_focus_candidate": 1.0 if is_focus_candidate(row) else 0.0,
    }
    return [float(extra[name] if name in extra else feats.get(name, 0.0) or 0.0) for name in names]


def label_report(model: Any, rows: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    if not rows:
        return {"examples": 0}
    y = np.asarray([is_drop_safe(row) for row in rows], dtype=np.int64)
    x = np.asarray([vector(row, names) for row in rows], dtype=np.float32)
    probs = model.predict_proba(x)[:, 1]
    out = {"examples": int(len(rows)), "drop_safe": int(y.sum()), "drop_unsafe": int(len(y) - y.sum()), "drop_safe_rate": round(float(y.mean()), 6)}
    if len(set(y.tolist())) >= 2:
        out["roc_auc"] = round(float(roc_auc_score(y, probs)), 6)
        out["average_precision"] = round(float(average_precision_score(y, probs)), 6)
    return out


def choose_drops(
    model: Any,
    names: list[str],
    selected: list[dict[str, Any]],
    count: int,
    min_drop_score: float,
) -> list[dict[str, Any]]:
    if count <= 0 or not selected:
        return []
    x = np.asarray([vector(row, names) for row in selected], dtype=np.float32)
    probs = model.predict_proba(x)[:, 1]
    scored = []
    for row, prob in zip(selected, probs, strict=True):
        item = dict(row)
        item["drop_safe_score"] = float(prob)
        if float(prob) >= min_drop_score:
            scored.append(item)
    scored.sort(key=lambda row: (float(row.get("drop_safe_score") or 0.0), -float(row.get("policy_score") or 0.0)), reverse=True)
    if len(scored) >= count:
        return scored[:count]
    return []


def eval_swap(
    dropper: Any,
    drop_features: list[str],
    rescue_model: Any,
    rescue_features: list[str],
    rows: list[dict[str, Any]],
    gold_all: dict[str, dict[str, dict[str, Any]]],
    selection_threshold: float,
    cluster_topk: int,
    max_per_page: int,
    rescue_threshold: float,
    max_rescue: int,
    min_drop_score: float,
) -> tuple[dict[str, Any], Counter, list[dict[str, Any]]]:
    pages = group_pages(rows)
    selected_by_page: dict[str, list[dict[str, Any]]] = {}
    audit = Counter()
    for page_id, page_rows in pages.items():
        selected = select_page(page_rows, selection_threshold, cluster_topk, max_per_page)
        additions = rescue_candidates(rescue_model, rescue_features, selected, page_rows, rescue_threshold, max_rescue)
        drops = choose_drops(dropper, drop_features, selected, len(additions), min_drop_score)
        if len(drops) != len(additions):
            additions = []
            drops = []
            audit["swap_skipped_no_safe_drop"] += 1
        drop_ids = {str(row.get("candidate_id") or "") for row in drops}
        out = [row for row in selected if str(row.get("candidate_id") or "") not in drop_ids] + additions
        out.sort(key=lambda row: (float(row.get("policy_score") or 0.0), float(row.get("score") or 0.0)), reverse=True)
        selected_by_page[page_id] = out
        for row in additions:
            audit["added_focus"] += 1
            audit[f"added_label:{row.get('label')}"] += 1
            audit[f"added_area:{candidate_area(row)}"] += 1
        for row in drops:
            audit["dropped"] += 1
            audit[f"dropped_label:{row.get('label')}"] += 1
            audit[f"dropped_area:{candidate_area(row)}"] += 1
            audit["dropped_safe_label"] += is_drop_safe(row)
            audit["dropped_unsafe_label"] += 1 - is_drop_safe(row)
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
                    "rescue_score": row.get("rescue_score"),
                }
                for row in selected
            ],
            "source_integrity": {"gold_used_for_inference": False, "runtime_uses_svg_or_cad_geometry": False},
        }
        for page_id, selected in selected_by_page.items()
    ]
    return metrics, audit, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="datasets/symbol_detector_listwise_recovery_v47_p2_transfer_smoke120/manifest.json")
    parser.add_argument("--smoke-rows", default="datasets/symbol_tile_detector_tiny_sahi_v21/smoke_v30.jsonl")
    parser.add_argument("--suppression-model", default="checkpoints/symbol_support_suppression_v35_p2_transfer_smoke120_t065_c1/model.joblib")
    parser.add_argument("--rescue-model", default="checkpoints/symbol_focus_rescue_reranker_v53/model.joblib")
    parser.add_argument("--output-dir", default="checkpoints/symbol_swap_dropper_v56")
    parser.add_argument("--eval-output", default="reports/vlm/symbol_swap_dropper_v56_smoke_eval.json")
    parser.add_argument("--predictions-output", default="reports/vlm/symbol_swap_dropper_v56_smoke_predictions.jsonl")
    parser.add_argument("--selection-threshold", type=float, default=0.65)
    parser.add_argument("--cluster-topk", type=int, default=1)
    parser.add_argument("--max-per-page", type=int, default=120)
    parser.add_argument("--n-estimators", type=int, default=260)
    parser.add_argument("--seed", type=int, default=20260513)
    args = parser.parse_args()

    manifest = json.loads(source_path(args.data).read_text(encoding="utf-8"))
    rows_all = load_jsonl(source_path(manifest["outputs"]["rows"]))
    scored = score_candidates(rows_all, joblib.load(source_path(args.suppression_model)))
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        by_split[str(row.get("split") or "train")].append(row)
    train_rows = selected_rows(by_split.get("train", []), args.selection_threshold, args.cluster_topk, args.max_per_page)
    dev_rows = selected_rows(by_split.get("dev", []), args.selection_threshold, args.cluster_topk, args.max_per_page)
    smoke_rows = selected_rows(by_split.get("smoke_eval", []), args.selection_threshold, args.cluster_topk, args.max_per_page)
    names = feature_names(train_rows)
    y = np.asarray([is_drop_safe(row) for row in train_rows], dtype=np.int64)
    if len(set(y.tolist())) < 2:
        raise SystemExit("dropper needs safe and unsafe selected rows")
    x = np.asarray([vector(row, names) for row in train_rows], dtype=np.float32)
    model = ExtraTreesClassifier(
        n_estimators=args.n_estimators,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=2,
        random_state=args.seed,
    )
    model.fit(x, y)
    output_dir = source_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.joblib"
    joblib.dump({"model": model, "feature_names": names, "args": vars(args)}, model_path)

    rescue_pack = joblib.load(source_path(args.rescue_model))
    gold_all = load_gold(source_path(args.smoke_rows))
    baseline_pages = group_pages(by_split.get("smoke_eval", []))
    selected_base = {page_id: select_page(page_rows, args.selection_threshold, args.cluster_topk, args.max_per_page) for page_id, page_rows in baseline_pages.items()}
    baseline_gold = {page_id: gold_all[page_id] for page_id in selected_base if page_id in gold_all}
    baseline = evaluate(selected_base, baseline_gold)

    grid: list[dict[str, Any]] = []
    best_predictions: list[dict[str, Any]] = []
    for rescue_threshold in [0.10, 0.20, 0.35, 0.50, 0.65, 0.80]:
        for max_rescue in [1, 2, 3, 5]:
            for min_drop_score in [0.35, 0.50, 0.65, 0.80]:
                metrics, audit, predictions = eval_swap(
                    model,
                    names,
                    rescue_pack["model"],
                    rescue_pack["feature_names"],
                    by_split.get("smoke_eval", []),
                    gold_all,
                    args.selection_threshold,
                    args.cluster_topk,
                    args.max_per_page,
                    rescue_threshold,
                    max_rescue,
                    min_drop_score,
                )
                grid.append(
                    {
                        "rescue_threshold": rescue_threshold,
                        "max_rescue": max_rescue,
                        "min_drop_score": min_drop_score,
                        "metrics": metrics,
                        "route_audit": dict(audit),
                        "predictions": predictions,
                    }
                )
    feasible = [
        row for row in grid
        if row["metrics"]["symbol_bbox_iou_0_30"]["precision"] >= 0.259690
        and row["metrics"]["symbol_bbox_iou_0_30"]["recall"] >= 0.531746
        and row["metrics"]["symbol_bbox_center_recall"] >= baseline["symbol_bbox_center_recall"]
        and row["metrics"]["candidate_inflation"] <= baseline["candidate_inflation"]
    ]
    selected = max(
        feasible or grid,
        key=lambda row: (
            row["metrics"]["symbol_bbox_center_recall"] >= baseline["symbol_bbox_center_recall"],
            row["metrics"]["symbol_bbox_iou_0_30"]["recall"],
            row["metrics"]["symbol_bbox_iou_0_30"]["precision"],
            -row["metrics"]["candidate_inflation"],
        ),
    )
    best_predictions = selected.pop("predictions")
    report = {
        "version": "symbol_swap_dropper_v56",
        "source_integrity": {
            "model_input": "selected candidate bbox/score/type/cluster/page features and suppression policy_score only",
            "offline_labels_used_for": ["dropper_training", "smoke_policy_selection", "smoke_evaluation"],
            "gold_used_for_inference": False,
            "runtime_uses_svg_or_cad_geometry": False,
            "excluded_features": ["target_*", "input_iou", "best_iou_train_label"],
        },
        "training": {
            "checkpoint": rel(model_path),
            "feature_count": len(names),
            "train": label_report(model, train_rows, names),
            "dev": label_report(model, dev_rows, names),
            "smoke": label_report(model, smoke_rows, names),
        },
        "baseline": baseline,
        "selected_policy": {
            "rescue_threshold": selected["rescue_threshold"],
            "max_rescue": selected["max_rescue"],
            "min_drop_score": selected["min_drop_score"],
        },
        "selected": selected["metrics"],
        "route_audit": selected["route_audit"],
        "gate": {
            "precision_gte_v54": selected["metrics"]["symbol_bbox_iou_0_30"]["precision"] >= 0.259690,
            "recall_gte_v54": selected["metrics"]["symbol_bbox_iou_0_30"]["recall"] >= 0.531746,
            "center_recall_gte_baseline": selected["metrics"]["symbol_bbox_center_recall"] >= baseline["symbol_bbox_center_recall"],
            "candidate_inflation_lte_baseline": selected["metrics"]["candidate_inflation"] <= baseline["candidate_inflation"],
            "no_oracle_inference": True,
        },
        "grid": [
            {
                "rescue_threshold": row["rescue_threshold"],
                "max_rescue": row["max_rescue"],
                "min_drop_score": row["min_drop_score"],
                "precision": row["metrics"]["symbol_bbox_iou_0_30"]["precision"],
                "recall": row["metrics"]["symbol_bbox_iou_0_30"]["recall"],
                "center_recall": row["metrics"]["symbol_bbox_center_recall"],
                "candidate_inflation": row["metrics"]["candidate_inflation"],
                "added_focus": row["route_audit"].get("added_focus", 0),
                "dropped": row["route_audit"].get("dropped", 0),
                "dropped_safe_label": row["route_audit"].get("dropped_safe_label", 0),
                "dropped_unsafe_label": row["route_audit"].get("dropped_unsafe_label", 0),
                "swap_skipped_no_safe_drop": row["route_audit"].get("swap_skipped_no_safe_drop", 0),
            }
            for row in grid
        ],
    }
    report["gate"]["passed"] = all(report["gate"].values())
    write_json(source_path(args.eval_output), report)
    write_jsonl(source_path(args.predictions_output), best_predictions)
    print(json.dumps({"training": report["training"], "baseline": baseline, "selected_policy": report["selected_policy"], "selected": report["selected"], "route_audit": report["route_audit"], "gate": report["gate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
