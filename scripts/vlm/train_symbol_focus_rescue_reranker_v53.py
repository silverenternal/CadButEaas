#!/usr/bin/env python3
"""Train a rescue reranker for unselected sink/tiny focus candidates."""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from apply_symbol_focus_rescue_policy_v52 import FOCUS_AREAS, FOCUS_LABELS, candidate_area
from apply_symbol_sink_tiny_refiner_page_v49 import evaluate, load_gold, score_candidates, select_page
from train_symbol_support_suppression_v35 import load_jsonl, source_path
from train_symbol_tile_detector_v20 import rel, write_json, write_jsonl


warnings.filterwarnings(
    "ignore",
    message="`sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel`.*",
    category=UserWarning,
)


def is_focus_candidate(row: dict[str, Any]) -> bool:
    return str(row.get("label") or "") in FOCUS_LABELS or candidate_area(row) in FOCUS_AREAS


def is_positive(row: dict[str, Any]) -> int:
    return int(float((row.get("labels") or {}).get("best_iou") or 0.0) >= 0.30)


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
    return base + ["policy_score", "candidate_score", "label_is_sink", "label_is_equipment", "pred_area_is_tiny", "pred_area_is_small"]


def vector(row: dict[str, Any], names: list[str]) -> list[float]:
    feats = row.get("features") or {}
    area = candidate_area(row)
    extra = {
        "policy_score": float(row.get("policy_score") or 0.0),
        "candidate_score": float(row.get("score") or 0.0),
        "label_is_sink": 1.0 if row.get("label") == "sink" else 0.0,
        "label_is_equipment": 1.0 if row.get("label") == "equipment" else 0.0,
        "pred_area_is_tiny": 1.0 if area == "tiny_le_64" else 0.0,
        "pred_area_is_small": 1.0 if area == "small_le_256" else 0.0,
    }
    return [float(extra[name] if name in extra else feats.get(name, 0.0) or 0.0) for name in names]


def group_pages(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    pages: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pages[str(row["page_id"])].append(row)
    return pages


def unselected_focus_rows(scored_rows: list[dict[str, Any]], threshold: float, cluster_topk: int, max_per_page: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for _page_id, page_rows in group_pages(scored_rows).items():
        selected = select_page(page_rows, threshold, cluster_topk, max_per_page)
        selected_ids = {str(row.get("candidate_id") or "") for row in selected}
        out.extend(
            row for row in page_rows
            if str(row.get("candidate_id") or "") not in selected_ids
            and is_focus_candidate(row)
        )
    return out


def label_report(model: Any, rows: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    if not rows:
        return {"examples": 0}
    y = np.asarray([is_positive(row) for row in rows], dtype=np.int64)
    x = np.asarray([vector(row, names) for row in rows], dtype=np.float32)
    probs = model.predict_proba(x)[:, 1]
    out = {"examples": int(len(rows)), "positive": int(y.sum()), "positive_rate": round(float(y.mean()), 6)}
    if len(set(y.tolist())) >= 2:
        out["roc_auc"] = round(float(roc_auc_score(y, probs)), 6)
        out["average_precision"] = round(float(average_precision_score(y, probs)), 6)
    return out


def choose_training_split(
    candidates_by_split: dict[str, list[dict[str, Any]]],
    preferred: str = "train",
    fallback: str = "dev",
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    summary: dict[str, Any] = {}
    for split_name, rows in candidates_by_split.items():
        labels = [is_positive(row) for row in rows]
        summary[split_name] = {
            "examples": len(rows),
            "positive": int(sum(labels)),
            "negative": int(len(labels) - sum(labels)),
            "has_both_classes": len(set(labels)) >= 2,
        }
    preferred_rows = candidates_by_split.get(preferred, [])
    preferred_labels = [is_positive(row) for row in preferred_rows]
    if len(set(preferred_labels)) >= 2:
        return preferred, preferred_rows, summary
    fallback_rows = candidates_by_split.get(fallback, [])
    fallback_labels = [is_positive(row) for row in fallback_rows]
    if len(set(fallback_labels)) >= 2:
        return fallback, fallback_rows, summary
    raise SystemExit(
        "rescue reranker needs positive and negative rows; "
        f"split_summary={json.dumps(summary, ensure_ascii=False)}"
    )


def rescue_page(
    model: Any,
    names: list[str],
    selected: list[dict[str, Any]],
    page_rows: list[dict[str, Any]],
    threshold: float,
    max_rescue: int,
) -> tuple[list[dict[str, Any]], Counter]:
    audit = Counter()
    selected_ids = {str(row.get("candidate_id") or "") for row in selected}
    pool = [
        row for row in page_rows
        if str(row.get("candidate_id") or "") not in selected_ids
        and is_focus_candidate(row)
    ]
    if not pool:
        return selected, audit
    x = np.asarray([vector(row, names) for row in pool], dtype=np.float32)
    probs = model.predict_proba(x)[:, 1]
    scored = []
    for row, prob in zip(pool, probs, strict=True):
        item = dict(row)
        item["rescue_score"] = float(prob)
        if float(prob) >= threshold:
            scored.append(item)
    scored.sort(key=lambda row: (float(row.get("rescue_score") or 0.0), float(row.get("policy_score") or 0.0), float(row.get("score") or 0.0)), reverse=True)
    rescued = scored[:max_rescue]
    for row in rescued:
        audit["rescued"] += 1
        audit[f"rescued_label:{row.get('label')}"] += 1
        audit[f"rescued_area:{candidate_area(row)}"] += 1
    out = list(selected) + rescued
    out.sort(key=lambda row: (float(row.get("policy_score") or 0.0), float(row.get("score") or 0.0)), reverse=True)
    return out, audit


def eval_policy(
    model: Any,
    names: list[str],
    rows: list[dict[str, Any]],
    gold_all: dict[str, dict[str, dict[str, Any]]],
    selection_threshold: float,
    cluster_topk: int,
    max_per_page: int,
    rescue_threshold: float,
    max_rescue: int,
) -> tuple[dict[str, Any], Counter, list[dict[str, Any]]]:
    pages = group_pages(rows)
    selected_base = {page_id: select_page(page_rows, selection_threshold, cluster_topk, max_per_page) for page_id, page_rows in pages.items()}
    selected_rescue: dict[str, list[dict[str, Any]]] = {}
    audit = Counter()
    for page_id, page_rows in pages.items():
        out, row_audit = rescue_page(model, names, selected_base.get(page_id, []), page_rows, rescue_threshold, max_rescue)
        selected_rescue[page_id] = out
        audit.update(row_audit)
    gold = {page_id: gold_all[page_id] for page_id in selected_rescue if page_id in gold_all}
    metrics = evaluate(selected_rescue, gold)
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
        for page_id, selected in selected_rescue.items()
    ]
    return metrics, audit, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="datasets/symbol_detector_listwise_recovery_v47_p2_transfer_smoke120/manifest.json")
    parser.add_argument("--smoke-rows", default="datasets/symbol_tile_detector_tiny_sahi_v21/smoke_v30.jsonl")
    parser.add_argument("--suppression-model", default="checkpoints/symbol_support_suppression_v35_p2_transfer_smoke120_t065_c1/model.joblib")
    parser.add_argument("--output-dir", default="checkpoints/symbol_focus_rescue_reranker_v53")
    parser.add_argument("--eval-output", default="reports/vlm/symbol_focus_rescue_reranker_v53_smoke_eval.json")
    parser.add_argument("--predictions-output", default="reports/vlm/symbol_focus_rescue_reranker_v53_smoke_predictions.jsonl")
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
    rescue_candidates_by_split = {
        split_name: unselected_focus_rows(rows, args.selection_threshold, args.cluster_topk, args.max_per_page)
        for split_name, rows in by_split.items()
    }
    train_split_used, train_rows, rescue_split_summary = choose_training_split(rescue_candidates_by_split)
    dev_rows = rescue_candidates_by_split.get("dev", [])
    smoke_pool_rows = rescue_candidates_by_split.get("smoke_eval", [])
    names = feature_names(train_rows)
    x = np.asarray([vector(row, names) for row in train_rows], dtype=np.float32)
    y = np.asarray([is_positive(row) for row in train_rows], dtype=np.int64)
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

    gold_all = load_gold(source_path(args.smoke_rows))
    baseline_pages = group_pages(by_split.get("smoke_eval", []))
    selected_base = {page_id: select_page(page_rows, args.selection_threshold, args.cluster_topk, args.max_per_page) for page_id, page_rows in baseline_pages.items()}
    baseline_gold = {page_id: gold_all[page_id] for page_id in selected_base if page_id in gold_all}
    baseline = evaluate(selected_base, baseline_gold)

    dev_grid: list[dict[str, Any]] = []
    for threshold in [0.10, 0.20, 0.35, 0.50, 0.65, 0.80]:
        for max_rescue in [1, 2, 3, 5]:
            metrics, audit, _preds = eval_policy(model, names, by_split.get("dev", []), gold_all, args.selection_threshold, args.cluster_topk, args.max_per_page, threshold, max_rescue)
            dev_grid.append({"threshold": threshold, "max_rescue": max_rescue, "metrics": metrics, "route_audit": dict(audit)})
    dev_selected = max(
        dev_grid,
        key=lambda row: (
            row["metrics"]["symbol_bbox_iou_0_30"]["precision"] >= baseline["symbol_bbox_iou_0_30"]["precision"],
            row["metrics"]["symbol_bbox_iou_0_30"]["recall"],
            row["metrics"]["symbol_bbox_iou_0_30"]["precision"],
            -row["metrics"]["candidate_inflation"],
        ),
    )
    smoke, smoke_audit, predictions = eval_policy(
        model,
        names,
        by_split.get("smoke_eval", []),
        gold_all,
        args.selection_threshold,
        args.cluster_topk,
        args.max_per_page,
        float(dev_selected["threshold"]),
        int(dev_selected["max_rescue"]),
    )
    report = {
        "version": "symbol_focus_rescue_reranker_v53",
        "source_integrity": {
            "model_input": "candidate bbox/score/type/cluster/page features and suppression policy_score only",
            "offline_labels_used_for": ["rescue_reranker_training", "dev_threshold_selection", "smoke_evaluation"],
            "gold_used_for_inference": False,
            "runtime_uses_svg_or_cad_geometry": False,
            "excluded_features": ["target_*", "input_iou", "best_iou_train_label"],
        },
        "training": {
            "checkpoint": rel(model_path),
            "feature_count": len(names),
            "train_split_used": train_split_used,
            "split_summary": rescue_split_summary,
            "train": label_report(model, train_rows, names),
            "dev": label_report(model, dev_rows, names),
            "smoke_pool": label_report(model, smoke_pool_rows, names),
        },
        "baseline": baseline,
        "selected_policy": {"threshold": float(dev_selected["threshold"]), "max_rescue": int(dev_selected["max_rescue"]), "selected_on": "dev"},
        "smoke": smoke,
        "route_audit": dict(smoke_audit),
        "gate": {
            "precision_gte_baseline": smoke["symbol_bbox_iou_0_30"]["precision"] >= baseline["symbol_bbox_iou_0_30"]["precision"],
            "recall_gt_baseline": smoke["symbol_bbox_iou_0_30"]["recall"] > baseline["symbol_bbox_iou_0_30"]["recall"],
            "candidate_inflation_lte_2_5": smoke["candidate_inflation"] <= 2.5,
            "no_oracle_inference": True,
        },
        "dev_grid": [
            {
                "threshold": row["threshold"],
                "max_rescue": row["max_rescue"],
                "precision": row["metrics"]["symbol_bbox_iou_0_30"]["precision"],
                "recall": row["metrics"]["symbol_bbox_iou_0_30"]["recall"],
                "f1": row["metrics"]["symbol_bbox_iou_0_30"]["f1"],
                "candidate_inflation": row["metrics"]["candidate_inflation"],
                "rescued": row["route_audit"].get("rescued", 0),
            }
            for row in dev_grid
        ],
    }
    report["gate"]["passed"] = all(report["gate"].values())
    write_json(source_path(args.eval_output), report)
    write_jsonl(source_path(args.predictions_output), predictions)
    print(json.dumps({"training": report["training"], "baseline": baseline, "selected_policy": report["selected_policy"], "smoke": smoke, "route_audit": report["route_audit"], "gate": report["gate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
