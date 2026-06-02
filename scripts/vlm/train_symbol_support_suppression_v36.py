#!/usr/bin/env python3
"""Train v36 source-calibrated suppression policy with dev threshold selection."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from train_symbol_support_suppression_v35 import load_jsonl, split_rows, vector, y
from train_symbol_tile_detector_v20 import rel, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]


def source_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def feature_names(rows: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for row in rows:
        names.update((row.get("features") or {}).keys())
    return sorted(names)


def sample_train(rows: list[dict[str, Any]], negative_ratio: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    positives = [row for row in rows if y(row)]
    negatives = [row for row in rows if not y(row)]
    negatives.sort(key=lambda row: (float(row.get("score", 0.0)), (row.get("labels") or {}).get("suppression_reason") == "center_only_no_iou"), reverse=True)
    hard_n = min(len(negatives), max(len(positives) * negative_ratio, 1000))
    selected = positives + negatives[:hard_n]
    tail = negatives[hard_n:]
    if tail:
        random_n = min(len(tail), max(len(positives), 1000))
        idx = rng.choice(np.arange(len(tail)), size=random_n, replace=False)
        selected.extend(tail[int(i)] for i in idx)
    return selected


def label_report(model: Any, rows: list[dict[str, Any]], names: list[str], max_rows: int = 120000) -> dict[str, Any]:
    if not rows:
        return {"examples": 0}
    sample = rows[:max_rows]
    xx = np.asarray([vector(row, names) for row in sample], dtype=np.float32)
    yy = np.asarray([y(row) for row in sample], dtype=np.int64)
    prob = model.predict_proba(xx)[:, 1]
    out = {"examples": int(len(sample)), "positive": int(yy.sum()), "positive_rate": round(float(yy.mean()), 6)}
    if len(set(yy.tolist())) >= 2:
        out["roc_auc"] = round(float(roc_auc_score(yy, prob)), 6)
        out["average_precision"] = round(float(average_precision_score(yy, prob)), 6)
    return out


def metric_view(report: dict[str, Any]) -> dict[str, float]:
    return {
        "center_recall": float(report["symbol_bbox_center_recall"]),
        "iou_0_30_recall": float(report["symbol_bbox_iou_0_30"]["recall"]),
        "precision": float(report["symbol_bbox_iou_0_30"]["precision"]),
        "f1": float(report["symbol_bbox_iou_0_30"]["f1"]),
        "candidate_inflation": float(report["candidate_inflation"]),
    }


def score_rows(model: Any, rows: list[dict[str, Any]], names: list[str], batch_size: int = 100000) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        xx = np.asarray([vector(row, names) for row in chunk], dtype=np.float32)
        probs = model.predict_proba(xx)[:, 1]
        for row, prob in zip(chunk, probs, strict=True):
            item = dict(row)
            item["policy_score"] = float(prob)
            scored.append(item)
    return scored


def evaluate_scored(rows: list[dict[str, Any]], split: str, threshold: float, max_per_page: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_page[str(row["page_id"])].append(row)
    totals = Counter()
    by_source = Counter()
    by_reason = Counter()
    by_label_miss = Counter()
    by_area_miss = Counter()
    predictions: list[dict[str, Any]] = []
    for page_id, page_rows in by_page.items():
        selected = [row for row in page_rows if float(row.get("policy_score", 0.0)) >= threshold]
        selected.sort(key=lambda row: (float(row.get("policy_score", 0.0)), float(row.get("score", 0.0))), reverse=True)
        selected = selected[:max_per_page]
        gold_targets: set[str] = set()
        target_label: dict[str, str] = {}
        target_area: dict[str, str] = {}
        center_hits: set[str] = set()
        iou_hits: set[str] = set()
        typed_correct = 0
        typed_total = 0
        for row in page_rows:
            labels = row.get("labels") or {}
            for gold in labels.get("page_gold_targets") or []:
                target = str(gold.get("target_id") or "")
                if not target:
                    continue
                gold_targets.add(target)
                target_label.setdefault(target, str(gold.get("label") or "generic_symbol"))
                target_area.setdefault(target, str(gold.get("area_bucket") or "unknown"))
        for row in selected:
            labels = row.get("labels") or {}
            best_iou = float(labels.get("best_iou", 0.0) or 0.0)
            target = labels.get("best_iou_target_id")
            for center_target in labels.get("center_target_ids") or []:
                center_hits.add(str(center_target))
            if target and best_iou >= 0.30:
                iou_hits.add(str(target))
                typed_total += 1
                if str(row.get("label") or "") == target_label.get(str(target), ""):
                    typed_correct += 1
            by_source[str(row.get("proposal_source") or "unknown")] += 1
            if best_iou < 0.30:
                by_reason[str(labels.get("suppression_reason") or "unknown")] += 1
        for target in gold_targets:
            if target not in iou_hits:
                by_label_miss[target_label.get(target, "unknown")] += 1
                by_area_miss[target_area.get(target, "unknown")] += 1
        totals["gold"] += len(gold_targets)
        totals["selected"] += len(selected)
        totals["iou_hit"] += len(iou_hits)
        totals["center_hit"] += len(center_hits & gold_targets)
        totals["typed_total"] += typed_total
        totals["typed_correct"] += typed_correct
        predictions.append(
            {
                "page_id": page_id,
                "predicted_symbols": [
                    {
                        "candidate_id": row["candidate_id"],
                        "bbox": row["bbox"],
                        "label": row["label"],
                        "confidence": round(float(row.get("policy_score", 0.0)), 6),
                        "proposal_source": row["proposal_source"],
                    }
                    for row in selected
                ],
                "source_integrity": {"gold_used_for_inference": False, "runtime_uses_svg_or_cad_geometry": False},
            }
        )
    precision = totals["iou_hit"] / max(totals["selected"], 1)
    recall = totals["iou_hit"] / max(totals["gold"], 1)
    center_recall = totals["center_hit"] / max(totals["gold"], 1)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return (
        {
            "split": split,
            "pages": len(by_page),
            "symbol_bbox_center_recall": round(center_recall, 6),
            "symbol_bbox_iou_0_30": {
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "true_positive": int(totals["iou_hit"]),
                "predicted": int(totals["selected"]),
                "gold": int(totals["gold"]),
            },
            "candidate_inflation": round(totals["selected"] / max(totals["gold"], 1), 6),
            "typed_accuracy_on_iou_matches": round(totals["typed_correct"] / max(totals["typed_total"], 1), 6),
            "selected_by_source": dict(by_source),
            "selected_negative_reasons": dict(by_reason),
            "missed_iou_by_label": dict(by_label_miss),
            "missed_iou_by_area": dict(by_area_miss),
        },
        predictions,
    )


def choose_policy(dev_rows: list[dict[str, Any]]) -> dict[str, Any]:
    grid = []
    for threshold in [0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
        for max_per_page in [60, 90, 120, 180, 240]:
            report, _ = evaluate_scored(dev_rows, "dev", threshold, max_per_page)
            grid.append({"threshold": threshold, "cluster_topk": 0, "max_per_page": max_per_page, "metrics": report, "view": metric_view(report)})
    selected = sorted(
        grid,
        key=lambda row: (
            row["view"]["precision"] >= 0.12,
            row["view"]["candidate_inflation"] <= 7.0,
            row["view"]["center_recall"],
            row["view"]["iou_0_30_recall"],
            row["view"]["precision"],
            -row["view"]["candidate_inflation"],
        ),
        reverse=True,
    )[0]
    return {
        "threshold": selected["threshold"],
        "cluster_topk": selected["cluster_topk"],
        "max_per_page": selected["max_per_page"],
        "dev_selected_metrics": selected["metrics"],
        "grid": [
            {"threshold": item["threshold"], "cluster_topk": item["cluster_topk"], "max_per_page": item["max_per_page"], **item["view"]}
            for item in grid
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="datasets/symbol_support_suppression_v36/manifest.json")
    parser.add_argument("--output-dir", default="checkpoints/symbol_support_suppression_v36")
    parser.add_argument("--eval-output", default="reports/vlm/symbol_support_suppression_v36_locked_eval.json")
    parser.add_argument("--predictions-output", default="reports/vlm/symbol_support_suppression_v36_locked_predictions.jsonl")
    parser.add_argument("--negative-ratio", type=int, default=8)
    parser.add_argument("--n-estimators", type=int, default=260)
    parser.add_argument("--seed", type=int, default=20260512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = source_path(args.data)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = load_jsonl(source_path(manifest["outputs"]["rows"]))
    by_split = split_rows(rows)
    train_rows = by_split.get("train", [])
    dev_rows = by_split.get("dev", [])
    locked_rows = by_split.get("locked", [])
    names = feature_names(rows)
    sampled_train = sample_train(train_rows, args.negative_ratio, args.seed)
    train_x = np.asarray([vector(row, names) for row in sampled_train], dtype=np.float32)
    train_y = np.asarray([y(row) for row in sampled_train], dtype=np.int64)
    model = ExtraTreesClassifier(
        n_estimators=args.n_estimators,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=args.seed,
    )
    model.fit(train_x, train_y)
    dev_scored = score_rows(model, dev_rows, names)
    locked_scored = score_rows(model, locked_rows, names)
    policy = choose_policy(dev_scored)
    threshold = float(policy["threshold"])
    cluster_topk = int(policy["cluster_topk"])
    max_per_page = int(policy["max_per_page"])
    dev_report, _ = evaluate_scored(dev_scored, "dev", threshold, max_per_page)
    locked_report, locked_predictions = evaluate_scored(locked_scored, "locked", threshold, max_per_page)
    output_dir = source_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.joblib"
    joblib.dump({"model": model, "feature_names": names, "policy": {"threshold": threshold, "cluster_topk": cluster_topk, "max_per_page": max_per_page}, "args": vars(args)}, model_path)
    report = {
        "version": "symbol_support_suppression_v36_locked_eval",
        "task": "P1-06-train-dev-v35-set-policy-generalization",
        "claim_boundary": "Train on train subset, select threshold on dev subset, evaluate locked with fixed dev-selected policy. Locked is not used for threshold selection.",
        "source_integrity": {
            "model_input": "raster-derived candidate bbox/score/source/type fields only",
            "offline_labels_used_for": ["train_supervision", "dev_threshold_selection", "locked_evaluation"],
            "gold_used_for_inference": False,
            "runtime_uses_svg_or_cad_geometry": False,
        },
        "training": {
            "checkpoint": rel(model_path),
            "data": rel(manifest_path),
            "feature_count": len(names),
            "sampled_train_rows": len(sampled_train),
            "sampled_train_positive": int(train_y.sum()),
            "train_label_report": label_report(model, sampled_train, names),
            "dev_label_report": label_report(model, dev_rows, names),
        },
        "policy": {"threshold": threshold, "cluster_topk": cluster_topk, "max_per_page": max_per_page, "selected_on": "dev"},
        "policy_search": policy,
        "dev": dev_report,
        "locked": locked_report,
        "stage_gate": {
            "locked_center_recall_min_0_92": locked_report["symbol_bbox_center_recall"] >= 0.92,
            "locked_iou_0_30_recall_min_0_68": locked_report["symbol_bbox_iou_0_30"]["recall"] >= 0.68,
            "locked_precision_min_0_12": locked_report["symbol_bbox_iou_0_30"]["precision"] >= 0.12,
            "locked_candidate_inflation_max_7": locked_report["candidate_inflation"] <= 7.0,
            "must_not_use_locked_for_threshold_selection": True,
            "no_oracle_inference": True,
        },
    }
    report["stage_gate"]["passed"] = all(report["stage_gate"].values())
    write_json(source_path(args.eval_output), report)
    write_jsonl(source_path(args.predictions_output), locked_predictions)
    print(json.dumps({"policy": report["policy"], "dev": dev_report, "locked": locked_report, "stage_gate": report["stage_gate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
