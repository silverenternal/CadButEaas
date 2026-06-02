#!/usr/bin/env python3
"""Train and evaluate a tiny/multiscale proposal expert for symbol pages."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from train_symbol_tile_detector_v20 import area_bucket, bbox_iou, center_covered, load_jsonl, rel, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
EPS = 1e-6


def matrix(rows: list[dict[str, Any]], feature_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray([[float((row.get("features") or {}).get(name, 0.0)) for name in feature_names] for row in rows], dtype=np.float32)
    y = np.asarray([int((row.get("labels") or {}).get("target_iou_0_30") or 0) for row in rows], dtype=np.int64)
    groups = np.asarray([str(row.get("row_id")) for row in rows])
    return x, y, groups


LEAKAGE_FEATURES = {
    "best_iou_hint",
    "center_hit_hint",
}


def page_metrics(predictions: dict[str, list[dict[str, Any]]], pages: list[dict[str, Any]]) -> dict[str, Any]:
    total = Counter()
    selected_total = 0
    tiny_total = 0
    tiny_selected = 0
    area_counts = Counter()
    for page in pages:
        row_id = str(page.get("row_id"))
        gold_rows = list(page.get("gold_symbols") or [])
        preds = predictions.get(row_id, [])
        selected_total += len(preds)
        for gold in gold_rows:
            gold_box = [float(v) for v in gold.get("bbox") or []]
            if len(gold_box) != 4:
                continue
            total["gold"] += 1
            bucket = str(gold.get("area_bucket") or area_bucket([int(round(v)) for v in gold_box]))
            area_counts[bucket] += 1
            best_iou = 0.0
            best_center = False
            for pred in preds:
                box = [float(v) for v in pred.get("bbox") or []]
                if len(box) != 4:
                    continue
                best_iou = max(best_iou, bbox_iou(box, gold_box))
                best_center = best_center or center_covered(box, gold_box)
            total["iou_hit"] += int(best_iou >= 0.30)
            total["center_hit"] += int(best_center)
            if bucket in {"tiny_le_64", "small_le_256"}:
                tiny_total += 1
                tiny_selected += int(best_iou >= 0.30)
    return {
        "gold_total": int(total["gold"]),
        "center_recall": round(total["center_hit"] / max(total["gold"], 1), 6),
        "iou_0_30_recall": round(total["iou_hit"] / max(total["gold"], 1), 6),
        "tiny_iou_recall": round(tiny_selected / max(tiny_total, 1), 6),
        "tiny_gold_total": tiny_total,
        "selected_candidates": selected_total,
        "area_counts": dict(area_counts),
    }


def select_page_candidates(preds: list[dict[str, Any]], threshold: float, topk: int) -> list[dict[str, Any]]:
    ordered = sorted(preds, key=lambda row: float(row.get("score", 0.0)), reverse=True)
    kept = [pred for pred in ordered if float(pred.get("score", 0.0)) >= threshold]
    if topk > 0:
        kept = kept[:topk]
    return kept


def evaluate_policy(rows: list[dict[str, Any]], pages: list[dict[str, Any]], model: Any, feature_names: list[str], threshold: float, topk: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_page[str(row.get("row_id"))].append(row)
    page_predictions: dict[str, list[dict[str, Any]]] = {}
    flat_predictions: list[dict[str, Any]] = []
    for row_id, page_rows in by_page.items():
        if not page_rows:
            continue
        x = np.asarray([[float((row.get("features") or {}).get(name, 0.0)) for name in feature_names] for row in page_rows], dtype=np.float32)
        probs = model.predict_proba(x)[:, 1]
        scored = []
        for row, score in zip(page_rows, probs.tolist(), strict=True):
            pred = dict(row.get("candidate") or {})
            pred["score"] = float(score)
            pred["proposal_source"] = pred.get("proposal_source") or "unknown"
            scored.append(pred)
        ordered = sorted(scored, key=lambda row: float(row.get("score", 0.0)), reverse=True)
        kept = [pred for pred in ordered if float(pred.get("score", 0.0)) >= threshold]
        if topk > 0:
            kept = kept[:topk]
        page_predictions[row_id] = kept
        flat_predictions.extend({"row_id": row_id, **pred} for pred in kept)
    metrics = page_metrics(page_predictions, pages)
    metrics["candidate_inflation"] = round(metrics["selected_candidates"] / max(len(rows), 1), 6)
    return metrics, flat_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/symbol_tiny_multiscale_v33/manifest.json")
    parser.add_argument("--output-dir", default="checkpoints/symbol_tiny_multiscale_v33")
    parser.add_argument("--eval-output", default="reports/vlm/symbol_tiny_multiscale_v33_smoke_eval.json")
    parser.add_argument("--predictions-output", default="reports/vlm/symbol_tiny_multiscale_v33_smoke_predictions.jsonl")
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--max-iter", type=int, default=180)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    args = parser.parse_args()

    manifest = json.loads(Path(args.data).read_text(encoding="utf-8"))
    train_rows = load_jsonl(ROOT / manifest["outputs"]["train"])
    smoke_rows = load_jsonl(ROOT / manifest["outputs"]["smoke"])
    train_pages = load_jsonl(ROOT / manifest["outputs"]["train_pages"])
    smoke_pages = load_jsonl(ROOT / manifest["outputs"]["smoke_pages"])
    if not train_rows or not smoke_rows:
        raise SystemExit("missing tiny multiscale rows")

    feature_names = [name for name in list(manifest.get("feature_names") or sorted(train_rows[0]["features"])) if name not in LEAKAGE_FEATURES]
    x, y, groups = matrix(train_rows, feature_names)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=args.seed)
    train_idx, val_idx = next(splitter.split(x, y, groups))
    pos = max(int(y[train_idx].sum()), 1)
    neg = max(int((1 - y[train_idx]).sum()), 1)
    sample_weight = np.where(y[train_idx] == 1, neg / pos, 1.0)
    tiny_bonus = []
    for row in train_rows:
        bucket = str((row.get("match") or {}).get("gold_area_bucket") or "")
        tiny_bonus.append(3.0 if bucket == "tiny_le_64" else 2.0 if bucket == "small_le_256" else 1.0)
    sample_weight = sample_weight * np.asarray(tiny_bonus, dtype=np.float32)[train_idx]

    model = HistGradientBoostingClassifier(
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        max_leaf_nodes=31,
        l2_regularization=0.02,
        random_state=args.seed,
    )
    model.fit(x[train_idx], y[train_idx], sample_weight=sample_weight)

    probs = model.predict_proba(x[val_idx])[:, 1]
    threshold_rows: list[dict[str, Any]] = []
    for threshold in [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15]:
        for topk in [16, 24, 32, 48, 64, 96, 128]:
            pred = (probs >= threshold).astype(np.int64)
            precision, recall, f1, _support = precision_recall_fscore_support(y[val_idx], pred, average="binary", zero_division=0)
            threshold_rows.append(
                {
                    "threshold": threshold,
                    "topk": topk,
                    "precision": round(float(precision), 6),
                    "recall": round(float(recall), 6),
                    "f1": round(float(f1), 6),
                    "kept_rate": round(float(pred.mean()), 6),
                }
            )
    selected = sorted(
        threshold_rows,
        key=lambda row: (row["recall"], row["f1"], row["precision"], -row["kept_rate"]),
        reverse=True,
    )[0]

    val_row_ids = {str(train_rows[i].get("row_id")) for i in val_idx}
    val_pages = [page for page in train_pages if str(page.get("row_id")) in val_row_ids]
    val_metrics, _val_preds = evaluate_policy([train_rows[i] for i in val_idx], val_pages, model, feature_names, float(selected["threshold"]), int(selected["topk"]))
    smoke_metrics, smoke_preds = evaluate_policy(smoke_rows, smoke_pages, model, feature_names, float(selected["threshold"]), int(selected["topk"]))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_names": feature_names,
            "selected_policy": selected,
            "model_type": "symbol_tiny_multiscale_v33_hist_gradient_boosting",
            "training_manifest": rel(Path(args.data)),
        },
        out_dir / "model.joblib",
    )
    write_jsonl(Path(args.predictions_output), smoke_preds)
    report = {
        "version": "symbol_tiny_multiscale_v33_smoke_eval",
        "task": "P1-01-tiny-multiscale-proposal-expert-v33",
        "manifest": rel(Path(args.data)),
        "checkpoint": rel(out_dir / "model.joblib"),
        "counts": {
            "train_rows": len(train_rows),
            "smoke_rows": len(smoke_rows),
            "train_pages": len(set(groups.tolist())),
            "smoke_pages": len({row.get("row_id") for row in smoke_rows}),
            "train_positives": int(y.sum()),
            "train_negatives": int((1 - y).sum()),
        },
        "feature_names": feature_names,
        "validation": {
            "roc_auc": round(float(roc_auc_score(y[val_idx], probs)), 6) if len(set(y[val_idx].tolist())) > 1 else None,
            "average_precision": round(float(average_precision_score(y[val_idx], probs)), 6),
            "threshold_grid": threshold_rows,
            "selected_policy": selected,
            "metrics": val_metrics,
        },
        "smoke": {
            "metrics": smoke_metrics,
            "selected_policy": selected,
        },
        "gate": {
            "smoke_center_recall_min_0_90": smoke_metrics["center_recall"] >= 0.90,
            "smoke_iou_0_30_recall_min_0_72": smoke_metrics["iou_0_30_recall"] >= 0.72,
            "smoke_tiny_iou_recall_min_0_48": smoke_metrics["tiny_iou_recall"] >= 0.48,
        },
    }
    report["gate"]["passed"] = all(bool(value) for value in report["gate"].values())
    write_json(Path(args.eval_output), report)
    print(
        json.dumps(
            {
                "selected_policy": selected,
                "smoke": smoke_metrics,
                "gate": report["gate"],
                "checkpoint": rel(out_dir / "model.joblib"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
