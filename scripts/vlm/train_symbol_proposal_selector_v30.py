#!/usr/bin/env python3
"""Train a lightweight candidate selector for v30 symbol proposals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from train_symbol_tile_detector_v20 import load_jsonl, rel, write_json


def matrix(rows: list[dict[str, Any]], feature_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray([[float((row.get("features") or {}).get(name, 0.0)) for name in feature_names] for row in rows], dtype=np.float32)
    y = np.asarray([int((row.get("labels") or {}).get("target") or 0) for row in rows], dtype=np.int64)
    groups = np.asarray([str(row.get("row_id")) for row in rows])
    return x, y, groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="datasets/symbol_proposal_selector_v30/smoke_features.jsonl")
    parser.add_argument("--output", default="checkpoints/symbol_proposal_merger_v30/listwise_selector.joblib")
    parser.add_argument("--report-output", default="reports/vlm/symbol_proposal_selector_v30_smoke_train_report.json")
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--max-iter", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.06)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.features))
    if not rows:
        raise SystemExit("empty feature table")
    feature_names = sorted(rows[0]["features"])
    x, y, groups = matrix(rows, feature_names)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=args.seed)
    train_idx, val_idx = next(splitter.split(x, y, groups))

    pos = max(int(y[train_idx].sum()), 1)
    neg = max(int((1 - y[train_idx]).sum()), 1)
    sample_weight = np.where(y[train_idx] == 1, neg / pos, 1.0)
    model = HistGradientBoostingClassifier(
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        max_leaf_nodes=31,
        l2_regularization=0.01,
        random_state=args.seed,
    )
    model.fit(x[train_idx], y[train_idx], sample_weight=sample_weight)
    probs = model.predict_proba(x[val_idx])[:, 1]
    threshold_rows: list[dict[str, Any]] = []
    for threshold in [0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
        pred = (probs >= threshold).astype(np.int64)
        precision, recall, f1, _support = precision_recall_fscore_support(y[val_idx], pred, average="binary", zero_division=0)
        threshold_rows.append(
            {
                "threshold": threshold,
                "precision": round(float(precision), 6),
                "recall": round(float(recall), 6),
                "f1": round(float(f1), 6),
                "kept_rate": round(float(pred.mean()), 6),
            }
        )
    selected = sorted(threshold_rows, key=lambda row: (row["recall"] >= 0.92, row["precision"], -row["kept_rate"], row["f1"]), reverse=True)[0]
    bundle = {
        "model_type": "symbol_proposal_selector_v30_hist_gradient_boosting",
        "feature_names": feature_names,
        "model": model,
        "selected_threshold": selected["threshold"],
        "training_features": rel(Path(args.features)),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output)
    report = {
        "version": "symbol_proposal_selector_v30_train_report",
        "metric_mode": "smoke_group_holdout",
        "features": rel(Path(args.features)),
        "output": rel(Path(args.output)),
        "counts": {
            "rows": len(rows),
            "pages": len(set(groups.tolist())),
            "positives": int(y.sum()),
            "negatives": int((1 - y).sum()),
            "train_rows": int(len(train_idx)),
            "val_rows": int(len(val_idx)),
        },
        "validation": {
            "roc_auc": round(float(roc_auc_score(y[val_idx], probs)), 6) if len(set(y[val_idx].tolist())) > 1 else None,
            "average_precision": round(float(average_precision_score(y[val_idx], probs)), 6),
            "threshold_grid": threshold_rows,
            "selected_threshold": selected,
        },
        "feature_names": feature_names,
    }
    write_json(Path(args.report_output), report)
    print(json.dumps({"output": rel(Path(args.output)), "validation": report["validation"], "counts": report["counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
