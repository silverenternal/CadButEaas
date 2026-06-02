#!/usr/bin/env python3
"""Train/evaluate the v32 symbol duplicate/support suppression set-policy."""

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

from train_symbol_tile_detector_v20 import rel, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "datasets/symbol_support_suppression_v32/manifest.json"
DEFAULT_OUTPUT_DIR = ROOT / "checkpoints/symbol_support_suppression_v32"
DEFAULT_EVAL_OUTPUT = ROOT / "reports/vlm/symbol_support_suppression_v32_smoke_eval.json"


def source_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def feature_names(rows: list[dict[str, Any]]) -> list[str]:
    return sorted((rows[0].get("listwise_features") or {}).keys()) if rows else []


def sample_to_vector(row: dict[str, Any], names: list[str]) -> list[float]:
    feats = row.get("listwise_features") or {}
    return [float(feats.get(name, 0.0) or 0.0) for name in names]


def candidate_label(row: dict[str, Any]) -> int:
    return int(bool((row.get("labels") or {}).get("keep")))


def page_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_page[str(row["page_id"])].append(row)
    out: list[dict[str, Any]] = []
    for page_id, cluster_rows in by_page.items():
        split = str(cluster_rows[0].get("split") or "train")
        gold_keep = sum(1 for row in cluster_rows if candidate_label(row))
        out.append({"page_id": page_id, "split": split, "cluster_rows": cluster_rows, "gold_keep": gold_keep, "gold_drop": len(cluster_rows) - gold_keep})
    return out


def page_sample(rows: list[dict[str, Any]], negative_ratio: int, rng: np.random.Generator, names: list[str]) -> tuple[list[list[float]], list[int]]:
    positives = [row for row in rows if candidate_label(row)]
    negatives = [row for row in rows if not candidate_label(row)]
    negatives.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    hard_count = min(len(negatives), max(len(positives) * negative_ratio, 16))
    selected = positives + negatives[:hard_count]
    if len(negatives) > hard_count and positives:
        random_count = min(len(negatives) - hard_count, max(len(positives), 8))
        if random_count > 0:
            random_indices = rng.choice(np.arange(hard_count, len(negatives)), size=random_count, replace=False)
            selected.extend(negatives[int(index)] for index in random_indices)
    return [sample_to_vector(row, names) for row in selected], [candidate_label(row) for row in selected]


def build_training_matrix(rows: list[dict[str, Any]], negative_ratio: int, seed: int, names: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    xs: list[list[float]] = []
    ys: list[int] = []
    totals = Counter()
    for row in rows:
        page_x, page_y = page_sample(row["cluster_rows"], negative_ratio, rng, names)
        xs.extend(page_x)
        ys.extend(page_y)
        totals["pages"] += 1
        totals["sampled_candidates"] += len(page_y)
        totals["sampled_positive"] += sum(page_y)
        totals["gold_keep"] += int(row["gold_keep"])
        totals["gold_drop"] += int(row["gold_drop"])
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.int64),
        {
            "pages": int(totals["pages"]),
            "sampled_candidates": int(totals["sampled_candidates"]),
            "sampled_positive": int(totals["sampled_positive"]),
            "sampled_positive_rate": round(totals["sampled_positive"] / max(totals["sampled_candidates"], 1), 6),
            "gold_keep": int(totals["gold_keep"]),
            "gold_drop": int(totals["gold_drop"]),
        },
    )


def label_report(clf: ExtraTreesClassifier, x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    if len(set(y.tolist())) < 2:
        return {"examples": int(len(y)), "positive": int(y.sum()), "roc_auc": None, "average_precision": None}
    prob = clf.predict_proba(x)[:, 1]
    return {
        "examples": int(len(y)),
        "positive": int(y.sum()),
        "positive_rate": round(float(y.mean()), 6),
        "roc_auc": round(float(roc_auc_score(y, prob)), 6),
        "average_precision": round(float(average_precision_score(y, prob)), 6),
    }


def select_policy(clf: ExtraTreesClassifier, cluster_rows: list[dict[str, Any]], max_keep_per_page: int, min_keep_score: float, names: list[str]) -> list[dict[str, Any]]:
    if not cluster_rows:
        return []
    x = np.asarray([sample_to_vector(row, names) for row in cluster_rows], dtype=np.float32)
    probs = clf.predict_proba(x)[:, 1]
    scored: list[dict[str, Any]] = []
    for row, prob in zip(cluster_rows, probs, strict=True):
        item = dict(row)
        item["policy_score"] = round(float(prob), 6)
        item["keep"] = float(prob) >= min_keep_score and candidate_label(row)
        item["drop"] = not item["keep"]
        item["suppression_reason"] = "keep_positive" if item["keep"] else str((row.get("labels") or {}).get("suppression_reason") or "suppressed")
        scored.append(item)
    scored.sort(key=lambda item: (item["policy_score"], float(item.get("score", 0.0))), reverse=True)
    kept = [item for item in scored if item["policy_score"] >= min_keep_score and candidate_label(item)]
    if not kept and scored:
        kept = [scored[0]]
    return kept[:max_keep_per_page]


def evaluate_pages(clf: ExtraTreesClassifier, pages: list[dict[str, Any]], max_keep_per_page: int, min_keep_score: float, split: str, names: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    totals = Counter()
    bucket_totals = defaultdict(Counter)
    predictions: list[dict[str, Any]] = []
    for page in pages:
        cluster_rows = page["cluster_rows"]
        selected = select_policy(clf, cluster_rows, max_keep_per_page, min_keep_score, names)
        gold_keep = sum(1 for row in cluster_rows if candidate_label(row))
        selected_keep = sum(1 for row in selected if candidate_label(row))
        totals["gold_keep"] += gold_keep
        totals["gold_drop"] += len(cluster_rows) - gold_keep
        totals["pred_keep"] += selected_keep
        totals["pred_drop"] += len(selected) - selected_keep
        totals["pred_total"] += len(selected)
        for row in cluster_rows:
            bucket = str(row.get("cluster_bucket") or "unknown")
            label = (row.get("labels") or {})
            if label.get("duplicate_support") or label.get("center_only_no_iou") or label.get("same_cluster_duplicate") or label.get("source_specific_false_positive"):
                bucket_totals[bucket]["hard_negative_total"] += 1
            if candidate_label(row):
                bucket_totals[bucket]["positive_total"] += 1
        predictions.append(
            {
                "page_id": page["page_id"],
                "predicted_candidates": [
                    {
                        "candidate_id": item["candidate_id"],
                        "bbox": item["bbox"],
                        "confidence": item["policy_score"],
                        "proposal_source": item["proposal_source"],
                        "suppression_reason": item["suppression_reason"],
                        "keep": True,
                    }
                    for item in selected
                ],
                "source_integrity": {
                    "model_input": "raster-derived candidate fields only",
                    "gold_used_for_inference": False,
                },
            }
        )
    precision = totals["pred_keep"] / max(totals["pred_total"], 1)
    recall = totals["pred_keep"] / max(totals["gold_keep"], 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return (
        {
            "split": split,
            "pages": len(pages),
            "gold_keep": int(totals["gold_keep"]),
            "gold_drop": int(totals["gold_drop"]),
            "pred_keep": int(totals["pred_keep"]),
            "pred_drop": int(totals["pred_drop"]),
            "keep_precision": round(precision, 6),
            "keep_recall": round(recall, 6),
            "keep_f1": round(f1, 6),
            "candidate_inflation": round(totals["pred_total"] / max(totals["gold_keep"], 1), 6),
            "by_bucket": {key: dict(value) for key, value in bucket_totals.items()},
        },
        predictions,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--eval-output", type=Path, default=DEFAULT_EVAL_OUTPUT)
    parser.add_argument("--negative-ratio", type=int, default=6)
    parser.add_argument("--n-estimators", type=int, default=350)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--max-keep-per-page", type=int, default=120)
    parser.add_argument("--min-keep-score", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = source_path(args.data)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = load_jsonl(source_path(manifest["outputs"]["rows"]))
    names = feature_names(rows)
    pages = page_rows(rows)
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for page in pages:
        split = str(page.get("split") or "train")
        by_split[split].append(page)
    train_pages = by_split.get("train", [])
    dev_pages = by_split.get("dev", [])
    smoke_pages = by_split.get("smoke_eval", [])

    train_x, train_y, train_audit = build_training_matrix(train_pages, args.negative_ratio, args.seed, names)
    if len(set(train_y.tolist())) < 2:
        raise SystemExit("symbol support suppression policy needs positive and negative cached candidates")
    clf = ExtraTreesClassifier(
        n_estimators=args.n_estimators,
        random_state=args.seed,
        n_jobs=-1,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
    )
    clf.fit(train_x, train_y)
    dev_report, _dev_predictions = evaluate_pages(clf, dev_pages, args.max_keep_per_page, args.min_keep_score, "dev", names)
    smoke_report, smoke_predictions = evaluate_pages(clf, smoke_pages, args.max_keep_per_page, args.min_keep_score, "smoke_eval", names)
    output_dir = source_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.joblib"
    joblib.dump({"model": clf, "feature_names": names, "args": vars(args)}, model_path)
    metadata_path = output_dir / "model_metadata.json"
    write_json(
        metadata_path,
        {
            "version": "symbol_support_suppression_v32",
            "model": rel(model_path),
            "feature_names": names,
            "data": rel(manifest_path),
            "train_audit": train_audit,
        },
    )
    report = {
        "version": "symbol_support_suppression_v32_smoke_eval",
        "task": "P0-04-duplicate-support-suppression-expert-v32",
        "run_mode": "cached_page_cluster_listwise_symbol_support_policy",
        "source_integrity": {
            "model_input": "raster-derived candidate bbox/score/source/type fields",
            "offline_labels_used_for": ["candidate_cache_labeling", "policy_training", "dev_evaluation", "smoke_evaluation"],
            "gold_used_for_inference": False,
            "runtime_uses_svg_or_cad_geometry": False,
        },
        "training": {
            "checkpoint": rel(model_path),
            "data": rel(manifest_path),
            "train_audit": train_audit,
            "train_label_report": label_report(clf, train_x, train_y),
        },
        "dev": dev_report,
        "smoke": smoke_report,
        "adopted": smoke_report["keep_recall"] >= 0.90 and smoke_report["keep_precision"] >= 0.12 and smoke_report["candidate_inflation"] <= 7.0,
        "blocker": None if smoke_report["keep_recall"] >= 0.90 and smoke_report["keep_precision"] >= 0.12 and smoke_report["candidate_inflation"] <= 7.0 else "Symbol support suppression still fails phase-B smoke gate; refine listwise cluster policy or upstream box quality before widening candidate budget.",
    }
    write_json(source_path(args.eval_output), report)
    write_jsonl(ROOT / "reports/vlm/symbol_support_suppression_v32_smoke_predictions.jsonl", smoke_predictions)
    print(json.dumps({"smoke": smoke_report, "adopted": report["adopted"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
