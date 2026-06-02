#!/usr/bin/env python3
"""Train a cached page-level listwise policy for v19 raster text candidates."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from train_text_heatmap_affinity_v19 import ROOT, bbox_iou, write_json, write_jsonl
from train_text_peak_reranker_v19 import FEATURE_NAMES


CACHE = ROOT / "datasets/text_peak_candidate_cache_v19"
REPORT = ROOT / "reports/vlm"
OUT = ROOT / "checkpoints/text_listwise_policy_v19"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def candidate_features(candidate: dict[str, Any]) -> list[float]:
    return [float(candidate["features"][name]) for name in FEATURE_NAMES]


def candidate_label(candidate: dict[str, Any]) -> int:
    return int(bool(candidate["labels"]["center_positive"]))


def page_training_examples(row: dict[str, Any], negative_ratio: int, rng: np.random.Generator) -> tuple[list[list[float]], list[int]]:
    positives = [candidate for candidate in row["candidates"] if candidate_label(candidate)]
    negatives = [candidate for candidate in row["candidates"] if not candidate_label(candidate)]
    negatives.sort(key=lambda item: float(item["confidence"]), reverse=True)
    hard_count = min(len(negatives), max(len(positives) * negative_ratio, 32))
    hard_negatives = negatives[:hard_count]
    if len(negatives) > hard_count and positives:
        random_count = min(len(negatives) - hard_count, max(len(positives), 8))
        random_indices = rng.choice(np.arange(hard_count, len(negatives)), size=random_count, replace=False)
        hard_negatives.extend(negatives[int(index)] for index in random_indices)
    selected = positives + hard_negatives
    return [candidate_features(candidate) for candidate in selected], [candidate_label(candidate) for candidate in selected]


def build_training_matrix(rows: list[dict[str, Any]], negative_ratio: int, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    xs: list[list[float]] = []
    ys: list[int] = []
    totals = Counter()
    for row in rows:
        page_x, page_y = page_training_examples(row, negative_ratio, rng)
        xs.extend(page_x)
        ys.extend(page_y)
        totals["pages"] += 1
        totals["gold"] += int(row["gold_text_count"])
        totals["raw_candidates"] += int(row["candidate_count"])
        totals["sampled_candidates"] += len(page_y)
        totals["sampled_positive"] += sum(page_y)
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.int64),
        {
            "pages": int(totals["pages"]),
            "gold": int(totals["gold"]),
            "raw_candidates": int(totals["raw_candidates"]),
            "sampled_candidates": int(totals["sampled_candidates"]),
            "sampled_positive": int(totals["sampled_positive"]),
            "sampled_positive_rate": round(totals["sampled_positive"] / max(totals["sampled_candidates"], 1), 6),
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


def select_candidates(clf: ExtraTreesClassifier, candidates: list[dict[str, Any]], budget: int, peak_diversity: bool) -> list[dict[str, Any]]:
    if not candidates:
        return []
    x = np.asarray([candidate_features(candidate) for candidate in candidates], dtype=np.float32)
    probs = clf.predict_proba(x)[:, 1]
    scored = []
    for candidate, prob in zip(candidates, probs):
        item = dict(candidate)
        item["policy_confidence"] = round(float(prob), 6)
        item["confidence"] = round(float(prob), 6)
        scored.append(item)
    scored.sort(key=lambda item: (item["policy_confidence"], float(item.get("candidate_confidence", item["confidence"]))), reverse=True)
    if not peak_diversity:
        return scored[:budget]
    selected: list[dict[str, Any]] = []
    used_peaks: set[tuple[int, int]] = set()
    for item in scored:
        peak = tuple(int(v) for v in item["peak_xy"])
        if peak in used_peaks:
            continue
        selected.append(item)
        used_peaks.add(peak)
        if len(selected) >= budget:
            break
    if len(selected) < budget:
        selected_ids = {id(item) for item in selected}
        for item in scored:
            if id(item) not in selected_ids:
                selected.append(item)
            if len(selected) >= budget:
                break
    return selected[:budget]


def evaluate_rows(clf: ExtraTreesClassifier, rows: list[dict[str, Any]], args: argparse.Namespace, split: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    totals = Counter()
    pred_rows: list[dict[str, Any]] = []
    for row in rows:
        selected = select_candidates(clf, row["candidates"], args.max_candidates_per_page, args.peak_diversity)
        used: set[int] = set()
        matched_center = 0
        matched_iou = 0
        covered_gold: set[int] = set()
        covered_iou_gold: set[int] = set()
        for pred_index, candidate in enumerate(selected):
            labels = candidate["labels"]
            if labels["center_gold_index"] is not None:
                covered_gold.add(int(labels["center_gold_index"]))
            if labels["iou_0_30_positive"] and labels["best_gold_index"] is not None:
                covered_iou_gold.add(int(labels["best_gold_index"]))
        matched_center = len(covered_gold)
        matched_iou = len(covered_iou_gold)
        totals["gold"] += int(row["gold_text_count"])
        totals["predicted"] += len(selected)
        totals["matched_center"] += matched_center
        totals["matched_iou"] += matched_iou
        pred_rows.append(
            {
                "id": row["id"],
                "image": row["image"],
                "predicted_text": [
                    {
                        "id": f"{row['id']}_text_listwise_policy_v19_{idx}",
                        "class": "text",
                        "family": "text",
                        "semantic_type": "unknown_text",
                        "bbox": pred["bbox"],
                        "confidence": pred["confidence"],
                        "proposal_source": "raster_text_listwise_policy_v19",
                        "payload": {"ocr_status": "not_invoked", "source": "raster_text_listwise_policy_v19"},
                    }
                    for idx, pred in enumerate(selected)
                ],
                "gold_text_count": row["gold_text_count"],
                "matched_center": matched_center,
                "matched_iou_0_30": matched_iou,
                "source_integrity": {"model_input": "raster_image_only", "gold_used_for_inference": False},
            }
        )
    precision = totals["matched_iou"] / max(totals["predicted"], 1)
    recall = totals["matched_iou"] / max(totals["gold"], 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return (
        {
            "split": split,
            "rows": len(rows),
            "max_candidates_per_page": args.max_candidates_per_page,
            "peak_diversity": bool(args.peak_diversity),
            "text_bbox_iou_0_30": {
                "matched": int(totals["matched_iou"]),
                "predicted": int(totals["predicted"]),
                "gold": int(totals["gold"]),
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
            },
            "text_bbox_center_recall": round(totals["matched_center"] / max(totals["gold"], 1), 6),
            "candidate_inflation": round(totals["predicted"] / max(totals["gold"], 1), 6),
        },
        pred_rows,
    )


def oracle_report(rows: list[dict[str, Any]], budget: int, split: str) -> dict[str, Any]:
    totals = Counter()
    for row in rows:
        center_gold = []
        iou_gold = []
        for candidate in row["candidates"]:
            labels = candidate["labels"]
            if labels["center_gold_index"] is not None:
                center_gold.append(int(labels["center_gold_index"]))
            if labels["iou_0_30_positive"] and labels["best_gold_index"] is not None:
                iou_gold.append(int(labels["best_gold_index"]))
        totals["gold"] += int(row["gold_text_count"])
        totals["oracle_center"] += min(len(set(center_gold)), budget)
        totals["oracle_iou"] += min(len(set(iou_gold)), budget)
    return {
        "split": split,
        "budget": budget,
        "center_recall_ceiling_under_cache_and_budget": round(totals["oracle_center"] / max(totals["gold"], 1), 6),
        "iou_0_30_recall_ceiling_under_cache_and_budget": round(totals["oracle_iou"] / max(totals["gold"], 1), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-candidates-per-page", type=int, default=55)
    parser.add_argument("--negative-ratio", type=int, default=6)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument("--peak-diversity", action="store_true")
    args = parser.parse_args()

    train_rows = read_jsonl(CACHE / "train.jsonl")
    dev_rows = read_jsonl(CACHE / "dev.jsonl")
    locked_rows = read_jsonl(CACHE / "locked.jsonl")
    x_train, y_train, train_audit = build_training_matrix(train_rows, args.negative_ratio, args.seed)
    if len(set(y_train.tolist())) < 2:
        raise SystemExit("listwise policy needs positive and negative cached candidates")
    clf = ExtraTreesClassifier(
        n_estimators=args.n_estimators,
        random_state=args.seed,
        n_jobs=-1,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
    )
    clf.fit(x_train, y_train)
    dev_report, _dev_predictions = evaluate_rows(clf, dev_rows, args, "dev")
    locked_report, locked_predictions = evaluate_rows(clf, locked_rows, args, "locked")
    report = {
        "version": "text_listwise_policy_v19_eval",
        "task": "P0-TEXT-001",
        "run_mode": "cached_page_level_budgeted_text_candidate_policy",
        "source_integrity": {
            "model_input": "raster_image_only",
            "offline_labels_used_for": ["candidate_cache_labeling", "policy_training", "dev_evaluation", "locked_evaluation"],
            "gold_used_for_inference": False,
            "runtime_uses_svg_or_cad_geometry": False,
        },
        "training": {
            "checkpoint": str((OUT / "model.joblib").relative_to(ROOT)),
            "cache": str(CACHE.relative_to(ROOT)),
            "feature_names": FEATURE_NAMES,
            "train_audit": train_audit,
            "train_label_report": label_report(clf, x_train, y_train),
        },
        "oracle": {
            "dev": oracle_report(dev_rows, args.max_candidates_per_page, "dev"),
            "locked": oracle_report(locked_rows, args.max_candidates_per_page, "locked"),
        },
        "dev": dev_report,
        "locked": locked_report,
        "adopted": locked_report["text_bbox_center_recall"] >= 0.80 and locked_report["candidate_inflation"] <= 5.0,
        "blocker": None
        if locked_report["text_bbox_center_recall"] >= 0.80 and locked_report["candidate_inflation"] <= 5.0
        else "Cached candidate pool/listwise policy still fails the text localization gate; if oracle ceiling is below 0.80, upstream heatmap candidate generation must be retrained before more reranking.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "feature_names": FEATURE_NAMES, "args": vars(args)}, OUT / "model.joblib")
    write_json(REPORT / "text_listwise_policy_v19_eval.json", report)
    write_jsonl(REPORT / "text_listwise_policy_v19_locked_predictions.jsonl", locked_predictions)
    print(json.dumps({"locked": locked_report, "oracle_locked": report["oracle"]["locked"], "adopted": report["adopted"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
