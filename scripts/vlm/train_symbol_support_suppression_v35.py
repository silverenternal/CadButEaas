#!/usr/bin/env python3
"""Train/evaluate v35 source-aware symbol suppression set-policy."""

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

from train_symbol_tile_detector_v20 import rel, write_json, write_jsonl


warnings.filterwarnings(
    "ignore",
    message="`sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel`.*",
    category=UserWarning,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "datasets/symbol_support_suppression_v35/manifest.json"
DEFAULT_OUTPUT_DIR = ROOT / "checkpoints/symbol_support_suppression_v35"
DEFAULT_EVAL_OUTPUT = ROOT / "reports/vlm/symbol_support_suppression_v35_smoke_eval.json"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def source_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def feature_names(rows: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for row in rows:
        names.update((row.get("features") or {}).keys())
    return sorted(names)


def vector(row: dict[str, Any], names: list[str]) -> list[float]:
    feats = row.get("features") or {}
    return [float(feats.get(name, 0.0) or 0.0) for name in names]


def y(row: dict[str, Any]) -> int:
    return int(bool((row.get("labels") or {}).get("keep")))


def split_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row.get("split") or "train")].append(row)
    return out


def label_report(model: Any, rows: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    if not rows:
        return {"examples": 0}
    yy = np.asarray([y(row) for row in rows], dtype=np.int64)
    xx = np.asarray([vector(row, names) for row in rows], dtype=np.float32)
    prob = model.predict_proba(xx)[:, 1]
    out = {"examples": int(len(rows)), "positive": int(yy.sum()), "positive_rate": round(float(yy.mean()), 6)}
    if len(set(yy.tolist())) >= 2:
        out["roc_auc"] = round(float(roc_auc_score(yy, prob)), 6)
        out["average_precision"] = round(float(average_precision_score(yy, prob)), 6)
    return out


def group_pages(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    pages: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pages[str(row["page_id"])].append(row)
    return pages


def choose_candidates(
    model: Any,
    rows: list[dict[str, Any]],
    names: list[str],
    threshold: float,
    cluster_topk: int,
    max_per_page: int,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    xx = np.asarray([vector(row, names) for row in rows], dtype=np.float32)
    probs = model.predict_proba(xx)[:, 1]
    scored: list[dict[str, Any]] = []
    for row, prob in zip(rows, probs, strict=True):
        item = dict(row)
        item["policy_score"] = float(prob)
        scored.append(item)
    by_cluster: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in scored:
        by_cluster[int(item.get("cluster_id") or 0)].append(item)
    selected: list[dict[str, Any]] = []
    for items in by_cluster.values():
        items.sort(key=lambda value: (float(value["policy_score"]), float(value.get("score", 0.0))), reverse=True)
        for rank, item in enumerate(items):
            if item["policy_score"] >= threshold or rank < cluster_topk:
                selected.append(item)
    selected.sort(key=lambda value: (float(value["policy_score"]), float(value.get("score", 0.0))), reverse=True)
    return selected[:max_per_page]


def evaluate(
    model: Any,
    rows: list[dict[str, Any]],
    names: list[str],
    split: str,
    threshold: float,
    cluster_topk: int,
    max_per_page: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pages = group_pages(rows)
    totals = Counter()
    by_source = Counter()
    by_reason = Counter()
    by_label_miss = Counter()
    by_area_miss = Counter()
    predictions: list[dict[str, Any]] = []
    for page_id, page_rows in pages.items():
        selected = choose_candidates(model, page_rows, names, threshold, cluster_topk, max_per_page)
        gold_targets: set[str] = set()
        center_hits: set[str] = set()
        iou_hits: set[str] = set()
        typed_correct = 0
        typed_total = 0
        target_label: dict[str, str] = {}
        target_area: dict[str, str] = {}
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
                        "confidence": round(float(row["policy_score"]), 6),
                        "proposal_source": row["proposal_source"],
                        "suppression_reason": "kept_by_v35_source_policy",
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
            "pages": len(pages),
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


def metric_view(report: dict[str, Any]) -> dict[str, float]:
    return {
        "center_recall": float(report["symbol_bbox_center_recall"]),
        "iou_0_30_recall": float(report["symbol_bbox_iou_0_30"]["recall"]),
        "precision": float(report["symbol_bbox_iou_0_30"]["precision"]),
        "f1": float(report["symbol_bbox_iou_0_30"]["f1"]),
        "candidate_inflation": float(report["candidate_inflation"]),
    }


def choose_policy(model: Any, rows: list[dict[str, Any]], names: list[str], max_per_page: int) -> dict[str, Any]:
    if not rows:
        return {"threshold": 0.10, "cluster_topk": 1, "max_per_page": max_per_page, "dev_metrics": None}
    grid: list[dict[str, Any]] = []
    for threshold in [0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50]:
        for cluster_topk in [0, 1, 2]:
            for page_cap in [120, 220, 400, max_per_page]:
                report, _ = evaluate(model, rows, names, "dev", threshold, cluster_topk, page_cap)
                view = metric_view(report)
                grid.append({"threshold": threshold, "cluster_topk": cluster_topk, "max_per_page": page_cap, "metrics": report, "view": view})
    selected = sorted(
        grid,
        key=lambda item: (
            item["view"]["center_recall"] >= 0.90,
            item["view"]["iou_0_30_recall"] >= 0.72,
            item["view"]["precision"] >= 0.12,
            item["view"]["candidate_inflation"] <= 7.0,
            item["view"]["iou_0_30_recall"],
            item["view"]["precision"],
            -item["view"]["candidate_inflation"],
        ),
        reverse=True,
    )[0]
    return {
        "threshold": selected["threshold"],
        "cluster_topk": selected["cluster_topk"],
        "max_per_page": selected["max_per_page"],
        "dev_metrics": selected["metrics"],
        "grid": [
            {
                "threshold": item["threshold"],
                "cluster_topk": item["cluster_topk"],
                "max_per_page": item["max_per_page"],
                **item["view"],
            }
            for item in grid
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--eval-output", type=Path, default=DEFAULT_EVAL_OUTPUT)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--cluster-topk", type=int, default=None)
    parser.add_argument("--max-per-page", type=int, default=220)
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=20260512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = source_path(args.data)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = load_jsonl(source_path(manifest["outputs"]["rows"]))
    names = [name for name in feature_names(rows) if not name.endswith("_audit_train_only")]
    by_split = split_rows(rows)
    train_rows = by_split.get("train", [])
    if not train_rows or len({y(row) for row in train_rows}) < 2:
        raise SystemExit("v35 support suppression needs train positives and negatives")
    train_x = np.asarray([vector(row, names) for row in train_rows], dtype=np.float32)
    train_y = np.asarray([y(row) for row in train_rows], dtype=np.int64)
    model: Any = ExtraTreesClassifier(
        n_estimators=args.n_estimators,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=args.n_jobs,
        random_state=args.seed,
    )
    model.fit(train_x, train_y)

    if args.threshold is not None and args.cluster_topk is not None:
        threshold = float(args.threshold)
        cluster_topk = int(args.cluster_topk)
        max_per_page = int(args.max_per_page)
        policy = {"threshold": threshold, "cluster_topk": cluster_topk, "max_per_page": max_per_page, "dev_metrics": None, "grid": []}
    else:
        policy = choose_policy(model, by_split.get("dev", []), names, args.max_per_page)
        threshold = float(args.threshold) if args.threshold is not None else float(policy["threshold"])
        cluster_topk = int(args.cluster_topk) if args.cluster_topk is not None else int(policy["cluster_topk"])
        max_per_page = int(policy["max_per_page"])
    dev_report, _ = evaluate(model, by_split.get("dev", []), names, "dev", threshold, cluster_topk, max_per_page)
    smoke_report, smoke_predictions = evaluate(model, by_split.get("smoke_eval", []), names, "smoke_eval", threshold, cluster_topk, max_per_page)
    all_report, _ = evaluate(model, rows, names, "all_cache_audit_not_final", threshold, cluster_topk, max_per_page)
    output_dir = source_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.joblib"
    joblib.dump({"model": model, "feature_names": names, "args": vars(args)}, model_path)
    report = {
        "version": "symbol_support_suppression_v35_smoke_eval",
        "task": "P1-04-v35-aware-source-calibrated-set-policy",
        "claim_boundary": "Smoke split is for direction only. all_cache_audit_not_final includes training pages and must not be claimed as final model quality.",
        "source_integrity": {
            "model_input": "raster-derived candidate bbox/score/source/type fields only",
            "offline_labels_used_for": ["policy_training", "dev_evaluation", "smoke_evaluation", "audit"],
            "gold_used_for_inference": False,
            "runtime_uses_svg_or_cad_geometry": False,
            "excluded_features": ["center_match_count_audit_train_only"],
        },
        "training": {
            "checkpoint": rel(model_path),
            "data": rel(manifest_path),
            "feature_count": len(names),
            "train_label_report": label_report(model, train_rows, names),
            "dev_label_report": label_report(model, by_split.get("dev", []), names),
            "smoke_label_report": label_report(model, by_split.get("smoke_eval", []), names),
        },
        "policy": {"threshold": threshold, "cluster_topk": cluster_topk, "max_per_page": max_per_page, "selected_from_dev": args.threshold is None or args.cluster_topk is None},
        "policy_search": {
            "dev_selected_metrics": policy.get("dev_metrics"),
            "grid": policy.get("grid", []),
        },
        "dev": dev_report,
        "smoke": smoke_report,
        "all_cache_audit_not_final": all_report,
        "stage_gate": {
            "smoke_center_recall_min_0_94": smoke_report["symbol_bbox_center_recall"] >= 0.94,
            "smoke_iou_0_30_recall_min_0_72": smoke_report["symbol_bbox_iou_0_30"]["recall"] >= 0.72,
            "smoke_precision_min_0_12": smoke_report["symbol_bbox_iou_0_30"]["precision"] >= 0.12,
            "smoke_candidate_inflation_max_7": smoke_report["candidate_inflation"] <= 7.0,
            "no_oracle_inference": True,
        },
    }
    report["stage_gate"]["passed"] = all(report["stage_gate"].values())
    write_json(source_path(args.eval_output), report)
    write_jsonl(ROOT / "reports/vlm/symbol_support_suppression_v35_smoke_predictions.jsonl", smoke_predictions)
    print(json.dumps({"smoke": smoke_report, "all_cache_audit_not_final": all_report, "stage_gate": report["stage_gate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
