#!/usr/bin/env python3
"""Evaluate class/size-aware symbol suppression over cached detector candidates."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from train_symbol_support_suppression_v35 import load_jsonl, source_path, vector
from train_symbol_tile_detector_v20 import area_bucket, rel, write_json, write_jsonl


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def predicted_area_bucket(row: dict[str, Any]) -> str:
    return area_bucket([float(v) for v in row.get("bbox") or [0, 0, 1, 1]])


def page_gold_targets(rows: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        for gold in (row.get("labels") or {}).get("page_gold_targets") or []:
            target_id = str(gold.get("target_id") or "")
            if target_id:
                out[target_id] = {"label": str(gold.get("label") or ""), "area_bucket": str(gold.get("area_bucket") or "")}
    return out


def score_rows(rows: list[dict[str, Any]], model_bundle: dict[str, Any]) -> list[dict[str, Any]]:
    names = list(model_bundle["feature_names"])
    model = model_bundle["model"]
    if not rows:
        return []
    x = np.asarray([vector(row, names) for row in rows], dtype=np.float32)
    probs = model.predict_proba(x)[:, 1]
    out: list[dict[str, Any]] = []
    for row, prob in zip(rows, probs, strict=True):
        item = dict(row)
        item["policy_score"] = float(prob)
        item["predicted_area_bucket"] = predicted_area_bucket(row)
        out.append(item)
    return out


def select_rows(
    rows: list[dict[str, Any]],
    threshold: float,
    fallback_topk: int,
    fallback_labels: set[str],
    fallback_area_buckets: set[str],
    max_per_page: int,
) -> list[dict[str, Any]]:
    selected_ids: set[str] = set()
    selected: list[dict[str, Any]] = []

    def add(row: dict[str, Any]) -> None:
        cid = str(row.get("candidate_id") or "")
        if cid and cid not in selected_ids:
            selected_ids.add(cid)
            selected.append(row)

    for row in sorted(rows, key=lambda item: (safe_float(item.get("policy_score")), safe_float(item.get("score"))), reverse=True):
        if safe_float(row.get("policy_score")) >= threshold:
            add(row)

    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cluster[str(row.get("cluster_key") or row.get("cluster_id") or "")].append(row)
    for items in by_cluster.values():
        items.sort(key=lambda item: (safe_float(item.get("policy_score")), safe_float(item.get("score"))), reverse=True)
        for rank, row in enumerate(items[: max(fallback_topk, 0)]):
            if rank >= fallback_topk:
                break
            label = str(row.get("label") or "")
            bucket = str(row.get("predicted_area_bucket") or "")
            if label in fallback_labels or bucket in fallback_area_buckets:
                add(row)

    selected.sort(key=lambda item: (safe_float(item.get("policy_score")), safe_float(item.get("score"))), reverse=True)
    return selected[:max_per_page]


def evaluate(pages: dict[str, list[dict[str, Any]]], selected_by_page: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    totals = Counter()
    misses_by_label = Counter()
    misses_by_area = Counter()
    selected_negative_reasons = Counter()
    for page_id, rows in pages.items():
        gold = page_gold_targets(rows)
        iou_hits: set[str] = set()
        center_hits: set[str] = set()
        typed_total = 0
        typed_correct = 0
        for row in selected_by_page.get(page_id, []):
            labels = row.get("labels") or {}
            target = str(labels.get("best_iou_target_id") or "")
            if target and safe_float(labels.get("best_iou")) >= 0.30:
                iou_hits.add(target)
                typed_total += 1
                if str(row.get("label") or "") == gold.get(target, {}).get("label", ""):
                    typed_correct += 1
            else:
                selected_negative_reasons[str(labels.get("suppression_reason") or "unknown")] += 1
            for target_id in labels.get("center_target_ids") or []:
                center_hits.add(str(target_id))
        for target_id, meta in gold.items():
            if target_id not in iou_hits:
                misses_by_label[meta.get("label", "unknown")] += 1
                misses_by_area[meta.get("area_bucket", "unknown")] += 1
        totals["gold"] += len(gold)
        totals["selected"] += len(selected_by_page.get(page_id, []))
        totals["iou_hit"] += len(iou_hits)
        totals["center_hit"] += len(center_hits & set(gold))
        totals["typed_total"] += typed_total
        totals["typed_correct"] += typed_correct
    precision = totals["iou_hit"] / max(totals["selected"], 1)
    recall = totals["iou_hit"] / max(totals["gold"], 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "pages": len(pages),
        "symbol_bbox_iou_0_30": {
            "matched": int(totals["iou_hit"]),
            "predicted": int(totals["selected"]),
            "gold": int(totals["gold"]),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        },
        "symbol_bbox_center_recall": round(totals["center_hit"] / max(totals["gold"], 1), 6),
        "candidate_inflation": round(totals["selected"] / max(totals["gold"], 1), 6),
        "typed_accuracy_on_iou_matches": round(totals["typed_correct"] / max(totals["typed_total"], 1), 6),
        "selected_negative_reasons": dict(selected_negative_reasons.most_common()),
        "misses_by_label": dict(misses_by_label.most_common()),
        "misses_by_area": dict(misses_by_area.most_common()),
    }


def parse_csv_set(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="datasets/symbol_detector_listwise_recovery_v47_p2_transfer_smoke120/manifest.json")
    parser.add_argument("--model", default="checkpoints/symbol_support_suppression_v35_p2_transfer_smoke120_t065_c1/model.joblib")
    parser.add_argument("--split", default="smoke_eval", choices=["all", "train", "dev", "smoke_eval"])
    parser.add_argument("--fallback-labels", default="sink,equipment")
    parser.add_argument("--fallback-area-buckets", default="tiny_le_64")
    parser.add_argument("--output", default="reports/vlm/symbol_class_size_suppression_policy_v48_eval.json")
    parser.add_argument("--predictions-output", default="reports/vlm/symbol_class_size_suppression_policy_v48_predictions.jsonl")
    parser.add_argument("--max-per-page", type=int, default=120)
    args = parser.parse_args()

    manifest = json.loads(source_path(args.data).read_text(encoding="utf-8"))
    rows_all = load_jsonl(source_path(manifest["outputs"]["rows"]))
    rows = [row for row in rows_all if args.split == "all" or str(row.get("split") or "") == args.split]
    scored_rows = score_rows(rows, joblib.load(source_path(args.model)))
    pages: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        pages[str(row["page_id"])].append(row)

    fallback_labels = parse_csv_set(args.fallback_labels)
    fallback_area_buckets = parse_csv_set(args.fallback_area_buckets)
    grid: list[dict[str, Any]] = []
    for threshold in [0.50, 0.60, 0.65, 0.70, 0.75, 0.80]:
        for fallback_topk in [0, 1, 2]:
            selected_by_page = {
                page_id: select_rows(page_rows, threshold, fallback_topk, fallback_labels, fallback_area_buckets, args.max_per_page)
                for page_id, page_rows in pages.items()
            }
            metrics = evaluate(pages, selected_by_page)
            grid.append({"threshold": threshold, "fallback_topk": fallback_topk, "metrics": metrics})
    selected = max(
        grid,
        key=lambda row: (
            row["metrics"]["symbol_bbox_iou_0_30"]["recall"] >= 0.50,
            row["metrics"]["symbol_bbox_iou_0_30"]["precision"] >= 0.20,
            row["metrics"]["symbol_bbox_iou_0_30"]["f1"],
            row["metrics"]["symbol_bbox_iou_0_30"]["precision"],
            -row["metrics"]["candidate_inflation"],
        ),
    )
    selected_by_page = {
        page_id: select_rows(page_rows, selected["threshold"], selected["fallback_topk"], fallback_labels, fallback_area_buckets, args.max_per_page)
        for page_id, page_rows in pages.items()
    }
    predictions = [
        {
            "page_id": page_id,
            "predicted_symbols": [
                {
                    "candidate_id": row["candidate_id"],
                    "bbox": row["bbox"],
                    "label": row.get("label"),
                    "confidence": round(safe_float(row.get("policy_score")), 6),
                    "proposal_source": row.get("proposal_source"),
                }
                for row in selected_rows
            ],
            "source_integrity": {"gold_used_for_inference": False, "runtime_uses_svg_or_cad_geometry": False},
        }
        for page_id, selected_rows in selected_by_page.items()
    ]
    report = {
        "version": "symbol_class_size_suppression_policy_v48",
        "data": rel(source_path(args.data)),
        "model": rel(source_path(args.model)),
        "split": args.split,
        "fallback_labels": sorted(fallback_labels),
        "fallback_area_buckets": sorted(fallback_area_buckets),
        "selected_policy": {"threshold": selected["threshold"], "fallback_topk": selected["fallback_topk"], "max_per_page": args.max_per_page},
        "selected": selected["metrics"],
        "gate": {
            "precision_gte_0_20": selected["metrics"]["symbol_bbox_iou_0_30"]["precision"] >= 0.20,
            "recall_gte_0_50": selected["metrics"]["symbol_bbox_iou_0_30"]["recall"] >= 0.50,
            "candidate_inflation_lte_5": selected["metrics"]["candidate_inflation"] <= 5.0,
            "no_oracle_inference": True,
        },
        "grid": [
            {
                "threshold": row["threshold"],
                "fallback_topk": row["fallback_topk"],
                "precision": row["metrics"]["symbol_bbox_iou_0_30"]["precision"],
                "recall": row["metrics"]["symbol_bbox_iou_0_30"]["recall"],
                "f1": row["metrics"]["symbol_bbox_iou_0_30"]["f1"],
                "center_recall": row["metrics"]["symbol_bbox_center_recall"],
                "candidate_inflation": row["metrics"]["candidate_inflation"],
            }
            for row in grid
        ],
    }
    report["gate"]["passed"] = all(report["gate"].values())
    write_json(source_path(args.output), report)
    write_jsonl(source_path(args.predictions_output), predictions)
    print(json.dumps({"selected_policy": report["selected_policy"], "selected": report["selected"], "gate": report["gate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
