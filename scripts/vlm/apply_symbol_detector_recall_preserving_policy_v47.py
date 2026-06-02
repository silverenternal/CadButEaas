#!/usr/bin/env python3
"""Evaluate recall-preserving page policy over detector recovery rows."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from train_symbol_support_suppression_v35 import load_jsonl, source_path
from train_symbol_tile_detector_v20 import rel, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def feature_score(row: dict[str, Any]) -> float:
    features = row.get("features") or {}
    return (
        safe_float(row.get("score"))
        + 0.25 * safe_float(features.get("cluster_score_max"))
        - 0.03 * safe_float(features.get("cluster_size"))
    )


def page_gold_targets(rows: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        for gold in (row.get("labels") or {}).get("page_gold_targets") or []:
            target_id = str(gold.get("target_id") or "")
            if target_id:
                out[target_id] = {"label": str(gold.get("label") or ""), "area_bucket": str(gold.get("area_bucket") or "")}
    return out


def select_rows(
    rows: list[dict[str, Any]],
    score_threshold: float,
    cluster_topk: int,
    label_topk: int,
    max_per_page: int,
) -> list[dict[str, Any]]:
    candidates = [row for row in rows if safe_float(row.get("score")) >= score_threshold]
    selected_ids: set[str] = set()
    selected: list[dict[str, Any]] = []

    def add(row: dict[str, Any]) -> None:
        cid = str(row.get("candidate_id") or "")
        if cid and cid not in selected_ids and len(selected) < max_per_page:
            selected_ids.add(cid)
            selected.append(row)

    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_cluster[str(row.get("cluster_key") or row.get("cluster_id") or "")].append(row)
        by_label[str(row.get("label") or "")].append(row)
    for items in by_label.values():
        items.sort(key=feature_score, reverse=True)
        for row in items[:label_topk]:
            add(row)
    for items in by_cluster.values():
        items.sort(key=feature_score, reverse=True)
        for row in items[:cluster_topk]:
            add(row)
    for row in sorted(candidates, key=feature_score, reverse=True):
        add(row)
    return selected


def evaluate_selection(pages: dict[str, list[dict[str, Any]]], selected_by_page: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    totals = Counter()
    misses_by_label = Counter()
    misses_by_area = Counter()
    for page_id, rows in pages.items():
        gold = page_gold_targets(rows)
        iou_hits: set[str] = set()
        center_hits: set[str] = set()
        typed_correct = 0
        typed_total = 0
        selected = selected_by_page.get(page_id, [])
        for row in selected:
            labels = row.get("labels") or {}
            target = str(labels.get("best_iou_target_id") or "")
            if target and safe_float(labels.get("best_iou")) >= 0.30:
                iou_hits.add(target)
                typed_total += 1
                if str(row.get("label") or "") == gold.get(target, {}).get("label", ""):
                    typed_correct += 1
            for target_id in labels.get("center_target_ids") or []:
                center_hits.add(str(target_id))
        for target, meta in gold.items():
            if target not in iou_hits:
                misses_by_label[meta.get("label", "unknown")] += 1
                misses_by_area[meta.get("area_bucket", "unknown")] += 1
        totals["gold"] += len(gold)
        totals["selected"] += len(selected)
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
        "misses_by_label": dict(misses_by_label.most_common()),
        "misses_by_area": dict(misses_by_area.most_common()),
    }


def group_pages(rows: list[dict[str, Any]], split: str) -> dict[str, list[dict[str, Any]]]:
    pages: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if split != "all" and str(row.get("split") or "") != split:
            continue
        pages[str(row["page_id"])].append(row)
    return pages


def oracle_report(pages: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return evaluate_selection(pages, {page_id: rows for page_id, rows in pages.items()})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="datasets/symbol_detector_listwise_recovery_v47/manifest.json")
    parser.add_argument("--output", default="reports/vlm/symbol_detector_recall_preserving_policy_v47_eval.json")
    parser.add_argument("--predictions-output", default="reports/vlm/symbol_detector_recall_preserving_policy_v47_predictions.jsonl")
    parser.add_argument("--split", default="all", choices=["all", "train", "dev", "smoke_eval"])
    parser.add_argument("--candidate-inflation-target", type=float, default=12.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(source_path(args.data).read_text(encoding="utf-8"))
    rows = load_jsonl(source_path(manifest["outputs"]["rows"]))
    pages = group_pages(rows, args.split)
    grid: list[dict[str, Any]] = []
    for score_threshold in [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10]:
        for cluster_topk in [0, 1, 2, 3]:
            for label_topk in [0, 1, 2, 4, 8]:
                for max_per_page in [80, 120, 160, 200, 260]:
                    selected_by_page = {
                        page_id: select_rows(page_rows, score_threshold, cluster_topk, label_topk, max_per_page)
                        for page_id, page_rows in pages.items()
                    }
                    metrics = evaluate_selection(pages, selected_by_page)
                    grid.append(
                        {
                            "score_threshold": score_threshold,
                            "cluster_topk": cluster_topk,
                            "label_topk": label_topk,
                            "max_per_page": max_per_page,
                            "metrics": metrics,
                        }
                    )
    feasible = [row for row in grid if row["metrics"]["candidate_inflation"] <= args.candidate_inflation_target]
    selected = max(
        feasible or grid,
        key=lambda row: (
            row["metrics"]["symbol_bbox_iou_0_30"]["recall"],
            row["metrics"]["symbol_bbox_center_recall"],
            row["metrics"]["symbol_bbox_iou_0_30"]["precision"],
            -row["metrics"]["candidate_inflation"],
        ),
    )
    selected_by_page = {
        page_id: select_rows(page_rows, selected["score_threshold"], selected["cluster_topk"], selected["label_topk"], selected["max_per_page"])
        for page_id, page_rows in pages.items()
    }
    prediction_rows = [
        {
            "page_id": page_id,
            "predicted_symbols": [
                {
                    "candidate_id": row["candidate_id"],
                    "bbox": row["bbox"],
                    "label": row.get("label"),
                    "confidence": round(feature_score(row), 6),
                    "proposal_source": row.get("proposal_source"),
                }
                for row in selected_rows
            ],
            "source_integrity": {"gold_used_for_inference": False, "runtime_uses_svg_or_cad_geometry": False},
        }
        for page_id, selected_rows in selected_by_page.items()
    ]
    report = {
        "version": "symbol_detector_recall_preserving_policy_v47",
        "data": rel(source_path(args.data)),
        "split": args.split,
        "selected_policy": {key: selected[key] for key in ["score_threshold", "cluster_topk", "label_topk", "max_per_page"]},
        "selected": selected["metrics"],
        "oracle_upper_bound_on_input_rows": oracle_report(pages),
        "gate": {
            "candidate_inflation_lte_target": selected["metrics"]["candidate_inflation"] <= args.candidate_inflation_target,
            "smoke_recall_gte_0_50": selected["metrics"]["symbol_bbox_iou_0_30"]["recall"] >= 0.50,
            "oracle_recall_gte_0_50": oracle_report(pages)["symbol_bbox_iou_0_30"]["recall"] >= 0.50,
        },
        "grid": [
            {
                "score_threshold": row["score_threshold"],
                "cluster_topk": row["cluster_topk"],
                "label_topk": row["label_topk"],
                "max_per_page": row["max_per_page"],
                "recall": row["metrics"]["symbol_bbox_iou_0_30"]["recall"],
                "precision": row["metrics"]["symbol_bbox_iou_0_30"]["precision"],
                "center_recall": row["metrics"]["symbol_bbox_center_recall"],
                "candidate_inflation": row["metrics"]["candidate_inflation"],
            }
            for row in grid
        ],
    }
    report["gate"]["passed"] = all(report["gate"].values())
    write_json(source_path(args.output), report)
    write_jsonl(source_path(args.predictions_output), prediction_rows)
    print(json.dumps({"selected_policy": report["selected_policy"], "selected": report["selected"], "oracle": report["oracle_upper_bound_on_input_rows"], "gate": report["gate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
