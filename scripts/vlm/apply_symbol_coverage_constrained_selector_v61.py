#!/usr/bin/env python3
"""Coverage-constrained page selector for full symbol recovery rows."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from typing import Any

from apply_symbol_detector_recall_preserving_policy_v47 import evaluate_selection, group_pages, safe_float
from apply_symbol_focus_rescue_policy_v52 import candidate_area
from train_symbol_support_suppression_v35 import load_jsonl, source_path
from train_symbol_tile_detector_v20 import rel, write_json, write_jsonl


FOCUS_LABELS = {"sink", "equipment", "shower", "stair"}
FOCUS_AREAS = {"tiny_le_64", "small_le_256"}
SCORE_BINS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.90]


def score_bin(score: float) -> str:
    for edge in SCORE_BINS:
        if score <= edge:
            return f"le_{edge:g}"
    return "gt_0.9"


def is_positive(row: dict[str, Any]) -> bool:
    return safe_float((row.get("labels") or {}).get("best_iou")) >= 0.30


def rate(pos: int, total: int, prior: float, strength: float = 8.0) -> float:
    return (pos + prior * strength) / max(total + strength, 1.0)


def fit_calibrator(rows: list[dict[str, Any]]) -> dict[str, Any]:
    global_counts = Counter()
    buckets: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        pos = int(is_positive(row))
        global_counts["pos"] += pos
        global_counts["total"] += 1
        label = str(row.get("label") or "unknown")
        area = candidate_area(row)
        sbin = score_bin(safe_float(row.get("score")))
        keys = [
            f"label_area_score|{label}|{area}|{sbin}",
            f"label_area|{label}|{area}",
            f"label|{label}",
            f"area|{area}",
            f"score|{sbin}",
        ]
        for key in keys:
            buckets[key]["pos"] += pos
            buckets[key]["total"] += 1
    prior = global_counts["pos"] / max(global_counts["total"], 1)
    return {
        "prior": prior,
        "global": dict(global_counts),
        "buckets": {key: dict(value) for key, value in buckets.items()},
    }


def bucket_rate(calibrator: dict[str, Any], key: str) -> float | None:
    counts = (calibrator.get("buckets") or {}).get(key)
    if not counts:
        return None
    return rate(int(counts.get("pos") or 0), int(counts.get("total") or 0), float(calibrator["prior"]))


def calibrated_score(row: dict[str, Any], calibrator: dict[str, Any]) -> float:
    label = str(row.get("label") or "unknown")
    area = candidate_area(row)
    raw_score = safe_float(row.get("score"))
    sbin = score_bin(raw_score)
    keys = [
        f"label_area_score|{label}|{area}|{sbin}",
        f"label_area|{label}|{area}",
        f"label|{label}",
        f"area|{area}",
        f"score|{sbin}",
    ]
    rates = [value for key in keys if (value := bucket_rate(calibrator, key)) is not None]
    prior = float(calibrator["prior"])
    empirical = rates[0] if rates else prior
    backoff = sum(rates) / max(len(rates), 1) if rates else prior
    features = row.get("features") or {}
    cluster_size = safe_float(features.get("cluster_size"))
    cluster_score_max = safe_float(features.get("cluster_score_max"))
    score = 2.0 * empirical + 0.8 * backoff + 0.45 * raw_score + 0.20 * cluster_score_max - 0.015 * cluster_size
    if label in FOCUS_LABELS:
        score += 0.08
    if area in FOCUS_AREAS:
        score += 0.06
    return score


def add_unique(selected: list[dict[str, Any]], selected_ids: set[str], row: dict[str, Any], max_per_page: int) -> None:
    cid = str(row.get("candidate_id") or "")
    if cid and cid not in selected_ids and len(selected) < max_per_page:
        selected_ids.add(cid)
        selected.append(row)


def select_page(
    rows: list[dict[str, Any]],
    calibrator: dict[str, Any],
    score_threshold: float,
    cluster_topk: int,
    label_quota: int,
    area_quota: int,
    focus_label_quota: int,
    focus_area_quota: int,
    max_per_page: int,
) -> list[dict[str, Any]]:
    candidates = [row for row in rows if safe_float(row.get("score")) >= score_threshold]
    scored = sorted(candidates, key=lambda row: calibrated_score(row, calibrator), reverse=True)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def add_from_groups(groups: dict[str, list[dict[str, Any]]], quota: int) -> None:
        if quota <= 0:
            return
        for items in groups.values():
            items.sort(key=lambda row: calibrated_score(row, calibrator), reverse=True)
            for row in items[:quota]:
                add_unique(selected, selected_ids, row, max_per_page)

    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_area: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_focus_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_focus_area: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        label = str(row.get("label") or "")
        area = candidate_area(row)
        by_cluster[str(row.get("cluster_key") or row.get("cluster_id") or "")].append(row)
        by_label[label].append(row)
        by_area[area].append(row)
        if label in FOCUS_LABELS:
            by_focus_label[label].append(row)
        if area in FOCUS_AREAS:
            by_focus_area[area].append(row)

    add_from_groups(by_cluster, cluster_topk)
    add_from_groups(by_label, label_quota)
    add_from_groups(by_area, area_quota)
    add_from_groups(by_focus_label, focus_label_quota)
    add_from_groups(by_focus_area, focus_area_quota)
    for row in scored:
        add_unique(selected, selected_ids, row, max_per_page)
    selected.sort(key=lambda row: calibrated_score(row, calibrator), reverse=True)
    return selected[:max_per_page]


def evaluate_policy(
    pages: dict[str, list[dict[str, Any]]],
    calibrator: dict[str, Any],
    policy: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    selected_by_page = {
        page_id: select_page(
            page_rows,
            calibrator,
            float(policy["score_threshold"]),
            int(policy["cluster_topk"]),
            int(policy["label_quota"]),
            int(policy["area_quota"]),
            int(policy["focus_label_quota"]),
            int(policy["focus_area_quota"]),
            int(policy["max_per_page"]),
        )
        for page_id, page_rows in pages.items()
    }
    return evaluate_selection(pages, selected_by_page), selected_by_page


def prediction_rows(selected_by_page: dict[str, list[dict[str, Any]]], calibrator: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "page_id": page_id,
            "predicted_symbols": [
                {
                    "candidate_id": row.get("candidate_id"),
                    "bbox": row.get("bbox"),
                    "label": row.get("label"),
                    "confidence": round(calibrated_score(row, calibrator), 6),
                    "proposal_source": row.get("proposal_source"),
                }
                for row in selected
            ],
            "source_integrity": {"gold_used_for_inference": False, "runtime_uses_svg_or_cad_geometry": False},
        }
        for page_id, selected in selected_by_page.items()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="datasets/symbol_detector_listwise_recovery_v47_p2_transfer_locked_full_v2/manifest.json")
    parser.add_argument("--output-dir", default="checkpoints/symbol_coverage_constrained_selector_v61")
    parser.add_argument("--eval-output", default="reports/vlm/symbol_coverage_constrained_selector_v61_eval.json")
    parser.add_argument("--smoke-predictions-output", default="reports/vlm/symbol_coverage_constrained_selector_v61_smoke_predictions.jsonl")
    parser.add_argument("--candidate-inflation-target", type=float, default=8.0)
    parser.add_argument("--fast-grid", action="store_true", help="Run a narrow grid around the current recall-preserving baseline.")
    args = parser.parse_args()

    manifest = json.loads(source_path(args.data).read_text(encoding="utf-8"))
    rows = load_jsonl(source_path(manifest["outputs"]["rows"]))
    train_rows = [row for row in rows if str(row.get("split") or "") == "train"]
    dev_pages = group_pages(rows, "dev")
    smoke_pages = group_pages(rows, "smoke_eval")
    calibrator = fit_calibrator(train_rows)

    grid: list[dict[str, Any]] = []
    if args.fast_grid:
        score_thresholds = [0.02]
        cluster_topks = [1]
        label_quotas = [0, 1, 2]
        area_quotas = [0, 1]
        focus_label_quotas = [0, 2, 4, 8]
        focus_area_quotas = [0, 2, 4]
        max_per_pages = [180, 200]
    else:
        score_thresholds = [0.001, 0.005, 0.01, 0.02]
        cluster_topks = [1, 2]
        label_quotas = [0, 1, 2]
        area_quotas = [0, 1, 2]
        focus_label_quotas = [0, 2, 4, 8]
        focus_area_quotas = [0, 2, 4]
        max_per_pages = [160, 180, 200, 220]
    for score_threshold in score_thresholds:
        for cluster_topk in cluster_topks:
            for label_quota in label_quotas:
                for area_quota in area_quotas:
                    for focus_label_quota in focus_label_quotas:
                        for focus_area_quota in focus_area_quotas:
                            for max_per_page in max_per_pages:
                                policy = {
                                    "score_threshold": score_threshold,
                                    "cluster_topk": cluster_topk,
                                    "label_quota": label_quota,
                                    "area_quota": area_quota,
                                    "focus_label_quota": focus_label_quota,
                                    "focus_area_quota": focus_area_quota,
                                    "max_per_page": max_per_page,
                                }
                                metrics, _ = evaluate_policy(dev_pages, calibrator, policy)
                                grid.append({"policy": policy, "metrics": metrics})
    feasible = [row for row in grid if row["metrics"]["candidate_inflation"] <= args.candidate_inflation_target]
    selected = max(
        feasible or grid,
        key=lambda row: (
            row["metrics"]["symbol_bbox_iou_0_30"]["recall"] >= 0.70,
            row["metrics"]["symbol_bbox_iou_0_30"]["recall"],
            row["metrics"]["symbol_bbox_center_recall"],
            row["metrics"]["symbol_bbox_iou_0_30"]["precision"],
            -row["metrics"]["candidate_inflation"],
        ),
    )
    smoke_metrics, smoke_selected = evaluate_policy(smoke_pages, calibrator, selected["policy"])
    output_dir = source_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calibrator_path = output_dir / "calibrator.json"
    write_json(calibrator_path, calibrator)
    report = {
        "version": "symbol_coverage_constrained_selector_v61",
        "data": rel(source_path(args.data)),
        "source_integrity": {
            "gold_used_for_inference": False,
            "runtime_uses_svg_or_cad_geometry": False,
            "offline_labels_used_for": ["train_calibration", "dev_policy_selection", "smoke_evaluation"],
            "runtime_features": ["candidate score", "predicted label", "bbox area", "cluster/page features"],
        },
        "calibrator": rel(calibrator_path),
        "selected_policy": selected["policy"],
        "dev": selected["metrics"],
        "smoke_eval": smoke_metrics,
        "gate": {
            "smoke_recall_gte_0_70": smoke_metrics["symbol_bbox_iou_0_30"]["recall"] >= 0.70,
            "smoke_candidate_inflation_lt_8": smoke_metrics["candidate_inflation"] < 8.0,
            "no_oracle_inference": True,
        },
        "grid": [
            {
                **row["policy"],
                "precision": row["metrics"]["symbol_bbox_iou_0_30"]["precision"],
                "recall": row["metrics"]["symbol_bbox_iou_0_30"]["recall"],
                "center_recall": row["metrics"]["symbol_bbox_center_recall"],
                "candidate_inflation": row["metrics"]["candidate_inflation"],
            }
            for row in grid
        ],
    }
    report["gate"]["passed"] = all(report["gate"].values())
    write_json(source_path(args.eval_output), report)
    write_jsonl(source_path(args.smoke_predictions_output), prediction_rows(smoke_selected, calibrator))
    print(json.dumps({"selected_policy": report["selected_policy"], "dev": report["dev"], "smoke_eval": report["smoke_eval"], "gate": report["gate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
