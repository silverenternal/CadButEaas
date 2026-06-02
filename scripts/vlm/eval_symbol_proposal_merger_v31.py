#!/usr/bin/env python3
"""Evaluate v31 coverage-aware symbol proposal selection."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from build_symbol_coverage_selector_features_v31 import build_clusters, peer_features, valid_box
from build_symbol_proposal_selector_features_v30 import box_features, load_preds
from eval_symbol_proposal_merger_v30 import build_golds_from_center_targets, rel_from_manifest, source_counts
from train_symbol_center_heatmap_probe_v24 import score_predictions
from train_symbol_tile_detector_v20 import area_bucket, bbox_iou, center_covered, rel, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]


def merge_sources(mask: dict[str, list[dict[str, Any]]], center: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row_id in set(mask) | set(center):
        rows = []
        for pred in mask.get(row_id, []):
            item = dict(pred)
            item["proposal_source"] = "mask_v28"
            rows.append(item)
        for pred in center.get(row_id, []):
            item = dict(pred)
            item["proposal_source"] = "center_branch_v30"
            rows.append(item)
        out[row_id] = [pred for pred in rows if valid_box(pred)]
    return out


def candidate_features(index: int, preds: list[dict[str, Any]], cluster_ids: list[int]) -> dict[str, float]:
    pred = preds[index]
    box = valid_box(pred) or [0.0, 0.0, 0.0, 0.0]
    source = str(pred.get("proposal_source") or "unknown")
    features = {
        "score": float(pred.get("score", 0.0)),
        "is_mask_v28": float(source == "mask_v28"),
        "is_center_branch_v30": float(source == "center_branch_v30"),
        "label_id": float(pred.get("label_id") or 5),
        "candidate_count_page": float(len(preds)),
        "best_iou_rank_for_gold": 0.0,
        "score_rank_for_gold": 0.0,
        "same_gold_positive_count": 0.0,
        "same_gold_coverage_count": 0.0,
    }
    features.update(box_features(box))
    features.update(peer_features(index, preds, cluster_ids))
    return features


def score_with_selector(page_preds: dict[str, list[dict[str, Any]]], selector_path: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    bundle = joblib.load(selector_path)
    model = bundle["model"]
    feature_names = list(bundle["feature_names"])
    out: dict[str, list[dict[str, Any]]] = {}
    for row_id, preds in page_preds.items():
        cluster_ids = build_clusters(preds, 12.0, 0.30)
        if not preds:
            out[row_id] = []
            continue
        x = np.asarray([[candidate_features(index, preds, cluster_ids).get(name, 0.0) for name in feature_names] for index in range(len(preds))], dtype=np.float32)
        probs = model.predict_proba(x)[:, 1]
        rescored = []
        for index, (pred, prob) in enumerate(zip(preds, probs.tolist(), strict=True)):
            item = dict(pred)
            item["pre_selector_score"] = float(item.get("score", 0.0))
            item["selector_score"] = float(prob)
            item["score"] = float(prob)
            item["cluster_id"] = int(cluster_ids[index])
            rescored.append(item)
        out[row_id] = rescored
    return out, {"selector": rel(selector_path), "selected_threshold": float(bundle.get("selected_threshold", 0.08)), "feature_names": feature_names}


def select_policy(
    preds: list[dict[str, Any]],
    threshold: float,
    topk: int,
    min_cluster_score: float,
    pre_nms_budget: int,
    pre_nms_budget_ratio: float,
    max_per_page: int,
) -> list[dict[str, Any]]:
    by_cluster: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for pred in preds:
        by_cluster[int(pred.get("cluster_id") or 0)].append(pred)
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for cluster_preds in by_cluster.values():
        ordered = sorted(cluster_preds, key=lambda row: float(row.get("selector_score", row.get("score", 0.0))), reverse=True)
        cluster_max = float(ordered[0].get("selector_score", ordered[0].get("score", 0.0))) if ordered else 0.0
        for rank, pred in enumerate(ordered):
            score = float(pred.get("selector_score", pred.get("score", 0.0)))
            keep = score >= threshold or (rank < topk and cluster_max >= min_cluster_score)
            if keep and id(pred) not in seen:
                selected.append(pred)
                seen.add(id(pred))
    selected.sort(key=lambda row: float(row.get("selector_score", row.get("score", 0.0))), reverse=True)
    if pre_nms_budget_ratio > 0.0:
        ratio_budget = max(1, int(round(len(preds) * pre_nms_budget_ratio)))
        pre_nms_budget = min(pre_nms_budget, ratio_budget) if pre_nms_budget > 0 else ratio_budget
    if pre_nms_budget > 0:
        selected = selected[:pre_nms_budget]
    return selected[:max_per_page]


def coverage_loss(
    predictions: list[dict[str, Any]],
    golds: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    pred_map = {str(row["row_id"]): list(row.get("predicted_symbols") or []) for row in predictions}
    counts = Counter()
    by_label = Counter()
    by_area = Counter()
    by_source_possible = Counter()
    examples = []
    for row_id, gold_map in golds.items():
        preds = pred_map.get(row_id, [])
        for gold in gold_map.values():
            gold_box = [float(v) for v in gold["bbox"]]
            label = str(gold.get("label") or "generic_symbol")
            bucket = area_bucket(gold_box)
            best_iou = 0.0
            center_hit = False
            best_source = None
            for pred in preds:
                box = [float(v) for v in pred.get("bbox") or []]
                if len(box) != 4:
                    continue
                iou = bbox_iou(box, gold_box)
                if iou > best_iou:
                    best_iou = iou
                    best_source = pred.get("proposal_source")
                center_hit = center_hit or center_covered(box, gold_box)
            if not center_hit:
                counts["selected_no_center_coverage"] += 1
                by_label[label] += 1
                by_area[bucket] += 1
                by_source_possible[str(best_source or "none")] += 1
                if len(examples) < 300:
                    examples.append({"row_id": row_id, "target_id": gold["target_id"], "label": label, "area_bucket": bucket, "best_iou": round(best_iou, 6), "best_source": best_source, "error": "selected_no_center_coverage"})
            if best_iou < 0.30:
                counts["selected_no_iou_0_30_coverage"] += 1
    return {
        "counts": dict(counts),
        "by_label": dict(by_label),
        "by_area": dict(by_area),
        "by_best_remaining_source": dict(by_source_possible),
        "examples": examples,
    }


def metric_summary(row: dict[str, Any]) -> dict[str, float]:
    metrics = row["metrics"]
    return {
        "center_recall": float(metrics["symbol_bbox_center_recall"]),
        "iou_0_30_recall": float(metrics["symbol_bbox_iou_0_30"]["recall"]),
        "precision": float(metrics["symbol_bbox_iou_0_30"]["precision"]),
        "f1": float(metrics["symbol_bbox_iou_0_30"]["f1"]),
        "candidate_inflation": float(metrics["candidate_inflation"]),
    }


def choose_views(grid: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    recall_first = sorted(grid, key=lambda row: (metric_summary(row)["center_recall"], metric_summary(row)["iou_0_30_recall"], metric_summary(row)["precision"], -metric_summary(row)["candidate_inflation"]), reverse=True)[0]
    balanced = sorted(grid, key=lambda row: (metric_summary(row)["center_recall"] >= 0.90, metric_summary(row)["iou_0_30_recall"] >= 0.72, metric_summary(row)["candidate_inflation"] <= 7.0, metric_summary(row)["precision"] >= 0.12, metric_summary(row)["f1"], metric_summary(row)["center_recall"], -metric_summary(row)["candidate_inflation"]), reverse=True)[0]
    coverage_guarded = sorted(grid, key=lambda row: (metric_summary(row)["center_recall"], metric_summary(row)["iou_0_30_recall"], metric_summary(row)["candidate_inflation"] <= 7.0, metric_summary(row)["precision"], -metric_summary(row)["candidate_inflation"]), reverse=True)[0]
    return {"recall_first": recall_first, "balanced_compression": balanced, "coverage_guarded": coverage_guarded}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/symbol_center_branch_v30/manifest.json")
    parser.add_argument("--center-predictions", default="reports/vlm/symbol_center_branch_v30_smoke_predictions.jsonl")
    parser.add_argument("--mask-predictions", default="reports/vlm/symbol_yolov8s_seg_rect_v28_smoke_v30_page_predictions.jsonl")
    parser.add_argument("--selector", default="checkpoints/symbol_proposal_merger_v31/coverage_selector.joblib")
    parser.add_argument("--split", default="smoke_v30")
    parser.add_argument("--eval-output", default="reports/vlm/symbol_proposal_merger_v31_coverage_smoke_page_eval.json")
    parser.add_argument("--predictions-output", default="reports/vlm/symbol_proposal_merger_v31_coverage_smoke_page_predictions.jsonl")
    parser.add_argument("--audit-output", default="reports/vlm/symbol_proposal_merger_v31_coverage_smoke_error_audit.json")
    parser.add_argument("--coverage-loss-output", default="reports/vlm/symbol_proposal_merger_v31_coverage_smoke_coverage_loss.json")
    parser.add_argument("--selector-threshold-grid", default="0.01,0.02,0.04,0.06,0.08,0.1,0.15,0.2,0.3,0.4")
    parser.add_argument("--cluster-topk-grid", default="0,1,2")
    parser.add_argument("--min-cluster-score-grid", default="0.0,0.01,0.03,0.06,0.1,0.2")
    parser.add_argument("--pre-nms-budget-grid", default="0,80,100,120,150,200")
    parser.add_argument("--pre-nms-budget-ratio-grid", default="0")
    parser.add_argument("--nms-threshold-grid", default="0.35,0.45,0.55,0.65")
    parser.add_argument("--max-per-page", type=int, default=1200)
    args = parser.parse_args()

    if args.split != "smoke_v30":
        raise SystemExit("v31 coverage selector currently supports smoke_v30 first; locked promotion follows after smoke phase gate.")
    manifest_path = Path(args.dataset)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    golds = build_golds_from_center_targets(rel_from_manifest(manifest_path, manifest["outputs"]["smoke_center_targets"]))
    merged = merge_sources(load_preds(Path(args.mask_predictions)), load_preds(Path(args.center_predictions)))
    rescored, selector_info = score_with_selector(merged, Path(args.selector))

    grid = []
    for threshold in [float(v) for v in args.selector_threshold_grid.split(",") if v.strip()]:
        for topk in [int(v) for v in args.cluster_topk_grid.split(",") if v.strip()]:
            for min_cluster_score in [float(v) for v in args.min_cluster_score_grid.split(",") if v.strip()]:
                for pre_nms_budget in [int(v) for v in args.pre_nms_budget_grid.split(",") if v.strip()]:
                    for pre_nms_budget_ratio in [float(v) for v in args.pre_nms_budget_ratio_grid.split(",") if v.strip()]:
                        selected_pages = {
                            row_id: select_policy(preds, threshold, topk, min_cluster_score, pre_nms_budget, pre_nms_budget_ratio, args.max_per_page)
                            for row_id, preds in rescored.items()
                        }
                        for nms_threshold in [float(v) for v in args.nms_threshold_grid.split(",") if v.strip()]:
                            metrics, predictions, errors = score_predictions(selected_pages, golds, 0.0, nms_threshold, args.max_per_page, 200)
                            grid.append(
                                {
                                    "policy": {
                                        "selector": rel(Path(args.selector)),
                                        "selector_score_threshold": threshold,
                                        "cluster_topk": topk,
                                        "min_cluster_score": min_cluster_score,
                                        "pre_nms_budget_per_page": pre_nms_budget,
                                        "pre_nms_budget_ratio": pre_nms_budget_ratio,
                                        "nms_threshold": nms_threshold,
                                        "max_per_page": args.max_per_page,
                                    },
                                    "metrics": metrics,
                                    "source_counts": source_counts(predictions),
                                    "error_count": len(errors),
                                    "predictions": predictions,
                                    "errors": errors,
                                }
                            )
    selected_views = choose_views(grid)
    selected = selected_views["coverage_guarded"]
    loss = coverage_loss(selected["predictions"], golds)
    report = {
        "version": "symbol_proposal_merger_v31_coverage_smoke_eval",
        "metric_mode": "smoke",
        "claim_boundary": "Offline coverage-aware selector evaluation. Runtime input remains raster-derived proposals and selector weights; gold is evaluation/audit only.",
        "dataset": rel(manifest_path),
        "inputs": {"mask_predictions": rel(Path(args.mask_predictions)), "center_predictions": rel(Path(args.center_predictions))},
        "selector_info": selector_info,
        "selected_policy": selected["policy"],
        "selected_metrics": selected["metrics"],
        "selected_source_counts": selected["source_counts"],
        "selection_views": {
            name: {"policy": row["policy"], "metrics": row["metrics"], "source_counts": row["source_counts"]}
            for name, row in selected_views.items()
        },
        "threshold_grid": [{key: value for key, value in row.items() if key not in {"predictions", "errors"}} for row in grid],
        "coverage_loss_summary": {key: value for key, value in loss.items() if key != "examples"},
        "stage_gate": {
            "phase_B_center_recall_min_0_90": selected["metrics"]["symbol_bbox_center_recall"] >= 0.90,
            "phase_B_iou_0_30_recall_min_0_72": selected["metrics"]["symbol_bbox_iou_0_30"]["recall"] >= 0.72,
            "phase_B_precision_min_0_12": selected["metrics"]["symbol_bbox_iou_0_30"]["precision"] >= 0.12,
            "phase_B_candidate_inflation_max_7": selected["metrics"]["candidate_inflation"] <= 7.0,
            "phase_C_center_recall_min_0_94": selected["metrics"]["symbol_bbox_center_recall"] >= 0.94,
            "phase_C_iou_0_30_recall_min_0_82": selected["metrics"]["symbol_bbox_iou_0_30"]["recall"] >= 0.82,
        },
    }
    report["stage_gate"]["phase_B_passed"] = all(report["stage_gate"][key] for key in report["stage_gate"] if key.startswith("phase_B"))
    report["stage_gate"]["phase_C_passed"] = report["stage_gate"]["phase_B_passed"] and all(report["stage_gate"][key] for key in report["stage_gate"] if key.startswith("phase_C"))
    write_json(Path(args.eval_output), report)
    write_jsonl(Path(args.predictions_output), selected["predictions"])
    write_json(Path(args.audit_output), {"errors": selected["errors"][:2000], "source_counts": selected["source_counts"]})
    write_json(Path(args.coverage_loss_output), loss)
    print(json.dumps({"selected_metrics": selected["metrics"], "source_counts": selected["source_counts"], "stage_gate": report["stage_gate"], "coverage_loss": loss["counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
