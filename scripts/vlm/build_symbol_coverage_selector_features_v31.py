#!/usr/bin/env python3
"""Build coverage/listwise features for the v31 symbol proposal selector."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_symbol_proposal_selector_features_v30 import box_features, center_distance, load_golds, load_preds, source_merge
from train_symbol_tile_detector_v20 import area_bucket, bbox_iou, center_covered, rel, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]


def valid_box(pred: dict[str, Any]) -> list[float] | None:
    box = [float(v) for v in pred.get("bbox") or []]
    if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def build_clusters(preds: list[dict[str, Any]], center_threshold: float, iou_threshold: float) -> list[int]:
    parent = list(range(len(preds)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    boxes = [valid_box(pred) for pred in preds]
    for left in range(len(preds)):
        left_box = boxes[left]
        if left_box is None:
            continue
        for right in range(left + 1, len(preds)):
            right_box = boxes[right]
            if right_box is None:
                continue
            if bbox_iou(left_box, right_box) >= iou_threshold or center_distance(left_box, right_box) <= center_threshold:
                union(left, right)
    roots: dict[int, int] = {}
    out: list[int] = []
    for index in range(len(preds)):
        root = find(index)
        if root not in roots:
            roots[root] = len(roots)
        out.append(roots[root])
    return out


def peer_features(index: int, preds: list[dict[str, Any]], cluster_ids: list[int]) -> dict[str, float]:
    box = valid_box(preds[index]) or [0.0, 0.0, 0.0, 0.0]
    source = str(preds[index].get("proposal_source") or "unknown")
    cluster_id = cluster_ids[index]
    max_iou = 0.0
    overlap_count = 0
    near_center_count = 0
    cluster_size = 0
    cluster_mask_count = 0
    cluster_center_count = 0
    source_peer_count = 0
    scores: list[float] = []
    for other_index, other in enumerate(preds):
        other_box = valid_box(other) or [0.0, 0.0, 0.0, 0.0]
        if other_index != index:
            iou = bbox_iou(box, other_box)
            max_iou = max(max_iou, iou)
            if iou >= 0.30:
                overlap_count += 1
            if center_distance(box, other_box) <= 12.0:
                near_center_count += 1
        if cluster_ids[other_index] == cluster_id:
            cluster_size += 1
            scores.append(float(other.get("score", 0.0)))
            other_source = str(other.get("proposal_source") or "unknown")
            cluster_mask_count += int(other_source == "mask_v28")
            cluster_center_count += int(other_source == "center_branch_v30")
            source_peer_count += int(other_source == source)
    score = float(preds[index].get("score", 0.0))
    sorted_scores = sorted(scores, reverse=True)
    rank = sorted_scores.index(score) + 1 if score in sorted_scores else len(sorted_scores)
    return {
        "max_peer_iou": max_iou,
        "overlap_peer_count": float(overlap_count),
        "near_center_peer_count": float(near_center_count),
        "cluster_size": float(cluster_size),
        "cluster_mask_count": float(cluster_mask_count),
        "cluster_center_count": float(cluster_center_count),
        "cluster_has_both_sources": float(cluster_mask_count > 0 and cluster_center_count > 0),
        "same_source_cluster_count": float(source_peer_count),
        "cluster_score_max": max(scores) if scores else 0.0,
        "cluster_score_mean": sum(scores) / max(len(scores), 1),
        "score_rank_in_cluster": float(rank),
        "score_rank_ratio_in_cluster": float(rank / max(cluster_size, 1)),
    }


def gold_labels(pred_box: list[float], pred_label: str, golds: list[dict[str, Any]]) -> dict[str, Any]:
    best_iou = 0.0
    best_center_iou = 0.0
    best_gold: dict[str, Any] | None = None
    best_center_gold: dict[str, Any] | None = None
    center_any = False
    for gold in golds:
        gold_box = [float(v) for v in gold.get("bbox") or []]
        if len(gold_box) != 4:
            continue
        iou = bbox_iou(pred_box, gold_box)
        if iou > best_iou:
            best_iou = iou
            best_gold = gold
        if center_covered(pred_box, gold_box):
            center_any = True
            if iou >= best_center_iou:
                best_center_iou = iou
                best_center_gold = gold
    positive = bool(best_gold and best_iou >= 0.30)
    typed_positive = bool(positive and str(best_gold.get("label")) == pred_label)
    coverage_gold = best_gold if positive else best_center_gold
    return {
        "target": int(positive),
        "typed_target": int(typed_positive),
        "coverage_target": int(positive or center_any),
        "best_iou": round(best_iou, 6),
        "center_covers_any_gold": int(center_any),
        "best_gold_label": None if coverage_gold is None else coverage_gold.get("label"),
        "best_gold_area_bucket": None if coverage_gold is None else coverage_gold.get("area_bucket"),
        "best_gold_target_id": None if coverage_gold is None else coverage_gold.get("target_id"),
    }


def annotate_gold_ranks(rows: list[dict[str, Any]]) -> None:
    by_gold: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        target_id = (row.get("labels") or {}).get("best_gold_target_id")
        if target_id:
            by_gold[str(target_id)].append(row)
    for gold_rows in by_gold.values():
        by_iou = sorted(gold_rows, key=lambda row: ((row.get("labels") or {}).get("best_iou") or 0.0, row["features"].get("score", 0.0)), reverse=True)
        by_score = sorted(gold_rows, key=lambda row: row["features"].get("score", 0.0), reverse=True)
        for rank, row in enumerate(by_iou, start=1):
            row["features"]["best_iou_rank_for_gold"] = float(rank)
            row["labels"]["is_best_iou_for_gold"] = int(rank == 1 and row["labels"].get("target") == 1)
        for rank, row in enumerate(by_score, start=1):
            row["features"]["score_rank_for_gold"] = float(rank)
        positive_count = sum(int((row.get("labels") or {}).get("target") or 0) for row in gold_rows)
        coverage_count = sum(int((row.get("labels") or {}).get("coverage_target") or 0) for row in gold_rows)
        for row in gold_rows:
            row["features"]["same_gold_positive_count"] = float(positive_count)
            row["features"]["same_gold_coverage_count"] = float(coverage_count)
            row["labels"]["sole_positive_for_gold"] = int(positive_count == 1 and row["labels"].get("target") == 1)


def audit_uncovered_golds(rows: list[dict[str, Any]], golds: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    covered_iou = {(row["row_id"], row["labels"].get("best_gold_target_id")) for row in rows if row["labels"].get("target")}
    covered_center = {(row["row_id"], row["labels"].get("best_gold_target_id")) for row in rows if row["labels"].get("coverage_target")}
    counts = Counter()
    examples = []
    for row_id, page_golds in golds.items():
        for gold in page_golds:
            key = (row_id, gold["target_id"])
            bucket = str(gold.get("area_bucket") or area_bucket(gold["bbox"]))
            label = str(gold.get("label") or "generic_symbol")
            if key not in covered_center:
                counts["raw_no_center_coverage"] += 1
                counts[f"raw_no_center_coverage_label:{label}"] += 1
                counts[f"raw_no_center_coverage_area:{bucket}"] += 1
                if len(examples) < 100:
                    examples.append({"row_id": row_id, "target_id": gold["target_id"], "label": label, "area_bucket": bucket, "bbox": gold["bbox"], "error": "raw_no_center_coverage"})
            if key not in covered_iou:
                counts["raw_no_iou_coverage"] += 1
                counts[f"raw_no_iou_coverage_label:{label}"] += 1
                counts[f"raw_no_iou_coverage_area:{bucket}"] += 1
    return {"counts": dict(counts), "examples": examples}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/symbol_center_branch_v30/manifest.json")
    parser.add_argument("--mask-predictions", default="reports/vlm/symbol_yolov8s_seg_rect_v28_smoke_v30_page_predictions.jsonl")
    parser.add_argument("--center-predictions", default="reports/vlm/symbol_center_branch_v30_smoke_predictions.jsonl")
    parser.add_argument("--output", default="datasets/symbol_proposal_selector_v31/smoke_coverage_features.jsonl")
    parser.add_argument("--manifest-output", default="datasets/symbol_proposal_selector_v31/manifest.json")
    parser.add_argument("--cluster-center-threshold", type=float, default=12.0)
    parser.add_argument("--cluster-iou-threshold", type=float, default=0.30)
    args = parser.parse_args()

    manifest_path = Path(args.dataset)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    golds = load_golds(ROOT / manifest["outputs"]["smoke_center_targets"])
    candidates = source_merge(load_preds(Path(args.mask_predictions)), load_preds(Path(args.center_predictions)))

    rows: list[dict[str, Any]] = []
    counts = Counter()
    for row_id, raw_preds in candidates.items():
        preds = [pred for pred in raw_preds if valid_box(pred)]
        cluster_ids = build_clusters(preds, args.cluster_center_threshold, args.cluster_iou_threshold)
        page_rows: list[dict[str, Any]] = []
        for index, pred in enumerate(preds):
            box = valid_box(pred)
            if box is None:
                continue
            source = str(pred.get("proposal_source") or "unknown")
            label = str(pred.get("label") or "generic_symbol")
            features = {
                "score": float(pred.get("score", 0.0)),
                "is_mask_v28": float(source == "mask_v28"),
                "is_center_branch_v30": float(source == "center_branch_v30"),
                "label_id": float(pred.get("label_id") or 5),
                "candidate_count_page": float(len(preds)),
            }
            features.update(box_features(box))
            features.update(peer_features(index, preds, cluster_ids))
            labels = gold_labels(box, label, golds.get(row_id, []))
            page_rows.append(
                {
                    "row_id": row_id,
                    "candidate_index": index,
                    "cluster_id": int(cluster_ids[index]),
                    "proposal_source": source,
                    "bbox": box,
                    "label": label,
                    "label_id": int(pred.get("label_id") or 5),
                    "features": features,
                    "labels": labels,
                }
            )
        annotate_gold_ranks(page_rows)
        for row in page_rows:
            for name in ["best_iou_rank_for_gold", "score_rank_for_gold", "same_gold_positive_count", "same_gold_coverage_count"]:
                row["features"].setdefault(name, 0.0)
            row["labels"].setdefault("is_best_iou_for_gold", 0)
            row["labels"].setdefault("sole_positive_for_gold", 0)
            rows.append(row)
            counts[f"source:{row['proposal_source']}"] += 1
            counts[f"target:{row['labels']['target']}"] += 1
            counts[f"coverage_target:{row['labels']['coverage_target']}"] += 1
            if row["labels"].get("is_best_iou_for_gold"):
                counts["best_iou_for_gold"] += 1
            if row["labels"].get("sole_positive_for_gold"):
                counts["sole_positive_for_gold"] += 1

    write_jsonl(Path(args.output), rows)
    raw_coverage = audit_uncovered_golds(rows, golds)
    out_manifest = {
        "version": "symbol_coverage_selector_features_v31",
        "metric_mode": "smoke",
        "claim_boundary": "Offline coverage-aware selector feature table. Gold labels only supervise/audit coverage; runtime selector uses proposal features only.",
        "source_integrity": {
            "runtime_model_input": ["raster-derived proposal bbox", "proposal score", "proposal source", "proposal cluster features"],
            "gold_used_for_runtime_feature": False,
            "gold_used_for_training_labels": True,
        },
        "inputs": {
            "dataset": rel(manifest_path),
            "mask_predictions": rel(Path(args.mask_predictions)),
            "center_predictions": rel(Path(args.center_predictions)),
        },
        "outputs": {"features": rel(Path(args.output))},
        "cluster_config": {"center_threshold": args.cluster_center_threshold, "iou_threshold": args.cluster_iou_threshold},
        "counts": dict(counts) | {"rows": len(rows), "pages": len({row["row_id"] for row in rows}), "clusters": len({(row["row_id"], row["cluster_id"]) for row in rows})},
        "feature_names": sorted(rows[0]["features"]) if rows else [],
        "raw_candidate_coverage_audit": raw_coverage,
    }
    write_json(Path(args.manifest_output), out_manifest)
    print(json.dumps({"features": rel(Path(args.output)), "counts": out_manifest["counts"], "raw_coverage": raw_coverage["counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
