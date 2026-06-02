#!/usr/bin/env python3
"""Build true-label listwise recovery rows from clean frontier detector predictions."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from eval_symbol_yolo_tile_detector_v22 import sample_tiles_area_aware
from train_symbol_tile_detector_v20 import area_bucket, bbox_iou, center_covered, load_jsonl, rel, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
SUPPORT_PATH = ROOT / "scripts" / "vlm" / "build_symbol_support_suppression_dataset_v32.py"
SPEC = importlib.util.spec_from_file_location("build_symbol_support_suppression_dataset_v32", SUPPORT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to load {SUPPORT_PATH}")
SUPPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUPPORT)
feature_vector = SUPPORT.feature_vector
valid_box = SUPPORT.valid_box


def source_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def split_for_page(row_id: str) -> str:
    score = sum(ord(ch) for ch in row_id)
    bucket = score % 10
    if bucket < 7:
        return "train"
    if bucket < 8:
        return "dev"
    return "smoke_eval"


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def page_golds_from_smoke(smoke_rows: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[float]]]:
    by_page: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    page_size: dict[str, list[float]] = {}
    for row in smoke_rows:
        page_id = str(row.get("row_id") or "")
        if not page_id:
            continue
        if page_id not in page_size:
            size = row.get("image_size") or [1.0, 1.0]
            page_size[page_id] = [float(size[0] or 1.0), float(size[1] or 1.0)]
        for gold in ((row.get("targets") or {}).get("boxes") or []):
            target_id = str(gold.get("target_id") or "")
            if not target_id:
                continue
            box = valid_box(gold.get("page_bbox") or gold.get("bbox"))
            if box is None:
                continue
            by_page[page_id][target_id] = {
                "target_id": target_id,
                "bbox": box,
                "label": str(gold.get("label") or "generic_symbol"),
                "label_id": int(gold.get("label_id") or 5),
                "area_bucket": str(gold.get("area_bucket") or area_bucket(box)),
            }
    return {key: list(value.values()) for key, value in by_page.items()}, page_size


def assign_cluster_ids(preds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clusters: list[list[float]] = []
    out: list[dict[str, Any]] = []
    for pred in sorted(preds, key=lambda item: float(item.get("score") or 0.0), reverse=True):
        box = valid_box(pred.get("bbox"))
        if box is None:
            continue
        label_id = int(pred.get("label_id") or 5)
        cluster_id = None
        for idx, cluster_box in enumerate(clusters):
            if bbox_iou(box, cluster_box) >= 0.25:
                cluster_id = idx
                break
        if cluster_id is None:
            cluster_id = len(clusters)
            clusters.append(box)
        item = dict(pred)
        item["cluster_id"] = int(cluster_id * 17 + label_id)
        out.append(item)
    return out


def match_candidate(pred: dict[str, Any], golds: list[dict[str, Any]]) -> dict[str, Any]:
    box = valid_box(pred.get("bbox"))
    if box is None:
        return {"best_iou": 0.0, "best_iou_target_id": None, "center_target_ids": []}
    best_iou = 0.0
    best_target = None
    center_targets: list[str] = []
    for gold in golds:
        gold_box = [float(v) for v in gold["bbox"]]
        iou = bbox_iou(box, gold_box)
        if iou > best_iou:
            best_iou = iou
            best_target = str(gold.get("target_id") or "")
        if center_covered(box, gold_box):
            center_targets.append(str(gold.get("target_id") or ""))
    return {"best_iou": best_iou, "best_iou_target_id": best_target, "center_target_ids": center_targets}


def build_rows(
    prediction_pages: list[dict[str, Any]],
    gold_by_page: dict[str, list[dict[str, Any]]],
    page_size: dict[str, list[float]],
    detector_source: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counts = Counter()
    for page in prediction_pages:
        page_id = str(page.get("row_id") or page.get("page_id") or "")
        golds = gold_by_page.get(page_id, [])
        if not page_id or not golds:
            continue
        preds = assign_cluster_ids(list(page.get("predicted_symbols") or []))
        cluster_map: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for pred in preds:
            cluster_map[int(pred.get("cluster_id") or 0)].append(pred)
        page_stats = {"page_candidate_count": float(len(preds))}
        size = page_size.get(page_id, [1.0, 1.0])
        page_gold_targets = [
            {"target_id": gold["target_id"], "label": gold["label"], "area_bucket": gold["area_bucket"]}
            for gold in golds
        ]
        cluster_stats_by_id: dict[int, dict[str, float]] = {}
        for cluster_id, items in cluster_map.items():
            scores = [safe_float(item.get("score")) for item in items]
            cluster_stats_by_id[cluster_id] = {
                "cluster_size": float(len(items)),
                "cluster_score_max": max(scores) if scores else 0.0,
                "cluster_score_mean": sum(scores) / max(len(scores), 1),
                "cluster_source_center_count": 0.0,
                "cluster_mask_count": 0.0,
            }
        for index, pred in enumerate(preds):
            box = valid_box(pred.get("bbox"))
            if box is None:
                continue
            cluster_id = int(pred.get("cluster_id") or 0)
            match = match_candidate(pred, golds)
            best_iou = safe_float(match["best_iou"])
            center_ids = [item for item in match["center_target_ids"] if item]
            keep = best_iou >= 0.30
            labels = {
                "keep": keep,
                "drop": not keep,
                "best_iou": round(best_iou, 6),
                "best_iou_target_id": match["best_iou_target_id"],
                "center_target_ids": center_ids,
                "page_gold_targets": page_gold_targets,
                "suppression_reason": "iou_positive" if keep else ("center_only_no_iou" if center_ids else "duplicate_or_background"),
            }
            row = {
                "row_id": page_id,
                "page_id": page_id,
                "candidate_id": f"{page_id}_frontier_cand_{index}",
                "candidate_index": index,
                "split": split_for_page(page_id),
                "cluster_id": cluster_id,
                "cluster_key": f"{page_id}|{cluster_id}",
                "proposal_source": detector_source,
                "detector_source": detector_source,
                "bbox": box,
                "score": safe_float(pred.get("score")),
                "selector_score": safe_float(pred.get("score")),
                "pre_selector_score": safe_float(pred.get("score")),
                "label": pred.get("label"),
                "label_id": int(pred.get("label_id") or 5),
                "page_stats": page_stats,
                "labels": labels,
            }
            row["features"] = feature_vector(row, row, match, cluster_stats_by_id[cluster_id], page_stats, size)
            rows.append(row)
            counts["rows"] += 1
            counts[f"split:{row['split']}"] += 1
            counts["positive" if keep else "negative"] += 1
            counts[f"reason:{labels['suppression_reason']}"] += 1
    manifest = {
        "version": "symbol_detector_listwise_recovery_v47",
        "rows": len(rows),
        "counts": dict(counts),
        "source_integrity": {
            "runtime_input": "raster detector prediction boxes, scores, and predicted types only",
            "offline_labels_used_for": ["policy_training", "smoke_evaluation", "audit"],
            "gold_used_for_inference": False,
            "runtime_uses_svg_or_cad_geometry": False,
        },
    }
    return rows, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default="reports/vlm/symbol_detector_frontier_yolo_v47_clean_grid_eval2_compression_predictions.jsonl")
    parser.add_argument("--smoke-rows", default="datasets/symbol_tile_detector_tiny_sahi_v21/smoke_v30.jsonl")
    parser.add_argument("--output-dir", default="datasets/symbol_detector_listwise_recovery_v47")
    parser.add_argument("--detector-source", default="clean_frontier_yolo_v47")
    parser.add_argument("--limit-smoke-tiles", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument("--positive-ratio", type=float, default=0.9)
    parser.add_argument("--small-positive-ratio", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = source_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    smoke_rows_all = load_jsonl(source_path(args.smoke_rows))
    smoke_rows = sample_tiles_area_aware(smoke_rows_all, args.limit_smoke_tiles, args.seed, args.positive_ratio, args.small_positive_ratio)
    gold_by_page, page_size = page_golds_from_smoke(smoke_rows)
    prediction_pages = load_jsonl(source_path(args.predictions))
    rows, manifest = build_rows(prediction_pages, gold_by_page, page_size, args.detector_source)
    rows_path = output_dir / "rows.jsonl"
    write_jsonl(rows_path, rows)
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_split[str(row.get("split") or "train")].append(row)
    outputs = {"rows": rel(rows_path)}
    for split, split_rows in sorted(by_split.items()):
        split_path = output_dir / f"{split}_rows.jsonl"
        write_jsonl(split_path, split_rows)
        outputs[f"{split}_rows"] = rel(split_path)
    manifest.update(
        {
            "predictions": rel(source_path(args.predictions)),
            "smoke_rows": rel(source_path(args.smoke_rows)),
            "sampled_tiles": len(smoke_rows),
            "pages_with_gold": len(gold_by_page),
            "outputs": outputs,
            "split_counts": {split: len(split_rows) for split, split_rows in sorted(by_split.items())},
        }
    )
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
