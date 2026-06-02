#!/usr/bin/env python3
"""Build train/dev/locked v36 suppression rows from v35 detector predictions."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_symbol_proposal_selector_features_v30 import load_preds
from cache_symbol_proposal_eval_v35 import candidate_gold_matches, merge_sources
from train_symbol_tile_detector_v20 import area_bucket, bbox_iou, load_jsonl, rel, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
LABELS = ["appliance", "bathtub", "column", "equipment", "generic_symbol", "shower", "sink", "stair", "table"]


def source_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def selected_tile_ids(pred_map: dict[str, list[dict[str, Any]]]) -> set[str]:
    out: set[str] = set()
    for preds in pred_map.values():
        for pred in preds:
            tile_id = str(pred.get("tile_id") or "")
            if tile_id:
                out.add(tile_id)
    return out


def load_golds(tile_jsonl: Path, row_ids: set[str], tile_ids: set[str]) -> dict[str, dict[str, dict[str, Any]]]:
    pages: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in load_jsonl(tile_jsonl):
        row_id = str(row.get("row_id") or "")
        if row_id not in row_ids:
            continue
        if tile_ids and str(row.get("id") or "") not in tile_ids:
            continue
        for gold in ((row.get("targets") or {}).get("boxes") or []):
            target_id = str(gold.get("target_id") or f"{row_id}_symbol_{len(pages[row_id])}")
            box = [float(v) for v in gold.get("page_bbox") or gold.get("bbox") or []]
            if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
                continue
            pages[row_id][target_id] = {
                "target_id": target_id,
                "bbox": box,
                "label": str(gold.get("label") or "generic_symbol"),
                "area_bucket": str(gold.get("area_bucket") or area_bucket(box)),
            }
    return pages


def valid_box(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    box = [float(v) for v in value]
    return box if box[2] > box[0] and box[3] > box[1] else None


def row_features(pred: dict[str, Any], box: list[float], page_candidate_count: int, source_rank: int) -> dict[str, float]:
    width = box[2] - box[0]
    height = box[3] - box[1]
    area = width * height
    score = float(pred.get("score", pred.get("selector_score", 0.0)) or 0.0)
    feats = {
        "score": score,
        "score_logit_safe": math.log(max(score, 1e-6) / max(1.0 - score, 1e-6)),
        "width": width,
        "height": height,
        "area": area,
        "log_area": math.log1p(area),
        "aspect": max(width, height) / max(min(width, height), 1e-6),
        "page_candidate_count": float(page_candidate_count),
        "source_rank": float(source_rank),
        "source_is_pretrained_tiny_v35": 1.0,
    }
    label = str(pred.get("label") or "generic_symbol")
    for name in LABELS:
        feats[f"label_is_{name}"] = 1.0 if label == name else 0.0
    bucket = area_bucket(box)
    for name in ["tiny_le_64", "small_le_256", "medium_le_1024", "large_le_4096", "xlarge_gt_4096"]:
        feats[f"area_is_{name}"] = 1.0 if bucket == name else 0.0
    return feats


def build_cache(predictions: Path, tile_jsonl: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tiny = load_preds(predictions)
    merged = merge_sources(("pretrained_tiny_v35", tiny))
    tile_ids = selected_tile_ids(tiny)
    golds = load_golds(tile_jsonl, set(merged), tile_ids)
    rows: list[dict[str, Any]] = []
    counts = Counter()
    for row_id, preds in sorted(merged.items()):
        gold_map = golds.get(row_id, {})
        gold_rows = [
            {"target_id": gold["target_id"], "bbox": gold["bbox"], "label": gold["label"], "area_bucket": gold["area_bucket"]}
            for gold in gold_map.values()
        ]
        rows.append({"row_id": row_id, "predicted_symbols": preds, "gold_symbols": gold_rows, "candidate_gold_matches": candidate_gold_matches(preds, gold_map)})
        counts["pages"] += 1
        counts["candidates"] += len(preds)
        counts["golds"] += len(gold_rows)
    counts["selected_tile_ids"] = len(tile_ids)
    return rows, dict(counts)


def build_listwise(cache_rows: list[dict[str, Any]], split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out: list[dict[str, Any]] = []
    counts = Counter()
    for page in cache_rows:
        page_id = str(page["row_id"])
        preds = list(page.get("predicted_symbols") or [])
        matches = {int(m["candidate_index"]): m for m in page.get("candidate_gold_matches") or []}
        page_gold_targets = [
            {"target_id": str(g["target_id"]), "label": str(g["label"]), "area_bucket": str(g["area_bucket"])}
            for g in page.get("gold_symbols") or []
        ]
        best_positive_for_gold: dict[str, tuple[float, int]] = {}
        source_seen = Counter()
        for index, _pred in enumerate(preds):
            match = matches.get(index, {})
            target = match.get("best_iou_target_id")
            best_iou = float(match.get("best_iou", 0.0) or 0.0)
            if target and best_iou >= 0.30:
                previous = best_positive_for_gold.get(str(target))
                if previous is None or best_iou > previous[0]:
                    best_positive_for_gold[str(target)] = (best_iou, index)
        keep_indices = {index for _target, (_iou, index) in best_positive_for_gold.items()}
        for index, pred in enumerate(preds):
            box = valid_box(pred.get("bbox"))
            if box is None:
                continue
            source = str(pred.get("proposal_source") or "pretrained_tiny_v35")
            source_seen[source] += 1
            match = matches.get(index, {})
            best_iou = float(match.get("best_iou", 0.0) or 0.0)
            center_ids = [str(v) for v in match.get("center_target_ids") or []]
            keep = index in keep_indices
            if keep:
                reason = "best_iou_positive"
            elif center_ids and best_iou < 0.30:
                reason = "center_only_no_iou"
            elif best_iou >= 0.30:
                reason = "same_gold_duplicate"
            else:
                reason = "source_or_background_negative"
            out.append(
                {
                    "page_id": page_id,
                    "split": split,
                    "candidate_id": f"{page_id}:{index}",
                    "candidate_index": index,
                    "cluster_id": index,
                    "bbox": box,
                    "label": str(pred.get("label") or "generic_symbol"),
                    "proposal_source": source,
                    "score": float(pred.get("score", 0.0) or 0.0),
                    "features": row_features(pred, box, len(preds), source_seen[source]),
                    "labels": {
                        "keep": keep,
                        "suppression_reason": reason,
                        "best_iou": best_iou,
                        "best_iou_target_id": match.get("best_iou_target_id"),
                        "center_target_ids": center_ids,
                        "page_gold_targets": page_gold_targets,
                    },
                    "source_integrity": {
                        "runtime_features_from": "raster-derived candidate bbox/score/source/type only",
                        "gold_used_for_inference": False,
                    },
                }
            )
            counts[f"split:{split}"] += 1
            counts[f"reason:{reason}"] += 1
            counts["positive" if keep else "negative"] += 1
    return out, dict(counts)


def add_split(name: str, predictions: Path, tiles: Path, output_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache_rows, cache_counts = build_cache(predictions, tiles)
    write_jsonl(output_dir / f"{name}_cache.jsonl", cache_rows)
    rows, row_counts = build_listwise(cache_rows, name)
    write_jsonl(output_dir / f"{name}_rows.jsonl", rows)
    return rows, {f"cache_{k}": v for k, v in cache_counts.items()} | {f"rows_{k}": v for k, v in row_counts.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="datasets/symbol_support_suppression_v36")
    parser.add_argument("--train-predictions", default="reports/vlm/symbol_pretrained_tiny_detector_v35_train_subset_predictions.jsonl")
    parser.add_argument("--train-tiles", default="datasets/symbol_tile_detector_tiny_sahi_v21/train.jsonl")
    parser.add_argument("--dev-predictions", default="reports/vlm/symbol_pretrained_tiny_detector_v35_dev_predictions.jsonl")
    parser.add_argument("--dev-tiles", default="datasets/symbol_tile_detector_tiny_sahi_v21/dev.jsonl")
    parser.add_argument("--locked-predictions", default="reports/vlm/symbol_pretrained_tiny_detector_v35_locked_predictions.jsonl")
    parser.add_argument("--locked-tiles", default="datasets/symbol_tile_detector_tiny_sahi_v21/locked.jsonl")
    args = parser.parse_args()
    output_dir = source_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    counts: dict[str, Any] = {}
    for split, pred, tiles in [
        ("train", source_path(args.train_predictions), source_path(args.train_tiles)),
        ("dev", source_path(args.dev_predictions), source_path(args.dev_tiles)),
        ("locked", source_path(args.locked_predictions), source_path(args.locked_tiles)),
    ]:
        rows, split_counts = add_split(split, pred, tiles, output_dir)
        all_rows.extend(rows)
        counts[split] = split_counts
    rows_path = output_dir / "listwise_rows.jsonl"
    write_jsonl(rows_path, all_rows)
    manifest = {
        "version": "symbol_support_suppression_v36",
        "task": "P1-06-train-dev-v35-set-policy-generalization",
        "outputs": {
            "rows": rel(rows_path),
            "train_rows": rel(output_dir / "train_rows.jsonl"),
            "dev_rows": rel(output_dir / "dev_rows.jsonl"),
            "locked_rows": rel(output_dir / "locked_rows.jsonl"),
        },
        "counts": counts,
        "source_integrity": {
            "runtime_input_allowed": ["raster-derived candidate bbox", "candidate score", "proposal source", "predicted type"],
            "offline_labels_used_for": ["training", "dev_threshold_selection", "locked_evaluation"],
            "gold_used_for_inference": False,
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps({"manifest": rel(output_dir / "manifest.json"), "counts": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
