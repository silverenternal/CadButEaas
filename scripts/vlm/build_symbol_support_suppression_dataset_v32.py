#!/usr/bin/env python3
"""Build a page/cluster listwise dataset for symbol support suppression v32."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from train_symbol_tile_detector_v20 import area_bucket, bbox_iou, rel, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = ROOT / "reports/vlm/symbol_proposal_eval_v31_smoke_cache.jsonl"
DEFAULT_AUDIT = ROOT / "reports/vlm/symbol_proposal_merger_v31_cache_sweep_support_negative_audit.json"
DEFAULT_CENTER_TARGETS = ROOT / "datasets/symbol_center_branch_v30/smoke_center_targets.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "datasets/symbol_support_suppression_v32"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def valid_box(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        box = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def choose_target_bucket(match: dict[str, Any], score: float, source: str) -> str:
    best_iou = safe_float(match.get("best_iou"))
    center_ids = match.get("center_target_ids") or []
    if best_iou >= 0.30:
        return "keep_positive"
    if center_ids:
        return "center_only_no_iou"
    if score >= 0.90:
        return "source_specific_false_positive"
    if source == "mask_v28" and score >= 0.50:
        return "same_cluster_duplicate"
    if source == "center_branch_v30" and score >= 0.30:
        return "support_negative"
    return "low_score_background"


def split_for_page(row_id: str) -> str:
    score = sum(ord(ch) for ch in row_id)
    bucket = score % 10
    if bucket < 7:
        return "train"
    if bucket < 8:
        return "dev"
    return "smoke_eval"


def cluster_key(pred: dict[str, Any]) -> str:
    return f"{pred.get('row_id')}|{int(pred.get('cluster_id') or 0)}"


def source_priority(source: str) -> int:
    return 1 if source == "center_branch_v30" else 0


def feature_vector(
    row: dict[str, Any],
    pred: dict[str, Any],
    match: dict[str, Any],
    cluster_stats: dict[str, float],
    page_stats: dict[str, float],
    page_size: list[float] | None,
) -> dict[str, float]:
    box = valid_box(pred.get("bbox")) or [0.0, 0.0, 0.0, 0.0]
    width = max(1e-6, box[2] - box[0])
    height = max(1e-6, box[3] - box[1])
    score = safe_float(pred.get("selector_score", pred.get("score")))
    source = str(pred.get("proposal_source") or "unknown")
    page_w = float(page_size[0]) if page_size and len(page_size) >= 2 else 1.0
    page_h = float(page_size[1]) if page_size and len(page_size) >= 2 else 1.0
    return {
        "score": score,
        "pre_selector_score": safe_float(pred.get("pre_selector_score", score)),
        "label_id": safe_float(pred.get("label_id") or 5),
        "is_mask_v28": 1.0 if source == "mask_v28" else 0.0,
        "is_center_branch_v30": 1.0 if source == "center_branch_v30" else 0.0,
        "cluster_id_mod_17": float(int(pred.get("cluster_id") or 0) % 17) / 17.0,
        "cluster_size": cluster_stats.get("cluster_size", 1.0),
        "cluster_score_max": cluster_stats.get("cluster_score_max", score),
        "cluster_score_mean": cluster_stats.get("cluster_score_mean", score),
        "cluster_source_center_count": cluster_stats.get("cluster_source_center_count", 0.0),
        "cluster_mask_count": cluster_stats.get("cluster_mask_count", 0.0),
        "page_candidate_count": page_stats.get("page_candidate_count", 0.0),
        "width_norm": width / max(page_w, 1.0),
        "height_norm": height / max(page_h, 1.0),
        "area_norm": (width * height) / max(page_w * page_h, 1.0),
        "aspect": width / height,
        "area_bucket_small": 1.0 if area_bucket(box) in {"tiny_le_64", "small_le_256"} else 0.0,
        "area_bucket_large": 1.0 if area_bucket(box) in {"large_le_4096", "xlarge_gt_4096"} else 0.0,
    }


def load_page_meta(center_targets_path: Path) -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(center_targets_path):
        row_id = str(row.get("row_id"))
        if row_id and row_id not in meta:
            meta[row_id] = {
                "image": row.get("image"),
                "image_size": row.get("image_size"),
            }
    return meta


def build_rows(cache_rows: list[dict[str, Any]], page_meta: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counts = Counter()
    bucket_counts = Counter()
    by_source = Counter()
    by_area = Counter()
    by_page = Counter()
    cluster_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cache_rows:
        row_id = str(row.get("row_id"))
        preds = list(row.get("predicted_symbols") or [])
        split = split_for_page(row_id)
        meta = page_meta.get(row_id, {})
        page_size = meta.get("image_size")
        matches = {int(match.get("candidate_index", -1)): match for match in row.get("candidate_gold_matches") or []}
        page_candidate_count = float(len(preds))
        page_stats = {"page_candidate_count": page_candidate_count}
        for index, pred in enumerate(preds):
            match = matches.get(index, {})
            source = str(pred.get("proposal_source") or "unknown")
            bucket = choose_target_bucket(match, safe_float(pred.get("selector_score", pred.get("score"))), source)
            key = cluster_key(pred)
            cluster_map[key].append(pred)
            area = area_bucket(valid_box(pred.get("bbox")) or [0.0, 0.0, 0.0, 0.0])
            by_area[area] += 1
            by_source[source] += 1
            bucket_counts[bucket] += 1
            by_page[row_id] += 1
            counts["rows"] += 1
            counts[f"bucket:{bucket}"] += 1
            counts[f"source:{source}"] += 1
            counts[f"area:{area}"] += 1
            rows.append(
                {
                    "row_id": row_id,
                    "page_id": row_id,
                    "candidate_id": f"{row_id}_cand_{index}",
                    "candidate_index": index,
                    "split": split,
                    "cluster_id": int(pred.get("cluster_id") or 0),
                    "cluster_key": key,
                    "proposal_source": source,
                    "bbox": pred.get("bbox"),
                    "score": safe_float(pred.get("score")),
                    "selector_score": safe_float(pred.get("selector_score", pred.get("score"))),
                    "pre_selector_score": safe_float(pred.get("pre_selector_score", pred.get("score"))),
                    "label": pred.get("label"),
                    "label_id": int(pred.get("label_id") or 5),
                    "cluster_bucket": bucket,
                    "gold_bucket": bucket,
                    "target_bucket": bucket,
                    "input_best_iou": round(safe_float(match.get("best_iou")), 6),
                    "input_center_target_count": len(match.get("center_target_ids") or []),
                    "target_keep": 1 if bucket == "keep_positive" else 0,
                    "target_drop": 1 if bucket != "keep_positive" else 0,
                    "target_suppression_reason": bucket,
                    "page_stats": page_stats,
                    "page_size": page_size,
                }
            )
    cluster_stats_by_key: dict[str, dict[str, float]] = {}
    for key, items in cluster_map.items():
        scores = [safe_float(item.get("selector_score", item.get("score"))) for item in items]
        cluster_source_center_count = sum(1.0 for item in items if str(item.get("proposal_source")) == "center_branch_v30")
        cluster_mask_count = sum(1.0 for item in items if str(item.get("proposal_source")) == "mask_v28")
        cluster_stats_by_key[key] = {
            "cluster_size": float(len(items)),
            "cluster_score_max": max(scores) if scores else 0.0,
            "cluster_score_mean": sum(scores) / max(len(scores), 1),
            "cluster_source_center_count": cluster_source_center_count,
            "cluster_mask_count": cluster_mask_count,
        }
    for row in rows:
        row.update(
            feature_vector(
                row,
                row,
                {"best_iou": row["input_best_iou"], "center_target_ids": [1] * row["input_center_target_count"]},
                cluster_stats_by_key[row["cluster_key"]],
                row["page_stats"],
                row.get("page_size"),
            )
        )
        row["listwise_features"] = {
            key: value
            for key, value in row.items()
            if key
            in {
                "score",
                "pre_selector_score",
                "label_id",
                "is_mask_v28",
                "is_center_branch_v30",
                "cluster_id_mod_17",
                "cluster_size",
                "cluster_score_max",
                "cluster_score_mean",
                "cluster_source_center_count",
                "cluster_mask_count",
                "page_candidate_count",
                "width_norm",
                "height_norm",
                "area_norm",
                "aspect",
                "area_bucket_small",
                "area_bucket_large",
            }
        }
        row["labels"] = {
            "keep": bool(row["target_keep"]),
            "drop": bool(row["target_drop"]),
            "suppression_reason": row["target_suppression_reason"],
            "positive": bool(row["target_keep"]),
            "duplicate_support": row["target_bucket"] in {"same_cluster_duplicate", "center_only_no_iou", "support_negative"},
            "center_only_no_iou": row["target_bucket"] == "center_only_no_iou",
            "same_cluster_duplicate": row["target_bucket"] == "same_cluster_duplicate",
            "source_specific_false_positive": row["target_bucket"] == "source_specific_false_positive",
            "low_score_background": row["target_bucket"] == "low_score_background",
        }
    manifest = {
        "version": "symbol_support_suppression_v32_dataset",
        "metric_mode": "smoke",
        "claim_boundary": "Offline listwise dataset for symbol support suppression. Runtime policy consumes raster-derived candidate fields only.",
        "counts": dict(counts),
        "bucket_counts": dict(bucket_counts),
        "by_source": dict(by_source),
        "by_area": dict(by_area),
        "by_page": dict(by_page),
    }
    return rows, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--support-negative-audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--center-targets", type=Path, default=DEFAULT_CENTER_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_path = args.cache if args.cache.is_absolute() else ROOT / args.cache
    center_targets_path = args.center_targets if args.center_targets.is_absolute() else ROOT / args.center_targets
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    rows, manifest = build_rows(load_jsonl(cache_path), load_page_meta(center_targets_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "listwise_rows.jsonl"
    write_jsonl(path, rows)
    manifest.update(
        {
            "inputs": {
                "cache": rel(cache_path),
                "center_targets": rel(center_targets_path),
                "support_negative_audit": rel(args.support_negative_audit if args.support_negative_audit.is_absolute() else ROOT / args.support_negative_audit),
            },
            "outputs": {
                "rows": rel(path),
            },
            "source_integrity": {
                "model_input": "raster-derived candidate bbox/score/source/type fields",
                "offline_gold_used_as_training_label": True,
                "svg_or_parser_geometry_used_at_runtime": False,
                "expected_json_used_at_runtime": False,
            },
        }
    )
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps({"manifest": rel(manifest_path), "counts": manifest["counts"], "bucket_counts": manifest["bucket_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
