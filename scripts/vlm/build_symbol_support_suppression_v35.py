#!/usr/bin/env python3
"""Build v35 source-aware symbol suppression rows from the union cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from train_symbol_tile_detector_v20 import area_bucket, bbox_iou, rel, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = ROOT / "reports/vlm/symbol_proposal_eval_v35_union_smoke_cache.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "datasets/symbol_support_suppression_v35"
SOURCES = ["mask_v28", "center_branch_v30", "pretrained_tiny_v35"]
LABELS = ["appliance", "bathtub", "column", "equipment", "generic_symbol", "shower", "sink", "stair", "table"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def valid_box(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        box = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    return box if box[2] > box[0] and box[3] > box[1] else None


def center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


def split_for_page(page_id: str) -> str:
    value = int(hashlib.sha1(page_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if value < 70:
        return "train"
    if value < 85:
        return "dev"
    return "smoke_eval"


def match_map(row: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for item in row.get("candidate_gold_matches") or []:
        index = int(item.get("candidate_index", -1))
        out[index] = {
            "best_iou": float(item.get("best_iou", 0.0) or 0.0),
            "best_iou_target_id": item.get("best_iou_target_id"),
            "center_target_ids": [str(v) for v in item.get("center_target_ids") or []],
        }
    return out


def cluster_candidates(preds: list[dict[str, Any]]) -> list[int]:
    boxes = [valid_box(pred.get("bbox")) for pred in preds]
    parent = list(range(len(preds)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, box_i in enumerate(boxes):
        if box_i is None:
            continue
        ci = center(box_i)
        wi, hi = box_i[2] - box_i[0], box_i[3] - box_i[1]
        for j in range(i + 1, len(boxes)):
            box_j = boxes[j]
            if box_j is None:
                continue
            cj = center(box_j)
            wj, hj = box_j[2] - box_j[0], box_j[3] - box_j[1]
            center_dist = math.hypot(ci[0] - cj[0], ci[1] - cj[1])
            radius = max(8.0, 0.35 * max(wi, hi, wj, hj))
            if bbox_iou(box_i, box_j) >= 0.25 or center_dist <= radius:
                union(i, j)
    cluster_ids: dict[int, int] = {}
    out: list[int] = []
    for index in range(len(preds)):
        root = find(index)
        if root not in cluster_ids:
            cluster_ids[root] = len(cluster_ids)
        out.append(cluster_ids[root])
    return out


def feature_row(
    pred: dict[str, Any],
    box: list[float],
    match: dict[str, Any],
    cluster_size: int,
    source_rank: int,
    page_candidate_count: int,
) -> dict[str, float]:
    width = box[2] - box[0]
    height = box[3] - box[1]
    area = width * height
    score = float(pred.get("selector_score", pred.get("score", 0.0)) or 0.0)
    feats: dict[str, float] = {
        "score": score,
        "score_logit_safe": math.log(max(score, 1e-6) / max(1.0 - score, 1e-6)),
        "width": width,
        "height": height,
        "area": area,
        "log_area": math.log1p(area),
        "aspect": max(width, height) / max(min(width, height), 1e-6),
        "cluster_size": float(cluster_size),
        "source_rank": float(source_rank),
        "page_candidate_count": float(page_candidate_count),
        "center_match_count_audit_train_only": float(len(match.get("center_target_ids") or [])),
    }
    source = str(pred.get("proposal_source") or "unknown")
    for name in SOURCES:
        feats[f"source_is_{name}"] = 1.0 if source == name else 0.0
    label = str(pred.get("label") or "generic_symbol")
    for name in LABELS:
        feats[f"label_is_{name}"] = 1.0 if label == name else 0.0
    bucket = area_bucket(box)
    for name in ["tiny_le_64", "small_le_256", "medium_le_1024", "large_le_4096", "xlarge_gt_4096"]:
        feats[f"area_is_{name}"] = 1.0 if bucket == name else 0.0
    return feats


def build_rows(cache_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counts = Counter()
    for page in cache_rows:
        page_id = str(page["row_id"])
        split = split_for_page(page_id)
        preds = list(page.get("predicted_symbols") or [])
        page_gold_targets = [
            {
                "target_id": str(gold.get("target_id") or gold.get("id") or ""),
                "label": str(gold.get("label") or "generic_symbol"),
                "area_bucket": str(gold.get("area_bucket") or area_bucket([float(v) for v in gold.get("bbox", [0, 0, 1, 1])])),
            }
            for gold in page.get("gold_symbols") or []
            if str(gold.get("target_id") or gold.get("id") or "")
        ]
        matches = match_map(page)
        cluster_ids = cluster_candidates(preds)
        cluster_sizes = Counter(cluster_ids)
        best_positive_for_gold: dict[str, tuple[float, int]] = {}
        source_order = defaultdict(int)
        for index, pred in enumerate(preds):
            source_order[str(pred.get("proposal_source") or "unknown")] += 1
            match = matches.get(index, {})
            target = match.get("best_iou_target_id")
            best_iou = float(match.get("best_iou", 0.0) or 0.0)
            if target and best_iou >= 0.30:
                prev = best_positive_for_gold.get(str(target))
                if prev is None or best_iou > prev[0]:
                    best_positive_for_gold[str(target)] = (best_iou, index)
        keep_indices = {index for _target, (_iou, index) in best_positive_for_gold.items()}
        source_seen = defaultdict(int)
        for index, pred in enumerate(preds):
            box = valid_box(pred.get("bbox"))
            if box is None:
                continue
            source = str(pred.get("proposal_source") or "unknown")
            source_seen[source] += 1
            match = matches.get(index, {})
            best_iou = float(match.get("best_iou", 0.0) or 0.0)
            center_ids = match.get("center_target_ids") or []
            keep = index in keep_indices
            if keep:
                reason = "best_iou_positive"
            elif center_ids and best_iou < 0.30:
                reason = "center_only_no_iou"
            elif best_iou >= 0.30:
                reason = "same_gold_duplicate"
            elif cluster_sizes[cluster_ids[index]] > 1:
                reason = "same_cluster_duplicate"
            else:
                reason = "source_or_background_negative"
            rows.append(
                {
                    "page_id": page_id,
                    "split": split,
                    "candidate_id": f"{page_id}:{index}",
                    "candidate_index": index,
                    "cluster_id": int(cluster_ids[index]),
                    "bbox": box,
                    "label": str(pred.get("label") or "generic_symbol"),
                    "proposal_source": source,
                    "score": float(pred.get("selector_score", pred.get("score", 0.0)) or 0.0),
                    "features": feature_row(pred, box, match, cluster_sizes[cluster_ids[index]], source_seen[source], len(preds)),
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
                        "gold_fields_for_training_and_eval_only": ["labels.best_iou", "labels.best_iou_target_id", "labels.center_target_ids", "labels.keep"],
                        "gold_used_for_inference": False,
                    },
                }
            )
            counts[f"split:{split}"] += 1
            counts[f"source:{source}"] += 1
            counts[f"label:{str(pred.get('label') or 'generic_symbol')}"] += 1
            counts[f"reason:{reason}"] += 1
            if keep:
                counts["positive"] += 1
            else:
                counts["negative"] += 1
        counts["pages"] += 1
        counts["gold_symbols"] += len(page.get("gold_symbols") or [])
    return rows, dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    cache_path = args.cache if args.cache.is_absolute() else ROOT / args.cache
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, counts = build_rows(load_jsonl(cache_path))
    rows_path = output_dir / "listwise_rows.jsonl"
    manifest_path = output_dir / "manifest.json"
    write_jsonl(rows_path, rows)
    manifest = {
        "version": "symbol_support_suppression_v35",
        "task": "P1-04-v35-aware-source-calibrated-set-policy",
        "inputs": {"cache": rel(cache_path)},
        "outputs": {"rows": rel(rows_path)},
        "counts": counts,
        "source_integrity": {
            "runtime_input_allowed": ["raster-derived candidate bbox", "candidate score", "proposal source", "predicted type"],
            "offline_labels_used_for": ["training", "evaluation", "audit"],
            "gold_used_for_inference": False,
        },
    }
    write_json(manifest_path, manifest)
    print(json.dumps({"manifest": rel(manifest_path), "counts": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
