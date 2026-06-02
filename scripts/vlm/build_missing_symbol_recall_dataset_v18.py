#!/usr/bin/env python3
"""Build an offline audit/training set for missing symbol recall.

This script intentionally reads gold labels only after detector/topology outputs
exist. It does not write gold-derived candidates back into an inference stream.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/vlm"
OUT = ROOT / "datasets/image_only_missing_symbol_recall_v18"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_topology_relations_v18 import bbox, center, center_covered, contains_point, integrity, iou, load_gold, write_json  # noqa: E402
from diagnose_contains_symbol_candidate_granularity_v18 import gold_centers_inside  # noqa: E402
from diagnose_contains_symbol_missing_gold_v18 import best_match, candidate_groups, gold_contains_symbols  # noqa: E402
from nms_topology_relations_v18 import load_by_id  # noqa: E402

DEFAULT_DATASET = REPORT / "contains_symbol_bipartite_dataset_symbol_proposal_combined.jsonl"
DEFAULT_ADAPTER = REPORT / "detector_adapter_v18_symbol_proposal_combined.jsonl"
DEFAULT_OUTPUT = OUT / "locked.jsonl"
DEFAULT_AUDIT = REPORT / "missing_symbol_recall_dataset_v18_audit.json"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def recoverable_keys_from_dataset(path: Path) -> set[str]:
    keys: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            labels = (json.loads(line).get("labels") or {})
            key = labels.get("gold_key") if isinstance(labels, dict) else None
            if key:
                keys.add(str(key))
    return keys


def box_area(box: list[float] | None) -> float:
    if box is None:
        return 0.0
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def box_size(box: list[float]) -> tuple[float, float]:
    return max(0.0, box[2] - box[0]), max(0.0, box[3] - box[1])


def expanded_box(box: list[float], factor: float, image_size: tuple[int, int]) -> list[int]:
    width, height = image_size
    cx, cy = center(box)
    bw, bh = box_size(box)
    nw = max(4.0, bw * factor)
    nh = max(4.0, bh * factor)
    x1 = max(0, int(math.floor(cx - nw / 2.0)))
    y1 = max(0, int(math.floor(cy - nh / 2.0)))
    x2 = min(width, int(math.ceil(cx + nw / 2.0)))
    y2 = min(height, int(math.ceil(cy + nh / 2.0)))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return [x1, y1, x2, y2]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_gray(path: str | Path) -> np.ndarray | None:
    image_path = Path(path)
    if not image_path.is_absolute():
        image_path = ROOT / image_path
    if not image_path.exists():
        return None
    with Image.open(image_path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def crop_stats(gray: np.ndarray | None, box: list[float] | None) -> dict[str, float]:
    if gray is None or box is None:
        return {
            "crop_dark_density_205": 0.0,
            "crop_dark_density_225": 0.0,
            "crop_mean_gray": 0.0,
            "crop_std_gray": 0.0,
            "crop_edge_touch_dark_ratio": 0.0,
        }
    height, width = gray.shape
    x1 = max(0, min(width - 1, int(math.floor(box[0]))))
    y1 = max(0, min(height - 1, int(math.floor(box[1]))))
    x2 = max(x1 + 1, min(width, int(math.ceil(box[2]))))
    y2 = max(y1 + 1, min(height, int(math.ceil(box[3]))))
    crop = gray[y1:y2, x1:x2]
    if crop.size == 0:
        return crop_stats(None, None)
    border = np.concatenate([crop[0, :], crop[-1, :], crop[:, 0], crop[:, -1]])
    return {
        "crop_dark_density_205": round(float((crop <= 205).mean()), 6),
        "crop_dark_density_225": round(float((crop <= 225).mean()), 6),
        "crop_mean_gray": round(float(crop.mean()), 6),
        "crop_std_gray": round(float(crop.std()), 6),
        "crop_edge_touch_dark_ratio": round(float((border <= 205).mean()), 6) if border.size else 0.0,
    }


def center_distance(left: list[float] | None, right: list[float] | None) -> float:
    if left is None or right is None:
        return 999999.0
    lx, ly = center(left)
    rx, ry = center(right)
    return math.hypot(lx - rx, ly - ry)


def candidate_kind(candidate: dict[str, Any] | None) -> str:
    payload = candidate.get("payload") if isinstance((candidate or {}).get("payload"), dict) else {}
    return str(payload.get("candidate_kind") or payload.get("proposal_source") or (candidate or {}).get("candidate_type") or "unknown")


def nearest_candidates(
    gold_box: list[float],
    candidates: list[dict[str, Any]],
    limit: int = 5,
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for cand in candidates:
        cb = bbox(cand.get("bbox"))
        if cb is None:
            continue
        ranked.append(
            {
                "candidate_id": cand.get("candidate_id"),
                "bbox": cb,
                "confidence": safe_float(cand.get("confidence")),
                "kind": candidate_kind(cand),
                "iou": round(iou(cb, gold_box), 6),
                "center_distance": round(center_distance(cb, gold_box), 6),
                "area_ratio_to_gold": round(box_area(cb) / max(box_area(gold_box), 1e-6), 6),
            }
        )
    ranked.sort(key=lambda item: (item["center_distance"], -item["iou"]))
    return ranked[:limit]


def symbol_candidate_summary(
    gold_box: list[float],
    groups: dict[str, list[dict[str, Any]]],
    row_gold_symbols: list[dict[str, Any]],
) -> dict[str, Any]:
    symbol_candidates = groups.get("symbol", [])
    best = best_match(gold_box, symbol_candidates, threshold=0.25)
    best_candidate = next((cand for cand in symbol_candidates if cand.get("candidate_id") == best.get("candidate_id")), None)
    covered_ids = gold_centers_inside(best_candidate, row_gold_symbols) if best_candidate else []
    return {
        "candidate_count": len(symbol_candidates),
        "best_match": best,
        "best_candidate_kind": candidate_kind(best_candidate),
        "best_candidate_area_ratio": round(box_area(bbox((best_candidate or {}).get("bbox"))) / max(box_area(gold_box), 1e-6), 6),
        "nearest": nearest_candidates(gold_box, symbol_candidates),
        "best_candidate_gold_centers_inside_count": len(covered_ids),
    }


def row_image_info(adapter: dict[str, Any]) -> tuple[str, tuple[int, int]]:
    image = str(adapter.get("image") or "")
    size = adapter.get("image_size")
    if isinstance(size, list) and len(size) >= 2:
        return image, (int(size[0]), int(size[1]))
    return image, (512, 512)


def proposal_boxes(gold_box: list[float], image_size: tuple[int, int]) -> list[tuple[str, list[int]]]:
    return [
        ("tight_gold_offline_positive", expanded_box(gold_box, 1.0, image_size)),
        ("padded_gold_offline_positive", expanded_box(gold_box, 1.8, image_size)),
        ("context_gold_offline_positive", expanded_box(gold_box, 3.0, image_size)),
    ]


def negative_boxes(
    gold_item: dict[str, Any],
    groups: dict[str, list[dict[str, Any]]],
    image_size: tuple[int, int],
    max_negatives: int,
) -> list[tuple[str, list[float], dict[str, Any]]]:
    gold_box = gold_item["symbol_bbox"]
    gold_center = center(gold_box)
    rows: list[tuple[str, list[float], dict[str, Any]]] = []
    for cand in groups.get("symbol", []):
        cb = bbox(cand.get("bbox"))
        if cb is None:
            continue
        match_score = max(iou(cb, gold_box), 0.5 if center_covered(cb, gold_box) else 0.0)
        if match_score >= 0.25:
            continue
        dist = center_distance(cb, gold_box)
        if dist > 120.0:
            continue
        rows.append(
            (
                "nearby_symbol_false_positive",
                cb,
                {
                    "source_candidate_id": cand.get("candidate_id"),
                    "source_candidate_confidence": safe_float(cand.get("confidence")),
                    "source_candidate_kind": candidate_kind(cand),
                    "distance_to_gold_center": round(dist, 6),
                },
            )
        )
    rows.sort(key=lambda item: item[2]["distance_to_gold_center"])
    selected = rows[:max_negatives]
    width, height = image_size
    gw, gh = box_size(gold_box)
    offsets = [(max(gw * 2.0, 10.0), 0.0), (-max(gw * 2.0, 10.0), 0.0), (0.0, max(gh * 2.0, 10.0)), (0.0, -max(gh * 2.0, 10.0))]
    for dx, dy in offsets:
        if len(selected) >= max_negatives:
            break
        cx = min(max(gold_center[0] + dx, 0.0), width - 1.0)
        cy = min(max(gold_center[1] + dy, 0.0), height - 1.0)
        box = expanded_box([cx - gw / 2.0, cy - gh / 2.0, cx + gw / 2.0, cy + gh / 2.0], 1.4, image_size)
        if iou(box, gold_box) < 0.05:
            selected.append(("nearby_background_hard_negative", box, {"distance_to_gold_center": round(math.hypot(dx, dy), 6)}))
    return selected


def build_records(
    recoverable: set[str],
    adapter_by_id: dict[str, dict[str, Any]],
    gold: dict[str, dict[str, Any]],
    max_negatives_per_positive: int,
    max_rows: int | None,
    dataset_path: str,
    adapter_path: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gold_rows = gold_contains_symbols(gold)
    gold_key_set = {str(row["gold_key"]) for row in gold_rows}
    missing_gold = [row for row in gold_rows if str(row["gold_key"]) not in recoverable]
    records: list[dict[str, Any]] = []
    counts = Counter()
    symbol_type_counts: dict[str, Counter[str]] = defaultdict(Counter)
    size_buckets = Counter()
    image_cache: dict[str, np.ndarray | None] = {}
    examples: list[dict[str, Any]] = []

    for gold_item in missing_gold:
        if max_rows is not None and counts["positive_gold_keys"] >= max_rows:
            break
        row_id = str(gold_item["row_id"])
        adapter = adapter_by_id.get(row_id)
        if not adapter:
            counts["missing_adapter_row"] += 1
            continue
        groups = candidate_groups(adapter)
        symbol_best = best_match(gold_item["symbol_bbox"], groups.get("symbol", []), threshold=0.25)
        if symbol_best.get("passes_match_threshold"):
            counts["not_missing_symbol_candidate"] += 1
            continue

        image, image_size = row_image_info(adapter)
        if image not in image_cache:
            image_cache[image] = load_gray(image)
        gray = image_cache[image]
        symbol_type = str(gold_item.get("symbol_type") or "symbol")
        gw, gh = box_size(gold_item["symbol_bbox"])
        area = gw * gh
        if area <= 16:
            size_bucket = "tiny_area_le_16"
        elif area <= 64:
            size_bucket = "small_area_17_64"
        elif area <= 256:
            size_bucket = "medium_area_65_256"
        else:
            size_bucket = "large_area_gt_256"
        size_buckets[size_bucket] += 1
        counts["positive_gold_keys"] += 1
        symbol_type_counts[symbol_type]["positive_gold_keys"] += 1

        base_features = {
            "gold_symbol_width": round(gw, 6),
            "gold_symbol_height": round(gh, 6),
            "gold_symbol_area": round(area, 6),
            "gold_symbol_aspect": round(gw / max(gh, 1e-6), 6),
            "best_existing_symbol_match_score": safe_float(symbol_best.get("match_score")),
            "existing_symbol_candidate_count": len(groups.get("symbol", [])),
            "space_candidate_count": len(groups.get("space", [])),
            "boundary_candidate_count": len(groups.get("boundary", [])),
            "text_candidate_count": len(groups.get("text", [])),
        }
        summary = symbol_candidate_summary(gold_item["symbol_bbox"], groups, gold["symbols"].get(row_id) or [])
        if len(examples) < 100:
            examples.append(
                {
                    "gold": gold_item,
                    "image": image,
                    "size_bucket": size_bucket,
                    "symbol_candidate_summary": summary,
                }
            )

        for kind, proposal in proposal_boxes(gold_item["symbol_bbox"], image_size):
            records.append(
                {
                    "id": f"{row_id}|{gold_item['gold_symbol_id']}|{kind}",
                    "split": "locked",
                    "row_id": row_id,
                    "image": image,
                    "image_size": list(image_size),
                    "bbox": proposal,
                    "candidate_family": "symbol",
                    "proposal_kind": kind,
                    "label_objectness": True,
                    "gold_key": gold_item["gold_key"],
                    "gold_symbol_id": gold_item["gold_symbol_id"],
                    "gold_room_id": gold_item["gold_room_id"],
                    "gold_symbol_type": symbol_type,
                    "features": {**base_features, **crop_stats(gray, proposal)},
                    "offline_label_scope": "training_or_locked_diagnosis_only",
                    "source_integrity": integrity(),
                }
            )
            counts["positive_rows"] += 1

        for neg_index, (kind, neg_box, neg_meta) in enumerate(negative_boxes(gold_item, groups, image_size, max_negatives_per_positive)):
            records.append(
                {
                    "id": f"{row_id}|{gold_item['gold_symbol_id']}|negative_{neg_index}",
                    "split": "locked",
                    "row_id": row_id,
                    "image": image,
                    "image_size": list(image_size),
                    "bbox": [round(float(v), 6) for v in neg_box],
                    "candidate_family": "symbol",
                    "proposal_kind": kind,
                    "label_objectness": False,
                    "gold_key": gold_item["gold_key"],
                    "gold_symbol_id": gold_item["gold_symbol_id"],
                    "gold_room_id": gold_item["gold_room_id"],
                    "gold_symbol_type": symbol_type,
                    "features": {**base_features, **crop_stats(gray, neg_box), **neg_meta},
                    "offline_label_scope": "hard_negative_for_training_only",
                    "source_integrity": integrity(),
                }
            )
            counts["negative_rows"] += 1
            symbol_type_counts[symbol_type]["negative_rows"] += 1

    audit = {
        "task": "IMG-MOE-V18-REBUILD-001.step_j_missing_symbol_recall_dataset",
        "input_dataset": dataset_path,
        "input_adapter": adapter_path,
        "gold_total": len(gold_rows),
        "dataset_recoverable_gold": len(recoverable & gold_key_set),
        "missing_gold": len(missing_gold),
        "missing_symbol_positive_gold_keys_exported": counts["positive_gold_keys"],
        "record_counts": dict(counts),
        "record_total": len(records),
        "symbol_type_counts": {key: dict(value) for key, value in sorted(symbol_type_counts.items())},
        "size_buckets": dict(size_buckets),
        "examples": examples,
        "source_integrity": integrity(),
        "gold_loaded_after_inference_for_training_and_diagnosis_only": True,
        "gold_used_for_inference": False,
    }
    return records, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--audit-output", default=str(DEFAULT_AUDIT))
    parser.add_argument("--max-negatives-per-positive", type=int, default=3)
    parser.add_argument("--max-positive-gold-keys", type=int, default=None)
    args = parser.parse_args()

    recoverable = recoverable_keys_from_dataset(Path(args.dataset))
    adapter_by_id = load_by_id(Path(args.adapter))
    records, audit = build_records(
        recoverable=recoverable,
        adapter_by_id=adapter_by_id,
        gold=load_gold(),
        max_negatives_per_positive=max(0, args.max_negatives_per_positive),
        max_rows=args.max_positive_gold_keys,
        dataset_path=str(args.dataset),
        adapter_path=str(args.adapter),
    )
    audit["output"] = str(args.output)
    write_jsonl(Path(args.output), records)
    write_json(
        Path(args.output).parent / "manifest.json",
        {
            "task": audit["task"],
            "dataset": str(Path(args.output).parent),
            "splits": {"locked": len(records)},
            "record_counts": audit["record_counts"],
            "symbol_type_counts": audit["symbol_type_counts"],
            "source_integrity": audit["source_integrity"],
            "gold_loaded_after_inference_for_training_and_diagnosis_only": True,
            "gold_used_for_inference": False,
        },
    )
    write_json(Path(args.audit_output), audit)
    print(
        json.dumps(
            {
                "record_total": audit["record_total"],
                "positive_gold_keys": audit["missing_symbol_positive_gold_keys_exported"],
                "positive_rows": audit["record_counts"].get("positive_rows", 0),
                "negative_rows": audit["record_counts"].get("negative_rows", 0),
                "top_symbol_types": Counter({k: v.get("positive_gold_keys", 0) for k, v in audit["symbol_type_counts"].items()}).most_common(10),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
