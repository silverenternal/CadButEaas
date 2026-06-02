#!/usr/bin/env python3
"""Mine patch-symbol hard cases for the next raster symbol-body detector.

Gold is used only offline to build training/audit records. The output dataset
contains positive symbol-body patches from still-unrecovered gold and current
patch true positives, plus hard negatives from high-score patch candidates that
do not recover canonical contains_symbol gold.
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

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/vlm"
DEFAULT_BASELINE = REPORT / "contains_symbol_bipartite_dataset_symbol_proposal_combined.jsonl"
DEFAULT_PATCH_DATASET = REPORT / "contains_symbol_bipartite_dataset_patch_heatmap_symbol_policy_oracle.jsonl"
DEFAULT_ADAPTER = REPORT / "detector_adapter_v18_patch_heatmap_symbol_policy_oracle.jsonl"
DEFAULT_OUTPUT = ROOT / "datasets/patch_symbol_hard_cases_v18/locked.jsonl"
DEFAULT_AUDIT = REPORT / "patch_symbol_hard_cases_v18_audit.json"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_topology_relations_v18 import bbox, center, integrity, iou, load_gold, write_json, write_jsonl  # noqa: E402
from generate_symbol_recall_candidates_v18 import image_array, parse_csv_ints  # noqa: E402
from nms_topology_relations_v18 import load_by_id, load_jsonl  # noqa: E402
from train_patch_symbol_body_segmenter_v18 import evidence_arrays, patch_features  # noqa: E402


def gold_key(row: dict[str, Any]) -> str | None:
    labels = row.get("labels") if isinstance(row.get("labels"), dict) else {}
    value = labels.get("gold_key")
    return str(value) if value else None


def recovered_keys(path: Path) -> set[str]:
    return {key for row in load_jsonl(path) if (key := gold_key(row))}


def gold_contains_symbols() -> list[dict[str, Any]]:
    gold = load_gold()
    out: list[dict[str, Any]] = []
    for row_id, symbols in gold["symbols"].items():
        rooms = (gold["rooms"].get(row_id) or {}).get("rooms") or []
        for symbol in symbols:
            sb = bbox(symbol.get("bbox"))
            if sb is None:
                continue
            sx, sy = center(sb)
            room_id = None
            for room in rooms:
                rb = bbox(room.get("bbox"))
                if rb and rb[0] - 2 <= sx <= rb[2] + 2 and rb[1] - 2 <= sy <= rb[3] + 2:
                    room_id = str(room.get("target_id"))
                    break
            if not room_id:
                continue
            symbol_id = str(symbol.get("target_id"))
            out.append(
                {
                    "row_id": str(row_id),
                    "room_id": room_id,
                    "symbol_id": symbol_id,
                    "gold_key": f"{row_id}|{room_id}|{symbol_id}",
                    "symbol_bbox": sb,
                    "symbol_type": str(symbol.get("type") or symbol.get("label") or "symbol"),
                }
            )
    return out


def clip_box(cx: float, cy: float, side: float, width: int, height: int) -> list[float]:
    half = side / 2.0
    return [
        max(0.0, min(float(width - 1), cx - half)),
        max(0.0, min(float(height - 1), cy - half)),
        max(1.0, min(float(width), cx + half)),
        max(1.0, min(float(height), cy + half)),
    ]


def image_size(row: dict[str, Any], arr: np.ndarray) -> list[int]:
    value = row.get("image_size")
    if isinstance(value, list) and len(value) == 2:
        return [int(value[0]), int(value[1])]
    return [int(arr.shape[1]), int(arr.shape[0])]


def candidate_stream(row: dict[str, Any]) -> list[dict[str, Any]]:
    return list(((row.get("scene_graph") or {}).get("candidate_stream") or []))


def patch_positive_candidate_ids(dataset_path: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for row in load_jsonl(dataset_path):
        cid = str(row.get("target_candidate_id") or "")
        key = gold_key(row)
        if key and "patch_heatmap_symbol_v18" in cid:
            out[cid].add(key)
    return out


def make_record(
    row: dict[str, Any],
    arr: np.ndarray,
    evidence: dict[str, np.ndarray],
    box: list[float],
    label: bool,
    record_id: str,
    source_bucket: str,
    gold_keys: list[str] | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": record_id,
        "row_id": row.get("id"),
        "image": row.get("image"),
        "image_size": image_size(row, arr),
        "bbox": [round(float(v), 6) for v in box],
        "label_objectness": bool(label),
        "gold_keys": gold_keys or [],
        "source_bucket": source_bucket,
        "features": {key: round(float(value), 8) for key, value in patch_features(row, arr, evidence, box).items()},
        "meta": meta or {},
        "offline_label_scope": "training_or_locked_diagnosis_only",
        "source_integrity": integrity(),
    }


def mine_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline_keys = recovered_keys(Path(args.baseline))
    patch_keys = recovered_keys(Path(args.patch_dataset))
    missing_after_patch = {item["gold_key"]: item for item in gold_contains_symbols() if item["gold_key"] not in patch_keys}
    adapter_by_id = load_by_id(Path(args.adapter))
    positive_patch_ids = patch_positive_candidate_ids(Path(args.patch_dataset))
    image_cache: dict[str, np.ndarray] = {}
    evidence_cache: dict[str, dict[str, np.ndarray]] = {}
    records: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    patch_sizes = [int(v) for v in args.patch_sizes]

    for gold_key_value, gold in sorted(missing_after_patch.items()):
        row = adapter_by_id.get(str(gold["row_id"]))
        if not row:
            counts["missing_adapter_for_gold"] += 1
            continue
        arr = image_array(row, image_cache)
        if arr is None:
            counts["missing_image_for_gold"] += 1
            continue
        evidence = evidence_cache.setdefault(str(row.get("image")), evidence_arrays(arr))
        gx, gy = center(gold["symbol_bbox"])
        for side in patch_sizes:
            for dx, dy in [(0, 0), (-2, 0), (2, 0), (0, -2), (0, 2)]:
                box = clip_box(gx + dx, gy + dy, float(side), arr.shape[1], arr.shape[0])
                records.append(
                    make_record(
                        row,
                        arr,
                        evidence,
                        box,
                        True,
                        f"{gold['row_id']}|missing_gold|{gold['symbol_id']}|{side}|{dx}_{dy}",
                        "positive_missing_after_patch_oracle",
                        [gold_key_value],
                        {"symbol_type": gold["symbol_type"], "symbol_bbox": gold["symbol_bbox"], "recovered_by_baseline": gold_key_value in baseline_keys},
                    )
                )
                counts["positive_missing_after_patch_oracle"] += 1

    for row in adapter_by_id.values():
        arr = image_array(row, image_cache)
        if arr is None:
            counts["missing_image_for_adapter"] += 1
            continue
        evidence = evidence_cache.setdefault(str(row.get("image")), evidence_arrays(arr))
        negatives_kept = 0
        positives_kept = 0
        for cand in candidate_stream(row):
            cid = str(cand.get("candidate_id") or "")
            if "patch_heatmap_symbol_v18" not in cid:
                continue
            cb = bbox(cand.get("bbox"))
            if cb is None:
                continue
            keys = sorted(positive_patch_ids.get(cid, set()))
            payload = cand.get("payload") if isinstance(cand.get("payload"), dict) else {}
            if keys and positives_kept < args.max_positive_patch_candidates_per_row:
                records.append(
                    make_record(
                        row,
                        arr,
                        evidence,
                        cb,
                        True,
                        f"{row.get('id')}|patch_tp|{cid}",
                        "positive_current_patch_candidate",
                        keys,
                        {"candidate_id": cid, "patch_score": payload.get("patch_score"), "cluster_size": payload.get("cluster_size")},
                    )
                )
                positives_kept += 1
                counts["positive_current_patch_candidate"] += 1
            elif not keys and negatives_kept < args.max_negative_patch_candidates_per_row:
                records.append(
                    make_record(
                        row,
                        arr,
                        evidence,
                        cb,
                        False,
                        f"{row.get('id')}|patch_fp|{cid}",
                        "negative_false_high_score_patch_candidate",
                        [],
                        {"candidate_id": cid, "patch_score": payload.get("patch_score"), "cluster_size": payload.get("cluster_size")},
                    )
                )
                negatives_kept += 1
                counts["negative_false_high_score_patch_candidate"] += 1

    audit = {
        "task": "IMG-MOE-V18-REBUILD-001.step_j10_mine_patch_symbol_hard_cases",
        "baseline": str(args.baseline),
        "patch_dataset": str(args.patch_dataset),
        "adapter": str(args.adapter),
        "output": str(args.output),
        "baseline_recoverable_keys": len(baseline_keys),
        "patch_recoverable_keys": len(patch_keys),
        "missing_gold_after_patch_oracle": len(missing_after_patch),
        "record_count": len(records),
        "record_counts": dict(counts),
        "patch_sizes": patch_sizes,
        "source_integrity": integrity(),
        "gold_loaded_after_inference_for_training_only": True,
        "gold_used_for_inference": False,
    }
    return records, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    parser.add_argument("--patch-dataset", default=str(DEFAULT_PATCH_DATASET))
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--patch-sizes", type=parse_csv_ints, default=parse_csv_ints("9,17,25"))
    parser.add_argument("--max-negative-patch-candidates-per-row", type=int, default=64)
    parser.add_argument("--max-positive-patch-candidates-per-row", type=int, default=64)
    args = parser.parse_args()
    rows, audit = mine_records(args)
    write_jsonl(Path(args.output), rows)
    write_json(Path(args.audit), audit)
    print(json.dumps({"records": len(rows), "record_counts": audit["record_counts"], "missing_gold_after_patch_oracle": audit["missing_gold_after_patch_oracle"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
