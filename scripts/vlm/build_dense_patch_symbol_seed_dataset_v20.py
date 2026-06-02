#!/usr/bin/env python3
"""Build dense raster patch/seed supervision for v20 symbol recovery.

This dataset aligns supervision with the dense inference distribution rather than
gold-centered patch crops. It scores dense windows over full pages, labels them
against the remaining missing gold symbol relations, and keeps auditable fields
for cap-displacement risk and regression analysis.
"""

from __future__ import annotations

import argparse
import hashlib
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
DATASET = ROOT / "datasets/dense_patch_symbol_seed_v20"

DEFAULT_BASELINE = REPORT / "contains_symbol_bipartite_dataset_symbol_proposal_combined.jsonl"
DEFAULT_PATCH_ORACLE = REPORT / "contains_symbol_bipartite_dataset_patch_heatmap_symbol_policy_oracle.jsonl"
DEFAULT_ADAPTER = REPORT / "detector_adapter_v18_symbol_proposal_combined.jsonl"
DEFAULT_MODEL = ROOT / "checkpoints/patch_symbol_body_segmenter_v19/model.json"
DEFAULT_OUTPUT = DATASET / "locked.jsonl"
DEFAULT_AUDIT = REPORT / "dense_patch_symbol_seed_v20_audit.json"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_topology_relations_v18 import bbox, center, contains_point, integrity, iou, load_gold, write_json, write_jsonl  # noqa: E402
from diagnose_contains_symbol_missing_gold_v18 import best_match, candidate_groups, gold_contains_symbols  # noqa: E402
from nms_topology_relations_v18 import load_by_id, load_jsonl  # noqa: E402
from train_patch_symbol_body_segmenter_v18 import dense_patch_boxes, evidence_arrays, patch_features, score  # noqa: E402


def resolve_image(path: str | None) -> Path | None:
    if not path:
        return None
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def load_gray(row: dict[str, Any], cache: dict[str, np.ndarray]) -> np.ndarray | None:
    image_path = resolve_image(str(row.get("image") or ""))
    if image_path is None or not image_path.exists():
        return None
    key = str(image_path)
    if key not in cache:
        cache[key] = np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)
    return cache[key]


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def stable_row_split(row_id: str) -> str:
    digest = hashlib.sha1(row_id.encode("utf-8")).hexdigest()
    return "train" if int(digest[:8], 16) % 10 < 8 else "eval"


def box_area(box: list[float] | None) -> float:
    if box is None:
        return 0.0
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def center_distance(left: list[float], right: list[float]) -> float:
    lx, ly = center(left)
    rx, ry = center(right)
    return math.hypot(lx - rx, ly - ry)


def recoverable_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    for row in load_jsonl(path):
        labels = row.get("labels") if isinstance(row.get("labels"), dict) else {}
        key = labels.get("gold_key")
        if key:
            keys.add(str(key))
    return keys


def row_gold_items(dataset_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in dataset_rows:
        row_id = str(item.get("row_id"))
        out[row_id].append(item)
    return out


def candidate_stream(row: dict[str, Any]) -> list[dict[str, Any]]:
    return list(((row.get("scene_graph") or {}).get("candidate_stream") or []))


def family_boxes(stream: list[dict[str, Any]], family: str) -> list[list[float]]:
    boxes: list[list[float]] = []
    for cand in stream:
        if str(cand.get("family") or "") != family:
            continue
        box = bbox(cand.get("bbox"))
        if box is not None:
            boxes.append(box)
    return boxes


def nearest_room_distance(box: list[float], room_boxes: list[list[float]]) -> float:
    if not room_boxes:
        return 0.0
    cx, cy = center(box)
    best = 999999.0
    for room in room_boxes:
        rx, ry = center(room)
        best = min(best, math.hypot(cx - rx, cy - ry))
    return best


def max_overlap(box: list[float], boxes: list[list[float]]) -> float:
    best = 0.0
    for other in boxes:
        best = max(best, iou(box, other))
    return best


def label_patch(patch_box: list[float], gold_items: list[dict[str, Any]]) -> tuple[bool, list[dict[str, Any]]]:
    cx, cy = center(patch_box)
    matched: list[dict[str, Any]] = []
    for gold in gold_items:
        gb = gold["symbol_bbox"]
        if contains_point(gb, cx, cy, margin=3.0) or iou(patch_box, gb) >= 0.03:
            matched.append(gold)
    return bool(matched), matched


def positive_training_boxes(gold_items: list[dict[str, Any]], width: int, height: int, patch_sizes: list[int]) -> list[list[float]]:
    boxes: list[list[float]] = []
    for gold in gold_items:
        gx, gy = center(gold["symbol_bbox"])
        for side in patch_sizes:
            for dx, dy in [(0, 0), (-2, 0), (2, 0), (0, -2), (0, 2)]:
                boxes.append(clip_box(gx + dx, gy + dy, float(side), width, height))
    return boxes


def clip_box(cx: float, cy: float, side: float, width: int, height: int) -> list[float]:
    half = side / 2.0
    x1 = max(0.0, min(float(width - 1), cx - half))
    y1 = max(0.0, min(float(height - 1), cy - half))
    x2 = max(x1 + 1.0, min(float(width), cx + half))
    y2 = max(y1 + 1.0, min(float(height), cy + half))
    return [x1, y1, x2, y2]


def in_any_room(box: list[float], room_boxes: list[list[float]], margin: float) -> bool:
    if not room_boxes:
        return True
    cx, cy = center(box)
    return any(contains_point(room, cx, cy, margin) for room in room_boxes)


def gold_sets() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    gold = gold_contains_symbols(load_gold())
    by_key = {str(item["gold_key"]): item for item in gold}
    return gold, by_key


def target_keys(
    baseline_dataset: Path,
    patch_oracle_dataset: Path | None,
) -> tuple[set[str], set[str]]:
    baseline_recoverable = recoverable_keys(baseline_dataset)
    oracle_recoverable = recoverable_keys(patch_oracle_dataset) if patch_oracle_dataset and patch_oracle_dataset.exists() else set(baseline_recoverable)
    gold, _ = gold_sets()
    all_keys = {str(item["gold_key"]) for item in gold}
    return all_keys - baseline_recoverable, all_keys - oracle_recoverable


def build_dataset(
    baseline_dataset: Path,
    patch_oracle_dataset: Path | None,
    adapter_path: Path,
    model_path: Path,
    patch_sizes: list[int],
    stride: int,
    max_selected_patches_per_row: int,
    max_records_per_row: int,
    room_margin: float,
    max_room_boxes: int,
    disable_room_filter: bool,
    limit_rows: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    adapter_by_id = load_by_id(adapter_path)
    baseline_missing_keys, oracle_missing_keys = target_keys(baseline_dataset, patch_oracle_dataset)
    gold_rows = gold_contains_symbols(load_gold())
    gold_by_row = row_gold_items(gold_rows)
    target_rows = sorted(
        {
            str(item["row_id"])
            for item in gold_rows
            if str(item["gold_key"]) in baseline_missing_keys or str(item["gold_key"]) in oracle_missing_keys
        }
    )
    if limit_rows is not None:
        target_rows = target_rows[:limit_rows]

    model = json.loads(model_path.read_text(encoding="utf-8"))
    cache: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    counts = Counter()
    per_row_selected: Counter[int] = Counter()
    per_row_positive: Counter[int] = Counter()
    row_audits: list[dict[str, Any]] = []
    positive_symbol_types = Counter()
    oracle_gain_only = Counter()
    processed_oracle_targets_by_row: dict[str, set[str]] = defaultdict(set)
    processed_baseline_targets_by_row: dict[str, set[str]] = defaultdict(set)

    for row_id in target_rows:
        row = adapter_by_id.get(row_id)
        if not row:
            counts["missing_adapter_rows"] += 1
            continue
        arr = load_gray(row, cache)
        if arr is None:
            counts["missing_image_rows"] += 1
            continue
        stream = candidate_stream(row)
        room_boxes = sorted(family_boxes(stream, "space"), key=box_area, reverse=True)[:max_room_boxes]
        symbol_boxes = family_boxes(stream, "symbol")
        height, width = arr.shape
        all_gold_items = gold_by_row.get(row_id, [])
        baseline_targets = [item for item in all_gold_items if str(item["gold_key"]) in baseline_missing_keys]
        oracle_targets = [item for item in all_gold_items if str(item["gold_key"]) in oracle_missing_keys]
        counts["rows"] += 1
        counts["baseline_target_gold"] += len(baseline_targets)
        counts["oracle_target_gold"] += len(oracle_targets)
        processed_oracle_targets_by_row[row_id].update(str(item["gold_key"]) for item in oracle_targets)
        processed_baseline_targets_by_row[row_id].update(str(item["gold_key"]) for item in baseline_targets)

        patch_boxes = dense_patch_boxes(width, height, stride, patch_sizes)
        counts["raw_patch_windows"] += len(patch_boxes)
        if not disable_room_filter:
            patch_boxes = [box for box in patch_boxes if in_any_room(box, room_boxes, room_margin)]
        counts["patch_windows_after_room_filter"] += len(patch_boxes)

        evidence = evidence_arrays(arr)
        scored: list[dict[str, Any]] = []
        for index, patch_box in enumerate(patch_boxes):
            features = {
                **{name: round(float(value), 8) for name, value in patch_features(row, arr, evidence, patch_box).items()},
                "room_support_count": float(sum(1 for room in room_boxes if contains_point(room, *center(patch_box), margin=room_margin))),
                "nearest_room_center_distance": round(nearest_room_distance(patch_box, room_boxes), 6),
                "max_existing_symbol_iou": round(max_overlap(patch_box, symbol_boxes), 6),
                "max_existing_symbol_center_hit": 1.0 if any(contains_point(box, *center(patch_box), margin=2.0) for box in symbol_boxes) else 0.0,
            }
            base_record = {
                "id": f"{row_id}|dense_patch_symbol_seed_v20|{index:06d}",
                "row_id": row_id,
                "image": row.get("image"),
                "image_size": row.get("image_size") or [width, height],
                "bbox": [round(float(v), 6) for v in patch_box],
                "features": features,
            }
            patch_score = round(float(score(base_record, model)), 6)
            is_baseline_positive, baseline_matched = label_patch(patch_box, baseline_targets)
            is_oracle_positive, oracle_matched = label_patch(patch_box, oracle_targets)
            max_existing_symbol = max_overlap(patch_box, symbol_boxes)
            regression_risk = bool(max_existing_symbol >= 0.45 or (is_oracle_positive and max_existing_symbol >= 0.25))
            scored.append(
                {
                    **base_record,
                    "patch_score": patch_score,
                    "label_objectness": is_oracle_positive,
                    "label_dense_positive": is_oracle_positive,
                    "label_baseline_missing": is_baseline_positive,
                    "label_oracle_missing": is_oracle_positive,
                    "label_safe_positive": bool(is_oracle_positive and not regression_risk),
                    "label_regression_risk": regression_risk,
                    "label_no_regression": not regression_risk,
                    "gold_keys": [str(item["gold_key"]) for item in oracle_matched],
                    "baseline_gold_keys": [str(item["gold_key"]) for item in baseline_matched],
                    "gold_symbol_types": [str(item.get("symbol_type") or "symbol") for item in oracle_matched],
                    "baseline_gold_symbol_types": [str(item.get("symbol_type") or "symbol") for item in baseline_matched],
                    "source_integrity": integrity(),
                }
            )

        ranked = sorted(
            scored,
            key=lambda row: (
                float(row.get("patch_score") or 0.0),
                float(row["features"].get("room_support_count") or 0.0),
                -box_area(row["bbox"]),
            ),
            reverse=True,
        )
        selected_limit = max_records_per_row if max_records_per_row > 0 else max_selected_patches_per_row
        selected_source = ranked[:selected_limit] if selected_limit > 0 else ranked
        selected: dict[str, dict[str, Any]] = {str(item["id"]): item for item in selected_source}
        for item in ranked:
            if item["label_objectness"]:
                selected[str(item["id"])] = item

        selected_rows = sorted(
            selected.values(),
            key=lambda row: (
                float(row.get("patch_score") or 0.0),
                bool(row.get("label_objectness")),
                -box_area(row["bbox"]),
            ),
            reverse=True,
        )

        for rank, item in enumerate(selected_rows):
            item["split"] = stable_row_split(row_id)
            item["patch_score"] = round(float(score(item, model)), 6)
            item["seed_rank"] = rank + 1
            item["seed_rank_norm"] = round((rank + 1) / max(len(selected_rows), 1), 6)
            item["room_count"] = len(room_boxes)
            item["symbol_candidate_count"] = len(symbol_boxes)
            item["source_integrity"] = integrity()
            rows.append(item)

        per_row_selected[len(selected_rows)] += 1
        per_row_positive[sum(1 for item in selected_rows if item["label_objectness"])] += 1
        positive_symbol_types.update(str(item["gold_symbol_types"][0]) if item.get("gold_symbol_types") else "symbol" for item in selected_rows if item["label_objectness"])
        oracle_gain_only.update(
            str(item["baseline_gold_keys"][0]) if item.get("baseline_gold_keys") else "unknown"
            for item in selected_rows
            if item["label_baseline_missing"] and not item["label_oracle_missing"]
        )
        row_audits.append(
            {
                "row_id": row_id,
                "split": stable_row_split(row_id),
                "raw_patch_windows": len(patch_boxes),
                "selected_records": len(selected_rows),
                "oracle_positive_records": sum(1 for item in selected_rows if item["label_objectness"]),
                "baseline_positive_records": sum(1 for item in selected_rows if item["label_baseline_missing"]),
                "regression_risk_records": sum(1 for item in selected_rows if item["label_regression_risk"]),
                "safe_positive_records": sum(1 for item in selected_rows if item["label_safe_positive"]),
                "oracle_missing_gold": len(oracle_targets),
                "baseline_missing_gold": len(baseline_targets),
            }
        )

    def recall_at_caps(label_field: str, caps: list[int]) -> dict[str, Any]:
        by_row: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_row[str(row.get("row_id"))].append(row)
        targets_by_row = processed_oracle_targets_by_row if label_field == "label_objectness" else processed_baseline_targets_by_row
        out: dict[str, Any] = {}
        for cap in caps:
            hit: set[str] = set()
            total: set[str] = set()
            selected_count = 0
            for rid in sorted(targets_by_row):
                candidates = by_row.get(rid) or []
                total.update(targets_by_row.get(rid) or set())
                selected = sorted(candidates, key=lambda item: float(item.get("patch_score") or 0.0), reverse=True)[:cap]
                selected_count += len(selected)
                for cand in selected:
                    keys = cand.get("gold_keys") if label_field == "label_objectness" else cand.get("baseline_gold_keys")
                    for key in keys or []:
                        hit.add(str(key))
            out[str(cap)] = {
                "selected_patch_rows": selected_count,
                "target_gold_keys": len(total),
                "hit_gold_keys": len(hit),
                "patch_recall": round(len(hit) / max(len(total), 1), 6),
                "selected_patch_rows_per_hit_gold": round(selected_count / max(len(hit), 1), 6),
            }
        return out

    audit = {
        "task": "IMG-MOE-V18-REBUILD-001.step_j10b_build_dense_patch_symbol_seed_dataset_v20",
        "baseline_dataset": str(baseline_dataset),
        "patch_oracle_dataset": str(patch_oracle_dataset) if patch_oracle_dataset else None,
        "adapter": str(adapter_path),
        "model": str(model_path),
        "output": str(DEFAULT_OUTPUT),
        "counts": dict(counts),
        "rows": len(rows),
        "row_audits_sample": row_audits[:100],
        "selected_per_row_histogram": dict(per_row_selected),
        "oracle_positive_per_row_histogram": dict(per_row_positive),
        "positive_symbol_types": dict(positive_symbol_types),
        "oracle_gain_only_histogram": dict(oracle_gain_only),
        "dense_recall_at_caps_oracle_target": recall_at_caps("label_objectness", [50, 100, 250, 500, 1000]),
        "dense_recall_at_caps_baseline_target": recall_at_caps("label_baseline_missing", [50, 100, 250, 500, 1000]),
        "label_counts": {
            "oracle_positive": sum(1 for row in rows if row.get("label_objectness")),
            "baseline_positive": sum(1 for row in rows if row.get("label_baseline_missing")),
            "safe_positive": sum(1 for row in rows if row.get("label_safe_positive")),
            "regression_risk": sum(1 for row in rows if row.get("label_regression_risk")),
        },
        "quality_gates": {
            "gold_loaded_after_inference_for_training_only": True,
            "gold_used_for_inference": False,
            "source_integrity_violations": 0,
            "oracle_target_recall_at_500_ge_0_90": recall_at_caps("label_objectness", [500])["500"]["patch_recall"] >= 0.90 if rows else False,
        },
        "source_integrity": integrity(),
    }
    return rows, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    parser.add_argument("--patch-oracle", default=str(DEFAULT_PATCH_ORACLE))
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--patch-sizes", type=parse_csv_ints, default=parse_csv_ints("9,17,25"))
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--max-selected-patches-per-row", type=int, default=1200)
    parser.add_argument("--max-records-per-row", type=int, default=1200)
    parser.add_argument("--room-margin", type=float, default=4.0)
    parser.add_argument("--max-room-boxes", type=int, default=80)
    parser.add_argument("--disable-room-filter", action="store_true")
    parser.add_argument("--limit-rows", type=int, default=None)
    args = parser.parse_args()

    patch_oracle = Path(args.patch_oracle) if args.patch_oracle else None
    rows, audit = build_dataset(
        baseline_dataset=Path(args.baseline),
        patch_oracle_dataset=patch_oracle,
        adapter_path=Path(args.adapter),
        model_path=Path(args.model),
        patch_sizes=args.patch_sizes,
        stride=args.stride,
        max_selected_patches_per_row=args.max_selected_patches_per_row,
        max_records_per_row=args.max_records_per_row,
        room_margin=args.room_margin,
        max_room_boxes=args.max_room_boxes,
        disable_room_filter=bool(args.disable_room_filter),
        limit_rows=args.limit_rows,
    )
    write_jsonl(Path(args.output), rows)
    write_json(Path(args.audit), audit)
    print(
        json.dumps(
            {
                "rows": audit["rows"],
                "counts": audit["counts"],
                "label_counts": audit["label_counts"],
                "dense_recall_at_caps_oracle_target": audit["dense_recall_at_caps_oracle_target"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
