#!/usr/bin/env python3
"""Train/audit a room-wall-band seed policy for missing raster symbols.

This step works below window generation: it scores connected components as
symbol-body seeds using raster geometry plus detector-output context. Gold is
used only offline to label/evaluate seeds.
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
CHECKPOINT = ROOT / "checkpoints/room_wall_band_symbol_seed_policy_v18"
DEFAULT_DATASET = REPORT / "contains_symbol_bipartite_dataset_symbol_proposal_combined.jsonl"
DEFAULT_ADAPTER = REPORT / "detector_adapter_v18_symbol_proposal_combined.jsonl"
DEFAULT_MODEL = CHECKPOINT / "model.json"
DEFAULT_AUDIT = REPORT / "room_wall_band_symbol_seed_policy_v18_audit.json"
DEFAULT_SCORED = REPORT / "room_wall_band_symbol_seed_policy_v18_scored.jsonl"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_topology_relations_v18 import bbox, center, center_covered, contains_point, integrity, iou, load_gold, write_json  # noqa: E402
from diagnose_contains_symbol_missing_gold_v18 import best_match, candidate_groups, gold_contains_symbols  # noqa: E402
from generate_symbol_recall_candidates_v18 import crop_stats, image_array  # noqa: E402
from nms_topology_relations_v18 import load_by_id, load_jsonl  # noqa: E402
from train_missing_symbol_recall_expert_v18 import auc, sigmoid, stable_bucket, threshold_metrics, write_jsonl  # noqa: E402

FEATURES = [
    "component_width_norm",
    "component_height_norm",
    "component_area_norm",
    "component_aspect_log",
    "component_fill",
    "component_center_x_norm",
    "component_center_y_norm",
    "component_border_distance_norm",
    "crop_dark_density_205",
    "crop_dark_density_225",
    "crop_mean_gray_norm",
    "crop_std_gray_norm",
    "crop_edge_touch_dark_ratio",
    "inside_any_room",
    "nearest_room_distance_norm",
    "room_edge_distance_norm",
    "nearest_boundary_distance_norm",
    "max_boundary_iou",
    "max_text_iou",
    "nearest_text_distance_norm",
    "symbol_candidate_count_log",
    "space_candidate_count_log",
    "boundary_candidate_count_log",
    "text_candidate_count_log",
]


def recovered_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    for row in load_jsonl(path):
        labels = row.get("labels") if isinstance(row.get("labels"), dict) else {}
        key = labels.get("gold_key")
        if key:
            keys.add(str(key))
    return keys


def resolve_image(path: str | None) -> Path | None:
    if not path:
        return None
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def load_gray(row: dict[str, Any], cache: dict[str, np.ndarray]) -> np.ndarray | None:
    path = resolve_image(str(row.get("image") or ""))
    if path is None or not path.exists():
        return None
    key = str(path)
    if key not in cache:
        cache[key] = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    return cache[key]


def box_area(box: list[float] | None) -> float:
    if box is None:
        return 0.0
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def box_distance(left: list[float], right: list[float]) -> float:
    lx, ly = center(left)
    dx = max(right[0] - lx, 0.0, lx - right[2])
    dy = max(right[1] - ly, 0.0, ly - right[3])
    return math.hypot(dx, dy)


def center_distance(left: list[float], right: list[float]) -> float:
    lx, ly = center(left)
    rx, ry = center(right)
    return math.hypot(lx - rx, ly - ry)


def nearest_distance(box: list[float], boxes: list[list[float]]) -> float:
    if not boxes:
        return 999999.0
    return min(box_distance(box, other) for other in boxes)


def max_iou(box: list[float], boxes: list[list[float]]) -> float:
    return max((iou(box, other) for other in boxes), default=0.0)


def room_features(box: list[float], room_boxes: list[list[float]], image_diag: float) -> dict[str, float]:
    cx, cy = center(box)
    containing = [room for room in room_boxes if contains_point(room, cx, cy, margin=2.0)]
    nearest = nearest_distance(box, room_boxes)
    if containing:
        edge_dist = min(min(abs(cx - room[0]), abs(cx - room[2]), abs(cy - room[1]), abs(cy - room[3])) for room in containing)
    else:
        edge_dist = nearest
    return {
        "inside_any_room": 1.0 if containing else 0.0,
        "nearest_room_distance_norm": min(nearest, image_diag) / max(image_diag, 1e-6),
        "room_edge_distance_norm": min(edge_dist, image_diag) / max(image_diag, 1e-6),
    }


def component_boxes(arr: np.ndarray, thresholds: list[int], args: argparse.Namespace) -> list[dict[str, Any]]:
    import cv2

    out: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, int, int, int]] = set()
    group_kernels = [int(v.strip()) for v in str(args.group_kernels).split(",") if v.strip()]
    for threshold in thresholds:
        base_mask = (arr <= threshold).astype("uint8")
        for kernel_size in group_kernels:
            if kernel_size <= 1:
                mask = base_mask
                max_area = args.max_component_area
                max_side = args.max_side
                seed_kind = "raw_component"
            else:
                kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
                mask = cv2.morphologyEx(base_mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.dilate(mask, kernel, iterations=1)
                max_area = args.max_group_component_area
                max_side = args.max_group_side
                seed_kind = f"grouped_component_k{kernel_size}"
            n_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
            for idx in range(1, int(n_labels)):
                x, y, w, h, area = [int(v) for v in stats[idx]]
                if area < args.min_component_area or area > max_area:
                    continue
                if w < args.min_side or h < args.min_side or w > max_side or h > max_side:
                    continue
                aspect = w / max(h, 1)
                if aspect < args.min_aspect or aspect > args.max_aspect:
                    continue
                fill = area / max(w * h, 1)
                if fill < args.min_fill:
                    continue
                key = (x, y, x + w, y + h, int(threshold), int(kernel_size))
                if key in seen:
                    continue
                seen.add(key)
                cx, cy = [float(v) for v in centroids[idx]]
                out.append(
                    {
                        "bbox": [float(x), float(y), float(x + w), float(y + h)],
                        "component_center": [round(cx, 3), round(cy, 3)],
                        "component_area": int(area),
                        "component_fill": round(float(fill), 6),
                        "threshold": int(threshold),
                        "seed_kind": seed_kind,
                        "group_kernel": int(kernel_size),
                    }
                )
    return out


def row_image_size(row: dict[str, Any], arr: np.ndarray | None) -> tuple[int, int]:
    size = row.get("image_size")
    if isinstance(size, list) and len(size) >= 2:
        return int(size[0]), int(size[1])
    if arr is not None:
        return int(arr.shape[1]), int(arr.shape[0])
    return 512, 512


def inference_features(
    row: dict[str, Any],
    arr: np.ndarray,
    component: dict[str, Any],
    groups: dict[str, list[dict[str, Any]]],
) -> dict[str, float]:
    image_w, image_h = row_image_size(row, arr)
    image_diag = math.hypot(image_w, image_h)
    box = component["bbox"]
    width = max(0.0, box[2] - box[0])
    height = max(0.0, box[3] - box[1])
    cx, cy = center(box)
    border = min(cx, cy, image_w - cx, image_h - cy)
    room_boxes = [b for cand in groups.get("space", []) if (b := bbox(cand.get("bbox"))) is not None]
    boundary_boxes = [b for cand in groups.get("boundary", []) if (b := bbox(cand.get("bbox"))) is not None]
    text_boxes = [b for cand in groups.get("text", []) if (b := bbox(cand.get("bbox"))) is not None]
    stats = crop_stats(arr, box)
    return {
        "component_width_norm": width / max(image_w, 1),
        "component_height_norm": height / max(image_h, 1),
        "component_area_norm": box_area(box) / max(image_w * image_h, 1),
        "component_aspect_log": math.log(width / max(height, 1e-6)),
        "component_fill": float(component.get("component_fill") or 0.0),
        "component_center_x_norm": cx / max(image_w, 1),
        "component_center_y_norm": cy / max(image_h, 1),
        "component_border_distance_norm": max(0.0, border) / max(min(image_w, image_h), 1),
        "crop_dark_density_205": float(stats.get("crop_dark_density_205") or 0.0),
        "crop_dark_density_225": float(stats.get("crop_dark_density_225") or 0.0),
        "crop_mean_gray_norm": float(stats.get("crop_mean_gray") or 0.0) / 255.0,
        "crop_std_gray_norm": float(stats.get("crop_std_gray") or 0.0) / 128.0,
        "crop_edge_touch_dark_ratio": float(stats.get("crop_edge_touch_dark_ratio") or 0.0),
        **room_features(box, room_boxes, image_diag),
        "nearest_boundary_distance_norm": min(nearest_distance(box, boundary_boxes), image_diag) / max(image_diag, 1e-6),
        "max_boundary_iou": max_iou(box, boundary_boxes),
        "max_text_iou": max_iou(box, text_boxes),
        "nearest_text_distance_norm": min(nearest_distance(box, text_boxes), image_diag) / max(image_diag, 1e-6),
        "symbol_candidate_count_log": math.log1p(len(groups.get("symbol", []))),
        "space_candidate_count_log": math.log1p(len(groups.get("space", []))),
        "boundary_candidate_count_log": math.log1p(len(groups.get("boundary", []))),
        "text_candidate_count_log": math.log1p(len(groups.get("text", []))),
    }


def is_seed_positive(component_box: list[float], gold_box: list[float]) -> bool:
    gcx, gcy = center(gold_box)
    ccx, ccy = center(component_box)
    gw = max(1.0, gold_box[2] - gold_box[0])
    gh = max(1.0, gold_box[3] - gold_box[1])
    expanded = [gold_box[0] - 2.0, gold_box[1] - 2.0, gold_box[2] + 2.0, gold_box[3] + 2.0]
    return (
        contains_point(expanded, ccx, ccy)
        or contains_point(component_box, gcx, gcy, margin=2.0)
        or iou(component_box, gold_box) >= 0.03
        or center_distance(component_box, gold_box) <= max(3.0, 0.5 * max(gw, gh))
    )


def target_missing_gold(
    dataset_path: Path,
    adapter_by_id: dict[str, dict[str, Any]],
    gold: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    recoverable = recovered_keys(dataset_path)
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in gold_contains_symbols(gold):
        if str(item["gold_key"]) in recoverable:
            continue
        adapter = adapter_by_id.get(str(item["row_id"]))
        if not adapter:
            continue
        groups = candidate_groups(adapter)
        symbol_best = best_match(item["symbol_bbox"], groups.get("symbol", []), threshold=0.25)
        if symbol_best.get("passes_match_threshold"):
            continue
        out[str(item["row_id"])].append(item)
    return out


def build_seed_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    adapter_by_id = load_by_id(Path(args.adapter))
    missing_by_row = target_missing_gold(Path(args.dataset), adapter_by_id, load_gold())
    image_cache: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    counts = Counter()
    positive_by_type: Counter[str] = Counter()
    positive_key_hits: dict[str, set[str]] = defaultdict(set)
    thresholds = [int(v.strip()) for v in str(args.thresholds).split(",") if v.strip()]

    for row_id, gold_items in sorted(missing_by_row.items()):
        row = adapter_by_id.get(row_id)
        if not row:
            continue
        arr = load_gray(row, image_cache)
        if arr is None:
            counts["missing_image_rows"] += 1
            continue
        groups = candidate_groups(row)
        components = component_boxes(arr, thresholds, args)
        counts["rows_with_missing_gold"] += 1
        counts["target_gold_keys"] += len(gold_items)
        counts["raw_components"] += len(components)
        for index, component in enumerate(components):
            comp_box = component["bbox"]
            matched = [gold for gold in gold_items if is_seed_positive(comp_box, gold["symbol_bbox"])]
            label = bool(matched)
            if label:
                counts["positive_seed_rows"] += 1
                for gold in matched:
                    positive_by_type[str(gold.get("symbol_type") or "symbol")] += 1
                    positive_key_hits[str(gold["gold_key"])].add(f"{row_id}|seed_{index}")
            else:
                counts["negative_seed_rows"] += 1
            features = inference_features(row, arr, component, groups)
            records.append(
                {
                    "id": f"{row_id}|seed_{index:05d}|t{component['threshold']}",
                    "row_id": row_id,
                    "image": row.get("image"),
                    "image_size": row.get("image_size"),
                    "bbox": [round(float(v), 6) for v in comp_box],
                    "threshold": component["threshold"],
                    "component_area": component["component_area"],
                    "component_fill": component["component_fill"],
                    "seed_kind": component.get("seed_kind"),
                    "group_kernel": component.get("group_kernel"),
                    "label_objectness": label,
                    "gold_keys": [str(gold["gold_key"]) for gold in matched],
                    "gold_symbol_types": [str(gold.get("symbol_type") or "symbol") for gold in matched],
                    "features": {key: round(float(value), 8) for key, value in features.items()},
                    "offline_label_scope": "training_or_locked_diagnosis_only",
                    "source_integrity": integrity(),
                }
            )

    audit = {
        "task": "IMG-MOE-V18-REBUILD-001.step_j5_build_room_wall_band_symbol_seed_dataset",
        "dataset": str(args.dataset),
        "adapter": str(args.adapter),
        "record_total": len(records),
        "counts": dict(counts),
        "positive_seed_gold_key_coverage": len(positive_key_hits),
        "positive_seed_gold_key_recall": round(len(positive_key_hits) / max(counts["target_gold_keys"], 1), 6),
        "positive_seed_rows_by_symbol_type": dict(positive_by_type),
        "thresholds": thresholds,
        "group_kernels": [int(v.strip()) for v in str(args.group_kernels).split(",") if v.strip()],
        "source_integrity": integrity(),
        "gold_loaded_after_inference_for_training_and_diagnosis_only": True,
        "gold_used_for_inference": False,
    }
    return records, audit


def vector(row: dict[str, Any]) -> np.ndarray:
    features = row.get("features") if isinstance(row.get("features"), dict) else {}
    return np.asarray([float(features.get(name) or 0.0) for name in FEATURES], dtype=np.float32)


def train_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [vector(row) for row in rows if row.get("label_objectness")]
    negatives = [vector(row) for row in rows if not row.get("label_objectness")]
    pos = np.stack(positives) if positives else np.zeros((1, len(FEATURES)), dtype=np.float32)
    neg = np.stack(negatives) if negatives else np.zeros((1, len(FEATURES)), dtype=np.float32)
    all_x = np.concatenate([pos, neg], axis=0)
    mean = all_x.mean(axis=0)
    std = all_x.std(axis=0) + 1e-6
    pos_z = (pos - mean) / std
    neg_z = (neg - mean) / std
    weights = pos_z.mean(axis=0) - neg_z.mean(axis=0)
    norm = float(np.linalg.norm(weights))
    if norm > 1e-9:
        weights = weights / norm
    pos_scores = pos_z @ weights
    neg_scores = neg_z @ weights
    bias = -min(float(np.percentile(pos_scores, 2)), float(np.percentile(neg_scores, 90)))
    return {
        "model_type": "room_wall_band_symbol_seed_centroid_ranker",
        "features": FEATURES,
        "mean": [float(v) for v in mean.tolist()],
        "std": [float(v) for v in std.tolist()],
        "weights": [float(v) for v in weights.tolist()],
        "bias": float(bias),
        "train_counts": {"rows": len(rows), "positive": len(positives), "negative": len(negatives)},
        "source_integrity": integrity(),
    }


def score(row: dict[str, Any], model: dict[str, Any]) -> float:
    x = vector(row)
    mean = np.asarray(model["mean"], dtype=np.float32)
    std = np.asarray(model["std"], dtype=np.float32)
    weights = np.asarray(model["weights"], dtype=np.float32)
    return sigmoid(float(((x - mean) / std) @ weights + float(model.get("bias") or 0.0)))


def split_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for row in rows:
        key = str(row.get("row_id") or row.get("id"))
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
        if int(digest[:8], 16) % 10 < 8:
            train.append(row)
        else:
            eval_rows.append(row)
    return train, eval_rows


def seed_recall_at_caps(scored: list[dict[str, Any]], caps: list[int]) -> dict[str, Any]:
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    target_keys_by_row: dict[str, set[str]] = defaultdict(set)
    for row in scored:
        row_id = str(row.get("row_id"))
        rows[row_id].append(row)
        for key in row.get("all_target_gold_keys") or []:
            target_keys_by_row[row_id].add(str(key))
    out: dict[str, Any] = {}
    for cap in caps:
        hit: set[str] = set()
        total: set[str] = set()
        selected_rows = 0
        for row_id, items in rows.items():
            total.update(target_keys_by_row[row_id])
            selected = sorted(items, key=lambda item: float(item.get("seed_score") or 0.0), reverse=True)[:cap]
            selected_rows += len(selected)
            for item in selected:
                for key in item.get("gold_keys") or []:
                    hit.add(str(key))
        out[str(cap)] = {
            "selected_seed_rows": selected_rows,
            "target_gold_keys": len(total),
            "hit_gold_keys": len(hit),
            "seed_recall": round(len(hit) / max(len(total), 1), 6),
            "selected_seed_rows_per_hit_gold": round(selected_rows / max(len(hit), 1), 6),
        }
    return out


def attach_all_targets(rows: list[dict[str, Any]]) -> None:
    targets_by_row: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        for key in row.get("gold_keys") or []:
            targets_by_row[str(row.get("row_id"))].add(str(key))
    for row in rows:
        row["all_target_gold_keys"] = sorted(targets_by_row[str(row.get("row_id"))])


def feature_weights(model: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"feature": name, "weight": round(float(weight), 6)}
        for name, weight in sorted(zip(FEATURES, model["weights"], strict=True), key=lambda item: abs(float(item[1])), reverse=True)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--model-output", default=str(DEFAULT_MODEL))
    parser.add_argument("--audit-output", default=str(DEFAULT_AUDIT))
    parser.add_argument("--scored-output", default=str(DEFAULT_SCORED))
    parser.add_argument("--thresholds", default="185,205,225")
    parser.add_argument("--min-component-area", type=int, default=1)
    parser.add_argument("--max-component-area", type=int, default=260)
    parser.add_argument("--min-side", type=int, default=1)
    parser.add_argument("--max-side", type=int, default=44)
    parser.add_argument("--min-aspect", type=float, default=0.08)
    parser.add_argument("--max-aspect", type=float, default=12.0)
    parser.add_argument("--min-fill", type=float, default=0.02)
    parser.add_argument("--group-kernels", default="1,3,5,9")
    parser.add_argument("--max-group-component-area", type=int, default=1800)
    parser.add_argument("--max-group-side", type=int, default=96)
    args = parser.parse_args()

    rows, dataset_audit = build_seed_rows(args)
    attach_all_targets(rows)
    train_rows, eval_rows = split_rows(rows)
    model = train_model(train_rows)
    scored: list[dict[str, Any]] = []
    for row in eval_rows:
        item = {
            "id": row.get("id"),
            "row_id": row.get("row_id"),
            "bbox": row.get("bbox"),
            "threshold": row.get("threshold"),
            "component_area": row.get("component_area"),
            "component_fill": row.get("component_fill"),
            "seed_kind": row.get("seed_kind"),
            "group_kernel": row.get("group_kernel"),
            "label_objectness": bool(row.get("label_objectness")),
            "gold_keys": row.get("gold_keys"),
            "gold_symbol_types": row.get("gold_symbol_types"),
            "all_target_gold_keys": row.get("all_target_gold_keys"),
            "seed_score": round(score(row, model), 6),
            "features": row.get("features"),
            "source_integrity": integrity(),
        }
        scored.append(item)
    thresholds = sorted({round(float(row["seed_score"]), 6) for row in scored})
    policy_sweep = [threshold_metrics([{"objectness_score": r["seed_score"], "label_objectness": r["label_objectness"]} for r in scored], t) for t in thresholds]
    feasible = [row for row in policy_sweep if row["recall"] >= 0.98]
    selected = sorted(feasible, key=lambda row: (row["candidate_reduction"], row["precision"]), reverse=True)[0] if feasible else sorted(policy_sweep, key=lambda row: (row["recall"], row["precision"]), reverse=True)[0]
    audit = {
        "task": "IMG-MOE-V18-REBUILD-001.step_j5_train_room_wall_band_symbol_seed_policy_before_more_window_expansion",
        "dataset_build": dataset_audit,
        "model_output": str(args.model_output),
        "scored_output": str(args.scored_output),
        "split": {
            "strategy": "stable_hash_by_row_id_80_20",
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "train_positive": sum(1 for row in train_rows if row.get("label_objectness")),
            "eval_positive": sum(1 for row in eval_rows if row.get("label_objectness")),
        },
        "eval": {
            "seed_auc": auc([{"objectness_score": r["seed_score"], "label_objectness": r["label_objectness"]} for r in scored]),
            "selected_policy": selected,
            "seed_recall_at_caps": seed_recall_at_caps(scored, [16, 32, 64, 128, 256, 500]),
            "top_feature_weights": feature_weights(model)[:16],
        },
        "adoption_decision": {
            "adopted_into_inference_stream": False,
            "reason": "This is a seed-level diagnostic. Adoption requires using the seed policy inside a generator and proving recoverable contains_symbol gain after topology replay.",
            "next_required_step": "Generate symbol windows only from top room-wall-band seeds, then compare recoverable gold and row cost against targeted_symbol_family_proposal_v18.",
        },
        "quality_gates": {
            "source_integrity_violations": 0,
            "gold_used_for_inference": False,
            "eval_seed_recall_at_128_ge_0_90": seed_recall_at_caps(scored, [128])["128"]["seed_recall"] >= 0.90,
            "ready_for_topology_adoption": False,
        },
        "source_integrity": integrity(),
        "gold_loaded_after_inference_for_training_and_diagnosis_only": True,
        "gold_used_for_inference": False,
    }
    model["selected_threshold"] = float(selected["threshold"])
    model["adopted_into_inference_stream"] = False
    write_json(Path(args.model_output), model)
    write_json(Path(args.audit_output), audit)
    write_jsonl(Path(args.scored_output), scored)
    print(
        json.dumps(
            {
                "records": dataset_audit["record_total"],
                "target_gold_keys": dataset_audit["counts"].get("target_gold_keys", 0),
                "positive_seed_gold_key_recall": dataset_audit["positive_seed_gold_key_recall"],
                "seed_auc": audit["eval"]["seed_auc"],
                "selected_policy": selected,
                "seed_recall_at_caps": audit["eval"]["seed_recall_at_caps"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
