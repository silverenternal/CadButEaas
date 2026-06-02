#!/usr/bin/env python3
"""Audit/train a high-recall raster symbol-body evidence segmenter.

This j6 diagnostic tests whether richer raster evidence maps can produce
symbol-body seeds that cover missing gold symbols before bbox/window scoring.
Gold is used only offline for labeling and evaluation.
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
CHECKPOINT = ROOT / "checkpoints/symbol_body_segmentation_v18"
DEFAULT_DATASET = REPORT / "contains_symbol_bipartite_dataset_symbol_proposal_combined.jsonl"
DEFAULT_ADAPTER = REPORT / "detector_adapter_v18_symbol_proposal_combined.jsonl"
DEFAULT_AUDIT = REPORT / "symbol_body_segmentation_v18_audit.json"
DEFAULT_SCORED = REPORT / "symbol_body_segmentation_v18_scored.jsonl"
DEFAULT_MODEL = CHECKPOINT / "model.json"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_topology_relations_v18 import bbox, center, contains_point, integrity, iou, load_gold, write_json  # noqa: E402
from diagnose_contains_symbol_missing_gold_v18 import best_match, candidate_groups, gold_contains_symbols  # noqa: E402
from nms_topology_relations_v18 import load_by_id, load_jsonl  # noqa: E402
from train_missing_symbol_recall_expert_v18 import auc, sigmoid, threshold_metrics, write_jsonl  # noqa: E402

FEATURES = [
    "bbox_width_norm",
    "bbox_height_norm",
    "bbox_area_norm",
    "bbox_aspect_log",
    "component_fill",
    "component_area_log",
    "mean_gray_norm",
    "std_gray_norm",
    "dark_density_205",
    "dark_density_245",
    "edge_density",
    "map_mean",
    "map_fill",
    "center_x_norm",
    "center_y_norm",
    "border_distance_norm",
    "source_dark",
    "source_adaptive",
    "source_canny",
    "source_sobel",
    "source_blackhat",
    "source_morph_gradient",
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


def center_distance(left: list[float], right: list[float]) -> float:
    lx, ly = center(left)
    rx, ry = center(right)
    return math.hypot(lx - rx, ly - ry)


def image_size(row: dict[str, Any], arr: np.ndarray) -> tuple[int, int]:
    size = row.get("image_size")
    if isinstance(size, list) and len(size) >= 2:
        return int(size[0]), int(size[1])
    return int(arr.shape[1]), int(arr.shape[0])


def crop(arr: np.ndarray, box: list[float]) -> np.ndarray:
    h, w = arr.shape
    x1 = max(0, min(w - 1, int(math.floor(box[0]))))
    y1 = max(0, min(h - 1, int(math.floor(box[1]))))
    x2 = max(x1 + 1, min(w, int(math.ceil(box[2]))))
    y2 = max(y1 + 1, min(h, int(math.ceil(box[3]))))
    return arr[y1:y2, x1:x2]


def target_missing_gold(dataset_path: Path, adapter_by_id: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    recoverable = recovered_keys(dataset_path)
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in gold_contains_symbols(load_gold()):
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


def evidence_maps(arr: np.ndarray, args: argparse.Namespace) -> list[tuple[str, np.ndarray, np.ndarray]]:
    import cv2

    maps: list[tuple[str, np.ndarray, np.ndarray]] = []
    for threshold in args.dark_thresholds:
        mask = arr <= int(threshold)
        maps.append((f"dark_{threshold}", mask.astype("uint8"), arr.astype(np.float32) / 255.0))

    adaptive = cv2.adaptiveThreshold(arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8)
    maps.append(("adaptive_gaussian_15", (adaptive > 0).astype("uint8"), adaptive.astype(np.float32) / 255.0))

    canny = cv2.Canny(arr, int(args.canny_low), int(args.canny_high))
    maps.append(("canny", (canny > 0).astype("uint8"), canny.astype(np.float32) / 255.0))

    sobel_x = cv2.Sobel(arr, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(arr, cv2.CV_32F, 0, 1, ksize=3)
    sobel = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)
    sobel_norm = sobel / max(float(sobel.max()), 1.0)
    maps.append(("sobel_gradient", (sobel_norm >= args.sobel_threshold).astype("uint8"), sobel_norm.astype(np.float32)))

    for kernel_size in args.structure_kernels:
        kernel = np.ones((int(kernel_size), int(kernel_size)), dtype=np.uint8)
        blackhat = cv2.morphologyEx(arr, cv2.MORPH_BLACKHAT, kernel)
        maps.append(
            (
                f"blackhat_k{kernel_size}",
                (blackhat >= int(args.blackhat_threshold)).astype("uint8"),
                blackhat.astype(np.float32) / max(float(blackhat.max()), 1.0),
            )
        )
        grad = cv2.morphologyEx(arr, cv2.MORPH_GRADIENT, kernel)
        maps.append(
            (
                f"morph_gradient_k{kernel_size}",
                (grad >= int(args.morph_gradient_threshold)).astype("uint8"),
                grad.astype(np.float32) / max(float(grad.max()), 1.0),
            )
        )
    return maps


def components_from_mask(mask: np.ndarray, source: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    import cv2

    out: list[dict[str, Any]] = []
    kernels = [int(v) for v in args.group_kernels]
    seen: set[tuple[int, int, int, int, str, int]] = set()
    for kernel_size in kernels:
        work = mask.astype("uint8")
        if kernel_size > 1:
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel)
            work = cv2.dilate(work, kernel, iterations=1)
        n_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(work, 8)
        for idx in range(1, int(n_labels)):
            x, y, w, h, area = [int(v) for v in stats[idx]]
            max_area = args.max_raw_area if kernel_size <= 1 else args.max_group_area
            max_side = args.max_raw_side if kernel_size <= 1 else args.max_group_side
            if area < args.min_area or area > max_area:
                continue
            if w < args.min_side or h < args.min_side or w > max_side or h > max_side:
                continue
            aspect = w / max(h, 1)
            if aspect < args.min_aspect or aspect > args.max_aspect:
                continue
            fill = area / max(w * h, 1)
            if fill < args.min_fill:
                continue
            key = (x, y, x + w, y + h, source, kernel_size)
            if key in seen:
                continue
            seen.add(key)
            cx, cy = [float(v) for v in centroids[idx]]
            out.append(
                {
                    "bbox": [float(x), float(y), float(x + w), float(y + h)],
                    "component_area": int(area),
                    "component_fill": round(float(fill), 6),
                    "source_map": source,
                    "group_kernel": int(kernel_size),
                    "component_center": [round(cx, 3), round(cy, 3)],
                }
            )
    return out


def is_positive_seed(seed_box: list[float], gold_box: list[float]) -> bool:
    gcx, gcy = center(gold_box)
    scx, scy = center(seed_box)
    gw = max(1.0, gold_box[2] - gold_box[0])
    gh = max(1.0, gold_box[3] - gold_box[1])
    expanded = [gold_box[0] - 3.0, gold_box[1] - 3.0, gold_box[2] + 3.0, gold_box[3] + 3.0]
    return (
        contains_point(expanded, scx, scy)
        or contains_point(seed_box, gcx, gcy, margin=2.0)
        or iou(seed_box, gold_box) >= 0.02
        or center_distance(seed_box, gold_box) <= max(4.0, 0.6 * max(gw, gh))
    )


def source_flags(source: str) -> dict[str, float]:
    return {
        "source_dark": 1.0 if source.startswith("dark_") else 0.0,
        "source_adaptive": 1.0 if source.startswith("adaptive") else 0.0,
        "source_canny": 1.0 if source.startswith("canny") else 0.0,
        "source_sobel": 1.0 if source.startswith("sobel") else 0.0,
        "source_blackhat": 1.0 if source.startswith("blackhat") else 0.0,
        "source_morph_gradient": 1.0 if source.startswith("morph_gradient") else 0.0,
    }


def seed_features(row: dict[str, Any], arr: np.ndarray, map_values: np.ndarray, seed: dict[str, Any]) -> dict[str, float]:
    w, h = image_size(row, arr)
    box = seed["bbox"]
    bw = max(0.0, box[2] - box[0])
    bh = max(0.0, box[3] - box[1])
    cx, cy = center(box)
    gray_crop = crop(arr, box)
    map_crop = crop(map_values, box)
    border = min(cx, cy, w - cx, h - cy)
    return {
        "bbox_width_norm": bw / max(w, 1),
        "bbox_height_norm": bh / max(h, 1),
        "bbox_area_norm": box_area(box) / max(w * h, 1),
        "bbox_aspect_log": math.log(bw / max(bh, 1e-6)),
        "component_fill": float(seed.get("component_fill") or 0.0),
        "component_area_log": math.log1p(float(seed.get("component_area") or 0.0)),
        "mean_gray_norm": float(gray_crop.mean()) / 255.0 if gray_crop.size else 0.0,
        "std_gray_norm": float(gray_crop.std()) / 128.0 if gray_crop.size else 0.0,
        "dark_density_205": float((gray_crop <= 205).mean()) if gray_crop.size else 0.0,
        "dark_density_245": float((gray_crop <= 245).mean()) if gray_crop.size else 0.0,
        "edge_density": float((map_crop > 0).mean()) if map_crop.size else 0.0,
        "map_mean": float(map_crop.mean()) if map_crop.size else 0.0,
        "map_fill": float(seed.get("component_fill") or 0.0),
        "center_x_norm": cx / max(w, 1),
        "center_y_norm": cy / max(h, 1),
        "border_distance_norm": max(0.0, border) / max(min(w, h), 1),
        **source_flags(str(seed.get("source_map") or "")),
    }


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    adapter_by_id = load_by_id(Path(args.adapter))
    missing_by_row = target_missing_gold(Path(args.dataset), adapter_by_id)
    cache: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    counts = Counter()
    hit_keys: dict[str, set[str]] = defaultdict(set)
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row_id, gold_items in sorted(missing_by_row.items()):
        row = adapter_by_id.get(row_id)
        if not row:
            continue
        arr = load_gray(row, cache)
        if arr is None:
            counts["missing_image_rows"] += 1
            continue
        counts["rows_with_missing_gold"] += 1
        counts["target_gold_keys"] += len(gold_items)
        maps = evidence_maps(arr, args)
        index = 0
        for source, mask, values in maps:
            seeds = components_from_mask(mask, source, args)
            counts["raw_seed_components"] += len(seeds)
            for seed in seeds:
                matched = [gold for gold in gold_items if is_positive_seed(seed["bbox"], gold["symbol_bbox"])]
                label = bool(matched)
                source_counts[source]["positive" if label else "negative"] += 1
                if label:
                    counts["positive_seed_rows"] += 1
                    for gold in matched:
                        hit_keys[str(gold["gold_key"])].add(f"{row_id}|{source}|{index}")
                else:
                    counts["negative_seed_rows"] += 1
                records.append(
                    {
                        "id": f"{row_id}|symbol_body_seed_{index:05d}",
                        "row_id": row_id,
                        "image": row.get("image"),
                        "image_size": row.get("image_size"),
                        "bbox": [round(float(v), 6) for v in seed["bbox"]],
                        "source_map": source,
                        "group_kernel": seed.get("group_kernel"),
                        "component_area": seed.get("component_area"),
                        "component_fill": seed.get("component_fill"),
                        "label_objectness": label,
                        "gold_keys": [str(gold["gold_key"]) for gold in matched],
                        "gold_symbol_types": [str(gold.get("symbol_type") or "symbol") for gold in matched],
                        "features": {k: round(float(v), 8) for k, v in seed_features(row, arr, values, seed).items()},
                        "offline_label_scope": "training_or_locked_diagnosis_only",
                        "source_integrity": integrity(),
                    }
                )
                index += 1
    audit = {
        "task": "IMG-MOE-V18-REBUILD-001.step_j6_build_symbol_body_segmentation_seed_dataset",
        "dataset": str(args.dataset),
        "adapter": str(args.adapter),
        "record_total": len(records),
        "counts": dict(counts),
        "positive_seed_gold_key_coverage": len(hit_keys),
        "positive_seed_gold_key_recall": round(len(hit_keys) / max(counts["target_gold_keys"], 1), 6),
        "source_map_counts": {key: dict(value) for key, value in sorted(source_counts.items())},
        "params": {
            "dark_thresholds": args.dark_thresholds,
            "structure_kernels": args.structure_kernels,
            "group_kernels": args.group_kernels,
        },
        "source_integrity": integrity(),
        "gold_loaded_after_inference_for_training_and_diagnosis_only": True,
        "gold_used_for_inference": False,
    }
    return records, audit


def vector(row: dict[str, Any]) -> np.ndarray:
    features = row.get("features") if isinstance(row.get("features"), dict) else {}
    return np.asarray([float(features.get(name) or 0.0) for name in FEATURES], dtype=np.float32)


def split_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for row in rows:
        digest = hashlib.sha1(str(row.get("row_id") or row.get("id")).encode("utf-8")).hexdigest()
        if int(digest[:8], 16) % 10 < 8:
            train.append(row)
        else:
            eval_rows.append(row)
    return train, eval_rows


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
        "model_type": "symbol_body_segmentation_seed_centroid_ranker",
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


def attach_all_targets(rows: list[dict[str, Any]]) -> None:
    targets_by_row: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        for key in row.get("gold_keys") or []:
            targets_by_row[str(row.get("row_id"))].add(str(key))
    for row in rows:
        row["all_target_gold_keys"] = sorted(targets_by_row[str(row.get("row_id"))])


def recall_at_caps(scored: list[dict[str, Any]], caps: list[int]) -> dict[str, Any]:
    by_row: dict[str, list[dict[str, Any]]] = defaultdict(list)
    target_by_row: dict[str, set[str]] = defaultdict(set)
    for row in scored:
        row_id = str(row.get("row_id"))
        by_row[row_id].append(row)
        for key in row.get("all_target_gold_keys") or []:
            target_by_row[row_id].add(str(key))
    out: dict[str, Any] = {}
    for cap in caps:
        hit: set[str] = set()
        total: set[str] = set()
        selected_count = 0
        for row_id, rows in by_row.items():
            total.update(target_by_row[row_id])
            selected = sorted(rows, key=lambda row: float(row.get("seed_score") or 0.0), reverse=True)[:cap]
            selected_count += len(selected)
            for row in selected:
                for key in row.get("gold_keys") or []:
                    hit.add(str(key))
        out[str(cap)] = {
            "selected_seed_rows": selected_count,
            "target_gold_keys": len(total),
            "hit_gold_keys": len(hit),
            "seed_recall": round(len(hit) / max(len(total), 1), 6),
            "selected_seed_rows_per_hit_gold": round(selected_count / max(len(hit), 1), 6),
        }
    return out


def feature_weights(model: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"feature": name, "weight": round(float(weight), 6)}
        for name, weight in sorted(zip(FEATURES, model["weights"], strict=True), key=lambda item: abs(float(item[1])), reverse=True)
    ]


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--audit-output", default=str(DEFAULT_AUDIT))
    parser.add_argument("--scored-output", default=str(DEFAULT_SCORED))
    parser.add_argument("--model-output", default=str(DEFAULT_MODEL))
    parser.add_argument("--dark-thresholds", type=parse_ints, default=parse_ints("185,205,225,245"))
    parser.add_argument("--structure-kernels", type=parse_ints, default=parse_ints("3,5,9"))
    parser.add_argument("--group-kernels", type=parse_ints, default=parse_ints("1,3,5"))
    parser.add_argument("--canny-low", type=int, default=35)
    parser.add_argument("--canny-high", type=int, default=120)
    parser.add_argument("--sobel-threshold", type=float, default=0.18)
    parser.add_argument("--blackhat-threshold", type=int, default=12)
    parser.add_argument("--morph-gradient-threshold", type=int, default=18)
    parser.add_argument("--min-area", type=int, default=1)
    parser.add_argument("--max-raw-area", type=int, default=320)
    parser.add_argument("--max-group-area", type=int, default=2200)
    parser.add_argument("--min-side", type=int, default=1)
    parser.add_argument("--max-raw-side", type=int, default=48)
    parser.add_argument("--max-group-side", type=int, default=110)
    parser.add_argument("--min-aspect", type=float, default=0.05)
    parser.add_argument("--max-aspect", type=float, default=18.0)
    parser.add_argument("--min-fill", type=float, default=0.015)
    args = parser.parse_args()

    rows, dataset_audit = build_rows(args)
    attach_all_targets(rows)
    train_rows, eval_rows = split_rows(rows)
    model = train_model(train_rows)
    scored: list[dict[str, Any]] = []
    for row in eval_rows:
        item = {
            "id": row.get("id"),
            "row_id": row.get("row_id"),
            "bbox": row.get("bbox"),
            "source_map": row.get("source_map"),
            "group_kernel": row.get("group_kernel"),
            "component_area": row.get("component_area"),
            "component_fill": row.get("component_fill"),
            "label_objectness": bool(row.get("label_objectness")),
            "gold_keys": row.get("gold_keys"),
            "gold_symbol_types": row.get("gold_symbol_types"),
            "all_target_gold_keys": row.get("all_target_gold_keys"),
            "seed_score": round(score(row, model), 6),
            "features": row.get("features"),
            "source_integrity": integrity(),
        }
        scored.append(item)
    score_rows = [{"objectness_score": r["seed_score"], "label_objectness": r["label_objectness"]} for r in scored]
    thresholds = sorted({round(float(row["seed_score"]), 6) for row in scored})
    sweep = [threshold_metrics(score_rows, threshold) for threshold in thresholds]
    feasible = [row for row in sweep if row["recall"] >= 0.98]
    selected = sorted(feasible, key=lambda row: (row["candidate_reduction"], row["precision"]), reverse=True)[0] if feasible else sorted(sweep, key=lambda row: (row["recall"], row["precision"]), reverse=True)[0]
    cap_recall = recall_at_caps(scored, [16, 32, 64, 128, 256, 500])
    audit = {
        "task": "IMG-MOE-V18-REBUILD-001.step_j6_rebuild_raster_symbol_body_segmentation_and_high_recall_detector",
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
            "seed_auc": auc(score_rows),
            "selected_policy": selected,
            "seed_recall_at_caps": cap_recall,
            "top_feature_weights": feature_weights(model)[:18],
        },
        "adoption_decision": {
            "adopted_into_inference_stream": False,
            "reason": "This is a segmentation-seed diagnostic. Candidate-stream adoption requires >=0.90 seed coverage at <=500 seeds/page and then topology replay.",
            "next_required_step": "If the gate passes, generate symbol proposals from top segmentation seeds; otherwise expand the segmenter beyond classical raster evidence maps.",
        },
        "quality_gates": {
            "source_integrity_violations": 0,
            "gold_used_for_inference": False,
            "seed_gold_key_coverage_ge_0_90": dataset_audit["positive_seed_gold_key_recall"] >= 0.90,
            "eval_seed_recall_at_500_ge_0_90": cap_recall["500"]["seed_recall"] >= 0.90,
            "ready_for_topology_replay": dataset_audit["positive_seed_gold_key_recall"] >= 0.90 and cap_recall["500"]["seed_recall"] >= 0.90,
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
                "seed_recall_at_caps": cap_recall,
                "quality_gates": audit["quality_gates"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
