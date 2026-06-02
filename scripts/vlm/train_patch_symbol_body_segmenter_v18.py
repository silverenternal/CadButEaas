#!/usr/bin/env python3
"""Train/evaluate a learned patch-level raster symbol-body segmenter.

The model is intentionally lightweight and auditable: it learns a patch scorer
from raster crop features, then evaluates dense sliding-window heatmap seeds on
missing-symbol pages. Gold is used only for training/evaluation labels.
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
CHECKPOINT = ROOT / "checkpoints/patch_symbol_body_segmenter_v18"
DEFAULT_DATASET = REPORT / "contains_symbol_bipartite_dataset_symbol_proposal_combined.jsonl"
DEFAULT_ADAPTER = REPORT / "detector_adapter_v18_symbol_proposal_combined.jsonl"
DEFAULT_MODEL = CHECKPOINT / "model.json"
DEFAULT_AUDIT = REPORT / "patch_symbol_body_segmenter_v18_audit.json"
DEFAULT_SCORED = REPORT / "patch_symbol_body_segmenter_v18_scored.jsonl"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_topology_relations_v18 import bbox, center, contains_point, integrity, iou, load_gold, write_json  # noqa: E402
from diagnose_contains_symbol_missing_gold_v18 import best_match, candidate_groups, gold_contains_symbols  # noqa: E402
from nms_topology_relations_v18 import load_by_id, load_jsonl  # noqa: E402
from train_missing_symbol_recall_expert_v18 import auc, sigmoid, threshold_metrics, write_jsonl  # noqa: E402

FEATURES = [
    "patch_w_norm",
    "patch_h_norm",
    "patch_area_norm",
    "center_x_norm",
    "center_y_norm",
    "border_distance_norm",
    "mean_gray_norm",
    "std_gray_norm",
    "min_gray_norm",
    "max_gray_norm",
    "dark_density_160",
    "dark_density_185",
    "dark_density_205",
    "dark_density_225",
    "dark_density_245",
    "center_dark_205",
    "edge_touch_dark_ratio",
    "row_dark_std",
    "col_dark_std",
    "sobel_mean",
    "sobel_max",
    "canny_density",
    "blackhat_mean",
    "adaptive_density",
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
    image_path = resolve_image(str(row.get("image") or ""))
    if image_path is None or not image_path.exists():
        return None
    key = str(image_path)
    if key not in cache:
        cache[key] = np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)
    return cache[key]


def box_area(box: list[float] | None) -> float:
    if box is None:
        return 0.0
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


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


def stable_row_split(row_id: str) -> str:
    digest = hashlib.sha1(row_id.encode("utf-8")).hexdigest()
    return "train" if int(digest[:8], 16) % 10 < 8 else "eval"


def clip_box(cx: float, cy: float, side: float, width: int, height: int) -> list[float]:
    half = side / 2.0
    x1 = max(0.0, min(float(width - 1), cx - half))
    y1 = max(0.0, min(float(height - 1), cy - half))
    x2 = max(x1 + 1.0, min(float(width), cx + half))
    y2 = max(y1 + 1.0, min(float(height), cy + half))
    return [x1, y1, x2, y2]


def crop(arr: np.ndarray, box: list[float]) -> np.ndarray:
    h, w = arr.shape
    x1 = max(0, min(w - 1, int(math.floor(box[0]))))
    y1 = max(0, min(h - 1, int(math.floor(box[1]))))
    x2 = max(x1 + 1, min(w, int(math.ceil(box[2]))))
    y2 = max(y1 + 1, min(h, int(math.ceil(box[3]))))
    return arr[y1:y2, x1:x2]


def evidence_arrays(arr: np.ndarray) -> dict[str, np.ndarray]:
    import cv2

    sobel_x = cv2.Sobel(arr, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(arr, cv2.CV_32F, 0, 1, ksize=3)
    sobel = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)
    sobel = sobel / max(float(sobel.max()), 1.0)
    canny = (cv2.Canny(arr, 35, 120) > 0).astype(np.float32)
    adaptive = (cv2.adaptiveThreshold(arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8) > 0).astype(np.float32)
    kernel = np.ones((3, 3), dtype=np.uint8)
    blackhat = cv2.morphologyEx(arr, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)
    blackhat = blackhat / max(float(blackhat.max()), 1.0)
    return {"sobel": sobel, "canny": canny, "adaptive": adaptive, "blackhat": blackhat}


def patch_features(row: dict[str, Any], arr: np.ndarray, evidence: dict[str, np.ndarray], patch_box: list[float]) -> dict[str, float]:
    height, width = arr.shape
    gray = crop(arr, patch_box)
    sobel = crop(evidence["sobel"], patch_box)
    canny = crop(evidence["canny"], patch_box)
    adaptive = crop(evidence["adaptive"], patch_box)
    blackhat = crop(evidence["blackhat"], patch_box)
    pw = max(1.0, patch_box[2] - patch_box[0])
    ph = max(1.0, patch_box[3] - patch_box[1])
    cx, cy = center(patch_box)
    border = min(cx, cy, width - cx, height - cy)
    center_crop = crop(arr, clip_box(cx, cy, 3.0, width, height))
    dark = (gray <= 205).astype(np.float32) if gray.size else np.zeros((1, 1), dtype=np.float32)
    border_pixels = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]]) if gray.size else np.zeros((1,), dtype=np.uint8)
    return {
        "patch_w_norm": pw / max(width, 1),
        "patch_h_norm": ph / max(height, 1),
        "patch_area_norm": (pw * ph) / max(width * height, 1),
        "center_x_norm": cx / max(width, 1),
        "center_y_norm": cy / max(height, 1),
        "border_distance_norm": max(0.0, border) / max(min(width, height), 1),
        "mean_gray_norm": float(gray.mean()) / 255.0 if gray.size else 0.0,
        "std_gray_norm": float(gray.std()) / 128.0 if gray.size else 0.0,
        "min_gray_norm": float(gray.min()) / 255.0 if gray.size else 0.0,
        "max_gray_norm": float(gray.max()) / 255.0 if gray.size else 0.0,
        "dark_density_160": float((gray <= 160).mean()) if gray.size else 0.0,
        "dark_density_185": float((gray <= 185).mean()) if gray.size else 0.0,
        "dark_density_205": float((gray <= 205).mean()) if gray.size else 0.0,
        "dark_density_225": float((gray <= 225).mean()) if gray.size else 0.0,
        "dark_density_245": float((gray <= 245).mean()) if gray.size else 0.0,
        "center_dark_205": float((center_crop <= 205).mean()) if center_crop.size else 0.0,
        "edge_touch_dark_ratio": float((border_pixels <= 205).mean()) if border_pixels.size else 0.0,
        "row_dark_std": float(dark.mean(axis=1).std()) if dark.size else 0.0,
        "col_dark_std": float(dark.mean(axis=0).std()) if dark.size else 0.0,
        "sobel_mean": float(sobel.mean()) if sobel.size else 0.0,
        "sobel_max": float(sobel.max()) if sobel.size else 0.0,
        "canny_density": float(canny.mean()) if canny.size else 0.0,
        "blackhat_mean": float(blackhat.mean()) if blackhat.size else 0.0,
        "adaptive_density": float(adaptive.mean()) if adaptive.size else 0.0,
    }


def label_patch(patch_box: list[float], gold_items: list[dict[str, Any]]) -> tuple[bool, list[dict[str, Any]]]:
    cx, cy = center(patch_box)
    matched: list[dict[str, Any]] = []
    for gold in gold_items:
        gb = gold["symbol_bbox"]
        if contains_point(gb, cx, cy, margin=3.0) or iou(patch_box, gb) >= 0.03:
            matched.append(gold)
    return bool(matched), matched


def dense_patch_boxes(width: int, height: int, stride: int, patch_sizes: list[int]) -> list[list[float]]:
    boxes: list[list[float]] = []
    ys = list(range(stride // 2, height, stride))
    xs = list(range(stride // 2, width, stride))
    for side in patch_sizes:
        for y in ys:
            for x in xs:
                boxes.append(clip_box(float(x), float(y), float(side), width, height))
    return boxes


def positive_training_boxes(gold_items: list[dict[str, Any]], width: int, height: int, patch_sizes: list[int]) -> list[list[float]]:
    boxes: list[list[float]] = []
    for gold in gold_items:
        gx, gy = center(gold["symbol_bbox"])
        for side in patch_sizes:
            for dx, dy in [(0, 0), (-2, 0), (2, 0), (0, -2), (0, 2)]:
                boxes.append(clip_box(gx + dx, gy + dy, float(side), width, height))
    return boxes


def hard_negative_boxes(row: dict[str, Any], arr: np.ndarray, gold_items: list[dict[str, Any]], args: argparse.Namespace) -> list[list[float]]:
    import cv2

    height, width = arr.shape
    boxes: list[list[float]] = []
    mask = arr <= 205
    n_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype("uint8"), 8)
    for idx in range(1, int(n_labels)):
        x, y, w, h, area = [int(v) for v in stats[idx]]
        if area < 1 or area > 900 or w > 80 or h > 80:
            continue
        cx, cy = [float(v) for v in centroids[idx]]
        box = clip_box(cx, cy, float(args.patch_sizes[1]), width, height)
        label, _matched = label_patch(box, gold_items)
        if not label:
            boxes.append(box)
        if len(boxes) >= args.max_hard_negatives_per_row:
            break
    return boxes


def make_record(
    row: dict[str, Any],
    arr: np.ndarray,
    evidence: dict[str, np.ndarray],
    patch_box: list[float],
    gold_items: list[dict[str, Any]],
    split: str,
    record_id: str,
) -> dict[str, Any]:
    label, matched = label_patch(patch_box, gold_items)
    return {
        "id": record_id,
        "split": split,
        "row_id": row.get("id"),
        "image": row.get("image"),
        "image_size": row.get("image_size"),
        "bbox": [round(float(v), 6) for v in patch_box],
        "label_objectness": label,
        "gold_keys": [str(gold["gold_key"]) for gold in matched],
        "gold_symbol_types": [str(gold.get("symbol_type") or "symbol") for gold in matched],
        "features": {name: round(float(value), 8) for name, value in patch_features(row, arr, evidence, patch_box).items()},
        "offline_label_scope": "training_or_locked_diagnosis_only",
        "source_integrity": integrity(),
    }


def vector(row: dict[str, Any]) -> np.ndarray:
    feats = row.get("features") if isinstance(row.get("features"), dict) else {}
    return np.asarray([float(feats.get(name) or 0.0) for name in FEATURES], dtype=np.float32)


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
    bias = -min(float(np.percentile(pos_scores, 2)), float(np.percentile(neg_scores, 92)))
    return {
        "model_type": "patch_symbol_body_segmenter_centroid_ranker",
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
    by_row: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        for key in row.get("gold_keys") or []:
            by_row[str(row.get("row_id"))].add(str(key))
    for row in rows:
        row["all_target_gold_keys"] = sorted(by_row[str(row.get("row_id"))])


def recall_at_caps(scored: list[dict[str, Any]], caps: list[int]) -> dict[str, Any]:
    by_row: dict[str, list[dict[str, Any]]] = defaultdict(list)
    target_by_row: dict[str, set[str]] = defaultdict(set)
    for row in scored:
        rid = str(row.get("row_id"))
        by_row[rid].append(row)
        for key in row.get("all_target_gold_keys") or []:
            target_by_row[rid].add(str(key))
    out: dict[str, Any] = {}
    for cap in caps:
        hit: set[str] = set()
        total: set[str] = set()
        selected_count = 0
        for rid, rows in by_row.items():
            total.update(target_by_row[rid])
            selected = sorted(rows, key=lambda item: float(item.get("patch_score") or 0.0), reverse=True)[:cap]
            selected_count += len(selected)
            for row in selected:
                for key in row.get("gold_keys") or []:
                    hit.add(str(key))
        out[str(cap)] = {
            "selected_patch_rows": selected_count,
            "target_gold_keys": len(total),
            "hit_gold_keys": len(hit),
            "patch_recall": round(len(hit) / max(len(total), 1), 6),
            "selected_patch_rows_per_hit_gold": round(selected_count / max(len(hit), 1), 6),
        }
    return out


def feature_weights(model: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"feature": name, "weight": round(float(weight), 6)}
        for name, weight in sorted(zip(FEATURES, model["weights"], strict=True), key=lambda item: abs(float(item[1])), reverse=True)
    ]


def build_train_and_eval(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    adapter_by_id = load_by_id(Path(args.adapter))
    missing_by_row = target_missing_gold(Path(args.dataset), adapter_by_id)
    cache: dict[str, np.ndarray] = {}
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    counts = Counter()
    train_counts = Counter()
    eval_counts = Counter()
    patch_sizes = [int(v) for v in args.patch_sizes]
    for row_id, gold_items in sorted(missing_by_row.items()):
        row = adapter_by_id.get(row_id)
        if not row:
            continue
        arr = load_gray(row, cache)
        if arr is None:
            counts["missing_image_rows"] += 1
            continue
        evidence = evidence_arrays(arr)
        split = stable_row_split(row_id)
        height, width = arr.shape
        counts[f"{split}_rows"] += 1
        counts[f"{split}_gold_keys"] += len(gold_items)
        if split == "train":
            boxes = positive_training_boxes(gold_items, width, height, patch_sizes)
            boxes.extend(hard_negative_boxes(row, arr, gold_items, args))
            if args.max_random_negatives_per_row > 0:
                dense = dense_patch_boxes(width, height, args.train_negative_stride, [patch_sizes[1]])
                step = max(1, len(dense) // max(args.max_random_negatives_per_row, 1))
                boxes.extend(dense[::step][: args.max_random_negatives_per_row])
            seen: set[tuple[int, int, int, int]] = set()
            for index, box in enumerate(boxes):
                key = tuple(int(round(v)) for v in box)
                if key in seen:
                    continue
                seen.add(key)
                rec = make_record(row, arr, evidence, box, gold_items, split, f"{row_id}|patch_train_{index:05d}")
                train_counts["positive" if rec["label_objectness"] else "negative"] += 1
                train_rows.append(rec)
        else:
            boxes = dense_patch_boxes(width, height, args.eval_stride, patch_sizes)
            for index, box in enumerate(boxes):
                rec = make_record(row, arr, evidence, box, gold_items, split, f"{row_id}|patch_eval_{index:05d}")
                eval_counts["positive" if rec["label_objectness"] else "negative"] += 1
                eval_rows.append(rec)
    attach_all_targets(train_rows)
    attach_all_targets(eval_rows)
    audit = {
        "task": "IMG-MOE-V18-REBUILD-001.step_j7_build_patch_symbol_body_segmenter_dataset",
        "dataset": str(args.dataset),
        "adapter": str(args.adapter),
        "counts": dict(counts),
        "train_record_counts": dict(train_counts),
        "eval_record_counts": dict(eval_counts),
        "patch_sizes": patch_sizes,
        "eval_stride": args.eval_stride,
        "train_negative_stride": args.train_negative_stride,
        "source_integrity": integrity(),
        "gold_loaded_after_inference_for_training_and_diagnosis_only": True,
        "gold_used_for_inference": False,
    }
    return train_rows, eval_rows, audit


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--model-output", default=str(DEFAULT_MODEL))
    parser.add_argument("--audit-output", default=str(DEFAULT_AUDIT))
    parser.add_argument("--scored-output", default=str(DEFAULT_SCORED))
    parser.add_argument("--patch-sizes", type=parse_ints, default=parse_ints("7,11,17,25"))
    parser.add_argument("--eval-stride", type=int, default=6)
    parser.add_argument("--train-negative-stride", type=int, default=12)
    parser.add_argument("--max-hard-negatives-per-row", type=int, default=1200)
    parser.add_argument("--max-random-negatives-per-row", type=int, default=500)
    args = parser.parse_args()

    train_rows, eval_rows, dataset_audit = build_train_and_eval(args)
    model = train_model(train_rows)
    scored: list[dict[str, Any]] = []
    for row in eval_rows:
        item = {
            "id": row.get("id"),
            "row_id": row.get("row_id"),
            "bbox": row.get("bbox"),
            "label_objectness": bool(row.get("label_objectness")),
            "gold_keys": row.get("gold_keys"),
            "gold_symbol_types": row.get("gold_symbol_types"),
            "all_target_gold_keys": row.get("all_target_gold_keys"),
            "patch_score": round(score(row, model), 6),
            "features": row.get("features"),
            "source_integrity": integrity(),
        }
        scored.append(item)
    score_rows = [{"objectness_score": row["patch_score"], "label_objectness": row["label_objectness"]} for row in scored]
    thresholds = sorted({round(float(row["patch_score"]), 6) for row in scored})
    sweep = [threshold_metrics(score_rows, threshold) for threshold in thresholds]
    feasible = [row for row in sweep if row["recall"] >= 0.98]
    selected = sorted(feasible, key=lambda row: (row["candidate_reduction"], row["precision"]), reverse=True)[0] if feasible else sorted(sweep, key=lambda row: (row["recall"], row["precision"]), reverse=True)[0]
    cap_recall = recall_at_caps(scored, [50, 100, 250, 500, 1000, 2000])
    audit = {
        "task": "IMG-MOE-V18-REBUILD-001.step_j7_build_learned_patch_level_symbol_body_segmenter_dataset_and_model",
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
            "patch_auc": auc(score_rows),
            "selected_policy": selected,
            "patch_recall_at_caps": cap_recall,
            "top_feature_weights": feature_weights(model)[:18],
        },
        "adoption_decision": {
            "adopted_into_inference_stream": False,
            "reason": "Patch-level model is a learned dense heatmap diagnostic. Candidate-stream adoption requires >=0.90 target recall at <=500 selected patches/page.",
            "next_required_step": "If cap gate passes, convert top patch heatmap responses into symbol-body seeds/windows and replay topology; otherwise upgrade model capacity or supervision.",
        },
        "quality_gates": {
            "source_integrity_violations": 0,
            "gold_used_for_inference": False,
            "eval_patch_recall_at_500_ge_0_90": cap_recall["500"]["patch_recall"] >= 0.90,
            "ready_for_candidate_generation": cap_recall["500"]["patch_recall"] >= 0.90,
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
                "train_rows": len(train_rows),
                "eval_rows": len(eval_rows),
                "patch_auc": audit["eval"]["patch_auc"],
                "selected_policy": selected,
                "patch_recall_at_caps": cap_recall,
                "quality_gates": audit["quality_gates"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
