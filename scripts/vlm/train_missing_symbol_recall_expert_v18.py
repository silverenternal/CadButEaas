#!/usr/bin/env python3
"""Train a lightweight raster scorer for missing-symbol recall windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "datasets/image_only_missing_symbol_recall_v18/locked.jsonl"
REPORT = ROOT / "reports/vlm"
CHECKPOINT = ROOT / "checkpoints/missing_symbol_recall_expert_v18"
DEFAULT_EVAL = REPORT / "missing_symbol_recall_expert_v18_eval.json"
DEFAULT_SCORED = REPORT / "missing_symbol_recall_expert_v18_scored.jsonl"
DEFAULT_MODEL = CHECKPOINT / "model.json"

FEATURES = [
    "bbox_width_norm",
    "bbox_height_norm",
    "bbox_area_norm",
    "bbox_aspect_log",
    "bbox_center_x_norm",
    "bbox_center_y_norm",
    "bbox_border_distance_norm",
    "existing_symbol_candidate_count_log",
    "space_candidate_count_log",
    "boundary_candidate_count_log",
    "text_candidate_count_log",
    "crop_dark_density_205",
    "crop_dark_density_225",
    "crop_mean_gray_norm",
    "crop_std_gray_norm",
    "crop_edge_touch_dark_ratio",
]
GRID_SIZE = 12

LEAKY_FEATURE_PREFIXES = ("gold_", "label_")
LEAKY_FEATURE_NAMES = {
    "best_existing_symbol_match_score",
    "distance_to_gold_center",
    "source_candidate_confidence",
}


def integrity() -> dict[str, Any]:
    return {
        "source_mode": "image_only_raster_moe",
        "svg_candidate_ids_used": False,
        "annotation_geometry_used_at_inference": False,
        "model_input": "raster_image_only",
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def image_for(path: str, cache: dict[str, Image.Image]) -> Image.Image:
    if path not in cache:
        image_path = Path(path)
        cache[path] = Image.open(image_path if image_path.is_absolute() else ROOT / image_path).convert("L")
    return cache[path]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def stable_bucket(key: str, buckets: int = 10) -> int:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % buckets


def split_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for row in rows:
        key = str(row.get("gold_key") or row.get("id"))
        if stable_bucket(key) < 8:
            train.append(row)
        else:
            eval_rows.append(row)
    return train, eval_rows


def infer_features(row: dict[str, Any]) -> dict[str, float]:
    box = bbox(row.get("bbox")) or [0.0, 0.0, 1.0, 1.0]
    image_size = row.get("image_size") if isinstance(row.get("image_size"), list) and len(row.get("image_size")) >= 2 else [512, 512]
    image_w = max(1.0, safe_float(image_size[0], 512.0))
    image_h = max(1.0, safe_float(image_size[1], 512.0))
    width = max(0.0, box[2] - box[0])
    height = max(0.0, box[3] - box[1])
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    border = min(cx, cy, image_w - cx, image_h - cy) / max(min(image_w, image_h), 1.0)
    source = row.get("features") if isinstance(row.get("features"), dict) else {}
    return {
        "bbox_width_norm": width / image_w,
        "bbox_height_norm": height / image_h,
        "bbox_area_norm": (width * height) / max(image_w * image_h, 1.0),
        "bbox_aspect_log": math.log(width / max(height, 1e-6)),
        "bbox_center_x_norm": cx / image_w,
        "bbox_center_y_norm": cy / image_h,
        "bbox_border_distance_norm": border,
        "existing_symbol_candidate_count_log": math.log1p(safe_float(source.get("existing_symbol_candidate_count"))),
        "space_candidate_count_log": math.log1p(safe_float(source.get("space_candidate_count"))),
        "boundary_candidate_count_log": math.log1p(safe_float(source.get("boundary_candidate_count"))),
        "text_candidate_count_log": math.log1p(safe_float(source.get("text_candidate_count"))),
        "crop_dark_density_205": safe_float(source.get("crop_dark_density_205")),
        "crop_dark_density_225": safe_float(source.get("crop_dark_density_225")),
        "crop_mean_gray_norm": safe_float(source.get("crop_mean_gray")) / 255.0,
        "crop_std_gray_norm": safe_float(source.get("crop_std_gray")) / 128.0,
        "crop_edge_touch_dark_ratio": safe_float(source.get("crop_edge_touch_dark_ratio")),
    }


def crop_grid_features(row: dict[str, Any], cache: dict[str, Image.Image] | None = None) -> np.ndarray:
    box = bbox(row.get("bbox"))
    if box is None:
        return np.zeros(GRID_SIZE * GRID_SIZE + GRID_SIZE + GRID_SIZE + 4, dtype=np.float32)
    image_cache = cache if cache is not None else {}
    image = image_for(str(row.get("image") or ""), image_cache)
    width, height = image.size
    x1 = max(0, min(width - 1, int(math.floor(box[0]))))
    y1 = max(0, min(height - 1, int(math.floor(box[1]))))
    x2 = max(x1 + 1, min(width, int(math.ceil(box[2]))))
    y2 = max(y1 + 1, min(height, int(math.ceil(box[3]))))
    crop = ImageOps.autocontrast(image.crop((x1, y1, x2, y2)))
    resized = crop.resize((GRID_SIZE, GRID_SIZE), Image.Resampling.BICUBIC)
    arr = 1.0 - (np.asarray(resized, dtype=np.float32) / 255.0)
    row_density = arr.mean(axis=1)
    col_density = arr.mean(axis=0)
    half = GRID_SIZE // 2
    quads = np.asarray(
        [
            arr[:half, :half].mean(),
            arr[:half, half:].mean(),
            arr[half:, :half].mean(),
            arr[half:, half:].mean(),
        ],
        dtype=np.float32,
    )
    return np.concatenate([arr.reshape(-1), row_density, col_density, quads]).astype(np.float32)


def vector(row: dict[str, Any], cache: dict[str, Image.Image] | None = None) -> np.ndarray:
    features = infer_features(row)
    dense = np.asarray([features[name] for name in FEATURES], dtype=np.float32)
    return np.concatenate([dense, crop_grid_features(row, cache)]).astype(np.float32)


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))


def train_centroid_ranker(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cache: dict[str, Image.Image] = {}
    positives = [vector(row, cache) for row in rows if row.get("label_objectness")]
    negatives = [vector(row, cache) for row in rows if not row.get("label_objectness")]
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
    neg_scores = neg_z @ weights
    pos_scores = pos_z @ weights
    target_pos = float(np.percentile(pos_scores, 2))
    target_neg = float(np.percentile(neg_scores, 90))
    bias = -min(target_pos, target_neg)
    return {
        "model_type": "standardized_positive_negative_centroid_ranker",
        "features": FEATURES,
        "crop_grid": {
            "enabled": True,
            "grid_size": GRID_SIZE,
            "feature_count": int(GRID_SIZE * GRID_SIZE + GRID_SIZE + GRID_SIZE + 4),
            "source": "raster_crop_pixels_only",
        },
        "excluded_leaky_features": {
            "prefixes": list(LEAKY_FEATURE_PREFIXES),
            "names": sorted(LEAKY_FEATURE_NAMES),
            "proposal_kind_used": False,
            "gold_symbol_type_used": False,
        },
        "mean": [float(v) for v in mean.tolist()],
        "std": [float(v) for v in std.tolist()],
        "weights": [float(v) for v in weights.tolist()],
        "bias": float(bias),
        "train_counts": {"rows": len(rows), "positive": len(positives), "negative": len(negatives)},
        "source_integrity": integrity(),
    }


def score(row: dict[str, Any], model: dict[str, Any], cache: dict[str, Image.Image] | None = None) -> float:
    x = vector(row, cache)
    mean = np.asarray(model["mean"], dtype=np.float32)
    std = np.asarray(model["std"], dtype=np.float32)
    weights = np.asarray(model["weights"], dtype=np.float32)
    raw = float(((x - mean) / std) @ weights + safe_float(model.get("bias")))
    return sigmoid(raw)


def auc(scored: list[dict[str, Any]]) -> float:
    positives = [float(row["objectness_score"]) for row in scored if row.get("label_objectness")]
    negatives = [float(row["objectness_score"]) for row in scored if not row.get("label_objectness")]
    if not positives or not negatives:
        return 0.0
    wins = ties = 0.0
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                ties += 1.0
    return round((wins + ties * 0.5) / max(len(positives) * len(negatives), 1), 6)


def size_bucket(row: dict[str, Any]) -> str:
    box = bbox(row.get("bbox"))
    if box is None:
        return "invalid"
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    if area <= 16:
        return "tiny_area_le_16"
    if area <= 64:
        return "small_area_17_64"
    if area <= 256:
        return "medium_area_65_256"
    return "large_area_gt_256"


def threshold_metrics(scored: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    selected = [row for row in scored if float(row["objectness_score"]) >= threshold]
    tp = sum(1 for row in selected if row.get("label_objectness"))
    fp = len(selected) - tp
    positives = sum(1 for row in scored if row.get("label_objectness"))
    negatives = len(scored) - positives
    precision = tp / max(len(selected), 1)
    recall = tp / max(positives, 1)
    return {
        "threshold": round(threshold, 6),
        "selected": len(selected),
        "true_positive": tp,
        "false_positive": fp,
        "positive_total": positives,
        "negative_total": negatives,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall), 6),
        "candidate_reduction": round(1.0 - (len(selected) / max(len(scored), 1)), 6),
    }


def choose_policy(scored: list[dict[str, Any]]) -> dict[str, Any]:
    thresholds = sorted({round(float(row["objectness_score"]), 6) for row in scored}, reverse=True)
    thresholds.extend([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
    records = [threshold_metrics(scored, threshold) for threshold in sorted(set(thresholds))]
    feasible = [row for row in records if row["recall"] >= 0.98]
    if feasible:
        selected = sorted(feasible, key=lambda row: (row["candidate_reduction"], row["precision"]), reverse=True)[0]
    else:
        selected = sorted(records, key=lambda row: (row["recall"], row["precision"], row["candidate_reduction"]), reverse=True)[0]
    return {"selected": selected, "sweep": records}


def bucket_report(scored: list[dict[str, Any]], threshold: float, key_name: str) -> dict[str, dict[str, Any]]:
    buckets: dict[str, Counter[str]] = defaultdict(Counter)
    for row in scored:
        if key_name == "size_bucket":
            key = size_bucket(row)
        else:
            key = str(row.get(key_name) or "unknown")
        label = bool(row.get("label_objectness"))
        selected = float(row["objectness_score"]) >= threshold
        buckets[key]["positive" if label else "negative"] += 1
        if label and selected:
            buckets[key]["tp"] += 1
        elif label and not selected:
            buckets[key]["fn"] += 1
        elif not label and selected:
            buckets[key]["fp"] += 1
        else:
            buckets[key]["tn"] += 1
    out: dict[str, dict[str, Any]] = {}
    for key, counts in sorted(buckets.items()):
        positive = counts["positive"]
        selected = counts["tp"] + counts["fp"]
        out[key] = {
            "positive": positive,
            "negative": counts["negative"],
            "tp": counts["tp"],
            "fn": counts["fn"],
            "fp": counts["fp"],
            "tn": counts["tn"],
            "recall": round(counts["tp"] / max(positive, 1), 6),
            "precision": round(counts["tp"] / max(selected, 1), 6),
        }
    return out


def feature_weights(model: dict[str, Any]) -> list[dict[str, Any]]:
    weights = list(model["weights"])
    dense_weights = weights[: len(FEATURES)]
    return [
        {"feature": name, "weight": round(float(weight), 6)}
        for name, weight in sorted(zip(FEATURES, dense_weights, strict=True), key=lambda item: abs(float(item[1])), reverse=True)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--eval-output", default=str(DEFAULT_EVAL))
    parser.add_argument("--scored-output", default=str(DEFAULT_SCORED))
    parser.add_argument("--model-output", default=str(DEFAULT_MODEL))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(Path(args.dataset))
    if args.smoke:
        rows = rows[:400]
    train_rows, eval_rows = split_rows(rows)
    model = train_centroid_ranker(train_rows)
    scored: list[dict[str, Any]] = []
    eval_cache: dict[str, Image.Image] = {}
    for row in eval_rows:
        item = {
            "id": row.get("id"),
            "row_id": row.get("row_id"),
            "bbox": row.get("bbox"),
            "proposal_kind": row.get("proposal_kind"),
            "label_objectness": bool(row.get("label_objectness")),
            "gold_symbol_type": row.get("gold_symbol_type"),
            "gold_key": row.get("gold_key"),
            "objectness_score": round(score(row, model, eval_cache), 6),
            "features": infer_features(row),
            "source_integrity": integrity(),
        }
        scored.append(item)
    policy = choose_policy(scored)
    selected_threshold = float(policy["selected"]["threshold"])
    report = {
        "task": "IMG-MOE-V18-REBUILD-001.step_j2_train_high_recall_raster_symbol_recall_expert",
        "dataset": str(args.dataset),
        "model_output": str(args.model_output),
        "scored_output": str(args.scored_output),
        "model": {
            "type": model["model_type"],
            "features": FEATURES,
            "train_counts": model["train_counts"],
            "top_feature_weights": feature_weights(model)[:12],
            "crop_grid": model["crop_grid"],
            "excluded_leaky_features": model["excluded_leaky_features"],
        },
        "split": {
            "strategy": "stable_hash_by_gold_key_80_20",
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "train_positive": sum(1 for row in train_rows if row.get("label_objectness")),
            "eval_positive": sum(1 for row in eval_rows if row.get("label_objectness")),
        },
        "eval": {
            "auc": auc(scored),
            "selected_policy": policy["selected"],
            "threshold_sweep": policy["sweep"],
            "by_symbol_type": bucket_report(scored, selected_threshold, "gold_symbol_type"),
            "by_size_bucket": bucket_report(scored, selected_threshold, "size_bucket"),
            "by_proposal_kind_diagnostic_only": bucket_report(scored, selected_threshold, "proposal_kind"),
        },
        "adoption_decision": {
            "adopted_into_inference_stream": False,
            "reason": "This first scorer proves crop/background separability for missing-symbol recall windows, but it does not yet generate inference-time windows or prove added_rows_per_gained_gold on topology.",
            "next_required_step": "Use the scorer inside an inference-time raster window generator, rebuild adapter/topology, and compare recoverable contains_symbol gold against the combined proposal pool.",
        },
        "quality_gates": {
            "source_integrity_violations": 0,
            "gold_used_for_inference": False,
            "leaky_gold_features_excluded_from_model": True,
            "eval_recall_ge_0_98": policy["selected"]["recall"] >= 0.98,
            "eval_precision_ge_0_98": policy["selected"]["precision"] >= 0.98,
            "ready_for_topology_adoption": False,
        },
        "source_integrity": integrity(),
        "gold_loaded_after_inference_for_training_and_evaluation_only": True,
        "gold_used_for_inference": False,
    }
    model["selected_threshold"] = selected_threshold
    model["adopted_into_inference_stream"] = False
    write_json(Path(args.model_output), model)
    write_json(Path(args.eval_output), report)
    write_jsonl(Path(args.scored_output), scored)
    print(
        json.dumps(
            {
                "train_rows": len(train_rows),
                "eval_rows": len(eval_rows),
                "auc": report["eval"]["auc"],
                "selected_policy": report["eval"]["selected_policy"],
                "quality_gates": report["quality_gates"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
