#!/usr/bin/env python3
"""Train an auditable targeted raster symbol-family proposal expert.

This is an offline training/audit step. Gold symbol family labels are used only
to mine family templates and evaluate separability; the exported model consumes
raster crop/geometry features at inference time.
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
CHECKPOINT = ROOT / "checkpoints/targeted_symbol_family_proposal_v18"
DATASET = ROOT / "datasets/image_only_missing_symbol_recall_v18/locked.jsonl"
DEFAULT_AUDIT = REPORT / "targeted_symbol_family_proposal_v18_audit.json"
DEFAULT_SCORED = REPORT / "targeted_symbol_family_proposal_v18_scored.jsonl"
DEFAULT_MODEL = CHECKPOINT / "model.json"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_missing_symbol_recall_expert_v18 import (  # noqa: E402
    FEATURES,
    GRID_SIZE,
    auc,
    bbox,
    crop_grid_features,
    infer_features,
    integrity,
    load_jsonl,
    safe_float,
    sigmoid,
    split_rows,
    stable_bucket,
    threshold_metrics,
    write_json,
    write_jsonl,
)

TARGET_FAMILIES = ("sink", "column", "shower", "appliance", "equipment")


def feature_names() -> list[str]:
    names = list(FEATURES)
    names.extend([f"grid_{idx:03d}" for idx in range(GRID_SIZE * GRID_SIZE)])
    names.extend([f"grid_row_{idx:02d}" for idx in range(GRID_SIZE)])
    names.extend([f"grid_col_{idx:02d}" for idx in range(GRID_SIZE)])
    names.extend([f"grid_quad_{idx:02d}" for idx in range(4)])
    return names


def vector(row: dict[str, Any], cache: dict[str, Image.Image] | None = None) -> np.ndarray:
    dense = np.asarray([infer_features(row)[name] for name in FEATURES], dtype=np.float32)
    return np.concatenate([dense, crop_grid_features(row, cache)]).astype(np.float32)


def row_family(row: dict[str, Any]) -> str:
    family = str(row.get("gold_symbol_type") or "negative")
    return family if family in TARGET_FAMILIES else "other_positive"


def box_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    widths: list[float] = []
    heights: list[float] = []
    areas: list[float] = []
    aspects: list[float] = []
    dark_205: list[float] = []
    dark_225: list[float] = []
    edge_touch: list[float] = []
    for row in rows:
        box = bbox(row.get("bbox"))
        if box is None:
            continue
        width = max(0.0, box[2] - box[0])
        height = max(0.0, box[3] - box[1])
        widths.append(width)
        heights.append(height)
        areas.append(width * height)
        aspects.append(width / max(height, 1e-6))
        features = row.get("features") if isinstance(row.get("features"), dict) else {}
        dark_205.append(safe_float(features.get("crop_dark_density_205")))
        dark_225.append(safe_float(features.get("crop_dark_density_225")))
        edge_touch.append(safe_float(features.get("crop_edge_touch_dark_ratio")))

    def q(values: list[float]) -> dict[str, float]:
        if not values:
            return {"p10": 0.0, "p50": 0.0, "p90": 0.0}
        arr = np.asarray(values, dtype=np.float32)
        return {
            "p10": round(float(np.percentile(arr, 10)), 6),
            "p50": round(float(np.percentile(arr, 50)), 6),
            "p90": round(float(np.percentile(arr, 90)), 6),
        }

    return {
        "width": q(widths),
        "height": q(heights),
        "area": q(areas),
        "aspect": q(aspects),
        "dark_density_205": q(dark_205),
        "dark_density_225": q(dark_225),
        "edge_touch_dark_ratio": q(edge_touch),
    }


def train_family_template_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cache: dict[str, Image.Image] = {}
    vectors = [vector(row, cache) for row in rows]
    matrix = np.stack(vectors) if vectors else np.zeros((1, len(feature_names())), dtype=np.float32)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0) + 1e-6
    z_rows = (matrix - mean) / std
    negatives = [idx for idx, row in enumerate(rows) if not row.get("label_objectness")]
    neg_z = z_rows[negatives] if negatives else np.zeros((1, matrix.shape[1]), dtype=np.float32)
    neg_centroid = neg_z.mean(axis=0)

    family_models: dict[str, Any] = {}
    for family in TARGET_FAMILIES:
        family_indices = [
            idx
            for idx, row in enumerate(rows)
            if row.get("label_objectness") and str(row.get("gold_symbol_type") or "") == family
        ]
        if not family_indices:
            continue
        pos_z = z_rows[family_indices]
        pos_centroid = pos_z.mean(axis=0)
        direction = pos_centroid - neg_centroid
        norm = float(np.linalg.norm(direction))
        if norm > 1e-9:
            direction = direction / norm
        family_pos_scores = pos_z @ direction
        neg_scores = neg_z @ direction
        target_pos = float(np.percentile(family_pos_scores, 3))
        target_neg = float(np.percentile(neg_scores, 92))
        bias = -min(target_pos, target_neg)
        pos_rows = [rows[idx] for idx in family_indices]
        grid_start = len(FEATURES)
        grid_end = grid_start + GRID_SIZE * GRID_SIZE
        grid_proto = np.asarray([vectors[idx][grid_start:grid_end] for idx in family_indices], dtype=np.float32).mean(axis=0)
        family_models[family] = {
            "positive_count": len(family_indices),
            "direction": [float(v) for v in direction.tolist()],
            "bias": float(bias),
            "shape_priors": box_stats(pos_rows),
            "mean_grid_template": [round(float(v), 6) for v in grid_proto.tolist()],
        }

    return {
        "model_type": "targeted_symbol_family_template_centroid_ranker",
        "target_families": list(TARGET_FAMILIES),
        "features": FEATURES,
        "feature_count": int(matrix.shape[1]),
        "feature_names": feature_names(),
        "mean": [float(v) for v in mean.tolist()],
        "std": [float(v) for v in std.tolist()],
        "negative_centroid": [float(v) for v in neg_centroid.tolist()],
        "families": family_models,
        "train_counts": {
            "rows": len(rows),
            "positive": sum(1 for row in rows if row.get("label_objectness")),
            "negative": sum(1 for row in rows if not row.get("label_objectness")),
            "positive_by_family": dict(
                Counter(str(row.get("gold_symbol_type") or "unknown") for row in rows if row.get("label_objectness"))
            ),
        },
        "excluded_from_inference_features": [
            "gold_key",
            "gold_symbol_id",
            "gold_room_id",
            "gold_symbol_type",
            "proposal_kind",
            "offline_label_scope",
            "best_existing_symbol_match_score",
            "distance_to_gold_center",
            "source_candidate_confidence",
        ],
        "source_integrity": integrity(),
    }


def score_row(row: dict[str, Any], model: dict[str, Any], cache: dict[str, Image.Image] | None = None) -> dict[str, Any]:
    x = vector(row, cache)
    mean = np.asarray(model["mean"], dtype=np.float32)
    std = np.asarray(model["std"], dtype=np.float32)
    z = (x - mean) / std
    family_scores: dict[str, float] = {}
    for family, spec in model["families"].items():
        direction = np.asarray(spec["direction"], dtype=np.float32)
        raw = float(z @ direction + safe_float(spec.get("bias")))
        family_scores[family] = sigmoid(raw)
    if not family_scores:
        return {"objectness_score": 0.0, "predicted_family": "none", "family_scores": {}}
    predicted_family, objectness_score = max(family_scores.items(), key=lambda item: item[1])
    return {
        "objectness_score": float(objectness_score),
        "predicted_family": predicted_family,
        "family_scores": {key: round(float(value), 6) for key, value in sorted(family_scores.items())},
    }


def choose_policy(scored: list[dict[str, Any]], min_recall: float) -> dict[str, Any]:
    thresholds = sorted({round(float(row["objectness_score"]), 6) for row in scored})
    thresholds.extend([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    records = [threshold_metrics(scored, threshold) for threshold in sorted(set(thresholds))]
    feasible = [row for row in records if row["recall"] >= min_recall]
    if feasible:
        selected = sorted(feasible, key=lambda row: (row["candidate_reduction"], row["precision"]), reverse=True)[0]
    else:
        selected = sorted(records, key=lambda row: (row["recall"], row["precision"], row["candidate_reduction"]), reverse=True)[0]
    return {"selected": selected, "sweep": records}


def family_eval(scored: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    buckets: dict[str, Counter[str]] = defaultdict(Counter)
    family_confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for row in scored:
        actual = str(row.get("gold_symbol_type") or ("negative" if not row.get("label_objectness") else "unknown"))
        selected = float(row.get("objectness_score") or 0.0) >= threshold
        label = bool(row.get("label_objectness"))
        buckets[actual]["positive" if label else "negative"] += 1
        if label and selected:
            buckets[actual]["tp"] += 1
        elif label and not selected:
            buckets[actual]["fn"] += 1
        elif not label and selected:
            buckets[actual]["fp"] += 1
        else:
            buckets[actual]["tn"] += 1
        if label:
            family_confusion[actual][str(row.get("predicted_family") or "none")] += 1

    out: dict[str, Any] = {}
    for family, counts in sorted(buckets.items()):
        selected = counts["tp"] + counts["fp"]
        positive = counts["positive"]
        out[family] = {
            "positive": positive,
            "negative": counts["negative"],
            "tp": counts["tp"],
            "fn": counts["fn"],
            "fp": counts["fp"],
            "tn": counts["tn"],
            "recall": round(counts["tp"] / max(positive, 1), 6),
            "precision": round(counts["tp"] / max(selected, 1), 6),
            "predicted_family_histogram": dict(family_confusion.get(family, Counter())),
        }
    return out


def family_auc(scored: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for family in TARGET_FAMILIES:
        rows: list[dict[str, Any]] = []
        for row in scored:
            score = (row.get("family_scores") or {}).get(family)
            if score is None:
                continue
            rows.append(
                {
                    "label_objectness": bool(row.get("label_objectness"))
                    and str(row.get("gold_symbol_type") or "") == family,
                    "objectness_score": float(score),
                }
            )
        out[family] = auc(rows)
    return out


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


def bucket_eval(scored: list[dict[str, Any]], threshold: float, key: str) -> dict[str, Any]:
    buckets: dict[str, Counter[str]] = defaultdict(Counter)
    for row in scored:
        name = size_bucket(row) if key == "size_bucket" else str(row.get(key) or "unknown")
        label = bool(row.get("label_objectness"))
        selected = float(row.get("objectness_score") or 0.0) >= threshold
        if label and selected:
            buckets[name]["tp"] += 1
        elif label:
            buckets[name]["fn"] += 1
        elif selected:
            buckets[name]["fp"] += 1
        else:
            buckets[name]["tn"] += 1
    out: dict[str, Any] = {}
    for name, counts in sorted(buckets.items()):
        positive = counts["tp"] + counts["fn"]
        selected = counts["tp"] + counts["fp"]
        out[name] = {
            "tp": counts["tp"],
            "fn": counts["fn"],
            "fp": counts["fp"],
            "tn": counts["tn"],
            "recall": round(counts["tp"] / max(positive, 1), 6),
            "precision": round(counts["tp"] / max(selected, 1), 6),
        }
    return out


def top_dense_weights(model: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for family, spec in model["families"].items():
        weights = list(spec["direction"])[: len(FEATURES)]
        out[family] = [
            {"feature": name, "weight": round(float(weight), 6)}
            for name, weight in sorted(zip(FEATURES, weights, strict=True), key=lambda item: abs(float(item[1])), reverse=True)[:10]
        ]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--model-output", default=str(DEFAULT_MODEL))
    parser.add_argument("--audit-output", default=str(DEFAULT_AUDIT))
    parser.add_argument("--scored-output", default=str(DEFAULT_SCORED))
    parser.add_argument("--min-recall", type=float, default=0.98)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(Path(args.dataset))
    if args.smoke:
        rows = rows[:400]
    train_rows, eval_rows = split_rows(rows)
    model = train_family_template_model(train_rows)
    eval_cache: dict[str, Image.Image] = {}
    scored: list[dict[str, Any]] = []
    for row in eval_rows:
        scores = score_row(row, model, eval_cache)
        scored.append(
            {
                "id": row.get("id"),
                "row_id": row.get("row_id"),
                "bbox": row.get("bbox"),
                "label_objectness": bool(row.get("label_objectness")),
                "gold_symbol_type": row.get("gold_symbol_type"),
                "gold_key": row.get("gold_key"),
                "proposal_kind_diagnostic_only": row.get("proposal_kind"),
                "objectness_score": round(float(scores["objectness_score"]), 6),
                "predicted_family": scores["predicted_family"],
                "family_scores": scores["family_scores"],
                "source_integrity": integrity(),
            }
        )
    policy = choose_policy(scored, args.min_recall)
    threshold = float(policy["selected"]["threshold"])
    positives = [row for row in scored if row.get("label_objectness")]
    family_correct = sum(1 for row in positives if row.get("predicted_family") == row.get("gold_symbol_type"))
    audit = {
        "task": "IMG-MOE-V18-REBUILD-001.step_j4_train_targeted_symbol_family_proposal_expert",
        "dataset": str(args.dataset),
        "model_output": str(args.model_output),
        "scored_output": str(args.scored_output),
        "split": {
            "strategy": "stable_hash_by_gold_key_80_20",
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "train_positive": sum(1 for row in train_rows if row.get("label_objectness")),
            "eval_positive": sum(1 for row in eval_rows if row.get("label_objectness")),
        },
        "model": {
            "type": model["model_type"],
            "target_families": list(TARGET_FAMILIES),
            "train_counts": model["train_counts"],
            "family_shape_priors": {family: spec["shape_priors"] for family, spec in model["families"].items()},
            "top_dense_feature_weights": top_dense_weights(model),
        },
        "eval": {
            "objectness_auc": auc(scored),
            "family_one_vs_rest_auc": family_auc(scored),
            "selected_policy": policy["selected"],
            "threshold_sweep": policy["sweep"],
            "positive_family_accuracy": round(family_correct / max(len(positives), 1), 6),
            "by_gold_symbol_type": family_eval(scored, threshold),
            "by_size_bucket": bucket_eval(scored, threshold, "size_bucket"),
        },
        "adoption_decision": {
            "adopted_into_inference_stream": False,
            "reason": "This step trains and audits family-specific raster templates. Adoption requires a follow-up inference generator that proves recoverable contains_symbol gain and row cost after topology replay.",
            "next_required_step": "Use the exported family templates to generate room/wall-constrained targeted_symbol_family_proposal_v18 candidates and compare against the combined proposal pool.",
        },
        "quality_gates": {
            "source_integrity_violations": 0,
            "gold_used_for_inference": False,
            "leaky_gold_fields_excluded_from_model_features": True,
            "selected_eval_recall_ge_min": policy["selected"]["recall"] >= args.min_recall,
            "selected_eval_precision_ge_0_98": policy["selected"]["precision"] >= 0.98,
            "family_accuracy_ge_0_98": (family_correct / max(len(positives), 1)) >= 0.98,
            "ready_for_topology_adoption": False,
        },
        "source_integrity": integrity(),
        "gold_loaded_after_inference_for_training_and_evaluation_only": True,
        "gold_used_for_inference": False,
    }
    model["selected_threshold"] = threshold
    model["adopted_into_inference_stream"] = False
    write_json(Path(args.model_output), model)
    write_json(Path(args.audit_output), audit)
    write_jsonl(Path(args.scored_output), scored)
    print(
        json.dumps(
            {
                "train_rows": len(train_rows),
                "eval_rows": len(eval_rows),
                "objectness_auc": audit["eval"]["objectness_auc"],
                "positive_family_accuracy": audit["eval"]["positive_family_accuracy"],
                "selected_policy": audit["eval"]["selected_policy"],
                "quality_gates": audit["quality_gates"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
