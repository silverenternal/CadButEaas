#!/usr/bin/env python3
"""Train a visual bbox refiner with a runtime-safe quality apply policy."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from train_symbol_tile_detector_v20 import bbox_iou, rel, write_json
from train_symbol_visual_box_refiner_v40 import apply_delta, features, load_jsonl, target


ROOT = Path(__file__).resolve().parents[2]


def source_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def input_iou(row: dict[str, Any]) -> float:
    return bbox_iou([float(v) for v in row["proposal"]["bbox"]], [float(v) for v in row["target"]["bbox"]])


def refined_iou(row: dict[str, Any], delta: list[float], clip: float) -> float:
    box = [float(v) for v in row["proposal"]["bbox"]]
    gold = [float(v) for v in row["target"]["bbox"]]
    return bbox_iou(apply_delta(box, delta, clip), gold)


def train_quality_labels(rows: list[dict[str, Any]], model: Any, clip: float) -> np.ndarray:
    x = np.asarray([features(row) for row in rows], dtype=np.float32)
    deltas = model.predict(x)
    labels: list[int] = []
    for row, delta in zip(rows, deltas, strict=True):
        before = input_iou(row)
        after = refined_iou(row, list(delta), clip)
        labels.append(int(after >= 0.30 and after >= before))
    return np.asarray(labels, dtype=np.int64)


def quality_score(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if len(proba) and proba.shape[1] > 1:
            return proba[:, 1]
    return model.predict(x).astype(np.float32)


def evaluate(
    refiner: Any,
    quality: Any,
    rows: list[dict[str, Any]],
    split: str,
    clip: float,
    threshold: float,
) -> dict[str, Any]:
    x = np.asarray([features(row) for row in rows], dtype=np.float32)
    deltas = refiner.predict(x)
    scores = quality_score(quality, x)
    totals = Counter()
    by_area = defaultdict(Counter)
    by_label = defaultdict(Counter)
    for row, delta, score in zip(rows, deltas, scores, strict=True):
        box = [float(v) for v in row["proposal"]["bbox"]]
        gold = [float(v) for v in row["target"]["bbox"]]
        refined = apply_delta(box, list(delta), clip)
        before = bbox_iou(box, gold)
        raw_after = bbox_iou(refined, gold)
        applied = float(score) >= threshold
        final_iou = raw_after if applied else before
        label = str(row["target"].get("label") or row["proposal"].get("label") or "")
        area = str(row["target"].get("area_bucket") or "")
        totals["rows"] += 1
        totals["applied"] += int(applied)
        totals["input_hit"] += int(before >= 0.30)
        totals["raw_refined_hit"] += int(raw_after >= 0.30)
        totals["policy_hit"] += int(final_iou >= 0.30)
        totals["policy_improved"] += int(final_iou > before)
        totals["policy_worse"] += int(final_iou < before)
        for bucket in [by_area[area], by_label[label]]:
            bucket["rows"] += 1
            bucket["applied"] += int(applied)
            bucket["input_hit"] += int(before >= 0.30)
            bucket["raw_refined_hit"] += int(raw_after >= 0.30)
            bucket["policy_hit"] += int(final_iou >= 0.30)

    def rates(c: Counter) -> dict[str, float]:
        n = max(int(c["rows"]), 1)
        return {
            "rows": int(c["rows"]),
            "applied": int(c["applied"]),
            "apply_rate": round(c["applied"] / n, 6),
            "input_iou_0_30_recall": round(c["input_hit"] / n, 6),
            "raw_refined_iou_0_30_recall": round(c["raw_refined_hit"] / n, 6),
            "policy_iou_0_30_recall": round(c["policy_hit"] / n, 6),
        }

    n = max(int(totals["rows"]), 1)
    return {
        "split": split,
        "rows": int(totals["rows"]),
        "applied": int(totals["applied"]),
        "apply_rate": round(totals["applied"] / n, 6),
        "input_iou_0_30_recall": round(totals["input_hit"] / n, 6),
        "raw_refined_iou_0_30_recall": round(totals["raw_refined_hit"] / n, 6),
        "policy_iou_0_30_recall": round(totals["policy_hit"] / n, 6),
        "policy_improved_rate": round(totals["policy_improved"] / n, 6),
        "policy_worse_rate": round(totals["policy_worse"] / n, 6),
        "by_area": {k: rates(v) for k, v in sorted(by_area.items())},
        "by_label": {k: rates(v) for k, v in sorted(by_label.items())},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="datasets/symbol_visual_box_refiner_v44_fulltarget")
    parser.add_argument("--output-dir", default="checkpoints/symbol_visual_box_refiner_v45_quality_policy")
    parser.add_argument("--eval-output", default="reports/vlm/symbol_visual_box_refiner_v45_quality_policy_locked_eval.json")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--quality-n-estimators", type=int, default=240)
    parser.add_argument("--quality-threshold", type=float, default=0.35)
    parser.add_argument("--clip", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=20260512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = source_path(args.data_dir)
    train = load_jsonl(data_dir / "train.jsonl")
    dev = load_jsonl(data_dir / "dev.jsonl")
    locked = load_jsonl(data_dir / "locked.jsonl")
    x_train = np.asarray([features(row) for row in train], dtype=np.float32)
    y_train = np.asarray([target(row) for row in train], dtype=np.float32)
    refiner = ExtraTreesRegressor(
        n_estimators=args.n_estimators,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=args.seed,
    )
    refiner.fit(x_train, y_train)
    y_quality = train_quality_labels(train, refiner, args.clip)
    quality = ExtraTreesClassifier(
        n_estimators=args.quality_n_estimators,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=args.seed + 17,
    )
    quality.fit(x_train, y_quality)
    out_dir = source_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.joblib"
    joblib.dump(
        {
            "refiner": refiner,
            "quality_model": quality,
            "quality_threshold": args.quality_threshold,
            "args": vars(args),
            "feature_type": "geom_plus_crop_intensity_stats_with_quality_policy",
        },
        model_path,
    )
    dev_eval = evaluate(refiner, quality, dev, "dev", args.clip, args.quality_threshold)
    locked_eval = evaluate(refiner, quality, locked, "locked", args.clip, args.quality_threshold)
    report = {
        "version": "symbol_visual_box_refiner_v45_quality_policy_locked_eval",
        "task": "P1-17-box-quality-policy-v45",
        "claim_boundary": "Quality-aware visual bbox refiner on v44 fulltarget crop rows. Quality labels are derived offline from training gold; runtime uses crop pixels and proposal fields only.",
        "source_integrity": {
            "model_input": "raster crop pixels plus proposal bbox/score/type",
            "offline_labels_used_for": ["training", "quality-label construction", "dev_evaluation", "locked_evaluation"],
            "gold_used_for_inference": False,
            "runtime_uses_svg_or_cad_geometry": False,
        },
        "training": {
            "checkpoint": rel(model_path),
            "train_rows": len(train),
            "quality_positive_rate": round(float(y_quality.mean()), 6),
            "quality_threshold": args.quality_threshold,
        },
        "dev": dev_eval,
        "locked": locked_eval,
        "stage_gate": {
            "locked_policy_iou_recall_improves": locked_eval["policy_iou_0_30_recall"] > locked_eval["input_iou_0_30_recall"],
            "locked_policy_not_worse_than_raw_refiner": locked_eval["policy_iou_0_30_recall"] >= locked_eval["raw_refined_iou_0_30_recall"],
            "locked_tiny_iou_recall_not_drop": locked_eval["by_area"].get("tiny_le_64", {}).get("policy_iou_0_30_recall", 0.0) >= locked_eval["by_area"].get("tiny_le_64", {}).get("input_iou_0_30_recall", 0.0),
            "no_oracle_inference": True,
        },
    }
    report["stage_gate"]["passed"] = all(report["stage_gate"].values())
    write_json(source_path(args.eval_output), report)
    print(json.dumps({"locked": locked_eval, "stage_gate": report["stage_gate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
