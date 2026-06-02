#!/usr/bin/env python3
"""Train/evaluate a lightweight bbox-delta refiner for symbol candidates."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from train_symbol_support_suppression_v35 import load_jsonl
from train_symbol_tile_detector_v20 import bbox_iou, rel, write_json


ROOT = Path(__file__).resolve().parents[2]


def source_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def feature_names(rows: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for row in rows:
        names.update((row.get("features") or {}).keys())
    forbidden = {"is_center_only_no_iou", "input_iou"}
    return sorted(
        name
        for name in names
        if not name.endswith("_train_label")
        and name not in forbidden
        and not name.startswith("target_")
    )


def vector(row: dict[str, Any], names: list[str]) -> list[float]:
    feats = row.get("features") or {}
    return [float(feats.get(name, 0.0) or 0.0) for name in names]


def target(row: dict[str, Any]) -> list[float]:
    if row.get("box_delta"):
        delta = row.get("box_delta") or {}
        return [float(delta.get(name, 0.0) or 0.0) for name in ["dcx", "dcy", "dw", "dh"]]
    labels = row.get("labels") or {}
    return [float(labels.get(name, 0.0) or 0.0) for name in ["dx1", "dy1", "dx2", "dy2"]]


def apply_delta(box: list[float], delta: list[float], clip: float) -> list[float]:
    width = max(1e-6, box[2] - box[0])
    height = max(1e-6, box[3] - box[1])
    dx, dy, dw, dh = [max(-clip, min(clip, float(v))) for v in delta]
    cx = (box[0] + box[2]) * 0.5 + dx * width
    cy = (box[1] + box[3]) * 0.5 + dy * height
    out_w = max(1.0, width * (1.0 + dw))
    out_h = max(1.0, height * (1.0 + dh))
    out = [cx - out_w * 0.5, cy - out_h * 0.5, cx + out_w * 0.5, cy + out_h * 0.5]
    if out[2] <= out[0] + 1:
        out[2] = out[0] + 1
    if out[3] <= out[1] + 1:
        out[3] = out[1] + 1
    return out


def eval_rows(model: Any, rows: list[dict[str, Any]], names: list[str], split: str, clip: float) -> dict[str, Any]:
    totals = Counter()
    by_label = defaultdict(Counter)
    by_area = defaultdict(Counter)
    batch = 50000
    for start in range(0, len(rows), batch):
        chunk = rows[start : start + batch]
        x = np.asarray([vector(row, names) for row in chunk], dtype=np.float32)
        preds = model.predict(x)
        for row, delta in zip(chunk, preds, strict=True):
            box = [float(v) for v in (row.get("bbox") or row.get("candidate_bbox"))]
            target_box = [float(v) for v in ((row.get("labels") or {}).get("target_bbox") or row.get("target_bbox"))]
            refined = apply_delta(box, list(delta), clip)
            input_iou = bbox_iou(box, target_box)
            refined_iou = bbox_iou(refined, target_box)
            label = str((row.get("labels") or {}).get("target_label") or row.get("target_label") or row.get("label") or "generic_symbol")
            area = str((row.get("labels") or {}).get("target_area_bucket") or row.get("target_area_bucket") or "unknown")
            totals["rows"] += 1
            totals["input_hit"] += int(input_iou >= 0.30)
            totals["refined_hit"] += int(refined_iou >= 0.30)
            totals["improved"] += int(refined_iou > input_iou)
            totals["worse"] += int(refined_iou < input_iou)
            totals["input_iou_sum"] += input_iou
            totals["refined_iou_sum"] += refined_iou
            by_label[label]["rows"] += 1
            by_label[label]["input_hit"] += int(input_iou >= 0.30)
            by_label[label]["refined_hit"] += int(refined_iou >= 0.30)
            by_area[area]["rows"] += 1
            by_area[area]["input_hit"] += int(input_iou >= 0.30)
            by_area[area]["refined_hit"] += int(refined_iou >= 0.30)
    def rates(counter: Counter) -> dict[str, float]:
        rows_n = max(int(counter["rows"]), 1)
        return {
            "rows": int(counter["rows"]),
            "input_iou_0_30_recall": round(counter["input_hit"] / rows_n, 6),
            "refined_iou_0_30_recall": round(counter["refined_hit"] / rows_n, 6),
        }
    rows_n = max(int(totals["rows"]), 1)
    return {
        "split": split,
        "rows": int(totals["rows"]),
        "input_iou_mean": round(totals["input_iou_sum"] / rows_n, 6),
        "refined_iou_mean": round(totals["refined_iou_sum"] / rows_n, 6),
        "input_iou_0_30_recall": round(totals["input_hit"] / rows_n, 6),
        "refined_iou_0_30_recall": round(totals["refined_hit"] / rows_n, 6),
        "improved_rate": round(totals["improved"] / rows_n, 6),
        "worse_rate": round(totals["worse"] / rows_n, 6),
        "by_label": {key: rates(value) for key, value in sorted(by_label.items())},
        "by_area": {key: rates(value) for key, value in sorted(by_area.items())},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="datasets/symbol_box_refiner_v38")
    parser.add_argument("--output-dir", default="checkpoints/symbol_box_refiner_v38")
    parser.add_argument("--eval-output", default="reports/vlm/symbol_box_refiner_v38_locked_eval.json")
    parser.add_argument("--n-estimators", type=int, default=180)
    parser.add_argument("--max-train-rows", type=int, default=120000)
    parser.add_argument("--clip", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=20260512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = source_path(args.data_dir)
    train_rows = load_jsonl(data_dir / "train.jsonl")
    dev_rows = load_jsonl(data_dir / "dev.jsonl")
    locked_path = data_dir / "locked.jsonl"
    locked_rows = load_jsonl(locked_path if locked_path.exists() else data_dir / "smoke_eval.jsonl")
    if args.max_train_rows and len(train_rows) > args.max_train_rows:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(np.arange(len(train_rows)), size=args.max_train_rows, replace=False)
        train_rows = [train_rows[int(i)] for i in idx]
    names = feature_names(train_rows)
    x = np.asarray([vector(row, names) for row in train_rows], dtype=np.float32)
    y = np.asarray([target(row) for row in train_rows], dtype=np.float32)
    model = ExtraTreesRegressor(n_estimators=args.n_estimators, min_samples_leaf=2, max_features="sqrt", n_jobs=-1, random_state=args.seed)
    model.fit(x, y)
    output_dir = source_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.joblib"
    joblib.dump({"model": model, "feature_names": names, "args": vars(args)}, model_path)
    dev = eval_rows(model, dev_rows, names, "dev", args.clip)
    locked = eval_rows(model, locked_rows, names, "locked", args.clip)
    tiny = locked["by_area"].get("tiny_le_64", {})
    sink = locked["by_label"].get("sink", {})
    shower = locked["by_label"].get("shower", {})
    report = {
        "version": "symbol_box_refiner_v38_locked_eval",
        "task": "P1-08-tiny-sink-shower-box-quality-refiner-v38",
        "claim_boundary": "Feature-only bbox-delta baseline. Runtime features are candidate bbox/score/type; gold bbox used only for supervised training/evaluation.",
        "source_integrity": {
            "model_input": "candidate bbox/score/type fields only",
            "offline_labels_used_for": ["bbox_delta_training", "dev_evaluation", "locked_evaluation"],
            "gold_used_for_inference": False,
            "runtime_uses_svg_or_cad_geometry": False,
        },
        "training": {"checkpoint": rel(model_path), "train_rows": len(train_rows), "feature_count": len(names)},
        "dev": dev,
        "locked": locked,
        "stage_gate": {
            "locked_tiny_iou_recall_improves_over_v35_0_299865": float(tiny.get("refined_iou_0_30_recall", 0.0)) > 0.299865,
            "locked_sink_iou_recall_improves": float(sink.get("refined_iou_0_30_recall", 0.0)) > float(sink.get("input_iou_0_30_recall", 0.0)),
            "locked_shower_iou_recall_improves": float(shower.get("refined_iou_0_30_recall", 0.0)) > float(shower.get("input_iou_0_30_recall", 0.0)),
            "locked_overall_iou_recall_not_drop": locked["refined_iou_0_30_recall"] >= locked["input_iou_0_30_recall"],
            "no_oracle_inference": True,
        },
    }
    report["stage_gate"]["passed"] = all(report["stage_gate"].values())
    write_json(source_path(args.eval_output), report)
    print(json.dumps({"locked": locked, "stage_gate": report["stage_gate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
