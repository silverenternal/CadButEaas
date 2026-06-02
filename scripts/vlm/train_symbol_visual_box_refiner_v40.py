#!/usr/bin/env python3
"""Train a smoke visual crop bbox refiner for v40."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from PIL import Image, ImageOps
from sklearn.ensemble import ExtraTreesRegressor

from train_symbol_tile_detector_v20 import bbox_iou, rel, write_json


ROOT = Path(__file__).resolve().parents[2]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def source_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def image_features(path: str) -> list[float]:
    with Image.open(source_path(path)) as img:
        gray = ImageOps.autocontrast(img.convert("L").resize((32, 32), Image.Resampling.BILINEAR))
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    dark = 1.0 - arr
    ys, xs = np.mgrid[0:32, 0:32].astype(np.float32)
    weight = dark + 1e-4
    total = float(weight.sum())
    cx = float((xs * weight).sum() / total) / 32.0
    cy = float((ys * weight).sum() / total) / 32.0
    return [
        float(arr.mean()),
        float(arr.std()),
        float(dark.mean()),
        float((arr < 0.35).mean()),
        cx,
        cy,
        float(((xs / 31.0 - cx) ** 2 * weight).sum() / total),
        float(((ys / 31.0 - cy) ** 2 * weight).sum() / total),
    ]


def geom_features(item: dict[str, Any]) -> list[float]:
    prop = item["proposal"]
    box = [float(v) for v in prop["bbox"]]
    width = max(1e-6, box[2] - box[0])
    height = max(1e-6, box[3] - box[1])
    score = float(prop.get("score") or 0.0)
    label = str(prop.get("label") or "")
    return [
        width,
        height,
        width * height,
        width / height,
        score,
        1.0 if label == "sink" else 0.0,
        1.0 if label == "shower" else 0.0,
        1.0 if label == "equipment" else 0.0,
        1.0 if label == "stair" else 0.0,
    ]


def features(item: dict[str, Any]) -> list[float]:
    return geom_features(item) + image_features((item["crop"] or {})["path"])


def target(item: dict[str, Any]) -> list[float]:
    return [float(v) for v in (item["target"] or {}).get("offset") or [0, 0, 0, 0]]


def apply_delta(box: list[float], delta: list[float], clip: float) -> list[float]:
    w = max(1e-6, box[2] - box[0])
    h = max(1e-6, box[3] - box[1])
    d = [max(-clip, min(clip, float(v))) for v in delta]
    out = [box[0] + d[0] * w, box[1] + d[1] * h, box[2] + d[2] * w, box[3] + d[3] * h]
    if out[2] <= out[0] + 1:
        out[2] = out[0] + 1
    if out[3] <= out[1] + 1:
        out[3] = out[1] + 1
    return out


def evaluate(model: Any, rows: list[dict[str, Any]], split: str, clip: float) -> dict[str, Any]:
    x = np.asarray([features(row) for row in rows], dtype=np.float32)
    pred = model.predict(x)
    totals = Counter()
    by_area = defaultdict(Counter)
    by_label = defaultdict(Counter)
    for row, delta in zip(rows, pred, strict=True):
        box = [float(v) for v in row["proposal"]["bbox"]]
        gold = [float(v) for v in row["target"]["bbox"]]
        refined = apply_delta(box, list(delta), clip)
        bi = bbox_iou(box, gold)
        ri = bbox_iou(refined, gold)
        label = str(row["target"].get("label") or row["proposal"].get("label") or "")
        area = str(row["target"].get("area_bucket") or "")
        totals["rows"] += 1
        totals["input_hit"] += int(bi >= 0.30)
        totals["refined_hit"] += int(ri >= 0.30)
        totals["improved"] += int(ri > bi)
        totals["worse"] += int(ri < bi)
        for bucket in [by_area[area], by_label[label]]:
            bucket["rows"] += 1
            bucket["input_hit"] += int(bi >= 0.30)
            bucket["refined_hit"] += int(ri >= 0.30)
    def rates(c: Counter) -> dict[str, float]:
        n = max(int(c["rows"]), 1)
        return {"rows": int(c["rows"]), "input_iou_0_30_recall": round(c["input_hit"] / n, 6), "refined_iou_0_30_recall": round(c["refined_hit"] / n, 6)}
    n = max(int(totals["rows"]), 1)
    return {
        "split": split,
        "rows": int(totals["rows"]),
        "input_iou_0_30_recall": round(totals["input_hit"] / n, 6),
        "refined_iou_0_30_recall": round(totals["refined_hit"] / n, 6),
        "improved_rate": round(totals["improved"] / n, 6),
        "worse_rate": round(totals["worse"] / n, 6),
        "by_area": {k: rates(v) for k, v in sorted(by_area.items())},
        "by_label": {k: rates(v) for k, v in sorted(by_label.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="datasets/symbol_visual_box_refiner_v40")
    parser.add_argument("--output-dir", default="checkpoints/symbol_visual_box_refiner_v40")
    parser.add_argument("--eval-output", default="reports/vlm/symbol_visual_box_refiner_v40_locked_eval.json")
    parser.add_argument("--n-estimators", type=int, default=160)
    parser.add_argument("--clip", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=20260512)
    args = parser.parse_args()
    data_dir = source_path(args.data_dir)
    train = load_jsonl(data_dir / "train.jsonl")
    dev = load_jsonl(data_dir / "dev.jsonl")
    locked = load_jsonl(data_dir / "locked.jsonl")
    x = np.asarray([features(row) for row in train], dtype=np.float32)
    y = np.asarray([target(row) for row in train], dtype=np.float32)
    model = ExtraTreesRegressor(n_estimators=args.n_estimators, min_samples_leaf=2, max_features="sqrt", n_jobs=-1, random_state=args.seed)
    model.fit(x, y)
    out = source_path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "model.joblib"
    joblib.dump({"model": model, "args": vars(args), "feature_type": "geom_plus_crop_intensity_stats"}, model_path)
    dev_eval = evaluate(model, dev, "dev", args.clip)
    locked_eval = evaluate(model, locked, "locked", args.clip)
    report = {
        "version": "symbol_visual_box_refiner_v40_locked_eval",
        "task": "P1-11-visual-crop-box-refiner-v40",
        "claim_boundary": "Smoke visual crop refiner using crop intensity statistics plus candidate geometry; crop pixels are runtime raster input.",
        "source_integrity": {"model_input": "raster crop pixels plus candidate bbox/score/type", "gold_used_for_inference": False, "runtime_uses_svg_or_cad_geometry": False},
        "training": {"checkpoint": rel(model_path), "train_rows": len(train)},
        "dev": dev_eval,
        "locked": locked_eval,
        "stage_gate": {
            "locked_iou_recall_improves": locked_eval["refined_iou_0_30_recall"] > locked_eval["input_iou_0_30_recall"],
            "locked_tiny_iou_recall_improves": locked_eval["by_area"].get("tiny_le_64", {}).get("refined_iou_0_30_recall", 0.0) > locked_eval["by_area"].get("tiny_le_64", {}).get("input_iou_0_30_recall", 0.0),
            "locked_sink_iou_recall_improves": locked_eval["by_label"].get("sink", {}).get("refined_iou_0_30_recall", 0.0) > locked_eval["by_label"].get("sink", {}).get("input_iou_0_30_recall", 0.0),
            "no_oracle_inference": True,
        },
    }
    report["stage_gate"]["passed"] = all(report["stage_gate"].values())
    write_json(source_path(args.eval_output), report)
    print(json.dumps({"locked": locked_eval, "stage_gate": report["stage_gate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
