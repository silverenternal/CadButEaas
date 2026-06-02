#!/usr/bin/env python3
"""P0-26: shrink-prior and lightweight regressor feasibility for P0-24 repair cases."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from train_symbol_box_refiner_v38 import apply_delta
from train_symbol_support_suppression_v35 import load_jsonl, source_path
from train_symbol_tile_detector_v20 import bbox_iou, write_json, write_jsonl

FOCUS_LABELS = {"sink", "shower", "stair", "equipment"}
FOCUS_AREAS = {"tiny_le_64", "small_le_256"}


def stats(box: list[float]) -> dict[str, float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2, "w": w, "h": h, "area": w * h, "aspect": w / h}


def target_delta(candidate: list[float], target: list[float]) -> list[float]:
    c = stats(candidate)
    t = stats(target)
    return [(t["cx"] - c["cx"]) / c["w"], (t["cy"] - c["cy"]) / c["h"], (t["w"] / c["w"]) - 1.0, (t["h"] / c["h"]) - 1.0]


def focus(row: dict[str, Any]) -> bool:
    return str(row.get("label") or "") in FOCUS_LABELS and str(row.get("area_bucket") or "") in FOCUS_AREAS


def feature_dict(row: dict[str, Any]) -> dict[str, float]:
    box = [float(v) for v in row["candidate_bbox"]]
    s = stats(box)
    label = str(row.get("candidate_label") or row.get("label") or "")
    candidate_area_bucket = "tiny_le_64" if s["area"] <= 64.0 else "small_le_256" if s["area"] <= 256.0 else "medium_or_larger"
    return {
        "score": float(row.get("candidate_score") or 0.0),
        "w": s["w"], "h": s["h"], "area": s["area"], "aspect": s["aspect"],
        "log_area": float(np.log(max(s["area"], 1e-6))),
        # Runtime-safe only: do not include gold-derived audit features such as
        # pred_area_over_target, input_iou, or center_dist_over_diag.
        "label_sink": 1.0 if label == "sink" else 0.0,
        "label_shower": 1.0 if label == "shower" else 0.0,
        "label_stair": 1.0 if label == "stair" else 0.0,
        "label_equipment": 1.0 if label == "equipment" else 0.0,
        "candidate_area_tiny": 1.0 if candidate_area_bucket == "tiny_le_64" else 0.0,
        "candidate_area_small": 1.0 if candidate_area_bucket == "small_le_256" else 0.0,
    }


def names_for(rows: list[dict[str, Any]]) -> list[str]:
    names = set()
    for row in rows:
        names.update(feature_dict(row).keys())
    return sorted(names)


def vector(row: dict[str, Any], names: list[str]) -> list[float]:
    feats = feature_dict(row)
    return [float(feats.get(name, 0.0)) for name in names]


def median_prior(train_rows: list[dict[str, Any]], key_mode: str) -> dict[str, list[float]]:
    groups: dict[str, list[list[float]]] = defaultdict(list)
    for row in train_rows:
        key = prior_key(row, key_mode)
        groups[key].append(target_delta(row["candidate_bbox"], row["target_bbox"]))
    out = {}
    for key, vals in groups.items():
        out[key] = [float(v) for v in np.median(np.asarray(vals, dtype=np.float64), axis=0)]
    out["__global__"] = [float(v) for v in np.median(np.asarray([target_delta(r["candidate_bbox"], r["target_bbox"]) for r in train_rows], dtype=np.float64), axis=0)]
    return out


def prior_key(row: dict[str, Any], mode: str) -> str:
    if mode == "label_area":
        return f"{row.get('label')}|{row.get('area_bucket')}"
    if mode == "area":
        return str(row.get("area_bucket") or "unknown")
    return "__global__"


def quant(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0}
    arr = np.asarray(vals, dtype=np.float64)
    return {"count": int(arr.size), "mean": round(float(arr.mean()), 6), "p50": round(float(np.quantile(arr, .5)), 6), "p90": round(float(np.quantile(arr, .9)), 6)}


def eval_deltas(rows: list[dict[str, Any]], deltas: list[list[float]], clip: float, name: str) -> dict[str, Any]:
    counters = Counter()
    by_label = defaultdict(Counter)
    by_area = defaultdict(Counter)
    input_ious, refined_ious = [], []
    for row, delta in zip(rows, deltas, strict=True):
        box = [float(v) for v in row["candidate_bbox"]]
        target = [float(v) for v in row["target_bbox"]]
        refined = apply_delta(box, delta, clip)
        input_iou = bbox_iou(box, target)
        refined_iou = bbox_iou(refined, target)
        input_ious.append(input_iou)
        refined_ious.append(refined_iou)
        counters["rows"] += 1
        counters["input_hit"] += int(input_iou >= 0.30)
        counters["refined_hit"] += int(refined_iou >= 0.30)
        counters["improved"] += int(refined_iou > input_iou)
        counters["worse"] += int(refined_iou < input_iou)
        for group in [by_label[str(row.get("label") or "")], by_area[str(row.get("area_bucket") or "")]]:
            group["rows"] += 1
            group["input_hit"] += int(input_iou >= 0.30)
            group["refined_hit"] += int(refined_iou >= 0.30)
    n = max(counters["rows"], 1)
    def rates(c: Counter) -> dict[str, Any]:
        m = max(c["rows"], 1)
        return {"rows": int(c["rows"]), "input_hit_rate": round(c["input_hit"] / m, 6), "refined_hit_rate": round(c["refined_hit"] / m, 6), "hit_gain": int(c["refined_hit"] - c["input_hit"])}
    return {
        "name": name,
        "rows": int(counters["rows"]),
        "input_hit_rate": round(counters["input_hit"] / n, 6),
        "refined_hit_rate": round(counters["refined_hit"] / n, 6),
        "hit_gain": int(counters["refined_hit"] - counters["input_hit"]),
        "improved_rate": round(counters["improved"] / n, 6),
        "worse_rate": round(counters["worse"] / n, 6),
        "input_iou": quant(input_ious),
        "refined_iou": quant(refined_ious),
        "by_label": {k: rates(v) for k, v in sorted(by_label.items())},
        "by_area": {k: rates(v) for k, v in sorted(by_area.items())},
    }


def train_regressor(train_rows: list[dict[str, Any]], names: list[str], kind: str):
    x = np.asarray([vector(row, names) for row in train_rows], dtype=np.float32)
    y = np.asarray([target_delta(row["candidate_bbox"], row["target_bbox"]) for row in train_rows], dtype=np.float32)
    if kind == "hgb":
        model = MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=120, learning_rate=0.04, max_leaf_nodes=12, l2_regularization=0.1, random_state=20260513))
    elif kind == "rf":
        model = RandomForestRegressor(n_estimators=200, min_samples_leaf=8, max_features=0.8, random_state=20260513, n_jobs=-1)
    else:
        model = ExtraTreesRegressor(n_estimators=240, min_samples_leaf=5, max_features=0.8, random_state=20260513, n_jobs=-1)
    model.fit(x, y)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="datasets/symbol_localization_repair_p024/dev_center_low_iou.jsonl")
    parser.add_argument("--test", default="datasets/symbol_localization_repair_p024/smoke_center_low_iou.jsonl")
    parser.add_argument("--clip", type=float, default=0.9)
    parser.add_argument("--output", default="reports/vlm/symbol_localization_repair_p026_eval.json")
    parser.add_argument("--checkpoint", default="checkpoints/symbol_localization_repair_p026/model.joblib")
    args = parser.parse_args()
    train_all = load_jsonl(source_path(args.train))
    test_all = load_jsonl(source_path(args.test))
    train_rows = [row for row in train_all if focus(row)]
    test_rows = [row for row in test_all if focus(row)]
    reports = []
    for mode in ["global", "area", "label_area"]:
        prior = median_prior(train_rows, mode)
        deltas = [prior.get(prior_key(row, mode), prior["__global__"]) for row in test_rows]
        reports.append(eval_deltas(test_rows, deltas, args.clip, f"median_prior_{mode}"))
    names = names_for(train_rows)
    models = {}
    for kind in ["extra_trees", "rf", "hgb"]:
        model = train_regressor(train_rows, names, kind)
        deltas = model.predict(np.asarray([vector(row, names) for row in test_rows], dtype=np.float32)).tolist()
        reports.append(eval_deltas(test_rows, deltas, args.clip, f"regressor_{kind}"))
        models[kind] = model
    best = max(reports, key=lambda r: (r["hit_gain"], r["refined_iou"]["mean"])) if reports else None
    source_path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "feature_names": names, "clip": args.clip, "train": args.train, "test": args.test, "best": best}, source_path(args.checkpoint))
    output = {
        "version": "symbol_localization_repair_p026",
        "train": args.train,
        "test": args.test,
        "train_focus_rows": len(train_rows),
        "test_focus_rows": len(test_rows),
        "clip": args.clip,
        "reports": reports,
        "decision": {
            "best_method": best["name"] if best else None,
            "best_hit_gain": best["hit_gain"] if best else None,
            "best_refined_hit_rate": best["refined_hit_rate"] if best else None,
            "baseline_input_hit_rate": best["input_hit_rate"] if best else None,
            "recommendation": "runtime_safe_features_only_integrate_if_page_level_gain_exceeds_budget_cost_else_pivot_to_proposal_generation",
        },
        "source_integrity": {"gold_used_for_inference": False, "runtime_uses_svg_or_cad_geometry": False, "runtime_features": ["candidate bbox", "candidate score", "predicted label", "candidate area bucket derived from candidate bbox"], "offline_labels_used_for": ["p026_refiner_training_and_eval"], "final_quality_claim_allowed": False},
    }
    write_json(source_path(args.output), output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
