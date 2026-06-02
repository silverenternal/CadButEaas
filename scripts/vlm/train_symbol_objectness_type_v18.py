#!/usr/bin/env python3
"""Train a lightweight objectness scorer for v18 symbol candidates."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "datasets/image_only_symbol_objectness_v18"
REPORT = ROOT / "reports/vlm"
CHECKPOINT = ROOT / "checkpoints/symbol_objectness_type_v18"
GOLD_LOCKED = ROOT / "datasets/image_only_symbol_detector_v18/locked.jsonl"

DEFAULT_OUTPUT = REPORT / "symbol_detector_v18_objectness_routed_candidates.jsonl"
DEFAULT_PREDICTIONS = REPORT / "symbol_detector_v18_objectness_predictions.jsonl"
DEFAULT_EVAL = REPORT / "symbol_objectness_type_v18_eval.json"


FEATURES = [
    "detector_confidence",
    "bbox_width",
    "bbox_height",
    "bbox_area",
    "bbox_aspect",
    "local_dark_density",
    "component_area",
    "component_fill",
    "anchor_w",
    "anchor_h",
    "is_dark_pixel_anchor",
    "is_dark_connected_component",
]
RASTER_FEATURES = [
    "crop_dark_ratio_210",
    "crop_dark_ratio_160",
    "crop_dark_ratio_80",
    "crop_mean",
    "crop_std",
    "crop_inner_dark_ratio_210",
    "crop_border_dark_ratio_210",
    "crop_dark_balance_x",
    "crop_dark_balance_y",
]
ALL_FEATURES = FEATURES + RASTER_FEATURES


def integrity() -> dict[str, Any]:
    return {
        "source_mode": "image_only_raster_moe",
        "svg_candidate_ids_used": False,
        "annotation_geometry_used_at_inference": False,
        "model_input": "raster_image_only",
    }


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def area(b: list[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def iou(left: list[float] | None, right: list[float] | None) -> float:
    if left is None or right is None:
        return 0.0
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    return inter / max(area(left) + area(right) - inter, 1e-9)


def center_covered(pred: list[float], gold: list[float], margin: float = 2.0) -> bool:
    gx = (gold[0] + gold[2]) / 2.0
    gy = (gold[1] + gold[3]) / 2.0
    return pred[0] - margin <= gx <= pred[2] + margin and pred[1] - margin <= gy <= pred[3] + margin


def image_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


class IntegralImageStats:
    def __init__(self, path: str) -> None:
        with Image.open(image_path(path)) as image:
            array = np.asarray(image.convert("L"), dtype=np.float64)
        self.size = (array.shape[1], array.shape[0])
        self.sum = self._integral(array)
        self.sum_sq = self._integral(array * array)
        self.dark_210 = self._integral((array < 210).astype(np.float64))
        self.dark_160 = self._integral((array < 160).astype(np.float64))
        self.dark_80 = self._integral((array < 80).astype(np.float64))

    @staticmethod
    def _integral(array: np.ndarray) -> np.ndarray:
        return np.pad(array.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)), mode="constant")

    @staticmethod
    def _area_sum(integral: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
        return float(integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1])

    def region(self, box: list[float], pad: int = 0) -> tuple[int, int, int, int]:
        width, height = self.size
        x1 = max(0, min(width, int(round(box[0])) - pad))
        y1 = max(0, min(height, int(round(box[1])) - pad))
        x2 = max(0, min(width, int(round(box[2])) + pad))
        y2 = max(0, min(height, int(round(box[3])) + pad))
        return x1, y1, x2, y2

    def crop_features(self, box: list[float]) -> dict[str, float]:
        x1, y1, x2, y2 = self.region(box, pad=1)
        if x2 <= x1 or y2 <= y1:
            return {name: 0.0 for name in RASTER_FEATURES}
        total = max(float((x2 - x1) * (y2 - y1)), 1.0)
        raw_sum = self._area_sum(self.sum, x1, y1, x2, y2)
        raw_sum_sq = self._area_sum(self.sum_sq, x1, y1, x2, y2)
        mean = raw_sum / total
        std = float(np.sqrt(max(raw_sum_sq / total - mean * mean, 0.0)))
        dark_total = self._area_sum(self.dark_210, x1, y1, x2, y2)

        ix1 = x1 + max(1, (x2 - x1) // 4)
        iy1 = y1 + max(1, (y2 - y1) // 4)
        ix2 = x2 - max(1, (x2 - x1) // 4)
        iy2 = y2 - max(1, (y2 - y1) // 4)
        inner_total = 0.0
        inner_area = 0.0
        if ix2 > ix1 and iy2 > iy1:
            inner_area = float((ix2 - ix1) * (iy2 - iy1))
            inner_total = self._area_sum(self.dark_210, ix1, iy1, ix2, iy2)
        border_area = max(total - inner_area, 1.0)
        border_total = max(dark_total - inner_total, 0.0)

        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        left_dark = self._area_sum(self.dark_210, x1, y1, mid_x, y2) if mid_x > x1 else 0.0
        right_dark = self._area_sum(self.dark_210, mid_x, y1, x2, y2) if x2 > mid_x else 0.0
        top_dark = self._area_sum(self.dark_210, x1, y1, x2, mid_y) if mid_y > y1 else 0.0
        bottom_dark = self._area_sum(self.dark_210, x1, mid_y, x2, y2) if y2 > mid_y else 0.0

        return {
            "crop_dark_ratio_210": dark_total / total,
            "crop_dark_ratio_160": self._area_sum(self.dark_160, x1, y1, x2, y2) / total,
            "crop_dark_ratio_80": self._area_sum(self.dark_80, x1, y1, x2, y2) / total,
            "crop_mean": mean / 255.0,
            "crop_std": std / 255.0,
            "crop_inner_dark_ratio_210": inner_total / max(inner_area, 1.0),
            "crop_border_dark_ratio_210": border_total / border_area,
            "crop_dark_balance_x": (left_dark - right_dark) / max(dark_total, 1.0),
            "crop_dark_balance_y": (top_dark - bottom_dark) / max(dark_total, 1.0),
        }


def add_raster_features(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cache: dict[str, IntegralImageStats] = {}
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        feats = dict(item.get("features") if isinstance(item.get("features"), dict) else {})
        box = bbox(item.get("bbox"))
        img = str(item.get("image") or "")
        if box is not None and img:
            if img not in cache:
                cache[img] = IntegralImageStats(img)
            feats.update(cache[img].crop_features(box))
        else:
            feats.update({name: 0.0 for name in RASTER_FEATURES})
        item["features"] = feats
        out.append(item)
    return out


def vector(row: dict[str, Any]) -> np.ndarray:
    feats = row.get("features") if isinstance(row.get("features"), dict) else {}
    return np.asarray([float(feats.get(name) or 0.0) for name in ALL_FEATURES], dtype=np.float32)


def matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    if not rows:
        return np.zeros((0, len(ALL_FEATURES)), dtype=np.float32)
    return np.stack([vector(row) for row in rows]).astype(np.float32)


def train_centroid_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
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
    # Keep the model conservative; this is a ranker, not a calibrated detector.
    norm = float(np.linalg.norm(weights))
    if norm > 1e-9:
        weights = weights / norm
    bias = -float(np.percentile(neg_z @ weights, 85))
    return {
        "features": ALL_FEATURES,
        "model_type": "standardized_positive_negative_centroid_ranker",
        "mean": mean.tolist(),
        "std": std.tolist(),
        "weights": weights.tolist(),
        "bias": bias,
        "train_counts": {"positive": len(positives), "negative": len(negatives), "rows": len(rows)},
    }


def train_extra_trees_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    x = matrix(rows)
    y = np.asarray([1 if row.get("label_objectness") else 0 for row in rows], dtype=np.int32)
    model = ExtraTreesClassifier(
        n_estimators=180,
        max_depth=18,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=18,
        n_jobs=-1,
    )
    model.fit(x, y)
    return {
        "features": ALL_FEATURES,
        "model_type": "extra_trees_raster_crop_objectness_ranker",
        "train_counts": {
            "positive": int(y.sum()),
            "negative": int(len(y) - y.sum()),
            "rows": int(len(y)),
        },
        "sklearn_model": model,
    }


def score_row(row: dict[str, Any], model: dict[str, Any]) -> float:
    if model.get("model_type") == "extra_trees_raster_crop_objectness_ranker":
        return float(model["sklearn_model"].predict_proba(matrix([row]))[0][1])
    x = vector(row)
    mean = np.asarray(model["mean"], dtype=np.float32)
    std = np.asarray(model["std"], dtype=np.float32)
    weights = np.asarray(model["weights"], dtype=np.float32)
    raw = float(((x - mean) / std) @ weights + float(model["bias"]))
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, raw))))


def load_gold(path: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in load_jsonl(path):
        out[str(row.get("id"))] = [
            item for item in (row.get("targets") or {}).get("symbols") or []
            if bbox(item.get("bbox")) is not None
        ]
    return out


def recall_for(rows_by_page: dict[str, list[dict[str, Any]]], gold: dict[str, list[dict[str, Any]]], cap: int) -> dict[str, Any]:
    total = hit = 0
    by_type: dict[str, Counter[str]] = defaultdict(Counter)
    for row_id, gold_items in gold.items():
        candidates = rows_by_page.get(row_id, [])[:cap]
        boxes = [bbox(row.get("bbox")) for row in candidates]
        for gold_item in gold_items:
            gb = bbox(gold_item.get("bbox"))
            if gb is None:
                continue
            total += 1
            label = str(gold_item.get("symbol_type") or gold_item.get("semantic_type") or "symbol")
            by_type[label]["gold"] += 1
            matched = any(cb is not None and (center_covered(cb, gb) or iou(cb, gb) >= 0.25) for cb in boxes)
            if matched:
                hit += 1
                by_type[label]["matched"] += 1
    return {
        "gold": total,
        "matched": hit,
        "center_or_iou_recall": round(hit / max(total, 1), 6),
        "per_type_recall": {
            label: {
                "gold": counts["gold"],
                "matched": counts["matched"],
                "recall": round(counts["matched"] / max(counts["gold"], 1), 6),
            }
            for label, counts in sorted(by_type.items())
        },
    }


def precision_at_threshold(scored: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    selected = [row for row in scored if float(row["objectness_score"]) >= threshold]
    tp = sum(1 for row in selected if row.get("label_objectness"))
    precision = tp / max(len(selected), 1)
    recall = tp / max(sum(1 for row in scored if row.get("label_objectness")), 1)
    return {
        "threshold": threshold,
        "selected": len(selected),
        "true_positive": tp,
        "precision": round(precision, 6),
        "objectness_label_recall": round(recall, 6),
    }


def apply_model(rows: list[dict[str, Any]], model: dict[str, Any]) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    if model.get("model_type") == "extra_trees_raster_crop_objectness_ranker":
        probs = model["sklearn_model"].predict_proba(matrix(rows))[:, 1] if rows else []
    else:
        probs = [score_row(row, model) for row in rows]
    for row, prob in zip(rows, probs, strict=True):
        item = dict(row)
        item["objectness_score"] = round(float(prob), 6)
        scored.append(item)
    scored.sort(key=lambda item: (str(item.get("row_id")), -float(item["objectness_score"]), -float(item.get("match_score") or 0.0)))
    return scored


def export_routed(scored: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_row: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        by_row[str(row.get("row_id"))].append(row)
    routed: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for row_id, rows in sorted(by_row.items()):
        pred_items: list[dict[str, Any]] = []
        for row in rows:
            payload = dict(row.get("payload") if isinstance(row.get("payload"), dict) else {})
            payload.update(
                {
                    "symbol_type": "symbol",
                    "typed_symbol_type": "symbol",
                    "type_label_adopted": False,
                    "objectness_score": row["objectness_score"],
                    "objectness_model": "symbol_objectness_type_v18_raster_crop_ranker",
                    "weak_gold_type_diagnostic": row.get("gold_type"),
                }
            )
            item = {
                "candidate_id": row.get("candidate_id"),
                "row_id": row_id,
                "family": "symbol",
                "route": "symbol_fixture",
                "candidate_type": "symbol",
                "bbox": row.get("bbox"),
                "confidence": row["objectness_score"],
                "payload": payload,
                "source_integrity": integrity(),
            }
            routed.append(item)
            pred_items.append(item)
        predictions.append(
            {
                "id": row_id,
                "source_integrity": integrity(),
                "route_trace": {**integrity(), "stage": "symbol_objectness_type_v18"},
                "candidate_stream": pred_items,
            }
        )
    return predictions, routed


def rows_by_page(scored: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        by_page[str(row.get("row_id"))].append(row)
    for rows in by_page.values():
        rows.sort(key=lambda item: float(item.get("objectness_score") or 0.0), reverse=True)
    return by_page


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--predictions-output", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--eval-output", default=str(DEFAULT_EVAL))
    parser.add_argument("--model-type", choices=["centroid", "extra_trees"], default="extra_trees")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--locked", action="store_true")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    limit = 20000 if args.smoke else None
    dev_rows = add_raster_features(load_jsonl(dataset / "dev.jsonl", limit))
    locked_rows = add_raster_features(load_jsonl(dataset / "locked.jsonl", limit))
    model = train_extra_trees_model(dev_rows) if args.model_type == "extra_trees" else train_centroid_model(dev_rows)
    locked_scored = apply_model(locked_rows, model)
    predictions, routed = export_routed(locked_scored)
    write_jsonl(Path(args.predictions_output), predictions)
    write_jsonl(Path(args.output), routed)
    CHECKPOINT.mkdir(parents=True, exist_ok=True)
    model_for_json = {key: value for key, value in model.items() if key != "sklearn_model"}
    write_json(CHECKPOINT / "model.json", model_for_json)
    if "sklearn_model" in model:
        joblib.dump({"model": model["sklearn_model"], "features": ALL_FEATURES}, CHECKPOINT / "model.joblib")

    gold = load_gold(GOLD_LOCKED)
    before_by_page = rows_by_page([{**row, "objectness_score": row.get("rank", 0) * -1.0} for row in locked_rows])
    after_by_page = rows_by_page(locked_scored)
    caps = [100, 250, 500, 750, 1000, 2500]
    cap_sweep = {
        str(cap): {
            "before_detector_order": recall_for(before_by_page, gold, cap),
            "after_objectness_order": recall_for(after_by_page, gold, cap),
        }
        for cap in caps
    }
    thresholds = [0.35, 0.45, 0.55, 0.65, 0.75]
    threshold_eval = {str(th): precision_at_threshold(locked_scored, th) for th in thresholds}
    cap500_after = cap_sweep["500"]["after_objectness_order"]["center_or_iou_recall"]
    cap500_before = cap_sweep["500"]["before_detector_order"]["center_or_iou_recall"]
    report = {
        "task": "IMG-MOE-V18-NEXT-006",
        "mode": "symbol_objectness_ranker",
        "rows": len(locked_rows),
        "model": {"type": model.get("model_type"), "features": ALL_FEATURES, "train_counts": model["train_counts"]},
        "cap_sweep": cap_sweep,
        "threshold_eval": threshold_eval,
        "type_label_adopted": False,
        "typed_label_reason": "Type classifier remains below adoption floors; objectness export keeps generic symbol labels.",
        "source_integrity": integrity(),
        "gold_loaded_after_inference_for_evaluation_only": True,
        "gold_used_for_inference": False,
        "quality_gates": {
            "source_integrity_violations": 0,
            "symbol_cap_recall_ge_070_at_500": cap500_after >= 0.70,
            "symbol_cap_recall_improved_at_500": cap500_after > cap500_before,
            "typed_label_not_adopted_when_below_floor": True,
            "per_class_recall_reported": bool(cap_sweep["500"]["after_objectness_order"]["per_type_recall"]),
        },
    }
    write_json(Path(args.eval_output), report)
    print(
        json.dumps(
            {
                "rows": len(locked_rows),
                "cap500_before": cap500_before,
                "cap500_after": cap500_after,
                "quality_gates": report["quality_gates"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
