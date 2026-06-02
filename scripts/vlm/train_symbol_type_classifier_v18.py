#!/usr/bin/env python3
"""Train and apply a raster crop prototype classifier for v18 symbols."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
CROP_DATA = ROOT / "datasets/image_only_symbol_crops_v18"
SYMBOL_DATA = ROOT / "datasets/image_only_symbol_detector_v18"
REPORT = ROOT / "reports/vlm"
CHECKPOINT = ROOT / "checkpoints/symbol_type_classifier_v18"
DETECTOR_PREDICTIONS = REPORT / "symbol_detector_v18_locked_predictions.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def bbox_iou(left: list[float] | None, right: list[float] | None) -> float:
    if not left or not right:
        return 0.0
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    la = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    ra = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return inter / max(la + ra - inter, 1e-9)


def center_covered(pred: list[int], gold: list[int], margin: int = 2) -> bool:
    cx = (gold[0] + gold[2]) / 2.0
    cy = (gold[1] + gold[3]) / 2.0
    return pred[0] - margin <= cx <= pred[2] + margin and pred[1] - margin <= cy <= pred[3] + margin


def crop_feature(image: Image.Image, bbox: list[int], pad: int = 3, size: int = 16) -> np.ndarray:
    width, height = image.size
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(width, x2 + pad)
    y2 = min(height, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        crop = Image.new("L", (size, size), 255)
        bw = bh = area = aspect = fill = 0.0
    else:
        crop = image.crop((x1, y1, x2, y2)).convert("L")
        crop = ImageOps.autocontrast(crop)
        bw = (x2 - x1) / max(width, 1)
        bh = (y2 - y1) / max(height, 1)
        area = bw * bh
        aspect = math.log((x2 - x1) / max(y2 - y1, 1))
        arr0 = np.asarray(crop, dtype=np.uint8)
        fill = float((arr0 <= 205).mean()) if arr0.size else 0.0
    resized = crop.resize((size, size), Image.Resampling.BICUBIC)
    arr = 1.0 - (np.asarray(resized, dtype=np.float32) / 255.0)
    row_density = arr.mean(axis=1)
    col_density = arr.mean(axis=0)
    quads = np.array([
        arr[: size // 2, : size // 2].mean(),
        arr[: size // 2, size // 2 :].mean(),
        arr[size // 2 :, : size // 2].mean(),
        arr[size // 2 :, size // 2 :].mean(),
    ], dtype=np.float32)
    geom = np.array([bw, bh, area, aspect, fill], dtype=np.float32)
    feat = np.concatenate([arr.reshape(-1), row_density, col_density, quads, geom]).astype(np.float32)
    norm = float(np.linalg.norm(feat))
    return feat / norm if norm > 1e-9 else feat


def image_for(path: str, cache: dict[str, Image.Image]) -> Image.Image:
    if path not in cache:
        p = Path(path)
        cache[path] = Image.open(p if p.is_absolute() else ROOT / p).convert("RGB")
    return cache[path]


def train_centroids(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cache: dict[str, Image.Image] = {}
    sums: dict[str, np.ndarray] = {}
    counts = Counter()
    for row in rows:
        label = str(row.get("symbol_type") or "generic_symbol")
        feat = crop_feature(image_for(row["image"], cache), [int(v) for v in row["bbox"]])
        sums[label] = feat if label not in sums else sums[label] + feat
        counts[label] += 1
    centroids = {}
    for label, total in sums.items():
        centroid = total / max(counts[label], 1)
        norm = float(np.linalg.norm(centroid))
        centroids[label] = (centroid / norm if norm > 1e-9 else centroid).astype(np.float32)
    return {
        "labels": sorted(centroids),
        "counts": dict(sorted(counts.items())),
        "centroids": centroids,
    }


def classify(feat: np.ndarray, model: dict[str, Any]) -> tuple[str, float, dict[str, float]]:
    scores = {label: float(np.dot(feat, centroid)) for label, centroid in model["centroids"].items()}
    label = max(scores, key=scores.get)
    ordered = sorted(scores.values(), reverse=True)
    margin = ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0)
    confidence = max(0.01, min(0.99, 0.50 + margin))
    return label, confidence, scores


def evaluate_gold_crops(rows: list[dict[str, Any]], model: dict[str, Any], limit: int | None = None) -> dict[str, Any]:
    cache: dict[str, Image.Image] = {}
    totals = Counter()
    by_type = Counter()
    by_type_hit = Counter()
    selected = rows[:limit] if limit else rows
    for row in selected:
        label = str(row.get("symbol_type") or "generic_symbol")
        pred, _conf, _scores = classify(crop_feature(image_for(row["image"], cache), row["bbox"]), model)
        totals["gold"] += 1
        totals["correct"] += 1 if pred == label else 0
        by_type[label] += 1
        by_type_hit[label] += 1 if pred == label else 0
    return {
        "rows": len(selected),
        "accuracy": round(totals["correct"] / max(totals["gold"], 1), 6),
        "correct": int(totals["correct"]),
        "gold": int(totals["gold"]),
        "per_type_accuracy": {key: round(by_type_hit[key] / max(by_type[key], 1), 6) for key in sorted(by_type)},
    }


def score_gold_crops(rows: list[dict[str, Any]], model: dict[str, Any], limit: int | None = None) -> list[dict[str, Any]]:
    cache: dict[str, Image.Image] = {}
    selected = rows[:limit] if limit else rows
    scored: list[dict[str, Any]] = []
    for row in selected:
        label = str(row.get("symbol_type") or "generic_symbol")
        pred, confidence, scores = classify(crop_feature(image_for(row["image"], cache), row["bbox"]), model)
        ordered = sorted(scores.values(), reverse=True)
        margin = ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0)
        scored.append({
            "id": row.get("id"),
            "label": label,
            "prediction": pred,
            "confidence": round(confidence, 6),
            "margin": round(float(margin), 6),
            "correct": pred == label,
        })
    return scored


def choose_abstain_threshold(
    scored: list[dict[str, Any]],
    precision_floor: float,
    f1_floor: float,
) -> dict[str, Any]:
    if not scored:
        return {
            "enabled": False,
            "confidence_threshold": 1.01,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "coverage": 0.0,
            "reason": "no_dev_scores",
        }
    candidates = sorted({float(row["confidence"]) for row in scored})
    best = None
    gold = len(scored)
    for threshold in candidates:
        selected = [row for row in scored if float(row["confidence"]) >= threshold]
        if not selected:
            continue
        tp = sum(1 for row in selected if row["correct"])
        precision = tp / max(len(selected), 1)
        recall = tp / max(gold, 1)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        item = {
            "confidence_threshold": round(threshold, 6),
            "selected": len(selected),
            "correct": tp,
            "gold": gold,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "coverage": round(len(selected) / max(gold, 1), 6),
        }
        if precision >= precision_floor and f1 >= f1_floor:
            if best is None or (item["coverage"], item["f1"], item["precision"]) > (best["coverage"], best["f1"], best["precision"]):
                best = item
    if best:
        best["enabled"] = True
        best["reason"] = "dev_precision_and_f1_floor_met"
        return best
    # Report the best precision/coverage tradeoff even when disabled.
    tradeoffs = []
    for threshold in candidates:
        selected = [row for row in scored if float(row["confidence"]) >= threshold]
        if not selected:
            continue
        tp = sum(1 for row in selected if row["correct"])
        precision = tp / max(len(selected), 1)
        recall = tp / max(gold, 1)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        tradeoffs.append((precision, f1, len(selected), threshold, tp))
    precision, f1, selected_count, threshold, tp = max(tradeoffs, default=(0.0, 0.0, 0, 1.01, 0))
    return {
        "enabled": False,
        "confidence_threshold": 1.01,
        "best_disabled_threshold": round(float(threshold), 6),
        "selected_at_best_disabled_threshold": int(selected_count),
        "correct_at_best_disabled_threshold": int(tp),
        "precision": round(float(precision), 6),
        "recall": round(float(tp) / max(gold, 1), 6),
        "f1": round(float(f1), 6),
        "coverage": round(float(selected_count) / max(gold, 1), 6),
        "reason": "no_dev_threshold_met_precision_and_f1_floor",
    }


def gold_symbols_by_row(path: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        row["id"]: [
            item for item in (row.get("targets") or {}).get("symbols") or []
            if item.get("bbox") and len(item["bbox"]) == 4
        ]
        for row in load_jsonl(path)
    }


def apply_to_detector(
    detector_rows: list[dict[str, Any]],
    model: dict[str, Any],
    limit_pages: int | None,
    limit_candidates_per_page: int | None,
    export_top_k: int,
    adopt_typed_labels: bool = True,
    type_confidence_threshold: float = 0.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cache: dict[str, Image.Image] = {}
    predictions: list[dict[str, Any]] = []
    routed: list[dict[str, Any]] = []
    selected_rows = detector_rows[:limit_pages] if limit_pages else detector_rows
    for row in selected_rows:
        image = image_for(row["image"], cache)
        candidates = list(row.get("predicted_symbols") or [])
        if limit_candidates_per_page:
            candidates = candidates[:limit_candidates_per_page]
        typed: list[dict[str, Any]] = []
        for cand in candidates:
            item = json.loads(json.dumps(cand))
            label, confidence, scores = classify(crop_feature(image, item["bbox"]), model)
            adopt_this_label = bool(adopt_typed_labels and confidence >= type_confidence_threshold)
            payload = dict(item.get("payload") or {})
            payload.update({
                "weak_typed_symbol_type": label,
                "typed_symbol_type": label if adopt_this_label else "symbol",
                "type_confidence": round(confidence, 6),
                "type_confidence_threshold": round(float(type_confidence_threshold), 6),
                "type_scores_top": dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]),
                "type_model": "symbol_type_classifier_v18_centroid_crop",
                "type_label_adopted": adopt_this_label,
            })
            if adopt_this_label:
                payload["symbol_type"] = label
                item["symbol_type"] = label
                item["semantic_type"] = label
            else:
                payload["symbol_type"] = "symbol"
                item["symbol_type"] = "symbol"
                item["semantic_type"] = "symbol"
            item["confidence"] = round(float(item.get("confidence", 0.0)) * 0.65 + confidence * 0.35, 6)
            item["payload"] = payload
            typed.append(item)
        export = typed[:export_top_k] if export_top_k else typed
        predictions.append({
            "id": row["id"],
            "image": row["image"],
            "predicted_symbols": export,
            "prediction_count_before_export_cap": len(typed),
            "source_integrity": {
                "model_input": "raster_image_only",
                "gold_used_for_inference": False,
            },
        })
        for pred in export:
            routed.append({
                "candidate_id": pred["id"],
                "row_id": row["id"],
                "family": "symbol",
                "route": "symbol_fixture",
                "bbox": pred["bbox"],
                "confidence": pred["confidence"],
                "payload": pred["payload"],
                "source_integrity": {
                    "model_input": "raster_image_only",
                    "gold_used_for_inference": False,
                },
            })
    return predictions, routed


def evaluate_typed_predictions(predictions: list[dict[str, Any]], gold_by_row: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    totals = Counter()
    by_type_gold = Counter()
    by_type_tp = Counter()
    false_positive_sources = Counter()
    misses: list[dict[str, Any]] = []
    for row in predictions:
        preds = row.get("predicted_symbols") or []
        used: set[int] = set()
        typed_preds = [
            pred for pred in preds
            if bool((pred.get("payload") or {}).get("type_label_adopted"))
        ]
        totals["objectness_predicted"] += len(preds)
        totals["typed_predicted"] += len(typed_preds)
        totals["typed_abstained"] += max(0, len(preds) - len(typed_preds))
        for pred in preds:
            false_positive_sources[(pred.get("payload") or {}).get("candidate_kind") or "unknown"] += 1
        for gold_index, gold in enumerate(gold_by_row.get(row["id"], [])):
            gb = [int(v) for v in gold["bbox"]]
            label = str(gold.get("symbol_type") or "generic_symbol")
            by_type_gold[label] += 1
            totals["gold"] += 1
            best_index: int | None = None
            best_key = (-1, 0.0)
            for pred_index, pred in enumerate(preds):
                if pred_index in used:
                    continue
                bbox = [int(v) for v in pred.get("bbox") or []]
                if len(bbox) != 4:
                    continue
                iou = bbox_iou(bbox, gb)
                localized = center_covered(bbox, gb) or iou >= 0.30
                key = (1 if localized else 0, iou)
                if key > best_key:
                    best_key = key
                    best_index = pred_index
            if best_index is None or best_key[0] == 0:
                misses.append({"row_id": row["id"], "gold_index": gold_index, "bbox": gb, "symbol_type": label, "reason": "not_localized"})
                continue
            used.add(best_index)
            pred = preds[best_index]
            totals["localized"] += 1
            if not bool((pred.get("payload") or {}).get("type_label_adopted")):
                misses.append({
                    "row_id": row["id"],
                    "gold_index": gold_index,
                    "bbox": gb,
                    "symbol_type": label,
                    "predicted_bbox": pred.get("bbox"),
                    "reason": "type_abstained",
                })
                continue
            if pred.get("symbol_type") == label:
                totals["typed_tp"] += 1
                by_type_tp[label] += 1
            else:
                misses.append({
                    "row_id": row["id"],
                    "gold_index": gold_index,
                    "bbox": gb,
                    "symbol_type": label,
                    "predicted_symbol_type": pred.get("symbol_type"),
                    "predicted_bbox": pred.get("bbox"),
                    "reason": "type_mismatch",
                })
    precision = totals["typed_tp"] / max(totals["typed_predicted"], 1)
    recall = totals["typed_tp"] / max(totals["gold"], 1)
    center_recall = totals["localized"] / max(totals["gold"], 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "symbol_bbox_center_recall": round(center_recall, 6),
        "typed_label": {
            "true_positive": int(totals["typed_tp"]),
            "predicted": int(totals["typed_predicted"]),
            "gold": int(totals["gold"]),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        },
        "objectness_predicted": int(totals["objectness_predicted"]),
        "typed_predictions_adopted": int(totals["typed_predicted"]),
        "typed_predictions_abstained": int(totals["typed_abstained"]),
        "typed_prediction_coverage": round(totals["typed_predicted"] / max(totals["objectness_predicted"], 1), 6),
        "per_type_typed_recall": {key: round(by_type_tp[key] / max(by_type_gold[key], 1), 6) for key in sorted(by_type_gold)},
        "false_positive_sources_top": dict(false_positive_sources.most_common(12)),
        "miss_examples": misses[:40],
    }


def serializable_model(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "labels": model["labels"],
        "counts": model["counts"],
        "centroids": {label: values.tolist() for label, values in model["centroids"].items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(CROP_DATA))
    parser.add_argument("--detector-predictions", default=str(DETECTOR_PREDICTIONS))
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-dev", type=int, default=None)
    parser.add_argument("--limit-pages", type=int, default=None)
    parser.add_argument("--limit-candidates-per-page", type=int, default=None)
    parser.add_argument("--export-top-k", type=int, default=2500)
    parser.add_argument("--safe-routed-output", default=str(REPORT / "symbol_detector_v18_safe_routed_candidates.jsonl"))
    parser.add_argument("--safe-predictions-output", default=str(REPORT / "symbol_detector_v18_safe_predictions.jsonl"))
    parser.add_argument("--adopt-precision-floor", type=float, default=0.98)
    parser.add_argument("--adopt-f1-floor", type=float, default=0.98)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    data = Path(args.data)
    train_rows = load_jsonl(data / "train.jsonl")
    dev_rows = load_jsonl(data / "dev.jsonl")
    if args.smoke:
        args.limit_train = args.limit_train or 300
        args.limit_dev = args.limit_dev or 200
        args.limit_pages = args.limit_pages or 2
        args.limit_candidates_per_page = args.limit_candidates_per_page or 200
        args.export_top_k = min(args.export_top_k, 200)
    model = train_centroids(train_rows[: args.limit_train] if args.limit_train else train_rows)
    dev_eval = evaluate_gold_crops(dev_rows, model, args.limit_dev)
    dev_scores = score_gold_crops(dev_rows, model, args.limit_dev)
    abstain_policy = choose_abstain_threshold(dev_scores, args.adopt_precision_floor, args.adopt_f1_floor)
    detector_rows = load_jsonl(Path(args.detector_predictions))
    predictions, routed = apply_to_detector(
        detector_rows,
        model,
        args.limit_pages,
        args.limit_candidates_per_page,
        args.export_top_k,
        adopt_typed_labels=True,
        type_confidence_threshold=0.0,
    )
    locked_eval = evaluate_typed_predictions(predictions, gold_symbols_by_row(SYMBOL_DATA / "locked.jsonl"))
    typed_label_adopted = bool(abstain_policy.get("enabled"))
    safe_predictions, safe_routed = apply_to_detector(
        detector_rows,
        model,
        args.limit_pages,
        args.limit_candidates_per_page,
        args.export_top_k,
        adopt_typed_labels=typed_label_adopted,
        type_confidence_threshold=float(abstain_policy.get("confidence_threshold") or 1.01),
    )
    safe_locked_eval = evaluate_typed_predictions(safe_predictions, gold_symbols_by_row(SYMBOL_DATA / "locked.jsonl"))
    success = locked_eval["symbol_bbox_center_recall"] >= 0.60 and typed_label_adopted
    report = {
        "task": "IMG-MOE-V18-P1-007",
        "run_mode": "raster_symbol_crop_centroid_classifier",
        "source_integrity": {
            "model_input": "raster_image_crop",
            "train_gold_use": "prototype_training_only",
            "dev_gold_use": "classifier_validation_only",
            "locked_gold_use": "post_inference_evaluation_only",
            "gold_used_for_inference": False,
        },
        "model": {
            "type": "nearest_centroid_normalized_crop_feature",
            "labels": model["labels"],
            "train_counts": model["counts"],
        },
        "dev_gold_crop_eval": dev_eval,
        "dev_abstain_policy": abstain_policy,
        "locked": locked_eval,
        "locked_safe_export": safe_locked_eval,
        "success_criteria": {
            "locked_symbol_candidate_center_recall_at_least_0_60": locked_eval["symbol_bbox_center_recall"] >= 0.60,
            "dev_abstain_policy_meets_type_floor": typed_label_adopted,
            "locked_safe_export_typed_label_precision_at_least_floor": safe_locked_eval["typed_label"]["precision"] >= args.adopt_precision_floor,
            "locked_safe_export_typed_label_f1_at_least_floor": safe_locked_eval["typed_label"]["f1"] >= args.adopt_f1_floor,
            "per_class_recall_reported": bool(locked_eval["per_type_typed_recall"]),
            "false_positive_sources_bucketed": bool(locked_eval["false_positive_sources_top"]),
        },
        "type_label_adoption": {
            "adopted": typed_label_adopted,
            "precision_floor": args.adopt_precision_floor,
            "f1_floor": args.adopt_f1_floor,
            "confidence_threshold": abstain_policy.get("confidence_threshold"),
            "abstain_policy": abstain_policy,
            "safe_export_policy": "typed labels remain diagnostic weak_typed_symbol_type unless locked precision and F1 meet production floor",
            "runtime_contract": "classifier input is raster crop only; offline labels are used only for training/evaluation",
        },
        "adopted": success,
    }

    CHECKPOINT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.smoke else ""
    write_json(CHECKPOINT / f"prototype_model{suffix}.json", serializable_model(model))
    write_json(REPORT / f"symbol_type_classifier_v18_eval{suffix}.json", report)
    write_jsonl(REPORT / f"symbol_detector_v18_typed_predictions{suffix}.jsonl", predictions)
    write_jsonl(REPORT / f"symbol_detector_v18_typed_routed_candidates{suffix}.jsonl", routed)
    if not args.smoke:
        write_jsonl(Path(args.safe_predictions_output), safe_predictions)
        write_jsonl(Path(args.safe_routed_output), safe_routed)

    print("task IMG-MOE-V18-P1-007 typed")
    print("labels", ",".join(model["labels"]))
    print("dev_crop_accuracy", dev_eval["accuracy"])
    print("locked_center_recall", locked_eval["symbol_bbox_center_recall"])
    print("locked_typed_f1", locked_eval["typed_label"]["f1"])
    print("type_label_adopted", typed_label_adopted)
    print("adopted", success)


if __name__ == "__main__":
    main()
