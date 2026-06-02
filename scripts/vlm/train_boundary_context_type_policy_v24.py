#!/usr/bin/env python3
"""Train a page-context boundary candidate policy for v24 YOLO proposals.

The policy is deliberately tabular and auditable.  It uses only raster-derived
candidate stream fields at inference time: bbox geometry, YOLO hint/confidence,
duplicate support, and local same-axis/neighbor support.  Gold boxes are used
only offline for supervised labels and evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report

from apply_boundary_proposals_with_graph_node_gnn_v24 import (
    BOUNDARY_TO_GRAPH_LABEL,
    LABELS,
    bbox,
    center,
    center_covered,
    iou,
    load_jsonl,
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[2]
ALL_LABELS = ["background", *LABELS]
FEATURE_NAMES = [
    "hint_hard_wall",
    "hint_door",
    "hint_window",
    "proposal_confidence",
    "conf_rank_pct",
    "bbox_width_norm",
    "bbox_height_norm",
    "bbox_area_norm",
    "bbox_aspect_log",
    "bbox_length_norm",
    "bbox_thickness_norm",
    "orientation_horizontal",
    "orientation_vertical",
    "duplicate_iou_050",
    "duplicate_iou_070",
    "same_hint_iou_050",
    "same_axis_near_8",
    "same_axis_near_16",
    "cross_axis_near_16",
    "hard_wall_overlap_iou_010",
    "opening_hint_neighbor_support",
]


def gold_items(row: dict[str, Any]) -> list[dict[str, Any]]:
    items = []
    for target in (row.get("targets") or {}).get("boxes") or []:
        b = bbox(target.get("bbox"))
        label = BOUNDARY_TO_GRAPH_LABEL.get(str(target.get("label")), str(target.get("label")))
        if b is not None and label in LABELS:
            items.append({"bbox": b, "label": label, "target_id": str(target.get("target_id") or "")})
    return items


def load_gold(path: Path, limit: int | None = None) -> dict[str, list[dict[str, Any]]]:
    return {str(row.get("id")): gold_items(row) for row in load_jsonl(path, limit)}


def box_metrics(box: list[float], image_size: list[int] | None) -> tuple[float, float, float, float, float, float, float]:
    width = max(box[2] - box[0], 1e-6)
    height = max(box[3] - box[1], 1e-6)
    image_w = float(image_size[0]) if image_size else 1.0
    image_h = float(image_size[1]) if image_size else 1.0
    image_area = max(image_w * image_h, 1.0)
    return (
        width / max(image_w, 1.0),
        height / max(image_h, 1.0),
        (width * height) / image_area,
        math.log(width / height),
        max(width, height) / max(math.hypot(image_w, image_h), 1.0),
        min(width, height) / max(math.hypot(image_w, image_h), 1.0),
        1.0 if width >= height else 0.0,
    )


def is_same_axis(left: list[float], right: list[float]) -> bool:
    return ((left[2] - left[0]) >= (left[3] - left[1])) == ((right[2] - right[0]) >= (right[3] - right[1]))


def axis_gap(left: list[float], right: list[float]) -> float:
    lx, ly = center(left)
    rx, ry = center(right)
    if (left[2] - left[0]) >= (left[3] - left[1]):
        return abs(ly - ry)
    return abs(lx - rx)


def candidate_features(row: dict[str, Any], cap: int) -> list[list[float]]:
    candidates = (row.get("candidate_stream") or [])[:cap]
    boxes = [bbox(candidate.get("bbox")) or [0.0, 0.0, 1.0, 1.0] for candidate in candidates]
    if not candidates:
        return []
    box_array = np.asarray(boxes, dtype=np.float32)
    widths = np.maximum(box_array[:, 2] - box_array[:, 0], 1e-6)
    heights = np.maximum(box_array[:, 3] - box_array[:, 1], 1e-6)
    horizontal_flags = widths >= heights
    areas = widths * heights
    ix1 = np.maximum(box_array[:, None, 0], box_array[None, :, 0])
    iy1 = np.maximum(box_array[:, None, 1], box_array[None, :, 1])
    ix2 = np.minimum(box_array[:, None, 2], box_array[None, :, 2])
    iy2 = np.minimum(box_array[:, None, 3], box_array[None, :, 3])
    inter = np.maximum(ix2 - ix1, 0.0) * np.maximum(iy2 - iy1, 0.0)
    union = np.maximum(areas[:, None] + areas[None, :] - inter, 1e-9)
    ious = inter / union
    np.fill_diagonal(ious, 0.0)
    centers_x = (box_array[:, 0] + box_array[:, 2]) * 0.5
    centers_y = (box_array[:, 1] + box_array[:, 3]) * 0.5
    same_axis = horizontal_flags[:, None] == horizontal_flags[None, :]
    axis_gaps = np.where(
        horizontal_flags[:, None],
        np.abs(centers_y[:, None] - centers_y[None, :]),
        np.abs(centers_x[:, None] - centers_x[None, :]),
    )
    np.fill_diagonal(axis_gaps, np.inf)
    hints = np.asarray([str(candidate.get("label_hint") or candidate.get("prediction") or "") for candidate in candidates], dtype=object)
    same_hint = hints[:, None] == hints[None, :]
    hard_wall_hint = hints == "hard_wall"
    opening_hint = np.isin(hints, ["door", "window"])
    confs = [float(candidate.get("proposal_confidence") or candidate.get("confidence") or 0.0) for candidate in candidates]
    order = np.argsort(np.asarray(confs, dtype=np.float32))[::-1] if confs else np.asarray([], dtype=np.int64)
    rank_pct = np.zeros(len(candidates), dtype=np.float32)
    for rank, idx in enumerate(order.tolist()):
        rank_pct[idx] = rank / max(len(candidates) - 1, 1)

    features: list[list[float]] = []
    for idx, candidate in enumerate(candidates):
        box = boxes[idx]
        hint = str(hints[idx])
        width_norm, height_norm, area_norm, aspect_log, length_norm, thickness_norm, horizontal = box_metrics(
            box, row.get("image_size") if isinstance(row.get("image_size"), list) else None
        )
        dup_050 = int((ious[idx] >= 0.50).sum())
        dup_070 = int((ious[idx] >= 0.70).sum())
        same_hint_050 = int(((ious[idx] >= 0.50) & same_hint[idx]).sum())
        same_axis_8 = int((same_axis[idx] & (axis_gaps[idx] <= 8.0)).sum())
        same_axis_16 = int((same_axis[idx] & (axis_gaps[idx] <= 16.0)).sum())
        cross_axis_16 = int((~same_axis[idx] & (axis_gaps[idx] <= 16.0)).sum())
        hard_wall_overlap = int(((ious[idx] >= 0.10) & hard_wall_hint).sum())
        opening_support = int(opening_hint[idx] and ((opening_hint & (axis_gaps[idx] <= 16.0)).sum()))
        features.append(
            [
                1.0 if hint == "hard_wall" else 0.0,
                1.0 if hint == "door" else 0.0,
                1.0 if hint == "window" else 0.0,
                confs[idx],
                float(rank_pct[idx]),
                width_norm,
                height_norm,
                area_norm,
                aspect_log,
                length_norm,
                thickness_norm,
                horizontal,
                1.0 - horizontal,
                float(min(dup_050, 25)),
                float(min(dup_070, 25)),
                float(min(same_hint_050, 25)),
                float(min(same_axis_8, 25)),
                float(min(same_axis_16, 25)),
                float(min(cross_axis_16, 25)),
                float(min(hard_wall_overlap, 25)),
                float(min(opening_support, 25)),
            ]
        )
    return features


def candidate_gold_label(candidate: dict[str, Any], golds: list[dict[str, Any]]) -> str:
    cb = bbox(candidate.get("bbox"))
    if cb is None:
        return "background"
    best_label = "background"
    best_score = 0.0
    for gold in golds:
        gb = gold["bbox"]
        score = max(iou(cb, gb), 1.0 if center_covered(cb, gb) else 0.0)
        if score > best_score:
            best_score = score
            best_label = gold["label"]
    return best_label if best_score > 0.0 else "background"


def build_dataset(
    rows: list[dict[str, Any]],
    gold_by_id: dict[str, list[dict[str, Any]]],
    train_cap: int,
    max_background: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, int]]:
    rng = np.random.default_rng(seed)
    positives: list[tuple[list[float], str, dict[str, Any]]] = []
    background: list[tuple[list[float], str, dict[str, Any]]] = []
    for row in rows:
        row_id = str(row.get("id"))
        features = candidate_features(row, train_cap)
        for candidate, feat in zip((row.get("candidate_stream") or [])[:train_cap], features, strict=True):
            label = candidate_gold_label(candidate, gold_by_id.get(row_id, []))
            ledger = {
                "row_id": row_id,
                "candidate_id": candidate.get("candidate_id"),
                "gold_label": label,
                "label_hint": candidate.get("label_hint"),
                "proposal_confidence": candidate.get("proposal_confidence"),
            }
            if label == "background":
                background.append((feat, label, ledger))
            else:
                positives.append((feat, label, ledger))
    if max_background > 0 and len(background) > max_background:
        keep = rng.choice(len(background), size=max_background, replace=False)
        background = [background[int(idx)] for idx in keep]
    combined = positives + background
    rng.shuffle(combined)
    x = np.asarray([item[0] for item in combined], dtype=np.float32)
    y = np.asarray([item[1] for item in combined])
    ledger = [item[2] for item in combined]
    return x, y, ledger, dict(Counter(y.tolist()))


def baseline_label(candidate: dict[str, Any]) -> str:
    label = str(candidate.get("prediction") or candidate.get("label_hint") or "")
    return label if label in LABELS else "hard_wall"


def label_from_probs(classes: list[str], probs: np.ndarray, thresholds: dict[str, float], fallback: str) -> str | None:
    prob_by_label = {label: float(probs[idx]) for idx, label in enumerate(classes)}
    if prob_by_label.get("background", 0.0) >= thresholds["background_drop"]:
        return None
    if fallback == "hard_wall":
        door_prob = prob_by_label.get("door", 0.0)
        window_prob = prob_by_label.get("window", 0.0)
        if window_prob >= thresholds["window"] and window_prob >= door_prob:
            return "window"
        if door_prob >= thresholds["door"]:
            return "door"
    return fallback


def score_rows(
    rows: list[dict[str, Any]],
    model: Any,
    cap: int,
) -> list[dict[str, Any]]:
    classes = [str(label) for label in model.classes_.tolist()]
    output = []
    for row in rows:
        copied = dict(row)
        stream = []
        candidates = (row.get("candidate_stream") or [])[:cap]
        features = candidate_features(row, cap)
        probs = model.predict_proba(np.asarray(features, dtype=np.float32)) if features else np.zeros((0, len(classes)))
        for candidate, prob in zip(candidates, probs, strict=True):
            item = dict(candidate)
            item["_context_policy_classes"] = classes
            item["_context_policy_probs"] = [float(value) for value in prob.tolist()]
            item["_context_policy_fallback"] = baseline_label(candidate)
            stream.append(item)
        copied["candidate_stream"] = stream
        output.append(copied)
    return output


def apply_thresholds_to_scored_rows(rows: list[dict[str, Any]], thresholds: dict[str, float], cap: int) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        copied = dict(row)
        stream = []
        for candidate in (row.get("candidate_stream") or [])[:cap]:
            classes = [str(label) for label in candidate.get("_context_policy_classes") or []]
            probs = np.asarray(candidate.get("_context_policy_probs") or [], dtype=np.float32)
            if not classes or probs.size != len(classes):
                continue
            fallback = str(candidate.get("_context_policy_fallback") or baseline_label(candidate))
            label = label_from_probs(classes, probs, thresholds, fallback)
            if label is None:
                continue
            item = {key: value for key, value in candidate.items() if not key.startswith("_context_policy_")}
            item["context_policy_prediction"] = label
            item["context_policy_probabilities"] = {cls: round(float(probs[idx]), 6) for idx, cls in enumerate(classes)}
            item["context_policy_fallback"] = fallback if label == fallback else None
            item["prediction"] = label
            item["confidence"] = item["context_policy_probabilities"].get(label, item.get("confidence", 0.0))
            stream.append(item)
        copied["candidate_stream"] = stream
        output.append(copied)
    return output


def predict_rows(
    rows: list[dict[str, Any]],
    model: Any,
    thresholds: dict[str, float],
    cap: int,
) -> list[dict[str, Any]]:
    return apply_thresholds_to_scored_rows(score_rows(rows, model, cap), thresholds, cap)


def evaluate(rows: list[dict[str, Any]], gold_by_id: dict[str, list[dict[str, Any]]], cap: int) -> dict[str, Any]:
    total = proposal_hit = classified_hit = predicted = 0
    per_label: dict[str, Counter[str]] = defaultdict(Counter)
    wrong_pairs = Counter()
    missed = []
    wrong = []
    for row in rows:
        row_id = str(row.get("id"))
        candidates = (row.get("candidate_stream") or [])[:cap]
        predicted += len(candidates)
        for gold in gold_by_id.get(row_id, []):
            total += 1
            label = gold["label"]
            per_label[label]["gold"] += 1
            matches = [
                candidate
                for candidate in candidates
                if (bbox(candidate.get("bbox")) is not None)
                and (center_covered(bbox(candidate.get("bbox")), gold["bbox"]) or iou(bbox(candidate.get("bbox")), gold["bbox"]) >= 0.30)
            ]
            if matches:
                proposal_hit += 1
                per_label[label]["proposal_matched"] += 1
            else:
                missed.append({"row_id": row_id, "target_id": gold.get("target_id"), "label": label, "bbox": gold["bbox"]})
            if any(candidate.get("prediction") == label for candidate in matches):
                classified_hit += 1
                per_label[label]["classified_matched"] += 1
            elif matches:
                pred_label = str(matches[0].get("prediction"))
                wrong_pairs[f"{label}->{pred_label}"] += 1
                wrong.append(
                    {
                        "row_id": row_id,
                        "target_id": gold.get("target_id"),
                        "gold_label": label,
                        "pred_label": pred_label,
                        "bbox": gold["bbox"],
                        "pred_bbox": matches[0].get("bbox"),
                    }
                )
    return {
        "gold": total,
        "predicted": predicted,
        "candidate_inflation": round(predicted / max(total, 1), 6),
        "proposal_recall": round(proposal_hit / max(total, 1), 6),
        "classified_recall": round(classified_hit / max(total, 1), 6),
        "classified_precision_proxy": round(classified_hit / max(predicted, 1), 6),
        "per_label": {
            label: {
                "gold": counts["gold"],
                "proposal_matched": counts["proposal_matched"],
                "classified_matched": counts["classified_matched"],
                "proposal_recall": round(counts["proposal_matched"] / max(counts["gold"], 1), 6),
                "classified_recall": round(counts["classified_matched"] / max(counts["gold"], 1), 6),
            }
            for label, counts in sorted(per_label.items())
        },
        "wrong_pairs": dict(wrong_pairs),
        "missed_summary": dict(Counter(item["label"] for item in missed)),
        "missed_examples": missed[:200],
        "wrong_type_examples": wrong[:200],
    }


def threshold_grid() -> list[dict[str, float]]:
    out = []
    for bg in [0.80, 0.88, 0.94, 0.98, 1.01]:
        for door in [0.35, 0.50, 0.65, 0.80]:
            for window in [0.35, 0.50, 0.65, 0.80]:
                out.append({"door": door, "window": window, "background_drop": bg})
    return out


def select_thresholds(rows: list[dict[str, Any]], gold_by_id: dict[str, list[dict[str, Any]]], model: Any, cap: int) -> tuple[dict[str, float], dict[str, Any]]:
    baseline = evaluate(rows, gold_by_id, cap)
    scored_rows = score_rows(rows, model, cap)
    best_thresholds = threshold_grid()[0]
    best_metrics = baseline
    best_score = -1.0
    for thresholds in threshold_grid():
        predicted = apply_thresholds_to_scored_rows(scored_rows, thresholds, cap)
        metrics = evaluate(predicted, gold_by_id, cap)
        hard_wall = metrics["per_label"].get("hard_wall", {}).get("classified_recall", 0.0)
        door = metrics["per_label"].get("door", {}).get("classified_recall", 0.0)
        window = metrics["per_label"].get("window", {}).get("classified_recall", 0.0)
        base_hard_wall = baseline["per_label"].get("hard_wall", {}).get("classified_recall", 0.0)
        base_door = baseline["per_label"].get("door", {}).get("classified_recall", 0.0)
        base_window = baseline["per_label"].get("window", {}).get("classified_recall", 0.0)
        no_large_drop = (
            metrics["classified_recall"] >= baseline["classified_recall"] - 0.001
            and hard_wall >= base_hard_wall - 0.002
            and door >= base_door - 0.002
            and window >= base_window - 0.002
        )
        score = metrics["classified_precision_proxy"] + 0.10 * max(door - base_door, 0.0) + 0.10 * max(window - base_window, 0.0)
        if no_large_drop:
            score += 1.0
        if score > best_score:
            best_score = score
            best_thresholds = thresholds
            best_metrics = metrics
    return best_thresholds, {"baseline": baseline, "selected": best_metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-predictions", default="reports/vlm/boundary_public_raster_v24_yolo_full_dev493_candidate_stream.jsonl")
    parser.add_argument("--locked-predictions", default="reports/vlm/boundary_public_raster_v24_yolo_full_locked50_candidate_stream.jsonl")
    parser.add_argument("--dataset", default="datasets/boundary_expert_public_raster_v19")
    parser.add_argument("--output-dir", default="checkpoints/boundary_context_type_policy_v24")
    parser.add_argument("--eval-output", default="reports/vlm/boundary_context_type_policy_v24_locked50_eval.json")
    parser.add_argument("--predictions-output", default="reports/vlm/boundary_context_type_policy_v24_locked50_predictions.jsonl")
    parser.add_argument("--train-ledger-output", default="reports/vlm/boundary_context_type_policy_v24_train_ledger.jsonl")
    parser.add_argument("--dev-limit", type=int, default=493)
    parser.add_argument("--locked-limit", type=int, default=50)
    parser.add_argument("--train-rows", type=int, default=400)
    parser.add_argument("--cap", type=int, default=800)
    parser.add_argument("--train-cap", type=int, default=240)
    parser.add_argument("--max-background", type=int, default=120000)
    parser.add_argument("--estimators", type=int, default=500)
    parser.add_argument("--model-backend", choices=["hist_gradient", "extra_trees"], default="hist_gradient")
    parser.add_argument("--seed", type=int, default=20260511)
    args = parser.parse_args()

    dataset = ROOT / args.dataset
    dev_rows = load_jsonl(ROOT / args.dev_predictions, args.dev_limit)
    train_rows = dev_rows[: args.train_rows]
    tune_rows = dev_rows[args.train_rows :]
    locked_rows = load_jsonl(ROOT / args.locked_predictions, args.locked_limit)
    train_gold = load_gold(dataset / "dev.jsonl", args.train_rows)
    tune_gold = load_gold(dataset / "dev.jsonl", args.dev_limit)
    locked_gold = load_gold(dataset / "locked.jsonl", args.locked_limit)

    x_train, y_train, ledger, train_counts = build_dataset(train_rows, train_gold, args.train_cap, args.max_background, args.seed)
    sample_weight = np.asarray(
        [{"background": 0.25, "hard_wall": 1.0, "door": 3.0, "window": 3.2}.get(str(label), 1.0) for label in y_train],
        dtype=np.float32,
    )
    if args.model_backend == "extra_trees":
        model = ExtraTreesClassifier(
            n_estimators=args.estimators,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight=None,
            random_state=args.seed,
            n_jobs=-1,
        )
    else:
        model = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=max(40, min(args.estimators, 180)),
            max_leaf_nodes=31,
            l2_regularization=0.01,
            random_state=args.seed,
        )
    model.fit(x_train, y_train, sample_weight=sample_weight)

    tune_gold_subset = {str(row.get("id")): tune_gold.get(str(row.get("id")), []) for row in tune_rows}
    thresholds, tune_eval = select_thresholds(tune_rows, tune_gold_subset, model, args.cap)
    locked_baseline = evaluate(locked_rows, locked_gold, args.cap)
    locked_predicted = predict_rows(locked_rows, model, thresholds, args.cap)
    locked_eval = evaluate(locked_predicted, locked_gold, args.cap)

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.joblib"
    joblib.dump(
        {
            "model": model,
            "labels": ALL_LABELS,
            "feature_names": FEATURE_NAMES,
            "thresholds": thresholds,
            "train_source": "full_dev_yolo_candidates_rows_0_to_train_rows_minus_1",
            "runtime_contract": "raster_candidate_stream_only_no_svg_no_gold_features",
            "model_backend": args.model_backend,
        },
        model_path,
    )
    write_jsonl(ROOT / args.predictions_output, locked_predicted)
    write_jsonl(ROOT / args.train_ledger_output, ledger)
    report = {
        "version": "boundary_context_type_policy_v24_locked50_eval",
        "task": "P0-BOUNDARY-PROPOSAL-002",
        "claim_boundary": "ExtraTrees page-context policy trained on full-dev YOLO candidate stream. Gold is used only for offline labels/evaluation; runtime features are bbox/conf/hint/context only.",
        "model": str(model_path),
        "feature_names": FEATURE_NAMES,
        "thresholds": thresholds,
        "train_counts": train_counts,
        "classification_report_train_sample": classification_report(y_train, model.predict(x_train), labels=ALL_LABELS, output_dict=True, zero_division=0),
        "tune_eval": tune_eval,
        "locked_baseline_yolo_hint": locked_baseline,
        "locked_eval": locked_eval,
        "success_gate": {
            "baseline_classified_recall": locked_baseline["classified_recall"],
            "baseline_precision_proxy": locked_baseline["classified_precision_proxy"],
            "locked_classified_recall": locked_eval["classified_recall"],
            "locked_precision_proxy": locked_eval["classified_precision_proxy"],
            "beats_yolo_hint_recall": locked_eval["classified_recall"] >= locked_baseline["classified_recall"],
            "beats_yolo_hint_precision_proxy": locked_eval["classified_precision_proxy"] > locked_baseline["classified_precision_proxy"],
            "door_recall_min": 0.9,
            "window_recall_min": 0.9,
            "passed": locked_eval["classified_recall"] >= locked_baseline["classified_recall"]
            and locked_eval["classified_precision_proxy"] > locked_baseline["classified_precision_proxy"]
            and locked_eval["per_label"].get("door", {}).get("classified_recall", 0.0) >= 0.9
            and locked_eval["per_label"].get("window", {}).get("classified_recall", 0.0) >= 0.9,
        },
    }
    write_json(ROOT / args.eval_output, report)
    print(
        json.dumps(
            {
                "train_counts": train_counts,
                "thresholds": thresholds,
                "locked_baseline": {
                    "classified_recall": locked_baseline["classified_recall"],
                    "precision_proxy": locked_baseline["classified_precision_proxy"],
                    "per_label": locked_baseline["per_label"],
                },
                "locked_eval": {
                    "classified_recall": locked_eval["classified_recall"],
                    "precision_proxy": locked_eval["classified_precision_proxy"],
                    "per_label": locked_eval["per_label"],
                    "wrong_pairs": locked_eval["wrong_pairs"],
                },
                "success_gate": report["success_gate"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
