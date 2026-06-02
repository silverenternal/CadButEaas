#!/usr/bin/env python3
"""Train an auditable boundary type fusion policy for v24 proposals."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report


ROOT = Path(__file__).resolve().parents[2]
LABELS = ["hard_wall", "door", "window"]
BOUNDARY_TO_GRAPH_LABEL = {"wall": "hard_wall", "opening": "door", "door": "door", "window": "window"}
FEATURE_NAMES = [
    "gnn_prob_hard_wall",
    "gnn_prob_door",
    "gnn_prob_window",
    "yolo_hint_hard_wall",
    "yolo_hint_door",
    "yolo_hint_window",
    "proposal_confidence",
    "bbox_width",
    "bbox_height",
    "bbox_area",
    "bbox_aspect_log",
    "bbox_length",
    "bbox_thickness",
    "orientation_horizontal",
    "orientation_vertical",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def iou(left: list[float] | None, right: list[float] | None) -> float:
    if left is None or right is None:
        return 0.0
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    return inter / max(area(left) + area(right) - inter, 1e-9)


def center(box: list[float]) -> tuple[float, float]:
    return (box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5


def center_covered(pred: list[float], gold: list[float], margin: float = 3.0) -> bool:
    gx, gy = center(gold)
    return pred[0] - margin <= gx <= pred[2] + margin and pred[1] - margin <= gy <= pred[3] + margin


def gold_by_row(path: Path, limit: int | None = None) -> dict[str, list[dict[str, Any]]]:
    output = {}
    for idx, row in enumerate(load_jsonl(path)):
        if limit is not None and idx >= limit:
            break
        items = []
        for item in (row.get("targets") or {}).get("boxes") or []:
            box = bbox(item.get("bbox"))
            label = BOUNDARY_TO_GRAPH_LABEL.get(str(item.get("label")))
            if box is not None and label in LABELS:
                items.append({"bbox": box, "label": label, "target_id": item.get("target_id")})
        output[str(row.get("id"))] = items
    return output


def feature_row(candidate: dict[str, Any]) -> list[float]:
    box = bbox(candidate.get("bbox")) or [0.0, 0.0, 1.0, 1.0]
    width = max(box[2] - box[0], 1e-6)
    height = max(box[3] - box[1], 1e-6)
    orient_horizontal = 1.0 if width >= height else 0.0
    hint = str(candidate.get("label_hint") or "")
    probs = candidate.get("probabilities") if isinstance(candidate.get("probabilities"), dict) else {}
    return [
        float(probs.get("hard_wall") or 0.0),
        float(probs.get("door") or 0.0),
        float(probs.get("window") or 0.0),
        1.0 if hint == "hard_wall" else 0.0,
        1.0 if hint == "door" else 0.0,
        1.0 if hint == "window" else 0.0,
        float(candidate.get("proposal_confidence") or 0.0),
        width,
        height,
        width * height,
        float(np.log(width / height)),
        max(width, height),
        min(width, height),
        orient_horizontal,
        1.0 - orient_horizontal,
    ]


def matched_training_rows(predictions_path: Path, gold_path: Path, limit: int | None = None, cap: int = 800) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    gold = gold_by_row(gold_path, limit)
    x_rows: list[list[float]] = []
    y_rows: list[str] = []
    ledger: list[dict[str, Any]] = []
    for pred_row in load_jsonl(predictions_path):
        row_id = str(pred_row.get("id"))
        candidates = (pred_row.get("candidate_stream") or [])[:cap]
        used: set[int] = set()
        for item in gold.get(row_id, []):
            gb = item["bbox"]
            best_index = None
            best_score = -1.0
            for idx, cand in enumerate(candidates):
                if idx in used:
                    continue
                cb = bbox(cand.get("bbox"))
                if cb is None:
                    continue
                score = max(iou(cb, gb), 1.0 if center_covered(cb, gb) else 0.0)
                if score > best_score:
                    best_index = idx
                    best_score = score
            if best_index is None or best_score <= 0.0:
                continue
            used.add(best_index)
            candidate = candidates[best_index]
            x_rows.append(feature_row(candidate))
            y_rows.append(item["label"])
            ledger.append(
                {
                    "row_id": row_id,
                    "target_id": item.get("target_id"),
                    "label": item["label"],
                    "candidate_id": candidate.get("candidate_id"),
                    "gnn_prediction": candidate.get("gnn_prediction"),
                    "label_hint": candidate.get("label_hint"),
                    "match_score": round(float(best_score), 6),
                }
            )
    return np.asarray(x_rows, dtype=np.float32), np.asarray(y_rows), ledger


def evaluate_predictions(
    predictions_path: Path,
    gold_path: Path,
    model: ExtraTreesClassifier,
    limit: int | None = None,
    cap: int = 800,
) -> dict[str, Any]:
    gold = gold_by_row(gold_path, limit)
    pred_by_id = {str(row.get("id")): (row.get("candidate_stream") or [])[:cap] for row in load_jsonl(predictions_path)}
    total = proposal_hit = classified_hit = predicted = 0
    per_label: dict[str, Counter[str]] = defaultdict(Counter)
    y_true: list[str] = []
    y_pred: list[str] = []
    for row_id, gold_items in gold.items():
        candidates = pred_by_id.get(row_id, [])
        predicted += len(candidates)
        if candidates:
            cand_x = np.asarray([feature_row(cand) for cand in candidates], dtype=np.float32)
            cand_pred = model.predict(cand_x).tolist()
        else:
            cand_pred = []
        for gold_item in gold_items:
            total += 1
            label = gold_item["label"]
            per_label[label]["gold"] += 1
            matches = []
            for idx, candidate in enumerate(candidates):
                cb = bbox(candidate.get("bbox"))
                if cb is not None and (center_covered(cb, gold_item["bbox"]) or iou(cb, gold_item["bbox"]) >= 0.30):
                    matches.append((idx, candidate))
            if matches:
                proposal_hit += 1
                per_label[label]["proposal_matched"] += 1
            if any(cand_pred[idx] == label for idx, _candidate in matches):
                classified_hit += 1
                per_label[label]["classified_matched"] += 1
                y_true.append(label)
                y_pred.append(label)
            elif matches:
                idx, _candidate = matches[0]
                y_true.append(label)
                y_pred.append(cand_pred[idx])
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
        "matched_classification_report": classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-predictions", default="reports/vlm/boundary_graph_node_gnn_v24_dev20_predictions.jsonl")
    parser.add_argument("--locked-predictions", default="reports/vlm/boundary_graph_node_gnn_v24_predictions.jsonl")
    parser.add_argument("--dataset", default="datasets/boundary_expert_public_raster_v19")
    parser.add_argument("--output-dir", default="checkpoints/boundary_type_fusion_v24")
    parser.add_argument("--eval-output", default="reports/vlm/boundary_type_fusion_v24_locked_eval.json")
    parser.add_argument("--train-ledger-output", default="reports/vlm/boundary_type_fusion_v24_train_ledger.jsonl")
    parser.add_argument("--dev-limit", type=int, default=20)
    parser.add_argument("--locked-limit", type=int, default=20)
    parser.add_argument("--cap", type=int, default=800)
    parser.add_argument("--estimators", type=int, default=300)
    parser.add_argument("--class-weight", default="balanced")
    parser.add_argument("--sample-weight-door", type=float, default=1.0)
    parser.add_argument("--sample-weight-window", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260511)
    args = parser.parse_args()

    dataset = ROOT / args.dataset
    x_train, y_train, ledger = matched_training_rows(ROOT / args.dev_predictions, dataset / "dev.jsonl", args.dev_limit, args.cap)
    class_weight: str | dict[str, float] | None
    if args.class_weight == "none":
        class_weight = None
    elif args.class_weight == "targeted":
        class_weight = {"hard_wall": 1.0, "door": 2.0, "window": 2.4}
    else:
        class_weight = args.class_weight
    model = ExtraTreesClassifier(
        n_estimators=args.estimators,
        min_samples_leaf=1,
        class_weight=class_weight,
        random_state=args.seed,
        n_jobs=-1,
    )
    sample_weight = np.ones(len(y_train), dtype=np.float32)
    sample_weight[y_train == "door"] *= float(args.sample_weight_door)
    sample_weight[y_train == "window"] *= float(args.sample_weight_window)
    model.fit(x_train, y_train, sample_weight=sample_weight)
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.joblib"
    joblib.dump({"model": model, "labels": LABELS, "feature_names": FEATURE_NAMES, "train_source": "dev_only"}, model_path)
    with (ROOT / args.train_ledger_output).open("w", encoding="utf-8") as handle:
        for row in ledger:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    dev_eval = evaluate_predictions(ROOT / args.dev_predictions, dataset / "dev.jsonl", model, args.dev_limit, args.cap)
    locked_eval = evaluate_predictions(ROOT / args.locked_predictions, dataset / "locked.jsonl", model, args.locked_limit, args.cap)
    report = {
        "version": "boundary_type_fusion_v24_locked_eval",
        "task": "P0-BOUNDARY-PROPOSAL-002",
        "claim_boundary": "Fusion model is trained on dev proposal matches only. Locked gold is used only for evaluation.",
        "source_integrity": {
            "runtime_inputs": ["raster_derived_yolo_label_hint", "raster_derived_yolo_score", "crop_gnn_probabilities", "bbox_geometry"],
            "gold_used_for_inference": False,
            "locked_gold_used_for_training": False,
        },
        "model": str(model_path),
        "feature_names": FEATURE_NAMES,
        "train": {
            "rows": int(len(y_train)),
            "label_counts": dict(Counter(y_train.tolist())),
            "class_weight": args.class_weight,
            "sample_weight_door": args.sample_weight_door,
            "sample_weight_window": args.sample_weight_window,
            "predictions": args.dev_predictions,
            "gold": str(dataset / "dev.jsonl"),
        },
        "dev_eval": dev_eval,
        "locked_eval": locked_eval,
        "success_gate": {
            "next_stage_classified_recall_min": 0.9,
            "next_stage_window_classified_recall_min": 0.75,
            "locked_classified_recall": locked_eval["classified_recall"],
            "locked_window_classified_recall": locked_eval["per_label"].get("window", {}).get("classified_recall", 0.0),
            "passed": locked_eval["classified_recall"] >= 0.9
            and locked_eval["per_label"].get("window", {}).get("classified_recall", 0.0) >= 0.75,
        },
    }
    write_json(ROOT / args.eval_output, report)
    print(json.dumps({"model": str(model_path), "train_rows": len(y_train), "locked_eval": locked_eval, "success_gate": report["success_gate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
