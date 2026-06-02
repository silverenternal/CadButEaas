#!/usr/bin/env python3
"""Train door/window specialist heads for boundary v24 candidates."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from train_boundary_type_fusion_v24 import LABELS, bbox, center_covered, feature_row, gold_by_row, iou


ROOT = Path(__file__).resolve().parents[2]
SPECIALIST_FEATURES = [
    *[f"base_{name}" for name in [
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
    ]],
    "fusion_pred_hard_wall",
    "fusion_pred_door",
    "fusion_pred_window",
    "hint_equals_fusion",
    "hint_is_opening",
    "gnn_is_hard_wall",
    "gnn_is_door",
    "gnn_is_window",
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def enrich_with_fusion(rows: list[dict[str, Any]], fusion_model: Any) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        copied = dict(row)
        stream = []
        candidates = row.get("candidate_stream") or []
        preds = fusion_model.predict(np.asarray([feature_row(c) for c in candidates], dtype=np.float32)).tolist() if candidates else []
        for candidate, pred in zip(candidates, preds, strict=True):
            item = dict(candidate)
            item["fusion_prediction"] = str(pred)
            stream.append(item)
        copied["candidate_stream"] = stream
        out.append(copied)
    return out


def candidate_features(candidate: dict[str, Any]) -> list[float]:
    base = feature_row(candidate)
    fusion = str(candidate.get("fusion_prediction") or candidate.get("prediction") or "")
    hint = str(candidate.get("label_hint") or "")
    gnn = str(candidate.get("gnn_prediction") or "")
    return [
        *base,
        1.0 if fusion == "hard_wall" else 0.0,
        1.0 if fusion == "door" else 0.0,
        1.0 if fusion == "window" else 0.0,
        1.0 if hint == fusion and hint in LABELS else 0.0,
        1.0 if hint in {"door", "window"} else 0.0,
        1.0 if gnn == "hard_wall" else 0.0,
        1.0 if gnn == "door" else 0.0,
        1.0 if gnn == "window" else 0.0,
    ]


def candidate_gold_label(candidate: dict[str, Any], gold_items: list[dict[str, Any]]) -> str | None:
    cb = bbox(candidate.get("bbox"))
    if cb is None:
        return None
    best_label = None
    best_score = 0.0
    for item in gold_items:
        gb = item["bbox"]
        score = max(iou(cb, gb), 1.0 if center_covered(cb, gb) else 0.0)
        if score > best_score:
            best_score = score
            best_label = item["label"]
    return best_label if best_score > 0.0 else None


def build_candidate_dataset(rows: list[dict[str, Any]], gold_path: Path, limit: int | None, cap: int) -> tuple[np.ndarray, dict[str, np.ndarray], list[dict[str, Any]]]:
    gold = gold_by_row(gold_path, limit)
    x_rows = []
    y = {"door": [], "window": []}
    ledger = []
    for row in rows:
        row_id = str(row.get("id"))
        for candidate in (row.get("candidate_stream") or [])[:cap]:
            label = candidate_gold_label(candidate, gold.get(row_id, []))
            x_rows.append(candidate_features(candidate))
            y["door"].append(1 if label == "door" else 0)
            y["window"].append(1 if label == "window" else 0)
            ledger.append(
                {
                    "row_id": row_id,
                    "candidate_id": candidate.get("candidate_id"),
                    "gold_label": label,
                    "fusion_prediction": candidate.get("fusion_prediction"),
                    "label_hint": candidate.get("label_hint"),
                    "proposal_confidence": candidate.get("proposal_confidence"),
                }
            )
    return np.asarray(x_rows, dtype=np.float32), {key: np.asarray(value, dtype=np.int64) for key, value in y.items()}, ledger


def train_head(x: np.ndarray, y: np.ndarray, seed: int) -> Any:
    if len(set(y.tolist())) < 2:
        raise ValueError("Specialist target has a single class")
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=31,
        l2_regularization=0.02,
        random_state=seed,
    )
    model.fit(x, y)
    return model


def score_head(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return model.decision_function(x)


def attach_specialist_scores(rows: list[dict[str, Any]], models: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        copied = dict(row)
        candidates = row.get("candidate_stream") or []
        if candidates:
            x = np.asarray([candidate_features(candidate) for candidate in candidates], dtype=np.float32)
            door_scores = score_head(models["door"], x)
            window_scores = score_head(models["window"], x)
        else:
            door_scores = []
            window_scores = []
        stream = []
        for candidate, door_score, window_score in zip(candidates, door_scores, window_scores, strict=True):
            item = dict(candidate)
            item["_specialist_door_score"] = float(door_score)
            item["_specialist_window_score"] = float(window_score)
            stream.append(item)
        copied["candidate_stream"] = stream
        output.append(copied)
    return output


def base_label(candidate: dict[str, Any]) -> str:
    value = str(candidate.get("fusion_prediction") or candidate.get("prediction") or "")
    return value if value in LABELS else "hard_wall"


def select_thresholds(rows: list[dict[str, Any]], gold_path: Path, models: dict[str, Any], limit: int | None, cap: int) -> dict[str, float]:
    grid = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.26, 0.30, 0.36, 0.44, 0.52, 0.60]
    best = {"door": 0.5, "window": 0.5}
    best_score = -1.0
    for door_t in grid:
        for window_t in grid:
            metrics = evaluate(rows, gold_path, models, {"door": door_t, "window": window_t}, limit, cap)
            score = (
                metrics["classified_recall"]
                + 0.45 * metrics["per_label"]["door"]["classified_recall"]
                + 0.45 * metrics["per_label"]["window"]["classified_recall"]
                + 0.05 * metrics["per_label"]["hard_wall"]["classified_recall"]
            )
            if score > best_score:
                best_score = score
                best = {"door": door_t, "window": window_t}
    return best


def predict_specialist(candidate: dict[str, Any], models: dict[str, Any] | None, thresholds: dict[str, float]) -> tuple[str, dict[str, float], str | None]:
    if "_specialist_door_score" in candidate and "_specialist_window_score" in candidate:
        door_score = float(candidate["_specialist_door_score"])
        window_score = float(candidate["_specialist_window_score"])
    else:
        if models is None:
            raise ValueError("models are required when candidate scores are not precomputed")
        x = np.asarray([candidate_features(candidate)], dtype=np.float32)
        door_score = float(score_head(models["door"], x)[0])
        window_score = float(score_head(models["window"], x)[0])
    label = base_label(candidate)
    override = None
    if door_score >= thresholds["door"] or window_score >= thresholds["window"]:
        if window_score >= thresholds["window"] and window_score >= door_score:
            label = "window"
            override = "window"
        elif door_score >= thresholds["door"]:
            label = "door"
            override = "door"
    return label, {"door": round(door_score, 6), "window": round(window_score, 6)}, override


def evaluate(rows: list[dict[str, Any]], gold_path: Path, models: dict[str, Any], thresholds: dict[str, float], limit: int | None, cap: int) -> dict[str, Any]:
    gold = gold_by_row(gold_path, limit)
    pred_by_id = {str(row.get("id")): (row.get("candidate_stream") or [])[:cap] for row in rows}
    total = proposal_hit = classified_hit = predicted = 0
    per_label: dict[str, Counter[str]] = defaultdict(Counter)
    wrong_pairs = Counter()
    for row_id, gold_items in gold.items():
        candidates = pred_by_id.get(row_id, [])
        predicted += len(candidates)
        pred_cache = [predict_specialist(c, None, thresholds)[0] for c in candidates]
        for item in gold_items:
            total += 1
            label = item["label"]
            per_label[label]["gold"] += 1
            matches = []
            for idx, candidate in enumerate(candidates):
                cb = bbox(candidate.get("bbox"))
                if cb is not None and (center_covered(cb, item["bbox"]) or iou(cb, item["bbox"]) >= 0.30):
                    matches.append(idx)
            if matches:
                proposal_hit += 1
                per_label[label]["proposal_matched"] += 1
            if any(pred_cache[idx] == label for idx in matches):
                classified_hit += 1
                per_label[label]["classified_matched"] += 1
            elif matches:
                wrong_pairs[f"{label}->{pred_cache[matches[0]]}"] += 1
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
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-predictions", default="reports/vlm/boundary_graph_node_gnn_v24_dev50_predictions.jsonl")
    parser.add_argument("--locked-predictions", default="reports/vlm/boundary_type_fusion_v24_locked50_predictions.jsonl")
    parser.add_argument("--dataset", default="datasets/boundary_expert_public_raster_v19")
    parser.add_argument("--fusion-model", default="checkpoints/boundary_type_fusion_v24/model.joblib")
    parser.add_argument("--output-dir", default="checkpoints/boundary_door_window_specialist_v24")
    parser.add_argument("--eval-output", default="reports/vlm/boundary_door_window_specialist_v24_locked50_eval.json")
    parser.add_argument("--predictions-output", default="reports/vlm/boundary_door_window_specialist_v24_locked50_predictions.jsonl")
    parser.add_argument("--train-ledger-output", default="reports/vlm/boundary_door_window_specialist_v24_train_ledger.jsonl")
    parser.add_argument("--dev-limit", type=int, default=50)
    parser.add_argument("--locked-limit", type=int, default=50)
    parser.add_argument("--cap", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20260511)
    args = parser.parse_args()

    bundle = joblib.load(ROOT / args.fusion_model)
    fusion_model = bundle["model"] if isinstance(bundle, dict) else bundle
    dev_rows = enrich_with_fusion(load_jsonl(ROOT / args.dev_predictions), fusion_model)
    locked_rows = enrich_with_fusion(load_jsonl(ROOT / args.locked_predictions), fusion_model)
    dataset = ROOT / args.dataset
    x_train, y_train, ledger = build_candidate_dataset(dev_rows, dataset / "dev.jsonl", args.dev_limit, args.cap)
    models = {target: train_head(x_train, y_train[target], args.seed + idx) for idx, target in enumerate(["door", "window"])}
    train_scores = {
        target: {
            "positive": int(y_train[target].sum()),
            "total": int(len(y_train[target])),
            "average_precision": round(float(average_precision_score(y_train[target], score_head(models[target], x_train))), 6),
            "roc_auc": round(float(roc_auc_score(y_train[target], score_head(models[target], x_train))), 6),
        }
        for target in ["door", "window"]
    }
    dev_scored = attach_specialist_scores(dev_rows, models)
    locked_scored = attach_specialist_scores(locked_rows, models)
    thresholds = select_thresholds(dev_scored, dataset / "dev.jsonl", models, args.dev_limit, args.cap)
    dev_eval = evaluate(dev_scored, dataset / "dev.jsonl", models, thresholds, args.dev_limit, args.cap)
    locked_eval = evaluate(locked_scored, dataset / "locked.jsonl", models, thresholds, args.locked_limit, args.cap)
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.joblib"
    joblib.dump(
        {
            "models": models,
            "thresholds": thresholds,
            "feature_names": SPECIALIST_FEATURES,
            "train_source": "dev_only",
            "fusion_model": args.fusion_model,
        },
        model_path,
    )
    write_jsonl(ROOT / args.train_ledger_output, ledger)
    write_specialist_predictions(ROOT / args.predictions_output, locked_scored, models, thresholds)
    report = {
        "version": "boundary_door_window_specialist_v24_locked50_eval",
        "claim_boundary": "Door/window specialist trained on dev50 candidate labels only; locked50 gold used only for final evaluation.",
        "model": str(model_path),
        "thresholds": thresholds,
        "train_scores": train_scores,
        "dev_eval": dev_eval,
        "locked_eval": locked_eval,
        "success_gate": {
            "classified_recall_min": 0.95,
            "door_recall_min": 0.9,
            "window_recall_min": 0.9,
            "locked_classified_recall": locked_eval["classified_recall"],
            "locked_door_recall": locked_eval["per_label"]["door"]["classified_recall"],
            "locked_window_recall": locked_eval["per_label"]["window"]["classified_recall"],
            "passed": locked_eval["classified_recall"] >= 0.95
            and locked_eval["per_label"]["door"]["classified_recall"] >= 0.9
            and locked_eval["per_label"]["window"]["classified_recall"] >= 0.9,
        },
    }
    write_json(ROOT / args.eval_output, report)
    print(json.dumps({"thresholds": thresholds, "locked_eval": locked_eval, "success_gate": report["success_gate"]}, ensure_ascii=False, indent=2))


def write_specialist_predictions(path: Path, rows: list[dict[str, Any]], models: dict[str, Any], thresholds: dict[str, float]) -> None:
    out_rows = []
    for row in rows:
        copied = dict(row)
        stream = []
        for candidate in row.get("candidate_stream") or []:
            item = dict(candidate)
            label, scores, override = predict_specialist(item, models, thresholds)
            item["specialist_prediction"] = label
            item["specialist_scores"] = scores
            item["specialist_override"] = override
            item["prediction"] = label
            stream.append(item)
        copied["candidate_stream"] = stream
        out_rows.append(copied)
    write_jsonl(path, out_rows)


if __name__ == "__main__":
    main()
