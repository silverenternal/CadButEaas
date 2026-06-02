#!/usr/bin/env python3
"""Search auditable YOLO-hint override thresholds for boundary fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib

from train_boundary_type_fusion_v24 import LABELS, bbox, center_covered, feature_row, gold_by_row, iou


ROOT = Path(__file__).resolve().parents[2]


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


def predict_label(candidate: dict[str, Any], thresholds: dict[str, float]) -> str:
    hint = str(candidate.get("label_hint") or "")
    score = float(candidate.get("proposal_confidence") or 0.0)
    base = str(candidate.get("fusion_prediction") or candidate.get("prediction") or "")
    if hint in {"door", "window"} and score >= float(thresholds.get(hint, 1.1)):
        return hint
    return base if base in LABELS else "hard_wall"


def with_fusion_predictions(rows: list[dict[str, Any]], model: Any) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        copied = dict(row)
        stream = []
        candidates = row.get("candidate_stream") or []
        if candidates:
            import numpy as np

            features = np.asarray([feature_row(candidate) for candidate in candidates], dtype=np.float32)
            preds = model.predict(features).tolist()
        else:
            preds = []
        for candidate, pred in zip(candidates, preds, strict=True):
            item = dict(candidate)
            item["fusion_prediction"] = str(pred)
            stream.append(item)
        copied["candidate_stream"] = stream
        output.append(copied)
    return output


def evaluate(pred_rows: list[dict[str, Any]], gold_path: Path, thresholds: dict[str, float], limit: int | None, cap: int) -> dict[str, Any]:
    gold = gold_by_row(gold_path, limit)
    pred_by_id = {str(row.get("id")): (row.get("candidate_stream") or [])[:cap] for row in pred_rows}
    total = proposal_hit = classified_hit = predicted = 0
    per_label = {label: {"gold": 0, "proposal_matched": 0, "classified_matched": 0} for label in LABELS}
    for row_id, gold_items in gold.items():
        candidates = pred_by_id.get(row_id, [])
        predicted += len(candidates)
        for gold_item in gold_items:
            total += 1
            label = gold_item["label"]
            per_label[label]["gold"] += 1
            matches = []
            for candidate in candidates:
                cb = bbox(candidate.get("bbox"))
                if cb is not None and (center_covered(cb, gold_item["bbox"]) or iou(cb, gold_item["bbox"]) >= 0.30):
                    matches.append(candidate)
            if matches:
                proposal_hit += 1
                per_label[label]["proposal_matched"] += 1
            if any(predict_label(candidate, thresholds) == label for candidate in matches):
                classified_hit += 1
                per_label[label]["classified_matched"] += 1
    per = {
        label: {
            **counts,
            "proposal_recall": round(counts["proposal_matched"] / max(counts["gold"], 1), 6),
            "classified_recall": round(counts["classified_matched"] / max(counts["gold"], 1), 6),
        }
        for label, counts in per_label.items()
    }
    return {
        "gold": total,
        "predicted": predicted,
        "candidate_inflation": round(predicted / max(total, 1), 6),
        "proposal_recall": round(proposal_hit / max(total, 1), 6),
        "classified_recall": round(classified_hit / max(total, 1), 6),
        "classified_precision_proxy": round(classified_hit / max(predicted, 1), 6),
        "per_label": per,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-predictions", default="reports/vlm/boundary_graph_node_gnn_v24_dev50_predictions.jsonl")
    parser.add_argument("--locked-predictions", default="reports/vlm/boundary_type_fusion_v24_locked50_predictions.jsonl")
    parser.add_argument("--dataset", default="datasets/boundary_expert_public_raster_v19")
    parser.add_argument("--fusion-model", default="checkpoints/boundary_type_fusion_v24/model.joblib")
    parser.add_argument("--output", default="reports/vlm/boundary_hint_override_v24_locked50_eval.json")
    parser.add_argument("--dev-limit", type=int, default=50)
    parser.add_argument("--locked-limit", type=int, default=50)
    parser.add_argument("--cap", type=int, default=800)
    args = parser.parse_args()

    bundle = joblib.load(ROOT / args.fusion_model)
    model = bundle["model"] if isinstance(bundle, dict) else bundle
    dev_pred_rows = with_fusion_predictions(load_jsonl(ROOT / args.dev_predictions), model)
    locked_pred_rows = with_fusion_predictions(load_jsonl(ROOT / args.locked_predictions), model)
    dataset = ROOT / args.dataset
    grid = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    dev_rows = []
    for door_t in grid:
        for window_t in grid:
            thresholds = {"door": door_t, "window": window_t}
            metrics = evaluate(dev_pred_rows, dataset / "dev.jsonl", thresholds, args.dev_limit, args.cap)
            score = (
                metrics["classified_recall"]
                + 0.35 * metrics["per_label"]["door"]["classified_recall"]
                + 0.35 * metrics["per_label"]["window"]["classified_recall"]
                + 0.10 * metrics["per_label"]["hard_wall"]["classified_recall"]
            )
            dev_rows.append({"thresholds": thresholds, "score": round(score, 6), "metrics": metrics})
    dev_rows.sort(key=lambda row: row["score"], reverse=True)
    best = dev_rows[0]
    locked = evaluate(locked_pred_rows, dataset / "locked.jsonl", best["thresholds"], args.locked_limit, args.cap)
    report = {
        "version": "boundary_hint_override_v24_locked50_eval",
        "claim_boundary": "Thresholds selected on dev predictions only; locked gold used only for final evaluation.",
        "selected": best,
        "top_dev_candidates": dev_rows[:10],
        "locked_eval": locked,
        "success_gate": {
            "classified_recall_min": 0.95,
            "door_recall_min": 0.9,
            "window_recall_min": 0.9,
            "locked_classified_recall": locked["classified_recall"],
            "locked_door_recall": locked["per_label"]["door"]["classified_recall"],
            "locked_window_recall": locked["per_label"]["window"]["classified_recall"],
            "passed": locked["classified_recall"] >= 0.95
            and locked["per_label"]["door"]["classified_recall"] >= 0.9
            and locked["per_label"]["window"]["classified_recall"] >= 0.9,
        },
    }
    write_json(ROOT / args.output, report)
    print(json.dumps({"selected": best["thresholds"], "locked_eval": locked, "success_gate": report["success_gate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
