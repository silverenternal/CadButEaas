#!/usr/bin/env python3
"""Evaluate RoomSpace predictions using a human review CSV for ambiguous rooms."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    from train_room_space_expert import evaluate_predictions
except ImportError:
    from scripts.vlm.train_room_space_expert import evaluate_predictions


ACCEPT_LABELS = {"accept_typed"}
EXCLUDE_LABELS = {"exclude"}
KEEP_LABELS = {"keep_room", "unclear", ""}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="checkpoints/cadstruct_moe_room_space_hierarchical_sklearn_v5_t046/locked_test_predictions.jsonl")
    parser.add_argument("--review-csv", default="reports/vlm/room_ambiguity_review_pack_v1/review_queue.csv")
    parser.add_argument("--output", default="reports/vlm/room_space_v5_t046_review_adjusted.json")
    args = parser.parse_args()

    predictions = load_jsonl(Path(args.predictions))
    review = load_review(Path(args.review_csv))
    adjusted = []
    stats = {"accept_typed": 0, "exclude": 0, "keep_or_unclear": 0, "missing": 0}

    for row in predictions:
        adjusted_rooms = []
        annotation = str(row.get("annotation") or "")
        for room in row.get("rooms") or []:
            item = dict(room)
            key = (annotation, str(item.get("id")))
            label = review.get(key, "")
            if label in ACCEPT_LABELS and item.get("gold") == "room" and item.get("prediction") != "room":
                item["gold"] = item["prediction"]
                item["review_adjusted"] = True
                stats["accept_typed"] += 1
            elif label in EXCLUDE_LABELS:
                stats["exclude"] += 1
                continue
            elif key in review:
                stats["keep_or_unclear"] += 1
            else:
                stats["missing"] += 1
            adjusted_rooms.append(item)
        adjusted.append({**row, "rooms": adjusted_rooms})

    strict_metrics = evaluate_predictions(predictions)
    review_metrics = evaluate_predictions(adjusted)
    report = {
        "predictions": args.predictions,
        "review_csv": args.review_csv,
        "review_policy": {
            "accept_typed": "Change gold room to prediction for reviewed generic-room boundary cases.",
            "keep_room": "Keep original strict gold label.",
            "unclear": "Keep original strict gold label.",
            "exclude": "Remove reviewed room from clean eval denominator.",
            "blank": "Treat as keep/unclear; dry-run mode before human review.",
        },
        "review_stats": stats,
        "strict": strict_metrics,
        "review_adjusted": review_metrics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"review_stats": stats, "strict": pick(strict_metrics), "review_adjusted": pick(review_metrics)}, ensure_ascii=False, indent=2))


def load_review(path: Path) -> dict[tuple[str, str], str]:
    review = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = str(row.get("review_label") or "").strip()
            if label not in ACCEPT_LABELS | EXCLUDE_LABELS | KEEP_LABELS:
                raise ValueError(f"Unknown review_label {label!r} for {row.get('review_id')}")
            review[(str(row.get("annotation") or ""), str(row.get("room_id") or ""))] = label
    return review


def pick(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "rooms": metrics.get("rooms"),
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "room_f1": ((metrics.get("per_label") or {}).get("room") or {}).get("f1"),
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    main()
