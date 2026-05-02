#!/usr/bin/env python3
"""Evaluate RoomSpace predictions with an explicit ambiguous-generic-room audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from audit_room_generic_boundary import is_typed_text, linked_texts, room_lookup, text_candidates
    from train_room_space_expert import evaluate_predictions
except ImportError:
    from scripts.vlm.audit_room_generic_boundary import is_typed_text, linked_texts, room_lookup, text_candidates
    from scripts.vlm.train_room_space_expert import evaluate_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/cadstruct_cubicasa5k_moe_locked/locked_test.jsonl")
    parser.add_argument("--predictions", default="checkpoints/cadstruct_moe_room_space_hierarchical_sklearn_v5_t046/locked_test_predictions.jsonl")
    parser.add_argument("--output", default="reports/vlm/room_space_v5_t046_ambiguity_adjusted.json")
    args = parser.parse_args()

    rows = {str(row.get("annotation_path")): row for row in load_jsonl(Path(args.dataset))}
    predictions = load_jsonl(Path(args.predictions))
    adjusted_predictions = []
    adjusted_count = 0
    candidates = []

    for pred_row in predictions:
        annotation = str(pred_row.get("annotation") or "")
        data_row = rows.get(annotation)
        expected = (data_row or {}).get("expected_json") or {}
        rooms = room_lookup(expected)
        texts = text_candidates(expected)
        adjusted_rooms = []
        for pred in pred_row.get("rooms") or []:
            item = dict(pred)
            if item.get("gold") == "room" and item.get("prediction") != "room":
                room = rooms.get(str(item.get("id")))
                linked = linked_texts(room.get("bbox") if room else None, texts)
                if any(is_typed_text(str(item.get("prediction")), text) for text in linked):
                    candidates.append(
                        {
                            "annotation": annotation,
                            "id": item.get("id"),
                            "gold": item.get("gold"),
                            "prediction": item.get("prediction"),
                            "texts": linked,
                            "confidence": item.get("confidence"),
                        }
                    )
                    item["gold"] = item["prediction"]
                    item["ambiguity_adjusted"] = True
                    adjusted_count += 1
            adjusted_rooms.append(item)
        adjusted_predictions.append({**pred_row, "rooms": adjusted_rooms})

    strict_metrics = evaluate_predictions(predictions)
    adjusted_metrics = evaluate_predictions(adjusted_predictions)
    report = {
        "dataset": args.dataset,
        "predictions": args.predictions,
        "policy": "Only gold=room and prediction!=room is adjusted when linked room-label text explicitly matches the predicted class.",
        "adjusted_cases": adjusted_count,
        "strict": strict_metrics,
        "ambiguity_adjusted": adjusted_metrics,
        "candidates": candidates[:200],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "adjusted_cases": adjusted_count,
                "strict": pick_metrics(strict_metrics),
                "ambiguity_adjusted": pick_metrics(adjusted_metrics),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def pick_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "rooms": metrics["rooms"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "room_f1": metrics["per_label"]["room"]["f1"],
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
