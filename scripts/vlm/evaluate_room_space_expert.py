#!/usr/bin/env python3
"""Evaluate a RoomSpaceExpert baseline checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from train_room_space_expert import evaluate_predictions, load_jsonl, predict_rows, write_jsonl
except ImportError:
    from scripts.vlm.train_room_space_expert import evaluate_predictions, load_jsonl, predict_rows, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/cadstruct_rooms_v1/smoke.jsonl")
    parser.add_argument("--model", default="checkpoints/cadstruct_moe_room_space_baseline/model.json")
    parser.add_argument("--output", default="reports/vlm/moe/room_space_baseline_smoke.json")
    parser.add_argument("--predictions-output", default="reports/vlm/moe/room_space_baseline_smoke_predictions.jsonl")
    args = parser.parse_args()

    model = json.loads(Path(args.model).read_text(encoding="utf-8"))
    rows = load_jsonl(Path(args.dataset))
    predictions = predict_rows(rows, model)
    report = {
        "dataset": args.dataset,
        "model": args.model,
        "predictions": args.predictions_output,
        "metrics": evaluate_predictions(predictions),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_jsonl(Path(args.predictions_output), predictions)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
