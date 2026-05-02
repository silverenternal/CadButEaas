#!/usr/bin/env python3
"""Profile token and vision-tile budgets for CadStruct SFT data."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from transformers import AutoProcessor

from sft_utils import SftBudget, encode_sft_row, encoded_sample_stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/cadstruct_sft/train.jsonl")
    parser.add_argument("--model", default="models/vlm/internvl3_5_14b_hf")
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-image-side", type=int, default=768)
    parser.add_argument("--max-vision-tiles", type=int, default=0)
    parser.add_argument("--skip-at-max-length", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    budget = SftBudget(
        max_length=args.max_length,
        max_vision_tiles=args.max_vision_tiles,
        skip_at_max_length=args.skip_at_max_length,
    )
    rows = []
    with Path(args.dataset).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if len(rows) >= args.limit:
                break

    samples = []
    for index, row in enumerate(rows):
        encoded = encode_sft_row(
            processor,
            row,
            max_length=args.max_length,
            max_image_side=args.max_image_side,
        )
        stats = encoded_sample_stats(encoded)
        samples.append(
            {
                "index": index,
                "image": row.get("image"),
                "source_dataset": row.get("source_dataset"),
                **stats.to_dict(),
                "budget_skip_reason": budget.skip_reason(stats),
            }
        )

    summary = {
        "dataset": args.dataset,
        "limit": len(samples),
        "max_length": args.max_length,
        "max_image_side": args.max_image_side,
        "budget": budget.to_dict(),
        "input_tokens": summarize([item["input_tokens"] for item in samples]),
        "supervised_tokens": summarize([item["supervised_tokens"] for item in samples]),
        "vision_tiles": summarize([item["vision_tiles"] for item in samples]),
        "zero_supervised": sum(1 for item in samples if item["supervised_tokens"] == 0),
        "at_max_length": sum(1 for item in samples if item["input_tokens"] >= args.max_length),
        "budget_skipped": sum(1 for item in samples if item["budget_skip_reason"] is not None),
        "budget_skip_reasons": count_values(
            item["budget_skip_reason"] for item in samples if item["budget_skip_reason"] is not None
        ),
        "top_by_tokens": sorted(samples, key=lambda item: item["input_tokens"], reverse=True)[:10],
        "top_by_tiles": sorted(samples, key=lambda item: item["vision_tiles"], reverse=True)[:10],
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")


def summarize(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "mean": None, "p95": None, "max": None}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(len(ordered) * 0.95))
    return {
        "min": ordered[0],
        "mean": round(statistics.mean(ordered), 2),
        "p95": ordered[p95_index],
        "max": ordered[-1],
    }


def count_values(values) -> dict[str, int]:
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


if __name__ == "__main__":
    main()
