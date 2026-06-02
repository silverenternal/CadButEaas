#!/usr/bin/env python3
"""Sweep fail-closed visual override thresholds for boundary door/window repair."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_boundary_type_fusion_v24 import bbox, center_covered, gold_by_row, iou  # noqa: E402


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def base_label(candidate: dict[str, Any]) -> str:
    value = str(candidate.get("fusion_prediction") or candidate.get("prediction") or "")
    return value if value in {"hard_wall", "door", "window"} else "hard_wall"


def visual_label(candidate: dict[str, Any], door_threshold: float, window_threshold: float) -> str:
    base = base_label(candidate)
    trace = candidate.get("visual_crop_context_trace") if isinstance(candidate.get("visual_crop_context_trace"), dict) else {}
    pred = str(trace.get("visual_prediction") or "")
    conf = float(trace.get("visual_confidence") or 0.0)
    if base == "hard_wall" and pred == "door" and conf >= door_threshold:
        return "door"
    if base == "hard_wall" and pred == "window" and conf >= window_threshold:
        return "window"
    return base


def evaluate(rows: list[dict[str, Any]], gold: dict[str, list[dict[str, Any]]], door_threshold: float, window_threshold: float, cap: int) -> dict[str, Any]:
    total = proposal_hit = classified_hit = predicted = 0
    per_label: dict[str, Counter[str]] = defaultdict(Counter)
    wrong_pairs = Counter()
    for row in rows:
        row_id = str(row.get("id"))
        candidates = (row.get("candidate_stream") or [])[:cap]
        predicted += len(candidates)
        labels = [visual_label(candidate, door_threshold, window_threshold) for candidate in candidates]
        for gold_item in gold.get(row_id, []):
            total += 1
            label = gold_item["label"]
            per_label[label]["gold"] += 1
            matches = []
            for idx, candidate in enumerate(candidates):
                cb = bbox(candidate.get("bbox"))
                if cb is not None and (center_covered(cb, gold_item["bbox"]) or iou(cb, gold_item["bbox"]) >= 0.30):
                    matches.append(idx)
            if matches:
                proposal_hit += 1
                per_label[label]["proposal_matched"] += 1
            if any(labels[idx] == label for idx in matches):
                classified_hit += 1
                per_label[label]["classified_matched"] += 1
            elif matches:
                wrong_pairs[f"{label}->{labels[matches[0]]}"] += 1
    return {
        "door_threshold": door_threshold,
        "window_threshold": window_threshold,
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
    parser.add_argument("--predictions", default="reports/vlm/boundary_door_window_crop_context_failclosed_v24_locked50_predictions.jsonl")
    parser.add_argument("--dataset", default="datasets/boundary_expert_public_raster_v19")
    parser.add_argument("--split", default="locked")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--cap", type=int, default=800)
    parser.add_argument("--output", default="reports/vlm/boundary_door_window_crop_context_failclosed_v24_threshold_sweep.json")
    args = parser.parse_args()

    rows = load_jsonl(ROOT / args.predictions, args.limit or None)
    gold = gold_by_row(ROOT / args.dataset / f"{args.split}.jsonl", args.limit or None)
    thresholds = [round(value / 100.0, 2) for value in range(50, 100, 5)]
    results = []
    for door_threshold in thresholds:
        for window_threshold in thresholds:
            result = evaluate(rows, gold, door_threshold, window_threshold, args.cap)
            results.append(result)
    baseline = evaluate(rows, gold, 1.01, 1.01, args.cap)
    no_drop = [
        result
        for result in results
        if result["classified_recall"] >= baseline["classified_recall"]
        and result["per_label"]["door"]["classified_recall"] >= baseline["per_label"]["door"]["classified_recall"]
        and result["per_label"]["window"]["classified_recall"] >= baseline["per_label"]["window"]["classified_recall"]
    ]
    best = max(results, key=lambda item: (item["classified_recall"], item["per_label"]["door"]["classified_recall"], item["per_label"]["window"]["classified_recall"]))
    best_no_drop = max(
        no_drop,
        key=lambda item: (item["classified_recall"], item["per_label"]["door"]["classified_recall"], item["per_label"]["window"]["classified_recall"]),
    ) if no_drop else None
    report = {
        "version": "boundary_failclosed_visual_threshold_sweep_v24",
        "predictions": args.predictions,
        "baseline_no_visual": baseline,
        "best": best,
        "best_no_drop": best_no_drop,
        "no_drop_count": len(no_drop),
        "top10": sorted(results, key=lambda item: item["classified_recall"], reverse=True)[:10],
    }
    write_json(ROOT / args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
