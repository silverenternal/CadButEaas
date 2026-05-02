#!/usr/bin/env python3
"""Build reviewed clean room-eval dataset from proposal/room boundary review labels.

This script replays human labels to a gold dataset split (typically
`locked_test`) and emits a parallel reviewed dataset. It never mutates the
source dataset in place.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


ACCEPT_LABELS = {
    "accept_typed",
    "accept",
    "accept_pred",
}
KEEP_LABELS = {
    "keep_gold",
    "keep",
    "keep_room",
    "unclear",
    "",
}
EXCLUDE_LABELS = {
    "exclude",
    "exclude_ambiguous",
}
REMAP_LABELS = {
    "remap_to_unknown",
    "remap_unknown",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="datasets/cadstruct_cubicasa5k_moe_locked")
    parser.add_argument("--predictions", default="checkpoints/cadstruct_moe_room_space_hierarchical_sklearn_v5_t046/locked_test_predictions.jsonl")
    parser.add_argument("--review-csv", default="reports/vlm/room_ambiguity_review_pack_v1/review_queue.csv")
    parser.add_argument("--output-dir", default="datasets/cadstruct_cubicasa5k_moe_locked_reviewed_v1")
    parser.add_argument("--splits", default="train,dev,locked_test,smoke")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    review = load_review(Path(args.review_csv))
    predictions = load_predictions(Path(args.predictions))

    manifest: dict[str, Any] = {
        "input_dir": str(input_dir),
        "predictions": str(args.predictions),
        "review_csv": str(args.review_csv),
        "splits": {},
        "review_stats": {},
    }
    total_stats = Counter()

    for split in splits:
        source_path = input_dir / f"{split}.jsonl"
        if not source_path.exists():
            print(f"warning: missing split {split}, skipping")
            continue

        rows = load_jsonl(source_path)
        split_rows: list[dict[str, Any]] = []
        split_stats = Counter()

        for row in rows:
            annotation = str(row.get("annotation_path") or "")
            row_predictions = predictions.get(annotation, {})
            adjusted = apply_review_to_row(row, review.get(annotation, {}), row_predictions, split_stats)
            split_rows.append(adjusted)

        split_path = output_dir / f"{split}.jsonl"
        write_jsonl(split_path, split_rows)

        split_stats["input_records"] = len(rows)
        split_stats["output_records"] = len(split_rows)
        split_stats["rooms_input"] = sum(len((row.get("expected_json") or {}).get("room_candidates") or []) for row in rows)
        split_stats["rooms_output"] = sum(len((row.get("expected_json") or {}).get("room_candidates") or []) for row in split_rows)

        # keep-room-type support with audit of review operations
        split_stats["label_counts_after"] = count_room_labels(split_rows)
        manifest["splits"][split] = dict(split_stats)
        for key, value in split_stats.items():
            if isinstance(value, (int, float)):
                total_stats[f"split:{split}:{key}"] += int(value)
            else:
                continue

    manifest["review_stats"] = {
        **{f"{key}": int(value) for key, value in total_stats.items()},
        "clean_total_records": sum(int(item["output_records"]) for item in manifest["splits"].values() if isinstance(item, dict)),
        "review_pairs_total": len({(item[0], item[1]) for item in review_to_pairs(review)}),
    }
    manifest["output_dir"] = str(output_dir)
    output_report = output_dir / "clean_room_review_manifest.json"
    output_report.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def review_to_pairs(review: dict[str, dict[str, str]]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for annotation, rooms in review.items():
        for room_id in rooms:
            pairs.add((annotation, room_id))
    return pairs


def apply_review_to_row(
    row: dict[str, Any],
    review_map: dict[str, str],
    pred_map: dict[str, str],
    stats: Counter,
) -> dict[str, Any]:
    output = deepcopy(row)
    expected = output.get("expected_json") or {}
    room_candidates = expected.get("room_candidates") or []

    adjusted_rooms: list[dict[str, Any]] = []
    removed = 0
    accepted = 0
    kept = 0
    remapped = 0
    for item in room_candidates:
        if not isinstance(item, dict):
            continue
        candidate = dict(item)
        room_id = str(candidate.get("id") or "")
        action = review_map.get(room_id, "")
        if action in KEEP_LABELS:
            kept += 1
            candidate["review_state"] = action or "keep"
        elif action in EXCLUDE_LABELS:
            removed += 1
            stats["exclude"] += 1
            candidate["review_state"] = action
            continue
        elif action in ACCEPT_LABELS:
            pred = pred_map.get(room_id)
            if pred and pred != str(candidate.get("room_type") or "room"):
                candidate["room_type"] = pred
                accepted += 1
                stats["accept_typed"] += 1
            else:
                stats["accept_typed_noop"] += 1
            candidate["review_state"] = action
        elif action in REMAP_LABELS:
            if candidate.get("room_type") == "room":
                candidate["room_type"] = "unknown_room"
            remapped += 1
            stats["remap_to_unknown"] += 1
            candidate["review_state"] = action
        elif action:
            # Unknown label is retained as a guardrail to avoid silent mis-handle.
            raise ValueError(f"Unknown review_label {action!r} for room {room_id}")
        else:
            candidate["review_state"] = "unchanged"

        adjusted_rooms.append(candidate)

    if removed:
        stats["rooms_removed"] += removed
    if accepted:
        stats["rooms_accepted"] += accepted
    if kept:
        stats["rooms_kept"] += kept
    if remapped:
        stats["rooms_remapped_unknown"] += remapped

    expected["room_candidates"] = adjusted_rooms
    output["expected_json"] = expected
    output.setdefault("review_metadata", {})
    output["review_metadata"].update(
        {
            "review_applied": True,
            "review_records": len(review_map),
            "review_rooms_removed": removed,
            "review_rooms_accepted": accepted,
            "review_rooms_remapped_unknown": remapped,
            "review_rooms_kept": kept,
        }
    )
    return output


def load_review(path: Path) -> dict[str, dict[str, str]]:
    review: dict[str, dict[str, str]] = {}
    if not path.exists():
        print(f"warning: missing review csv {path}, continue with empty review")
        return review
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            annotation = str(row.get("annotation") or "").strip()
            room_id = str(row.get("room_id") or "").strip()
            label = str(row.get("review_label") or "").strip()
            if not annotation:
                continue
            if room_id == "":
                continue
            review.setdefault(annotation, {})[room_id] = normalize_label(label)
    return review


def normalize_label(label: str) -> str:
    if not label:
        return ""
    normalized = str(label).strip().lower().replace(" ", "_")
    return normalized


def load_predictions(path: Path) -> dict[str, dict[str, str]]:
    pred_rows: dict[str, dict[str, str]] = {}
    if not path.exists():
        print(f"warning: missing predictions {path}, review accept-tuned rooms cannot inherit predictions")
        return pred_rows
    for row in load_jsonl(path):
        annotation = str(row.get("annotation") or "")
        pred_map: dict[str, str] = {}
        for room in row.get("rooms") or []:
            room_id = str(room.get("id") or "")
            pred = str(room.get("prediction") or "")
            if room_id:
                pred_map[room_id] = pred
        pred_rows[annotation] = pred_map
    return pred_rows


def count_room_labels(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        rooms = (row.get("expected_json") or {}).get("room_candidates") or []
        for room in rooms:
            if isinstance(room, dict):
                counts[str(room.get("room_type") or "room")] += 1
    return dict(counts)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
