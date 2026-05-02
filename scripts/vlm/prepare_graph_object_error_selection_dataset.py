#!/usr/bin/env python3
"""Build keep/suppress proposal labels from semantic prediction errors."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


OPENING_LABELS = {"door", "window"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal-dir", default="datasets/cadstruct_graph_objects_topology_singleton_proposals")
    parser.add_argument("--prediction-dir", default="reports/vlm")
    parser.add_argument("--prediction-prefix", default="graph_object_topology_singleton_classifier")
    parser.add_argument("--output-dir", default="datasets/cadstruct_graph_object_error_selection")
    parser.add_argument("--min-keep-purity", type=float, default=0.98)
    parser.add_argument("--suppress-low-confidence-errors", action="store_true", default=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "proposal_dir": args.proposal_dir,
        "prediction_dir": args.prediction_dir,
        "prediction_prefix": args.prediction_prefix,
        "min_keep_purity": args.min_keep_purity,
        "splits": {},
        "note": "keep/suppress labels derived from semantic prediction correctness and proposal purity",
    }
    for split in ["train", "dev", "smoke"]:
        proposal_path = Path(args.proposal_dir) / f"{split}.jsonl"
        prediction_path = Path(args.prediction_dir) / f"{args.prediction_prefix}_{split}_predictions.jsonl"
        if split == "smoke" and not prediction_path.exists():
            prediction_path = Path(args.prediction_dir) / f"{args.prediction_prefix}_smoke_predictions.jsonl"
        if not proposal_path.exists() or not prediction_path.exists():
            continue
        manifest["splits"][split] = convert_split(
            proposal_path,
            prediction_path,
            output_dir / f"{split}.jsonl",
            args.min_keep_purity,
        )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def convert_split(proposal_path: Path, prediction_path: Path, output_path: Path, min_keep_purity: float) -> dict[str, Any]:
    proposals = load_jsonl(proposal_path)
    predictions = load_jsonl(prediction_path)
    if len(proposals) != len(predictions):
        raise SystemExit(f"row mismatch for {proposal_path}: {len(proposals)} != {len(predictions)}")

    rows = 0
    groups = 0
    keep_counts = Counter()
    reason_counts = Counter()
    keep_by_semantic_label = defaultdict(Counter)
    with output_path.open("w", encoding="utf-8") as target:
        for proposal_row, prediction_row in zip(proposals, predictions):
            pred_by_id = {int(group["id"]): group for group in prediction_row.get("groups") or []}
            labeled_groups = []
            for group in proposal_row.get("groups") or []:
                pred = pred_by_id.get(int(group["id"]))
                if pred is None:
                    continue
                item = label_group(group, pred, min_keep_purity)
                labeled_groups.append(item)
                groups += 1
                keep_counts[str(item["keep_label"])] += 1
                reason_counts[item["selection_reason"]] += 1
                keep_by_semantic_label[item["semantic_label"]][str(item["keep_label"])] += 1
            if not labeled_groups:
                continue
            proposal_row["groups"] = labeled_groups
            target.write(json.dumps(proposal_row, ensure_ascii=False) + "\n")
            rows += 1
    return {
        "rows": rows,
        "groups": groups,
        "keep_counts": dict(keep_counts),
        "selection_reasons": dict(reason_counts),
        "keep_by_semantic_label": {label: dict(counts) for label, counts in sorted(keep_by_semantic_label.items())},
    }


def label_group(group: dict[str, Any], prediction: dict[str, Any], min_keep_purity: float) -> dict[str, Any]:
    semantic_label = str(group.get("label"))
    predicted_label = str(prediction.get("prediction"))
    purity = float(group.get("label_purity", 1.0) or 0.0)
    member_count = len(group.get("member_ids") or [])
    correct = predicted_label == semantic_label
    keep = correct and purity >= min_keep_purity
    reason = "correct_high_purity" if keep else "semantic_error"
    if correct and purity < min_keep_purity:
        reason = "low_purity"
    if semantic_label == "hard_wall" and predicted_label in OPENING_LABELS:
        keep = False
        reason = "false_opening_on_wall"
    if semantic_label in OPENING_LABELS and predicted_label == "hard_wall":
        keep = False
        reason = "missed_opening_as_wall"
    if semantic_label in OPENING_LABELS and member_count > 1 and purity < 1.0:
        keep = False
        reason = "mixed_opening_component"

    item = dict(group)
    item["semantic_label"] = semantic_label
    item["semantic_prediction"] = predicted_label
    item["semantic_confidence"] = float(prediction.get("confidence", 0.0) or 0.0)
    item["keep_label"] = 1 if keep else 0
    item["selection_label"] = "keep" if keep else "suppress"
    item["selection_reason"] = reason
    item["label"] = item["selection_label"]
    return item


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    main()
