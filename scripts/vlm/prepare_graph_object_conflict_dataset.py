#!/usr/bin/env python3
"""Prepare per-primitive conflict-ranking rows from object proposals."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SEMANTIC_LABELS = ["hard_wall", "door", "window"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal-dir", default="datasets/cadstruct_graph_objects_topology_singleton_proposals")
    parser.add_argument("--prediction-dir", default="reports/vlm")
    parser.add_argument("--prediction-prefix", default="graph_object_topology_singleton_classifier")
    parser.add_argument("--output-dir", default="datasets/cadstruct_graph_object_conflicts")
    parser.add_argument("--require-singleton-ground-truth", action="store_true", default=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "proposal_dir": args.proposal_dir,
        "prediction_dir": args.prediction_dir,
        "prediction_prefix": args.prediction_prefix,
        "labels": ["suppress", "keep"],
        "splits": {},
        "note": "per-primitive candidate rows; label=keep if candidate semantic prediction matches primitive ground truth",
    }
    for split in ["train", "dev", "smoke"]:
        proposal_path = Path(args.proposal_dir) / f"{split}.jsonl"
        prediction_path = Path(args.prediction_dir) / f"{args.prediction_prefix}_{split}_predictions.jsonl"
        if split == "smoke" and not prediction_path.exists():
            prediction_path = Path(args.prediction_dir) / f"{args.prediction_prefix}_smoke_predictions.jsonl"
        if not proposal_path.exists() or not prediction_path.exists():
            continue
        manifest["splits"][split] = convert_split(proposal_path, prediction_path, output_dir / f"{split}.jsonl")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def convert_split(proposal_path: Path, prediction_path: Path, output_path: Path) -> dict[str, Any]:
    proposals = load_jsonl(proposal_path)
    predictions = load_jsonl(prediction_path)
    if len(proposals) != len(predictions):
        raise SystemExit(f"row mismatch for {proposal_path}: {len(proposals)} != {len(predictions)}")

    rows = 0
    candidates = 0
    label_counts = Counter()
    target_candidate_counts = []
    keep_by_semantic = defaultdict(Counter)
    with output_path.open("w", encoding="utf-8") as target:
        for proposal_row, prediction_row in zip(proposals, predictions):
            sample = to_conflict_sample(proposal_row, prediction_row)
            if not sample["groups"]:
                continue
            target.write(json.dumps(sample, ensure_ascii=False) + "\n")
            rows += 1
            candidates += len(sample["groups"])
            per_target = Counter()
            for row in sample["groups"]:
                label_counts[row["label"]] += 1
                keep_by_semantic[row["candidate_semantic"]][row["label"]] += 1
                per_target[int(row["target_id"])] += 1
            target_candidate_counts.extend(per_target.values())
    return {
        "rows": rows,
        "candidate_rows": candidates,
        "label_counts": dict(label_counts),
        "keep_by_candidate_semantic": {label: dict(counts) for label, counts in sorted(keep_by_semantic.items())},
        "candidates_per_primitive": summarize(target_candidate_counts),
    }


def to_conflict_sample(proposal_row: dict[str, Any], prediction_row: dict[str, Any]) -> dict[str, Any]:
    groups = {int(group["id"]): group for group in proposal_row.get("groups") or []}
    predictions = {int(group["id"]): group for group in prediction_row.get("groups") or []}
    gt_by_target = singleton_ground_truth(groups.values())
    rows = []
    row_id = 0
    for group_id, group in groups.items():
        pred = predictions.get(group_id)
        if pred is None:
            continue
        candidate_semantic = str(pred.get("prediction"))
        if candidate_semantic not in SEMANTIC_LABELS:
            continue
        member_ids = [int(node_id) for node_id in group.get("member_ids") or []]
        if not member_ids:
            continue
        for target_id in member_ids:
            target_label = gt_by_target.get(target_id)
            if target_label is None:
                continue
            features = conflict_features(group.get("features") or {}, pred, len(member_ids))
            keep = candidate_semantic == target_label
            rows.append(
                {
                    "id": row_id,
                    "proposal_id": group_id,
                    "target_id": target_id,
                    "target_semantic": target_label,
                    "candidate_semantic": candidate_semantic,
                    "semantic_confidence": float(pred.get("confidence", 0.0) or 0.0),
                    "features": features,
                    "label": "keep" if keep else "suppress",
                }
            )
            row_id += 1
    return {
        "image": proposal_row.get("image"),
        "source_dataset": proposal_row.get("source_dataset"),
        "groups": rows,
    }


def conflict_features(base: dict[str, Any], pred: dict[str, Any], member_count: int) -> dict[str, Any]:
    features = dict(base)
    semantic = str(pred.get("prediction"))
    confidence = float(pred.get("confidence", 0.0) or 0.0)
    features["semantic_confidence"] = confidence
    for label in SEMANTIC_LABELS:
        features[f"candidate_semantic_{label}"] = 1.0 if semantic == label else 0.0
    features["candidate_member_fraction"] = 1.0 / max(member_count, 1)
    features["candidate_is_singleton"] = 1.0 if member_count == 1 else 0.0
    return features


def singleton_ground_truth(groups: Any) -> dict[int, str]:
    result = {}
    for group in groups:
        member_ids = group.get("member_ids") or []
        label = group.get("label")
        if len(member_ids) == 1 and label in SEMANTIC_LABELS:
            result[int(member_ids[0])] = str(label)
    return result


def summarize(values: list[int]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "max": 0.0, "p50": 0.0, "p95": 0.0}
    ordered = sorted(values)
    return {
        "mean": round(sum(values) / len(values), 3),
        "max": float(max(values)),
        "p50": float(percentile(ordered, 0.50)),
        "p95": float(percentile(ordered, 0.95)),
    }


def percentile(ordered: list[int], q: float) -> int:
    if not ordered:
        return 0
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[index]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    main()
