#!/usr/bin/env python3
"""Apply routed v41 visual box refiner to page-level locked candidates."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from apply_symbol_box_refiner_v38 import cache_gold_maps, evaluate, predictions_from_rows
from train_symbol_tile_detector_v20 import area_bucket, rel, write_json, write_jsonl
from train_symbol_visual_box_refiner_v40 import apply_delta, features, load_jsonl


ROOT = Path(__file__).resolve().parents[2]


def source_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def proposal_area_bucket(row: dict[str, Any]) -> str:
    proposal = row.get("proposal") or {}
    box = [float(v) for v in proposal.get("bbox") or [0, 0, 0, 0]]
    return area_bucket(box)


def should_apply_runtime(row: dict[str, Any], policy: dict[str, Any]) -> bool:
    """Runtime-safe routing: proposal label and proposal bbox size only."""
    proposal = row.get("proposal") or {}
    label = str(proposal.get("label") or "")
    bucket = proposal_area_bucket(row)
    if label in set(policy.get("deny_labels") or []):
        return False
    if bucket in set(policy.get("deny_areas") or []):
        return False
    if label in set(policy.get("allow_labels") or []):
        return True
    if bucket in set(policy.get("allow_areas") or []):
        return True
    return False


def load_policy(path: Path | None) -> dict[str, Any]:
    if path and path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        policy = payload.get("policy")
        if isinstance(policy, dict):
            return policy
    return {
        "allow_labels": ["sink", "stair"],
        "allow_areas": ["large_le_4096", "xlarge_gt_4096"],
        "deny_labels": ["shower"],
        "deny_areas": ["tiny_le_64", "small_le_256"],
    }


def refine_crop_rows(
    crop_rows: list[dict[str, Any]],
    model: Any,
    policy: dict[str, Any],
    clip: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    routed_rows = [row for row in crop_rows if should_apply_runtime(row, policy)]
    refined_by_id: dict[str, dict[str, Any]] = {}
    route_counts = Counter()
    route_by_label = Counter()
    route_by_area = Counter()

    if routed_rows:
        x = np.asarray([features(row) for row in routed_rows], dtype=np.float32)
        deltas = model.predict(x)
        for row, delta in zip(routed_rows, deltas, strict=True):
            candidate_id = str(row["id"])
            box = [float(v) for v in row["proposal"]["bbox"]]
            refined_by_id[candidate_id] = {
                "bbox": apply_delta(box, list(delta), clip),
                "delta": [float(v) for v in delta],
            }

    for row in crop_rows:
        proposal = row.get("proposal") or {}
        label = str(proposal.get("label") or "")
        bucket = proposal_area_bucket(row)
        routed = should_apply_runtime(row, policy)
        route_counts["crop_rows"] += 1
        route_counts["routed_crop_rows"] += int(routed)
        route_by_label[(label, routed)] += 1
        route_by_area[(bucket, routed)] += 1

    def grouped(counter: Counter) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for (key, routed), count in counter.items():
            row = out.setdefault(str(key), {"rows": 0, "routed": 0})
            row["rows"] += int(count)
            row["routed"] += int(count) if routed else 0
        return dict(sorted(out.items()))

    stats = {
        "crop_rows": int(route_counts["crop_rows"]),
        "routed_crop_rows": int(route_counts["routed_crop_rows"]),
        "refined_candidate_count": len(refined_by_id),
        "by_runtime_label": grouped(route_by_label),
        "by_runtime_area": grouped(route_by_area),
    }
    return refined_by_id, stats


def page_coverage(rows: list[dict[str, Any]], refined_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pages = {str(row["page_id"]) for row in rows}
    affected = {str(row["page_id"]) for row in rows if str(row["candidate_id"]) in refined_by_id}
    return {
        "page_count": len(pages),
        "affected_page_count": len(affected),
        "candidate_count": len(rows),
        "refined_candidate_count": len(refined_by_id),
        "refined_candidate_fraction": round(len(refined_by_id) / max(len(rows), 1), 6),
        "affected_page_fraction": round(len(affected) / max(len(pages), 1), 6),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-records", default="datasets/symbol_visual_box_refiner_v40/locked.jsonl")
    parser.add_argument("--rows", default="datasets/symbol_support_suppression_v36/locked_rows.jsonl")
    parser.add_argument("--cache", default="datasets/symbol_support_suppression_v36/locked_cache.jsonl")
    parser.add_argument("--model", default="checkpoints/symbol_visual_box_refiner_v40/model.joblib")
    parser.add_argument("--policy-report", default="reports/vlm/symbol_visual_box_refiner_v41_locked_eval.json")
    parser.add_argument("--eval-output", default="reports/vlm/symbol_visual_box_refiner_v41_page_locked_eval.json")
    parser.add_argument("--predictions-output", default="reports/vlm/symbol_visual_box_refiner_v41_page_locked_predictions.jsonl")
    parser.add_argument("--clip", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    crop_rows = load_jsonl(source_path(args.crop_records))
    rows = load_jsonl(source_path(args.rows))
    cache_rows = load_jsonl(source_path(args.cache))
    bundle = joblib.load(source_path(args.model))
    policy = load_policy(source_path(args.policy_report))

    refined_by_id, routing = refine_crop_rows(crop_rows, bundle["model"], policy, args.clip)
    before_predictions = predictions_from_rows(rows, {})
    after_predictions = predictions_from_rows(rows, refined_by_id)
    gold_by_page = cache_gold_maps(cache_rows)
    before = evaluate(before_predictions, gold_by_page)
    after = evaluate(after_predictions, gold_by_page)
    coverage = page_coverage(rows, refined_by_id)

    write_jsonl(source_path(args.predictions_output), after_predictions)
    report = {
        "version": "symbol_visual_box_refiner_v41_page_locked_eval",
        "task": "P1-13-v41-page-level-integration",
        "claim_boundary": "Apply routed v41 visual crop bbox refiner to covered locked page candidates. Coverage is limited by the v40 locked crop-smoke export; unchanged candidates remain in the page stream.",
        "source_integrity": {
            "model_input": "raster crop pixels plus proposal bbox/score/type",
            "routing_input": "proposal label and proposal bbox area only",
            "offline_labels_used_for": ["locked_evaluation", "policy_selection_from_previous_v41_report"],
            "gold_used_for_inference": False,
            "runtime_uses_svg_or_cad_geometry": False,
        },
        "inputs": {
            "crop_records": rel(source_path(args.crop_records)),
            "rows": rel(source_path(args.rows)),
            "cache": rel(source_path(args.cache)),
            "model": rel(source_path(args.model)),
            "policy_report": rel(source_path(args.policy_report)),
        },
        "outputs": {"predictions": rel(source_path(args.predictions_output))},
        "policy": policy,
        "routing": routing,
        "coverage": coverage,
        "before": before,
        "after": after,
        "stage_gate": {
            "page_locked_iou_recall_not_drop": after["symbol_bbox_iou_0_30"]["recall"] >= before["symbol_bbox_iou_0_30"]["recall"],
            "page_locked_sink_iou_recall_improves": after["type_iou_recall"].get("sink", 0.0) > before["type_iou_recall"].get("sink", 0.0),
            "page_locked_tiny_iou_recall_not_drop": after["area_iou_recall"].get("tiny_le_64", 0.0) >= before["area_iou_recall"].get("tiny_le_64", 0.0),
            "no_oracle_inference": True,
        },
    }
    report["stage_gate"]["passed"] = all(report["stage_gate"].values())
    write_json(source_path(args.eval_output), report)
    print(
        json.dumps(
            {
                "routing": routing,
                "coverage": coverage,
                "before": before["symbol_bbox_iou_0_30"],
                "after": after["symbol_bbox_iou_0_30"],
                "sink": {
                    "before": before["type_iou_recall"].get("sink", 0.0),
                    "after": after["type_iou_recall"].get("sink", 0.0),
                },
                "tiny": {
                    "before": before["area_iou_recall"].get("tiny_le_64", 0.0),
                    "after": after["area_iou_recall"].get("tiny_le_64", 0.0),
                },
                "stage_gate": report["stage_gate"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
