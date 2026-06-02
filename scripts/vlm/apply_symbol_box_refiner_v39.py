#!/usr/bin/env python3
"""Apply v38 refiner with a dev-selected runtime safety gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from apply_symbol_box_refiner_v38 import cache_gold_maps, evaluate, predictions_from_rows, refine_rows, source_path
from train_symbol_box_refiner_v38 import vector
from train_symbol_support_suppression_v35 import load_jsonl
from train_symbol_tile_detector_v20 import rel, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]


def candidate_area(row: dict[str, Any]) -> float:
    box = [float(v) for v in row.get("bbox") or [0, 0, 1, 1]]
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def delta_magnitude(delta: list[float]) -> float:
    return max(abs(float(v)) for v in delta)


def gate_refined(rows: list[dict[str, Any]], refined_all: dict[str, dict[str, Any]], policy: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    labels = set(policy.get("labels") or [])
    min_area = float(policy.get("min_area", 0.0))
    max_delta = float(policy.get("max_delta", 999.0))
    min_score = float(policy.get("min_score", 0.0))
    never_tiny = bool(policy.get("never_tiny", True))
    for row in rows:
        cid = str(row["candidate_id"])
        refined = refined_all.get(cid)
        if not refined:
            continue
        label = str(row.get("label") or "")
        area = candidate_area(row)
        score = float(row.get("score", 0.0) or 0.0)
        if labels and label not in labels:
            continue
        if area < min_area:
            continue
        if never_tiny and area <= 64:
            continue
        if score < min_score:
            continue
        if delta_magnitude(refined.get("delta") or []) > max_delta:
            continue
        out[cid] = refined
    return out


def load_bundle(model_path: Path) -> tuple[Any, list[str]]:
    bundle = joblib.load(model_path)
    return bundle["model"], list(bundle["feature_names"])


def make_refined_all(rows: list[dict[str, Any]], model: Any, names: list[str], clip: float) -> dict[str, dict[str, Any]]:
    return refine_rows(rows, model, names, clip)


def eval_policy(rows: list[dict[str, Any]], cache_rows: list[dict[str, Any]], refined_all: dict[str, dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    gated = gate_refined(rows, refined_all, policy)
    preds = predictions_from_rows(rows, gated)
    metrics = evaluate(preds, cache_gold_maps(cache_rows))
    metrics["refined_candidate_count"] = len(gated)
    return metrics


def choose_policy(dev_rows: list[dict[str, Any]], dev_cache: list[dict[str, Any]], refined_all: dict[str, dict[str, Any]]) -> dict[str, Any]:
    before = eval_policy(dev_rows, dev_cache, {}, {"labels": []})
    grid = []
    label_sets = [["sink"], ["sink", "stair"], ["appliance", "bathtub", "generic_symbol", "sink", "stair"]]
    for labels in label_sets:
        for min_area in [1024, 4096]:
            for max_delta in [0.10, 0.20, 0.35]:
                for min_score in [0.0, 0.50]:
                    policy = {"labels": labels, "min_area": min_area, "max_delta": max_delta, "min_score": min_score, "never_tiny": True}
                    after = eval_policy(dev_rows, dev_cache, refined_all, policy)
                    view = {
                        "iou_recall_delta": after["symbol_bbox_iou_0_30"]["recall"] - before["symbol_bbox_iou_0_30"]["recall"],
                        "tiny_delta": after["area_iou_recall"].get("tiny_le_64", 0.0) - before["area_iou_recall"].get("tiny_le_64", 0.0),
                        "small_delta": after["area_iou_recall"].get("small_le_256", 0.0) - before["area_iou_recall"].get("small_le_256", 0.0),
                        "sink_delta": after["type_iou_recall"].get("sink", 0.0) - before["type_iou_recall"].get("sink", 0.0),
                    }
                    grid.append({"policy": policy, "after": after, "view": view})
    selected = sorted(
        grid,
        key=lambda row: (
            row["view"]["tiny_delta"] >= 0.0,
            row["view"]["small_delta"] >= 0.0,
            row["view"]["iou_recall_delta"],
            row["view"]["sink_delta"],
            row["after"].get("refined_candidate_count", 0),
        ),
        reverse=True,
    )[0]
    return {"before": before, "selected": selected, "grid_size": len(grid)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="datasets/symbol_support_suppression_v36")
    parser.add_argument("--model", default="checkpoints/symbol_box_refiner_v38/model.joblib")
    parser.add_argument("--eval-output", default="reports/vlm/symbol_box_refiner_v39_page_locked_eval.json")
    parser.add_argument("--predictions-output", default="reports/vlm/symbol_box_refiner_v39_page_locked_predictions.jsonl")
    parser.add_argument("--clip", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = source_path(args.data_dir)
    model, names = load_bundle(source_path(args.model))
    dev_rows = load_jsonl(data_dir / "dev_rows.jsonl")
    locked_rows = load_jsonl(data_dir / "locked_rows.jsonl")
    dev_cache = load_jsonl(data_dir / "dev_cache.jsonl")
    locked_cache = load_jsonl(data_dir / "locked_cache.jsonl")
    dev_refined = make_refined_all(dev_rows, model, names, args.clip)
    locked_refined = make_refined_all(locked_rows, model, names, args.clip)
    choice = choose_policy(dev_rows, dev_cache, dev_refined)
    policy = choice["selected"]["policy"]
    locked_before = eval_policy(locked_rows, locked_cache, {}, {"labels": []})
    locked_after = eval_policy(locked_rows, locked_cache, locked_refined, policy)
    locked_gated = gate_refined(locked_rows, locked_refined, policy)
    locked_predictions = predictions_from_rows(locked_rows, locked_gated)
    write_jsonl(source_path(args.predictions_output), locked_predictions)
    report = {
        "version": "symbol_box_refiner_v39_page_locked_eval",
        "task": "P1-10-refiner-safety-gate-v39",
        "claim_boundary": "Safety gate selected on dev, applied once to locked. Runtime gate uses candidate bbox/score/type and predicted delta magnitude only.",
        "source_integrity": {
            "model_input": "candidate bbox/score/type fields and predicted delta only",
            "offline_labels_used_for": ["dev_gate_selection", "locked_evaluation"],
            "gold_used_for_inference": False,
            "runtime_uses_svg_or_cad_geometry": False,
        },
        "policy": policy,
        "dev_before": choice["before"],
        "dev_selected_after": choice["selected"]["after"],
        "dev_selected_view": choice["selected"]["view"],
        "policy_search": {"grid_size": choice["grid_size"]},
        "locked_before": locked_before,
        "locked_after": locked_after,
        "outputs": {"predictions": rel(source_path(args.predictions_output))},
        "stage_gate": {
            "page_locked_iou_recall_not_drop_vs_before": locked_after["symbol_bbox_iou_0_30"]["recall"] >= locked_before["symbol_bbox_iou_0_30"]["recall"],
            "page_locked_tiny_iou_recall_not_drop_vs_before": locked_after["area_iou_recall"].get("tiny_le_64", 0.0) >= locked_before["area_iou_recall"].get("tiny_le_64", 0.0),
            "page_locked_sink_iou_recall_improves_or_not_drop": locked_after["type_iou_recall"].get("sink", 0.0) >= locked_before["type_iou_recall"].get("sink", 0.0),
            "no_oracle_inference": True,
        },
    }
    report["stage_gate"]["passed"] = all(report["stage_gate"].values())
    write_json(source_path(args.eval_output), report)
    print(json.dumps({"policy": policy, "dev_delta": choice["selected"]["view"], "locked_before": locked_before, "locked_after": locked_after, "stage_gate": report["stage_gate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
