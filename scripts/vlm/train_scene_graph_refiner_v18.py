#!/usr/bin/env python3
"""Train/evaluate a quality-gated keep/drop refiner for v18 scene candidates."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/vlm"
CHECKPOINT = ROOT / "checkpoints/scene_graph_refiner_v18/refiner_policy.json"
DEFAULT_DATASET = ROOT / "datasets/image_only_scene_graph_refiner_v18/locked.jsonl"
DEFAULT_INPUT = REPORT / "detector_adapter_v18_routed_candidates.jsonl"
DEFAULT_EVAL = REPORT / "scene_graph_refiner_v18_eval.json"
DEFAULT_DECISIONS = REPORT / "scene_graph_refiner_v18_decisions.jsonl"
DEFAULT_FINAL = REPORT / "scene_graph_refiner_v18_final_predictions.jsonl"


def integrity() -> dict[str, Any]:
    return {
        "source_mode": "image_only_raster_moe",
        "svg_candidate_ids_used": False,
        "annotation_geometry_used_at_inference": False,
        "model_input": "raster_image_only",
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def score(item: dict[str, Any], weights: dict[str, float]) -> float:
    feats = item.get("features") if isinstance(item.get("features"), dict) else {}
    value = 0.0
    for key, weight in weights.items():
        value += float(feats.get(key) or 0.0) * weight
    return value


def policy_grid() -> list[dict[str, Any]]:
    weights = {
        "relation_support_then_confidence": {
            "relation_support_score_max": 2.5,
            "relation_support_count": 0.25,
            "detector_confidence": 1.0,
            "ocr_confidence": 0.3,
            "type_confidence": 0.4,
        },
        "confidence_only": {"detector_confidence": 1.0},
    }
    cap_sets = [
        {"boundary": 200, "space": 20, "symbol": 100, "text": 20},
        {"boundary": 300, "space": 30, "symbol": 150, "text": 20},
        {"boundary": 400, "space": 40, "symbol": 250, "text": 25},
        {"boundary": 500, "space": 50, "symbol": 300, "text": 30},
        {"boundary": 600, "space": 60, "symbol": 350, "text": 35},
        {"boundary": 700, "space": 70, "symbol": 425, "text": 40},
        {"boundary": 800, "space": 100, "symbol": 500, "text": 50},
    ]
    return [
        {"name": f"{ranker}_b{caps['boundary']}_s{caps['space']}_y{caps['symbol']}_t{caps['text']}", "ranker": ranker, "weights": weight, "caps": caps}
        for ranker, weight in weights.items()
        for caps in cap_sets
    ]


def evaluate_policy(items: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    by_page_family: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    total = Counter()
    positives = Counter()
    for item in items:
        family = str(item.get("family"))
        by_page_family[(str(item.get("row_id")), family)].append(item)
        total[family] += 1
        if item.get("label_keep"):
            positives[family] += 1

    kept_items: list[dict[str, Any]] = []
    for (row_id, family), group in by_page_family.items():
        cap = int((policy.get("caps") or {}).get(family, len(group)))
        ordered = sorted(group, key=lambda item: score(item, policy.get("weights") or {}), reverse=True)
        kept_items.extend(ordered[:cap])

    kept = Counter()
    kept_positive = Counter()
    typed_pred = typed_tp = 0
    for item in kept_items:
        family = str(item.get("family"))
        kept[family] += 1
        if item.get("label_keep"):
            kept_positive[family] += 1
        if family == "symbol":
            typed_pred += 1
            if item.get("typed_correct"):
                typed_tp += 1

    metrics: dict[str, Any] = {}
    for family in sorted(total):
        baseline_recall = positives[family] / max(positives[family], 1)
        refined_recall = kept_positive[family] / max(positives[family], 1)
        precision = kept_positive[family] / max(kept[family], 1)
        reduction = 1.0 - kept[family] / max(total[family], 1)
        metrics[family] = {
            "baseline_candidates": total[family],
            "baseline_positive": positives[family],
            "kept_candidates": kept[family],
            "kept_positive": kept_positive[family],
            "candidate_reduction": round(reduction, 6),
            "keep_precision": round(precision, 6),
            "recall": round(refined_recall, 6),
            "recall_drop_from_adapter": round(baseline_recall - refined_recall, 6),
        }
    all_total = sum(total.values())
    all_kept = sum(kept.values())
    all_pos = sum(positives.values())
    all_kept_pos = sum(kept_positive.values())
    symbol_precision = typed_tp / max(typed_pred, 1)
    symbol_gold = positives["symbol"]
    symbol_f1 = 0.0
    symbol_recall = typed_tp / max(symbol_gold, 1)
    if symbol_precision + symbol_recall:
        symbol_f1 = 2 * symbol_precision * symbol_recall / (symbol_precision + symbol_recall)
    return {
        "policy": {"name": policy["name"], "ranker": policy["ranker"], "caps": policy["caps"]},
        "candidate_total": all_total,
        "candidate_kept": all_kept,
        "candidate_reduction": round(1.0 - all_kept / max(all_total, 1), 6),
        "keep_precision": round(all_kept_pos / max(all_kept, 1), 6),
        "keep_recall": round(all_kept_pos / max(all_pos, 1), 6),
        "family_metrics": metrics,
        "symbol_typed_after_refiner": {
            "predicted": typed_pred,
            "typed_true_positive": typed_tp,
            "precision": round(symbol_precision, 6),
            "recall": round(symbol_recall, 6),
            "f1": round(symbol_f1, 6),
        },
    }


def choose_policy(results: list[dict[str, Any]]) -> dict[str, Any]:
    feasible = [
        item for item in results
        if item["candidate_reduction"] >= 0.70
        and item["family_metrics"].get("boundary", {}).get("recall_drop_from_adapter", 1.0) <= 0.05
        and item["family_metrics"].get("space", {}).get("recall_drop_from_adapter", 1.0) <= 0.05
    ]
    if feasible:
        return max(feasible, key=lambda item: (item["keep_precision"], item["keep_recall"], item["candidate_reduction"]))
    # Fail closed: when no policy satisfies both reduction and recall gates,
    # preserve candidate recall in the emitted stream and leave aggressive
    # reduction as a diagnostic candidate, not an adopted final output.
    return max(results, key=lambda item: (item["keep_recall"], item["keep_precision"], item["candidate_reduction"]))


def choose_aggressive_policy(results: list[dict[str, Any]]) -> dict[str, Any]:
    return max(results, key=lambda item: (item["candidate_reduction"] >= 0.70, item["keep_recall"], item["keep_precision"]))


def write_outputs(items: list[dict[str, Any]], policy_eval: dict[str, Any], input_rows: list[dict[str, Any]], decisions_path: Path, final_path: Path) -> None:
    caps = policy_eval["policy"]["caps"]
    weights = next(p["weights"] for p in policy_grid() if p["name"] == policy_eval["policy"]["name"])
    scores_by_id = {item["candidate_id"]: score(item, weights) for item in items}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[(item["row_id"], item["family"])].append(item)
    keep_ids: set[str] = set()
    for (row_id, family), group in grouped.items():
        ordered = sorted(group, key=lambda item: scores_by_id.get(item["candidate_id"], 0.0), reverse=True)
        keep_ids.update(str(item["candidate_id"]) for item in ordered[: int(caps.get(family, len(group)))])

    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    with decisions_path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps({
                "row_id": item["row_id"],
                "candidate_id": item["candidate_id"],
                "family": item["family"],
                "score": round(scores_by_id.get(item["candidate_id"], 0.0), 6),
                "decision": "keep" if item["candidate_id"] in keep_ids else "drop",
                "label_keep": item.get("label_keep"),
                "source_integrity": integrity(),
            }, ensure_ascii=False) + "\n")

    with final_path.open("w", encoding="utf-8") as handle:
        for row in input_rows:
            stream = []
            for cand in ((row.get("scene_graph") or {}).get("candidate_stream") or []):
                if cand.get("candidate_id") in keep_ids:
                    kept = dict(cand)
                    kept["refiner_score"] = round(scores_by_id.get(cand.get("candidate_id"), 0.0), 6)
                    kept["refiner_decision"] = "keep"
                    stream.append(kept)
            handle.write(json.dumps({
                "id": row.get("id"),
                "image": row.get("image"),
                "image_size": row.get("image_size") or [512, 512],
                "source_integrity": integrity(),
                "route_trace": {**integrity(), "stage": "scene_graph_refiner_v18_keep_drop"},
                "scene_graph": {
                    "nodes": [],
                    "relations": [],
                    "candidate_stream": stream,
                    "candidate_counts": dict(Counter(c.get("family") for c in stream)),
                },
            }, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--checkpoint", default=str(CHECKPOINT))
    parser.add_argument("--eval-output", default=str(DEFAULT_EVAL))
    parser.add_argument("--decisions-output", default=str(DEFAULT_DECISIONS))
    parser.add_argument("--final-output", default=str(DEFAULT_FINAL))
    args = parser.parse_args()

    items = load_jsonl(Path(args.dataset))
    results = [evaluate_policy(items, policy) for policy in policy_grid()]
    selected = choose_policy(results)
    aggressive = choose_aggressive_policy(results)
    impossible_reasons = []
    if selected["candidate_reduction"] < 0.70:
        impossible_reasons.append("best policy could not reduce candidates by 70% while retaining useful recall")
    if selected["family_metrics"].get("boundary", {}).get("recall_drop_from_adapter", 1.0) > 0.05:
        impossible_reasons.append("boundary recall drop exceeds 0.05 under any >=70% reduction policy")
    if selected["family_metrics"].get("space", {}).get("recall_drop_from_adapter", 1.0) > 0.05:
        impossible_reasons.append("space recall drop exceeds 0.05 under selected policy")
    if selected["symbol_typed_after_refiner"]["precision"] < 0.05 or selected["symbol_typed_after_refiner"]["f1"] < 0.05:
        impossible_reasons.append("symbol typed precision/F1 remain below 0.05 adoption floor")

    report = {
        "task": "IMG-MOE-V18-P1-010",
        "status": "adoptable" if not impossible_reasons else "not_adoptable_blocked_by_detector_quality",
        "selected_policy": selected,
        "aggressive_diagnostic_policy": aggressive,
        "policy_sweep": results,
        "quality_gates": {
            "candidate_reduction_at_least_70_percent": selected["candidate_reduction"] >= 0.70,
            "boundary_recall_drop_at_most_0_05": selected["family_metrics"].get("boundary", {}).get("recall_drop_from_adapter", 1.0) <= 0.05,
            "space_recall_drop_at_most_0_05": selected["family_metrics"].get("space", {}).get("recall_drop_from_adapter", 1.0) <= 0.05,
            "symbol_typed_precision_at_least_0_05": selected["symbol_typed_after_refiner"]["precision"] >= 0.05,
            "symbol_typed_f1_at_least_0_05": selected["symbol_typed_after_refiner"]["f1"] >= 0.05,
        },
        "blockers": impossible_reasons,
        "source_integrity": integrity(),
        "gold_loaded_after_inference_for_training_and_evaluation_only": True,
        "gold_used_for_inference": False,
    }
    write_json(Path(args.eval_output), report)
    checkpoint = {"model_type": "scene_graph_refiner_v18_rule_policy", "selected_policy": selected["policy"], "source_integrity": integrity()}
    write_json(Path(args.checkpoint), checkpoint)
    write_outputs(items, selected, load_jsonl(Path(args.input)), Path(args.decisions_output), Path(args.final_output))
    print(json.dumps({"status": report["status"], "selected_policy": selected["policy"], "candidate_reduction": selected["candidate_reduction"], "blockers": impossible_reasons}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
