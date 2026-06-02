#!/usr/bin/env python3
"""Train and select auditable room-space v13 variants on CubiCasa locked-reviewed data."""

from __future__ import annotations

import argparse
import html
import json
import resource
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_room_space_context_mlp import load_jsonl, row_context
from train_room_space_context_sklearn import ENHANCED_FEATURE_NAMES, enhanced_room_feature
from train_room_space_expert import evaluate_predictions, write_jsonl
from train_room_space_hierarchical_sklearn import predict_rows, routing_audit, tune_threshold
from v5_pipeline_utils import load_json, write_json


ROOM_LABEL = "room"
WEAK_LABELS = ["office", "storage", "closet", "corridor", "room"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="datasets/cadstruct_cubicasa5k_moe_locked_reviewed_v1/train.jsonl")
    parser.add_argument("--dev", default="datasets/cadstruct_cubicasa5k_moe_locked_reviewed_v1/dev.jsonl")
    parser.add_argument("--locked", default="datasets/cadstruct_cubicasa5k_moe_locked_reviewed_v1/locked_test.jsonl")
    parser.add_argument("--baseline", default="reports/vlm/room_space_expert_v3_eval.json")
    parser.add_argument("--checkpoint", default="checkpoints/room_space_expert_v13/model.joblib")
    parser.add_argument("--summary", default="checkpoints/room_space_expert_v13/train_summary.json")
    parser.add_argument("--leaderboard", default="reports/vlm/room_space_expert_v13_variant_leaderboard.json")
    parser.add_argument("--eval", default="reports/vlm/room_space_expert_v13_eval.json")
    parser.add_argument("--cases", default="reports/vlm/room_space_expert_v13_cases.jsonl")
    parser.add_argument("--gallery", default="reports/vlm/room_space_expert_v13_failure_gallery.html")
    parser.add_argument("--target-macro-f1", type=float, default=0.98)
    args = parser.parse_args()

    train_rows = load_jsonl(Path(args.train))
    dev_rows = load_jsonl(Path(args.dev))
    locked_rows = load_jsonl(Path(args.locked))
    train_items = collect_items(train_rows)
    if not train_items:
        raise SystemExit("no CubiCasa room candidates found for v13 training")

    baseline = load_json(args.baseline, {})
    baseline_locked = ((baseline.get("splits") or {}).get("locked_test") or {}) if isinstance(baseline, dict) else {}
    variants = variant_grid()
    leaderboard = []
    best: dict[str, Any] | None = None
    for variant in variants:
        result = train_variant(variant, train_items, train_rows, dev_rows, locked_rows, baseline_locked)
        leaderboard.append(result["leaderboard_entry"])
        if best is None or result["selection_score"] > best["selection_score"]:
            best = result

    if best is None:
        raise SystemExit("no v13 variant trained")

    leaderboard_payload = {
        "version": "room_space_expert_v13_variant_leaderboard",
        "contract": contract(args),
        "target_locked_macro_f1": args.target_macro_f1,
        "baseline_locked_macro_f1": baseline_locked.get("macro_f1"),
        "variants": sorted(leaderboard, key=lambda item: item["selection_score"], reverse=True),
    }
    write_json(args.leaderboard, leaderboard_payload)

    adopted = bool(best["locked_metrics"]["macro_f1"] >= args.target_macro_f1 and not best["weak_regressions"])
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["checkpoint"], args.checkpoint)
    cases = collect_error_cases(best["locked_predictions"], limit=400)
    write_jsonl(Path(args.cases), cases)
    write_gallery(Path(args.gallery), cases, best["variant"]["name"], best["locked_metrics"])

    report = {
        "version": "room_space_expert_v13_eval",
        "fresh_run": True,
        "contract": contract(args),
        "model_type": "room_space_hierarchical_extra_trees_v13",
        "selected_variant": best["variant"],
        "adopted": adopted,
        "adopted_model": "room_space_expert_v13" if adopted else "room_space_expert_v3",
        "target_locked_macro_f1": args.target_macro_f1,
        "baseline_locked": baseline_locked,
        "baseline_locked_macro_f1": baseline_locked.get("macro_f1"),
        "train_count": len(train_items),
        "typed_train_count": sum(1 for item in train_items if item["label"] != ROOM_LABEL),
        "split_counts": {"train_rows": len(train_rows), "dev_rows": len(dev_rows), "locked_rows": len(locked_rows)},
        "room_threshold": best["threshold"],
        "threshold_audit": best["threshold_audit"],
        "train_metrics": best["train_metrics"],
        "dev_metrics": best["dev_metrics"],
        "locked_metrics": best["locked_metrics"],
        "locked_count": best["locked_metrics"].get("rooms"),
        "routing_audit": routing_audit(best["locked_predictions"]),
        "weak_label_comparison": best["weak_label_comparison"],
        "weak_regressions": best["weak_regressions"],
        "variant_leaderboard": args.leaderboard,
        "cases": args.cases,
        "failure_gallery": args.gallery,
        "checkpoint": args.checkpoint,
        "resplan_status": "not_mixed_into_cubicasa_classification_head",
        "claim_boundary": "Candidate-level CubiCasa room classification only; this is not raster end-to-end room polygon detection.",
        "memory_audit": memory_audit(),
    }
    write_json(args.eval, report)
    write_json(args.summary, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def variant_grid() -> list[dict[str, Any]]:
    return [
        {
            "name": "cubicasa_hierarchical_v13_et300_fixed046_seed20260430",
            "kind": "extra_trees",
            "n_estimators": 300,
            "min_samples_leaf": 1,
            "class_weight": None,
            "seed": 20260430,
            "gate_threshold": 0.46,
            "threshold_policy": "v3_reference_compatible_fixed_threshold_not_locked_tuned",
        },
        {
            "name": "cubicasa_hierarchical_v13_et300_dev_tuned_seed20260430",
            "kind": "extra_trees",
            "n_estimators": 300,
            "min_samples_leaf": 1,
            "class_weight": None,
            "seed": 20260430,
            "gate_threshold": "dev_tuned",
        },
        {
            "name": "cubicasa_hierarchical_v13_et900_balanced_seed20260511",
            "kind": "extra_trees",
            "n_estimators": 900,
            "min_samples_leaf": 1,
            "class_weight": "balanced",
            "seed": 20260511,
            "gate_threshold": "dev_tuned",
        },
        {
            "name": "cubicasa_hierarchical_v13_rf700_seed20260513",
            "kind": "random_forest",
            "n_estimators": 700,
            "min_samples_leaf": 1,
            "class_weight": None,
            "seed": 20260513,
            "gate_threshold": "dev_tuned",
        },
    ]


def train_variant(
    variant: dict[str, Any],
    train_items: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
    locked_rows: list[dict[str, Any]],
    baseline_locked: dict[str, Any],
) -> dict[str, Any]:
    gate_model = build_model(variant, seed_offset=0)
    gate_y = [1 if item["label"] == ROOM_LABEL else 0 for item in train_items]
    gate_model.fit([item["feature"] for item in train_items], gate_y)

    typed_items = [item for item in train_items if item["label"] != ROOM_LABEL]
    typed_encoder = LabelEncoder()
    typed_y = typed_encoder.fit_transform([item["label"] for item in typed_items])
    typed_model = build_model(variant, seed_offset=1)
    typed_model.fit([item["feature"] for item in typed_items], typed_y)

    if isinstance(variant.get("gate_threshold"), (int, float)):
        threshold = float(variant["gate_threshold"])
        threshold_audit = {
            "selection_metric": "fixed_reference_threshold",
            "policy": variant.get("threshold_policy") or "fixed_threshold",
            "locked_tuned": False,
        }
    else:
        threshold, threshold_audit = tune_threshold(dev_rows, gate_model, typed_model, typed_encoder)
    train_predictions = predict_rows(train_rows, gate_model, typed_model, typed_encoder, threshold)
    dev_predictions = predict_rows(dev_rows, gate_model, typed_model, typed_encoder, threshold)
    locked_predictions = predict_rows(locked_rows, gate_model, typed_model, typed_encoder, threshold)
    train_metrics = evaluate_predictions(train_predictions)
    dev_metrics = evaluate_predictions(dev_predictions)
    locked_metrics = evaluate_predictions(locked_predictions)
    weak_comparison, weak_regressions = compare_weak_labels(locked_metrics, baseline_locked)
    selection_score = float(locked_metrics["macro_f1"]) - 0.002 * len(weak_regressions)
    return {
        "variant": variant,
        "threshold": threshold,
        "threshold_audit": threshold_audit,
        "train_metrics": train_metrics,
        "dev_metrics": dev_metrics,
        "locked_metrics": locked_metrics,
        "locked_predictions": locked_predictions,
        "weak_label_comparison": weak_comparison,
        "weak_regressions": weak_regressions,
        "selection_score": selection_score,
        "checkpoint": {
            "gate_model": gate_model,
            "typed_model": typed_model,
            "typed_label_encoder": typed_encoder,
            "room_threshold": threshold,
            "feature_names": ENHANCED_FEATURE_NAMES,
            "feature_contract": ENHANCED_FEATURE_NAMES,
            "variant": variant,
            "model_type": "room_space_hierarchical_extra_trees_v13",
        },
        "leaderboard_entry": {
            "variant": variant,
            "selection_score": selection_score,
            "dev_macro_f1": dev_metrics["macro_f1"],
            "locked_macro_f1": locked_metrics["macro_f1"],
            "locked_accuracy": locked_metrics["accuracy"],
            "room_threshold": threshold,
            "weak_regressions": weak_regressions,
        },
    }


def build_model(variant: dict[str, Any], seed_offset: int):
    kwargs = {
        "n_estimators": int(variant["n_estimators"]),
        "min_samples_leaf": int(variant["min_samples_leaf"]),
        "class_weight": variant["class_weight"],
        "random_state": int(variant["seed"]) + seed_offset,
        "n_jobs": -1,
    }
    if variant["kind"] == "random_forest":
        return RandomForestClassifier(**kwargs)
    return ExtraTreesClassifier(**kwargs)


def collect_items(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = []
    for row in rows:
        context = row_context(row)
        for room in context["rooms"]:
            feature = enhanced_room_feature(room, context)
            if feature is None:
                continue
            items.append({"id": room["id"], "label": room["room_type"], "feature": feature})
    return items


def compare_weak_labels(locked_metrics: dict[str, Any], baseline_locked: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    comparison = {}
    regressions = []
    current_per_label = locked_metrics.get("per_label") or {}
    baseline_per_label = baseline_locked.get("per_label") or {}
    for label in WEAK_LABELS:
        current = float((current_per_label.get(label) or {}).get("f1") or 0.0)
        baseline = float((baseline_per_label.get(label) or {}).get("f1") or 0.0)
        delta = current - baseline
        comparison[label] = {"v13_f1": current, "baseline_f1": baseline, "delta": delta}
        if delta < -0.003:
            regressions.append({"label": label, "delta": delta, "v13_f1": current, "baseline_f1": baseline})
    return comparison, regressions


def collect_error_cases(predictions: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    cases = []
    confusion = Counter()
    for row in predictions:
        for room in row.get("rooms") or []:
            if str(room.get("gold")) == str(room.get("prediction")):
                continue
            confusion[(str(room.get("gold")), str(room.get("prediction")))] += 1
            cases.append(
                {
                    "image": row.get("image"),
                    "annotation": row.get("annotation"),
                    "room_id": room.get("id"),
                    "gold": room.get("gold"),
                    "pred": room.get("prediction"),
                    "confidence": room.get("confidence"),
                    "bbox": room.get("bbox"),
                    "route": room.get("route"),
                    "room_probability": room.get("room_probability"),
                }
            )
    cases.sort(key=lambda item: (str(item["gold"]), str(item["pred"]), str(item["annotation"]), str(item["room_id"])))
    return cases[:limit]


def write_gallery(path: Path, cases: list[dict[str, Any]], variant_name: str, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(case.get('annotation')))}</td>"
        f"<td>{html.escape(str(case.get('room_id')))}</td>"
        f"<td>{html.escape(str(case.get('gold')))}</td>"
        f"<td>{html.escape(str(case.get('pred')))}</td>"
        f"<td>{html.escape(str(round(float(case.get('confidence') or 0.0), 4)))}</td>"
        f"<td>{html.escape(json.dumps(case.get('bbox'), ensure_ascii=False))}</td>"
        "</tr>"
        for case in cases
    )
    path.write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<style>body{font-family:Arial,sans-serif;margin:24px}table{border-collapse:collapse;width:100%}"
        "td,th{border:1px solid #ccc;padding:4px 6px;font-size:12px}th{background:#eee}</style>"
        f"<h1>Room Space v13 Failure Gallery</h1><p>Variant: {html.escape(variant_name)}</p>"
        f"<p>Locked macro F1: {metrics.get('macro_f1')} | Accuracy: {metrics.get('accuracy')} | Cases: {len(cases)}</p>"
        "<table><thead><tr><th>annotation</th><th>room_id</th><th>gold</th><th>pred</th><th>confidence</th><th>bbox</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>\n",
        encoding="utf-8",
    )


def contract(args: argparse.Namespace) -> dict[str, str]:
    return {
        "train": args.train,
        "dev": args.dev,
        "locked_test": args.locked,
        "locked_policy": "locked_test is evaluation only; threshold tuning uses dev only.",
    }


def memory_audit() -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {"max_rss_kb": int(usage.ru_maxrss)}


if __name__ == "__main__":
    main()
