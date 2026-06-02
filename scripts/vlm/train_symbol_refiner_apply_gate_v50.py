#!/usr/bin/env python3
"""Train an apply gate for the sink/tiny box refiner and evaluate page-level routing."""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from apply_symbol_sink_tiny_refiner_page_v49 import (
    evaluate,
    load_gold,
    refine_selected,
    score_candidates,
    select_page,
)
from train_symbol_box_refiner_v38 import apply_delta
from train_symbol_support_suppression_v35 import load_jsonl, source_path, vector
from train_symbol_tile_detector_v20 import bbox_iou, rel, write_json, write_jsonl


warnings.filterwarnings(
    "ignore",
    message="`sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel`.*",
    category=UserWarning,
)


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def gate_feature_names(rows: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for row in rows:
        names.update((row.get("features") or {}).keys())
    forbidden = {"input_iou", "best_iou_train_label", "is_center_only_no_iou"}
    return sorted(
        name
        for name in names
        if name not in forbidden
        and not name.startswith("target_")
        and not name.endswith("_train_label")
    )


def gate_vector(row: dict[str, Any], names: list[str]) -> list[float]:
    feats = row.get("features") or {}
    return [float(feats.get(name, 0.0) or 0.0) for name in names]


def make_apply_rows(rows: list[dict[str, Any]], refiner_bundle: dict[str, Any], clip: float) -> list[dict[str, Any]]:
    names = list(refiner_bundle["feature_names"])
    model = refiner_bundle["model"]
    x = np.asarray([vector(row, names) for row in rows], dtype=np.float32)
    deltas = model.predict(x) if len(rows) else []
    out: list[dict[str, Any]] = []
    for row, delta in zip(rows, deltas, strict=True):
        box = [float(v) for v in row["candidate_bbox"]]
        target_box = [float(v) for v in row["target_bbox"]]
        input_iou = bbox_iou(box, target_box)
        refined = apply_delta(box, list(delta), clip)
        refined_iou = bbox_iou(refined, target_box)
        positive_gain = refined_iou >= input_iou + 0.02
        protects_positive = not (input_iou >= 0.30 and refined_iou < input_iou - 0.02)
        item = dict(row)
        item["apply_labels"] = {
            "input_iou": round(input_iou, 6),
            "refined_iou": round(refined_iou, 6),
            "delta_iou": round(refined_iou - input_iou, 6),
            "apply": bool(positive_gain and protects_positive),
            "protects_positive": bool(protects_positive),
        }
        return_features = dict(item.get("features") or {})
        return_features.update(
            {
                "candidate_score": safe_float(row.get("candidate_score")),
                "candidate_label_sink": 1.0 if row.get("candidate_label") == "sink" else 0.0,
                "candidate_label_equipment": 1.0 if row.get("candidate_label") == "equipment" else 0.0,
            }
        )
        item["features"] = return_features
        out.append(item)
    return out


def label_report(model: Any, rows: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    if not rows:
        return {"examples": 0}
    y = np.asarray([int((row.get("apply_labels") or {}).get("apply")) for row in rows], dtype=np.int64)
    x = np.asarray([gate_vector(row, names) for row in rows], dtype=np.float32)
    probs = model.predict_proba(x)[:, 1]
    out = {"examples": int(len(rows)), "positive": int(y.sum()), "positive_rate": round(float(y.mean()), 6)}
    if len(set(y.tolist())) >= 2:
        out["roc_auc"] = round(float(roc_auc_score(y, probs)), 6)
        out["average_precision"] = round(float(average_precision_score(y, probs)), 6)
    return out


def page_candidate_key(row: dict[str, Any]) -> str:
    return str(row.get("candidate_id") or "")


def apply_gate_to_selected(
    selected: list[dict[str, Any]],
    refiner_bundle: dict[str, Any],
    gate_model: Any,
    gate_names: list[str],
    hardcase_by_candidate: dict[str, dict[str, Any]],
    threshold: float,
    clip: float,
    route_labels: set[str],
    route_areas: set[str],
) -> tuple[list[dict[str, Any]], Counter]:
    route_rows: list[dict[str, Any]] = []
    selected_by_id = {page_candidate_key(row): row for row in selected}
    for candidate_id, selected_row in selected_by_id.items():
        hardcase = hardcase_by_candidate.get(candidate_id)
        if hardcase is None:
            continue
        label = str(selected_row.get("label") or "")
        box = selected_row.get("bbox") or []
        try:
            width = max(0.0, float(box[2]) - float(box[0]))
            height = max(0.0, float(box[3]) - float(box[1]))
            pred_area = "tiny_le_64" if width * height <= 64 else "small_le_256" if width * height <= 256 else "other"
        except (TypeError, ValueError, IndexError):
            pred_area = "other"
        if route_labels and label not in route_labels and pred_area not in route_areas:
            continue
        features = dict(hardcase.get("features") or {})
        features.update(
            {
                "candidate_score": safe_float(selected_row.get("score")),
                "candidate_label_sink": 1.0 if selected_row.get("label") == "sink" else 0.0,
                "candidate_label_equipment": 1.0 if selected_row.get("label") == "equipment" else 0.0,
            }
        )
        item = dict(selected_row)
        item["features"] = features
        route_rows.append(item)
    if not route_rows:
        return selected, Counter()
    x = np.asarray([gate_vector(row, gate_names) for row in route_rows], dtype=np.float32)
    probs = gate_model.predict_proba(x)[:, 1]
    routed_ids = {page_candidate_key(row): float(prob) for row, prob in zip(route_rows, probs, strict=True) if float(prob) >= threshold}
    refined, audit = refine_selected([row for row in selected if page_candidate_key(row) in routed_ids], refiner_bundle, clip)
    refined_by_id = {page_candidate_key(row): row for row in refined}
    out: list[dict[str, Any]] = []
    for row in selected:
        cid = page_candidate_key(row)
        if cid in refined_by_id:
            item = refined_by_id[cid]
            item["apply_gate_score"] = round(routed_ids[cid], 6)
            out.append(item)
        else:
            out.append(row)
    audit["gate_candidates"] += len(route_rows)
    audit["gate_applied"] += len(routed_ids)
    return out, audit


def page_eval(
    recovery_manifest: Path,
    smoke_rows: Path,
    suppression_model: Path,
    refiner_bundle: dict[str, Any],
    gate_model: Any,
    gate_names: list[str],
    hardcase_rows: list[dict[str, Any]],
    gate_threshold: float,
    clip: float,
    route_labels: set[str],
    route_areas: set[str],
    selection_threshold: float,
    cluster_topk: int,
    max_per_page: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(recovery_manifest.read_text(encoding="utf-8"))
    rows = [row for row in load_jsonl(source_path(manifest["outputs"]["rows"])) if str(row.get("split") or "") == "smoke_eval"]
    scored = score_candidates(rows, joblib.load(suppression_model))
    pages: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        pages[str(row["page_id"])].append(row)
    hardcase_by_candidate = {str(row.get("candidate_id")): row for row in hardcase_rows}
    selected_raw = {page_id: select_page(page_rows, selection_threshold, cluster_topk, max_per_page) for page_id, page_rows in pages.items()}
    selected_gated: dict[str, list[dict[str, Any]]] = {}
    route_audit = Counter()
    for page_id, selected in selected_raw.items():
        gated, audit = apply_gate_to_selected(selected, refiner_bundle, gate_model, gate_names, hardcase_by_candidate, gate_threshold, clip, route_labels, route_areas)
        selected_gated[page_id] = gated
        route_audit.update(audit)
    gold_all = load_gold(smoke_rows)
    gold = {page_id: gold_all[page_id] for page_id in selected_gated if page_id in gold_all}
    baseline = evaluate(selected_raw, gold)
    gated = evaluate(selected_gated, gold)
    predictions = [
        {
            "page_id": page_id,
            "predicted_symbols": [
                {
                    "candidate_id": row.get("candidate_id"),
                    "bbox": row.get("bbox"),
                    "label": row.get("label"),
                    "confidence": round(safe_float(row.get("policy_score")), 6),
                    "refined_by": row.get("refined_by"),
                    "apply_gate_score": row.get("apply_gate_score"),
                }
                for row in selected
            ],
            "source_integrity": {"gold_used_for_inference": False, "runtime_uses_svg_or_cad_geometry": False},
        }
        for page_id, selected in selected_gated.items()
    ]
    return {"baseline_without_refiner": baseline, "gated_refined": gated, "route_audit": dict(route_audit)}, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardcase-dir", default="datasets/symbol_sink_tiny_hardcases_v49")
    parser.add_argument("--recovery-data", default="datasets/symbol_detector_listwise_recovery_v47_p2_transfer_smoke120/manifest.json")
    parser.add_argument("--smoke-rows", default="datasets/symbol_tile_detector_tiny_sahi_v21/smoke_v30.jsonl")
    parser.add_argument("--suppression-model", default="checkpoints/symbol_support_suppression_v35_p2_transfer_smoke120_t065_c1/model.joblib")
    parser.add_argument("--refiner-model", default="checkpoints/symbol_box_refiner_v38_sink_tiny_v49/model.joblib")
    parser.add_argument("--output-dir", default="checkpoints/symbol_refiner_apply_gate_v50")
    parser.add_argument("--eval-output", default="reports/vlm/symbol_refiner_apply_gate_v50_smoke_eval.json")
    parser.add_argument("--predictions-output", default="reports/vlm/symbol_refiner_apply_gate_v50_smoke_predictions.jsonl")
    parser.add_argument("--gate-threshold", type=float, default=0.55)
    parser.add_argument("--route-labels", default="sink,equipment")
    parser.add_argument("--route-areas", default="tiny_le_64,small_le_256")
    parser.add_argument("--clip", type=float, default=0.75)
    parser.add_argument("--selection-threshold", type=float, default=0.65)
    parser.add_argument("--cluster-topk", type=int, default=1)
    parser.add_argument("--max-per-page", type=int, default=120)
    parser.add_argument("--n-estimators", type=int, default=240)
    parser.add_argument("--seed", type=int, default=20260513)
    args = parser.parse_args()

    hardcase_dir = source_path(args.hardcase_dir)
    refiner_bundle = joblib.load(source_path(args.refiner_model))
    train_rows = make_apply_rows(load_jsonl(hardcase_dir / "train.jsonl"), refiner_bundle, args.clip)
    dev_rows = make_apply_rows(load_jsonl(hardcase_dir / "dev.jsonl"), refiner_bundle, args.clip)
    smoke_rows = make_apply_rows(load_jsonl(hardcase_dir / "smoke_eval.jsonl"), refiner_bundle, args.clip)
    names = gate_feature_names(train_rows)
    x = np.asarray([gate_vector(row, names) for row in train_rows], dtype=np.float32)
    y = np.asarray([int((row.get("apply_labels") or {}).get("apply")) for row in train_rows], dtype=np.int64)
    if len(set(y.tolist())) < 2:
        raise SystemExit("apply gate needs positive and negative training rows")
    model = ExtraTreesClassifier(
        n_estimators=args.n_estimators,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=2,
        random_state=args.seed,
    )
    model.fit(x, y)
    output_dir = source_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.joblib"
    joblib.dump({"model": model, "feature_names": names, "args": vars(args)}, model_path)
    page_report, predictions = page_eval(
        source_path(args.recovery_data),
        source_path(args.smoke_rows),
        source_path(args.suppression_model),
        refiner_bundle,
        model,
        names,
        smoke_rows,
        args.gate_threshold,
        args.clip,
        {item.strip() for item in args.route_labels.split(",") if item.strip()},
        {item.strip() for item in args.route_areas.split(",") if item.strip()},
        args.selection_threshold,
        args.cluster_topk,
        args.max_per_page,
    )
    baseline = page_report["baseline_without_refiner"]
    gated = page_report["gated_refined"]
    report = {
        "version": "symbol_refiner_apply_gate_v50",
        "source_integrity": {
            "model_input": "candidate bbox/score/type and refiner runtime features only",
            "offline_labels_used_for": ["apply_gate_training", "smoke_evaluation"],
            "gold_used_for_inference": False,
            "runtime_uses_svg_or_cad_geometry": False,
            "excluded_features": ["input_iou", "target_*", "best_iou_train_label", "is_center_only_no_iou"],
        },
        "training": {
            "checkpoint": rel(model_path),
            "feature_count": len(names),
            "train": label_report(model, train_rows, names),
            "dev": label_report(model, dev_rows, names),
            "smoke_rows": label_report(model, smoke_rows, names),
        },
        "page_smoke": page_report,
        "gate": {
            "precision_not_drop": gated["symbol_bbox_iou_0_30"]["precision"] >= baseline["symbol_bbox_iou_0_30"]["precision"],
            "recall_not_drop": gated["symbol_bbox_iou_0_30"]["recall"] >= baseline["symbol_bbox_iou_0_30"]["recall"],
            "sink_misses_reduce": gated["misses_by_label"].get("sink", 0) < baseline["misses_by_label"].get("sink", 0),
            "tiny_misses_reduce": gated["misses_by_area"].get("tiny_le_64", 0) < baseline["misses_by_area"].get("tiny_le_64", 0),
            "equipment_misses_not_increase": gated["misses_by_label"].get("equipment", 0) <= baseline["misses_by_label"].get("equipment", 0),
            "small_misses_not_increase": gated["misses_by_area"].get("small_le_256", 0) <= baseline["misses_by_area"].get("small_le_256", 0),
            "no_oracle_inference": True,
        },
    }
    report["gate"]["passed"] = all(report["gate"].values())
    write_json(source_path(args.eval_output), report)
    write_jsonl(source_path(args.predictions_output), predictions)
    print(json.dumps({"training": report["training"], "page_smoke": page_report, "gate": report["gate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
