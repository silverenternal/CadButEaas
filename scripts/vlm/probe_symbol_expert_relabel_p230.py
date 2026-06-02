#!/usr/bin/env python3
"""Probe P230 symbol expert relabeling over the P229 raster contract stream.

The v13 SymbolFixture checkpoint is large, so this script is intentionally
bounded: it can run on a row/candidate sample, records checkpoint load latency,
and emits a non-promotable diagnostic if the expert cannot complete in time.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

import sys

sys.path.insert(0, str(ROOT / "scripts" / "vlm"))

from build_raster_symbol_contract_adapter_p229 import prediction_to_metric_item, write_json, write_jsonl  # noqa: E402
from cadstruct_moe.experts.symbol_fixture import SymbolFixtureExpert  # noqa: E402
from cadstruct_moe.schema import ExpertPrediction, RoutedCandidate  # noqa: E402
from freeze_symbol_p222_p221a_sink_tiny import bootstrap, metrics, score_rows  # noqa: E402
from fuse_symbol_p206g_with_p211_p212 import load_p206g  # noqa: E402


DEFAULT_CONTRACT = ROOT / "reports" / "vlm" / "p229_raster_symbol_contract_predictions.jsonl"
DEFAULT_OVERLAY = ROOT / "reports" / "vlm" / "symbol_p224a_column_frozen_overlay.jsonl"
DEFAULT_OUTPUT = ROOT / "reports" / "vlm" / "p230_symbol_expert_relabel_predictions.jsonl"
DEFAULT_EVAL = ROOT / "reports" / "vlm" / "p230_symbol_expert_relabel_eval.json"


def load_contract_rows(path: Path, max_rows: int | None = None, max_candidates: int | None = None) -> list[dict[str, Any]]:
    rows = []
    seen_candidates = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            candidates = row.get("routed_candidates") or []
            if max_candidates is not None:
                remaining = max_candidates - seen_candidates
                if remaining <= 0:
                    break
                candidates = candidates[:remaining]
                row = dict(row)
                row["routed_candidates"] = candidates
            seen_candidates += len(candidates)
            rows.append(row)
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def to_candidate(item: dict[str, Any]) -> RoutedCandidate:
    return RoutedCandidate(
        candidate_id=str(item["candidate_id"]),
        expert=str(item.get("expert") or "symbol_fixture"),
        family=str(item.get("family") or "symbol"),
        candidate_type=str(item.get("candidate_type") or "generic_symbol"),
        confidence=float(item.get("confidence") or 0.0),
        bbox=item.get("bbox"),
        source=str(item.get("source") or "p229_raster_symbol_contract_adapter"),
        payload=dict(item.get("payload") or {}),
        route_trace=dict(item.get("route_trace") or {}),
    )


def per_label_metrics(preds_by_row: dict[str, list[dict[str, Any]]], golds_by_row: dict[str, dict[str, dict[str, Any]]], row_ids: list[str]) -> dict[str, Any]:
    labels = sorted({gold["label"] for rid in row_ids for gold in golds_by_row[rid].values()} | {pred["label"] for preds in preds_by_row.values() for pred in preds})
    out: dict[str, Any] = {}
    for label in labels:
        label_preds = defaultdict(list)
        label_golds: dict[str, dict[str, dict[str, Any]]] = {}
        for rid in row_ids:
            label_preds[rid] = [pred for pred in preds_by_row.get(rid, []) if pred.get("label") == label]
            label_golds[rid] = {gid: gold for gid, gold in golds_by_row[rid].items() if gold.get("label") == label}
        out[label] = metrics(score_rows(label_preds, label_golds, row_ids))
    return out


def baseline_preds_from_contract(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for row in rows:
        row_id = str(row["row_id"])
        preds = []
        for item in row.get("routed_candidates") or []:
            preds.append({
                "id": str(item["candidate_id"]),
                "target_id": str(item["candidate_id"]),
                "label": str(item.get("candidate_type") or "generic_symbol"),
                "symbol_type": str(item.get("candidate_type") or "generic_symbol"),
                "bbox": item.get("bbox"),
                "confidence": float(item.get("confidence") or 0.0),
                "score": float(item.get("confidence") or 0.0),
                "source": "p229_raster_contract_passthrough_baseline",
            })
        out[row_id] = preds
    return out


def prediction_rows(rows: list[dict[str, Any]], predictions_by_id: dict[str, ExpertPrediction], load_seconds: float, predict_seconds: float) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], Counter]:
    output = []
    preds_by_row: dict[str, list[dict[str, Any]]] = {}
    label_counts = Counter()
    for row in rows:
        row_id = str(row["row_id"])
        routed = row.get("routed_candidates") or []
        predictions = [predictions_by_id[str(item["candidate_id"])] for item in routed if str(item["candidate_id"]) in predictions_by_id]
        preds_by_row[row_id] = [prediction_to_metric_item(item) for item in predictions]
        for item in predictions:
            label_counts[item.label] += 1
        output.append({
            "row_id": row_id,
            "source": "p230_symbol_expert_relabel_probe",
            "expert_predictions": [item.to_dict() for item in predictions],
            "probe_metadata": {
                "contract_version": "p230_symbol_expert_relabel_probe_v0",
                "checkpoint_load_seconds": round(load_seconds, 3),
                "predict_seconds": round(predict_seconds, 3),
                "runtime_source_integrity": "raster_contract_only_no_svg_no_expected_json_no_offline_labels",
            },
        })
    return output, preds_by_row, label_counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--eval-out", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    args = parser.parse_args()

    rows = load_contract_rows(args.contract, args.max_rows, args.max_candidates)
    selected_ids = [str(row["row_id"]) for row in rows]
    _overlay_rows, _overlay_preds, all_golds = load_p206g(args.overlay)
    golds_by_row = {rid: all_golds[rid] for rid in selected_ids}
    baseline_preds = baseline_preds_from_contract(rows)
    baseline_per_row = score_rows(baseline_preds, golds_by_row, selected_ids)

    candidates = [to_candidate(item) for row in rows for item in row.get("routed_candidates") or []]
    load_start = time.time()
    expert = SymbolFixtureExpert()
    load_seconds = time.time() - load_start
    predict_start = time.time()
    predictions = expert.predict(candidates)
    predict_seconds = time.time() - predict_start
    predictions_by_id = {item.candidate_id: item for item in predictions}
    output_rows, preds_by_row, label_counts = prediction_rows(rows, predictions_by_id, load_seconds, predict_seconds)
    candidate_per_row = score_rows(preds_by_row, golds_by_row, selected_ids)

    baseline_metrics = metrics(baseline_per_row)
    candidate_metrics = metrics(candidate_per_row)
    report = {
        "id": "p230_symbol_expert_relabel_eval",
        "phase": "P230_registry_checkpoint_alignment_and_expert_relabel_probe",
        "contract_input": str(args.contract),
        "overlay_eval_source": str(args.overlay),
        "output": str(args.output),
        "sample": {"rows": len(rows), "candidates": len(candidates), "max_rows": args.max_rows, "max_candidates": args.max_candidates},
        "expert_loaded": expert.is_loaded(),
        "checkpoint_load_seconds": round(load_seconds, 3),
        "predict_seconds": round(predict_seconds, 3),
        "baseline_metrics_iou_0_30": baseline_metrics,
        "candidate_metrics_iou_0_30": candidate_metrics,
        "delta_vs_p229b": {key: round(candidate_metrics[key] - baseline_metrics[key], 6) for key in ["precision", "recall", "f1"]},
        "bootstrap_vs_p229b": bootstrap(baseline_per_row, candidate_per_row, iterations=1000, seed=230),
        "per_label_metrics_iou_0_30": per_label_metrics(preds_by_row, golds_by_row, selected_ids),
        "label_counts": dict(label_counts),
        "promotion_recommendation": "promote_only_if_full_locked_run_has_positive_f1_and_no_precision_regression",
        "claim_boundary": "Runtime predictions consume only the P229 raster-contract JSONL. Offline targets are used only for locked scoring.",
    }
    write_jsonl(args.output, output_rows)
    write_json(args.eval_out, report)
    print(json.dumps({"wrote": str(args.output), "eval": str(args.eval_out), "metrics": candidate_metrics, "delta": report["delta_vs_p229b"], "load_seconds": report["checkpoint_load_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
