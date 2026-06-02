#!/usr/bin/env python3
"""Build a runtime-safe raster-symbol contract stream for CadStruct MoE.

P229 intentionally bridges raster symbol proposals into the existing contract
expert stack. The output JSONL is sanitized for runtime use and excludes gold,
SVG/parser, and expected_json fields.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

import sys

sys.path.insert(0, str(ROOT / "scripts" / "vlm"))

from cadstruct_moe.schema import ExpertPrediction, RoutedCandidate  # noqa: E402
from freeze_symbol_p222_p221a_sink_tiny import metrics, score_rows  # noqa: E402
from fuse_symbol_p206g_with_p211_p212 import load_p206g  # noqa: E402


DEFAULT_INPUT = ROOT / "reports" / "vlm" / "symbol_p224a_column_frozen_overlay.jsonl"
DEFAULT_OUTPUT = ROOT / "reports" / "vlm" / "p229_raster_symbol_contract_predictions.jsonl"
DEFAULT_EVAL = ROOT / "reports" / "vlm" / "p229_raster_symbol_contract_eval.json"
LABELS = {"appliance", "bathtub", "column", "equipment", "generic_symbol", "shower", "sink", "stair", "table"}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(text + ("\n" if rows else ""), encoding="utf-8")


def row_identifier(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("row_id") or "unknown_row")


def image_size(row: dict[str, Any]) -> tuple[float, float]:
    value = row.get("image_size") or row.get("page_size") or {}
    if isinstance(value, dict):
        return float(value.get("width") or 1.0), float(value.get("height") or 1.0)
    if isinstance(value, list | tuple) and len(value) >= 2:
        return float(value[0] or 1.0), float(value[1] or 1.0)
    metadata = row.get("metadata") or {}
    return float(metadata.get("width") or 1.0), float(metadata.get("height") or 1.0)


def safe_label(candidate: dict[str, Any]) -> str:
    label = str(candidate.get("symbol_type") or candidate.get("label") or "generic_symbol")
    return label if label in LABELS else "generic_symbol"


def safe_score(candidate: dict[str, Any]) -> float:
    try:
        return float(candidate.get("confidence", candidate.get("score", 0.0)) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def safe_bbox(candidate: dict[str, Any]) -> list[float] | None:
    value = candidate.get("bbox")
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def to_routed_candidate(row_id: str, index: int, candidate: dict[str, Any], page_width: float, page_height: float) -> RoutedCandidate | None:
    bbox = safe_bbox(candidate)
    if bbox is None:
        return None
    label = safe_label(candidate)
    confidence = safe_score(candidate)
    candidate_id = str(candidate.get("id") or candidate.get("target_id") or f"{row_id}_p229_symbol_{index:05d}")
    payload = {
        "bbox": bbox,
        "symbol_type": label,
        "confidence": confidence,
        "score": confidence,
        "rotation": float(candidate.get("rotation") or 0.0),
        "hard_case_focus": float(candidate.get("hard_case_focus") or 0.0),
        "metadata": {"width": page_width, "height": page_height},
        "proposal_stage": "raster_symbol_overlay_to_contract",
    }
    return RoutedCandidate(
        candidate_id=candidate_id,
        expert="symbol_fixture",
        family="symbol",
        candidate_type=label,
        confidence=confidence,
        bbox=bbox,
        source="p229_raster_symbol_contract_adapter",
        payload=payload,
        route_trace={
            "matched_hint": label,
            "routing_confidence": confidence,
            "abstain": False,
            "router": "p229_direct_symbol_contract_route",
        },
    )


def prediction_to_metric_item(prediction: ExpertPrediction) -> dict[str, Any]:
    return {
        "id": prediction.candidate_id,
        "target_id": prediction.candidate_id,
        "label": prediction.label,
        "symbol_type": prediction.label,
        "bbox": prediction.bbox,
        "confidence": prediction.confidence,
        "score": prediction.confidence,
        "source": prediction.source,
    }


def passthrough_prediction(candidate: RoutedCandidate) -> ExpertPrediction:
    return ExpertPrediction(
        candidate_id=candidate.candidate_id,
        expert="symbol_fixture",
        family="symbol",
        label=candidate.candidate_type,
        confidence=candidate.confidence,
        bbox=candidate.bbox,
        geometry={"bbox": candidate.bbox or []},
        source="p229_raster_contract_passthrough_baseline",
        metadata={
            "candidate_type": candidate.candidate_type,
            "contract_version": "p229_symbol_contract_v0",
            "inference_mode": "passthrough_baseline_no_oracle",
        },
    )


def build_rows(input_path: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    rows, _baseline_preds_by_row, golds_by_row = load_p206g(input_path)
    row_ids = [row_identifier(row) for row in rows]
    row_contexts: list[dict[str, Any]] = []
    preds_by_row: dict[str, list[dict[str, Any]]] = {}
    totals = Counter()
    source_counts = Counter()
    label_counts = Counter()

    for row in rows:
        rid = row_identifier(row)
        page_width, page_height = image_size(row)
        routed = [
            item
            for index, candidate in enumerate(row.get("symbol_candidates") or [])
            if (item := to_routed_candidate(rid, index, candidate, page_width, page_height)) is not None
        ]
        totals["rows"] += 1
        totals["routed_candidates"] += len(routed)
        row_contexts.append({
            "row_id": rid,
            "page_width": page_width,
            "page_height": page_height,
            "routed": routed,
        })

    output_rows: list[dict[str, Any]] = []
    for context in row_contexts:
        rid = context["row_id"]
        routed = context["routed"]
        predictions = [passthrough_prediction(item) for item in routed]
        totals["expert_predictions"] += len(predictions)
        preds_by_row[rid] = [prediction_to_metric_item(item) for item in predictions]
        for item in predictions:
            label_counts[item.label] += 1
            source_counts[item.source] += 1
        output_rows.append({
            "row_id": rid,
            "source": "p229_raster_symbol_contract_adapter",
            "routed_candidates": [item.to_dict() for item in routed],
            "expert_predictions": [item.to_dict() for item in predictions],
            "adapter_metadata": {
                "page_width": context["page_width"],
                "page_height": context["page_height"],
                "expert_loaded": False,
                "contract_version": "p229_symbol_contract_v0",
                "inference_mode": "passthrough_baseline_no_oracle",
                "runtime_source_integrity": "raster_proposals_only_no_svg_no_expected_json_no_offline_labels",
            },
        })

    summary = {
        "input_rows": len(rows),
        "scored_rows": len(row_ids),
        "expert_loaded": False,
        "inference_mode": "passthrough_baseline_no_oracle",
        "totals": dict(totals),
        "prediction_sources": dict(source_counts),
        "label_counts": dict(label_counts),
    }
    return output_rows, preds_by_row, {"golds_by_row": golds_by_row, "row_ids": row_ids, "summary": summary}


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--eval-out", type=Path, default=DEFAULT_EVAL)
    args = parser.parse_args()

    output_rows, preds_by_row, context = build_rows(args.input)
    golds_by_row = context["golds_by_row"]
    row_ids = context["row_ids"]
    per_row = score_rows(preds_by_row, golds_by_row, row_ids)
    report = {
        "id": "p229_raster_symbol_contract_eval",
        "phase": "P229b_contract_inventory_and_symbol_adapter_v0",
        "input": str(args.input),
        "output": str(args.output),
        "summary": context["summary"],
        "overall_metrics_iou_0_30": metrics(per_row),
        "per_label_metrics_iou_0_30": per_label_metrics(preds_by_row, golds_by_row, row_ids),
        "claim_boundary": "Offline eval uses locked targets only for scoring. Runtime JSONL excludes target/gold/SVG/parser/expected_json fields.",
        "next_step": "Compare v13 contract-expert labels against passthrough raster labels; promote only if precision-safe on locked scoring.",
    }
    write_jsonl(args.output, output_rows)
    write_json(args.eval_out, report)
    print(json.dumps({"wrote": str(args.output), "eval": str(args.eval_out), "metrics": report["overall_metrics_iou_0_30"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
