#!/usr/bin/env python3
"""Build the v18 image-only MoE diagnostic dashboard.

This report freezes the raster-only evaluation view used by the v18 recovery
plan. It consumes model-credit prediction streams plus offline gold only for
audit/evaluation, and it does not create inference candidates from gold labels.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from image_only_moe_v17_pipeline import (  # noqa: E402
    CONTRACT,
    FAMILY_TO_CLASS,
    REPORT,
    _family_f1,
    _gold_structured,
    _load_v16_rows,
    _normalize_box,
    _proposal_family,
    load_json,
    load_jsonl,
)
from validate_image_only_moe_stream import validate_rows  # noqa: E402

TASK_ID = "IMG-MOE-V18-P0-001"
FAMILIES = ("boundary", "space", "symbol", "text", "sheet")

FAMILY_IOU = {
    "boundary": 0.20,
    "space": 0.25,
    "symbol": 0.25,
    "text": 0.25,
    "sheet": 0.25,
}

FIRST_MILESTONE_RECALL = {
    "boundary": 0.70,
    "space": 0.80,
    "symbol": 0.60,
    "text": 0.60,
    "sheet": 0.50,
}

GOLD_KEYS = {
    "boundary": "edges",
    "space": "rooms",
    "symbol": "symbols",
    "text": "texts",
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(text + ("\n" if rows else ""), encoding="utf-8")


def round_metric(value: Any, digits: int = 6) -> Any:
    if value is None:
        return None
    try:
        if math.isnan(value):
            return None
    except TypeError:
        pass
    if isinstance(value, float):
        return round(value, digits)
    return value


def metric_view(metric: dict[str, Any] | None) -> dict[str, Any]:
    metric = metric or {}
    keys = [
        "matched",
        "labeled_tp",
        "precision",
        "recall",
        "f1",
        "label_precision",
        "label_recall",
        "label_f1",
        "predicted",
        "gold",
    ]
    return {key: round_metric(metric.get(key)) for key in keys if key in metric}


def gold_rows_for_family(source_rows: dict[str, dict[str, Any]], row_id: str, family: str) -> list[dict[str, Any]]:
    gold = _gold_structured(source_rows.get(row_id, {}))
    if family == "boundary":
        cls_set = FAMILY_TO_CLASS["boundary"]
        return [
            {
                "bbox": item.get("bbox"),
                "semantic_type": item.get("class") or item.get("semantic_type"),
                "class": item.get("class") or item.get("semantic_type"),
            }
            for item in gold.get("edges") or []
            if str(item.get("class") or item.get("semantic_type")) in cls_set
        ]
    if family == "space":
        return [
            {"bbox": item.get("bbox"), "semantic_type": item.get("semantic_type") or item.get("class"), "class": "room"}
            for item in gold.get("rooms") or []
        ]
    if family == "symbol":
        return [
            {"bbox": item.get("bbox"), "semantic_type": item.get("semantic_type") or item.get("class"), "class": "symbol"}
            for item in gold.get("symbols") or []
        ]
    if family == "text":
        return [
            {"bbox": item.get("bbox"), "semantic_type": item.get("semantic_type") or item.get("class"), "class": "text"}
            for item in gold.get("texts") or []
        ]
    return []


def candidate_rows_by_family(candidate_rows: list[dict[str, Any]], family: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in candidate_rows:
        for candidate in row.get("candidate_stream") or []:
            if _proposal_family(candidate) == family:
                out.append(candidate)
    return out


def prediction_rows_by_family(prediction_rows: list[dict[str, Any]], family: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in prediction_rows:
        for prediction in row.get("predictions") or []:
            if str(prediction.get("family")) == family:
                out.append(prediction)
    return out


def all_gold_by_family(source_rows: dict[str, dict[str, Any]], row_ids: list[str], family: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row_id in row_ids:
        out.extend(gold_rows_for_family(source_rows, row_id, family))
    return out


def family_limit(candidate_metric: dict[str, Any], expert_metric: dict[str, Any], warning_count: int, family: str) -> str:
    if int(candidate_metric.get("gold") or 0) == 0 and int(candidate_metric.get("predicted") or 0) == 0:
        return "no_locked_gold_non_core"
    recall = float(candidate_metric.get("recall") or 0.0)
    label_f1 = float(expert_metric.get("label_f1") or 0.0)
    candidate_f1 = float(candidate_metric.get("f1") or 0.0)
    target_recall = FIRST_MILESTONE_RECALL.get(family, 0.60)

    if recall < target_recall:
        return "candidate_limited"
    if label_f1 < max(0.20, candidate_f1 * 0.50):
        return "classifier_limited"
    if warning_count and family in {"boundary", "space", "text"}:
        return "fusion_limited"
    return "not_primary_bottleneck"


def warning_summary(prediction_rows: list[dict[str, Any]]) -> tuple[Counter[str], dict[str, int]]:
    counter: Counter[str] = Counter()
    by_family: Counter[str] = Counter()
    for row in prediction_rows:
        for warning in row.get("fusion_warnings") or []:
            warning_type = str(warning).split(":", 1)[0]
            counter[warning_type] += 1
            if warning_type.startswith("room_"):
                by_family["space"] += 1
            elif warning_type.startswith("opening_"):
                by_family["boundary"] += 1
            elif warning_type.startswith("dimension_"):
                by_family["text"] += 1
            else:
                by_family["unknown"] += 1
    return counter, dict(by_family)


def detector_source_breakdown(
    candidate_rows: list[dict[str, Any]],
    source_rows: dict[str, dict[str, Any]],
    row_ids: list[str],
) -> dict[str, Any]:
    gold_by_family = {family: all_gold_by_family(source_rows, row_ids, family) for family in FAMILIES}
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in candidate_rows:
        for candidate in row.get("candidate_stream") or []:
            family = _proposal_family(candidate)
            source = str(
                candidate.get("proposal_source")
                or (candidate.get("payload") or {}).get("proposal_source")
                or candidate.get("source")
                or "unknown"
            )
            grouped[family][source].append(candidate)

    report: dict[str, Any] = {}
    for family, by_source in grouped.items():
        family_report = {}
        for source, preds in sorted(by_source.items()):
            metric = _family_f1(preds, gold_by_family.get(family, []), FAMILY_IOU.get(family, 0.25))
            family_report[source] = {
                "candidate_count": len(preds),
                "recall_at_iou": round_metric(metric.get("recall")),
                "precision_at_iou": round_metric(metric.get("precision")),
                "f1_at_iou": round_metric(metric.get("f1")),
                "ap": None,
                "ap_note": "AP requires scored threshold sweep; v18 P0-001 freezes the recall/F1 proxy from the existing stream.",
            }
        report[family] = family_report
    return report


def per_page_rows(
    prediction_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    source_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates_by_id = {str(row.get("id")): row for row in candidate_rows}
    rows: list[dict[str, Any]] = []
    for row in prediction_rows:
        row_id = str(row.get("id"))
        gold = _gold_structured(source_rows.get(row_id, {}))
        scene_graph = row.get("scene_graph") or {}
        candidate_stream = candidates_by_id.get(row_id, {}).get("candidate_stream") or []
        family_candidate_counts = Counter(_proposal_family(candidate) for candidate in candidate_stream)
        family_prediction_counts = Counter(str(pred.get("family") or "unknown") for pred in row.get("predictions") or [])
        warnings = [str(item) for item in row.get("fusion_warnings") or []]
        warning_types = Counter(item.split(":", 1)[0] for item in warnings)
        rows.append(
            {
                "id": row_id,
                "image": row.get("image"),
                "source_dataset": row.get("source_dataset"),
                "gold_counts": {
                    "boundary": len(gold.get("edges") or []),
                    "space": len(gold.get("rooms") or []),
                    "symbol": len(gold.get("symbols") or []),
                    "text": len(gold.get("texts") or []),
                },
                "candidate_counts": dict(sorted(family_candidate_counts.items())),
                "prediction_counts": dict(sorted(family_prediction_counts.items())),
                "graph_counts": {
                    "nodes": len(scene_graph.get("nodes") or []),
                    "edges": len(scene_graph.get("edges") or []),
                },
                "fusion_warning_count": len(warnings),
                "fusion_warning_types": dict(sorted(warning_types.items())),
            }
        )
    return rows


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    predictions_path = ROOT / args.predictions
    candidates_path = ROOT / args.candidates
    expert_predictions_path = ROOT / args.expert_predictions

    prediction_rows = load_jsonl(predictions_path)
    candidate_rows = load_jsonl(candidates_path)
    expert_rows = load_jsonl(expert_predictions_path)
    source_rows = {str(row["id"]): row for row in _load_v16_rows()}
    row_ids = [str(row.get("id")) for row in prediction_rows]

    contract = load_json(CONTRACT)
    source_integrity_gate = validate_rows(prediction_rows, contract)
    v17_eval = load_json(REPORT / "image_only_moe_v17_eval.json", {})
    warning_counts, warning_counts_by_family = warning_summary(prediction_rows)

    family_breakdown: dict[str, Any] = {}
    for family in FAMILIES:
        candidate_metric = metric_view((v17_eval.get("candidate_metrics") or {}).get(family))
        expert_metric = metric_view((v17_eval.get("expert_metrics") or {}).get(family))
        if not candidate_metric:
            candidate_metric = metric_view(
                _family_f1(
                    candidate_rows_by_family(candidate_rows, family),
                    all_gold_by_family(source_rows, row_ids, family),
                    FAMILY_IOU.get(family, 0.25),
                )
            )
        if family != "sheet" and not expert_metric:
            expert_metric = metric_view(
                _family_f1(
                    prediction_rows_by_family(prediction_rows, family),
                    all_gold_by_family(source_rows, row_ids, family),
                    FAMILY_IOU.get(family, 0.25),
                )
            )

        family_breakdown[family] = {
            "candidate_metric": candidate_metric,
            "expert_metric": expert_metric,
            "warning_count": warning_counts_by_family.get(family, 0),
            "diagnosis": family_limit(candidate_metric, expert_metric, warning_counts_by_family.get(family, 0), family),
            "first_milestone_recall": FIRST_MILESTONE_RECALL.get(family),
        }

    page_rows = per_page_rows(prediction_rows, candidate_rows, source_rows)
    expert_counter = Counter(row.get("family") for row in expert_rows)
    source_counter = Counter(
        str(
            candidate.get("proposal_source")
            or (candidate.get("payload") or {}).get("proposal_source")
            or candidate.get("source")
            or "unknown"
        )
        for row in candidate_rows
        for candidate in row.get("candidate_stream") or []
    )

    total_nodes = sum(len((row.get("scene_graph") or {}).get("nodes") or []) for row in prediction_rows)
    total_edges = sum(len((row.get("scene_graph") or {}).get("edges") or []) for row in prediction_rows)
    gold_counts = Counter()
    for row_id in row_ids:
        gold = _gold_structured(source_rows.get(row_id, {}))
        for family, key in GOLD_KEYS.items():
            gold_counts[family] += len(gold.get(key) or [])

    report = {
        "task": TASK_ID,
        "protocol_version": "image_only_moe_v18_diagnostics_v1",
        "created": "2026-05-08",
        "input_streams": {
            "predictions": args.predictions,
            "candidates": args.candidates,
            "expert_predictions": args.expert_predictions,
            "contract": str(CONTRACT.relative_to(ROOT)),
            "offline_gold_source": "reports/vlm/image_only_structured_moe_proposals_v16.jsonl",
        },
        "source_integrity_gate": source_integrity_gate,
        "source_integrity_policy": {
            "model_credit_input": "raster_image_only",
            "offline_gold_use": "audit_and_locked_evaluation_only",
            "oracle_credit_allowed": False,
        },
        "summary": {
            "rows": len(prediction_rows),
            "candidate_rows": len(candidate_rows),
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "gold_counts": dict(sorted(gold_counts.items())),
            "candidate_mean_f1": v17_eval.get("candidate_mean_f1"),
            "final_mean_f1": v17_eval.get("final_mean_f1"),
            "primary_dashboard_diagnosis": "candidate_recall_frontend_limited",
        },
        "family_breakdown": family_breakdown,
        "detector_source_breakdown": detector_source_breakdown(candidate_rows, source_rows, row_ids),
        "expert_usage": {
            "prediction_counts_by_family": dict(sorted(expert_counter.items())),
            "nonzero_core_experts": {
                family: expert_counter.get(family, 0) > 0
                for family in ("boundary", "space", "symbol", "text")
            },
        },
        "fusion_diagnostics": {
            "warning_counts": dict(sorted(warning_counts.items())),
            "warning_counts_by_family": dict(sorted(warning_counts_by_family.items())),
            "relation_f1": None,
            "relation_metric_status": "blocked_without_locked_relation_gold",
            "edge_count_proxy": {
                "predicted_edges": total_edges,
                "gold_relation_edges": None,
                "note": "v17 structured source rows expose gold objects and boundary primitives, but not a locked relation-gold graph for relation F1.",
            },
        },
        "page_count_artifact": args.per_page_output,
        "adoption_policy": {
            "minimum_required_before_adoption": [
                "source_integrity_gate.passed == true",
                "final_mean_f1 beats v15 baseline",
                "room and text family metrics are nonzero",
                "visual pack confirms no systematic coordinate offset",
            ],
            "adopted": False,
        },
        "next_unblocked_tasks": [
            "IMG-MOE-V18-P0-002",
            "IMG-MOE-V18-P0-003",
            "IMG-MOE-V18-P0-004",
            "IMG-MOE-V18-P0-006",
        ],
    }

    write_json(ROOT / args.output, report)
    write_jsonl(ROOT / args.per_page_output, page_rows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="reports/vlm/image_only_moe_predictions_v17.jsonl")
    parser.add_argument("--candidates", default="reports/vlm/image_only_moe_candidates_with_crops_v17.jsonl")
    parser.add_argument("--expert-predictions", default="reports/vlm/image_only_moe_expert_predictions_v17.jsonl")
    parser.add_argument("--output", default="reports/vlm/image_only_moe_v18_diagnostic_dashboard.json")
    parser.add_argument("--per-page-output", default="reports/vlm/image_only_moe_v18_page_diagnostics.jsonl")
    args = parser.parse_args()
    report = build_report(args)
    print(json.dumps({
        "task": report["task"],
        "output": args.output,
        "per_page_output": args.per_page_output,
        "source_integrity_passed": report["source_integrity_gate"]["passed"],
        "primary_dashboard_diagnosis": report["summary"]["primary_dashboard_diagnosis"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
