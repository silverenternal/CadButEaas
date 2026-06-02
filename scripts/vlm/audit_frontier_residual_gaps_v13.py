#!/usr/bin/env python3
"""Audit residual frontier gaps for the v13 specialist recovery plan."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from v5_pipeline_utils import load_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundary-hard-cases", default="datasets/cadstruct_hard_cases_v3/boundary_expert_v3_hard_cases/manifest.jsonl")
    parser.add_argument("--room-train", default="datasets/cadstruct_rooms_v1/train.jsonl")
    parser.add_argument("--room-dev", default="datasets/cadstruct_rooms_v1/dev.jsonl")
    parser.add_argument("--room-locked", default="datasets/cadstruct_rooms_v1/smoke.jsonl")
    parser.add_argument("--symbol-hard-cases", default="datasets/symbol_fixture_expert_v13_hard_cases/train.jsonl")
    parser.add_argument("--text-train", default="datasets/text_dimension_expert_v4_full_ocr_augmented/train.jsonl")
    parser.add_argument("--text-dev", default="datasets/text_dimension_expert_v4_full_ocr_augmented/dev.jsonl")
    parser.add_argument("--text-locked", default="datasets/text_dimension_expert_v4_full_ocr_augmented/locked_test.jsonl")
    parser.add_argument("--floorplancad-locked", default="datasets/cadstruct_real_world_benchmark_v1/wall_opening/floorplancad_locked_test.jsonl")
    parser.add_argument("--output", default="reports/vlm/frontier_residual_gap_audit_v13.json")
    parser.add_argument("--target-gate", default="reports/vlm/frontier_target_metric_gate_v13.json")
    args = parser.parse_args()

    report = {
        "version": "frontier_residual_gap_audit_v13",
        "source_audits": {
            "boundary": audit_boundary(load_jsonl(args.boundary_hard_cases)),
            "room_space": audit_rooms([*load_jsonl(args.room_train), *load_jsonl(args.room_dev), *load_jsonl(args.room_locked)]),
            "symbol_fixture": audit_symbol(load_jsonl(args.symbol_hard_cases)),
            "text_dimension": audit_text([*load_jsonl(args.text_train), *load_jsonl(args.text_dev), *load_jsonl(args.text_locked)]),
            "floorplancad": audit_floorplancad(load_jsonl(args.floorplancad_locked)),
        },
    }
    report["family_decision"] = {
        "boundary": "model_and_postprocess",
        "room_space": "model_first",
        "symbol_fixture": "model_first_with_open_vocab_retrieval",
        "text_dimension": "model_first_with_ocr_gate",
        "floorplancad": "model_and_threshold_gate",
    }
    report["target_gate"] = {
        "locked_metric_order": ["boundary", "room_space", "symbol_fixture", "text_dimension", "floorplancad"],
        "retrain_priority": rank_priorities(report["source_audits"]),
        "claim_boundary": "Only families with source-held-out trainable signals move into v13 retraining; locked sets remain eval-only.",
    }
    write_json(args.output, report)
    write_json(args.target_gate, report["target_gate"])
    print(json.dumps({"output": args.output, "retrain_priority": report["target_gate"]["retrain_priority"]}, ensure_ascii=False, indent=2))


def audit_boundary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row.get("defect_type") or row.get("review_reason") or "unknown") for row in rows)
    by_semantic = Counter(str(row.get("semantic_type") or row.get("base_raw_label") or "unknown") for row in rows)
    return {
        "rows": len(rows),
        "defect_counts": dict(counts.most_common()),
        "semantic_counts": dict(by_semantic.most_common()),
        "trainable_now": len(rows) > 0,
        "priority_score": len(rows) + counts.get("boundary_drift", 0) * 2 + counts.get("false_wall", 0) * 2,
        "recommended_action": "train boundary geometry refiner with line-aware hard cases and keep postprocess cleanup separate",
    }


def audit_rooms(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rooms = [room for row in rows for room in row.get("rooms") or []]
    labels = Counter(str(room.get("room_type") or "room") for room in rooms)
    with_labels = sum(1 for room in rooms if str(room.get("room_type") or "") != "room")
    return {
        "rows": len(rows),
        "rooms": len(rooms),
        "label_counts": dict(labels.most_common()),
        "labeled_rooms": with_labels,
        "trainable_now": len(rooms) > 0,
        "priority_score": len(rooms) + with_labels * 2,
        "recommended_action": "pretrain room-aware polygon completion and room-label linkage before any MoE fusion refresh",
    }


def audit_symbol(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(str(row.get("label") or row.get("gold_label") or row.get("raw_label") or "unknown") for row in rows)
    hard_focus = sum(1 for row in rows if str(row.get("label") or row.get("gold_label") or "").lower() in {"appliance", "equipment"})
    return {
        "rows": len(rows),
        "label_counts": dict(labels.most_common()),
        "hard_focus_rows": hard_focus,
        "trainable_now": len(rows) > 0,
        "priority_score": len(rows) + hard_focus * 3,
        "recommended_action": "train symbol long-tail recovery with exemplar retrieval and hard negatives",
    }


def audit_text(rows: list[dict[str, Any]]) -> dict[str, Any]:
    text_candidates = [item for row in rows for item in row.get("text_candidates") or []]
    label_counts = Counter(str(item.get("text_type") or "note_text") for item in text_candidates)
    numeric_count = sum(1 for item in text_candidates if has_digit(str(item.get("raw_text") or item.get("text") or "")))
    dimension_links = sum(len(row.get("dimension_links") or []) for row in rows)
    return {
        "rows": len(rows),
        "text_candidates": len(text_candidates),
        "label_counts": dict(label_counts.most_common()),
        "numeric_text_candidates": numeric_count,
        "dimension_links": dimension_links,
        "trainable_now": len(text_candidates) > 0,
        "priority_score": len(text_candidates) + numeric_count * 2 + dimension_links,
        "recommended_action": "separate OCR, numeric text, and dimension linking with layout-aware training",
    }


def audit_floorplancad(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(str(node.get("label") or "unknown") for row in rows for node in row.get("nodes") or [])
    high_aspect = sum(1 for row in rows for node in row.get("nodes") or [] if float(node.get("features", {}).get("length") or 0) > 80 and str(node.get("label") or "") == "door")
    return {
        "rows": len(rows),
        "node_counts": dict(labels.most_common()),
        "high_aspect_door_nodes": high_aspect,
        "trainable_now": len(rows) > 0,
        "priority_score": len(rows) + high_aspect * 2,
        "recommended_action": "apply threshold and few-shot adaptation on source-shifted wall/opening nodes",
    }


def rank_priorities(audits: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"family": family, "score": int(info.get("priority_score") or 0), "trainable_now": bool(info.get("trainable_now"))}
        for family, info in sorted(audits.items(), key=lambda item: float(item[1].get("priority_score") or 0), reverse=True)
    ]


def has_digit(text: str) -> bool:
    return any(ch.isdigit() for ch in text)


if __name__ == "__main__":
    main()
