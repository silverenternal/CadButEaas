#!/usr/bin/env python3
"""Summarize learned CadStruct MoE expert baselines into one audit report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--room-context", default="checkpoints/cadstruct_moe_room_space_context_mlp_streamed/train_summary.json")
    parser.add_argument("--room-context-previous", default="checkpoints/cadstruct_moe_room_space_context_mlp/train_summary.json")
    parser.add_argument("--room-crop", default="checkpoints/cadstruct_moe_room_space_crop_mlp/train_summary.json")
    parser.add_argument("--room-random-forest", default="checkpoints/cadstruct_moe_room_space_context_random_forest/train_summary.json")
    parser.add_argument("--room-enhanced-random-forest", default="checkpoints/cadstruct_moe_room_space_context_enhanced_random_forest/train_summary.json")
    parser.add_argument("--room-enhanced-hist-gbdt", default="checkpoints/cadstruct_moe_room_space_context_enhanced_hist_gbdt/train_summary.json")
    parser.add_argument("--room-shape-random-forest", default="checkpoints/cadstruct_moe_room_space_context_shape_random_forest/train_summary.json")
    parser.add_argument("--symbol-crop", default="checkpoints/cadstruct_moe_symbol_fixture_crop_mlp/train_summary.json")
    parser.add_argument("--text-crop", default="checkpoints/cadstruct_moe_text_dimension_crop_mlp/train_summary.json")
    parser.add_argument("--room-predicted-upstream", default="reports/vlm/room_space_predicted_upstream_comparison.json")
    parser.add_argument("--output", default="reports/vlm/moe_learned_baseline_summary.json")
    args = parser.parse_args()

    room_context = load_json(Path(args.room_context))
    room_context_previous = load_json(Path(args.room_context_previous))
    room_crop = load_json(Path(args.room_crop))
    room_random_forest = load_json(Path(args.room_random_forest))
    room_enhanced_random_forest = load_json(Path(args.room_enhanced_random_forest))
    room_enhanced_hist_gbdt = load_json(Path(args.room_enhanced_hist_gbdt))
    room_shape_random_forest = load_json(Path(args.room_shape_random_forest))
    symbol_crop = load_json(Path(args.symbol_crop))
    text_crop = load_json(Path(args.text_crop))
    room_predicted_upstream = load_json(Path(args.room_predicted_upstream))

    summary = {
        "date": "2026-04-30",
        "purpose": "Single audit view for learned CadStruct MoE expert baselines.",
        "expert_results": {
            "room_space_context_mlp_streamed": expert_metrics(room_context, "rooms"),
            "room_space_crop_mlp": expert_metrics(room_crop, "rooms"),
            "room_space_context_random_forest": expert_metrics(room_random_forest, "rooms"),
            "room_space_context_enhanced_random_forest": expert_metrics(room_enhanced_random_forest, "rooms"),
            "room_space_context_enhanced_hist_gbdt": expert_metrics(room_enhanced_hist_gbdt, "rooms"),
            "room_space_context_shape_random_forest": expert_metrics(room_shape_random_forest, "rooms"),
            "symbol_fixture_crop_mlp": expert_metrics(symbol_crop, "symbols"),
            "text_dimension_crop_mlp": expert_metrics(text_crop, "text_candidates"),
        },
        "room_space_context_delta_vs_crop": metric_delta(room_context, room_crop),
        "room_space_random_forest_delta_vs_context_mlp": metric_delta(room_random_forest, room_context),
        "room_space_enhanced_random_forest_delta_vs_base_random_forest": metric_delta(room_enhanced_random_forest, room_random_forest),
        "room_space_shape_random_forest_delta_vs_enhanced_random_forest": metric_delta(room_shape_random_forest, room_enhanced_random_forest),
        "room_space_predicted_upstream_delta": room_predicted_upstream.get("splits", {}),
        "room_space_streaming_memory_delta": memory_delta(room_context, room_context_previous),
        "readiness_assessment": {
            "wall_opening": "paper-grade for the current structural primitive task, based on existing locked-test reports",
            "room_space": "improved by structure-aware context, but still gold-box and below paper-grade macro F1",
            "symbol_fixture": "usable learned baseline, but long-tail classes remain weak",
            "text_dimension": "high accuracy baseline, but note_text and OCR/content understanding remain incomplete",
            "integrated_moe": "engineering scaffold is in place; final paper claims should wait for CubiCasa-aligned boundary prediction, predicted room proposals, and cross-dataset tests",
        },
        "known_metric_caveats": [
            "Room mean IoU is 1.0 because current RoomSpace baselines reuse gold room boxes.",
            "Symbol host-link F1 is deterministic from gold boxes and is not a detector host-link claim.",
            "Room predicted-upstream symbol/text ablation is done; boundary semantics and room boxes are still gold.",
            "Macro F1, not only accuracy, should drive checkpoint selection because room and symbol classes are imbalanced.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def expert_metrics(summary: dict[str, Any], count_key: str) -> dict[str, Any]:
    dev = summary.get("splits", {}).get("dev", {})
    smoke = summary.get("splits", {}).get("smoke", {})
    memory = summary.get("memory_audit", {})
    return {
        "checkpoint": str(summary.get("model", "")),
        "dev_count": dev.get(count_key),
        "dev_accuracy": dev.get("accuracy"),
        "dev_macro_f1": dev.get("macro_f1"),
        "smoke_count": smoke.get(count_key),
        "smoke_accuracy": smoke.get("accuracy"),
        "smoke_macro_f1": smoke.get("macro_f1"),
        "peak_rss_kb": memory.get("max_rss_kb"),
        "cuda_peak_allocated_mb": memory.get("cuda_peak_allocated_mb"),
        "cuda_peak_reserved_mb": memory.get("cuda_peak_reserved_mb"),
    }


def metric_delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_dev = left.get("splits", {}).get("dev", {})
    right_dev = right.get("splits", {}).get("dev", {})
    return {
        "dev_accuracy": safe_delta(left_dev.get("accuracy"), right_dev.get("accuracy")),
        "dev_macro_f1": safe_delta(left_dev.get("macro_f1"), right_dev.get("macro_f1")),
    }


def memory_delta(current: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    current_rss = current.get("memory_audit", {}).get("max_rss_kb")
    previous_rss = previous.get("memory_audit", {}).get("max_rss_kb")
    return {
        "current_peak_rss_kb": current_rss,
        "previous_peak_rss_kb": previous_rss,
        "absolute_reduction_kb": safe_delta(previous_rss, current_rss),
        "relative_reduction": safe_ratio(safe_delta(previous_rss, current_rss), previous_rss),
    }


def safe_delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def safe_ratio(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
