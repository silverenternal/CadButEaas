#!/usr/bin/env python3
"""Inventory high-metric historical assets for raster-only MoE reuse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def metric(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return cur if cur is not None else default


def scan_high_metrics(min_value: float) -> list[dict[str, Any]]:
    metric_tokens = (
        "precision",
        "recall",
        "f1",
        "accuracy",
        "macro_f1",
        "micro_f1",
        "auc",
        "average_precision",
    )
    rows: list[dict[str, Any]] = []
    paths = list((ROOT / "reports/vlm").glob("*.json")) + list((ROOT / "checkpoints").glob("**/*.json"))
    for path in paths:
        try:
            data = load_json(path)
        except Exception:
            continue

        def walk(obj: Any, key_path: str = "") -> None:
            if isinstance(obj, dict):
                for key, value in obj.items():
                    next_path = f"{key_path}.{key}" if key_path else key
                    if isinstance(value, bool):
                        continue
                    if isinstance(value, (int, float)):
                        lower = key.lower()
                        if any(token in lower for token in metric_tokens) and min_value <= float(value) <= 1.000001:
                            rows.append({
                                "value": round(float(value), 6),
                                "path": str(path.relative_to(ROOT)),
                                "metric_path": next_path,
                            })
                    elif isinstance(value, (dict, list)):
                        walk(value, next_path)
            elif isinstance(obj, list):
                for index, value in enumerate(obj[:1000]):
                    walk(value, f"{key_path}[{index}]")

        walk(data)
    rows.sort(key=lambda item: (item["value"], item["path"], item["metric_path"]), reverse=True)
    return rows


def known_assets() -> list[dict[str, Any]]:
    symbol_v8 = load_json(ROOT / "reports/vlm/symbol_visual_evidence_v8_eval.json")
    v8_runtime = load_json(ROOT / "reports/vlm/v8_runtime_reproducibility_audit.json")
    boundary = load_json(ROOT / "reports/vlm/boundary_label_arbitration_v1_eval.json")
    visual_gate = load_json(ROOT / "reports/vlm/symbol_visual_gate_v14_eval.json")
    zero_shot = load_json(ROOT / "reports/vlm/zero_shot_performance_audit.json")
    training_index = load_json(ROOT / "reports/vlm/training_runs_index.json")
    return [
        {
            "asset_id": "symbol_visual_evidence_v8",
            "paths": [
                "reports/vlm/symbol_visual_evidence_v8_eval.json",
                "reports/vlm/v8_runtime_reproducibility_audit.json",
            ],
            "best_metrics": {
                "locked_reject_precision": metric(symbol_v8, "locked_eval", "reject_precision"),
                "locked_reject_recall": metric(symbol_v8, "locked_eval", "reject_recall"),
                "locked_macro_f1": metric(
                    v8_runtime,
                    "adopted_component_guards",
                    "symbol_visual_evidence_v8",
                    "locked_eval",
                    "classification_report",
                    "macro avg",
                    "f1-score",
                ),
                "locked_accuracy": metric(
                    v8_runtime,
                    "adopted_component_guards",
                    "symbol_visual_evidence_v8",
                    "locked_eval",
                    "classification_report",
                    "accuracy",
                ),
            },
            "locked_evaluated": True,
            "raster_only_reuse": "yes_as_visual_evidence_gate",
            "direct_current_v18_score": "no",
            "reason": "It classifies keep vs empty/review visual evidence, not full symbol type or contains_symbol relation.",
            "recommended_reuse": "Plug into v18 symbol candidate stream before type classification to suppress empty/false symbol crops while preserving raster-only inference.",
        },
        {
            "asset_id": "boundary_label_arbitration_v1",
            "paths": ["reports/vlm/boundary_label_arbitration_v1_eval.json"],
            "best_metrics": {
                "locked_accuracy": metric(boundary, "locked_boundary_metrics", "accuracy"),
                "locked_macro_f1": metric(boundary, "locked_boundary_metrics", "macro_f1"),
            },
            "locked_evaluated": True,
            "raster_only_reuse": "maybe_after_input_audit",
            "direct_current_v18_score": "no",
            "reason": "Strong boundary label arbitration exists, but it was evaluated on a real-world boundary-label task and must be checked against v18 detector payloads.",
            "recommended_reuse": "Audit feature contract and, if raster-derived, expose as boundary expert route in v18.",
        },
        {
            "asset_id": "symbol_visual_gate_v14",
            "paths": ["reports/vlm/symbol_visual_gate_v14_eval.json"],
            "best_metrics": {
                "raw_recovered_symbol_precision": metric(visual_gate, "raw_recovered_symbol_f1", "precision"),
                "raw_recovered_symbol_recall": metric(visual_gate, "raw_recovered_symbol_f1", "recall"),
                "raw_recovered_symbol_f1": metric(visual_gate, "raw_recovered_symbol_f1", "f1"),
                "existing_plus_missing_symbol_f1": metric(visual_gate, "current_existing_plus_missing_symbol_f1", "f1"),
            },
            "locked_evaluated": False,
            "raster_only_reuse": "no_direct_reuse",
            "direct_current_v18_score": "no",
            "reason": "Report explicitly says it uses CubiCasa SVG/parser symbol candidates and raw labels to repair saved visual-chain errors.",
            "recommended_reuse": "Use as diagnostic upper-bound only, not runtime model input.",
        },
        {
            "asset_id": "graph_node_training_runs",
            "paths": ["reports/vlm/training_runs_index.json"],
            "best_metrics": {
                "train_summary_count": metric(training_index, "train_summary_count"),
                "runs": metric(training_index, "aggregate", "runs"),
            },
            "locked_evaluated": False,
            "raster_only_reuse": "candidate_backbone_after_dataset_audit",
            "direct_current_v18_score": "no",
            "reason": "Many 0.98-1.0 values are dev/per-class/smoke metrics on graph-node datasets, not v18 raster-to-graph locked metrics.",
            "recommended_reuse": "Mine best raster graph-node checkpoints as candidate feature/backbone assets, then re-evaluate under v18 manifest gates.",
        },
        {
            "asset_id": "zero_shot_or_trained_structure_breakdown",
            "paths": ["reports/vlm/zero_shot_performance_audit.json"],
            "best_metrics": {
                "primitive_node_weighted_ensemble_accuracy": metric(
                    zero_shot, "trained_structure_source_breakdown", "primitive_node_weighted_ensemble", "overall", "accuracy"
                ),
                "primitive_node_weighted_ensemble_macro_f1": metric(
                    zero_shot, "trained_structure_source_breakdown", "primitive_node_weighted_ensemble", "overall", "macro_f1"
                ),
            },
            "locked_evaluated": False,
            "raster_only_reuse": "maybe_for_boundary_opening",
            "direct_current_v18_score": "no",
            "reason": "Smoke/source-breakdown asset, useful for primitive hard_wall/door/window but not current symbol/type/relation scores.",
            "recommended_reuse": "Use for boundary/opening expert migration, not for symbol contains relation.",
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/vlm/high_metric_asset_audit_v18.json")
    parser.add_argument("--min-value", type=float, default=0.98)
    args = parser.parse_args()

    high_metrics = scan_high_metrics(args.min_value)
    assets = known_assets()
    audit = {
        "schema_version": "cadstruct_high_metric_asset_audit_v18",
        "purpose": "Separate historical high metrics from current raster-only MoE v18 locked scores.",
        "summary": {
            "high_metric_entries_found": len(high_metrics),
            "known_assets_reviewed": len(assets),
            "main_conclusion": (
                "The project does contain many >0.98 historical metrics, but most are component/dev/per-class/smoke "
                "or SVG/parser-assisted diagnostics. Only locked raster-compatible assets should be promoted into v18."
            ),
        },
        "known_assets": assets,
        "top_high_metric_entries": high_metrics[:250],
        "promotion_rules": [
            "Must consume raster pixels or raster-derived detector payloads at inference.",
            "Offline SVG/parser labels are allowed only for training, calibration, evaluation, and upper-bound analysis.",
            "Must be re-run through cadstruct_moe_smoke/locked manifests before counting as current v18 score.",
            "Per-class/dev/smoke 0.98+ metrics are reusable evidence, not production adoption.",
        ],
        "recommended_next": [
            "Promote symbol_visual_evidence_v8 as the first reusable asset for v18 symbol candidate filtering.",
            "Audit boundary_label_arbitration_v1 feature contract for v18 boundary expert reuse.",
            "Mine graph-node checkpoints for raster-compatible backbone/feature reuse, then re-evaluate under v18 locked manifests.",
        ],
    }
    write_json(ROOT / args.output, audit)


if __name__ == "__main__":
    main()
