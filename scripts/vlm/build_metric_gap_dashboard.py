#!/usr/bin/env python3
"""Build a single metric gap dashboard for CadStruct-MoE.

The dashboard intentionally reads existing reports only.  It is used as the
stable starting point for gap-closure work so smoke-only gains do not get mixed
with locked-test or end-to-end claims.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TARGETS = {
    "wall_opening.locked_macro_f1": 0.98,
    "wall_opening.locked_accuracy": 0.99,
    "wall_opening.locked_probability_r2": 0.98,
    "wall_opening.floorplancad_macro_f1": 0.98,
    "room_space.gold_strict_macro_f1": 0.98,
    "room_space.e2e_macro_f1_first_milestone": 0.90,
    "symbol_fixture.dev_macro_f1_first_milestone": 0.90,
    "text_dimension.dev_macro_f1_first_milestone": 0.90,
    "text_dimension.dimension_link_f1_first_milestone": 0.90,
    "scene_graph.node_macro_f1_first_milestone": 0.90,
    "scene_graph.relation_f1_first_milestone": 0.80,
    "generalization.leave_one_source_out_macro_f1": 0.95,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", default="reports/vlm")
    parser.add_argument("--output", default="reports/vlm/metric_gap_dashboard_v1.json")
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    matrix = load_json(reports_dir / "real_world_capability_matrix.json")
    generalization = load_json(reports_dir / "generalization_benchmark_v1.json")
    room_e2e = load_json(reports_dir / "room_space_predicted_upstream_comparison.json")
    room_adjusted = load_json(reports_dir / "room_space_v5_t046_ambiguity_adjusted.json")
    scene_graph = load_json(reports_dir / "moe" / "fused_scene_graph_smoke_audit.json")
    zero_shot = load_json(reports_dir / "zero_shot_performance_audit.json")

    dashboard: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "metric_gap_dashboard_v1",
        "inputs": {
            "real_world_capability_matrix": str(reports_dir / "real_world_capability_matrix.json"),
            "generalization_benchmark": str(reports_dir / "generalization_benchmark_v1.json"),
            "room_predicted_upstream": str(reports_dir / "room_space_predicted_upstream_comparison.json"),
            "room_ambiguity_adjusted": str(reports_dir / "room_space_v5_t046_ambiguity_adjusted.json"),
            "scene_graph_audit": str(reports_dir / "moe" / "fused_scene_graph_smoke_audit.json"),
            "zero_shot_audit": str(reports_dir / "zero_shot_performance_audit.json"),
        },
        "targets": TARGETS,
        "experts": {},
        "blocking_summary": [],
    }

    experts = dashboard["experts"]
    experts["wall_opening"] = wall_opening_section(matrix, generalization)
    experts["room_space"] = room_space_section(matrix, room_e2e, room_adjusted)
    experts["symbol_fixture"] = symbol_section(matrix)
    experts["text_dimension"] = text_section(matrix)
    experts["scene_graph"] = scene_graph_section(scene_graph)
    experts["zero_shot_vlm"] = zero_shot_section(zero_shot)
    experts["generalization"] = generalization_section(generalization, matrix)

    for name, section in experts.items():
        blockers = section.get("blockers") or []
        for blocker in blockers:
            dashboard["blocking_summary"].append({"expert": name, **blocker})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dashboard, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "blockers": len(dashboard["blocking_summary"])}, ensure_ascii=False))


def wall_opening_section(matrix: dict[str, Any], generalization: dict[str, Any]) -> dict[str, Any]:
    metrics = nested(matrix, "experts", "wall_opening", "metrics") or {}
    locked = metrics.get("strict_paper_locked") or {}
    current = nested(generalization, "current_evidence", "wall_opening_source_mixed_selected") or {}
    by_source = current.get("by_source_macro_f1") or {}
    values = {
        "locked_accuracy": locked.get("accuracy") or current.get("locked_accuracy"),
        "locked_macro_f1": locked.get("macro_f1") or current.get("locked_macro_f1"),
        "locked_probability_r2": locked.get("probability_r2") or current.get("locked_probability_r2"),
        "by_source_macro_f1": by_source,
    }
    blockers = []
    add_gap(blockers, "floorplancad_macro_f1", by_source.get("floorplancad"), TARGETS["wall_opening.floorplancad_macro_f1"], "FloorPlanCAD source remains below 98%.")
    add_gap(blockers, "locked_probability_r2", values["locked_probability_r2"], TARGETS["wall_opening.locked_probability_r2"], "Overall probability R2 is below target.")
    return {"status": status_from_blockers(blockers), "current": values, "target": "98%+ locked and source-wise", "blockers": blockers, "next_phase": "P4"}


def room_space_section(matrix: dict[str, Any], room_e2e: dict[str, Any], room_adjusted: dict[str, Any]) -> dict[str, Any]:
    strict = nested(matrix, "experts", "room_space", "metrics", "strict") or {}
    adjusted = nested(matrix, "experts", "room_space", "metrics", "ambiguity_adjusted") or {}
    if room_adjusted:
        strict = room_adjusted.get("strict") or strict
        adjusted = room_adjusted.get("ambiguity_adjusted") or adjusted
    dev_e2e = nested(room_e2e, "splits", "dev") or {}
    smoke_e2e = nested(room_e2e, "splits", "smoke") or {}
    current = {
        "gold_strict_macro_f1": strict.get("macro_f1"),
        "gold_strict_accuracy": strict.get("accuracy"),
        "ambiguity_adjusted_macro_f1": adjusted.get("macro_f1"),
        "predicted_upstream_dev_macro_f1": dev_e2e.get("predicted_upstream_macro_f1"),
        "predicted_upstream_smoke_macro_f1": smoke_e2e.get("predicted_upstream_macro_f1"),
    }
    blockers = []
    add_gap(blockers, "predicted_upstream_dev_macro_f1", current["predicted_upstream_dev_macro_f1"], TARGETS["room_space.e2e_macro_f1_first_milestone"], "Gold polygon classifier is strong, but end-to-end upstream path is not.")
    return {"status": status_from_blockers(blockers), "current": current, "target": "end-to-end room macro F1 >= 0.90 first, then 0.98", "blockers": blockers, "next_phase": "P1"}


def symbol_section(matrix: dict[str, Any]) -> dict[str, Any]:
    dev = nested(matrix, "experts", "symbol_fixture", "metrics", "dev") or {}
    smoke = nested(matrix, "experts", "symbol_fixture", "metrics", "smoke") or {}
    current = {"dev_accuracy": dev.get("accuracy"), "dev_macro_f1": dev.get("macro_f1"), "smoke_macro_f1": smoke.get("macro_f1")}
    blockers = []
    add_gap(blockers, "dev_macro_f1", current["dev_macro_f1"], TARGETS["symbol_fixture.dev_macro_f1_first_milestone"], "Symbol/fixture expert is far below the first 90% milestone.")
    return {"status": status_from_blockers(blockers), "current": current, "target": "macro F1 >= 0.90 first, then 0.95/0.98", "blockers": blockers, "next_phase": "P2"}


def text_section(matrix: dict[str, Any]) -> dict[str, Any]:
    dev = nested(matrix, "experts", "text_dimension", "metrics", "dev") or {}
    smoke = nested(matrix, "experts", "text_dimension", "metrics", "smoke") or {}
    current = {
        "dev_accuracy": dev.get("accuracy"),
        "dev_macro_f1": dev.get("macro_f1"),
        "dev_dimension_link_f1": dev.get("dimension_link_f1"),
        "smoke_macro_f1": smoke.get("macro_f1"),
        "smoke_dimension_link_f1": smoke.get("dimension_link_f1"),
    }
    blockers = []
    add_gap(blockers, "dev_macro_f1", current["dev_macro_f1"], TARGETS["text_dimension.dev_macro_f1_first_milestone"], "Text type classification is below the first 90% milestone.")
    add_gap(blockers, "dev_dimension_link_f1", current["dev_dimension_link_f1"], TARGETS["text_dimension.dimension_link_f1_first_milestone"], "Dimension relation linking is below the first 90% milestone.")
    return {"status": status_from_blockers(blockers), "current": current, "target": "text macro F1 and relation F1 >= 0.90 first", "blockers": blockers, "next_phase": "P3"}


def scene_graph_section(scene_graph: dict[str, Any]) -> dict[str, Any]:
    warnings = scene_graph.get("warnings") or scene_graph.get("warning_counts") or {}
    current = {
        "records": scene_graph.get("records"),
        "total_nodes": scene_graph.get("total_nodes"),
        "total_edges": scene_graph.get("total_edges"),
        "nodes_per_record": scene_graph.get("nodes_per_record"),
        "edges_per_record": scene_graph.get("edges_per_record"),
        "status": scene_graph.get("status"),
        "warning_counts": warnings,
        "node_macro_f1": scene_graph.get("node_macro_f1"),
        "relation_f1": scene_graph.get("relation_f1"),
        "invalid_graph_rate": scene_graph.get("invalid_graph_rate"),
    }
    blockers = [
        {
            "metric": "node_macro_f1",
            "current": current["node_macro_f1"],
            "target": TARGETS["scene_graph.node_macro_f1_first_milestone"],
            "gap": None,
            "reason": "Scene graph audit currently reports graph volume/warnings but not node/relation F1.",
        }
    ]
    if warning_value(warnings, "room_label_without_link") > 0:
        blockers.append(
            {
                "metric": "room_label_without_link",
                "current": warning_value(warnings, "room_label_without_link"),
                "target": 0,
                "gap": warning_value(warnings, "room_label_without_link"),
                "reason": "Room labels are generated without reliable labeled_by links.",
            }
        )
    return {"status": "blocked", "current": current, "target": "node/relation F1 with invalid graph rate", "blockers": blockers, "next_phase": "P5"}


def zero_shot_section(zero_shot: dict[str, Any]) -> dict[str, Any]:
    best = zero_shot.get("best_base_vlm") or zero_shot.get("best") or {}
    metrics = best.get("metrics") or best
    current = {
        "samples": metrics.get("total") or metrics.get("samples"),
        "semantic_exact_f1_mean": metrics.get("semantic_exact_f1_mean"),
        "relation_f1_mean": metrics.get("relation_f1_mean"),
        "geometry_consistency_mean": metrics.get("geometry_consistency_mean"),
        "json_success": metrics.get("json_success"),
    }
    return {
        "status": "baseline_only",
        "current": current,
        "target": ">=30 samples per model; useful as teacher/baseline, not main path",
        "blockers": [
            {
                "metric": "samples",
                "current": current["samples"],
                "target": 30,
                "gap": None if current["samples"] is None else max(0, 30 - current["samples"]),
                "reason": "Existing zero-shot result is smoke-scale only.",
            }
        ],
        "next_phase": "P7",
    }


def generalization_section(generalization: dict[str, Any], matrix: dict[str, Any]) -> dict[str, Any]:
    loso = nested(generalization, "current_evidence", "wall_opening_leave_one_source_out") or nested(matrix, "experts", "wall_opening", "metrics", "cross_source_locked") or {}
    cvc_to_floor = nested(loso, "cvc_fp_train_floorplancad_test", "macro_f1")
    floor_to_cvc = nested(loso, "floorplancad_train_cvc_fp_test", "macro_f1")
    current = {"cvc_fp_train_floorplancad_test_macro_f1": cvc_to_floor, "floorplancad_train_cvc_fp_test_macro_f1": floor_to_cvc}
    blockers = []
    add_gap(blockers, "cvc_fp_train_floorplancad_test_macro_f1", cvc_to_floor, TARGETS["generalization.leave_one_source_out_macro_f1"], "Pure cross-source transfer fails for CVC-FP -> FloorPlanCAD.")
    add_gap(blockers, "floorplancad_train_cvc_fp_test_macro_f1", floor_to_cvc, TARGETS["generalization.leave_one_source_out_macro_f1"], "Pure cross-source transfer fails for FloorPlanCAD -> CVC-FP.")
    return {"status": status_from_blockers(blockers), "current": current, "target": "source-heldout macro F1 >= 0.95 first", "blockers": blockers, "next_phase": "P6"}


def add_gap(blockers: list[dict[str, Any]], metric: str, current: Any, target: float, reason: str) -> None:
    value = safe_float(current)
    if value is None or value < target:
        blockers.append({"metric": metric, "current": current, "target": target, "gap": None if value is None else round(target - value, 6), "reason": reason})


def status_from_blockers(blockers: list[dict[str, Any]]) -> str:
    return "ready" if not blockers else "blocked"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def nested(data: dict[str, Any], *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def warning_value(warnings: Any, key: str) -> int:
    if isinstance(warnings, dict):
        value = warnings.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    if isinstance(warnings, list):
        return sum(1 for item in warnings if isinstance(item, str) and item == key)
    return 0


if __name__ == "__main__":
    main()
