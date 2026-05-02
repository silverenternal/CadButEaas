#!/usr/bin/env python3
"""Build a leave-one-source-out v3 matrix from available expert evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPERTS = ["WallOpening", "RoomSpace", "SymbolFixture", "TextDimension", "SheetLayout", "SceneGraph"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/vlm/loso_eval_matrix_v3.json")
    args = parser.parse_args()

    wall_report = load_first(
        [
            "reports/vlm/source_heldout_eval_batch_v1.json",
            "reports/vlm/generalization_benchmark_v1.json",
            "reports/vlm/paper_v2_generalization_followup_summary.json",
            "reports/vlm/paper_v2_two_stage_router_summary.json",
        ]
    )
    e2e_report = load_json(Path("reports/vlm/e2e_scene_graph_v1_eval.json"))
    symbol_report = load_first(["reports/vlm/symbol_fixture_crop_context_encoder_v3_eval.json", "reports/vlm/symbol_fixture_expert_v1_eval.json"])
    text_report = load_first(["reports/vlm/text_dimension_expert_v2_eval.json", "reports/vlm/text_dimension_expert_v1_eval.json"])
    room_report = load_first(["reports/vlm/room_space_v5_t046_review_adjusted_auto_accept.json", "reports/vlm/room_space_context_eval_v1.json"])

    matrix = {
        "WallOpening": wall_entry(wall_report),
        "RoomSpace": generic_entry("RoomSpace", room_report, ["cubicasa5k"], source_drop=None),
        "SymbolFixture": generic_entry("SymbolFixture", symbol_report, ["cubicasa5k"], source_drop=None),
        "TextDimension": generic_entry("TextDimension", text_report, ["cubicasa5k"], source_drop=None),
        "SheetLayout": {
            "status": "data_insufficient",
            "sources": [],
            "metrics": {},
            "source_drop": None,
            "reason": "No sheet layout train/eval artifact or non-CubiCasa layout labels are available yet.",
        },
        "SceneGraph": scene_graph_entry(e2e_report),
    }

    for expert, row in matrix.items():
        row.setdefault("expert", expert)
        row.setdefault("loso_protocol", "leave_one_source_out_v3")
        row.setdefault("source_drop", None)

    report = {
        "version": "loso_eval_matrix_v3",
        "created": "2026-05-01",
        "policy": "Each expert reports LOSO metrics when available; otherwise it reports data_insufficient with the missing source/label reason.",
        "experts": matrix,
        "summary": {
            "expert_count": len(EXPERTS),
            "metric_or_data_insufficient_count": sum(1 for row in matrix.values() if row.get("metrics") or row.get("status") == "data_insufficient"),
            "data_insufficient": [name for name, row in matrix.items() if row.get("status") == "data_insufficient"],
        },
        "done_when_checks": {
            "each_expert_has_metric_or_data_insufficient": all(matrix[name].get("metrics") or matrix[name].get("status") == "data_insufficient" for name in EXPERTS),
            "reports_source_drop": all("source_drop" in matrix[name] for name in EXPERTS),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def wall_entry(report: dict[str, Any]) -> dict[str, Any]:
    by_source = nested(report, "by_source_macro_f1") or nested(report, "wall_opening", "by_source_macro_f1") or {}
    floor = by_source.get("floorplancad") if isinstance(by_source, dict) else None
    cvc = by_source.get("cvc_fp") if isinstance(by_source, dict) else None
    drop = None
    if floor is not None and cvc is not None:
        drop = round(abs(float(cvc) - float(floor)), 6)
    metrics = {}
    if by_source:
        metrics["by_source_macro_f1"] = by_source
    if nested(report, "wall_opening"):
        metrics["snapshot"] = report["wall_opening"]
    return {
        "status": "metrics_available" if metrics else "data_insufficient",
        "sources": sorted(by_source.keys()) if isinstance(by_source, dict) else [],
        "metrics": metrics,
        "source_drop": drop,
        "reason": None if metrics else "No wall/opening LOSO report found.",
    }


def generic_entry(name: str, report: dict[str, Any], sources: list[str], source_drop: float | None) -> dict[str, Any]:
    if not report:
        return {
            "status": "data_insufficient",
            "sources": sources,
            "metrics": {},
            "source_drop": source_drop,
            "reason": f"{name} has no source-heldout artifact with multiple sources yet.",
        }
    metrics = extract_metrics(report)
    return {
        "status": "metrics_available" if metrics else "data_insufficient",
        "sources": sources,
        "metrics": metrics,
        "source_drop": source_drop,
        "reason": None if metrics else f"{name} artifact exists but does not expose LOSO metrics.",
    }


def scene_graph_entry(report: dict[str, Any]) -> dict[str, Any]:
    metrics = {}
    if report:
        metrics = {
            "node_f1": report.get("node_f1"),
            "relation_f1": report.get("relation_f1"),
            "invalid_graph_rate": report.get("invalid_graph_rate"),
            "by_source": report.get("by_source"),
        }
    return {
        "status": "metrics_available" if metrics else "data_insufficient",
        "sources": sorted((report.get("by_source") or {}).keys()) if report else [],
        "metrics": metrics,
        "source_drop": None,
        "reason": None if metrics else "No scene graph e2e report found.",
    }


def extract_metrics(report: dict[str, Any]) -> dict[str, Any]:
    keys = ["accuracy", "macro_f1", "relation_f1", "locked_macro_f1", "locked_relation_f1", "by_source", "splits"]
    return {key: report.get(key) for key in keys if key in report}


def nested(data: dict[str, Any], *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def load_first(paths: list[str]) -> dict[str, Any]:
    for path in paths:
        data = load_json(Path(path))
        if data:
            return data
    return {}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


if __name__ == "__main__":
    main()

