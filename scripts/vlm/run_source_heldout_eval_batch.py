#!/usr/bin/env python3
"""Collect source-heldout evaluation evidence across CadStruct experts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generalization", default="reports/vlm/generalization_benchmark_v1.json")
    parser.add_argument("--registry", default="reports/vlm/locked_benchmark_source_registry_v1.json")
    parser.add_argument("--output", default="reports/vlm/source_heldout_eval_batch_v1.json")
    args = parser.parse_args()

    generalization = load_json(Path(args.generalization))
    registry = load_json(Path(args.registry))
    evidence = generalization.get("current_evidence") or {}
    loso = evidence.get("wall_opening_leave_one_source_out") or {}
    sources = registry.get("sources") or {}

    report = {
        "version": "source_heldout_eval_batch_v1",
        "inputs": {
            "generalization": args.generalization,
            "registry": args.registry,
        },
        "experts": {
            "WallOpening": {
                "status": "evaluated",
                "protocol": "leave-one-source-out",
                "sources": ["cvc_fp", "floorplancad"],
                "results": {
                    "cvc_fp_train_floorplancad_test": loso.get("cvc_fp_train_floorplancad_test"),
                    "floorplancad_train_cvc_fp_test": loso.get("floorplancad_train_cvc_fp_test"),
                },
                "claim": "zero-shot source transfer is not supported by current metrics",
            },
            "RoomSpace": single_source_entry("cubicasa5k", sources, "room_space", "needs non-CubiCasa locked room source"),
            "SymbolFixture": single_source_entry("cubicasa5k", sources, "symbol_fixture", "needs FloorPlanCAD/internal symbol fixture locked labels"),
            "TextDimension": single_source_entry("cubicasa5k", sources, "text_dimension", "needs FloorPlanCAD/internal OCR/text locked labels"),
        },
    }
    report["summary"] = {
        "covered_experts": sorted(report["experts"]),
        "evaluated_experts": sorted(name for name, item in report["experts"].items() if item["status"] == "evaluated"),
        "blocked_experts": sorted(name for name, item in report["experts"].items() if item["status"] != "evaluated"),
        "status": "partial_coverage_with_explicit_blocks",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def single_source_entry(source: str, sources: dict[str, Any], family: str, blocker: str) -> dict[str, Any]:
    info = sources.get(source) or {}
    family_count = (info.get("element_families") or {}).get(family, 0)
    return {
        "status": "blocked_single_locked_source",
        "protocol": "source-heldout requested but not statistically valid with one locked source",
        "sources": [source],
        "locked_files": [
            item for item in info.get("locked_files") or []
            if family in item or (family == "room_space" and "room_space" in item)
        ],
        "element_family_count": family_count,
        "blocker": blocker,
        "claim": "do not claim source-heldout generalization for this expert yet",
    }


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
