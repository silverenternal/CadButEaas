#!/usr/bin/env python3
"""Build runtime/reproducibility audit for CadStruct v8."""

from __future__ import annotations

import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path

from v8_raster_e2e_utils import ROOT, load_json, update_todo_remove, write_json


COMMANDS = [
    "uv run python scripts/vlm/audit_raster_e2e_assets_v8.py",
    "uv run python scripts/vlm/build_raster_detection_dataset_v8.py",
    "uv run python scripts/vlm/train_raster_candidate_detector_v8.py",
    "uv run python scripts/vlm/build_symbol_visual_evidence_dataset_v8.py",
    "uv run python scripts/vlm/train_symbol_visual_evidence_v8.py",
    "uv run python scripts/vlm/build_raster_e2e_predictions_v8.py",
    "uv run python scripts/vlm/build_hybrid_visual_model_v8.py",
    "uv run python scripts/vlm/render_scene_graph_visual_demo.py --predictions reports/vlm/hybrid_visual_model_v8_predictions.jsonl --output-dir reports/vlm/visual_demo_hybrid_v8_model --source-dataset cubicasa5k --limit 5",
    "uv run python scripts/vlm/build_visual_model_v8_comparison.py",
    "uv run python scripts/vlm/audit_raster_e2e_defects_v8.py",
]


def main() -> None:
    deps = {name: bool(importlib.util.find_spec(name)) for name in ["torch", "sklearn", "PIL", "numpy", "joblib"]}
    detector = load_json("reports/vlm/raster_candidate_detector_v8_eval.json", {})
    symbol = load_json("reports/vlm/symbol_visual_evidence_v8_eval.json", {})
    report = {
        "version": "v8_runtime_reproducibility_audit",
        "created": "2026-05-07",
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": deps,
        "commands": COMMANDS,
        "adopted_component_guards": {
            "raster_candidate_detector_v8": {"adopted": detector.get("adopted"), "run_mode": (detector.get("train_summary") or {}).get("run_mode"), "macro_f1": detector.get("macro_f1")},
            "symbol_visual_evidence_v8": {"adopted": symbol.get("adopted"), "locked_eval": symbol.get("locked_eval")},
        },
        "full_locked_claim": {
            "raster_candidate_detector_v8": "full locked evaluated but rejected",
            "symbol_visual_evidence_v8": "locked evaluated; train/dev sampled for runtime and reported in dataset audit",
        },
        "claim_boundary": "Adopted components must come from locked evaluation. raster_e2e remains rejected.",
    }
    write_json("reports/vlm/v8_runtime_reproducibility_audit.json", report)
    write_runbook(report)
    update_todo_remove(["RASTER-V8-T9"])
    print(json.dumps({"output": "reports/vlm/v8_runtime_reproducibility_audit.json"}, ensure_ascii=False, indent=2))


def write_runbook(report: dict) -> None:
    lines = [
        "# CadStruct v8 Raster E2E Runbook",
        "",
        "This runbook reproduces the v8 source-mode separation work.",
        "",
        "## Commands",
        "",
    ]
    lines.extend(f"```bash\n{cmd}\n```\n" for cmd in report["commands"])
    lines.extend(
        [
            "## Claim Boundary",
            "",
            "Pure raster E2E is not adopted because `raster_candidate_detector_v8` failed locked metrics.",
            "Hybrid v8 uses SVG/parser candidate geometry plus adopted raster crop visual-evidence review flags.",
            "Postprocess cleanup remains separate from model recognition credit.",
            "",
        ]
    )
    path = ROOT / "docs/cadstruct-v8-raster-e2e-runbook.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
