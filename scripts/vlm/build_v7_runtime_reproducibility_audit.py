#!/usr/bin/env python3
"""Summarize v7 runtime and reproducibility evidence."""

from __future__ import annotations

import json

from v5_pipeline_utils import load_json, write_json


def main() -> None:
    boundary = load_json("reports/vlm/boundary_geometry_refiner_v7_eval.json", {})
    symbol = load_json("reports/vlm/symbol_fixture_expert_v13_eval.json", {})
    hard = load_json("reports/vlm/symbol_fixture_v13_hard_case_audit.json", {})
    report = {
        "version": "v7_runtime_reproducibility_audit",
        "boundary_geometry_refiner_v7": {
            "run_mode": boundary.get("run_mode"),
            "elapsed_seconds": boundary.get("elapsed_seconds"),
            "record_limits": boundary.get("record_limits"),
            "train_count": boundary.get("train_count"),
            "locked_count": boundary.get("locked_count"),
            "leakage_check": boundary.get("leakage_check"),
            "adopted": boundary.get("adopted"),
        },
        "symbol_fixture_expert_v13": {
            "elapsed_seconds": symbol.get("elapsed_seconds"),
            "train_count": symbol.get("train_count"),
            "dev_count": symbol.get("dev_count"),
            "locked_count": symbol.get("locked_count"),
            "leakage_check": symbol.get("leakage_check"),
            "adopted": symbol.get("adopted"),
            "adoption_checks": symbol.get("adoption_checks"),
        },
        "hard_case_dataset": {
            "split_counts": hard.get("split_counts"),
            "leakage_check": hard.get("leakage_check"),
        },
        "commands": [
            "uv run python scripts/vlm/train_boundary_geometry_refiner_v7.py",
            "uv run python scripts/vlm/build_symbol_fixture_hard_cases_v13.py",
            "uv run python scripts/vlm/train_symbol_fixture_expert_v13.py",
            "uv run python scripts/vlm/build_real_upstream_model_predictions_v7.py",
            "uv run python scripts/vlm/apply_visual_postprocess_v7.py",
            "uv run python scripts/vlm/build_visual_defect_ablation_v7.py",
        ],
        "claim_boundary": "Full train/dev and full locked results are separated from smoke. Rejected model candidates remain in reports but are not integrated.",
    }
    write_json("reports/vlm/v7_runtime_reproducibility_audit.json", report)
    print(json.dumps({"output": "reports/vlm/v7_runtime_reproducibility_audit.json"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
