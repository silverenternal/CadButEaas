#!/usr/bin/env python3
"""Estimate CadStruct training memory risk before launching a run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/vlm/memory_budget_check_v2.json")
    args = parser.parse_args()

    checks = [
        check("14B LoRA structural", 45224, 32768, ["4-bit LoRA", "max_length=6144", "vision_tiles=8"]),
        check("crop-GNN h1024 multi-scale", 18000, 32768, ["tile_size<=2048", "CPU crop staging", "no all-crop CUDA cache"]),
        check("symbol crop MLP v3b", 1758, 32768, ["batch_size=4096", "image_cache_size<=12"]),
        check("scene graph all-pairs 8k nodes", 65536, 32768, ["cap edge candidates", "spatial index", "degraded mode"]),
    ]
    report = {
        "version": "memory_budget_check_v2",
        "hardware_profiles_mib": {"primary_96gb": 98304, "secondary_32gb": 32768},
        "checks": checks,
        "high_risk_configs": [item for item in checks if item["risk_32gb"] == "high"],
        "status": "ok",
    }
    write_json(Path(args.output), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def check(name: str, estimated_mib: int, budget_mib: int, mitigations: list[str]) -> dict[str, Any]:
    ratio = estimated_mib / budget_mib
    risk = "high" if ratio > 0.9 else "medium" if ratio > 0.5 else "low"
    return {
        "name": name,
        "estimated_peak_mib": estimated_mib,
        "budget_mib": budget_mib,
        "budget_ratio": round(ratio, 4),
        "risk_32gb": risk,
        "oom_guardrails": mitigations,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
