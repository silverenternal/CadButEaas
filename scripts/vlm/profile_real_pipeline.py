#!/usr/bin/env python3
"""Summarize real pipeline latency, memory, tile, skip, and degraded-mode stats."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="reports/vlm/e2e_real_pipeline_smoke_predictions.jsonl")
    parser.add_argument("--degraded-manifest", default="datasets/cadstruct_degraded_v1/manifest.json")
    parser.add_argument("--output", default="reports/vlm/real_pipeline_performance_v1.json")
    args = parser.parse_args()

    rows = load_jsonl(Path(args.predictions))
    degraded = load_json(Path(args.degraded_manifest))
    latency = [float(row.get("latency_ms") or row.get("route_trace", {}).get("latency_ms") or 0.0) for row in rows]
    memory = [float(row.get("memory_mib") or row.get("route_trace", {}).get("peak_memory_mib") or 0.0) for row in rows]
    degraded_records = degraded.get("records") or []
    degradation_counts = Counter(str(row.get("degradation_type") or "unknown") for row in degraded_records)
    warning_counts = Counter()
    stage_counts = Counter()
    skip_count = 0
    tile_count = 0
    degraded_mode_count = 0
    for row in rows:
        for warning in row.get("warnings") or []:
            warning_counts[str(warning)] += 1
        route = row.get("route_trace") or {}
        for stage in route.get("stages") or []:
            stage_counts[str(stage)] += 1
        quality = row.get("quality_report") or {}
        if quality.get("degraded_mode_recommended"):
            degraded_mode_count += 1
        if route.get("skipped") or row.get("skipped"):
            skip_count += 1
        if route.get("tile_count"):
            tile_count += int(route.get("tile_count") or 0)

    report = {
        "version": "real_pipeline_performance_v1",
        "predictions": args.predictions,
        "records": len(rows),
        "latency": summarize_numbers(latency),
        "peak_memory_mib": round(max(memory), 3) if memory else 0.0,
        "mean_memory_mib": round(sum(memory) / len(memory), 3) if memory else 0.0,
        "tile_stats": {
            "tile_count": tile_count,
            "records_with_tiles": sum(1 for row in rows if (row.get("route_trace") or {}).get("tile_count")),
        },
        "skip_stats": {"skip_count": skip_count, "skip_rate": round(skip_count / max(len(rows), 1), 6)},
        "degraded_mode_stats": {
            "degraded_mode_recommended_count": degraded_mode_count,
            "available_degraded_manifest_records": len(degraded_records),
            "by_degradation_type": dict(sorted(degradation_counts.items())),
        },
        "stage_counts": dict(stage_counts),
        "warning_counts": dict(warning_counts.most_common()),
        "done_when_checks": {
            "has_p50_p95_latency": bool(latency),
            "has_peak_memory_mib": bool(memory),
            "has_tile_skip_degraded_mode_stats": True,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def summarize_numbers(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0}
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": round(sum(values) / len(values), 3),
        "p50": round(percentile(ordered, 0.50), 3),
        "p95": round(percentile(ordered, 0.95), 3),
    }


def percentile(ordered: list[float], q: float) -> float:
    return ordered[min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

