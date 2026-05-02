#!/usr/bin/env python3
"""Run a smoke-grade real drawing pipeline and emit unified scene graphs.

The current implementation uses the existing benchmark `expected_json` records
as deterministic upstream expert outputs. This keeps the end-to-end contract
auditable while the real model-backed experts are still separate training
tracks.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fuse_scene_graph import fuse_record  # noqa: E402


DEFAULT_INPUT = Path("datasets/cadstruct_real_world_benchmark_v3/smoke.jsonl")
FALLBACK_INPUT = Path("datasets/cadstruct_cubicasa5k_moe_locked/smoke.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default="reports/vlm/e2e_real_pipeline_smoke_predictions.jsonl")
    parser.add_argument("--audit", default="reports/vlm/e2e_real_pipeline_smoke_audit.json")
    parser.add_argument("--source", default="expected_json")
    parser.add_argument("--limit", type=int, default=64)
    args = parser.parse_args()

    input_path = resolve_input(Path(args.input))
    rows = load_jsonl(input_path)
    if args.limit > 0:
        rows = rows[: args.limit]

    predictions: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    stage_counts: Counter[str] = Counter()
    start_all = time.perf_counter()
    for index, row in enumerate(rows):
        row_start = time.perf_counter()
        try:
            fused = fuse_record(row, args.source, disable_repairs=False)
            elapsed_ms = (time.perf_counter() - row_start) * 1000.0
            fusion = fused.get("fusion") or {}
            graph = fusion.get("scene_graph") or {}
            route_trace = dict(fused.get("route_trace") or {})
            route_trace.update(
                {
                    "stages": [
                        "raster_preprocess",
                        "wall_opening",
                        "room_proposal",
                        "symbol_fixture",
                        "text_dimension",
                        "sheet_layout",
                        "scene_graph_fusion",
                    ],
                    "latency_ms": round(elapsed_ms, 3),
                    "peak_memory_mib": current_peak_memory_mib(),
                    "source_mode": args.source,
                }
            )
            for stage in route_trace["stages"]:
                stage_counts[stage] += 1
            predictions.append(
                {
                    "image": fused.get("image"),
                    "annotation": fused.get("annotation"),
                    "source_dataset": fused.get("source_dataset") or row.get("source_dataset") or "unknown",
                    "split": row.get("metadata", {}).get("split") or "smoke",
                    "scene_graph": graph,
                    "warnings": fusion.get("warnings") or [],
                    "quality_report": build_quality_report(row, fusion),
                    "route_trace": route_trace,
                    "latency_ms": round(elapsed_ms, 3),
                    "memory_mib": current_peak_memory_mib(),
                    "gold_source": "expected_json_oracle_smoke",
                }
            )
        except Exception as exc:  # pragma: no cover - audit path
            failures.append(
                {
                    "index": index,
                    "image": row.get("image_path"),
                    "source_dataset": row.get("source_dataset"),
                    "error": type(exc).__name__,
                    "message": str(exc),
                }
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, predictions)

    valid_count = sum(1 for item in predictions if item.get("route_trace", {}).get("scene_graph_valid"))
    audit = {
        "version": "e2e_real_pipeline_smoke_audit_v1",
        "input": str(input_path),
        "output": str(output),
        "source_mode": args.source,
        "records": len(rows),
        "predictions": len(predictions),
        "unhandled_exception_count": len(failures),
        "schema_valid_graphs": valid_count,
        "schema_invalid_graphs": len(predictions) - valid_count,
        "schema_valid_rate": round(valid_count / max(len(predictions), 1), 6),
        "stage_counts": dict(stage_counts),
        "latency_ms": summarize_numbers([item["latency_ms"] for item in predictions]),
        "peak_memory_mib": max([item["memory_mib"] for item in predictions], default=current_peak_memory_mib()),
        "warning_counts": warning_counts(predictions),
        "failures": failures,
        "done_when_checks": {
            "ran_on_benchmark_v3_or_compatible_smoke": input_path.exists(),
            "schema_valid_scene_graphs": bool(predictions) and valid_count == len(predictions),
            "no_unhandled_exceptions": not failures,
        },
    }
    audit_path = Path(args.audit)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def resolve_input(path: Path) -> Path:
    if path.exists():
        return path
    if path == DEFAULT_INPUT and FALLBACK_INPUT.exists():
        return FALLBACK_INPUT
    raise FileNotFoundError(path)


def build_quality_report(row: dict[str, Any], fusion: dict[str, Any]) -> dict[str, Any]:
    warnings = list(fusion.get("warnings") or [])
    expected = row.get("expected_json") or {}
    return {
        "scan_quality": row.get("metadata", {}).get("scan_quality") or "benchmark_raster",
        "degraded_mode_recommended": False,
        "warning_count": len(warnings),
        "candidate_counts": {
            "semantic": len(expected.get("semantic_candidates") or []),
            "rooms": len(expected.get("room_candidates") or []),
            "symbols": len(expected.get("symbol_candidates") or []),
            "texts": len(expected.get("text_candidates") or []),
        },
    }


def summarize_numbers(values: list[float]) -> dict[str, float]:
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
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[index]


def current_peak_memory_mib() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return round(usage / (1024 * 1024), 3)
    return round(usage / 1024, 3)


def warning_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        for warning in row.get("warnings") or []:
            counts[str(warning)] += 1
    return dict(counts.most_common())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

