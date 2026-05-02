#!/usr/bin/env python3
"""Evaluate a raster VLM sidecar against JSONL samples."""

from __future__ import annotations

import argparse
import base64
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image

from eval_metrics import (
    count_warnings,
    dimension_hit,
    geometry_consistency,
    relation_f1,
    safe_rate,
    semantic_exact_f1,
    semantic_hit,
)


def load_samples(path: Path, limit: int | None) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
            if limit is not None and len(samples) >= limit:
                break
    return samples


def build_request(sample: dict[str, Any]) -> dict[str, Any]:
    image_path = Path(sample["image_path"])
    with Image.open(image_path) as image:
        width, height = image.size
    expected = sample.get("expected_json", {})
    hints = sample.get("request_hints") or {}
    if hints:
        return {
            "schema_version": "raster-vlm-1.0",
            "image": {"width": width, "height": height, "color": "luma8"},
            "thumbnail_png_base64": base64.b64encode(image_path.read_bytes()).decode("ascii"),
            "polylines": hints.get("polylines", []),
            "primitive_graph": hints.get("primitive_graph", {"nodes": [], "edges": []}),
            "text_candidates": hints.get("text_candidates", []),
            "symbol_candidates": hints.get("symbol_candidates", []),
        }
    expected_dimensions = expected.get("dimension_candidates") or []
    expected_semantics = expected.get("semantic_candidates") or []

    text_candidates = []
    for candidate in expected_dimensions:
        text_candidates.append(
            {
                "content": str(candidate.get("raw_text", "")),
                "confidence": 0.9,
                "bbox": candidate.get("bbox", [0.0, 0.0, 0.0, 0.0]),
                "rotation": 0.0,
                "accepted": True,
            }
        )

    polylines = []
    for _candidate in expected_semantics:
        polylines.append([[30.0, 40.0], [290.0, 40.0]])

    return {
        "schema_version": "raster-vlm-1.0",
        "image": {"width": width, "height": height, "color": "luma8"},
        "thumbnail_png_base64": base64.b64encode(image_path.read_bytes()).decode("ascii"),
        "polylines": polylines,
        "text_candidates": text_candidates,
        "symbol_candidates": [],
    }


def post_json(url: str, payload: dict[str, Any], timeout: float) -> tuple[dict[str, Any] | None, str | None, float]:
    started = time.perf_counter()
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data, None, (time.perf_counter() - started) * 1000
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return None, str(exc), (time.perf_counter() - started) * 1000


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    samples = load_samples(Path(args.dataset), args.limit)
    rows = []
    for index, sample in enumerate(samples):
        payload = build_request(sample)
        response, error, latency_ms = post_json(args.url, payload, args.timeout)
        expected = sample.get("expected_json", {})
        row = {
            "index": index,
            "image_path": sample.get("image_path"),
            "ok": response is not None,
            "error": error,
            "latency_ms": round(latency_ms, 3),
            "dimension_hit": bool(response and dimension_hit(expected, response)),
            "semantic_hit": bool(response and semantic_hit(expected, response)),
            "semantic_exact_f1": semantic_exact_f1(expected, response) if response else 0.0,
            "relation_f1": relation_f1(expected, response) if response else 0.0,
            "geometry_consistency": geometry_consistency(sample, response) if response else 0.0,
            "dimension_count": len(response.get("dimension_candidates", [])) if response else 0,
            "semantic_count": len(response.get("semantic_candidates", [])) if response else 0,
            "warnings": response.get("warnings", []) if response else [],
        }
        rows.append(row)
        if args.verbose:
            print(json.dumps(row, ensure_ascii=False))

    latencies = [row["latency_ms"] for row in rows if row["ok"]]
    warning_counts = count_warnings(rows)
    semantic_counts = [row["semantic_count"] for row in rows if row["ok"]]
    summary = {
        "dataset": args.dataset,
        "url": args.url,
        "total": len(rows),
        "ok": sum(1 for row in rows if row["ok"]),
        "json_success_rate": safe_rate(sum(1 for row in rows if row["ok"]), len(rows)),
        "dimension_hit_rate": safe_rate(sum(1 for row in rows if row["dimension_hit"]), len(rows)),
        "semantic_hit_rate": safe_rate(sum(1 for row in rows if row["semantic_hit"]), len(rows)),
        "semantic_exact_f1_mean": round(statistics.mean([row["semantic_exact_f1"] for row in rows]), 4)
        if rows
        else 0.0,
        "relation_f1_mean": round(statistics.mean([row["relation_f1"] for row in rows]), 4) if rows else 0.0,
        "geometry_consistency_mean": round(statistics.mean([row["geometry_consistency"] for row in rows]), 4)
        if rows
        else 0.0,
        "empty_semantic_rate": safe_rate(sum(1 for row in rows if row["ok"] and row["semantic_count"] == 0), len(rows)),
        "semantic_count_mean": round(statistics.mean(semantic_counts), 3) if semantic_counts else 0.0,
        "partial_recovery_count": warning_counts.get("partial_json_recovered", 0),
        "warning_counts": warning_counts,
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 3) if latencies else None,
            "median": round(statistics.median(latencies), 3) if latencies else None,
            "max": round(max(latencies), 3) if latencies else None,
        },
        "rows": rows,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/raster_vlm/smoke.jsonl")
    parser.add_argument("--url", default="http://127.0.0.1:8765/analyze_raster")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    summary = evaluate(args)
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
