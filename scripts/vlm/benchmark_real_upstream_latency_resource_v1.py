#!/usr/bin/env python3
"""Benchmark reproducible real-upstream replay/fusion stages."""

from __future__ import annotations

import json
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT / "scripts" / "vlm") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "vlm"))

from audit_relation_gold_id_repair_sensitivity_v1 import build_nodes  # noqa: E402
from audit_relation_no_repair_rule_sweep_v1 import edges_for_rule  # noqa: E402
from fuse_real_upstream import evaluate_nodes, evaluate_relations, extract_gold, load_jsonl  # noqa: E402

PREDICTIONS = ROOT / "reports" / "vlm" / "real_upstream_predictions_dev_symbol_label_arbitrated_v1.jsonl"
DEV_SPLIT = ROOT / "datasets" / "cadstruct_real_world_benchmark_v1" / "room_space" / "cubicasa5k_reviewed_locked_test.jsonl"
OUTPUT = ROOT / "reports" / "vlm" / "real_upstream_latency_resource_v1.json"


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def timed(name: str, fn: Callable[[], Any]) -> tuple[str, float, Any]:
    start = time.perf_counter()
    out = fn()
    elapsed = (time.perf_counter() - start) * 1000.0
    return name, elapsed, out


def summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    p95_index = min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95)))
    return {
        "mean_ms": round(statistics.mean(ordered), 3),
        "p50_ms": round(statistics.median(ordered), 3),
        "p95_ms": round(ordered[p95_index], 3),
    }


def main() -> int:
    stages: dict[str, list[float]] = {"load_inputs": [], "build_nodes": [], "no_repair_fusion_v2": [], "evaluate": []}
    last_counts: dict[str, Any] = {}
    selected_rule = {"name": "iou_center_hybrid_pad_0_overlap_0.5", "rule": "iou_center_hybrid", "padding": 0.0, "min_symbol_overlap": 0.5}
    for _ in range(5):
        name, ms, loaded = timed("load_inputs", lambda: (load_jsonl(PREDICTIONS), load_jsonl(DEV_SPLIT)))
        stages[name].append(ms)
        predictions, records = loaded

        name, ms, node_records = timed("build_nodes", lambda: build_nodes(predictions, records))
        stages[name].append(ms)

        def fuse() -> list[dict[str, Any]]:
            return [edge for nodes in node_records for edge in edges_for_rule(nodes, selected_rule)]

        name, ms, edges = timed("no_repair_fusion_v2", fuse)
        stages[name].append(ms)

        def eval_all() -> dict[str, Any]:
            gold_nodes, gold_edges = extract_gold(records)
            nodes = [node for nodes in node_records for node in nodes]
            return {
                "node_evaluation": evaluate_nodes(nodes, gold_nodes),
                "relation_evaluation": evaluate_relations(edges, gold_edges),
                "gold_nodes": len(gold_nodes),
                "gold_edges": len(gold_edges),
                "fused_nodes": len(nodes),
                "fused_edges": len(edges),
            }

        name, ms, metrics = timed("evaluate", eval_all)
        stages[name].append(ms)
        last_counts = {
            "records": len(records),
            "predictions": len(predictions),
            **{k: metrics[k] for k in ["gold_nodes", "gold_edges", "fused_nodes", "fused_edges"]},
            "node_macro_f1": metrics["node_evaluation"]["macro_f1"],
            "relation_f1": metrics["relation_evaluation"]["f1"],
        }

    total_by_run = [sum(stages[name][i] for name in stages) for i in range(5)]
    report = {
        "version": "real_upstream_latency_resource_v1",
        "created": "2026-05-03",
        "benchmark_type": "local_replay_from_saved_real_upstream_predictions",
        "includes": ["JSONL input load", "gold-compatible node reconstruction", "symbol label-arbitrated no-repair v2 fusion", "node/relation evaluation"],
        "excludes": ["OCR backend runtime", "VLM teacher calls", "expert model inference time before saved prediction stream"],
        "inputs": {
            "predictions": str(PREDICTIONS.relative_to(ROOT)),
            "dev_split": str(DEV_SPLIT.relative_to(ROOT)),
        },
        "counts": last_counts,
        "stage_timings": {name: summarize(values) for name, values in stages.items()},
        "total_replay": summarize(total_by_run),
        "peak_rss_mb": round(rss_mb(), 3),
        "paper_table_latency": {
            "value_ms_p50": summarize(total_by_run)["p50_ms"],
            "label": "replay/fusion p50, excludes OCR/VLM/expert inference",
            "legacy_latency_policy": "Do not compare directly with historical 12.1ms legacy smoke latency.",
        },
        "status": "passed",
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT}")
    print(json.dumps({"p50_ms": report["total_replay"]["p50_ms"], "p95_ms": report["total_replay"]["p95_ms"], "peak_rss_mb": report["peak_rss_mb"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
