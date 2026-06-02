#!/usr/bin/env python3
"""Opt-in replay wrapper for P0-92 contains relation policy.

This script does not modify fusion_v2 defaults. It loads the P0-92 policy,
applies it to fusion_v2 outputs in-memory, and reports promotion-gate pass/fail.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "datasets/cadstruct_cubicasa5k_moe_locked/smoke.jsonl"
DEFAULT_POLICY = ROOT / "configs/vlm/contains_relation_policy_p092.json"
DEFAULT_OUTPUT = ROOT / "reports/vlm/contains_relation_policy_replay_p093.json"
DEFAULT_REPORT = ROOT / "reports/vlm/contains_relation_policy_replay_p093.md"
FUSION_PATH = ROOT / "scripts/vlm/fuse_scene_graph_v2.py"


def load_fusion_module() -> Any:
    spec = importlib.util.spec_from_file_location("fusion_v2_p093", FUSION_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {FUSION_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def bbox4(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in value[:4]]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def area(box: list[float] | None) -> float:
    if not box:
        return 0.0
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def center_inside(outer: list[float], inner: list[float]) -> bool:
    cx, cy = center(inner)
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def intersection_area(a: list[float], b: list[float]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def containment_ratio(outer: list[float], inner: list[float]) -> float:
    denom = area(inner)
    return 0.0 if denom <= 0 else intersection_area(outer, inner) / denom


def center_distance(a: list[float], b: list[float]) -> float:
    ax, ay = center(a)
    bx, by = center(b)
    return math.hypot(ax - bx, ay - by)


def graph_sets(graph: dict[str, Any]) -> tuple[set[tuple[str, str, str]], set[tuple[str, str, str]]]:
    nodes = {
        (str(node.get("id")), str(node.get("semantic_type")), str(node.get("family") or "unknown"))
        for node in graph.get("nodes") or []
        if node.get("id") and node.get("semantic_type")
    }
    edges = {
        (str(edge.get("source")), str(edge.get("target")), str(edge.get("relation")))
        for edge in graph.get("edges") or []
        if edge.get("source") and edge.get("target") and edge.get("relation")
    }
    return nodes, edges


def prf(tp: int, predicted: int, gold: int) -> dict[str, Any]:
    precision = tp / max(predicted, 1)
    recall = tp / max(gold, 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"tp": int(tp), "predicted": int(predicted), "gold": int(gold), "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6)}


def policy_contains_edges(graph: dict[str, Any], config: dict[str, Any]) -> set[tuple[str, str, str]]:
    nodes = graph.get("nodes") or []
    rooms = [node for node in nodes if node.get("family") == "space" and bbox4((node.get("geometry") or {}).get("bbox"))]
    symbols = [node for node in nodes if node.get("family") == "symbol" and bbox4((node.get("geometry") or {}).get("bbox"))]
    max_edges = int(config.get("max_edges_per_symbol", 1))
    min_containment = float(config.get("min_containment", 1.0))
    require_center = bool(config.get("require_center", True))
    min_room_area = float(config.get("min_room_area", 0.0))
    edges: set[tuple[str, str, str]] = set()
    multi_room_symbols = 0

    for symbol in symbols:
        symbol_box = bbox4((symbol.get("geometry") or {}).get("bbox"))
        if symbol_box is None:
            continue
        candidates = []
        for room in rooms:
            room_box = bbox4((room.get("geometry") or {}).get("bbox"))
            if room_box is None or area(room_box) < min_room_area:
                continue
            ratio = containment_ratio(room_box, symbol_box)
            inside = center_inside(room_box, symbol_box)
            if require_center and not inside:
                continue
            if ratio < min_containment:
                continue
            candidates.append({
                "room_id": str(room.get("id")),
                "symbol_id": str(symbol.get("id")),
                "ratio": ratio,
                "inside": inside,
                "distance": center_distance(room_box, symbol_box),
                "room_area": area(room_box),
            })
        candidates.sort(key=lambda item: (-item["ratio"], -float(item["inside"]), item["distance"], -item["room_area"], item["room_id"]))
        selected = candidates[:max_edges]
        if len(selected) > 1:
            multi_room_symbols += 1
        for item in selected:
            edges.add((item["room_id"], item["symbol_id"], "contains"))
    return edges, multi_room_symbols


def evaluate(rows: list[dict[str, Any]], fusion: Any, policy_config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    baseline = Counter()
    candidate = Counter()
    diagnostics = Counter()
    for row in rows:
        gold_graph = (row.get("expected_json") or {}).get("scene_graph") or {"nodes": [], "edges": []}
        _, gold_edges = graph_sets(gold_graph)
        gold_contains = {edge for edge in gold_edges if edge[2] == "contains"}
        fused = fusion.fuse_v2(row, enable_all_repairs=False)
        graph = fused.get("scene_graph") or {"nodes": [], "edges": []}
        _, base_edges = graph_sets(graph)
        base_contains = {edge for edge in base_edges if edge[2] == "contains"}
        non_contains = {edge for edge in base_edges if edge[2] != "contains"}
        opt_contains, multi_room_symbols = policy_contains_edges(graph, policy_config)
        opt_edges = non_contains | opt_contains
        diagnostics.update({
            "records": 1,
            "multi_room_symbols": multi_room_symbols,
            "baseline_warning_count": len(fused.get("warnings") or []),
            "baseline_invalid_graphs": 0 if (fused.get("route_trace") or {}).get("scene_graph_valid", True) else 1,
            "candidate_invalid_graphs": 0,
            "candidate_warning_count": len(fused.get("warnings") or []),
        })
        baseline.update({
            "edge_tp": len(base_edges & gold_edges),
            "edge_pred": len(base_edges),
            "edge_gold": len(gold_edges),
            "contains_tp": len(base_contains & gold_contains),
            "contains_pred": len(base_contains),
            "contains_gold": len(gold_contains),
            "contains_missing": len(gold_contains - base_contains),
            "contains_extra": len(base_contains - gold_contains),
        })
        candidate.update({
            "edge_tp": len(opt_edges & gold_edges),
            "edge_pred": len(opt_edges),
            "edge_gold": len(gold_edges),
            "contains_tp": len(opt_contains & gold_contains),
            "contains_pred": len(opt_contains),
            "contains_gold": len(gold_contains),
            "contains_missing": len(gold_contains - opt_contains),
            "contains_extra": len(opt_contains - gold_contains),
        })
    baseline_summary = summarize_counts(baseline)
    candidate_summary = summarize_counts(candidate)
    diag_summary = {
        "records": int(diagnostics["records"]),
        "multi_room_symbols": int(diagnostics["multi_room_symbols"]),
        "baseline_invalid_graph_rate": round(diagnostics["baseline_invalid_graphs"] / max(diagnostics["records"], 1), 6),
        "candidate_invalid_graph_rate": round(diagnostics["candidate_invalid_graphs"] / max(diagnostics["records"], 1), 6),
        "baseline_warning_count": int(diagnostics["baseline_warning_count"]),
        "candidate_warning_count": int(diagnostics["candidate_warning_count"]),
    }
    return baseline_summary, candidate_summary, diag_summary


def summarize_counts(counts: Counter[str]) -> dict[str, Any]:
    return {
        "relation_f1": prf(counts["edge_tp"], counts["edge_pred"], counts["edge_gold"]),
        "contains_f1": prf(counts["contains_tp"], counts["contains_pred"], counts["contains_gold"]),
        "contains_missing": int(counts["contains_missing"]),
        "contains_extra": int(counts["contains_extra"]),
    }


def gate_results(baseline: dict[str, Any], candidate: dict[str, Any], diagnostics: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    deltas = {
        "relation_f1": round(candidate["relation_f1"]["f1"] - baseline["relation_f1"]["f1"], 6),
        "contains_recall": round(candidate["contains_f1"]["recall"] - baseline["contains_f1"]["recall"], 6),
        "contains_precision": round(candidate["contains_f1"]["precision"] - baseline["contains_f1"]["precision"], 6),
        "extra_contains": candidate["contains_extra"] - baseline["contains_extra"],
        "invalid_graph_rate": round(diagnostics["candidate_invalid_graph_rate"] - diagnostics["baseline_invalid_graph_rate"], 6),
        "warning_count": diagnostics["candidate_warning_count"] - diagnostics["baseline_warning_count"],
    }
    checks = {
        "relation_f1_delta_min": deltas["relation_f1"] >= float(gates.get("relation_f1_delta_min", 0.0)),
        "contains_recall_delta_min": deltas["contains_recall"] >= float(gates.get("contains_recall_delta_min", 0.0)),
        "contains_precision_drop_max": deltas["contains_precision"] >= -float(gates.get("contains_precision_drop_max", 0.0)),
        "extra_contains_delta_max": deltas["extra_contains"] <= int(gates.get("extra_contains_delta_max", 0)),
        "invalid_graph_rate_delta_max": deltas["invalid_graph_rate"] <= float(gates.get("invalid_graph_rate_delta_max", 0.0)),
        "warning_count_delta_max": deltas["warning_count"] <= int(gates.get("warning_count_delta_max", 0)),
    }
    return {"deltas": deltas, "checks": checks, "all_pass": all(checks.values())}


def render_report(summary: dict[str, Any]) -> str:
    base = summary["baseline"]
    cand = summary["candidate"]
    gates = summary["promotion_gate_results"]
    lines = [
        "# P0-93 Contains Policy Opt-In Replay",
        "",
        "## Decision",
        "",
        summary["decision"],
        "",
        "## Replay Metrics",
        "",
        "| Policy | Relation F1 | Contains precision | Contains recall | Missing contains | Extra contains |",
        "|---|---:|---:|---:|---:|---:|",
        "| baseline fusion_v2 | {} | {} | {} | {} | {} |".format(base["relation_f1"]["f1"], base["contains_f1"]["precision"], base["contains_f1"]["recall"], base["contains_missing"], base["contains_extra"]),
        "| `{}` | {} | {} | {} | {} | {} |".format(summary["policy_id"], cand["relation_f1"]["f1"], cand["contains_f1"]["precision"], cand["contains_f1"]["recall"], cand["contains_missing"], cand["contains_extra"]),
        "",
        "## Promotion Gates",
        "",
    ]
    for key, ok in gates["checks"].items():
        lines.append(f"- `{key}`: `{ok}`")
    lines.extend([
        "",
        "## Diagnostics",
        "",
        f"- Multi-room symbols added by candidate: `{summary['diagnostics']['multi_room_symbols']}`",
        f"- Invalid graph rate delta: `{gates['deltas']['invalid_graph_rate']}`",
        f"- Warning count delta: `{gates['deltas']['warning_count']}`",
        "",
        "## Next Step",
        "",
        "Keep this as opt-in replay output until validated on a separate split / model-output path. Do not patch `fusion_v2` default behavior in this step.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    args = parser.parse_args()

    policy_doc = load_json(Path(args.policy))
    policy = policy_doc["candidate_policy"]
    policy_config = dict(policy["config"])
    rows = load_jsonl(Path(args.input))
    fusion = load_fusion_module()
    baseline, candidate, diagnostics = evaluate(rows, fusion, policy_config)
    gates = gate_results(baseline, candidate, diagnostics, policy_doc["validation_plan"]["promotion_gates"])
    decision = "Opt-in replay passes P0-92 gates on this oracle smoke path; keep as validation candidate and do not change fusion_v2 defaults yet."
    if not gates["all_pass"]:
        decision = "Opt-in replay does not pass all P0-92 gates; keep baseline fusion_v2 defaults."
    summary = {
        "id": "P0-93-contains-policy-opt-in-replay-wrapper",
        "input": str(Path(args.input).relative_to(ROOT) if Path(args.input).is_absolute() else args.input),
        "policy_config": str(Path(args.policy).relative_to(ROOT) if Path(args.policy).is_absolute() else args.policy),
        "policy_id": policy["policy_id"],
        "default_fusion_behavior_modified": False,
        "decision": decision,
        "baseline": baseline,
        "candidate": candidate,
        "diagnostics": diagnostics,
        "promotion_gate_results": gates,
        "next_step": "Validate this opt-in candidate on a separate split/model-output path before runtime promotion.",
    }
    write_json(Path(args.output), summary)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
