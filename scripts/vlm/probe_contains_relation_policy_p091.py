#!/usr/bin/env python3
"""Probe geometry-only contains relation policies without changing runtime fusion.

The probe reuses fusion_v2 nodes/edges, removes predicted contains edges, then
rebuilds contains edges from room-symbol geometry under simple thresholds. It
reports relation precision/recall/F1 and missing/extra contains deltas.
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
DEFAULT_OUTPUT = ROOT / "reports/vlm/contains_relation_policy_probe_p091.json"
DEFAULT_REPORT = ROOT / "reports/vlm/contains_relation_policy_probe_p091.md"
FUSION_PATH = ROOT / "scripts/vlm/fuse_scene_graph_v2.py"


def load_fusion_module() -> Any:
    spec = importlib.util.spec_from_file_location("fusion_v2_p091", FUSION_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {FUSION_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
        (str(n.get("id")), str(n.get("semantic_type")), str(n.get("family") or "unknown"))
        for n in graph.get("nodes") or []
        if n.get("id") and n.get("semantic_type")
    }
    edges = {
        (str(e.get("source")), str(e.get("target")), str(e.get("relation")))
        for e in graph.get("edges") or []
        if e.get("source") and e.get("target") and e.get("relation")
    }
    return nodes, edges


def prf(tp: int, predicted: int, gold: int) -> dict[str, Any]:
    precision = tp / max(predicted, 1)
    recall = tp / max(gold, 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"tp": int(tp), "predicted": int(predicted), "gold": int(gold), "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6)}


def candidate_contains_edges(graph: dict[str, Any], config: dict[str, Any]) -> set[tuple[str, str, str]]:
    nodes = graph.get("nodes") or []
    rooms = [node for node in nodes if node.get("family") == "space" and bbox4((node.get("geometry") or {}).get("bbox"))]
    symbols = [node for node in nodes if node.get("family") == "symbol" and bbox4((node.get("geometry") or {}).get("bbox"))]
    max_edges_per_symbol = int(config.get("max_edges_per_symbol", 1))
    min_containment = float(config.get("min_containment", 1.0))
    require_center = bool(config.get("require_center", True))
    max_top2_gap = config.get("max_top2_gap")
    max_top2_gap = None if max_top2_gap is None else float(max_top2_gap)
    allow_multi_when_ambiguous = bool(config.get("allow_multi_when_ambiguous", False))
    min_room_area = float(config.get("min_room_area", 0.0))
    edges: set[tuple[str, str, str]] = set()

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
                "center_inside": inside,
                "room_area": area(room_box),
                "distance": center_distance(room_box, symbol_box),
            })
        if not candidates:
            continue
        candidates.sort(key=lambda item: (-item["ratio"], -float(item["center_inside"]), item["distance"], -item["room_area"], item["room_id"]))
        if allow_multi_when_ambiguous and len(candidates) > 1 and max_top2_gap is not None:
            top_ratio = candidates[0]["ratio"]
            selected = [item for item in candidates if top_ratio - item["ratio"] <= max_top2_gap][:max_edges_per_symbol]
        else:
            selected = candidates[:max_edges_per_symbol]
        for item in selected:
            edges.add((item["room_id"], item["symbol_id"], "contains"))
    return edges


def evaluate_config(rows: list[dict[str, Any]], fusion: Any, config: dict[str, Any]) -> dict[str, Any]:
    totals = Counter()
    for row in rows:
        gold_graph = (row.get("expected_json") or {}).get("scene_graph") or {"nodes": [], "edges": []}
        _, gold_edges = graph_sets(gold_graph)
        gold_contains = {edge for edge in gold_edges if edge[2] == "contains"}
        fused = fusion.fuse_v2(row, enable_all_repairs=False)
        pred_graph = fused.get("scene_graph") or {"nodes": [], "edges": []}
        _, base_edges = graph_sets(pred_graph)
        base_non_contains = {edge for edge in base_edges if edge[2] != "contains"}
        proposed_contains = candidate_contains_edges(pred_graph, config)
        proposed_edges = base_non_contains | proposed_contains
        totals.update({
            "records": 1,
            "edge_tp": len(proposed_edges & gold_edges),
            "edge_pred": len(proposed_edges),
            "edge_gold": len(gold_edges),
            "contains_tp": len(proposed_contains & gold_contains),
            "contains_pred": len(proposed_contains),
            "contains_gold": len(gold_contains),
            "contains_missing": len(gold_contains - proposed_contains),
            "contains_extra": len(proposed_contains - gold_contains),
        })
    return {
        "config": config,
        "records": int(totals["records"]),
        "relation_f1": prf(totals["edge_tp"], totals["edge_pred"], totals["edge_gold"]),
        "contains_f1": prf(totals["contains_tp"], totals["contains_pred"], totals["contains_gold"]),
        "contains_missing": int(totals["contains_missing"]),
        "contains_extra": int(totals["contains_extra"]),
    }


def sweep_configs() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for min_containment in [1.0, 0.95, 0.90, 0.75, 0.50]:
        for max_edges in [1, 2, 4]:
            configs.append({
                "name": f"center_ratio_{min_containment:g}_top{max_edges}",
                "require_center": True,
                "min_containment": min_containment,
                "max_edges_per_symbol": max_edges,
                "allow_multi_when_ambiguous": False,
                "max_top2_gap": None,
                "min_room_area": 0.0,
            })
    for gap in [0.0, 0.01, 0.05, 0.10]:
        configs.append({
            "name": f"center_full_multi_gap_{gap:g}",
            "require_center": True,
            "min_containment": 1.0,
            "max_edges_per_symbol": 4,
            "allow_multi_when_ambiguous": True,
            "max_top2_gap": gap,
            "min_room_area": 0.0,
        })
    return configs


def render_report(summary: dict[str, Any]) -> str:
    base = summary["baseline"]
    best = summary["best_policy"]
    lines = [
        "# P0-91 Geometry-Only Contains Relation Policy Probe",
        "",
        "## Decision",
        "",
        summary["decision"],
        "",
        "## Baseline vs Best Probe",
        "",
        "| Policy | Relation F1 | Contains recall | Contains precision | Missing contains | Extra contains |",
        "|---|---:|---:|---:|---:|---:|",
        "| baseline fusion_v2 | {} | {} | {} | {} | {} |".format(base["relation_f1"]["f1"], base["contains_f1"]["recall"], base["contains_f1"]["precision"], base["contains_missing"], base["contains_extra"]),
        "| `{}` | {} | {} | {} | {} | {} |".format(best["config"]["name"], best["relation_f1"]["f1"], best["contains_f1"]["recall"], best["contains_f1"]["precision"], best["contains_missing"], best["contains_extra"]),
        "",
        "## Best Config",
        "",
    ]
    for key, value in best["config"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend([
        "",
        "## Top Sweep Results",
        "",
        "| Rank | Config | Relation F1 | Contains recall | Contains precision | Missing | Extra |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ])
    for idx, item in enumerate(summary["top_results"][:10], start=1):
        lines.append("| {} | `{}` | {} | {} | {} | {} | {} |".format(idx, item["config"]["name"], item["relation_f1"]["f1"], item["contains_f1"]["recall"], item["contains_f1"]["precision"], item["contains_missing"], item["contains_extra"]))
    lines.extend([
        "",
        "## Next Step",
        "",
        "Do not patch runtime fusion yet. Package the best geometry-only policy as an opt-in replay candidate, then validate on a separate split or locked replay before replacing the current relation repair rule.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    fusion = load_fusion_module()
    baseline_config = {
        "name": "baseline_fusion_v2",
        "require_center": True,
        "min_containment": 1.0,
        "max_edges_per_symbol": 1,
        "allow_multi_when_ambiguous": False,
        "max_top2_gap": None,
        "min_room_area": 0.0,
    }
    # Baseline should use actual fusion contains, not rebuilt policy.
    baseline_totals = Counter()
    for row in rows:
        gold_graph = (row.get("expected_json") or {}).get("scene_graph") or {"nodes": [], "edges": []}
        _, gold_edges = graph_sets(gold_graph)
        gold_contains = {edge for edge in gold_edges if edge[2] == "contains"}
        fused = fusion.fuse_v2(row, enable_all_repairs=False)
        _, base_edges = graph_sets(fused.get("scene_graph") or {"nodes": [], "edges": []})
        base_contains = {edge for edge in base_edges if edge[2] == "contains"}
        baseline_totals.update({
            "records": 1,
            "edge_tp": len(base_edges & gold_edges),
            "edge_pred": len(base_edges),
            "edge_gold": len(gold_edges),
            "contains_tp": len(base_contains & gold_contains),
            "contains_pred": len(base_contains),
            "contains_gold": len(gold_contains),
            "contains_missing": len(gold_contains - base_contains),
            "contains_extra": len(base_contains - gold_contains),
        })
    baseline = {
        "config": baseline_config,
        "records": int(baseline_totals["records"]),
        "relation_f1": prf(baseline_totals["edge_tp"], baseline_totals["edge_pred"], baseline_totals["edge_gold"]),
        "contains_f1": prf(baseline_totals["contains_tp"], baseline_totals["contains_pred"], baseline_totals["contains_gold"]),
        "contains_missing": int(baseline_totals["contains_missing"]),
        "contains_extra": int(baseline_totals["contains_extra"]),
    }

    results = [evaluate_config(rows, fusion, config) for config in sweep_configs()]
    results.sort(key=lambda item: (item["relation_f1"]["f1"], item["contains_f1"]["f1"], -item["contains_extra"]), reverse=True)
    best = results[0]
    decision = "Geometry-only contains policy probe is positive as an opt-in candidate: it improves relation F1 and sharply reduces missing contains, but runtime fusion should not be changed until separate validation."
    if best["relation_f1"]["f1"] <= baseline["relation_f1"]["f1"]:
        decision = "Geometry-only contains policy probe is not yet better than baseline; keep auditing before runtime changes."
    summary = {
        "id": "P0-91-geometry-only-contains-relation-policy-probe",
        "input": str(Path(args.input).relative_to(ROOT) if Path(args.input).is_absolute() else args.input),
        "records": len(rows),
        "decision": decision,
        "baseline": baseline,
        "best_policy": best,
        "top_results": results[:20],
        "all_results_count": len(results),
        "next_step": "Package best policy as an opt-in relation replay candidate and validate separately before touching fusion_v2 defaults.",
    }
    write_json(Path(args.output), summary)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
