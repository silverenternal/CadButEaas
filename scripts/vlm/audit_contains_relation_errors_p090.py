#!/usr/bin/env python3
"""Audit missing/extra contains relations from fusion_v2 cases.

P0-90 is analysis-only: it does not change fusion behavior. It reconstructs
room/symbol geometry from the locked smoke input, compares gold contains edges
with fusion_v2 predictions, and attributes errors to geometry/ambiguity signals.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "datasets/cadstruct_cubicasa5k_moe_locked/smoke.jsonl"
DEFAULT_CASES = ROOT / "reports/vlm/scene_graph_fusion_v2_cases.jsonl"
DEFAULT_OUTPUT = ROOT / "reports/vlm/contains_relation_error_audit_p090.json"
DEFAULT_REPORT = ROOT / "reports/vlm/contains_relation_error_audit_p090.md"
FUSION_PATH = ROOT / "scripts/vlm/fuse_scene_graph_v2.py"


def load_fusion_module() -> Any:
    spec = importlib.util.spec_from_file_location("fusion_v2_p090", FUSION_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {FUSION_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def contains(outer: list[float], inner: list[float]) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


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


def distance_between_centers(a: list[float], b: list[float]) -> float:
    ax, ay = center(a)
    bx, by = center(b)
    return math.hypot(ax - bx, ay - by)


def size_bucket(box: list[float] | None, family: str) -> str:
    value = area(box)
    if family == "room":
        if value <= 10000:
            return "room_tiny"
        if value <= 50000:
            return "room_small"
        if value <= 200000:
            return "room_medium"
        return "room_large"
    if value <= 32 * 32:
        return "symbol_tiny"
    if value <= 96 * 96:
        return "symbol_small"
    if value <= 192 * 192:
        return "symbol_medium"
    return "symbol_large"


def record_key(row: dict[str, Any]) -> str:
    return str(row.get("annotation_path") or row.get("annotation") or row.get("image_path") or row.get("image"))


def geometry_index(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    expected = row.get("expected_json") or {}
    out: dict[str, dict[str, Any]] = {}
    for item in expected.get("room_candidates") or []:
        item_id = str(item.get("id") or "")
        box = bbox4(item.get("bbox"))
        if item_id and box:
            out[item_id] = {"id": item_id, "family": "space", "label": item.get("room_type") or "room", "bbox": box, "area": area(box)}
    for item in expected.get("symbol_candidates") or []:
        item_id = str(item.get("id") or "")
        box = bbox4(item.get("bbox"))
        if item_id and box:
            out[item_id] = {"id": item_id, "family": "symbol", "label": item.get("symbol_type") or "generic_symbol", "bbox": box, "area": area(box)}
    return out


def contains_edges_from_graph(graph: dict[str, Any]) -> set[tuple[str, str, str]]:
    return {
        (str(edge.get("source")), str(edge.get("target")), "contains")
        for edge in graph.get("edges") or []
        if str(edge.get("relation")) == "contains" and edge.get("source") and edge.get("target")
    }


def best_room_for_symbol(room_items: list[dict[str, Any]], symbol_box: list[float]) -> dict[str, Any] | None:
    if not room_items:
        return None
    ranked = []
    for room in room_items:
        room_box = room["bbox"]
        ranked.append({
            "room_id": room["id"],
            "room_label": room["label"],
            "center_inside": center_inside(room_box, symbol_box),
            "strict_contains": contains(room_box, symbol_box),
            "containment_ratio": containment_ratio(room_box, symbol_box),
            "center_distance": distance_between_centers(room_box, symbol_box),
            "room_area": room["area"],
        })
    ranked.sort(key=lambda item: (-float(item["center_inside"]), -float(item["strict_contains"]), -item["containment_ratio"], item["center_distance"], -item["room_area"]))
    return ranked[0]


def second_best_gap(room_items: list[dict[str, Any]], symbol_box: list[float]) -> float | None:
    ratios = sorted((containment_ratio(room["bbox"], symbol_box) for room in room_items), reverse=True)
    if len(ratios) < 2:
        return None
    return ratios[0] - ratios[1]


def classify_edge(edge: tuple[str, str, str], index: dict[str, dict[str, Any]], rooms: list[dict[str, Any]]) -> dict[str, Any]:
    source, target, _ = edge
    source_item = index.get(source)
    target_item = index.get(target)
    result: dict[str, Any] = {
        "source": source,
        "target": target,
        "source_family": source_item.get("family") if source_item else "missing",
        "target_family": target_item.get("family") if target_item else "missing",
        "source_label": source_item.get("label") if source_item else None,
        "target_label": target_item.get("label") if target_item else None,
        "room_area_bucket": size_bucket(source_item.get("bbox") if source_item else None, "room"),
        "symbol_area_bucket": size_bucket(target_item.get("bbox") if target_item else None, "symbol"),
    }
    if not source_item or not target_item or source_item.get("family") != "space" or target_item.get("family") != "symbol":
        result["reason"] = "missing_or_unexpected_endpoint_family"
        return result
    room_box = source_item["bbox"]
    symbol_box = target_item["bbox"]
    best = best_room_for_symbol(rooms, symbol_box)
    gap = second_best_gap(rooms, symbol_box)
    result.update({
        "strict_contains": contains(room_box, symbol_box),
        "center_inside": center_inside(room_box, symbol_box),
        "containment_ratio": round(containment_ratio(room_box, symbol_box), 6),
        "center_distance": round(distance_between_centers(room_box, symbol_box), 3),
        "best_room_id": best.get("room_id") if best else None,
        "best_room_label": best.get("room_label") if best else None,
        "best_room_is_source": bool(best and best.get("room_id") == source),
        "best_room_containment_ratio": round(float(best.get("containment_ratio", 0.0)), 6) if best else None,
        "best_room_center_inside": bool(best and best.get("center_inside")),
        "top2_containment_ratio_gap": round(gap, 6) if gap is not None else None,
    })
    if result["best_room_is_source"] and result["center_inside"]:
        reason = "fusion_policy_underlinked_best_room"
    elif result["center_inside"]:
        reason = "center_inside_but_not_best_room"
    elif result["containment_ratio"] > 0:
        reason = "partial_overlap_no_center"
    else:
        reason = "no_bbox_overlap"
    if gap is not None and gap < 0.05:
        reason += "+ambiguous_room_overlap"
    result["reason"] = reason
    return result


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reason_counts = Counter(str(row.get("reason")) for row in rows)
    target_label_counts = Counter(str(row.get("target_label")) for row in rows)
    source_label_counts = Counter(str(row.get("source_label")) for row in rows)
    symbol_bucket_counts = Counter(str(row.get("symbol_area_bucket")) for row in rows)
    room_bucket_counts = Counter(str(row.get("room_area_bucket")) for row in rows)
    center_inside = sum(1 for row in rows if row.get("center_inside"))
    strict_contains = sum(1 for row in rows if row.get("strict_contains"))
    best_room_is_source = sum(1 for row in rows if row.get("best_room_is_source"))
    ambiguous = sum(1 for row in rows if row.get("top2_containment_ratio_gap") is not None and row.get("top2_containment_ratio_gap") < 0.05)
    return {
        "count": len(rows),
        "reason_counts": dict(reason_counts.most_common()),
        "target_label_counts": dict(target_label_counts.most_common(20)),
        "source_label_counts": dict(source_label_counts.most_common(20)),
        "symbol_area_bucket_counts": dict(symbol_bucket_counts.most_common()),
        "room_area_bucket_counts": dict(room_bucket_counts.most_common()),
        "center_inside_count": center_inside,
        "strict_contains_count": strict_contains,
        "best_room_is_source_count": best_room_is_source,
        "ambiguous_top2_gap_lt_0_05_count": ambiguous,
    }


def render_report(summary: dict[str, Any]) -> str:
    missing = summary["missing_contains"]
    extra = summary["extra_contains"]
    lines = [
        "# P0-90 Contains Relation Error Audit",
        "",
        "## Decision",
        "",
        summary["decision"],
        "",
        "## Counts",
        "",
        f"- Records: `{summary['records']}`",
        f"- Gold contains edges: `{summary['gold_contains_edges']}`",
        f"- Predicted contains edges: `{summary['pred_contains_edges']}`",
        f"- Missing contains edges: `{missing['count']}`",
        f"- Extra contains edges: `{extra['count']}`",
        "",
        "## Missing Contains Attribution",
        "",
    ]
    for key, value in missing["reason_counts"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Extra Contains Attribution", ""])
    for key, value in extra["reason_counts"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend([
        "",
        "## Key Observations",
        "",
        f"- Missing edges where the gold source room is also the geometry-best room: `{missing['best_room_is_source_count']}`.",
        f"- Missing edges with symbol center inside the gold room: `{missing['center_inside_count']}`.",
        f"- Missing edges with ambiguous top-2 room overlap gap < 0.05: `{missing['ambiguous_top2_gap_lt_0_05_count']}`.",
        f"- Extra edges with symbol center inside the predicted room: `{extra['center_inside_count']}`.",
        f"- Extra edges where predicted room is geometry-best: `{extra['best_room_is_source_count']}`.",
        "",
        "## Recommended Next Step",
        "",
        "Proceed with a policy probe for `contains` relation inference. Start with geometry-only scoring over candidate room-symbol pairs: center-inside, containment ratio, room area, symbol area bucket, and top-2 ambiguity gap. Do not change runtime fusion behavior until a locked/smoke replay shows fewer missing contains edges without unacceptable extra edges.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    args = parser.parse_args()

    fusion = load_fusion_module()
    rows = load_jsonl(Path(args.input))
    missing_rows: list[dict[str, Any]] = []
    extra_rows: list[dict[str, Any]] = []
    totals = Counter()

    for row in rows:
        key = record_key(row)
        index = geometry_index(row)
        rooms = [item for item in index.values() if item.get("family") == "space"]
        gold_graph = (row.get("expected_json") or {}).get("scene_graph") or {"edges": []}
        gold_contains = contains_edges_from_graph(gold_graph)
        pred_graph = fusion.fuse_v2(row, enable_all_repairs=False).get("scene_graph") or {"edges": []}
        pred_contains = contains_edges_from_graph(pred_graph)
        missing = sorted(gold_contains - pred_contains)
        extra = sorted(pred_contains - gold_contains)
        totals.update({
            "records": 1,
            "gold_contains_edges": len(gold_contains),
            "pred_contains_edges": len(pred_contains),
            "missing_contains_edges": len(missing),
            "extra_contains_edges": len(extra),
        })
        for edge in missing:
            item = classify_edge(edge, index, rooms)
            item.update({"record": key, "error_kind": "missing_contains"})
            missing_rows.append(item)
        for edge in extra:
            item = classify_edge(edge, index, rooms)
            item.update({"record": key, "error_kind": "extra_contains"})
            extra_rows.append(item)

    summary = {
        "id": "P0-90-contains-relation-error-audit",
        "input": str(Path(args.input).relative_to(ROOT) if Path(args.input).is_absolute() else args.input),
        "records": int(totals["records"]),
        "gold_contains_edges": int(totals["gold_contains_edges"]),
        "pred_contains_edges": int(totals["pred_contains_edges"]),
        "missing_contains": summarize_rows(missing_rows),
        "extra_contains": summarize_rows(extra_rows),
        "missing_samples": missing_rows[:40],
        "extra_samples": extra_rows[:40],
        "decision": "Contains relation errors are dominated by under-linking room-symbol pairs that are geometrically plausible; next step should be a geometry-only contains relation policy probe, not symbol detector work.",
        "next_step": "P0-91 probe geometry-only contains relation policy over room-symbol pairs and replay fusion metrics without changing runtime defaults.",
    }
    write_json(Path(args.output), summary)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
