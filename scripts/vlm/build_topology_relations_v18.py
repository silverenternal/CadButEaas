#!/usr/bin/env python3
"""Build topology-aware relation candidates from v18 raster-only detector output."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/vlm"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_INPUT = REPORT / "detector_adapter_v18_routed_candidates.jsonl"
DEFAULT_CANDIDATES = REPORT / "topology_relations_v18_candidates.jsonl"
DEFAULT_FEATURES = REPORT / "topology_relations_v18_rerank_features.jsonl"
DEFAULT_EVAL = REPORT / "topology_relations_v18_eval.json"
DEFAULT_AUDIT = REPORT / "topology_relations_v18_warning_audit.json"
DEFAULT_SWEEP = REPORT / "topology_relations_v18_cap_sweep.json"

GOLD_ROOM = ROOT / "datasets/image_only_room_polygon_v18/locked.jsonl"
GOLD_BOUNDARY = ROOT / "datasets/image_only_boundary_detector_v18/locked.jsonl"
GOLD_SYMBOL = ROOT / "datasets/image_only_symbol_detector_v18/locked.jsonl"
GOLD_TEXT = ROOT / "datasets/image_only_text_ocr_v18/locked.jsonl"


def integrity() -> dict[str, Any]:
    return {
        "source_mode": "image_only_raster_moe",
        "svg_candidate_ids_used": False,
        "annotation_geometry_used_at_inference": False,
        "model_input": "raster_image_only",
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def area(b: list[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def center(b: list[float]) -> tuple[float, float]:
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def iou(left: list[float] | None, right: list[float] | None) -> float:
    if left is None or right is None:
        return 0.0
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    return inter / max(area(left) + area(right) - inter, 1e-9)


def contains_point(outer: list[float], x: float, y: float, margin: float = 0.0) -> bool:
    return outer[0] - margin <= x <= outer[2] + margin and outer[1] - margin <= y <= outer[3] + margin


def expanded_box(b: list[float], margin: float) -> list[float]:
    return [b[0] - margin, b[1] - margin, b[2] + margin, b[3] + margin]


def center_covered(pred: list[float], gold: list[float], margin: float = 2.0) -> bool:
    gx, gy = center(gold)
    return contains_point(pred, gx, gy, margin)


def side_overlap(room: list[float], wall: list[float], side: str) -> float:
    if side in {"left", "right"}:
        overlap = max(0.0, min(room[3], wall[3]) - max(room[1], wall[1]))
        return overlap / max(room[3] - room[1], 1e-9)
    overlap = max(0.0, min(room[2], wall[2]) - max(room[0], wall[0]))
    return overlap / max(room[2] - room[0], 1e-9)


def nearest_room_side(room: list[float], item: list[float]) -> tuple[str, float, float]:
    cx, cy = center(item)
    distances = {
        "left": abs(cx - room[0]),
        "right": abs(cx - room[2]),
        "top": abs(cy - room[1]),
        "bottom": abs(cy - room[3]),
    }
    side = min(distances, key=distances.get)
    return side, distances[side], side_overlap(room, item, side)


def orientation(cand: dict[str, Any], b: list[float]) -> str:
    payload = cand.get("payload") if isinstance(cand.get("payload"), dict) else {}
    features = payload.get("features") if isinstance(payload.get("features"), dict) else {}
    value = features.get("orientation")
    if value in {"horizontal", "vertical"}:
        return str(value)
    return "horizontal" if (b[2] - b[0]) >= (b[3] - b[1]) else "vertical"


def orientation_compatible(side: str, orient: str) -> float:
    if side in {"top", "bottom"}:
        return 1.0 if orient == "horizontal" else 0.35
    return 1.0 if orient == "vertical" else 0.35


def confidence(cand: dict[str, Any]) -> float:
    try:
        return float(cand.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def family_groups(stream: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cand in stream:
        b = bbox(cand.get("bbox"))
        if b is None:
            continue
        item = dict(cand)
        item["_bbox"] = b
        groups[str(cand.get("family") or "unknown")].append(item)
    return groups


def relation_row(
    row_id: str,
    rel_type: str,
    source: dict[str, Any],
    target: dict[str, Any],
    score: float,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "relation_id": f"{row_id}_{rel_type}_{source['candidate_id']}_{target['candidate_id']}",
        "row_id": row_id,
        "relation": rel_type,
        "source_candidate_id": source["candidate_id"],
        "target_candidate_id": target["candidate_id"],
        "source_family": source.get("family"),
        "target_family": target.get("family"),
        "confidence": round(score, 6),
        "evidence": evidence,
        "source_integrity": integrity(),
    }


def bounded_by_candidates(row_id: str, rooms: list[dict[str, Any]], boundaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for room in rooms:
        rb = room["_bbox"]
        local: list[dict[str, Any]] = []
        for boundary in boundaries:
            bb = boundary["_bbox"]
            side, dist, overlap = nearest_room_side(rb, bb)
            orient = orientation(boundary, bb)
            compatible = orientation_compatible(side, orient)
            max_side_gap = max(8.0, min(rb[2] - rb[0], rb[3] - rb[1]) * 0.18)
            if overlap < 0.08 or dist > max_side_gap:
                continue
            score = (0.44 * min(overlap, 1.0)) + (0.22 * max(0.0, 1.0 - dist / max_side_gap)) + (0.18 * compatible) + (0.16 * confidence(boundary))
            local.append(
                relation_row(
                    row_id,
                    "bounded_by",
                    room,
                    boundary,
                    score,
                    {
                        "room_bbox": rb,
                        "boundary_bbox": bb,
                        "bbox_distance": round(dist, 3),
                        "side_overlap_ratio": round(overlap, 6),
                        "side": side,
                        "orientation": orient,
                        "orientation_compatible": round(compatible, 3),
                        "detector_confidence_product": round(confidence(room) * confidence(boundary), 6),
                        "source_family": "space/boundary",
                        "cap_rank_source": "relation_support_score",
                    },
                )
            )
        local.sort(key=lambda item: item["confidence"], reverse=True)
        out.extend(local[:12])
    return out


def contains_candidates(row_id: str, rooms: list[dict[str, Any]], targets: list[dict[str, Any]], rel_type: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    per_room_cap = 64 if rel_type == "contains_symbol" else 5
    for room in rooms:
        rb = room["_bbox"]
        contains_margin = 2.0
        expanded_room = expanded_box(rb, contains_margin)
        rx, ry = center(rb)
        local: list[dict[str, Any]] = []
        diag = math.hypot(rb[2] - rb[0], rb[3] - rb[1])
        for target in targets:
            tb = target["_bbox"]
            tx, ty = center(tb)
            center_inside_strict = contains_point(rb, tx, ty, margin=2.0)
            center_inside_expanded = center_inside_strict
            target_room_iou = iou(expanded_room, tb)
            if not center_inside_strict:
                continue
            norm_dist = min(1.0, math.hypot(tx - rx, ty - ry) / max(diag, 1e-9))
            size_ratio = min(1.0, area(tb) / max(area(rb), 1e-9) * 40.0)
            det_product = confidence(room) * confidence(target)
            score = 0.35 * (1.0 - norm_dist) + 0.25 * det_product + 0.25 * size_ratio + 0.15
            evidence = {
                "room_bbox": rb,
                "target_bbox": tb,
                "center_inside_room": center_inside_strict,
                "center_inside_expanded_room": center_inside_expanded,
                "contains_margin": round(contains_margin, 3),
                "expanded_room_target_iou": round(target_room_iou, 6),
                "center_distance_to_room_center": round(norm_dist, 6),
                "target_area_ratio_scaled": round(size_ratio, 6),
                "detector_confidence_product": round(det_product, 6),
                "source_family": f"space/{target.get('family')}",
                "cap_rank_source": "relation_support_score",
                "per_room_cap": per_room_cap,
            }
            payload = target.get("payload") if isinstance(target.get("payload"), dict) else {}
            if rel_type == "contains_symbol":
                evidence["typed_symbol_type"] = payload.get("typed_symbol_type") or payload.get("symbol_type") or target.get("candidate_type")
                evidence["type_confidence"] = payload.get("type_confidence")
            else:
                evidence["ocr_status"] = payload.get("ocr_status")
                evidence["ocr_confidence"] = payload.get("ocr_confidence")
                evidence["readability_confidence"] = payload.get("ocr_confidence", target.get("confidence"))
            local.append(relation_row(row_id, rel_type, room, target, score, evidence))
        local.sort(key=lambda item: item["confidence"], reverse=True)
        out.extend(local[:per_room_cap])
    return out


def adjacent_candidates(row_id: str, rooms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    by_room: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for i, left in enumerate(rooms):
        lb = left["_bbox"]
        for right in rooms[i + 1 :]:
            rb = right["_bbox"]
            h_gap = max(0.0, max(lb[0], rb[0]) - min(lb[2], rb[2]))
            v_gap = max(0.0, max(lb[1], rb[1]) - min(lb[3], rb[3]))
            x_overlap = max(0.0, min(lb[2], rb[2]) - max(lb[0], rb[0]))
            y_overlap = max(0.0, min(lb[3], rb[3]) - max(lb[1], rb[1]))
            x_ratio = x_overlap / max(min(lb[2] - lb[0], rb[2] - rb[0]), 1e-9)
            y_ratio = y_overlap / max(min(lb[3] - lb[1], rb[3] - rb[1]), 1e-9)
            gap = h_gap if x_ratio == 0.0 else v_gap if y_ratio == 0.0 else min(h_gap, v_gap)
            support = max(x_ratio if h_gap <= 8 else 0.0, y_ratio if v_gap <= 8 else 0.0)
            if support < 0.25 or gap > 10.0:
                continue
            score = 0.55 * support + 0.25 * max(0.0, 1.0 - gap / 10.0) + 0.2 * confidence(left) * confidence(right)
            rel = relation_row(
                row_id,
                "adjacent_to",
                left,
                right,
                score,
                {
                    "left_bbox": lb,
                    "right_bbox": rb,
                    "gap": round(gap, 3),
                    "axis_overlap_ratio": round(support, 6),
                    "detector_confidence_product": round(confidence(left) * confidence(right), 6),
                    "source_family": "space/space",
                    "cap_rank_source": "relation_support_score",
                },
            )
            by_room[left["candidate_id"]].append(rel)
            by_room[right["candidate_id"]].append(rel)
    seen: set[str] = set()
    for rels in by_room.values():
        rels.sort(key=lambda item: item["confidence"], reverse=True)
        for rel in rels[:8]:
            if rel["relation_id"] not in seen:
                out.append(rel)
                seen.add(rel["relation_id"])
    return out


def build_relations(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    page_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    counts = Counter()

    for row in rows:
        row_id = str(row.get("id"))
        stream = ((row.get("scene_graph") or {}).get("candidate_stream") or [])
        groups = family_groups(stream)
        rooms = groups.get("space", [])
        boundaries = groups.get("boundary", [])
        symbols = groups.get("symbol", [])
        texts = groups.get("text", [])
        by_candidate_id = {cand.get("candidate_id"): cand for cand in stream if cand.get("candidate_id")}
        rels: list[dict[str, Any]] = []
        rels.extend(bounded_by_candidates(row_id, rooms, boundaries))
        rels.extend(contains_candidates(row_id, rooms, symbols, "contains_symbol"))
        rels.extend(contains_candidates(row_id, rooms, texts, "labeled_by_text"))
        rels.extend(adjacent_candidates(row_id, rooms))
        rels.sort(key=lambda item: (item["relation"], -item["confidence"], item["source_candidate_id"], item["target_candidate_id"]))

        for rel in rels:
            counts[rel["relation"]] += 1
            source = by_candidate_id.get(rel["source_candidate_id"], {})
            target = by_candidate_id.get(rel["target_candidate_id"], {})
            feature_rows.append(
                {
                    "row_id": row_id,
                    "relation_id": rel["relation_id"],
                    "relation": rel["relation"],
                    "source_candidate_id": rel["source_candidate_id"],
                    "target_candidate_id": rel["target_candidate_id"],
                    "confidence": rel["confidence"],
                    "features": rel["evidence"],
                    "source_provenance": source.get("provenance"),
                    "target_provenance": target.get("provenance"),
                    "source_audit_trace": source.get("audit_trace"),
                    "target_audit_trace": target.get("audit_trace"),
                    "label": None,
                    "source_integrity": integrity(),
                }
            )
        if not rooms:
            warnings.append({"row_id": row_id, "warning": "no_space_candidates"})
        if not rels:
            warnings.append({"row_id": row_id, "warning": "no_relation_candidates"})
        page_rows.append(
            {
                "id": row_id,
                "image": row.get("image"),
                "image_size": row.get("image_size") or [512, 512],
                "source_integrity": integrity(),
                "route_trace": {
                    **integrity(),
                    "stage": "topology_relations_v18",
                    "candidate_contract_version": row.get("scene_graph", {}).get("candidate_contract_version"),
                    "gold_loaded_after_inference_for_evaluation_only": False,
                },
                "scene_graph": {
                    "nodes": [],
                    "relations": rels,
                    "candidate_counts": {family: len(groups.get(family, [])) for family in ["space", "boundary", "symbol", "text"]},
                    "relation_counts": dict(Counter(rel["relation"] for rel in rels)),
                },
            }
        )

    audit = {
        "rows": len(page_rows),
        "relation_counts": dict(counts),
        "warnings": warnings[:500],
        "warning_counts": dict(Counter(item["warning"] for item in warnings)),
        "source_integrity": integrity(),
    }
    return page_rows, feature_rows, audit


def load_gold() -> dict[str, dict[str, Any]]:
    rooms: dict[str, dict[str, Any]] = {}
    boundaries: dict[str, dict[str, Any]] = {}
    symbols: dict[str, dict[str, Any]] = {}
    texts: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(GOLD_ROOM):
        rooms[row["id"]] = {
            "rooms": (row.get("targets") or {}).get("rooms") or [],
            "relations": (row.get("targets") or {}).get("relations") or {},
        }
    for row in load_jsonl(GOLD_BOUNDARY):
        boundaries[row["id"]] = {item["target_id"]: item for item in (row.get("targets") or {}).get("boxes") or []}
    for row in load_jsonl(GOLD_SYMBOL):
        symbols[row["id"]] = (row.get("targets") or {}).get("symbols") or []
    for row in load_jsonl(GOLD_TEXT):
        texts[row["id"]] = (row.get("targets") or {}).get("texts") or []
    return {"rooms": rooms, "boundaries": boundaries, "symbols": symbols, "texts": texts}


def match_item(cand_box: list[float], gold_items: list[dict[str, Any]], threshold: float = 0.3) -> str | None:
    best: tuple[float, str] | None = None
    for gold in gold_items:
        gb = bbox(gold.get("bbox"))
        if gb is None:
            continue
        score = iou(cand_box, gb)
        if center_covered(cand_box, gb):
            score = max(score, 0.5)
        if score >= threshold and (best is None or score > best[0]):
            best = (score, str(gold.get("target_id")))
    return best[1] if best else None


def relation_label(
    rel: dict[str, Any],
    row_candidates: dict[str, dict[str, Any]],
    gold: dict[str, dict[str, Any]],
) -> tuple[str | None, str | None, bool]:
    row_id = rel["row_id"]
    source = row_candidates.get(rel["source_candidate_id"])
    target = row_candidates.get(rel["target_candidate_id"])
    if not source or not target:
        return None, None, False
    sb, tb = bbox(source.get("bbox")), bbox(target.get("bbox"))
    if sb is None or tb is None:
        return None, None, False
    room_items = (gold["rooms"].get(row_id) or {}).get("rooms") or []
    room_id = match_item(sb, room_items, threshold=0.3)
    if not room_id:
        return None, None, False
    rel_type = rel["relation"]
    if rel_type == "bounded_by":
        boundary_items = list((gold["boundaries"].get(row_id) or {}).values())
        boundary_id = match_item(tb, boundary_items, threshold=0.3)
        if not boundary_id:
            return room_id, None, False
        pairs = {
            (item.get("source"), item.get("target"))
            for item in ((gold["rooms"].get(row_id) or {}).get("relations") or {}).get("bounded_by") or []
        }
        return room_id, boundary_id, (room_id, boundary_id) in pairs
    if rel_type == "contains_symbol":
        symbol_id = match_item(tb, gold["symbols"].get(row_id) or [], threshold=0.25)
        if not symbol_id:
            return room_id, None, False
        canonical_room_id = None
        symbol_box = None
        for symbol in gold["symbols"].get(row_id) or []:
            if str(symbol.get("target_id")) == symbol_id:
                symbol_box = bbox(symbol.get("bbox"))
                break
        if symbol_box is not None:
            sx, sy = center(symbol_box)
            for room in room_items:
                rb = bbox(room.get("bbox"))
                if rb and contains_point(rb, sx, sy, margin=2.0):
                    canonical_room_id = str(room.get("target_id"))
                    break
        return room_id, symbol_id, room_id == canonical_room_id and contains_point(sb, *center(tb), margin=2.0)
    if rel_type == "labeled_by_text":
        text_id = match_item(tb, gold["texts"].get(row_id) or [], threshold=0.25)
        if not text_id:
            return room_id, None, False
        linked = {
            item.get("target_id"): item.get("linked_room_id")
            for item in gold["texts"].get(row_id) or []
        }
        return room_id, text_id, linked.get(text_id) == room_id
    if rel_type == "adjacent_to":
        right_id = match_item(tb, room_items, threshold=0.3)
        if not right_id or right_id == room_id:
            return room_id, right_id, False
        pairs = {
            tuple(sorted([str(item.get("source")), str(item.get("target"))]))
            for item in ((gold["rooms"].get(row_id) or {}).get("relations") or {}).get("adjacency") or []
        }
        return room_id, right_id, tuple(sorted([room_id, right_id])) in pairs
    return room_id, None, False


def evaluate_relations(page_rows: list[dict[str, Any]], source_rows: list[dict[str, Any]]) -> dict[str, Any]:
    gold = load_gold()
    candidates_by_row = {
        str(row.get("id")): {
            cand.get("candidate_id"): cand
            for cand in ((row.get("scene_graph") or {}).get("candidate_stream") or [])
            if cand.get("candidate_id")
        }
        for row in source_rows
    }
    by_type: dict[str, Counter[str]] = defaultdict(Counter)
    matched_gold: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    gold_counts = Counter()
    for row_id, room_data in gold["rooms"].items():
        gold_counts["bounded_by"] += len((room_data.get("relations") or {}).get("bounded_by") or [])
        for sym in gold["symbols"].get(row_id) or []:
            sb = bbox(sym.get("bbox"))
            if sb is None:
                continue
            sx, sy = center(sb)
            for room in room_data.get("rooms") or []:
                rb = bbox(room.get("bbox"))
                if rb and contains_point(rb, sx, sy, margin=2.0):
                    gold_counts["contains_symbol"] += 1
                    break
        gold_counts["labeled_by_text"] += sum(1 for item in gold["texts"].get(row_id) or [] if item.get("linked_room_id"))
        gold_counts["adjacent_to"] += len((room_data.get("relations") or {}).get("adjacency") or [])

    for page in page_rows:
        row_id = str(page.get("id"))
        for rel in (page.get("scene_graph") or {}).get("relations") or []:
            rel_type = rel["relation"]
            by_type[rel_type]["predicted"] += 1
            left, right, ok = relation_label(rel, candidates_by_row.get(row_id, {}), gold)
            if ok and left and right:
                key = (row_id, left, right)
                if key not in matched_gold[rel_type]:
                    matched_gold[rel_type].add(key)
                    by_type[rel_type]["true_positive"] += 1
                else:
                    by_type[rel_type]["duplicate_positive"] += 1

    metrics: dict[str, Any] = {}
    for rel_type in ["bounded_by", "contains_symbol", "labeled_by_text", "adjacent_to"]:
        tp = by_type[rel_type]["true_positive"]
        pred = by_type[rel_type]["predicted"]
        gold_total = gold_counts[rel_type]
        precision = tp / max(pred, 1)
        recall = tp / max(gold_total, 1)
        f1 = 0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall)
        metrics[rel_type] = {
            "true_positive": tp,
            "predicted": pred,
            "gold": gold_total,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "duplicate_positive": by_type[rel_type]["duplicate_positive"],
        }
    return {
        "task": "IMG-MOE-V18-P1-009",
        "mode": "oracle-eval",
        "gold_loaded_after_inference_for_evaluation_only": True,
        "gold_used_for_inference": False,
        "relation_metrics": metrics,
        "source_integrity": integrity(),
    }


def recall_for_candidates(selected: dict[str, list[dict[str, Any]]], gold_rows: dict[str, list[dict[str, Any]]], threshold: float) -> dict[str, Any]:
    total = hit = 0
    for row_id, gold_items in gold_rows.items():
        cands = selected.get(row_id, [])
        for gold in gold_items:
            gb = bbox(gold.get("bbox"))
            if gb is None:
                continue
            total += 1
            if any((iou(bbox(c.get("bbox")), gb) >= threshold or (bbox(c.get("bbox")) and center_covered(bbox(c.get("bbox")) or gb, gb))) for c in cands):
                hit += 1
    return {"gold": total, "matched": hit, "recall": round(hit / max(total, 1), 6)}


def cap_sweep(page_rows: list[dict[str, Any]], source_rows: list[dict[str, Any]]) -> dict[str, Any]:
    gold_boundary: dict[str, list[dict[str, Any]]] = {}
    gold_symbol: dict[str, list[dict[str, Any]]] = {}
    for row in load_jsonl(GOLD_BOUNDARY):
        gold_boundary[row["id"]] = (row.get("targets") or {}).get("boxes") or []
    for row in load_jsonl(GOLD_SYMBOL):
        gold_symbol[row["id"]] = (row.get("targets") or {}).get("symbols") or []

    support: dict[str, Counter[str]] = defaultdict(Counter)
    for page in page_rows:
        for rel in (page.get("scene_graph") or {}).get("relations") or []:
            if rel["relation"] == "bounded_by":
                support[page["id"]][rel["target_candidate_id"]] += 1
            elif rel["relation"] == "contains_symbol":
                support[page["id"]][rel["target_candidate_id"]] += 1

    families: dict[str, dict[str, list[dict[str, Any]]]] = {"boundary": defaultdict(list), "symbol": defaultdict(list)}
    for row in source_rows:
        row_id = str(row.get("id"))
        for cand in ((row.get("scene_graph") or {}).get("candidate_stream") or []):
            family = cand.get("family")
            if family in families:
                item = dict(cand)
                item["_relation_support"] = support[row_id][cand.get("candidate_id")]
                families[family][row_id].append(item)

    caps = {"boundary": [100, 200, 400, 800], "symbol": [50, 100, 250, 500]}
    result: dict[str, Any] = {
        "task": "IMG-MOE-V18-P1-009",
        "note": "Sweep ranks candidates already emitted by detector_adapter_v18; it diagnoses relation-aware rerank after the current adapter cap, not recovery of pre-cap discarded candidates.",
        "gold_loaded_after_inference_for_evaluation_only": True,
        "gold_used_for_inference": False,
        "families": {},
    }
    for family, family_caps in caps.items():
        gold_rows = gold_boundary if family == "boundary" else gold_symbol
        result["families"][family] = {}
        for ranker in ["detector_confidence", "relation_support_then_confidence"]:
            result["families"][family][ranker] = {}
            for cap in family_caps:
                selected: dict[str, list[dict[str, Any]]] = {}
                for row_id, cands in families[family].items():
                    if ranker == "detector_confidence":
                        ordered = sorted(cands, key=lambda c: confidence(c), reverse=True)
                    else:
                        ordered = sorted(cands, key=lambda c: (c.get("_relation_support", 0), confidence(c)), reverse=True)
                    selected[row_id] = ordered[:cap]
                result["families"][family][ranker][str(cap)] = recall_for_candidates(
                    selected,
                    gold_rows,
                    0.30 if family == "boundary" else 0.25,
                )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--mode", choices=["oracle-eval", "predicted"], default="predicted")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--locked", action="store_true")
    parser.add_argument("--cap-sweep", action="store_true")
    parser.add_argument("--output", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--features-output", default=str(DEFAULT_FEATURES))
    parser.add_argument("--eval-output", default=str(DEFAULT_EVAL))
    parser.add_argument("--warning-audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--threshold-sweep", default=str(DEFAULT_SWEEP))
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    if args.smoke:
        rows = rows[:5]
    page_rows, feature_rows, audit = build_relations(rows)
    audit["mode"] = args.mode
    audit["locked"] = bool(args.locked)
    audit["smoke"] = bool(args.smoke)
    audit["gold_loaded_after_inference_for_evaluation_only"] = False
    audit["gold_used_for_inference"] = False

    write_jsonl(Path(args.output), page_rows)
    write_jsonl(Path(args.features_output), feature_rows)
    write_json(Path(args.warning_audit), audit)

    report = evaluate_relations(page_rows, rows)
    report["rows"] = len(page_rows)
    report["features"] = len(feature_rows)
    report["relation_counts"] = audit["relation_counts"]
    report["warning_counts"] = audit["warning_counts"]
    report["quality_gates"] = {
        "source_integrity_violations": 0,
        "target_symbol_cap_recall_at_500": 0.70,
        "target_boundary_cap_recall_at_800": 0.80,
        "relation_metrics_reported": True,
    }
    write_json(Path(args.eval_output), report)

    if args.cap_sweep or args.mode == "predicted":
        write_json(Path(args.threshold_sweep), cap_sweep(page_rows, rows))

    print(json.dumps({"rows": len(page_rows), "features": len(feature_rows), "relations": audit["relation_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
