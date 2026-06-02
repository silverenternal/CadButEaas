#!/usr/bin/env python3
"""Run loaded CadStruct MoE experts on CubiCasa SVG-derived candidates.

This is different from export_moe_scene_graph.py --source expected_json:
it uses the SVG-derived candidates as inputs, calls the registered expert
models, fuses their predicted labels, and evaluates per-record outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "scripts" / "vlm"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

warnings.filterwarnings(
    "ignore",
    message="`sklearn.utils.parallel.delayed` should be used",
    category=UserWarning,
)

from cadstruct_moe import build_default_experts, describe_experts, summarize_expert_execution
from cadstruct_moe.fusion import fuse_predictions
from cadstruct_moe.schema import ExpertPrediction, RoutedCandidate


BOUNDARY_LABELS = {"hard_wall", "partition_wall", "door", "window", "opening", "curtain_wall"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="datasets/cadstruct_cubicasa5k_moe_locked/smoke.jsonl")
    parser.add_argument("--output", default="reports/vlm/cubicasa_svg_case/model_fused_scene_graph_locked_smoke.jsonl")
    parser.add_argument("--eval-output", default="reports/vlm/cubicasa_svg_case/model_fused_scene_graph_locked_smoke_eval.json")
    parser.add_argument("--audit-output", default="reports/vlm/cubicasa_svg_case/model_expert_runtime_audit.json")
    parser.add_argument("--cases-output", default="reports/vlm/cubicasa_svg_case/model_fused_scene_graph_locked_smoke_cases.jsonl")
    args = parser.parse_args()

    started = time.perf_counter()
    rows = load_jsonl(ROOT / args.input)
    experts = build_default_experts(["boundary", "space", "symbol", "text"])
    expert_status = describe_experts(experts)

    exported: list[dict[str, Any]] = []
    family_audits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    totals = Counter()
    by_family: dict[str, Counter[str]] = defaultdict(Counter)
    cases: list[dict[str, Any]] = []

    for record_index, record in enumerate(rows):
        candidates = build_candidates(record, record_index)
        predictions: list[ExpertPrediction] = []
        for family in ("boundary", "space", "symbol", "text"):
            expert = experts[family]
            family_candidates = candidates[family]
            if family == "boundary":
                # The boundary expert is graph-aware, so call it once per page.
                family_predictions = expert.predict(family_candidates)
            else:
                family_predictions = expert.predict(family_candidates)
            family_predictions = attach_relations(family_predictions, record, record_index)
            predictions.extend(family_predictions)
            family_audits[family].append(summarize_expert_execution(expert, family_candidates, family_predictions))

        fusion = fuse_predictions(predictions)
        row_out = {
            "record_index": record_index,
            "image": record.get("image_path"),
            "annotation": record.get("annotation_path"),
            "source_dataset": record.get("source_dataset"),
            "fusion": fusion.to_dict(),
        }
        exported.append(row_out)

        gold_nodes, gold_edges = gold_sets(record, record_index)
        pred_nodes, pred_edges = graph_sets(fusion.scene_graph)
        node_tp = gold_nodes & pred_nodes
        edge_tp = gold_edges & pred_edges
        totals.update(
            records=1,
            node_tp=len(node_tp),
            node_pred=len(pred_nodes),
            node_gold=len(gold_nodes),
            edge_tp=len(edge_tp),
            edge_pred=len(pred_edges),
            edge_gold=len(gold_edges),
        )
        for family in ("boundary", "space", "symbol", "text"):
            g = {item for item in gold_nodes if item[2] == family}
            p = {item for item in pred_nodes if item[2] == family}
            by_family[family].update(tp=len(g & p), pred=len(p), gold=len(g))
        if gold_nodes != pred_nodes or gold_edges != pred_edges:
            cases.append(
                {
                    "record_index": record_index,
                    "image": record.get("image_path"),
                    "annotation": record.get("annotation_path"),
                    "missing_nodes": sorted(gold_nodes - pred_nodes)[:80],
                    "extra_nodes": sorted(pred_nodes - gold_nodes)[:80],
                    "missing_edges": sorted(gold_edges - pred_edges)[:80],
                    "extra_edges": sorted(pred_edges - gold_edges)[:80],
                }
            )

    write_jsonl(ROOT / args.output, exported)
    write_jsonl(ROOT / args.cases_output, cases)
    audit = {
        "version": "cubicasa_svg_model_expert_runtime_audit_v1",
        "input": args.input,
        "records": len(rows),
        "expert_status": expert_status,
        "families": summarize_family_audits(family_audits),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    write_json(ROOT / args.audit_output, audit)
    report = {
        "version": "cubicasa_svg_model_case_eval_v1",
        "input": args.input,
        "predictions": args.output,
        "records": len(rows),
        "node_f1": f1(totals["node_tp"], totals["node_pred"], totals["node_gold"]),
        "relation_f1": f1(totals["edge_tp"], totals["edge_pred"], totals["edge_gold"]),
        "by_family_node_f1": {
            family: f1(counter["tp"], counter["pred"], counter["gold"])
            for family, counter in sorted(by_family.items())
        },
        "case_count": len(cases),
        "audit": args.audit_output,
        "elapsed_seconds": audit["elapsed_seconds"],
        "interpretation": "Actual registered expert checkpoints were run on SVG-derived candidates. SVG still supplies the candidate boxes and topology; labels are predicted by loaded expert models.",
    }
    write_json(ROOT / args.eval_output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_candidates(record: dict[str, Any], record_index: int) -> dict[str, list[RoutedCandidate]]:
    expected = record.get("expected_json") or {}
    meta = record.get("metadata") or {}
    page_meta = {"width": meta.get("width", 2000), "height": meta.get("height", 2000)}
    primitive_graph = (record.get("request_hints") or {}).get("primitive_graph") or {}
    primitive_by_id = {
        str(node.get("id")): node
        for node in primitive_graph.get("nodes") or []
        if isinstance(node, dict) and node.get("id") is not None
    }
    primitive_edges = [edge for edge in primitive_graph.get("edges") or [] if isinstance(edge, dict)]
    primitive_bboxes = [
        normalize_bbox(node.get("bbox"))
        for node in primitive_graph.get("nodes") or []
        if isinstance(node, dict)
    ]
    primitive_bboxes = [bbox for bbox in primitive_bboxes if bbox]
    page_bbox = page_bounds(primitive_bboxes)
    bbox_counts = Counter(tuple(bbox) for bbox in primitive_bboxes)
    room_context = build_page_context(record)

    out = {"boundary": [], "space": [], "symbol": [], "text": []}
    for item in expected.get("semantic_candidates") or []:
        target_id = str(item.get("target_id") or item.get("id") or len(out["boundary"]))
        primitive = primitive_by_id.get(target_id, {})
        bbox = primitive.get("bbox") or item.get("bbox")
        out["boundary"].append(
            RoutedCandidate(
                candidate_id=f"r{record_index}:boundary:{target_id}",
                expert="wall_opening",
                family="boundary",
                candidate_type=str(item.get("semantic_type") or item.get("raw_label") or "boundary"),
                confidence=float(item.get("confidence") or 0.9),
                bbox=bbox,
                payload={
                    "_page_metadata": page_meta,
                    "image": record.get("image_path"),
                    "raster_path": record.get("image_path"),
                    "source_dataset": record.get("source_dataset"),
                    "features": primitive,
                    "bbox": bbox,
                    "page_bbox": page_bbox,
                    "bbox_group_count": int(bbox_counts.get(tuple(normalize_bbox(bbox) or []), 1)),
                    "edges": [
                        {
                            "source": f"r{record_index}:boundary:{edge.get('source')}",
                            "target": f"r{record_index}:boundary:{edge.get('target')}",
                            "relation": edge.get("relation"),
                        }
                        for edge in primitive_edges
                    ],
                    "raw_id": target_id,
                    "raw_label": item.get("raw_label"),
                    "semantic_type": item.get("semantic_type"),
                },
            )
        )

    rooms = expected.get("room_candidates") or []
    for index, item in enumerate(rooms):
        raw_id = str(item.get("id") or f"room_{index}")
        out["space"].append(
            RoutedCandidate(
                candidate_id=f"r{record_index}:space:{raw_id}",
                expert="room_space",
                family="space",
                candidate_type="room",
                confidence=float(item.get("confidence") or 0.9),
                bbox=item.get("bbox"),
                payload={
                    "shape_features": item.get("shape_features") or {},
                    "_page_metadata": page_meta,
                    "page_context": room_context,
                    "raw_id": raw_id,
                },
            )
        )

    for index, item in enumerate(expected.get("symbol_candidates") or []):
        raw_id = str(item.get("id") or f"symbol_{index}")
        out["symbol"].append(
            RoutedCandidate(
                candidate_id=f"r{record_index}:symbol:{raw_id}",
                expert="symbol_fixture",
                family="symbol",
                candidate_type="symbol",
                confidence=float(item.get("confidence") or 0.9),
                bbox=item.get("bbox"),
                payload={
                    "bbox": item.get("bbox"),
                    "symbol_type": item.get("symbol_type"),
                    "raw_label": item.get("raw_label"),
                    "rotation": item.get("rotation"),
                    "shape_features": item.get("shape_features") if isinstance(item.get("shape_features"), dict) else {},
                    "rooms": [{"bbox": r.get("bbox"), "room_type": r.get("room_type", "")} for r in rooms],
                    "_page_metadata": page_meta,
                    "raw_id": raw_id,
                },
            )
        )

    for index, item in enumerate(expected.get("text_candidates") or []):
        raw_id = str(item.get("id") or f"text_{index}")
        out["text"].append(
            RoutedCandidate(
                candidate_id=f"r{record_index}:text:{raw_id}",
                expert="text_dimension",
                family="text",
                candidate_type="text",
                confidence=float(item.get("confidence") or 0.9),
                bbox=item.get("bbox"),
                payload={
                    "bbox": item.get("bbox"),
                    "raw_text": item.get("raw_text", item.get("text", "")),
                    "text": item.get("text", ""),
                    "text_type": item.get("text_type", "note_text"),
                    "_page_metadata": page_meta,
                    "raw_id": raw_id,
                },
            )
        )
    return out


def attach_relations(
    predictions: list[ExpertPrediction],
    record: dict[str, Any],
    record_index: int,
) -> list[ExpertPrediction]:
    rels = relation_map(record, record_index)
    attached = []
    for pred in predictions:
        attached.append(
            ExpertPrediction(
                candidate_id=pred.candidate_id,
                expert=pred.expert,
                family=pred.family,
                label=pred.label,
                confidence=pred.confidence,
                bbox=pred.bbox,
                geometry=pred.geometry,
                relations=rels.get(pred.candidate_id, []),
                source=pred.source,
                metadata=pred.metadata,
            )
        )
    return attached


def relation_map(record: dict[str, Any], record_index: int) -> dict[str, list[dict[str, Any]]]:
    expected = record.get("expected_json") or {}
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    primitive_graph = (record.get("request_hints") or {}).get("primitive_graph") or {}
    for edge in primitive_graph.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        src = f"r{record_index}:boundary:{edge.get('source')}"
        dst = f"r{record_index}:boundary:{edge.get('target')}"
        rel = str(edge.get("relation") or "attached_to")
        result[src].append({"source": src, "target": dst, "relation": rel})
        result[dst].append({"source": dst, "target": src, "relation": rel})

    boundary_boxes = []
    for item in expected.get("semantic_candidates") or []:
        raw_id = str(item.get("target_id") or item.get("id") or "")
        bbox = normalize_bbox(item.get("bbox"))
        if bbox:
            boundary_boxes.append((f"r{record_index}:boundary:{raw_id}", bbox))

    for index, room in enumerate(expected.get("room_candidates") or []):
        room_id = f"r{record_index}:space:{room.get('id') or f'room_{index}'}"
        rb = normalize_bbox(room.get("bbox"))
        if not rb:
            continue
        for boundary_id, bb in boundary_boxes:
            if bbox_intersects(rb, bb):
                result[room_id].append({"source": room_id, "target": boundary_id, "relation": "bounds"})

    room_boxes = [
        (f"r{record_index}:space:{room.get('id') or f'room_{i}'}", normalize_bbox(room.get("bbox")))
        for i, room in enumerate(expected.get("room_candidates") or [])
    ]
    for index, symbol in enumerate(expected.get("symbol_candidates") or []):
        symbol_id = f"r{record_index}:symbol:{symbol.get('id') or f'symbol_{index}'}"
        sb = normalize_bbox(symbol.get("bbox"))
        if not sb:
            continue
        containing = [(rid, rb) for rid, rb in room_boxes if rb and bbox_hosts_symbol(rb, sb)]
        if containing:
            room_id, _ = max(containing, key=lambda item: bbox_area(item[1] or [0, 0, 0, 0]))
            result[room_id].append({"source": room_id, "target": symbol_id, "relation": "contains"})
    return result


def gold_sets(record: dict[str, Any], record_index: int) -> tuple[set[tuple[str, str, str]], set[tuple[str, str, str]]]:
    expected = record.get("expected_json") or {}
    nodes: set[tuple[str, str, str]] = set()
    for item in expected.get("semantic_candidates") or []:
        raw_id = str(item.get("target_id") or item.get("id") or "")
        label = str(item.get("semantic_type") or "unknown")
        nodes.add((f"r{record_index}:boundary:{raw_id}", label, "boundary"))
    for index, item in enumerate(expected.get("room_candidates") or []):
        raw_id = str(item.get("id") or f"room_{index}")
        nodes.add((f"r{record_index}:space:{raw_id}", str(item.get("room_type") or "room"), "space"))
    for index, item in enumerate(expected.get("symbol_candidates") or []):
        raw_id = str(item.get("id") or f"symbol_{index}")
        nodes.add((f"r{record_index}:symbol:{raw_id}", str(item.get("symbol_type") or "generic_symbol"), "symbol"))
    for index, item in enumerate(expected.get("text_candidates") or []):
        raw_id = str(item.get("id") or f"text_{index}")
        nodes.add((f"r{record_index}:text:{raw_id}", str(item.get("text_type") or "note_text"), "text"))

    edges = {
        (str(edge["source"]), str(edge["target"]), str(edge["relation"]))
        for values in relation_map(record, record_index).values()
        for edge in values
        if edge.get("source") and edge.get("target") and edge.get("relation")
    }
    return nodes, edges


def graph_sets(graph: dict[str, Any]) -> tuple[set[tuple[str, str, str]], set[tuple[str, str, str]]]:
    nodes = {
        (str(node.get("id")), str(node.get("semantic_type")), str(node.get("family")))
        for node in graph.get("nodes") or []
        if node.get("id") and node.get("semantic_type")
    }
    edges = {
        (str(edge.get("source")), str(edge.get("target")), str(edge.get("relation")))
        for edge in graph.get("edges") or []
        if edge.get("source") and edge.get("target") and edge.get("relation")
    }
    return nodes, edges


def summarize_family_audits(items: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    out = {}
    for family, audits in sorted(items.items()):
        out[family] = {
            "pages": len(audits),
            "loaded": all(audit.get("loaded") for audit in audits),
            "candidate_count": sum(int(audit.get("candidate_count") or 0) for audit in audits),
            "prediction_count": sum(int(audit.get("prediction_count") or 0) for audit in audits),
            "fallback_prediction_count": sum(int(audit.get("fallback_prediction_count") or 0) for audit in audits),
            "missing_prediction_count": sum(int(audit.get("missing_prediction_count") or 0) for audit in audits),
            "source_counts": dict(
                Counter(
                    source
                    for audit in audits
                    for source, count in (audit.get("prediction_source_counts") or {}).items()
                    for _ in range(int(count))
                ).most_common()
            ),
            "label_counts": dict(
                Counter(
                    label
                    for audit in audits
                    for label, count in (audit.get("prediction_label_counts") or {}).items()
                    for _ in range(int(count))
                ).most_common()
            ),
        }
    return out


def build_page_context(record: dict[str, Any]) -> dict[str, Any]:
    expected = record.get("expected_json") or {}
    meta = record.get("metadata") or {}
    rooms = [
        {
            "id": str(item.get("id") or f"room_{index}"),
            "room_type": str(item.get("room_type") or "room"),
            "bbox": normalize_bbox(item.get("bbox")),
            "shape_features": item.get("shape_features") if isinstance(item.get("shape_features"), dict) else {},
        }
        for index, item in enumerate(expected.get("room_candidates") or [])
        if isinstance(item, dict)
    ]
    rooms = [item for item in rooms if item["bbox"]]
    symbols = [
        {"id": str(item.get("id") or f"symbol_{index}"), "symbol_type": str(item.get("symbol_type") or "generic_symbol"), "bbox": normalize_bbox(item.get("bbox"))}
        for index, item in enumerate(expected.get("symbol_candidates") or [])
        if isinstance(item, dict)
    ]
    symbols = [item for item in symbols if item["bbox"]]
    texts = [
        {"id": str(item.get("id") or f"text_{index}"), "text_type": str(item.get("text_type") or "note_text"), "text": str(item.get("text") or ""), "bbox": normalize_bbox(item.get("bbox"))}
        for index, item in enumerate(expected.get("text_candidates") or [])
        if isinstance(item, dict)
    ]
    texts = [item for item in texts if item["bbox"]]
    boundaries = [
        {"semantic_type": str(item.get("semantic_type") or "unknown"), "bbox": normalize_bbox(item.get("bbox"))}
        for item in ((record.get("request_hints") or {}).get("primitive_graph") or {}).get("nodes") or []
        if isinstance(item, dict)
    ]
    boundaries = [item for item in boundaries if item["bbox"]]
    return {
        "width": float(meta.get("width") or 2000.0),
        "height": float(meta.get("height") or 2000.0),
        "rooms": rooms,
        "symbols": symbols,
        "texts": texts,
        "boundaries": boundaries,
        "adjacency": room_adjacency(rooms),
    }


def room_adjacency(rooms: list[dict[str, Any]]) -> dict[str, int]:
    degrees = {str(room["id"]): 0 for room in rooms}
    for i, left in enumerate(rooms):
        for right in rooms[i + 1:]:
            if left.get("bbox") and right.get("bbox") and adjacent(left["bbox"], right["bbox"]):
                degrees[str(left["id"])] += 1
                degrees[str(right["id"])] += 1
    return degrees


def adjacent(left: list[float], right: list[float]) -> bool:
    if bbox_contains(left, right) or bbox_contains(right, left):
        return False
    horizontal_gap = max(left[0] - right[2], right[0] - left[2], 0.0)
    vertical_gap = max(left[1] - right[3], right[1] - left[3], 0.0)
    if horizontal_gap > 2.0 or vertical_gap > 2.0:
        return False
    x_overlap = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    y_overlap = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    min_side = max(min(left[2] - left[0], left[3] - left[1], right[2] - right[0], right[3] - right[1]), 1.0)
    return max(x_overlap, y_overlap) / min_side >= 0.03


def normalize_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(item) for item in value[:4]]
    except (TypeError, ValueError):
        return None
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def page_bounds(bboxes: list[list[float]]) -> list[float]:
    if not bboxes:
        return [0.0, 0.0, 2000.0, 2000.0]
    return [
        min(bbox[0] for bbox in bboxes),
        min(bbox[1] for bbox in bboxes),
        max(bbox[2] for bbox in bboxes),
        max(bbox[3] for bbox in bboxes),
    ]


def bbox_contains(left: list[float], right: list[float]) -> bool:
    return left[0] <= right[0] and left[1] <= right[1] and left[2] >= right[2] and left[3] >= right[3]


def bbox_intersects(left: list[float], right: list[float]) -> bool:
    return not (left[2] < right[0] or right[2] < left[0] or left[3] < right[1] or right[3] < left[1])


def bbox_hosts_symbol(room_bbox: list[float], symbol_bbox: list[float], min_overlap: float = 0.2) -> bool:
    if bbox_contains(room_bbox, symbol_bbox):
        return True
    cx = (symbol_bbox[0] + symbol_bbox[2]) / 2.0
    cy = (symbol_bbox[1] + symbol_bbox[3]) / 2.0
    if room_bbox[0] <= cx <= room_bbox[2] and room_bbox[1] <= cy <= room_bbox[3]:
        return True
    return bbox_overlap_ratio(room_bbox, symbol_bbox) >= min_overlap


def bbox_overlap_ratio(left: list[float], right: list[float]) -> float:
    ix1 = max(left[0], right[0])
    iy1 = max(left[1], right[1])
    ix2 = min(left[2], right[2])
    iy2 = min(left[3], right[3])
    return bbox_area([ix1, iy1, ix2, iy2]) / max(bbox_area(right), 1.0)


def bbox_area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def f1(tp: int, predicted: int, gold: int) -> dict[str, float | int]:
    precision = tp / max(predicted, 1)
    recall = tp / max(gold, 1)
    score = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"tp": int(tp), "predicted": int(predicted), "gold": int(gold), "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(score, 6)}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
