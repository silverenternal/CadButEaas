#!/usr/bin/env python3
"""Audit visual scene-graph defects for CubiCasa5K review packs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional pixel evidence
    Image = None  # type: ignore[assignment]

try:
    from roomspace_geometry import (
        adaptive_margin as room_adaptive_margin,
        best_room_for_label,
        node_bbox as room_node_bbox,
        node_polygon,
        room_contains_label,
    )
except ImportError:  # pragma: no cover
    from scripts.vlm.roomspace_geometry import (
        adaptive_margin as room_adaptive_margin,
        best_room_for_label,
        node_bbox as room_node_bbox,
        node_polygon,
        room_contains_label,
    )


EQUIPMENT_RAW_LABEL_HINTS = {
    "equipment",
    "fireplace",
    "firebox",
    "fireplacecorner",
    "fireplaceround",
    "placeforfireplace",
    "placeforfireplacecorner",
    "heater",
    "highheater",
    "boiler",
    "pipe",
    "chimney",
    "watertap",
    "tap",
    "faucet",
    "heatersign",
}
BOUNDARY_AREA_LABELS = {"hard_wall", "partition_wall"}
BOUNDARY_OPENING_LABELS = {"door", "window", "opening"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="reports/vlm/e2e_cubicasa_visual_demo_predictions.jsonl")
    parser.add_argument("--converted", default="datasets/cadstruct_cubicasa5k_moe_locked/smoke.jsonl")
    parser.add_argument("--render-dir", default="reports/vlm/visual_demo")
    parser.add_argument("--output-json", default="reports/vlm/visual_demo/defect_audit_v2.json")
    parser.add_argument("--output-md", default="reports/vlm/visual_demo/defect_audit_v2.md")
    parser.add_argument("--output-cases", default="reports/vlm/visual_demo/model_defect_cases_v3.jsonl")
    parser.add_argument("--output-summary-v3", default="reports/vlm/visual_demo/model_defect_summary_v3.json")
    parser.add_argument("--output-review-dir", default="reports/vlm/visual_demo/model_defect_review_v3")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--symbol-ink-ratio-threshold", type=float, default=0.006)
    parser.add_argument("--large-boundary-area-ratio", type=float, default=0.08)
    parser.add_argument("--canvas-tolerance", type=float, default=0.5)
    parser.add_argument("--tiny-room-area-threshold", type=float, default=5000.0)
    parser.add_argument("--tiny-room-side-threshold", type=float, default=80.0)
    args = parser.parse_args()

    prediction_rows = load_jsonl(Path(args.predictions))
    if args.limit > 0:
        prediction_rows = prediction_rows[: args.limit]
    converted_by_image = {str(row.get("image_path") or row.get("image")): row for row in load_jsonl(Path(args.converted))}
    summaries_by_image = load_render_summaries(Path(args.render_dir))

    samples = [
        audit_sample(row, converted_by_image, summaries_by_image, args)
        for row in prediction_rows
    ]
    defect_counts: Counter[str] = Counter()
    layer_counts: Counter[str] = Counter()
    symbol_counts: Counter[str] = Counter()
    equipment_raw_label_counts: Counter[str] = Counter()
    needs_review_symbol_raw_label_counts: Counter[str] = Counter()
    source_expert_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    cases: list[dict[str, Any]] = []
    for sample in samples:
        for defect in sample["defects"]:
            defect_counts[str(defect["type"])] += 1
            layer_counts[str(defect["recommended_layer"])] += 1
            source_expert_counts[str(defect.get("source_expert") or "unknown")] += 1
            family_counts[str(defect.get("family") or "unknown")] += 1
            if str(defect.get("type")) == "needs_review_symbol":
                needs_review_symbol_raw_label_counts[str(defect.get("raw_label") or "unknown")] += 1
            cases.append(enrich_case(defect, sample, Path(args.render_dir)))
        for label, count in (sample["metrics"].get("symbol_semantic_counts") or {}).items():
            symbol_counts[str(label)] += int(count)
        for label, count in (sample["metrics"].get("equipment_raw_label_counts") or {}).items():
            equipment_raw_label_counts[str(label)] += int(count)

    report = {
        "version": "cadstruct_moe_visual_defect_audit_v2",
        "inputs": {
            "predictions": str(args.predictions),
            "converted": str(args.converted),
            "render_dir": str(args.render_dir),
        },
        "scope": {
            "records": len(samples),
            "claim_boundary": "Audits the current CubiCasa5K scene-graph visual chain. With source_mode=expected_json, defects are parser/export/fusion/render contract issues unless a real expert-model path is explicitly used.",
        },
        "summary": {
            "defect_counts": dict(defect_counts.most_common()),
            "recommended_layer_counts": dict(layer_counts.most_common()),
            "source_expert_counts": dict(source_expert_counts.most_common()),
            "family_counts": dict(family_counts.most_common()),
            "text_nodes_with_content": sum(item["metrics"]["text_nodes_with_content"] for item in samples),
            "text_nodes_without_content": sum(item["metrics"]["text_nodes_without_content"] for item in samples),
            "symbol_semantic_counts": dict(symbol_counts.most_common()),
            "equipment_raw_label_counts": dict(equipment_raw_label_counts.most_common()),
            "needs_review_symbol_raw_label_counts": dict(needs_review_symbol_raw_label_counts.most_common()),
        },
        "samples": samples,
        "done_when_checks": {
            "ran_on_visual_samples": bool(samples),
            "has_equipment_audit": defect_counts.get("empty_symbol", 0) > 0 or any(sample["metrics"]["equipment_nodes"] for sample in samples),
            "has_boundary_audit": defect_counts.get("unsupported_wall", 0) > 0 or any(sample["metrics"]["boundary_nodes"] for sample in samples),
            "has_room_audit": any(key in defect_counts for key in ["missing_room", "extra_room", "room_without_label", "label_without_room"]),
            "has_text_audit": any(key in defect_counts for key in ["metadata_missing", "missing_visible_text"]),
        },
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.output_md).write_text(render_markdown(report), encoding="utf-8")
    write_jsonl(Path(args.output_cases), sorted(cases, key=case_sort_key))
    summary_v3 = {
        "version": "cadstruct_moe_model_defect_summary_v3",
        "inputs": report["inputs"],
        "records": len(samples),
        "case_count": len(cases),
        "defect_counts": dict(defect_counts.most_common()),
        "recommended_layer_counts": dict(layer_counts.most_common()),
        "source_expert_counts": dict(source_expert_counts.most_common()),
        "family_counts": dict(family_counts.most_common()),
        "top_high_confidence_cases": sorted(cases, key=case_sort_key)[:50],
        "claim_boundary": real_model_claim_boundary(samples),
    }
    summary_path = Path(args.output_summary_v3)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_v3, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    render_review_html(Path(args.output_review_dir), summary_v3)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


def audit_sample(
    row: dict[str, Any],
    converted_by_image: dict[str, dict[str, Any]],
    summaries_by_image: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    image = str(row.get("image") or row.get("image_path") or "")
    converted = converted_by_image.get(image) or {}
    summary = summaries_by_image.get(image) or {}
    graph = get_scene_graph(row)
    nodes = [node for node in graph.get("nodes") or [] if isinstance(node, dict)]
    edges = [edge for edge in graph.get("edges") or [] if isinstance(edge, dict)]
    bbox_by_id = {str(node.get("id")): normalize_bbox((node.get("geometry") or {}).get("bbox") or node.get("bbox")) for node in nodes}
    defects: list[dict[str, Any]] = []

    source_bbox = normalize_bbox(summary.get("source_scene_canvas_bbox") or summary.get("inferred_scene_canvas_bbox"))
    if source_bbox is None:
        source_bbox = metadata_canvas_bbox(converted)
    canvas_area = bbox_area(source_bbox) if source_bbox else 0.0
    room_nodes = [node for node in nodes if str(node.get("family")) == "space"]
    text_nodes = [node for node in nodes if str(node.get("family")) == "text"]
    room_label_nodes = [node for node in text_nodes if str(node.get("semantic_type")) == "room_label"]

    image_ink = load_image_ink_probe(Path(image))
    for node in nodes:
        node_id = str(node.get("id") or "")
        family = str(node.get("family") or "unknown")
        semantic = str(node.get("semantic_type") or "unknown")
        bbox = bbox_by_id.get(node_id)
        common = defect_common(row, node, bbox)
        if bbox is None:
            defects.append({**common, "type": "metadata_missing", "severity": "P1", "reason": "node has no valid bbox", "recommended_layer": "fusion/export"})
            continue
        if source_bbox is not None and bbox_outside_canvas(bbox, source_bbox, tolerance=float(args.canvas_tolerance)):
            defects.append({**common, "type": "bbox_outside_canvas", "severity": "P0", "reason": "node bbox is outside source canvas", "recommended_layer": "parser_renderer"})
        if bbox_area(bbox) <= 1.0 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            defects.append({**common, "type": "tiny_or_degenerate_bbox", "severity": "P1", "reason": "node bbox is degenerate or nearly invisible", "recommended_layer": "parser"})

        metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
        if family == "text" and not str(metadata.get("text") or "").strip() and semantic not in {"dimension_line", "leader_line"}:
            defects.append({**common, "type": "metadata_missing", "severity": "P0", "reason": "text-like node has no readable metadata.text", "recommended_layer": "text_dimension"})
        if family == "symbol" and semantic == "equipment":
            ink_ratio = image_ink(bbox, source_bbox) if image_ink and source_bbox else None
            raw_label = str(metadata.get("raw_label") or "").lower()
            reason_parts = []
            if ink_ratio is not None and ink_ratio < float(args.symbol_ink_ratio_threshold):
                reason_parts.append(f"low dark-pixel ratio {ink_ratio:.5f}")
            if raw_label and raw_label not in EQUIPMENT_RAW_LABEL_HINTS:
                reason_parts.append(f"unexpected raw_label={raw_label}")
            if not containing_room(bbox, room_nodes, symbol_mode=True):
                reason_parts.append("symbol bbox is not contained by any room bbox")
            if reason_parts:
                quality_flags = node.get("quality_flags") if isinstance(node.get("quality_flags"), list) else []
                is_reviewed = "needs_review_outside_room" in quality_flags
                defects.append(
                    {
                        **common,
                        "type": "needs_review_symbol" if is_reviewed else "empty_symbol",
                        "severity": "P1" if is_reviewed else "P0",
                        "reason": "; ".join(reason_parts),
                        "recommended_layer": "parser_and_symbol_expert",
                        "evidence": {"ink_ratio": ink_ratio, "raw_label": metadata.get("raw_label"), "quality_flags": quality_flags},
                    }
                )
        if family == "boundary":
            area_ratio = bbox_area(bbox) / canvas_area if canvas_area else 0.0
            aspect = bbox_aspect_ratio(bbox)
            geometry = node.get("geometry") if isinstance(node.get("geometry"), dict) else {}
            render_geometry = geometry.get("source_geometry") if isinstance(geometry.get("source_geometry"), dict) else geometry
            geometry_type = str(render_geometry.get("type") or geometry.get("type") or "bbox") if isinstance(render_geometry, dict) else str(geometry.get("type") or "bbox")
            render_hint = str(geometry.get("render_hint") or "")
            if geometry_type == "bbox" and (area_ratio > float(args.large_boundary_area_ratio) or (semantic in BOUNDARY_OPENING_LABELS and aspect > 50.0)):
                defects.append(
                    {
                        **common,
                        "type": "unsupported_wall",
                        "severity": "P0",
                        "reason": f"boundary rendered from bbox may be misleading: area_ratio={area_ratio:.4f}, aspect={aspect:.2f}",
                        "recommended_layer": "parser_boundary_renderer",
                        "evidence": {"area_ratio": area_ratio, "aspect_ratio": aspect},
                    }
                )
            elif render_hint == "line_like_boundary_centerline" and semantic in BOUNDARY_OPENING_LABELS and aspect > 50.0:
                # The geometry is still semantically review-worthy, but it is no
                # longer a misleading filled/bbox wall rendering defect.
                pass

    room_label_edges = linked_pairs(edges, {"labels", "contains"})
    labeled_room_ids = {source for source, target in room_label_edges if target in {str(node.get("id")) for node in room_label_nodes}}
    labeled_room_ids.update({target for source, target in room_label_edges if source in {str(node.get("id")) for node in room_label_nodes}})
    label_ids_with_room_edge = {
        source
        for source, target in room_label_edges
        if source in {str(node.get("id")) for node in room_label_nodes}
        and target in {str(node.get("id")) for node in room_nodes}
    }
    label_ids_with_room_edge.update(
        target
        for source, target in room_label_edges
        if target in {str(node.get("id")) for node in room_label_nodes}
        and source in {str(node.get("id")) for node in room_nodes}
    )
    room_label_relation_evidence: list[dict[str, Any]] = []
    for room in room_nodes:
        room_id = str(room.get("id") or "")
        room_bbox = bbox_by_id.get(room_id)
        if room_bbox is None:
            continue
        room_metadata = room.get("metadata") if isinstance(room.get("metadata"), dict) else {}
        if is_suspicious_tiny_room(room, room_bbox, args):
            defects.append(
                {
                    **defect_common(row, room, room_bbox),
                    "type": "extra_room",
                    "severity": "P0",
                    "reason": f"tiny {room_metadata.get('raw_label') or room.get('semantic_type')} space likely comes from a fixture/subpath, not a room region",
                    "recommended_layer": "room_parser_and_room_expert",
                    "evidence": {"area": bbox_area(room_bbox), "max_side": max(room_bbox[2] - room_bbox[0], room_bbox[3] - room_bbox[1])},
                }
            )
            continue
        labels_inside = []
        for text in room_label_nodes:
            relation = room_contains_label(room, text, source_bbox)
            if relation.get("contains"):
                labels_inside.append(text)
                room_label_relation_evidence.append(
                    {
                        "room_id": room_id,
                        "label_id": text.get("id"),
                        "method": relation.get("method"),
                        "distance": relation.get("distance"),
                        "margin": relation.get("margin"),
                    }
                )
        if not labels_inside and room_id not in labeled_room_ids:
            nearest_label = nearest_label_evidence(room, room_label_nodes, source_bbox)
            defects.append(
                {
                    **defect_common(row, room, room_bbox),
                    "type": "room_without_label",
                    "severity": "P1",
                    "reason": "room geometry contains no room_label text center/bbox within adaptive margin",
                    "recommended_layer": "room_parser_and_room_expert",
                    "evidence": {
                        "nearest_label": nearest_label,
                        "geometry_mode": "polygon" if node_polygon(room) else "bbox",
                    },
                }
            )
        if not any(bbox_intersects(room_bbox, other) for node_id, other in bbox_by_id.items() if node_id.startswith("boundary_") and other):
            defects.append({**defect_common(row, room, room_bbox), "type": "missing_room", "severity": "P1", "reason": "room has no intersecting boundary candidate", "recommended_layer": "room_parser_and_room_expert"})
    for text in room_label_nodes:
        text_id = str(text.get("id") or "")
        text_bbox = bbox_by_id.get(text_id)
        if text_bbox:
            best_room, relation = best_room_for_label(text, room_nodes, source_bbox)
            if relation.get("contains") or text_id in label_ids_with_room_edge:
                room_label_relation_evidence.append(
                    {
                        "room_id": best_room.get("id") if best_room else None,
                        "label_id": text_id,
                        "method": relation.get("method") if relation.get("contains") else "explicit_labels_edge",
                        "distance": relation.get("distance"),
                        "margin": relation.get("margin"),
                    }
                )
            else:
                defects.append(
                    {
                        **defect_common(row, text, text_bbox),
                        "type": "label_without_room",
                        "severity": "P1",
                        "reason": "room_label text center is outside all room geometries after adaptive margin",
                        "recommended_layer": "room_parser_and_room_expert",
                        "evidence": {
                            "nearest_room": {
                                "node_id": best_room.get("id") if best_room else None,
                                "bbox": room_node_bbox(best_room) if best_room else None,
                                "method": relation.get("method"),
                                "distance": relation.get("distance"),
                                "margin": relation.get("margin"),
                            },
                            "geometry_mode": "polygon" if best_room and node_polygon(best_room) else "bbox",
                        },
                    }
                )

    expected_texts = ((converted.get("expected_json") or {}).get("text_candidates") or [])
    scene_text_ids = {str(node.get("id") or "") for node in text_nodes}
    suppressed_missing_text_counts: Counter[str] = Counter()
    for item in expected_texts:
        if str(item.get("id") or "") not in scene_text_ids:
            item_bbox = normalize_bbox(item.get("bbox"))
            item_text = str(item.get("text") or "").strip()
            if not item_text:
                suppressed_missing_text_counts["empty_text_candidate"] += 1
                continue
            if source_bbox is not None and item_bbox is not None and bbox_outside_area_ratio(item_bbox, source_bbox) >= 0.15:
                suppressed_missing_text_counts["outside_source_canvas"] += 1
                continue
            defects.append(
                {
                    "sample_id": sample_id(row),
                    "node_id": str(item.get("id") or ""),
                    "family": "text",
                    "semantic_type": str(item.get("text_type") or "text"),
                    "bbox": item_bbox,
                    "source_expert": str(item.get("source") or ""),
                    "source_mode": source_mode(row),
                    "raw_label": item.get("raw_label") or item.get("text_type"),
                    "confidence": item.get("confidence"),
                    "type": "missing_visible_text",
                    "severity": "P0",
                    "reason": "converted text candidate is absent from scene graph",
                    "recommended_layer": "fusion/export",
                    "evidence": {"text": item.get("text")},
                }
            )

    text_with_content = sum(1 for node in text_nodes if str(((node.get("metadata") or {}).get("text") if isinstance(node.get("metadata"), dict) else "") or "").strip())
    metrics = {
        "nodes": len(nodes),
        "edges": len(edges),
        "boundary_nodes": sum(1 for node in nodes if str(node.get("family")) == "boundary"),
        "room_nodes": len(room_nodes),
        "symbol_nodes": sum(1 for node in nodes if str(node.get("family")) == "symbol"),
        "equipment_nodes": sum(1 for node in nodes if str(node.get("semantic_type")) == "equipment"),
        "symbol_semantic_counts": dict(Counter(str(node.get("semantic_type") or "unknown") for node in nodes if str(node.get("family")) == "symbol").most_common()),
        "equipment_raw_label_counts": dict(
            Counter(
                str(((node.get("metadata") or {}).get("raw_label") if isinstance(node.get("metadata"), dict) else None) or "unknown")
                for node in nodes
                if str(node.get("family")) == "symbol" and str(node.get("semantic_type")) == "equipment"
            ).most_common()
        ),
        "text_nodes": len(text_nodes),
        "text_nodes_with_content": text_with_content,
        "text_nodes_without_content": len(text_nodes) - text_with_content,
        "suppressed_missing_text_counts": dict(suppressed_missing_text_counts.most_common()),
        "room_nodes_with_polygon": sum(1 for room in room_nodes if node_polygon(room)),
        "room_label_relation_methods": dict(Counter(str(item.get("method") or "unknown") for item in room_label_relation_evidence).most_common()),
    }
    return {
        "sample_id": sample_id(row),
        "image": image,
        "annotation": row.get("annotation"),
        "source_mode": source_mode(row),
        "metrics": metrics,
        "defect_counts": dict(Counter(str(item["type"]) for item in defects).most_common()),
        "defects": defects[:200],
    }


def load_image_ink_probe(path: Path):
    if Image is None or not path.exists():
        return None
    try:
        image = Image.open(path).convert("L")
    except Exception:  # pragma: no cover - corrupt or unsupported local image
        return None
    width, height = image.size
    pixels = image.load()

    def probe(scene_bbox: list[float], source_bbox: list[float] | None) -> float:
        if source_bbox is None:
            return 0.0
        x1 = int(round((scene_bbox[0] - source_bbox[0]) / max(source_bbox[2] - source_bbox[0], 1.0) * width))
        y1 = int(round((scene_bbox[1] - source_bbox[1]) / max(source_bbox[3] - source_bbox[1], 1.0) * height))
        x2 = int(round((scene_bbox[2] - source_bbox[0]) / max(source_bbox[2] - source_bbox[0], 1.0) * width))
        y2 = int(round((scene_bbox[3] - source_bbox[1]) / max(source_bbox[3] - source_bbox[1], 1.0) * height))
        x1, x2 = sorted((max(0, min(width, x1)), max(0, min(width, x2))))
        y1, y2 = sorted((max(0, min(height, y1)), max(0, min(height, y2))))
        total = max(0, x2 - x1) * max(0, y2 - y1)
        if total == 0:
            return 0.0
        dark = 0
        for y in range(y1, y2):
            for x in range(x1, x2):
                if pixels[x, y] < 220:
                    dark += 1
        return dark / total

    return probe


def load_render_summaries(render_dir: Path) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    if not render_dir.exists():
        return summaries
    for path in render_dir.glob("*/summary.json"):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        image = str(item.get("image") or "")
        if image:
            summaries[image] = item
    return summaries


def get_scene_graph(row: dict[str, Any]) -> dict[str, Any]:
    if isinstance(row.get("scene_graph"), dict):
        return row["scene_graph"]
    fusion = row.get("fusion") if isinstance(row.get("fusion"), dict) else {}
    graph = fusion.get("scene_graph") if isinstance(fusion.get("scene_graph"), dict) else {}
    return graph


def defect_common(row: dict[str, Any], node: dict[str, Any], bbox: list[float] | None) -> dict[str, Any]:
    metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
    upstream_metadata = metadata.get("upstream_metadata") if isinstance(metadata.get("upstream_metadata"), dict) else {}
    return {
        "sample_id": sample_id(row),
        "node_id": str(node.get("id") or ""),
        "family": str(node.get("family") or "unknown"),
        "semantic_type": str(node.get("semantic_type") or "unknown"),
        "bbox": bbox,
        "source_expert": str(node.get("source_expert") or node.get("expert") or ""),
        "source_mode": source_mode(row),
        "raw_label": metadata.get("raw_label"),
        "base_raw_label": metadata.get("base_raw_label"),
        "base_label": upstream_metadata.get("base_label"),
        "model_source": metadata.get("model_source"),
        "proposal_source": metadata.get("proposal_source"),
        "confidence": node.get("confidence"),
    }


def enrich_case(defect: dict[str, Any], sample: dict[str, Any], render_dir: Path) -> dict[str, Any]:
    sample_dir = render_dir / str(sample.get("sample_id") or "")
    case = dict(defect)
    case["image"] = sample.get("image")
    case["annotation"] = sample.get("annotation")
    case["recommended_layer"] = normalize_recommended_layer(case)
    case["render_assets"] = {
        "review_html": str(render_dir / "review_pack_v2" / "index.html"),
        "sample_summary": str(sample_dir / "summary.json"),
        "input_reference": str(sample_dir / "input_reference.png"),
        "input_aligned_from_svg": str(sample_dir / "input_aligned_from_svg.png"),
        "overlay": str(sample_dir / "overlay.png"),
    }
    case["review_priority"] = review_priority(case)
    return case


def normalize_recommended_layer(case: dict[str, Any]) -> str:
    defect_type = str(case.get("type") or "")
    family = str(case.get("family") or "")
    if defect_type in {"bbox_outside_canvas", "tiny_or_degenerate_bbox"}:
        return "candidate_parser"
    if family == "text" or defect_type in {"metadata_missing", "missing_visible_text"}:
        return "text_dimension"
    if family == "symbol" or defect_type in {"empty_symbol", "needs_review_symbol"}:
        return "symbol_fixture"
    if family == "boundary" or defect_type == "unsupported_wall":
        return "wall_opening"
    if family == "space" or defect_type in {"extra_room", "room_without_label", "label_without_room", "missing_room"}:
        return "room_space"
    return str(case.get("recommended_layer") or "fusion_postprocess")


def review_priority(case: dict[str, Any]) -> float:
    severity_weight = {"P0": 2.0, "P1": 1.0}.get(str(case.get("severity") or ""), 0.5)
    try:
        confidence = float(case.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return round(severity_weight + confidence, 6)


def case_sort_key(case: dict[str, Any]) -> tuple[float, str, str]:
    return (-float(case.get("review_priority") or 0.0), str(case.get("sample_id") or ""), str(case.get("node_id") or ""))


def real_model_claim_boundary(samples: list[dict[str, Any]]) -> str:
    modes = {str(sample.get("source_mode") or "") for sample in samples}
    if "expected_json" in modes:
        return "Contains expected_json/oracle-smoke rows; do not report as true model recognition."
    if "real_upstream_saved_model_predictions" in modes:
        return "Saved expert models classify parser/SVG candidate geometry; this is not pure raster end-to-end detection."
    return "Inspect source_mode before using this audit in claims."


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def render_review_html(output_dir: Path, summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in summary.get("top_high_confidence_cases") or []:
        assets = case.get("render_assets") if isinstance(case.get("render_assets"), dict) else {}
        rows.append(
            "<tr>"
            f"<td>{html_escape(case.get('review_priority'))}</td>"
            f"<td>{html_escape(case.get('sample_id'))}</td>"
            f"<td>{html_escape(case.get('node_id'))}</td>"
            f"<td>{html_escape(case.get('type'))}</td>"
            f"<td>{html_escape(case.get('family'))}/{html_escape(case.get('semantic_type'))}</td>"
            f"<td>{html_escape(case.get('confidence'))}</td>"
            f"<td>{html_escape(case.get('source_expert'))}</td>"
            f"<td>{html_escape(case.get('recommended_layer'))}</td>"
            f"<td>{html_escape(case.get('reason'))}</td>"
            f"<td><a href=\"{html_attr(relpath(assets.get('sample_summary'), output_dir))}\">summary</a> "
            f"<a href=\"{html_attr(relpath(assets.get('overlay'), output_dir))}\">overlay</a></td>"
            "</tr>"
        )
    html = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>CadStruct Model Defect Review v3</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #d8dee6; padding: 6px 8px; vertical-align: top; }}
    th {{ background: #eef3f8; text-align: left; }}
    code {{ background: #f3f5f7; padding: 1px 4px; }}
  </style>
</head>
<body>
  <h1>CadStruct Model Defect Review v3</h1>
  <p><code>{claim_boundary}</code></p>
  <p>Cases: {case_count}; defects: <code>{defect_counts}</code></p>
  <table>
    <thead><tr><th>Priority</th><th>Sample</th><th>Node</th><th>Type</th><th>Label</th><th>Conf</th><th>Expert</th><th>Layer</th><th>Reason</th><th>Assets</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>
""".format(
        claim_boundary=html_escape(summary.get("claim_boundary")),
        case_count=html_escape(summary.get("case_count")),
        defect_counts=html_escape(summary.get("defect_counts")),
        rows="\n".join(rows),
    )
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def relpath(path_value: Any, base: Path) -> str:
    if not path_value:
        return ""
    path = Path(str(path_value))
    try:
        return str(path.relative_to(base))
    except ValueError:
        try:
            return str(path.resolve().relative_to(base.resolve()))
        except ValueError:
            return str(path)


def html_escape(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def html_attr(value: Any) -> str:
    return html_escape(value).replace(" ", "%20")


def linked_pairs(edges: list[dict[str, Any]], relations: set[str]) -> set[tuple[str, str]]:
    return {
        (str(edge.get("source") or ""), str(edge.get("target") or ""))
        for edge in edges
        if str(edge.get("relation") or "") in relations
    }


def containing_room(bbox: list[float], rooms: list[dict[str, Any]], tolerance: float = 0.0, symbol_mode: bool = False) -> bool:
    for room in rooms:
        room_bbox = normalize_bbox((room.get("geometry") or {}).get("bbox") or room.get("bbox"))
        if not room_bbox:
            continue
        padded = pad_bbox(room_bbox, tolerance)
        if symbol_mode:
            if bbox_hosts_symbol(padded, bbox):
                return True
        elif bbox_contains(padded, bbox):
            return True
    return False


def nearest_label_evidence(room: dict[str, Any], labels: list[dict[str, Any]], canvas_bbox: list[float] | None) -> dict[str, Any] | None:
    room_bbox = room_node_bbox(room)
    if room_bbox is None:
        return None
    best: dict[str, Any] | None = None
    for label in labels:
        relation = room_contains_label(room, label, canvas_bbox)
        distance = relation.get("distance")
        if distance is None:
            label_bbox = room_node_bbox(label)
            distance = bbox_distance(room_bbox, label_bbox) if label_bbox else None
        item = {
            "node_id": label.get("id"),
            "bbox": room_node_bbox(label),
            "method": relation.get("method"),
            "contains": relation.get("contains"),
            "distance": distance,
            "margin": relation.get("margin") if relation.get("margin") is not None else room_adaptive_margin(room_bbox, canvas_bbox),
            "text": ((label.get("metadata") or {}).get("text") if isinstance(label.get("metadata"), dict) else None),
        }
        if best is None or float(item.get("distance") if item.get("distance") is not None else 1e9) < float(best.get("distance") if best.get("distance") is not None else 1e9):
            best = item
    return best


def pad_bbox(bbox: list[float], padding: float) -> list[float]:
    return [bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding]


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
    intersection = bbox_area([ix1, iy1, ix2, iy2])
    return intersection / max(bbox_area(right), 1.0)


def is_suspicious_tiny_room(room: dict[str, Any], bbox: list[float], args: argparse.Namespace) -> bool:
    semantic = str(room.get("semantic_type") or "").lower()
    metadata = room.get("metadata") if isinstance(room.get("metadata"), dict) else {}
    raw_label = str(metadata.get("raw_label") or "").lower()
    if semantic not in {"toilet", "bathroom", "unknown_room"} and raw_label not in {"toilet", "toiletseat"}:
        return False
    width = max(0.0, bbox[2] - bbox[0])
    height = max(0.0, bbox[3] - bbox[1])
    return bbox_area(bbox) < float(args.tiny_room_area_threshold) and max(width, height) < float(args.tiny_room_side_threshold)


def sample_id(row: dict[str, Any]) -> str:
    image = Path(str(row.get("image") or row.get("image_path") or "sample"))
    return image.parent.name or image.stem


def source_mode(row: dict[str, Any]) -> str:
    return str((row.get("route_trace") or {}).get("source_mode") or row.get("source") or "unknown")


def metadata_canvas_bbox(row: dict[str, Any]) -> list[float] | None:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    width = metadata.get("width")
    height = metadata.get("height")
    try:
        return [0.0, 0.0, float(width), float(height)]
    except (TypeError, ValueError):
        return None


def normalize_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def bbox_area(bbox: list[float] | None) -> float:
    if bbox is None:
        return 0.0
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def bbox_outside_area_ratio(bbox: list[float], canvas: list[float]) -> float:
    intersection = [max(bbox[0], canvas[0]), max(bbox[1], canvas[1]), min(bbox[2], canvas[2]), min(bbox[3], canvas[3])]
    outside = bbox_area(bbox) - bbox_area(intersection)
    return max(0.0, outside) / max(bbox_area(bbox), 1.0)


def bbox_aspect_ratio(bbox: list[float]) -> float:
    width = max(0.0, bbox[2] - bbox[0])
    height = max(0.0, bbox[3] - bbox[1])
    if min(width, height) <= 0.0:
        return float("inf")
    return max(width, height) / min(width, height)


def bbox_contains(left: list[float], right: list[float]) -> bool:
    return left[0] <= right[0] and left[1] <= right[1] and left[2] >= right[2] and left[3] >= right[3]


def bbox_intersects(left: list[float], right: list[float]) -> bool:
    return not (left[2] < right[0] or right[2] < left[0] or left[3] < right[1] or right[3] < left[1])


def bbox_outside_canvas(bbox: list[float], canvas: list[float], tolerance: float = 0.0) -> bool:
    return (
        bbox[0] < canvas[0] - tolerance
        or bbox[1] < canvas[1] - tolerance
        or bbox[2] > canvas[2] + tolerance
        or bbox[3] > canvas[3] + tolerance
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# CadStruct-MoE Visual Defect Audit v2",
        "",
        f"Predictions: `{report['inputs']['predictions']}`",
        f"Converted records: `{report['inputs']['converted']}`",
        "",
        "## Summary",
        "",
        f"- Samples: {report['scope']['records']}",
        f"- Defect counts: {report['summary']['defect_counts']}",
        f"- Recommended layers: {report['summary']['recommended_layer_counts']}",
        f"- Text nodes with content: {report['summary']['text_nodes_with_content']}",
        f"- Text nodes without content: {report['summary']['text_nodes_without_content']}",
        "",
        "## Samples",
        "",
    ]
    for sample in report["samples"]:
        lines.extend(
            [
                f"### {sample['sample_id']}",
                "",
                f"- Source mode: `{sample['source_mode']}`",
                f"- Metrics: {sample['metrics']}",
                f"- Defects: {sample['defect_counts']}",
                "",
            ]
        )
        for defect in sample["defects"][:12]:
            lines.append(
                f"- `{defect['type']}` `{defect['node_id']}` {defect['family']}/{defect['semantic_type']}: {defect['reason']} -> {defect['recommended_layer']}"
            )
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
