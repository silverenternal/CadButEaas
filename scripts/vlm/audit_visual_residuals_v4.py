#!/usr/bin/env python3
"""Build a per-case residual ledger for the CubiCasa visual demo v4 chain."""

from __future__ import annotations

import argparse
import json
import math
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="reports/vlm/visual_demo/model_defect_cases_roomspace_v4.jsonl")
    parser.add_argument("--predictions", default="reports/vlm/real_upstream_model_postprocessed_predictions_roomlink_v3.jsonl")
    parser.add_argument("--converted", default="datasets/cadstruct_real_world_benchmark_v1/room_space/cubicasa5k_reviewed_locked_test.jsonl")
    parser.add_argument("--output-json", default="reports/vlm/visual_residual_ledger_v4.json")
    parser.add_argument("--output-md", default="reports/vlm/visual_residual_ledger_v4.md")
    args = parser.parse_args()

    cases = load_jsonl(Path(args.cases))
    predictions = load_jsonl(Path(args.predictions))
    converted = load_jsonl(Path(args.converted))
    prediction_by_sample = {sample_id(row): row for row in predictions}
    converted_by_sample = {sample_id(row): row for row in converted}
    svg_cache: dict[str, list[dict[str, Any]]] = {}

    ledger: list[dict[str, Any]] = []
    for case in cases:
        sid = str(case.get("sample_id") or sample_id(case))
        pred = prediction_by_sample.get(sid, {})
        conv = converted_by_sample.get(sid, {})
        node_id = str(case.get("node_id") or "")
        defect_type = str(case.get("type") or "")
        node = find_node(pred, node_id)
        converted_candidate = find_converted_candidate(conv, case)
        annotation = str(case.get("annotation") or pred.get("annotation") or conv.get("annotation_path") or "")
        svg_probe = probe_svg(annotation, case, converted_candidate, svg_cache)
        quality_gate = node.get("quality_gate") if isinstance(node, dict) and isinstance(node.get("quality_gate"), dict) else {}
        attribution = attribute_case(case, node, converted_candidate, svg_probe)
        ledger.append(
            {
                "sample_id": sid,
                "node_id": node_id,
                "defect_type": defect_type,
                "family": case.get("family"),
                "semantic_type": case.get("semantic_type"),
                "bbox": case.get("bbox"),
                "raw_label": case.get("raw_label"),
                "base_raw_label": case.get("base_raw_label") or metadata_value(node, "base_raw_label"),
                "model_label": metadata_value(node, "model_label") or case.get("semantic_type"),
                "confidence": case.get("confidence"),
                "quality_gate": {
                    "decision": quality_gate.get("decision"),
                    "reasons": quality_gate.get("reasons") or [],
                    "evidence": quality_gate.get("evidence") or {},
                },
                "converted_candidate": compact_candidate(converted_candidate),
                "svg_probe": svg_probe,
                "model_probabilities": model_probabilities(node),
                "attribution": attribution,
                "recommended_fix": recommended_fix(attribution),
                "source_mode": case.get("source_mode"),
                "image": case.get("image") or pred.get("image") or conv.get("image_path"),
                "annotation": annotation,
            }
        )

    summary = summarize(ledger)
    report = {
        "version": "cadstruct_moe_visual_residual_ledger_v4",
        "inputs": {
            "cases": str(args.cases),
            "predictions": str(args.predictions),
            "converted": str(args.converted),
        },
        "claim_boundary": "This ledger attributes visual-demo residuals over saved real upstream expert labels and CubiCasa SVG/parser candidate geometry. It is not an end-to-end raster detector evaluation.",
        "summary": summary,
        "cases": ledger,
    }
    write_json(Path(args.output_json), report)
    Path(args.output_md).write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def attribute_case(case: dict[str, Any], node: dict[str, Any], candidate: dict[str, Any], svg_probe: dict[str, Any]) -> dict[str, Any]:
    defect_type = str(case.get("type") or "")
    evidence_text = str(((case.get("evidence") or {}).get("text") if isinstance(case.get("evidence"), dict) else "") or "")
    if defect_type == "missing_visible_text":
        if not evidence_text.strip() and not str(candidate.get("text") or "").strip():
            if svg_probe.get("nearest_visible_text"):
                return {
                    "primary": "candidate_extraction",
                    "secondary": "text_metadata_recovery",
                    "reason": "Converted text candidate has empty text, but nearby SVG text exists; recover from SVG text/tspan source.",
                    "fix_type": "parser",
                    "retrain_needed": False,
                }
            return {
                "primary": "audit_contract",
                "secondary": "candidate_extraction_empty_text_artifact",
                "reason": "Converted candidate is absent from the scene graph because it has no readable text; current audit counted an empty dimension-text artifact as missing visible text.",
                "fix_type": "parser_or_audit_suppression",
                "retrain_needed": False,
            }
        return {
            "primary": "postprocess_gate",
            "secondary": "text_without_readable_content_drop",
            "reason": "Readable converted text is missing from scene graph after gating/export.",
            "fix_type": "quality_gate_or_export",
            "retrain_needed": False,
        }

    if defect_type == "unsupported_wall":
        aspect = float(((case.get("evidence") or {}).get("aspect_ratio") if isinstance(case.get("evidence"), dict) else 0.0) or 0.0)
        semantic = str(case.get("semantic_type") or "")
        raw = str(case.get("raw_label") or "").lower()
        geometry_type = str(((node.get("geometry") or {}).get("type") if isinstance(node.get("geometry"), dict) else "") or "")
        if geometry_type in {"line", "polyline", "polygon"}:
            return {
                "primary": "audit_contract",
                "secondary": "geometry_supported_but_audit_flagged",
                "reason": f"Boundary node has explicit {geometry_type} geometry; audit should not treat it as bbox-only.",
                "fix_type": "audit",
                "retrain_needed": False,
            }
        if aspect > 50:
            return {
                "primary": "renderer_geometry",
                "secondary": "line_like_bbox_without_source_geometry",
                "reason": f"Boundary candidate is line-like bbox-only (aspect={aspect:.2f}); renderer should draw centerline/review styling instead of a wall-like rectangle.",
                "fix_type": "renderer_postprocess",
                "retrain_needed": semantic == "door" and raw in {"hard_wall", "window"},
            }
        return {
            "primary": "model_classification",
            "secondary": "boundary_label_conflict",
            "reason": "Boundary semantic/raw label conflict remains after geometry checks.",
            "fix_type": "boundary_recalibration",
            "retrain_needed": True,
        }

    if defect_type == "empty_symbol":
        probs = model_probabilities(node)
        equipment_prob = probs.get("equipment")
        appliance_prob = probs.get("appliance")
        raw = str(case.get("raw_label") or metadata_value(node, "raw_label") or "").lower()
        if raw == "appliance" and equipment_prob is not None and appliance_prob is not None:
            margin = float(equipment_prob) - float(appliance_prob)
            return {
                "primary": "model_classification",
                "secondary": "low_margin_raw_label_conflict",
                "reason": f"Raw label is appliance but model chose equipment with low margin {margin:.3f}; apply raw-label-aware threshold or retrain appliance/equipment hard cases.",
                "fix_type": "symbol_threshold_or_retrain",
                "retrain_needed": margin > 0.25,
            }
        return {
            "primary": "model_classification",
            "secondary": "symbol_semantic_confusion",
            "reason": "Symbol class is semantically implausible for its raw label.",
            "fix_type": "symbol_threshold_or_retrain",
            "retrain_needed": True,
        }

    return {
        "primary": "unknown",
        "secondary": "unclassified",
        "reason": "Residual type is not covered by v4 ledger rules.",
        "fix_type": "manual_review",
        "retrain_needed": False,
    }


def recommended_fix(attribution: dict[str, Any]) -> str:
    fix_type = str(attribution.get("fix_type") or "")
    if fix_type == "parser_or_audit_suppression":
        return "Suppress empty non-readable text artifacts in visual audit; keep them out of recognition-success claims."
    if fix_type == "parser":
        return "Recover text from SVG text/tspan metadata before quality gating."
    if fix_type == "renderer_postprocess":
        return "Render line-like boundary bboxes as centerlines with review styling and preserve source geometry when available."
    if fix_type == "symbol_threshold_or_retrain":
        return "Apply raw-label-aware appliance/equipment arbitration first; retrain only if locked split still shows confusion."
    if fix_type == "boundary_recalibration":
        return "Add boundary hard cases and run renderer-only vs model-v4 ablation."
    return "Manual review."


def find_converted_candidate(record: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    expected = record.get("expected_json") if isinstance(record.get("expected_json"), dict) else {}
    node_id = str(case.get("node_id") or "")
    family = str(case.get("family") or "")
    pools: list[tuple[str, str]] = []
    if family == "text":
        pools.append(("text_candidates", "id"))
    elif family == "symbol":
        pools.append(("symbol_candidates", "id"))
    elif family == "boundary":
        target = node_id.replace("boundary_", "", 1)
        for item in expected.get("semantic_candidates") or []:
            if str(item.get("target_id")) == target:
                return dict(item)
        graph = ((record.get("request_hints") or {}).get("primitive_graph") or {})
        for item in graph.get("nodes") or []:
            if str(item.get("id")) == target:
                return dict(item)
    for key, id_key in pools:
        for item in expected.get(key) or []:
            if str(item.get(id_key)) == node_id:
                return dict(item)
    return {}


def compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    if not candidate:
        return {}
    keys = ["id", "target_id", "source_id", "text_type", "semantic_type", "symbol_type", "bbox", "text", "raw_label", "tag", "geometry", "shape_features"]
    return {key: candidate.get(key) for key in keys if key in candidate}


def probe_svg(annotation: str, case: dict[str, Any], candidate: dict[str, Any], cache: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    path = Path(annotation)
    if not annotation or not path.exists():
        return {"available": False, "reason": "missing_annotation"}
    if annotation not in cache:
        cache[annotation] = collect_svg_elements(path)
    bbox = normalize_bbox(case.get("bbox") or candidate.get("bbox"))
    nearest = nearest_svg_elements(cache[annotation], bbox, limit=5) if bbox else []
    nearest_visible_text = next((item for item in nearest if item.get("text") and not item.get("hidden")), None)
    nearest_hidden_text = next((item for item in nearest if item.get("text") and item.get("hidden")), None)
    return {
        "available": True,
        "nearest": nearest[:3],
        "nearest_visible_text": trim_svg_item(nearest_visible_text),
        "nearest_hidden_text": trim_svg_item(nearest_hidden_text),
        "has_readable_nearby_text": bool(nearest_visible_text),
    }


def collect_svg_elements(path: Path) -> list[dict[str, Any]]:
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return []
    items: list[dict[str, Any]] = []
    index = 0

    def visit(element: ET.Element, hidden_parent: bool) -> None:
        nonlocal index
        tag = local_name(element.tag)
        attrib = element.attrib
        hidden = hidden_parent or "display: none" in str(attrib.get("style") or "") or str(attrib.get("display") or "") == "none"
        text = "".join(element.itertext()).strip() if tag == "text" else ""
        bbox = rough_element_bbox(tag, attrib)
        classes = str(attrib.get("class") or "")
        if tag in {"text", "polygon", "rect", "line", "path", "polyline"}:
            items.append(
                {
                    "ordinal": index,
                    "tag": tag,
                    "id": attrib.get("id"),
                    "class": classes,
                    "hidden": hidden,
                    "text": text,
                    "bbox": bbox,
                }
            )
            index += 1
        for child in list(element):
            visit(child, hidden)

    visit(root, False)
    return items


def rough_element_bbox(tag: str, attrib: dict[str, str]) -> list[float] | None:
    if tag == "line":
        vals = [num(attrib.get(key)) for key in ["x1", "y1", "x2", "y2"]]
        if all(v is not None for v in vals):
            xs = [float(vals[0]), float(vals[2])]
            ys = [float(vals[1]), float(vals[3])]
            return [min(xs), min(ys), max(xs), max(ys)]
    if tag in {"polygon", "polyline"}:
        values = [float(item) for item in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", attrib.get("points") or "")]
        if len(values) >= 4:
            xs = values[0::2]
            ys = values[1::2]
            return [min(xs), min(ys), max(xs), max(ys)]
    if tag == "rect":
        x, y, w, h = (num(attrib.get(key)) for key in ["x", "y", "width", "height"])
        if None not in {x, y, w, h}:
            return [float(x), float(y), float(x) + float(w), float(y) + float(h)]
    if tag == "text":
        x, y = num(attrib.get("x")), num(attrib.get("y"))
        if x is not None and y is not None:
            return [float(x), float(y) - 12.0, float(x) + 10.0, float(y) + 2.0]
    return None


def nearest_svg_elements(items: list[dict[str, Any]], bbox: list[float], limit: int) -> list[dict[str, Any]]:
    scored = []
    for item in items:
        ibox = normalize_bbox(item.get("bbox"))
        if ibox is None:
            continue
        scored.append((bbox_center_distance(bbox, ibox), item))
    return [trim_svg_item(item) for _distance, item in sorted(scored, key=lambda pair: pair[0])[:limit]]


def trim_svg_item(item: dict[str, Any] | None) -> dict[str, Any] | None:
    if not item:
        return None
    return {key: item.get(key) for key in ["ordinal", "tag", "id", "class", "hidden", "text", "bbox"] if item.get(key) not in (None, "")}


def model_probabilities(node: dict[str, Any]) -> dict[str, float]:
    metadata = node.get("metadata") if isinstance(node, dict) and isinstance(node.get("metadata"), dict) else {}
    upstream = metadata.get("upstream_metadata") if isinstance(metadata.get("upstream_metadata"), dict) else {}
    for key in ["symbol_long_tail_model_v1_probs", "arbitration_v2_probs", "model_probs", "probabilities"]:
        probs = upstream.get(key) or metadata.get(key)
        if isinstance(probs, dict):
            return {str(label): float(value) for label, value in probs.items() if is_number(value)}
    return {}


def summarize(ledger: list[dict[str, Any]]) -> dict[str, Any]:
    defect_counts = Counter(str(item.get("defect_type")) for item in ledger)
    attribution_counts = Counter(str((item.get("attribution") or {}).get("primary")) for item in ledger)
    fix_counts = Counter(str((item.get("attribution") or {}).get("fix_type")) for item in ledger)
    retrain_counts = Counter("retrain_candidate" if (item.get("attribution") or {}).get("retrain_needed") else "no_retrain_first" for item in ledger)
    by_sample: dict[str, Counter[str]] = defaultdict(Counter)
    for item in ledger:
        by_sample[str(item.get("sample_id"))][str(item.get("defect_type"))] += 1
    return {
        "case_count": len(ledger),
        "defect_counts": dict(defect_counts.most_common()),
        "attribution_counts": dict(attribution_counts.most_common()),
        "fix_type_counts": dict(fix_counts.most_common()),
        "retrain_counts": dict(retrain_counts.most_common()),
        "by_sample": {sample: dict(counts.most_common()) for sample, counts in sorted(by_sample.items())},
        "done_when": {
            "all_38_cases_attributed": len(ledger) == 38 and "unknown" not in attribution_counts,
            "has_text_boundary_symbol": all(defect_counts.get(key, 0) > 0 for key in ["missing_visible_text", "unsupported_wall", "empty_symbol"]),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Visual Residual Ledger v4",
        "",
        f"- Cases: {summary['case_count']}",
        f"- Defects: `{summary['defect_counts']}`",
        f"- Attribution: `{summary['attribution_counts']}`",
        f"- Fix types: `{summary['fix_type_counts']}`",
        f"- Retrain flags: `{summary['retrain_counts']}`",
        "",
        "## Cases",
        "",
        "| sample | node | defect | attribution | fix | retrain | reason |",
        "|---|---:|---|---|---|---|---|",
    ]
    for item in report["cases"]:
        attr = item["attribution"]
        reason = str(attr.get("reason") or "").replace("|", "/")
        lines.append(
            f"| {item['sample_id']} | `{item['node_id']}` | {item['defect_type']} | {attr.get('primary')} / {attr.get('secondary')} | {attr.get('fix_type')} | {bool(attr.get('retrain_needed'))} | {reason} |"
        )
    return "\n".join(lines) + "\n"


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


def find_node(row: dict[str, Any], node_id: str) -> dict[str, Any]:
    graph = row.get("scene_graph") if isinstance(row.get("scene_graph"), dict) else {}
    for node in graph.get("nodes") or []:
        if isinstance(node, dict) and str(node.get("id")) == node_id:
            return node
    return {}


def metadata_value(node: dict[str, Any], key: str) -> Any:
    metadata = node.get("metadata") if isinstance(node, dict) and isinstance(node.get("metadata"), dict) else {}
    return metadata.get(key)


def sample_id(row: dict[str, Any]) -> str:
    for key in ["sample_id", "image", "image_path"]:
        value = str(row.get(key) or "")
        if value:
            parts = Path(value).parts
            if len(parts) >= 2 and parts[-1].lower().endswith(".png"):
                return parts[-2]
            return Path(value).stem
    return ""


def normalize_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(value[index]) for index in range(4)]
    except (TypeError, ValueError):
        return None
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def bbox_center_distance(a: list[float], b: list[float]) -> float:
    ax, ay = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
    bx, by = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    return math.hypot(ax - bx, ay - by)


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def num(value: str | None) -> float | None:
    if value is None:
        return None
    match = re.search(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", value)
    return float(match.group(0)) if match else None


def is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


if __name__ == "__main__":
    main()
