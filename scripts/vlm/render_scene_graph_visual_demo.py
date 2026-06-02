#!/usr/bin/env python3
"""Render configured CadStruct-MoE scene graph visual demos.

The primary renderer is SVG/HTML so the script can run without Pillow, OpenCV,
or browser automation. When `rsvg-convert` is available, PNG copies are emitted
for PPT/report use.
"""

from __future__ import annotations

import argparse
import zlib
import html
import json
import math
import re
import shutil
import struct
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_FAMILY_STYLES = {
    "boundary": {"stroke": "#1f77b4", "fill": "#1f77b4", "name": "Boundary: walls / doors / windows"},
    "wall": {"stroke": "#1f77b4", "fill": "#1f77b4", "name": "Wall boundary"},
    "opening": {"stroke": "#17becf", "fill": "#17becf", "name": "Door / opening"},
    "window": {"stroke": "#007f7f", "fill": "#007f7f", "name": "Window / glass"},
    "space": {"stroke": "#2ca02c", "fill": "#2ca02c", "name": "Space: rooms / regions"},
    "symbol": {"stroke": "#ff7f0e", "fill": "#ff7f0e", "name": "Symbol: fixtures / equipment"},
    "text": {"stroke": "#9467bd", "fill": "#9467bd", "name": "Text: labels / dimensions"},
    "dimension": {"stroke": "#8c564b", "fill": "#8c564b", "name": "Dimension aids: lines / leaders"},
    "sheet": {"stroke": "#7f7f7f", "fill": "#7f7f7f", "name": "Sheet: title block / layout"},
    "warning": {"stroke": "#d62728", "fill": "#d62728", "name": "Warning / needs review"},
    "unknown": {"stroke": "#444444", "fill": "#444444", "name": "Unknown family"},
}

DEFAULT_PREDICTIONS = Path("reports/vlm/e2e_cubicasa_visual_demo_predictions.jsonl")
DEFAULT_OUTPUT_DIR = Path("reports/vlm/visual_demo")
DEFAULT_CONFIG = Path("configs/vlm/cubicasa_visual_demo.json")
DEFAULT_CREATED = "2026-05-06"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--source-dataset", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-labels", type=int, default=None)
    parser.add_argument("--min-box-pixels", type=float, default=None)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    prediction_path = Path(args.predictions or config.get("default_predictions") or DEFAULT_PREDICTIONS)
    output_dir = Path(args.output_dir or config.get("default_output_dir") or DEFAULT_OUTPUT_DIR)
    source_dataset = args.source_dataset if args.source_dataset is not None else str(config.get("source_dataset") or "cubicasa5k")
    limit = args.limit if args.limit is not None else int(config.get("limit", 5))
    max_labels = args.max_labels if args.max_labels is not None else int(config.get("max_labels", 28))
    min_box_pixels = args.min_box_pixels if args.min_box_pixels is not None else float(config.get("min_box_pixels", 2.0))
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        row
        for row in load_jsonl(prediction_path)
        if not source_dataset or row.get("source_dataset") == source_dataset
    ]
    if limit > 0:
        rows = rows[:limit]
    if not rows:
        raise SystemExit(f"No prediction rows found for source_dataset={source_dataset!r}")
    sample_ids = [sample_id_for(row, index, config) for index, row in enumerate(rows)]
    clean_stale_sample_dirs(output_dir, set(sample_ids), config)

    sample_summaries: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        sample_id = sample_ids[index]
        sample_dir = output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        summary = render_sample(
            row,
            sample_id=sample_id,
            sample_dir=sample_dir,
            max_labels=max_labels,
            min_box_pixels=min_box_pixels,
            config=config,
        )
        sample_summaries.append(summary)

    manifest = build_manifest(prediction_path, output_dir, sample_summaries, config)
    write_json(output_dir / "sample_manifest_v1.json", manifest)
    coverage = build_coverage_audit(manifest, sample_summaries, config)
    write_json(output_dir / "coverage_audit_v1.json", coverage)
    write_text(output_dir / "coverage_audit_v1.md", render_coverage_markdown(coverage, config))
    write_review_pack(output_dir, manifest, coverage, config)
    write_notes(Path("docs/cadstruct/archive/cadstruct-visual-result-demo-notes.md"), manifest, config)
    update_todo_status(Path("todo.json"), config)
    print(json.dumps({"output_dir": str(output_dir), "samples": len(sample_summaries)}, ensure_ascii=False, indent=2))


def render_sample(
    row: dict[str, Any],
    sample_id: str,
    sample_dir: Path,
    max_labels: int,
    min_box_pixels: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    image_path = Path(str(row.get("image") or row.get("image_path") or ""))
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    image_w, image_h = png_size(image_path)
    graph = row.get("scene_graph") or {}
    nodes = list(graph.get("nodes") or [])
    warning_node_ids = extract_warning_node_ids(row.get("warnings") or [])
    raw_canvas_bbox = infer_canvas_bbox(nodes, image_w, image_h)
    source_canvas_bbox = source_canvas_bbox_for(row, image_w, image_h) if canvas_policy(config).get("prefer_svg_viewbox", True) else None
    raw_canvas_area = max(1.0, (raw_canvas_bbox[2] - raw_canvas_bbox[0]) * (raw_canvas_bbox[3] - raw_canvas_bbox[1]))
    canvas_nodes = []
    for node in nodes:
        bbox = normalize_bbox((node.get("geometry") or {}).get("bbox") or node.get("bbox"))
        if bbox is not None and not is_suspect_large_boundary_bbox(node, bbox, raw_canvas_area, config):
            canvas_nodes.append(node)
    canvas_bbox = source_canvas_bbox or infer_canvas_bbox(canvas_nodes, image_w, image_h)
    alignment_bbox = image_alignment_bbox(image_path, config)
    canvas_transform = canvas_to_image_transform(canvas_bbox, image_w, image_h, config, target_bbox=alignment_bbox)

    reference_input = sample_dir / "input_reference.png"
    shutil.copyfile(image_path, reference_input)
    annotation_background = annotation_background_path(row, sample_dir, image_w, image_h, config)
    display_input = annotation_background or reference_input
    display_w, display_h = png_size(display_input)
    if display_w != image_w or display_h != image_h:
        image_w, image_h = display_w, display_h
        canvas_transform = canvas_to_image_transform(canvas_bbox, image_w, image_h, config, target_bbox=alignment_bbox)

    draw_items: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for node in nodes:
        bbox = normalize_bbox((node.get("geometry") or {}).get("bbox") or node.get("bbox"))
        if bbox is None:
            skipped.append({"id": str(node.get("id") or ""), "reason": "missing_or_invalid_bbox"})
            continue
        if is_suspect_large_boundary_bbox(node, bbox, raw_canvas_area, config):
            skipped.append({"id": str(node.get("id") or ""), "reason": suspicious_bbox_policy(config).get("skip_reason", "suspect_large_boundary_bbox")})
            continue
        source_tolerance = float(canvas_policy(config).get("source_canvas_tolerance", 0.0))
        if source_canvas_bbox is not None and bbox_outside_canvas(bbox, source_canvas_bbox, tolerance=source_tolerance):
            skipped.append({"id": str(node.get("id") or ""), "reason": "bbox_outside_source_canvas"})
            continue
        mapped = map_bbox(bbox, canvas_transform)
        if mapped is None:
            skipped.append({"id": str(node.get("id") or ""), "reason": "bbox_outside_canvas"})
            continue
        x, y, w, h = mapped
        if w < min_box_pixels and h < min_box_pixels:
            skipped.append({"id": str(node.get("id") or ""), "reason": "too_small_after_scaling"})
            continue
        draw_items.append(
            {
                "node": node,
                "bbox": bbox,
                "canvas_bbox": canvas_bbox,
                "canvas_transform": canvas_transform,
                "image_size": [image_w, image_h],
                "rect": [x, y, w, h],
                "warning": str(node.get("id")) in warning_node_ids or node_needs_review(node, bbox, nodes, config),
            }
        )

    counts = Counter(display_family(item["node"], config) for item in draw_items)
    semantic_counts = Counter(str(item["node"].get("semantic_type") or "unknown") for item in draw_items)
    warning_counts = Counter(str(item["node"].get("family") or "unknown") for item in draw_items if item["warning"])
    label_items = choose_label_items(draw_items, max_labels=max_labels, config=config)

    overlay_svg = sample_dir / "overlay_only.svg"
    overlay_on_image_svg = sample_dir / "overlay_on_image.svg"
    side_by_side_svg = sample_dir / "side_by_side.svg"
    write_text(overlay_svg, render_overlay_svg(None, image_w, image_h, draw_items, label_items, sample_id, config))
    write_text(overlay_on_image_svg, render_overlay_svg(display_input.name, image_w, image_h, draw_items, label_items, sample_id, config))
    write_text(side_by_side_svg, render_side_by_side_svg(display_input.name, image_w, image_h, draw_items, label_items, sample_id, config))
    png_outputs = convert_sample_svgs(sample_dir, [overlay_svg, overlay_on_image_svg, side_by_side_svg])

    summary = {
        "sample_id": sample_id,
        "image": str(image_path),
        "annotation": row.get("annotation"),
        "source_dataset": row.get("source_dataset"),
        "split": row.get("split"),
        "gold_source": row.get("gold_source"),
        "source_mode": (row.get("route_trace") or {}).get("source_mode"),
        "image_size": {"width": image_w, "height": image_h},
        "raw_scene_canvas_bbox": raw_canvas_bbox,
        "source_scene_canvas_bbox": source_canvas_bbox,
        "inferred_scene_canvas_bbox": canvas_bbox,
        "image_alignment_bbox": alignment_bbox,
        "canvas_to_image_transform": canvas_transform,
        "node_count": len(nodes),
        "edge_count": len(graph.get("edges") or []),
        "rendered_nodes": len(draw_items),
        "skipped_nodes": len(skipped),
        "skipped_reasons": dict(Counter(item["reason"] for item in skipped)),
        "skipped_examples": skipped[:20],
        "per_family_counts": dict(counts),
        "top_semantic_counts": dict(semantic_counts.most_common(20)),
        "warning_count": len(row.get("warnings") or []),
        "warning_node_family_counts": dict(warning_counts),
        "needs_review_nodes": sum(1 for item in draw_items if item["warning"]),
        "boundary_render_mode": str(config.get("boundary_render_mode", "geometry_preferred_bbox_fallback")),
        "warnings": list(row.get("warnings") or [])[:50],
        "quality_report": row.get("quality_report") or {},
        "route_trace": row.get("route_trace") or {},
        "files": {
            "input": str(display_input),
            "input_reference": str(reference_input),
            "input_aligned_from_annotation": str(annotation_background or ""),
            "overlay_only": str(overlay_svg),
            "overlay_only_png": str(png_outputs.get("overlay_only", "")),
            "overlay_on_image": str(overlay_on_image_svg),
            "overlay_on_image_png": str(png_outputs.get("overlay_on_image", "")),
            "side_by_side": str(side_by_side_svg),
            "side_by_side_png": str(png_outputs.get("side_by_side", "")),
            "summary": str(sample_dir / "summary.json"),
        },
        "selection_reason": selection_reason(counts, row.get("warnings") or [], config),
        "recognized_summary": recognized_summary(counts, config),
        "limitation_summary": limitation_summary(row, skipped),
    }
    write_json(sample_dir / "summary.json", summary)
    return summary


def render_overlay_svg(
    image_href: str | None,
    width: int,
    height: int,
    draw_items: list[dict[str, Any]],
    label_items: list[dict[str, Any]],
    sample_id: str,
    config: dict[str, Any],
) -> str:
    legend_cfg = legend_config(config)
    legend_h = int(legend_cfg.get("overlay_extra_height", 58))
    total_h = height + legend_h
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{total_h}" viewBox="0 0 {width} {total_h}">',
        "<defs>",
        '<style>text{font-family:Arial,Helvetica,sans-serif}.label{font-size:10px;font-weight:700;paint-order:stroke;stroke:#fff;stroke-width:2px}.legend{font-size:11px}.title{font-size:14px;font-weight:700}</style>',
        "</defs>",
    ]
    if image_href:
        parts.append(f'<image href="{escape_attr(image_href)}" x="0" y="0" width="{width}" height="{height}" preserveAspectRatio="none"/>')
        parts.append('<rect x="0" y="0" width="100%" height="100%" fill="#ffffff" opacity="0.08"/>')
    else:
        parts.append('<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>')
    parts.append(render_boxes(draw_items, config))
    parts.append(render_labels(label_items, config))
    parts.append(render_legend(8, height + 18, sample_id, config, max_width=width - 16))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def render_side_by_side_svg(
    image_href: str,
    width: int,
    height: int,
    draw_items: list[dict[str, Any]],
    label_items: list[dict[str, Any]],
    sample_id: str,
    config: dict[str, Any],
) -> str:
    legend_cfg = legend_config(config)
    gap = 24
    panel_w = width
    legend_h = int(legend_cfg.get("side_by_side_extra_height", 64))
    total_w = panel_w * 2 + gap
    total_h = height + legend_h
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{total_h}" viewBox="0 0 {total_w} {total_h}">',
        "<defs>",
        '<style>text{font-family:Arial,Helvetica,sans-serif}.label{font-size:10px;font-weight:700;paint-order:stroke;stroke:#fff;stroke-width:2px}.legend{font-size:11px}.title{font-size:14px;font-weight:700}.panel{font-size:13px;font-weight:700}</style>',
        "</defs>",
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>',
        f'<text class="panel" x="0" y="14">{escape_text(review_pack_config(config).get("input_panel_title", "Input floorplan"))}</text>',
        f'<text class="panel" x="{panel_w + gap}" y="14">{escape_text(review_pack_config(config).get("overlay_panel_title", "MoE scene graph overlay"))}</text>',
        f'<image href="{escape_attr(image_href)}" x="0" y="22" width="{width}" height="{height}" preserveAspectRatio="none"/>',
        f'<g transform="translate({panel_w + gap},22)">',
        f'<image href="{escape_attr(image_href)}" x="0" y="0" width="{width}" height="{height}" preserveAspectRatio="none"/>',
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff" opacity="0.08"/>',
        render_boxes(draw_items, config),
        render_labels(label_items, config),
        "</g>",
        render_legend(0, height + 42, sample_id, config),
        "</svg>",
    ]
    return "\n".join(parts) + "\n"


def annotation_background_path(
    row: dict[str, Any],
    sample_dir: Path,
    image_w: int,
    image_h: int,
    config: dict[str, Any],
) -> Path | None:
    policy = background_policy(config)
    if str(policy.get("mode") or "source_image") != "annotation_render":
        return None
    annotation = row.get("annotation") or row.get("annotation_path")
    if not annotation:
        return None
    annotation_path = Path(str(annotation))
    if not annotation_path.exists() or not shutil.which("rsvg-convert"):
        return None
    output_name = str(policy.get("annotation_render_name") or "input_aligned_from_svg.png")
    output = sample_dir / output_name
    try:
        subprocess.run(
            ["rsvg-convert", "-w", str(image_w), "-h", str(image_h), str(annotation_path), "-o", str(output)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return None
    return output if output.exists() else None


def render_boxes(draw_items: list[dict[str, Any]], config: dict[str, Any]) -> str:
    styles = family_styles(config)
    ordered = sorted(draw_items, key=lambda item: family_order(display_family(item["node"], config), config))
    parts = ['<g class="boxes">']
    for item in ordered:
        node = item["node"]
        x, y, w, h = item["rect"]
        family = display_family(node, config)
        style = styles.get(family, styles["unknown"])
        opacity = str(style.get("fill_opacity", 0.08))
        stroke_width = str(styles["warning"].get("stroke_width", 2.2) if item["warning"] else style.get("stroke_width", 1.3))
        stroke = styles["warning"]["stroke"] if item["warning"] else style["stroke"]
        fill = style["fill"]
        dash = ' stroke-dasharray="5 3"' if item["warning"] else ""
        geometry_svg = render_geometry_item(item, stroke, fill, opacity, stroke_width, dash)
        if geometry_svg:
            parts.append(geometry_svg)
        else:
            parts.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
                f'fill="{fill}" fill-opacity="{opacity}" stroke="{stroke}" stroke-width="{stroke_width}"{dash}/>'
            )
    parts.append("</g>")
    return "\n".join(parts)


def render_geometry_item(item: dict[str, Any], stroke: str, fill: str, opacity: str, stroke_width: str, dash: str) -> str | None:
    node = item["node"]
    geometry = (node.get("geometry") or {}).get("source_geometry") or (node.get("geometry") or {}).get("geometry") or node.get("geometry") or {}
    if not isinstance(geometry, dict):
        return None
    geometry_type = str(geometry.get("type") or "")
    points = geometry.get("points")
    if geometry_type not in {"polygon", "polyline", "line"} or not isinstance(points, list) or len(points) < 2:
        return None
    mapped = [map_point(point, item["canvas_transform"]) for point in points]
    if any(point is None for point in mapped):
        return None
    point_text = " ".join(f"{point[0]:.2f},{point[1]:.2f}" for point in mapped if point is not None)
    if geometry_type == "polygon":
        return f'<polygon points="{point_text}" fill="{fill}" fill-opacity="{opacity}" stroke="{stroke}" stroke-width="{stroke_width}"{dash}/>'
    if geometry_type == "polyline":
        return f'<polyline points="{point_text}" fill="none" stroke="{stroke}" stroke-width="{stroke_width}"{dash}/>'
    first, second = mapped[0], mapped[1]
    if first is None or second is None:
        return None
    return f'<line x1="{first[0]:.2f}" y1="{first[1]:.2f}" x2="{second[0]:.2f}" y2="{second[1]:.2f}" stroke="{stroke}" stroke-width="{stroke_width}"{dash}/>'


def render_labels(label_items: list[dict[str, Any]], config: dict[str, Any]) -> str:
    styles = family_styles(config)
    parts = ['<g class="labels">']
    for item in label_items:
        node = item["node"]
        x, y, _w, _h = item["rect"]
        family = display_family(node, config)
        style = styles.get(family, styles["unknown"])
        label = short_node_label(node, config)
        parts.append(f'<text class="label" x="{x + 2:.2f}" y="{max(10, y - 2):.2f}" fill="{style["stroke"]}">{escape_text(label)}</text>')
    parts.append("</g>")
    return "\n".join(parts)


def render_legend(x: int, y: int, sample_id: str, config: dict[str, Any], max_width: int | None = None) -> str:
    styles = family_styles(config)
    legend_cfg = legend_config(config)
    parts = [f'<g transform="translate({x},{y})">', f'<text class="title" x="0" y="0">{escape_text(sample_id)}</text>']
    dx = 0
    row_y = 14
    for family in legend_cfg.get("family_order", ["boundary", "space", "symbol", "text", "dimension", "sheet", "warning"]):
        style = styles[family]
        label = style.get("short_name", family.title()) if max_width is not None else style["name"]
        item_w = int(legend_cfg.get("compact_item_width", 120)) if max_width is not None else int(legend_cfg.get("full_warning_item_width" if family == "warning" else "full_item_width", 185))
        if max_width is not None and dx > 0 and dx + item_w > max_width:
            dx = 0
            row_y += int(legend_cfg.get("row_height", 18))
        parts.append(f'<rect x="{dx}" y="{row_y}" width="14" height="10" fill="{style["fill"]}" fill-opacity="0.35" stroke="{style["stroke"]}"/>')
        parts.append(f'<text class="legend" x="{dx + 18}" y="{row_y + 10}">{escape_text(label)}</text>')
        dx += item_w
    parts.append("</g>")
    return "\n".join(parts)


def choose_label_items(draw_items: list[dict[str, Any]], max_labels: int, config: dict[str, Any]) -> list[dict[str, Any]]:
    label_cfg = label_policy(config)
    priorities = {family: index for index, family in enumerate(label_cfg.get("family_priority", ["space", "symbol", "boundary", "text", "sheet"]))}
    min_x_gap = float(label_cfg.get("min_x_gap", 52))
    min_y_gap = float(label_cfg.get("min_y_gap", 16))
    candidates = sorted(
        draw_items,
        key=lambda item: (
            priorities.get(display_family(item["node"], config), 9),
            -area(item["rect"]),
            str(item["node"].get("id") or ""),
        ),
    )
    used: list[tuple[float, float]] = []
    selected: list[dict[str, Any]] = []
    for item in candidates:
        x, y, _w, _h = item["rect"]
        if any(abs(x - ox) < min_x_gap and abs(y - oy) < min_y_gap for ox, oy in used):
            continue
        selected.append(item)
        used.append((x, y))
        if len(selected) >= max_labels:
            break
    return selected


def build_manifest(prediction_path: Path, output_dir: Path, summaries: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": str(config.get("manifest_version", "cadstruct_moe_visual_demo_manifest_v1")),
        "created": str(config.get("created", DEFAULT_CREATED)),
        "config": str(config.get("_config_path", "")),
        "prediction_path": str(prediction_path),
        "output_dir": str(output_dir),
        "source_dataset_policy": str(config.get("source_dataset_policy", "Only CubiCasa5K/CubiCasa samples are used in this visual demo.")),
        "samples": summaries,
        "done_when_checks": {
            "all_images_exist": all(Path(item["image"]).exists() for item in summaries),
            "all_source_dataset_selected": all(item.get("source_dataset") == config.get("source_dataset") for item in summaries),
            "all_have_scene_graph": all(item.get("node_count", 0) > 0 for item in summaries),
            "has_uncertain_or_warning_case": any(item.get("warning_count", 0) > 0 or item.get("skipped_nodes", 0) > 0 for item in summaries),
        },
    }


def build_coverage_audit(manifest: dict[str, Any], summaries: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": str(config.get("coverage_version", "cadstruct_moe_visual_coverage_audit_v1")),
        "created": str(config.get("created", DEFAULT_CREATED)),
        "source_boundary": {
            "dataset": str(config.get("source_dataset", "selected")),
            "claim": "Visualization demo of the current scene_graph output contract.",
            "not_claimed": "This does not by itself prove external scanned-drawing generalization.",
        },
        "samples": [
            {
                "sample_id": item["sample_id"],
                "image": item["image"],
                "source_dataset": item["source_dataset"],
                "recognized": item["recognized_summary"],
                "limitations": item["limitation_summary"],
                "counts": {
                    "nodes": item["node_count"],
                    "edges": item["edge_count"],
                    "rendered_nodes": item["rendered_nodes"],
                    "skipped_nodes": item["skipped_nodes"],
                    "per_family": item["per_family_counts"],
                    "warnings": item["warning_count"],
                    "needs_review_nodes": item.get("needs_review_nodes", 0),
                    "warning_node_family_counts": item.get("warning_node_family_counts", {}),
                },
                "visual_files": item["files"],
            }
            for item in summaries
        ],
        "manifest_done_when_checks": manifest["done_when_checks"],
    }


def render_coverage_markdown(coverage: dict[str, Any], config: dict[str, Any]) -> str:
    lines = [
        f"# {review_pack_config(config).get('coverage_title', 'CadStruct-MoE Visual Coverage Audit v1')}",
        "",
        str(review_pack_config(config).get("question_text", "This review pack answers the advisor-facing visual review question.")),
        "",
        f"Scope: {review_pack_config(config).get('scope_text', 'samples are selected by the configured dataset policy.')}",
        "",
    ]
    for sample in coverage["samples"]:
        counts = sample["counts"]
        lines.extend(
            [
                f"## {sample['sample_id']}",
                "",
                f"- Input: `{sample['image']}`",
                f"- Visual comparison: `{sample['visual_files']['side_by_side']}`",
                f"- Recognized: {sample['recognized']}",
                f"- Uncertain / not reliably shown: {sample['limitations']}",
                f"- Counts: nodes={counts['nodes']}, edges={counts['edges']}, rendered={counts['rendered_nodes']}, skipped={counts['skipped_nodes']}, warnings={counts['warnings']}, needs_review={counts.get('needs_review_nodes', 0)}, per_family={counts['per_family']}",
                "",
            ]
        )
    return "\n".join(lines)


def write_review_pack(output_dir: Path, manifest: dict[str, Any], coverage: dict[str, Any], config: dict[str, Any]) -> None:
    pack_dir = output_dir / str(config.get("review_pack_name", "review_pack_v1"))
    pack_dir.mkdir(parents=True, exist_ok=True)
    readme = [
        f"# {review_pack_config(config).get('versioned_title', 'CadStruct-MoE Visual Review Pack v1')}",
        "",
        f"Config: `{config.get('_config_path', '')}`",
        "",
        f"Purpose: {review_pack_config(config).get('purpose', 'show input floorplans next to recognition overlays.')}",
        "",
        f"Color legend: {review_pack_config(config).get('legend_text', 'configured overlay legend.')}",
        "",
    ]
    cards = []
    for sample in coverage["samples"]:
        side_file = sample["visual_files"].get("side_by_side_png") or sample["visual_files"]["side_by_side"]
        side = relpath_for_html(Path(side_file), pack_dir)
        overlay = relpath_for_html(Path(sample["visual_files"]["overlay_on_image"]), pack_dir)
        readme.extend(
            [
                f"## {sample['sample_id']}",
                "",
                f"- Side-by-side: `{sample['visual_files']['side_by_side']}`",
                f"- Recognized: {sample['recognized']}",
                f"- Uncertain / limitation: {sample['limitations']}",
                "",
            ]
        )
        cards.append(
            f"""
<section class="sample">
  <h2>{escape_text(sample['sample_id'])}</h2>
  <p><strong>Recognized:</strong> {escape_text(sample['recognized'])}</p>
  <p><strong>Uncertain / limitation:</strong> {escape_text(sample['limitations'])}</p>
  <p><strong>Needs review:</strong> {escape_text(str(sample['counts'].get('needs_review_nodes', 0)))} nodes; {escape_text(str(sample['counts'].get('warning_node_family_counts', {})))}</p>
  <img src="{escape_attr(side)}" alt="{escape_attr(sample['sample_id'])} side by side">
  <p class="link"><a href="{escape_attr(overlay)}">Overlay-only view</a></p>
</section>
""".strip()
        )
    write_text(pack_dir / "README.md", "\n".join(readme))
    html_doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape_text(review_pack_config(config).get('title', 'CadStruct-MoE Visual Review Pack'))}</title>
  <style>
    body {{ margin: 0; font-family: Arial, Helvetica, sans-serif; color: #202124; background: #f7f7f5; }}
    header {{ padding: 24px 28px 12px; background: #fff; border-bottom: 1px solid #ddd; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 20px; }}
    .sample {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 0 0 18px; }}
    h1 {{ font-size: 22px; margin: 0 0 8px; }}
    h2 {{ font-size: 18px; margin: 0 0 10px; }}
    p {{ line-height: 1.45; margin: 8px 0; }}
    img {{ display: block; width: 100%; height: auto; border: 1px solid #ddd; background: #fff; }}
    .legend {{ font-size: 13px; color: #444; }}
    .link {{ font-size: 13px; }}
  </style>
</head>
<body>
  <header>
    <h1>{escape_text(review_pack_config(config).get('versioned_title', 'CadStruct-MoE Visual Review Pack v1'))}</h1>
    <p class="legend">{escape_text(review_pack_config(config).get('legend_text', 'configured overlay legend.'))}</p>
  </header>
  <main>
    {' '.join(cards)}
  </main>
</body>
</html>
"""
    write_text(pack_dir / "index.html", html_doc)
    first = coverage["samples"][0]["visual_files"]["side_by_side"]
    paper_name = str(config.get("paper_candidate_figure", "paper_candidate_figure_v1"))
    paper_svg = output_dir / f"{paper_name}.svg"
    write_text(paper_svg, Path(first).read_text(encoding="utf-8"))
    convert_svg_to_png(paper_svg, output_dir / f"{paper_name}.png")


def convert_sample_svgs(sample_dir: Path, svg_paths: list[Path]) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    for svg_path in svg_paths:
        png_path = sample_dir / f"{svg_path.stem}.png"
        if convert_svg_to_png(svg_path, png_path):
            outputs[svg_path.stem] = png_path
    return outputs


def convert_svg_to_png(svg_path: Path, png_path: Path) -> bool:
    if shutil.which("rsvg-convert") is None:
        return False
    png_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["rsvg-convert", str(svg_path), "-o", str(png_path)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.returncode == 0 and png_path.exists()


def write_notes(path: Path, manifest: dict[str, Any], config: dict[str, Any]) -> None:
    notes = config.get("notes") if isinstance(config.get("notes"), dict) else {}
    lines = [
        f"# {notes.get('title', 'CadStruct-MoE Visual Result Demo Notes')}",
        "",
        str(notes.get("claim_boundary", "This note records the claim boundary for the CubiCasa5K visual result demo.")),
        "",
        f"- Config: `{manifest.get('config', '')}`",
        f"- Dataset: {notes.get('dataset_line', 'this demo uses CubiCasa5K/CubiCasa floorplan samples only.')}",
        f"- Pipeline: {notes.get('pipeline_line', 'each visual is generated from `scene_graph.nodes` and `scene_graph.edges` emitted by `scripts/vlm/run_real_drawing_pipeline.py`.')}",
        f"- Source mode: {notes.get('source_mode_line', 'the current smoke run may use `expected_json` / SVG supervision as upstream expert outputs. In that case, the figure demonstrates the output contract and visualization method, not standalone real-world model generalization.')}",
        f"- Interpretation: {notes.get('interpretation_line', 'colored overlays show recognized scene-graph elements. Red marks warnings or elements that need review. Skipped/missing bbox counts in the audit are treated as limitations.')}",
        f"- Evaluation boundary: {notes.get('evaluation_line', 'quantitative claims should remain tied to locked evaluation reports; the visual pack is for advisor review and qualitative communication.')}",
        "",
        f"Generated samples: {len(manifest.get('samples') or [])}",
        "",
        f"## {boundary_overlay_audit(config).get('title', 'Boundary Overlay Audit')}",
        "",
        f"- Finding: {boundary_overlay_audit(config).get('finding', 'oversized boxes can come from visualization artifacts.')}",
        f"- Evidence: {boundary_overlay_audit(config).get('evidence', 'legacy source metadata should be audited before model claims.')}",
        f"- Parser fix: {boundary_overlay_audit(config).get('parser_fix', 'SVG path bbox parsing is handled before rendering.')}",
        f"- Renderer fix: {boundary_overlay_audit(config).get('renderer_fix', 'source canvas and suspicious bbox policies are applied before rendering.')}",
        f"- Residual interpretation: {boundary_overlay_audit(config).get('residual_interpretation', 'remaining broad boundaries should be interpreted according to source annotations.')}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text(path, "\n".join(lines))


def update_todo_status(path: Path, config: dict[str, Any]) -> None:
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for task in data.get("tasks") or []:
        if str(task.get("id", "")).startswith("VIS-V1-"):
            task["status"] = "completed"
            changed = True
    if not changed:
        return
    tasks = data.get("tasks") or []
    completed = sum(1 for task in tasks if task.get("status") == "completed")
    data["overall_status"] = {
        "total_tasks": len(tasks),
        "completed": completed,
        "pending": len(tasks) - completed,
        "readiness": str(config.get("todo_readiness", "Executed visual demo artifacts, coverage audit, review pack, and claim-boundary notes are generated.")),
    }
    write_json(path, data)


def selection_reason(counts: Counter[str], warnings: list[str], config: dict[str, Any]) -> str:
    selection_cfg = config.get("selection_policy") if isinstance(config.get("selection_policy"), dict) else {}
    if warnings:
        return str(selection_cfg.get("warning_reason", "Includes warning/uncertain cases so the review pack does not only show best-case recognition."))
    if counts.get("symbol", 0) >= int(selection_cfg.get("symbol_dense_threshold", 50)):
        return str(selection_cfg.get("symbol_dense_reason", "Symbol-dense drawing, useful for fixture/equipment recognition visualization."))
    if counts.get("text", 0) >= int(selection_cfg.get("text_dense_threshold", 100)):
        return str(selection_cfg.get("text_dense_reason", "Text/dimension-dense drawing, useful for TextDimensionExpert visualization."))
    return str(selection_cfg.get("default_reason", "Representative drawing with boundary, room, symbol, and text outputs."))


def recognized_summary(counts: Counter[str] | dict[str, int], config: dict[str, Any]) -> str:
    c = dict(counts)
    parts = []
    for family in sorted(family_styles(config), key=lambda item: family_order(item, config)):
        label = str(family_styles(config)[family].get("summary_label", "items"))
        if c.get(family, 0):
            parts.append(f"{c[family]} {label}")
    return ", ".join(parts) if parts else "No rendered scene-graph nodes."


def display_family(node: dict[str, Any], config: dict[str, Any]) -> str:
    semantic_type = str(node.get("semantic_type") or "")
    semantic_map = config.get("semantic_display_family")
    if isinstance(semantic_map, dict) and semantic_type in semantic_map:
        return str(semantic_map[semantic_type])
    return str(node.get("family") or "unknown")


def clean_stale_sample_dirs(output_dir: Path, active_sample_ids: set[str], config: dict[str, Any]) -> None:
    if not config.get("clean_stale_sample_dirs", False):
        return
    prefix = sanitize_id(str(config.get("sample_id_prefix") or config.get("source_dataset") or "sample"))
    pattern = f"{prefix}_*"
    for path in output_dir.glob(pattern):
        if not path.is_dir() or path.name in active_sample_ids:
            continue
        shutil.rmtree(path)


def limitation_summary(row: dict[str, Any], skipped: list[dict[str, str]]) -> str:
    warnings = list(row.get("warnings") or [])
    details = []
    if warnings:
        details.append(f"{len(warnings)} graph warnings, including {warnings[0]}")
    if skipped:
        reasons = Counter(item["reason"] for item in skipped)
        details.append(f"{len(skipped)} nodes not rendered because {dict(reasons)}")
    source_mode = (row.get("route_trace") or {}).get("source_mode") or "unknown"
    if source_mode == "expected_json":
        details.append("source_mode=expected_json; visualization/oracle-smoke output, not external generalization.")
    elif "real_upstream" in str(source_mode):
        details.append(f"source_mode={source_mode}; labels are saved model predictions over parser/SVG candidate geometry.")
    else:
        details.append(f"source_mode={source_mode}")
    return "; ".join(details)


def infer_canvas_bbox(nodes: list[dict[str, Any]], image_w: int, image_h: int) -> list[float]:
    xs: list[float] = []
    ys: list[float] = []
    for node in nodes:
        bbox = normalize_bbox((node.get("geometry") or {}).get("bbox") or node.get("bbox"))
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        xs.extend([x0, x1])
        ys.extend([y0, y1])
    if not xs or not ys:
        return [0.0, 0.0, float(image_w), float(image_h)]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if min_x >= -1 and min_y >= -1 and max_x <= image_w * 1.1 and max_y <= image_h * 1.1:
        return [0.0, 0.0, float(image_w), float(image_h)]
    return [min(0.0, min_x), min(0.0, min_y), max_x, max_y]


def source_canvas_bbox_for(row: dict[str, Any], image_w: int, image_h: int) -> list[float] | None:
    annotation = row.get("annotation") or row.get("annotation_path")
    if not annotation:
        return None
    path = Path(str(annotation))
    if not path.exists():
        return None
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return None
    view_box = root.attrib.get("viewBox")
    if view_box:
        values = [float(item) for item in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", view_box)]
        if len(values) >= 4 and values[2] > 0 and values[3] > 0:
            return [values[0], values[1], values[0] + values[2], values[1] + values[3]]
    width = first_svg_number(root.attrib.get("width"))
    height = first_svg_number(root.attrib.get("height"))
    if width and height:
        return [0.0, 0.0, width, height]
    return [0.0, 0.0, float(image_w), float(image_h)]


def first_svg_number(value: str | None) -> float | None:
    if not value:
        return None
    match = re.search(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", value)
    if match is None:
        return None
    number = float(match.group(0))
    return number if number > 0 else None


def bbox_outside_canvas(bbox: list[float], canvas: list[float], tolerance: float) -> bool:
    c0, c1, c2, c3 = canvas
    pad_x = max(1.0, (c2 - c0) * tolerance)
    pad_y = max(1.0, (c3 - c1) * tolerance)
    return bbox[0] < c0 - pad_x or bbox[1] < c1 - pad_y or bbox[2] > c2 + pad_x or bbox[3] > c3 + pad_y


def is_suspect_large_boundary_bbox(node: dict[str, Any], bbox: list[float], canvas_area: float, config: dict[str, Any]) -> bool:
    policy = suspicious_bbox_policy(config)
    if str(node.get("family") or "") not in set(policy.get("families", ["boundary"])):
        return False
    semantic_type = str(node.get("semantic_type") or "")
    if semantic_type not in set(policy.get("semantic_types", ["door", "opening", "window"])):
        return False
    width = max(0.0, bbox[2] - bbox[0])
    height = max(0.0, bbox[3] - bbox[1])
    area = width * height
    canvas_side = math.sqrt(canvas_area)
    if semantic_type in set(policy.get("large_local_semantic_types", ["door", "opening"])) and max(width, height) / max(canvas_side, 1.0) > float(policy.get("max_side_over_canvas_side", 0.25)):
        return True
    if area / canvas_area > float(policy.get("max_area_over_canvas_area", 0.08)):
        return True
    aspect = max(width, height) / max(min(width, height), 1.0)
    return aspect > float(policy.get("max_aspect_ratio", 60.0)) and area / canvas_area > float(policy.get("min_area_over_canvas_area_for_aspect_rule", 0.01))


def node_needs_review(node: dict[str, Any], bbox: list[float], nodes: list[dict[str, Any]], config: dict[str, Any]) -> bool:
    policy = config.get("needs_review_policy") if isinstance(config.get("needs_review_policy"), dict) else {}
    semantic = str(node.get("semantic_type") or "")
    family = str(node.get("family") or "")
    if family == "symbol" and semantic in set(policy.get("symbol_requires_room", ["equipment"])):
        return not any(
            str(other.get("family") or "") == "space"
            and (other_bbox := normalize_bbox((other.get("geometry") or {}).get("bbox") or other.get("bbox"))) is not None
            and bbox_contains(other_bbox, bbox)
            for other in nodes
        )
    return False


def canvas_to_image_transform(
    canvas: list[float],
    image_w: int,
    image_h: int,
    config: dict[str, Any],
    target_bbox: list[float] | None = None,
) -> dict[str, Any]:
    c0, c1, c2, c3 = canvas
    cw = c2 - c0
    ch = c3 - c1
    if cw <= 0 or ch <= 0:
        return {
            "mode": "invalid_canvas",
            "canvas_bbox": canvas,
            "image_size": [image_w, image_h],
            "scale_x": 0.0,
            "scale_y": 0.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
            "content_bbox": [0.0, 0.0, 0.0, 0.0],
        }
    policy = canvas_policy(config)
    mode = str(policy.get("svg_image_fit", "meet")).lower()
    target = target_bbox if target_bbox is not None else [0.0, 0.0, float(image_w), float(image_h)]
    tx0, ty0, tx1, ty1 = target
    target_w = max(1.0, tx1 - tx0)
    target_h = max(1.0, ty1 - ty0)
    raw_scale_x = target_w / cw
    raw_scale_y = target_h / ch
    if mode in {"stretch", "fill"}:
        scale_x = raw_scale_x
        scale_y = raw_scale_y
        offset_x = tx0
        offset_y = ty0
        mode = "stretch"
    elif mode in {"slice", "cover"}:
        scale = max(raw_scale_x, raw_scale_y)
        scale_x = scale
        scale_y = scale
        offset_x = tx0 + (target_w - cw * scale) / 2.0
        offset_y = ty0 + (target_h - ch * scale) / 2.0
        mode = "slice"
    else:
        scale = min(raw_scale_x, raw_scale_y)
        scale_x = scale
        scale_y = scale
        offset_x = tx0 + (target_w - cw * scale) / 2.0
        offset_y = ty0 + (target_h - ch * scale) / 2.0
        mode = "meet"
    return {
        "mode": mode,
        "canvas_bbox": [round(float(value), 6) for value in canvas],
        "image_size": [int(image_w), int(image_h)],
        "target_bbox": [round(float(value), 6) for value in target],
        "scale_x": round(float(scale_x), 9),
        "scale_y": round(float(scale_y), 9),
        "offset_x": round(float(offset_x), 6),
        "offset_y": round(float(offset_y), 6),
        "content_bbox": [
            round(float(offset_x), 6),
            round(float(offset_y), 6),
            round(float(offset_x + cw * scale_x), 6),
            round(float(offset_y + ch * scale_y), 6),
        ],
        "raw_stretch_scale_x": round(float(raw_scale_x), 9),
        "raw_stretch_scale_y": round(float(raw_scale_y), 9),
    }


def map_bbox(bbox: list[float], transform: dict[str, Any]) -> list[float] | None:
    canvas = transform.get("canvas_bbox")
    if not isinstance(canvas, list) or len(canvas) != 4:
        return None
    c0, c1, c2, c3 = [float(value) for value in canvas]
    image_size = transform.get("image_size")
    if not isinstance(image_size, list) or len(image_size) != 2:
        return None
    image_w = float(image_size[0])
    image_h = float(image_size[1])
    scale_x = float(transform.get("scale_x") or 0.0)
    scale_y = float(transform.get("scale_y") or 0.0)
    offset_x = float(transform.get("offset_x") or 0.0)
    offset_y = float(transform.get("offset_y") or 0.0)
    if scale_x <= 0 or scale_y <= 0:
        return None
    x0, y0, x1, y1 = bbox
    x = offset_x + (x0 - c0) * scale_x
    y = offset_y + (y0 - c1) * scale_y
    w = (x1 - x0) * scale_x
    h = (y1 - y0) * scale_y
    if x > image_w or y > image_h or x + w < 0 or y + h < 0:
        return None
    x = max(0.0, min(float(image_w), x))
    y = max(0.0, min(float(image_h), y))
    w = max(0.0, min(float(image_w) - x, w))
    h = max(0.0, min(float(image_h) - y, h))
    return [x, y, w, h]


def map_point(point: Any, transform: dict[str, Any]) -> list[float] | None:
    if not isinstance(point, list | tuple) or len(point) != 2:
        return None
    canvas = transform.get("canvas_bbox")
    if not isinstance(canvas, list) or len(canvas) != 4:
        return None
    c0, c1, _c2, _c3 = [float(value) for value in canvas]
    image_size = transform.get("image_size")
    if not isinstance(image_size, list) or len(image_size) != 2:
        return None
    image_w = float(image_size[0])
    image_h = float(image_size[1])
    scale_x = float(transform.get("scale_x") or 0.0)
    scale_y = float(transform.get("scale_y") or 0.0)
    offset_x = float(transform.get("offset_x") or 0.0)
    offset_y = float(transform.get("offset_y") or 0.0)
    if scale_x <= 0 or scale_y <= 0:
        return None
    try:
        x0 = float(point[0])
        y0 = float(point[1])
    except (TypeError, ValueError):
        return None
    x = offset_x + (x0 - c0) * scale_x
    y = offset_y + (y0 - c1) * scale_y
    return [max(0.0, min(float(image_w), x)), max(0.0, min(float(image_h), y))]


def image_alignment_bbox(image_path: Path, config: dict[str, Any]) -> list[float] | None:
    policy = image_alignment_policy(config)
    if not policy.get("enabled", False):
        return None
    try:
        width, height, channels, pixels = read_png_pixels(image_path)
    except (OSError, ValueError, zlib.error):
        return None
    bbox = dark_pixel_bbox(
        width,
        height,
        channels,
        pixels,
        threshold=int(policy.get("dark_threshold", 210)),
        min_component_delta=int(policy.get("min_component_delta", 28)),
    )
    if bbox is None:
        return None
    min_coverage = float(policy.get("min_coverage", 0.2))
    area_ratio = bbox_xyxy_area(bbox) / max(float(width * height), 1.0)
    if area_ratio < min_coverage:
        return None
    margin = float(policy.get("margin_pixels", 0.0))
    if margin:
        bbox = [
            max(0.0, bbox[0] - margin),
            max(0.0, bbox[1] - margin),
            min(float(width), bbox[2] + margin),
            min(float(height), bbox[3] + margin),
        ]
    return [round(float(value), 3) for value in bbox]


def dark_pixel_bbox(
    width: int,
    height: int,
    channels: int,
    pixels: bytearray,
    threshold: int,
    min_component_delta: int,
) -> list[float] | None:
    xs: list[int] = []
    ys: list[int] = []
    stride = width * channels
    for y in range(height):
        row = y * stride
        for x in range(width):
            index = row + x * channels
            r = pixels[index]
            g = pixels[index + 1]
            b = pixels[index + 2]
            if channels == 4 and pixels[index + 3] == 0:
                continue
            if min(r, g, b) <= threshold and (255 - min(r, g, b)) >= min_component_delta:
                xs.append(x)
                ys.append(y)
    if not xs:
        return None
    return [float(min(xs)), float(min(ys)), float(max(xs) + 1), float(max(ys) + 1)]


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def read_png_pixels(path: Path) -> tuple[int, int, int, bytearray]:
    data = path.read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("not_png")
    pos = len(PNG_SIGNATURE)
    width = height = color_type = bit_depth = interlace = None
    idat = bytearray()
    while pos < len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        chunk_type = data[pos + 4 : pos + 8]
        chunk_data = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _, _, interlace = struct.unpack(">IIBBBBB", chunk_data)
        elif chunk_type == b"IDAT":
            idat.extend(chunk_data)
        elif chunk_type == b"IEND":
            break
    if width is None or height is None or color_type is None:
        raise ValueError("missing_ihdr")
    if bit_depth != 8 or color_type not in (2, 6) or interlace != 0:
        raise ValueError(f"unsupported_png:bit_depth={bit_depth}:color_type={color_type}:interlace={interlace}")
    channels = 3 if color_type == 2 else 4
    raw = zlib.decompress(bytes(idat))
    stride = width * channels
    pixels = bytearray(width * height * channels)
    prev = bytearray(stride)
    offset = 0
    for y in range(height):
        filter_type = raw[offset]
        offset += 1
        scan = bytearray(raw[offset : offset + stride])
        offset += stride
        recon = unfilter_scanline(filter_type, scan, prev, channels)
        pixels[y * stride : (y + 1) * stride] = recon
        prev = recon
    return int(width), int(height), channels, pixels


def unfilter_scanline(filter_type: int, scan: bytearray, prev: bytearray, bpp: int) -> bytearray:
    out = bytearray(scan)
    for i in range(len(out)):
        left = out[i - bpp] if i >= bpp else 0
        up = prev[i] if prev else 0
        up_left = prev[i - bpp] if prev and i >= bpp else 0
        if filter_type == 0:
            continue
        if filter_type == 1:
            out[i] = (out[i] + left) & 0xFF
        elif filter_type == 2:
            out[i] = (out[i] + up) & 0xFF
        elif filter_type == 3:
            out[i] = (out[i] + ((left + up) // 2)) & 0xFF
        elif filter_type == 4:
            out[i] = (out[i] + paeth(left, up, up_left)) & 0xFF
        else:
            raise ValueError(f"unsupported_filter:{filter_type}")
    return out


def paeth(left: int, up: int, up_left: int) -> int:
    p = left + up - up_left
    pa = abs(p - left)
    pb = abs(p - up)
    pc = abs(p - up_left)
    if pa <= pb and pa <= pc:
        return left
    if pb <= pc:
        return up
    return up_left


def normalize_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list | tuple) or len(value) != 4:
        return None
    try:
        vals = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in vals):
        return None
    x0, y0, x1, y1 = vals
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return [x0, y0, x1, y1]


def bbox_contains(left: list[float], right: list[float]) -> bool:
    return left[0] <= right[0] and left[1] <= right[1] and left[2] >= right[2] and left[3] >= right[3]


def extract_warning_node_ids(warnings: list[str]) -> set[str]:
    ids: set[str] = set()
    for warning in warnings:
        for part in str(warning).split(":")[1:]:
            token = re.split(r"[^A-Za-z0-9_.-]", part)[0]
            if token:
                ids.add(token)
    return ids


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a PNG image: {path}")
    width, height = struct.unpack(">II", header[16:24])
    return int(width), int(height)


def sample_id_for(row: dict[str, Any], index: int, config: dict[str, Any]) -> str:
    image = Path(str(row.get("image") or row.get("image_path") or f"sample_{index}"))
    parent = image.parent.name if image.parent.name else "unknown"
    prefix = str(config.get("sample_id_prefix") or row.get("source_dataset") or "sample")
    return f"{sanitize_id(prefix)}_{index:02d}_{sanitize_id(parent)}"


def short_node_label(node: dict[str, Any], config: dict[str, Any]) -> str:
    metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
    if str(node.get("family") or "") == "text" and metadata.get("text"):
        semantic = str(metadata.get("text"))
    else:
        semantic = str(node.get("semantic_type") or node.get("family") or "node")
    node_id = str(node.get("id") or "")
    if node_id.startswith("boundary_"):
        node_id = node_id.replace("boundary_", "b")
    elif node_id.startswith("svg_"):
        node_id = node_id.replace("svg_", "s")
    max_chars = int(label_policy(config).get("max_label_chars", 28))
    return f"{semantic}:{node_id}"[:max_chars]


def family_order(family: str, config: dict[str, Any]) -> int:
    return int(family_styles(config).get(family, family_styles(config)["unknown"]).get("order", 9))


def area(rect: list[float]) -> float:
    return max(0.0, rect[2]) * max(0.0, rect[3])


def bbox_xyxy_area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        config: dict[str, Any] = {}
    else:
        config = json.loads(path.read_text(encoding="utf-8"))
    config["_config_path"] = str(path)
    config.setdefault("family_styles", DEFAULT_FAMILY_STYLES)
    return config


def family_styles(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    styles = config.get("family_styles")
    if not isinstance(styles, dict):
        return DEFAULT_FAMILY_STYLES
    merged = {key: dict(value) for key, value in DEFAULT_FAMILY_STYLES.items()}
    for family, style in styles.items():
        if isinstance(style, dict):
            base = dict(merged.get(family, {}))
            base.update(style)
            merged[family] = base
    return merged


def legend_config(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("legend")
    return value if isinstance(value, dict) else {}


def label_policy(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("label_policy")
    return value if isinstance(value, dict) else {}


def canvas_policy(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("canvas_policy")
    return value if isinstance(value, dict) else {}


def image_alignment_policy(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("image_alignment_policy")
    return value if isinstance(value, dict) else {}


def background_policy(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("background_policy")
    return value if isinstance(value, dict) else {}


def suspicious_bbox_policy(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("suspicious_bbox_policy")
    return value if isinstance(value, dict) else {}


def review_pack_config(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("review_pack")
    return value if isinstance(value, dict) else {}


def boundary_overlay_audit(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("boundary_overlay_audit")
    return value if isinstance(value, dict) else {}


def sanitize_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "sample"


def relpath_for_html(path: Path, base: Path) -> str:
    return Path("../" + str(path.relative_to(base.parent))).as_posix() if path.is_relative_to(base.parent) else path.as_posix()


def escape_text(value: Any) -> str:
    return html.escape(str(value), quote=False)


def escape_attr(value: Any) -> str:
    return html.escape(str(value), quote=True)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
