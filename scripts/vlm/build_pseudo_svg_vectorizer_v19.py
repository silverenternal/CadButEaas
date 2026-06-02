#!/usr/bin/env python3
"""Build a raster-only pseudo-SVG vectorization bridge for MoE experiments.

This is a diagnostic branch: it converts PNG/raster pages into SVG-like paths
and detector candidates without using offline SVG/CAD geometry at inference.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import xml.sax.saxutils as xml_escape
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/vlm"
OUT = ROOT / "datasets/pseudo_svg_vectorizer_v19"


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def abs_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def source_integrity() -> dict[str, Any]:
    return {
        "source_mode": "image_only_raster_pseudo_svg",
        "svg_candidate_ids_used": False,
        "annotation_geometry_used_at_inference": False,
        "model_input": "raster_image_only",
        "runtime_uses_svg_or_cad_geometry": False,
    }


def bbox_iou(left: list[float], right: list[float]) -> float:
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    la = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    ra = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return inter / max(la + ra - inter, 1e-9)


def center_covered(pred: list[int], gold: list[int], margin: int = 2) -> bool:
    cx = (gold[0] + gold[2]) / 2.0
    cy = (gold[1] + gold[3]) / 2.0
    return pred[0] - margin <= cx <= pred[2] + margin and pred[1] - margin <= cy <= pred[3] + margin


def preprocess(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.medianBlur(gray, 3)
    binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = np.ones((2, 2), dtype=np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


def hough_boundary_candidates(binary: np.ndarray, row_id: str, image_path: str, max_candidates: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lines = cv2.HoughLinesP(binary, 1, np.pi / 180.0, threshold=24, minLineLength=12, maxLineGap=4)
    candidates: list[dict[str, Any]] = []
    paths: list[dict[str, Any]] = []
    if lines is None:
        return candidates, paths
    for index, line in enumerate(lines[:, 0].tolist()):
        x1, y1, x2, y2 = [int(v) for v in line]
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 8:
            continue
        pad = 2
        bbox = [max(0, min(x1, x2) - pad), max(0, min(y1, y2) - pad), min(binary.shape[1], max(x1, x2) + pad), min(binary.shape[0], max(y1, y2) + pad)]
        orientation = "horizontal" if abs(x2 - x1) >= abs(y2 - y1) else "vertical"
        confidence = round(min(1.0, length / 120.0), 6)
        path_id = f"{row_id}_pseudo_svg_line_{index:04d}"
        paths.append({"id": path_id, "kind": "line", "points": [[x1, y1], [x2, y2]], "bbox": bbox, "length": round(length, 3), "confidence": confidence})
        candidates.append(
            {
                "candidate_id": f"{row_id}_pseudo_svg_boundary_{index:04d}",
                "row_id": row_id,
                "family": "boundary",
                "route": "wall_opening",
                "candidate_type": "wall_or_vector_line",
                "bbox": bbox,
                "confidence": confidence,
                "payload": {
                    "image": image_path,
                    "raster_path": image_path,
                    "pseudo_svg_path_id": path_id,
                    "features": {"primitive_type": "hough_line", "length": round(length, 3), "orientation": orientation, "points": [[x1, y1], [x2, y2]]},
                    "proposal_source": "pseudo_svg_vectorizer_v19",
                    "source_integrity": source_integrity(),
                },
                "source_integrity": source_integrity(),
            }
        )
        if len(candidates) >= max_candidates:
            break
    return candidates, paths


def connected_component_candidates(binary: np.ndarray, row_id: str, image_path: str, max_text: int, max_symbol: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
    candidates: list[dict[str, Any]] = []
    blobs: list[dict[str, Any]] = []
    text_count = 0
    symbol_count = 0
    for idx in range(1, int(n_labels)):
        x, y, w, h, area = [int(v) for v in stats[idx]]
        if area < 3 or w <= 0 or h <= 0:
            continue
        bbox = [x, y, x + w, y + h]
        density = float(area) / max(w * h, 1)
        blob_id = f"{row_id}_pseudo_svg_blob_{idx:04d}"
        blob = {"id": blob_id, "kind": "component", "bbox": bbox, "area": area, "density": round(density, 6)}
        blobs.append(blob)
        is_text_like = area <= 260 and w <= 48 and h <= 24 and 0.05 <= density <= 0.95
        is_symbol_like = 12 <= area <= 1800 and 4 <= w <= 80 and 4 <= h <= 80 and not is_text_like
        if is_text_like and text_count < max_text:
            confidence = round(min(1.0, 0.35 + density), 6)
            candidates.append(
                {
                    "candidate_id": f"{row_id}_pseudo_svg_text_{text_count:04d}",
                    "row_id": row_id,
                    "family": "text",
                    "route": "text_dimension",
                    "candidate_type": "text_blob",
                    "bbox": bbox,
                    "confidence": confidence,
                    "payload": {
                        "raw_text": "",
                        "normalized_text": "",
                        "ocr_status": "not_invoked",
                        "pseudo_svg_path_id": blob_id,
                        "features": {"primitive_type": "connected_component", "area": area, "density": round(density, 6), "width": w, "height": h},
                        "proposal_source": "pseudo_svg_vectorizer_v19",
                        "source_integrity": source_integrity(),
                    },
                    "source_integrity": source_integrity(),
                }
            )
            text_count += 1
        elif is_symbol_like and symbol_count < max_symbol:
            confidence = round(min(1.0, 0.25 + density), 6)
            candidates.append(
                {
                    "candidate_id": f"{row_id}_pseudo_svg_symbol_{symbol_count:04d}",
                    "row_id": row_id,
                    "family": "symbol",
                    "route": "symbol_fixture",
                    "candidate_type": "pseudo_vector_blob",
                    "bbox": bbox,
                    "confidence": confidence,
                    "payload": {
                        "symbol_type": "unknown",
                        "pseudo_svg_path_id": blob_id,
                        "features": {"primitive_type": "connected_component", "area": area, "density": round(density, 6), "width": w, "height": h},
                        "proposal_source": "pseudo_svg_vectorizer_v19",
                        "source_integrity": source_integrity(),
                    },
                    "source_integrity": source_integrity(),
                }
            )
            symbol_count += 1
    return candidates, blobs


def contours_as_paths(binary: np.ndarray, row_id: str, max_paths: int) -> list[dict[str, Any]]:
    contours, _hier = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    paths: list[dict[str, Any]] = []
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_paths]
    for index, contour in enumerate(contours):
        epsilon = max(1.0, 0.01 * cv2.arcLength(contour, True))
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = [[int(p[0][0]), int(p[0][1])] for p in approx]
        if len(points) < 2:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        paths.append({"id": f"{row_id}_pseudo_svg_contour_{index:04d}", "kind": "contour", "points": points, "bbox": [x, y, x + w, y + h], "area": round(float(cv2.contourArea(contour)), 3), "confidence": 0.5})
    return paths


def svg_polyline(points: list[list[int]], attrs: str = "") -> str:
    point_text = " ".join(f"{x},{y}" for x, y in points)
    return f'<polyline points="{xml_escape.escape(point_text)}" fill="none" stroke="black" stroke-width="1" {attrs}/>'


def write_svg(path: Path, width: int, height: int, paths: list[dict[str, Any]], image_path: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" data-source-image="{xml_escape.escape(image_path)}" data-vectorizer="pseudo_svg_vectorizer_v19">',
    ]
    for item in paths:
        if "points" in item and len(item["points"]) >= 2:
            lines.append("  " + svg_polyline(item["points"], f'data-id="{xml_escape.escape(item["id"])}" data-kind="{xml_escape.escape(item["kind"])}"'))
        elif item.get("kind") == "component":
            x1, y1, x2, y2 = item["bbox"]
            lines.append(f'  <rect x="{x1}" y="{y1}" width="{x2 - x1}" height="{y2 - y1}" fill="none" stroke="red" stroke-width="0.5" data-id="{xml_escape.escape(item["id"])}" data-kind="component"/>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_vtracer_svg(image_path: str, output_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []

    if args.vtracer_backend in {"python", "auto"}:
        report = write_vtracer_svg_python(image_path, output_path, args)
        reports.append({"backend": "python", **report})
        if report.get("available"):
            return {"backend": "python", "attempts": reports, **report}
        if args.vtracer_backend == "python":
            return {"available": False, "backend": "python", "attempts": reports}

    if args.vtracer_backend in {"cli", "auto"}:
        report = write_vtracer_svg_cli(image_path, output_path, args)
        reports.append({"backend": "cli", **report})
        if report.get("available"):
            return {"backend": "cli", "attempts": reports, **report}
        return {"available": False, "backend": "cli", "attempts": reports}

    return {"available": False, "backend": args.vtracer_backend, "attempts": reports}


def write_vtracer_svg_python(image_path: str, output_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    code = """
import sys
import vtracer
vtracer.convert_image_to_svg_py(
    sys.argv[1],
    sys.argv[2],
    colormode=sys.argv[3],
    hierarchical=sys.argv[4],
    mode=sys.argv[5],
    filter_speckle=int(sys.argv[6]),
    color_precision=int(sys.argv[7]),
    layer_difference=int(sys.argv[8]),
    corner_threshold=int(sys.argv[9]),
    length_threshold=float(sys.argv[10]),
    max_iterations=int(sys.argv[11]),
    splice_threshold=int(sys.argv[12]),
    path_precision=int(sys.argv[13]),
)
"""
    cmd = [
        sys.executable,
        "-c",
        code,
        str(abs_path(image_path)),
        str(output_path),
        args.vtracer_colormode,
        args.vtracer_hierarchical,
        args.vtracer_mode,
        str(args.vtracer_filter_speckle),
        str(args.vtracer_color_precision),
        str(args.vtracer_layer_difference),
        str(args.vtracer_corner_threshold),
        str(args.vtracer_length_threshold),
        str(args.vtracer_max_iterations),
        str(args.vtracer_splice_threshold),
        str(args.vtracer_path_precision),
    ]
    result = subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=args.vtracer_timeout)
    if result.returncode != 0:
        return {"available": False, "returncode": result.returncode, "stderr": result.stderr[-500:]}
    return {"available": True, "path": str(output_path.relative_to(ROOT)), **count_svg_elements(output_path)}


def write_vtracer_svg_cli(image_path: str, output_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    binary = args.vtracer_cli or shutil.which("vtracer") or str(Path.home() / ".cargo/bin/vtracer")
    if not binary or not Path(binary).exists():
        return {"available": False, "error": "vtracer_cli_not_found", "searched": [args.vtracer_cli, shutil.which("vtracer"), str(Path.home() / ".cargo/bin/vtracer")]}
    colormode = "bw" if args.vtracer_colormode in {"binary", "bw"} else args.vtracer_colormode
    cmd = [
        binary,
        "--input",
        str(abs_path(image_path)),
        "--output",
        str(output_path),
        "--colormode",
        colormode,
        "--hierarchical",
        args.vtracer_hierarchical,
        "--mode",
        args.vtracer_mode,
        "--filter_speckle",
        str(args.vtracer_filter_speckle),
        "--color_precision",
        str(args.vtracer_color_precision),
        "--gradient_step",
        str(args.vtracer_layer_difference),
        "--corner_threshold",
        str(args.vtracer_corner_threshold),
        "--segment_length",
        str(args.vtracer_length_threshold),
        "--splice_threshold",
        str(args.vtracer_splice_threshold),
        "--path_precision",
        str(args.vtracer_path_precision),
    ]
    result = subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=args.vtracer_timeout)
    if result.returncode != 0:
        return {"available": False, "binary": binary, "returncode": result.returncode, "stderr": result.stderr[-500:], "stdout": result.stdout[-500:]}
    return {"available": True, "binary": binary, "path": str(output_path.relative_to(ROOT)), **count_svg_elements(output_path)}


def count_svg_elements(path: Path) -> dict[str, int]:
    try:
        root = ET.parse(path).getroot()
    except Exception:
        return {"svg_paths": 0, "svg_polylines": 0, "svg_rects": 0, "svg_elements": 0}
    counts = Counter()
    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        counts[tag] += 1
    return {
        "svg_paths": int(counts["path"]),
        "svg_polylines": int(counts["polyline"]),
        "svg_rects": int(counts["rect"]),
        "svg_elements": int(sum(counts.values())),
    }


def text_targets(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in row.get("text_targets") or [] if item.get("bbox") and len(item["bbox"]) == 4]


def evaluate_text_recall(row: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    golds = text_targets(row)
    text_candidates = [candidate for candidate in candidates if candidate["family"] == "text"]
    center_hits = 0
    iou_hits = 0
    for gold in golds:
        gb = [int(v) for v in gold["bbox"]]
        center_hits += int(any(center_covered(candidate["bbox"], gb) for candidate in text_candidates))
        iou_hits += int(any(bbox_iou(candidate["bbox"], gb) >= 0.30 for candidate in text_candidates))
    return {"gold": len(golds), "text_candidates": len(text_candidates), "center_hits": center_hits, "iou_hits": iou_hits}


def process_row(row: dict[str, Any], args: argparse.Namespace, out_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    row_id = str(row.get("source_row_id") or row.get("id"))
    image_path = str(row["image"])
    image = Image.open(abs_path(image_path)).convert("L")
    width, height = image.size
    gray = np.asarray(image, dtype=np.uint8)
    binary = preprocess(gray)
    boundary_candidates, line_paths = hough_boundary_candidates(binary, row_id, image_path, args.max_boundary_candidates)
    blob_candidates, blob_paths = connected_component_candidates(binary, row_id, image_path, args.max_text_candidates, args.max_symbol_candidates)
    contour_paths = contours_as_paths(binary, row_id, args.max_contour_paths)
    all_paths = line_paths + contour_paths + blob_paths
    svg_path = out_dir / "svg" / f"{row_id}.svg"
    vtracer_path = out_dir / "vtracer_svg" / f"{row_id}.svg"
    vtracer_report: dict[str, Any] = {"available": False, "skipped": True}
    if args.vectorizer in {"opencv", "both"}:
        write_svg(svg_path, width, height, all_paths, image_path)
    if args.vectorizer in {"vtracer", "both"}:
        vtracer_report = write_vtracer_svg(image_path, vtracer_path, args)
    candidates = boundary_candidates + blob_candidates
    predicted_symbols = [
        {
            "id": candidate["candidate_id"],
            "class": "symbol",
            "family": "symbol",
            "symbol_type": "unknown",
            "semantic_type": "unknown_symbol",
            "bbox": candidate["bbox"],
            "confidence": candidate["confidence"],
            "payload": candidate["payload"],
            "source_integrity": candidate["source_integrity"],
        }
        for candidate in candidates
        if candidate["family"] == "symbol"
    ]
    candidate_row = {
        "id": row_id,
        "image": image_path,
        "image_size": [width, height],
        "pseudo_svg": str(svg_path.relative_to(ROOT)) if svg_path.exists() else None,
        "vtracer_svg": str(vtracer_path.relative_to(ROOT)) if vtracer_path.exists() else None,
        "predicted_symbols": predicted_symbols,
        "source_integrity": source_integrity(),
        "route_trace": {"stage": "pseudo_svg_vectorizer_v19", **source_integrity()},
        "scene_graph": {"nodes": [], "relations": [], "candidate_stream": candidates},
    }
    path_row = {
        "id": row_id,
        "image": image_path,
        "image_size": [width, height],
        "pseudo_svg": str(svg_path.relative_to(ROOT)) if svg_path.exists() else None,
        "vtracer_svg": str(vtracer_path.relative_to(ROOT)) if vtracer_path.exists() else None,
        "vtracer_report": vtracer_report,
        "paths": all_paths,
        "source_integrity": source_integrity(),
    }
    text_eval = evaluate_text_recall(row, candidates)
    audit_row = {
        "id": row_id,
        "image": image_path,
        "pseudo_svg": str(svg_path.relative_to(ROOT)) if svg_path.exists() else None,
        "vtracer_svg": str(vtracer_path.relative_to(ROOT)) if vtracer_path.exists() else None,
        "vtracer_report": vtracer_report,
        "path_count": len(all_paths),
        "candidate_count": len(candidates),
        "family_counts": dict(Counter(candidate["family"] for candidate in candidates)),
        "text_eval": text_eval,
    }
    return candidate_row, path_row, audit_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="datasets/text_expert_raster_v19/locked.jsonl")
    parser.add_argument("--output-dir", default=str(OUT.relative_to(ROOT)))
    parser.add_argument("--audit", default="reports/vlm/pseudo_svg_vectorizer_v19_audit.json")
    parser.add_argument("--max-rows", type=int, default=20)
    parser.add_argument("--vectorizer", choices=["opencv", "vtracer", "both"], default="both")
    parser.add_argument("--max-boundary-candidates", type=int, default=1200)
    parser.add_argument("--max-text-candidates", type=int, default=220)
    parser.add_argument("--max-symbol-candidates", type=int, default=600)
    parser.add_argument("--max-contour-paths", type=int, default=500)
    parser.add_argument("--vtracer-colormode", default="binary")
    parser.add_argument("--vtracer-hierarchical", default="stacked")
    parser.add_argument("--vtracer-mode", default="polygon")
    parser.add_argument("--vtracer-filter-speckle", type=int, default=4)
    parser.add_argument("--vtracer-color-precision", type=int, default=6)
    parser.add_argument("--vtracer-layer-difference", type=int, default=16)
    parser.add_argument("--vtracer-corner-threshold", type=int, default=60)
    parser.add_argument("--vtracer-length-threshold", type=float, default=4.0)
    parser.add_argument("--vtracer-max-iterations", type=int, default=10)
    parser.add_argument("--vtracer-splice-threshold", type=int, default=45)
    parser.add_argument("--vtracer-path-precision", type=int, default=1)
    parser.add_argument("--vtracer-timeout", type=int, default=30)
    parser.add_argument("--vtracer-backend", choices=["auto", "python", "cli"], default="auto")
    parser.add_argument("--vtracer-cli", default="")
    args = parser.parse_args()

    rows = load_jsonl(abs_path(args.input))
    if args.max_rows:
        rows = rows[: args.max_rows]
    out_dir = abs_path(args.output_dir)
    candidate_rows = []
    path_rows = []
    audit_rows = []
    totals = Counter()
    for row in rows:
        candidate_row, path_row, audit_row = process_row(row, args, out_dir)
        candidate_rows.append(candidate_row)
        path_rows.append(path_row)
        audit_rows.append(audit_row)
        totals["rows"] += 1
        totals["paths"] += audit_row["path_count"]
        totals["vtracer_paths"] += int(audit_row["vtracer_report"].get("svg_paths") or 0)
        totals["vtracer_elements"] += int(audit_row["vtracer_report"].get("svg_elements") or 0)
        totals["candidates"] += audit_row["candidate_count"]
        for family, count in audit_row["family_counts"].items():
            totals[f"family_{family}"] += count
        text_eval = audit_row["text_eval"]
        totals["text_gold"] += text_eval["gold"]
        totals["text_candidates"] += text_eval["text_candidates"]
        totals["text_center_hits"] += text_eval["center_hits"]
        totals["text_iou_hits"] += text_eval["iou_hits"]

    write_jsonl(out_dir / "pseudo_svg_paths.jsonl", path_rows)
    write_jsonl(out_dir / "pseudo_svg_candidates.jsonl", candidate_rows)
    manifest = {
        "version": "pseudo_svg_vectorizer_v19",
        "task": "P0-PSEUDO-SVG-001",
        "purpose": "Raster-only PNG-to-pseudo-SVG bridge for testing whether SVG-scene strong experts can be reused on vectorized raster drawings.",
        "input": args.input,
        "output_dir": str(out_dir.relative_to(ROOT)),
        "artifacts": {
            "paths": str((out_dir / "pseudo_svg_paths.jsonl").relative_to(ROOT)),
            "candidates": str((out_dir / "pseudo_svg_candidates.jsonl").relative_to(ROOT)),
            "svg_dir": str((out_dir / "svg").relative_to(ROOT)),
            "audit": args.audit,
        },
        "method": "opencv_otsu_hough_connected_components plus optional VTracer SVG export",
        "vectorizer": args.vectorizer,
        "source_integrity": source_integrity(),
        "summary": {
            "rows": int(totals["rows"]),
            "paths": int(totals["paths"]),
            "vtracer_paths": int(totals["vtracer_paths"]),
            "vtracer_elements": int(totals["vtracer_elements"]),
            "candidates": int(totals["candidates"]),
            "family_counts": {
                "boundary": int(totals["family_boundary"]),
                "text": int(totals["family_text"]),
                "symbol": int(totals["family_symbol"]),
            },
            "text_gold": int(totals["text_gold"]),
            "text_candidate_inflation": round(totals["text_candidates"] / max(totals["text_gold"], 1), 6),
            "text_center_recall_ceiling": round(totals["text_center_hits"] / max(totals["text_gold"], 1), 6),
            "text_iou_0_30_recall_ceiling": round(totals["text_iou_hits"] / max(totals["text_gold"], 1), 6),
        },
        "row_audit": audit_rows,
    }
    write_json(out_dir / "manifest.json", manifest)
    write_json(abs_path(args.audit), manifest)
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
