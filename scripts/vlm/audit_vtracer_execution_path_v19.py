#!/usr/bin/env python3
"""Audit stable VTracer execution paths for the raster pseudo-SVG bridge."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/vlm"


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


def abs_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def bbox_iou(left: list[float], right: list[float]) -> float:
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    la = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    ra = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return inter / max(la + ra - inter, 1e-9)


def center_covered(pred: list[float], gold: list[float], margin: float = 2.0) -> bool:
    cx = (gold[0] + gold[2]) / 2.0
    cy = (gold[1] + gold[3]) / 2.0
    return pred[0] - margin <= cx <= pred[2] + margin and pred[1] - margin <= cy <= pred[3] + margin


def count_svg(path: Path) -> dict[str, int]:
    if not path.exists():
        return {"svg_paths": 0, "svg_polylines": 0, "svg_rects": 0, "svg_elements": 0}
    try:
        root = ET.parse(path).getroot()
    except Exception:
        return {"svg_paths": 0, "svg_polylines": 0, "svg_rects": 0, "svg_elements": 0}
    counts: dict[str, int] = {}
    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        counts[tag] = counts.get(tag, 0) + 1
    return {
        "svg_paths": counts.get("path", 0),
        "svg_polylines": counts.get("polyline", 0),
        "svg_rects": counts.get("rect", 0),
        "svg_elements": sum(counts.values()),
    }


def run_python_api(image: Path, out: Path, timeout: int) -> dict[str, Any]:
    out.parent.mkdir(parents=True, exist_ok=True)
    code = """
import sys
import vtracer
vtracer.convert_image_to_svg_py(
    sys.argv[1],
    sys.argv[2],
    colormode="binary",
    hierarchical="stacked",
    mode="polygon",
    filter_speckle=4,
    color_precision=6,
    layer_difference=16,
    corner_threshold=60,
    length_threshold=4.0,
    max_iterations=10,
    splice_threshold=45,
    path_precision=1,
)
"""
    result = subprocess.run([sys.executable, "-c", code, str(image), str(out)], cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    report: dict[str, Any] = {"returncode": result.returncode, "stdout_tail": result.stdout[-300:], "stderr_tail": result.stderr[-300:]}
    if result.returncode == 0:
        report.update({"available": True, "path": str(out.relative_to(ROOT)), **count_svg(out)})
    else:
        report.update({"available": False})
    return report


def run_cli(image: Path, out: Path, timeout: int, binary: str = "") -> dict[str, Any]:
    out.parent.mkdir(parents=True, exist_ok=True)
    exe = binary or shutil.which("vtracer") or str(Path.home() / ".cargo/bin/vtracer")
    if not exe or not Path(exe).exists():
        return {"available": False, "error": "vtracer_cli_not_found", "searched": [binary, shutil.which("vtracer"), str(Path.home() / ".cargo/bin/vtracer")]}
    cmd = [
        exe,
        "--input",
        str(image),
        "--output",
        str(out),
        "--colormode",
        "bw",
        "--hierarchical",
        "stacked",
        "--mode",
        "polygon",
        "--filter_speckle",
        "4",
        "--color_precision",
        "6",
        "--gradient_step",
        "16",
        "--corner_threshold",
        "60",
        "--segment_length",
        "4.0",
        "--splice_threshold",
        "45",
        "--path_precision",
        "1",
    ]
    result = subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    report: dict[str, Any] = {"binary": exe, "returncode": result.returncode, "stdout_tail": result.stdout[-300:], "stderr_tail": result.stderr[-300:]}
    if result.returncode == 0:
        report.update({"available": True, "path": str(out.relative_to(ROOT)), **count_svg(out)})
    else:
        report.update({"available": False})
    return report


def path_bboxes_from_svg(path: Path, image_size: list[int] | None) -> list[list[float]]:
    if not path.exists():
        return []
    root = ET.parse(path).getroot()
    bboxes: list[list[float]] = []
    for elem in root.iter():
        if elem.tag.split("}")[-1] != "path":
            continue
        d = elem.attrib.get("d") or ""
        nums = [float(item) for item in re.findall(r"-?\d+(?:\.\d+)?", d)]
        if len(nums) < 4:
            continue
        xs = nums[0::2]
        ys = nums[1::2]
        transform = elem.attrib.get("transform") or ""
        match = re.search(r"translate\((-?\d+(?:\.\d+)?)[, ]+(-?\d+(?:\.\d+)?)\)", transform)
        tx, ty = (float(match.group(1)), float(match.group(2))) if match else (0.0, 0.0)
        bbox = [min(xs) + tx, min(ys) + ty, max(xs) + tx, max(ys) + ty]
        if image_size:
            width, height = float(image_size[0]), float(image_size[1])
            bbox = [max(0.0, bbox[0]), max(0.0, bbox[1]), min(width, bbox[2]), min(height, bbox[3])]
        if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
            bboxes.append(bbox)
    return bboxes


def filtered_bboxes(bboxes: list[list[float]], image_size: list[int] | None) -> list[list[float]]:
    if not image_size:
        return bboxes
    page_area = max(float(image_size[0] * image_size[1]), 1.0)
    out = []
    for box in bboxes:
        w = box[2] - box[0]
        h = box[3] - box[1]
        if w * h > page_area * 0.25:
            continue
        if w > image_size[0] * 0.95 or h > image_size[1] * 0.95:
            continue
        out.append(box)
    return out


def gold_boundary(path: Path) -> dict[str, list[list[float]]]:
    gold: dict[str, list[list[float]]] = {}
    for row in load_jsonl(path):
        gold[str(row["id"])] = [
            [float(v) for v in item["bbox"]]
            for item in (row.get("targets") or {}).get("boxes") or []
            if item.get("bbox") and len(item["bbox"]) == 4
        ]
    return gold


def recall(rows: list[dict[str, Any]], predictions: dict[str, list[list[float]]], gold: dict[str, list[list[float]]]) -> dict[str, Any]:
    total = matched = predicted = 0
    for row in rows:
        row_id = str(row["id"])
        preds = predictions.get(row_id, [])
        predicted += len(preds)
        used: set[int] = set()
        for gold_box in gold.get(row_id, []):
            total += 1
            best = None
            best_iou = 0.0
            for index, pred in enumerate(preds):
                if index in used:
                    continue
                overlap = bbox_iou(pred, gold_box)
                if center_covered(pred, gold_box) or overlap >= 0.30:
                    if overlap >= best_iou:
                        best = index
                        best_iou = overlap
            if best is not None:
                used.add(best)
                matched += 1
    return {
        "gold": total,
        "predicted": predicted,
        "matched": matched,
        "center_or_iou_recall": round(matched / max(total, 1), 6),
        "candidate_inflation": round(predicted / max(total, 1), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="datasets/pseudo_svg_vectorizer_v19_locked/pseudo_svg_paths.jsonl")
    parser.add_argument("--gold-boundary", default="datasets/image_only_boundary_detector_v18/locked.jsonl")
    parser.add_argument("--out-dir", default="reports/vlm/vtracer_execution_path_v19")
    parser.add_argument("--audit", default="reports/vlm/vtracer_execution_path_v19_audit.json")
    parser.add_argument("--max-rows", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--vtracer-cli", default="")
    args = parser.parse_args()

    rows = load_jsonl(abs_path(args.input))[: args.max_rows]
    out_dir = abs_path(args.out_dir)
    gold = gold_boundary(abs_path(args.gold_boundary))
    python_predictions_all: dict[str, list[list[float]]] = {}
    python_predictions_filtered: dict[str, list[list[float]]] = {}
    cli_predictions_all: dict[str, list[list[float]]] = {}
    cli_predictions_filtered: dict[str, list[list[float]]] = {}
    opencv_predictions: dict[str, list[list[float]]] = {}
    row_reports: list[dict[str, Any]] = []

    for row in rows:
        row_id = str(row["id"])
        image = abs_path(row["image"])
        py_out = out_dir / "python_api" / f"{row_id}.svg"
        cli_out = out_dir / "cli" / f"{row_id}.svg"
        python_report = run_python_api(image, py_out, args.timeout)
        cli_report = run_cli(image, cli_out, args.timeout, args.vtracer_cli)
        python_bboxes = path_bboxes_from_svg(py_out, row.get("image_size"))
        cli_bboxes = path_bboxes_from_svg(cli_out, row.get("image_size"))
        python_predictions_all[row_id] = python_bboxes
        python_predictions_filtered[row_id] = filtered_bboxes(python_bboxes, row.get("image_size"))
        cli_predictions_all[row_id] = cli_bboxes
        cli_predictions_filtered[row_id] = filtered_bboxes(cli_bboxes, row.get("image_size"))
        opencv_predictions[row_id] = [[float(v) for v in item["bbox"]] for item in row.get("paths") or [] if item.get("bbox")]
        row_reports.append(
            {
                "id": row_id,
                "image": row["image"],
                "opencv_path_count": len(row.get("paths") or []),
                "python_api": python_report,
                "cli": cli_report,
                "python_bbox_count_all": len(python_predictions_all[row_id]),
                "python_bbox_count_filtered": len(python_predictions_filtered[row_id]),
                "cli_bbox_count_all": len(cli_predictions_all[row_id]),
                "cli_bbox_count_filtered": len(cli_predictions_filtered[row_id]),
            }
        )

    audit = {
        "version": "vtracer_execution_path_v19",
        "task": "P0-PSEUDO-SVG-001",
        "purpose": "Find a stable raster PNG to pseudo-SVG vectorizer path before reusing SVG/scene experts.",
        "source_integrity": {
            "source_mode": "image_only_raster_pseudo_svg",
            "runtime_uses_svg_or_cad_geometry": False,
            "annotation_geometry_used_at_inference": False,
        },
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "vtracer_cli": shutil.which("vtracer") or str(Path.home() / ".cargo/bin/vtracer"),
        },
        "inputs": {"paths": args.input, "gold_boundary": args.gold_boundary, "rows": len(rows)},
        "summary": {
            "python_api_success_rows": sum(1 for item in row_reports if item["python_api"].get("available")),
            "cli_success_rows": sum(1 for item in row_reports if item["cli"].get("available")),
            "opencv_total_paths": sum(item["opencv_path_count"] for item in row_reports),
            "python_api_total_svg_paths": sum(int(item["python_api"].get("svg_paths") or 0) for item in row_reports),
            "cli_total_svg_paths": sum(int(item["cli"].get("svg_paths") or 0) for item in row_reports),
            "python_api_total_bbox_all": sum(item["python_bbox_count_all"] for item in row_reports),
            "python_api_total_bbox_filtered": sum(item["python_bbox_count_filtered"] for item in row_reports),
            "cli_total_bbox_all": sum(item["cli_bbox_count_all"] for item in row_reports),
            "cli_total_bbox_filtered": sum(item["cli_bbox_count_filtered"] for item in row_reports),
            "opencv_boundary_bbox_recall_ceiling": recall(rows, opencv_predictions, gold),
            "vtracer_python_api_boundary_bbox_recall_ceiling_all_paths": recall(rows, python_predictions_all, gold),
            "vtracer_python_api_boundary_bbox_recall_ceiling_filtered_paths": recall(rows, python_predictions_filtered, gold),
            "vtracer_cli_boundary_bbox_recall_ceiling_all_paths": recall(rows, cli_predictions_all, gold),
            "vtracer_cli_boundary_bbox_recall_ceiling_filtered_paths": recall(rows, cli_predictions_filtered, gold),
        },
        "decision": "adopt_cli_for_export_only_not_semantic_candidates",
        "rationale": [
            "Rust CLI is stable in-process isolation and avoids the Python 3.14 wheel segfault.",
            "Raw VTracer paths are filled contours, not line/room semantics; semantic lifting still needs a learned or rule-based normalizer before scene-graph reuse.",
        ],
        "row_reports": row_reports,
    }
    write_json(abs_path(args.audit), audit)
    print(json.dumps(audit["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
