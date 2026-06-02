#!/usr/bin/env python3
"""Add auditable raster-derived symbol subcandidates to an existing adapter stream.

The script uses only the page raster and existing detector candidates. It does
not load gold labels. Gold can be used later by separate evaluation scripts.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from build_topology_relations_v18 import bbox, integrity
from nms_topology_relations_v18 import load_jsonl

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/vlm"

DEFAULT_INPUT = REPORT / "detector_adapter_v18_symbol_boundary_fixed_candidates.jsonl"
DEFAULT_OUTPUT = REPORT / "detector_adapter_v18_symbol_subcandidate_augmented.jsonl"
DEFAULT_AUDIT = REPORT / "detector_adapter_v18_symbol_subcandidate_augmented_audit.json"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_image(path: str | None) -> Path | None:
    if not path:
        return None
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def image_array(row: dict[str, Any], cache: dict[str, np.ndarray]) -> np.ndarray | None:
    image_path = resolve_image(row.get("image"))
    if image_path is None or not image_path.exists():
        return None
    key = str(image_path)
    if key not in cache:
        cache[key] = np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)
    return cache[key]


def bbox_iou(left: list[float], right: list[float]) -> float:
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    la = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    ra = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return inter / max(la + ra - inter, 1e-9)


def dark_component_boxes(crop: np.ndarray, threshold: int) -> list[dict[str, Any]]:
    import cv2

    mask = crop <= threshold
    n_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype("uint8"), 8)
    out: list[dict[str, Any]] = []
    for idx in range(1, int(n_labels)):
        x, y, w, h, area = [int(v) for v in stats[idx]]
        if area < 3 or w < 2 or h < 2:
            continue
        if w > 44 or h > 30:
            continue
        fill = area / max(w * h, 1)
        if fill < 0.06:
            continue
        out.append({"bbox": [x, y, x + w, y + h], "area": area, "fill": fill})
    out.sort(key=lambda item: (item["area"], item["fill"]), reverse=True)
    return out


def make_child(parent: dict[str, Any], child_box: list[int], component: dict[str, Any], index: int) -> dict[str, Any]:
    row_id = str(parent.get("row_id"))
    parent_id = str(parent.get("candidate_id"))
    confidence = float(parent.get("confidence") or 0.0)
    fill = float(component.get("fill") or 0.0)
    area = float(component.get("area") or 0.0)
    child_id = f"{parent_id}_subcc_{index:02d}_{child_box[0]}_{child_box[1]}_{child_box[2]}_{child_box[3]}"
    payload = dict(parent.get("payload") if isinstance(parent.get("payload"), dict) else {})
    payload.update(
        {
            "candidate_kind": "symbol_subcandidate_connected_component",
            "source": "raster_symbol_subcandidate_fanout_v18",
            "parent_candidate_id": parent_id,
            "component_area": int(area),
            "component_fill": round(fill, 6),
            "type_label_adopted": False,
            "objectness_score": round(min(0.99, confidence * (0.72 + min(fill, 1.0) * 0.18)), 6),
        }
    )
    return {
        "candidate_id": child_id,
        "candidate_contract_version": parent.get("candidate_contract_version") or "detector_candidate_contract_v1",
        "row_id": row_id,
        "family": "symbol",
        "route": parent.get("route") or "symbol_fixture",
        "candidate_type": parent.get("candidate_type") or "symbol",
        "bbox": child_box,
        "confidence": round(min(0.99, confidence * (0.70 + min(fill, 1.0) * 0.20)), 6),
        "payload": payload,
        "source_integrity": integrity(),
        "provenance": {
            "input_source": "raster_symbol_subcandidate_fanout_v18",
            "raw_candidate_id": child_id,
            "parent_candidate_id": parent_id,
            "row_id": row_id,
            "family": "symbol",
            "route": parent.get("route") or "symbol_fixture",
            "image": (parent.get("provenance") or {}).get("image"),
        },
        "audit_trace": {
            "stage": "symbol_subcandidate_fanout_v18",
            "parent_candidate_id": parent_id,
            "parent_bbox": parent.get("bbox"),
            "component_area": int(area),
            "component_fill": round(fill, 6),
            "source_integrity": integrity(),
        },
    }


def parent_is_eligible(candidate: dict[str, Any], min_area: float) -> bool:
    if candidate.get("family") != "symbol":
        return False
    b = bbox(candidate.get("bbox"))
    if b is None:
        return False
    payload = candidate.get("payload") if isinstance(candidate.get("payload"), dict) else {}
    kind = payload.get("candidate_kind")
    area = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return area >= min_area and kind in {"dark_pixel_anchor", "dark_connected_component"}


def child_candidates_for_parent(
    row: dict[str, Any],
    parent: dict[str, Any],
    arr: np.ndarray,
    threshold: int,
    pad: int,
    max_children: int,
) -> list[dict[str, Any]]:
    b = bbox(parent.get("bbox"))
    if b is None:
        return []
    height, width = arr.shape
    x1, y1, x2, y2 = [int(round(v)) for v in b]
    crop = arr[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)]
    if crop.size == 0:
        return []
    children: list[dict[str, Any]] = []
    parent_area = max(1.0, (x2 - x1) * (y2 - y1))
    for component in dark_component_boxes(crop, threshold=threshold):
        cx1, cy1, cx2, cy2 = component["bbox"]
        child_box = [
            max(0, x1 + cx1 - pad),
            max(0, y1 + cy1 - pad),
            min(width, x1 + cx2 + pad),
            min(height, y1 + cy2 + pad),
        ]
        child_area = max(1.0, (child_box[2] - child_box[0]) * (child_box[3] - child_box[1]))
        if child_area >= parent_area * 0.78:
            continue
        if bbox_iou(child_box, [x1, y1, x2, y2]) > 0.84:
            continue
        if any(bbox_iou(child_box, existing["bbox"]) > 0.75 for existing in children):
            continue
        children.append(make_child(parent, child_box, component, len(children)))
        if len(children) >= max_children:
            break
    return children


def augment(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = load_jsonl(Path(args.input))
    image_cache: dict[str, np.ndarray] = {}
    out_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    per_row_added: dict[str, int] = {}
    source_kinds: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []

    for row in rows:
        row_id = str(row.get("id"))
        arr = image_array(row, image_cache)
        stream = list(((row.get("scene_graph") or {}).get("candidate_stream") or []))
        counts["input_candidates"] += len(stream)
        if arr is None:
            counts["missing_image_rows"] += 1
            out_rows.append(row)
            continue
        added: list[dict[str, Any]] = []
        for candidate in stream:
            if not parent_is_eligible(candidate, min_area=float(args.min_parent_area)):
                continue
            payload = candidate.get("payload") if isinstance(candidate.get("payload"), dict) else {}
            source_kinds[str(payload.get("candidate_kind") or "unknown")] += 1
            children = child_candidates_for_parent(
                row,
                candidate,
                arr,
                threshold=int(args.threshold),
                pad=int(args.pad),
                max_children=int(args.max_children_per_parent),
            )
            for child in children:
                if any(bbox_iou(child["bbox"], existing.get("bbox") or []) > 0.90 for existing in stream + added):
                    continue
                added.append(child)
                if len(examples) < 50:
                    examples.append(
                        {
                            "row_id": row_id,
                            "parent_candidate_id": candidate.get("candidate_id"),
                            "child_candidate_id": child.get("candidate_id"),
                            "parent_bbox": candidate.get("bbox"),
                            "child_bbox": child.get("bbox"),
                        }
                    )
                if len(added) >= int(args.max_added_per_row):
                    break
            if len(added) >= int(args.max_added_per_row):
                break
        new_row = dict(row)
        new_scene = dict(new_row.get("scene_graph") if isinstance(new_row.get("scene_graph"), dict) else {})
        new_stream = stream + added
        new_scene["candidate_stream"] = new_stream
        counts["added_symbol_subcandidates"] += len(added)
        counts["output_candidates"] += len(new_stream)
        candidate_counts = dict(new_scene.get("candidate_counts") if isinstance(new_scene.get("candidate_counts"), dict) else {})
        candidate_counts["symbol"] = sum(1 for item in new_stream if item.get("family") == "symbol")
        new_scene["candidate_counts"] = candidate_counts
        new_scene["symbol_subcandidate_fanout_v18"] = {
            "enabled": True,
            "added_symbol_subcandidates": len(added),
            "threshold": int(args.threshold),
            "max_children_per_parent": int(args.max_children_per_parent),
            "max_added_per_row": int(args.max_added_per_row),
        }
        new_row["scene_graph"] = new_scene
        out_rows.append(new_row)
        per_row_added[row_id] = len(added)

    audit = {
        "task": "IMG-MOE-V18-REBUILD-001.step_g_symbol_subcandidate_fanout",
        "input": str(args.input),
        "output": str(args.output),
        "rows": len(rows),
        "counts": dict(counts),
        "source_candidate_kind_counts": dict(source_kinds),
        "added_per_row_histogram": dict(Counter(per_row_added.values())),
        "worst_rows": [{"row_id": row_id, "added": count} for row_id, count in sorted(per_row_added.items(), key=lambda item: item[1], reverse=True)[:50]],
        "examples": examples,
        "source_integrity": integrity(),
        "gold_loaded": False,
        "gold_used_for_inference": False,
    }
    return out_rows, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--threshold", type=int, default=190)
    parser.add_argument("--pad", type=int, default=2)
    parser.add_argument("--min-parent-area", type=float, default=220.0)
    parser.add_argument("--max-children-per-parent", type=int, default=3)
    parser.add_argument("--max-added-per-row", type=int, default=300)
    args = parser.parse_args()

    rows, audit = augment(args)
    write_jsonl(Path(args.output), rows)
    write_json(Path(args.audit), audit)
    print(json.dumps(audit["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
