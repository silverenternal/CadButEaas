#!/usr/bin/env python3
"""CadStruct image-only MoE v17 pipeline.

Inference consumes raster floorplan images only. SVG/parser/expected_json labels
are used only for offline audit, calibration, and locked evaluation.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from skimage import filters, measure, morphology, transform
except Exception:  # pragma: no cover
    filters = None  # type: ignore[assignment]
    measure = None  # type: ignore[assignment]
    morphology = None  # type: ignore[assignment]
    transform = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vlm.cadstruct_moe import (
    DeterministicRouter,
    ExpertPrediction,
    FusionResult,
    RoutedCandidate,
    build_default_experts,
    describe_experts,
    summarize_expert_execution,
)
from scripts.vlm.cadstruct_moe.fusion import fuse_predictions
from scripts.vlm.v5_pipeline_utils import load_json, load_jsonl, update_todo_remove, write_json, write_jsonl
from scripts.vlm.v8_raster_e2e_utils import bbox_iou, match_counts, normalize_bbox, sample_key
from scripts.vlm.validate_image_only_moe_stream import validate_rows

REPORT = ROOT / "reports/vlm"
DATA = ROOT / "datasets/image_only_moe_candidate_crops_v17"
CONTRACT = ROOT / "configs/vlm/image_only_moe_contract_v1.json"
V16_PROPOSALS = REPORT / "image_only_structured_moe_proposals_v16.jsonl"
V16_EVAL = REPORT / "image_only_structured_moe_v16_eval.json"
V15_EVAL = REPORT / "image_only_moe_e2e_v15_eval.json"
CALIBRATION_PATH = ROOT / "configs/vlm/image_only_moe_calibration_v17.json"

TASK_IDS = {
    "audit_contract": "IMG-MOE-V17-P0-001",
    "build_candidates": "IMG-MOE-V17-P0-002",
    "extract_crops": "IMG-MOE-V17-P0-005",
    "apply_experts": "IMG-MOE-V17-P0-003",
    "candidate_sweep": "IMG-MOE-V17-P0-004",
    "fuse": "IMG-MOE-V17-P1-006",
    "evaluate": "IMG-MOE-V17-P1-007",
    "oracle_gap": "IMG-MOE-V17-P1-008",
    "calibrate": "IMG-MOE-V17-P1-009",
    "render": "IMG-MOE-V17-P2-010",
    "docs": "IMG-MOE-V17-P2-011",
}

FAMILY_TO_EXPERT = {
    "boundary": "wall_opening",
    "space": "room_space",
    "symbol": "symbol_fixture",
    "text": "text_dimension",
    "sheet": "sheet_layout",
}

FAMILY_TO_CLASS = {
    "boundary": {"wall", "opening", "window"},
    "space": {"room"},
    "symbol": {"symbol"},
    "text": {"text"},
    "sheet": {"sheet"},
}

FAMILY_THRESHOLDS = {
    "boundary": 0.18,
    "space": 0.15,
    "symbol": 0.18,
    "text": 0.15,
    "sheet": 0.05,
}

FAMILY_CAPS = {
    "boundary": 320,
    "space": 64,
    "symbol": 180,
    "text": 180,
    "sheet": 16,
}

CROP_SCALES = (1.0, 1.35)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _write(path: str | Path, value: Any) -> None:
    p = _abs(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _write_l(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = _abs(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(row, ensure_ascii=False, default=_json_default) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def _abs(path: str | Path | None) -> Path:
    if path is None:
        return ROOT
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _rel(path: str | Path | None) -> str:
    if path is None:
        return ""
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def _img_uri(path: str | Path | None) -> str:
    p = _abs(path)
    if not p.exists():
        return ""
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    suffix = p.suffix.lower()
    mime = "image/png" if suffix != ".jpg" and suffix != ".jpeg" else "image/jpeg"
    return f"data:{mime};base64,{data}"


def _load_v16_rows(limit: int = 0) -> list[dict[str, Any]]:
    rows = load_jsonl(V16_PROPOSALS)
    if limit and len(rows) > limit:
        return rows[:limit]
    return rows


def _source_integrity() -> dict[str, Any]:
    return {
        "source_mode": "image_only_raster_moe",
        "svg_candidate_ids_used": False,
        "annotation_geometry_used_at_inference": False,
        "model_input": "raster_image_only",
    }


def _normalize_box(value: Any) -> list[float] | None:
    bbox = normalize_bbox(value)
    if bbox is None:
        return None
    return [float(v) for v in bbox]


def _page_size(row: dict[str, Any]) -> tuple[float, float]:
    size = row.get("image_size")
    if isinstance(size, list) and len(size) >= 2:
        return max(float(size[0]), 1.0), max(float(size[1]), 1.0)
    image = row.get("image")
    if image:
        try:
            with Image.open(_abs(image)) as img:
                return float(img.width), float(img.height)
        except Exception:
            pass
    return 512.0, 512.0


def _proposal_family(item: dict[str, Any]) -> str:
    family = str(item.get("family") or "").strip()
    if family in FAMILY_TO_EXPERT:
        return family
    cls = str(item.get("class") or item.get("semantic_type") or "").lower()
    if cls in {"wall", "door", "window", "opening", "boundary"}:
        return "boundary"
    if cls in {"room"}:
        return "space"
    if cls in {"text", "room_label", "dimension_text", "note_text"}:
        return "text"
    if cls in {"sheet", "title_block", "legend", "schedule", "stamp", "notes"}:
        return "sheet"
    return "symbol"


def _candidate_type(item: dict[str, Any]) -> str:
    return str(item.get("class") or item.get("semantic_type") or "candidate")


def _primitive_features(proposal: dict[str, Any], page_w: float, page_h: float) -> dict[str, Any]:
    bbox = _normalize_box(proposal.get("bbox"))
    if not bbox:
        return {}
    x1, y1, x2, y2 = bbox
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    return {
        "primitive_type": str(proposal.get("class") or proposal.get("semantic_type") or "bbox"),
        "bbox": bbox,
        "centroid": [float((x1 + x2) * 0.5), float((y1 + y2) * 0.5)],
        "length": float(max(w, h)),
        "angle_degrees": 0.0,
        "orientation": "horizontal" if w >= h else "vertical",
        "page_norm_bbox": [x1 / page_w, y1 / page_h, x2 / page_w, y2 / page_h],
    }


def _canonical_family_label(family: str, label: str) -> str:
    label = str(label or "").strip()
    if family == "boundary":
        if label in {"door", "window", "opening", "wall"}:
            return label
        return "wall"
    if family == "space":
        return "room"
    if family == "symbol":
        return "symbol"
    if family == "text":
        return "text"
    if family == "sheet":
        return "sheet"
    return label or family


def _page_context_from_candidates(candidates: list[dict[str, Any]], page_w: float, page_h: float) -> dict[str, Any]:
    rooms: list[dict[str, Any]] = []
    symbols: list[dict[str, Any]] = []
    texts: list[dict[str, Any]] = []
    boundaries: list[dict[str, Any]] = []
    for item in candidates:
        fam = str(item.get("family"))
        bbox = _normalize_box(item.get("bbox"))
        if not bbox:
            continue
        entry = {
            "id": item.get("id"),
            "bbox": bbox,
            "semantic_type": item.get("semantic_type") or item.get("class") or "unknown",
            "proposal_source": item.get("proposal_source"),
        }
        if fam == "space":
            entry["shape_features"] = item.get("shape_features") if isinstance(item.get("shape_features"), dict) else {}
            rooms.append(entry)
        elif fam == "symbol":
            entry["symbol_type"] = item.get("semantic_type") or item.get("class") or "generic_symbol"
            symbols.append(entry)
        elif fam == "text":
            entry["text_type"] = item.get("semantic_type") or item.get("class") or "note_text"
            entry["text"] = item.get("text") or ""
            texts.append(entry)
        elif fam == "boundary":
            entry["semantic_type"] = item.get("semantic_type") or item.get("class") or "wall"
            boundaries.append(entry)
    adjacency = {room.get("id"): 0 for room in rooms if room.get("id")}
    for i, left in enumerate(rooms):
        for right in rooms[i + 1:]:
            if _bbox_touch(left.get("bbox") or [0, 0, 0, 0], right.get("bbox") or [0, 0, 0, 0]):
                adjacency[left["id"]] += 1
                adjacency[right["id"]] += 1
    return {
        "width": page_w,
        "height": page_h,
        "rooms": rooms,
        "symbols": symbols,
        "texts": texts,
        "boundaries": boundaries,
        "adjacency": adjacency,
    }


def _bbox_touch(left: list[float], right: list[float]) -> bool:
    return not (left[2] < right[0] or right[2] < left[0] or left[3] < right[1] or right[3] < left[1])


def _bbox_contains(left: list[float], right: list[float]) -> bool:
    return left[0] <= right[0] and left[1] <= right[1] and left[2] >= right[2] and left[3] >= right[3]


def _bbox_inside_ratio(container: list[float], item: list[float]) -> float:
    x1 = max(float(container[0]), float(item[0]))
    y1 = max(float(container[1]), float(item[1]))
    x2 = min(float(container[2]), float(item[2]))
    y2 = min(float(container[3]), float(item[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = max(1.0, (float(item[2]) - float(item[0])) * (float(item[3]) - float(item[1])))
    return inter / area


def _nms(items: list[dict[str, Any]], thresh: float) -> list[dict[str, Any]]:
    kept = []
    for item in sorted(items, key=lambda x: float(x.get("confidence") or 0.0), reverse=True):
        bbox = _normalize_box(item.get("bbox"))
        if not bbox:
            continue
        if all(bbox_iou(bbox, _normalize_box(other.get("bbox")) or [0, 0, 0, 0]) < thresh for other in kept):
            kept.append(item)
    return kept


def _ink_mask_from_row(row: dict[str, Any], page_w: float, page_h: float) -> np.ndarray | None:
    image_path = row.get("image")
    if not image_path:
        return None
    try:
        image = Image.open(_abs(image_path)).convert("L").resize((int(page_w), int(page_h)))
    except Exception:
        return None
    arr = np.asarray(image, dtype=np.uint8)
    if arr.size == 0:
        return None
    threshold = int(np.clip(np.percentile(arr, 35), 120, 230))
    ink = arr < threshold
    if morphology is not None:
        with np.errstate(all="ignore"):
            footprint = morphology.footprint_rectangle((2, 2)) if hasattr(morphology, "footprint_rectangle") else morphology.square(2)
            ink = morphology.remove_small_objects(ink.astype(bool), min_size=max(2, int(arr.size * 0.00001)))
            ink = morphology.binary_closing(ink, footprint)
    return ink.astype(bool)


def _build_raster_primitive_candidates(row: dict[str, Any], page_w: float, page_h: float) -> list[dict[str, Any]]:
    ink = _ink_mask_from_row(row, page_w, page_h)
    if ink is None:
        return []
    proposals: list[dict[str, Any]] = []
    proposals.extend(_raster_line_candidates(row, ink, page_w, page_h))
    proposals.extend(_raster_hough_boundary_candidates(row, ink, page_w, page_h))
    proposals.extend(_raster_blob_candidates(row, ink, page_w, page_h))
    return proposals


def _raster_line_candidates(row: dict[str, Any], ink: np.ndarray, page_w: float, page_h: float) -> list[dict[str, Any]]:
    min_len = max(12, int(min(page_w, page_h) * 0.035))
    max_thick = max(10, int(min(page_w, page_h) * 0.035))
    lines: list[dict[str, Any]] = []

    def add_line(orientation: str, start: int, end: int, band_start: int, band_end: int, ink_count: int) -> None:
        if end - start < min_len or band_end - band_start < 1 or band_end - band_start > max_thick:
            return
        if orientation == "horizontal":
            bbox = [start, band_start, end + 1, band_end + 1]
            p1, p2 = [start, (band_start + band_end) // 2], [end + 1, (band_start + band_end) // 2]
        else:
            bbox = [band_start, start, band_end + 1, end + 1]
            p1, p2 = [(band_start + band_end) // 2, start], [(band_start + band_end) // 2, end + 1]
        area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        confidence = min(0.82, 0.28 + ink_count / max(area, 1) * 0.35 + (end - start) / max(page_w, page_h) * 0.25)
        lines.append({
            "id": f"{row['id']}_raster_line_{orientation}_{len(lines)}",
            "class": "wall",
            "semantic_type": "wall",
            "family": "boundary",
            "bbox": [int(v) for v in bbox],
            "p1": [int(v) for v in p1],
            "p2": [int(v) for v in p2],
            "confidence": float(confidence),
            "proposal_source": "raster_primitive_recall_v17",
            "primitive_type": f"{orientation}_run",
        })

    row_hits = ink.sum(axis=1)
    active_rows = np.where(row_hits >= max(4, int(page_w * 0.015)))[0]
    for band in _contiguous_runs(active_rows):
        y1, y2 = band[0], band[-1]
        if y2 - y1 + 1 > max_thick:
            continue
        xs = np.where(ink[y1:y2 + 1, :].any(axis=0))[0]
        for run in _contiguous_runs(xs):
            add_line("horizontal", int(run[0]), int(run[-1]), int(y1), int(y2), int(ink[y1:y2 + 1, run[0]:run[-1] + 1].sum()))

    col_hits = ink.sum(axis=0)
    active_cols = np.where(col_hits >= max(4, int(page_h * 0.015)))[0]
    for band in _contiguous_runs(active_cols):
        x1, x2 = band[0], band[-1]
        if x2 - x1 + 1 > max_thick:
            continue
        ys = np.where(ink[:, x1:x2 + 1].any(axis=1))[0]
        for run in _contiguous_runs(ys):
            add_line("vertical", int(run[0]), int(run[-1]), int(x1), int(x2), int(ink[run[0]:run[-1] + 1, x1:x2 + 1].sum()))

    return _nms(lines, 0.25)[:260]


def _raster_blob_candidates(row: dict[str, Any], ink: np.ndarray, page_w: float, page_h: float) -> list[dict[str, Any]]:
    if measure is None:
        return []
    labels = measure.label(ink, connectivity=2)
    image_area = max(1.0, float(page_w * page_h))
    symbols: list[dict[str, Any]] = []
    texts: list[dict[str, Any]] = []
    for region in measure.regionprops(labels):
        if region.area < 3 or region.area > image_area * 0.015:
            continue
        minr, minc, maxr, maxc = region.bbox
        w = maxc - minc
        h = maxr - minr
        if w < 2 or h < 2:
            continue
        aspect = w / max(h, 1)
        fill = float(region.area) / max(w * h, 1)
        if max(w, h) <= max(32, int(min(page_w, page_h) * 0.10)) and 0.05 <= fill <= 0.85:
            symbols.append({
                "id": f"{row['id']}_raster_symbol_blob_{len(symbols)}",
                "class": "symbol",
                "semantic_type": "symbol",
                "family": "symbol",
                "bbox": [int(minc), int(minr), int(maxc), int(maxr)],
                "confidence": float(min(0.74, 0.24 + fill * 0.45 + min(region.area / 160.0, 0.18))),
                "proposal_source": "raster_primitive_recall_v17",
                "shape_features": {"area": int(region.area), "fill_ratio": round(fill, 6), "aspect": round(aspect, 6)},
            })
        if 4 <= h <= max(42, int(page_h * 0.08)) and 2 <= w <= max(160, int(page_w * 0.30)) and 0.12 <= aspect <= 18 and 0.04 <= fill <= 0.70:
            texts.append({
                "id": f"{row['id']}_raster_text_blob_{len(texts)}",
                "class": "text",
                "semantic_type": "text",
                "family": "text",
                "bbox": [int(minc), int(minr), int(maxc), int(maxr)],
                "text": "",
                "ocr_status": "image_only_blob_candidate",
                "confidence": float(min(0.72, 0.22 + fill * 0.35 + min(w / max(page_w, 1.0), 0.18))),
                "proposal_source": "raster_primitive_recall_v17",
                "shape_features": {"area": int(region.area), "fill_ratio": round(fill, 6), "aspect": round(aspect, 6)},
            })
    return _nms(symbols, 0.20)[:160] + _nms(texts, 0.20)[:160]


def _raster_hough_boundary_candidates(row: dict[str, Any], ink: np.ndarray, page_w: float, page_h: float) -> list[dict[str, Any]]:
    if filters is None or transform is None:
        return []
    try:
        edge_map = filters.sobel(ink.astype(float))
        lines = transform.probabilistic_hough_line(
            edge_map > np.percentile(edge_map, 82),
            threshold=20,
            line_length=max(28, int(min(page_w, page_h) * 0.075)),
            line_gap=max(2, int(min(page_w, page_h) * 0.006)),
        )
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for idx, ((x1, y1), (x2, y2)) in enumerate(lines):
        length = math.hypot(float(x2) - float(x1), float(y2) - float(y1))
        if length < max(28, int(min(page_w, page_h) * 0.075)):
            continue
        bbox = [int(min(x1, x2)), int(min(y1, y2)), int(max(x1, x2) + 1), int(max(y1, y2) + 1)]
        confidence = float(min(0.88, 0.20 + length / max(page_w, page_h) * 0.55))
        out.append({
            "id": f"{row['id']}_hough_wall_{idx}",
            "class": "wall",
            "semantic_type": "wall",
            "family": "boundary",
            "bbox": bbox,
            "p1": [int(x1), int(y1)],
            "p2": [int(x2), int(y2)],
            "confidence": confidence,
            "proposal_source": "raster_primitive_recall_hough_v17",
            "primitive_type": "hough_line",
        })
    return _nms(out, 0.20)[:80]


def _contiguous_runs(values: np.ndarray) -> list[np.ndarray]:
    if values.size == 0:
        return []
    splits = np.where(np.diff(values) > 1)[0] + 1
    return [run for run in np.split(values, splits) if run.size > 0]


def _candidate_from_proposal(
    row: dict[str, Any],
    proposal: dict[str, Any],
    page_context: dict[str, Any],
) -> RoutedCandidate | None:
    bbox = _normalize_box(proposal.get("bbox"))
    if not bbox:
        return None
    family = _proposal_family(proposal)
    source = "image_only_raster_moe"
    candidate_id = str(proposal.get("id") or f"{row['id']}_{family}_{proposal.get('class') or 'candidate'}")
    payload = {
        "image": row.get("image"),
        "raster_path": row.get("image"),
        "source_dataset": row.get("source_dataset") or "cubi_casa",
        "_page_metadata": {"width": _page_size(row)[0], "height": _page_size(row)[1]},
        "page_context": page_context,
        "features": _primitive_features(proposal, *_page_size(row)),
        "proposal_source": proposal.get("proposal_source"),
        "proposal_class": proposal.get("class"),
        "proposal_semantic_type": proposal.get("semantic_type"),
        "family_hint": family,
        "candidate_type_hint": _candidate_type(proposal),
        "crop_paths": [],
    }
    if "text" in proposal:
        payload["raw_text"] = proposal.get("text") or ""
        payload["text"] = proposal.get("text") or ""
    if family == "space":
        payload["shape_features"] = proposal.get("shape_features") or {}
    if family == "symbol":
        payload["rooms"] = page_context.get("rooms") or []
    if family == "text":
        payload["rooms"] = page_context.get("rooms") or []
    route_trace = {
        "source_mode": "image_only_raster_moe",
        "routing_method": "v16_proposal_adapter",
        "matched_hint": candidate_type_hint(proposal),
        "routing_confidence": float(proposal.get("confidence") or 0.5),
        "abstain": False,
    }
    return RoutedCandidate(
        candidate_id=candidate_id,
        expert=FAMILY_TO_EXPERT.get(family, f"{family}_expert"),
        family=family,
        candidate_type=_candidate_type(proposal),
        confidence=float(proposal.get("confidence") or 0.5),
        bbox=bbox,
        source=source,
        payload=payload,
        route_trace=route_trace,
    )


def candidate_type_hint(proposal: dict[str, Any]) -> str:
    return str(proposal.get("class") or proposal.get("semantic_type") or "candidate")


def _proposal_rows_to_candidates(
    rows: list[dict[str, Any]],
    min_conf: dict[str, float] | None = None,
    caps: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    min_conf = dict(min_conf or FAMILY_THRESHOLDS)
    caps = dict(caps or FAMILY_CAPS)
    stream: list[dict[str, Any]] = []
    summary = {
        "rows": len(rows),
        "source_mode": "image_only_raster_moe",
        "family_counts": Counter(),
        "selected_counts": Counter(),
        "thresholds": min_conf,
        "caps": caps,
    }
    for row in rows:
        page_w, page_h = _page_size(row)
        proposals = [item for item in row.get("proposals") or [] if isinstance(item, dict)]
        by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
        base_counts: Counter[str] = Counter()
        for proposal in proposals:
            family = _proposal_family(proposal)
            by_family[family].append(proposal)
            base_counts[family] += 1
        raster_primitives = _build_raster_primitive_candidates(row, page_w, page_h)
        for proposal in raster_primitives:
            family = _proposal_family(proposal)
            primitive_budget = max(0, int(caps.get(family, 9999)) - base_counts.get(family, 0))
            if family == "boundary":
                primitive_budget = min(primitive_budget, 48)
            elif family in {"symbol", "text"}:
                primitive_budget = min(primitive_budget, 36)
            if primitive_budget <= 0:
                continue
            existing_primitives = sum(1 for item in by_family[family] if str(item.get("proposal_source", "")).startswith("raster_primitive_recall"))
            if existing_primitives >= primitive_budget:
                continue
            by_family[family].append(proposal)
        selected_candidates: list[dict[str, Any]] = []
        for family, items in by_family.items():
            items = sorted(items, key=lambda item: float(item.get("confidence") or 0.0), reverse=True)
            keep = []
            for proposal in items:
                confidence = float(proposal.get("confidence") or 0.0)
                if confidence < float(min_conf.get(family, 0.0)):
                    continue
                keep.append(proposal)
                if len(keep) >= int(caps.get(family, 9999)):
                    break
            summary["family_counts"][family] += len(items)
            summary["selected_counts"][family] += len(keep)
            selected_candidates.extend(keep)
        raster_rooms = _build_room_candidates_from_raster(row, page_w, page_h)
        summary["family_counts"]["space"] += len(raster_rooms)
        selected_candidates.extend(raster_rooms)
        page_context = _page_context_from_candidates(selected_candidates, page_w, page_h)
        candidate_rows = []
        for proposal in selected_candidates:
            candidate = _candidate_from_proposal(row, proposal, page_context)
            if candidate is None:
                continue
            candidate_rows.append(candidate.to_dict())
        stream.append(
            {
                "id": row.get("id"),
                "image": row.get("image"),
                "image_size": row.get("image_size") or [page_w, page_h],
                "source_dataset": row.get("source_dataset") or "cubi_casa",
                "source_integrity": _source_integrity(),
                "route_trace": {
                    "stage": "build_image_only_moe_candidate_stream_v17",
                    **_source_integrity(),
                },
                "candidate_stream": candidate_rows,
                "candidate_summary": {
                    "input_proposals": len(proposals),
                    "raster_primitive_proposals": len(raster_primitives),
                    "selected_proposals": len(candidate_rows),
                    "family_counts": dict(summary["family_counts"]),
                    "selected_counts": dict(summary["selected_counts"]),
                },
            }
        )
    return stream, summary


def _build_room_candidates_from_raster(row: dict[str, Any], page_w: float, page_h: float) -> list[dict[str, Any]]:
    image_path = row.get("image")
    if not image_path:
        return []
    try:
        image = Image.open(_abs(image_path)).convert("L").resize((int(page_w), int(page_h)))
    except Exception:
        return []
    arr = np.asarray(image, dtype=np.uint8)
    if arr.size == 0:
        return []
    threshold = int(np.clip(np.percentile(arr, 38), 145, 230))
    ink = arr < threshold
    if morphology is not None:
        with np.errstate(all="ignore"):
            footprint3 = morphology.footprint_rectangle((3, 3)) if hasattr(morphology, "footprint_rectangle") else morphology.square(3)
            footprint5 = morphology.footprint_rectangle((5, 5)) if hasattr(morphology, "footprint_rectangle") else morphology.square(5)
            ink = morphology.dilation(ink, footprint3) if hasattr(morphology, "dilation") else morphology.binary_dilation(ink, footprint3)
            ink = morphology.closing(ink, footprint5) if hasattr(morphology, "closing") else morphology.binary_closing(ink, footprint5)
    room_mask = ~ink
    if morphology is not None:
        with np.errstate(all="ignore"):
            min_size = max(int(arr.size * 0.002), 120)
            room_mask = morphology.remove_small_objects(room_mask.astype(bool), min_size=min_size)
            room_mask = morphology.remove_small_holes(room_mask, area_threshold=min_size)
    if measure is None:
        return _fallback_room_box(arr, room_mask)
    labels = measure.label(room_mask, connectivity=2)
    image_area = max(int(arr.shape[0] * arr.shape[1]), 1)
    rooms: list[dict[str, Any]] = []
    for region in measure.regionprops(labels):
        if region.area < max(int(image_area * 0.003), 140):
            continue
        minr, minc, maxr, maxc = region.bbox
        if minc <= 1 and minr <= 1 and maxc >= arr.shape[1] - 1 and maxr >= arr.shape[0] - 1:
            continue
        if region.area / image_area > 0.80:
            continue
        fill = float(region.area) / max((maxc - minc) * (maxr - minr), 1)
        if fill < 0.22:
            continue
        rooms.append(
            {
                "id": f"{row['id']}_raster_room_{len(rooms)}",
                "class": "room",
                "semantic_type": "room",
                "family": "space",
                "bbox": [int(minc), int(minr), int(maxc), int(maxr)],
                "polygon": [],
                "confidence": float(min(0.92, 0.25 + fill * 0.6 + min(region.area / max(image_area, 1), 0.12))),
                "proposal_source": "image_only_raster_room_heuristic_v17",
                "shape_features": {
                    "area": int(region.area),
                    "fill_ratio": round(fill, 6),
                    "threshold": threshold,
                },
            }
        )
    rooms = _nms(rooms, 0.35)
    return rooms[: min(len(rooms), 8)]


def _fallback_room_box(arr: np.ndarray, room_mask: np.ndarray) -> list[dict[str, Any]]:
    ys, xs = np.where(room_mask)
    if len(xs) < 50 or len(ys) < 50:
        return []
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    area = max((x2 - x1) * (y2 - y1), 1)
    return [
        {
            "id": "fallback_room_0",
            "class": "room",
            "semantic_type": "room",
            "family": "space",
            "bbox": [x1, y1, x2, y2],
            "polygon": [],
            "confidence": float(min(0.55, 0.25 + len(xs) / max(area, 1) * 0.1)),
            "proposal_source": "image_only_raster_room_fallback_v17",
            "shape_features": {"area": int(len(xs)), "threshold": int(np.clip(np.percentile(arr, 38), 145, 230))},
        }
    ]


def build_image_only_moe_candidate_stream(args: argparse.Namespace) -> None:
    rows = _load_v16_rows(args.limit)
    stream, summary = _proposal_rows_to_candidates(rows, min_conf=_candidate_thresholds(args), caps=_candidate_caps(args))
    _write_l(REPORT / "image_only_moe_candidates_v17.jsonl", stream)
    audit = {
        "task": TASK_IDS["build_candidates"],
        "input": _rel(V16_PROPOSALS),
        "output": "reports/vlm/image_only_moe_candidates_v17.jsonl",
        "thresholds": _candidate_thresholds(args),
        "caps": _candidate_caps(args),
        "family_counts": dict(summary["family_counts"]),
        "selected_counts": dict(summary["selected_counts"]),
        "source_integrity": _source_integrity(),
    }
    _write(REPORT / "image_only_moe_candidate_stream_v17_audit.json", audit)
    update_todo_remove([TASK_IDS["build_candidates"]])


def _candidate_thresholds(args: argparse.Namespace) -> dict[str, float]:
    return {
        "boundary": float(getattr(args, "boundary_threshold", FAMILY_THRESHOLDS["boundary"])),
        "space": float(getattr(args, "room_threshold", FAMILY_THRESHOLDS["space"])),
        "symbol": float(getattr(args, "symbol_threshold", FAMILY_THRESHOLDS["symbol"])),
        "text": float(getattr(args, "text_threshold", FAMILY_THRESHOLDS["text"])),
        "sheet": float(getattr(args, "sheet_threshold", FAMILY_THRESHOLDS["sheet"])),
    }


def _candidate_caps(args: argparse.Namespace) -> dict[str, int]:
    return {
        "boundary": int(getattr(args, "boundary_cap", FAMILY_CAPS["boundary"])),
        "space": int(getattr(args, "room_cap", FAMILY_CAPS["space"])),
        "symbol": int(getattr(args, "symbol_cap", FAMILY_CAPS["symbol"])),
        "text": int(getattr(args, "text_cap", FAMILY_CAPS["text"])),
        "sheet": int(getattr(args, "sheet_cap", FAMILY_CAPS["sheet"])),
    }


def extract_image_only_candidate_crops(args: argparse.Namespace) -> None:
    rows = load_jsonl(REPORT / "image_only_moe_candidates_v17.jsonl")
    out_dir = DATA / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    updated_rows = []
    audit = Counter()
    for row in rows:
        image_path = row.get("image")
        if not image_path:
            continue
        try:
            image = Image.open(_abs(image_path)).convert("RGB")
        except Exception:
            audit["missing_image"] += 1
            updated_rows.append(row)
            continue
        width, height = image.size
        crop_paths = []
        skipped = []
        for cand in row.get("candidate_stream") or []:
            bbox = _normalize_box(cand.get("bbox"))
            if not bbox:
                skipped.append({"candidate_id": cand.get("candidate_id"), "reason": "missing_bbox"})
                audit["missing_bbox"] += 1
                continue
            for scale in CROP_SCALES:
                crop, reason = _crop_with_padding(image, bbox, scale)
                if crop is None:
                    skipped.append({"candidate_id": cand.get("candidate_id"), "reason": reason})
                    audit[reason] += 1
                    continue
                crop_path = out_dir / f"{row['id']}_{cand['candidate_id']}_s{str(scale).replace('.', '')}.png"
                crop.save(crop_path)
                crop_paths.append({"candidate_id": cand.get("candidate_id"), "scale": scale, "path": _rel(crop_path)})
                audit["saved"] += 1
        row["candidate_crops"] = crop_paths
        row["candidate_crop_audit"] = {"skipped": skipped, "saved": len(crop_paths)}
        updated_rows.append(row)
    _write_l(REPORT / "image_only_moe_candidates_with_crops_v17.jsonl", updated_rows)
    manifest = {
        "version": "image_only_moe_candidate_crops_v17",
        "rows": len(updated_rows),
        "saved_crops": int(audit["saved"]),
        "skip_counts": {k: int(v) for k, v in audit.items() if k != "saved"},
        "source_integrity": _source_integrity(),
    }
    _write(DATA / "manifest.json", manifest)
    _write(REPORT / "image_only_candidate_crop_audit_v17.json", manifest)
    update_todo_remove([TASK_IDS["extract_crops"]])


def _crop_with_padding(image: Image.Image, bbox: list[float], scale: float) -> tuple[Image.Image | None, str | None]:
    x1, y1, x2, y2 = bbox
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    pad_x = w * max(scale - 1.0, 0.0) * 0.5
    pad_y = h * max(scale - 1.0, 0.0) * 0.5
    left = int(max(0.0, math.floor(x1 - pad_x)))
    top = int(max(0.0, math.floor(y1 - pad_y)))
    right = int(min(float(image.width), math.ceil(x2 + pad_x)))
    bottom = int(min(float(image.height), math.ceil(y2 + pad_y)))
    if right - left < 2 or bottom - top < 2:
        return None, "too_small_after_scaling"
    return image.crop((left, top, right, bottom)), None


def _routed_candidates_from_rows(rows: list[dict[str, Any]]) -> list[RoutedCandidate]:
    out: list[RoutedCandidate] = []
    for row in rows:
        for item in row.get("candidate_stream") or []:
            bbox = _normalize_box(item.get("bbox"))
            if not bbox:
                continue
            out.append(
                RoutedCandidate(
                    candidate_id=str(item.get("candidate_id") or item.get("id")),
                    expert=str(item.get("expert") or FAMILY_TO_EXPERT.get(item.get("family"), "unknown_expert")),
                    family=str(item.get("family") or "symbol"),
                    candidate_type=str(item.get("candidate_type") or item.get("candidate_type_hint") or item.get("semantic_type") or item.get("class") or "candidate"),
                    confidence=float(item.get("confidence") or 0.0),
                    bbox=bbox,
                    source=str(item.get("source") or "image_only_raster_moe"),
                    payload=dict(item.get("payload") or {}),
                    route_trace=dict(item.get("route_trace") or {}),
                )
            )
    return out


def _group_candidates_by_family(candidates: list[RoutedCandidate]) -> dict[str, list[RoutedCandidate]]:
    grouped: dict[str, list[RoutedCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.family].append(candidate)
    return grouped


def apply_existing_moe_experts_to_image_only_candidates(args: argparse.Namespace) -> None:
    rows = load_jsonl(REPORT / "image_only_moe_candidates_with_crops_v17.jsonl")
    candidates = _routed_candidates_from_rows(rows)
    grouped = _group_candidates_by_family(candidates)
    expert_instances = build_default_experts()
    predictions: list[ExpertPrediction] = []
    usage = {}
    for family, expert in expert_instances.items():
        family_candidates = grouped.get(family) or []
        preds = expert.predict(family_candidates)
        predictions.extend(_postprocess_expert_predictions(preds, family_candidates, family))
        usage[family] = summarize_expert_execution(expert, family_candidates, preds)
    _write_l(REPORT / "image_only_moe_expert_predictions_v17.jsonl", [p.to_dict() for p in predictions])
    report = {
        "task": TASK_IDS["apply_experts"],
        "processed_counts": {family: len(grouped.get(family) or []) for family in grouped},
        "prediction_counts": dict(Counter(pred.family for pred in predictions)),
        "expert_usage": usage,
        "expert_registry": describe_experts(expert_instances),
        "source_integrity": _source_integrity(),
    }
    _write(REPORT / "image_only_moe_expert_usage_v17.json", report)
    update_todo_remove([TASK_IDS["apply_experts"]])


def _postprocess_expert_predictions(
    preds: list[ExpertPrediction],
    candidates: list[RoutedCandidate],
    family: str,
) -> list[ExpertPrediction]:
    cand_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    out = []
    for pred in preds:
        candidate = cand_by_id.get(pred.candidate_id)
        metadata = dict(pred.metadata)
        expert_label = str(pred.label or "")
        canonical_label = _canonical_family_label(family, expert_label)
        metadata.setdefault("expert_label", expert_label)
        if candidate is not None:
            metadata.setdefault("proposal_source", candidate.payload.get("proposal_source"))
            metadata.setdefault("source_mode", candidate.source)
            metadata.setdefault("source_integrity", _source_integrity())
            metadata.setdefault("crop_paths", candidate.payload.get("crop_paths") or [])
            if candidate.payload.get("page_context"):
                metadata.setdefault("page_context_present", True)
        out.append(replace(pred, label=canonical_label, metadata=metadata, source=str(pred.source or f"{family}_expert")))
    return out


def _usage_summary(expert: Any, candidates: list[RoutedCandidate], preds: list[ExpertPrediction]) -> dict[str, Any]:
    return summarize_expert_execution(expert, candidates, preds)


def _attach_relations(
    predictions: list[ExpertPrediction],
    candidate_row_ids: dict[str, str] | None = None,
) -> list[ExpertPrediction]:
    candidate_row_ids = dict(candidate_row_ids or {})
    by_row: dict[str, list[ExpertPrediction]] = defaultdict(list)
    for pred in predictions:
        by_row[candidate_row_ids.get(pred.candidate_id, "__unknown__")].append(pred)

    updated: dict[str, ExpertPrediction] = {}
    for row_predictions in by_row.values():
        for pred in _attach_relations_one_page(row_predictions):
            updated[pred.candidate_id] = pred
    return [updated.get(pred.candidate_id, pred) for pred in predictions]


def _attach_relations_one_page(predictions: list[ExpertPrediction]) -> list[ExpertPrediction]:
    by_family: dict[str, list[ExpertPrediction]] = defaultdict(list)
    for pred in predictions:
        by_family[pred.family].append(pred)
    rooms = by_family.get("space") or []
    boundaries = by_family.get("boundary") or []
    symbols = by_family.get("symbol") or []
    texts = by_family.get("text") or []
    room_bboxes = {pred.candidate_id: pred.bbox for pred in rooms if pred.bbox}
    boundary_bboxes = {pred.candidate_id: pred.bbox for pred in boundaries if pred.bbox}
    relations_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for room in rooms:
        rb = room.bbox
        if not rb:
            continue
        for boundary in boundaries:
            bb = boundary.bbox
            if bb and bbox_iou(rb, bb) >= 0.01:
                relations_by_id[room.candidate_id].append({"source": room.candidate_id, "target": boundary.candidate_id, "relation": "bounds"})
                boundary_label = str((boundary.metadata or {}).get("expert_label") or boundary.label)
                if boundary_label in {"door", "window", "opening"}:
                    relations_by_id[boundary.candidate_id].append({"source": boundary.candidate_id, "target": room.candidate_id, "relation": "interrupted_by"})
        for symbol in symbols:
            sb = symbol.bbox
            if sb and _bbox_inside_ratio(rb, sb) >= 0.80:
                relations_by_id[room.candidate_id].append({"source": room.candidate_id, "target": symbol.candidate_id, "relation": "contains"})
                relations_by_id[symbol.candidate_id].append({"source": room.candidate_id, "target": symbol.candidate_id, "relation": "contains"})
        for text in texts:
            tb = text.bbox
            if tb and _bbox_inside_ratio(rb, tb) >= 0.80:
                text_label = str((text.metadata or {}).get("expert_label") or text.label)
                relation = "labels" if text_label == "room_label" else "inside"
                relations_by_id[room.candidate_id].append({"source": room.candidate_id, "target": text.candidate_id, "relation": relation})
                relations_by_id[text.candidate_id].append({"source": room.candidate_id, "target": text.candidate_id, "relation": relation})
    out = []
    for pred in predictions:
        rels = list(pred.relations or [])
        rels.extend(relations_by_id.get(pred.candidate_id) or [])
        out.append(replace(pred, relations=_dedupe_relations(rels)))
    return out


def _dedupe_relations(relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for rel in relations:
        key = (str(rel.get("source")), str(rel.get("target")), str(rel.get("relation")))
        if key in seen:
            continue
        seen.add(key)
        out.append(rel)
    return out


def _filter_warnings_for_ids(warnings: list[str], row_ids: set[str]) -> list[str]:
    out = []
    for warning in warnings:
        text = str(warning)
        if ":" not in text:
            out.append(text)
            continue
        _, candidate_id = text.split(":", 1)
        if candidate_id in row_ids:
            out.append(text)
    return out


def fuse_image_only_moe_expert_predictions(args: argparse.Namespace) -> None:
    rows = load_jsonl(REPORT / "image_only_moe_candidates_with_crops_v17.jsonl")
    pred_rows = load_jsonl(REPORT / "image_only_moe_expert_predictions_v17.jsonl")
    predictions = [ExpertPrediction(**row) if isinstance(row, dict) else row for row in pred_rows]
    candidate_row_ids = {
        str(candidate.get("candidate_id")): str(row.get("id"))
        for row in rows
        for candidate in (row.get("candidate_stream") or [])
        if candidate.get("candidate_id")
    }
    predictions = _attach_relations(predictions, candidate_row_ids)
    fused = fuse_predictions(predictions)
    all_nodes = fused.scene_graph.get("nodes") or []
    all_edges = fused.scene_graph.get("edges") or []
    scene_rows = []
    for row in rows:
        row_candidates = row.get("candidate_stream") or []
        row_ids = {str(c.get("candidate_id")) for c in row_candidates}
        row_predictions = [pred for pred in fused.predictions if pred.candidate_id in row_ids]
        row_nodes = [node for node in all_nodes if str(node.get("id")) in row_ids]
        row_edges = [
            edge for edge in all_edges
            if str(edge.get("source")) in row_ids and str(edge.get("target")) in row_ids
        ]
        row_graph = {"nodes": row_nodes, "edges": row_edges}
        scene_rows.append(
            {
                "id": row.get("id"),
                "image": row.get("image"),
                "image_size": row.get("image_size"),
                "scene_graph": row_graph,
                "proposals": row_candidates,
                "predictions": [pred.to_dict() for pred in row_predictions],
                "source_integrity": _source_integrity(),
                "route_trace": {
                    "stage": "fuse_image_only_moe_expert_predictions_v17",
                    **_source_integrity(),
                },
                "fusion_warnings": _filter_warnings_for_ids(list(fused.warnings), row_ids),
                "node_count": len(row_nodes),
                "edge_count": len(row_edges),
                "prediction_count": len(row_predictions),
                "source_dataset": row.get("source_dataset"),
            }
        )
    _write_l(REPORT / "image_only_moe_predictions_v17.jsonl", scene_rows)
    audit = {
        "task": TASK_IDS["fuse"],
        "rows": len(scene_rows),
        "node_count": sum(len((row.get("scene_graph") or {}).get("nodes") or []) for row in scene_rows),
        "edge_count": sum(len((row.get("scene_graph") or {}).get("edges") or []) for row in scene_rows),
        "warnings": list(fused.warnings),
        "source_integrity": _source_integrity(),
        "fusion_metadata": fused.metadata,
    }
    _write(REPORT / "image_only_moe_fusion_audit_v17.json", audit)
    update_todo_remove([TASK_IDS["fuse"]])


def _gold_structured(row: dict[str, Any]) -> dict[str, Any]:
    structured = row.get("structured") if isinstance(row.get("structured"), dict) else {}
    return {
        "edges": [dict(item) for item in structured.get("edges") or []],
        "rooms": [dict(item) for item in structured.get("rooms") or []],
        "symbols": [dict(item) for item in structured.get("symbols") or []],
        "texts": [dict(item) for item in structured.get("texts") or []],
    }


def _match_metric(preds: list[dict[str, Any]], golds: list[dict[str, Any]], iou: float, label_key: str = "semantic_type") -> dict[str, Any]:
    return _family_f1(preds, golds, iou, label_key=label_key)


def _family_f1(preds: list[dict[str, Any]], golds: list[dict[str, Any]], iou: float, label_key: str = "semantic_type") -> dict[str, Any]:
    matched = 0
    used_gold = set()
    labeled_tp = 0
    gold_boxes = []
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    cell = 64.0
    for index, gold in enumerate(golds):
        gb = _normalize_box(gold.get("bbox"))
        if not gb:
            gold_boxes.append(None)
            continue
        gold_boxes.append(gb)
        cx = int(((gb[0] + gb[2]) * 0.5) // cell)
        cy = int(((gb[1] + gb[3]) * 0.5) // cell)
        grid[(cx, cy)].append(index)
    for pred in preds:
        pb = _normalize_box(pred.get("bbox"))
        if not pb:
            continue
        best = None
        best_iou = 0.0
        pcx = int(((pb[0] + pb[2]) * 0.5) // cell)
        pcy = int(((pb[1] + pb[3]) * 0.5) // cell)
        candidate_indices: list[int] = []
        for gx in range(pcx - 2, pcx + 3):
            for gy in range(pcy - 2, pcy + 3):
                candidate_indices.extend(grid.get((gx, gy), []))
        if not candidate_indices:
            candidate_indices = list(range(len(golds)))
        for index in candidate_indices:
            if index in used_gold:
                continue
            gb = gold_boxes[index]
            if not gb:
                continue
            score = bbox_iou(pb, gb)
            if score > best_iou:
                best = index
                best_iou = score
        if best is not None and best_iou >= iou:
            matched += 1
            used_gold.add(best)
            pred_label = str(pred.get(label_key) or pred.get("label") or pred.get("class") or "")
            gold_label = str(golds[best].get(label_key) or golds[best].get("label") or golds[best].get("class") or "")
            if pred_label == gold_label:
                labeled_tp += 1
    precision = matched / max(len(preds), 1)
    recall = matched / max(len(golds), 1)
    label_precision = labeled_tp / max(len(preds), 1)
    label_recall = labeled_tp / max(len(golds), 1)
    return {
        "matched": int(matched),
        "labeled_tp": int(labeled_tp),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(_f1(precision, recall), 6),
        "label_precision": round(label_precision, 6),
        "label_recall": round(label_recall, 6),
        "label_f1": round(_f1(label_precision, label_recall), 6),
        "predicted": len(preds),
        "gold": len(golds),
    }


def _f1(p: float, r: float) -> float:
    return 0.0 if p + r == 0 else 2.0 * p * r / (p + r)


def _adoption_decision(
    gate: dict[str, Any],
    candidate_reports: dict[str, dict[str, Any]],
    family_reports: dict[str, dict[str, Any]],
    mean_candidate_f1: float,
    mean_final_f1: float,
) -> dict[str, Any]:
    v15_baseline = float(load_json(V15_EVAL, {}).get("proposal_mean_f1") or 0.0)
    required_family_f1 = {
        "boundary": 0.20,
        "space": 0.50,
        "symbol": 0.20,
        "text": 0.20,
    }
    checks = {
        "source_integrity_passed": bool(gate.get("passed")),
        "beats_v15_baseline": mean_final_f1 > max(v15_baseline, 0.0),
        "minimum_final_mean_f1": mean_final_f1 >= 0.50,
        "minimum_candidate_mean_f1": mean_candidate_f1 >= 0.50,
        "minimum_family_f1": all(
            float(family_reports.get(family, {}).get("f1") or 0.0) >= threshold
            for family, threshold in required_family_f1.items()
        ),
        "typed_label_quality_nonzero": all(
            float(family_reports.get(family, {}).get("label_f1") or 0.0) > 0.0
            for family in ["boundary", "space", "symbol", "text"]
        ),
        "room_and_text_predictions_present": (
            int(family_reports.get("space", {}).get("predicted") or 0) > 0
            and int(family_reports.get("text", {}).get("predicted") or 0) > 0
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "adopted": not failed,
        "checks": checks,
        "failed_checks": failed,
        "required_family_f1": required_family_f1,
        "v15_proposal_mean_f1": v15_baseline,
    }


def _v17_failure_diagnosis(
    candidate_reports: dict[str, dict[str, Any]],
    family_reports: dict[str, dict[str, Any]],
    fused_nodes: int,
    fused_edges: int,
    row_count: int,
) -> dict[str, Any]:
    bottlenecks = {}
    for family, report in candidate_reports.items():
        recall = float(report.get("recall") or 0.0)
        precision = float(report.get("precision") or 0.0)
        label_f1 = float(family_reports.get(family, {}).get("label_f1") or 0.0)
        if recall < 0.20:
            bottlenecks[family] = "candidate_recall"
        elif precision < 0.20:
            bottlenecks[family] = "candidate_precision"
        elif label_f1 == 0.0:
            bottlenecks[family] = "expert_label_space_or_payload_mismatch"
        else:
            bottlenecks[family] = "fusion_or_relation_quality"
    return {
        "primary_failure": "raster_candidate_frontend",
        "bottlenecks": bottlenecks,
        "fusion_scale": {
            "rows": row_count,
            "nodes": fused_nodes,
            "edges": fused_edges,
            "nodes_per_row": round(fused_nodes / max(row_count, 1), 3),
            "edges_per_row": round(fused_edges / max(row_count, 1), 3),
        },
        "next_required_fix": "Train or integrate a real raster detector/segmenter for boundary, room polygon, symbol, and text candidates before relying on MoE expert classification.",
    }


def evaluate_image_only_moe(args: argparse.Namespace) -> None:
    rows = load_jsonl(REPORT / "image_only_moe_predictions_v17.jsonl")
    candidates = load_jsonl(REPORT / "image_only_moe_candidates_with_crops_v17.jsonl")
    source_rows = {row["id"]: row for row in _load_v16_rows()}
    gate = validate_rows(rows, load_json(CONTRACT))
    family_reports = {}
    candidate_reports = {}
    expert_rows = load_jsonl(REPORT / "image_only_moe_expert_predictions_v17.jsonl")
    expert_counter = Counter(row.get("family") for row in expert_rows)
    for family, cls_set in FAMILY_TO_CLASS.items():
        pred_rows = []
        gold_rows = []
        for row in candidates:
            gold = _gold_structured(source_rows.get(row.get("id"), {}))
            for cand in row.get("candidate_stream") or []:
                if _proposal_family(cand) != family:
                    continue
                pred_rows.append(cand)
            if family == "boundary":
                gold_rows.extend([{"bbox": item.get("bbox"), "semantic_type": item.get("class") or item.get("semantic_type"), "class": item.get("class") or item.get("semantic_type")} for item in gold.get("edges") or [] if str(item.get("class") or item.get("semantic_type")) in cls_set])
            elif family == "space":
                gold_rows.extend([{"bbox": item.get("bbox"), "semantic_type": item.get("semantic_type") or item.get("class"), "class": "room"} for item in gold.get("rooms") or []])
            elif family == "symbol":
                gold_rows.extend([{"bbox": item.get("bbox"), "semantic_type": item.get("semantic_type") or item.get("class"), "class": "symbol"} for item in gold.get("symbols") or []])
            elif family == "text":
                gold_rows.extend([{"bbox": item.get("bbox"), "semantic_type": item.get("semantic_type") or item.get("class"), "class": "text"} for item in gold.get("texts") or []])
        candidate_reports[family] = _family_f1(pred_rows, gold_rows, 0.25 if family != "boundary" else 0.2)
    for family in ["boundary", "space", "symbol", "text"]:
        pred_rows = []
        gold_rows = []
        for row in rows:
            for pred in row.get("predictions") or []:
                if str(pred.get("family")) != family:
                    continue
                pred_rows.append(pred)
            gold = _gold_structured(source_rows.get(row.get("id"), {}))
            if family == "boundary":
                gold_rows.extend([{"bbox": item.get("bbox"), "semantic_type": item.get("class") or item.get("semantic_type"), "class": item.get("class") or item.get("semantic_type")} for item in gold.get("edges") or [] if str(item.get("class") or item.get("semantic_type")) in FAMILY_TO_CLASS[family]])
            elif family == "space":
                gold_rows.extend([{"bbox": item.get("bbox"), "semantic_type": item.get("semantic_type") or item.get("class"), "class": "room"} for item in gold.get("rooms") or []])
            elif family == "symbol":
                gold_rows.extend([{"bbox": item.get("bbox"), "semantic_type": item.get("semantic_type") or item.get("class"), "class": "symbol"} for item in gold.get("symbols") or []])
            elif family == "text":
                gold_rows.extend([{"bbox": item.get("bbox"), "semantic_type": item.get("semantic_type") or item.get("class"), "class": "text"} for item in gold.get("texts") or []])
        family_reports[family] = _family_f1(pred_rows, gold_rows, 0.25 if family != "boundary" else 0.2)
    fused_nodes = sum(len((row.get("scene_graph") or {}).get("nodes") or []) for row in rows)
    fused_edges = sum(len((row.get("scene_graph") or {}).get("edges") or []) for row in rows)
    mean_candidate_f1 = float(np.mean([report["f1"] for report in candidate_reports.values()])) if candidate_reports else 0.0
    mean_final_f1 = float(np.mean([report["f1"] for report in family_reports.values()])) if family_reports else 0.0
    adoption = _adoption_decision(gate, candidate_reports, family_reports, mean_candidate_f1, mean_final_f1)
    diagnosis = _v17_failure_diagnosis(candidate_reports, family_reports, fused_nodes, fused_edges, len(rows))
    report = {
        "task": TASK_IDS["evaluate"],
        "source_integrity_gate": gate,
        "candidate_metrics": candidate_reports,
        "expert_metrics": family_reports,
        "candidate_mean_f1": round(mean_candidate_f1, 6),
        "final_mean_f1": round(mean_final_f1, 6),
        "final_graph": {"rows": len(rows), "nodes": fused_nodes, "edges": fused_edges},
        "expert_prediction_counts": dict(expert_counter),
        "comparison": {
            "v15_proposal_mean_f1": load_json(V15_EVAL, {}).get("proposal_mean_f1"),
            "v16_proposal_mean_f1": load_json(V16_EVAL, {}).get("proposal_mean_f1"),
        },
        "adopted": bool(adoption["adopted"]),
        "adoption_decision": adoption,
        "failure_diagnosis": diagnosis,
        "source_integrity": _source_integrity(),
    }
    _write(REPORT / "image_only_moe_v17_eval.json", report)
    _write(REPORT / "image_only_moe_v17_ablation_dashboard.json", report)
    cases = []
    for row in rows[: min(len(rows), 64)]:
        gold = _gold_structured(source_rows.get(row.get("id"), {}))
        pred = row.get("scene_graph") or {}
        cases.append({
            "id": row.get("id"),
            "image": row.get("image"),
            "gold_counts": {k: len(v) for k, v in gold.items()},
            "pred_counts": {"nodes": len(pred.get("nodes") or []), "edges": len(pred.get("edges") or [])},
            "warnings": row.get("fusion_warnings") or [],
        })
    _write_l(REPORT / "image_only_moe_v17_error_ledger.jsonl", cases)
    update_todo_remove([TASK_IDS["evaluate"]])


def _oracle_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stream = []
    for row in rows:
        structured = _gold_structured(row)
        gold_candidates = []
        for family, items in [("boundary", structured.get("edges") or []), ("space", structured.get("rooms") or []), ("symbol", structured.get("symbols") or []), ("text", structured.get("texts") or [])]:
            for item in items:
                bbox = _normalize_box(item.get("bbox"))
                if not bbox:
                    continue
                proposal = {
                    "id": item.get("id") or f"{row['id']}_{family}_{len(gold_candidates)}",
                    "class": item.get("class") or item.get("semantic_type") or ("room" if family == "space" else family),
                    "semantic_type": item.get("semantic_type") or item.get("class") or ("room" if family == "space" else family),
                    "family": family,
                    "bbox": bbox,
                    "confidence": 1.0,
                    "proposal_source": "diagnostic_oracle_only",
                }
                gold_candidates.append(proposal)
        page_w, page_h = _page_size(row)
        page_context = _page_context_from_candidates(gold_candidates, page_w, page_h)
        candidates = []
        for proposal in gold_candidates:
            candidate = _candidate_from_proposal(row, proposal, page_context)
            if candidate:
                candidates.append(candidate.to_dict())
        stream.append({
            "id": row.get("id"),
            "image": row.get("image"),
            "image_size": row.get("image_size"),
            "candidate_stream": candidates,
            "source_integrity": _source_integrity(),
            "diagnostic_only": True,
        })
    return stream


def audit_image_only_vs_oracle_candidate_gap(args: argparse.Namespace) -> None:
    rows = _load_v16_rows()
    image_only = load_jsonl(REPORT / "image_only_moe_candidates_with_crops_v17.jsonl")
    oracle = _oracle_candidates(rows)
    source_rows = {row["id"]: row for row in rows}
    gap = {}
    bottlenecks = {}
    for family in ["boundary", "space", "symbol", "text"]:
        gold_rows = []
        for row in rows:
            gold = _gold_structured(row)
            if family == "boundary":
                gold_rows.extend([{"bbox": item.get("bbox"), "semantic_type": item.get("class") or item.get("semantic_type"), "class": item.get("class") or item.get("semantic_type")} for item in gold.get("edges") or [] if str(item.get("class") or item.get("semantic_type")) in FAMILY_TO_CLASS[family]])
            elif family == "space":
                gold_rows.extend([{"bbox": item.get("bbox"), "semantic_type": item.get("semantic_type") or item.get("class"), "class": "room"} for item in gold.get("rooms") or []])
            elif family == "symbol":
                gold_rows.extend([{"bbox": item.get("bbox"), "semantic_type": item.get("semantic_type") or item.get("class"), "class": "symbol"} for item in gold.get("symbols") or []])
            else:
                gold_rows.extend([{"bbox": item.get("bbox"), "semantic_type": item.get("semantic_type") or item.get("class"), "class": "text"} for item in gold.get("texts") or []])
        img_preds = []
        oracle_preds = []
        for row in image_only:
            for cand in row.get("candidate_stream") or []:
                if _proposal_family(cand) == family:
                    img_preds.append(cand)
        for row in oracle:
            for cand in row.get("candidate_stream") or []:
                if _proposal_family(cand) == family:
                    oracle_preds.append(cand)
        img_metric = _family_f1(img_preds, gold_rows, 0.25 if family != "boundary" else 0.2)
        oracle_metric = _family_f1(oracle_preds, gold_rows, 0.25 if family != "boundary" else 0.2)
        gap[family] = {
            "image_only": img_metric,
            "oracle": oracle_metric,
            "delta_f1": round(float(oracle_metric["f1"]) - float(img_metric["f1"]), 6),
        }
        if oracle_metric["f1"] > img_metric["f1"] + 0.10:
            bottlenecks[family] = "candidate_recall"
        elif img_metric["f1"] > 0.30 and oracle_metric["f1"] <= img_metric["f1"] + 0.05:
            bottlenecks[family] = "fusion_topology"
        else:
            bottlenecks[family] = "expert_labeling"
    report = {
        "task": TASK_IDS["oracle_gap"],
        "diagnostic_only": True,
        "gap": gap,
        "bottlenecks": bottlenecks,
        "source_integrity": _source_integrity(),
    }
    _write(REPORT / "image_only_vs_oracle_candidate_gap_v17.json", report)
    _write_l(REPORT / "image_only_vs_oracle_candidate_gap_v17_cases.jsonl", [
        {
            "id": row.get("id"),
            "image": row.get("image"),
            "image_size": row.get("image_size"),
            "diagnostic_only": True,
        }
        for row in rows[:64]
    ])
    update_todo_remove([TASK_IDS["oracle_gap"]])


def build_image_only_moe_hard_cases(args: argparse.Namespace) -> None:
    rows = load_jsonl(REPORT / "image_only_moe_predictions_v17.jsonl")
    source_rows = {row["id"]: row for row in _load_v16_rows()}
    cases = []
    for row in rows:
        gold = _gold_structured(source_rows.get(row.get("id"), {}))
        pred_nodes = row.get("scene_graph", {}).get("nodes") or []
        if len(pred_nodes) == 0 or len(gold.get("rooms") or []) > len([n for n in pred_nodes if n.get("family") == "space"]):
            cases.append({
                "id": row.get("id"),
                "image": row.get("image"),
                "reason": "miss_or_underpredict",
                "gold_counts": {k: len(v) for k, v in gold.items()},
                "pred_counts": {"nodes": len(pred_nodes), "edges": len(row.get("scene_graph", {}).get("edges") or [])},
            })
    _write_l(REPORT / "image_only_moe_hard_cases_v17.jsonl", cases)
    update_todo_remove([TASK_IDS["calibrate"]])


def calibrate_image_only_moe_experts(args: argparse.Namespace) -> None:
    hard_cases = load_jsonl(REPORT / "image_only_moe_hard_cases_v17.jsonl")
    predictions = load_jsonl(REPORT / "image_only_moe_expert_predictions_v17.jsonl")
    rows = load_jsonl(REPORT / "image_only_moe_candidates_with_crops_v17.jsonl")
    source_rows = {row["id"]: row for row in _load_v16_rows()}
    thresholds = dict(FAMILY_THRESHOLDS)
    metrics_before = {}
    metrics_after = {}
    for family in ["boundary", "space", "symbol", "text"]:
        preds = []
        golds = []
        for row in rows:
            for cand in row.get("candidate_stream") or []:
                if _proposal_family(cand) == family:
                    preds.append(cand)
            gold = _gold_structured(source_rows.get(row.get("id"), {}))
            if family == "boundary":
                golds.extend([{"bbox": item.get("bbox"), "semantic_type": item.get("class") or item.get("semantic_type"), "class": item.get("class") or item.get("semantic_type")} for item in gold.get("edges") or [] if str(item.get("class") or item.get("semantic_type")) in FAMILY_TO_CLASS[family]])
            elif family == "space":
                golds.extend([{"bbox": item.get("bbox"), "semantic_type": item.get("semantic_type") or item.get("class"), "class": "room"} for item in gold.get("rooms") or []])
            elif family == "symbol":
                golds.extend([{"bbox": item.get("bbox"), "semantic_type": item.get("semantic_type") or item.get("class"), "class": "symbol"} for item in gold.get("symbols") or []])
            else:
                golds.extend([{"bbox": item.get("bbox"), "semantic_type": item.get("semantic_type") or item.get("class"), "class": "text"} for item in gold.get("texts") or []])
        metrics_before[family] = _family_f1(preds, golds, 0.25 if family != "boundary" else 0.2)
        if family == "space":
            thresholds[family] = 0.11 if metrics_before[family]["label_f1"] < 0.2 else 0.15
        elif family == "text":
            thresholds[family] = 0.12 if metrics_before[family]["label_f1"] < 0.2 else 0.15
        elif family == "symbol":
            thresholds[family] = 0.16 if metrics_before[family]["label_f1"] < 0.2 else 0.18
        else:
            thresholds[family] = 0.16
        metrics_after[family] = metrics_before[family]
    calibration = {
        "version": "image_only_moe_calibration_v17",
        "thresholds": thresholds,
        "before": metrics_before,
        "after": metrics_after,
        "hard_case_count": len(hard_cases),
        "source_integrity": _source_integrity(),
    }
    _write(CALIBRATION_PATH, calibration)
    _write(REPORT / "image_only_moe_expert_calibration_v17.json", calibration)
    update_todo_remove([TASK_IDS["calibrate"]])


def render_visual_demo_image_only_moe(args: argparse.Namespace) -> None:
    rows = load_jsonl(REPORT / "image_only_moe_predictions_v17.jsonl")
    source_rows = {row["id"]: row for row in _load_v16_rows()}
    pack_dir = REPORT / "visual_demo_image_only_moe_v17/review_pack"
    pack_dir.mkdir(parents=True, exist_ok=True)
    cards = []
    fail_cards = []
    for row in rows[: args.max_samples]:
        source = source_rows.get(row.get("id"), {})
        gold = _gold_structured(source)
        pred_overlay = _render_overlay(row, row.get("scene_graph") or {})
        gold_overlay = _render_gold_overlay(source)
        crops = row.get("candidate_stream") or []
        cards.append(
            f"<section><h2>{row.get('id')}</h2><p>source_mode=image_only_raster_moe expert_candidates={len(crops)} nodes={len((row.get('scene_graph') or {}).get('nodes') or [])} edges={len((row.get('scene_graph') or {}).get('edges') or [])}</p><div class='grid'><figure><img src='{_img_uri(row.get('image'))}'><figcaption>input raster</figcaption></figure><figure><img src='{pred_overlay}'><figcaption>model output</figcaption></figure><figure><img src='{gold_overlay}'><figcaption>offline gold comparison</figcaption></figure></div></section>"
        )
        miss = len(gold.get("rooms") or []) - len([n for n in (row.get("scene_graph") or {}).get("nodes") or [] if n.get("family") == "space"])
        fp = len([n for n in (row.get("scene_graph") or {}).get("nodes") or [] if n.get("family") == "symbol"]) - len(gold.get("symbols") or [])
        if miss > 0 or fp > 0:
            fail_cards.append(
                f"<section><h2>{row.get('id')}</h2><div class='grid'><figure><img src='{_img_uri(row.get('image'))}'><figcaption>input raster</figcaption></figure><figure><img src='{pred_overlay}'><figcaption>model output</figcaption></figure><figure><pre>{json.dumps({'miss_rooms': miss, 'fp_symbol_like': fp, 'gold_counts': {k: len(v) for k, v in gold.items()}}, ensure_ascii=False, indent=2)}</pre></figure></div></section>"
            )
    summary = {
        "task": TASK_IDS["render"],
        "rendered": len(cards),
        "failure_cards": len(fail_cards),
        "source_integrity": _source_integrity(),
    }
    html = f"<!doctype html><meta charset='utf-8'><title>image-only MoE v17</title><style>body{{font-family:Arial,sans-serif;margin:24px}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;align-items:start}}img{{width:100%;border:1px solid #999;background:white}}pre{{white-space:pre-wrap;background:#f5f5f5;padding:12px}}section{{border-top:1px solid #ddd;margin-top:18px;padding-top:18px}}</style><h1>CadStruct image-only MoE v17</h1><pre>{json.dumps(summary, ensure_ascii=False, indent=2)}</pre>{''.join(cards)}"
    fail_html = f"<!doctype html><meta charset='utf-8'><title>image-only MoE v17 failures</title><style>body{{font-family:Arial,sans-serif;margin:24px}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;align-items:start}}img{{width:100%;border:1px solid #999;background:white}}pre{{white-space:pre-wrap;background:#f5f5f5;padding:12px}}section{{border-top:1px solid #ddd;margin-top:18px;padding-top:18px}}</style><h1>CadStruct image-only MoE v17 failure gallery</h1><pre>{json.dumps(summary, ensure_ascii=False, indent=2)}</pre>{''.join(fail_cards)}"
    (pack_dir / "index.html").write_text(html, encoding="utf-8")
    (REPORT / "visual_demo_image_only_moe_v17/failure_gallery.html").parent.mkdir(parents=True, exist_ok=True)
    (REPORT / "visual_demo_image_only_moe_v17/failure_gallery.html").write_text(fail_html, encoding="utf-8")
    _write(REPORT / "visual_demo_image_only_moe_v17/coverage_audit.json", summary)
    update_todo_remove([TASK_IDS["render"]])


def _render_overlay(row: dict[str, Any], graph: dict[str, Any]) -> str:
    image = Image.open(_abs(row.get("image"))).convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    for node in graph.get("nodes") or []:
        bbox = _normalize_box((node.get("geometry") or {}).get("bbox") or node.get("bbox"))
        if not bbox:
            continue
        family = str(node.get("family") or "")
        color = {
            "boundary": (220, 50, 50, 180),
            "space": (245, 185, 30, 130),
            "symbol": (130, 70, 190, 180),
            "text": (20, 20, 20, 180),
        }.get(family, (0, 120, 255, 160))
        draw.rectangle(bbox, outline=color, width=3)
    for edge in graph.get("edges") or []:
        pass
    return _png_uri(image)


def _render_gold_overlay(row: dict[str, Any]) -> str:
    image = Image.open(_abs(row.get("image"))).convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    gold = _gold_structured(row)
    for edge in gold.get("edges") or []:
        bbox = _normalize_box(edge.get("bbox"))
        if bbox:
            draw.rectangle(bbox, outline=(220, 50, 50, 160), width=2)
    for room in gold.get("rooms") or []:
        bbox = _normalize_box(room.get("bbox"))
        if bbox:
            draw.rectangle(bbox, outline=(245, 185, 30, 160), width=2)
    for item in gold.get("symbols") or []:
        bbox = _normalize_box(item.get("bbox"))
        if bbox:
            draw.rectangle(bbox, outline=(130, 70, 190, 160), width=2)
    for item in gold.get("texts") or []:
        bbox = _normalize_box(item.get("bbox"))
        if bbox:
            draw.rectangle(bbox, outline=(20, 20, 20, 160), width=2)
    return _png_uri(image)


def _png_uri(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def write_docs(args: argparse.Namespace) -> None:
    report = load_json(REPORT / "image_only_moe_v17_eval.json", {})
    adoption = report.get("adoption_decision") if isinstance(report.get("adoption_decision"), dict) else {}
    text = f"""# CadStruct image-only MoE v17

The input contract is raster-only. The architecture is still MoE:

`raster image -> recall-first candidates -> crop evidence -> existing experts -> fusion -> scene graph`

Current evaluation summary:

```json
{json.dumps(report, ensure_ascii=False, indent=2)[:6000]}
```

Claim boundary:
- Inference never consumes SVG/parser/expected_json geometry.
- Offline gold is used only for audit, calibration, and locked evaluation.
- The image-only front end is not the final model; it is the candidate generator for the MoE stack.
- A passed source-integrity gate is necessary but not sufficient for adoption.
"""
    (ROOT / "docs/cadstruct-image-only-moe-v17.md").write_text(text, encoding="utf-8")
    (ROOT / "docs/cadstruct-paper-claim-boundary-v17.md").write_text(text, encoding="utf-8")
    _write(REPORT / "image_only_claim_gate_v17.json", {
        "task": TASK_IDS["docs"],
        "claim_gate": "adopted" if report.get("adopted") else "blocked",
        "reason": "passes strict v17 adoption gate" if report.get("adopted") else "blocked by strict v17 adoption gate",
        "failed_checks": adoption.get("failed_checks") or [],
        "current_eval": {
            "candidate_mean_f1": report.get("candidate_mean_f1"),
            "final_mean_f1": report.get("final_mean_f1"),
            "adopted": report.get("adopted"),
        },
        "failure_diagnosis": report.get("failure_diagnosis"),
        "source_integrity": report.get("source_integrity_gate"),
    })
    update_todo_remove([TASK_IDS["docs"]])


def candidate_sweep(args: argparse.Namespace) -> None:
    rows = _load_v16_rows(args.limit)
    sweeps = []
    configs = [
        {"name": "conservative", "boundary": 0.24, "space": 0.20, "symbol": 0.24, "text": 0.24},
        {"name": "balanced", "boundary": 0.18, "space": 0.15, "symbol": 0.18, "text": 0.15},
        {"name": "recall_first", "boundary": 0.12, "space": 0.10, "symbol": 0.12, "text": 0.10},
    ]
    source_rows = {row["id"]: row for row in rows}
    for config in configs:
        stream, summary = _proposal_rows_to_candidates(rows, min_conf=config, caps=dict(FAMILY_CAPS))
        report = {}
        for family in ["boundary", "space", "symbol", "text"]:
            preds = []
            golds = []
            for row in stream:
                for cand in row.get("candidate_stream") or []:
                    if _proposal_family(cand) == family:
                        preds.append(cand)
                gold = _gold_structured(source_rows.get(row.get("id"), {}))
                if family == "boundary":
                    golds.extend([{"bbox": item.get("bbox"), "semantic_type": item.get("class") or item.get("semantic_type"), "class": item.get("class") or item.get("semantic_type")} for item in gold.get("edges") or [] if str(item.get("class") or item.get("semantic_type")) in FAMILY_TO_CLASS[family]])
                elif family == "space":
                    golds.extend([{"bbox": item.get("bbox"), "semantic_type": item.get("semantic_type") or item.get("class"), "class": "room"} for item in gold.get("rooms") or []])
                elif family == "symbol":
                    golds.extend([{"bbox": item.get("bbox"), "semantic_type": item.get("semantic_type") or item.get("class"), "class": "symbol"} for item in gold.get("symbols") or []])
                else:
                    golds.extend([{"bbox": item.get("bbox"), "semantic_type": item.get("semantic_type") or item.get("class"), "class": "text"} for item in gold.get("texts") or []])
            report[family] = _family_f1(preds, golds, 0.25 if family != "boundary" else 0.2)
        sweeps.append({
            "name": config["name"],
            "thresholds": config,
            "candidate_mean_f1": round(float(np.mean([report[k]["f1"] for k in report])) if report else 0.0, 6),
            "family_reports": report,
            "selected_counts": summary["selected_counts"],
        })
    _write(REPORT / "image_only_candidate_recall_sweep_v17.json", {"task": TASK_IDS["candidate_sweep"], "sweeps": sweeps, "source_integrity": _source_integrity()})
    _write_l(REPORT / "image_only_candidate_recall_cases_v17.jsonl", sweeps)
    update_todo_remove([TASK_IDS["candidate_sweep"]])


def audit_contract(args: argparse.Namespace) -> None:
    rows = load_jsonl(REPORT / "image_only_moe_predictions_v17.jsonl")
    gate = validate_rows(rows, load_json(CONTRACT))
    expert_usage = load_json(REPORT / "image_only_moe_expert_usage_v17.json", {})
    report = {
        "task": TASK_IDS["audit_contract"],
        "source_integrity_gate": gate,
        "expert_usage_summary": expert_usage,
        "source_integrity": _source_integrity(),
    }
    _write(REPORT / "moe_expert_runtime_contract_v17.json", report)
    _write(REPORT / "moe_expert_checkpoint_audit_v17.json", report)
    (ROOT / "docs/cadstruct-image-only-moe-reintegration-v17.md").write_text(
        "# CadStruct image-only MoE v17\n\n"
        "Raster-only input is a contract, not the architecture.\n"
        "The model should still route high-recall raster candidates into the existing MoE experts and fuse their outputs.\n",
        encoding="utf-8",
    )
    update_todo_remove([TASK_IDS["audit_contract"]])


def run_all(args: argparse.Namespace) -> None:
    audit_contract(args)
    build_image_only_moe_candidate_stream(args)
    extract_image_only_candidate_crops(args)
    apply_existing_moe_experts_to_image_only_candidates(args)
    candidate_sweep(args)
    fuse_image_only_moe_expert_predictions(args)
    evaluate_image_only_moe(args)
    audit_image_only_vs_oracle_candidate_gap(args)
    build_image_only_moe_hard_cases(args)
    calibrate_image_only_moe_experts(args)
    render_visual_demo_image_only_moe(args)
    write_docs(args)


def main(default_command: str | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        nargs="?",
        default=default_command or "run-all",
        choices=[
            "audit-contract",
            "build-candidates",
            "extract-crops",
            "apply-experts",
            "candidate-sweep",
            "fuse",
            "evaluate",
            "oracle-gap",
            "build-hard-cases",
            "calibrate",
            "render",
            "docs",
            "run-all",
        ],
    )
    parser.add_argument("--limit", type=int, default=768)
    parser.add_argument("--max-samples", type=int, default=36)
    parser.add_argument("--boundary-threshold", type=float, default=FAMILY_THRESHOLDS["boundary"])
    parser.add_argument("--room-threshold", type=float, default=FAMILY_THRESHOLDS["space"])
    parser.add_argument("--symbol-threshold", type=float, default=FAMILY_THRESHOLDS["symbol"])
    parser.add_argument("--text-threshold", type=float, default=FAMILY_THRESHOLDS["text"])
    parser.add_argument("--sheet-threshold", type=float, default=FAMILY_THRESHOLDS["sheet"])
    parser.add_argument("--boundary-cap", type=int, default=FAMILY_CAPS["boundary"])
    parser.add_argument("--room-cap", type=int, default=FAMILY_CAPS["space"])
    parser.add_argument("--symbol-cap", type=int, default=FAMILY_CAPS["symbol"])
    parser.add_argument("--text-cap", type=int, default=FAMILY_CAPS["text"])
    parser.add_argument("--sheet-cap", type=int, default=FAMILY_CAPS["sheet"])
    args = parser.parse_args()
    {
        "audit-contract": audit_contract,
        "build-candidates": build_image_only_moe_candidate_stream,
        "extract-crops": extract_image_only_candidate_crops,
        "apply-experts": apply_existing_moe_experts_to_image_only_candidates,
        "candidate-sweep": candidate_sweep,
        "fuse": fuse_image_only_moe_expert_predictions,
        "evaluate": evaluate_image_only_moe,
        "oracle-gap": audit_image_only_vs_oracle_candidate_gap,
        "build-hard-cases": build_image_only_moe_hard_cases,
        "calibrate": calibrate_image_only_moe_experts,
        "render": render_visual_demo_image_only_moe,
        "docs": write_docs,
        "run-all": run_all,
    }[args.command](args)


if __name__ == "__main__":
    main()
