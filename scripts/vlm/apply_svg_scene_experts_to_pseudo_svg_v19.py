#!/usr/bin/env python3
"""Run existing CadStruct SVG/scene-era experts on raster-derived pseudo-SVG candidates."""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning:sklearn.utils.parallel")

from scripts.vlm.cadstruct_moe import ExpertPrediction, RoutedCandidate, build_default_experts, describe_experts

warnings.filterwarnings(
    "ignore",
    message=r"`sklearn\.utils\.parallel\.delayed` should be used.*",
    category=UserWarning,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def rel_path(value: str | Path) -> str:
    path = Path(value)
    try:
        return str(path.relative_to(ROOT))
    except Exception:
        return str(path)


def source_integrity() -> dict[str, Any]:
    return {
        "source_mode": "image_only_raster_pseudo_svg_expert_adapter",
        "runtime_input": "raster_image_only",
        "pseudo_svg_status": "inference_derived_intermediate",
        "svg_candidate_ids_used": False,
        "annotation_geometry_used_at_inference": False,
        "runtime_uses_svg_or_cad_geometry": False,
    }


def normalize_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in value[:4]]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def candidate_stream(row: dict[str, Any]) -> list[dict[str, Any]]:
    return list(((row.get("scene_graph") or {}).get("candidate_stream") or []))


def rows_by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("id")): row for row in load_jsonl(path)}


def pick_candidates(
    row_id: str,
    raw_row: dict[str, Any] | None,
    normalized_row: dict[str, Any] | None,
    boundary_source: str,
) -> list[dict[str, Any]]:
    raw = candidate_stream(raw_row or {})
    normalized = candidate_stream(normalized_row or {})
    out: list[dict[str, Any]] = []
    if boundary_source in {"raw", "both"}:
        out.extend(item for item in raw if item.get("family") == "boundary")
    if boundary_source in {"normalized", "both"}:
        out.extend(item for item in normalized if item.get("family") == "boundary")
    out.extend(item for item in normalized if item.get("family") == "space")
    out.extend(item for item in raw if item.get("family") in {"symbol", "text", "sheet"})

    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for item in out:
        candidate_id = str(item.get("candidate_id") or item.get("id") or "")
        if not candidate_id or candidate_id in seen:
            continue
        seen.add(candidate_id)
        copied = json.loads(json.dumps(item, ensure_ascii=False))
        copied["row_id"] = copied.get("row_id") or row_id
        unique.append(copied)
    return unique


def page_size(row: dict[str, Any] | None, candidates: list[dict[str, Any]]) -> tuple[float, float]:
    size = (row or {}).get("image_size")
    if isinstance(size, list) and len(size) >= 2:
        return max(float(size[0]), 1.0), max(float(size[1]), 1.0)
    max_x = max((float((item.get("bbox") or [0, 0, 0, 0])[2]) for item in candidates if normalize_bbox(item.get("bbox"))), default=512.0)
    max_y = max((float((item.get("bbox") or [0, 0, 0, 0])[3]) for item in candidates if normalize_bbox(item.get("bbox"))), default=512.0)
    return max(max_x, 1.0), max(max_y, 1.0)


def build_page_context(candidates: list[dict[str, Any]], width: float, height: float) -> dict[str, Any]:
    rooms: list[dict[str, Any]] = []
    symbols: list[dict[str, Any]] = []
    texts: list[dict[str, Any]] = []
    boundaries: list[dict[str, Any]] = []
    for item in candidates:
        bbox = normalize_bbox(item.get("bbox"))
        if bbox is None:
            continue
        family = str(item.get("family") or "")
        candidate_id = str(item.get("candidate_id") or item.get("id"))
        payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
        if family == "space":
            rooms.append({"id": candidate_id, "room_type": "room", "bbox": bbox, "shape_features": payload.get("shape_features") or {}})
        elif family == "symbol":
            symbols.append({"id": candidate_id, "symbol_type": payload.get("symbol_type") or "generic_symbol", "bbox": bbox})
        elif family == "text":
            texts.append(
                {
                    "id": candidate_id,
                    "text_type": payload.get("text_type") or item.get("candidate_type") or "note_text",
                    "text": payload.get("raw_text") or payload.get("text") or "",
                    "bbox": bbox,
                }
            )
        elif family == "boundary":
            boundaries.append(
                {
                    "id": candidate_id,
                    "semantic_type": payload.get("semantic_type") or item.get("candidate_type") or "hard_wall",
                    "bbox": bbox,
                }
            )
    adjacency = {room["id"]: 0 for room in rooms}
    for i, left in enumerate(rooms):
        for right in rooms[i + 1:]:
            if bbox_touch(left["bbox"], right["bbox"]):
                adjacency[left["id"]] += 1
                adjacency[right["id"]] += 1
    return {
        "width": width,
        "height": height,
        "rooms": rooms,
        "symbols": symbols,
        "texts": texts,
        "boundaries": boundaries,
        "adjacency": adjacency,
    }


def bbox_touch(left: list[float], right: list[float]) -> bool:
    return not (left[2] < right[0] or right[2] < left[0] or left[3] < right[1] or right[3] < left[1])


def to_routed_candidate(item: dict[str, Any], row: dict[str, Any], context: dict[str, Any]) -> RoutedCandidate | None:
    bbox = normalize_bbox(item.get("bbox"))
    if bbox is None:
        return None
    family = str(item.get("family") or "symbol")
    route = str(item.get("route") or family_to_expert(family))
    payload = dict(item.get("payload") or {})
    payload["bbox"] = bbox
    payload["_page_metadata"] = {"width": context["width"], "height": context["height"]}
    payload["page_context"] = context
    payload["rooms"] = context["rooms"]
    payload["image"] = payload.get("image") or row.get("image")
    payload["raster_path"] = payload.get("raster_path") or row.get("image")
    payload["proposal_source"] = payload.get("proposal_source") or "pseudo_svg_expert_adapter_v19"
    payload["source_integrity"] = source_integrity()
    return RoutedCandidate(
        candidate_id=str(item.get("candidate_id") or item.get("id")),
        expert=route,
        family=family,
        candidate_type=str(item.get("candidate_type") or "candidate"),
        confidence=float(item.get("confidence") or 0.0),
        bbox=bbox,
        source="image_only_raster_pseudo_svg",
        payload=payload,
        route_trace={
            "stage": "pseudo_svg_scene_expert_adapter_v19",
            "routing_confidence": float(item.get("confidence") or 0.0),
            "abstain": False,
            **source_integrity(),
        },
    )


def family_to_expert(family: str) -> str:
    return {
        "boundary": "wall_opening",
        "space": "room_space",
        "symbol": "symbol_fixture",
        "text": "text_dimension",
        "sheet": "sheet_layout",
    }.get(family, f"{family}_expert")


def canonical_label(family: str, label: str) -> str:
    if family == "boundary":
        return {"hard_wall": "wall", "partition_wall": "wall", "door": "opening", "opening": "opening", "window": "window"}.get(label, label or "wall")
    if family == "space":
        return label or "room"
    if family == "symbol":
        return label or "generic_symbol"
    if family == "text":
        return label or "note_text"
    return label or family


def postprocess(pred: ExpertPrediction, candidate: RoutedCandidate) -> ExpertPrediction:
    metadata = dict(pred.metadata)
    metadata.setdefault("expert_raw_label", pred.label)
    metadata.setdefault("source_mode", "image_only_raster_pseudo_svg")
    metadata.setdefault("source_integrity", source_integrity())
    metadata.setdefault("proposal_source", candidate.payload.get("proposal_source"))
    metadata.setdefault("pseudo_svg_adapter", "pseudo_svg_scene_expert_adapter_v19")
    return replace(pred, label=canonical_label(pred.family, str(pred.label)), metadata=metadata)


def scene_node(row_id: str, pred: ExpertPrediction) -> dict[str, Any]:
    return {
        "id": pred.candidate_id,
        "row_id": row_id,
        "family": pred.family,
        "semantic_type": pred.label,
        "confidence": pred.confidence,
        "bbox": pred.bbox,
        "source_expert": pred.expert,
        "source": pred.source,
        "metadata": pred.metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-input", default="datasets/pseudo_svg_vectorizer_v19_locked/pseudo_svg_candidates.jsonl")
    parser.add_argument("--normalized-input", default="datasets/pseudo_svg_vectorizer_v19_locked/pseudo_svg_normalized_candidates_recall.jsonl")
    parser.add_argument("--output", default="reports/vlm/pseudo_svg_scene_expert_predictions_v19.jsonl")
    parser.add_argument("--scene-output", default="reports/vlm/pseudo_svg_scene_expert_rows_v19.jsonl")
    parser.add_argument("--audit", default="reports/vlm/pseudo_svg_scene_expert_adapter_v19_audit.json")
    parser.add_argument("--boundary-source", choices=["raw", "normalized", "both"], default="raw")
    parser.add_argument("--families", default="boundary,space,symbol,text")
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    raw_rows = rows_by_id(abs_path(args.raw_input))
    normalized_rows = rows_by_id(abs_path(args.normalized_input))
    row_ids = sorted(set(raw_rows) | set(normalized_rows))
    if args.max_rows:
        row_ids = row_ids[: args.max_rows]
    families = {item.strip() for item in args.families.split(",") if item.strip()}
    experts = build_default_experts(sorted(families))

    prediction_rows: list[dict[str, Any]] = []
    scene_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    family_usage: dict[str, Counter[str]] = defaultdict(Counter)

    for row_id in row_ids:
        raw_row = raw_rows.get(row_id)
        normalized_row = normalized_rows.get(row_id)
        row = raw_row or normalized_row or {"id": row_id}
        picked = pick_candidates(row_id, raw_row, normalized_row, args.boundary_source)
        width, height = page_size(row, picked)
        context = build_page_context(picked, width, height)
        routed = [candidate for item in picked if item.get("family") in families for candidate in [to_routed_candidate(item, row, context)] if candidate]
        by_family: dict[str, list[RoutedCandidate]] = defaultdict(list)
        for candidate in routed:
            by_family[candidate.family].append(candidate)
            counts[f"input_{candidate.family}"] += 1
            source_counts[str(candidate.payload.get("proposal_source") or "unknown")] += 1

        row_predictions: list[ExpertPrediction] = []
        for family in sorted(families):
            expert = experts.get(family)
            if expert is None:
                continue
            family_candidates = by_family.get(family) or []
            preds = expert.predict(family_candidates)
            family_usage[family]["candidates"] += len(family_candidates)
            family_usage[family]["predictions"] += len(preds)
            for pred in preds:
                candidate = next((item for item in family_candidates if item.candidate_id == pred.candidate_id), None)
                if candidate is None:
                    continue
                processed = postprocess(pred, candidate)
                row_predictions.append(processed)
                counts[f"pred_{processed.family}"] += 1
                label_counts[f"{processed.family}:{processed.label}"] += 1

        for pred in row_predictions:
            item = pred.to_dict()
            item["row_id"] = row_id
            prediction_rows.append(item)
        scene_rows.append(
            {
                "id": row_id,
                "image": row.get("image"),
                "image_size": [width, height],
                "source_integrity": source_integrity(),
                "route_trace": {"stage": "pseudo_svg_scene_expert_adapter_v19", **source_integrity()},
                "scene_graph": {
                    "nodes": [scene_node(row_id, pred) for pred in row_predictions],
                    "relations": [],
                    "candidate_stream": [candidate.to_dict() for candidate in routed],
                },
            }
        )

    audit = {
        "version": "pseudo_svg_scene_expert_adapter_v19",
        "task": "P0-PSEUDO-SVG-001",
        "purpose": "Connect existing CadStruct SVG/scene-era strong expert wrappers to raster-derived pseudo-SVG candidate streams.",
        "inputs": {
            "raw": args.raw_input,
            "normalized": args.normalized_input,
            "boundary_source": args.boundary_source,
            "families": sorted(families),
            "rows": len(row_ids),
        },
        "outputs": {"predictions": args.output, "scene_rows": args.scene_output, "audit": args.audit},
        "source_integrity": source_integrity(),
        "expert_registry": describe_experts(experts),
        "counts": {key: int(value) for key, value in counts.items()},
        "label_counts": {key: int(value) for key, value in sorted(label_counts.items())},
        "proposal_source_counts": {key: int(value) for key, value in sorted(source_counts.items())},
        "family_usage": {family: {key: int(value) for key, value in stats.items()} for family, stats in family_usage.items()},
        "adoption_decision": {
            "connected": True,
            "production_adopted": False,
            "reason": "Experts are now callable on pseudo-SVG candidates, but upstream candidate recall and domain-shift audits are still below production gates.",
        },
    }
    write_jsonl(abs_path(args.output), prediction_rows)
    write_jsonl(abs_path(args.scene_output), scene_rows)
    write_json(abs_path(args.audit), audit)
    print(json.dumps({"rows": len(row_ids), "counts": audit["counts"], "loaded": {k: v.get("loaded") for k, v in audit["expert_registry"].items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
