#!/usr/bin/env python3
"""Raster VLM output parsing, repair, and normalization.

This module is deliberately independent from the HTTP sidecar. It gives us a
single auditable contract for model output, offline evaluation, and unit-level
regression checks.
"""

from __future__ import annotations

import json
import re
from typing import Any


def parse_model_json(raw_text: str) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    candidate = raw_text.strip()
    if "```" in candidate:
        candidate = re.sub(r"^```(?:json)?", "", candidate.strip(), flags=re.IGNORECASE).strip()
        candidate = re.sub(r"```$", "", candidate.strip()).strip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start >= 0 and end > start:
        candidate = candidate[start : end + 1]
    candidate = re.sub(r",(\s*[}\]])", r"\1", candidate)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        warnings.append(f"json_parse_failed: {exc}")
        recovered = recover_partial_json(candidate)
        if recovered:
            warnings.append("partial_json_recovered")
            return recovered, warnings
        return {}, warnings
    if not isinstance(parsed, dict):
        warnings.append("json_parse_failed: top-level value is not object")
        return {}, warnings
    return parsed, warnings


def recover_partial_json(candidate: str) -> dict[str, Any]:
    recovered: dict[str, Any] = {}
    semantics = recover_object_list(candidate, "semantic_candidates")
    if semantics:
        recovered["semantic_candidates"] = semantics
    symbols = recover_object_list(candidate, "symbol_candidates")
    if symbols:
        recovered["symbol_candidates"] = symbols
    dimensions = recover_object_list(candidate, "dimension_candidates")
    if dimensions:
        recovered["dimension_candidates"] = dimensions
    return recovered


def recover_object_list(candidate: str, field_name: str) -> list[dict[str, Any]]:
    field_match = re.search(rf'"{re.escape(field_name)}"\s*:\s*\[', candidate)
    if not field_match:
        return []
    start = field_match.end()
    end_match = re.search(r'\]\s*,\s*"\w+"\s*:', candidate[start:])
    end = start + end_match.start() if end_match else len(candidate)
    section = candidate[start:end]
    objects = []
    for match in re.finditer(r"\{[^{}]*\}", section, flags=re.DOTALL):
        text = re.sub(r",(\s*[}\]])", r"\1", match.group(0))
        try:
            item = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            objects.append(item)
    return objects


def normalize_output(parsed: dict[str, Any], backend: str) -> dict[str, Any]:
    warnings = parsed.get("warnings") if isinstance(parsed.get("warnings"), list) else []
    scene_graph = normalize_scene_graph(parsed.get("scene_graph"))
    semantics = [normalize_semantic(item, backend) for item in list_field(parsed, "semantic_candidates")]
    repair_warnings = []
    if not semantics and scene_graph:
        semantics = semantic_candidates_from_scene_graph(scene_graph, backend)
        if semantics:
            repair_warnings.append("semantic_candidates_repaired_from_scene_graph")
    return {
        "dimension_candidates": [
            normalize_dimension(item, backend) for item in list_field(parsed, "dimension_candidates")
        ],
        "symbol_candidates": [normalize_symbol(item) for item in list_field(parsed, "symbol_candidates")],
        "semantic_candidates": semantics,
        "scene_graph": scene_graph,
        "warnings": [str(item) for item in warnings] + repair_warnings,
    }


def normalize_dimension(item: dict[str, Any], backend: str) -> dict[str, Any]:
    return {
        "raw_text": str(item.get("raw_text", "")),
        "nominal_value": number_or_none(item.get("nominal_value")),
        "tolerance_type": str_or_none(item.get("tolerance_type")),
        "upper_deviation": number_or_none(item.get("upper_deviation")),
        "lower_deviation": number_or_none(item.get("lower_deviation")),
        "geometric_type": str_or_none(item.get("geometric_type")),
        "datums": string_list(item.get("datums")),
        "roughness": number_or_none(item.get("roughness")),
        "bbox": bbox(item.get("bbox")),
        "confidence": confidence(item.get("confidence")),
        "source": str(item.get("source") or f"{backend}_vlm_http"),
    }


def normalize_symbol(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "symbol_type": str(item.get("symbol_type", "unknown")),
        "confidence": confidence(item.get("confidence")),
        "bbox": bbox(item.get("bbox")),
        "rotation": float(item.get("rotation", 0.0) or 0.0),
    }


def normalize_semantic(item: dict[str, Any], backend: str) -> dict[str, Any]:
    semantic_type = str(item.get("semantic_type", "detail_line"))
    semantic_aliases = {
        "wall": "hard_wall",
        "structural_wall": "hard_wall",
        "line": "detail_line",
    }
    return {
        "target_id": int_or_default(item.get("target_id", item.get("primitive_id", 0))),
        "semantic_type": semantic_aliases.get(semantic_type, semantic_type),
        "confidence": confidence(item.get("confidence")),
        "source": str(item.get("source") or f"{backend}_vlm_http"),
    }


def normalize_scene_graph(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    nodes = []
    for item in value.get("nodes", []):
        if not isinstance(item, dict):
            continue
        try:
            nodes.append(
                {
                    "id": int(item.get("id")),
                    "semantic_type": str(item.get("semantic_type", "detail_line")),
                    "primitive_id": int(item.get("primitive_id", item.get("id"))),
                }
            )
        except (TypeError, ValueError):
            continue
    edges = []
    for item in value.get("edges", []):
        if not isinstance(item, dict):
            continue
        try:
            edges.append(
                {
                    "source": int(item.get("source")),
                    "target": int(item.get("target")),
                    "relation": str(item.get("relation", "related_to")),
                }
            )
        except (TypeError, ValueError):
            continue
    return {"nodes": nodes, "edges": edges}


def semantic_candidates_from_scene_graph(scene_graph: dict[str, Any], backend: str) -> list[dict[str, Any]]:
    candidates = []
    for node in scene_graph.get("nodes", []):
        if not isinstance(node, dict):
            continue
        candidates.append(
            {
                "target_id": int_or_default(node.get("primitive_id", node.get("id", 0))),
                "semantic_type": str(node.get("semantic_type", "detail_line")),
                "confidence": 0.5,
                "source": f"{backend}_scene_graph_repair",
            }
        )
    return candidates


def list_field(parsed: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = parsed.get(key)
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def bbox(value: Any) -> list[float]:
    if not isinstance(value, list):
        return [0.0, 0.0, 0.0, 0.0]
    values = [float(item or 0.0) for item in value[:4]]
    return values + [0.0] * (4 - len(values))


def confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


def number_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def str_or_none(value: Any) -> str | None:
    return None if value is None else str(value)


def int_or_default(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
