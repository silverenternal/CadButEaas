#!/usr/bin/env python3
"""Reusable metrics for CadStruct VLM evaluation."""

from __future__ import annotations

from typing import Any


def dimension_hit(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    expected_values = [
        item.get("nominal_value")
        for item in expected.get("dimension_candidates", [])
        if item.get("nominal_value") is not None
    ]
    actual_values = [
        item.get("nominal_value")
        for item in actual.get("dimension_candidates", [])
        if item.get("nominal_value") is not None
    ]
    for expected_value in expected_values:
        for actual_value in actual_values:
            try:
                if abs(float(expected_value) - float(actual_value)) <= 1e-6:
                    return True
            except (TypeError, ValueError):
                continue
    return not expected_values


def semantic_hit(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    expected_types = {item.get("semantic_type") for item in expected.get("semantic_candidates", [])}
    actual_types = {item.get("semantic_type") for item in actual.get("semantic_candidates", [])}
    return bool(expected_types & actual_types) if expected_types else True


def semantic_exact_f1(expected: dict[str, Any], actual: dict[str, Any]) -> float:
    expected_items = semantic_pair_set(expected.get("semantic_candidates", []))
    actual_items = semantic_pair_set(actual.get("semantic_candidates", []))
    if not expected_items:
        return 1.0
    if not actual_items:
        return 0.0
    true_positive = len(expected_items & actual_items)
    precision = true_positive / len(actual_items)
    recall = true_positive / len(expected_items)
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def semantic_pair_set(items: Any) -> set[tuple[int, str]]:
    result: set[tuple[int, str]] = set()
    if not isinstance(items, list):
        return result
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            target_id = int(item.get("target_id"))
        except (TypeError, ValueError):
            continue
        result.add((target_id, str(item.get("semantic_type", "detail_line"))))
    return result


def relation_f1(expected: dict[str, Any], actual: dict[str, Any]) -> float:
    expected_edges = edge_set((expected.get("scene_graph") or {}).get("edges", []))
    actual_edges = edge_set((actual.get("scene_graph") or {}).get("edges", []))
    if not expected_edges:
        return 1.0
    if not actual_edges:
        return 0.0
    true_positive = len(expected_edges & actual_edges)
    precision = true_positive / len(actual_edges)
    recall = true_positive / len(expected_edges)
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def edge_set(edges: Any) -> set[tuple[int, int, str]]:
    result: set[tuple[int, int, str]] = set()
    if not isinstance(edges, list):
        return result
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        try:
            source = int(edge.get("source"))
            target = int(edge.get("target"))
        except (TypeError, ValueError):
            continue
        relation = str(edge.get("relation", "related_to"))
        result.add((min(source, target), max(source, target), relation))
    return result


def geometry_consistency(sample: dict[str, Any], actual: dict[str, Any]) -> float:
    graph = ((sample.get("request_hints") or {}).get("primitive_graph") or {})
    nodes = {int(node["id"]): node for node in graph.get("nodes", []) if isinstance(node, dict) and "id" in node}
    candidates = actual.get("semantic_candidates", [])
    if not candidates:
        return 1.0
    checks = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        try:
            target_id = int(item.get("target_id"))
        except (TypeError, ValueError):
            checks.append(False)
            continue
        node = nodes.get(target_id)
        semantic_type = str(item.get("semantic_type", "detail_line"))
        checks.append(semantic_candidate_is_geometrically_plausible(semantic_type, node, graph))
    return round(sum(1 for item in checks if item) / len(checks), 4) if checks else 1.0


def semantic_candidate_is_geometrically_plausible(
    semantic_type: str, node: dict[str, Any] | None, graph: dict[str, Any]
) -> bool:
    if node is None:
        return False
    length = float(node.get("length", 0.0) or 0.0)
    orientation = str(node.get("orientation", ""))
    if semantic_type in {"hard_wall", "partition_wall", "centerline"}:
        return length >= 40.0 and orientation in {"horizontal", "vertical", "diagonal"}
    if semantic_type in {"door", "window", "opening"}:
        return length >= 10.0 and has_graph_contact(int(node["id"]), graph)
    return True


def has_graph_contact(node_id: int, graph: dict[str, Any]) -> bool:
    for edge in graph.get("edges", []):
        if not isinstance(edge, dict):
            continue
        if edge.get("source") == node_id or edge.get("target") == node_id:
            return True
    return False


def safe_rate(count: int, total: int) -> float:
    return round(count / total, 4) if total else 0.0


def count_warnings(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for warning in row.get("warnings", []):
            key = str(warning).split(":", 1)[0]
            counts[key] = counts.get(key, 0) + 1
    return counts
