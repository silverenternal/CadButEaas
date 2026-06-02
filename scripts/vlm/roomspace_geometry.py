#!/usr/bin/env python3
"""Shared RoomSpace geometry helpers for visual audit and postprocess."""

from __future__ import annotations

import math
from typing import Any


BBox = list[float]
Point = tuple[float, float]


def normalize_bbox(value: Any) -> BBox | None:
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


def node_bbox(node: dict[str, Any]) -> BBox | None:
    geometry = node.get("geometry") if isinstance(node.get("geometry"), dict) else {}
    return normalize_bbox(geometry.get("bbox") or node.get("bbox"))


def bbox_area(bbox: BBox | None) -> float:
    if bbox is None:
        return 0.0
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def bbox_center(bbox: BBox | None) -> Point | None:
    if bbox is None:
        return None
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def bbox_diag(bbox: BBox | None) -> float:
    if bbox is None:
        return 0.0
    return math.hypot(max(0.0, bbox[2] - bbox[0]), max(0.0, bbox[3] - bbox[1]))


def bbox_contains(left: BBox, right: BBox, padding: float = 0.0) -> bool:
    return (
        left[0] - padding <= right[0]
        and left[1] - padding <= right[1]
        and left[2] + padding >= right[2]
        and left[3] + padding >= right[3]
    )


def point_in_bbox(point: Point, bbox: BBox, padding: float = 0.0) -> bool:
    return bbox[0] - padding <= point[0] <= bbox[2] + padding and bbox[1] - padding <= point[1] <= bbox[3] + padding


def bbox_intersects(left: BBox, right: BBox) -> bool:
    return not (left[2] < right[0] or right[2] < left[0] or left[3] < right[1] or right[3] < left[1])


def bbox_distance(left: BBox, right: BBox) -> float:
    dx = max(left[0] - right[2], right[0] - left[2], 0.0)
    dy = max(left[1] - right[3], right[1] - left[3], 0.0)
    return math.hypot(dx, dy)


def point_bbox_distance(point: Point, bbox: BBox) -> float:
    dx = max(bbox[0] - point[0], point[0] - bbox[2], 0.0)
    dy = max(bbox[1] - point[1], point[1] - bbox[3], 0.0)
    return math.hypot(dx, dy)


def adaptive_margin(room_bbox: BBox, canvas_bbox: BBox | None = None, ratio: float = 0.035, cap_ratio: float = 0.012) -> float:
    base = bbox_diag(room_bbox) * ratio
    if canvas_bbox is not None:
        base = min(base, bbox_diag(canvas_bbox) * cap_ratio)
    return max(2.0, base)


def geometry_polygon(geometry: dict[str, Any] | None) -> list[Point]:
    if not isinstance(geometry, dict):
        return []
    source = geometry.get("source_geometry") if isinstance(geometry.get("source_geometry"), dict) else None
    nested = geometry.get("geometry") if isinstance(geometry.get("geometry"), dict) else None
    for item in (source, nested, geometry):
        if not isinstance(item, dict) or str(item.get("type")) != "polygon":
            continue
        points = []
        for point in item.get("points") or []:
            if not isinstance(point, list | tuple) or len(point) != 2:
                continue
            try:
                x, y = float(point[0]), float(point[1])
            except (TypeError, ValueError):
                continue
            if math.isfinite(x) and math.isfinite(y):
                points.append((x, y))
        if len(points) >= 3:
            return points
    return []


def node_polygon(node: dict[str, Any]) -> list[Point]:
    geometry = node.get("geometry") if isinstance(node.get("geometry"), dict) else {}
    return geometry_polygon(geometry)


def point_in_polygon(point: Point, polygon: list[Point]) -> bool:
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i, pi in enumerate(polygon):
        xi, yi = pi
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def room_contains_label(room: dict[str, Any], label: dict[str, Any], canvas_bbox: BBox | None = None) -> dict[str, Any]:
    room_bbox = node_bbox(room)
    label_bbox = node_bbox(label)
    if room_bbox is None or label_bbox is None:
        return {"contains": False, "method": "missing_bbox", "distance": None, "margin": None}
    center = bbox_center(label_bbox)
    if center is None:
        return {"contains": False, "method": "missing_label_center", "distance": None, "margin": None}
    polygon = node_polygon(room)
    if polygon and point_in_polygon(center, polygon):
        return {"contains": True, "method": "polygon_contains_center", "distance": 0.0, "margin": 0.0}
    if point_in_bbox(center, room_bbox):
        if polygon:
            return {"contains": True, "method": "bbox_contains_center_polygon_miss", "distance": 0.0, "margin": 0.0}
        if bbox_contains(room_bbox, label_bbox):
            return {"contains": True, "method": "bbox_contains_label", "distance": 0.0, "margin": 0.0}
        return {"contains": True, "method": "bbox_contains_center", "distance": 0.0, "margin": 0.0}
    if bbox_contains(room_bbox, label_bbox):
        return {"contains": True, "method": "bbox_contains_label", "distance": 0.0, "margin": 0.0}
    margin = adaptive_margin(room_bbox, canvas_bbox)
    distance = point_bbox_distance(center, room_bbox)
    if distance <= margin:
        return {"contains": True, "method": "nearest_with_adaptive_margin", "distance": round(distance, 6), "margin": round(margin, 6)}
    return {"contains": False, "method": "outside", "distance": round(distance, 6), "margin": round(margin, 6)}


def best_room_for_label(label: dict[str, Any], rooms: list[dict[str, Any]], canvas_bbox: BBox | None = None) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    best_room = None
    best_result: dict[str, Any] = {"contains": False, "method": "no_room", "distance": None, "margin": None}
    best_score: tuple[int, float, float] | None = None
    for room in rooms:
        result = room_contains_label(room, label, canvas_bbox)
        room_bbox = node_bbox(room)
        distance = float(result.get("distance") if result.get("distance") is not None else 0.0)
        contains_rank = 0 if result.get("contains") else 1
        area = bbox_area(room_bbox)
        score = (contains_rank, distance, area)
        if best_score is None or score < best_score:
            best_score = score
            best_room = room
            best_result = result
    return best_room, best_result
