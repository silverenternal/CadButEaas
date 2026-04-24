"""DXF 解析器 — 基于 ezdxf 库

本模块解析 DXF 文件，输出与 Rust `RawEntity` 兼容的 JSON 格式。
作为 CadButEaas 系统中 Rust dxf crate 的高精度替代方案。

用法:
    python dxf_parser.py <path_to_dxf> [--layers L1,L2] [--exclude-layers L3,L4]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import ezdxf
from ezdxf.entities import (
    Arc,
    Circle,
    Dimension,
    Ellipse,
    Hatch,
    Insert,
    Line,
    LWPolyline,
    MText,
    Polyline,
    Spline,
    Text,
)


# ============================================================================
# 工具函数
# ============================================================================

def _make_metadata(
    entity: Any,
    *,
    layer_filter: set[str] | None = None,
    exclude_layers: set[str] | None = None,
) -> dict[str, Any] | None:
    """从 DXF 实体提取元数据，返回 None 表示应过滤掉该实体。"""
    dxf = entity.dxf
    layer = getattr(dxf, "layer", "0") or "0"

    if layer_filter and layer not in layer_filter:
        return None
    if exclude_layers and layer in exclude_layers:
        return None

    # 颜色
    color = None
    if hasattr(dxf, "color") and dxf.color is not None:
        c = dxf.color
        if isinstance(c, int):
            color = f"ACI:{c}"

    # 线宽
    lineweight = None
    if hasattr(dxf, "lineweight") and dxf.lineweight is not None:
        lw = dxf.lineweight
        if lw >= 0:
            lineweight = lw / 100.0  # ezdxf lineweight 单位是 0.01mm

    # 线型
    line_type = None
    if hasattr(dxf, "linetype") and dxf.linetype:
        line_type = dxf.linetype

    # handle
    handle = None
    if hasattr(dxf, "handle") and dxf.handle:
        handle = dxf.handle

    return {
        "layer": layer,
        "color": color,
        "lineweight": lineweight,
        "line_type": line_type,
        "handle": handle,
        "material": None,
        "width": None,
    }


def _default_metadata() -> dict[str, Any]:
    return {
        "layer": None,
        "color": None,
        "lineweight": None,
        "line_type": None,
        "handle": None,
        "material": None,
        "width": None,
    }


# ============================================================================
# 实体转换器
# ============================================================================

def _to_line(entity: Line, meta: dict) -> dict:
    return {
        "type": "line",
        "start": [entity.dxf.start.x, entity.dxf.start.y],
        "end": [entity.dxf.end.x, entity.dxf.end.y],
        "metadata": meta,
        "semantic": None,
    }


def _to_lwpolyline(entity: LWPolyline, meta: dict) -> list[dict]:
    """LWPOLYLINE 可能包含 bulge 信息，需要分段处理。

    如果存在 bulge != 0，拆分为 Path；否则合并为 Polyline。
    """
    # get_points("xyb") 返回 (x, y, bulge) 元组
    points = entity.get_points("xyb")
    closed = entity.closed
    has_bulge = False

    for pt in points:
        bulge = pt[2] if len(pt) > 2 else 0.0
        if abs(bulge) > 1e-10:
            has_bulge = True
            break

    if not has_bulge:
        pts = [[p[0], p[1]] for p in points]
        return [{
            "type": "polyline",
            "points": pts,
            "closed": closed,
            "metadata": meta,
            "semantic": None,
        }]

    # 有 bulge → 使用 Path 命令
    commands = []
    for i, pt in enumerate(points):
        cmd = {
            "cmd": "MoveTo" if i == 0 else "LineTo",
            "x": pt[0],
            "y": pt[1],
        }
        commands.append(cmd)
    if closed:
        commands.append({"cmd": "Close"})

    return [{
        "type": "path",
        "commands": commands,
        "metadata": meta,
        "semantic": None,
    }]


def _to_polyline(entity: Polyline, meta: dict) -> list[dict]:
    """3D POLYLINE 或含 bulge 的 2D polyline。"""
    try:
        points = list(entity.points)
    except Exception:
        return []

    if not points:
        return []

    closed = entity.is_closed

    # 检查 bulge
    has_bulge = False
    for pt in points:
        if hasattr(pt, "bulge") and abs(pt.bulge) > 1e-10:
            has_bulge = True
            break

    if not has_bulge:
        pts = [[p[0], p[1]] for p in points]
        return [{
            "type": "polyline",
            "points": pts,
            "closed": closed,
            "metadata": meta,
            "semantic": None,
        }]

    # 有 bulge → Path
    commands = []
    for i, pt in enumerate(points):
        cmd = {
            "cmd": "MoveTo" if i == 0 else "LineTo",
            "x": pt[0],
            "y": pt[1],
        }
        commands.append(cmd)
    if closed:
        commands.append({"cmd": "Close"})

    return [{
        "type": "path",
        "commands": commands,
        "metadata": meta,
        "semantic": None,
    }]


def _to_arc(entity: Arc, meta: dict) -> dict:
    return {
        "type": "arc",
        "center": [entity.dxf.center.x, entity.dxf.center.y],
        "radius": entity.dxf.radius,
        "start_angle": entity.dxf.start_angle,
        "end_angle": entity.dxf.end_angle,
        "metadata": meta,
        "semantic": None,
    }


def _to_circle(entity: Circle, meta: dict) -> dict:
    return {
        "type": "circle",
        "center": [entity.dxf.center.x, entity.dxf.center.y],
        "radius": entity.dxf.radius,
        "metadata": meta,
        "semantic": None,
    }


def _to_ellipse(entity: Ellipse, meta: dict) -> list[dict]:
    """椭圆 → 使用 Path 的 ArcTo 命令近似。"""
    center = entity.dxf.center
    major_axis = entity.major_axis
    minor_ratio = entity.dxf.ratio
    start_param = entity.dxf.start_param
    end_param = entity.dxf.end_param

    # 计算长短轴
    a = math.sqrt(major_axis[0]**2 + major_axis[1]**2)
    b = a * minor_ratio

    # 椭圆的旋转角
    angle = math.atan2(major_axis[1], major_axis[0])
    angle_deg = math.degrees(angle)

    # 参数转角度
    start_deg = math.degrees(start_param)
    end_deg = math.degrees(end_param)

    # 使用多段线近似椭圆弧
    num_segments = max(32, int(abs(end_deg - start_deg) / 5))
    if num_segments < 4:
        num_segments = 32

    pts = []
    for i in range(num_segments + 1):
        t = start_param + (end_param - start_param) * i / num_segments
        x = a * math.cos(t)
        y = b * math.sin(t)
        # 旋转到世界坐标
        x_rot = x * math.cos(angle) - y * math.sin(angle) + center[0]
        y_rot = x * math.sin(angle) + y * math.cos(angle) + center[1]
        pts.append([x_rot, y_rot])

    return [{
        "type": "polyline",
        "points": pts,
        "closed": abs(end_deg - start_deg) >= 359.9,
        "metadata": meta,
        "semantic": None,
    }]


def _spline_to_polyline(spline: Spline, num_points: int = 64) -> list[float]:
    """将 NURBS 样条曲线离散化为点集。"""
    try:
        points = list(spline.flattening(0.01))
        return [[p[0], p[1]] for p in points]
    except Exception:
        # Fallback: 均匀采样
        pts = []
        for i in range(num_points + 1):
            t = i / num_points
            try:
                pt = spline.point(t)
                pts.append([pt[0], pt[1]])
            except Exception:
                pass
        return pts


def _to_spline(entity: Spline, meta: dict) -> list[dict]:
    """SPLINE → 离散化为 Polyline（与 Rust 解析器一致）。"""
    pts = _spline_to_polyline(entity)
    if not pts:
        return []

    closed = entity.closed
    return [{
        "type": "polyline",
        "points": pts,
        "closed": closed,
        "metadata": meta,
        "semantic": None,
    }]


def _to_text(entity: Text | MText, meta: dict) -> dict:
    dxf = entity.dxf
    position = [dxf.insert.x, dxf.insert.y]
    content = dxf.text if hasattr(dxf, "text") else str(dxf.text)
    height = getattr(dxf, "height", 2.5) or 2.5
    rotation = getattr(dxf, "rotation", 0.0) or 0.0

    style_name = None
    if hasattr(dxf, "style") and dxf.style:
        style_name = dxf.style

    return {
        "type": "text",
        "position": position,
        "content": content,
        "height": height,
        "rotation": rotation,
        "style_name": style_name,
        "align_left": None,
        "align_right": None,
        "metadata": meta,
        "semantic": None,
    }


def _to_insert(entity: Insert, meta: dict, doc: ezdxf.document.Drawing) -> dict:
    dxf = entity.dxf
    return {
        "type": "block_reference",
        "block_name": dxf.name,
        "insertion_point": [dxf.insert.x, dxf.insert.y],
        "scale": [
            getattr(dxf, "xscale", 1.0) or 1.0,
            getattr(dxf, "yscale", 1.0) or 1.0,
            getattr(dxf, "zscale", 1.0) or 1.0,
        ],
        "rotation": getattr(dxf, "rotation", 0.0) or 0.0,
        "metadata": meta,
        "semantic": None,
    }


_DIMENSION_TYPE_MAP = {
    0: "linear",
    1: "aligned",
    2: "angular",
    3: "diameter",
    4: "radial",
    5: "angular",  # 3-point angular
}


def _to_dimension(entity: Dimension, meta: dict) -> dict:
    dxf = entity.dxf
    dim_type_raw = getattr(dxf, "dxftype", 0)
    if hasattr(dxf, "dimtype"):
        dim_type_raw = dxf.dimtype

    dim_type = _DIMENSION_TYPE_MAP.get(dim_type_raw, "linear")

    measurement = getattr(dxf, "measurement", 0.0) or 0.0
    text = None
    if hasattr(dxf, "text") and dxf.text:
        text = dxf.text

    # 定义点
    def_pts = []
    for attr in ["defpoint", "defpoint2", "defpoint3", "defpoint4"]:
        if hasattr(dxf, attr):
            pt = getattr(dxf, attr)
            if pt is not None:
                def_pts.append([pt[0], pt[1]])

    return {
        "type": "dimension",
        "dimension_type": dim_type,
        "measurement": measurement,
        "text": text,
        "definition_points": def_pts,
        "metadata": meta,
        "semantic": None,
    }


def _to_attrib(entity, meta: dict) -> dict:
    """ATTRIB → RawEntity::Attribute（块属性值）。"""
    dxf = entity.dxf
    return {
        "type": "attribute",
        "tag": getattr(dxf, "tag", "") or "",
        "value": getattr(dxf, "text", "") or "",
        "position": [getattr(dxf, "insert", (0, 0, 0))[0], getattr(dxf, "insert", (0, 0, 0))[1]],
        "height": getattr(dxf, "height", 1.0) or 1.0,
        "rotation": getattr(dxf, "rotation", 0.0) or 0.0,
        "metadata": meta,
        "semantic": None,
    }


def _to_attdef(entity, meta: dict) -> dict:
    """ATTDEF → RawEntity::AttributeDefinition（属性定义/模板）。"""
    dxf = entity.dxf
    return {
        "type": "attribute_definition",
        "tag": getattr(dxf, "tag", "") or "",
        "default_value": getattr(dxf, "text", "") or "",
        "prompt": getattr(dxf, "prompt", "") or "",
        "position": [getattr(dxf, "insert", (0, 0, 0))[0], getattr(dxf, "insert", (0, 0, 0))[1]],
        "height": getattr(dxf, "height", 1.0) or 1.0,
        "rotation": getattr(dxf, "rotation", 0.0) or 0.0,
        "metadata": meta,
        "semantic": None,
    }


def _to_leader(entity, meta: dict) -> dict | None:
    """LEADER → RawEntity::Leader（引线标注）。"""
    try:
        pts = list(entity.vertices)
    except Exception:
        pts = []

    if len(pts) < 2:
        return None

    return {
        "type": "leader",
        "points": [[p[0], p[1]] for p in pts],
        "annotation_text": None,
        "metadata": meta,
        "semantic": None,
    }


def _to_point(entity, meta: dict) -> dict:
    """POINT → RawEntity::Point（测量标记/参考点）。"""
    dxf = entity.dxf
    loc = getattr(dxf, "location", (0, 0, 0)) or (0, 0, 0)
    return {
        "type": "point",
        "position": [loc[0], loc[1]],
        "metadata": meta,
        "semantic": None,
    }


def _to_mline(entity, meta: dict) -> dict | None:
    """MLINE → RawEntity::MLine（多线，建筑图纸中常用于表示墙体）。"""
    dxf = entity.dxf
    try:
        verts = list(entity.vertices)
    except Exception:
        verts = []

    if len(verts) < 2:
        return None

    flags = getattr(dxf, "flags", 0) or 0
    return {
        "type": "mline",
        "center_line": [[v[0], v[1]] for v in verts],
        "closed": (flags & 1) != 0,
        "style_name": getattr(dxf, "style_name", "") or "",
        "scale_factor": getattr(dxf, "scale", 1.0) or 1.0,
        "metadata": meta,
        "semantic": None,
    }


def _to_ray(entity, meta: dict) -> dict:
    """RAY/XLINE → RawEntity::Ray（射线/构造线）。"""
    dxf = entity.dxf
    dxftype = getattr(dxf, "dxftype", "") or ""
    if dxftype == "XLINE":
        base = getattr(dxf, "base_point", (0, 0, 0)) or (0, 0, 0)
    else:
        base = getattr(dxf, "start", (0, 0, 0)) or (0, 0, 0)

    direction = getattr(dxf, "unit_direction", (1, 0, 0)) or (1, 0, 0)
    return {
        "type": "ray",
        "start": [base[0], base[1]],
        "direction": [direction[0], direction[1]],
        "metadata": meta,
        "semantic": None,
    }


def _bulge_to_arc(start, end, bulge, meta) -> dict:
    """将 bulge 转换为圆弧实体。"""
    # bulge = tan(theta/4)
    theta = 4.0 * math.atan(bulge)
    chord_len = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)

    if chord_len < 1e-10:
        return None

    # 半径
    radius = abs(chord_len / (2.0 * math.sin(theta / 2.0))) if abs(math.sin(theta / 2.0)) > 1e-10 else chord_len / 2.0

    # 中点
    mid_x = (start[0] + end[0]) / 2.0
    mid_y = (start[1] + end[1]) / 2.0

    # 垂直方向
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    perp_x = -dy
    perp_y = dx

    # 中心到弦中点的距离
    sagitta = abs(bulge) * chord_len / 2.0
    dist = radius - sagitta if radius > sagitta else 0

    # 规范化垂直方向
    perp_len = math.sqrt(perp_x**2 + perp_y**2)
    if perp_len < 1e-10:
        return None
    perp_x /= perp_len
    perp_y /= perp_len

    # 中心
    sign = 1.0 if bulge > 0 else -1.0
    cx = mid_x + sign * perp_x * dist
    cy = mid_y + sign * perp_y * dist

    # 起始和终止角度
    start_angle = math.degrees(math.atan2(start[1] - cy, start[0] - cx))
    end_angle = math.degrees(math.atan2(end[1] - cy, end[0] - cx))

    return {
        "type": "arc",
        "center": [cx, cy],
        "radius": radius,
        "start_angle": start_angle,
        "end_angle": end_angle,
        "metadata": meta,
        "semantic": None,
    }


def _to_hatch(entity: Hatch, meta: dict) -> list[dict]:
    """HATCH → 边界路径 + 图案信息。"""
    results = []

    dxf = entity.dxf
    solid_fill = getattr(dxf, "solid_fill", False) or False

    # 图案
    pattern_name = getattr(dxf, "pattern_name", "") or ""
    if solid_fill:
        pattern = {"type": "solid", "color": {"r": 0, "g": 0, "b": 0, "a": 255}}
    elif pattern_name:
        pattern = {"type": "predefined", "name": pattern_name}
    else:
        pattern = {"type": "predefined", "name": "ANSI31"}

    scale = getattr(dxf, "pattern_scale", 1.0) or 1.0
    angle = getattr(dxf, "pattern_angle", 0.0) or 0.0

    # 边界路径
    boundary_paths = []
    try:
        for edge_path in entity.paths.paths:
            path_type = getattr(edge_path, "path_type_flags", 0)
            has_lines = path_type & 1
            has_arcs = path_type & 2
            has_polylines = path_type & 4
            has_ellipses = path_type & 8
            has_splines = path_type & 16

            if has_polylines and hasattr(edge_path, "polyline_path"):
                poly = edge_path.polyline_path
                pts_with_bulges = list(poly)
                pts = [[p[0], p[1]] for p in pts_with_bulges]
                bulges = [p[2] if len(p) > 2 else 0.0 for p in pts_with_bulges]
                has_any_bulge = any(abs(b) > 1e-10 for b in bulges)
                boundary_paths.append({
                    "type": "polyline",
                    "points": pts,
                    "closed": True,
                    "bulges": bulges if has_any_bulge else None,
                })

            if has_lines and hasattr(edge_path, "edges"):
                for edge in edge_path.edges:
                    edge_type = getattr(edge, "edge_type", 0)
                    if edge_type == 1:  # line
                        boundary_paths.append({
                            "type": "polyline",
                            "points": [
                                [edge.start_point.x, edge.start_point.y],
                                [edge.end_point.x, edge.end_point.y],
                            ],
                            "closed": False,
                        })
                    elif edge_type == 2:  # arc
                        boundary_paths.append({
                            "type": "arc",
                            "center": [edge.center_point.x, edge.center_point.y],
                            "radius": edge.radius,
                            "start_angle": math.degrees(edge.start_angle),
                            "end_angle": math.degrees(edge.end_angle),
                            "ccw": edge.is_counter_clockwise,
                        })
                    elif edge_type == 3:  # ellipse
                        boundary_paths.append({
                            "type": "ellipse_arc",
                            "center": [edge.center_point.x, edge.center_point.y],
                            "major_axis": [edge.end_point.x, edge.end_point.y],
                            "minor_axis_ratio": edge.minor_axis_ratio,
                            "start_angle": math.degrees(edge.start_angle),
                            "end_angle": math.degrees(edge.end_angle),
                            "ccw": edge.is_counter_clockwise,
                        })
                    elif edge_type == 4:  # spline
                        try:
                            cp = list(edge.control_points)
                            knots = list(edge.knot_values)
                            degree = getattr(edge, "degree", 3) or 3
                            weights = list(edge.weights) if hasattr(edge, "weights") and edge.weights else None
                            boundary_paths.append({
                                "type": "spline",
                                "control_points": [[p.x, p.y] for p in cp],
                                "knots": knots,
                                "degree": degree,
                                "weights": weights,
                                "fit_points": None,
                                "flags": None,
                            })
                        except Exception:
                            pass
    except Exception:
        pass

    if boundary_paths:
        results.append({
            "type": "hatch",
            "boundary_paths": boundary_paths,
            "pattern": pattern,
            "solid_fill": solid_fill,
            "metadata": meta,
            "semantic": None,
            "scale": scale,
            "angle": angle,
        })

    return results


# ============================================================================
# 主解析逻辑
# ============================================================================

def parse_dxf(
    filepath: str,
    *,
    layers: list[str] | None = None,
    exclude_layers: list[str] | None = None,
) -> dict[str, Any]:
    """解析 DXF 文件并返回与 RawEntity 兼容的 JSON。

    返回值:
        {
            "success": bool,
            "entities": [...],
            "errors": [...],
            "stats": {...},
        }
    """
    result: dict[str, Any] = {
        "success": False,
        "entities": [],
        "errors": [],
        "stats": {},
    }

    layer_filter = set(layers) if layers else None
    exclude_set = set(exclude_layers) if exclude_layers else None

    if not os.path.isfile(filepath):
        result["errors"].append(f"File not found: {filepath}")
        return result

    try:
        doc = ezdxf.readfile(filepath)
        msp = doc.modelspace()
    except Exception as e:
        result["errors"].append(f"Failed to read DXF file: {e}")
        return result

    entities = []
    errors = []
    stats: dict[str, int] = {}

    for entity in msp:
        dxftype = entity.dxftype()

        try:
            meta = _make_metadata(
                entity,
                layer_filter=layer_filter,
                exclude_layers=exclude_set,
            )
            if meta is None:
                continue

            converted: list[dict] | dict | None = None

            if dxftype == "LINE":
                converted = _to_line(entity, meta)
            elif dxftype == "LWPOLYLINE":
                converted = _to_lwpolyline(entity, meta)
            elif dxftype == "POLYLINE":
                converted = _to_polyline(entity, meta)
            elif dxftype == "ARC":
                converted = _to_arc(entity, meta)
            elif dxftype == "CIRCLE":
                converted = _to_circle(entity, meta)
            elif dxftype == "ELLIPSE":
                converted = _to_ellipse(entity, meta)
            elif dxftype == "SPLINE":
                converted = _to_spline(entity, meta)
            elif dxftype in ("TEXT", "MTEXT"):
                converted = _to_text(entity, meta)
            elif dxftype == "INSERT":
                converted = _to_insert(entity, meta, doc)
            elif dxftype in ("DIMENSION", "LinearDimension", "AlignedDimension",
                             "AngularDimension", "RadialDimension", "DiametricDimension"):
                converted = _to_dimension(entity, meta)
            elif dxftype == "HATCH":
                converted = _to_hatch(entity, meta)
            elif dxftype == "ATTRIB":
                converted = _to_attrib(entity, meta)
            elif dxftype == "ATTDEF":
                converted = _to_attdef(entity, meta)
            elif dxftype == "LEADER":
                converted = _to_leader(entity, meta)
            elif dxftype == "POINT":
                converted = _to_point(entity, meta)
            elif dxftype == "MLINE":
                converted = _to_mline(entity, meta)
            elif dxftype in ("RAY", "XLINE"):
                converted = _to_ray(entity, meta)
            else:
                continue  # 跳过不支持的实体类型

            if converted is not None:
                if isinstance(converted, list):
                    entities.extend(converted)
                else:
                    entities.append(converted)

            stats[dxftype] = stats.get(dxftype, 0) + 1

        except Exception as e:
            errors.append({
                "entity_type": dxftype,
                "handle": getattr(entity.dxf, "handle", "unknown"),
                "error": str(e),
            })

    result["success"] = True
    result["entities"] = entities
    result["errors"] = errors
    result["stats"] = stats

    return result


# ============================================================================
# CLI 入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Parse DXF file using ezdxf")
    parser.add_argument("filepath", help="Path to DXF file")
    parser.add_argument("--layers", help="Comma-separated layer whitelist")
    parser.add_argument("--exclude-layers", help="Comma-separated layer blacklist")
    parser.add_argument("--python-path", help="Path to Python interpreter (for diagnostics)")

    args = parser.parse_args()

    layers = None
    if args.layers:
        layers = [l.strip() for l in args.layers.split(",") if l.strip()]

    exclude_layers = None
    if args.exclude_layers:
        exclude_layers = [l.strip() for l in args.exclude_layers.split(",") if l.strip()]

    result = parse_dxf(args.filepath, layers=layers, exclude_layers=exclude_layers)

    # 输出 JSON 到 stdout
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")

    # 退出码: 0 = 成功, 1 = 失败
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
