#!/usr/bin/env python3
"""Shared helpers for the v18 relation-graph reconstruction rebuild."""

from __future__ import annotations

import json
import math
import zlib
from bisect import bisect_right
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageStat

from build_topology_relations_v18 import evaluate_relations, integrity, load_gold, relation_label, write_json, write_jsonl
from nms_topology_relations_v18 import cluster_candidates, load_by_id, load_jsonl, row_candidate_map
from train_relation_reranker_v18 import DEFAULT_SPLITTER_MODEL, build_symbol_instance_ids, feature_vector, safe_float, topology_pair_key

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/vlm"
CHECKPOINT = ROOT / "checkpoints/relation_graph_policy_v18"

DEFAULT_INPUT = REPORT / "topology_relations_v18_symbol_boundary_fixed_candidates.jsonl"
DEFAULT_ADAPTER = REPORT / "detector_adapter_v18_symbol_boundary_fixed_candidates.jsonl"
DEFAULT_DATASET = REPORT / "relation_graph_reconstruction_v18_dataset.jsonl"
DEFAULT_MODEL = CHECKPOINT / "model.json"
DEFAULT_EVAL = REPORT / "relation_graph_reconstruction_v18_eval.json"
DEFAULT_AUDIT = REPORT / "relation_graph_reconstruction_v18_audit.json"
DEFAULT_REVIEW = REPORT / "relation_graph_reconstruction_v18_review_pack.json"

RELATIONS = ["bounded_by", "contains_symbol", "labeled_by_text", "adjacent_to"]
_RASTER_CACHE: dict[str, tuple[Image.Image, Image.Image] | None] = {}
_VISUAL_FEATURE_CACHE: dict[tuple[str, str, tuple[float, float, float, float] | None, float], dict[str, float]] = {}


def split_name(row_id: str) -> str:
    bucket = zlib.crc32(row_id.encode("utf-8")) % 10
    if bucket < 7:
        return "train"
    if bucket < 8:
        return "dev"
    return "test"


def load_pages(path: Path, smoke: bool = False, limit: int | None = None) -> list[dict[str, Any]]:
    rows = load_jsonl(path, limit=limit)
    return rows[:5] if smoke else rows


def transform_value(name: str, value: float) -> float:
    if name.endswith("_area") or name in {"component_node_count", "component_edge_count", "component_bridge_count"}:
        return math.log1p(max(0.0, value)) / 12.0
    if name.endswith("_width") or name.endswith("_height") or name.endswith("_degree"):
        return math.log1p(max(0.0, value)) / 8.0
    if name.endswith("_rank") or name.endswith("_cap_rank"):
        return 1.0 / (1.0 + max(0.0, value))
    if name.endswith("_aspect"):
        return math.log(max(value, 1e-6))
    if name.endswith("_ratio"):
        return max(0.0, min(1.0, value))
    return max(0.0, value)


def vectorize(features: dict[str, Any], names: list[str]) -> np.ndarray:
    return np.asarray([transform_value(name, safe_float(features.get(name))) for name in names], dtype=np.float32)


def candidate_bbox(candidate: dict[str, Any] | None) -> list[float] | None:
    if not isinstance(candidate, dict):
        return None
    value = candidate.get("bbox")
    if not isinstance(value, list) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(item) for item in value[:4]]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def candidate_raster_path(candidate: dict[str, Any] | None) -> str:
    if not isinstance(candidate, dict):
        return ""
    payload = candidate.get("payload") if isinstance(candidate.get("payload"), dict) else {}
    return str(payload.get("raster_path") or payload.get("image") or candidate.get("image") or "")


def load_raster(path: str) -> tuple[Image.Image, Image.Image] | None:
    if not path:
        return None
    if path in _RASTER_CACHE:
        return _RASTER_CACHE[path]
    try:
        image_path = Path(path)
        if not image_path.is_absolute():
            image_path = ROOT / image_path
        image = Image.open(image_path).convert("L")
        _RASTER_CACHE[path] = (image, image.filter(ImageFilter.FIND_EDGES))
    except (FileNotFoundError, OSError):
        _RASTER_CACHE[path] = None
    return _RASTER_CACHE[path]


def crop_image(image: Image.Image, bbox: list[float], pad: float) -> Image.Image | None:
    x1 = max(0, int(math.floor(bbox[0] - pad)))
    y1 = max(0, int(math.floor(bbox[1] - pad)))
    x2 = min(image.width, int(math.ceil(bbox[2] + pad)))
    y2 = min(image.height, int(math.ceil(bbox[3] + pad)))
    if x2 <= x1 or y2 <= y1:
        return None
    return image.crop((x1, y1, x2, y2))


def histogram_fraction(image: Image.Image, start: int, end: int) -> float:
    hist = image.histogram()
    total = sum(hist)
    if total <= 0:
        return 0.0
    return float(sum(hist[start:end]) / total)


def raster_layout_features(patch: Image.Image) -> dict[str, float]:
    width, height = patch.size
    if width <= 2 or height <= 2:
        return {
            "dark_center_ratio": 1.0,
            "dark_border_ratio": 1.0,
            "dark_horizontal_balance": 0.0,
            "dark_vertical_balance": 0.0,
            "aspect": float(width / max(height, 1)),
            "area": float(width * height),
        }
    center = patch.crop((width // 4, height // 4, max(width // 4 + 1, width * 3 // 4), max(height // 4 + 1, height * 3 // 4)))
    top = patch.crop((0, 0, width, max(1, height // 4)))
    bottom = patch.crop((0, max(0, height * 3 // 4), width, height))
    left = patch.crop((0, 0, max(1, width // 4), height))
    right = patch.crop((max(0, width * 3 // 4), 0, width, height))
    dark = max(histogram_fraction(patch, 0, 128), 1e-6)
    left_dark = histogram_fraction(left, 0, 128)
    right_dark = histogram_fraction(right, 0, 128)
    top_dark = histogram_fraction(top, 0, 128)
    bottom_dark = histogram_fraction(bottom, 0, 128)
    return {
        "dark_center_ratio": histogram_fraction(center, 0, 128) / dark,
        "dark_border_ratio": ((left_dark + right_dark + top_dark + bottom_dark) / 4.0) / dark,
        "dark_horizontal_balance": (left_dark - right_dark) / max(left_dark + right_dark, 1e-6),
        "dark_vertical_balance": (top_dark - bottom_dark) / max(top_dark + bottom_dark, 1e-6),
        "aspect": float(width / max(height, 1)),
        "area": float(width * height),
    }


def raster_patch_features(
    raster: tuple[Image.Image, Image.Image] | None,
    bbox: list[float] | None,
    *,
    prefix: str,
    pad: float,
) -> dict[str, float]:
    defaults = {
        f"{prefix}_mean": 0.0,
        f"{prefix}_std": 0.0,
        f"{prefix}_dark_density": 0.0,
        f"{prefix}_very_dark_density": 0.0,
        f"{prefix}_mid_dark_density": 0.0,
        f"{prefix}_edge_density": 0.0,
        f"{prefix}_edge_strong_density": 0.0,
        f"{prefix}_context_dark_density": 0.0,
        f"{prefix}_context_edge_density": 0.0,
        f"{prefix}_dark_ratio": 0.0,
        f"{prefix}_edge_ratio": 0.0,
        f"{prefix}_dark_center_ratio": 0.0,
        f"{prefix}_dark_border_ratio": 0.0,
        f"{prefix}_dark_horizontal_balance": 0.0,
        f"{prefix}_dark_vertical_balance": 0.0,
        f"{prefix}_aspect": 0.0,
        f"{prefix}_area": 0.0,
    }
    box_key = tuple(round(float(value), 3) for value in bbox) if bbox is not None else None
    cache_key = (str(id(raster)), prefix, box_key, float(pad))
    if cache_key in _VISUAL_FEATURE_CACHE:
        return _VISUAL_FEATURE_CACHE[cache_key]
    if raster is None or bbox is None:
        _VISUAL_FEATURE_CACHE[cache_key] = defaults
        return defaults
    image, edge = raster
    patch = crop_image(image, bbox, pad)
    edge_patch = crop_image(edge, bbox, pad)
    context_pad = max(8.0, max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 0.75)
    context = crop_image(image, bbox, context_pad)
    edge_context = crop_image(edge, bbox, context_pad)
    if patch is None or edge_patch is None or context is None or edge_context is None:
        _VISUAL_FEATURE_CACHE[cache_key] = defaults
        return defaults
    stat = ImageStat.Stat(patch)
    dark = histogram_fraction(patch, 0, 128)
    edge_density = histogram_fraction(edge_patch, 33, 256)
    context_dark = histogram_fraction(context, 0, 128)
    context_edge = histogram_fraction(edge_context, 33, 256)
    out = {
        f"{prefix}_mean": float(stat.mean[0]) / 255.0,
        f"{prefix}_std": float(stat.stddev[0]) / 255.0,
        f"{prefix}_dark_density": dark,
        f"{prefix}_very_dark_density": histogram_fraction(patch, 0, 64),
        f"{prefix}_mid_dark_density": histogram_fraction(patch, 64, 160),
        f"{prefix}_edge_density": edge_density,
        f"{prefix}_edge_strong_density": histogram_fraction(edge_patch, 96, 256),
        f"{prefix}_context_dark_density": context_dark,
        f"{prefix}_context_edge_density": context_edge,
        f"{prefix}_dark_ratio": dark / max(context_dark, 1e-6),
        f"{prefix}_edge_ratio": edge_density / max(context_edge, 1e-6),
    }
    for name, value in raster_layout_features(patch).items():
        out[f"{prefix}_{name}"] = value
    _VISUAL_FEATURE_CACHE[cache_key] = out
    return out


def union_bbox(left: list[float] | None, right: list[float] | None) -> list[float] | None:
    if left is None or right is None:
        return None
    return [min(left[0], right[0]), min(left[1], right[1]), max(left[2], right[2]), max(left[3], right[3])]


def contains_symbol_visual_features(rel: dict[str, Any], candidates: dict[str, dict[str, Any]]) -> dict[str, float]:
    if str(rel.get("relation")) != "contains_symbol":
        return {}
    source = candidates.get(str(rel.get("source_candidate_id")))
    target = candidates.get(str(rel.get("target_candidate_id")))
    source_bbox = candidate_bbox(source)
    target_bbox = candidate_bbox(target)
    raster = load_raster(candidate_raster_path(target) or candidate_raster_path(source))
    visual: dict[str, float] = {}
    visual.update(raster_patch_features(raster, target_bbox, prefix="visual_symbol", pad=2.0))
    visual.update(raster_patch_features(raster, source_bbox, prefix="visual_room", pad=2.0))
    visual.update(raster_patch_features(raster, union_bbox(source_bbox, target_bbox), prefix="visual_union", pad=4.0))
    if source_bbox is None or target_bbox is None:
        visual.update(
            {
                "visual_target_center_x_in_source": 0.0,
                "visual_target_center_y_in_source": 0.0,
                "visual_target_area_to_source": 0.0,
                "visual_target_inside_source": 0.0,
            }
        )
        return visual
    sw = max(source_bbox[2] - source_bbox[0], 1e-6)
    sh = max(source_bbox[3] - source_bbox[1], 1e-6)
    tw = max(target_bbox[2] - target_bbox[0], 1e-6)
    th = max(target_bbox[3] - target_bbox[1], 1e-6)
    tcx = (target_bbox[0] + target_bbox[2]) / 2.0
    tcy = (target_bbox[1] + target_bbox[3]) / 2.0
    ix1 = max(source_bbox[0], target_bbox[0])
    iy1 = max(source_bbox[1], target_bbox[1])
    ix2 = min(source_bbox[2], target_bbox[2])
    iy2 = min(source_bbox[3], target_bbox[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    visual.update(
        {
            "visual_target_center_x_in_source": (tcx - source_bbox[0]) / sw,
            "visual_target_center_y_in_source": (tcy - source_bbox[1]) / sh,
            "visual_target_area_to_source": (tw * th) / max(sw * sh, 1e-6),
            "visual_target_inside_source": inter / max(tw * th, 1e-6),
        }
    )
    return visual


def graph_node_key(rel: dict[str, Any], endpoint: str) -> str:
    cluster_id = str(rel.get(f"{endpoint}_cluster_id") or rel.get(f"{endpoint}_candidate_id") or "missing")
    return f"{endpoint}:{cluster_id}"


def connected_components(relations: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, int], dict[str, int], dict[str, int]]:
    graph: dict[str, set[str]] = defaultdict(set)
    degree: Counter[str] = Counter()
    for rel in relations:
        left = graph_node_key(rel, "source")
        right = graph_node_key(rel, "target")
        graph[left].add(right)
        graph[right].add(left)
        degree[left] += 1
        degree[right] += 1

    node_component: dict[str, str] = {}
    component_index = 0
    for node in sorted(graph):
        if node in node_component:
            continue
        component_id = f"relation_component_{component_index:05d}"
        component_index += 1
        queue: deque[str] = deque([node])
        node_component[node] = component_id
        while queue:
            current = queue.popleft()
            for nxt in graph[current]:
                if nxt in node_component:
                    continue
                node_component[nxt] = component_id
                queue.append(nxt)

    component_sizes: Counter[str] = Counter(node_component.values())
    component_edges: Counter[str] = Counter()
    for rel in relations:
        component_id = node_component.get(graph_node_key(rel, "source")) or node_component.get(graph_node_key(rel, "target")) or "relation_component_missing"
        component_edges[component_id] += 1
    return node_component, dict(component_sizes), dict(degree), dict(component_edges)


def relation_features(
    rel: dict[str, Any],
    candidates: dict[str, dict[str, Any]],
    duplicate_count: int,
    node_component: dict[str, str],
    component_sizes: dict[str, int],
    degree: dict[str, int],
    component_edges: dict[str, int],
) -> dict[str, float]:
    base = feature_vector(rel, candidates, duplicate_count=duplicate_count)
    src_node = graph_node_key(rel, "source")
    tgt_node = graph_node_key(rel, "target")
    component_id = node_component.get(src_node) or node_component.get(tgt_node) or "relation_component_missing"
    component_node_count = float(component_sizes.get(component_id, 0))
    source_degree = float(degree.get(src_node, 0))
    target_degree = float(degree.get(tgt_node, 0))
    bridge_source = 1.0 if source_degree > 1.0 else 0.0
    bridge_target = 1.0 if target_degree > 1.0 else 0.0
    bridge_count = float((1 if bridge_source else 0) + (1 if bridge_target else 0))
    features = {
        **base,
        "component_node_count": component_node_count,
        "component_edge_count": float(component_edges.get(component_id, 0)),
        "component_bridge_count": bridge_count,
        "source_graph_degree": source_degree,
        "target_graph_degree": target_degree,
        "source_bridge_flag": bridge_source,
        "target_bridge_flag": bridge_target,
        "duplicate_relation_count": float(duplicate_count),
        "pair_support_ratio": min(1.0, float(duplicate_count) / max(source_degree + target_degree, 1.0)),
        "bridge_source_degree_ratio": source_degree / max(component_node_count, 1.0),
        "bridge_target_degree_ratio": target_degree / max(component_node_count, 1.0),
    }
    features.update(contains_symbol_visual_features(rel, candidates))
    return features


def graph_role(rel: dict[str, Any], features: dict[str, float]) -> str:
    if not bool(rel.get("_label_positive")):
        return "negative"
    if features.get("component_bridge_count", 0.0) > 0.0 or features.get("duplicate_relation_count", 0.0) > 1.0:
        return "bridge"
    return "positive"


def build_dataset_rows(
    relation_pages: list[dict[str, Any]],
    adapter_by_id: dict[str, dict[str, Any]],
    smoke: bool = False,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gold = load_gold()
    rows: list[dict[str, Any]] = []
    page_audits: list[dict[str, Any]] = []
    missing_adapter_rows = 0
    for page in relation_pages:
        row_id = str(page.get("id"))
        adapter = adapter_by_id.get(row_id)
        if not adapter:
            missing_adapter_rows += 1
            continue
        candidates = row_candidate_map(adapter)
        cluster_ids, cluster_summaries, cluster_warnings = cluster_candidates(row_id, candidates)
        symbol_instance_ids, split_audit = build_symbol_instance_ids(row_id, candidates, cluster_ids, DEFAULT_SPLITTER_MODEL)
        pair_buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        relations = list(((page.get("scene_graph") or {}).get("relations") or []))
        for rel in relations:
            pair_buckets[topology_pair_key(rel, cluster_ids, symbol_instance_ids)].append(rel)
        node_component, component_sizes, degree, component_edges = connected_components(
            [
                {
                    **rel,
                    "source_cluster_id": cluster_ids.get(str(rel.get("source_candidate_id")), f"missing:{rel.get('source_candidate_id')}"),
                    "target_cluster_id": cluster_ids.get(str(rel.get("target_candidate_id")), f"missing:{rel.get('target_candidate_id')}"),
                }
                for rel in relations
            ]
        )

        page_relation_rows: list[dict[str, Any]] = []
        for rel in relations:
            source_cluster_id = cluster_ids.get(str(rel.get("source_candidate_id")), f"missing:{rel.get('source_candidate_id')}")
            target_cluster_id = cluster_ids.get(str(rel.get("target_candidate_id")), f"missing:{rel.get('target_candidate_id')}")
            enriched_rel = {
                **rel,
                "source_cluster_id": source_cluster_id,
                "target_cluster_id": target_cluster_id,
            }
            pair_key = topology_pair_key(rel, cluster_ids, symbol_instance_ids)
            duplicate_count = len(pair_buckets[pair_key])
            if duplicate_count <= 0:
                duplicate_count = 1
            labeled = relation_label({**rel, "row_id": row_id}, candidates, gold)
            left_gold, right_gold, ok = labeled
            rel = dict(rel)
            rel["_label_positive"] = bool(ok and left_gold and right_gold)
            feats = relation_features(
                enriched_rel,
                candidates,
                duplicate_count,
                node_component,
                component_sizes,
                degree,
                component_edges,
            )
            page_relation_rows.append(
                {
                    "row_id": row_id,
                    "relation_id": rel.get("relation_id"),
                    "relation": rel.get("relation"),
                    "source_candidate_id": rel.get("source_candidate_id"),
                    "target_candidate_id": rel.get("target_candidate_id"),
                    "source_cluster_id": source_cluster_id,
                    "target_cluster_id": target_cluster_id,
                    "symbol_instance_cluster_id": symbol_instance_ids.get(str(rel.get("target_candidate_id")), target_cluster_id),
                    "component_id": node_component.get(graph_node_key(enriched_rel, "source")) or node_component.get(graph_node_key(enriched_rel, "target")) or "relation_component_missing",
                    "duplicate_relation_count": duplicate_count,
                    "graph_role": graph_role(rel, feats),
                    "edge_features": feats,
                    "labels": {
                        "label_positive": bool(ok and left_gold and right_gold),
                        "graph_role": graph_role(rel, feats),
                        "gold_room_id": left_gold,
                        "gold_target_id": right_gold,
                        "gold_key": f"{row_id}|{left_gold}|{right_gold}" if ok and left_gold and right_gold else None,
                        "gold_loaded_after_inference_for_training_only": True,
                        "gold_used_for_inference": False,
                    },
                    "source_integrity": integrity(),
                }
            )
        rows.extend(page_relation_rows)
        page_audits.append(
            {
                "row_id": row_id,
                "relation_rows": len(page_relation_rows),
                "positive_rows": sum(1 for row in page_relation_rows if row["labels"]["label_positive"]),
                "cluster_count": len(cluster_summaries),
                "component_count": len(component_sizes),
                "cluster_warnings": cluster_warnings[:25],
                "symbol_instance_splitter": split_audit,
            }
        )
    positives = sum(1 for row in rows if row["labels"]["label_positive"])
    by_relation = Counter(row["relation"] for row in rows)
    by_role = Counter(row["graph_role"] for row in rows)
    audit = {
        "rows": len({row["row_id"] for row in rows}),
        "edge_rows": len(rows),
        "positive_edges": positives,
        "negative_edges": len(rows) - positives,
        "positive_rate": round(positives / max(len(rows), 1), 6),
        "by_relation": dict(by_relation),
        "by_graph_role": dict(by_role),
        "missing_adapter_rows": missing_adapter_rows,
        "page_audits": page_audits[:100],
        "source_integrity": integrity(),
        "gold_loaded_after_inference_for_training_only": True,
        "gold_used_for_inference": False,
        "smoke": bool(smoke),
        "limit": limit,
    }
    return rows, audit


def feature_names(rows: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for row in rows:
        feats = row.get("edge_features") if isinstance(row.get("edge_features"), dict) else {}
        names.update(str(key) for key in feats.keys())
    return sorted(names)


def train_logistic(rows: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    x = np.asarray([vectorize(row.get("edge_features") or {}, names) for row in rows], dtype=np.float32)
    y = np.asarray([1.0 if row.get("labels", {}).get("label_positive") else 0.0 for row in rows], dtype=np.float32)
    if len(rows) == 0:
        return {
            "features": names,
            "mean": [0.0] * len(names),
            "std": [1.0] * len(names),
            "weights": [0.0] * len(names),
            "bias": 0.0,
            "train_counts": {"rows": 0, "positive": 0, "negative": 0},
        }
    mean = x.mean(axis=0)
    std = x.std(axis=0) + 1e-6
    z = (x - mean) / std
    pos_n = max(float(y.sum()), 1.0)
    neg_n = max(float(len(y) - y.sum()), 1.0)
    sample_weight = np.where(y > 0.5, len(y) / (2.0 * pos_n), len(y) / (2.0 * neg_n)).astype(np.float32)
    weights = np.zeros((len(names),), dtype=np.float32)
    bias = 0.0
    lr = 0.06
    reg = 0.002
    for _ in range(240):
        logits = np.clip(z @ weights + bias, -30.0, 30.0)
        pred = 1.0 / (1.0 + np.exp(-logits))
        err = (pred - y) * sample_weight
        grad_w = (z.T @ err) / max(len(y), 1) + reg * weights
        grad_b = float(err.mean())
        weights -= lr * grad_w
        bias -= lr * grad_b
    return {
        "features": names,
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "weights": weights.astype(float).tolist(),
        "bias": float(bias),
        "train_counts": {"rows": int(len(rows)), "positive": int(y.sum()), "negative": int(len(y) - y.sum())},
    }


def score_row(row: dict[str, Any], model: dict[str, Any]) -> float:
    names = list(model.get("features") or [])
    x = vectorize(row.get("edge_features") or {}, names)
    mean = np.asarray(model.get("mean") or [0.0] * len(names), dtype=np.float32)
    std = np.asarray(model.get("std") or [1.0] * len(names), dtype=np.float32)
    weights = np.asarray(model.get("weights") or [0.0] * len(names), dtype=np.float32)
    raw = float(((x - mean) / np.maximum(std, 1e-9)) @ weights + safe_float(model.get("bias")))
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, raw))))


def relation_model(model: dict[str, Any], relation: str) -> dict[str, Any]:
    """Return the per-relation scorer, falling back to legacy flat checkpoints."""
    relations = model.get("relations") if isinstance(model.get("relations"), dict) else {}
    rel_model = relations.get(relation)
    if isinstance(rel_model, dict):
        return rel_model
    return model


def listwise_model(model: dict[str, Any], relation: str) -> dict[str, Any] | None:
    listwise = model.get("listwise") if isinstance(model.get("listwise"), dict) else {}
    relations = listwise.get("relations") if isinstance(listwise.get("relations"), dict) else {}
    rel_model = relations.get(relation)
    return rel_model if isinstance(rel_model, dict) else None


def listwise_features(row: dict[str, Any], bucket_rows: list[dict[str, Any]]) -> dict[str, float]:
    source_id = str(row.get("source_cluster_id") or row.get("source_candidate_id") or "")
    target_id = str(row.get("target_cluster_id") or row.get("target_candidate_id") or "")
    component_id = str(row.get("component_id") or "relation_component_missing")
    score = safe_float(row.get("relation_score"))
    scores = sorted((safe_float(item.get("relation_score")) for item in bucket_rows), reverse=True)
    source_scores = sorted(
        (safe_float(item.get("relation_score")) for item in bucket_rows if str(item.get("source_cluster_id") or item.get("source_candidate_id") or "") == source_id),
        reverse=True,
    )
    target_scores = sorted(
        (safe_float(item.get("relation_score")) for item in bucket_rows if str(item.get("target_cluster_id") or item.get("target_candidate_id") or "") == target_id),
        reverse=True,
    )
    component_scores = sorted(
        (safe_float(item.get("relation_score")) for item in bucket_rows if str(item.get("component_id") or "relation_component_missing") == component_id),
        reverse=True,
    )
    pair_count = sum(
        1
        for item in bucket_rows
        if str(item.get("source_cluster_id") or item.get("source_candidate_id") or "") == source_id
        and str(item.get("target_cluster_id") or item.get("target_candidate_id") or "") == target_id
    )
    def rank(values: list[float]) -> int:
        return 1 + sum(1 for value in values if value > score + 1e-12)

    component_size = max(len(component_scores), 1)
    source_size = max(len(source_scores), 1)
    target_size = max(len(target_scores), 1)
    global_size = max(len(scores), 1)
    edge_features = row.get("edge_features") if isinstance(row.get("edge_features"), dict) else {}
    source_degree = safe_float(edge_features.get("source_graph_degree"))
    target_degree = safe_float(edge_features.get("target_graph_degree"))
    component_edges = safe_float(edge_features.get("component_edge_count"))
    feats = {
        "base_relation_score": score,
        "global_score_rank": float(rank(scores)),
        "source_score_rank": float(rank(source_scores)),
        "target_score_rank": float(rank(target_scores)),
        "component_score_rank": float(rank(component_scores)),
        "global_score_percentile": rank(scores) / global_size,
        "source_score_percentile": rank(source_scores) / source_size,
        "target_score_percentile": rank(target_scores) / target_size,
        "component_score_percentile": rank(component_scores) / component_size,
        "source_cluster_edge_count": float(source_size),
        "target_cluster_edge_count": float(target_size),
        "component_relation_edge_count": float(component_size),
        "cluster_pair_edge_count": float(pair_count),
        "duplicate_relation_count": safe_float(row.get("duplicate_relation_count")),
        "component_edge_count": component_edges,
        "component_density_ratio": component_edges / max(safe_float(edge_features.get("component_node_count")), 1.0),
        "bridge_degree_sum": source_degree + target_degree,
        "bridge_degree_max": max(source_degree, target_degree),
        "component_bridge_count": safe_float(edge_features.get("component_bridge_count")),
        "pair_support_ratio": safe_float(edge_features.get("pair_support_ratio")),
        "relation_confidence": safe_float(edge_features.get("relation_confidence")),
        "detector_confidence_product": safe_float(edge_features.get("detector_confidence_product")),
    }
    if str(row.get("relation")) == "bounded_by":
        feats.update(bounded_by_geometry_features(edge_features))
    return feats


def bounded_by_geometry_features(edge_features: dict[str, Any]) -> dict[str, float]:
    """Geometry cues for room-wall representative selection.

    These are inference-available detector geometry features, not labels. They
    matter most for bounded_by because duplicate wall support edges often share
    the same clustered source/target pair but differ in raw wall-side alignment.
    """
    source_area = safe_float(edge_features.get("source_bbox_area"))
    source_width = safe_float(edge_features.get("source_bbox_width"))
    source_height = safe_float(edge_features.get("source_bbox_height"))
    target_area = safe_float(edge_features.get("target_bbox_area"))
    target_width = safe_float(edge_features.get("target_bbox_width"))
    target_height = safe_float(edge_features.get("target_bbox_height"))
    long_side = max(target_width, target_height)
    short_side = min(target_width, target_height)
    thinness = 0.0 if long_side <= 0.0 else 1.0 - min(1.0, short_side / max(long_side, 1e-6))
    side_overlap = safe_float(edge_features.get("side_overlap_ratio"))
    axis_overlap = safe_float(edge_features.get("axis_overlap_ratio"))
    bbox_distance = safe_float(edge_features.get("bbox_distance"))
    gap = safe_float(edge_features.get("gap"))
    orientation = safe_float(edge_features.get("orientation_compatible"))
    edge_margin_ratio = safe_float(edge_features.get("target_room_min_edge_distance_ratio"))
    edge_proximity = 1.0 - max(0.0, min(1.0, edge_margin_ratio))
    edge_margin_direct = safe_float(edge_features.get("target_room_edge_margin_ratio"))
    edge_direct_proximity = 1.0 - max(0.0, min(1.0, edge_margin_direct))
    center_x_offset = abs(safe_float(edge_features.get("target_room_center_x_offset")))
    center_y_offset = abs(safe_float(edge_features.get("target_room_center_y_offset")))
    side_axis_offset = min(1.0, max(center_x_offset, center_y_offset) * 2.0)
    parallel_axis_centering = 1.0 - min(1.0, min(center_x_offset, center_y_offset) * 2.0)
    distance_score = 1.0 / (1.0 + max(0.0, bbox_distance + gap))
    target_area_ratio = 0.0 if source_area <= 0.0 else min(1.0, target_area / max(source_area, 1e-6))
    source_aspect_balance = 1.0 - min(1.0, abs(source_width - source_height) / max(source_width, source_height, 1.0))
    wall_side_alignment_score = (
        0.22 * side_overlap
        + 0.18 * orientation
        + 0.16 * thinness
        + 0.16 * edge_proximity
        + 0.10 * edge_direct_proximity
        + 0.10 * distance_score
        + 0.08 * parallel_axis_centering
    )
    return {
        "bounded_wall_thinness_score": thinness,
        "bounded_side_overlap_ratio": side_overlap,
        "bounded_axis_overlap_ratio": axis_overlap,
        "bounded_orientation_compatible": orientation,
        "bounded_edge_proximity_score": edge_proximity,
        "bounded_edge_direct_proximity_score": edge_direct_proximity,
        "bounded_side_axis_offset_score": side_axis_offset,
        "bounded_parallel_axis_centering_score": parallel_axis_centering,
        "bounded_distance_score": distance_score,
        "bounded_target_area_ratio": target_area_ratio,
        "bounded_source_aspect_balance": source_aspect_balance,
        "bounded_wall_side_alignment_score": wall_side_alignment_score,
    }


def bounded_by_room_side(row: dict[str, Any]) -> str:
    edge_features = row.get("edge_features") if isinstance(row.get("edge_features"), dict) else {}
    x_offset = safe_float(edge_features.get("target_room_center_x_offset"))
    y_offset = safe_float(edge_features.get("target_room_center_y_offset"))
    if abs(x_offset) >= abs(y_offset):
        return "east" if x_offset >= 0.0 else "west"
    return "south" if y_offset >= 0.0 else "north"


def bbox_from_candidate_id(candidate_id: Any) -> list[float] | None:
    parts = str(candidate_id or "").split("_")
    if len(parts) < 4:
        return None
    try:
        bbox = [float(value) for value in parts[-4:]]
    except ValueError:
        return None
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    return bbox


def bounded_by_wall_segment_key(row: dict[str, Any], bin_size: int = 16) -> str:
    """Stable wall-support bucket for bounded_by duplicate collapse.

    The key is inference-available: it is derived from detector candidate ids
    that already carry raster proposal bboxes. It groups support edges along a
    room side by the wall segment's parallel-axis center, so duplicated raw wall
    fragments can be capped without imposing a page-level hard NMS.
    """
    side = bounded_by_room_side(row)
    bbox = bbox_from_candidate_id(row.get("target_candidate_id"))
    if not bbox:
        return f"{side}:missing_bbox"
    center_x = (bbox[0] + bbox[2]) / 2.0
    center_y = (bbox[1] + bbox[3]) / 2.0
    bin_width = max(1, int(bin_size or 16))
    projected = center_y if side in {"east", "west"} else center_x
    segment_bin = int(projected // bin_width)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    orientation = "vertical" if height >= width else "horizontal"
    return f"{side}:{orientation}:{segment_bin}"


def score_listwise_row(row: dict[str, Any], rel_model: dict[str, Any]) -> float:
    names = list(rel_model.get("features") or [])
    x = vectorize(row.get("listwise_features") or {}, names)
    mean = np.asarray(rel_model.get("mean") or [0.0] * len(names), dtype=np.float32)
    std = np.asarray(rel_model.get("std") or [1.0] * len(names), dtype=np.float32)
    weights = np.asarray(rel_model.get("weights") or [0.0] * len(names), dtype=np.float32)
    raw = float(((x - mean) / np.maximum(std, 1e-9)) @ weights + safe_float(rel_model.get("bias")))
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, raw))))


def contains_symbol_support_features(row: dict[str, Any]) -> dict[str, float]:
    listwise = row.get("listwise_features") if isinstance(row.get("listwise_features"), dict) else {}
    edge = row.get("edge_features") if isinstance(row.get("edge_features"), dict) else {}
    feats: dict[str, float] = {
        "relation_score": safe_float(row.get("relation_score")),
        "base_relation_score": safe_float(row.get("base_relation_score")),
        "confidence": safe_float(row.get("confidence")),
        "contains_support_set_size": safe_float(listwise.get("contains_support_set_size")),
        "contains_support_rank": safe_float(listwise.get("contains_support_rank")),
        "contains_support_tail_rank": safe_float(listwise.get("contains_support_tail_rank")),
        "contains_support_rank_percentile": safe_float(listwise.get("contains_support_rank_percentile")),
        "contains_support_tail_percentile": safe_float(listwise.get("contains_support_tail_percentile")),
    }
    for name in [
        "global_score_rank",
        "source_score_rank",
        "target_score_rank",
        "component_score_rank",
        "cluster_pair_score_rank",
        "cluster_pair_tail_rank",
        "global_score_percentile",
        "source_score_percentile",
        "target_score_percentile",
        "component_score_percentile",
        "cluster_pair_score_percentile",
        "cluster_pair_tail_percentile",
        "source_cluster_edge_count",
        "target_cluster_edge_count",
        "component_relation_edge_count",
        "cluster_pair_edge_count",
        "duplicate_relation_count",
        "component_edge_count",
        "component_density_ratio",
        "bridge_degree_sum",
        "bridge_degree_max",
        "component_bridge_count",
        "pair_support_ratio",
        "relation_confidence",
        "detector_confidence_product",
    ]:
        feats[f"listwise_{name}"] = safe_float(listwise.get(name))
    for name in [
        "bbox_distance",
        "center_distance",
        "source_bbox_area",
        "target_bbox_area",
        "bbox_iou",
        "target_inside_source_ratio",
        "source_inside_target_ratio",
        "side_overlap_ratio",
        "axis_overlap_ratio",
        "source_graph_degree",
        "target_graph_degree",
        "component_node_count",
        "component_edge_count",
        "component_bridge_count",
        "duplicate_relation_count",
        "pair_support_ratio",
        "detector_confidence_product",
    ]:
        feats[f"edge_{name}"] = safe_float(edge.get(name))
    for name, value in edge.items():
        if str(name).startswith("visual_"):
            feats[str(name)] = safe_float(value)
    return feats


def score_contains_symbol_support_row(row: dict[str, Any], model: dict[str, Any]) -> float:
    names = list(model.get("features") or [])
    x = np.asarray([transform_value(name, safe_float(contains_symbol_support_features(row).get(name))) for name in names], dtype=np.float32)
    mean = np.asarray(model.get("mean") or [0.0] * len(names), dtype=np.float32)
    std = np.asarray(model.get("std") or [1.0] * len(names), dtype=np.float32)
    weights = np.asarray(model.get("weights") or [0.0] * len(names), dtype=np.float32)
    raw = float(((x - mean) / np.maximum(std, 1e-9)) @ weights + safe_float(model.get("bias")))
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, raw))))


def contains_symbol_assignment_features(row: dict[str, Any]) -> dict[str, float]:
    feats = contains_symbol_support_features(row)
    if row.get("support_criticality_score") is not None:
        feats["prior_support_criticality_score"] = safe_float(row.get("support_criticality_score"))
    return feats


def score_contains_symbol_assignment_row(row: dict[str, Any], model: dict[str, Any]) -> float:
    names = list(model.get("features") or [])
    feats = contains_symbol_assignment_features(row)
    x = np.asarray([transform_value(name, safe_float(feats.get(name))) for name in names], dtype=np.float32)
    mean = np.asarray(model.get("mean") or [0.0] * len(names), dtype=np.float32)
    std = np.asarray(model.get("std") or [1.0] * len(names), dtype=np.float32)
    weights = np.asarray(model.get("weights") or [0.0] * len(names), dtype=np.float32)
    raw = float(((x - mean) / np.maximum(std, 1e-9)) @ weights + safe_float(model.get("bias")))
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, raw))))


def apply_listwise_scores(rows: list[dict[str, Any]], model: dict[str, Any]) -> list[dict[str, Any]]:
    by_page_relation: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_page_relation[(str(row.get("row_id")), str(row.get("relation")))].append(row)
    out: list[dict[str, Any]] = []
    for (_, relation), bucket in by_page_relation.items():
        rel_model = listwise_model(model, relation)
        global_scores = sorted(safe_float(item.get("relation_score")) for item in bucket)
        source_scores: dict[str, list[float]] = defaultdict(list)
        target_scores: dict[str, list[float]] = defaultdict(list)
        component_scores: dict[str, list[float]] = defaultdict(list)
        pair_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
        pair_alignment_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
        wall_segment_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
        contains_support_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
        pair_counts: Counter[tuple[str, str]] = Counter()
        for item in bucket:
            source_id = str(item.get("source_cluster_id") or item.get("source_candidate_id") or "")
            target_id = str(item.get("target_cluster_id") or item.get("target_candidate_id") or "")
            symbol_id = str(item.get("symbol_instance_cluster_id") or target_id)
            component_id = str(item.get("component_id") or "relation_component_missing")
            score = safe_float(item.get("relation_score"))
            edge_features = item.get("edge_features") if isinstance(item.get("edge_features"), dict) else {}
            alignment_score = 0.0
            if relation == "bounded_by":
                alignment_score = bounded_by_geometry_features(edge_features).get("bounded_wall_side_alignment_score", 0.0)
                wall_segment_scores[(source_id, bounded_by_wall_segment_key(item))].append(score)
            if relation == "contains_symbol":
                contains_support_scores[(source_id, symbol_id)].append(score)
            source_scores[source_id].append(score)
            target_scores[target_id].append(score)
            component_scores[component_id].append(score)
            pair_counts[(source_id, target_id)] += 1
            pair_scores[(source_id, target_id)].append(score)
            pair_alignment_scores[(source_id, target_id)].append(alignment_score)
        for values in source_scores.values():
            values.sort()
        for values in target_scores.values():
            values.sort()
        for values in component_scores.values():
            values.sort()
        for values in pair_scores.values():
            values.sort()
        for values in pair_alignment_scores.values():
            values.sort()
        for values in wall_segment_scores.values():
            values.sort()
        for values in contains_support_scores.values():
            values.sort()

        def rank(values: list[float], score: float) -> int:
            return 1 + len(values) - bisect_right(values, score + 1e-12)

        for row in bucket:
            item = dict(row)
            source_id = str(item.get("source_cluster_id") or item.get("source_candidate_id") or "")
            target_id = str(item.get("target_cluster_id") or item.get("target_candidate_id") or "")
            symbol_id = str(item.get("symbol_instance_cluster_id") or target_id)
            component_id = str(item.get("component_id") or "relation_component_missing")
            score = safe_float(item.get("relation_score"))
            src_scores = source_scores.get(source_id, [])
            tgt_scores = target_scores.get(target_id, [])
            comp_scores = component_scores.get(component_id, [])
            pr_scores = pair_scores.get((source_id, target_id), [])
            pr_alignment_scores = pair_alignment_scores.get((source_id, target_id), [])
            wall_segment_key = bounded_by_wall_segment_key(item) if relation == "bounded_by" else ""
            wall_scores = wall_segment_scores.get((source_id, wall_segment_key), [])
            support_scores = contains_support_scores.get((source_id, symbol_id), [])
            edge_features = item.get("edge_features") if isinstance(item.get("edge_features"), dict) else {}
            source_degree = safe_float(edge_features.get("source_graph_degree"))
            target_degree = safe_float(edge_features.get("target_graph_degree"))
            component_edges = safe_float(edge_features.get("component_edge_count"))
            geom = bounded_by_geometry_features(edge_features) if relation == "bounded_by" else {}
            alignment_score = safe_float(geom.get("bounded_wall_side_alignment_score"))
            feats = {
                "base_relation_score": score,
                "global_score_rank": float(rank(global_scores, score)),
                "source_score_rank": float(rank(src_scores, score)),
                "target_score_rank": float(rank(tgt_scores, score)),
                "component_score_rank": float(rank(comp_scores, score)),
                "cluster_pair_score_rank": float(rank(pr_scores, score)),
                "cluster_pair_tail_rank": float(bisect_right(pr_scores, score + 1e-12)),
                "cluster_pair_alignment_rank": float(rank(pr_alignment_scores, alignment_score)),
                "cluster_pair_alignment_tail_rank": float(bisect_right(pr_alignment_scores, alignment_score + 1e-12)),
                "bounded_wall_segment_score_rank": float(rank(wall_scores, score)) if relation == "bounded_by" else 0.0,
                "bounded_wall_segment_tail_rank": float(bisect_right(wall_scores, score + 1e-12)) if relation == "bounded_by" else 0.0,
                "contains_support_rank": float(rank(support_scores, score)) if relation == "contains_symbol" else 0.0,
                "contains_support_tail_rank": float(bisect_right(support_scores, score + 1e-12)) if relation == "contains_symbol" else 0.0,
                "global_score_percentile": rank(global_scores, score) / max(len(global_scores), 1),
                "source_score_percentile": rank(src_scores, score) / max(len(src_scores), 1),
                "target_score_percentile": rank(tgt_scores, score) / max(len(tgt_scores), 1),
                "component_score_percentile": rank(comp_scores, score) / max(len(comp_scores), 1),
                "cluster_pair_score_percentile": rank(pr_scores, score) / max(len(pr_scores), 1),
                "cluster_pair_tail_percentile": bisect_right(pr_scores, score + 1e-12) / max(len(pr_scores), 1),
                "cluster_pair_alignment_percentile": rank(pr_alignment_scores, alignment_score) / max(len(pr_alignment_scores), 1),
                "cluster_pair_alignment_tail_percentile": bisect_right(pr_alignment_scores, alignment_score + 1e-12) / max(len(pr_alignment_scores), 1),
                "bounded_wall_segment_score_percentile": rank(wall_scores, score) / max(len(wall_scores), 1) if relation == "bounded_by" else 0.0,
                "bounded_wall_segment_tail_percentile": bisect_right(wall_scores, score + 1e-12) / max(len(wall_scores), 1) if relation == "bounded_by" else 0.0,
                "contains_support_rank_percentile": rank(support_scores, score) / max(len(support_scores), 1) if relation == "contains_symbol" else 0.0,
                "contains_support_tail_percentile": bisect_right(support_scores, score + 1e-12) / max(len(support_scores), 1) if relation == "contains_symbol" else 0.0,
                "source_cluster_edge_count": float(len(src_scores)),
                "target_cluster_edge_count": float(len(tgt_scores)),
                "component_relation_edge_count": float(len(comp_scores)),
                "cluster_pair_edge_count": float(pair_counts[(source_id, target_id)]),
                "bounded_wall_segment_edge_count": float(len(wall_scores)) if relation == "bounded_by" else 0.0,
                "contains_support_set_size": float(len(support_scores)) if relation == "contains_symbol" else 0.0,
                "duplicate_relation_count": safe_float(item.get("duplicate_relation_count")),
                "component_edge_count": component_edges,
                "component_density_ratio": component_edges / max(safe_float(edge_features.get("component_node_count")), 1.0),
                "bridge_degree_sum": source_degree + target_degree,
                "bridge_degree_max": max(source_degree, target_degree),
                "component_bridge_count": safe_float(edge_features.get("component_bridge_count")),
                "pair_support_ratio": safe_float(edge_features.get("pair_support_ratio")),
                "relation_confidence": safe_float(edge_features.get("relation_confidence")),
                "detector_confidence_product": safe_float(edge_features.get("detector_confidence_product")),
            }
            feats.update(geom)
            if relation == "bounded_by":
                side = bounded_by_room_side(item)
                feats.update(
                    {
                        "bounded_room_side_east": 1.0 if side == "east" else 0.0,
                        "bounded_room_side_west": 1.0 if side == "west" else 0.0,
                        "bounded_room_side_north": 1.0 if side == "north" else 0.0,
                        "bounded_room_side_south": 1.0 if side == "south" else 0.0,
                    }
                )
            item["listwise_features"] = feats
            if rel_model:
                item["listwise_score"] = score_listwise_row(item, rel_model)
                item["relation_score"] = item["listwise_score"]
                item["base_relation_score"] = row.get("relation_score")
            support_model = model.get("contains_symbol_support_criticality") if isinstance(model.get("contains_symbol_support_criticality"), dict) else None
            if relation == "contains_symbol" and support_model:
                scorer = support_model.get("scorer") if isinstance(support_model.get("scorer"), dict) else support_model
                item["support_criticality_score"] = score_contains_symbol_support_row(item, scorer)
            assignment_model = model.get("contains_symbol_assignment") if isinstance(model.get("contains_symbol_assignment"), dict) else None
            if relation == "contains_symbol" and assignment_model:
                scorer = assignment_model.get("scorer") if isinstance(assignment_model.get("scorer"), dict) else assignment_model
                item["assignment_score"] = score_contains_symbol_assignment_row(item, scorer)
            out.append(item)
    return out


def score_rows(rows: list[dict[str, Any]], model: dict[str, Any]) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        rel_type = str(item.get("relation"))
        item["relation_score"] = score_row(item, relation_model(model, rel_type))
        scored.append(item)
    return apply_listwise_scores(scored, model)


def threshold_sweep(scored: list[dict[str, Any]], thresholds: list[float]) -> list[dict[str, Any]]:
    positives = sum(1 for row in scored if row.get("labels", {}).get("label_positive"))
    out: list[dict[str, Any]] = []
    for threshold in thresholds:
        selected = [row for row in scored if safe_float(row.get("relation_score")) >= threshold]
        tp = sum(1 for row in selected if row.get("labels", {}).get("label_positive"))
        precision = tp / max(len(selected), 1)
        recall = tp / max(positives, 1)
        f1 = 0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall)
        out.append(
            {
                "threshold": round(threshold, 6),
                "selected": len(selected),
                "true_positive": tp,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
            }
        )
    return out


def default_policy() -> dict[str, Any]:
    return {
        "bounded_by": {"threshold": 0.45, "max_per_source": 4, "max_per_target": 4, "max_per_component": 6},
        "contains_symbol": {"threshold": 0.52, "max_per_source": 2, "max_per_target": 1, "max_per_component": 4},
        "labeled_by_text": {"threshold": 0.55, "max_per_source": 2, "max_per_target": 1, "max_per_component": 3},
        "adjacent_to": {"threshold": 0.50, "max_per_source": 3, "max_per_target": 3, "max_per_component": 8},
    }


def select_rows(rows: list[dict[str, Any]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    by_relation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        relation = str(row.get("relation"))
        threshold = safe_float((policy.get(relation) or {}).get("threshold"))
        if safe_float(row.get("relation_score")) >= threshold:
            by_relation[relation].append(row)

    for relation, bucket in by_relation.items():
        params = policy.get(relation) or {}
        if relation == "contains_symbol" and params.get("assignment_policy"):
            selected.extend(select_contains_symbol_assignment_rows(bucket, params))
            continue
        max_per_source = int(params.get("max_per_source") or 999999)
        max_per_target = int(params.get("max_per_target") or 999999)
        max_per_component = int(params.get("max_per_component") or 999999)
        max_per_pair = int(params.get("max_per_pair") or 999999)
        max_pair_tail_slots = int(params.get("max_pair_tail_slots") or 0)
        max_pair_alignment_slots = int(params.get("max_pair_alignment_slots") or 0)
        max_per_source_side = int(params.get("max_per_source_side") or 999999)
        max_per_source_side_segment = int(params.get("max_per_source_side_segment") or 999999)
        max_wall_segment_tail_slots = int(params.get("max_wall_segment_tail_slots") or 0)
        max_wall_segment_support_overflow_slots = int(params.get("max_wall_segment_support_overflow_slots") or 0)
        wall_segment_support_score_rank_max = int(params.get("wall_segment_support_score_rank_max") or 0)
        wall_segment_support_tail_rank_max = int(params.get("wall_segment_support_tail_rank_max") or 0)
        wall_segment_bin_size = int(params.get("wall_segment_bin_size") or 16)
        max_per_room_symbol_instance = int(params.get("max_per_room_symbol_instance") or 999999)
        max_contains_symbol_support_tail_slots = int(params.get("max_contains_symbol_support_tail_slots") or 0)
        max_contains_symbol_support_overflow_slots = int(params.get("max_contains_symbol_support_overflow_slots") or 0)
        contains_symbol_support_criticality_threshold = safe_float(params.get("contains_symbol_support_criticality_threshold"))
        source_counts: Counter[str] = Counter()
        target_counts: Counter[str] = Counter()
        component_counts: Counter[str] = Counter()
        pair_counts: Counter[tuple[str, str]] = Counter()
        pair_tail_counts: Counter[tuple[str, str]] = Counter()
        pair_alignment_counts: Counter[tuple[str, str]] = Counter()
        source_side_counts: Counter[tuple[str, str]] = Counter()
        source_side_segment_counts: Counter[tuple[str, str, str]] = Counter()
        source_side_segment_tail_counts: Counter[tuple[str, str, str]] = Counter()
        source_side_segment_support_counts: Counter[tuple[str, str, str]] = Counter()
        room_symbol_counts: Counter[tuple[str, str]] = Counter()
        room_symbol_tail_counts: Counter[tuple[str, str]] = Counter()
        room_symbol_support_overflow_counts: Counter[tuple[str, str]] = Counter()
        ordered = sorted(bucket, key=lambda item: (safe_float(item.get("relation_score")), safe_float(item.get("confidence"))), reverse=True)
        room_symbol_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
        if relation == "contains_symbol" and (max_per_room_symbol_instance < 999999 or max_contains_symbol_support_tail_slots > 0):
            for item in bucket:
                source_id = str(item.get("source_cluster_id") or item.get("source_candidate_id"))
                symbol_id = str(item.get("symbol_instance_cluster_id") or item.get("target_cluster_id") or item.get("target_candidate_id"))
                room_symbol_scores[(source_id, symbol_id)].append(safe_float(item.get("relation_score")))
            for values in room_symbol_scores.values():
                values.sort()

        def support_overflow_candidate(row: dict[str, Any]) -> bool:
            if relation != "bounded_by" or max_wall_segment_support_overflow_slots <= 0:
                return False
            listwise = row.get("listwise_features") if isinstance(row.get("listwise_features"), dict) else {}
            segment_score_rank = safe_float(listwise.get("bounded_wall_segment_score_rank"))
            segment_tail_rank = safe_float(listwise.get("bounded_wall_segment_tail_rank"))
            return (
                wall_segment_support_score_rank_max > 0
                and wall_segment_support_tail_rank_max > 0
                and 0 < segment_score_rank <= wall_segment_support_score_rank_max
                and 0 < segment_tail_rank <= wall_segment_support_tail_rank_max
            )

        selected_ids: set[str] = set()

        def contains_symbol_support_overflow_candidate(row: dict[str, Any]) -> bool:
            return (
                relation == "contains_symbol"
                and max_contains_symbol_support_overflow_slots > 0
                and safe_float(row.get("support_criticality_score")) >= contains_symbol_support_criticality_threshold
            )

        def try_select(row: dict[str, Any], *, allow_support_overflow: bool = False, allow_contains_support_overflow: bool = False) -> bool:
            source_id = str(row.get("source_cluster_id") or row.get("source_candidate_id"))
            target_id = str(row.get("target_cluster_id") or row.get("target_candidate_id"))
            component_id = str(row.get("component_id") or "relation_component_missing")
            relation_id = str(row.get("relation_id") or "")
            if relation_id in selected_ids:
                return False
            pair_id = (source_id, target_id)
            symbol_id = str(row.get("symbol_instance_cluster_id") or row.get("target_cluster_id") or row.get("target_candidate_id"))
            room_symbol_id = (source_id, symbol_id)
            side = bounded_by_room_side(row) if relation == "bounded_by" else relation
            segment_key = bounded_by_wall_segment_key(row, wall_segment_bin_size) if relation == "bounded_by" else relation
            source_side_id = (source_id, side)
            source_side_segment_id = (source_id, side, segment_key)
            listwise = row.get("listwise_features") if isinstance(row.get("listwise_features"), dict) else {}
            pair_tail_rank = safe_float(listwise.get("cluster_pair_tail_rank"))
            pair_alignment_rank = safe_float(listwise.get("cluster_pair_alignment_rank"))
            wall_segment_tail_rank = safe_float(listwise.get("bounded_wall_segment_tail_rank"))
            use_tail_slot = max_pair_tail_slots > 0 and pair_tail_rank > 0 and pair_tail_rank <= max_pair_tail_slots
            use_alignment_slot = max_pair_alignment_slots > 0 and pair_alignment_rank > 0 and pair_alignment_rank <= max_pair_alignment_slots
            use_wall_segment_tail_slot = (
                relation == "bounded_by"
                and max_wall_segment_tail_slots > 0
                and wall_segment_tail_rank > 0
                and wall_segment_tail_rank <= max_wall_segment_tail_slots
            )
            if source_counts[source_id] >= max_per_source:
                return False
            if target_counts[target_id] >= max_per_target:
                return False
            if component_counts[component_id] >= max_per_component:
                return False
            if source_side_counts[source_side_id] >= max_per_source_side:
                return False
            can_use_wall_segment_tail = (
                use_wall_segment_tail_slot
                and source_side_segment_tail_counts[source_side_segment_id] < max_wall_segment_tail_slots
            )
            can_use_support_overflow = (
                allow_support_overflow
                and support_overflow_candidate(row)
                and source_side_segment_support_counts[source_side_segment_id] < max_wall_segment_support_overflow_slots
            )
            if source_side_segment_counts[source_side_segment_id] >= max_per_source_side_segment and not (can_use_wall_segment_tail or can_use_support_overflow):
                return False
            if relation == "contains_symbol" and room_symbol_counts[room_symbol_id] >= max_per_room_symbol_instance:
                scores = room_symbol_scores.get(room_symbol_id, [])
                support_tail_rank = bisect_right(scores, safe_float(row.get("relation_score")) + 1e-12)
                can_use_support_tail = (
                    max_contains_symbol_support_tail_slots > 0
                    and 0 < support_tail_rank <= max_contains_symbol_support_tail_slots
                    and room_symbol_tail_counts[room_symbol_id] < max_contains_symbol_support_tail_slots
                )
                can_use_contains_support_overflow = (
                    allow_contains_support_overflow
                    and contains_symbol_support_overflow_candidate(row)
                    and room_symbol_support_overflow_counts[room_symbol_id] < max_contains_symbol_support_overflow_slots
                )
                if not (can_use_support_tail or can_use_contains_support_overflow):
                    return False
            can_use_tail = use_tail_slot and pair_tail_counts[pair_id] < max_pair_tail_slots
            can_use_alignment = use_alignment_slot and pair_alignment_counts[pair_id] < max_pair_alignment_slots
            if pair_counts[pair_id] >= max_per_pair and not (can_use_tail or can_use_alignment):
                return False
            selected.append(row)
            selected_ids.add(relation_id)
            source_counts[source_id] += 1
            target_counts[target_id] += 1
            component_counts[component_id] += 1
            source_side_counts[source_side_id] += 1
            source_side_segment_counts[source_side_segment_id] += 1
            if use_wall_segment_tail_slot:
                source_side_segment_tail_counts[source_side_segment_id] += 1
            if can_use_support_overflow:
                source_side_segment_support_counts[source_side_segment_id] += 1
            if relation == "contains_symbol":
                scores = room_symbol_scores.get(room_symbol_id, [])
                support_tail_rank = bisect_right(scores, safe_float(row.get("relation_score")) + 1e-12)
                if max_contains_symbol_support_tail_slots > 0 and 0 < support_tail_rank <= max_contains_symbol_support_tail_slots:
                    room_symbol_tail_counts[room_symbol_id] += 1
                if allow_contains_support_overflow and contains_symbol_support_overflow_candidate(row):
                    room_symbol_support_overflow_counts[room_symbol_id] += 1
                room_symbol_counts[room_symbol_id] += 1
            pair_counts[pair_id] += 1
            if use_tail_slot:
                pair_tail_counts[pair_id] += 1
            if use_alignment_slot:
                pair_alignment_counts[pair_id] += 1
            return True

        for row in ordered:
            try_select(row)

        if relation == "bounded_by" and max_wall_segment_support_overflow_slots > 0:
            support_ordered = [row for row in bucket if support_overflow_candidate(row)]
            support_ordered.sort(
                key=lambda item: (
                    safe_float((item.get("listwise_features") or {}).get("bounded_wall_segment_tail_rank")),
                    safe_float((item.get("listwise_features") or {}).get("bounded_wall_segment_score_rank")),
                    -safe_float(item.get("relation_score")),
                    str(item.get("relation_id")),
                )
            )
            for row in support_ordered:
                try_select(row, allow_support_overflow=True)
        if relation == "contains_symbol" and max_contains_symbol_support_overflow_slots > 0:
            support_ordered = [row for row in bucket if contains_symbol_support_overflow_candidate(row)]
            support_ordered.sort(
                key=lambda item: (
                    safe_float(item.get("support_criticality_score")),
                    safe_float(item.get("relation_score")),
                    -safe_float((item.get("listwise_features") or {}).get("contains_support_rank")),
                    str(item.get("relation_id")),
                ),
                reverse=True,
            )
            for row in support_ordered:
                try_select(row, allow_contains_support_overflow=True)
    selected.sort(key=lambda item: (str(item.get("relation")), -safe_float(item.get("relation_score")), str(item.get("source_candidate_id")), str(item.get("target_candidate_id"))))
    return selected


def select_contains_symbol_assignment_rows(bucket: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    threshold = safe_float(params.get("assignment_score_threshold"))
    max_rooms = int(params.get("max_rooms_per_symbol") or 999999)
    max_edges = int(params.get("max_edges_per_room_symbol") or 999999)
    overflow_slots = int(params.get("critical_overflow_edges_per_symbol") or 0)
    by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in bucket:
        if safe_float(row.get("assignment_score")) < threshold:
            continue
        symbol_id = str(row.get("symbol_instance_cluster_id") or row.get("target_cluster_id") or row.get("target_candidate_id"))
        page_symbol_key = f"{row.get('row_id')}|{symbol_id}"
        by_symbol[page_symbol_key].append(row)

    out: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def commit(row: dict[str, Any], reason: str) -> None:
        relation_id = str(row.get("relation_id") or "")
        if relation_id in selected_ids:
            return
        item = dict(row)
        item["assignment_selection_reason"] = reason
        out.append(item)
        selected_ids.add(relation_id)

    for rows_for_symbol in by_symbol.values():
        by_room: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows_for_symbol:
            room_id = str(row.get("source_cluster_id") or row.get("source_candidate_id"))
            by_room[room_id].append(row)
        room_order = []
        for room_id, room_rows in by_room.items():
            ordered_room = sorted(
                room_rows,
                key=lambda item: (safe_float(item.get("assignment_score")), safe_float(item.get("relation_score"))),
                reverse=True,
            )
            best = ordered_room[0]
            room_order.append((room_id, ordered_room, safe_float(best.get("assignment_score")), safe_float(best.get("relation_score"))))
        room_order.sort(key=lambda item: (item[2], item[3]), reverse=True)
        for _room_id, room_rows, _best_assignment, _best_relation in room_order[:max_rooms]:
            for row in room_rows[:max_edges]:
                commit(row, "symbol_room_assignment")
        if overflow_slots > 0:
            overflowed = 0
            overflow_rows = sorted(
                rows_for_symbol,
                key=lambda item: (safe_float(item.get("assignment_score")), safe_float(item.get("relation_score"))),
                reverse=True,
            )
            for row in overflow_rows:
                if str(row.get("relation_id") or "") in selected_ids:
                    continue
                commit(row, "learned_symbol_critical_overflow")
                overflowed += 1
                if overflowed >= overflow_slots:
                    break
    return out


def render_pages(
    relation_pages: list[dict[str, Any]],
    adapter_by_id: dict[str, dict[str, Any]],
    model: dict[str, Any],
    policy: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    page_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    warning_counts: Counter[str] = Counter()
    page_stats: list[dict[str, Any]] = []
    for page in relation_pages:
        row_id = str(page.get("id"))
        adapter = adapter_by_id.get(row_id)
        if not adapter:
            warning_counts["missing_adapter_row"] += 1
            continue
        candidates = row_candidate_map(adapter)
        cluster_ids, cluster_summaries, cluster_warnings = cluster_candidates(row_id, candidates)
        symbol_instance_ids, split_audit = build_symbol_instance_ids(row_id, candidates, cluster_ids, DEFAULT_SPLITTER_MODEL)
        relations = list(((page.get("scene_graph") or {}).get("relations") or []))
        pair_buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for rel in relations:
            pair_buckets[topology_pair_key(rel, cluster_ids, symbol_instance_ids)].append(rel)
        node_component, component_sizes, degree, component_edges = connected_components(
            [
                {
                    **rel,
                    "source_cluster_id": cluster_ids.get(str(rel.get("source_candidate_id")), f"missing:{rel.get('source_candidate_id')}"),
                    "target_cluster_id": cluster_ids.get(str(rel.get("target_candidate_id")), f"missing:{rel.get('target_candidate_id')}"),
                }
                for rel in relations
            ]
        )
        scored_rows: list[dict[str, Any]] = []
        for rel in relations:
            source_cluster_id = cluster_ids.get(str(rel.get("source_candidate_id")), f"missing:{rel.get('source_candidate_id')}")
            target_cluster_id = cluster_ids.get(str(rel.get("target_candidate_id")), f"missing:{rel.get('target_candidate_id')}")
            enriched_rel = {
                **rel,
                "source_cluster_id": source_cluster_id,
                "target_cluster_id": target_cluster_id,
            }
            pair_key = topology_pair_key(rel, cluster_ids, symbol_instance_ids)
            duplicate_count = max(1, len(pair_buckets[pair_key]))
            features = relation_features(
                enriched_rel,
                candidates,
                duplicate_count,
                node_component,
                component_sizes,
                degree,
                component_edges,
            )
            row = {
                "row_id": row_id,
                "relation_id": rel.get("relation_id"),
                "relation": rel.get("relation"),
                "source_candidate_id": rel.get("source_candidate_id"),
                "target_candidate_id": rel.get("target_candidate_id"),
                "source_cluster_id": source_cluster_id,
                "target_cluster_id": target_cluster_id,
                "symbol_instance_cluster_id": symbol_instance_ids.get(str(rel.get("target_candidate_id")), target_cluster_id),
                "component_id": node_component.get(graph_node_key(enriched_rel, "source")) or node_component.get(graph_node_key(enriched_rel, "target")) or "relation_component_missing",
                "duplicate_relation_count": duplicate_count,
                "edge_features": features,
                "confidence": safe_float(rel.get("confidence")),
                "labels": {},
            }
            scored_rows.append(row)
        scored_rows = score_rows(scored_rows, model)
        kept = select_rows(scored_rows, policy)
        kept_ids = {str(row.get("relation_id")) for row in kept}
        relations_out: list[dict[str, Any]] = []
        for rel in relations:
            rid = str(rel.get("relation_id"))
            if rid not in kept_ids:
                continue
            row = next((item for item in kept if str(item.get("relation_id")) == rid), None)
            out = dict(rel)
            evidence = dict(out.get("evidence") if isinstance(out.get("evidence"), dict) else {})
            evidence.update(
                {
                    "relation_graph_policy": "relation_graph_reconstruction_v18",
                    "relation_graph_score": row.get("relation_score") if row else None,
                    "relation_graph_component_id": row.get("component_id") if row else None,
                    "relation_graph_duplicate_count": row.get("duplicate_relation_count") if row else None,
                    "source_cluster_id": row.get("source_cluster_id") if row else None,
                    "target_cluster_id": row.get("target_cluster_id") if row else None,
                }
            )
            out["evidence"] = evidence
            out["relation_graph_score"] = row.get("relation_score") if row else None
            out["relation_graph_component_id"] = row.get("component_id") if row else None
            out["relation_graph_duplicate_count"] = row.get("duplicate_relation_count") if row else None
            out["source_cluster_id"] = row.get("source_cluster_id") if row else None
            out["target_cluster_id"] = row.get("target_cluster_id") if row else None
            out["symbol_instance_cluster_id"] = row.get("symbol_instance_cluster_id") if row else None
            out["source_integrity"] = integrity()
            relations_out.append(out)
            feature_rows.append(
                {
                    "row_id": row_id,
                    "relation_id": rid,
                    "relation": rel.get("relation"),
                    "source_candidate_id": rel.get("source_candidate_id"),
                    "target_candidate_id": rel.get("target_candidate_id"),
                    "source_cluster_id": row.get("source_cluster_id") if row else None,
                    "target_cluster_id": row.get("target_cluster_id") if row else None,
                    "symbol_instance_cluster_id": row.get("symbol_instance_cluster_id") if row else None,
                    "component_id": row.get("component_id") if row else None,
                    "relation_score": row.get("relation_score") if row else None,
                    "edge_features": row.get("edge_features") if row else {},
                    "source_integrity": integrity(),
                }
            )
        warning_counts.update(item.get("warning") for item in cluster_warnings)
        page_stats.append(
            {
                "row_id": row_id,
                "before_relations": len(relations),
                "after_relations": len(relations_out),
                "selected_reduction": round(1.0 - len(relations_out) / max(len(relations), 1), 6),
                "cluster_count": len(cluster_summaries),
                "component_count": len(component_sizes),
                "split_audit": split_audit,
            }
        )
        page_rows.append(
            {
                "id": row_id,
                "image": page.get("image") or adapter.get("image"),
                "image_size": page.get("image_size") or adapter.get("image_size") or [512, 512],
                "source_integrity": integrity(),
                "route_trace": {
                    **integrity(),
                    "stage": "relation_graph_reconstruction_v18",
                    "gold_loaded_after_inference_for_evaluation_only": False,
                },
                "scene_graph": {
                    "nodes": [],
                    "relations": relations_out,
                    "candidate_counts": ((adapter.get("scene_graph") or {}).get("candidate_counts") or {}),
                    "relation_counts": dict(Counter(str(rel.get("relation")) for rel in relations_out)),
                },
            }
        )
    audit = {
        "task": "IMG-MOE-V18-REBUILD-005",
        "rows": len(page_rows),
        "features": len(feature_rows),
        "warning_counts": {str(k): int(v) for k, v in warning_counts.items() if k},
        "page_stats": page_stats[:100],
        "source_integrity": integrity(),
        "gold_loaded_after_inference_for_evaluation_only": False,
        "gold_used_for_inference": False,
    }
    return page_rows, feature_rows, audit


def evaluate_selection(page_rows: list[dict[str, Any]], source_rows: list[dict[str, Any]]) -> dict[str, Any]:
    report = evaluate_relations(page_rows, source_rows)
    report["source_integrity"] = integrity()
    report["gold_loaded_after_inference_for_evaluation_only"] = True
    report["gold_used_for_inference"] = False
    return report


def summarize_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    before_metrics = before.get("relation_metrics") or {}
    after_metrics = after.get("relation_metrics") or {}
    for rel_type in RELATIONS:
        before_row = before_metrics.get(rel_type) or {}
        after_row = after_metrics.get(rel_type) or {}
        before_pred = int(before_row.get("predicted") or 0)
        after_pred = int(after_row.get("predicted") or 0)
        before_dup = int(before_row.get("duplicate_positive") or 0)
        after_dup = int(after_row.get("duplicate_positive") or 0)
        out[rel_type] = {
            "predicted_before": before_pred,
            "predicted_after": after_pred,
            "predicted_reduction": round(1.0 - after_pred / max(before_pred, 1), 6),
            "recall_before": before_row.get("recall"),
            "recall_after": after_row.get("recall"),
            "recall_drop_abs": round(float(before_row.get("recall") or 0.0) - float(after_row.get("recall") or 0.0), 6),
            "precision_before": before_row.get("precision"),
            "precision_after": after_row.get("precision"),
            "duplicate_positive_before": before_dup,
            "duplicate_positive_after": after_dup,
            "duplicate_positive_reduction": round(1.0 - after_dup / max(before_dup, 1), 6),
        }
    return out
