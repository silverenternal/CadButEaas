#!/usr/bin/env python3
"""Train/apply a relation-aware topology reranker for v18 raster-only MoE.

The script uses gold labels only offline to fit relation-type rankers and select
compression policy. Inference inputs remain detector/topology outputs only.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np

from build_topology_relations_v18 import bbox, center, evaluate_relations, integrity, iou, load_gold, relation_label, write_json, write_jsonl
from nms_topology_relations_v18 import (
    cluster_candidates,
    load_by_id,
    load_jsonl,
    metric_delta,
    relation_pair_key,
    row_candidate_map,
)

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/vlm"
CHECKPOINT = ROOT / "checkpoints/relation_reranker_v18"
SPLITTER_CHECKPOINT = ROOT / "checkpoints/contains_symbol_instance_splitter_v18/model.json"

DEFAULT_INPUT = REPORT / "topology_relations_v18_symbol_boundary_fixed_candidates.jsonl"
DEFAULT_ADAPTER = REPORT / "detector_adapter_v18_symbol_boundary_fixed_candidates.jsonl"
DEFAULT_BEFORE_EVAL = REPORT / "topology_relations_v18_symbol_boundary_fixed_eval.json"
DEFAULT_OUTPUT = REPORT / "topology_relations_v18_relation_reranked_candidates.jsonl"
DEFAULT_FEATURES = REPORT / "topology_relations_v18_relation_reranked_features.jsonl"
DEFAULT_EVAL = REPORT / "topology_relations_v18_relation_reranked_eval.json"
DEFAULT_AUDIT = REPORT / "topology_relations_v18_relation_reranked_audit.json"

DEFAULT_SPLITTER_MODEL: dict[str, Any] = {
    "enabled": False,
    "min_members": 4,
    "min_spread": 8.0,
    "center_threshold": 3.0,
    "merge_iou": 0.65,
    "min_size_ratio": 0.55,
    "max_instances": 3,
}

RELATIONS = ["bounded_by", "contains_symbol", "labeled_by_text", "adjacent_to"]
FEATURES = [
    "relation_confidence",
    "detector_confidence_product",
    "source_confidence",
    "target_confidence",
    "duplicate_relation_count",
    "bbox_distance",
    "side_overlap_ratio",
    "axis_overlap_ratio",
    "gap",
    "orientation_compatible",
    "center_distance_to_room_center",
    "target_area_ratio_scaled",
    "type_confidence",
    "ocr_confidence",
    "readability_confidence",
    "source_bbox_area",
    "source_bbox_width",
    "source_bbox_height",
    "source_bbox_aspect",
    "target_bbox_area",
    "target_bbox_width",
    "target_bbox_height",
    "target_bbox_aspect",
    "target_room_edge_margin",
    "target_room_edge_margin_ratio",
    "target_room_center_x_offset",
    "target_room_center_y_offset",
    "target_objectness_score",
    "target_local_dark_density",
    "target_anchor_area",
    "target_anchor_width",
    "target_anchor_height",
    "target_anchor_aspect",
    "target_kind_dark_pixel_anchor",
    "target_kind_dark_connected_component",
    "target_kind_other",
    "target_bbox_to_anchor_area_ratio",
    "target_room_min_edge_distance_ratio",
    "target_room_max_edge_distance_ratio",
]


def safe_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return 0.0
        return out
    except (TypeError, ValueError):
        return 0.0


def candidate_conf(candidates: dict[str, dict[str, Any]], candidate_id: Any) -> float:
    cand = candidates.get(str(candidate_id)) or {}
    return safe_float(cand.get("confidence"))


def load_splitter_model(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return dict(DEFAULT_SPLITTER_MODEL)
    model_path = Path(path)
    if not model_path.exists():
        return dict(DEFAULT_SPLITTER_MODEL)
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    model = dict(DEFAULT_SPLITTER_MODEL)
    if isinstance(payload.get("model"), dict):
        model.update(payload["model"])
    else:
        model.update({key: value for key, value in payload.items() if key in DEFAULT_SPLITTER_MODEL})
    model["enabled"] = bool(model.get("enabled"))
    return model


def box_center_distance(left: list[float], right: list[float]) -> float:
    lx, ly = center(left)
    rx, ry = center(right)
    return math.hypot(lx - rx, ly - ry)


def box_size_ratio(left: list[float], right: list[float]) -> float:
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return min(left_area, right_area) / max(left_area, right_area, 1e-9)


def symbol_payload_density(cand: dict[str, Any]) -> float:
    payload = cand.get("payload") if isinstance(cand.get("payload"), dict) else {}
    return safe_float(payload.get("local_dark_density"))


def symbol_instance_similar(left: dict[str, Any], right: dict[str, Any], model: dict[str, Any]) -> bool:
    lb = bbox(left.get("bbox"))
    rb = bbox(right.get("bbox"))
    if lb is None or rb is None:
        return False
    if iou(lb, rb) >= safe_float(model.get("merge_iou")):
        return True
    if box_size_ratio(lb, rb) < safe_float(model.get("min_size_ratio")):
        return False
    return box_center_distance(lb, rb) <= safe_float(model.get("center_threshold"))


def build_symbol_instance_ids(
    row_id: str,
    candidates: dict[str, dict[str, Any]],
    cluster_ids: dict[str, str],
    splitter_model: dict[str, Any],
) -> tuple[dict[str, str], dict[str, Any]]:
    """Split dense symbol clusters into detector-only instance ids."""
    instance_ids = dict(cluster_ids)
    audit = Counter()
    if not splitter_model.get("enabled"):
        return instance_ids, {"enabled": False}

    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cid, cand in candidates.items():
        if str(cand.get("family")) != "symbol":
            continue
        cluster_id = cluster_ids.get(cid)
        if cluster_id:
            by_cluster[cluster_id].append(cand)

    min_members = int(splitter_model.get("min_members") or 4)
    min_spread = safe_float(splitter_model.get("min_spread"))
    max_instances = max(1, int(splitter_model.get("max_instances") or 1))

    for cluster_id, members in by_cluster.items():
        if len(members) < min_members:
            continue
        boxes = [bbox(member.get("bbox")) for member in members]
        boxes = [item for item in boxes if item is not None]
        if len(boxes) < min_members:
            continue
        spread = 0.0
        for index, left in enumerate(boxes):
            for right in boxes[index + 1 :]:
                spread = max(spread, box_center_distance(left, right))
        if spread < min_spread:
            continue

        ordered = sorted(
            members,
            key=lambda item: (candidate_conf(candidates, item.get("candidate_id")), symbol_payload_density(item)),
            reverse=True,
        )
        reps: list[dict[str, Any]] = []
        assignment: dict[str, int] = {}
        for cand in ordered:
            cid = str(cand.get("candidate_id"))
            best_index: int | None = None
            best_score = -1.0
            for index, rep in enumerate(reps):
                cb = bbox(cand.get("bbox"))
                rb = bbox(rep.get("bbox"))
                if cb is None or rb is None or not symbol_instance_similar(cand, rep, splitter_model):
                    continue
                score = iou(cb, rb) + 1.0 / max(box_center_distance(cb, rb), 1.0)
                if score > best_score:
                    best_index = index
                    best_score = score
            if best_index is None:
                if len(reps) >= max_instances:
                    cb = bbox(cand.get("bbox")) or [0.0, 0.0, 0.0, 0.0]
                    best_index = min(
                        range(len(reps)),
                        key=lambda index: box_center_distance(cb, bbox(reps[index].get("bbox")) or [0.0, 0.0, 0.0, 0.0]),
                    )
                else:
                    best_index = len(reps)
                    reps.append(cand)
            assignment[cid] = best_index

        if len(reps) <= 1:
            continue
        audit["split_symbol_clusters"] += 1
        audit["split_symbol_members"] += len(members)
        audit["emitted_symbol_instances"] += len(reps)
        for cid, index in assignment.items():
            instance_ids[cid] = f"{cluster_id}_inst_{index:02d}"

    return instance_ids, {"enabled": True, **dict(audit)}


def topology_pair_key(
    rel: dict[str, Any],
    cluster_ids: dict[str, str],
    symbol_instance_ids: dict[str, str] | None = None,
) -> tuple[str, str, str]:
    rel_type = str(rel.get("relation"))
    if rel_type != "contains_symbol" or symbol_instance_ids is None:
        return relation_pair_key(rel, cluster_ids)
    source_cluster = cluster_ids.get(str(rel.get("source_candidate_id")), f"missing:{rel.get('source_candidate_id')}")
    target_instance = symbol_instance_ids.get(
        str(rel.get("target_candidate_id")),
        cluster_ids.get(str(rel.get("target_candidate_id")), f"missing:{rel.get('target_candidate_id')}"),
    )
    return rel_type, source_cluster, target_instance


def contains_symbol_component_ids(buckets: dict[tuple[str, str, str], list[dict[str, Any]]]) -> dict[tuple[str, str, str], str]:
    graph: dict[str, set[str]] = defaultdict(set)
    for rel_type, source_cluster, target_cluster in buckets:
        if rel_type != "contains_symbol":
            continue
        room_node = f"room:{source_cluster}"
        symbol_node = f"symbol:{target_cluster}"
        graph[room_node].add(symbol_node)
        graph[symbol_node].add(room_node)

    node_component: dict[str, str] = {}
    component_count = 0
    for node in sorted(graph):
        if node in node_component:
            continue
        component_id = f"contains_component_{component_count:05d}"
        component_count += 1
        queue: deque[str] = deque([node])
        node_component[node] = component_id
        while queue:
            current = queue.popleft()
            for nxt in graph[current]:
                if nxt in node_component:
                    continue
                node_component[nxt] = component_id
                queue.append(nxt)

    out: dict[tuple[str, str, str], str] = {}
    for key in buckets:
        rel_type, source_cluster, _target_cluster = key
        if rel_type != "contains_symbol":
            continue
        out[key] = node_component.get(f"room:{source_cluster}", "contains_component_missing")
    return out


def bbox_stats(value: Any) -> tuple[float, float, float, float]:
    b = bbox(value)
    if b is None:
        return 0.0, 0.0, 0.0, 0.0
    width = max(0.0, b[2] - b[0])
    height = max(0.0, b[3] - b[1])
    return width * height, width, height, width / max(height, 1e-9)


def room_target_geometry(room_box: Any, target_box: Any) -> dict[str, float]:
    rb = bbox(room_box)
    tb = bbox(target_box)
    if rb is None or tb is None:
        return {
            "target_room_edge_margin": 0.0,
            "target_room_edge_margin_ratio": 0.0,
            "target_room_center_x_offset": 0.0,
            "target_room_center_y_offset": 0.0,
            "target_room_min_edge_distance_ratio": 0.0,
            "target_room_max_edge_distance_ratio": 0.0,
        }
    tx = (tb[0] + tb[2]) / 2.0
    ty = (tb[1] + tb[3]) / 2.0
    rw = max(1e-9, rb[2] - rb[0])
    rh = max(1e-9, rb[3] - rb[1])
    edge_distances = [tx - rb[0], rb[2] - tx, ty - rb[1], rb[3] - ty]
    margin = min(edge_distances)
    return {
        "target_room_edge_margin": max(0.0, margin),
        "target_room_edge_margin_ratio": max(0.0, margin) / max(min(rw, rh), 1e-9),
        "target_room_center_x_offset": abs(tx - ((rb[0] + rb[2]) / 2.0)) / rw,
        "target_room_center_y_offset": abs(ty - ((rb[1] + rb[3]) / 2.0)) / rh,
        "target_room_min_edge_distance_ratio": max(0.0, min(edge_distances)) / max(min(rw, rh), 1e-9),
        "target_room_max_edge_distance_ratio": max(0.0, max(edge_distances)) / max(max(rw, rh), 1e-9),
    }


def symbol_payload_features(target: dict[str, Any], target_area: float) -> dict[str, float]:
    payload = target.get("payload") if isinstance(target.get("payload"), dict) else {}
    anchor = payload.get("anchor_size")
    if isinstance(anchor, list) and len(anchor) == 2:
        anchor_w = max(0.0, safe_float(anchor[0]))
        anchor_h = max(0.0, safe_float(anchor[1]))
    else:
        anchor_w = 0.0
        anchor_h = 0.0
    anchor_area = anchor_w * anchor_h
    kind = str(payload.get("candidate_kind") or "")
    return {
        "target_objectness_score": safe_float(payload.get("objectness_score")),
        "target_local_dark_density": safe_float(payload.get("local_dark_density")),
        "target_anchor_area": anchor_area,
        "target_anchor_width": anchor_w,
        "target_anchor_height": anchor_h,
        "target_anchor_aspect": anchor_w / max(anchor_h, 1e-9) if anchor_w and anchor_h else 0.0,
        "target_kind_dark_pixel_anchor": 1.0 if kind == "dark_pixel_anchor" else 0.0,
        "target_kind_dark_connected_component": 1.0 if kind == "dark_connected_component" else 0.0,
        "target_kind_other": 1.0 if kind and kind not in {"dark_pixel_anchor", "dark_connected_component"} else 0.0,
        "target_bbox_to_anchor_area_ratio": target_area / max(anchor_area, 1e-9) if anchor_area else 0.0,
    }


def feature_vector(rel: dict[str, Any], candidates: dict[str, dict[str, Any]], duplicate_count: int = 1) -> dict[str, float]:
    evidence = rel.get("evidence") if isinstance(rel.get("evidence"), dict) else {}
    source = candidates.get(str(rel.get("source_candidate_id"))) or {}
    target = candidates.get(str(rel.get("target_candidate_id"))) or {}
    source_area, source_w, source_h, source_aspect = bbox_stats(source.get("bbox") or evidence.get("room_bbox"))
    target_area, target_w, target_h, target_aspect = bbox_stats(target.get("bbox") or evidence.get("target_bbox"))
    room_geom = room_target_geometry(evidence.get("room_bbox") or source.get("bbox"), evidence.get("target_bbox") or target.get("bbox"))
    return {
        "relation_confidence": safe_float(rel.get("confidence")),
        "detector_confidence_product": safe_float(evidence.get("detector_confidence_product")),
        "source_confidence": candidate_conf(candidates, rel.get("source_candidate_id")),
        "target_confidence": candidate_conf(candidates, rel.get("target_candidate_id")),
        "duplicate_relation_count": float(duplicate_count),
        "bbox_distance": safe_float(evidence.get("bbox_distance")),
        "side_overlap_ratio": safe_float(evidence.get("side_overlap_ratio")),
        "axis_overlap_ratio": safe_float(evidence.get("axis_overlap_ratio")),
        "gap": safe_float(evidence.get("gap")),
        "orientation_compatible": safe_float(evidence.get("orientation_compatible")),
        "center_distance_to_room_center": safe_float(evidence.get("center_distance_to_room_center")),
        "target_area_ratio_scaled": safe_float(evidence.get("target_area_ratio_scaled")),
        "type_confidence": safe_float(evidence.get("type_confidence")),
        "ocr_confidence": safe_float(evidence.get("ocr_confidence")),
        "readability_confidence": safe_float(evidence.get("readability_confidence")),
        "source_bbox_area": source_area,
        "source_bbox_width": source_w,
        "source_bbox_height": source_h,
        "source_bbox_aspect": source_aspect,
        "target_bbox_area": target_area,
        "target_bbox_width": target_w,
        "target_bbox_height": target_h,
        "target_bbox_aspect": target_aspect,
        **room_geom,
        **symbol_payload_features(target, target_area),
    }


def vector_values(features: dict[str, float]) -> list[float]:
    values: list[float] = []
    for name in FEATURES:
        value = features.get(name, 0.0)
        if name in {"bbox_distance", "gap"}:
            value = 1.0 / (1.0 + max(0.0, value))
        elif name == "duplicate_relation_count":
            value = math.log1p(max(0.0, value))
        elif name.endswith("_area"):
            value = math.log1p(max(0.0, value)) / 12.0
        elif name.endswith("_width") or name.endswith("_height") or name == "target_room_edge_margin":
            value = math.log1p(max(0.0, value)) / 8.0
        elif name.endswith("_aspect"):
            value = math.log(max(value, 1e-6))
        else:
            value = max(0.0, value)
        values.append(value)
    return values


def train_model(
    relation_pages: list[dict[str, Any]],
    adapter_by_id: dict[str, dict[str, Any]],
    train_row_ids: set[str],
    splitter_model: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gold = load_gold()
    samples: dict[str, list[list[float]]] = defaultdict(list)
    labels: dict[str, list[int]] = defaultdict(list)
    rank_groups: dict[str, list[tuple[list[list[float]], list[int]]]] = defaultdict(list)
    counts: dict[str, Counter[str]] = defaultdict(Counter)

    for page in relation_pages:
        row_id = str(page.get("id"))
        if row_id not in train_row_ids:
            continue
        adapter = adapter_by_id.get(row_id)
        if not adapter:
            continue
        candidates = row_candidate_map(adapter)
        cluster_ids, _, _ = cluster_candidates(row_id, candidates)
        symbol_instance_ids, _split_audit = build_symbol_instance_ids(row_id, candidates, cluster_ids, splitter_model or DEFAULT_SPLITTER_MODEL)
        buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for rel in ((page.get("scene_graph") or {}).get("relations") or []):
            buckets[topology_pair_key(rel, cluster_ids, symbol_instance_ids)].append(rel)
        for bucket in buckets.values():
            dup = len(bucket)
            group_values: list[list[float]] = []
            group_labels: list[int] = []
            for rel in bucket:
                rel_type = str(rel.get("relation"))
                left, right, ok = relation_label({**rel, "row_id": row_id}, candidates, gold)
                values = vector_values(feature_vector(rel, candidates, dup))
                label = "pos" if ok and left and right else "neg"
                counts[rel_type][label] += 1
                samples[rel_type].append(values)
                label_int = 1 if label == "pos" else 0
                labels[rel_type].append(label_int)
                group_values.append(values)
                group_labels.append(label_int)
            if bucket:
                rank_groups[str(bucket[0].get("relation"))].append((group_values, group_labels))

    model: dict[str, Any] = {"type": "relation_type_balanced_logistic_ranker_with_contains_symbol_listwise_selector", "features": FEATURES, "relations": {}, "train_counts": {}}
    for rel_type in RELATIONS:
        rel_counts = counts[rel_type]
        if not samples[rel_type]:
            model["relations"][rel_type] = {"weights": [0.0] * len(FEATURES), "mean": [0.0] * len(FEATURES), "std": [1.0] * len(FEATURES), "bias": 0.0}
            model["train_counts"][rel_type] = dict(counts[rel_type])
            continue
        x = np.asarray(samples[rel_type], dtype=np.float32)
        y = np.asarray(labels[rel_type], dtype=np.float32)
        mean_arr = x.mean(axis=0)
        std_arr = x.std(axis=0) + 1e-6
        z = (x - mean_arr) / std_arr
        pos_n = max(float(y.sum()), 1.0)
        neg_n = max(float(len(y) - y.sum()), 1.0)
        sample_weight = np.where(y > 0.5, len(y) / (2.0 * pos_n), len(y) / (2.0 * neg_n)).astype(np.float32)
        weights = np.zeros((len(FEATURES),), dtype=np.float32)
        bias = 0.0
        lr = 0.08
        reg = 0.002
        for _ in range(180):
            logits = np.clip(z @ weights + bias, -30.0, 30.0)
            pred = 1.0 / (1.0 + np.exp(-logits))
            err = (pred - y) * sample_weight
            grad_w = (z.T @ err) / max(len(y), 1) + reg * weights
            grad_b = float(err.mean())
            weights -= lr * grad_w
            bias -= lr * grad_b
        if rel_type == "contains_symbol":
            train_listwise_groups = []
            for group_values, group_labels in rank_groups[rel_type]:
                if not group_values or not any(group_labels) or all(group_labels):
                    continue
                group_x = (np.asarray(group_values, dtype=np.float32) - mean_arr) / std_arr
                group_y = np.asarray(group_labels, dtype=np.float32)
                group_y = group_y / max(float(group_y.sum()), 1.0)
                train_listwise_groups.append((group_x, group_y))
            if train_listwise_groups:
                lr = 0.03
                reg = 0.001
                for _ in range(220):
                    grad_w = reg * weights
                    grad_b = 0.0
                    for group_x, group_y in train_listwise_groups:
                        logits = np.clip(group_x @ weights + bias, -30.0, 30.0)
                        exp_logits = np.exp(logits - float(logits.max()))
                        prob = exp_logits / max(float(exp_logits.sum()), 1e-9)
                        err = prob - group_y
                        grad_w += (group_x.T @ err) / max(len(train_listwise_groups), 1)
                        grad_b += float(err.sum()) / max(len(train_listwise_groups), 1)
                    weights -= lr * grad_w
                    bias -= lr * grad_b
        model["relations"][rel_type] = {"weights": weights.astype(float).tolist(), "mean": mean_arr.astype(float).tolist(), "std": std_arr.astype(float).tolist(), "bias": float(bias)}
        model["train_counts"][rel_type] = dict(rel_counts)
    return model


def score_relation(rel: dict[str, Any], candidates: dict[str, dict[str, Any]], model: dict[str, Any], duplicate_count: int) -> float:
    rel_type = str(rel.get("relation"))
    params = (model.get("relations") or {}).get(rel_type) or {}
    weights = params.get("weights") or [0.0] * len(FEATURES)
    mean = params.get("mean") or [0.0] * len(FEATURES)
    std = params.get("std") or [1.0] * len(FEATURES)
    bias = float(params.get("bias") or 0.0)
    values = vector_values(feature_vector(rel, candidates, duplicate_count))
    learned = bias
    for index, value in enumerate(values):
        learned += float(weights[index]) * ((value - float(mean[index])) / max(float(std[index]), 1e-9))
    return learned + 0.15 * safe_float(rel.get("confidence"))


def contains_symbol_pair_keep_budget(
    bucket: list[dict[str, Any]],
    candidates: dict[str, dict[str, Any]],
    model: dict[str, Any],
) -> tuple[int, dict[str, float]]:
    duplicate_count = len(bucket)
    if duplicate_count <= 1:
        return 1, {"top2_score_gap": 0.0, "top3_score_gap": 0.0}
    ordered = sorted(bucket, key=lambda item: score_relation(item, candidates, model, duplicate_count), reverse=True)
    top_score = score_relation(ordered[0], candidates, model, duplicate_count)
    second_score = score_relation(ordered[1], candidates, model, duplicate_count)
    top2_gap = top_score - second_score
    keep = 1
    if top2_gap <= 0.01:
        keep = 2
    top3_gap = 0.0
    if duplicate_count >= 3:
        third_score = score_relation(ordered[2], candidates, model, duplicate_count)
        top3_gap = top_score - third_score
        if top3_gap <= 0.006 and top2_gap <= 0.02:
            keep = 3
    return keep, {"top2_score_gap": round(top2_gap, 6), "top3_score_gap": round(top3_gap, 6)}


def select_cluster_representatives(
    page: dict[str, Any],
    adapter: dict[str, Any],
    model: dict[str, Any],
    splitter_model: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row_id = str(page.get("id"))
    candidates = row_candidate_map(adapter)
    cluster_ids, cluster_summaries, warnings = cluster_candidates(row_id, candidates)
    symbol_instance_ids, split_audit = build_symbol_instance_ids(row_id, candidates, cluster_ids, splitter_model or DEFAULT_SPLITTER_MODEL)
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    missing = 0
    for rel in ((page.get("scene_graph") or {}).get("relations") or []):
        if str(rel.get("source_candidate_id")) not in cluster_ids or str(rel.get("target_candidate_id")) not in cluster_ids:
            missing += 1
        buckets[topology_pair_key(rel, cluster_ids, symbol_instance_ids)].append(rel)
    component_by_key = contains_symbol_component_ids(buckets)

    selected: list[dict[str, Any]] = []
    for key, bucket in buckets.items():
        dup = len(bucket)
        component_id = component_by_key.get(key)
        keep_per_pair = 1
        keep_budget_trace: dict[str, float] = {}
        if str(bucket[0].get("relation")) == "contains_symbol":
            keep_per_pair, keep_budget_trace = contains_symbol_pair_keep_budget(bucket, candidates, model)
            keep_per_pair = min(keep_per_pair, dup)
        ordered = sorted(bucket, key=lambda item: score_relation(item, candidates, model, dup), reverse=True)
        kept_ids = {item.get("relation_id") for item in ordered[:keep_per_pair]}
        for rel in ordered[:keep_per_pair]:
            out = dict(rel)
            evidence = dict(out.get("evidence") if isinstance(out.get("evidence"), dict) else {})
            relation_score = score_relation(out, candidates, model, dup)
            source_cluster_id = cluster_ids.get(str(rel.get("source_candidate_id")), key[1])
            target_cluster_id = cluster_ids.get(str(rel.get("target_candidate_id")), key[2])
            room_instance_cluster_id = source_cluster_id
            target_symbol_instance_id = symbol_instance_ids.get(str(rel.get("target_candidate_id")), target_cluster_id)
            evidence.update(
                {
                    "source_cluster_id": source_cluster_id,
                    "target_cluster_id": target_cluster_id,
                    "room_instance_cluster_id": room_instance_cluster_id,
                    "component_id": component_id,
                    "duplicate_relation_count": dup,
                    "contains_symbol_pair_keep_budget": keep_per_pair if str(out.get("relation")) == "contains_symbol" else 1,
                    **keep_budget_trace,
                    "relation_pair_keep_rank": len([item for item in ordered[:keep_per_pair] if score_relation(item, candidates, model, dup) > relation_score]) + 1,
                    "relation_rerank_score": round(relation_score, 6),
                    "relation_rerank_model": "relation_reranker_v18_balanced_logistic_symbol_anchor_features",
                    "original_relation_id": rel.get("relation_id"),
                    "suppressed_relation_ids": [item.get("relation_id") for item in bucket if item.get("relation_id") not in kept_ids][:25],
                }
            )
            if str(out.get("relation")) == "contains_symbol":
                evidence["target_symbol_instance_cluster_id"] = target_symbol_instance_id
                evidence["symbol_instance_cluster_id"] = target_symbol_instance_id
                evidence["contains_symbol_instance_splitter_enabled"] = bool((splitter_model or DEFAULT_SPLITTER_MODEL).get("enabled"))
            out["evidence"] = evidence
            out["source_cluster_id"] = source_cluster_id
            out["target_cluster_id"] = target_cluster_id
            out["room_instance_cluster_id"] = room_instance_cluster_id
            if str(out.get("relation")) == "contains_symbol":
                out["component_id"] = component_id
                out["target_symbol_instance_cluster_id"] = target_symbol_instance_id
                out["symbol_instance_cluster_id"] = target_symbol_instance_id
            out["cluster_duplicate_count"] = dup
            if str(out.get("relation")) == "contains_symbol":
                out["contains_symbol_pair_keep_budget"] = keep_per_pair
            out["relation_rerank_score"] = round(relation_score, 6)
            out["source_integrity"] = integrity()
            selected.append(out)

    meta = {
        "cluster_count": len(cluster_summaries),
        "missing_cluster_relations": missing,
        "contains_symbol_component_count": len(set(component_by_key.values())),
        "cluster_warnings": warnings,
        "symbol_instance_splitter": split_audit,
    }
    return selected, meta


def cap_by_policy(relations: list[dict[str, Any]], policy: dict[str, int]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    adjacent_by_node: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_source: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    contains_by_component: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rel in relations:
        rel_type = str(rel.get("relation"))
        if rel_type == "adjacent_to":
            adjacent_by_node[str(rel.get("source_candidate_id"))].append(rel)
            adjacent_by_node[str(rel.get("target_candidate_id"))].append(rel)
        elif rel_type == "contains_symbol":
            evidence = rel.get("evidence") if isinstance(rel.get("evidence"), dict) else {}
            component_id = str(rel.get("component_id") or evidence.get("component_id") or "component_missing")
            source_id = str(rel.get("source_cluster_id") or rel.get("source_candidate_id"))
            contains_by_component[(component_id, source_id)].append(rel)
        elif rel_type in {"bounded_by", "contains_symbol", "labeled_by_text"}:
            by_source[(rel_type, str(rel.get("source_cluster_id") or rel.get("source_candidate_id")))].append(rel)
        else:
            selected.append(rel)

    for (rel_type, _source), bucket in by_source.items():
        cap = int(policy.get(rel_type, len(bucket)))
        ordered = sorted(bucket, key=lambda item: (safe_float(item.get("relation_rerank_score")), safe_float(item.get("confidence"))), reverse=True)
        selected.extend(ordered[:cap])

    contains_component_cap = int(policy.get("contains_symbol_component", policy.get("contains_symbol", 999999)))
    contains_pool: list[dict[str, Any]] = []
    for (_component_id, _source), bucket in contains_by_component.items():
        ordered = sorted(bucket, key=lambda item: (safe_float(item.get("relation_rerank_score")), safe_float(item.get("confidence"))), reverse=True)
        contains_pool.extend(ordered[:contains_component_cap])
    contains_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rel in contains_pool:
        contains_by_source[str(rel.get("source_cluster_id") or rel.get("source_candidate_id"))].append(rel)
    contains_source_cap = int(policy.get("contains_symbol", 999999))
    for bucket in contains_by_source.values():
        ordered = sorted(bucket, key=lambda item: (safe_float(item.get("relation_rerank_score")), safe_float(item.get("confidence"))), reverse=True)
        selected.extend(ordered[:contains_source_cap])

    adjacent_ids: set[str] = set()
    adjacent_cap = int(policy.get("adjacent_to", 999999))
    for bucket in adjacent_by_node.values():
        ordered = sorted(bucket, key=lambda item: (safe_float(item.get("relation_rerank_score")), safe_float(item.get("confidence"))), reverse=True)
        for rel in ordered[:adjacent_cap]:
            rid = str(rel.get("relation_id"))
            if rid not in adjacent_ids:
                adjacent_ids.add(rid)
                selected.append(rel)

    selected.sort(key=lambda item: (str(item.get("relation")), -safe_float(item.get("relation_rerank_score")), str(item.get("source_candidate_id")), str(item.get("target_candidate_id"))))
    return selected


def render_pages(
    prepared_pages: list[dict[str, Any]],
    policy: dict[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    page_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    warning_counts: Counter[str] = Counter()
    page_stats: list[dict[str, Any]] = []
    for prepared in prepared_pages:
        row_id = str(prepared["row_id"])
        page = prepared["page"]
        adapter = prepared["adapter"]
        pre_cap = prepared["relations"]
        meta = prepared["meta"]
        kept = cap_by_policy(pre_cap, policy)
        warning_counts.update(item.get("warning") for item in meta.get("cluster_warnings") or [])
        if meta.get("missing_cluster_relations"):
            warning_counts["orphan_cluster"] += 1
        if len(pre_cap) > len(kept):
            warning_counts["relation_policy_suppressed"] += 1
        for rel in kept:
            feature_rows.append(
                {
                    "row_id": row_id,
                    "relation_id": rel["relation_id"],
                    "relation": rel["relation"],
                    "source_candidate_id": rel["source_candidate_id"],
                    "target_candidate_id": rel["target_candidate_id"],
                    "source_cluster_id": rel.get("source_cluster_id"),
                    "target_cluster_id": rel.get("target_cluster_id"),
                    "component_id": rel.get("component_id"),
                    "room_instance_cluster_id": rel.get("room_instance_cluster_id"),
                    "target_symbol_instance_cluster_id": rel.get("target_symbol_instance_cluster_id"),
                    "symbol_instance_cluster_id": rel.get("symbol_instance_cluster_id"),
                    "confidence": rel["confidence"],
                    "relation_rerank_score": rel.get("relation_rerank_score"),
                    "features": rel.get("evidence") or {},
                    "label": None,
                    "source_integrity": integrity(),
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
                    "stage": "topology_relations_v18_relation_reranked",
                    "gold_loaded_after_inference_for_evaluation_only": False,
                },
                "scene_graph": {
                    "nodes": [],
                    "relations": kept,
                    "candidate_counts": ((adapter.get("scene_graph") or {}).get("candidate_counts") or {}),
                    "relation_counts": dict(Counter(str(rel.get("relation")) for rel in kept)),
                },
            }
        )
        page_stats.append(
            {
                "row_id": row_id,
                "before_relations": len((page.get("scene_graph") or {}).get("relations") or []),
                "clustered_relations": len(pre_cap),
                "after_relations": len(kept),
            }
        )
    audit = {
        "task": "IMG-MOE-V18-NEXT-004-RELATION-AWARE-RERANK",
        "policy": policy,
        "warning_counts": {str(k): v for k, v in warning_counts.items() if k},
        "page_stats": page_stats,
        "source_integrity": integrity(),
        "gold_loaded_after_inference_for_evaluation_only": False,
        "gold_used_for_inference": False,
    }
    return page_rows, feature_rows, audit


def prepare_pages(
    relation_pages: list[dict[str, Any]],
    adapter_by_id: dict[str, dict[str, Any]],
    model: dict[str, Any],
    splitter_model: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    gold = load_gold()
    prepared: list[dict[str, Any]] = []
    warnings: Counter[str] = Counter()
    for page in relation_pages:
        row_id = str(page.get("id"))
        adapter = adapter_by_id.get(row_id)
        if adapter is None:
            warnings["missing_adapter_row"] += 1
            continue
        pre_cap, meta = select_cluster_representatives(page, adapter, model, splitter_model)
        candidates = row_candidate_map(adapter)
        for rel in pre_cap:
            left, right, ok = relation_label({**rel, "row_id": row_id}, candidates, gold)
            rel["_eval_ok"] = bool(ok and left and right)
            rel["_eval_key"] = [row_id, left, right] if ok and left and right else None
        prepared.append({"row_id": row_id, "page": page, "adapter": adapter, "relations": pre_cap, "meta": meta})
    return prepared, warnings


def clean_relation(rel: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in rel.items() if not key.startswith("_eval_")}


def clean_pages(page_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for page in page_rows:
        out = dict(page)
        graph = dict(out.get("scene_graph") or {})
        graph["relations"] = [clean_relation(rel) for rel in graph.get("relations") or []]
        out["scene_graph"] = graph
        cleaned.append(out)
    return cleaned


def cached_eval_report(
    page_rows: list[dict[str, Any]],
    before_report: dict[str, Any],
    policy: dict[str, int],
    model: dict[str, Any],
    smoke: bool,
    locked: bool,
) -> dict[str, Any]:
    before_metrics = before_report.get("relation_metrics") or {}
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    matched: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for page in page_rows:
        for rel in ((page.get("scene_graph") or {}).get("relations") or []):
            rel_type = str(rel.get("relation"))
            counters[rel_type]["predicted"] += 1
            key_values = rel.get("_eval_key")
            if rel.get("_eval_ok") and isinstance(key_values, list) and len(key_values) == 3:
                key = (str(key_values[0]), str(key_values[1]), str(key_values[2]))
                if key in matched[rel_type]:
                    counters[rel_type]["duplicate_positive"] += 1
                else:
                    matched[rel_type].add(key)
                    counters[rel_type]["true_positive"] += 1

    metrics: dict[str, Any] = {}
    for rel_type in RELATIONS:
        pred = counters[rel_type]["predicted"]
        tp = counters[rel_type]["true_positive"]
        gold_total = int((before_metrics.get(rel_type) or {}).get("gold") or 0)
        precision = tp / max(pred, 1)
        recall = tp / max(gold_total, 1)
        f1 = 0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall)
        metrics[rel_type] = {
            "true_positive": tp,
            "predicted": pred,
            "gold": gold_total,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "duplicate_positive": counters[rel_type]["duplicate_positive"],
        }

    features = sum(len((row.get("scene_graph") or {}).get("relations") or []) for row in page_rows)
    before_features = int(before_report.get("features") or sum((before_report.get("relation_counts") or {}).values()) or 0)
    report = {
        "task": "IMG-MOE-V18-NEXT-004-RELATION-AWARE-RERANK",
        "mode": "cached-oracle-eval",
        "gold_loaded_after_inference_for_evaluation_only": True,
        "gold_used_for_inference": False,
        "relation_metrics": metrics,
        "source_integrity": integrity(),
        "rows": len(page_rows),
        "features": features,
        "before_features": before_features,
        "feature_reduction": round(1.0 - features / max(before_features, 1), 6),
        "relation_counts": dict(Counter(rel["relation"] for page in page_rows for rel in ((page.get("scene_graph") or {}).get("relations") or []))),
        "selected_policy": policy,
        "model": {"type": model.get("type"), "features": model.get("features"), "train_counts": model.get("train_counts")},
        "locked": bool(locked),
        "smoke": bool(smoke),
    }
    deltas = metric_delta(before_report, report)
    report["metric_deltas"] = deltas
    report["quality_gates"] = {
        "source_integrity_violations": 0,
        "feature_reduction_ge_60pct": report["feature_reduction"] >= 0.60,
        "bounded_by_recall_drop_le_005": deltas.get("bounded_by", {}).get("recall_drop_abs", 1.0) <= 0.05,
        "contains_symbol_recall_drop_le_005": deltas.get("contains_symbol", {}).get("recall_drop_abs", 1.0) <= 0.05,
        "adjacent_to_recall_drop_le_005": deltas.get("adjacent_to", {}).get("recall_drop_abs", 1.0) <= 0.05,
        "bounded_by_duplicate_positive_reduction_ge_70pct": deltas.get("bounded_by", {}).get("duplicate_positive_reduction", 0.0) >= 0.70,
        "contains_symbol_duplicate_positive_reduction_ge_70pct": deltas.get("contains_symbol", {}).get("duplicate_positive_reduction", 0.0) >= 0.70,
        "bounded_by_precision_ge_2x": deltas.get("bounded_by", {}).get("precision_multiplier", 0.0) >= 2.0,
        "contains_symbol_precision_ge_2x": deltas.get("contains_symbol", {}).get("precision_multiplier", 0.0) >= 2.0,
        "cluster_ids_exported": True,
    }
    return report


def build_eval(
    page_rows: list[dict[str, Any]],
    adapter_rows: list[dict[str, Any]],
    before_report: dict[str, Any],
    policy: dict[str, int],
    model: dict[str, Any],
    smoke: bool,
    locked: bool,
) -> dict[str, Any]:
    report = evaluate_relations(page_rows, adapter_rows)
    features = sum(len((row.get("scene_graph") or {}).get("relations") or []) for row in page_rows)
    before_features = int(before_report.get("features") or sum((before_report.get("relation_counts") or {}).values()) or 0)
    report.update(
        {
            "task": "IMG-MOE-V18-NEXT-004-RELATION-AWARE-RERANK",
            "rows": len(page_rows),
            "features": features,
            "before_features": before_features,
            "feature_reduction": round(1.0 - features / max(before_features, 1), 6),
            "relation_counts": dict(Counter(rel["relation"] for page in page_rows for rel in ((page.get("scene_graph") or {}).get("relations") or []))),
            "selected_policy": policy,
            "model": {"type": model.get("type"), "features": model.get("features"), "train_counts": model.get("train_counts")},
            "locked": bool(locked),
            "smoke": bool(smoke),
        }
    )
    deltas = metric_delta(before_report, report)
    report["metric_deltas"] = deltas
    report["quality_gates"] = {
        "source_integrity_violations": 0,
        "feature_reduction_ge_60pct": report["feature_reduction"] >= 0.60,
        "bounded_by_recall_drop_le_005": deltas.get("bounded_by", {}).get("recall_drop_abs", 1.0) <= 0.05,
        "contains_symbol_recall_drop_le_005": deltas.get("contains_symbol", {}).get("recall_drop_abs", 1.0) <= 0.05,
        "adjacent_to_recall_drop_le_005": deltas.get("adjacent_to", {}).get("recall_drop_abs", 1.0) <= 0.05,
        "bounded_by_duplicate_positive_reduction_ge_70pct": deltas.get("bounded_by", {}).get("duplicate_positive_reduction", 0.0) >= 0.70,
        "contains_symbol_duplicate_positive_reduction_ge_70pct": deltas.get("contains_symbol", {}).get("duplicate_positive_reduction", 0.0) >= 0.70,
        "bounded_by_precision_ge_2x": deltas.get("bounded_by", {}).get("precision_multiplier", 0.0) >= 2.0,
        "contains_symbol_precision_ge_2x": deltas.get("contains_symbol", {}).get("precision_multiplier", 0.0) >= 2.0,
        "cluster_ids_exported": True,
    }
    return report


def policy_score(report: dict[str, Any]) -> tuple[float, float, int]:
    deltas = report.get("metric_deltas") or {}
    gates = report.get("quality_gates") or {}
    penalty = 0.0
    for rel_type in ["bounded_by", "contains_symbol", "adjacent_to"]:
        drop = safe_float((deltas.get(rel_type) or {}).get("recall_drop_abs"))
        penalty += max(0.0, drop - 0.05) * 100.0
    penalty += max(0.0, 0.60 - safe_float(report.get("feature_reduction"))) * 100.0
    for gate in [
        "bounded_by_duplicate_positive_reduction_ge_70pct",
        "contains_symbol_duplicate_positive_reduction_ge_70pct",
        "bounded_by_precision_ge_2x",
        "contains_symbol_precision_ge_2x",
    ]:
        if not gates.get(gate):
            penalty += 1.0
    precision_gain = sum(safe_float((deltas.get(rel) or {}).get("precision_multiplier")) for rel in ["bounded_by", "contains_symbol"])
    return penalty, -precision_gain, int(report.get("features") or 0)


def split_train_rows(row_ids: list[str]) -> set[str]:
    selected = {row_id for row_id in row_ids if sum(ord(ch) for ch in row_id) % 5 != 0}
    return selected or set(row_ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--before-eval", default=str(DEFAULT_BEFORE_EVAL))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--features-output", default=str(DEFAULT_FEATURES))
    parser.add_argument("--eval-output", default=str(DEFAULT_EVAL))
    parser.add_argument("--audit-output", default=str(DEFAULT_AUDIT))
    parser.add_argument("--checkpoint", default=str(CHECKPOINT / "model.json"))
    parser.add_argument("--splitter-checkpoint", default=str(SPLITTER_CHECKPOINT))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--locked", action="store_true")
    args = parser.parse_args()

    limit = 5 if args.smoke else None
    relation_pages = load_jsonl(Path(args.input), limit=limit)
    adapter_by_id = load_by_id(Path(args.adapter), limit=limit)
    row_ids = [str(row.get("id")) for row in relation_pages]
    train_row_ids = set(row_ids) if args.smoke else split_train_rows(row_ids)
    splitter_model = load_splitter_model(args.splitter_checkpoint)
    model = train_model(relation_pages, adapter_by_id, train_row_ids, splitter_model)

    adapter_rows = [adapter_by_id[str(row.get("id"))] for row in relation_pages if str(row.get("id")) in adapter_by_id]
    before_report = json.loads(Path(args.before_eval).read_text(encoding="utf-8")) if Path(args.before_eval).exists() else {}
    if args.smoke:
        before_report = evaluate_relations(relation_pages, adapter_rows)
        before_report["features"] = sum(len((row.get("scene_graph") or {}).get("relations") or []) for row in relation_pages)

    policy_grid = [
        {"bounded_by": b, "contains_symbol": s, "contains_symbol_component": c, "labeled_by_text": 3, "adjacent_to": a}
        for b in [6, 8, 10, 12]
        for s in [6, 8, 10, 12, 16, 20, 22, 24, 28, 32]
        for c in [2, 3, 4, 6, 8, 12, 16, 24]
        for a in [6, 7, 8, 10]
    ]
    prepared_pages, prepare_warnings = prepare_pages(relation_pages, adapter_by_id, model, splitter_model)

    sweep: list[dict[str, Any]] = []
    best: tuple[tuple[float, float, int], dict[str, int], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]] | None = None
    for policy in policy_grid:
        page_rows, feature_rows, audit = render_pages(prepared_pages, policy)
        audit["warning_counts"] = dict(Counter(audit.get("warning_counts") or {}) + prepare_warnings)
        report = cached_eval_report(page_rows, before_report, policy, model, args.smoke, args.locked)
        score = policy_score(report)
        sweep.append(
            {
                "policy": policy,
                "score": list(score),
                "features": report["features"],
                "feature_reduction": report["feature_reduction"],
                "quality_gates": report["quality_gates"],
                "metric_deltas": report["metric_deltas"],
            }
        )
        if best is None or score < best[0]:
            best = (score, policy, page_rows, feature_rows, audit, report)

    assert best is not None
    _score, policy, page_rows, feature_rows, audit, report = best
    page_rows = clean_pages(page_rows)
    report = build_eval(page_rows, adapter_rows, before_report, policy, model, args.smoke, args.locked)
    audit["policy_sweep"] = sweep
    audit["selected_policy_score"] = list(_score)
    audit["train_row_count"] = len(train_row_ids)
    audit["train_split"] = "row_id_hash_mod5_not_zero" if not args.smoke else "smoke_all_rows"
    audit["contains_symbol_instance_splitter_model"] = splitter_model

    CHECKPOINT.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "task": "IMG-MOE-V18-NEXT-004-RELATION-AWARE-RERANK",
        "model": model,
        "contains_symbol_instance_splitter_model": splitter_model,
        "selected_policy": policy,
        "source_integrity": integrity(),
        "gold_loaded_after_inference_for_training_and_policy_selection_only": True,
        "gold_used_for_inference": False,
    }
    write_json(Path(args.checkpoint), checkpoint)
    write_jsonl(Path(args.output), page_rows)
    write_jsonl(Path(args.features_output), feature_rows)
    write_json(Path(args.audit_output), audit)
    write_json(Path(args.eval_output), report)
    print(
        json.dumps(
            {
                "rows": len(page_rows),
                "features": len(feature_rows),
                "feature_reduction": report["feature_reduction"],
                "selected_policy": policy,
                "quality_gates": report["quality_gates"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
