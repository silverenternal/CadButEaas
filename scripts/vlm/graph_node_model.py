#!/usr/bin/env python3
"""Shared CadStruct graph node model utilities."""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


DEFAULT_LABELS = ["hard_wall", "door", "window"]
ORIENTATIONS = ["horizontal", "vertical", "diagonal", "rectangular", "unknown"]
PRIMITIVE_TYPES = ["line", "polyline", "bbox", "arc", "circle", "object_group", "unknown"]
BASE_NUMERIC_FEATURES = [
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "width",
    "height",
    "area",
    "cx",
    "cy",
    "length",
    "angle_degrees",
]
TOPOLOGY_NUMERIC_FEATURES = [
    "graph_degree",
    "graph_in_degree",
    "graph_out_degree",
    "relation_touches",
    "relation_opens_in_wall",
    "relation_window_in_wall",
    "relation_contained_in",
    "relation_contains",
]
LIE_NUMERIC_FEATURES = [
    "se2_cx",
    "se2_cy",
    "se2_width",
    "se2_height",
    "se2_area",
    "log_area_frac",
    "log_length_frac",
    "aspect_log",
    "radial_norm",
    "cos2_local_angle",
    "sin2_local_angle",
]
RASTER_NUMERIC_FEATURES = [
    "raster_mean",
    "raster_std",
    "raster_dark_density",
    "raster_very_dark_density",
    "raster_mid_dark_density",
    "raster_edge_density",
    "raster_edge_strong_density",
    "raster_context_dark_density",
    "raster_dark_ratio",
    "raster_context_edge_density",
    "raster_edge_ratio",
    "raster_dark_center_ratio",
    "raster_dark_border_ratio",
    "raster_dark_horizontal_balance",
    "raster_dark_vertical_balance",
]
SOURCE_NUMERIC_FEATURES = [
    "source_cvc_fp",
    "source_floorplancad",
    "source_unknown",
]
GROUP_NUMERIC_FEATURES = [
    "member_count",
    "group_width",
    "group_height",
    "group_area",
    "member_length_mean",
    "member_length_max",
    "member_length_std",
    "member_width_mean",
    "member_width_max",
    "member_width_std",
    "member_height_mean",
    "member_height_max",
    "member_height_std",
    "member_area_mean",
    "member_area_max",
    "member_area_std",
    "member_aspect_log_mean",
    "member_aspect_log_std",
    *RASTER_NUMERIC_FEATURES,
    "internal_edge_count",
    "boundary_edge_count",
    "internal_relation_touches",
    "internal_relation_opens_in_wall",
    "internal_relation_window_in_wall",
    "internal_relation_contains",
    "internal_relation_contained_in",
    "boundary_relation_touches",
    "boundary_relation_opens_in_wall",
    "boundary_relation_window_in_wall",
    "boundary_relation_contains",
    "boundary_relation_contained_in",
    "member_orientation_horizontal",
    "member_orientation_vertical",
    "member_orientation_diagonal",
    "member_orientation_rectangular",
    "member_orientation_unknown",
    "member_primitive_line",
    "member_primitive_polyline",
    "member_primitive_bbox",
    "member_primitive_arc",
    "member_primitive_circle",
    "member_primitive_unknown",
    "semantic_confidence",
    "candidate_semantic_hard_wall",
    "candidate_semantic_door",
    "candidate_semantic_window",
    "candidate_member_fraction",
    "candidate_is_singleton",
]


@dataclass
class FeatureSpec:
    labels: list[str]
    mean: list[float]
    std: list[float]
    numeric_features: list[str]
    orientations: list[str]
    primitive_types: list[str]


class NodeClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatedNodeClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
        experts: int = 3,
        gate_temperature: float = 1.0,
        top_k: int = 0,
        expert_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.expert_count = experts
        self.gate_temperature = gate_temperature
        self.top_k = top_k
        self.expert_dropout = expert_dropout
        self.gate = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, experts))
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, output_dim),
                )
                for _ in range(experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = self.routing_weights(x)
        gates = apply_expert_dropout(gates, self.expert_dropout, self.training)
        expert_logits = torch.stack([expert(x) for expert in self.experts], dim=1)
        return torch.sum(gates.unsqueeze(-1) * expert_logits, dim=1)

    def routing_weights(self, x: torch.Tensor) -> torch.Tensor:
        return route_weights(self.gate(x), self.gate_temperature, self.top_k)


class TensorRingLinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, rank: int = 4, bias: bool = True) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank
        self.input_factors = factor_pair(input_dim)
        self.output_factors = factor_pair(output_dim)
        in_a, in_b = self.input_factors
        out_a, out_b = self.output_factors
        self.core_a = nn.Parameter(torch.empty(rank, in_a, out_a, rank))
        self.core_b = nn.Parameter(torch.empty(rank, in_b, out_b, rank))
        self.bias = nn.Parameter(torch.zeros(output_dim)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        scale = 1.0 / math.sqrt(max(self.input_dim, 1))
        nn.init.uniform_(self.core_a, -scale, scale)
        nn.init.uniform_(self.core_b, -scale, scale)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -scale, scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.einsum("a i o b, b j p a -> i j o p", self.core_a, self.core_b)
        weight = weight.reshape(self.input_dim, self.output_dim)
        output = x.matmul(weight)
        if self.bias is not None:
            output = output + self.bias
        return output


class TensorRingNodeClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float, rank: int = 4) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.tr_rank = rank
        self.net = nn.Sequential(
            TensorRingLinear(input_dim, hidden_dim, rank),
            nn.GELU(),
            nn.Dropout(dropout),
            TensorRingLinear(hidden_dim, hidden_dim, rank),
            nn.GELU(),
            nn.Dropout(dropout),
            TensorRingLinear(hidden_dim, output_dim, rank),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TensorRingGatedNodeClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
        experts: int = 3,
        rank: int = 4,
        gate_temperature: float = 1.0,
        top_k: int = 0,
        expert_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.expert_count = experts
        self.tr_rank = rank
        self.gate_temperature = gate_temperature
        self.top_k = top_k
        self.expert_dropout = expert_dropout
        self.gate = nn.Sequential(TensorRingLinear(input_dim, hidden_dim, rank), nn.GELU(), nn.Linear(hidden_dim, experts))
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    TensorRingLinear(input_dim, hidden_dim, rank),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    TensorRingLinear(hidden_dim, hidden_dim, rank),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    TensorRingLinear(hidden_dim, output_dim, rank),
                )
                for _ in range(experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = self.routing_weights(x)
        gates = apply_expert_dropout(gates, self.expert_dropout, self.training)
        expert_logits = torch.stack([expert(x) for expert in self.experts], dim=1)
        return torch.sum(gates.unsqueeze(-1) * expert_logits, dim=1)

    def routing_weights(self, x: torch.Tensor) -> torch.Tensor:
        return route_weights(self.gate(x), self.gate_temperature, self.top_k)


def build_model(
    model_type: str,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    dropout: float,
    experts: int = 3,
    tr_rank: int = 4,
    gate_temperature: float = 1.0,
    top_k: int = 0,
    expert_dropout: float = 0.0,
) -> nn.Module:
    if model_type == "mlp":
        return NodeClassifier(input_dim, hidden_dim, output_dim, dropout)
    if model_type == "gated":
        return GatedNodeClassifier(
            input_dim, hidden_dim, output_dim, dropout, experts, gate_temperature, top_k, expert_dropout
        )
    if model_type == "tr_mlp":
        return TensorRingNodeClassifier(input_dim, hidden_dim, output_dim, dropout, tr_rank)
    if model_type == "tr_gated":
        return TensorRingGatedNodeClassifier(
            input_dim, hidden_dim, output_dim, dropout, experts, tr_rank, gate_temperature, top_k, expert_dropout
        )
    raise ValueError(f"unsupported model_type: {model_type}")


def route_weights(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0) -> torch.Tensor:
    temperature = max(float(temperature), 1e-6)
    weights = torch.softmax(logits / temperature, dim=-1)
    if top_k and 0 < top_k < weights.shape[-1]:
        indices = torch.topk(weights, k=top_k, dim=-1).indices
        mask = torch.zeros_like(weights).scatter_(-1, indices, 1.0)
        weights = weights * mask
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    return weights


def apply_expert_dropout(gates: torch.Tensor, dropout: float, training: bool) -> torch.Tensor:
    if not training or dropout <= 0.0:
        return gates
    dropped = F.dropout(gates, p=dropout, training=True)
    return dropped / dropped.sum(dim=-1, keepdim=True).clamp_min(1e-9)


def factor_pair(value: int) -> tuple[int, int]:
    root = int(math.sqrt(value))
    for first in range(root, 0, -1):
        if value % first == 0:
            return first, value // first
    return 1, value


def build_feature_spec(rows: list[dict[str, Any]], labels: list[str]) -> FeatureSpec:
    numeric_feature_names = feature_names_for_rows(rows)
    numeric = [numeric_features(row["features"], numeric_feature_names) for row in rows]
    cols = list(zip(*numeric))
    mean = [sum(col) / len(col) for col in cols]
    std = []
    for col, mu in zip(cols, mean):
        var = sum((value - mu) ** 2 for value in col) / len(col)
        std.append(max(math.sqrt(var), 1e-6))
    return FeatureSpec(
        labels=labels,
        mean=mean,
        std=std,
        numeric_features=numeric_feature_names,
        orientations=ORIENTATIONS,
        primitive_types=PRIMITIVE_TYPES,
    )


def tensorize(
    rows: list[dict[str, Any]], feature_spec: FeatureSpec, label_to_id: dict[str, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    xs = [encode_features(row["features"], feature_spec) for row in rows]
    ys = [label_to_id[row["label"]] for row in rows]
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.long)


def encode_features(features: dict[str, Any], spec: FeatureSpec) -> list[float]:
    values = numeric_features(features, spec.numeric_features)
    encoded = [(value - mean) / std for value, mean, std in zip(values, spec.mean, spec.std)]
    orientation = str(features.get("orientation", "unknown"))
    primitive_type = str(features.get("primitive_type", "unknown"))
    encoded.extend(1.0 if orientation == item else 0.0 for item in spec.orientations)
    encoded.extend(1.0 if primitive_type == item else 0.0 for item in spec.primitive_types)
    return encoded


def feature_names_for_rows(rows: list[dict[str, Any]]) -> list[str]:
    names = list(BASE_NUMERIC_FEATURES)
    if any(any(name in row.get("features", {}) for name in TOPOLOGY_NUMERIC_FEATURES) for row in rows):
        names.extend(TOPOLOGY_NUMERIC_FEATURES)
    if any(any(name in row.get("features", {}) for name in LIE_NUMERIC_FEATURES) for row in rows):
        names.extend(LIE_NUMERIC_FEATURES)
    has_group_features = any("member_count" in row.get("features", {}) for row in rows)
    if has_group_features:
        names.extend(GROUP_NUMERIC_FEATURES)
    elif any(any(name in row.get("features", {}) for name in RASTER_NUMERIC_FEATURES) for row in rows):
        names.extend(RASTER_NUMERIC_FEATURES)
    if any(any(name in row.get("features", {}) for name in SOURCE_NUMERIC_FEATURES) for row in rows):
        names.extend(SOURCE_NUMERIC_FEATURES)
    return names


def numeric_features(features: dict[str, Any], names: list[str] | None = None) -> list[float]:
    bbox = features.get("bbox") if isinstance(features.get("bbox"), list) else [0.0, 0.0, 0.0, 0.0]
    x1, y1, x2, y2 = [float(value or 0.0) for value in (bbox[:4] + [0.0] * 4)[:4]]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    centroid = features.get("centroid") if isinstance(features.get("centroid"), list) else [0.0, 0.0]
    cx, cy = [float(value or 0.0) for value in (centroid[:2] + [0.0] * 2)[:2]]
    values = {
        "bbox_x1": x1,
        "bbox_y1": y1,
        "bbox_x2": x2,
        "bbox_y2": y2,
        "width": width,
        "height": height,
        "area": width * height,
        "cx": cx,
        "cy": cy,
        "length": float(features.get("length", 0.0) or 0.0),
        "angle_degrees": float(features.get("angle_degrees", 0.0) or 0.0),
    }
    for name in TOPOLOGY_NUMERIC_FEATURES:
        values[name] = float(features.get(name, 0.0) or 0.0)
    for name in LIE_NUMERIC_FEATURES:
        values[name] = float(features.get(name, 0.0) or 0.0)
    for name in GROUP_NUMERIC_FEATURES:
        values[name] = float(features.get(name, 0.0) or 0.0)
    for name in SOURCE_NUMERIC_FEATURES:
        values[name] = float(features.get(name, 0.0) or 0.0)
    return [values.get(name, 0.0) for name in (names or BASE_NUMERIC_FEATURES)]


def raw_node_features(node: dict[str, Any]) -> dict[str, Any]:
    bbox = node.get("bbox") if isinstance(node.get("bbox"), list) else [0.0, 0.0, 0.0, 0.0]
    centroid = node.get("centroid") if isinstance(node.get("centroid"), list) else [0.0, 0.0]
    return {
        "primitive_type": str(node.get("primitive_type", "unknown")),
        "bbox": [float(value or 0.0) for value in (bbox[:4] + [0.0] * 4)[:4]],
        "centroid": [float(value or 0.0) for value in (centroid[:2] + [0.0] * 2)[:2]],
        "length": float(node.get("length", 0.0) or 0.0),
        "angle_degrees": float(node.get("angle_degrees", 0.0) or 0.0),
        "orientation": str(node.get("orientation", "unknown")),
    }


def graph_node_features(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]] | None = None,
    include_topology: bool = False,
    include_lie_features: bool = False,
) -> dict[int, dict[str, Any]]:
    features_by_id = {}
    for node in nodes:
        if not int_like(node.get("id")):
            continue
        node_id = int(node["id"])
        features_by_id[node_id] = raw_node_features(node)
    if include_lie_features:
        add_lie_canonical_features(features_by_id, edges or [])
    if not include_topology:
        return features_by_id

    for features in features_by_id.values():
        for name in TOPOLOGY_NUMERIC_FEATURES:
            features[name] = 0.0
    for edge in edges or []:
        if not isinstance(edge, dict) or not int_like(edge.get("source")) or not int_like(edge.get("target")):
            continue
        source = int(edge["source"])
        target = int(edge["target"])
        relation = str(edge.get("relation", "unknown"))
        source_features = features_by_id.get(source)
        target_features = features_by_id.get(target)
        if source_features is not None:
            source_features["graph_degree"] += 1.0
            source_features["graph_out_degree"] += 1.0
            add_relation_count(source_features, relation)
        if target_features is not None:
            target_features["graph_degree"] += 1.0
            target_features["graph_in_degree"] += 1.0
            add_relation_count(target_features, relation)
    return features_by_id


def add_lie_canonical_features(features_by_id: dict[int, dict[str, Any]], edges: list[dict[str, Any]]) -> None:
    if not features_by_id:
        return
    centers = {node_id: feature_center(features) for node_id, features in features_by_id.items()}
    xs = [center[0] for center in centers.values()]
    ys = [center[1] for center in centers.values()]
    bounds = graph_bounds(features_by_id)
    graph_cx = sum(xs) / len(xs)
    graph_cy = sum(ys) / len(ys)
    graph_width = max(bounds[2] - bounds[0], max(xs) - min(xs), 1e-6)
    graph_height = max(bounds[3] - bounds[1], max(ys) - min(ys), 1e-6)
    graph_scale = max(math.sqrt(graph_width * graph_width + graph_height * graph_height), 1e-6)
    theta = dominant_graph_angle(centers, edges)
    cos_t = math.cos(-theta)
    sin_t = math.sin(-theta)
    graph_area = max(graph_width * graph_height, 1e-6)

    for features in features_by_id.values():
        cx, cy = feature_center(features)
        dx = (cx - graph_cx) / graph_scale
        dy = (cy - graph_cy) / graph_scale
        se2_cx = cos_t * dx - sin_t * dy
        se2_cy = sin_t * dx + cos_t * dy
        bbox = features.get("bbox") if isinstance(features.get("bbox"), list) else [0.0, 0.0, 0.0, 0.0]
        width = max(0.0, float(bbox[2]) - float(bbox[0]))
        height = max(0.0, float(bbox[3]) - float(bbox[1]))
        length = float(features.get("length", 0.0) or 0.0)
        angle = math.radians(float(features.get("angle_degrees", 0.0) or 0.0)) - theta
        features.update(
            {
                "se2_cx": se2_cx,
                "se2_cy": se2_cy,
                "se2_width": width / graph_scale,
                "se2_height": height / graph_scale,
                "se2_area": (width * height) / graph_area,
                "log_area_frac": math.log1p((width * height) / graph_area),
                "log_length_frac": math.log1p(length / graph_scale),
                "aspect_log": math.log((width + 1e-6) / (height + 1e-6)),
                "radial_norm": math.sqrt(dx * dx + dy * dy),
                "cos2_local_angle": math.cos(2.0 * angle),
                "sin2_local_angle": math.sin(2.0 * angle),
            }
        )


def graph_bounds(features_by_id: dict[int, dict[str, Any]]) -> tuple[float, float, float, float]:
    x1s = []
    y1s = []
    x2s = []
    y2s = []
    for features in features_by_id.values():
        bbox = features.get("bbox") if isinstance(features.get("bbox"), list) else None
        if not bbox or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = [float(value or 0.0) for value in bbox[:4]]
        x1s.append(min(x1, x2))
        y1s.append(min(y1, y2))
        x2s.append(max(x1, x2))
        y2s.append(max(y1, y2))
    if not x1s:
        return (0.0, 0.0, 1e-6, 1e-6)
    return (min(x1s), min(y1s), max(x2s), max(y2s))


def dominant_graph_angle(centers: dict[int, tuple[float, float]], edges: list[dict[str, Any]]) -> float:
    cos_sum = 0.0
    sin_sum = 0.0
    for edge in edges:
        if not isinstance(edge, dict) or not int_like(edge.get("source")) or not int_like(edge.get("target")):
            continue
        source = centers.get(int(edge["source"]))
        target = centers.get(int(edge["target"]))
        if source is None or target is None:
            continue
        dx = target[0] - source[0]
        dy = target[1] - source[1]
        length = math.sqrt(dx * dx + dy * dy)
        if length <= 1e-6:
            continue
        angle = math.atan2(dy, dx)
        cos_sum += math.cos(2.0 * angle) * length
        sin_sum += math.sin(2.0 * angle) * length
    if abs(cos_sum) <= 1e-9 and abs(sin_sum) <= 1e-9:
        return 0.0
    return 0.5 * math.atan2(sin_sum, cos_sum)


def feature_center(features: dict[str, Any]) -> tuple[float, float]:
    centroid = features.get("centroid") if isinstance(features.get("centroid"), list) else [0.0, 0.0]
    values = (centroid[:2] + [0.0] * 2)[:2]
    return float(values[0] or 0.0), float(values[1] or 0.0)


def add_relation_count(features: dict[str, Any], relation: str) -> None:
    if relation in {"touches", "opens_in_wall", "window_in_wall", "contained_in", "contains"}:
        features[f"relation_{relation}"] += 1.0


def class_weight_tensor(labels: torch.Tensor, class_count: int) -> torch.Tensor:
    counts = torch.bincount(labels.detach().cpu(), minlength=class_count).float()
    weights = counts.sum() / counts.clamp_min(1.0)
    weights[counts == 0] = 0.0
    return weights / weights[weights > 0].mean()


def int_like(value: Any) -> bool:
    try:
        int(value)
        return True
    except (TypeError, ValueError):
        return False


def evaluate_model(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    labels: list[str],
    batch_size: int | None = None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    model.eval()
    model_device = device or next(model.parameters()).device
    tile_size = batch_size or int(x.shape[0]) or 1
    predictions = []
    probability_chunks = []
    with torch.inference_mode():
        for start in range(0, int(x.shape[0]), tile_size):
            batch_x = x[start : start + tile_size].to(model_device, non_blocking=True)
            probs = torch.softmax(model(batch_x), dim=-1).detach().cpu()
            probability_chunks.append(probs)
            predictions.append(probs.argmax(dim=-1))
    pred = torch.cat(predictions, dim=0) if predictions else torch.empty(0, dtype=torch.long)
    probs = torch.cat(probability_chunks, dim=0) if probability_chunks else torch.empty(0, len(labels))
    y_cpu = y.detach().cpu()
    confusion = torch.zeros((len(labels), len(labels)), dtype=torch.long)
    for target, output in zip(y_cpu, pred):
        confusion[int(target), int(output)] += 1
    correct = int((pred == y_cpu).sum().detach().cpu())
    total = int(y_cpu.numel())
    per_label = {}
    f1s = []
    for index, label in enumerate(labels):
        tp = int(confusion[index, index].detach().cpu())
        fp = int(confusion[:, index].sum().detach().cpu()) - tp
        fn = int(confusion[index, :].sum().detach().cpu()) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        per_label[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": int(confusion[index, :].sum()),
        }
    return {
        "accuracy": round(correct / total, 6) if total else 0.0,
        "macro_f1": round(sum(f1s) / len(f1s), 6) if f1s else 0.0,
        "probability_r2": probability_r2(probs, y_cpu, len(labels)),
        "per_label_r2": per_label_probability_r2(probs, y_cpu, labels),
        "per_label": per_label,
        "confusion": confusion.tolist(),
    }


def probability_r2(probs: torch.Tensor, y: torch.Tensor, class_count: int) -> float:
    if probs.numel() == 0 or y.numel() == 0:
        return 0.0
    target = F.one_hot(y.to(torch.long), num_classes=class_count).to(dtype=probs.dtype)
    residual = torch.sum((target - probs) ** 2)
    centered = target - target.mean()
    total = torch.sum(centered**2)
    if float(total) <= 1e-12:
        return 0.0
    return round(float(1.0 - residual / total), 6)


def per_label_probability_r2(probs: torch.Tensor, y: torch.Tensor, labels: list[str]) -> dict[str, float]:
    if probs.numel() == 0 or y.numel() == 0:
        return {label: 0.0 for label in labels}
    target = F.one_hot(y.to(torch.long), num_classes=len(labels)).to(dtype=probs.dtype)
    output = {}
    for index, label in enumerate(labels):
        residual = torch.sum((target[:, index] - probs[:, index]) ** 2)
        centered = target[:, index] - target[:, index].mean()
        total = torch.sum(centered**2)
        output[label] = round(float(1.0 - residual / total), 6) if float(total) > 1e-12 else 0.0
    return output


def routing_summary(
    model: nn.Module, x: torch.Tensor, batch_size: int | None = None, device: torch.device | str | None = None
) -> dict[str, Any] | None:
    if not hasattr(model, "routing_weights"):
        return None
    model.eval()
    model_device = device or next(model.parameters()).device
    tile_size = batch_size or int(x.shape[0]) or 1
    chunks = []
    with torch.inference_mode():
        for start in range(0, int(x.shape[0]), tile_size):
            batch_x = x[start : start + tile_size].to(model_device, non_blocking=True)
            chunks.append(model.routing_weights(batch_x).detach().cpu())
    weights = torch.cat(chunks, dim=0) if chunks else torch.empty(0, 0)
    mean = weights.mean(dim=0)
    hard = weights.argmax(dim=-1)
    counts = torch.bincount(hard, minlength=weights.shape[1])
    total = max(int(hard.numel()), 1)
    return {
        "mean_gate_weight": [round(float(value), 6) for value in mean],
        "hard_route_fraction": [round(float(value) / total, 6) for value in counts],
    }


def routing_balance_loss(model: nn.Module, x: torch.Tensor) -> torch.Tensor | None:
    if not hasattr(model, "routing_weights"):
        return None
    weights = model.routing_weights(x)
    mean = weights.mean(dim=0)
    return mean.numel() * torch.sum(mean * mean) - 1.0


def save_checkpoint(path: Path, model: nn.Module, feature_spec: FeatureSpec, args: argparse.Namespace, metrics: dict[str, Any]) -> None:
    model_type = getattr(args, "model_type", "mlp")
    experts = int(getattr(args, "experts", 3))
    tr_rank = int(getattr(args, "tr_rank", 4))
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_spec": asdict(feature_spec),
            "model_config": {
                "input_dim": int(getattr(model, "input_dim", next(model.parameters()).shape[1])),
                "hidden_dim": args.hidden_dim,
                "output_dim": len(feature_spec.labels),
                "dropout": args.dropout,
                "model_type": model_type,
                "experts": experts,
                "tr_rank": tr_rank,
                "routing_balance_weight": float(getattr(args, "routing_balance_weight", 0.0)),
                "gate_temperature": float(getattr(args, "gate_temperature", 1.0)),
                "top_k": int(getattr(args, "top_k", 0)),
                "expert_dropout": float(getattr(args, "expert_dropout", 0.0)),
            },
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(path: str | Path, device: str | torch.device) -> tuple[nn.Module, FeatureSpec, list[str], dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    feature_spec = FeatureSpec(**checkpoint["feature_spec"])
    labels = feature_spec.labels
    model_config = checkpoint["model_config"]
    model = build_model(
        model_type=model_config.get("model_type", "mlp"),
        input_dim=model_config["input_dim"],
        hidden_dim=model_config["hidden_dim"],
        output_dim=model_config["output_dim"],
        dropout=model_config["dropout"],
        experts=model_config.get("experts", 3),
        tr_rank=model_config.get("tr_rank", 4),
        gate_temperature=model_config.get("gate_temperature", 1.0),
        top_k=model_config.get("top_k", 0),
        expert_dropout=model_config.get("expert_dropout", 0.0),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, feature_spec, labels, checkpoint
