#!/usr/bin/env python3
"""Train a line-token panoptic component MoE expert.

This is the model-side PQ/RQ upgrade over the semantic+embedding line-token
expert: it predicts component queries with class logits and primitive membership
masks, trained by semantic CE plus component class/BCE/Dice losses.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue
from typing import Any, Iterable
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.floorplancad_train_line_token_transformer_moe import (  # noqa: E402
    BASE_FEATURES,
    IGNORE_LABEL,
    POSITION_ENCODING_VERSION,
    POSITION_MAX_FREQUENCY_LOG2,
    import_torch,
    iter_jsonl,
    parse_float,
    parse_int,
    record_to_arrays,
    rel,
    semantic_class_weights,
    sinusoidal_position,
    utc_now,
    write_json,
)
from experiments.floorplancad_target_schema_v2 import (  # noqa: E402
    SCHEMA_VERSION as TARGET_SCHEMA_V2,
    load_target_schema_v2,
    semantic_primitive_weights,
    task_availability,
)
from experiments.floorplancad_target_schema_v3 import (  # noqa: E402
    INPUT_SCHEMA_VERSION as INPUT_SCHEMA_V3,
    SCHEMA_VERSION as TARGET_SCHEMA_V3,
    V3_FEATURE_NAMES,
    load_target_schema_v3,
)
from experiments.floorplancad_target_schema_v4 import (  # noqa: E402
    INPUT_SCHEMA_VERSION as INPUT_SCHEMA_V4,
    INPUT_SCHEMA_VERSION_V5,
    INPUT_SCHEMA_VERSION_V6,
    SCHEMA_VERSION as TARGET_SCHEMA_V4,
    SCHEMA_VERSION_V5 as TARGET_SCHEMA_V5,
    SCHEMA_VERSION_V6 as TARGET_SCHEMA_V6,
    V4_FEATURE_NAMES,
    V5_FEATURE_NAMES,
    V6_FEATURE_NAMES,
    load_target_schema_v4,
)
from experiments.floorplancad_window_identity_embedding import (  # noqa: E402
    QueryIdentityHead,
    adjacent_window_identity_loss,
)
from experiments.floorplancad_sq_rq_cross_attention import SqRqCrossAttention  # noqa: E402
from experiments.floorplancad_query_ownership import (  # noqa: E402
    DEFAULT_MIN_QUERY_SCORE,
    OWNERSHIP_VERSION,
    ownership_config,
    ownership_cross_entropy,
    ownership_mask_consistency_loss,
    ownership_targets,
)
from experiments.floorplancad_multitask_gradient_control import (  # noqa: E402
    OBJECTIVE_ABI_FIELDS as GRADIENT_CONTROL_ABI_FIELDS,
    assign_multitask_gradients,
)
from experiments.floorplancad_panoptic_matching import (  # noqa: E402
    assignment_cost_diagnostics,
    component_assignment_cost,
    greedy_assignment_gpu,
    greedy_assignment_gpu_tensor,
    linear_sum_assignment_fallback,
    match_component_queries,
)
from experiments.floorplancad_panoptic_protocol import (  # noqa: E402
    PANOPTIC_OBJECT_NORMALIZATION_THRESHOLD,
    PANOPTIC_QUALITY_MASK_THRESHOLD,
    PANOPTIC_QUALITY_OBJECTIVE_VERSION,
    STUFF_LABELS,
    THING_LABELS,
)
from experiments.floorplancad_panoptic_runtime_config import DEFAULT_RUNTIME_PROFILE  # noqa: E402
from experiments.floorplancad_panoptic_scoring import (  # noqa: E402
    query_mask_objectness_scores,
    rq_sq_quality_deployment_scores,
)

DEFAULT_TRAIN = DEFAULT_RUNTIME_PROFILE.train_path(ROOT)
DEFAULT_VAL = DEFAULT_RUNTIME_PROFILE.val_path(ROOT)
DEFAULT_MODEL = ROOT / "reports/vlm/floorplancad_line_token_panoptic_moe/panoptic_component_moe.pt"
DEFAULT_REPORT = ROOT / "results/floorplancad_line_token_panoptic_moe_train.json"
DEFAULT_BOTTLENECK_LEDGER = ROOT / "results/floorplancad_true_moe_pq95_runtime_locked_full433_adapter_bottleneck_ledger.json"
ONTOLOGY_PATH = ROOT / "configs/floorplancad_semantic_map.json"
PANOPTIC_CHECKPOINT_ABI_VERSION = "floorplancad_panoptic_checkpoint_abi_v2"
PANOPTIC_QUALITY_HEAD_VERSION = "independent_query_decoded_mask_iou_v3"
PANOPTIC_IDENTITY_HEAD_VERSION = "normalized_query_identity_embedding_v1"
PANOPTIC_IDENTITY_DIM = 32
PANOPTIC_GEOMETRY_DECODER_VERSION = "thing_query_relative_multiscale_v4_padding_safe_all_segment_neighbors"
PANOPTIC_GEOMETRY_CHECKPOINT_ABI_VERSION = "floorplancad_panoptic_checkpoint_abi_v3_geometry_v2"
PANOPTIC_SQ_RQ_VERSION = "prediction_only_cross_attention_v7_persisted_deployment_state"
PANOPTIC_SQ_RQ_DEPLOYMENT_VERSION = "sq_rq_fail_closed_deployment_v1"
PANOPTIC_SQ_RQ_CHECKPOINT_ABI_VERSION = "floorplancad_panoptic_checkpoint_abi_v4_geometry_v2_sq_rq"
PANOPTIC_OWNERSHIP_CHECKPOINT_ABI_VERSION = "floorplancad_panoptic_checkpoint_abi_v5_geometry_v2_sq_rq_ownership"
PANOPTIC_GRADIENT_CONTROL_CHECKPOINT_ABI_VERSION = "floorplancad_panoptic_checkpoint_abi_v6_geometry_v2_sq_rq_ownership_pcgrad"
PANOPTIC_GRADIENT_CONTROL_VERSION = "deterministic_pcgrad_v3_shared_sq_rq_cross_attention"
PANOPTIC_SPARSE_ROUTER_VERSION = "prediction_only_topk_router_v2_typed_branch"
PANOPTIC_SPARSE_ROUTER_CHECKPOINT_ABI_VERSION = "floorplancad_panoptic_checkpoint_abi_v9_semantic35_typed_branch_router"
PANOPTIC_SPARSE_ROUTER_SCHEMA_VERSION = "floorplancad_line_token_panoptic_moe_checkpoint_v9_semantic35_typed_branch_router"
PANOPTIC_LEGACY_SPARSE_ROUTER_SCHEMA_VERSION = "floorplancad_line_token_panoptic_moe_checkpoint_v7_learned_sparse_router"
PANOPTIC_SEGMENT_INPUT_PROTOCOL_VERSION = "floorplancad_panoptic_input_v3_lossless_segments"
PANOPTIC_V4_SEGMENT_INPUT_PROTOCOL_VERSION = "floorplancad_panoptic_input_v4_raw_semantic_segments"
MODEL_FEATURE_NAMES = V3_FEATURE_NAMES
V4_MODEL_FEATURE_NAMES = V4_FEATURE_NAMES
V5_MODEL_FEATURE_NAMES = V5_FEATURE_NAMES
V6_MODEL_FEATURE_NAMES = V6_FEATURE_NAMES
FAMILY_LABELS = {
    "doors_windows": tuple(range(0, 10)),
    "furniture": (10, 11, 12, 13, 14, 15, 16),
    "appliances_plumbing": (17, 18, 19, 20, 21, 22, 23, 24, 25, 26),
    "vertical_transport": (27, 28, 29),
    "stuff_boundary": STUFF_LABELS,
}
FAMILY_NAMES = tuple(FAMILY_LABELS)
FURNITURE_RECALL_DIAGNOSTIC_THRESHOLDS = (0.05, 0.10, 0.15, 0.20)
LABEL_TO_FAMILY = {
    label: family
    for family, labels in FAMILY_LABELS.items()
    for label in labels
}
CLASS_NAMES = (
    "single door", "double door", "sliding door", "folding door", "revolving door", "rolling door",
    "window", "bay window", "blind window", "opening symbol", "sofa", "bed", "chair", "table",
    "TV cabinet", "Wardrobe", "cabinet", "gas stove", "sink", "refrigerator", "airconditioner",
    "bath", "bath tub", "washing machine", "squat toilet", "urinal", "toilet", "stairs",
    "elevator", "escalator", "row chairs", "parking spot", "wall", "curtain wall", "railing",
)


def label_family(label: int) -> str:
    return LABEL_TO_FAMILY.get(int(label), "ignore_or_unknown")


def label_family_index(label: int) -> int | None:
    family = LABEL_TO_FAMILY.get(int(label))
    if family is None:
        return None
    return FAMILY_NAMES.index(family)


def diagnostic_threshold_key(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "p")


LOSS_EXPERT_GROUPS = {
    "sq_semantic": ("semantic",),
    "rq_admission": ("query", "query_objectness", "rq_admission_hard_recall", "component_seed"),
    "mask_shape": ("mask", "mask_hard_recall_floor"),
    "quality_deployment": ("quality_calibration", "quality_deployment_floor"),
    "route_control": ("route_classification",),
    "topology_merge": ("ownership", "geometry_aux", "content_anchor", "identity", "offset_vote", "affinity"),
    "teacher_aux": ("teacher", "candidate_mask_prior"),
}


def parse_int_set_csv(value: str | None) -> set[int]:
    if value is None or not str(value).strip():
        return set()
    labels: set[int] = set()
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        labels.add(int(item))
    return labels


def parse_family_set_csv(value: str | None) -> set[str]:
    if value is None or not str(value).strip():
        return set()
    families: set[str] = set()
    for item in str(value).split(","):
        family = item.strip()
        if not family:
            continue
        if family not in FAMILY_LABELS:
            raise ValueError(f"unknown family in hard recall curriculum: {family}")
        families.add(family)
    return families


def input_feature_schema_from_args(args: argparse.Namespace) -> str:
    schema = str(getattr(args, "input_feature_schema", "v3"))
    if bool(getattr(args, "require_target_schema_v4", False)):
        schema = "v4"
    if bool(getattr(args, "require_target_schema_v5", False)):
        schema = "v5"
    if bool(getattr(args, "require_target_schema_v6", False)):
        schema = "v6"
    if schema not in {"v3", "v4", "v5", "v6"}:
        raise ValueError(f"unsupported input feature schema: {schema}")
    return schema


INPUT_FEATURE_SCHEMA_ORDER = {"v3": 3, "v4": 4, "v5": 5, "v6": 6}
OUTPUT_PROTOCOL_CLAIM_PATH_FIELDS = (
    "model_output",
    "last_model_output",
    "report",
    "diagnostic_checkpoint_dir",
    "checkpoint_archive_dir",
    "final_instance_gate_checkpoint",
    "final_instance_gate_report",
)


def protocol_schema_claims_from_path(path: Path | None) -> set[str]:
    if path is None:
        return set()
    text = path.as_posix().lower()
    return {
        schema
        for schema in INPUT_FEATURE_SCHEMA_ORDER
        if re.search(rf"(?<![a-z0-9]){schema}(?![a-z0-9])", text)
    }


def output_protocol_claim_blockers(args: argparse.Namespace, input_feature_schema: str) -> list[str]:
    if bool(getattr(args, "allow_output_protocol_name_mismatch", False)):
        return []
    input_order = INPUT_FEATURE_SCHEMA_ORDER[input_feature_schema]
    blockers: list[str] = []
    for field in OUTPUT_PROTOCOL_CLAIM_PATH_FIELDS:
        path = getattr(args, field, None)
        higher_claims = sorted(
            schema
            for schema in protocol_schema_claims_from_path(path)
            if INPUT_FEATURE_SCHEMA_ORDER[schema] > input_order
        )
        if higher_claims:
            claims = ",".join(higher_claims)
            blockers.append(
                f"{field}_claims_{claims}_with_input_{input_feature_schema}:{rel(path)}"
            )
    return blockers


def model_feature_names_for_schema(schema: str) -> tuple[str, ...]:
    if schema == "v3":
        return MODEL_FEATURE_NAMES
    if schema == "v4":
        return V4_MODEL_FEATURE_NAMES
    if schema == "v5":
        return V5_MODEL_FEATURE_NAMES
    if schema == "v6":
        return V6_MODEL_FEATURE_NAMES
    raise ValueError(f"unsupported input feature schema: {schema}")


def input_protocol_for_schema(schema: str, *, content_seeded_queries: bool = False) -> dict[str, Any]:
    if schema == "v3":
        return {
            "version": PANOPTIC_SEGMENT_INPUT_PROTOCOL_VERSION,
            "target_schema_version": TARGET_SCHEMA_V3,
            "input_schema_version": INPUT_SCHEMA_V3,
            "segment_features": True,
            "content_seeded_queries": bool(content_seeded_queries),
        }
    if schema == "v4":
        return {
            "version": PANOPTIC_V4_SEGMENT_INPUT_PROTOCOL_VERSION,
            "target_schema_version": TARGET_SCHEMA_V4,
            "input_schema_version": INPUT_SCHEMA_V4,
            "segment_features": True,
            "content_seeded_queries": bool(content_seeded_queries),
        }
    if schema == "v5":
        return {
            "version": "floorplancad_panoptic_input_v5_structural_segments",
            "target_schema_version": TARGET_SCHEMA_V5,
            "input_schema_version": INPUT_SCHEMA_VERSION_V5,
            "segment_features": True,
            "content_seeded_queries": bool(content_seeded_queries),
        }
    if schema == "v6":
        return {
            "version": "floorplancad_panoptic_input_v6_structural_relation_segments",
            "target_schema_version": TARGET_SCHEMA_V6,
            "input_schema_version": INPUT_SCHEMA_VERSION_V6,
            "segment_features": True,
            "content_seeded_queries": bool(content_seeded_queries),
        }
    raise ValueError(f"unsupported input feature schema: {schema}")


def input_protocol_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return input_protocol_for_schema(
        input_feature_schema_from_args(args),
        content_seeded_queries=bool(getattr(args, "content_seeded_queries", False)),
    )


def feature_names_for_input_protocol(input_protocol: dict[str, Any] | None) -> tuple[str, ...]:
    if input_protocol is None:
        return BASE_FEATURES
    target_schema = input_protocol.get("target_schema_version")
    if target_schema == TARGET_SCHEMA_V3:
        return MODEL_FEATURE_NAMES
    if target_schema == TARGET_SCHEMA_V4:
        return V4_MODEL_FEATURE_NAMES
    if target_schema == TARGET_SCHEMA_V5:
        return V5_MODEL_FEATURE_NAMES
    if target_schema == TARGET_SCHEMA_V6:
        return V6_MODEL_FEATURE_NAMES
    raise ValueError(f"unsupported checkpoint target schema: {target_schema}")


def validate_segment_input_protocol(input_protocol: dict[str, Any]) -> str:
    if input_protocol.get("segment_features") is not True:
        raise ValueError("checkpoint input protocol must require per-primitive segment features")
    normalized = dict(input_protocol)
    normalized.setdefault("content_seeded_queries", False)
    for schema in ("v3", "v4", "v5", "v6"):
        expected = input_protocol_for_schema(
            schema,
            content_seeded_queries=bool(normalized.get("content_seeded_queries", False)),
        )
        if normalized == expected:
            return schema
    raise ValueError(f"unsupported checkpoint input protocol: {input_protocol}")


def _balanced_factors(value: int) -> list[int]:
    """Small factors for a tensor-ring mode decomposition."""
    if value <= 1:
        return [1]
    factors: list[int] = []
    remaining = int(value)
    divisor = 2
    while divisor * divisor <= remaining:
        while remaining % divisor == 0:
            factors.append(divisor)
            remaining //= divisor
        divisor += 1
    if remaining > 1:
        factors.append(remaining)
    return factors or [value]


def morton_spatial_order(torch: Any, centers: Any, grid_size: int) -> Any:
    """Return a deterministic, input-order-independent locality order for 2-D tokens."""
    if centers.ndim != 3 or centers.shape[-1] != 2:
        raise ValueError("centers must have shape [batch, tokens, 2]")
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    coordinates = torch.floor(centers.clamp(0.0, 1.0 - 1e-6) * grid_size).long()
    x_coordinate, y_coordinate = coordinates.unbind(dim=-1)
    morton_code = torch.zeros_like(x_coordinate)
    for bit in range(max(1, int(math.ceil(math.log2(grid_size))))):
        morton_code |= ((x_coordinate >> bit) & 1) << (2 * bit)
        morton_code |= ((y_coordinate >> bit) & 1) << (2 * bit + 1)
    return torch.argsort(morton_code, dim=1, stable=True)


def endpoint_pairwise_min_distance(torch: Any, left: Any, right: Any) -> Any:
    """Minimum endpoint distance, invariant to either segment's orientation."""
    if left.shape[-2:] != (2, 2) or right.shape[-2:] != (2, 2):
        raise ValueError("segments must end in [endpoint, xy] = [2, 2]")
    distance = torch.linalg.vector_norm(left[..., :, None, :] - right[..., None, :, :], dim=-1)
    return distance.amin(dim=-1).amin(dim=-1)


def sparse_endpoint_neighbor_graph(
    torch: Any,
    features: Any,
    *,
    neighbors: int,
    valid_mask: Any | None = None,
) -> tuple[Any, Any, Any]:
    """Build a bounded Morton-local endpoint graph without an N-by-N tensor."""
    if features.ndim != 3 or features.shape[-1] < 6:
        raise ValueError("features must be [batch, tokens, >=6]")
    batch, tokens, _ = features.shape
    if tokens < 2 or int(neighbors) < 1:
        empty_indices = torch.empty((batch, tokens, 0), dtype=torch.long, device=features.device)
        empty_valid = torch.empty((batch, tokens, 0), dtype=torch.bool, device=features.device)
        empty_distance = features.new_empty((batch, tokens, 0))
        return empty_indices, empty_valid, empty_distance
    valid = (
        valid_mask.to(torch.bool)
        if valid_mask is not None
        else torch.ones((batch, tokens), dtype=torch.bool, device=features.device)
    )
    if valid.shape != (batch, tokens):
        raise ValueError("valid_mask must be [batch, tokens]")
    count = min(int(neighbors), tokens - 1)
    centers = features[..., 4:6].clamp(0.0, 1.0)
    endpoints = features[..., :4].reshape(batch, tokens, 2, 2)
    candidate_radius = max(count * 2, 8)
    spatial_grid = max(int(math.sqrt(tokens)) * 2, 2)
    spatial_order = morton_spatial_order(torch, centers, spatial_grid)
    ranks = torch.empty_like(spatial_order)
    ranks.scatter_(1, spatial_order, torch.arange(tokens, device=features.device).expand_as(spatial_order))
    offsets = torch.arange(-candidate_radius, candidate_radius + 1, device=features.device)
    raw_candidate_ranks = ranks[..., None] + offsets
    candidate_in_range = (raw_candidate_ranks >= 0) & (raw_candidate_ranks < tokens)
    candidate_ranks = raw_candidate_ranks.clamp(0, tokens - 1)
    candidate_indices = torch.gather(
        spatial_order[:, None, :].expand(-1, tokens, -1), 2, candidate_ranks
    )
    batch_indices = torch.arange(batch, device=features.device)[:, None, None]
    candidate_endpoints = endpoints[batch_indices, candidate_indices]
    candidate_distance = endpoint_pairwise_min_distance(
        torch, endpoints[:, :, None], candidate_endpoints
    )
    candidate_valid = valid[batch_indices, candidate_indices]
    self_indices = torch.arange(tokens, device=features.device)[None, :, None]
    candidate_valid &= candidate_in_range & (candidate_indices != self_indices)
    candidate_distance = candidate_distance.masked_fill(~(valid[..., None] & candidate_valid), float("inf"))
    neighbor_distance, neighbor_slot = torch.topk(candidate_distance, count, dim=-1, largest=False)
    neighbor_indices = torch.gather(candidate_indices, 2, neighbor_slot)
    neighbor_valid = torch.isfinite(neighbor_distance)
    return neighbor_indices, neighbor_valid, neighbor_distance


def sparse_all_segment_neighbor_graph(
    torch: Any,
    features: Any,
    segment_features: Any,
    segment_valid: Any,
    *,
    neighbors: int,
    valid_mask: Any | None = None,
    max_segments: int = 32,
) -> tuple[Any, Any, Any]:
    """Build primitive neighbors from all retained sampled-segment endpoints."""
    if segment_features.ndim != 4 or segment_features.shape[:2] != features.shape[:2] or segment_features.shape[-1] < 4:
        raise ValueError("segment_features must be [batch, tokens, segments, >=4]")
    if segment_valid.shape != segment_features.shape[:3]:
        raise ValueError("segment_valid must be [batch, tokens, segments]")
    batch, tokens, _ = features.shape
    if tokens < 2 or int(neighbors) < 1:
        empty_indices = torch.empty((batch, tokens, 0), dtype=torch.long, device=features.device)
        empty_valid = torch.empty((batch, tokens, 0), dtype=torch.bool, device=features.device)
        empty_distance = features.new_empty((batch, tokens, 0))
        return empty_indices, empty_valid, empty_distance
    segment_cap = min(max(1, int(max_segments)), segment_features.shape[2])
    segment_features = segment_features[:, :, :segment_cap]
    segment_valid = segment_valid[:, :, :segment_cap].to(torch.bool)
    primitive_valid = (
        valid_mask.to(torch.bool)
        if valid_mask is not None
        else torch.ones((batch, tokens), dtype=torch.bool, device=features.device)
    )
    count = min(int(neighbors), tokens - 1)
    left_endpoints = segment_features[..., :4].reshape(batch, tokens, segment_cap, 2, 2)
    best_distance = features.new_full((batch, tokens, count), float("inf"))
    best_indices = torch.zeros((batch, tokens, count), dtype=torch.long, device=features.device)
    chunk = max(1, min(tokens, 16 if segment_cap >= 16 else 32))
    candidate_positions = torch.arange(tokens, device=features.device)
    left_valid = primitive_valid & segment_valid.any(dim=-1)
    for start in range(0, tokens, chunk):
        stop = min(start + chunk, tokens)
        candidate_indices = candidate_positions[start:stop]
        right_endpoints = left_endpoints[:, start:stop]
        right_segment_valid = segment_valid[:, start:stop]
        distances = torch.linalg.vector_norm(
            left_endpoints[:, :, None, :, :, None, None, :]
            - right_endpoints[:, None, :, None, None, :, :, :],
            dim=-1,
        )
        valid_segment_pairs = (
            segment_valid[:, :, None, :, None, None, None]
            & right_segment_valid[:, None, :, None, None, :, None]
        )
        distances = distances.masked_fill(~valid_segment_pairs, float("inf"))
        distances = distances.amin(dim=(-1, -2, -3, -4))
        right_valid = primitive_valid[:, start:stop] & right_segment_valid.any(dim=-1)
        candidate_valid = (
            left_valid[:, :, None]
            & right_valid[:, None, :]
            & (candidate_indices[None, None, :] != torch.arange(tokens, device=features.device)[None, :, None])
        )
        distances = distances.masked_fill(~candidate_valid, float("inf"))
        merged_distance = torch.cat([best_distance, distances], dim=-1)
        merged_indices = torch.cat([
            best_indices,
            candidate_indices.view(1, 1, -1).expand(batch, tokens, -1),
        ], dim=-1)
        best_distance, order = torch.topk(merged_distance, count, dim=-1, largest=False)
        best_indices = torch.gather(merged_indices, 2, order)

    neighbor_distance = best_distance
    neighbor_indices = best_indices
    neighbor_valid = torch.isfinite(neighbor_distance)
    return neighbor_indices, neighbor_valid, neighbor_distance


def sparse_router_config(*, enabled: bool, hidden_dim: int, num_experts: int, top_k: int, temperature: float, typed_branch_routers: bool = False, branch_num_experts: int = 2, branch_top_k: int = 1, branch_capacity_factor: float = 1.25, branch_dropless: bool = False) -> dict[str, Any]:
    if typed_branch_routers and not enabled:
        raise ValueError("typed branch routers require the learned sparse router ABI")
    if int(num_experts) < 2:
        raise ValueError("sparse router requires at least two experts")
    if not 1 <= int(top_k) <= int(num_experts):
        raise ValueError("sparse router top_k must be in [1, num_experts]")
    if float(temperature) <= 0.0:
        raise ValueError("sparse router temperature must be positive")
    if int(branch_num_experts) < 2:
        raise ValueError("branch router requires at least two experts")
    if not 1 <= int(branch_top_k) <= int(branch_num_experts):
        raise ValueError("branch router top_k must be in [1, branch_num_experts]")
    if float(branch_capacity_factor) < 1.0:
        raise ValueError("branch capacity factor must be at least 1.0")
    expert_manifest = [
        {
            "expert_id": index,
            "role": "shared_token_backbone",
            "typed_capabilities": ["semantic", "rq", "sq", "bridge"],
            "branch_specific": False,
        }
        for index in range(int(num_experts))
    ]
    return {
        "version": PANOPTIC_SPARSE_ROUTER_VERSION,
        "abi_generation": "v8_typed_branch_router_generation_1",
        "enabled": bool(enabled),
        "hidden_dim": int(hidden_dim),
        "num_experts": int(num_experts),
        "top_k": int(top_k),
        "temperature": float(temperature),
        "expert_manifest": expert_manifest,
        "route_trace_schema": "token_topk_v1",
        "branch_router_status": "typed_branch_routers_enabled" if typed_branch_routers else "shared_router_only_pending_typed_branch_routers",
        "typed_branch_routers": bool(typed_branch_routers),
        "branch_num_experts": int(branch_num_experts),
        "branch_top_k": int(branch_top_k),
        "branch_capacity_factor": float(branch_capacity_factor),
        "branch_dropless": bool(branch_dropless),
        "branch_names": ["semantic", "rq", "sq", "bridge"],
        "expert_sharing_topology": "shared_encoder_pool_plus_private_task_adapters_v1",
        "branch_specialization": "private_pool_low_entropy_with_shared_pool_load_balance_v1",
        "input_source": "predicted_encoder_token_features",
        "forbidden_context": ["gt_labels", "gt_masks", "page_instance_id", "matched_target_indices"],
        "load_balance_diagnostic": "mean_router_probability_cv_squared",
    }


def gradient_control_config(mode: str) -> dict[str, Any]:
    if mode not in {"sum", "pcgrad"}:
        raise ValueError("gradient_control must be sum or pcgrad")
    return {
        "version": PANOPTIC_GRADIENT_CONTROL_VERSION if mode == "pcgrad" else "legacy_sum_v1",
        "mode": mode,
        "shared_parameters": [
            "input_proj", "encoder", "sparse_router", "sparse_experts", "sparse_router_norm",
            "branch_routers", "bridge_gate", "sq_rq_cross_attention",
        ],
        "task_groups": ["semantic", "query_mask_quality", "teacher", "identity", "ownership", "router"],
        "auxiliary_loss_owner": "query_mask_quality",
        "router_load_balance_loss_owner": "router",
        "amp_scale": 1.0,
        "clip_after_projection": True,
        "objective_abi_fields": GRADIENT_CONTROL_ABI_FIELDS if mode == "pcgrad" else None,
    }


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def feature_schema_sha256(feature_names: tuple[str, ...] = BASE_FEATURES) -> str:
    return canonical_json_sha256({"feature_names": feature_names, "feature_dim": len(feature_names)})


def ontology_sha256() -> str:
    return sha256_path(ONTOLOGY_PATH)


def model_state_schema_sha256(state: dict[str, Any]) -> str:
    schema = [(name, list(value.shape), str(value.dtype)) for name, value in sorted(state.items())]
    return canonical_json_sha256(schema)


def window_contract(max_tokens_per_record: int, *, target_schema_version: str = TARGET_SCHEMA_V3) -> dict[str, Any]:
    if target_schema_version not in {TARGET_SCHEMA_V3, TARGET_SCHEMA_V4, TARGET_SCHEMA_V5, TARGET_SCHEMA_V6}:
        raise ValueError(f"unsupported window contract target schema: {target_schema_version}")
    return {
        "input_level": "primitive",
        "target_schema_version": target_schema_version,
        "long_record_policy": "deterministic_contiguous_max_supervision_window_v1",
        "max_tokens_per_record": int(max_tokens_per_record),
        "page_inference": "continuous_overlap_windows",
    }


def geometry_decoder_config(
    *, hidden_dim: int, heads: int, num_queries: int, decoder_layers: int, identity_dim: int,
    num_stuff_queries: int = 32, local_neighbors: int = 4, coarse_grid_size: int = 4,
    typed_stuff_slots: bool = False, tensor_ring_rank: int = 0,
    geometry_attention_tile_size: int = 0,
) -> dict[str, Any]:
    if typed_stuff_slots and int(num_stuff_queries) != len(STUFF_LABELS):
        raise ValueError("typed_stuff_slots requires exactly one slot for each FloorPlanCAD stuff label")
    stuff_queries = min(int(num_stuff_queries), int(num_queries))
    return {
        "version": PANOPTIC_GEOMETRY_DECODER_VERSION,
        "hidden_dim": int(hidden_dim), "attention_heads": int(heads),
        "num_queries": int(num_queries), "num_thing_queries": int(num_queries) - stuff_queries,
        "num_stuff_queries": stuff_queries, "decoder_layers": max(int(decoder_layers), 1),
        "typed_stuff_slots": bool(typed_stuff_slots),
        "stuff_slot_labels": list(STUFF_LABELS) if typed_stuff_slots else None,
        "identity_dim": int(identity_dim), "local_neighbors": int(local_neighbors),
        "coarse_grid_size": int(coarse_grid_size),
        "tensor_ring_rank": int(tensor_ring_rank),
        "geometry_attention_tile_size": int(geometry_attention_tile_size),
    }


def typed_thing_query_count(model: Any, num_queries: int, num_stuff_queries: int) -> int | None:
    runtime_model = getattr(model, "_orig_mod", model)
    if not bool(getattr(runtime_model, "typed_stuff_slots", False)):
        return None
    return int(num_queries) - int(num_stuff_queries)


def sq_rq_config(
    *,
    enabled: bool,
    hidden_dim: int,
    heads: int,
    num_labels: int,
    gradient_scale: float,
    query_confidence_threshold: float = 0.6,
    token_membership_threshold: float = 0.5,
    training_membership_temperature: float = 0.1,
    semantic_query_residual_enabled: bool = False,
) -> dict[str, Any]:
    if not 0.0 <= float(gradient_scale) <= 0.1:
        raise ValueError("sq_rq_gradient_scale must be in [0, 0.1]")
    if not 0.0 <= float(query_confidence_threshold) <= 1.0:
        raise ValueError("sq_rq_query_confidence_threshold must be in [0, 1]")
    if not 0.0 <= float(token_membership_threshold) <= 1.0:
        raise ValueError("sq_rq_token_membership_threshold must be in [0, 1]")
    if not 0.0 <= float(training_membership_temperature) <= 1.0:
        raise ValueError("sq_rq_training_membership_temperature must be in [0, 1]")
    return {
        "version": PANOPTIC_SQ_RQ_VERSION,
        "enabled": bool(enabled),
        "primitive_dim": int(hidden_dim),
        "rq_query_dim": int(hidden_dim),
        "hidden_dim": int(hidden_dim),
        "heads": int(heads),
        "num_labels": int(num_labels),
        "gradient_scale": float(gradient_scale),
        "context_policy": "factorized_admission_soft_train_hard_eval_membership_topk_adaptive_gate_v3",
        "context_top_k": 8,
        "query_confidence_threshold": float(query_confidence_threshold),
        "query_confidence_semantics": "sigmoid_factorized_admission_probability",
        "token_membership_threshold": float(token_membership_threshold),
        "training_membership_temperature": float(training_membership_temperature),
        "semantic_compatibility_floor": 0.05,
        "semantic_query_residual": "mask_weighted_semantic_logits_zero_initialized_gate",
        "semantic_query_residual_gate_init": 0.0,
        "semantic_query_residual_enabled": bool(semantic_query_residual_enabled),
        "sq_mask_residual": "sq_token_mask_logits_zero_initialized_gate",
        "sq_mask_residual_gate_init": 0.0,
        "sq_ownership_residual": "sq_token_ownership_logits_zero_initialized_gate",
        "sq_ownership_residual_gate_init": 0.0,
        "no_object_class": int(num_labels) - 1,
        "context_source": "predicted_final_query_embedding_mask_class_and_factorized_admission_logits",
        "forbidden_context": ["gt_masks", "page_instance_id", "matched_target_indices"],
    }


def sq_rq_deployment_config(
    *,
    enabled: bool,
    query_confidence_threshold: float,
    token_membership_threshold: float,
    auto_fused: bool = False,
    auto_fuse_reason: str | None = None,
) -> dict[str, Any]:
    if not isinstance(enabled, bool) or not isinstance(auto_fused, bool):
        raise ValueError("SQ<-RQ deployment enabled/auto_fused flags must be boolean")
    if not 0.0 <= float(query_confidence_threshold) <= 1.0:
        raise ValueError("SQ<-RQ deployment query confidence threshold must be in [0, 1]")
    if not 0.0 <= float(token_membership_threshold) <= 1.0:
        raise ValueError("SQ<-RQ deployment token membership threshold must be in [0, 1]")
    if auto_fused and not auto_fuse_reason:
        raise ValueError("auto-fused SQ<-RQ deployment requires a reason")
    if not auto_fused and auto_fuse_reason is not None:
        raise ValueError("non-fused SQ<-RQ deployment cannot carry an auto-fuse reason")
    effective_enabled = bool(enabled) and not bool(auto_fused)
    return {
        "version": PANOPTIC_SQ_RQ_DEPLOYMENT_VERSION,
        "enabled": effective_enabled,
        "auto_fused": bool(auto_fused),
        "auto_fuse_reason": str(auto_fuse_reason) if auto_fuse_reason else None,
        "policy": "persist_runtime_state_and_fail_closed_after_auto_fuse",
        "query_confidence_threshold": float(query_confidence_threshold),
        "token_membership_threshold": float(token_membership_threshold),
        "training_membership_temperature": 0.0,
        "phase": "deployment_hard_thresholds",
    }


def quality_objective_contract(
    mask_threshold: float = PANOPTIC_QUALITY_MASK_THRESHOLD,
) -> dict[str, Any]:
    if not 0.0 <= float(mask_threshold) <= 1.0:
        raise ValueError("quality objective mask_threshold must be in [0, 1]")
    return {
        "version": PANOPTIC_QUALITY_OBJECTIVE_VERSION,
        "target": "length_weighted_hard_mask_iou_with_recall_safe_soft_iou_floor",
        "positive_score": "quality_probability",
        "negative_score": "foreground_probability_times_quality_probability",
        "negative_target": 0.0,
        "mask_threshold": float(mask_threshold),
        "metric_weighting": "floorplancad_log1p_length_weighted_primitive_set",
        "decoder_scope": "oracle_positive_window_ownership_null_competition_premerge_when_available",
        "hard_negative_scope": "deployment_composite_score",
        "ceiling_penalty": "probability_space_squared_hinge",
        "ranking": "deployment_composite_score_positive_lift",
        "object_normalization_threshold": PANOPTIC_OBJECT_NORMALIZATION_THRESHOLD,
        "deployment_use": "class_probability_times_quality_probability_times_mask_objectness",
    }


def checkpoint_abi_metadata(max_tokens_per_record: int, *, geometry_config: dict[str, Any] | None = None, geometry_state_schema_sha256: str | None = None, sq_rq: dict[str, Any] | None = None, sq_rq_deployment: dict[str, Any] | None = None, ownership: dict[str, Any] | None = None, gradient_control: dict[str, Any] | None = None, sparse_router: dict[str, Any] | None = None, quality_head_trained: bool = True, quality_admission_promoted: bool = False, input_protocol: dict[str, Any] | None = None, quality_mask_threshold: float = PANOPTIC_QUALITY_MASK_THRESHOLD) -> dict[str, Any]:
    if input_protocol is not None:
        validate_segment_input_protocol(input_protocol)
    active_feature_names = feature_names_for_input_protocol(input_protocol)
    target_schema_version = input_protocol.get("target_schema_version", TARGET_SCHEMA_V3) if input_protocol is not None else TARGET_SCHEMA_V3
    contract = window_contract(max_tokens_per_record, target_schema_version=target_schema_version)
    quality_contract = quality_objective_contract(mask_threshold=quality_mask_threshold)
    metadata = {
        "abi_version": PANOPTIC_CHECKPOINT_ABI_VERSION,
        "feature_schema_sha256": feature_schema_sha256(active_feature_names),
        "ontology_sha256": ontology_sha256(),
        "window_contract": contract,
        "window_contract_sha256": canonical_json_sha256(contract),
        "position_encoding_version": POSITION_ENCODING_VERSION,
        "quality_head": PANOPTIC_QUALITY_HEAD_VERSION,
        "quality_objective_version": PANOPTIC_QUALITY_OBJECTIVE_VERSION,
        "quality_objective_config": quality_contract,
        "quality_objective_config_sha256": canonical_json_sha256(quality_contract),
        "quality_head_trained": bool(quality_head_trained),
        "quality_admission_promoted": bool(quality_admission_promoted),
        "identity_head_version": PANOPTIC_IDENTITY_HEAD_VERSION,
        "identity_dim": PANOPTIC_IDENTITY_DIM,
    }
    if geometry_config is not None:
        if not isinstance(geometry_state_schema_sha256, str) or len(geometry_state_schema_sha256) != 64:
            raise ValueError("geometry-v2 ABI requires geometry_state_schema_sha256")
        metadata.update({
            "abi_version": PANOPTIC_GEOMETRY_CHECKPOINT_ABI_VERSION,
            "geometry_decoder_version": PANOPTIC_GEOMETRY_DECODER_VERSION,
            "geometry_config": geometry_config,
            "geometry_config_sha256": canonical_json_sha256(geometry_config),
            "geometry_state_schema_sha256": geometry_state_schema_sha256,
            "query_allocation_sha256": canonical_json_sha256({
                "thing": [0, geometry_config["num_thing_queries"]],
                "stuff": [geometry_config["num_thing_queries"], geometry_config["num_queries"]],
            }),
        })
    if sq_rq is not None and sq_rq.get("enabled"):
        if geometry_config is None:
            raise ValueError("SQ<-RQ production ABI requires geometry-v2")
        if sq_rq_deployment is None:
            raise ValueError("SQ<-RQ production ABI requires explicit deployment state")
        deployment = sq_rq_deployment
        if deployment.get("version") != PANOPTIC_SQ_RQ_DEPLOYMENT_VERSION:
            raise ValueError("SQ<-RQ deployment ABI version mismatch")
        if bool(deployment.get("auto_fused", False)) and bool(deployment.get("enabled", False)):
            raise ValueError("auto-fused SQ<-RQ deployment cannot remain enabled")
        metadata.update({
            "abi_version": PANOPTIC_SQ_RQ_CHECKPOINT_ABI_VERSION,
            "sq_rq_version": PANOPTIC_SQ_RQ_VERSION,
            "sq_rq_config": sq_rq,
            "sq_rq_config_sha256": canonical_json_sha256(sq_rq),
            "sq_rq_deployment": deployment,
            "sq_rq_deployment_sha256": canonical_json_sha256(deployment),
        })
    if ownership is not None:
        if geometry_config is None or sq_rq is None or not sq_rq.get("enabled"):
            raise ValueError("ownership production ABI requires geometry-v2 and SQ<-RQ")
        metadata.update({
            "abi_version": PANOPTIC_OWNERSHIP_CHECKPOINT_ABI_VERSION,
            "ownership_version": OWNERSHIP_VERSION,
            "ownership_config": ownership,
            "ownership_config_sha256": canonical_json_sha256(ownership),
        })
    if gradient_control is not None:
        if ownership is None:
            raise ValueError("gradient-control ABI requires geometry-v2, SQ<-RQ, and ownership")
        metadata.update({
            "abi_version": PANOPTIC_GRADIENT_CONTROL_CHECKPOINT_ABI_VERSION,
            "gradient_control_version": gradient_control["version"],
            "gradient_control_config": gradient_control,
            "gradient_control_config_sha256": canonical_json_sha256(gradient_control),
        })
    if sparse_router is not None and sparse_router.get("enabled"):
        if geometry_config is None or sq_rq is None or not sq_rq.get("enabled") or ownership is None:
            raise ValueError("ABI-v8 sparse router requires the complete geometry-v2 + SQ-RQ + ownership + PCGrad stack")
        metadata.update({
            "abi_version": PANOPTIC_SPARSE_ROUTER_CHECKPOINT_ABI_VERSION,
            "semantic_head_num_labels": IGNORE_LABEL,
            "sparse_router_version": PANOPTIC_SPARSE_ROUTER_VERSION,
            "sparse_router_config": sparse_router,
            "sparse_router_config_sha256": canonical_json_sha256(sparse_router),
        })
    if input_protocol is not None:
        metadata["input_protocol"] = input_protocol
        metadata["input_protocol_sha256"] = canonical_json_sha256(input_protocol)
    return metadata


def validate_checkpoint_abi(
    ckpt: dict[str, Any],
    *,
    legacy_position_compat: bool = False,
    allow_quality_objective_mismatch: bool = False,
) -> dict[str, Any]:
    state = ckpt.get("state_dict") if isinstance(ckpt.get("state_dict"), dict) else {}
    quality_keys = {"query_quality_head.weight", "query_quality_head.bias"}
    present_quality = quality_keys & set(state)
    identity_prefix = "query_identity_head."
    present_identity = {key for key in state if key.startswith(identity_prefix)}
    ownership_keys = {
        "query_ownership_head.weight", "query_ownership_head.bias",
        "token_ownership_head.weight", "token_ownership_head.bias",
        "null_ownership_head.weight", "null_ownership_head.bias",
        "ownership_residual_gate",
    }
    present_ownership = ownership_keys & set(state)
    legacy_ownership_without_residual_gate = present_ownership == ownership_keys - {"ownership_residual_gate"}
    identity_keys = {
        "query_identity_head.projection.0.weight", "query_identity_head.projection.0.bias",
        "query_identity_head.projection.2.weight", "query_identity_head.projection.2.bias",
        "query_identity_head.projection.3.weight", "query_identity_head.projection.3.bias",
    }
    if present_identity and present_identity != identity_keys:
        raise ValueError(f"partial query_identity_head state is forbidden: present={sorted(present_identity)}")
    if present_quality and present_quality != quality_keys:
        raise ValueError(f"partial query_quality_head state is forbidden: present={sorted(present_quality)}")
    if present_ownership and present_ownership != ownership_keys and not legacy_ownership_without_residual_gate:
        raise ValueError(f"partial ownership head state is forbidden: present={sorted(present_ownership)}")
    metadata = ckpt.get("checkpoint_abi") if isinstance(ckpt.get("checkpoint_abi"), dict) else None
    if metadata is None:
        if not legacy_position_compat:
            raise ValueError("checkpoint ABI metadata missing; use --legacy-position-compat only for explicit diagnostic replay")
        if present_quality:
            raise ValueError("legacy compatibility is restricted to checkpoints with no quality head state")
        return {
            "status": "legacy_diagnostic_explicit",
            "production_compatible": False,
            "position_encoding_version": "continuous_fourier_legacy_v1",
            "quality_head_trained": False,
            "quality_multiplier": 1.0,
            "identity_head_trained": False,
            "warning": "Legacy ABI is unknown; position v1 and quality multiplier 1 are explicit diagnostic assumptions.",
        }
    legacy_identity = "identity_head_version" not in metadata or "identity_dim" not in metadata
    if legacy_identity and legacy_position_compat:
        if present_identity:
            raise ValueError("legacy identity compatibility forbids unversioned identity head state")
        return {
            "status": "legacy_identity_diagnostic_explicit",
            "production_compatible": False,
            "position_encoding_version": metadata.get("position_encoding_version", "continuous_fourier_legacy_v1"),
            "quality_head_trained": present_quality == quality_keys,
            "quality_multiplier": "sigmoid_trained_head" if present_quality == quality_keys else 1.0,
            "identity_head_trained": False,
            "warning": "Identity ABI is absent; inference falls back to the legacy overlap tracker for diagnostics only.",
        }
    geometry_mode = metadata.get("geometry_decoder_version") == PANOPTIC_GEOMETRY_DECODER_VERSION
    sq_rq_enabled = metadata.get("sq_rq_version") == PANOPTIC_SQ_RQ_VERSION
    ownership_enabled = metadata.get("ownership_version") == OWNERSHIP_VERSION
    gradient_control_enabled = metadata.get("gradient_control_version") == PANOPTIC_GRADIENT_CONTROL_VERSION
    sparse_router_enabled = metadata.get("sparse_router_version") == PANOPTIC_SPARSE_ROUTER_VERSION
    if not ownership_enabled and (geometry_mode or sq_rq_enabled):
        if not legacy_position_compat:
            raise ValueError("pre-v5 checkpoint lacks global ownership ABI; explicit legacy compatibility is diagnostic only")
    required = {
        "abi_version", "feature_schema_sha256", "ontology_sha256", "window_contract",
        "window_contract_sha256", "position_encoding_version", "quality_head", "quality_head_trained",
        "quality_objective_version", "quality_objective_config", "quality_objective_config_sha256",
        "identity_head_version", "identity_dim",
    }
    if metadata.get("quality_objective_version") == PANOPTIC_QUALITY_OBJECTIVE_VERSION:
        required.add("quality_admission_promoted")
    if geometry_mode:
        required.update({"geometry_decoder_version", "geometry_config", "geometry_config_sha256", "geometry_state_schema_sha256", "query_allocation_sha256"})
    if sq_rq_enabled:
        required.update({
            "sq_rq_version", "sq_rq_config", "sq_rq_config_sha256",
            "sq_rq_deployment", "sq_rq_deployment_sha256",
        })
    if ownership_enabled:
        required.update({"ownership_version", "ownership_config", "ownership_config_sha256"})
    if gradient_control_enabled:
        required.update({"gradient_control_version", "gradient_control_config", "gradient_control_config_sha256"})
    legacy_sparse_schema = ckpt.get("schema_version") == PANOPTIC_LEGACY_SPARSE_ROUTER_SCHEMA_VERSION and not bool((metadata.get("sparse_router_config") or {}).get("typed_branch_routers", False))
    if sparse_router_enabled:
        if not (geometry_mode and sq_rq_enabled and ownership_enabled):
            raise ValueError("ABI-v8 sparse router requires the complete geometry-v2 + SQ-RQ + ownership stack")
        required.update({"sparse_router_version", "sparse_router_config", "sparse_router_config_sha256", "semantic_head_num_labels"})
    input_protocol = metadata.get("input_protocol")
    if input_protocol is not None:
        validate_segment_input_protocol(input_protocol)
        required.update({"input_protocol", "input_protocol_sha256"})
    missing = required - set(metadata)
    if missing:
        raise ValueError(f"checkpoint ABI metadata incomplete: missing={sorted(missing)}")
    expected_quality_version = (
        metadata.get("quality_objective_version")
        if allow_quality_objective_mismatch
        else PANOPTIC_QUALITY_OBJECTIVE_VERSION
    )
    metadata_quality_config = (
        metadata.get("quality_objective_config")
        if isinstance(metadata.get("quality_objective_config"), dict)
        else {}
    )
    expected_quality_config = (
        metadata.get("quality_objective_config")
        if allow_quality_objective_mismatch
        else quality_objective_contract(
            mask_threshold=metadata_quality_config.get(
                "mask_threshold",
                PANOPTIC_QUALITY_MASK_THRESHOLD,
            )
        )
    )
    expected = {
        "abi_version": PANOPTIC_SPARSE_ROUTER_CHECKPOINT_ABI_VERSION if sparse_router_enabled else (PANOPTIC_GRADIENT_CONTROL_CHECKPOINT_ABI_VERSION if gradient_control_enabled else (PANOPTIC_OWNERSHIP_CHECKPOINT_ABI_VERSION if ownership_enabled else (PANOPTIC_SQ_RQ_CHECKPOINT_ABI_VERSION if sq_rq_enabled else (PANOPTIC_GEOMETRY_CHECKPOINT_ABI_VERSION if geometry_mode else PANOPTIC_CHECKPOINT_ABI_VERSION)))),
        "feature_schema_sha256": feature_schema_sha256(feature_names_for_input_protocol(input_protocol)),
        "ontology_sha256": ontology_sha256(),
        "position_encoding_version": POSITION_ENCODING_VERSION,
        "quality_head": PANOPTIC_QUALITY_HEAD_VERSION,
        "quality_objective_version": expected_quality_version,
        "quality_objective_config": expected_quality_config,
        "quality_head_trained": bool(metadata.get("quality_head_trained", True)),
        "identity_head_version": PANOPTIC_IDENTITY_HEAD_VERSION,
        "identity_dim": PANOPTIC_IDENTITY_DIM,
        "semantic_head_num_labels": IGNORE_LABEL if sparse_router_enabled else metadata.get("semantic_head_num_labels"),
    }
    if metadata.get("quality_objective_version") == PANOPTIC_QUALITY_OBJECTIVE_VERSION:
        expected["quality_admission_promoted"] = bool(
            metadata.get("quality_admission_promoted", False)
        )
    actual = {key: metadata.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"checkpoint ABI mismatch: expected={expected}, actual={actual}")
    if metadata["window_contract_sha256"] != canonical_json_sha256(metadata["window_contract"]):
        raise ValueError("checkpoint window contract hash mismatch")
    if metadata["quality_objective_config_sha256"] != canonical_json_sha256(metadata["quality_objective_config"]):
        raise ValueError("checkpoint quality objective config hash mismatch")
    if input_protocol is not None:
        if metadata["input_protocol_sha256"] != canonical_json_sha256(input_protocol):
            raise ValueError("checkpoint input protocol hash mismatch")
        validate_segment_input_protocol(input_protocol)
    if geometry_mode:
        geometry_config = metadata["geometry_config"]
        if metadata["geometry_config_sha256"] != canonical_json_sha256(geometry_config):
            raise ValueError("checkpoint geometry config hash mismatch")
        allocation = {"thing": [0, geometry_config["num_thing_queries"]], "stuff": [geometry_config["num_thing_queries"], geometry_config["num_queries"]]}
        if metadata["query_allocation_sha256"] != canonical_json_sha256(allocation):
            raise ValueError("checkpoint geometry query allocation hash mismatch")
        if metadata["geometry_state_schema_sha256"] != model_state_schema_sha256(state):
            raise ValueError("checkpoint geometry state schema hash mismatch")
    if sq_rq_enabled:
        sq_rq = metadata["sq_rq_config"]
        if not geometry_mode or not sq_rq.get("enabled") or sq_rq.get("version") != PANOPTIC_SQ_RQ_VERSION:
            raise ValueError("checkpoint SQ<-RQ requires enabled geometry-v2 prediction-only config")
        if metadata["sq_rq_config_sha256"] != canonical_json_sha256(sq_rq):
            raise ValueError("checkpoint SQ<-RQ config hash mismatch")
        if not 0.0 <= float(sq_rq.get("gradient_scale", -1.0)) <= 0.1:
            raise ValueError("checkpoint SQ<-RQ gradient scale is outside [0, 0.1]")
        if not 0.0 <= float(sq_rq.get("training_membership_temperature", 0.0)) <= 1.0:
            raise ValueError("checkpoint SQ<-RQ training membership temperature is outside [0, 1]")
        sq_rq_deployment = metadata["sq_rq_deployment"]
        if not isinstance(sq_rq_deployment, dict):
            raise ValueError("checkpoint SQ<-RQ deployment config must be a mapping")
        if sq_rq_deployment.get("version") != PANOPTIC_SQ_RQ_DEPLOYMENT_VERSION:
            raise ValueError("checkpoint SQ<-RQ deployment ABI mismatch")
        if metadata["sq_rq_deployment_sha256"] != canonical_json_sha256(sq_rq_deployment):
            raise ValueError("checkpoint SQ<-RQ deployment config hash mismatch")
        expected_sq_rq_deployment = sq_rq_deployment_config(
            enabled=sq_rq_deployment.get("enabled"),
            query_confidence_threshold=sq_rq_deployment.get("query_confidence_threshold", -1.0),
            token_membership_threshold=sq_rq_deployment.get("token_membership_threshold", -1.0),
            auto_fused=sq_rq_deployment.get("auto_fused"),
            auto_fuse_reason=sq_rq_deployment.get("auto_fuse_reason"),
        )
        if sq_rq_deployment != expected_sq_rq_deployment:
            raise ValueError("checkpoint SQ<-RQ deployment config is not canonical")
        if (
            float(sq_rq_deployment["query_confidence_threshold"])
            != float(sq_rq["query_confidence_threshold"])
            or float(sq_rq_deployment["token_membership_threshold"])
            != float(sq_rq["token_membership_threshold"])
        ):
            raise ValueError("checkpoint SQ<-RQ deployment thresholds differ from architecture ABI")
    if ownership_enabled:
        ownership = metadata["ownership_config"]
        if not geometry_mode or not sq_rq_enabled or ownership.get("version") != OWNERSHIP_VERSION:
            raise ValueError("checkpoint ownership requires geometry-v2 and SQ<-RQ")
        if metadata["ownership_config_sha256"] != canonical_json_sha256(ownership):
            raise ValueError("checkpoint ownership config hash mismatch")
    if gradient_control_enabled:
        gradient_control = metadata["gradient_control_config"]
        if not ownership_enabled or gradient_control != gradient_control_config("pcgrad"):
            raise ValueError("checkpoint production gradient control must be deterministic PCGrad over the v7 SQ-RQ stack")
        if metadata["gradient_control_config_sha256"] != canonical_json_sha256(gradient_control):
            raise ValueError("checkpoint gradient control config hash mismatch")
    if sparse_router_enabled:
        sparse_router = metadata["sparse_router_config"]
        expected_router = sparse_router_config(
            enabled=True, hidden_dim=sparse_router.get("hidden_dim", -1),
            num_experts=sparse_router.get("num_experts", -1), top_k=sparse_router.get("top_k", -1),
            temperature=sparse_router.get("temperature", -1.0),
            typed_branch_routers=bool(sparse_router.get("typed_branch_routers", False)),
            branch_num_experts=int(sparse_router.get("branch_num_experts", 2)),
            branch_top_k=int(sparse_router.get("branch_top_k", 1)),
            branch_capacity_factor=float(sparse_router.get("branch_capacity_factor", 1.25)),
            branch_dropless=bool(sparse_router.get("branch_dropless", False)),
        )
        legacy_router = dict(sparse_router)
        legacy_router.pop("branch_dropless", None)
        legacy_expected_router = dict(expected_router)
        legacy_expected_router.pop("branch_dropless", None)
        current_router_valid = (
            sparse_router == expected_router
            and metadata["sparse_router_config_sha256"] == canonical_json_sha256(sparse_router)
        )
        legacy_router_valid = (
            "branch_dropless" not in sparse_router
            and legacy_router == legacy_expected_router
            and metadata["sparse_router_config_sha256"] == canonical_json_sha256(legacy_router)
        )
        if not current_router_valid and not legacy_router_valid:
            raise ValueError("checkpoint sparse router config/hash mismatch")
    top_level_hashes = {
        "feature_schema_sha256": ckpt.get("feature_schema_sha256"),
        "ontology_sha256": ckpt.get("ontology_sha256"),
        "window_contract_sha256": ckpt.get("window_contract_sha256"),
    }
    expected_hashes = {
        "feature_schema_sha256": metadata["feature_schema_sha256"],
        "ontology_sha256": metadata["ontology_sha256"],
        "window_contract_sha256": metadata["window_contract_sha256"],
    }
    if top_level_hashes != expected_hashes:
        raise ValueError(f"checkpoint top-level ABI hashes missing or inconsistent: expected={expected_hashes}, actual={top_level_hashes}")
    top_level_contract = {
        "schema_version": ckpt.get("schema_version"),
        "position_encoding_version": ckpt.get("position_encoding_version"),
        "quality_head": ckpt.get("quality_head"),
        "geometry_decoder_mode": ckpt.get("geometry_decoder_mode", "legacy_debug"),
        "geometry_config": ckpt.get("geometry_config"),
        "sq_rq_config": ckpt.get("sq_rq_config"),
        "sq_rq_deployment": ckpt.get("sq_rq_deployment"),
        "ownership_config": ckpt.get("ownership_config"),
        "gradient_control_config": ckpt.get("gradient_control_config"),
        "sparse_router_config": ckpt.get("sparse_router_config"),
        "input_protocol": ckpt.get("input_protocol"),
    }
    expected_contract = {
        "schema_version": PANOPTIC_LEGACY_SPARSE_ROUTER_SCHEMA_VERSION if legacy_sparse_schema else (PANOPTIC_SPARSE_ROUTER_SCHEMA_VERSION if sparse_router_enabled else ("floorplancad_line_token_panoptic_moe_checkpoint_v6_geometry_v2_sq_rq_ownership_pcgrad" if gradient_control_enabled else ("floorplancad_line_token_panoptic_moe_checkpoint_v5_geometry_v2_sq_rq_ownership" if ownership_enabled else ("floorplancad_line_token_panoptic_moe_checkpoint_v4_geometry_v2_sq_rq" if sq_rq_enabled else ("floorplancad_line_token_panoptic_moe_checkpoint_v3_geometry_v2" if geometry_mode else "floorplancad_line_token_panoptic_moe_checkpoint_v2"))))),
        "position_encoding_version": metadata["position_encoding_version"],
        "quality_head": metadata["quality_head"],
        "geometry_decoder_mode": "geometry_v2" if geometry_mode else "legacy_debug",
        "geometry_config": metadata.get("geometry_config"),
        "sq_rq_config": metadata.get("sq_rq_config"),
        "sq_rq_deployment": metadata.get("sq_rq_deployment"),
        "ownership_config": metadata.get("ownership_config"),
        "gradient_control_config": metadata.get("gradient_control_config"),
        "sparse_router_config": metadata.get("sparse_router_config"),
        "input_protocol": metadata.get("input_protocol"),
    }
    if top_level_contract != expected_contract:
        raise ValueError(f"checkpoint top-level ABI contract mismatch: expected={expected_contract}, actual={top_level_contract}")
    if sq_rq_enabled:
        deployment = metadata["sq_rq_deployment"]
        if bool(ckpt.get("sq_rq_auto_fused", False)) != bool(deployment["auto_fused"]):
            raise ValueError("checkpoint top-level SQ<-RQ auto-fuse state differs from deployment ABI")
        if ckpt.get("sq_rq_auto_fuse_reason") != deployment.get("auto_fuse_reason"):
            raise ValueError("checkpoint top-level SQ<-RQ auto-fuse reason differs from deployment ABI")
    identity_top = {"identity_head_version": ckpt.get("identity_head_version"), "identity_dim": ckpt.get("identity_dim")}
    if any(value is not None for value in identity_top.values()) and identity_top != {
        "identity_head_version": metadata["identity_head_version"], "identity_dim": metadata["identity_dim"]
    }:
        raise ValueError("checkpoint top-level identity ABI is partial or inconsistent")
    if canonical_json_sha256({"feature_names": ckpt.get("feature_names"), "feature_dim": len(ckpt.get("feature_names") or [])}) != metadata["feature_schema_sha256"]:
        raise ValueError("checkpoint feature_names do not match feature schema hash")
    if present_quality != quality_keys:
        raise ValueError("trained quality metadata requires complete query_quality_head state")
    if present_identity != identity_keys:
        raise ValueError("trained identity metadata requires complete query_identity_head state")
    if ownership_enabled and present_ownership != ownership_keys and not legacy_ownership_without_residual_gate:
        raise ValueError("ownership metadata requires complete ownership head state")
    objective_config = ckpt.get("objective_config")
    objective_hash_value = ckpt.get("objective_config_hash")
    if not isinstance(objective_config, dict) or not isinstance(objective_hash_value, str) or len(objective_hash_value) != 64:
        raise ValueError("checkpoint full objective config/hash missing or malformed")
    if objective_hash_value != objective_config_hash(objective_config):
        raise ValueError("checkpoint objective config hash mismatch")
    objective_quality_version = objective_config.get("quality_objective_version")
    if objective_quality_version != metadata.get("quality_objective_version"):
        raise ValueError(
            "checkpoint quality objective ABI is not cross-bound to objective_config: "
            f"metadata={metadata.get('quality_objective_version')!r}, "
            f"objective_config={objective_quality_version!r}"
        )
    if objective_config.get("quality_objective_config") != metadata.get("quality_objective_config"):
        raise ValueError("checkpoint quality objective config is not cross-bound to objective_config")
    if (
        "quality_admission_promoted" in metadata
        and bool(ckpt.get("quality_admission_promoted", False))
        != bool(metadata.get("quality_admission_promoted", False))
    ):
        raise ValueError("checkpoint quality admission promotion state is inconsistent")
    if bool(metadata.get("quality_admission_promoted", False)):
        promoted_gate = ckpt.get("selection_gate") or {}
        if (
            ckpt.get("checkpoint_boundary") != "best_selection_checkpoint"
            or promoted_gate.get("passed") is not True
            or not math.isfinite(float(ckpt.get("selection_score", -float("inf"))))
        ):
            raise ValueError(
                "quality admission promotion requires a finite, gate-passed best checkpoint"
            )
    return {
        "status": "validated",
        "production_compatible": ownership_enabled or not (geometry_mode or sq_rq_enabled),
        "position_encoding_version": POSITION_ENCODING_VERSION,
        "quality_head_trained": bool(metadata.get("quality_head_trained", True)),
        "quality_admission_promoted": bool(
            metadata.get("quality_admission_promoted", False)
        ),
        "quality_objective_version": metadata.get("quality_objective_version"),
        "quality_objective_config": metadata.get("quality_objective_config"),
        "quality_admission_compatible": (
            metadata.get("quality_objective_version") == PANOPTIC_QUALITY_OBJECTIVE_VERSION
            and metadata.get("quality_objective_config") == quality_objective_contract()
        ),
        "quality_objective_migration_allowed": bool(
            allow_quality_objective_mismatch
            and metadata.get("quality_objective_version") != PANOPTIC_QUALITY_OBJECTIVE_VERSION
        ),
        "quality_multiplier": "sigmoid_trained_head" if metadata.get("quality_head_trained", True) else 1.0,
        "identity_head_trained": True,
        "geometry_decoder_mode": "geometry_v2" if geometry_mode else "legacy_debug",
        "geometry_config": metadata.get("geometry_config"),
        "sq_rq_enabled": sq_rq_enabled,
        "sq_rq_config": metadata.get("sq_rq_config"),
        "sq_rq_deployment": metadata.get("sq_rq_deployment"),
        "sq_rq_deployment_enabled": bool(
            sq_rq_enabled and (metadata.get("sq_rq_deployment") or {}).get("enabled", False)
        ),
        "ownership_enabled": ownership_enabled,
        "ownership_config": metadata.get("ownership_config"),
        "gradient_control_enabled": gradient_control_enabled,
        "gradient_control_config": metadata.get("gradient_control_config"),
        "sparse_router_enabled": sparse_router_enabled,
        "sparse_router_config": metadata.get("sparse_router_config"),
        "input_protocol": input_protocol,
        "requires_segment_features": bool(input_protocol and input_protocol.get("segment_features")),
        "warning": None if ownership_enabled or not (geometry_mode or sq_rq_enabled) else "Pre-v5 geometry checkpoint is diagnostic-only because ownership-before-mask is absent.",
        "checkpoint_abi": metadata,
    }


def read_json_file(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"status": "invalid_json", "path": rel(path)}
    return data if isinstance(data, dict) else {"status": "non_object_json", "path": rel(path)}


def deterministic_stratified_validation_pages(path: Path, page_limit: int, seed: int) -> tuple[str, ...] | None:
    """Choose a deterministic page-level diagnostic subset with label/type coverage."""
    if int(page_limit) <= 0:
        return None
    page_strata: dict[str, set[tuple[str, int]]] = {}
    for record in iter_jsonl(path):
        page_id = str(record.get("original_record_id") or record.get("record_id") or "")
        if not page_id:
            continue
        strata = page_strata.setdefault(page_id, set())
        for row in record.get("primitive_rows") or []:
            if not isinstance(row, dict):
                continue
            label = parse_int(row.get("semantic_id"), IGNORE_LABEL)
            if 0 <= label < IGNORE_LABEL:
                strata.add(("thing" if label in THING_LABELS else "stuff", label))
    universe = set().union(*page_strata.values()) if page_strata else set()
    if not universe:
        raise ValueError("validation cache has no labeled thing/stuff strata")
    ranking = {
        page_id: hashlib.sha256(f"{seed}:{page_id}".encode("utf-8")).hexdigest()
        for page_id in page_strata
    }
    selected: list[str] = []
    uncovered = set(universe)
    while uncovered and len(selected) < int(page_limit):
        candidates = [page_id for page_id in page_strata if page_id not in selected]
        if not candidates:
            break
        page_id = min(
            candidates,
            key=lambda value: (-len(page_strata[value] & uncovered), ranking[value], value),
        )
        if not (page_strata[page_id] & uncovered):
            break
        selected.append(page_id)
        uncovered -= page_strata[page_id]
    if uncovered:
        raise ValueError(
            f"val_limit_pages={page_limit} cannot cover all diagnostic strata; missing={sorted(uncovered)}"
        )
    remaining = sorted((page_id for page_id in page_strata if page_id not in selected), key=lambda value: (ranking[value], value))
    selected.extend(remaining[: max(0, int(page_limit) - len(selected))])
    return tuple(selected)


def load_panoptic_target_arrays(
    record: dict[str, Any],
    max_tokens: int,
    *,
    training: bool = True,
    legacy_diagnostic: bool = False,
    num_queries: int | None = None,
    max_segments_per_primitive: int = 32,
) -> tuple[Any, ...] | None:
    target_schema = record.get("target_schema_version")
    if target_schema not in {TARGET_SCHEMA_V2, TARGET_SCHEMA_V3, TARGET_SCHEMA_V4, TARGET_SCHEMA_V5, TARGET_SCHEMA_V6}:
        if not legacy_diagnostic:
            raise ValueError(f"panoptic production requires {TARGET_SCHEMA_V2}; v1/fallback cache is forbidden")
        if training:
            raise ValueError("legacy target schema is diagnostic-only and cannot be used for training")
        arrays = record_to_arrays(record, max_tokens, random.Random(0), "primitive")
        if arrays is None:
            return None
        x, labels, instances, primitives = arrays
        ones = np.ones(len(labels), dtype=np.float32)
        return x, labels, instances, primitives, ones, ones, np.ones(len(labels), dtype=bool), tuple(str(value) for value in instances.tolist()), None, None
    if int(max_segments_per_primitive) < 1:
        raise ValueError("max_segments_per_primitive must be positive")
    segment_features = None
    segment_padding = None
    if target_schema in {TARGET_SCHEMA_V3, TARGET_SCHEMA_V4, TARGET_SCHEMA_V5, TARGET_SCHEMA_V6}:
        batch_segments = load_target_schema_v4(record) if target_schema in {TARGET_SCHEMA_V4, TARGET_SCHEMA_V5, TARGET_SCHEMA_V6} else load_target_schema_v3(record)
        batch = batch_segments.base
        max_segments = min(max(len(segments) for segments in batch_segments.segment_features), int(max_segments_per_primitive))
        segment_features = np.zeros((len(batch_segments.segment_features), max_segments, batch.features.shape[1]), dtype=np.float32)
        segment_padding = np.ones((len(batch_segments.segment_features), max_segments), dtype=bool)
        for primitive_index, segments in enumerate(batch_segments.segment_features):
            if len(segments) <= max_segments:
                selected = np.arange(len(segments), dtype=np.int64)
            else:
                selected = np.unique(np.linspace(0, len(segments) - 1, num=max_segments, dtype=np.int64))
                if selected.size < max_segments:
                    selected = np.pad(selected, (0, max_segments - selected.size), mode="edge")
            count = min(len(selected), max_segments)
            segment_features[primitive_index, :count] = segments[selected[:count]]
            segment_padding[primitive_index, :count] = False
    else:
        batch = load_target_schema_v2(record)
    cache_capacity = int(record.get("query_target_capacity", -1))
    cache_count = int(record.get("query_target_count", -1))
    if num_queries is not None and (cache_capacity > int(num_queries) or cache_count > int(num_queries)):
        raise ValueError(
            f"target-schema-v2 query capacity/count exceed runtime num_queries: "
            f"capacity={cache_capacity}, count={cache_count}, num_queries={num_queries}"
        )
    if len(batch.labels) > max_tokens:
        raise ValueError(f"target-schema-v2 window exceeds max_tokens without an explicit continuous window: {len(batch.labels)}>{max_tokens}")
    # Semantic supervision keeps inverse-exposure weighting. RQ matching, masks,
    # quality targets, ownership, and proxy IoU use the FloorPlanCAD/VecFormer
    # primitive-IoU protocol: log(1 + primitive_length) only.
    semantic_weights = semantic_primitive_weights(batch, normalize=False)
    length_weights = batch.log1p_length.astype(np.float32)
    return (
        batch.features,
        batch.labels,
        batch.instances,
        batch.primitive_ids,
        semantic_weights,
        length_weights,
        batch.mask_loss_valid,
        batch.page_instance_ids,
        segment_features,
        segment_padding,
    )


def prefetch_iterable(source: Iterable[Any], max_prefetch: int) -> Iterable[Any]:
    if max_prefetch <= 0:
        yield from source
        return
    sentinel = object()
    queue: Queue[Any] = Queue(maxsize=max(1, int(max_prefetch)))

    def worker() -> None:
        try:
            for item in source:
                queue.put(item)
        finally:
            queue.put(sentinel)

    thread = threading.Thread(target=worker, name="panoptic_moe_prefetch", daemon=True)
    thread.start()
    while True:
        item = queue.get()
        if item is sentinel:
            break
        yield item


def prefetch_training_arrays(
    source: Iterable[dict[str, Any]],
    *,
    max_prefetch: int,
    workers: int,
    max_tokens: int,
    rng: random.Random,
    input_level: str,
    num_queries: int,
) -> Iterable[tuple[dict[str, Any], Any, Any, Any, Any]]:
    if max_prefetch <= 0:
        for record in source:
            arrays = load_panoptic_target_arrays(record, max_tokens, training=True, num_queries=num_queries)
            yield record, arrays, rng.getstate(), None, None
        return

    if workers > 1:
        seed_rng = random.Random()
        seed_rng.setstate(rng.getstate())
        sentinel = object()
        queue: Queue[Any] = Queue(maxsize=max(1, int(max_prefetch)))
        executor = ThreadPoolExecutor(max_workers=max(1, int(workers)), thread_name_prefix="panoptic_moe_parse")
        stop_event = threading.Event()

        def record_row_count(record: dict[str, Any]) -> int:
            cached_rows = record.get("primitive_rows") if isinstance(record.get("primitive_rows"), list) else None
            if cached_rows is not None:
                return len(cached_rows)
            tokens = record.get("line_tokens") if isinstance(record.get("line_tokens"), list) else []
            return len(tokens)

        def parse_record(record: dict[str, Any], rng_state_after_record: Any) -> tuple[dict[str, Any], Any, Any, Any, Any]:
            try:
                arrays = load_panoptic_target_arrays(record, max_tokens, training=True, num_queries=num_queries)
                return record, arrays, rng_state_after_record, None, None
            except Exception as exc:  # noqa: BLE001 - propagate worker parsing failures to the training thread.
                return record, None, rng_state_after_record, type(exc).__name__, str(exc)

        def producer() -> None:
            pending: dict[int, Future[Any]] = {}
            next_submit = 0
            next_emit = 0
            try:
                source_iter = iter(source)
                exhausted = False
                while not stop_event.is_set():
                    while not exhausted and len(pending) < max(1, int(max_prefetch)):
                        try:
                            record = next(source_iter)
                        except StopIteration:
                            exhausted = True
                            break
                        if record_row_count(record) > max_tokens:
                            arrays = load_panoptic_target_arrays(record, max_tokens, training=True, num_queries=num_queries)
                            immediate: Future[Any] = Future()
                            immediate.set_result((record, arrays, seed_rng.getstate(), None, None))
                            pending[next_submit] = immediate
                        else:
                            pending[next_submit] = executor.submit(parse_record, record, seed_rng.getstate())
                        next_submit += 1
                    if next_emit in pending:
                        queue.put(pending.pop(next_emit).result())
                        next_emit += 1
                        continue
                    if exhausted:
                        break
            finally:
                stop_event.set()
                queue.put(sentinel)

        thread = threading.Thread(target=producer, name="panoptic_moe_array_prefetch_producer", daemon=True)
        thread.start()
        try:
            while True:
                item = queue.get()
                if item is sentinel:
                    break
                yield item
        finally:
            stop_event.set()
            executor.shutdown(wait=False, cancel_futures=True)
        return

    sentinel = object()
    queue: Queue[Any] = Queue(maxsize=max(1, int(max_prefetch)))
    worker_rng = random.Random()
    worker_rng.setstate(rng.getstate())

    def worker() -> None:
        try:
            for record in source:
                try:
                    arrays = load_panoptic_target_arrays(record, max_tokens, training=True, num_queries=num_queries)
                    queue.put((record, arrays, worker_rng.getstate(), None, None))
                except Exception as exc:  # noqa: BLE001 - propagate worker parsing failures to the training thread.
                    queue.put((record, None, worker_rng.getstate(), type(exc).__name__, str(exc)))
                    break
        finally:
            queue.put(sentinel)

    thread = threading.Thread(target=worker, name="panoptic_moe_array_prefetch", daemon=True)
    thread.start()
    while True:
        item = queue.get()
        if item is sentinel:
            break
        yield item


def _distribution_stats(values: Any, names: tuple[str, ...]) -> dict[str, Any]:
    if values is None or getattr(values, "size", 0) == 0:
        return {
            "rows": 0,
            "feature_dim": len(names),
            "zero_std_columns": [],
            "low_nonzero_columns": [],
            "per_column": [],
        }
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        array = array.reshape(-1, array.shape[-1])
    mean = array.mean(axis=0)
    std = array.std(axis=0)
    nonzero = (np.abs(array) > 1e-8).mean(axis=0)
    per_column = [
        {
            "index": index,
            "name": names[index] if index < len(names) else f"feature_{index}",
            "mean": float(mean[index]),
            "std": float(std[index]),
            "nonzero_fraction": float(nonzero[index]),
        }
        for index in range(array.shape[1])
    ]
    return {
        "rows": int(array.shape[0]),
        "feature_dim": int(array.shape[1]),
        "zero_std_columns": [row["name"] for row in per_column if row["std"] < 1e-8],
        "low_nonzero_columns": [row["name"] for row in per_column if row["nonzero_fraction"] < 1e-6],
        "per_column": per_column,
    }


def feature_ingress_audit(
    path: Path,
    *,
    max_tokens: int,
    num_queries: int,
    feature_names: tuple[str, ...],
    input_protocol: dict[str, Any],
    limit_records: int = 8,
    fail_closed: bool = True,
) -> dict[str, Any]:
    """Verify the cache feature contract before training can hide a mismatch."""
    primitive_rows: list[Any] = []
    segment_rows: list[Any] = []
    records = 0
    segment_records = 0
    target_schemas = Counter()
    input_schemas = Counter()
    segment_valid = 0
    segment_total = 0
    blockers: list[str] = []
    for record in iter_jsonl(path, max(1, int(limit_records))):
        records += 1
        target_schemas[str(record.get("target_schema_version"))] += 1
        input_schemas[str(record.get("input_schema_version"))] += 1
        arrays = load_panoptic_target_arrays(record, max_tokens, training=True, num_queries=num_queries)
        if arrays is None:
            blockers.append(f"record_{records}_unloadable")
            continue
        x_np = arrays[0]
        segment_features_np = arrays[8] if len(arrays) > 8 else None
        segment_padding_np = arrays[9] if len(arrays) > 9 else None
        if x_np.shape[-1] != len(feature_names):
            blockers.append(f"feature_dim_mismatch:{x_np.shape[-1]}!={len(feature_names)}")
        primitive_rows.append(x_np)
        if segment_features_np is not None and segment_padding_np is not None:
            segment_records += 1
            valid = ~segment_padding_np.astype(bool)
            segment_valid += int(valid.sum())
            segment_total += int(valid.size)
            if valid.any():
                segment_rows.append(segment_features_np[valid])
            if segment_features_np.shape[-1] != len(feature_names):
                blockers.append(f"segment_feature_dim_mismatch:{segment_features_np.shape[-1]}!={len(feature_names)}")
    primitive = np.concatenate(primitive_rows, axis=0) if primitive_rows else np.empty((0, len(feature_names)), dtype=np.float32)
    segments = np.concatenate(segment_rows, axis=0) if segment_rows else np.empty((0, len(feature_names)), dtype=np.float32)
    expected_input_schema = str(input_protocol.get("input_schema_version"))
    expected_target_schema = str(input_protocol.get("target_schema_version"))
    if records <= 0:
        blockers.append("empty_cache")
    if target_schemas and set(target_schemas) != {expected_target_schema}:
        blockers.append(f"target_schema_mismatch:{dict(target_schemas)}")
    if input_schemas and set(input_schemas) != {expected_input_schema}:
        blockers.append(f"input_schema_mismatch:{dict(input_schemas)}")
    if bool(input_protocol.get("segment_features")) and segment_records != records:
        blockers.append(f"segment_features_missing:{segment_records}/{records}")
    if bool(input_protocol.get("segment_features")) and segment_valid <= 0:
        blockers.append("segment_valid_coverage_zero")
    primitive_stats = _distribution_stats(primitive, feature_names)
    segment_stats_payload = _distribution_stats(segments, feature_names)
    if fail_closed:
        if expected_input_schema.endswith("_raw_semantic_segments"):
            required_nonconstant = {"primitive_kind_path", "path_cmd_line", "segment_order_norm", "tangent_dx", "tangent_dy", "same_layer_fraction"}
            zero_std = set(primitive_stats["zero_std_columns"]) | set(segment_stats_payload["zero_std_columns"])
            missing_signal = sorted(required_nonconstant & zero_std)
            if missing_signal:
                blockers.append(f"raw_semantic_feature_signal_missing:{missing_signal}")
    return {
        "schema_version": "floorplancad_feature_ingress_audit_v1",
        "path": rel(path),
        "records": records,
        "input_protocol": input_protocol,
        "feature_schema_sha256": feature_schema_sha256(feature_names),
        "feature_names": list(feature_names),
        "target_schema_counts": dict(target_schemas),
        "input_schema_counts": dict(input_schemas),
        "segment_enabled": bool(segment_records),
        "segment_record_coverage": segment_records / max(records, 1),
        "segment_valid_ratio": segment_valid / max(segment_total, 1),
        "primitive_stats": primitive_stats,
        "segment_stats": segment_stats_payload,
        "passed": not blockers,
        "blockers": blockers,
    }


def feature_sensitivity_smoke(
    model: Any,
    torch: Any,
    path: Path,
    device: Any,
    *,
    max_tokens: int,
    num_queries: int,
    feature_names: tuple[str, ...],
    sample_tokens: int = 256,
) -> dict[str, Any]:
    """Check that primitive/segment feature groups affect model outputs."""
    record = next(iter(iter_jsonl(path, 1)), None)
    if record is None:
        return {"schema_version": "floorplancad_feature_sensitivity_smoke_v1", "passed": False, "blockers": ["empty_cache"]}
    arrays = load_panoptic_target_arrays(record, max_tokens, training=True, num_queries=num_queries)
    if arrays is None:
        return {"schema_version": "floorplancad_feature_sensitivity_smoke_v1", "passed": False, "blockers": ["unloadable_record"]}
    x_np = arrays[0][: int(sample_tokens)]
    segment_features_np = arrays[8][: int(sample_tokens)] if arrays[8] is not None else None
    segment_padding_np = arrays[9][: int(sample_tokens)] if arrays[9] is not None else None
    model_was_training = bool(model.training)
    model.eval()
    blockers: list[str] = []
    with torch.no_grad():
        x = torch.from_numpy(x_np).to(device).unsqueeze(0)
        segment_features = None if segment_features_np is None else torch.from_numpy(segment_features_np).to(device).unsqueeze(0)
        segment_padding = None if segment_padding_np is None else torch.from_numpy(segment_padding_np).to(device).unsqueeze(0)
        base_outputs = model(
            x,
            segment_features=segment_features,
            segment_padding_mask=segment_padding,
            return_quality=True,
            return_identity=True,
        )[:4]
        extra_columns = [
            index
            for index, name in enumerate(feature_names)
            if index >= len(V3_FEATURE_NAMES) or name.startswith(("rgb_", "layer_", "same_layer", "page_"))
        ]
        if extra_columns:
            zero_extra_x = x.clone()
            zero_extra_x[..., extra_columns] = 0
            zero_extra_segments = None
            if segment_features is not None:
                zero_extra_segments = segment_features.clone()
                zero_extra_segments[..., extra_columns] = 0
            extra_outputs = model(
                zero_extra_x,
                segment_features=zero_extra_segments,
                segment_padding_mask=segment_padding,
                return_quality=True,
                return_identity=True,
            )[:4]
        else:
            extra_outputs = base_outputs
        if segment_features is not None:
            no_segment_outputs = model(
                x,
                segment_features=None,
                segment_padding_mask=None,
                return_quality=True,
                return_identity=True,
            )[:4]
        else:
            no_segment_outputs = base_outputs
    if model_was_training:
        model.train()
    names = ("semantic", "query", "mask", "quality")
    zero_extra_delta = {
        name: float((base.detach().float() - changed.detach().float()).abs().mean().item())
        for name, base, changed in zip(names, base_outputs, extra_outputs, strict=True)
    }
    no_segment_delta = {
        name: float((base.detach().float() - changed.detach().float()).abs().mean().item())
        for name, base, changed in zip(names, base_outputs, no_segment_outputs, strict=True)
    }
    if extra_columns and max(zero_extra_delta.values(), default=0.0) <= 1e-7:
        blockers.append("zero_extra_features_no_output_delta")
    if segment_features is not None and max(no_segment_delta.values(), default=0.0) <= 1e-7:
        blockers.append("remove_segment_features_no_output_delta")
    return {
        "schema_version": "floorplancad_feature_sensitivity_smoke_v1",
        "path": rel(path),
        "sample_tokens": int(x_np.shape[0]),
        "extra_feature_columns": [feature_names[index] for index in extra_columns],
        "zero_extra_mean_abs_delta": zero_extra_delta,
        "remove_segment_features_mean_abs_delta": no_segment_delta,
        "passed": not blockers,
        "blockers": blockers,
    }


def atomic_torch_save(torch: Any, payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def prune_checkpoint_archives(directory: Path, keep: int) -> list[str]:
    if keep <= 0 or not directory.exists():
        return []
    archives = sorted(directory.glob("*.pt"), key=lambda item: item.stat().st_mtime, reverse=True)
    kept = archives[:keep]
    for stale in archives[keep:]:
        try:
            stale.unlink()
        except OSError:
            pass
    return [rel(path) or str(path) for path in kept]


def save_top_k_diagnostic_checkpoint(
    torch: Any,
    payload: dict[str, Any],
    directory: Path | None,
    *,
    score: float,
    keep: int,
) -> list[str]:
    """Preserve inspectable peak epochs independently of promotion gates."""
    if directory is None or int(keep) <= 0:
        return []
    directory.mkdir(parents=True, exist_ok=True)
    epoch = int(payload.get("epoch", 0))
    path = directory / f"epoch{epoch:04d}.pt"
    diagnostic_payload = dict(payload)
    diagnostic_payload["diagnostic_selection_score"] = float(score)
    diagnostic_payload["checkpoint_boundary"] = "top_k_diagnostic_checkpoint"
    atomic_torch_save(torch, diagnostic_payload, path)
    ranked: list[tuple[float, Path]] = []
    for candidate in directory.glob("epoch*.pt"):
        try:
            candidate_payload = torch.load(candidate, map_location="cpu", weights_only=False)
            candidate_score = float(candidate_payload.get("diagnostic_selection_score", candidate_payload.get("selection_score", -float("inf"))))
        except Exception:  # noqa: BLE001 - a damaged diagnostic artifact must not stop training.
            candidate_score = -float("inf")
        ranked.append((candidate_score if math.isfinite(candidate_score) else -float("inf"), candidate))
    ranked.sort(key=lambda item: (item[0], item[1].name), reverse=True)
    for _candidate_score, stale in ranked[int(keep):]:
        stale.unlink(missing_ok=True)
    return [rel(candidate) or str(candidate) for _candidate_score, candidate in ranked[: int(keep)]]


def checkpoint_state_dict(model: Any) -> Any:
    source = getattr(model, "_orig_mod", model)
    return source.state_dict()


def amp_dtype_from_arg(torch: Any, device: Any, amp: str) -> Any | None:
    if device.type != "cuda" or amp == "off":
        return None
    if amp == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 AMP requested but CUDA device does not support bfloat16")
        return torch.bfloat16
    if amp == "fp16":
        return torch.float16
    raise ValueError(f"unsupported amp mode: {amp}")


def autocast_context(torch: Any, device: Any, dtype: Any | None) -> Any:
    if device.type == "cuda" and dtype is not None:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def cuda_memory_summary(torch: Any, device: Any) -> dict[str, Any]:
    if getattr(device, "type", None) != "cuda":
        return {"enabled": False}
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    except Exception as exc:  # noqa: BLE001 - runtime telemetry must not break training.
        return {"enabled": False, "error": type(exc).__name__, "message": str(exc)}
    props = torch.cuda.get_device_properties(device)
    return {
        "enabled": True,
        "device": str(device),
        "name": getattr(props, "name", None),
        "total_memory_mib": int(total_bytes // (1024 * 1024)),
        "free_memory_mib": int(free_bytes // (1024 * 1024)),
        "allocated_mib": int(torch.cuda.memory_allocated(device) // (1024 * 1024)),
        "reserved_mib": int(torch.cuda.memory_reserved(device) // (1024 * 1024)),
    }


def apply_auto_throughput_profile(args: argparse.Namespace, torch: Any, device: Any) -> dict[str, Any]:
    requested = {
        "batch_records": int(args.batch_records),
        "train_prefetch_records": int(args.train_prefetch_records),
        "train_prefetch_workers": int(args.train_prefetch_workers),
        "progress_checkpoint_records": int(args.progress_checkpoint_records),
        "amp": args.amp,
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "geometry_attention_tile_size": int(args.geometry_attention_tile_size),
        "train_component_matching": getattr(args, "train_component_matching", None),
    }
    memory = cuda_memory_summary(torch, device)
    applied = False
    launch_blocked = False
    reason = "disabled"
    if getattr(device, "type", None) == "cuda" and args.auto_throughput_profile != "off":
        total_mib = int(memory.get("total_memory_mib") or 0)
        if args.auto_throughput_profile == "gpu_32gb_safe" and 24000 <= total_mib <= 40000:
            free_mib = int(memory.get("free_memory_mib") or 0)
            args.batch_records = min(int(args.batch_records), int(args.auto_profile_32gb_max_batch_records))
            args.train_prefetch_records = min(
                max(int(args.train_prefetch_records), 1), int(args.auto_profile_32gb_prefetch_records)
            )
            args.train_prefetch_workers = min(
                max(int(args.train_prefetch_workers), 1), int(args.auto_profile_32gb_prefetch_workers)
            )
            args.gradient_checkpointing = True
            if getattr(args, "train_component_matching", None) is None:
                args.train_component_matching = "greedy_gpu_train"
            if int(args.geometry_attention_tile_size) == 0:
                args.geometry_attention_tile_size = int(args.auto_profile_32gb_geometry_attention_tile_size)
            if args.amp == "off":
                args.amp = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
            applied = True
            required_free_mib = int(args.auto_profile_32gb_min_free_mib)
            if free_mib < required_free_mib:
                launch_blocked = True
                reason = f"cuda_32gb_insufficient_free_memory:{free_mib}<{required_free_mib}_mib"
            else:
                reason = "cuda_32gb_memory_safe_profile"
        elif args.auto_throughput_profile in {"gpu0_high_memory", "gpu0_full_memory"} and total_mib >= int(args.auto_profile_min_total_mib):
            free_mib = int(memory.get("free_memory_mib") or 0)
            target_batch = int(args.auto_profile_batch_records)
            target_prefetch = int(args.auto_profile_prefetch_records)
            target_workers = int(args.auto_profile_prefetch_workers)
            target_progress_checkpoint = int(args.auto_profile_progress_checkpoint_records)
            if args.auto_throughput_profile == "gpu0_full_memory" and free_mib >= int(args.auto_profile_full_free_mib):
                target_batch = max(target_batch, int(args.auto_profile_full_batch_records))
                target_prefetch = max(target_prefetch, int(args.auto_profile_full_prefetch_records))
                target_workers = max(target_workers, int(args.auto_profile_full_prefetch_workers))
                target_progress_checkpoint = max(target_progress_checkpoint, int(args.auto_profile_full_progress_checkpoint_records))
            args.batch_records = max(int(args.batch_records), target_batch)
            args.train_prefetch_records = max(int(args.train_prefetch_records), target_prefetch)
            args.train_prefetch_workers = max(int(args.train_prefetch_workers), target_workers)
            args.progress_checkpoint_records = max(int(args.progress_checkpoint_records), target_progress_checkpoint)
            if args.amp == "off":
                args.amp = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
            applied = True
            reason = "cuda_full_memory_profile" if args.auto_throughput_profile == "gpu0_full_memory" else "cuda_high_memory_profile"
        else:
            reason = f"cuda_total_memory_below_{args.auto_profile_min_total_mib}_mib"
    return {
        "enabled": args.auto_throughput_profile != "off",
        "profile": args.auto_throughput_profile,
        "applied": applied,
        "launch_blocked": launch_blocked,
        "reason": reason,
        "requested": requested,
        "effective": {
            "batch_records": int(args.batch_records),
            "train_prefetch_records": int(args.train_prefetch_records),
            "train_prefetch_workers": int(args.train_prefetch_workers),
            "progress_checkpoint_records": int(args.progress_checkpoint_records),
            "amp": args.amp,
            "gradient_checkpointing": bool(args.gradient_checkpointing),
            "geometry_attention_tile_size": int(args.geometry_attention_tile_size),
            "train_component_matching": getattr(args, "train_component_matching", None),
        },
        "cuda_memory_before_profile": memory,
    }


def maybe_pin_empty(torch: Any, shape: tuple[int, ...], *, dtype: Any, pin: bool) -> Any:
    if pin:
        return torch.empty(shape, dtype=dtype, pin_memory=True)
    return torch.empty(shape, dtype=dtype)


def maybe_pin_full(torch: Any, shape: tuple[int, ...], fill_value: Any, *, dtype: Any, pin: bool) -> Any:
    tensor = maybe_pin_empty(torch, shape, dtype=dtype, pin=pin)
    tensor.fill_(fill_value)
    return tensor


def augment_geometry_features(
    torch: Any,
    features: Any,
    *,
    scale_min: float,
    scale_max: float,
    translation: float,
    parameters: dict[str, Any] | None = None,
) -> Any:
    """Apply one topology-preserving affine transform per page to line features."""
    if features.ndim not in {3, 4} or features.shape[-1] < 11:
        raise ValueError("features must be [B,N,F] or [B,N,S,F] with geometric columns")
    if not 0.0 < scale_min <= scale_max:
        raise ValueError("geometry augmentation scale range is invalid")
    if translation < 0.0:
        raise ValueError("geometry augmentation translation must be non-negative")
    batch = features.shape[0]
    if parameters is None:
        parameters = {
            "flip_x": torch.rand((batch, 1, 1), device=features.device) < 0.5,
            "flip_y": torch.rand((batch, 1, 1), device=features.device) < 0.5,
            "rotations": torch.randint(0, 4, (batch, 1, 1), device=features.device),
            "scale": torch.empty((batch, 1, 1), device=features.device).uniform_(float(scale_min), float(scale_max)),
            "shift": torch.empty((batch, 1, 1, 2), device=features.device).uniform_(-float(translation), float(translation)),
        }
    required = {"flip_x", "flip_y", "rotations", "scale", "shift"}
    if required.difference(parameters):
        raise ValueError("geometry augmentation parameters are incomplete")
    result = features.clone()
    flat = result.reshape(batch, -1, result.shape[-1])
    points = flat[..., :4].reshape(batch, -1, 2, 2)
    flip_x = parameters["flip_x"]
    flip_y = parameters["flip_y"]
    points[..., 0] = torch.where(flip_x, 1.0 - points[..., 0], points[..., 0])
    points[..., 1] = torch.where(flip_y, 1.0 - points[..., 1], points[..., 1])
    rotations = parameters["rotations"]
    centered = points - 0.5
    x_coord, y_coord = centered[..., 0], centered[..., 1]
    rotated_x = torch.where(rotations == 1, -y_coord, torch.where(rotations == 2, -x_coord, torch.where(rotations == 3, y_coord, x_coord)))
    rotated_y = torch.where(rotations == 1, x_coord, torch.where(rotations == 2, -y_coord, torch.where(rotations == 3, -x_coord, y_coord)))
    scale = parameters["scale"]
    shift = parameters["shift"]
    points = torch.stack([rotated_x, rotated_y], dim=-1) * scale.unsqueeze(-1) + 0.5 + shift
    points = points.clamp(0.0, 1.0)
    flat[..., :4] = points.reshape(batch, -1, 4)
    delta = points[..., 1, :] - points[..., 0, :]
    flat[..., 4:6] = points.mean(dim=-2)
    orientation = torch.atan2(delta[..., 1], delta[..., 0]) / math.pi
    flat[..., 8] = torch.where(orientation < 0.0, orientation + 1.0, orientation)
    flat[..., 9] = (delta[..., 0].abs() >= delta[..., 1].abs()).to(flat.dtype)
    flat[..., 10] = (delta[..., 1].abs() > delta[..., 0].abs()).to(flat.dtype)
    if flat.shape[-1] > 29:
        tangent_x = flat[..., 28]
        tangent_y = flat[..., 29]
        tangent_x = torch.where(flip_x.squeeze(-1), -tangent_x, tangent_x)
        tangent_y = torch.where(flip_y.squeeze(-1), -tangent_y, tangent_y)
        rotation_index = rotations.squeeze(-1)
        rotated_tangent_x = torch.where(
            rotation_index == 1,
            -tangent_y,
            torch.where(rotation_index == 2, -tangent_x, torch.where(rotation_index == 3, tangent_y, tangent_x)),
        )
        rotated_tangent_y = torch.where(
            rotation_index == 1,
            tangent_x,
            torch.where(rotation_index == 2, -tangent_y, torch.where(rotation_index == 3, -tangent_x, tangent_y)),
        )
        tangent_norm = torch.sqrt(rotated_tangent_x.square() + rotated_tangent_y.square()).clamp_min(1e-12)
        flat[..., 28] = rotated_tangent_x / tangent_norm
        flat[..., 29] = rotated_tangent_y / tangent_norm
    return result


def augment_candidate_descriptor_features(torch: Any, candidate_features: Any, parameters: dict[str, Any]) -> Any:
    """Apply the same page-level geometry transform to candidate bbox/center descriptors."""
    if candidate_features.ndim != 3 or candidate_features.shape[-1] < 7:
        raise ValueError("candidate_features must be [B,C,F] with bbox/center descriptor columns")
    required = {"flip_x", "flip_y", "rotations", "scale", "shift"}
    if required.difference(parameters):
        raise ValueError("geometry augmentation parameters are incomplete")
    result = candidate_features.clone()
    bbox = result[..., 1:5].reshape(result.shape[0], result.shape[1], 2, 2)
    x1, y1, x2, y2 = result[..., 1], result[..., 2], result[..., 3], result[..., 4]
    corners = torch.stack(
        [
            torch.stack([x1, y1], dim=-1),
            torch.stack([x1, y2], dim=-1),
            torch.stack([x2, y1], dim=-1),
            torch.stack([x2, y2], dim=-1),
            result[..., 5:7],
        ],
        dim=-2,
    )
    flip_x = parameters["flip_x"]
    flip_y = parameters["flip_y"]
    points = corners
    points[..., 0] = torch.where(flip_x, 1.0 - points[..., 0], points[..., 0])
    points[..., 1] = torch.where(flip_y, 1.0 - points[..., 1], points[..., 1])
    rotations = parameters["rotations"]
    centered = points - 0.5
    x_coord, y_coord = centered[..., 0], centered[..., 1]
    rotated_x = torch.where(rotations == 1, -y_coord, torch.where(rotations == 2, -x_coord, torch.where(rotations == 3, y_coord, x_coord)))
    rotated_y = torch.where(rotations == 1, x_coord, torch.where(rotations == 2, -y_coord, torch.where(rotations == 3, -x_coord, y_coord)))
    transformed = torch.stack([rotated_x, rotated_y], dim=-1) * parameters["scale"].unsqueeze(-1) + 0.5 + parameters["shift"]
    transformed = transformed.clamp(0.0, 1.0)
    transformed_bbox = transformed[..., :4, :]
    mins = transformed_bbox.amin(dim=-2)
    maxs = transformed_bbox.amax(dim=-2)
    result[..., 1] = mins[..., 0]
    result[..., 2] = mins[..., 1]
    result[..., 3] = maxs[..., 0]
    result[..., 4] = maxs[..., 1]
    result[..., 5:7] = transformed[..., 4, :]
    result[..., 7] = ((maxs[..., 0] - mins[..., 0]).clamp_min(0.0) * (maxs[..., 1] - mins[..., 1]).clamp_min(0.0))
    return result


def unique_page_semantic_class_weights(
    torch: Any,
    path: Path,
    *,
    limit: int | None,
    beta: float = 0.999,
    minimum: float = 0.25,
    maximum: float = 4.0,
) -> Any:
    """Compute smoothed class weights once per original page primitive."""
    if not 0.0 < beta < 1.0 or minimum <= 0.0 or maximum < minimum:
        raise ValueError("invalid effective-number class-weight configuration")
    counts = np.zeros(IGNORE_LABEL, dtype=np.float64)
    seen: set[tuple[str, int]] = set()
    for record in iter_jsonl(path, limit):
        page_id = str(record.get("original_record_id") or record.get("record_id") or "")
        for row in record.get("primitive_rows", []) if isinstance(record.get("primitive_rows"), list) else []:
            primitive_id = parse_int(row.get("primitive_id"), -1)
            label = parse_int(row.get("semantic_id"), IGNORE_LABEL)
            key = (page_id, primitive_id)
            if primitive_id < 0 or label < 0 or label >= IGNORE_LABEL or key in seen:
                continue
            seen.add(key)
            counts[label] += max(parse_float(row.get("log1p_primitive_length"), 1.0), 1e-6)
    effective = 1.0 - np.power(float(beta), counts)
    weights = np.divide(1.0 - float(beta), effective, out=np.zeros_like(effective), where=effective > 0.0)
    positive = weights > 0.0
    if not np.any(positive):
        return torch.ones(IGNORE_LABEL, dtype=torch.float32)
    weights[positive] /= weights[positive].mean()
    weights = np.clip(weights, float(minimum), float(maximum))
    return torch.as_tensor(weights, dtype=torch.float32)


def make_panoptic_model(
    nn: Any,
    torch: Any,
    feature_dim: int,
    hidden_dim: int,
    layers: int,
    heads: int,
    num_queries: int,
    num_labels: int = 36,
    query_decoder_layers: int = 3,
    dropout: float = 0.1,
    position_encoding_version: str = POSITION_ENCODING_VERSION,
    identity_dim: int = PANOPTIC_IDENTITY_DIM,
    geometry_decoder_mode: str = "legacy_debug",
    num_stuff_queries: int = 32,
    geometry_local_neighbors: int = 4,
    geometry_coarse_grid_size: int = 4,
    sq_rq_enabled: bool = False,
    sq_rq_gradient_scale: float = 0.0,
    sq_rq_query_confidence_threshold: float = 0.6,
    sq_rq_token_membership_threshold: float = 0.5,
    sq_rq_training_membership_temperature: float = 0.1,
    ownership_enabled: bool = True,
    learned_sparse_router: bool = False,
    router_num_experts: int = 4,
    router_top_k: int = 2,
    router_temperature: float = 1.0,
    typed_branch_routers: bool = False,
    branch_num_experts: int = 2,
    branch_top_k: int = 1,
    branch_capacity_factor: float = 1.25,
    branch_dropless: bool = False,
    typed_stuff_slots: bool = False,
    semantic_query_residual_enabled: bool = True,
    geometry_attention_tile_size: int = 0,
    tensor_ring_rank: int = 0,
    gradient_checkpointing: bool = False,
    content_seeded_queries: bool = False,
    repeated_group_fusion: bool = False,
    relation_bias_enabled: bool = False,
    component_seeded_queries: bool = False,
    offset_vote_enabled: bool = False,
    candidate_aware_queries: bool = False,
    candidate_feature_dim: int = 0,
    candidate_mask_prior_logit: float = 0.0,
    weak_family_feature_fusion: bool = False,
    quality_query_gradient_scale: float = 0.0,
    explicit_route_classifier: bool = False,
    dense_attention_feature_adapter: bool = False,
    dense_attention_window_size: int = 128,
    route_conditioning_residual_scale: float = 0.10,
    dense_attention_adapter_residual_scale: float = 0.10,
) -> Any:
    if geometry_decoder_mode not in {"legacy_debug", "geometry_v2"}:
        raise ValueError("geometry_decoder_mode must be legacy_debug or geometry_v2")
    router_config = sparse_router_config(
        enabled=learned_sparse_router, hidden_dim=hidden_dim, num_experts=router_num_experts,
        top_k=router_top_k, temperature=router_temperature,
        typed_branch_routers=typed_branch_routers, branch_num_experts=branch_num_experts,
        branch_top_k=branch_top_k,
        branch_capacity_factor=branch_capacity_factor,
        branch_dropless=branch_dropless,
    )
    class TensorRingLinear(nn.Module):
        """Tensor-ring factorized linear map used for geometry projections."""
        def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True) -> None:
            super().__init__()
            self.in_features, self.out_features = int(in_features), int(out_features)
            self.in_factors, self.out_factors = _balanced_factors(in_features), _balanced_factors(out_features)
            if len(self.in_factors) != len(self.out_factors):
                modes = max(len(self.in_factors), len(self.out_factors))
                self.in_factors += [1] * (modes - len(self.in_factors))
                self.out_factors += [1] * (modes - len(self.out_factors))
            self.rank = int(rank)
            if self.rank < 1:
                raise ValueError("tensor ring rank must be positive")
            self.cores = nn.ParameterList([
                nn.Parameter(torch.randn(self.rank, in_mode, out_mode, self.rank) * (1.0 / math.sqrt(self.rank * in_mode)))
                for in_mode, out_mode in zip(self.in_factors, self.out_factors)
            ])
            self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        def _weight(self) -> Any:
            tensor = self.cores[0]
            for core in self.cores[1:]:
                tensor = torch.einsum("a i o b, b j p c -> a i j o p c", tensor, core)
                tensor = tensor.reshape(tensor.shape[0], tensor.shape[1] * tensor.shape[2], tensor.shape[3] * tensor.shape[4], tensor.shape[5])
            weight = tensor.diagonal(dim1=0, dim2=3).sum(-1)
            return weight.reshape(self.in_features, self.out_features).transpose(0, 1)

        def forward(self, inputs: Any) -> Any:
            return torch.nn.functional.linear(inputs, self._weight(), self.bias)

    def projection(in_features: int, out_features: int, *, bias: bool = True) -> Any:
        if int(tensor_ring_rank) > 0:
            return TensorRingLinear(in_features, out_features, int(tensor_ring_rank), bias=bias)
        return nn.Linear(in_features, out_features, bias=bias)

    class QueryDecoderLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = nn.MultiheadAttention(hidden_dim, heads, dropout=float(dropout), batch_first=True)
            self.cross_attn = nn.MultiheadAttention(hidden_dim, heads, dropout=float(dropout), batch_first=True)
            self.ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(float(dropout)), nn.Linear(hidden_dim * 4, hidden_dim))
            self.norm_self = nn.LayerNorm(hidden_dim)
            self.norm_cross = nn.LayerNorm(hidden_dim)
            self.norm_ffn = nn.LayerNorm(hidden_dim)

        def forward(self, queries: Any, tokens: Any, token_padding_mask: Any | None = None) -> Any:
            residual = queries
            q_self, _ = self.self_attn(queries, queries, queries, need_weights=False)
            queries = self.norm_self(residual + q_self)
            residual = queries
            q_cross, _ = self.cross_attn(queries, tokens, tokens, key_padding_mask=token_padding_mask, need_weights=False)
            queries = self.norm_cross(residual + q_cross)
            residual = queries
            queries = self.norm_ffn(residual + self.ffn(queries))
            return queries

    class RelativeGeometryDecoderLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.heads = int(heads)
            self.head_dim = hidden_dim // int(heads)
            if hidden_dim % int(heads):
                raise ValueError("hidden_dim must be divisible by heads")
            self.self_attn = nn.MultiheadAttention(hidden_dim, heads, dropout=float(dropout), batch_first=True)
            self.q_proj, self.k_proj, self.v_proj = projection(hidden_dim, hidden_dim), projection(hidden_dim, hidden_dim), projection(hidden_dim, hidden_dim)
            self.out_proj = projection(hidden_dim, hidden_dim)
            self.geometry_bias = nn.Sequential(nn.Linear(5, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, heads))
            self.ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(float(dropout)), nn.Linear(hidden_dim * 4, hidden_dim))
            self.norm_self, self.norm_cross, self.norm_ffn = nn.LayerNorm(hidden_dim), nn.LayerNorm(hidden_dim), nn.LayerNorm(hidden_dim)

        def _streaming_cross_attention(self, projected_q: Any, projected_k: Any, projected_v: Any,
                                       query_positions: Any, memory_positions: Any,
                                       memory_padding: Any, tile_size: int) -> Any:
            """Exact softmax attention without materialising the full BxHxQxM tensor."""
            batch, heads, query_count, _ = projected_q.shape
            memory_count = projected_k.shape[2]
            scale = math.sqrt(float(self.head_dim))
            running_max = projected_q.new_full((batch, heads, query_count), -float("inf"), dtype=torch.float32)
            running_sum = projected_q.new_zeros((batch, heads, query_count), dtype=torch.float32)
            running_value = projected_q.new_zeros((batch, heads, query_count, self.head_dim), dtype=torch.float32)
            for start in range(0, memory_count, tile_size):
                stop = min(start + tile_size, memory_count)
                key_tile = projected_k[:, :, start:stop]
                value_tile = projected_v[:, :, start:stop]
                relative = memory_positions[:, None, start:stop] - query_positions[:, :, None]
                distance = torch.linalg.vector_norm(relative, dim=-1, keepdim=True)
                bias = self.geometry_bias(torch.cat([relative, relative.abs(), torch.log1p(distance)], dim=-1)).permute(0, 3, 1, 2)
                scores = torch.einsum("bhqd,bhmd->bhqm", projected_q, key_tile) / scale + bias
                scores = scores.masked_fill(memory_padding[:, None, None, start:stop], -1e4)
                scores_float = scores.float()
                tile_max = scores_float.max(dim=-1).values
                next_max = torch.maximum(running_max, tile_max)
                old_scale = torch.exp(running_max - next_max)
                tile_exp = torch.exp(scores_float - next_max.unsqueeze(-1))
                running_sum = old_scale * running_sum + tile_exp.sum(dim=-1)
                running_value = old_scale.unsqueeze(-1) * running_value + torch.einsum("bhqm,bhmd->bhqd", tile_exp, value_tile.float())
                running_max = next_max
            cross = (running_value / running_sum.clamp_min(1e-12).unsqueeze(-1)).to(projected_q.dtype)
            return cross.transpose(1, 2).reshape(batch, query_count, hidden_dim)

        def forward(self, queries: Any, query_positions: Any, memory: Any, memory_positions: Any, memory_padding: Any) -> tuple[Any, Any]:
            attended, _ = self.self_attn(queries, queries, queries, need_weights=False)
            queries = self.norm_self(queries + attended)
            batch, query_count, _ = queries.shape
            memory_count = memory.shape[1]
            projected_q = self.q_proj(queries).reshape(batch, query_count, self.heads, self.head_dim).transpose(1, 2)
            projected_k = self.k_proj(memory).reshape(batch, memory_count, self.heads, self.head_dim).transpose(1, 2)
            projected_v = self.v_proj(memory).reshape(batch, memory_count, self.heads, self.head_dim).transpose(1, 2)
            if geometry_attention_tile_size > 0 and geometry_attention_tile_size < memory_count:
                cross = self._streaming_cross_attention(
                    projected_q, projected_k, projected_v, query_positions, memory_positions,
                    memory_padding, int(geometry_attention_tile_size),
                )
                weights = None
            else:
                content = torch.einsum("bhqd,bhmd->bhqm", projected_q, projected_k) / math.sqrt(float(self.head_dim))
                relative = memory_positions[:, None] - query_positions[:, :, None]
                distance = torch.linalg.vector_norm(relative, dim=-1, keepdim=True)
                bias = self.geometry_bias(torch.cat([relative, relative.abs(), torch.log1p(distance)], dim=-1)).permute(0, 3, 1, 2)
                logits = (content + bias).masked_fill(memory_padding[:, None, None, :], -1e4)
                weights = torch.softmax(logits, dim=-1)
                cross = torch.einsum("bhqm,bhmd->bhqd", weights, projected_v).transpose(1, 2).reshape(batch, query_count, hidden_dim)
            queries = self.norm_cross(queries + self.out_proj(cross))
            return self.norm_ffn(queries + self.ffn(queries)), weights

    class LineTokenPanopticMoE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_proj = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
            self.segment_encoder = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            )
            segment_context_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=heads,
                dim_feedforward=hidden_dim * 2,
                dropout=float(dropout),
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.segment_context_encoder = nn.TransformerEncoder(
                segment_context_layer,
                num_layers=1,
                enable_nested_tensor=False,
            )
            self.segment_pool_score = nn.Linear(hidden_dim, 1)
            self.segment_aggregate_encoder = nn.Sequential(
                nn.Linear(10, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            )
            self.segment_fusion_gate = nn.Linear(hidden_dim * 3, hidden_dim)
            self.segment_fusion_norm = nn.LayerNorm(hidden_dim)
            self.page_global_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
            self.layer_context_proj = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
            self.context_fusion_gate = nn.Linear(hidden_dim * 3, hidden_dim)
            self.context_fusion_norm = nn.LayerNorm(hidden_dim)
            self.repeated_group_fusion = bool(repeated_group_fusion)
            self.relation_bias_enabled = bool(relation_bias_enabled)
            if self.repeated_group_fusion:
                relation_feature_dim = min(int(feature_dim), 18)
                self.relation_feature_proj = nn.Sequential(
                    nn.Linear(relation_feature_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
                )
                self.relation_fusion_gate = nn.Linear(hidden_dim * 3, hidden_dim)
                self.relation_fusion_norm = nn.LayerNorm(hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=heads,
                dim_feedforward=hidden_dim * 4,
                dropout=float(dropout),
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=layers,
                enable_nested_tensor=False,
            )
            self.gradient_checkpointing = bool(gradient_checkpointing)
            self.learned_sparse_router = bool(learned_sparse_router)
            self.router_temperature = float(router_temperature)
            self.router_top_k = int(router_top_k)
            self.typed_branch_routers = bool(typed_branch_routers)
            self.branch_num_experts = int(branch_num_experts)
            self.branch_top_k = int(branch_top_k)
            self.branch_capacity_factor = float(branch_capacity_factor)
            if self.learned_sparse_router:
                self.sparse_router = nn.Linear(hidden_dim, int(router_num_experts))
                self.sparse_experts = nn.ModuleList(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Dropout(float(dropout)),
                        nn.Linear(hidden_dim * 2, hidden_dim),
                    )
                    for _ in range(int(router_num_experts))
                )
                self.sparse_router_norm = nn.LayerNorm(hidden_dim)
            if self.typed_branch_routers:
                self.branch_routers = nn.ModuleDict()
                for branch_name in ("semantic", "rq", "sq", "bridge"):
                    self.branch_routers[branch_name] = nn.ModuleDict({
                        "router": nn.Linear(hidden_dim, self.branch_num_experts),
                        "experts": nn.ModuleList(
                            nn.Sequential(
                                nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
                            )
                            for _ in range(self.branch_num_experts)
                        ),
                        "norm": nn.LayerNorm(hidden_dim),
                    })
                self.bridge_gate = nn.Parameter(torch.zeros(()))
            self.last_router_diagnostics: dict[str, Any] | None = None
            self.last_typed_outputs: dict[str, Any] | None = None
            self.last_branch_router_diagnostics: dict[str, Any] | None = None
            self.last_geometry_neighbor_indices: Any | None = None
            self.last_geometry_neighbor_valid: Any | None = None
            self.last_segment_diagnostics: dict[str, Any] | None = None
            self.last_page_context_diagnostics: dict[str, Any] | None = None
            self.last_query_seed_diagnostics: dict[str, Any] | None = None
            self.last_candidate_query_diagnostics: dict[str, Any] | None = None
            self.last_weak_family_feature_diagnostics: dict[str, Any] | None = None
            self.last_dense_attention_adapter_diagnostics: dict[str, Any] | None = None
            self.last_explicit_route_logits: Any | None = None
            self.last_explicit_route_diagnostics: dict[str, Any] | None = None
            self.last_component_seed_logits: Any | None = None
            self.last_token_offsets: Any | None = None
            self.last_token_affinity_embeddings: Any | None = None

            self.semantic_head = nn.Linear(hidden_dim, int(num_labels) - 1)
            self.semantic_query_residual_enabled = bool(semantic_query_residual_enabled)
            self.semantic_query_residual_gate = nn.Parameter(torch.zeros(()))
            self.sq_mask_residual_gate = nn.Parameter(torch.zeros(()))
            self.sq_mask_residual_projection = nn.Linear(hidden_dim, hidden_dim)
            self.sq_ownership_residual_gate = nn.Parameter(torch.zeros(()))
            self.sq_rq_enabled = bool(sq_rq_enabled)
            self.sq_rq_runtime_enabled = bool(sq_rq_enabled)
            self.sq_rq_cross_attention = SqRqCrossAttention(
                primitive_dim=hidden_dim, rq_query_dim=hidden_dim, hidden_dim=hidden_dim,
                num_classes=num_labels, heads=heads,
                controlled_gradient_scale=float(sq_rq_gradient_scale),
                query_confidence_threshold=float(sq_rq_query_confidence_threshold),
                token_membership_threshold=float(sq_rq_token_membership_threshold),
                training_membership_temperature=float(sq_rq_training_membership_temperature),
                no_object_class=IGNORE_LABEL,
                include_fixture_semantic_heads=False,
            ) if self.sq_rq_enabled else None
            self.geometry_decoder_mode = geometry_decoder_mode
            self.content_seeded_queries = bool(content_seeded_queries)
            self.component_seeded_queries = bool(component_seeded_queries)
            self.offset_vote_enabled = bool(offset_vote_enabled)
            self.candidate_aware_queries = bool(candidate_aware_queries)
            self.candidate_feature_dim = int(candidate_feature_dim)
            self.candidate_mask_prior_logit = float(candidate_mask_prior_logit)
            self.weak_family_feature_fusion = bool(weak_family_feature_fusion)
            self.quality_query_gradient_scale = float(quality_query_gradient_scale)
            self.explicit_route_classifier = bool(explicit_route_classifier)
            self.dense_attention_feature_adapter = bool(dense_attention_feature_adapter)
            self.dense_attention_window_size = max(1, int(dense_attention_window_size))
            self.route_conditioning_residual_scale = max(0.0, float(route_conditioning_residual_scale))
            self.dense_attention_adapter_residual_scale = max(0.0, float(dense_attention_adapter_residual_scale))
            self.route_conditioning_runtime_scale = 1.0
            self.dense_attention_adapter_runtime_scale = 1.0
            if self.candidate_aware_queries:
                if self.candidate_feature_dim <= 0:
                    raise ValueError("candidate-aware queries require a positive candidate_feature_dim")
                self.candidate_feature_proj = nn.Sequential(
                    nn.Linear(self.candidate_feature_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
                )
                self.candidate_anchor_proj = nn.Linear(self.candidate_feature_dim, 2)
            if self.weak_family_feature_fusion:
                self.weak_family_feature_proj = nn.Sequential(
                    nn.Linear(12, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
                )
                self.weak_family_feature_gate = nn.Linear(hidden_dim * 2, hidden_dim)
                self.weak_family_feature_norm = nn.LayerNorm(hidden_dim)
            if self.explicit_route_classifier:
                self.route_family_head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, len(FAMILY_NAMES))
                )
                self.route_family_embed = nn.Embedding(len(FAMILY_NAMES), hidden_dim)
                self.route_token_gate = nn.Linear(hidden_dim * 2, hidden_dim)
                self.route_token_norm = nn.LayerNorm(hidden_dim)
                self.route_residual_logit_gate = nn.Parameter(torch.tensor(-4.0))
            if self.dense_attention_feature_adapter:
                self.dense_adapter_attn = nn.MultiheadAttention(
                    hidden_dim, heads, dropout=float(dropout), batch_first=True
                )
                self.dense_adapter_ffn = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Dropout(float(dropout)),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                )
                self.dense_adapter_norm_attn = nn.LayerNorm(hidden_dim)
                self.dense_adapter_norm_ffn = nn.LayerNorm(hidden_dim)
                self.dense_adapter_residual_logit_gate = nn.Parameter(torch.tensor(-4.0))
            self.typed_stuff_slots = bool(typed_stuff_slots)
            if self.typed_stuff_slots and int(num_stuff_queries) != len(STUFF_LABELS):
                raise ValueError("typed stuff slots require exactly five stuff queries")
            stuff_count = min(int(num_stuff_queries), int(num_queries))
            self.num_thing_queries = int(num_queries) - stuff_count
            self.num_stuff_queries = stuff_count
            if geometry_decoder_mode == "geometry_v2":
                self.thing_query_embed = nn.Embedding(self.num_thing_queries, hidden_dim)
                self.stuff_query_embed = nn.Embedding(self.num_stuff_queries, hidden_dim)
                self.query_type_embed = nn.Embedding(2, hidden_dim)
                self.query_anchor = nn.Linear(hidden_dim, 2)
                self.thing_seed_score = nn.Linear(hidden_dim, 1)
                if self.content_seeded_queries or self.component_seeded_queries:
                    self.family_seed_head = nn.Linear(hidden_dim, len(FAMILY_NAMES))
                if self.component_seeded_queries:
                    self.component_seed_head = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, len(FAMILY_NAMES) + 4),
                    )
                if self.offset_vote_enabled:
                    self.token_offset_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2))
                    self.token_affinity_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
                    self.offset_ownership_gate = nn.Parameter(torch.zeros(()))
                self.local_graph_proj = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
                self.coarse_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
                self.query_decoder = nn.ModuleList(RelativeGeometryDecoderLayer() for _ in range(max(int(query_decoder_layers), 1)))
            else:
                self.query_embed = nn.Embedding(num_queries, hidden_dim)
                self.query_decoder = nn.ModuleList(QueryDecoderLayer() for _ in range(max(int(query_decoder_layers), 1)))
            self.query_class_head = nn.Linear(hidden_dim, num_labels)
            self.query_objectness_head = nn.Linear(hidden_dim, 1)
            self.stuff_presence_head = nn.Linear(hidden_dim, 1)
            self.query_quality_head = nn.Linear(hidden_dim, 1)
            self.query_identity_head = QueryIdentityHead(hidden_dim, identity_dim)
            self.query_mask_head = nn.Linear(hidden_dim, hidden_dim)
            self.token_mask_head = nn.Linear(hidden_dim, hidden_dim)
            self.ownership_enabled = bool(ownership_enabled)
            self.query_ownership_head = nn.Linear(hidden_dim, hidden_dim)
            self.token_ownership_head = nn.Linear(hidden_dim, hidden_dim)
            self.null_ownership_head = nn.Linear(hidden_dim, 1)
            self.ownership_residual_gate = nn.Parameter(torch.zeros(()))
            self.last_aux_outputs: list[dict[str, Any]] = []

        def _mask_query_label_domains(self, query_logits: Any) -> Any:
            """Enforce the FloorPlanCAD thing/stuff contract at every decoder output."""
            if not self.typed_stuff_slots:
                return query_logits
            masked = query_logits.clone()
            invalid = -1.0e4
            masked[:, : self.num_thing_queries, 30:IGNORE_LABEL] = invalid
            for slot_index, label in enumerate(STUFF_LABELS):
                query_index = self.num_thing_queries + slot_index
                invalid_labels = [index for index in range(IGNORE_LABEL) if index != label]
                masked[:, query_index, invalid_labels] = invalid
            return masked

        @staticmethod
        def _semantic_logits_with_no_object(semantic_logits: Any) -> Any:
            """Expose an explicit impossible no-object slot only to RQ-conditioned SQ."""
            no_object = semantic_logits.new_full((*semantic_logits.shape[:-1], 1), -1.0e4)
            return torch.cat([semantic_logits, no_object], dim=-1)

        def _apply_query_presence(
            self,
            query_logits: Any,
            queries: Any,
            admission_logits: Any | None = None,
        ) -> tuple[Any, Any]:
            """Factorize admission into object presence and conditional foreground class."""
            objectness_logits = (
                self.query_objectness_head(queries).squeeze(-1)
                if admission_logits is None else admission_logits
            )
            if objectness_logits.shape != query_logits.shape[:2]:
                raise ValueError("admission_logits must be [batch, queries]")
            unified_admission_logits = objectness_logits.clone()
            fused = query_logits.new_full(query_logits.shape, -1.0e4)
            no_object_label = fused.shape[-1] - 1
            if self.typed_stuff_slots:
                thing_objectness = objectness_logits[:, : self.num_thing_queries]
                thing_conditional = torch.log_softmax(query_logits[:, : self.num_thing_queries, :30], dim=-1)
                fused[:, : self.num_thing_queries, :30] = (
                    thing_conditional + torch.nn.functional.logsigmoid(thing_objectness).unsqueeze(-1)
                )
                fused[:, : self.num_thing_queries, no_object_label] = torch.nn.functional.logsigmoid(-thing_objectness)
                stuff_presence = (
                    self.stuff_presence_head(queries[:, self.num_thing_queries:]).squeeze(-1)
                    if admission_logits is None
                    else objectness_logits[:, self.num_thing_queries:]
                )
                unified_admission_logits[:, self.num_thing_queries:] = stuff_presence
                for slot_index, label in enumerate(STUFF_LABELS):
                    query_index = self.num_thing_queries + slot_index
                    fused[:, query_index, label] = torch.nn.functional.logsigmoid(stuff_presence[:, slot_index])
                    fused[:, query_index, no_object_label] = torch.nn.functional.logsigmoid(-stuff_presence[:, slot_index])
            else:
                conditional = torch.log_softmax(query_logits[..., :no_object_label], dim=-1)
                fused[..., :no_object_label] = conditional + torch.nn.functional.logsigmoid(objectness_logits).unsqueeze(-1)
                fused[..., no_object_label] = torch.nn.functional.logsigmoid(-objectness_logits)
            return self._mask_query_label_domains(fused), unified_admission_logits

        @staticmethod
        def _dispatch_selected_experts(features: Any, experts: Any, indices: Any, weights: Any) -> Any:
            """Run only selected expert/token pairs and scatter their weighted outputs."""
            flat_features = features.reshape(-1, features.shape[-1])
            flat_indices = indices.reshape(-1, indices.shape[-1])
            flat_weights = weights.reshape(-1, weights.shape[-1])
            output = torch.zeros_like(flat_features)
            for expert_index, expert in enumerate(experts):
                assignments = torch.nonzero(
                    (flat_indices == expert_index) & (flat_weights > 0), as_tuple=False
                )
                if assignments.numel() == 0:
                    continue
                token_rows = assignments[:, 0]
                slot_rows = assignments[:, 1]
                expert_output = expert(flat_features.index_select(0, token_rows))
                weighted_output = expert_output.to(output.dtype) * flat_weights[token_rows, slot_rows].to(output.dtype).unsqueeze(-1)
                output.index_add_(0, token_rows, weighted_output)
            return output.reshape_as(features)

        def _route_typed_branch(self, features: Any, branch_name: str, valid_mask: Any | None = None) -> tuple[Any, dict[str, Any]]:
            if not self.typed_branch_routers:
                return features, {"enabled": False, "branch": branch_name}
            block = self.branch_routers[branch_name]
            if valid_mask is None:
                valid_mask = torch.ones(features.shape[:-1], dtype=torch.bool, device=features.device)
            valid_mask = valid_mask.to(torch.bool)
            logits = block["router"](features)
            scaled_logits = logits / self.router_temperature
            values, indices = torch.topk(scaled_logits, self.branch_top_k, dim=-1)
            probability = torch.softmax(scaled_logits, dim=-1)
            if self.branch_top_k == 1:
                selected_probability = probability.gather(-1, indices)
                hard_weight = torch.ones_like(selected_probability)
                weights = hard_weight + selected_probability - selected_probability.detach()
                routing_strategy = "top1_straight_through_selected_probability_v1"
            else:
                weights = torch.softmax(values, dim=-1)
                routing_strategy = "topk_softmax_selected_logits_v1"
            flat_indices = indices.reshape(-1, indices.shape[-1])
            flat_weights = weights.reshape(-1, weights.shape[-1])
            valid_flat = valid_mask.reshape(-1)
            capacity = None
            if not bool(branch_dropless):
                valid_assignments = valid_flat.sum().to(logits.dtype) * self.branch_top_k
                capacity = torch.ceil(
                    valid_assignments * self.branch_capacity_factor / self.branch_num_experts
                ).to(torch.long).clamp_min(1)
            keep = torch.ones_like(flat_indices, dtype=torch.bool)
            keep &= valid_flat[:, None]
            overflow = torch.zeros((), dtype=torch.long, device=features.device)
            if not bool(branch_dropless):
                for expert_index in range(self.branch_num_experts):
                    assigned = (flat_indices == expert_index) & keep
                    flat_assigned = assigned.reshape(-1)
                    scores = flat_weights.masked_fill(~assigned, -float("inf")).reshape(-1)
                    order = torch.argsort(scores, descending=True, stable=True)
                    rank = torch.empty_like(order)
                    rank.scatter_(0, order, torch.arange(order.numel(), device=order.device))
                    dropped = flat_assigned & (rank >= capacity)
                    overflow = overflow + dropped.sum()
                    keep &= ~dropped.reshape_as(keep)
            flat_weights = flat_weights * keep.to(flat_weights.dtype)
            weight_sum = flat_weights.sum(dim=-1, keepdim=True)
            flat_weights = torch.where(weight_sum > 0, flat_weights / weight_sum.clamp_min(1e-12), flat_weights)
            weights = flat_weights.reshape_as(weights)
            routed = block["norm"](features + self._dispatch_selected_experts(features, block["experts"], indices, weights))
            routed = torch.where(valid_mask[..., None], routed, features)
            probability_mask = valid_mask[..., None].to(probability.dtype)
            mean_probability = (probability * probability_mask).sum(dim=tuple(range(probability.ndim - 1))) / probability_mask.sum().clamp_min(1.0)
            load_balance = mean_probability.var(unbiased=False) / mean_probability.mean().square().clamp_min(1e-12)
            routing_entropy = (-(probability * probability.clamp_min(1e-12).log()).sum(dim=-1) * valid_mask.to(probability.dtype)).sum() / valid_mask.sum().clamp_min(1)
            assignment_counts = torch.bincount(flat_indices[keep].flatten(), minlength=self.branch_num_experts).to(logits.dtype)
            assignment_fraction = assignment_counts / assignment_counts.sum().clamp_min(1.0)
            switch_load_balance = self.branch_num_experts * (mean_probability * assignment_fraction.detach()).sum()
            return routed, {
                "enabled": True,
                "branch": branch_name,
                "expert_pool": f"{branch_name}_private",
                "topk_indices": indices,
                "topk_weights": weights,
                "mean_expert_probability": mean_probability,
                "load_balance_cv_squared": load_balance,
                "switch_load_balance_loss": switch_load_balance,
                "router_z_loss": scaled_logits.square().mean(),
                "routing_entropy": routing_entropy,
                "capacity": capacity,
                "overflow_assignments": overflow,
                "assignment_fraction": assignment_fraction,
                "usage_gate_passed": (assignment_fraction >= 0.05).all(),
                "branch_dropless": bool(branch_dropless),
                "routing_strategy": routing_strategy,
            }

        @staticmethod
        def _content_seed_indices(seed_scores: Any, geometry: Any, valid_mask: Any, count: int) -> tuple[Any, Any, Any]:
            """Rank content seeds by score with geometry-stable tie breaking."""
            batch, token_count = seed_scores.shape
            canonical = torch.arange(token_count, device=seed_scores.device).expand(batch, -1)
            # Stable lexicographic geometry order makes tied scores independent of
            # cache/window token order. Exact duplicate segments are equivalent.
            for feature_index in reversed((0, 1, 2, 3, 6, 14)):
                values = geometry[..., feature_index].gather(1, canonical)
                order = torch.argsort(values, dim=-1, stable=True)
                canonical = canonical.gather(1, order)
            canonical_scores = seed_scores.gather(1, canonical)
            ranked = torch.argsort(canonical_scores, dim=-1, descending=True, stable=True)
            selected = canonical.gather(1, ranked[:, :count])
            selected_scores = seed_scores.gather(1, selected)
            selected_valid = valid_mask.gather(1, selected)
            return selected_scores, selected, selected_valid

        @staticmethod
        def _deterministic_segment_aggregates(segment_features: Any, segment_valid: Any, primitive_valid: Any) -> Any:
            """Derive bounded primitive geometry summaries from retained segments."""
            valid = segment_valid & primitive_valid[..., None]
            valid_float = valid.to(segment_features.dtype)
            x1, y1, x2, y2 = (segment_features[..., index] for index in range(4))
            inf = torch.full_like(x1, float("inf"))
            negative_inf = torch.full_like(x1, -float("inf"))
            bbox_x1 = torch.where(valid, torch.minimum(x1, x2), inf).amin(dim=-1)
            bbox_y1 = torch.where(valid, torch.minimum(y1, y2), inf).amin(dim=-1)
            bbox_x2 = torch.where(valid, torch.maximum(x1, x2), negative_inf).amax(dim=-1)
            bbox_y2 = torch.where(valid, torch.maximum(y1, y2), negative_inf).amax(dim=-1)
            dx, dy = x2 - x1, y2 - y1
            lengths = torch.sqrt(dx.square() + dy.square()) * valid_float
            total_length = lengths.sum(dim=-1)
            unit = lengths.clamp_min(1e-12)
            direction_x = (dx * valid_float / unit).sum(dim=-1) / valid_float.sum(dim=-1).clamp_min(1.0)
            direction_y = (dy * valid_float / unit).sum(dim=-1) / valid_float.sum(dim=-1).clamp_min(1.0)
            horizontal = ((dx.abs() >= dy.abs()) & valid).to(segment_features.dtype).sum(dim=-1) / valid_float.sum(dim=-1).clamp_min(1.0)
            vertical = ((dy.abs() > dx.abs()) & valid).to(segment_features.dtype).sum(dim=-1) / valid_float.sum(dim=-1).clamp_min(1.0)
            aggregate = torch.stack([
                bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                total_length, torch.log1p(total_length), horizontal, vertical,
                direction_x, direction_y,
            ], dim=-1)
            return torch.where(primitive_valid[..., None], aggregate, torch.zeros_like(aggregate))

        def _quality_query_input(self, queries: Any) -> Any:
            scale = max(0.0, min(float(self.quality_query_gradient_scale), 1.0))
            if scale <= 0.0:
                return queries.detach()
            return queries.detach() + scale * (queries - queries.detach())

        def _apply_weak_family_feature_fusion(self, x: Any, h: Any, primitive_valid: Any) -> Any:
            if not self.weak_family_feature_fusion:
                return h
            feature_indices = [6, 7, 8, 9, 10, 11, 13, 15, 31, 32, 36, 37]
            values = []
            for index in feature_indices:
                if index < x.shape[-1]:
                    values.append(x[..., index])
                else:
                    values.append(torch.zeros(x.shape[:2], dtype=x.dtype, device=x.device))
            weak_features = torch.stack(values, dim=-1).to(h.dtype)
            weak_tokens = self.weak_family_feature_proj(weak_features)
            gate = torch.sigmoid(self.weak_family_feature_gate(torch.cat([h, weak_tokens], dim=-1)))
            fused = self.weak_family_feature_norm(h + gate * weak_tokens)
            self.last_weak_family_feature_diagnostics = {
                "enabled": True,
                "feature_indices": feature_indices,
                "feature_count": len(feature_indices),
                "valid_token_count": primitive_valid.sum(dim=-1),
                "mean_abs_feature": weak_features.detach().abs().mean(),
            }
            return torch.where(primitive_valid[..., None], fused, h)

        def _apply_dense_attention_feature_adapter(self, h: Any, primitive_valid: Any) -> Any:
            if not self.dense_attention_feature_adapter:
                return h
            window = max(1, min(int(self.dense_attention_window_size), int(h.shape[1])))
            base_scale = float(self.dense_attention_adapter_residual_scale)
            runtime_scale = float(getattr(self, "dense_attention_adapter_runtime_scale", 1.0))
            learned_gate = torch.sigmoid(self.dense_adapter_residual_logit_gate)
            residual_scale = (
                h.new_tensor(base_scale)
                * h.new_tensor(runtime_scale)
                * learned_gate.to(h.dtype)
            )
            chunks = []
            valid_chunks = []
            attended_windows = 0
            for start in range(0, int(h.shape[1]), window):
                stop = min(start + window, int(h.shape[1]))
                chunk = h[:, start:stop]
                chunk_valid = primitive_valid[:, start:stop].to(torch.bool)
                if bool(chunk_valid.any().item()) and base_scale > 0.0 and runtime_scale > 0.0:
                    key_padding_mask = ~chunk_valid
                    row_has_valid = chunk_valid.any(dim=-1)
                    if not bool(row_has_valid.all().item()):
                        key_padding_mask = key_padding_mask.clone()
                        key_padding_mask[~row_has_valid, 0] = False
                    normalized = self.dense_adapter_norm_attn(chunk)
                    attended, _ = self.dense_adapter_attn(
                        normalized,
                        normalized,
                        normalized,
                        key_padding_mask=key_padding_mask,
                        need_weights=False,
                    )
                    attended = attended.masked_fill(~chunk_valid[..., None], 0)
                    chunk = chunk + residual_scale * attended
                    ffn_input = self.dense_adapter_norm_ffn(chunk)
                    ffn_delta = self.dense_adapter_ffn(ffn_input).masked_fill(~chunk_valid[..., None], 0)
                    chunk = chunk + residual_scale * ffn_delta
                    attended_windows += 1
                chunks.append(chunk)
                valid_chunks.append(chunk_valid)
            fused = torch.cat(chunks, dim=1)
            valid = torch.cat(valid_chunks, dim=1)
            self.last_dense_attention_adapter_diagnostics = {
                "enabled": True,
                "adapter": "local_window_multihead_self_attention_v1",
                "window_size": int(window),
                "attended_windows": int(attended_windows),
                "valid_token_count": valid.sum(dim=-1),
                "residual_scale": residual_scale.detach(),
                "learned_residual_gate": learned_gate.detach(),
                "runtime_scale": runtime_scale,
            }
            return torch.where(valid[..., None], fused, h)

        def _apply_explicit_route_conditioning(self, h: Any, primitive_valid: Any) -> tuple[Any, Any | None]:
            if not self.explicit_route_classifier:
                return h, None
            route_logits = self.route_family_head(h)
            route_probability = torch.softmax(route_logits.float(), dim=-1).to(h.dtype)
            route_context = route_probability @ self.route_family_embed.weight.to(h.dtype)
            learned_gate = torch.sigmoid(self.route_residual_logit_gate)
            residual_scale = (
                h.new_tensor(float(self.route_conditioning_residual_scale))
                * h.new_tensor(float(getattr(self, "route_conditioning_runtime_scale", 1.0)))
                * learned_gate.to(h.dtype)
            )
            token_gate = torch.sigmoid(self.route_token_gate(torch.cat([h, route_context], dim=-1)))
            route_delta = self.route_token_norm(token_gate * route_context)
            routed = h + residual_scale * route_delta
            valid = primitive_valid.to(torch.bool)
            self.last_explicit_route_logits = route_logits
            self.last_explicit_route_diagnostics = {
                "enabled": True,
                "router": "explicit_family_route_classifier_v1",
                "route_conditioning": "scheduled_soft_family_embedding_residual_gate_v2",
                "valid_token_count": valid.sum(dim=-1),
                "residual_scale": residual_scale.detach(),
                "learned_residual_gate": learned_gate.detach(),
                "runtime_scale": float(getattr(self, "route_conditioning_runtime_scale", 1.0)),
                "mean_route_entropy": (
                    -(route_probability * route_probability.clamp_min(1e-12).log()).sum(dim=-1)
                    * valid.to(route_probability.dtype)
                ).sum() / valid.sum().clamp_min(1),
            }
            return torch.where(valid[..., None], routed, h), route_logits

        def _final_quality_logits(
            self,
            queries: Any,
            query_logits: Any,
            mask_logits: Any,
            token_valid_mask: Any,
            ownership_logits: Any | None,
        ) -> Any:
            """Final-mask aware quality score with all deployment statistics detached."""
            base = self.query_quality_head(self._quality_query_input(queries)).squeeze(-1)
            valid = token_valid_mask.to(mask_logits.dtype)
            token_denominator = valid.sum(dim=-1, keepdim=True).clamp_min(1.0)
            mask_probability = mask_logits.detach().sigmoid() * valid[:, None, :]
            area = mask_probability.sum(dim=-1) / token_denominator
            entropy = -(
                mask_probability * mask_probability.clamp_min(1e-6).log()
                + (1.0 - mask_probability) * (1.0 - mask_probability).clamp_min(1e-6).log()
            )
            entropy = (entropy * valid[:, None, :]).sum(dim=-1) / token_denominator
            foreground = query_logits.detach().float()[..., :IGNORE_LABEL].max(dim=-1).values
            no_object = query_logits.detach().float()[..., IGNORE_LABEL]
            class_margin = (foreground - no_object).clamp(min=-8.0, max=8.0)
            # Use a bounded empty-mask prior. The previous log(area) term strongly
            # suppressed small but valid CAD components, which hurts recall before
            # the mask head is fully calibrated.
            mask_token_count = mask_probability.sum(dim=-1).clamp_min(0.0)
            empty_penalty = (
                torch.log1p(mask_token_count)
                / torch.log1p(token_denominator).clamp_min(1e-6)
            ) - 1.0
            owner_bonus = torch.zeros_like(base)
            null_margin = torch.zeros_like(base)
            if ownership_logits is not None and ownership_logits.shape[:2] == (mask_logits.shape[0], mask_logits.shape[-1]):
                owner_probability = ownership_logits.detach().float().softmax(dim=-1)
                query_owner = owner_probability[..., : mask_logits.shape[1]].transpose(1, 2)
                null_owner = owner_probability[..., mask_logits.shape[1]].unsqueeze(1)
                owner_support = (query_owner * mask_probability).sum(dim=-1) / token_denominator
                owner_bonus = torch.log1p(owner_support * token_denominator)
                null_margin = ((query_owner - null_owner) * mask_probability).sum(dim=-1) / token_denominator
            return base + 0.20 * class_margin + 0.15 * empty_penalty - 0.10 * entropy + 0.10 * owner_bonus + 0.10 * null_margin

        def _geometry_memory(
            self,
            x: Any,
            h: Any,
            padding: Any,
            segment_features: Any | None = None,
            segment_valid: Any | None = None,
        ) -> tuple[Any, Any, Any]:
            valid = ~padding if padding is not None else torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
            centers = x[..., 4:6].clamp(0.0, 1.0)
            if x.shape[1] > 1:
                if segment_features is not None and segment_valid is not None:
                    neighbor_indices, neighbor_valid, neighbor_distance = sparse_all_segment_neighbor_graph(
                        torch,
                        x,
                        segment_features,
                        segment_valid,
                        neighbors=geometry_local_neighbors,
                        valid_mask=valid,
                    )
                    topology_source = "all_segment_endpoint_min_distance_v1"
                else:
                    neighbor_indices, neighbor_valid, neighbor_distance = sparse_endpoint_neighbor_graph(
                        torch, x, neighbors=geometry_local_neighbors, valid_mask=valid
                    )
                    topology_source = "representative_endpoint_min_distance_v3"
                batch_indices = torch.arange(x.shape[0], device=x.device)[:, None, None]
                self.last_geometry_neighbor_indices = neighbor_indices.detach()
                self.last_geometry_neighbor_valid = neighbor_valid.detach()
                self.last_geometry_topology_source = topology_source
                neighbors = h[batch_indices, neighbor_indices]
                weights = torch.softmax((-neighbor_distance).masked_fill(~neighbor_valid, -1.0e4), dim=-1)
                weights = weights * neighbor_valid.to(weights.dtype)
                weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                local_context = (neighbors * weights.unsqueeze(-1)).sum(-2)
            else:
                local_context = torch.zeros_like(h)
                self.last_geometry_topology_source = "none_single_token"
                self.last_geometry_neighbor_indices = torch.empty(
                    (x.shape[0], x.shape[1], 0), dtype=torch.long, device=x.device
                )
                self.last_geometry_neighbor_valid = torch.empty(
                    (x.shape[0], x.shape[1], 0), dtype=torch.bool, device=x.device
                )
            local = self.local_graph_proj(torch.cat([h, local_context], -1))
            grid = int(geometry_coarse_grid_size)
            cell_ids = (torch.floor(centers.clamp(max=1 - 1e-6) * grid).long()[..., 1] * grid + torch.floor(centers.clamp(max=1 - 1e-6) * grid).long()[..., 0])
            coarse = h.new_zeros((x.shape[0], grid * grid, hidden_dim)); counts = h.new_zeros((x.shape[0], grid * grid, 1))
            for batch_index in range(x.shape[0]):
                ids = cell_ids[batch_index, valid[batch_index]]
                coarse[batch_index].index_add_(0, ids, h[batch_index, valid[batch_index]])
                counts[batch_index].index_add_(0, ids, torch.ones((ids.numel(), 1), dtype=h.dtype, device=h.device))
            coarse = self.coarse_proj(coarse / counts.clamp_min(1.0)); coarse_valid = counts.squeeze(-1) > 0
            axis = (torch.arange(grid, dtype=x.dtype, device=x.device) + 0.5) / grid
            gy, gx = torch.meshgrid(axis, axis, indexing="ij")
            coarse_positions = torch.stack([gx.flatten(), gy.flatten()], -1).unsqueeze(0).expand(x.shape[0], -1, -1)
            return torch.cat([h, local, coarse], 1), torch.cat([centers, centers, coarse_positions], 1), ~torch.cat([valid, valid, coarse_valid], 1)

        def _apply_page_layer_context(self, x: Any, h: Any, primitive_valid: Any) -> Any:
            valid_float = primitive_valid.to(h.dtype)
            page_mean = (h * valid_float.unsqueeze(-1)).sum(dim=1, keepdim=True) / valid_float.sum(dim=1, keepdim=True).clamp_min(1.0).unsqueeze(-1)
            page_context = self.page_global_proj(page_mean).expand_as(h)
            layer_values = torch.round(x[..., 13].detach() * 512.0).to(torch.long) if x.shape[-1] > 13 else torch.zeros(x.shape[:2], dtype=torch.long, device=x.device)
            layer_context = torch.zeros_like(h)
            layer_groups = 0
            for batch_index in range(h.shape[0]):
                valid_layers = torch.unique(layer_values[batch_index, primitive_valid[batch_index]])
                layer_groups += int(valid_layers.numel())
                for layer_id in valid_layers:
                    members = primitive_valid[batch_index] & (layer_values[batch_index] == layer_id)
                    if bool(members.any().item()):
                        layer_context[batch_index, members] = h[batch_index, members].mean(dim=0, keepdim=True)
            layer_context = self.layer_context_proj(torch.cat([h, layer_context], dim=-1))
            gate = torch.sigmoid(self.context_fusion_gate(torch.cat([h, page_context, layer_context], dim=-1)))
            self.last_page_context_diagnostics = {
                "enabled": True,
                "page_global_context": "valid_token_mean_v1",
                "layer_context": "same_layer_mean_v1",
                "layer_groups": layer_groups,
            }
            return self.context_fusion_norm(h + gate * (page_context + layer_context))

        def _apply_repeated_group_context(self, x: Any, h: Any, primitive_valid: Any) -> Any:
            if not self.repeated_group_fusion:
                return h
            page_layer = self._apply_page_layer_context(x, h, primitive_valid)
            relation_width = min(int(x.shape[-1]), 18)
            relation_features = x[..., -relation_width:].to(h.dtype)
            relation_context = self.relation_feature_proj(relation_features)
            edge_coverage = relation_context.new_tensor(0.0)
            if self.relation_bias_enabled and x.shape[1] > 1:
                centers = x[..., 4:6].clamp(0.0, 1.0)
                distance = torch.cdist(centers.float(), centers.float()).to(h.dtype)
                valid_pair = primitive_valid[:, :, None] & primitive_valid[:, None, :]
                weights = torch.softmax((-distance).masked_fill(~valid_pair, -1.0e4), dim=-1)
                relation_context = relation_context + torch.einsum("bij,bjh->bih", weights, h)
                edge_coverage = valid_pair.to(h.dtype).mean()
            gate = torch.sigmoid(self.relation_fusion_gate(torch.cat([h, page_layer, relation_context], dim=-1)))
            self.last_page_context_diagnostics = {
                **(self.last_page_context_diagnostics or {}),
                "repeated_group_fusion": True,
                "relation_bias_enabled": bool(self.relation_bias_enabled),
                "relation_edge_coverage": edge_coverage,
            }
            return self.relation_fusion_norm(h + gate * (page_layer + relation_context - h))

        def forward(
            self,
            x: Any,
            token_padding_mask: Any | None = None,
            segment_features: Any | None = None,
            segment_padding_mask: Any | None = None,
            candidate_features: Any | None = None,
            candidate_padding_mask: Any | None = None,
            candidate_token_masks: Any | None = None,
            return_quality: bool = False,
            return_identity: bool = False,
        ) -> tuple[Any, ...]:
            h = self.input_proj(x)
            self.last_segment_diagnostics = None
            self.last_page_context_diagnostics = None
            self.last_candidate_query_diagnostics = None
            self.last_weak_family_feature_diagnostics = None
            self.last_dense_attention_adapter_diagnostics = None
            self.last_explicit_route_logits = None
            self.last_explicit_route_diagnostics = None
            self.last_component_seed_logits = None
            self.last_token_offsets = None
            self.last_token_affinity_embeddings = None
            if segment_features is not None:
                if segment_features.ndim != 4 or segment_features.shape[:2] != x.shape[:2] or segment_features.shape[-1] != x.shape[-1]:
                    raise ValueError("segment_features must have shape [batch, primitive, segment, feature_dim]")
                if segment_padding_mask is None:
                    segment_padding_mask = torch.zeros(segment_features.shape[:3], dtype=torch.bool, device=segment_features.device)
                if segment_padding_mask.shape != segment_features.shape[:3]:
                    raise ValueError("segment_padding_mask must have shape [batch, primitive, segment]")
                segment_valid = ~segment_padding_mask.to(torch.bool)
                primitive_valid = (
                    ~token_padding_mask.to(torch.bool)
                    if token_padding_mask is not None
                    else torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
                )
                if not bool((segment_valid.any(dim=-1) | ~primitive_valid).all().item()):
                    raise ValueError("every primitive must retain at least one valid sampled segment")
                encoded_segments = self.segment_encoder(segment_features)
                flat_segments = encoded_segments.reshape(-1, encoded_segments.shape[-2], encoded_segments.shape[-1])
                flat_padding = segment_padding_mask.reshape(-1, segment_padding_mask.shape[-1]).to(torch.bool)
                flat_valid_rows = ~flat_padding.all(dim=-1)
                contextual_segments = flat_segments.clone()
                if bool(flat_valid_rows.any().item()):
                    contextual_segments[flat_valid_rows] = self.segment_context_encoder(
                        flat_segments[flat_valid_rows],
                        src_key_padding_mask=flat_padding[flat_valid_rows],
                    )
                encoded_segments = contextual_segments.reshape_as(encoded_segments)
                segment_scores = self.segment_pool_score(encoded_segments).squeeze(-1)
                segment_scores = segment_scores.masked_fill(~segment_valid, -torch.finfo(segment_scores.dtype).max)
                segment_scores = torch.where(
                    primitive_valid[..., None], segment_scores,
                    torch.zeros_like(segment_scores),
                )
                segment_weights = torch.softmax(segment_scores, dim=-1)
                segment_weights = segment_weights * segment_valid.to(segment_weights.dtype)
                segment_weights = segment_weights / segment_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                pooled_segments = torch.einsum("bnsh,bns->bnh", encoded_segments, segment_weights)
                pooled_segments = pooled_segments * primitive_valid.unsqueeze(-1).to(pooled_segments.dtype)
                if x.shape[1] > 1:
                    cross_indices, cross_valid, cross_distance = sparse_all_segment_neighbor_graph(
                        torch,
                        x,
                        segment_features,
                        segment_valid,
                        neighbors=geometry_local_neighbors,
                        valid_mask=primitive_valid,
                    )
                    batch_indices = torch.arange(x.shape[0], device=x.device)[:, None, None]
                    cross_neighbors = pooled_segments[batch_indices, cross_indices]
                    cross_weights = torch.softmax((-cross_distance).masked_fill(~cross_valid, -1.0e4), dim=-1)
                    cross_weights = cross_weights * cross_valid.to(cross_weights.dtype)
                    cross_weights = cross_weights / cross_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    cross_segment_context = (cross_neighbors * cross_weights.unsqueeze(-1)).sum(-2)
                else:
                    cross_valid = torch.empty((x.shape[0], x.shape[1], 0), dtype=torch.bool, device=x.device)
                    cross_segment_context = torch.zeros_like(pooled_segments)
                pooled_segments = pooled_segments + cross_segment_context
                aggregate_features = self._deterministic_segment_aggregates(
                    segment_features, segment_valid, primitive_valid,
                )
                aggregate_tokens = self.segment_aggregate_encoder(aggregate_features)
                fusion_gate = torch.sigmoid(self.segment_fusion_gate(torch.cat([h, pooled_segments, aggregate_tokens], dim=-1)))
                h = self.segment_fusion_norm(h + fusion_gate * (pooled_segments + aggregate_tokens))
                self.last_segment_diagnostics = {
                    "enabled": True,
                    "segment_first_interaction": "per_primitive_self_attention_plus_cross_primitive_all_segment_neighbors_v1",
                    "cross_primitive_edge_coverage": cross_valid.to(torch.float32).mean(),
                    "mean_valid_segments": segment_valid.to(torch.float32).sum(dim=-1).mean(),
                    "max_pool_weight": segment_weights.max(dim=-1).values.mean(),
                    "mean_total_segment_length": aggregate_features[..., 4].mean(),
                }
            primitive_valid_for_context = (
                ~token_padding_mask.to(torch.bool)
                if token_padding_mask is not None
                else torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
            )
            h = self._apply_weak_family_feature_fusion(x, h, primitive_valid_for_context)
            h = self._apply_page_layer_context(x, h, primitive_valid_for_context)
            h = self._apply_dense_attention_feature_adapter(h, primitive_valid_for_context)
            pos = sinusoidal_position(x, h.shape[-1], version=position_encoding_version)
            if pos.shape[-1] >= h.shape[-1]:
                h = h + pos[..., : h.shape[-1]]
            else:
                h[..., : pos.shape[-1]] = h[..., : pos.shape[-1]] + pos
            if self.gradient_checkpointing and self.training and hasattr(torch.utils, "checkpoint"):
                for encoder_layer in self.encoder.layers:
                    h = torch.utils.checkpoint.checkpoint(
                        lambda layer_input, layer=encoder_layer: layer(layer_input, src_key_padding_mask=token_padding_mask),
                        h,
                        use_reentrant=False,
                    )
                if self.encoder.norm is not None:
                    h = self.encoder.norm(h)
            else:
                h = self.encoder(h, src_key_padding_mask=token_padding_mask)
            h = self._apply_repeated_group_context(x, h, primitive_valid_for_context)
            self.last_router_diagnostics = None
            self.last_branch_router_diagnostics = {}
            self.last_typed_outputs = None
            self.last_query_seed_diagnostics = None
            self.last_family_seed_logits = None
            self.last_query_positions = None
            if self.learned_sparse_router:
                router_logits = self.sparse_router(h) / self.router_temperature
                top_values, top_indices = torch.topk(router_logits, self.router_top_k, dim=-1)
                probabilities = torch.softmax(router_logits, dim=-1)
                if self.router_top_k == 1:
                    selected_probability = probabilities.gather(-1, top_indices)
                    hard_weight = torch.ones_like(selected_probability)
                    top_weights = hard_weight + selected_probability - selected_probability.detach()
                    routing_strategy = "top1_straight_through_selected_probability_v1"
                else:
                    top_weights = torch.softmax(top_values, dim=-1).to(router_logits.dtype)
                    routing_strategy = "topk_softmax_selected_logits_v1"
                h = self.sparse_router_norm(h + self._dispatch_selected_experts(h, self.sparse_experts, top_indices, top_weights))
                if token_padding_mask is not None:
                    valid = (~token_padding_mask).unsqueeze(-1).to(probabilities.dtype)
                    mean_probability = (probabilities * valid).sum(dim=(0, 1)) / valid.sum().clamp_min(1.0)
                else:
                    mean_probability = probabilities.mean(dim=(0, 1))
                load_balance = mean_probability.var(unbiased=False) / mean_probability.mean().square().clamp_min(1e-12)
                flat_indices = top_indices.reshape(-1, top_indices.shape[-1])
                if token_padding_mask is not None:
                    assignment_valid = (~token_padding_mask).reshape(-1)
                else:
                    assignment_valid = torch.ones(flat_indices.shape[0], dtype=torch.bool, device=h.device)
                assignment_counts = torch.bincount(
                    flat_indices[assignment_valid].flatten(), minlength=len(self.sparse_experts)
                ).to(router_logits.dtype)
                assignment_fraction = assignment_counts / assignment_counts.sum().clamp_min(1.0)
                routing_entropy = (-(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1) * assignment_valid.reshape(h.shape[:2]).to(probabilities.dtype)).sum() / assignment_valid.sum().clamp_min(1)
                self.last_router_diagnostics = {
                    "router_logits": router_logits,
                    "topk_indices": top_indices,
                    "topk_weights": top_weights,
                    "mean_expert_probability": mean_probability,
                    "load_balance_cv_squared": load_balance,
                    "switch_load_balance_loss": len(self.sparse_experts) * (mean_probability * assignment_fraction.detach()).sum(),
                    "router_z_loss": router_logits.square().mean(),
                    "assignment_fraction": assignment_fraction,
                    "routing_entropy": routing_entropy,
                    "routing_strategy": routing_strategy,
                    "config": router_config,
                    "route_trace": {
                    "schema_version": "token_topk_v1",
                    "branch": "shared_token_backbone",
                    "expert_pool": "shared_encoder",
                        "expert_manifest": router_config["expert_manifest"],
                        "valid_token_count": (~token_padding_mask).sum(dim=1) if token_padding_mask is not None else torch.full((h.shape[0],), h.shape[1], device=h.device, dtype=torch.long),
                    },
                }
            h_shared = h
            token_valid_mask = ~token_padding_mask if token_padding_mask is not None else torch.ones(h_shared.shape[:2], dtype=torch.bool, device=h_shared.device)
            h_semantic, semantic_route = self._route_typed_branch(h_shared, "semantic", token_valid_mask)
            self.last_branch_router_diagnostics["semantic"] = semantic_route
            semantic_base_logits = self._semantic_logits_with_no_object(self.semantic_head(h_semantic))
            h_rq, route_family_logits = self._apply_explicit_route_conditioning(h_shared, token_valid_mask)
            tmask = self.token_mask_head(h_rq)
            self.last_aux_outputs = []
            candidate_valid = None
            candidate_count = 0
            component_seed_logits = None
            if self.geometry_decoder_mode == "geometry_v2":
                thing = self.thing_query_embed.weight + self.query_type_embed.weight[0]
                stuff = self.stuff_query_embed.weight + self.query_type_embed.weight[1]
                thing_queries = thing.unsqueeze(0).expand(x.shape[0], -1, -1)
                candidate_query_positions = None
                if self.candidate_aware_queries and candidate_features is not None:
                    if candidate_features.ndim != 3 or candidate_features.shape[0] != x.shape[0] or candidate_features.shape[-1] != self.candidate_feature_dim:
                        raise ValueError("candidate_features must have shape [batch, candidates, candidate_feature_dim]")
                    if candidate_padding_mask is None:
                        candidate_padding_mask = torch.zeros(candidate_features.shape[:2], dtype=torch.bool, device=candidate_features.device)
                    if candidate_padding_mask.shape != candidate_features.shape[:2]:
                        raise ValueError("candidate_padding_mask must have shape [batch, candidates]")
                    candidate_valid = ~candidate_padding_mask.to(torch.bool)
                    candidate_count = min(self.num_thing_queries, int(candidate_features.shape[1]))
                    if candidate_count > 0:
                        candidate_tokens = self.candidate_feature_proj(candidate_features[:, :candidate_count].to(h_rq.dtype))
                        candidate_valid_prefix = candidate_valid[:, :candidate_count]
                        thing_queries = thing_queries.clone()
                        thing_queries[:, :candidate_count] = (
                            thing_queries[:, :candidate_count]
                            + candidate_tokens * candidate_valid_prefix.unsqueeze(-1).to(candidate_tokens.dtype)
                        )
                        candidate_query_positions = torch.sigmoid(
                            self.candidate_anchor_proj(candidate_features[:, :candidate_count].to(h_rq.dtype))
                        )
                        self.last_candidate_query_diagnostics = {
                            "enabled": True,
                            "candidate_feature_dim": self.candidate_feature_dim,
                            "candidate_count": candidate_count,
                            "valid_candidate_count": candidate_valid_prefix.sum(dim=-1),
                            "query_policy": "candidate_feature_bias_first_thing_queries_v1",
                        }
                family_seed_logits = None
                if self.content_seeded_queries or self.component_seeded_queries or self.explicit_route_classifier:
                    family_seed_logits = (
                        route_family_logits
                        if route_family_logits is not None
                        else self.family_seed_head(h_rq)
                    )
                    self.last_family_seed_logits = family_seed_logits
                if self.component_seeded_queries:
                    component_seed_logits = self.component_seed_head(h_rq)
                    self.last_component_seed_logits = component_seed_logits
                seed_mode = "disabled"
                if self.content_seeded_queries or self.component_seeded_queries:
                    family_seed_scores = family_seed_logits.max(dim=-1).values
                    seed_scores = self.thing_seed_score(h_rq).squeeze(-1) + family_seed_scores
                    if self.component_seeded_queries:
                        component_family_scores = component_seed_logits[..., :len(FAMILY_NAMES)].max(dim=-1).values
                        component_size_scores = component_seed_logits[..., len(FAMILY_NAMES):].mean(dim=-1)
                        seed_scores = seed_scores + component_family_scores + 0.25 * component_size_scores
                        seed_mode = "component_family_bbox_topk_v1"
                    else:
                        seed_mode = "family_token_topk_v1"
                    seed_scores = seed_scores.masked_fill(~token_valid_mask, -torch.finfo(seed_scores.dtype).max)
                    seed_count = min(self.num_thing_queries, h_rq.shape[1])
                    seed_values, seed_indices, seed_valid = self._content_seed_indices(
                        seed_scores, x, token_valid_mask, seed_count,
                    )
                    seed_tokens = torch.gather(
                        h_rq, 1, seed_indices.unsqueeze(-1).expand(-1, -1, h_rq.shape[-1]),
                    )
                    thing_queries = thing_queries.clone()
                    thing_queries[:, :seed_count] = thing_queries[:, :seed_count] + seed_tokens * seed_valid.unsqueeze(-1).to(seed_tokens.dtype)
                    seed_centers = torch.gather(
                        x[..., 4:6].clamp(0.0, 1.0), 1,
                        seed_indices.unsqueeze(-1).expand(-1, -1, 2),
                    )
                    self.last_query_seed_diagnostics = {
                        "enabled": True,
                        "seed_mode": seed_mode,
                        "seed_count": seed_count,
                        "selected_valid_count": seed_valid.sum(dim=-1),
                        "mean_seed_score": (seed_values * seed_valid.to(seed_values.dtype)).sum() / seed_valid.sum().clamp_min(1),
                        "seed_indices": seed_indices,
                        "seed_centers": seed_centers,
                    }
                q = torch.cat([thing_queries, stuff.unsqueeze(0).expand(x.shape[0], -1, -1)], 1)
                memory, memory_positions, memory_padding = self._geometry_memory(
                    x,
                    h_rq,
                    token_padding_mask,
                    segment_features=segment_features,
                    segment_valid=(None if segment_padding_mask is None else ~segment_padding_mask.to(torch.bool)),
                )
                if (
                    self.candidate_aware_queries
                    and candidate_features is not None
                    and candidate_count > 0
                    and candidate_valid is not None
                ):
                    candidate_memory = self.candidate_feature_proj(candidate_features[:, :candidate_count].to(h_rq.dtype))
                    candidate_positions = (
                        candidate_query_positions
                        if candidate_query_positions is not None
                        else torch.sigmoid(self.candidate_anchor_proj(candidate_features[:, :candidate_count].to(h_rq.dtype)))
                    )
                    candidate_padding = ~candidate_valid[:, :candidate_count]
                    memory = torch.cat([memory, candidate_memory], dim=1)
                    memory_positions = torch.cat([memory_positions, candidate_positions], dim=1)
                    memory_padding = torch.cat([memory_padding, candidate_padding], dim=1)
                for layer_index, layer in enumerate(self.query_decoder):
                    query_positions = torch.sigmoid(self.query_anchor(q))
                    if candidate_query_positions is not None and candidate_count > 0:
                        query_positions = query_positions.clone()
                        query_positions[:, :candidate_count] = candidate_query_positions
                    if self.content_seeded_queries and self.last_query_seed_diagnostics is not None:
                        seed_count = int(self.last_query_seed_diagnostics["seed_count"])
                        seed_valid = self.last_query_seed_diagnostics["selected_valid_count"]
                        seed_centers = self.last_query_seed_diagnostics["seed_centers"]
                        if seed_count > 0:
                            query_positions = query_positions.clone()
                            query_positions[:, :seed_count] = 0.5 * query_positions[:, :seed_count] + 0.5 * seed_centers
                    self.last_query_positions = query_positions
                    q, attention = layer(q, query_positions, memory, memory_positions, memory_padding)
                    layer_query, layer_admission = self._apply_query_presence(
                        self._mask_query_label_domains(self.query_class_head(q)), q
                    )
                    layer_quality = self.query_quality_head(self._quality_query_input(q)).squeeze(-1)
                    layer_identity = self.query_identity_head(q)
                    layer_mask = torch.einsum("bqh,bnh->bqn", self.query_mask_head(q), tmask) / math.sqrt(float(h_rq.shape[-1]))
                    self.last_aux_outputs.append({"layer_index": layer_index, "query_logits": layer_query, "admission_logits": layer_admission, "mask_logits": layer_mask, "quality_logits": layer_quality, "identity_embeddings": layer_identity, "query_positions": query_positions, "geometry_attention": attention})
                final = self.last_aux_outputs[-1]
                query_logits, mask_logits, quality_logits, identity_embeddings = final["query_logits"], final["mask_logits"], final["quality_logits"], final["identity_embeddings"]
            else:
                q = self.query_embed.weight.unsqueeze(0).expand(x.shape[0], -1, -1)
                for layer in self.query_decoder:
                    q = layer(q, h_rq, token_padding_mask)
                query_logits = self.query_class_head(q); quality_logits = self.query_quality_head(q).squeeze(-1)
                identity_embeddings = self.query_identity_head(q)
                mask_logits = torch.einsum("bqh,bnh->bqn", self.query_mask_head(q), tmask) / math.sqrt(float(h_rq.shape[-1]))
            q, rq_route = self._route_typed_branch(q, "rq", torch.ones(q.shape[:2], dtype=torch.bool, device=q.device))
            self.last_branch_router_diagnostics["rq"] = rq_route
            conditional_query_logits = self._mask_query_label_domains(self.query_class_head(q))
            query_logits, query_admission_logits = self._apply_query_presence(conditional_query_logits, q)
            quality_logits = self.query_quality_head(self._quality_query_input(q)).squeeze(-1)
            identity_embeddings = self.query_identity_head(q)
            mask_logits = torch.einsum("bqh,bnh->bqn", self.query_mask_head(q), tmask) / math.sqrt(float(h_rq.shape[-1]))
            valid_tokens = torch.ones(mask_logits.shape[0], mask_logits.shape[-1], dtype=mask_logits.dtype, device=mask_logits.device)
            if token_padding_mask is not None:
                valid_tokens = (~token_padding_mask).to(mask_logits.dtype)
            membership = mask_logits.sigmoid() * valid_tokens[:, None, :]
            evidence_denominator = membership.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            semantic_evidence = torch.einsum("bqn,bnc->bqc", membership, semantic_base_logits) / evidence_denominator
            semantic_evidence = semantic_evidence.clone()
            semantic_evidence[..., semantic_evidence.shape[-1] - 1] = 0.0
            semantic_query_residual = (
                torch.tanh(self.semantic_query_residual_gate) * semantic_evidence
                if self.semantic_query_residual_enabled else torch.zeros_like(semantic_evidence)
            )
            query_admission_gate = query_admission_logits.sigmoid().to(semantic_query_residual.dtype)
            semantic_query_residual = semantic_query_residual * query_admission_gate.unsqueeze(-1)
            conditional_query_logits = self._mask_query_label_domains(
                conditional_query_logits + semantic_query_residual
            )
            query_logits, _ = self._apply_query_presence(
                conditional_query_logits, q, admission_logits=query_admission_logits
            )
            if (
                self.candidate_aware_queries
                and candidate_token_masks is not None
                and candidate_count > 0
                and self.candidate_mask_prior_logit > 0.0
            ):
                if candidate_token_masks.ndim != 3 or candidate_token_masks.shape[0] != x.shape[0]:
                    raise ValueError("candidate_token_masks must have shape [batch, candidates, tokens]")
                if candidate_token_masks.shape[1] < candidate_count or candidate_token_masks.shape[2] != mask_logits.shape[-1]:
                    raise ValueError("candidate_token_masks must match candidate count and token count")
                candidate_mask = candidate_token_masks[:, :candidate_count].to(mask_logits.dtype)
                if candidate_valid is not None:
                    candidate_mask = candidate_mask * candidate_valid[:, :candidate_count].unsqueeze(-1).to(mask_logits.dtype)
                mask_logits = mask_logits.clone()
                mask_logits[:, :candidate_count, :] = (
                    mask_logits[:, :candidate_count, :]
                    + self.candidate_mask_prior_logit * candidate_mask
                )
                self.last_candidate_query_diagnostics = {
                    **(self.last_candidate_query_diagnostics or {"enabled": True}),
                    "mask_prior_policy": "primitive_id_candidate_mask_additive_logit_v1",
                    "mask_prior_logit": self.candidate_mask_prior_logit,
                    "candidate_mask_token_total": candidate_mask.sum(dim=-1),
                }
            self.last_sq_rq_outputs = None
            semantic_pre_cross_tokens = h_semantic
            semantic_post_cross_tokens = semantic_pre_cross_tokens
            if self.sq_rq_cross_attention is not None and self.sq_rq_runtime_enabled:
                self.last_sq_rq_outputs = self.sq_rq_cross_attention(
                    h_semantic, q, mask_logits, query_logits,
                    rq_admission_logits=query_admission_logits,
                    base_semantic_logits=semantic_base_logits,
                    primitive_padding_mask=token_padding_mask,
                )
                semantic_post_cross_tokens = self.last_sq_rq_outputs["sq_tokens"]
            semantic_post_private_tokens, sq_route = self._route_typed_branch(
                semantic_post_cross_tokens, "sq", token_valid_mask
            )
            self.last_branch_router_diagnostics["sq"] = sq_route
            bridge_tokens, bridge_route = self._route_typed_branch(semantic_post_private_tokens, "bridge", token_valid_mask)
            semantic_tokens = semantic_post_private_tokens
            if self.typed_branch_routers:
                semantic_tokens = semantic_post_private_tokens + torch.tanh(self.bridge_gate) * (bridge_tokens - semantic_post_private_tokens)
            self.last_branch_router_diagnostics["bridge"] = bridge_route
            rq_mask_logits_for_quality = mask_logits
            sq_mask_residual = torch.zeros_like(mask_logits)
            if self.sq_rq_cross_attention is not None and self.sq_rq_runtime_enabled:
                sq_context_delta = semantic_post_cross_tokens - semantic_pre_cross_tokens
                projected_context_delta = self.sq_mask_residual_projection(sq_context_delta)
                if self.sq_mask_residual_projection.bias is not None:
                    projected_context_delta = projected_context_delta - self.sq_mask_residual_projection.bias
                context_edge_feedback_gate = self.last_sq_rq_outputs["context_edge_feedback_gate"].transpose(1, 2)
                sq_mask_delta = torch.einsum(
                    "bqh,bnh->bqn", self.query_mask_head(q).detach(), projected_context_delta
                ) / math.sqrt(float(h_rq.shape[-1]))
                sq_mask_residual = (
                    torch.tanh(self.sq_mask_residual_gate)
                    * context_edge_feedback_gate
                    * sq_mask_delta
                )
                sq_mask_residual = sq_mask_residual * valid_tokens[:, None, :]
                mask_logits = mask_logits + sq_mask_residual
            token_offsets = self.token_offset_head(h_rq) if self.offset_vote_enabled else None
            token_affinity = self.token_affinity_head(h_rq) if self.offset_vote_enabled else None
            self.last_token_offsets = token_offsets
            self.last_token_affinity_embeddings = token_affinity
            semantic_logits = self.semantic_head(semantic_tokens)
            self.last_typed_outputs = {
                "schema_version": "semantic_panoptic_typed_outputs_v2_staged_sq_rq",
                "semantic_logits": semantic_logits,
                "semantic_base_logits": semantic_base_logits,
                "semantic_post_cross_logits": self.semantic_head(semantic_post_cross_tokens),
                "semantic_post_private_logits": self.semantic_head(semantic_post_private_tokens),
                "rq_query_logits": query_logits,
                "rq_mask_logits": mask_logits,
                "query_objectness_logits": query_admission_logits,
                "query_admission_logits": query_admission_logits,
                "semantic_query_residual": semantic_query_residual,
                "sq_mask_residual": sq_mask_residual,
                "sq_cross_tokens": semantic_post_cross_tokens,
                "sq_private_tokens": semantic_post_private_tokens,
                "sq_tokens": semantic_tokens,
                "ownership_logits": None,
                "sq_ownership_residual": None,
                "component_seed_logits": component_seed_logits,
                "token_offsets": token_offsets,
                "token_affinity_embeddings": token_affinity,
            }
            if self.last_sq_rq_outputs is not None:
                self.last_sq_rq_outputs["semantic_base_logits"] = semantic_base_logits
                self.last_sq_rq_outputs["semantic_post_cross_logits"] = self.last_typed_outputs["semantic_post_cross_logits"]
                self.last_sq_rq_outputs["semantic_post_private_logits"] = self.last_typed_outputs["semantic_post_private_logits"]
                self.last_sq_rq_outputs["semantic_context_logits"] = semantic_logits
            ownership_logits = None
            sq_ownership_residual = None
            rq_ownership_logits_for_quality = None
            if self.ownership_enabled:
                query_owner = self.query_ownership_head(q)
                token_owner = self.token_ownership_head(h_rq)
                ownership_residual = torch.einsum("bnh,bqh->bnq", token_owner, query_owner) / math.sqrt(float(h_rq.shape[-1]))
                if token_offsets is not None:
                    centers = x[..., 4:6].clamp(0.0, 1.0)
                    voted_centers = (centers + 0.25 * torch.tanh(token_offsets)).clamp(0.0, 1.0)
                    query_positions_for_vote = torch.sigmoid(self.query_anchor(q))
                    vote_distance = torch.cdist(voted_centers.float(), query_positions_for_vote.float()).to(ownership_residual.dtype)
                    ownership_residual = ownership_residual - torch.tanh(self.offset_ownership_gate) * vote_distance
                rq_query_token_owner = rq_mask_logits_for_quality.transpose(1, 2) + query_admission_logits[:, None, :]
                rq_query_token_owner = rq_query_token_owner + torch.tanh(self.ownership_residual_gate) * ownership_residual
                rq_ownership_logits_for_quality = torch.cat([rq_query_token_owner, self.null_ownership_head(h_rq)], dim=-1)
                mask_owner = mask_logits.transpose(1, 2)
                query_token_owner = mask_owner + query_admission_logits[:, None, :]
                query_token_owner = query_token_owner + torch.tanh(self.ownership_residual_gate) * ownership_residual
                ownership_logits = torch.cat([query_token_owner, self.null_ownership_head(h_rq)], dim=-1)
                if self.sq_rq_cross_attention is not None and self.sq_rq_runtime_enabled:
                    sq_token_owner = self.token_ownership_head(semantic_tokens)
                    sq_query_delta = torch.einsum(
                        "bnh,bqh->bnq", sq_token_owner - token_owner, query_owner.detach()
                    ) / math.sqrt(float(h_rq.shape[-1]))
                    context_edge_feedback_gate = self.last_sq_rq_outputs["context_edge_feedback_gate"]
                    sq_query_delta = sq_query_delta * context_edge_feedback_gate
                    sq_null_delta = self.null_ownership_head(semantic_tokens) - self.null_ownership_head(h_rq)
                    sq_null_delta = sq_null_delta * self.last_sq_rq_outputs["adaptive_context_gate"].unsqueeze(-1)
                    sq_ownership_delta = torch.cat([sq_query_delta, sq_null_delta], dim=-1)
                    sq_ownership_residual = torch.tanh(self.sq_ownership_residual_gate) * sq_ownership_delta
                    sq_ownership_residual = sq_ownership_residual * token_valid_mask.unsqueeze(-1).to(sq_ownership_residual.dtype)
                    ownership_logits = ownership_logits + sq_ownership_residual
            quality_logits = self._final_quality_logits(
                q,
                query_logits,
                rq_mask_logits_for_quality,
                token_valid_mask,
                rq_ownership_logits_for_quality,
            )
            self.last_ownership_logits = ownership_logits
            self.last_query_objectness_logits = query_admission_logits
            self.last_query_admission_logits = query_admission_logits
            if self.last_typed_outputs is not None:
                self.last_typed_outputs["ownership_logits"] = ownership_logits
                self.last_typed_outputs["sq_ownership_residual"] = sq_ownership_residual
            if return_quality and return_identity:
                return semantic_logits, query_logits, mask_logits, quality_logits, identity_embeddings
            if return_quality:
                return semantic_logits, query_logits, mask_logits, quality_logits
            if return_identity:
                return semantic_logits, query_logits, mask_logits, identity_embeddings
            return semantic_logits, query_logits, mask_logits

    return LineTokenPanopticMoE()


def target_priority_rows(
    groups: dict[tuple[int, int], list[int]],
    *,
    profile: dict[str, Any] | None,
    small_component_size: int,
) -> list[tuple[tuple[int, int], list[int]]]:
    recall_limited = {int(label) for label in (profile or {}).get("recall_limited_labels", [])}
    grouping_limited = {int(label) for label in (profile or {}).get("grouping_limited_labels", [])}
    bottleneck_labels = recall_limited | grouping_limited
    rows = list(groups.items())

    def priority(item: tuple[tuple[int, int], list[int]]) -> tuple[int, int, int, int, int]:
        (label, instance_id), indices = item
        size = len(indices)
        is_stuff = 1 if 30 <= int(label) <= 34 else 0
        is_bottleneck = 1 if int(label) in bottleneck_labels else 0
        is_small = 1 if size <= int(small_component_size) else 0
        # Lower tuple values are kept first. Small and bottleneck components
        # protect RQ; size remains the final tie-breaker for stable masks.
        return (-is_bottleneck, -is_small, -is_stuff, size, int(instance_id))

    return sorted(rows, key=priority)


def select_component_targets(
    groups: dict[tuple[int, int], list[int]],
    *,
    num_queries: int,
    profile: dict[str, Any] | None,
    small_component_size: int,
) -> tuple[list[tuple[tuple[int, int], list[int]]], dict[str, Any]]:
    total_components = len(groups)
    recall_limited = {int(label) for label in (profile or {}).get("recall_limited_labels", [])}
    grouping_limited = {int(label) for label in (profile or {}).get("grouping_limited_labels", [])}
    bottleneck_labels = recall_limited | grouping_limited
    if total_components <= num_queries:
        ordered = sorted(groups.items(), key=lambda item: len(item[1]), reverse=True)
        return ordered, {
            "target_components_total": total_components,
            "target_components_kept": len(ordered),
            "target_components_dropped": 0,
            "small_components_total": sum(1 for _key, indices in groups.items() if len(indices) <= small_component_size),
            "small_components_kept": sum(1 for _key, indices in ordered if len(indices) <= small_component_size),
            "bottleneck_components_total": sum(1 for (label, _instance), _indices in groups.items() if int(label) in bottleneck_labels),
            "bottleneck_components_kept": sum(1 for (label, _instance), _indices in ordered if int(label) in bottleneck_labels),
            "selection_policy": "all_components_fit",
        }

    by_label: dict[int, list[tuple[tuple[int, int], list[int]]]] = {}
    for row in target_priority_rows(groups, profile=profile, small_component_size=small_component_size):
        by_label.setdefault(int(row[0][0]), []).append(row)

    selected: list[tuple[tuple[int, int], list[int]]] = []
    used: set[tuple[int, int]] = set()
    labels_by_priority = sorted(
        by_label,
        key=lambda label: (
            -(1 if label in bottleneck_labels else 0),
            -sum(1 for _key, indices in by_label[label] if len(indices) <= small_component_size),
            -len(by_label[label]),
            label,
        ),
    )
    while len(selected) < num_queries:
        added = False
        for label in labels_by_priority:
            bucket = by_label[label]
            while bucket and bucket[0][0] in used:
                bucket.pop(0)
            if not bucket:
                continue
            row = bucket.pop(0)
            used.add(row[0])
            selected.append(row)
            added = True
            if len(selected) >= num_queries:
                break
        if not added:
            break

    selected_keys = {key for key, _indices in selected}
    small_total = sum(1 for _key, indices in groups.items() if len(indices) <= small_component_size)
    bottleneck_total = sum(1 for (label, _instance), _indices in groups.items() if int(label) in bottleneck_labels)
    diagnostics = {
        "target_components_total": total_components,
        "target_components_kept": len(selected),
        "target_components_dropped": max(total_components - len(selected), 0),
        "small_components_total": small_total,
        "small_components_kept": sum(1 for key, indices in selected if len(indices) <= small_component_size),
        "bottleneck_components_total": bottleneck_total,
        "bottleneck_components_kept": sum(1 for (label, _instance), _indices in selected if int(label) in bottleneck_labels),
        "selection_policy": "bottleneck_small_component_round_robin",
        "dropped_labels": sorted({int(label) for (label, instance), _indices in groups.items() if (label, instance) not in selected_keys}),
    }
    return selected, diagnostics


def component_targets(
    torch: Any,
    labels: Any,
    instances: Any,
    num_queries: int,
    profile: dict[str, Any] | None = None,
    small_component_size: int = 3,
) -> tuple[Any, Any, int, dict[str, Any]]:
    device = labels.device
    groups: dict[tuple[int, int], list[int]] = {}
    labels_cpu = labels.detach().cpu().tolist()
    inst_cpu = instances.detach().cpu().tolist()
    for idx, (label, inst) in enumerate(zip(labels_cpu, inst_cpu, strict=True)):
        label = int(label)
        inst = int(inst)
        if label == IGNORE_LABEL:
            continue
        if 30 <= label <= 34:
            key = (label, -label)
        elif inst >= 0:
            key = (label, inst)
        else:
            continue
        groups.setdefault(key, []).append(idx)
    ordered, diagnostics = select_component_targets(groups, num_queries=num_queries, profile=profile, small_component_size=small_component_size)
    target_labels = torch.empty((len(ordered),), dtype=torch.long, device=device)
    masks = torch.zeros((len(ordered), labels.numel()), dtype=torch.float32, device=device)
    for tidx, ((label, _inst), indices) in enumerate(ordered):
        target_labels[tidx] = int(label)
        masks[tidx, torch.tensor(indices, dtype=torch.long, device=device)] = 1.0
    return target_labels, masks, len(ordered), diagnostics


def component_targets_schema_v2(
    torch: Any,
    labels: Any,
    page_instance_ids: tuple[str | None, ...],
    mask_loss_valid: Any,
    primitive_weights: Any,
    num_queries: int,
    *,
    partial_component_policy: str = "exclude",
    partial_component_min_tokens: int = 1,
) -> tuple[Any, Any, Any, int, dict[str, Any]]:
    if partial_component_policy not in {"exclude", "window_visible"}:
        raise ValueError(f"unsupported partial component policy: {partial_component_policy}")
    if partial_component_min_tokens < 1:
        raise ValueError("partial_component_min_tokens must be at least 1")
    groups: dict[str, list[int]] = {}
    group_labels: dict[str, int] = {}
    labels_cpu = labels.detach().cpu().tolist()
    mask_valid_cpu = mask_loss_valid.detach().cpu().tolist()
    for index, identity in enumerate(page_instance_ids):
        label = int(labels_cpu[index])
        if identity is None or label == IGNORE_LABEL:
            continue
        groups.setdefault(identity, []).append(index)
        group_labels.setdefault(identity, label)
        if group_labels[identity] != label:
            raise ValueError(f"page_instance_id has inconsistent semantic labels: {identity}")
    if len(groups) > num_queries:
        raise ValueError(f"target-schema-v2 query overflow at consumer: {len(groups)}>{num_queries}")
    valid_groups = []
    partial_groups = []
    partial_groups_kept = []
    partial_groups_too_small = []
    for identity, indices in sorted(groups.items()):
        if not all(bool(mask_valid_cpu[index]) for index in indices):
            partial_groups.append(identity)
            if partial_component_policy == "window_visible" and len(indices) >= partial_component_min_tokens:
                valid_groups.append((identity, indices))
                partial_groups_kept.append(identity)
            else:
                if len(indices) < partial_component_min_tokens:
                    partial_groups_too_small.append(identity)
                continue
            continue
        valid_groups.append((identity, indices))
    target_labels = torch.as_tensor([group_labels[identity] for identity, _indices in valid_groups], dtype=torch.long, device=labels.device)
    target_masks_cpu = np.zeros((len(valid_groups), len(labels_cpu)), dtype=np.float32)
    for target_index, (identity, indices) in enumerate(valid_groups):
        target_masks_cpu[target_index, indices] = 1.0
    target_masks = torch.from_numpy(target_masks_cpu).to(labels.device, non_blocking=True)
    diagnostics = {
        "raw_target_components_total": len(groups),
        "policy_eligible_target_components": len(valid_groups),
        "policy_excluded_target_components": len(groups) - len(valid_groups),
        "capacity_target_components_total": len(valid_groups),
        "capacity_target_components_kept": len(valid_groups),
        "capacity_target_components_dropped": 0,
        "target_components_total": len(groups),
        "target_components_kept": len(valid_groups),
        "target_components_dropped": len(groups) - len(valid_groups),
        "partial_mask_components_excluded": len(partial_groups) - len(partial_groups_kept),
        "partial_mask_components_kept_window_visible": len(partial_groups_kept),
        "partial_mask_components_too_small": len(partial_groups_too_small),
        "partial_page_instance_ids": partial_groups,
        "partial_component_policy": partial_component_policy,
        "partial_component_min_tokens": int(partial_component_min_tokens),
        "identity_source": "page_instance_id",
    }
    return target_labels, target_masks, primitive_weights.float(), len(valid_groups), diagnostics


def matched_query_page_instance_ids(
    torch: Any,
    query_labels: Any,
    query_masks: Any,
    page_instance_ids: tuple[str | None, ...],
    mask_loss_valid: Any,
) -> tuple[list[str | None], Any]:
    """Recover identity supervision only from matched, complete schema-v2 targets."""
    query_labels_cpu = query_labels.detach().cpu().tolist()
    query_masks_cpu = query_masks.detach().float().cpu().numpy()
    mask_valid_cpu = mask_loss_valid.detach().cpu().tolist()
    identities: list[str | None] = [None] * len(query_labels_cpu)
    valid_queries_cpu = np.zeros((len(query_labels_cpu),), dtype=np.bool_)
    for query_index, label in enumerate(query_labels_cpu):
        if int(label) == IGNORE_LABEL:
            continue
        token_indices = np.flatnonzero(query_masks_cpu[query_index] > 0.5).tolist()
        target_ids = {
            page_instance_ids[token_index]
            for token_index in token_indices
            if bool(mask_valid_cpu[token_index]) and page_instance_ids[token_index] is not None
        }
        if len(target_ids) == 1 and token_indices and all(bool(mask_valid_cpu[token_index]) for token_index in token_indices):
            identities[query_index] = next(iter(target_ids))
            valid_queries_cpu[query_index] = True
    return identities, torch.from_numpy(valid_queries_cpu).to(query_labels.device, non_blocking=True)


def weighted_semantic_loss_schema_v2(
    torch: Any, logits: Any, labels: Any, primitive_weights: Any,
    class_weights: Any | None = None, label_smoothing: float = 0.0,
) -> Any:
    if logits.shape[-1] == IGNORE_LABEL + 1:
        logits = logits[..., :IGNORE_LABEL]
    if logits.shape[-1] != IGNORE_LABEL:
        raise ValueError(f"semantic logits must expose exactly {IGNORE_LABEL} foreground classes")
    if class_weights is not None and int(class_weights.numel()) == IGNORE_LABEL + 1:
        class_weights = class_weights[:IGNORE_LABEL]
    if class_weights is not None and int(class_weights.numel()) != IGNORE_LABEL:
        raise ValueError("semantic class weights must exclude the ignored no-object label")
    losses = torch.nn.functional.cross_entropy(
        logits.float(), labels, weight=class_weights, ignore_index=IGNORE_LABEL,
        label_smoothing=float(label_smoothing), reduction="none",
    )
    weights = primitive_weights.float() * (labels != IGNORE_LABEL).float()
    return (losses * weights).sum() / weights.sum().clamp_min(1e-6)


def matching_assignment_churn(torch: Any, candidate_labels: Any, reference_labels: Any) -> Any:
    active = (candidate_labels != IGNORE_LABEL) | (reference_labels != IGNORE_LABEL)
    if not bool(active.any().item()):
        return candidate_labels.float().sum() * 0.0
    return (candidate_labels[active] != reference_labels[active]).float().mean()


def query_selected_primitive_indices(
    torch: Any,
    seed_diagnostics: dict[str, Any] | None,
    *,
    batch_index: int,
    num_queries: int,
    token_count: int,
    device: Any,
) -> Any | None:
    """Return VecFormer-style selected primitive ids for seeded thing queries."""
    if not seed_diagnostics or "seed_indices" not in seed_diagnostics:
        return None
    seed_indices = seed_diagnostics["seed_indices"]
    if seed_indices is None or seed_indices.ndim != 2 or batch_index >= int(seed_indices.shape[0]):
        return None
    result = torch.full((int(num_queries),), -1, dtype=torch.long, device=device)
    count = min(int(seed_indices.shape[1]), int(num_queries))
    if count <= 0:
        return result
    selected = seed_indices[int(batch_index), :count].to(device=device, dtype=torch.long)
    valid = (selected >= 0) & (selected < int(token_count))
    result[:count] = torch.where(valid, selected, torch.full_like(selected, -1))
    return result


TEACHER_PROVENANCE_FIELDS = ("split", "record_ids_sha256", "source_checkpoint_sha256", "gt_schema_sha256", "command")


def validate_teacher_provenance(path: Path) -> dict[str, Any]:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not manifest_path.is_file():
        raise RuntimeError(f"teacher provenance manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    provenance = manifest.get("artifact_provenance") if isinstance(manifest, dict) else None
    if not isinstance(provenance, dict):
        raise RuntimeError(f"teacher provenance manifest is malformed: {manifest_path}")
    missing = [field for field in TEACHER_PROVENANCE_FIELDS if not provenance.get(field)]
    if missing:
        raise RuntimeError(f"teacher provenance missing required fields: {', '.join(missing)}")
    if provenance.get("diagnostic_only") is True or provenance.get("training_use_allowed") is False:
        raise RuntimeError("diagnostic-only teacher artifact is forbidden for training")
    return provenance


def load_teacher_proposals(path: Path | None, *, positive_only: bool, min_gt_iou: float) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if path is None:
        return {}, {"enabled": False, "path": None, "records": 0, "proposals": 0}
    if not path.exists():
        raise FileNotFoundError(f"teacher proposals not found: {path}")
    provenance = validate_teacher_provenance(path)
    by_record: dict[str, list[dict[str, Any]]] = {}
    counters = Counter()
    for row in iter_jsonl(path):
        row_provenance = row.get("artifact_provenance")
        if not isinstance(row_provenance, dict) or any(row_provenance.get(field) != provenance.get(field) for field in TEACHER_PROVENANCE_FIELDS):
            raise RuntimeError(f"teacher row provenance does not match manifest: {row.get('record_id')}")
        record_id = str(row.get("record_id"))
        proposals = []
        for item in row.get("teacher_proposals", []) if isinstance(row.get("teacher_proposals"), list) else []:
            match = item.get("gt_match") if isinstance(item.get("gt_match"), dict) else {}
            is_positive = bool(match.get("positive"))
            if is_positive and parse_float(match.get("gt_iou"), 0.0) < min_gt_iou:
                counters["filtered_low_iou"] += 1
                continue
            positions = item.get("primitive_positions")
            primitive_ids = item.get("primitive_ids")
            if not isinstance(positions, list) or not isinstance(primitive_ids, list) or len(positions) != len(primitive_ids):
                counters["malformed_positions"] += 1
                continue
            positions = sorted({parse_int(idx, -1) for idx in positions if parse_int(idx, -1) >= 0})
            primitive_ids = [parse_int(idx, -1) for idx in primitive_ids]
            if len(set(primitive_ids)) != len(primitive_ids) or any(idx < 0 for idx in primitive_ids):
                counters["malformed_primitive_ids"] += 1
                continue
            label = parse_int(item.get("label"), -1)
            if not positions or label < 0 or label >= IGNORE_LABEL:
                counters["empty_or_ignore"] += 1
                continue
            if positive_only and not is_positive:
                counters["filtered_nonpositive"] += 1
                continue
            proposals.append(
                {
                    "label": label if is_positive else IGNORE_LABEL,
                    "primitive_positions": positions,
                    "primitive_ids": primitive_ids,
                    "score": parse_float(item.get("score"), 1.0),
                    "teacher_positive": is_positive,
                }
            )
            counters["positive_proposals" if is_positive else "hard_negative_objectness_proposals"] += 1
            counters["proposals"] += 1
        if proposals:
            by_record[record_id] = proposals
            counters["records_with_teacher"] += 1
        counters["records"] += 1
    return by_record, {
        "enabled": True,
        "path": rel(path),
        "positive_only": positive_only,
        "min_gt_iou": min_gt_iou,
        "artifact_provenance": provenance,
        **dict(counters),
    }


def teacher_targets(
    torch: Any,
    record_id: str,
    teacher_by_record: dict[str, list[dict[str, Any]]],
    token_count: int,
    num_queries: int,
    device: Any,
) -> tuple[Any, Any, int, dict[str, Any]]:
    proposals = teacher_by_record.get(str(record_id), [])
    if not proposals:
        return (
            torch.empty((0,), dtype=torch.long, device=device),
            torch.zeros((0, token_count), dtype=torch.float32, device=device),
            0,
            {"teacher_available": False, "teacher_components_total": 0, "teacher_components_kept": 0, "teacher_components_dropped": 0},
        )
    ranked = sorted(proposals, key=lambda item: (parse_float(item.get("score"), 1.0), len(item.get("primitive_positions") or [])), reverse=True)
    kept = ranked[:num_queries]
    labels = torch.empty((len(kept),), dtype=torch.long, device=device)
    masks = torch.zeros((len(kept), token_count), dtype=torch.float32, device=device)
    for idx, item in enumerate(kept):
        labels[idx] = int(item["label"])
        valid_positions = [pos for pos in item["primitive_positions"] if 0 <= int(pos) < token_count]
        if valid_positions:
            masks[idx, torch.tensor(valid_positions, dtype=torch.long, device=device)] = 1.0
    return labels, masks, len(kept), {
        "teacher_available": True,
        "teacher_components_total": len(proposals),
        "teacher_components_kept": len(kept),
        "teacher_components_dropped": max(len(proposals) - len(kept), 0),
    }


def load_candidate_proposals(path: Path | None, *, max_candidates: int, feature_dim: int) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if path is None:
        return {}, {"enabled": False, "path": None, "records": 0, "candidates": 0}
    if not path.exists():
        raise FileNotFoundError(f"candidate proposals not found: {path}")
    if int(max_candidates) <= 0 or int(feature_dim) <= 0:
        raise ValueError("candidate proposals require positive max_candidates and feature_dim")
    forbidden = {"semantic_id", "instance_id", "page_instance_id", "gt_iou", "IoU"}
    by_record: dict[str, list[dict[str, Any]]] = {}
    counters = Counter()
    for row in iter_jsonl(path):
        contract = row.get("runtime_contract") if isinstance(row.get("runtime_contract"), dict) else {}
        if contract.get("gt_free") is not True:
            raise ValueError(f"candidate proposal row is not marked gt_free: {row.get('record_id')}")
        record_id = str(row.get("record_id"))
        candidates = []
        for item in row.get("candidates", []) if isinstance(row.get("candidates"), list) else []:
            if any(key in item for key in forbidden):
                raise ValueError(f"candidate proposal leaked GT-only key for record {record_id}")
            features = item.get("candidate_features")
            if not isinstance(features, list) or len(features) != int(feature_dim):
                counters["feature_dim_mismatch"] += 1
                continue
            primitive_ids = item.get("primitive_ids")
            if not isinstance(primitive_ids, list):
                counters["missing_primitive_ids"] += 1
                continue
            parsed_ids = sorted({parse_int(value, -1) for value in primitive_ids})
            if not parsed_ids or any(value < 0 for value in parsed_ids):
                counters["malformed_primitive_ids"] += 1
                continue
            candidates.append({
                "candidate_features": [parse_float(value, 0.0) for value in features],
                "primitive_ids": parsed_ids,
                "proposal_source": item.get("proposal_source"),
                "expert_owner": item.get("expert_owner"),
            })
            if len(candidates) >= int(max_candidates):
                break
        if candidates:
            by_record[record_id] = candidates
            counters["records_with_candidates"] += 1
            counters["candidates"] += len(candidates)
        counters["records"] += 1
    return by_record, {
        "enabled": True,
        "path": rel(path),
        "max_candidates_per_record": int(max_candidates),
        "candidate_feature_dim": int(feature_dim),
        **dict(counters),
    }


def candidate_arrays_for_record(
    record: dict[str, Any],
    candidate_by_record: dict[str, list[dict[str, Any]]] | None,
    *,
    max_candidates: int,
    feature_dim: int,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if not candidate_by_record or int(max_candidates) <= 0 or int(feature_dim) <= 0:
        return None, None, None
    candidates = candidate_by_record.get(str(record.get("record_id")), [])[: int(max_candidates)]
    if not candidates:
        return None, None, None
    features = np.zeros((int(max_candidates), int(feature_dim)), dtype=np.float32)
    padding = np.ones((int(max_candidates),), dtype=np.bool_)
    primitive_rows = record.get("primitive_rows") if isinstance(record.get("primitive_rows"), list) else []
    primitive_ids = [
        parse_int(row.get("primitive_id"), -1) if isinstance(row, dict) else -1
        for row in primitive_rows
    ]
    local_by_primitive_id = {
        primitive_id: local
        for local, primitive_id in enumerate(primitive_ids)
        if primitive_id >= 0
    }
    token_count = len(primitive_ids)
    token_masks = np.zeros((int(max_candidates), token_count), dtype=np.float32)
    for index, row in enumerate(candidates):
        features[index] = np.asarray(row["candidate_features"], dtype=np.float32)
        padding[index] = False
        for primitive_id in row.get("primitive_ids", []):
            local = local_by_primitive_id.get(int(primitive_id))
            if local is not None:
                token_masks[index, local] = 1.0
    return features, padding, token_masks


def candidate_record_coverage(
    path: Path,
    *,
    limit: int | None,
    record_id_allowlist: set[str] | None,
    candidate_by_record: dict[str, list[dict[str, Any]]] | None,
    max_candidates: int,
    feature_dim: int,
) -> dict[str, Any]:
    records = 0
    records_with_candidates = 0
    candidates = 0
    tokens_in_candidate_masks = 0
    for record in iter_jsonl(path, limit):
        if record_id_allowlist is not None:
            page_id = str(record.get("original_record_id") or record.get("record_id") or "")
            if page_id not in record_id_allowlist:
                continue
        records += 1
        _features, padding, token_masks = candidate_arrays_for_record(
            record,
            candidate_by_record,
            max_candidates=max_candidates,
            feature_dim=feature_dim,
        )
        if padding is None or token_masks is None:
            continue
        valid = ~padding
        if bool(valid.any()):
            records_with_candidates += 1
            candidates += int(valid.sum())
            tokens_in_candidate_masks += int(token_masks[valid].sum())
    return {
        "records": records,
        "records_with_candidates": records_with_candidates,
        "record_coverage": records_with_candidates / max(records, 1),
        "candidates": candidates,
        "tokens_in_candidate_masks": tokens_in_candidate_masks,
    }


def teacher_record_key(record: dict[str, Any]) -> str:
    original = record.get("original_record_id")
    if original is not None:
        return str(original)
    record_id = str(record.get("record_id"))
    return record_id.split("::", 1)[0]


def should_flush_page_aware_batch(pending_count: int, pending_page_id: str | None, current_page_id: str, batch_size: int) -> bool:
    return pending_count >= batch_size and pending_page_id is not None and current_page_id != pending_page_id


def planned_page_aware_optimizer_steps(
    path: Path,
    *,
    batch_records: int,
    limit_records: int = 0,
) -> int:
    if int(batch_records) < 1:
        raise ValueError("batch_records must be positive")
    pending_count = 0
    pending_page_id: str | None = None
    optimizer_steps = 0
    limit = int(limit_records) if int(limit_records) > 0 else None
    for record in iter_jsonl(path, limit):
        current_page_id = teacher_record_key(record)
        if should_flush_page_aware_batch(
            pending_count,
            pending_page_id,
            current_page_id,
            int(batch_records),
        ):
            optimizer_steps += 1
            pending_count = 0
        pending_count += 1
        pending_page_id = current_page_id
    if pending_count:
        optimizer_steps += 1
    return optimizer_steps


def page_aware_checkpoint_completed_records(records_seen: int, *, current_record_pending: bool) -> int:
    completed = int(records_seen) - int(bool(current_record_pending))
    if completed < 0:
        raise ValueError("completed record count cannot be negative")
    return completed


def next_progress_threshold(records_completed: int, interval: int) -> int:
    interval = int(interval)
    if interval <= 0:
        return 0
    return ((int(records_completed) // interval) + 1) * interval


def capture_training_rng_state(torch: Any, np: Any, local_rng: random.Random) -> dict[str, Any]:
    state = {
        "schema_version": "floorplancad_training_rng_state_v1",
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "local_record_rng": local_rng.getstate(),
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda_all": None,
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = [value.cpu() for value in torch.cuda.get_rng_state_all()]
    return state


def restore_training_rng_state(torch: Any, np: Any, local_rng: random.Random, state: Any) -> bool:
    if not isinstance(state, dict) or state.get("schema_version") != "floorplancad_training_rng_state_v1":
        return False
    random.setstate(state["python_random"])
    np.random.set_state(state["numpy_random"])
    local_rng.setstate(state["local_record_rng"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = state.get("torch_cuda_all")
    if cuda_states is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA RNG device count mismatch: checkpoint={len(cuda_states)}, runtime={torch.cuda.device_count()}"
            )
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])
    return True


def reportable_resume_checkpoint(resume_report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in resume_report.items()
        if key not in {"rng_state", "training_rng_state"}
    }


def window_teacher_targets(
    torch: Any,
    record: dict[str, Any],
    teacher_by_record: dict[str, list[dict[str, Any]]],
    token_count: int,
    num_queries: int,
    device: Any,
) -> tuple[Any, Any, int, dict[str, Any]]:
    base_id = teacher_record_key(record)
    proposals = teacher_by_record.get(base_id, [])
    if not proposals:
        return (
            torch.empty((0,), dtype=torch.long, device=device),
            torch.zeros((0, token_count), dtype=torch.float32, device=device),
            0,
            {"teacher_available": False, "teacher_components_total": 0, "teacher_components_kept": 0, "teacher_components_dropped": 0},
        )
    primitive_rows = record.get("primitive_rows")
    if not isinstance(primitive_rows, list) or len(primitive_rows) != token_count:
        raise ValueError("teacher window requires one primitive_row per model token")
    window_primitive_ids = [parse_int(row.get("primitive_id"), -1) if isinstance(row, dict) else -1 for row in primitive_rows]
    if any(value < 0 for value in window_primitive_ids) or len(set(window_primitive_ids)) != len(window_primitive_ids):
        raise ValueError("teacher window primitive_ids must be valid and unique")
    local_by_primitive_id = {primitive_id: local for local, primitive_id in enumerate(window_primitive_ids)}
    clipped: list[dict[str, Any]] = []
    for item in proposals:
        teacher_ids = item.get("primitive_ids")
        if not isinstance(teacher_ids, list):
            raise ValueError("teacher proposal primitive_ids missing after validated load")
        positions = [local_by_primitive_id[primitive_id] for primitive_id in map(int, teacher_ids) if primitive_id in local_by_primitive_id]
        if positions:
            clipped.append({**item, "primitive_positions": sorted(set(positions))})
    if not clipped:
        return (
            torch.empty((0,), dtype=torch.long, device=device),
            torch.zeros((0, token_count), dtype=torch.float32, device=device),
            0,
            {
                "teacher_available": True,
                "teacher_components_total": len(proposals),
                "teacher_components_kept": 0,
                "teacher_components_dropped": len(proposals),
            },
        )
    return teacher_targets(torch, base_id, {base_id: clipped}, token_count, num_queries, device)


def update_teacher_diagnostics(counters: Counter, diagnostics: dict[str, Any], *, prefix: str = "teacher") -> None:
    counters[f"{prefix}_records_with_teacher"] += int(bool(diagnostics.get("teacher_available")))
    counters[f"{prefix}_components_total"] += int(diagnostics.get("teacher_components_total", 0))
    counters[f"{prefix}_components_kept"] += int(diagnostics.get("teacher_components_kept", 0))
    counters[f"{prefix}_components_dropped"] += int(diagnostics.get("teacher_components_dropped", 0))


def teacher_positive_query_loss(torch: Any, ce_query: Any, query_logits: Any, teacher_labels: Any) -> Any:
    positive = teacher_labels != IGNORE_LABEL
    if int(positive.sum().item()) <= 0:
        return query_logits.sum() * 0.0
    return ce_query(query_logits[positive], teacher_labels[positive])


def teacher_hard_negative_objectness_loss(torch: Any, ce_query: Any, query_logits: Any, mask_logits: Any, q_labels: Any, teacher_labels: Any, teacher_masks: Any) -> tuple[Any, int]:
    """Map hard negatives only onto GT-unmatched queries and supervise no-object."""
    negative_indices = torch.nonzero(teacher_labels == IGNORE_LABEL, as_tuple=False).flatten()
    available_queries = torch.nonzero(q_labels == IGNORE_LABEL, as_tuple=False).flatten().tolist()
    selected: list[int] = []
    for teacher_index in negative_indices.tolist():
        if not available_queries:
            break
        teacher_mask = teacher_masks[int(teacher_index)].float()
        probabilities = torch.sigmoid(mask_logits[available_queries].float())
        overlap = torch.einsum("qn,n->q", probabilities, teacher_mask) / teacher_mask.sum().clamp_min(1.0)
        best_local = int(torch.argmax(overlap).item())
        selected.append(int(available_queries.pop(best_local)))
    if not selected:
        return query_logits.sum() * 0.0, 0
    indices = torch.tensor(selected, dtype=torch.long, device=query_logits.device)
    targets = torch.full((len(selected),), IGNORE_LABEL, dtype=torch.long, device=query_logits.device)
    return ce_query(query_logits[indices], targets), len(selected)


def update_teacher_match_conflicts(
    torch: Any,
    counters: Counter,
    q_labels: Any,
    q_masks: Any,
    teacher_labels: Any,
    teacher_masks: Any,
) -> None:
    supervised_positive = q_labels != IGNORE_LABEL
    teacher_positive = teacher_labels != IGNORE_LABEL
    overlap = supervised_positive & teacher_positive
    overlap_count = int(overlap.sum().item())
    counters["teacher_matched_positive_queries"] += int(teacher_positive.sum().item())
    counters["teacher_supervision_overlap_queries"] += overlap_count
    if overlap_count <= 0:
        return
    counters["teacher_label_conflict_queries"] += int((q_labels[overlap] != teacher_labels[overlap]).sum().item())
    supervised_masks = q_masks[overlap].float()
    distilled_masks = teacher_masks[overlap].float()
    intersection = (supervised_masks * distilled_masks).sum(dim=-1)
    union = ((supervised_masks + distilled_masks) > 0.0).float().sum(dim=-1)
    iou = intersection / union.clamp_min(1.0)
    counters["teacher_mask_conflict_queries"] += int((iou < 0.5).sum().item())


def align_teacher_to_gt_queries(
    torch: Any,
    q_labels: Any,
    q_masks: Any,
    teacher_labels: Any,
    teacher_masks: Any,
    *,
    min_identity_iou: float = 0.5,
) -> tuple[Any, Any, dict[str, int]]:
    """Attach positive teacher targets only to the query already assigned to that GT identity."""
    aligned_labels = torch.full_like(q_labels, IGNORE_LABEL)
    aligned_masks = torch.zeros_like(q_masks)
    used_queries: set[int] = set()
    diagnostics = Counter()
    for teacher_index in range(int(teacher_labels.numel())):
        label = int(teacher_labels[teacher_index].item())
        if label == IGNORE_LABEL:
            diagnostics["teacher_negative_objectness_only"] += 1
            continue
        candidates = torch.nonzero(q_labels == label, as_tuple=False).flatten().tolist()
        best_query = None
        best_iou = -1.0
        teacher_mask = teacher_masks[teacher_index].float()
        for query_index in candidates:
            if int(query_index) in used_queries:
                continue
            gt_mask = q_masks[int(query_index)].float()
            intersection = float((teacher_mask * gt_mask).sum().item())
            union = float(((teacher_mask + gt_mask) > 0.0).float().sum().item())
            iou = intersection / max(union, 1.0)
            if iou > best_iou:
                best_iou = iou
                best_query = int(query_index)
        if best_query is None or best_iou < float(min_identity_iou):
            diagnostics["teacher_identity_unaligned"] += 1
            continue
        used_queries.add(best_query)
        aligned_labels[best_query] = q_labels[best_query]
        aligned_masks[best_query] = teacher_mask
        diagnostics["teacher_identity_aligned"] += 1
    diagnostics["gt_positive_teacher_negative_conflicts"] = 0
    return aligned_labels, aligned_masks, dict(diagnostics)


def _iter_bottleneck_provenance_strings(data: Any, prefix: str = "") -> Iterable[tuple[str, str]]:
    if not isinstance(data, dict):
        return
    for raw_key, value in data.items():
        key = str(raw_key)
        key_lower = key.lower()
        field = f"{prefix}.{key}" if prefix else key
        is_source_field = (
            key_lower in {"dataset", "input", "inputs", "provenance", "scope", "source", "split"}
            or key_lower.startswith(("input_", "source_"))
            or key_lower.endswith(("_dataset", "_input", "_provenance", "_scope", "_split"))
        )
        if is_source_field:
            stack: list[tuple[str, Any]] = [(field, value)]
            while stack:
                leaf_field, leaf_value = stack.pop()
                if isinstance(leaf_value, str):
                    yield leaf_field, leaf_value
                elif isinstance(leaf_value, dict):
                    stack.extend(
                        (f"{leaf_field}.{child_key}", child_value)
                        for child_key, child_value in leaf_value.items()
                    )
                elif isinstance(leaf_value, list):
                    stack.extend(
                        (f"{leaf_field}[{index}]", child_value)
                        for index, child_value in enumerate(leaf_value)
                    )
        elif isinstance(value, dict):
            yield from _iter_bottleneck_provenance_strings(value, field)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, dict):
                    yield from _iter_bottleneck_provenance_strings(item, f"{field}[{index}]")


def _reject_forbidden_bottleneck_provenance(data: dict[str, Any], path: Path) -> None:
    forbidden: list[tuple[str, str]] = []
    for field, value in _iter_bottleneck_provenance_strings(data):
        lowered = value.strip().lower()
        if "internal_full433" in lowered or re.search(r"(?:^|[^a-z])test(?:$|[^a-z])", lowered):
            forbidden.append((field, value))
    if forbidden:
        details = ", ".join(f"{field}={value!r}" for field, value in forbidden)
        raise ValueError(
            f"bottleneck ledger {path} has forbidden test/internal_full433 provenance: {details}; "
            "derive training weights from train/validation artifacts only"
        )


def load_bottleneck_profile(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {
            "enabled": False,
            "path": rel(path) if path is not None else None,
            "recall_limited_labels": [],
            "precision_limited_labels": [],
            "grouping_limited_labels": [],
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"bottleneck ledger {path} must contain a JSON object")
    _reject_forbidden_bottleneck_provenance(data, path)
    if "recall_limited_labels" in data:
        recall = data.get("recall_limited_labels") or []
        precision = data.get("precision_limited_labels") or []
        grouping = data.get("grouping_limited_labels") or []
    else:
        rows = data.get("top_bottlenecks") if isinstance(data.get("top_bottlenecks"), list) else []
        recall = [row.get("label") for row in rows if row.get("priority_reason") == "recall_limited_reduce_fn"]
        precision = [row.get("label") for row in rows if row.get("priority_reason") == "precision_limited_reduce_fp"]
        grouping = [row.get("label") for row in rows if row.get("priority_reason") == "balanced_rq_failure_fix_grouping"]
    return {
        "enabled": True,
        "path": rel(path),
        "recall_limited_labels": sorted({int(label) for label in recall if label is not None}),
        "precision_limited_labels": sorted({int(label) for label in precision if label is not None}),
        "grouping_limited_labels": sorted({int(label) for label in grouping if label is not None}),
    }


def parse_label_list(raw: str) -> list[int]:
    labels: list[int] = []
    for part in str(raw or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if 0 <= value < IGNORE_LABEL:
            labels.append(value)
    return sorted(set(labels))


def apply_label_overrides(profile: dict[str, Any], *, extra_recall: list[int], extra_grouping: list[int], extra_precision: list[int]) -> dict[str, Any]:
    out = dict(profile)
    out["enabled"] = bool(out.get("enabled")) or bool(extra_recall or extra_grouping or extra_precision)
    out["recall_limited_labels"] = sorted(set(int(v) for v in out.get("recall_limited_labels", [])) | set(extra_recall))
    out["grouping_limited_labels"] = sorted(set(int(v) for v in out.get("grouping_limited_labels", [])) | set(extra_grouping))
    out["precision_limited_labels"] = sorted(set(int(v) for v in out.get("precision_limited_labels", [])) | set(extra_precision))
    out["label_overrides"] = {
        "extra_recall_labels": extra_recall,
        "extra_grouping_labels": extra_grouping,
        "extra_precision_labels": extra_precision,
    }
    return out


def build_query_class_weights(
    torch: Any,
    profile: dict[str, Any],
    *,
    no_object_weight: float,
    recall_class_weight: float,
    precision_class_weight: float,
    grouping_class_weight: float,
) -> Any:
    weights = torch.ones(36, dtype=torch.float32)
    weights[IGNORE_LABEL] = float(no_object_weight)
    if profile.get("enabled"):
        for label in profile.get("recall_limited_labels", []):
            if 0 <= int(label) < IGNORE_LABEL:
                weights[int(label)] = max(float(weights[int(label)]), float(recall_class_weight))
        for label in profile.get("grouping_limited_labels", []):
            if 0 <= int(label) < IGNORE_LABEL:
                weights[int(label)] = max(float(weights[int(label)]), float(grouping_class_weight))
        for label in profile.get("precision_limited_labels", []):
            if 0 <= int(label) < IGNORE_LABEL:
                weights[int(label)] = min(float(weights[int(label)]), float(precision_class_weight))
    return weights


def load_init_checkpoint(torch: Any, model: Any, path: Path | None, args: argparse.Namespace, device: Any) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"init checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    checkpoint_abi = validate_checkpoint_abi(
        ckpt,
        allow_quality_objective_mismatch=True,
    )
    state_dict = dict(ckpt["state_dict"])
    initialized_owner_residual_gate = False
    if "ownership_residual_gate" in model.state_dict() and "ownership_residual_gate" not in state_dict:
        state_dict["ownership_residual_gate"] = torch.zeros_like(model.state_dict()["ownership_residual_gate"])
        initialized_owner_residual_gate = True
    active_feature_names = model_feature_names_for_schema(input_feature_schema_from_args(args))
    expected = {
        "feature_dim": len(active_feature_names),
        "feature_names": list(active_feature_names),
        "hidden_dim": args.hidden_dim,
        "layers": args.layers,
        "heads": args.heads,
        "num_queries": args.num_queries,
        "query_decoder_layers": args.query_decoder_layers,
        "num_labels": 36,
        "dropout": args.dropout,
        "position_encoding_version": POSITION_ENCODING_VERSION,
        "position_max_frequency_log2": POSITION_MAX_FREQUENCY_LOG2,
        "quality_head": PANOPTIC_QUALITY_HEAD_VERSION,
    }
    actual = {
        "feature_dim": len(ckpt.get("feature_names", [])),
        "feature_names": list(ckpt.get("feature_names", [])),
        "hidden_dim": int(ckpt.get("hidden_dim", -1)),
        "layers": int(ckpt.get("layers", -1)),
        "heads": int(ckpt.get("heads", -1)),
        "num_queries": int(ckpt.get("num_queries", -1)),
        "query_decoder_layers": int(ckpt.get("query_decoder_layers", -1)),
        "num_labels": int(ckpt.get("num_labels", -1)),
        "dropout": float(ckpt.get("dropout", 0.1)),
        "position_encoding_version": ckpt.get("position_encoding_version"),
        "position_max_frequency_log2": ckpt.get("position_max_frequency_log2"),
        "quality_head": ckpt.get("quality_head"),
    }
    if actual != expected:
        raise ValueError(f"init checkpoint architecture mismatch: expected={expected}, actual={actual}, path={rel(path)}")
    migrating_router = (
        bool(getattr(args, "learned_sparse_router", False))
        and ckpt.get("sparse_router_config") is None
        and (ckpt.get("checkpoint_abi") or {}).get("abi_version") == PANOPTIC_GRADIENT_CONTROL_CHECKPOINT_ABI_VERSION
    )
    source_router = ckpt.get("sparse_router_config") or {}
    target_typed = bool(getattr(args, "typed_branch_routers", False))
    source_typed = bool(source_router.get("typed_branch_routers", False))
    migrating_typed_branch = target_typed and not source_typed
    if bool(getattr(args, "learned_sparse_router", False)) and ckpt.get("sparse_router_config") is None and not migrating_router:
        raise ValueError("learned sparse router migration is restricted to an explicitly validated ABI-v6 checkpoint")
    migrating_family_seed_head = (
        getattr(args, "geometry_decoder_mode", "legacy_debug") == "geometry_v2"
        and (bool(getattr(args, "content_seeded_queries", False)) or bool(getattr(args, "component_seeded_queries", False)))
        and "family_seed_head.weight" in model.state_dict()
        and "family_seed_head.weight" not in state_dict
    )
    migrating_component_seed_head = (
        bool(getattr(args, "component_seeded_queries", False))
        and "component_seed_head.0.weight" in model.state_dict()
        and "component_seed_head.0.weight" not in state_dict
    )
    migrating_candidate_adapter = (
        bool(getattr(args, "candidate_aware_queries", False))
        and "candidate_feature_proj.0.weight" in model.state_dict()
        and "candidate_feature_proj.0.weight" not in state_dict
    )
    migrating_weak_family_fusion = (
        bool(getattr(args, "weak_family_feature_fusion", False))
        and "weak_family_feature_proj.0.weight" in model.state_dict()
        and "weak_family_feature_proj.0.weight" not in state_dict
    )
    migrating_explicit_route_classifier = (
        bool(getattr(args, "explicit_route_classifier", False))
        and "route_family_head.0.weight" in model.state_dict()
        and "route_family_head.0.weight" not in state_dict
    )
    migrating_dense_attention_adapter = (
        bool(getattr(args, "dense_attention_feature_adapter", False))
        and "dense_adapter_attn.in_proj_weight" in model.state_dict()
        and "dense_adapter_attn.in_proj_weight" not in state_dict
    )
    if migrating_router:
        incompatible = model.load_state_dict(state_dict, strict=False)
        allowed_prefixes = ("sparse_router.", "sparse_experts.", "sparse_router_norm.")
        if target_typed:
            allowed_prefixes += ("branch_routers.", "bridge_gate")
        if migrating_family_seed_head:
            allowed_prefixes += ("family_seed_head.",)
        if migrating_component_seed_head:
            allowed_prefixes += ("component_seed_head.", "thing_seed_score.")
        if migrating_candidate_adapter:
            allowed_prefixes += ("candidate_feature_proj.", "candidate_anchor_proj.")
        if migrating_weak_family_fusion:
            allowed_prefixes += ("weak_family_feature_proj.", "weak_family_feature_gate.", "weak_family_feature_norm.")
        if migrating_explicit_route_classifier:
            allowed_prefixes += (
                "route_family_head.", "route_family_embed.", "route_token_gate.",
                "route_token_norm.", "route_residual_logit_gate",
            )
        if migrating_dense_attention_adapter:
            allowed_prefixes += (
                "dense_adapter_attn.", "dense_adapter_ffn.",
                "dense_adapter_norm_attn.", "dense_adapter_norm_ffn.",
                "dense_adapter_residual_logit_gate",
            )
        unexpected = list(incompatible.unexpected_keys)
        forbidden_missing = [key for key in incompatible.missing_keys if not key.startswith(allowed_prefixes)]
        if unexpected or forbidden_missing:
            raise ValueError(
                f"v6-to-v7 router migration has incompatible state: missing={forbidden_missing}, unexpected={unexpected}"
            )
    else:
        if (
            migrating_typed_branch
            or migrating_family_seed_head
            or migrating_component_seed_head
            or migrating_candidate_adapter
            or migrating_weak_family_fusion
            or migrating_explicit_route_classifier
            or migrating_dense_attention_adapter
        ):
            incompatible = model.load_state_dict(state_dict, strict=False)
            allowed_prefixes = ("branch_routers.", "bridge_gate")
            if migrating_family_seed_head:
                allowed_prefixes += ("family_seed_head.",)
            if migrating_component_seed_head:
                allowed_prefixes += ("component_seed_head.", "thing_seed_score.")
            if migrating_candidate_adapter:
                allowed_prefixes += ("candidate_feature_proj.", "candidate_anchor_proj.")
            if migrating_weak_family_fusion:
                allowed_prefixes += ("weak_family_feature_proj.", "weak_family_feature_gate.", "weak_family_feature_norm.")
            if migrating_explicit_route_classifier:
                allowed_prefixes += (
                    "route_family_head.", "route_family_embed.", "route_token_gate.",
                    "route_token_norm.", "route_residual_logit_gate",
                )
            if migrating_dense_attention_adapter:
                allowed_prefixes += (
                    "dense_adapter_attn.", "dense_adapter_ffn.",
                    "dense_adapter_norm_attn.", "dense_adapter_norm_ffn.",
                    "dense_adapter_residual_logit_gate",
                )
            forbidden_missing = [key for key in incompatible.missing_keys if not key.startswith(allowed_prefixes)]
            if incompatible.unexpected_keys or forbidden_missing:
                raise ValueError(
                    "checkpoint migration has incompatible state: "
                    f"missing={forbidden_missing}, unexpected={list(incompatible.unexpected_keys)}"
                )
        else:
            model.load_state_dict(state_dict, strict=True)
    return {
        "path": rel(path),
        "schema_version": ckpt.get("schema_version"),
        "source_epoch": ckpt.get("epoch"),
        "source_selection_score": ckpt.get("selection_score"),
        "checkpoint_boundary": ckpt.get("checkpoint_boundary"),
        "source_quality_objective_version": checkpoint_abi.get("quality_objective_version"),
        "quality_objective_migration": checkpoint_abi.get("quality_objective_migration_allowed", False),
        "sq_rq_deployment": checkpoint_abi.get("sq_rq_deployment"),
        "strict_architecture_match": not (
            migrating_router
            or migrating_typed_branch
            or migrating_family_seed_head
            or migrating_component_seed_head
            or migrating_explicit_route_classifier
            or migrating_dense_attention_adapter
            or initialized_owner_residual_gate
        ),
        "migration": (
            "v6_weights_plus_new_random_router_parameters" if migrating_router else
            "v7_weights_plus_new_random_typed_branch_parameters" if migrating_typed_branch else
            "legacy_weights_plus_new_random_family_seed_head" if migrating_family_seed_head else
            "legacy_weights_plus_new_random_explicit_route_classifier" if migrating_explicit_route_classifier else
            "legacy_weights_plus_new_random_component_seed_head" if migrating_component_seed_head else
            "legacy_weights_plus_new_random_dense_attention_adapter" if migrating_dense_attention_adapter else
            ("legacy_ownership_weights_plus_zero_owner_residual_gate" if initialized_owner_residual_gate else None)
        ),
    }


def positive_label_weights(torch: Any, labels: Any, class_weights: Any) -> Any:
    if labels.numel() == 0:
        return labels.float()
    safe_labels = labels.clamp(min=0, max=IGNORE_LABEL)
    return class_weights.to(labels.device)[safe_labels].float().clamp_min(1e-6)


def dice_loss_per_item(torch: Any, logits: Any, targets: Any) -> Any:
    logits = logits.float()
    targets = targets.float()
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=-1)
    denom = probs.sum(dim=-1) + targets.sum(dim=-1)
    return 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)


def dice_loss(torch: Any, logits: Any, targets: Any) -> Any:
    return dice_loss_per_item(torch, logits, targets).mean()


def tversky_loss_per_item(
    torch: Any,
    probs: Any,
    targets: Any,
    *,
    alpha: float,
    beta: float,
    token_weights: Any | None = None,
) -> Any:
    weights = 1.0 if token_weights is None else token_weights.float().to(probs.device).unsqueeze(0)
    tp = (probs * targets * weights).sum(dim=-1)
    fp = (probs * (1.0 - targets) * weights).sum(dim=-1)
    fn = ((1.0 - probs) * targets * weights).sum(dim=-1)
    return 1.0 - (tp + 1.0) / (tp + float(alpha) * fp + float(beta) * fn + 1.0)


def weighted_mask_loss(
    torch: Any,
    logits: Any,
    targets: Any,
    labels: Any,
    class_weights: Any,
    *,
    positive_weight: float,
    negative_weight: float,
    focal_gamma: float,
    area_ratio_loss_weight: float,
    area_overcoverage_weight: float,
    tversky_loss_weight: float,
    tversky_alpha: float,
    tversky_beta: float,
    positive_prob_floor_loss_weight: float,
    positive_prob_floor: float,
    primitive_weights: Any | None = None,
) -> Any:
    if logits.numel() == 0:
        return logits.sum() * 0.0
    logits = logits.float()
    targets = targets.float()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if positive_weight != 1.0 or negative_weight != 1.0:
        pos_weight = torch.full_like(targets, float(positive_weight))
        neg_weight = torch.full_like(targets, float(negative_weight))
        bce = bce * torch.where(targets >= 0.5, pos_weight, neg_weight)
    if focal_gamma > 0.0:
        probs = torch.sigmoid(logits)
        pt = torch.where(targets >= 0.5, probs, 1.0 - probs)
        bce = bce * torch.pow((1.0 - pt).clamp_min(1e-6), float(focal_gamma))
    else:
        probs = torch.sigmoid(logits)
    token_weights = torch.ones_like(targets[0]) if primitive_weights is None else primitive_weights.float().to(targets.device)
    token_weights = token_weights.clamp_min(0.0)
    bce = (bce * token_weights.unsqueeze(0)).sum(dim=-1) / token_weights.sum().clamp_min(1e-6)
    weighted_probs = probs * token_weights.unsqueeze(0)
    weighted_targets = targets * token_weights.unsqueeze(0)
    inter = (probs * targets * token_weights.unsqueeze(0)).sum(dim=-1)
    denom = weighted_probs.sum(dim=-1) + weighted_targets.sum(dim=-1)
    dice = 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)
    if tversky_loss_weight > 0.0:
        dice = dice + float(tversky_loss_weight) * tversky_loss_per_item(
            torch,
            probs,
            targets,
            alpha=tversky_alpha,
            beta=tversky_beta,
            token_weights=token_weights,
        )
    if positive_prob_floor_loss_weight > 0.0:
        target_area_for_mean = weighted_targets.sum(dim=-1).clamp_min(1.0)
        positive_prob_mean = (probs * weighted_targets).sum(dim=-1) / target_area_for_mean
        floor_gap = torch.relu(float(positive_prob_floor) - positive_prob_mean)
        dice = dice + float(positive_prob_floor_loss_weight) * floor_gap.square()
    if area_ratio_loss_weight > 0.0:
        pred_area = weighted_probs.sum(dim=-1) + 1.0
        target_area = weighted_targets.sum(dim=-1).clamp_min(1.0) + 1.0
        log_area_ratio = torch.log(pred_area / target_area)
        area_loss = torch.nn.functional.smooth_l1_loss(log_area_ratio, torch.zeros_like(log_area_ratio), reduction="none")
        if area_overcoverage_weight != 1.0:
            over_weight = torch.full_like(area_loss, float(area_overcoverage_weight))
            area_loss = area_loss * torch.where(log_area_ratio > 0.0, over_weight, torch.ones_like(area_loss))
        dice = dice + float(area_ratio_loss_weight) * area_loss
    weights = positive_label_weights(torch, labels, class_weights)
    return ((bce + dice) * weights).sum() / weights.sum().clamp_min(1e-6)


def family_seed_targets(torch: Any, labels: Any, valid_mask: Any) -> tuple[Any, Any]:
    targets = torch.zeros((labels.shape[0], len(FAMILY_NAMES)), dtype=torch.float32, device=labels.device)
    valid = valid_mask.to(torch.bool) & (labels >= 0) & (labels < IGNORE_LABEL)
    for label, family in LABEL_TO_FAMILY.items():
        targets[:, FAMILY_NAMES.index(family)] = torch.where(
            labels == int(label),
            torch.ones_like(targets[:, FAMILY_NAMES.index(family)]),
            targets[:, FAMILY_NAMES.index(family)],
        )
    return targets, valid


def family_seed_loss(torch: Any, seed_logits: Any | None, labels: Any, valid_mask: Any, primitive_weights: Any | None = None) -> Any | None:
    if seed_logits is None:
        return None
    targets, valid = family_seed_targets(torch, labels, valid_mask)
    if seed_logits.shape != targets.shape:
        raise ValueError("family seed logits must be [tokens, families]")
    if not bool(valid.any().item()):
        return seed_logits.sum() * 0.0
    weights = torch.where(targets > 0.5, torch.full_like(targets, 4.0), torch.ones_like(targets))
    if primitive_weights is not None:
        weights = weights * primitive_weights.to(weights.dtype).unsqueeze(-1).clamp_min(0.0)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(seed_logits.float(), targets, reduction="none")
    return (loss[valid] * weights[valid]).sum() / weights[valid].sum().clamp_min(1e-8)


def component_seed_loss(
    torch: Any,
    component_seed_logits: Any | None,
    labels: Any,
    valid_mask: Any,
    primitive_weights: Any | None = None,
) -> Any | None:
    if component_seed_logits is None:
        return None
    family_logits = component_seed_logits[..., :len(FAMILY_NAMES)]
    return family_seed_loss(torch, family_logits, labels, valid_mask, primitive_weights)


def update_family_seed_proxy(torch: Any, counters: Counter, seed_logits: Any | None, labels: Any, valid_mask: Any) -> None:
    if seed_logits is None:
        counters["family_seed_missing_records"] += 1
        return
    targets, valid = family_seed_targets(torch, labels, valid_mask)
    if not bool(valid.any().item()):
        return
    predictions = torch.sigmoid(seed_logits.float()) >= 0.5
    counters["family_seed_records"] += 1
    counters["family_seed_targets"] += int(targets[valid].sum().item())
    counters["family_seed_predicted"] += int(predictions[valid].sum().item())
    counters["family_seed_tp"] += int((predictions[valid] & (targets[valid] > 0.5)).sum().item())
    for index, family in enumerate(FAMILY_NAMES):
        family_target = (targets[:, index] > 0.5) & valid
        family_pred = predictions[:, index] & valid
        counters[f"family_seed_{family}_target"] += int(family_target.sum().item())
        counters[f"family_seed_{family}_predicted"] += int(family_pred.sum().item())
        counters[f"family_seed_{family}_tp"] += int((family_target & family_pred).sum().item())


def family_seed_proxy_payload(counters: Counter) -> dict[str, Any]:
    tp = int(counters["family_seed_tp"])
    predicted = int(counters["family_seed_predicted"])
    target = int(counters["family_seed_targets"])
    return {
        "schema_version": "family_content_anchor_proxy_v1",
        "active": int(counters["family_seed_records"]) > 0,
        "inactive_reason": None if int(counters["family_seed_records"]) > 0 else "family_seed_logits_absent",
        "records": int(counters["family_seed_records"]),
        "missing_records": int(counters["family_seed_missing_records"]),
        "target": target,
        "predicted": predicted,
        "tp": tp,
        "precision": tp / max(predicted, 1),
        "recall": tp / max(target, 1),
        "per_family": [
            {
                "family": family,
                "target": int(counters[f"family_seed_{family}_target"]),
                "predicted": int(counters[f"family_seed_{family}_predicted"]),
                "tp": int(counters[f"family_seed_{family}_tp"]),
                "precision": int(counters[f"family_seed_{family}_tp"]) / max(int(counters[f"family_seed_{family}_predicted"]), 1),
                "recall": int(counters[f"family_seed_{family}_tp"]) / max(int(counters[f"family_seed_{family}_target"]), 1),
            }
            for family in FAMILY_NAMES
        ],
    }


def unmatched_query_empty_mask_loss(
    torch: Any,
    mask_logits: Any,
    q_labels: Any,
    admission_logits: Any,
    *,
    top_k: int,
    primitive_weights: Any | None = None,
) -> Any:
    """Penalize masks from the highest-admission unmatched queries only."""
    negative = q_labels == IGNORE_LABEL
    if int(negative.sum().item()) == 0 or int(top_k) <= 0:
        return mask_logits.sum() * 0.0
    negative_indices = torch.nonzero(negative, as_tuple=False).flatten()
    count = min(int(top_k), int(negative_indices.numel()))
    ranked = admission_logits.detach().float().index_select(0, negative_indices)
    selected = negative_indices.index_select(0, ranked.topk(count).indices)
    selected_logits = mask_logits.float().index_select(0, selected)
    loss = torch.nn.functional.softplus(selected_logits)
    if primitive_weights is None:
        return loss.mean()
    weights = primitive_weights.float().to(loss.device).clamp_min(0.0)
    return (loss * weights.unsqueeze(0)).sum() / (weights.sum().clamp_min(1e-6) * count)


def candidate_mask_prior_loss(
    torch: Any,
    mask_logits: Any,
    candidate_token_masks: Any | None,
    candidate_padding_mask: Any | None,
    *,
    thing_query_count: int,
    primitive_weights: Any | None = None,
) -> Any | None:
    if candidate_token_masks is None or candidate_padding_mask is None:
        return None
    count = min(int(thing_query_count), int(mask_logits.shape[0]), int(candidate_token_masks.shape[0]))
    if count <= 0:
        return None
    target = candidate_token_masks[:count, : mask_logits.shape[-1]].float().to(mask_logits.device)
    valid = (~candidate_padding_mask[:count].bool().to(mask_logits.device)) & (target.sum(dim=-1) > 0)
    if not bool(valid.any().item()):
        return None
    logits = mask_logits[:count].float()[valid]
    target = target[valid]
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if primitive_weights is not None:
        weights = primitive_weights.float().to(loss.device).clamp_min(0.0)
        return (loss * weights.unsqueeze(0)).sum() / (weights.sum().clamp_min(1e-6) * max(int(valid.sum().item()), 1))
    return loss.mean()


def token_offset_vote_loss(
    torch: Any,
    token_offsets: Any | None,
    features: Any,
    q_labels: Any,
    q_masks: Any,
    primitive_weights: Any | None = None,
) -> Any | None:
    if token_offsets is None:
        return None
    positive = q_labels != IGNORE_LABEL
    if not bool(positive.any().item()):
        return token_offsets.sum() * 0.0
    if token_offsets.ndim != 2 or token_offsets.shape[-1] != 2:
        raise ValueError("token_offsets must have shape [tokens, 2]")
    if features.ndim != 2 or features.shape[0] != token_offsets.shape[0] or features.shape[-1] < 6:
        raise ValueError("features must contain normalized primitive centers at columns 4 and 5")
    masks = q_masks[positive].float().to(token_offsets.device)
    if masks.shape[-1] != token_offsets.shape[0]:
        raise ValueError("q_masks width must match token_offsets")
    centers = features[:, 4:6].float().to(token_offsets.device)
    weights = primitive_weight_vector_for_masks(
        torch, primitive_weights, token_offsets.shape[0], token_offsets.device
    ).unsqueeze(0)
    component_mass = (masks * weights).sum(dim=-1, keepdim=True).clamp_min(1e-6)
    component_centers = torch.einsum("pn,nc,pn->pc", masks, centers, weights.expand_as(masks)) / component_mass
    predicted_centers = (centers.unsqueeze(0) + 0.25 * torch.tanh(token_offsets.float()).unsqueeze(0)).clamp(0.0, 1.0)
    token_error = torch.nn.functional.smooth_l1_loss(
        predicted_centers,
        component_centers.unsqueeze(1).expand_as(predicted_centers),
        reduction="none",
    ).sum(dim=-1)
    active_weights = masks * weights
    return (token_error * active_weights).sum() / active_weights.sum().clamp_min(1e-6)


def token_affinity_component_loss(
    torch: Any,
    token_affinity_embeddings: Any | None,
    q_labels: Any,
    q_masks: Any,
    primitive_weights: Any | None = None,
    *,
    negative_margin: float = 0.2,
) -> Any | None:
    if token_affinity_embeddings is None:
        return None
    positive = q_labels != IGNORE_LABEL
    if not bool(positive.any().item()):
        return token_affinity_embeddings.sum() * 0.0
    embeddings = torch.nn.functional.normalize(token_affinity_embeddings.float(), dim=-1)
    masks = q_masks[positive].float().to(embeddings.device)
    if masks.shape[-1] != embeddings.shape[0]:
        raise ValueError("q_masks width must match token affinity embeddings")
    weights = primitive_weight_vector_for_masks(
        torch, primitive_weights, embeddings.shape[0], embeddings.device
    ).unsqueeze(0)
    active_weights = masks * weights
    valid_components = active_weights.sum(dim=-1) > 0.0
    if not bool(valid_components.any().item()):
        return embeddings.sum() * 0.0
    masks = masks[valid_components]
    active_weights = active_weights[valid_components]
    prototypes = torch.einsum("pn,nd,pn->pd", masks, embeddings, active_weights)
    prototypes = torch.nn.functional.normalize(
        prototypes / active_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6),
        dim=-1,
    )
    similarity = embeddings @ prototypes.transpose(0, 1)
    positive_pull = ((1.0 - similarity.transpose(0, 1)) * active_weights).sum() / active_weights.sum().clamp_min(1e-6)
    if prototypes.shape[0] <= 1:
        return positive_pull
    prototype_similarity = prototypes @ prototypes.transpose(0, 1)
    off_diagonal = ~torch.eye(prototypes.shape[0], dtype=torch.bool, device=prototypes.device)
    negative_push = torch.relu(prototype_similarity[off_diagonal] - float(negative_margin)).square().mean()
    return positive_pull + negative_push


def focused_family_positive_query_mask(torch: Any, q_labels: Any, family: str) -> Any:
    if not family:
        return torch.zeros_like(q_labels, dtype=torch.bool)
    labels = q_labels.detach().cpu().tolist()
    selected = [
        int(label) != IGNORE_LABEL and label_family(int(label)) == family
        for label in labels
    ]
    return torch.as_tensor(selected, dtype=torch.bool, device=q_labels.device)


def family_recall_focus_loss(
    torch: Any,
    query_logits: Any,
    admission_logits: Any | None,
    quality_logits: Any | None,
    mask_logits: Any,
    q_labels: Any,
    q_masks: Any,
    *,
    family: str,
    admission_floor: float,
    mask_positive_prob_floor: float,
    quality_floor: float,
    primitive_weights: Any | None = None,
) -> Any | None:
    focus = focused_family_positive_query_mask(torch, q_labels, family)
    if not bool(focus.any().item()):
        return None
    losses = []
    if admission_logits is not None:
        floor = torch.logit(
            admission_logits.new_tensor(float(admission_floor)).clamp(min=1e-6, max=1.0 - 1e-6)
        )
        losses.append(torch.nn.functional.softplus(floor - admission_logits.float()[focus]).mean())
    focus_masks = q_masks[focus].float()
    focus_logits = mask_logits.float()[focus]
    positive_tokens = focus_masks >= 0.5
    if bool(positive_tokens.any().item()):
        probs = torch.sigmoid(focus_logits)
        if primitive_weights is None:
            token_weights = torch.ones_like(focus_masks)
        else:
            token_weights = primitive_weights.float().to(focus_masks.device).clamp_min(0.0).unsqueeze(0).expand_as(focus_masks)
        weighted_positive = positive_tokens.to(token_weights.dtype) * token_weights
        denom = weighted_positive.sum(dim=-1).clamp_min(1.0)
        mean_positive_probability = (probs * weighted_positive).sum(dim=-1) / denom
        losses.append(torch.relu(float(mask_positive_prob_floor) - mean_positive_probability).square().mean())
    if quality_logits is not None:
        foreground_probability = query_logits.float().softmax(dim=-1)[..., :IGNORE_LABEL].max(dim=-1).values
        deployment_score = foreground_probability * torch.sigmoid(quality_logits.float())
        losses.append(torch.relu(float(quality_floor) - deployment_score[focus]).square().mean())
    if not losses:
        return None
    return torch.stack(losses).mean()


def hard_recall_label_mask(torch: Any, q_labels: Any, labels: set[int]) -> Any:
    if not labels:
        return torch.zeros_like(q_labels, dtype=torch.bool)
    selected = [int(label) in labels for label in q_labels.detach().cpu().tolist()]
    return torch.as_tensor(selected, dtype=torch.bool, device=q_labels.device)


def hard_recall_admission_margin_loss(
    torch: Any,
    admission_logits: Any | None,
    q_labels: Any,
    labels: set[int],
    *,
    probability_floor: float,
) -> Any | None:
    focus = hard_recall_label_mask(torch, q_labels, labels)
    if admission_logits is None or not bool(focus.any().item()):
        return None
    floor = torch.logit(
        admission_logits.new_tensor(float(probability_floor)).clamp(min=1e-6, max=1.0 - 1e-6)
    )
    return torch.nn.functional.softplus(floor - admission_logits.float()[focus]).mean()


def hard_recall_mask_floor_loss(
    torch: Any,
    mask_logits: Any,
    q_labels: Any,
    q_masks: Any,
    labels: set[int],
    *,
    probability_floor: float,
    primitive_weights: Any | None = None,
) -> Any | None:
    focus = hard_recall_label_mask(torch, q_labels, labels)
    if not bool(focus.any().item()):
        return None
    target = q_masks[focus].float()
    positive = target >= 0.5
    if not bool(positive.any().item()):
        return None
    probs = torch.sigmoid(mask_logits.float()[focus])
    if primitive_weights is None:
        weights = torch.ones_like(target)
    else:
        weights = primitive_weights.float().to(target.device).clamp_min(0.0).unsqueeze(0).expand_as(target)
    positive_weights = positive.to(weights.dtype) * weights
    denom = positive_weights.sum(dim=-1).clamp_min(1.0)
    positive_mean = (probs * positive_weights).sum(dim=-1) / denom
    return torch.relu(float(probability_floor) - positive_mean).square().mean()


def hard_recall_quality_deployment_loss(
    torch: Any,
    query_logits: Any,
    quality_logits: Any | None,
    q_labels: Any,
    labels: set[int],
    *,
    deployment_floor: float,
) -> Any | None:
    focus = hard_recall_label_mask(torch, q_labels, labels)
    if quality_logits is None or not bool(focus.any().item()):
        return None
    foreground_probability = query_logits.float().softmax(dim=-1)[..., :IGNORE_LABEL].max(dim=-1).values
    deployment_score = foreground_probability * torch.sigmoid(quality_logits.float())
    return torch.relu(float(deployment_floor) - deployment_score[focus]).square().mean()


def explicit_route_classification_loss(
    torch: Any,
    route_logits: Any | None,
    semantic_labels: Any,
    valid_mask: Any,
) -> Any | None:
    if route_logits is None:
        return None
    if route_logits.shape[:2] != semantic_labels.shape:
        raise ValueError("route_logits must be [batch, tokens, families]")
    target = torch.full(
        semantic_labels.shape,
        -100,
        dtype=torch.long,
        device=semantic_labels.device,
    )
    valid = valid_mask.to(torch.bool) & (semantic_labels >= 0) & (semantic_labels < IGNORE_LABEL)
    for label in torch.unique(semantic_labels[valid]).detach().cpu().tolist():
        family_index = label_family_index(int(label))
        if family_index is None:
            continue
        target = torch.where(
            valid & (semantic_labels == int(label)),
            torch.full_like(target, int(family_index)),
            target,
        )
    if not bool((target >= 0).any().item()):
        return route_logits.sum() * 0.0
    return torch.nn.functional.cross_entropy(
        route_logits.reshape(-1, route_logits.shape[-1]).float(),
        target.reshape(-1),
        ignore_index=-100,
    )


def geometry_mask_connectivity_loss(
    torch: Any,
    mask_logits: Any,
    features: Any,
    targets: Any | None = None,
    *,
    neighbors: int = 4,
    boundary_margin: float = 0.25,
    neighbor_indices: Any | None = None,
    neighbor_valid: Any | None = None,
) -> Any:
    if mask_logits.numel() == 0 or features.shape[0] < 2:
        return mask_logits.sum() * 0.0
    if neighbor_indices is None or neighbor_valid is None:
        neighbor_indices, neighbor_valid, _ = sparse_endpoint_neighbor_graph(
            torch, features.unsqueeze(0), neighbors=neighbors
        )
        neighbor_indices, neighbor_valid = neighbor_indices.squeeze(0), neighbor_valid.squeeze(0)
    if neighbor_indices.ndim != 2 or neighbor_valid.shape != neighbor_indices.shape:
        raise ValueError("neighbor indices and validity must be [tokens, neighbors]")
    probabilities = torch.sigmoid(mask_logits.float())
    token_count = probabilities.shape[-1]
    if neighbor_indices.shape[0] != token_count:
        raise ValueError("neighbor indices must contain one row per mask token")
    valid_indices = neighbor_indices[neighbor_valid]
    if valid_indices.numel() and bool(((valid_indices < 0) | (valid_indices >= token_count)).any().item()):
        raise ValueError("valid geometry neighbor index is outside the local mask token range")
    safe_neighbor_indices = neighbor_indices.masked_fill(~neighbor_valid, 0)
    neighbor_probabilities = probabilities[:, safe_neighbor_indices]
    difference = probabilities.unsqueeze(-1) - neighbor_probabilities
    pair_valid = neighbor_valid.unsqueeze(0).to(difference.dtype)
    if targets is None:
        return (difference.square() * pair_valid).sum() / pair_valid.sum().clamp_min(1.0)
    target_values = targets.float()
    neighbor_targets = target_values[:, safe_neighbor_indices]
    same_component = target_values.unsqueeze(-1) * neighbor_targets * pair_valid
    across_boundary = (target_values.unsqueeze(-1) - neighbor_targets).abs() * pair_valid
    positive_pairs = same_component.sum().clamp_min(1.0)
    boundary_pairs = across_boundary.sum().clamp_min(1.0)
    connectivity = (difference.square() * same_component).sum() / positive_pairs
    separation = (torch.relu(float(boundary_margin) - difference.abs()).square() * across_boundary).sum() / boundary_pairs
    return connectivity + separation


def moe_branch_specialization_loss(torch: Any, diagnostics: dict[str, Any]) -> Any | None:
    entropies = [value["routing_entropy"] for value in diagnostics.values() if value.get("enabled") and "routing_entropy" in value]
    if not entropies:
        return None
    return torch.stack(entropies).mean()


def geometry_v2_auxiliary_loss(
    torch: Any, aux_outputs: list[dict[str, Any]], target_labels: Any, target_masks: Any, *,
    num_queries: int, primitive_weights: Any | None = None,
    intermediate_weight: float = 0.5, matching: str = "hungarian_cpu",
    query_class_weights: Any | None = None, thing_query_count: int | None = None,
    typed_stuff_slots: bool = False, selected_primitive_indices: Any | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    """Per-layer class/focal/Dice supervision for geometry-v2 query states.

    Cross-window identity supervision is computed separately from matched thing
    queries. A blanket Q-by-Q orthogonality target is impossible when Q exceeds
    the identity embedding dimension and conflicts with same-instance tracking.
    """
    if not aux_outputs:
        return target_masks.sum() * 0.0, []
    diagnostics, total = [], target_masks.sum() * 0.0
    for layer_index, layer in enumerate(aux_outputs):
        query_logits, mask_logits = layer["query_logits"].squeeze(0), layer["mask_logits"].squeeze(0)
        query_labels, query_masks, positives, _ = match_component_queries(
            torch, query_logits, mask_logits, target_labels, target_masks, num_queries,
            matching=matching, primitive_weights=primitive_weights,
            thing_query_count=thing_query_count, typed_stuff_slots=typed_stuff_slots,
            selected_primitive_indices=selected_primitive_indices,
        )
        class_weights = None if query_class_weights is None else query_class_weights.float()
        class_term = torch.nn.functional.cross_entropy(query_logits.float(), query_labels, weight=class_weights)
        positive = query_labels != IGNORE_LABEL
        if positives:
            logits, targets = mask_logits[positive], query_masks[positive]
            weights = torch.ones_like(targets) if primitive_weights is None else primitive_weights.unsqueeze(0).expand_as(targets)
            probability = torch.sigmoid(logits)
            binary_ce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            target_probability = probability * targets + (1.0 - probability) * (1.0 - targets)
            focal_term = (binary_ce * (1.0 - target_probability).pow(2.0) * weights).sum() / weights.sum().clamp_min(1.0)
            numerator = 2.0 * (probability * targets * weights).sum(-1) + 1.0
            denominator = ((probability + targets) * weights).sum(-1) + 1.0
            dice_term = (1.0 - numerator / denominator).mean()
        else:
            focal_term = mask_logits.sum() * 0.0
            dice_term = mask_logits.sum() * 0.0
        layer_weight = 1.0 if layer_index == len(aux_outputs) - 1 else float(intermediate_weight)
        total = total + layer_weight * (class_term + focal_term + dice_term)
        diagnostics.append({"layer_index": layer_index, "weight": layer_weight, "positives": int(positives), "matching": matching, "terms": ["class", "focal", "dice"]})
    return total, diagnostics


def semantic_loss_or_zero(torch: Any, ce_loss: Any, logits: Any, labels: Any) -> Any:
    valid = labels != IGNORE_LABEL
    if int(valid.sum().item()) <= 0:
        return logits.sum() * 0.0
    return ce_loss(logits, labels)


def query_objectness_loss(
    torch: Any,
    query_logits: Any,
    q_labels: Any,
    *,
    admission_logits: Any | None = None,
    positive_weight: float,
    negative_weight: float,
    positive_margin_floor_loss_weight: float,
    positive_margin_floor: float,
    negative_margin_ceiling_loss_weight: float,
    negative_margin_ceiling: float,
) -> Any:
    """Auxiliary object-vs-no-object loss aligned with apply-time admission."""
    if query_logits.numel() == 0:
        return query_logits.sum() * 0.0
    query_logits = query_logits.float()
    # Inference admits an object query when the best object class beats
    # no-object.  Using logsumexp here rewards diffuse object mass across many
    # classes and can create false admissions that hurt RQ, so train the same
    # margin used by diagnostics and application-time argmax.
    if admission_logits is None:
        object_logits = query_logits[:, :IGNORE_LABEL].max(dim=-1).values
        no_object_logits = query_logits[:, IGNORE_LABEL]
        logits = object_logits - no_object_logits
    else:
        if admission_logits.shape != q_labels.shape:
            raise ValueError("admission_logits must match q_labels")
        logits = admission_logits.float()
    targets = (q_labels != IGNORE_LABEL).float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    positive = targets >= 0.5
    negative = ~positive
    group_losses = []
    group_weights = []
    if bool(positive.any().item()):
        group_losses.append(loss[positive].mean())
        group_weights.append(float(positive_weight))
    if bool(negative.any().item()):
        group_losses.append(loss[negative].mean())
        group_weights.append(float(negative_weight))
    weight_total = max(sum(group_weights), 1e-6)
    objectness = sum(
        group_loss * group_weight for group_loss, group_weight in zip(group_losses, group_weights, strict=True)
    ) / weight_total
    if positive_margin_floor_loss_weight > 0.0:
        positive_logits = logits[targets >= 0.5]
        if positive_logits.numel() > 0:
            margin_gap = torch.relu(float(positive_margin_floor) - positive_logits)
            objectness = objectness + float(positive_margin_floor_loss_weight) * margin_gap.square().mean()
    if negative_margin_ceiling_loss_weight > 0.0:
        negative_logits = logits[targets < 0.5]
        if negative_logits.numel() > 0:
            margin_gap = torch.relu(negative_logits - float(negative_margin_ceiling))
            objectness = objectness + float(negative_margin_ceiling_loss_weight) * margin_gap.square().mean()
    return objectness


def rq_query_supervision_losses(
    torch: Any,
    ce_query: Any,
    query_logits: Any,
    q_labels: Any,
    *,
    rq_available: bool,
    admission_logits: Any | None = None,
    positive_weight: float,
    negative_weight: float,
    positive_margin_floor_loss_weight: float,
    positive_margin_floor: float,
    negative_margin_ceiling_loss_weight: float,
    negative_margin_ceiling: float,
) -> tuple[Any | None, Any | None]:
    """Mask query classification and no-object supervision when RQ targets are absent."""
    if not rq_available:
        return None, None
    query = ce_query(query_logits, q_labels)
    objectness = query_objectness_loss(
        torch,
        query_logits,
        q_labels,
        admission_logits=admission_logits,
        positive_weight=positive_weight,
        negative_weight=negative_weight,
        positive_margin_floor_loss_weight=positive_margin_floor_loss_weight,
        positive_margin_floor=positive_margin_floor,
        negative_margin_ceiling_loss_weight=negative_margin_ceiling_loss_weight,
        negative_margin_ceiling=negative_margin_ceiling,
    )
    return query, objectness


def deployment_decoded_query_masks(
    torch: Any,
    mask_logits: Any,
    ownership_logits: Any | None = None,
    *,
    admitted_queries: Any | None = None,
    mask_threshold: float = PANOPTIC_QUALITY_MASK_THRESHOLD,
) -> Any:
    if mask_logits.ndim != 2:
        raise ValueError("mask_logits must have shape [queries, tokens]")
    if not 0.0 <= float(mask_threshold) <= 1.0:
        raise ValueError("decoded mask threshold must be in [0, 1]")
    query_count, token_count = mask_logits.shape
    membership = torch.sigmoid(mask_logits.float()) >= float(mask_threshold)
    if admitted_queries is None:
        admitted = torch.ones(query_count, dtype=torch.bool, device=mask_logits.device)
    else:
        if admitted_queries.shape != (query_count,):
            raise ValueError("admitted_queries must contain one boolean per query")
        admitted = admitted_queries.bool().to(mask_logits.device)
    membership = membership & admitted.unsqueeze(-1)
    if ownership_logits is None:
        return membership
    if ownership_logits.shape != (token_count, query_count + 1):
        raise ValueError("ownership_logits must have shape [tokens, queries + null]")
    ownership = ownership_logits.float().to(mask_logits.device)
    query_evidence = ownership[:, :query_count].transpose(0, 1)
    query_evidence = query_evidence.masked_fill(~membership, -float("inf"))
    winner = torch.cat([query_evidence, ownership[:, query_count].unsqueeze(0)], dim=0).argmax(dim=0)
    query_indices = torch.arange(query_count, device=mask_logits.device).unsqueeze(1)
    return winner.unsqueeze(0) == query_indices


def primitive_weight_vector_for_masks(torch: Any, primitive_weights: Any | None, token_count: int, device: Any) -> Any:
    if primitive_weights is None:
        return torch.ones((token_count,), dtype=torch.float32, device=device)
    weights = primitive_weights.float().to(device).flatten().clamp_min(0.0)
    if weights.shape != (token_count,):
        raise ValueError("primitive_weights must contain one value per mask token")
    if not bool((weights > 0.0).any().item()):
        return torch.ones((token_count,), dtype=torch.float32, device=device)
    return weights


def rq_sq_quality_targets(
    torch: Any,
    mask_logits: Any,
    q_labels: Any,
    q_masks: Any,
    primitive_weights: Any | None = None,
    *,
    query_logits: Any | None = None,
    wrong_class_quality_scale: float = 0.25,
    positive_quality_floor_labels: set[int] | None = None,
    positive_quality_floor: float = 0.0,
    ownership_logits: Any | None = None,
    mask_threshold: float = PANOPTIC_QUALITY_MASK_THRESHOLD,
    soft_target_weight: float = 0.0,
) -> tuple[Any, Any]:
    if not 0.0 <= float(mask_threshold) <= 1.0:
        raise ValueError("quality target mask threshold must be in [0, 1]")
    if not 0.0 <= float(soft_target_weight) <= 1.0:
        raise ValueError("quality soft target weight must be in [0, 1]")
    positive = q_labels != IGNORE_LABEL
    quality_target = torch.zeros(q_labels.shape, dtype=torch.float32, device=mask_logits.device)
    if bool(positive.any().item()):
        predicted = deployment_decoded_query_masks(
            torch,
            mask_logits,
            ownership_logits,
            admitted_queries=positive,
            mask_threshold=mask_threshold,
        )[positive]
        target = q_masks[positive].bool()
        weights_1d = primitive_weight_vector_for_masks(torch, primitive_weights, mask_logits.shape[-1], mask_logits.device)
        weights = weights_1d.unsqueeze(0).expand_as(target)
        intersection = ((predicted & target).to(weights.dtype) * weights).sum(dim=-1)
        union = ((predicted | target).to(weights.dtype) * weights).sum(dim=-1)
        target_iou = intersection / union.clamp_min(1e-6)
        if float(soft_target_weight) > 0.0:
            target_float = q_masks[positive].float()
            probability = torch.sigmoid(mask_logits.float()[positive])
            soft_intersection = (probability * target_float * weights).sum(dim=-1)
            soft_union = ((probability + target_float - probability * target_float) * weights).sum(dim=-1)
            soft_iou = soft_intersection / soft_union.clamp_min(1e-6)
            target_iou = torch.maximum(target_iou, soft_iou * float(soft_target_weight))
        if query_logits is not None:
            if query_logits.shape[0] != q_labels.shape[0]:
                raise ValueError("query_logits must contain one row per quality target")
            predicted_label = query_logits.float()[positive, :IGNORE_LABEL].argmax(dim=-1)
            class_correct = predicted_label == q_labels[positive]
            target_iou = torch.where(
                class_correct,
                target_iou,
                target_iou * float(wrong_class_quality_scale),
            )
        floor_labels = positive_quality_floor_labels or set()
        if floor_labels and float(positive_quality_floor) > 0.0:
            positive_labels = q_labels[positive]
            visible_target = target.sum(dim=-1) > 0
            floor_mask = hard_recall_label_mask(torch, positive_labels, floor_labels) & visible_target
            if bool(floor_mask.any().item()):
                floor = target_iou.new_full(target_iou.shape, float(positive_quality_floor))
                target_iou = torch.where(floor_mask, torch.maximum(target_iou, floor), target_iou)
        quality_target[positive] = target_iou.detach()
    return quality_target, positive


def rq_sq_quality_ranking_pairs(
    torch: Any,
    prediction: Any,
    quality_target: Any,
    positive: Any,
    *,
    ranking_margin: float,
    ranking_top_k: int,
) -> tuple[Any, Any, Any]:
    """Select target-consistent positive-vs-unmatched ranking pairs.

    The ranking margin is capped by the observed decoded hard-IoU gap. This avoids
    contradicting calibration when a matched query has a genuinely small IoU.
    """
    negative = ~positive
    if not bool(positive.any().item()) or not bool(negative.any().item()):
        empty = prediction.new_empty((0,))
        return empty, empty, empty
    positive_scores = prediction[positive]
    positive_targets = quality_target[positive]
    negative_scores = prediction[negative]
    negative_targets = quality_target[negative]
    target_gap = positive_targets.unsqueeze(1) - negative_targets.unsqueeze(0)
    valid = target_gap > 1e-6
    if not bool(valid.any().item()):
        empty = prediction.new_empty((0,))
        return empty, empty, empty
    top_count = min(int(ranking_top_k), int(negative_scores.numel()))
    candidate_scores = negative_scores.unsqueeze(0).expand_as(target_gap).masked_fill(~valid, -float("inf"))
    selected_scores, selected_indices = candidate_scores.topk(top_count, dim=1)
    selected_valid = torch.isfinite(selected_scores)
    selected_negative_scores = negative_scores[selected_indices]
    selected_gap = target_gap.gather(1, selected_indices)
    selected_positive_scores = positive_scores.unsqueeze(1).expand_as(selected_negative_scores)
    desired_margin = selected_gap.clamp(min=0.0, max=float(ranking_margin))
    return (
        selected_positive_scores[selected_valid],
        selected_negative_scores[selected_valid],
        desired_margin[selected_valid],
    )


def rq_sq_quality_calibration_loss(
    torch: Any,
    quality_logits: Any,
    mask_logits: Any,
    q_labels: Any,
    q_masks: Any,
    primitive_weights: Any | None = None,
    *,
    ranking_weight: float = 0.25,
    ranking_margin: float = 0.05,
    ranking_top_k: int = 1,
    hard_negative_weight: float = 0.1,
    unmatched_ceiling_weight: float = 0.0,
    unmatched_ceiling_probability: float = 0.05,
    foreground_scores: Any | None = None,
    query_logits: Any | None = None,
    positive_quality_floor_labels: set[int] | None = None,
    positive_quality_floor: float = 0.0,
    ownership_logits: Any | None = None,
    mask_threshold: float = PANOPTIC_QUALITY_MASK_THRESHOLD,
    soft_target_weight: float = 0.0,
) -> Any:
    if (
        ranking_weight < 0.0
        or ranking_margin < 0.0
        or int(ranking_top_k) < 1
        or hard_negative_weight < 0.0
        or unmatched_ceiling_weight < 0.0
    ):
        raise ValueError(
            "quality ranking/hard-negative/ceiling weights must be non-negative "
            "and ranking_top_k must be positive"
        )
    if not 0.0 < float(unmatched_ceiling_probability) < 1.0:
        raise ValueError("unmatched quality ceiling probability must be in (0, 1)")
    quality_target, positive = rq_sq_quality_targets(
        torch,
        mask_logits,
        q_labels,
        q_masks,
        primitive_weights,
        query_logits=query_logits,
        positive_quality_floor_labels=positive_quality_floor_labels,
        positive_quality_floor=positive_quality_floor,
        ownership_logits=ownership_logits,
        mask_threshold=mask_threshold,
        soft_target_weight=soft_target_weight,
    )
    logits = quality_logits.float()
    quality_probability = torch.sigmoid(logits)
    if foreground_scores is None:
        foreground_probability = torch.ones_like(quality_probability)
    else:
        if foreground_scores.shape != quality_probability.shape:
            raise ValueError("foreground_scores must contain one scalar per quality logit")
        foreground_probability = foreground_scores.float().detach().clamp(min=0.0, max=1.0)
    deployment_score = foreground_probability * quality_probability
    losses = []
    if bool(positive.any().item()):
        losses.append(
            torch.nn.functional.binary_cross_entropy_with_logits(
                logits[positive], quality_target[positive], reduction="mean"
            )
        )
    negative = ~positive
    hard_negative_scores = None
    if bool(negative.any().item()):
        negative_scores = deployment_score[negative]
        safe_negative_scores = negative_scores.clamp(max=1.0 - 1e-6)
        negative_mean = -torch.log1p(-safe_negative_scores).mean()
        hard_count = min(int(ranking_top_k), int(negative_scores.numel()))
        hard_negative_scores = negative_scores.topk(hard_count).values
        safe_hard_negative_scores = hard_negative_scores.clamp(max=1.0 - 1e-6)
        hard_negative = -torch.log1p(-safe_hard_negative_scores).mean()
        losses.append(negative_mean + float(hard_negative_weight) * hard_negative)
    calibration = torch.stack(losses).mean() if losses else quality_logits.sum() * 0.0
    if float(unmatched_ceiling_weight) > 0.0 and hard_negative_scores is not None:
        ceiling_probability = logits.new_tensor(float(unmatched_ceiling_probability))
        unmatched_ceiling = torch.relu(hard_negative_scores - ceiling_probability).square().mean()
        calibration = calibration + float(unmatched_ceiling_weight) * unmatched_ceiling
    if float(ranking_weight) <= 0.0 or not bool(positive.any().item()) or not bool(negative.any().item()):
        return calibration
    positive_scores, negative_scores, desired_margin = rq_sq_quality_ranking_pairs(
        torch,
        deployment_score,
        quality_target,
        positive,
        ranking_margin=ranking_margin,
        ranking_top_k=ranking_top_k,
    )
    if positive_scores.numel() == 0:
        return calibration
    negative_probability = negative_scores.detach()
    required_positive_probability = (negative_probability + desired_margin).clamp(max=1.0 - 1e-6)
    required_positive_logits = torch.logit(required_positive_probability, eps=1e-6)
    positive_deployment_logits = torch.logit(
        positive_scores.clamp(min=1e-6, max=1.0 - 1e-6),
        eps=1e-6,
    )
    ranking = torch.relu(required_positive_logits - positive_deployment_logits).mean()
    return calibration + float(ranking_weight) * ranking


def stuff_overlap_union_consistency_loss(torch: Any, rows: list[tuple[str, int, Any, Any]]) -> tuple[Any | None, int]:
    pairs: list[Any] = []
    previous: dict[str, tuple[int, Any, Any]] = {}
    for page_id, window_index, primitive_ids, stuff_masks in sorted(rows, key=lambda row: (row[0], row[1])):
        prior = previous.get(page_id)
        if prior is not None and window_index == prior[0] + 1:
            previous_ids, previous_masks = prior[1], prior[2]
            current_lookup = {int(value): index for index, value in enumerate(primitive_ids.detach().cpu().tolist())}
            prior_lookup = {int(value): index for index, value in enumerate(previous_ids.detach().cpu().tolist())}
            shared = sorted(set(current_lookup) & set(prior_lookup))
            if shared:
                current_index = torch.as_tensor([current_lookup[value] for value in shared], device=stuff_masks.device)
                prior_index = torch.as_tensor([prior_lookup[value] for value in shared], device=previous_masks.device)
                current_probability = torch.sigmoid(stuff_masks[:, current_index])
                prior_probability = torch.sigmoid(previous_masks[:, prior_index])
                pairs.append(0.5 * (
                    torch.nn.functional.smooth_l1_loss(current_probability, prior_probability.detach())
                    + torch.nn.functional.smooth_l1_loss(prior_probability, current_probability.detach())
                ))
        previous[page_id] = (window_index, primitive_ids, stuff_masks)
    if not pairs:
        return None, 0
    return torch.stack(pairs).mean(), len(pairs)


def compose_task_loss_diagnostic(
    terms: dict[str, Any], weights: dict[str, float], *, pcgrad: bool,
) -> tuple[Any, dict[str, Any]]:
    """Compose the production-weighted scalar used only for loss diagnostics.

    PCGrad owns backward task gradients and therefore has no single optimized
    scalar.  The returned sum is nevertheless an exact, stable train/eval
    diagnostic over the same task terms and weights.
    """
    if set(terms) != set(weights):
        raise ValueError("task loss terms and weights must have identical keys")
    weighted = {
        name: (None if terms[name] is None else terms[name] * float(weights[name]))
        for name in terms
    }
    active = [value for value in weighted.values() if value is not None]
    if not active:
        raise ValueError("at least one task loss term must be available")
    total = sum(active)
    return total, {
        "aggregation": "production_weighted_sum_diagnostic_pcgrad_backward" if pcgrad else "production_weighted_sum",
        "weights": {name: float(weights[name]) for name in sorted(weights)},
        "weighted_terms": weighted,
    }


def active_loss_experts_from_args(args: argparse.Namespace) -> set[str]:
    value = str(getattr(args, "active_loss_experts", "joint_routed") or "joint_routed")
    if value == "joint_routed":
        return set(LOSS_EXPERT_GROUPS)
    aliases = {
        "quality_deployment_only": {"quality_deployment"},
        "rq_admission_only": {"rq_admission"},
        "mask_shape_only": {"mask_shape"},
        "semantic_only": {"semantic"},
        "topology_only": {"topology"},
        "teacher_aux_only": {"teacher_aux"},
    }
    if value in aliases:
        return set(aliases[value])
    requested = {item.strip() for item in value.split(",") if item.strip()}
    if not requested:
        raise ValueError("active_loss_experts must name at least one expert")
    unknown = requested - set(LOSS_EXPERT_GROUPS)
    if unknown:
        raise ValueError(f"unknown active loss experts: {sorted(unknown)}")
    return requested


def loss_weights_for_active_experts(weights: dict[str, float], active_experts: set[str]) -> dict[str, float]:
    active_terms = {
        term
        for expert, terms in LOSS_EXPERT_GROUPS.items()
        if expert in active_experts
        for term in terms
    }
    return {
        name: (float(weight) if name in active_terms else 0.0)
        for name, weight in weights.items()
    }


def auxiliary_loss_weight_for_active_experts(
    base_weight: float,
    active_experts: set[str],
    allowed_experts: set[str],
    *,
    joint_only: bool = False,
) -> float:
    if joint_only and active_experts != set(LOSS_EXPERT_GROUPS):
        return 0.0
    return float(base_weight) if active_experts & allowed_experts else 0.0


def update_loss_expert_counters(
    counters: Counter,
    terms: dict[str, Any],
    weights: dict[str, float],
    active_experts: set[str],
) -> None:
    weighted_by_term = {
        name: 0.0 if value is None else float((value.detach() * float(weights[name])).item())
        for name, value in terms.items()
    }
    for expert, expert_terms in LOSS_EXPERT_GROUPS.items():
        raw_sum = 0.0
        weighted_sum = 0.0
        active_count = 0
        for term in expert_terms:
            value = terms.get(term)
            if value is None:
                continue
            raw_sum += float(value.detach().item())
            weighted_sum += weighted_by_term.get(term, 0.0)
            active_count += 1
        counters[f"loss_expert_{expert}_raw_sum"] += raw_sum
        counters[f"loss_expert_{expert}_weighted_sum"] += weighted_sum
        counters[f"loss_expert_{expert}_active_terms"] += active_count
        counters[f"loss_expert_{expert}_records"] += int(active_count > 0)
        counters[f"loss_expert_{expert}_enabled_records"] += int(expert in active_experts)


def loss_expert_payload(counters: Counter) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    total_weighted = sum(float(counters[f"loss_expert_{expert}_weighted_sum"]) for expert in LOSS_EXPERT_GROUPS)
    for expert, terms in LOSS_EXPERT_GROUPS.items():
        records = int(counters[f"loss_expert_{expert}_records"])
        weighted = float(counters[f"loss_expert_{expert}_weighted_sum"])
        rows[expert] = {
            "terms": list(terms),
            "records": records,
            "enabled_records": int(counters[f"loss_expert_{expert}_enabled_records"]),
            "active_terms": int(counters[f"loss_expert_{expert}_active_terms"]),
            "raw_mean": float(counters[f"loss_expert_{expert}_raw_sum"]) / max(records, 1),
            "weighted_mean": weighted / max(records, 1),
            "weighted_total": weighted,
            "weighted_fraction": weighted / max(total_weighted, 1e-12),
        }
    return {
        "schema_version": "multi_loss_expert_contribution_v1",
        "groups": rows,
        "total_weighted": total_weighted,
    }


def update_target_diagnostics(counters: Counter, diagnostics: dict[str, Any]) -> None:
    counters["target_components_total"] += int(diagnostics.get("target_components_total", 0))
    counters["target_components_kept"] += int(diagnostics.get("target_components_kept", 0))
    counters["target_components_dropped"] += int(diagnostics.get("target_components_dropped", 0))
    counters["small_components_total"] += int(diagnostics.get("small_components_total", 0))
    counters["small_components_kept"] += int(diagnostics.get("small_components_kept", 0))
    counters["bottleneck_components_total"] += int(diagnostics.get("bottleneck_components_total", 0))
    counters["bottleneck_components_kept"] += int(diagnostics.get("bottleneck_components_kept", 0))
    counters["partial_mask_components_excluded"] += int(diagnostics.get("partial_mask_components_excluded", 0))
    counters["partial_mask_components_kept_window_visible"] += int(diagnostics.get("partial_mask_components_kept_window_visible", 0))
    counters["partial_mask_components_too_small"] += int(diagnostics.get("partial_mask_components_too_small", 0))
    counters["raw_target_components_total"] += int(diagnostics.get("raw_target_components_total", diagnostics.get("target_components_total", 0)))
    counters["policy_eligible_target_components"] += int(diagnostics.get("policy_eligible_target_components", diagnostics.get("target_components_kept", 0)))
    counters["policy_excluded_target_components"] += int(diagnostics.get("policy_excluded_target_components", diagnostics.get("target_components_dropped", 0)))
    counters["capacity_target_components_total"] += int(diagnostics.get("capacity_target_components_total", diagnostics.get("target_components_kept", 0)))
    counters["capacity_target_components_kept"] += int(diagnostics.get("capacity_target_components_kept", diagnostics.get("target_components_kept", 0)))
    counters["capacity_target_components_dropped"] += int(diagnostics.get("capacity_target_components_dropped", 0))


def target_diagnostic_payload(counters: Counter) -> dict[str, Any]:
    total = int(counters["target_components_total"])
    small_total = int(counters["small_components_total"])
    bottleneck_total = int(counters["bottleneck_components_total"])
    return {
        "target_components_total": total,
        "target_components_kept": int(counters["target_components_kept"]),
        "target_components_dropped": int(counters["target_components_dropped"]),
        "target_component_keep_rate": int(counters["target_components_kept"]) / max(total, 1),
        "raw_target_components_total": int(counters["raw_target_components_total"]),
        "policy_eligible_target_components": int(counters["policy_eligible_target_components"]),
        "policy_excluded_target_components": int(counters["policy_excluded_target_components"]),
        "capacity_target_components_total": int(counters["capacity_target_components_total"]),
        "capacity_target_components_kept": int(counters["capacity_target_components_kept"]),
        "capacity_target_components_dropped": int(counters["capacity_target_components_dropped"]),
        "capacity_target_keep_rate": int(counters["capacity_target_components_kept"]) / max(int(counters["capacity_target_components_total"]), 1),
        "small_components_total": small_total,
        "small_components_kept": int(counters["small_components_kept"]),
        "small_component_keep_rate": int(counters["small_components_kept"]) / max(small_total, 1),
        "bottleneck_components_total": bottleneck_total,
        "bottleneck_components_kept": int(counters["bottleneck_components_kept"]),
        "bottleneck_component_keep_rate": int(counters["bottleneck_components_kept"]) / max(bottleneck_total, 1),
        "partial_mask_components_excluded": int(counters["partial_mask_components_excluded"]),
        "partial_mask_components_kept_window_visible": int(counters["partial_mask_components_kept_window_visible"]),
        "partial_mask_components_too_small": int(counters["partial_mask_components_too_small"]),
    }


def update_component_proxy(
    torch: Any,
    counters: Counter,
    query_logits: Any,
    mask_logits: Any,
    q_labels: Any,
    q_masks: Any,
    quality_logits: Any | None = None,
    primitive_weights: Any | None = None,
    *,
    ownership_logits: Any | None = None,
    deployment_min_query_score: float = DEFAULT_MIN_QUERY_SCORE,
    deployment_mask_threshold: float = PANOPTIC_QUALITY_MASK_THRESHOLD,
) -> None:
    if not 0.0 <= float(deployment_min_query_score) <= 1.0:
        raise ValueError("deployment_min_query_score must be in [0, 1]")
    if not 0.0 <= float(deployment_mask_threshold) <= 1.0:
        raise ValueError("deployment_mask_threshold must be in [0, 1]")
    prior_min_query_score = counters.get("deployment_min_query_score")
    if prior_min_query_score is not None and float(prior_min_query_score) != float(deployment_min_query_score):
        raise ValueError("component proxy cannot mix deployment_min_query_score values")
    prior_mask_threshold = counters.get("deployment_mask_threshold")
    if prior_mask_threshold is not None and float(prior_mask_threshold) != float(deployment_mask_threshold):
        raise ValueError("component proxy cannot mix deployment_mask_threshold values")
    counters["deployment_min_query_score"] = float(deployment_min_query_score)
    counters["deployment_mask_threshold"] = float(deployment_mask_threshold)
    all_pred_labels = query_logits.argmax(dim=-1)
    positive_query_mask = q_labels != IGNORE_LABEL
    negative_query_mask = ~positive_query_mask
    predicted_object_query_mask = all_pred_labels != IGNORE_LABEL
    class_probability = query_logits.float().softmax(dim=-1)
    foreground_score = class_probability[:, :IGNORE_LABEL].max(dim=-1).values
    mask_objectness_score = query_mask_objectness_scores(torch, mask_logits)
    if quality_logits is None:
        quality_probability = torch.ones_like(foreground_score)
    else:
        if quality_logits.shape != foreground_score.shape:
            raise ValueError("quality_logits must contain one scalar per query")
        quality_probability = torch.sigmoid(quality_logits.float())
    calibrated_score = foreground_score * quality_probability * mask_objectness_score
    calibrated_admitted_query_mask = (
        predicted_object_query_mask & (calibrated_score >= float(deployment_min_query_score))
    )
    all_pred_binary = deployment_decoded_query_masks(
        torch,
        mask_logits,
        ownership_logits,
        admitted_queries=predicted_object_query_mask,
        mask_threshold=deployment_mask_threshold,
    )
    calibrated_pred_binary = deployment_decoded_query_masks(
        torch,
        mask_logits,
        ownership_logits,
        admitted_queries=calibrated_admitted_query_mask,
        mask_threshold=deployment_mask_threshold,
    )
    calibrated_pred_binary_pre_ownership = deployment_decoded_query_masks(
        torch,
        mask_logits,
        None,
        admitted_queries=calibrated_admitted_query_mask,
        mask_threshold=deployment_mask_threshold,
    )
    calibrated_proposal_query_mask = calibrated_admitted_query_mask & calibrated_pred_binary.any(dim=-1)
    calibrated_pre_ownership_proposal_query_mask = calibrated_admitted_query_mask & calibrated_pred_binary_pre_ownership.any(dim=-1)
    ownership_erased_query_mask = calibrated_pre_ownership_proposal_query_mask & ~calibrated_proposal_query_mask
    object_logits = query_logits[:, :IGNORE_LABEL].max(dim=-1).values
    no_object_logits = query_logits[:, IGNORE_LABEL]
    object_margins = object_logits - no_object_logits
    positive_margins = object_margins[positive_query_mask]
    negative_margins = object_margins[negative_query_mask]
    counters["query_total"] += int(q_labels.numel())
    counters["query_target_positive_total"] += int(positive_query_mask.sum().item())
    counters["query_target_negative_total"] += int(negative_query_mask.sum().item())
    counters["query_predicted_object_total"] += int(predicted_object_query_mask.sum().item())
    counters["calibrated_query_admitted_total"] += int(calibrated_admitted_query_mask.sum().item())
    counters["calibrated_query_proposal_total"] += int(calibrated_proposal_query_mask.sum().item())
    counters["calibrated_query_pre_ownership_proposal_total"] += int(
        calibrated_pre_ownership_proposal_query_mask.sum().item()
    )
    counters["calibrated_query_ownership_erased_total"] += int(ownership_erased_query_mask.sum().item())
    counters["calibrated_query_rejected_empty_mask_total"] += int(
        (calibrated_admitted_query_mask & ~calibrated_proposal_query_mask).sum().item()
    )
    counters["calibrated_query_rejected_low_score_total"] += int(
        (predicted_object_query_mask & ~calibrated_admitted_query_mask).sum().item()
    )
    counters["query_predicted_no_object_total"] += int((all_pred_labels == IGNORE_LABEL).sum().item())
    counters["query_positive_predicted_object"] += int((positive_query_mask & predicted_object_query_mask).sum().item())
    counters["query_negative_predicted_object"] += int((negative_query_mask & predicted_object_query_mask).sum().item())
    counters["query_negative_predicted_no_object"] += int((negative_query_mask & (all_pred_labels == IGNORE_LABEL)).sum().item())
    if int(positive_margins.numel()) > 0:
        counters["query_positive_object_margin_sum"] += float(positive_margins.sum().item())
        counters["query_positive_object_margin_items"] += int(positive_margins.numel())
        counters["query_positive_object_margin_positive"] += int((positive_margins > 0).sum().item())
    if int(negative_margins.numel()) > 0:
        counters["query_negative_object_margin_sum"] += float(negative_margins.sum().item())
        counters["query_negative_object_margin_items"] += int(negative_margins.numel())
        counters["query_negative_object_margin_positive"] += int((negative_margins > 0).sum().item())
    pos = q_labels != IGNORE_LABEL
    if int(pos.sum().item()) <= 0:
        counters["instance_proxy_fp"] += int(predicted_object_query_mask.sum().item())
        counters["calibrated_instance_proxy_fp"] += int(calibrated_proposal_query_mask.sum().item())
        return
    pos_logits = query_logits[pos]
    pos_labels = q_labels[pos]
    pos_masks = mask_logits[pos]
    target_masks = q_masks[pos]
    pred_labels_all = query_logits[:, :IGNORE_LABEL].argmax(dim=-1)
    pred_labels = pred_labels_all[pos]
    counters["component_query_positives"] += int(pos_labels.numel())
    counters["component_query_label_correct"] += int((pred_labels == pos_labels).sum().item())
    dice = 1.0 - dice_loss_per_item(torch, pos_masks, target_masks)
    counters["component_mask_dice_sum"] += float(dice.sum().item())
    counters["component_mask_items"] += int(dice.numel())
    mask_prob = torch.sigmoid(pos_masks)
    pred_binary = all_pred_binary[pos]
    target_binary = target_masks >= 0.5
    proxy_weights_1d = primitive_weight_vector_for_masks(torch, primitive_weights, mask_logits.shape[-1], mask_logits.device)
    proxy_weights = proxy_weights_1d.unsqueeze(0).expand_as(target_binary).to(mask_prob.dtype)
    intersection = ((pred_binary & target_binary).to(mask_prob.dtype) * proxy_weights).sum(dim=-1)
    union = ((pred_binary | target_binary).to(mask_prob.dtype) * proxy_weights).sum(dim=-1)
    matched_iou = intersection / union.clamp_min(1.0)
    admitted_positive = all_pred_labels[pos] != IGNORE_LABEL
    correctly_labeled = pred_labels == pos_labels
    true_positive = admitted_positive & correctly_labeled & (matched_iou >= 0.5)
    query_true_positive = int(true_positive.sum().item())
    query_false_negative = int(pos_labels.numel()) - query_true_positive
    query_false_positive = int(predicted_object_query_mask.sum().item()) - query_true_positive
    counters["instance_proxy_tp"] += query_true_positive
    counters["instance_proxy_fp"] += query_false_positive
    counters["instance_proxy_fn"] += query_false_negative
    if query_true_positive:
        counters["instance_proxy_iou_sum"] += float(matched_iou[true_positive].sum().item())
    calibrated_proposal_positive = calibrated_proposal_query_mask[pos]
    calibrated_positive_binary = calibrated_pred_binary[pos]
    calibrated_pre_ownership_positive_binary = calibrated_pred_binary_pre_ownership[pos]
    calibrated_intersection = ((calibrated_positive_binary & target_binary).to(mask_prob.dtype) * proxy_weights).sum(dim=-1)
    calibrated_union = ((calibrated_positive_binary | target_binary).to(mask_prob.dtype) * proxy_weights).sum(dim=-1)
    calibrated_iou = calibrated_intersection / calibrated_union.clamp_min(1.0)
    calibrated_true_positive = calibrated_proposal_positive & correctly_labeled & (calibrated_iou >= 0.5)
    calibrated_tp = int(calibrated_true_positive.sum().item())
    counters["calibrated_instance_proxy_tp"] += calibrated_tp
    counters["calibrated_instance_proxy_fp"] += int(calibrated_proposal_query_mask.sum().item()) - calibrated_tp
    counters["calibrated_instance_proxy_fn"] += int(pos_labels.numel()) - calibrated_tp
    if calibrated_tp:
        counters["calibrated_instance_proxy_iou_sum"] += float(
            calibrated_iou[calibrated_true_positive].sum().item()
        )
    pos_indices = torch.nonzero(pos, as_tuple=False).flatten()
    positive_predicted_object = all_pred_labels[pos] != IGNORE_LABEL
    positive_calibrated_score = calibrated_score[pos]
    positive_quality_probability = quality_probability[pos]
    positive_foreground_score = foreground_score[pos]
    positive_mask_objectness_score = mask_objectness_score[pos]
    positive_object_margins = object_margins[pos]
    positive_calibrated_admitted = calibrated_admitted_query_mask[pos]
    positive_empty_mask = positive_calibrated_admitted & ~calibrated_positive_binary.any(dim=-1)
    positive_pre_ownership_empty_mask = positive_calibrated_admitted & ~calibrated_pre_ownership_positive_binary.any(dim=-1)
    positive_ownership_erased_mask = (
        positive_calibrated_admitted
        & calibrated_pre_ownership_positive_binary.any(dim=-1)
        & ~calibrated_positive_binary.any(dim=-1)
    )
    for local_index, label_value in enumerate(pos_labels.detach().cpu().tolist()):
        label = int(label_value)
        family = label_family(label)
        prefix = f"class_{label}"
        family_prefix = f"family_{family}"
        counters[f"{prefix}_target"] += 1
        counters[f"{family_prefix}_target"] += 1
        deployment_value = float(positive_calibrated_score[local_index].detach().item())
        quality_value = float(positive_quality_probability[local_index].detach().item())
        foreground_value = float(positive_foreground_score[local_index].detach().item())
        mask_objectness_value = float(positive_mask_objectness_score[local_index].detach().item())
        margin_value = float(positive_object_margins[local_index].detach().item())
        for diagnostic_prefix in (prefix, family_prefix):
            counters.setdefault(f"{diagnostic_prefix}_positive_deployment_scores", []).append(deployment_value)
            counters.setdefault(f"{diagnostic_prefix}_positive_quality_scores", []).append(quality_value)
            counters.setdefault(f"{diagnostic_prefix}_positive_foreground_scores", []).append(foreground_value)
            counters.setdefault(f"{diagnostic_prefix}_positive_mask_objectness_scores", []).append(mask_objectness_value)
            counters[f"{diagnostic_prefix}_positive_object_margin_sum"] += margin_value
            counters[f"{diagnostic_prefix}_positive_object_margin_items"] += 1
            counters[f"{diagnostic_prefix}_positive_empty_mask"] += int(
                bool(positive_empty_mask[local_index].detach().item())
            )
            counters[f"{diagnostic_prefix}_positive_pre_ownership_empty_mask"] += int(
                bool(positive_pre_ownership_empty_mask[local_index].detach().item())
            )
            counters[f"{diagnostic_prefix}_positive_ownership_erased_mask"] += int(
                bool(positive_ownership_erased_mask[local_index].detach().item())
            )
        if bool(calibrated_true_positive[local_index].detach().item()):
            counters[f"{prefix}_tp"] += 1
            counters[f"{prefix}_iou_sum"] += float(calibrated_iou[local_index].detach().item())
            counters[f"{family_prefix}_tp"] += 1
            counters[f"{family_prefix}_iou_sum"] += float(calibrated_iou[local_index].detach().item())
        else:
            counters[f"{prefix}_fn"] += 1
            counters[f"{family_prefix}_fn"] += 1
            if not bool(positive_predicted_object[local_index].detach().item()):
                reason = "no_object"
            elif int(pred_labels[local_index].detach().item()) != label:
                reason = "wrong_class"
            elif not bool(positive_calibrated_admitted[local_index].detach().item()):
                reason = "low_score"
            elif bool(positive_empty_mask[local_index].detach().item()):
                reason = "empty_mask"
            elif not bool(calibrated_proposal_positive[local_index].detach().item()):
                reason = "low_score_or_empty_mask"
            elif float(calibrated_iou[local_index].detach().item()) <= 0.0:
                reason = "zero_overlap"
            else:
                reason = "low_iou"
            counters[f"{prefix}_fn_{reason}"] += 1
            counters[f"{family_prefix}_fn_{reason}"] += 1
            if reason in {"low_score", "empty_mask"}:
                counters[f"{prefix}_fn_low_score_or_empty_mask"] += 1
                counters[f"{family_prefix}_fn_low_score_or_empty_mask"] += 1
        if family == "furniture":
            for threshold in FURNITURE_RECALL_DIAGNOSTIC_THRESHOLDS:
                key = diagnostic_threshold_key(threshold)
                threshold_admitted = (
                    bool(positive_predicted_object[local_index].detach().item())
                    and deployment_value >= float(threshold)
                )
                threshold_proposal = threshold_admitted and bool(pred_binary[local_index].any().detach().item())
                threshold_tp = (
                    threshold_proposal
                    and bool(correctly_labeled[local_index].detach().item())
                    and float(matched_iou[local_index].detach().item()) >= 0.5
                )
                counters[f"family_furniture_threshold_{key}_target"] += 1
                counters[f"family_furniture_threshold_{key}_tp"] += int(threshold_tp)
    tp_query_indices = set(int(pos_indices[index].detach().item()) for index in torch.nonzero(calibrated_true_positive, as_tuple=False).flatten())
    pred_families = [label_family(int(label)) for label in pred_labels_all.detach().cpu().tolist()]
    all_pred_has_mask = all_pred_binary.any(dim=-1)
    for threshold in FURNITURE_RECALL_DIAGNOSTIC_THRESHOLDS:
        key = diagnostic_threshold_key(threshold)
        threshold_admitted = predicted_object_query_mask & (calibrated_score >= float(threshold))
        furniture_proposals = [
            index
            for index in torch.nonzero(threshold_admitted & all_pred_has_mask, as_tuple=False).flatten().detach().cpu().tolist()
            if pred_families[int(index)] == "furniture"
        ]
        counters[f"family_furniture_threshold_{key}_proposal"] += len(furniture_proposals)
    for query_index in torch.nonzero(calibrated_proposal_query_mask, as_tuple=False).flatten().detach().cpu().tolist():
        if int(query_index) in tp_query_indices:
            continue
        pred_label = int(pred_labels_all[int(query_index)].detach().cpu().item())
        family = label_family(pred_label)
        counters[f"class_{pred_label}_fp"] += 1
        counters[f"family_{family}_fp"] += 1
    counters["component_mask_tp"] += int((pred_binary & target_binary).sum().item())
    counters["component_mask_fp"] += int((pred_binary & ~target_binary).sum().item())
    counters["component_mask_fn"] += int((~pred_binary & target_binary).sum().item())
    counters["component_mask_predicted_positive_tokens"] += int(pred_binary.sum().item())
    counters["component_mask_target_positive_tokens"] += int(target_binary.sum().item())
    counters["component_mask_total_tokens"] += int(target_binary.numel())
    positive_prob = mask_prob[target_binary]
    negative_prob = mask_prob[~target_binary]
    if int(positive_prob.numel()) > 0:
        counters["component_mask_positive_prob_sum"] += float(positive_prob.sum().item())
        counters["component_mask_positive_prob_items"] += int(positive_prob.numel())
    if int(negative_prob.numel()) > 0:
        counters["component_mask_negative_prob_sum"] += float(negative_prob.sum().item())
        counters["component_mask_negative_prob_items"] += int(negative_prob.numel())


def update_supervised_component_proxy(
    torch: Any,
    counters: Counter,
    query_logits: Any,
    mask_logits: Any,
    q_labels: Any,
    q_masks: Any,
    quality_logits: Any | None = None,
    primitive_weights: Any | None = None,
    *,
    rq_available: bool,
    ownership_logits: Any | None = None,
    deployment_min_query_score: float = DEFAULT_MIN_QUERY_SCORE,
    deployment_mask_threshold: float = PANOPTIC_QUALITY_MASK_THRESHOLD,
) -> None:
    if not rq_available:
        counters["rq_supervision_missing_records"] += 1
        return
    counters["rq_supervision_available_records"] += 1
    update_component_proxy(
        torch,
        counters,
        query_logits,
        mask_logits,
        q_labels,
        q_masks,
        quality_logits,
        primitive_weights=primitive_weights,
        ownership_logits=ownership_logits,
        deployment_min_query_score=deployment_min_query_score,
        deployment_mask_threshold=deployment_mask_threshold,
    )


def counter_quantile(counters: Counter, key: str, quantile: float) -> float | None:
    values = counters.get(key)
    if not isinstance(values, list) or not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(float(quantile), 1.0)) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def score_diagnostics_for_prefix(counters: Counter, prefix: str, target: int) -> dict[str, Any]:
    margin_items = int(counters[f"{prefix}_positive_object_margin_items"])
    return {
        "positive_deployment_score_p10": counter_quantile(counters, f"{prefix}_positive_deployment_scores", 0.10),
        "positive_deployment_score_p50": counter_quantile(counters, f"{prefix}_positive_deployment_scores", 0.50),
        "positive_quality_score_p10": counter_quantile(counters, f"{prefix}_positive_quality_scores", 0.10),
        "positive_foreground_score_p10": counter_quantile(counters, f"{prefix}_positive_foreground_scores", 0.10),
        "positive_mask_objectness_score_p10": counter_quantile(counters, f"{prefix}_positive_mask_objectness_scores", 0.10),
        "positive_object_margin_mean": (
            float(counters[f"{prefix}_positive_object_margin_sum"]) / max(margin_items, 1)
        ),
        "positive_empty_mask_rate": int(counters[f"{prefix}_positive_empty_mask"]) / max(target, 1),
    }


def furniture_threshold_sweep_payload(counters: Counter) -> list[dict[str, Any]]:
    rows = []
    for threshold in FURNITURE_RECALL_DIAGNOSTIC_THRESHOLDS:
        key = diagnostic_threshold_key(threshold)
        target = int(counters[f"family_furniture_threshold_{key}_target"])
        tp = int(counters[f"family_furniture_threshold_{key}_tp"])
        proposal = int(counters[f"family_furniture_threshold_{key}_proposal"])
        fp = max(proposal - tp, 0)
        fn = max(target - tp, 0)
        rows.append({
            "min_query_score": float(threshold),
            "target": target,
            "proposal": proposal,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "RQ": tp / max(tp + 0.5 * fp + 0.5 * fn, 1.0),
        })
    return rows


def component_proxy_payload(counters: Counter) -> dict[str, Any]:
    positives = int(counters["component_query_positives"])
    query_total = int(counters["query_total"])
    query_target_positive_total = int(counters["query_target_positive_total"])
    query_target_negative_total = int(counters["query_target_negative_total"])
    query_predicted_object_total = int(counters["query_predicted_object_total"])
    query_negative_predicted_object = int(counters["query_negative_predicted_object"])
    query_negative_predicted_no_object = int(counters["query_negative_predicted_no_object"])
    query_positive_predicted_object = int(counters["query_positive_predicted_object"])
    positive_margin_items = int(counters["query_positive_object_margin_items"])
    negative_margin_items = int(counters["query_negative_object_margin_items"])
    mask_items = int(counters["component_mask_items"])
    tp = int(counters["component_mask_tp"])
    fp = int(counters["component_mask_fp"])
    fn = int(counters["component_mask_fn"])
    predicted_positive_tokens = int(counters["component_mask_predicted_positive_tokens"])
    target_positive_tokens = int(counters["component_mask_target_positive_tokens"])
    total_tokens = int(counters["component_mask_total_tokens"])
    label_accuracy = int(counters["component_query_label_correct"]) / max(positives, 1)
    mean_mask_dice = float(counters["component_mask_dice_sum"]) / max(mask_items, 1)
    mask_f1 = (2.0 * tp) / max(2 * tp + fp + fn, 1)
    positive_precision = tp / max(tp + fp, 1)
    positive_recall = tp / max(tp + fn, 1)
    predicted_positive_rate = predicted_positive_tokens / max(total_tokens, 1)
    target_positive_rate = target_positive_tokens / max(total_tokens, 1)
    mean_positive_mask_probability = float(counters["component_mask_positive_prob_sum"]) / max(int(counters["component_mask_positive_prob_items"]), 1)
    mean_negative_mask_probability = float(counters["component_mask_negative_prob_sum"]) / max(int(counters["component_mask_negative_prob_items"]), 1)
    instance_tp = int(counters["instance_proxy_tp"])
    instance_fp = int(counters["instance_proxy_fp"])
    instance_fn = int(counters["instance_proxy_fn"])
    instance_precision = instance_tp / max(instance_tp + instance_fp, 1)
    instance_recall = instance_tp / max(instance_tp + instance_fn, 1)
    instance_rq = instance_tp / max(instance_tp + 0.5 * instance_fp + 0.5 * instance_fn, 1.0)
    instance_sq = float(counters["instance_proxy_iou_sum"]) / max(instance_tp, 1)
    calibrated_instance_tp = int(counters["calibrated_instance_proxy_tp"])
    calibrated_instance_fp = int(counters["calibrated_instance_proxy_fp"])
    calibrated_instance_fn = int(counters["calibrated_instance_proxy_fn"])
    calibrated_instance_precision = calibrated_instance_tp / max(
        calibrated_instance_tp + calibrated_instance_fp, 1
    )
    calibrated_instance_recall = calibrated_instance_tp / max(
        calibrated_instance_tp + calibrated_instance_fn, 1
    )
    calibrated_instance_rq = calibrated_instance_tp / max(
        calibrated_instance_tp + 0.5 * calibrated_instance_fp + 0.5 * calibrated_instance_fn,
        1.0,
    )
    calibrated_instance_sq = float(counters["calibrated_instance_proxy_iou_sum"]) / max(
        calibrated_instance_tp, 1
    )
    calibrated_query_admitted_total = int(counters["calibrated_query_admitted_total"])
    calibrated_query_proposal_total = int(counters["calibrated_query_proposal_total"])
    admission_recall = query_positive_predicted_object / max(query_target_positive_total, 1)
    admission_specificity = query_negative_predicted_no_object / max(query_target_negative_total, 1)
    legacy_admission_balanced_rq = (
        2.0 * admission_recall * admission_specificity
        / max(admission_recall + admission_specificity, 1e-12)
    )
    per_class = []
    for label in range(IGNORE_LABEL):
        target = int(counters[f"class_{label}_target"])
        tp_class = int(counters[f"class_{label}_tp"])
        fp_class = int(counters[f"class_{label}_fp"])
        fn_class = int(counters[f"class_{label}_fn"])
        rq = tp_class / max(tp_class + 0.5 * fp_class + 0.5 * fn_class, 1.0)
        sq = float(counters[f"class_{label}_iou_sum"]) / max(tp_class, 1)
        per_class.append({
            "label": label,
            "class_name": CLASS_NAMES[label] if label < len(CLASS_NAMES) else f"class_{label}",
            "family": label_family(label),
            "kind": "stuff" if label in STUFF_LABELS else "thing",
            "support": target,
            "TP": tp_class,
            "FP": fp_class,
            "FN": fn_class,
            "RQ": rq,
            "SQ": sq,
            "PQ": rq * sq,
            "score_diagnostics": score_diagnostics_for_prefix(counters, f"class_{label}", target),
            "fn_attribution": {
                "no_object": int(counters[f"class_{label}_fn_no_object"]),
                "wrong_class": int(counters[f"class_{label}_fn_wrong_class"]),
                "low_score": int(counters[f"class_{label}_fn_low_score"]),
                "empty_mask": int(counters[f"class_{label}_fn_empty_mask"]),
                "low_score_or_empty_mask": int(counters[f"class_{label}_fn_low_score_or_empty_mask"]),
                "zero_overlap": int(counters[f"class_{label}_fn_zero_overlap"]),
                "low_iou": int(counters[f"class_{label}_fn_low_iou"]),
            },
        })
    per_family = []
    for family in FAMILY_LABELS:
        target = int(counters[f"family_{family}_target"])
        tp_family = int(counters[f"family_{family}_tp"])
        fp_family = int(counters[f"family_{family}_fp"])
        fn_family = int(counters[f"family_{family}_fn"])
        rq = tp_family / max(tp_family + 0.5 * fp_family + 0.5 * fn_family, 1.0)
        sq = float(counters[f"family_{family}_iou_sum"]) / max(tp_family, 1)
        per_family.append({
            "family": family,
            "support": target,
            "TP": tp_family,
            "FP": fp_family,
            "FN": fn_family,
            "RQ": rq,
            "SQ": sq,
            "PQ": rq * sq,
            "score_diagnostics": score_diagnostics_for_prefix(counters, f"family_{family}", target),
            "fn_attribution": {
                "no_object": int(counters[f"family_{family}_fn_no_object"]),
                "wrong_class": int(counters[f"family_{family}_fn_wrong_class"]),
                "low_score": int(counters[f"family_{family}_fn_low_score"]),
                "empty_mask": int(counters[f"family_{family}_fn_empty_mask"]),
                "low_score_or_empty_mask": int(counters[f"family_{family}_fn_low_score_or_empty_mask"]),
                "zero_overlap": int(counters[f"family_{family}_fn_zero_overlap"]),
                "low_iou": int(counters[f"family_{family}_fn_low_iou"]),
            },
        })
    weakest_supported_classes = sorted(
        [row for row in per_class if row["support"] >= 20],
        key=lambda row: (row["RQ"], -row["support"], row["label"]),
    )[:12]
    proxy_conservation = {
        "protocol": "calibrated_proxy_tp_fp_fn_conservation_v1",
        "target_equals_tp_plus_fn": positives == calibrated_instance_tp + calibrated_instance_fn,
        "proposal_equals_tp_plus_fp": calibrated_query_proposal_total == calibrated_instance_tp + calibrated_instance_fp,
        "class_target_equals_tp_plus_fn": sum(int(row["support"]) for row in per_class) == sum(
            int(row["TP"]) + int(row["FN"]) for row in per_class
        ),
        "class_fp_equals_calibrated_fp": sum(int(row["FP"]) for row in per_class) == calibrated_instance_fp,
        "positive_targets": positives,
        "calibrated_proposals": calibrated_query_proposal_total,
    }
    proxy_conservation["ok"] = bool(
        proxy_conservation["target_equals_tp_plus_fn"]
        and proxy_conservation["proposal_equals_tp_plus_fp"]
        and proxy_conservation["class_target_equals_tp_plus_fn"]
        and proxy_conservation["class_fp_equals_calibrated_fp"]
    )
    return {
        "proxy_protocol": "window_local_length_weighted_primitive_iou_diagnostic_only",
        "paper_metric_eligible": False,
        "component_query_positives": positives,
        "query_label_accuracy": label_accuracy,
        "query_total": query_total,
        "query_target_positive_total": query_target_positive_total,
        "query_target_negative_total": query_target_negative_total,
        "query_predicted_object_total": query_predicted_object_total,
        "query_predicted_object_rate": query_predicted_object_total / max(query_total, 1),
        "query_object_to_target_ratio": query_predicted_object_total / max(query_target_positive_total, 1),
        "query_positive_object_recall": admission_recall,
        "query_negative_object_false_positive_rate": query_negative_predicted_object / max(query_target_negative_total, 1),
        "query_negative_no_object_accuracy": admission_specificity,
        "query_positive_object_margin_mean": float(counters["query_positive_object_margin_sum"]) / max(positive_margin_items, 1),
        "query_negative_object_margin_mean": float(counters["query_negative_object_margin_sum"]) / max(negative_margin_items, 1),
        "query_positive_object_margin_positive_rate": int(counters["query_positive_object_margin_positive"]) / max(positive_margin_items, 1),
        "query_negative_object_margin_positive_rate": int(counters["query_negative_object_margin_positive"]) / max(negative_margin_items, 1),
        "mean_mask_dice": mean_mask_dice,
        "mask_token_f1": mask_f1,
        "mask_token_precision": positive_precision,
        "mask_token_recall": positive_recall,
        "predicted_positive_tokens": predicted_positive_tokens,
        "target_positive_tokens": target_positive_tokens,
        "mask_total_tokens": total_tokens,
        "predicted_positive_rate": predicted_positive_rate,
        "target_positive_rate": target_positive_rate,
        "positive_rate_ratio": predicted_positive_rate / max(target_positive_rate, 1e-12),
        "mean_positive_mask_probability": mean_positive_mask_probability,
        "mean_negative_mask_probability": mean_negative_mask_probability,
        "positive_negative_probability_gap": mean_positive_mask_probability - mean_negative_mask_probability,
        "component_proxy_score": 0.5 * label_accuracy + 0.25 * mean_mask_dice + 0.25 * mask_f1,
        "instance_proxy_iou_threshold": 0.5,
        "instance_proxy_tp": instance_tp,
        "instance_proxy_fp": instance_fp,
        "instance_proxy_fn": instance_fn,
        "instance_proxy_precision": instance_precision,
        "instance_proxy_recall": instance_recall,
        "instance_proxy_rq": instance_rq,
        "instance_proxy_protocol": "query_class_and_length_weighted_binary_mask_iou_0p5_tp_fp_fn_v2",
        "instance_proxy_sq": instance_sq,
        "instance_proxy_pq": instance_rq * instance_sq,
        "calibrated_instance_proxy_protocol": "respect_no_object_class_times_quality_times_mask_objectness_threshold_then_length_weighted_binary_mask_iou_0p5_tp_fp_fn_v3",
        "calibrated_instance_proxy_min_query_score": float(
            counters.get("deployment_min_query_score", DEFAULT_MIN_QUERY_SCORE)
        ),
        "calibrated_instance_proxy_mask_threshold": float(
            counters.get("deployment_mask_threshold", PANOPTIC_QUALITY_MASK_THRESHOLD)
        ),
        "calibrated_query_admitted_total": calibrated_query_admitted_total,
        "calibrated_query_admitted_coverage": calibrated_query_admitted_total / max(query_total, 1),
        "calibrated_query_proposal_total": calibrated_query_proposal_total,
        "calibrated_query_proposal_coverage": calibrated_query_proposal_total / max(query_total, 1),
        "calibrated_query_rejected_low_score_total": int(
            counters["calibrated_query_rejected_low_score_total"]
        ),
        "calibrated_query_rejected_empty_mask_total": int(
            counters["calibrated_query_rejected_empty_mask_total"]
        ),
        "calibrated_instance_proxy_tp": calibrated_instance_tp,
        "calibrated_instance_proxy_fp": calibrated_instance_fp,
        "calibrated_instance_proxy_fn": calibrated_instance_fn,
        "calibrated_instance_proxy_precision": calibrated_instance_precision,
        "calibrated_instance_proxy_recall": calibrated_instance_recall,
        "calibrated_instance_proxy_rq": calibrated_instance_rq,
        "calibrated_instance_proxy_sq": calibrated_instance_sq,
        "calibrated_instance_proxy_pq": calibrated_instance_rq * calibrated_instance_sq,
        "legacy_admission_specificity": admission_specificity,
        "legacy_admission_balanced_rq": legacy_admission_balanced_rq,
        "legacy_admission_balanced_rq_protocol": "harmonic_positive_object_recall_and_negative_no_object_accuracy_not_instance_rq",
        "legacy_matched_mask_token_iou": positive_precision * positive_recall / max(positive_precision + positive_recall - positive_precision * positive_recall, 1e-12),
        "proxy_conservation": proxy_conservation,
        "per_class": per_class,
        "per_family": per_family,
        "furniture_threshold_sweep": furniture_threshold_sweep_payload(counters),
        "weakest_supported_classes": weakest_supported_classes,
    }


def evaluate(
    model: Any,
    pack: dict[str, Any],
    path: Path,
    device: Any,
    max_tokens: int,
    limit: int | None,
    num_queries: int,
    query_class_weights: Any,
    bottleneck_profile: dict[str, Any] | None,
    small_component_size: int,
    mask_positive_weight: float,
    mask_negative_weight: float,
    mask_focal_gamma: float,
    mask_area_ratio_loss_weight: float,
    mask_area_overcoverage_weight: float,
    mask_tversky_loss_weight: float,
    mask_tversky_alpha: float,
    mask_tversky_beta: float,
    mask_positive_prob_floor_loss_weight: float,
    mask_positive_prob_floor: float,
    objectness_loss_weight: float,
    objectness_positive_weight: float,
    objectness_negative_weight: float,
    objectness_positive_margin_floor_loss_weight: float,
    objectness_positive_margin_floor: float,
    objectness_negative_margin_ceiling_loss_weight: float,
    objectness_negative_margin_ceiling: float,
    amp_dtype: Any | None = None,
    teacher_by_record: dict[str, list[dict[str, Any]]] | None = None,
    teacher_loss_weight: float = 0.0,
    teacher_mask_loss_weight: float = 1.0,
    teacher_query_loss_weight: float = 0.5,
    matching: str = "hungarian_cpu",
    semantic_loss_weight: float = 1.0,
    query_loss_weight: float = 1.0,
    mask_loss_weight: float = 1.0,
    rq_sq_quality_calibration_loss_weight: float = 0.0,
    rq_sq_quality_ranking_weight: float = 0.25,
    rq_sq_quality_ranking_margin: float = 0.05,
    rq_sq_quality_ranking_top_k: int = 1,
    rq_sq_quality_hard_negative_weight: float = 0.1,
    rq_sq_quality_unmatched_ceiling_weight: float = 0.0,
    rq_sq_quality_unmatched_ceiling_probability: float = 0.05,
    quality_soft_target_weight: float = 0.0,
    positive_quality_floor_labels: set[int] | None = None,
    positive_quality_floor: float = 0.0,
    ownership_loss_weight: float = 0.0,
    ownership_mask_consistency_loss_weight: float = 0.0,
    identity_loss_weight: float = 0.0,
    identity_temperature: float = 0.1,
    identity_negative_margin: float = 0.25,
    geometry_aux_loss_weight: float = 0.5,
    content_anchor_loss_weight: float = 0.0,
    offset_vote_loss_weight: float = 0.0,
    affinity_loss_weight: float = 0.0,
    geometry_decoder_mode: str = "legacy_debug",
    thing_query_count: int | None = None,
    semantic_class_weights_value: Any | None = None,
    pcgrad_diagnostic: bool = False,
    router_load_balance_loss_weight: float = 0.0,
    partial_component_policy: str = "exclude",
    partial_component_min_tokens: int = 1,
    semantic_label_smoothing: float = 0.0,
    query_label_smoothing: float = 0.0,
    unmatched_mask_negative_loss_weight: float = 0.0,
    unmatched_mask_negative_top_k: int = 0,
    deployment_min_query_score: float = DEFAULT_MIN_QUERY_SCORE,
    deployment_mask_threshold: float = PANOPTIC_QUALITY_MASK_THRESHOLD,
    record_id_allowlist: set[str] | None = None,
    candidate_by_record: dict[str, list[list[float]]] | None = None,
    max_candidate_queries: int = 0,
    candidate_feature_dim: int = 0,
) -> dict[str, Any]:
    torch = pack["torch"]
    nn = pack["nn"]
    ce_sem = nn.CrossEntropyLoss(
        ignore_index=IGNORE_LABEL, weight=semantic_class_weights_value,
        label_smoothing=float(semantic_label_smoothing),
    )
    ce_query = nn.CrossEntropyLoss(
        weight=query_class_weights.to(device), label_smoothing=float(query_label_smoothing),
    )
    model.eval()
    counters = Counter()
    quality_positive_predictions: list[float] = []
    quality_negative_predictions: list[float] = []
    quality_positive_deployment_predictions: list[float] = []
    quality_negative_deployment_predictions: list[float] = []
    loss_sum = 0.0
    identity_previous: dict[str, tuple[Any, list[str | None], Any, int]] = {}
    diagnostic_weights = {
        "semantic": semantic_loss_weight,
        "query": query_loss_weight,
        "query_objectness": objectness_loss_weight,
        "quality_calibration": rq_sq_quality_calibration_loss_weight,
        "mask": mask_loss_weight,
        "ownership": ownership_loss_weight,
        "teacher": teacher_loss_weight,
        "geometry_aux": geometry_aux_loss_weight,
        "content_anchor": content_anchor_loss_weight,
        "offset_vote": offset_vote_loss_weight,
        "affinity": affinity_loss_weight,
        "router_load_balance": router_load_balance_loss_weight,
    }
    with torch.no_grad():
        for record in iter_jsonl(path, limit):
            if record_id_allowlist is not None:
                page_id = str(record.get("original_record_id") or record.get("record_id") or "")
                if page_id not in record_id_allowlist:
                    continue
            arrays = load_panoptic_target_arrays(record, max_tokens, training=True, num_queries=num_queries)
            if arrays is None:
                continue
            if len(arrays) == 8:
                arrays = (*arrays, None, None)
            x_np, y_np, inst_np, _prim_np, semantic_weights_np, length_weights_np, mask_valid_np, page_instance_ids, segment_features_np, segment_padding_np = arrays
            record_id = str(record.get("record_id"))
            availability = task_availability(record)
            x = torch.from_numpy(x_np).to(device).unsqueeze(0)
            y = torch.from_numpy(y_np).to(device)
            inst = torch.from_numpy(inst_np).to(device)
            semantic_weights = torch.from_numpy(semantic_weights_np).to(device)
            length_weights = torch.from_numpy(length_weights_np).to(device)
            mask_valid = torch.from_numpy(mask_valid_np).to(device)
            segment_features = None if segment_features_np is None else torch.from_numpy(segment_features_np).to(device).unsqueeze(0)
            segment_padding = None if segment_padding_np is None else torch.from_numpy(segment_padding_np).to(device).unsqueeze(0)
            candidate_np, candidate_padding_np, candidate_mask_np = candidate_arrays_for_record(
                record,
                candidate_by_record,
                max_candidates=max_candidate_queries,
                feature_dim=candidate_feature_dim,
            )
            candidate_features = None if candidate_np is None else torch.from_numpy(candidate_np).to(device).unsqueeze(0)
            candidate_padding = None if candidate_padding_np is None else torch.from_numpy(candidate_padding_np).to(device).unsqueeze(0)
            candidate_token_masks = None if candidate_mask_np is None else torch.from_numpy(candidate_mask_np).to(device).unsqueeze(0)
            if candidate_padding_np is not None:
                counters["candidate_records_with_candidates"] += 1
                counters["candidate_valid_total"] += int((~candidate_padding_np).sum())
                counters["candidate_mask_token_total"] += int(candidate_mask_np.sum()) if candidate_mask_np is not None else 0
            with autocast_context(torch, device, amp_dtype):
                outputs = model(
                    x, segment_features=segment_features, segment_padding_mask=segment_padding,
                    candidate_features=candidate_features, candidate_padding_mask=candidate_padding,
                    candidate_token_masks=candidate_token_masks,
                    return_quality=True, return_identity=True,
                )
                semantic_logits, query_logits, mask_logits = outputs[:3]
                quality_logits = outputs[3] if len(outputs) > 3 else None
                identity_embeddings = outputs[4] if len(outputs) > 4 else None
            runtime_model = getattr(model, "_orig_mod", model)
            ownership_logits = getattr(runtime_model, "last_ownership_logits", None)
            admission_logits = getattr(runtime_model, "last_query_admission_logits", None)
            geometry_aux_outputs = getattr(runtime_model, "last_aux_outputs", []) if geometry_decoder_mode == "geometry_v2" else []
            sq_rq_outputs = getattr(runtime_model, "last_sq_rq_outputs", None)
            router_diagnostics = getattr(runtime_model, "last_router_diagnostics", None)
            query_seed_diagnostics = getattr(runtime_model, "last_query_seed_diagnostics", None)
            family_seed_logits_b = getattr(runtime_model, "last_family_seed_logits", None)
            component_seed_logits_b = getattr(runtime_model, "last_component_seed_logits", None)
            token_offsets_b = getattr(runtime_model, "last_token_offsets", None)
            token_affinity_b = getattr(runtime_model, "last_token_affinity_embeddings", None)
            semantic_logits = semantic_logits.squeeze(0).float()
            query_logits = query_logits.squeeze(0).float()
            mask_logits = mask_logits.squeeze(0).float()
            quality_logits = quality_logits.squeeze(0).float() if quality_logits is not None else query_logits.sum(dim=-1) * 0.0
            family_seed_logits = None if family_seed_logits_b is None else family_seed_logits_b.squeeze(0).float()
            component_seed_logits = None if component_seed_logits_b is None else component_seed_logits_b.squeeze(0).float()
            token_offsets = None if token_offsets_b is None else token_offsets_b.squeeze(0).float()
            token_affinity = None if token_affinity_b is None else token_affinity_b.squeeze(0).float()
            admission_logits = (
                None if admission_logits is None else admission_logits.squeeze(0).float()
            )
            target_labels, target_masks, target_weights, positives, target_diag = component_targets_schema_v2(
                torch, y, page_instance_ids, mask_valid, length_weights, num_queries,
                partial_component_policy=partial_component_policy,
                partial_component_min_tokens=partial_component_min_tokens,
            )
            update_target_diagnostics(counters, target_diag)
            selected_primitive_indices = query_selected_primitive_indices(
                torch,
                query_seed_diagnostics,
                batch_index=0,
                num_queries=num_queries,
                token_count=y.numel(),
                device=device,
            )
            if availability["rq"]:
                q_labels, q_masks, positives, matched = match_component_queries(
                    torch, query_logits, mask_logits, target_labels, target_masks, num_queries, matching=matching, primitive_weights=target_weights,
                    thing_query_count=thing_query_count, typed_stuff_slots=bool(getattr(runtime_model, "typed_stuff_slots", False)),
                    selected_primitive_indices=selected_primitive_indices,
                )
            else:
                q_labels = torch.full((num_queries,), IGNORE_LABEL, dtype=torch.long, device=device)
                q_masks = torch.zeros((num_queries, y.numel()), dtype=torch.float32, device=device)
                positives = matched = 0
            matched_ids, matched_valid = matched_query_page_instance_ids(
                torch, q_labels, q_masks, page_instance_ids, mask_valid
            )
            update_supervised_component_proxy(
                torch,
                counters,
                query_logits,
                mask_logits,
                q_labels,
                q_masks,
                quality_logits if availability["quality"] else None,
                primitive_weights=target_weights,
                rq_available=availability["rq"],
                ownership_logits=(
                    None
                    if ownership_logits is None
                    else ownership_logits[0, : y.numel()].float()
                ),
                deployment_min_query_score=deployment_min_query_score,
                deployment_mask_threshold=deployment_mask_threshold,
            )
            update_family_seed_proxy(torch, counters, family_seed_logits, y, mask_valid)
            teacher_loss = query_logits.sum() * 0.0
            teacher_pos_count = 0
            if availability["teacher"] and teacher_by_record and teacher_loss_weight > 0.0:
                t_labels, t_masks, teacher_pos_count, t_diag = window_teacher_targets(torch, record, teacher_by_record, y.numel(), num_queries, device)
                update_teacher_diagnostics(counters, t_diag, prefix="teacher")
                if int(t_diag.get("teacher_components_kept", 0)) > 0:
                    tq_labels, tq_masks, identity_diag = align_teacher_to_gt_queries(torch, q_labels, q_masks, t_labels, t_masks)
                    counters.update(identity_diag)
                    tpos = tq_labels != IGNORE_LABEL
                    update_teacher_match_conflicts(torch, counters, q_labels, q_masks, tq_labels, tq_masks)
                    teacher_negative, teacher_negative_count = teacher_hard_negative_objectness_loss(
                        torch, ce_query, query_logits, mask_logits, q_labels, t_labels, t_masks
                    )
                    counters["teacher_hard_negative_objectness_queries"] += teacher_negative_count
                    if int(tpos.sum().item()) > 0:
                        teacher_query = teacher_positive_query_loss(torch, ce_query, query_logits, tq_labels)
                        teacher_mask = weighted_mask_loss(
                            torch,
                            mask_logits[tpos],
                            tq_masks[tpos],
                            tq_labels[tpos],
                            query_class_weights,
                            positive_weight=mask_positive_weight,
                            negative_weight=mask_negative_weight,
                            focal_gamma=mask_focal_gamma,
                            area_ratio_loss_weight=mask_area_ratio_loss_weight,
                            area_overcoverage_weight=mask_area_overcoverage_weight,
                            tversky_loss_weight=mask_tversky_loss_weight,
                            tversky_alpha=mask_tversky_alpha,
                            tversky_beta=mask_tversky_beta,
                            positive_prob_floor_loss_weight=mask_positive_prob_floor_loss_weight,
                            positive_prob_floor=mask_positive_prob_floor,
                            primitive_weights=target_weights,
                        )
                        teacher_loss = teacher_query_loss_weight * (teacher_query + teacher_negative) + teacher_mask_loss_weight * teacher_mask
                    else:
                        teacher_loss = teacher_query_loss_weight * teacher_negative
            pos = q_labels != IGNORE_LABEL
            semantic = weighted_semantic_loss_schema_v2(
                torch, semantic_logits, y, semantic_weights,
                semantic_class_weights_value, semantic_label_smoothing,
            ) if availability["semantic"] else None
            query, objectness = rq_query_supervision_losses(
                torch,
                ce_query,
                query_logits,
                q_labels,
                rq_available=availability["rq"],
                admission_logits=admission_logits,
                positive_weight=objectness_positive_weight,
                negative_weight=objectness_negative_weight,
                positive_margin_floor_loss_weight=objectness_positive_margin_floor_loss_weight,
                positive_margin_floor=objectness_positive_margin_floor,
                negative_margin_ceiling_loss_weight=objectness_negative_margin_ceiling_loss_weight,
                negative_margin_ceiling=objectness_negative_margin_ceiling,
            )
            quality_prediction = quality_foreground_score = quality_deployment_prediction = None
            if availability["quality"]:
                quality_prediction, quality_foreground_score, quality_deployment_prediction = (
                    rq_sq_quality_deployment_scores(torch, query_logits, quality_logits, mask_logits)
                )
            quality_calibration = rq_sq_quality_calibration_loss(
                torch, quality_logits, mask_logits, q_labels, q_masks, target_weights,
                ranking_weight=rq_sq_quality_ranking_weight,
                ranking_margin=rq_sq_quality_ranking_margin,
                ranking_top_k=rq_sq_quality_ranking_top_k,
                hard_negative_weight=rq_sq_quality_hard_negative_weight,
                unmatched_ceiling_weight=rq_sq_quality_unmatched_ceiling_weight,
                unmatched_ceiling_probability=rq_sq_quality_unmatched_ceiling_probability,
                foreground_scores=quality_foreground_score,
                query_logits=query_logits,
                positive_quality_floor_labels=positive_quality_floor_labels,
                positive_quality_floor=positive_quality_floor,
                ownership_logits=(
                    None
                    if ownership_logits is None
                    else ownership_logits[0, : y.numel()].float()
                ),
                mask_threshold=deployment_mask_threshold,
                soft_target_weight=quality_soft_target_weight,
            ) if availability["quality"] else None
            if availability["quality"]:
                quality_target, quality_positive = rq_sq_quality_targets(
                    torch,
                    mask_logits,
                    q_labels,
                    q_masks,
                    target_weights,
                    query_logits=query_logits,
                    positive_quality_floor_labels=positive_quality_floor_labels,
                    positive_quality_floor=positive_quality_floor,
                    ownership_logits=(
                        None
                        if ownership_logits is None
                        else ownership_logits[0, : y.numel()].float()
                    ),
                    mask_threshold=deployment_mask_threshold,
                    soft_target_weight=quality_soft_target_weight,
                )
                quality_negative = ~quality_positive
                if bool(quality_positive.any().item()):
                    quality_positive_predictions.extend(
                        float(value) for value in quality_prediction[quality_positive].cpu().tolist()
                    )
                    quality_positive_deployment_predictions.extend(
                        float(value)
                        for value in quality_deployment_prediction[quality_positive].cpu().tolist()
                    )
                    counters["quality_items"] += int(quality_positive.sum().item())
                    counters["quality_prediction_sum"] += float(quality_prediction[quality_positive].sum().item())
                    counters["quality_deployment_prediction_sum"] += float(
                        quality_deployment_prediction[quality_positive].sum().item()
                    )
                    counters["quality_target_sum"] += float(quality_target[quality_positive].sum().item())
                    counters["quality_absolute_error_sum"] += float((quality_prediction[quality_positive] - quality_target[quality_positive]).abs().sum().item())
                if bool(quality_negative.any().item()):
                    quality_negative_predictions.extend(
                        float(value) for value in quality_prediction[quality_negative].cpu().tolist()
                    )
                    quality_negative_deployment_predictions.extend(
                        float(value)
                        for value in quality_deployment_prediction[quality_negative].cpu().tolist()
                    )
                    counters["quality_negative_items"] += int(quality_negative.sum().item())
                    counters["quality_negative_prediction_sum"] += float(quality_prediction[quality_negative].sum().item())
                    counters["quality_negative_deployment_prediction_sum"] += float(
                        quality_deployment_prediction[quality_negative].sum().item()
                    )
                if bool(quality_positive.any().item()) and bool(quality_negative.any().item()):
                    positive_scores, negative_scores, desired_margin = rq_sq_quality_ranking_pairs(
                        torch,
                        quality_deployment_prediction,
                        quality_target,
                        quality_positive,
                        ranking_margin=rq_sq_quality_ranking_margin,
                        ranking_top_k=rq_sq_quality_ranking_top_k,
                    )
                    counters["quality_ranking_pairs"] += int(positive_scores.numel())
                    counters["quality_ranking_violations"] += int(
                        (positive_scores - negative_scores < desired_margin).sum().item()
                    )
            if ownership_logits is None or not availability["ownership"]:
                ownership_loss = None
            else:
                counters["ownership_available_records"] += 1
                owner_target = ownership_targets(torch, q_masks, q_labels, mask_valid)
                ownership_ce = ownership_cross_entropy(
                    torch, ownership_logits[0, : y.numel()].float(), owner_target, length_weights
                )
                ownership_consistency = ownership_mask_consistency_loss(
                    torch, ownership_logits[0, : y.numel()].float(), mask_logits, q_labels, mask_valid,
                    no_object_label=IGNORE_LABEL, primitive_weights=length_weights,
                )
                ownership_loss = ownership_ce + float(ownership_mask_consistency_loss_weight) * ownership_consistency
            mask = mask_logits[pos]
            target = q_masks[pos]
            mask_labels = q_labels[pos]
            mask_loss = (
                mask.sum() * 0.0
                if positives == 0
                else weighted_mask_loss(
                    torch,
                    mask,
                    target,
                    mask_labels,
                    query_class_weights,
                    positive_weight=mask_positive_weight,
                    negative_weight=mask_negative_weight,
                    focal_gamma=mask_focal_gamma,
                    area_ratio_loss_weight=mask_area_ratio_loss_weight,
                    area_overcoverage_weight=mask_area_overcoverage_weight,
                    tversky_loss_weight=mask_tversky_loss_weight,
                    tversky_alpha=mask_tversky_alpha,
                    tversky_beta=mask_tversky_beta,
                    positive_prob_floor_loss_weight=mask_positive_prob_floor_loss_weight,
                    positive_prob_floor=mask_positive_prob_floor,
                    primitive_weights=target_weights,
                )
            ) if availability["rq"] else None
            if mask_loss is not None and admission_logits is not None and unmatched_mask_negative_loss_weight > 0.0:
                hard_negative_mask = unmatched_query_empty_mask_loss(
                    torch,
                    mask_logits,
                    q_labels,
                    admission_logits,
                    top_k=unmatched_mask_negative_top_k,
                    primitive_weights=target_weights,
                )
                mask_loss = mask_loss + float(unmatched_mask_negative_loss_weight) * hard_negative_mask
                counters["unmatched_mask_negative_loss_sum"] += float(hard_negative_mask.item())
            geometry_aux_loss = query_logits.sum() * 0.0 if availability["rq"] else None
            if availability["rq"] and len(geometry_aux_outputs) > 1:
                record_aux = [
                    {
                        key: value[..., : y.numel()] if key == "mask_logits" else value
                        for key, value in layer.items() if key != "layer_index"
                    }
                    for layer in geometry_aux_outputs[:-1]
                ]
                geometry_aux_loss, geometry_diag = geometry_v2_auxiliary_loss(
                    torch, record_aux, target_labels, target_masks,
                    num_queries=num_queries, primitive_weights=target_weights, query_class_weights=query_class_weights,
                    thing_query_count=thing_query_count, typed_stuff_slots=bool(getattr(runtime_model, "typed_stuff_slots", False)),
                    selected_primitive_indices=selected_primitive_indices,
                )
                counters["geometry_aux_layers"] += len(geometry_diag)
            router_load_balance = (
                router_diagnostics["load_balance_cv_squared"]
                if router_diagnostics is not None
                else query_logits.sum() * 0.0
            )
            content_anchor = (
                family_seed_loss(torch, family_seed_logits, y, mask_valid, length_weights)
                if availability["rq"]
                else None
            )
            offset_vote = (
                token_offset_vote_loss(torch, token_offsets, x.squeeze(0), q_labels, q_masks, length_weights)
                if availability["rq"]
                else None
            )
            affinity = (
                token_affinity_component_loss(torch, token_affinity, q_labels, q_masks, length_weights)
                if availability["rq"]
                else None
            )
            loss, loss_diagnostic = compose_task_loss_diagnostic(
                {
                    "semantic": semantic,
                    "query": query,
                    "query_objectness": objectness,
                    "quality_calibration": quality_calibration,
                    "mask": mask_loss,
                    "ownership": ownership_loss,
                    "teacher": teacher_loss,
                    "geometry_aux": geometry_aux_loss,
                    "content_anchor": content_anchor,
                    "offset_vote": offset_vote,
                    "affinity": affinity,
                    "router_load_balance": router_load_balance,
                },
                diagnostic_weights,
                pcgrad=pcgrad_diagnostic,
            )
            page_id = str(record.get("original_record_id") or record.get("record_id"))
            window_index = parse_int(record.get("window_index"), parse_int(record.get("window_start"), 0))
            previous = identity_previous.get(page_id)
            if availability["identity"] and identity_embeddings is not None and identity_loss_weight > 0.0 and previous is not None and window_index - previous[3] == 1:
                identity_loss, identity_diag = adjacent_window_identity_loss(
                    torch.stack([previous[0], identity_embeddings.squeeze(0)]),
                    [previous[1], matched_ids],
                    torch.stack([previous[2], matched_valid]),
                    [previous[3], window_index],
                    [page_id, page_id],
                    temperature=identity_temperature,
                    negative_margin=identity_negative_margin,
                )
                loss = loss + float(identity_loss_weight) * identity_loss
                counters["identity_loss_sum"] += float(identity_loss.item())
                counters["identity_edges"] += 1
                counters["identity_positive_assignment_directions"] += int(identity_diag["positive_assignment_directions"])
                counters["identity_negative_pairs"] += int(identity_diag["negative_pairs"])
            if availability["identity"] and identity_embeddings is not None:
                identity_previous[page_id] = (
                    identity_embeddings.squeeze(0).detach(), matched_ids, matched_valid.detach(), window_index
                )
            if sq_rq_outputs is not None:
                base_logits = sq_rq_outputs["semantic_base_logits"].squeeze(0).float()
                base_loss = weighted_semantic_loss_schema_v2(
                    torch, base_logits, y, semantic_weights,
                    semantic_class_weights_value, semantic_label_smoothing,
                ) if availability["sq"] else None
                post_cross_logits = sq_rq_outputs["semantic_post_cross_logits"].squeeze(0).float()
                post_cross_loss = weighted_semantic_loss_schema_v2(
                    torch, post_cross_logits, y, semantic_weights,
                    semantic_class_weights_value, semantic_label_smoothing,
                ) if availability["sq"] else None
                post_private_logits = sq_rq_outputs["semantic_post_private_logits"].squeeze(0).float()
                post_private_loss = weighted_semantic_loss_schema_v2(
                    torch, post_private_logits, y, semantic_weights,
                    semantic_class_weights_value, semantic_label_smoothing,
                ) if availability["sq"] else None
                counters["sq_rq_base_semantic_loss_sum"] += float(base_loss.item()) if base_loss is not None else 0.0
                counters["sq_rq_post_cross_semantic_loss_sum"] += float(post_cross_loss.item()) if post_cross_loss is not None else 0.0
                counters["sq_rq_post_private_semantic_loss_sum"] += float(post_private_loss.item()) if post_private_loss is not None else 0.0
                counters["sq_rq_context_semantic_loss_sum"] += float(semantic.item()) if semantic is not None else 0.0
                attention = sq_rq_outputs["attention_weights"]
                consistency = sq_rq_outputs["semantic_consistency_mask"]
                admitted = sq_rq_outputs["admitted_rq_queries"]
                raw_admitted = sq_rq_outputs["raw_admitted_rq_queries"]
                membership_supported = sq_rq_outputs["membership_supported_rq_queries"]
                adaptive_gate = sq_rq_outputs["adaptive_context_gate"]
                counters["sq_rq_attention_nonzero"] += int((attention > 0).sum().item())
                counters["sq_rq_attention_total"] += int(attention.numel())
                counters["sq_rq_context_edges"] += int(consistency.sum().item())
                counters["sq_rq_context_possible_edges"] += int(consistency.numel())
                counters["sq_rq_admitted_queries"] += int(admitted.sum().item())
                counters["sq_rq_raw_admitted_queries"] += int(raw_admitted.sum().item())
                counters["sq_rq_membership_supported_queries"] += int(membership_supported.sum().item())
                counters["sq_rq_total_queries"] += int(admitted.numel())
                counters["sq_rq_adaptive_gate_sum"] += float(adaptive_gate.sum().item())
                counters["sq_rq_adaptive_gate_items"] += int(adaptive_gate.numel())
            pred = semantic_logits.argmax(dim=-1)
            valid = y != IGNORE_LABEL
            counters["records"] += 1
            counters["tokens"] += int(y.numel())
            counters["labeled_tokens"] += int(valid.sum().item())
            counters["semantic_correct"] += int(((pred == y) & valid).sum().item())
            counters["target_components"] += int(positives)
            counters["matched_components"] += int(matched)
            loss_sum += float(loss.item())
            counters["semantic_loss_sum"] += float(semantic.item()) if semantic is not None else 0.0
            counters["query_loss_sum"] += float(query.item()) if query is not None else 0.0
            counters["query_objectness_loss_sum"] += float(objectness.item()) if objectness is not None else 0.0
            counters["rq_sq_quality_calibration_loss_sum"] += float(quality_calibration.item()) if quality_calibration is not None else 0.0
            counters["mask_loss_sum"] += float(mask_loss.item()) if mask_loss is not None else 0.0
            counters["ownership_loss_sum"] += float(ownership_loss.item()) if ownership_loss is not None else 0.0
            counters["geometry_aux_loss_sum"] += float(geometry_aux_loss.item()) if geometry_aux_loss is not None else 0.0
            counters["content_anchor_loss_sum"] += float(content_anchor.item()) if content_anchor is not None else 0.0
            counters["router_load_balance_loss_sum"] += float(router_load_balance.item())
            if router_diagnostics is not None:
                mean_probability = router_diagnostics["mean_expert_probability"]
                assignment_fraction = router_diagnostics["assignment_fraction"]
                counters["router_diagnostic_records"] += 1
                counters["router_routing_entropy_sum"] += float(router_diagnostics["routing_entropy"].item())
                for expert_index, probability in enumerate(mean_probability.tolist()):
                    counters[f"router_expert_probability_{expert_index}"] += float(probability)
                for expert_index, fraction in enumerate(assignment_fraction.tolist()):
                    counters[f"router_expert_assignment_fraction_{expert_index}"] += float(fraction)
            counters["teacher_loss_sum"] += float(teacher_loss.item()) if teacher_loss is not None else 0.0
            counters["teacher_positive_components"] += int(teacher_pos_count)
    loss_breakdown = {
        "semantic": counters["semantic_loss_sum"] / max(counters["records"], 1),
        "query": counters["query_loss_sum"] / max(counters["records"], 1),
        "query_objectness": counters["query_objectness_loss_sum"] / max(counters["records"], 1),
        "quality_calibration": counters["rq_sq_quality_calibration_loss_sum"] / max(counters["records"], 1),
        "mask": counters["mask_loss_sum"] / max(counters["records"], 1),
        "unmatched_mask_negative": counters["unmatched_mask_negative_loss_sum"] / max(counters["records"], 1),
        "router_load_balance": counters["router_load_balance_loss_sum"] / max(counters["records"], 1),
        "content_anchor": counters["content_anchor_loss_sum"] / max(counters["records"], 1),
        "teacher": counters["teacher_loss_sum"] / max(counters["records"], 1),
        "teacher_loss_weight": teacher_loss_weight,
        "teacher_mask_loss_weight": teacher_mask_loss_weight,
        "teacher_query_loss_weight": teacher_query_loss_weight,
        "query_objectness_weight": objectness_loss_weight,
        "semantic_weight": semantic_loss_weight,
        "query_weight": query_loss_weight,
        "mask_weight": mask_loss_weight,
        "quality_calibration_weight": rq_sq_quality_calibration_loss_weight,
        "ownership_weight": ownership_loss_weight,
        "geometry_aux_weight": geometry_aux_loss_weight,
        "content_anchor_weight": content_anchor_loss_weight,
        "router_load_balance_weight": router_load_balance_loss_weight,
        "identity_weight": identity_loss_weight,
        "aggregation": loss_diagnostic["aggregation"] if counters["records"] else (
            "production_weighted_sum_diagnostic_pcgrad_backward" if pcgrad_diagnostic else "production_weighted_sum"
        ),
    }
    if counters["ownership_available_records"]:
        loss_breakdown["ownership"] = counters["ownership_loss_sum"] / max(counters["records"], 1)
    if counters["geometry_aux_layers"]:
        loss_breakdown["geometry_aux"] = counters["geometry_aux_loss_sum"] / max(counters["records"], 1)
    if counters["identity_edges"]:
        loss_breakdown["identity"] = counters["identity_loss_sum"] / max(counters["identity_edges"], 1)

    def empirical_quantile(values: list[float], probability: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        position = (len(ordered) - 1) * float(probability)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return float(ordered[lower])
        fraction = position - lower
        return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)

    hard_negative_count = min(int(rq_sq_quality_ranking_top_k), len(quality_negative_predictions))
    unmatched_topk_mean = (
        sum(sorted(quality_negative_predictions, reverse=True)[:hard_negative_count]) / hard_negative_count
        if hard_negative_count
        else 0.0
    )
    deployment_hard_negative_count = min(
        int(rq_sq_quality_ranking_top_k),
        len(quality_negative_deployment_predictions),
    )
    unmatched_deployment_topk_mean = (
        sum(
            sorted(quality_negative_deployment_predictions, reverse=True)[
                :deployment_hard_negative_count
            ]
        )
        / deployment_hard_negative_count
        if deployment_hard_negative_count
        else 0.0
    )
    return {
        "records": counters["records"],
        "tokens": counters["tokens"],
        "labeled_tokens": counters["labeled_tokens"],
        "semantic_token_accuracy": counters["semantic_correct"] / max(counters["labeled_tokens"], 1),
        "target_components": counters["target_components"],
        "matched_components": counters["matched_components"],
        "target_selection": target_diagnostic_payload(counters),
        "candidate_proxy": {
            "records_with_candidates": counters["candidate_records_with_candidates"],
            "candidate_valid_total": counters["candidate_valid_total"],
            "candidate_mask_token_total": counters["candidate_mask_token_total"],
            "candidate_record_coverage": counters["candidate_records_with_candidates"] / max(counters["records"], 1),
        },
        "component_proxy": component_proxy_payload(counters),
        "content_anchor_proxy": family_seed_proxy_payload(counters),
        "loss": loss_sum / max(counters["records"], 1),
        "loss_breakdown": loss_breakdown,
        "identity_proxy": {
            "adjacent_page_window_edges": counters["identity_edges"],
            "positive_assignment_directions": counters["identity_positive_assignment_directions"],
            "negative_pairs": counters["identity_negative_pairs"],
        },
        "quality_proxy": {
            "score_protocol": "foreground_probability_times_quality_probability_times_mask_objectness_v2",
            "items": counters["quality_items"],
            "predicted_mean": counters["quality_prediction_sum"] / max(counters["quality_items"], 1),
            "positive_deployment_score_mean": counters["quality_deployment_prediction_sum"] / max(counters["quality_items"], 1),
            "decoded_hard_iou_target_mean": counters["quality_target_sum"] / max(counters["quality_items"], 1),
            "mean_absolute_calibration_error": counters["quality_absolute_error_sum"] / max(counters["quality_items"], 1),
            "unmatched_items": counters["quality_negative_items"],
            "unmatched_predicted_mean": counters["quality_negative_prediction_sum"] / max(counters["quality_negative_items"], 1),
            "unmatched_predicted_p95": empirical_quantile(quality_negative_predictions, 0.95),
            "unmatched_predicted_p99": empirical_quantile(quality_negative_predictions, 0.99),
            "unmatched_predicted_max": max(quality_negative_predictions, default=0.0),
            "unmatched_topk_mean": unmatched_topk_mean,
            "positive_predicted_p10": empirical_quantile(quality_positive_predictions, 0.10),
            "positive_deployment_score_p10": empirical_quantile(
                quality_positive_deployment_predictions, 0.10
            ),
            "unmatched_deployment_score_mean": counters["quality_negative_deployment_prediction_sum"] / max(counters["quality_negative_items"], 1),
            "unmatched_deployment_score_p95": empirical_quantile(
                quality_negative_deployment_predictions, 0.95
            ),
            "unmatched_deployment_score_p99": empirical_quantile(
                quality_negative_deployment_predictions, 0.99
            ),
            "unmatched_deployment_score_max": max(
                quality_negative_deployment_predictions, default=0.0
            ),
            "unmatched_deployment_score_topk_mean": unmatched_deployment_topk_mean,
            "ranking_pairs": counters["quality_ranking_pairs"],
            "ranking_violations": counters["quality_ranking_violations"],
            "ranking_violation_rate": counters["quality_ranking_violations"] / max(counters["quality_ranking_pairs"], 1),
        },
        "sq_rq_proxy": {
            "semantic_base_loss": counters["sq_rq_base_semantic_loss_sum"] / max(counters["records"], 1),
            "semantic_post_cross_loss": counters["sq_rq_post_cross_semantic_loss_sum"] / max(counters["records"], 1),
            "semantic_post_private_loss": counters["sq_rq_post_private_semantic_loss_sum"] / max(counters["records"], 1),
            "semantic_context_loss": counters["sq_rq_context_semantic_loss_sum"] / max(counters["records"], 1),
            "semantic_cross_minus_base_loss": (
                counters["sq_rq_post_cross_semantic_loss_sum"] - counters["sq_rq_base_semantic_loss_sum"]
            ) / max(counters["records"], 1),
            "semantic_private_minus_cross_loss": (
                counters["sq_rq_post_private_semantic_loss_sum"] - counters["sq_rq_post_cross_semantic_loss_sum"]
            ) / max(counters["records"], 1),
            "semantic_bridge_minus_private_loss": (
                counters["sq_rq_context_semantic_loss_sum"] - counters["sq_rq_post_private_semantic_loss_sum"]
            ) / max(counters["records"], 1),
            "semantic_context_minus_base_loss": (
                counters["sq_rq_context_semantic_loss_sum"] - counters["sq_rq_base_semantic_loss_sum"]
            ) / max(counters["records"], 1),
            "attention_nonzero_coverage": counters["sq_rq_attention_nonzero"] / max(counters["sq_rq_attention_total"], 1),
            "context_edge_coverage": counters["sq_rq_context_edges"] / max(counters["sq_rq_context_possible_edges"], 1),
            "admitted_query_coverage": counters["sq_rq_admitted_queries"] / max(counters["sq_rq_total_queries"], 1),
            "raw_admission_coverage": counters["sq_rq_raw_admitted_queries"] / max(counters["sq_rq_total_queries"], 1),
            "membership_supported_query_coverage": (
                counters["sq_rq_membership_supported_queries"] / max(counters["sq_rq_total_queries"], 1)
            ),
            "adaptive_context_gate_mean": counters["sq_rq_adaptive_gate_sum"] / max(counters["sq_rq_adaptive_gate_items"], 1),
            "scale0_gradient_isolation": "checkpoint_abi_static_diagnostic",
        },
        "router_proxy": {
            "enabled": counters["router_diagnostic_records"] > 0,
            "load_balance_cv_squared": counters["router_load_balance_loss_sum"] / max(counters["router_diagnostic_records"], 1),
            "mean_expert_probability": [
                counters[f"router_expert_probability_{index}"] / max(counters["router_diagnostic_records"], 1)
                for index in range(len((getattr(getattr(model, "_orig_mod", model), "sparse_experts", []))))
            ],
            "assignment_fraction": [
                counters[f"router_expert_assignment_fraction_{index}"] / max(counters["router_diagnostic_records"], 1)
                for index in range(len((getattr(getattr(model, "_orig_mod", model), "sparse_experts", []))))
            ],
            "routing_entropy": counters["router_routing_entropy_sum"] / max(counters["router_diagnostic_records"], 1),
        },
        "teacher_proxy": {
            "records_with_teacher": counters["teacher_records_with_teacher"],
            "teacher_components_total": counters["teacher_components_total"],
            "teacher_components_kept": counters["teacher_components_kept"],
            "teacher_components_dropped": counters["teacher_components_dropped"],
            "teacher_positive_components": counters["teacher_positive_components"],
            "teacher_matched_positive_queries": counters["teacher_matched_positive_queries"],
            "teacher_supervision_overlap_queries": counters["teacher_supervision_overlap_queries"],
            "teacher_label_conflict_queries": counters["teacher_label_conflict_queries"],
            "teacher_mask_conflict_queries": counters["teacher_mask_conflict_queries"],
            "teacher_identity_aligned": counters["teacher_identity_aligned"],
            "teacher_identity_unaligned": counters["teacher_identity_unaligned"],
            "gt_positive_teacher_negative_conflicts": counters["gt_positive_teacher_negative_conflicts"],
            "teacher_hard_negative_objectness_queries": counters["teacher_hard_negative_objectness_queries"],
        },
    }


def precision_phase_admission_ready(
    history: list[dict[str, Any]],
    required_epochs: int,
    min_recall: float,
    max_negative_margin_rate: float,
    min_sq_proxy: float,
    min_calibrated_rq: float = 0.0,
    min_calibrated_proposal_coverage: float = 0.0,
) -> bool:
    if required_epochs <= 0:
        return True
    if len(history) < required_epochs:
        return False
    for row in history[-required_epochs:]:
        component = ((row.get("val") or {}).get("component_proxy") or {})
        if int(component.get("query_predicted_object_total", 0) or 0) <= 0:
            return False
        if int(component.get("query_target_positive_total", 0) or 0) <= 0:
            return False
        if int(component.get("target_positive_tokens", 0) or 0) <= 0:
            return False
        metrics = {
            "object_recall": component.get("query_positive_object_recall"),
            "negative_margin_rate": component.get("query_negative_object_margin_positive_rate"),
            "mask_precision": component.get("mask_token_precision"),
            "mask_recall": component.get("mask_token_recall"),
        }
        if min_calibrated_rq > 0.0:
            metrics["calibrated_rq"] = component.get("calibrated_instance_proxy_rq")
        if min_calibrated_proposal_coverage > 0.0:
            metrics["calibrated_proposal_coverage"] = component.get(
                "calibrated_query_proposal_coverage"
            )
        if any(value is None for value in metrics.values()):
            return False
        try:
            metrics = {key: float(value) for key, value in metrics.items()}
        except (TypeError, ValueError):
            return False
        if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in metrics.values()):
            return False
        if metrics["object_recall"] < min_recall:
            return False
        if metrics["negative_margin_rate"] > max_negative_margin_rate:
            return False
        if metrics.get("calibrated_rq", 1.0) < min_calibrated_rq:
            return False
        if (
            metrics.get("calibrated_proposal_coverage", 1.0)
            < min_calibrated_proposal_coverage
        ):
            return False
        precision = metrics["mask_precision"]
        recall = metrics["mask_recall"]
        denominator = precision + recall - precision * recall
        sq_proxy = precision * recall / denominator if denominator > 0.0 else 0.0
        if sq_proxy < min_sq_proxy:
            return False
    return True


def precision_phase_progress(
    *,
    epoch: int,
    start_epoch: int,
    transition_epochs: int,
    allowed: bool,
    previous_progress: float,
) -> float:
    if start_epoch < 0 or epoch < start_epoch:
        return 0.0
    step = 1.0 / max(int(transition_epochs), 1)
    previous = min(max(float(previous_progress), 0.0), 1.0)
    delta = step if allowed else -step
    return min(max(previous + delta, 0.0), 1.0)


def blend_schedule(
    base: dict[str, float],
    target: dict[str, float],
    progress: float,
) -> dict[str, float]:
    alpha = min(max(float(progress), 0.0), 1.0)
    return {
        key: float(base[key]) + alpha * (float(target[key]) - float(base[key]))
        for key in base
    }


def objectness_schedule(
    args: argparse.Namespace,
    epoch: int,
    *,
    precision_phase_allowed: bool = True,
    previous_precision_phase_progress: float = 0.0,
) -> dict[str, Any]:
    base = {
        "query_objectness_loss_weight": args.query_objectness_loss_weight,
        "query_objectness_positive_weight": args.query_objectness_positive_weight,
        "query_objectness_negative_weight": args.query_objectness_negative_weight,
        "query_objectness_positive_margin_floor_loss_weight": args.query_objectness_positive_margin_floor_loss_weight,
        "query_objectness_positive_margin_floor": args.objectness_positive_margin_floor,
        "query_objectness_negative_margin_ceiling_loss_weight": args.query_objectness_negative_margin_ceiling_loss_weight,
        "query_objectness_negative_margin_ceiling": args.objectness_negative_margin_ceiling,
    }
    if args.objectness_warmup_epochs > 0 and epoch <= args.objectness_warmup_epochs:
        warmup = {
            "query_objectness_loss_weight": args.query_objectness_loss_weight * args.objectness_warmup_loss_multiplier,
            "query_objectness_positive_weight": args.query_objectness_positive_weight * args.objectness_warmup_positive_multiplier,
            "query_objectness_negative_weight": args.query_objectness_negative_weight * args.objectness_warmup_negative_multiplier,
            "query_objectness_positive_margin_floor_loss_weight": args.objectness_warmup_positive_margin_floor_loss_weight,
            "query_objectness_positive_margin_floor": args.objectness_positive_margin_floor,
            "query_objectness_negative_margin_ceiling_loss_weight": args.objectness_warmup_negative_margin_ceiling_loss_weight,
            "query_objectness_negative_margin_ceiling": args.objectness_negative_margin_ceiling,
        }
        return {**warmup, "precision_phase_progress": 0.0, "precision_phase_allowed": False}

    target = {
        "query_objectness_loss_weight": args.objectness_precision_phase_loss_weight,
        "query_objectness_positive_weight": args.objectness_precision_phase_positive_weight,
        "query_objectness_negative_weight": args.objectness_precision_phase_negative_weight,
        "query_objectness_positive_margin_floor_loss_weight": args.objectness_precision_phase_positive_margin_floor_loss_weight,
        "query_objectness_positive_margin_floor": args.objectness_positive_margin_floor,
        "query_objectness_negative_margin_ceiling_loss_weight": args.objectness_precision_phase_negative_margin_ceiling_loss_weight,
        "query_objectness_negative_margin_ceiling": args.objectness_negative_margin_ceiling,
    }
    configured_start_epoch = int(args.objectness_precision_phase_start_epoch)
    warmup_stop_epoch = int(args.objectness_warmup_epochs) + 1 if int(args.objectness_warmup_epochs) > 0 else 0
    effective_start_epoch = (
        -1 if configured_start_epoch < 0
        else max(configured_start_epoch, warmup_stop_epoch)
    )
    progress = precision_phase_progress(
        epoch=epoch,
        start_epoch=effective_start_epoch,
        transition_epochs=int(getattr(args, "precision_phase_transition_epochs", 8)),
        allowed=precision_phase_allowed,
        previous_progress=previous_precision_phase_progress,
    )
    return {
        **blend_schedule(base, target, progress),
        "precision_phase_progress": progress,
        "precision_phase_allowed": bool(precision_phase_allowed),
    }


def mask_loss_schedule(
    args: argparse.Namespace,
    epoch: int,
    *,
    precision_phase_allowed: bool = True,
    previous_precision_phase_progress: float = 0.0,
) -> dict[str, Any]:
    base = {
        "mask_positive_weight": float(getattr(args, "mask_positive_weight", 1.0)),
        "mask_negative_weight": args.mask_negative_weight,
        "mask_area_ratio_loss_weight": args.mask_area_ratio_loss_weight,
        "mask_area_overcoverage_weight": args.mask_area_overcoverage_weight,
        "mask_tversky_loss_weight": args.mask_tversky_loss_weight,
        "mask_positive_prob_floor_loss_weight": args.mask_positive_prob_floor_loss_weight,
    }
    target = {
        "mask_positive_weight": float(getattr(args, "mask_precision_phase_positive_weight", 1.0)),
        "mask_negative_weight": args.mask_precision_phase_negative_weight,
        "mask_area_ratio_loss_weight": args.mask_precision_phase_area_ratio_loss_weight,
        "mask_area_overcoverage_weight": args.mask_precision_phase_area_overcoverage_weight,
        "mask_tversky_loss_weight": args.mask_precision_phase_tversky_loss_weight,
        "mask_positive_prob_floor_loss_weight": args.mask_precision_phase_positive_prob_floor_loss_weight,
    }
    configured_start_epoch = int(args.mask_precision_phase_start_epoch)
    warmup_stop_epoch = int(getattr(args, "objectness_warmup_epochs", 0)) + 1 if int(getattr(args, "objectness_warmup_epochs", 0)) > 0 else 0
    effective_start_epoch = (
        -1 if configured_start_epoch < 0
        else max(configured_start_epoch, warmup_stop_epoch)
    )
    progress = precision_phase_progress(
        epoch=epoch,
        start_epoch=effective_start_epoch,
        transition_epochs=int(getattr(args, "precision_phase_transition_epochs", 8)),
        allowed=precision_phase_allowed,
        previous_progress=previous_precision_phase_progress,
    )
    return {
        **blend_schedule(base, target, progress),
        "precision_phase_progress": progress,
        "precision_phase_allowed": bool(precision_phase_allowed),
    }


def recall_gated_selection_score(val: dict[str, Any], min_object_recall: float, min_mask_recall: float) -> tuple[float, dict[str, Any]]:
    component = val.get("component_proxy") or {}
    object_recall = float(component.get("query_positive_object_recall", 0.0))
    object_total = int(component.get("query_predicted_object_total", 0))
    mask_recall = float(component.get("mask_token_recall", 0.0))
    proxy = float(component.get("component_proxy_score", 0.0))
    semantic = float(val.get("semantic_token_accuracy", 0.0))
    passed = object_total > 0 and object_recall >= min_object_recall and mask_recall >= min_mask_recall
    if not passed:
        return -float("inf"), {
            "passed": False,
            "reason": "zero_or_insufficient_validation_object_recall_or_mask_recall",
            "query_predicted_object_total": object_total,
            "query_positive_object_recall": object_recall,
            "mask_token_recall": mask_recall,
            "min_query_positive_object_recall": min_object_recall,
            "min_mask_token_recall": min_mask_recall,
        }
    return 0.45 * object_recall + 0.35 * mask_recall + 0.15 * proxy + 0.05 * semantic, {
        "passed": True,
        "query_predicted_object_total": object_total,
        "query_positive_object_recall": object_recall,
        "mask_token_recall": mask_recall,
        "proxy": proxy,
        "semantic_token_accuracy": semantic,
    }


def pq_aware_selection_score(
    val: dict[str, Any],
    min_object_recall: float,
    min_mask_recall: float,
    min_mask_precision: float,
    max_positive_rate_ratio: float,
) -> tuple[float, dict[str, Any]]:
    component = val.get("component_proxy") or {}
    object_recall = float(component.get("query_positive_object_recall", 0.0))
    object_total = int(component.get("query_predicted_object_total", 0))
    mask_recall = float(component.get("mask_token_recall", 0.0))
    mask_precision = float(component.get("mask_token_precision", 0.0))
    mask_f1 = float(component.get("mask_token_f1", 0.0))
    positive_rate_ratio = float(component.get("positive_rate_ratio", float("inf")))
    proxy = float(component.get("component_proxy_score", 0.0))
    semantic = float(val.get("semantic_token_accuracy", 0.0))
    overcoverage_ok = positive_rate_ratio <= max_positive_rate_ratio if max_positive_rate_ratio > 0 else True
    passed = (
        object_total > 0
        and object_recall >= min_object_recall
        and mask_recall >= min_mask_recall
        and mask_precision >= min_mask_precision
        and overcoverage_ok
    )
    gate = {
        "passed": passed,
        "query_predicted_object_total": object_total,
        "query_positive_object_recall": object_recall,
        "mask_token_precision": mask_precision,
        "mask_token_recall": mask_recall,
        "mask_token_f1": mask_f1,
        "positive_rate_ratio": positive_rate_ratio,
        "max_positive_rate_ratio": max_positive_rate_ratio,
        "min_query_positive_object_recall": min_object_recall,
        "min_mask_token_recall": min_mask_recall,
        "min_mask_token_precision": min_mask_precision,
        "proxy": proxy,
        "semantic_token_accuracy": semantic,
    }
    if not passed:
        gate["reason"] = "insufficient_object_recall_mask_precision_recall_or_overcoverage"
        return -float("inf"), gate
    score = 0.35 * object_recall + 0.30 * mask_f1 + 0.20 * mask_precision + 0.10 * proxy + 0.05 * semantic
    return score, gate


def admission_aware_selection_score(
    val: dict[str, Any],
    min_object_recall: float,
    min_mask_recall: float,
    min_mask_precision: float,
    max_positive_rate_ratio: float,
    min_positive_margin_rate: float,
    min_positive_margin_mean: float,
    max_negative_margin_rate: float,
) -> tuple[float, dict[str, Any]]:
    selection_score, gate = pq_aware_selection_score(
        val,
        min_object_recall,
        min_mask_recall,
        min_mask_precision,
        max_positive_rate_ratio,
    )
    component = val.get("component_proxy") or {}
    positive_margin_rate = float(component.get("query_positive_object_margin_positive_rate", 0.0))
    positive_margin_mean = float(component.get("query_positive_object_margin_mean", 0.0))
    negative_margin_rate = float(component.get("query_negative_object_margin_positive_rate", 1.0))
    admission_passed = (
        positive_margin_rate >= min_positive_margin_rate
        and positive_margin_mean >= min_positive_margin_mean
        and (max_negative_margin_rate <= 0.0 or negative_margin_rate <= max_negative_margin_rate)
    )
    gate = {
        **gate,
        "query_positive_object_margin_positive_rate": positive_margin_rate,
        "query_positive_object_margin_mean": positive_margin_mean,
        "query_negative_object_margin_positive_rate": negative_margin_rate,
        "min_query_positive_object_margin_positive_rate": min_positive_margin_rate,
        "min_query_positive_object_margin_mean": min_positive_margin_mean,
        "max_query_negative_object_margin_positive_rate": max_negative_margin_rate,
        "admission_aware": True,
    }
    if not gate.get("passed") or not admission_passed:
        gate["passed"] = False
        gate["reason"] = "insufficient_pq_proxy_or_final_admission_margin"
        return -float("inf"), gate
    score = (
        0.25 * float(gate.get("query_positive_object_recall", 0.0))
        + 0.20 * float(gate.get("mask_token_f1", 0.0))
        + 0.20 * float(gate.get("mask_token_precision", 0.0))
        + 0.20 * positive_margin_rate
        + 0.10 * max(0.0, min((positive_margin_mean + 1.0) / 2.0, 1.0))
        + 0.05 * max(0.0, 1.0 - negative_margin_rate)
    )
    return max(score, selection_score), gate


def joint_rq_sq_selection_score(
    val: dict[str, Any],
    min_rq_proxy: float,
    min_sq_proxy: float,
    max_negative_margin_rate: float,
    min_mask_precision: float = 0.0,
    min_mask_recall: float = 0.0,
    min_instance_tp: int = 0,
    min_proposal_coverage: float = 0.0,
) -> tuple[float, dict[str, Any]]:
    component = val.get("component_proxy") or {}
    proxy_protocol = str(component.get("calibrated_instance_proxy_protocol", ""))
    proxy_conservation = component.get("proxy_conservation") or {}
    proxy_conservation_ok = bool(proxy_conservation.get("ok", False))
    object_recall = float(component.get("query_positive_object_recall", 0.0))
    negative_margin_rate = float(component.get("query_negative_object_margin_positive_rate", 1.0))
    raw_object_total = int(component.get("query_predicted_object_total", 0))
    admitted_total = int(component.get("calibrated_query_admitted_total", 0))
    object_total = int(component.get("calibrated_query_proposal_total", 0))
    instance_precision = float(component.get("calibrated_instance_proxy_precision", 0.0))
    legacy_mask_precision = float(component.get("mask_token_precision", 0.0))
    legacy_mask_recall = float(component.get("mask_token_recall", 0.0))
    instance_tp = int(component.get("calibrated_instance_proxy_tp", 0))
    proposal_coverage = float(component.get("calibrated_query_proposal_coverage", 0.0))
    rq_proxy = float(component.get("calibrated_instance_proxy_rq", 0.0))
    sq_proxy = float(component.get("calibrated_instance_proxy_sq", 0.0))
    pq_proxy = float(component.get("calibrated_instance_proxy_pq", 0.0))
    accepted_proxy_protocols = {
        "respect_no_object_class_times_quality_threshold_then_binary_mask_iou_0p5_tp_fp_fn_v1",
        "respect_no_object_class_times_quality_threshold_then_length_weighted_binary_mask_iou_0p5_tp_fp_fn_v2",
        "respect_no_object_class_times_quality_times_mask_objectness_threshold_then_length_weighted_binary_mask_iou_0p5_tp_fp_fn_v3",
    }
    passed = (
        proxy_protocol in accepted_proxy_protocols
        and proxy_conservation_ok
        and object_total > 0
        and rq_proxy >= min_rq_proxy
        and sq_proxy >= min_sq_proxy
        and legacy_mask_precision >= min_mask_precision
        and legacy_mask_recall >= min_mask_recall
        and instance_tp >= int(min_instance_tp)
        and proposal_coverage >= min_proposal_coverage
        and (max_negative_margin_rate <= 0.0 or negative_margin_rate <= max_negative_margin_rate)
    )
    gate = {
        "passed": passed,
        "joint_rq_sq": True,
        "selection_protocol": "quality_calibrated_deployment_admission_v1",
        "component_proxy_protocol": proxy_protocol,
        "proxy_conservation_ok": proxy_conservation_ok,
        "proxy_conservation": proxy_conservation,
        "query_predicted_object_total": raw_object_total,
        "calibrated_query_admitted_total": admitted_total,
        "calibrated_query_proposal_total": object_total,
        "calibrated_query_admitted_coverage": float(
            component.get("calibrated_query_admitted_coverage", 0.0)
        ),
        "deployment_min_query_score": float(
            component.get("calibrated_instance_proxy_min_query_score", DEFAULT_MIN_QUERY_SCORE)
        ),
        "deployment_mask_threshold": float(
            component.get("calibrated_instance_proxy_mask_threshold", PANOPTIC_QUALITY_MASK_THRESHOLD)
        ),
        "query_positive_object_recall": object_recall,
        "query_negative_object_margin_positive_rate": negative_margin_rate,
        "admission_precision_proxy": instance_precision,
        "mask_token_precision": legacy_mask_precision,
        "mask_token_recall": legacy_mask_recall,
        "min_mask_token_precision": min_mask_precision,
        "min_mask_token_recall": min_mask_recall,
        "calibrated_query_proposal_coverage": proposal_coverage,
        "min_calibrated_query_proposal_coverage": min_proposal_coverage,
        "instance_proxy_tp": instance_tp,
        "min_instance_proxy_tp": int(min_instance_tp),
        "instance_proxy_fp": int(component.get("calibrated_instance_proxy_fp", 0)),
        "instance_proxy_fn": int(component.get("calibrated_instance_proxy_fn", 0)),
        "instance_proxy_iou_threshold": float(component.get("instance_proxy_iou_threshold", 0.5)),
        "rq_proxy": rq_proxy,
        "sq_proxy": sq_proxy,
        "pq_proxy": pq_proxy,
        "min_rq_proxy": min_rq_proxy,
        "min_sq_proxy": min_sq_proxy,
        "max_query_negative_object_margin_positive_rate": max_negative_margin_rate,
    }
    if not passed:
        gate["reason"] = "invalid_proxy_conservation_or_insufficient_joint_rq_sq"
        return -float("inf"), gate
    return pq_proxy, gate


def quality_checkpoint_selection_gate(
    val: dict[str, Any],
    *,
    max_unmatched_quality: float,
    max_ranking_violation_rate: float,
) -> dict[str, Any]:
    if not 0.0 <= float(max_unmatched_quality) <= 1.0:
        raise ValueError("maximum unmatched quality must be in [0, 1]")
    if not 0.0 <= float(max_ranking_violation_rate) <= 1.0:
        raise ValueError("maximum quality ranking violation rate must be in [0, 1]")
    enabled = float(max_unmatched_quality) < 1.0 or float(max_ranking_violation_rate) < 1.0
    if not enabled:
        return {"enabled": False, "passed": True, "reason": "disabled"}
    quality = val.get("quality_proxy") or {}
    items = int(quality.get("items", 0))
    unmatched_items = int(quality.get("unmatched_items", 0))
    ranking_pairs = int(quality.get("ranking_pairs", 0))
    unmatched_deployment_score_supported = "unmatched_deployment_score_mean" in quality
    unmatched_deployment_score = float(
        quality.get("unmatched_deployment_score_mean", float("inf"))
    )
    ranking_violation_rate = float(quality.get("ranking_violation_rate", float("inf")))
    blockers = []
    if items <= 0 or unmatched_items <= 0:
        blockers.append("quality_support_missing")
    if ranking_pairs <= 0:
        blockers.append("quality_ranking_support_missing")
    if not unmatched_deployment_score_supported or not math.isfinite(unmatched_deployment_score):
        blockers.append("unmatched_deployment_score_support_missing")
    elif unmatched_deployment_score > float(max_unmatched_quality):
        blockers.append("unmatched_deployment_score_above_maximum")
    if ranking_violation_rate > float(max_ranking_violation_rate):
        blockers.append("quality_ranking_violation_above_maximum")
    return {
        "enabled": True,
        "passed": not blockers,
        "reason": "ok" if not blockers else ";".join(blockers),
        "blockers": blockers,
        "items": items,
        "unmatched_items": unmatched_items,
        "ranking_pairs": ranking_pairs,
        "unmatched_predicted_mean": float(
            quality.get("unmatched_predicted_mean", float("inf"))
        ),
        "unmatched_deployment_score_mean": unmatched_deployment_score,
        "max_unmatched_deployment_score": float(max_unmatched_quality),
        "ranking_violation_rate": ranking_violation_rate,
        "max_ranking_violation_rate": float(max_ranking_violation_rate),
    }


def sq_rq_coverage_selection_gate(
    val: dict[str, Any],
    min_admitted_query_coverage: float,
    min_context_edge_coverage: float,
) -> dict[str, Any]:
    if not 0.0 <= float(min_admitted_query_coverage) <= 1.0:
        raise ValueError("minimum admitted-query coverage must be in [0, 1]")
    if not 0.0 <= float(min_context_edge_coverage) <= 1.0:
        raise ValueError("minimum context-edge coverage must be in [0, 1]")
    proxy = val.get("sq_rq_proxy") or {}
    admitted = float(proxy.get("admitted_query_coverage", 0.0))
    context_edges = float(proxy.get("context_edge_coverage", 0.0))
    blockers = []
    if admitted < min_admitted_query_coverage:
        blockers.append("sq_rq_admitted_query_coverage_below_minimum")
    if context_edges < min_context_edge_coverage:
        blockers.append("sq_rq_context_edge_coverage_below_minimum")
    return {
        "passed": not blockers,
        "reason": "ok" if not blockers else ";".join(blockers),
        "admitted_query_coverage": admitted,
        "context_edge_coverage": context_edges,
        "min_admitted_query_coverage": float(min_admitted_query_coverage),
        "min_context_edge_coverage": float(min_context_edge_coverage),
        "blockers": blockers,
    }


def sq_rq_training_threshold_schedule(args: argparse.Namespace, epoch: int) -> dict[str, float | int]:
    warmup_epochs = int(getattr(args, "sq_rq_coverage_warmup_epochs", 0))
    enable_epoch = max(int(getattr(args, "sq_rq_enable_after_epoch", 1)), 1)
    active_epoch = int(epoch) - enable_epoch + 1
    final_confidence = float(args.sq_rq_query_confidence_threshold)
    final_membership = float(args.sq_rq_token_membership_threshold)
    warmup_confidence = float(getattr(args, "sq_rq_warmup_query_confidence_threshold", 0.2))
    warmup_membership = float(getattr(args, "sq_rq_warmup_token_membership_threshold", 0.2))
    maximum_membership_temperature = float(
        getattr(args, "sq_rq_training_membership_temperature", 0.0)
    )
    if active_epoch <= 0:
        return {
            "epoch": int(epoch), "enable_epoch": enable_epoch, "active_epoch": active_epoch,
            "warmup_epochs": warmup_epochs,
            "query_confidence_threshold": warmup_confidence,
            "token_membership_threshold": warmup_membership,
            "training_membership_temperature": maximum_membership_temperature,
            "phase": "inactive_before_enable",
        }
    if warmup_epochs <= 0 or active_epoch > warmup_epochs:
        return {
            "epoch": int(epoch), "enable_epoch": enable_epoch, "active_epoch": active_epoch,
            "warmup_epochs": warmup_epochs,
            "query_confidence_threshold": final_confidence,
            "token_membership_threshold": final_membership,
            "training_membership_temperature": 0.0,
            "phase": "frozen_hard_thresholds",
        }
    progress = float(active_epoch - 1) / float(max(warmup_epochs, 1))
    return {
        "epoch": int(epoch), "enable_epoch": enable_epoch, "active_epoch": active_epoch,
        "warmup_epochs": warmup_epochs,
        "query_confidence_threshold": warmup_confidence + progress * (final_confidence - warmup_confidence),
        "token_membership_threshold": warmup_membership + progress * (final_membership - warmup_membership),
        "training_membership_temperature": maximum_membership_temperature * (1.0 - progress),
        "phase": "scheduled_soft_coverage",
    }


def sq_rq_deployment_thresholds(args: argparse.Namespace, epoch: int) -> dict[str, float | int]:
    return {
        "epoch": int(epoch),
        "query_confidence_threshold": float(args.sq_rq_query_confidence_threshold),
        "token_membership_threshold": float(args.sq_rq_token_membership_threshold),
        "training_membership_temperature": 0.0,
        "phase": "deployment_hard_thresholds",
    }


def sq_rq_checkpoint_promotion_ready(
    runtime_enabled: bool,
    training_thresholds: dict[str, float | int],
    *,
    auto_fused: bool = False,
) -> bool:
    if auto_fused:
        return not runtime_enabled
    return runtime_enabled and training_thresholds.get("phase") == "frozen_hard_thresholds"


def set_sq_rq_runtime_thresholds(model: Any, thresholds: dict[str, float | int]) -> None:
    runtime_model = getattr(model, "_orig_mod", model)
    module = getattr(runtime_model, "sq_rq_cross_attention", None)
    if module is None:
        return
    module.query_confidence_threshold = float(thresholds["query_confidence_threshold"])
    module.token_membership_threshold = float(thresholds["token_membership_threshold"])
    if "training_membership_temperature" in thresholds:
        module.training_membership_temperature = float(
            thresholds["training_membership_temperature"]
        )


def router_usage_selection_gate(
    val: dict[str, Any],
    max_dominant_probability: float,
    min_assignment_fraction: float,
) -> dict[str, Any]:
    if not 0.0 < float(max_dominant_probability) <= 1.0:
        raise ValueError("maximum dominant expert probability must be in (0, 1]")
    if not 0.0 <= float(min_assignment_fraction) <= 1.0:
        raise ValueError("minimum expert assignment fraction must be in [0, 1]")
    router = val.get("router_proxy") or {}
    probabilities = [float(value) for value in router.get("mean_expert_probability") or []]
    assignments = [float(value) for value in router.get("assignment_fraction") or []]
    blockers = []
    if not probabilities:
        blockers.append("router_diagnostics_missing")
    elif max(probabilities) > max_dominant_probability:
        blockers.append("router_dominant_expert_probability_above_maximum")
    if assignments and min(assignments) < min_assignment_fraction:
        blockers.append("router_expert_assignment_fraction_below_minimum")
    return {
        "passed": not blockers,
        "reason": "ok" if not blockers else ";".join(blockers),
        "mean_expert_probability": probabilities,
        "assignment_fraction": assignments,
        "max_dominant_expert_probability": float(max_dominant_probability),
        "min_expert_assignment_fraction": float(min_assignment_fraction),
        "blockers": blockers,
    }


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_provenance(args: argparse.Namespace) -> dict[str, Any]:
    inputs = {}
    for split, path in (("train", args.train), ("val", args.val)):
        stat = path.stat()
        inputs[split] = {
            "path": rel(path),
            "bytes": int(stat.st_size),
            "sha256": sha256_path(path),
        }
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=False, capture_output=True, text=True,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=False, capture_output=True, text=True,
    )
    return {
        "inputs": inputs,
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "git_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
        "git_dirty_path_count": len(dirty.stdout.splitlines()) if dirty.returncode == 0 else None,
    }


def validate_per_epoch_final_gate_report(
    report_path: Path,
    checkpoint_path: Path,
    *,
    epoch: int,
    protocol: str,
    launched_after_ns: int,
    min_tp: int,
    min_rq: float,
    min_sq: float,
    max_fp: int,
    min_pq: float | None = None,
) -> dict[str, Any]:
    blockers: list[str] = []
    if not report_path.is_file():
        return {"passed": False, "reason": "report_missing", "blockers": ["report_missing"]}
    report_mtime_ns = report_path.stat().st_mtime_ns
    if report_mtime_ns < launched_after_ns:
        blockers.append("report_stale_mtime")
    report = read_json_file(report_path)
    expected_hash = sha256_path(checkpoint_path)
    if report.get("checkpoint_sha256") != expected_hash:
        blockers.append("checkpoint_sha256_mismatch")
    if parse_int(report.get("epoch"), -1) != int(epoch):
        blockers.append("epoch_mismatch")
    if str(report.get("protocol") or "") != protocol:
        blockers.append("protocol_mismatch")
    metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
    required_metrics = ("tp", "RQ", "SQ", "fp")
    if min_pq is not None:
        required_metrics += ("PQ",)
    if any(name not in metrics for name in required_metrics):
        blockers.append("required_metrics_missing")
    tp = parse_int(metrics.get("tp"), -1)
    fp = parse_int(metrics.get("fp"), -1)
    rq = parse_float(metrics.get("RQ"), float("nan"))
    sq = parse_float(metrics.get("SQ"), float("nan"))
    pq = parse_float(metrics.get("PQ"), float("nan")) if min_pq is not None else None
    if not math.isfinite(rq) or not math.isfinite(sq) or tp < 0 or fp < 0:
        blockers.append("required_metrics_invalid")
    if tp < min_tp:
        blockers.append("tp_below_threshold")
    if not math.isfinite(rq) or rq < min_rq:
        blockers.append("rq_below_threshold")
    if not math.isfinite(sq) or sq < min_sq:
        blockers.append("sq_below_threshold")
    if max_fp >= 0 and fp > max_fp:
        blockers.append("fp_above_threshold")
    if min_pq is not None and (not math.isfinite(float(pq)) or float(pq) < min_pq):
        blockers.append("pq_below_threshold")
    return {
        "passed": not blockers,
        "reason": "passed" if not blockers else "per_epoch_final_gate_validation_failed",
        "blockers": sorted(set(blockers)),
        "report": rel(report_path),
        "checkpoint": rel(checkpoint_path),
        "checkpoint_sha256": expected_hash,
        "epoch": int(epoch),
        "protocol": protocol,
        "report_mtime_ns": report_mtime_ns,
        "launched_after_ns": launched_after_ns,
        "metrics": {"tp": tp, "RQ": rq, "SQ": sq, "fp": fp, **({"PQ": pq} if min_pq is not None else {})},
        "thresholds": {"min_tp": min_tp, "min_RQ": min_rq, "min_SQ": min_sq, "max_fp": max_fp, **({"min_PQ": min_pq} if min_pq is not None else {})},
    }


def run_per_epoch_final_gate(
    args: argparse.Namespace,
    *,
    checkpoint_path: Path,
    epoch: int,
) -> dict[str, Any]:
    if not args.require_final_instance_gate_for_best:
        return {"required": False, "passed": True, "reason": "not_required"}
    if not args.final_instance_gate_command_template or args.final_instance_gate_report is None:
        return {"required": True, "passed": False, "reason": "command_or_report_not_configured", "blockers": ["command_or_report_not_configured"]}
    report_path = args.final_instance_gate_report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.exists():
        report_path.unlink()
    launched_after_ns = time.time_ns()
    replacements = {
        "checkpoint": shlex.quote(str(checkpoint_path.resolve())),
        "epoch": str(int(epoch)),
        "report": shlex.quote(str(report_path.resolve())),
    }
    try:
        command = args.final_instance_gate_command_template.format(**replacements)
    except (KeyError, ValueError) as exc:
        return {"required": True, "passed": False, "reason": "invalid_command_template", "blockers": [f"invalid_command_template:{exc}"]}
    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=ROOT,
            timeout=max(int(args.final_instance_gate_timeout_seconds), 1),
            check=False,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"required": True, "passed": False, "reason": "gate_command_exception", "blockers": [f"gate_command_exception:{type(exc).__name__}"]}
    if completed.returncode != 0:
        return {
            "required": True,
            "passed": False,
            "reason": "gate_command_failed",
            "blockers": [f"gate_command_returncode:{completed.returncode}"],
            "command": command,
            "stdout_tail": completed.stdout[-2000:],
            "stderr_tail": completed.stderr[-2000:],
        }
    configured_min_pq = float(getattr(args, "min_final_instance_pq_for_best", 0.0))
    require_pq = bool(getattr(args, "select_best_by_final_instance_pq", False))
    result = validate_per_epoch_final_gate_report(
        report_path,
        checkpoint_path,
        epoch=epoch,
        protocol=args.final_instance_gate_protocol,
        launched_after_ns=launched_after_ns,
        min_tp=int(args.min_final_instance_tp_for_best),
        min_rq=float(args.min_final_instance_rq_for_best),
        min_sq=float(args.min_final_instance_sq_for_best),
        max_fp=int(args.max_final_instance_fp_for_best),
        min_pq=configured_min_pq if configured_min_pq > 0.0 else (0.0 if require_pq else None),
    )
    return {"required": True, "command": command, **result}


def checkpoint_payload(
    model: Any,
    args: argparse.Namespace,
    *,
    run_id: str,
    pid: int,
    epoch: int,
    selection_score: float,
    selection_gate: dict[str, Any],
    schedule: dict[str, float],
    mask_schedule: dict[str, float],
    bottleneck_profile: dict[str, Any],
    query_class_weights: Any,
    boundary: str,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    semantic_class_weights_value: Any | None = None,
    history: list[dict[str, Any]] | None = None,
    best_score: float | None = None,
    best_val_semantic: float | None = None,
    best_checkpoint_written: bool | None = None,
    rng_state: Any | None = None,
    training_rng_state: dict[str, Any] | None = None,
    intra_epoch_progress: dict[str, Any] | None = None,
) -> dict[str, Any]:
    objective_config = objective_config_from_args(args)
    geometry_config = None
    if getattr(args, "geometry_decoder_mode", "legacy_debug") == "geometry_v2":
        geometry_config = geometry_decoder_config(
            hidden_dim=args.hidden_dim, heads=args.heads, num_queries=args.num_queries,
            decoder_layers=args.query_decoder_layers, identity_dim=PANOPTIC_IDENTITY_DIM,
            num_stuff_queries=getattr(args, "num_stuff_queries", 32),
            local_neighbors=getattr(args, "geometry_local_neighbors", 4),
            coarse_grid_size=getattr(args, "geometry_coarse_grid_size", 4),
            typed_stuff_slots=getattr(args, "typed_stuff_slots", False),
            tensor_ring_rank=getattr(args, "tensor_ring_rank", 0),
            geometry_attention_tile_size=getattr(args, "geometry_attention_tile_size", 0),
        )
    sq_rq = sq_rq_config(
        enabled=bool(getattr(args, "sq_rq_enabled", False)), hidden_dim=args.hidden_dim,
        heads=args.heads, num_labels=36, gradient_scale=getattr(args, "sq_rq_gradient_scale", 0.0),
        query_confidence_threshold=getattr(args, "sq_rq_query_confidence_threshold", 0.6),
        token_membership_threshold=getattr(args, "sq_rq_token_membership_threshold", 0.5),
        training_membership_temperature=getattr(args, "sq_rq_training_membership_temperature", 0.1),
        semantic_query_residual_enabled=getattr(args, "semantic_query_residual_enabled", False),
    )
    runtime_model = getattr(model, "_orig_mod", model)
    sq_rq_auto_fused = bool(getattr(args, "_sq_rq_auto_fused", False))
    sq_rq_deployment = (
        sq_rq_deployment_config(
            enabled=bool(
                sq_rq["enabled"]
                and getattr(runtime_model, "sq_rq_runtime_enabled", sq_rq["enabled"])
            ),
            query_confidence_threshold=getattr(args, "sq_rq_query_confidence_threshold", 0.6),
            token_membership_threshold=getattr(args, "sq_rq_token_membership_threshold", 0.5),
            auto_fused=sq_rq_auto_fused,
            auto_fuse_reason=getattr(args, "_sq_rq_auto_fuse_reason", None),
        )
        if sq_rq["enabled"]
        else None
    )
    ownership = ownership_config(hidden_dim=args.hidden_dim, num_queries=args.num_queries) if geometry_config is not None and sq_rq["enabled"] else None
    gradient_control = gradient_control_config(args.gradient_control) if ownership is not None and args.gradient_control == "pcgrad" else None
    sparse_router = sparse_router_config(
        enabled=True, hidden_dim=args.hidden_dim, num_experts=args.router_num_experts,
        top_k=args.router_top_k, temperature=args.router_temperature,
        typed_branch_routers=getattr(args, "typed_branch_routers", False),
        branch_num_experts=getattr(args, "branch_num_experts", 2),
        branch_top_k=getattr(args, "branch_top_k", 1),
        branch_capacity_factor=getattr(args, "branch_capacity_factor", 1.25),
        branch_dropless=getattr(args, "branch_dropless", False),
    ) if getattr(args, "learned_sparse_router", False) else None
    input_protocol = input_protocol_from_args(args)
    active_feature_names = feature_names_for_input_protocol(input_protocol)
    state_dict = checkpoint_state_dict(model)
    quality_head_trained = float(
        getattr(args, "rq_sq_quality_calibration_loss_weight", 0.0)
    ) > 0.0
    quality_admission_promoted = bool(
        quality_head_trained
        and boundary == "best_selection_checkpoint"
        and selection_gate.get("passed") is True
        and math.isfinite(float(selection_score))
    )
    payload = {
        "schema_version": PANOPTIC_SPARSE_ROUTER_SCHEMA_VERSION if sparse_router else ("floorplancad_line_token_panoptic_moe_checkpoint_v6_geometry_v2_sq_rq_ownership_pcgrad" if gradient_control else ("floorplancad_line_token_panoptic_moe_checkpoint_v5_geometry_v2_sq_rq_ownership" if ownership else ("floorplancad_line_token_panoptic_moe_checkpoint_v4_geometry_v2_sq_rq" if sq_rq["enabled"] else ("floorplancad_line_token_panoptic_moe_checkpoint_v3_geometry_v2" if geometry_config else "floorplancad_line_token_panoptic_moe_checkpoint_v2")))),
        "state_dict": state_dict,
        "feature_names": active_feature_names,
        "hidden_dim": args.hidden_dim,
        "layers": args.layers,
        "heads": args.heads,
        "num_queries": args.num_queries,
        "query_decoder_layers": args.query_decoder_layers,
        "geometry_decoder_mode": getattr(args, "geometry_decoder_mode", "legacy_debug"),
        "geometry_config": geometry_config,
        "sq_rq_config": sq_rq if sq_rq["enabled"] else None,
        "sq_rq_training_thresholds": sq_rq_training_threshold_schedule(args, epoch) if sq_rq["enabled"] else None,
        "sq_rq_validation_thresholds": sq_rq_deployment_thresholds(args, epoch) if sq_rq["enabled"] else None,
        "sq_rq_deployment": sq_rq_deployment,
        "sq_rq_auto_fused": sq_rq_auto_fused,
        "sq_rq_auto_fuse_reason": getattr(args, "_sq_rq_auto_fuse_reason", None),
        "ownership_config": ownership,
        "gradient_control_config": gradient_control,
        "sparse_router_config": sparse_router,
        "weak_family_feature_fusion": bool(getattr(args, "weak_family_feature_fusion", False)),
        "quality_query_gradient_scale": float(getattr(args, "quality_query_gradient_scale", 0.0)),
        "content_seeded_queries": bool(getattr(args, "content_seeded_queries", False)),
        "component_seeded_queries": bool(getattr(args, "component_seeded_queries", False)),
        "component_seed_loss_weight": float(getattr(args, "component_seed_loss_weight", 0.0)),
        "explicit_route_classifier": bool(getattr(args, "explicit_route_classifier", False)),
        "route_conditioning_residual_scale": float(getattr(args, "route_conditioning_residual_scale", 0.10)),
        "route_conditioning_enable_after_epoch": int(getattr(args, "route_conditioning_enable_after_epoch", 2)),
        "route_conditioning_warmup_epochs": int(getattr(args, "route_conditioning_warmup_epochs", 3)),
        "dense_attention_feature_adapter": bool(getattr(args, "dense_attention_feature_adapter", False)),
        "dense_attention_window_size": int(getattr(args, "dense_attention_window_size", 128)),
        "dense_attention_adapter_residual_scale": float(getattr(args, "dense_attention_adapter_residual_scale", 0.10)),
        "dense_attention_adapter_enable_after_epoch": int(getattr(args, "dense_attention_adapter_enable_after_epoch", 2)),
        "dense_attention_adapter_warmup_epochs": int(getattr(args, "dense_attention_adapter_warmup_epochs", 3)),
        "input_protocol": input_protocol,
        "dropout": args.dropout,
        "position_encoding_version": POSITION_ENCODING_VERSION,
        "position_max_frequency_log2": POSITION_MAX_FREQUENCY_LOG2,
        "quality_head": PANOPTIC_QUALITY_HEAD_VERSION,
        "identity_head_version": PANOPTIC_IDENTITY_HEAD_VERSION,
        "identity_dim": PANOPTIC_IDENTITY_DIM,
        "checkpoint_abi": checkpoint_abi_metadata(
            args.max_tokens_per_record, geometry_config=geometry_config,
            geometry_state_schema_sha256=model_state_schema_sha256(state_dict) if geometry_config else None,
            sq_rq=sq_rq,
            sq_rq_deployment=sq_rq_deployment,
            ownership=ownership,
            gradient_control=gradient_control,
            sparse_router=sparse_router,
            quality_head_trained=quality_head_trained,
            quality_admission_promoted=quality_admission_promoted,
            input_protocol=input_protocol,
            quality_mask_threshold=args.deployment_mask_threshold,
        ),
        "feature_schema_sha256": feature_schema_sha256(active_feature_names),
        "ontology_sha256": ontology_sha256(),
        "window_contract_sha256": canonical_json_sha256(window_contract(
            args.max_tokens_per_record,
            target_schema_version=input_protocol["target_schema_version"],
        )),
        "num_labels": 36,
        "ignore_label": IGNORE_LABEL,
        "run_id": run_id,
        "pid": pid,
        "epoch": epoch,
        "source": "CadStruct-MoE line_token_component panoptic query expert",
        "train_path": rel(args.train),
        "val_path": rel(args.val),
        "training_provenance": getattr(args, "_training_provenance", None),
        "checkpoint_boundary": boundary,
        "quality_admission_promoted": quality_admission_promoted,
        "batch_records": args.batch_records,
        "train_prefetch_records": args.train_prefetch_records,
        "train_prefetch_workers": args.train_prefetch_workers,
        "amp": args.amp,
        "gradient_control": args.gradient_control,
        "gradient_control_version": gradient_control["version"] if gradient_control else "legacy_sum_v1",
        "gradient_control_config_sha256": canonical_json_sha256(gradient_control_config(args.gradient_control)),
        "disable_tf32": args.disable_tf32,
        "enable_cudnn_benchmark": args.enable_cudnn_benchmark,
        "compile_model": bool(args.compile_model),
        "compile_mode": args.compile_mode,
        "progress_checkpoint_records": args.progress_checkpoint_records,
        "progress_checkpoint_seconds": args.progress_checkpoint_seconds,
        "checkpoint_archive_dir": rel(args.checkpoint_archive_dir),
        "checkpoint_archive_keep": int(args.checkpoint_archive_keep),
        "graceful_signal_checkpoint": True,
        "component_matching": args.component_matching,
        "losses": [
            "semantic_ce",
            "hungarian_component_query_ce",
            "query_objectness_bce",
            "query_objectness_positive_margin_floor",
            "query_objectness_negative_margin_ceiling",
            "rq_sq_quality_calibration",
            "hungarian_component_mask_bce",
            "hungarian_component_mask_dice",
            "mask_area_ratio_regularizer",
            "mask_tversky_recall_regularizer",
            "mask_positive_probability_floor",
            "adjacent_window_query_identity",
            "global_query_token_ownership_ce",
        ],
        "matching": "class_plus_focal_mask_plus_dice_exact_hungarian_v1",
        "checkpoint_metric": args.checkpoint_metric,
        "objective_config": objective_config,
        "objective_config_hash": objective_config_hash(objective_config),
        "selection_score": selection_score,
        "selection_gate": selection_gate,
        "objectness_schedule": schedule,
        "mask_loss_schedule": mask_schedule,
        "mask_positive_weight": mask_schedule["mask_positive_weight"],
        "mask_negative_weight": mask_schedule["mask_negative_weight"],
        "mask_focal_gamma": args.mask_focal_gamma,
        "mask_area_ratio_loss_weight": mask_schedule["mask_area_ratio_loss_weight"],
        "mask_area_overcoverage_weight": mask_schedule["mask_area_overcoverage_weight"],
        "mask_tversky_loss_weight": mask_schedule["mask_tversky_loss_weight"],
        "mask_tversky_alpha": args.mask_tversky_alpha,
        "mask_tversky_beta": args.mask_tversky_beta,
        "mask_positive_prob_floor_loss_weight": mask_schedule["mask_positive_prob_floor_loss_weight"],
        "mask_positive_prob_floor": args.mask_positive_prob_floor,
        "mask_precision_phase_start_epoch": args.mask_precision_phase_start_epoch,
        "mask_precision_phase_positive_weight": args.mask_precision_phase_positive_weight,
        "mask_precision_phase_negative_weight": args.mask_precision_phase_negative_weight,
        "mask_precision_phase_area_ratio_loss_weight": args.mask_precision_phase_area_ratio_loss_weight,
        "mask_precision_phase_area_overcoverage_weight": args.mask_precision_phase_area_overcoverage_weight,
        "mask_precision_phase_tversky_loss_weight": args.mask_precision_phase_tversky_loss_weight,
        "mask_precision_phase_positive_prob_floor_loss_weight": args.mask_precision_phase_positive_prob_floor_loss_weight,
        "query_objectness_loss_weight": args.query_objectness_loss_weight,
        "query_objectness_positive_weight": args.query_objectness_positive_weight,
        "query_objectness_negative_weight": args.query_objectness_negative_weight,
        "query_objectness_positive_margin_floor_loss_weight": args.query_objectness_positive_margin_floor_loss_weight,
        "query_objectness_negative_margin_ceiling_loss_weight": args.query_objectness_negative_margin_ceiling_loss_weight,
        "rq_sq_quality_calibration_loss_weight": args.rq_sq_quality_calibration_loss_weight,
        "rq_sq_quality_ranking_weight": args.rq_sq_quality_ranking_weight,
        "rq_sq_quality_ranking_margin": args.rq_sq_quality_ranking_margin,
        "rq_sq_quality_ranking_top_k": args.rq_sq_quality_ranking_top_k,
        "rq_sq_quality_hard_negative_weight": args.rq_sq_quality_hard_negative_weight,
        "rq_sq_quality_unmatched_ceiling_weight": args.rq_sq_quality_unmatched_ceiling_weight,
        "rq_sq_quality_unmatched_ceiling_probability": args.rq_sq_quality_unmatched_ceiling_probability,
        "family_recall_focus": getattr(args, "family_recall_focus", ""),
        "family_recall_loss_weight": getattr(args, "family_recall_loss_weight", 0.0),
        "family_recall_admission_floor": getattr(args, "family_recall_admission_floor", 0.25),
        "family_recall_mask_prob_floor": getattr(args, "family_recall_mask_prob_floor", 0.35),
        "family_recall_quality_floor": getattr(args, "family_recall_quality_floor", 0.10),
        "active_loss_experts": getattr(args, "active_loss_experts", "joint_routed"),
        "hard_recall_labels": getattr(args, "hard_recall_labels", ""),
        "rq_admission_expert_weight": getattr(args, "rq_admission_expert_weight", 0.0),
        "mask_recall_expert_weight": getattr(args, "mask_recall_expert_weight", 0.0),
        "quality_deployment_expert_weight": getattr(args, "quality_deployment_expert_weight", 0.0),
        "hard_recall_admission_floor": getattr(args, "hard_recall_admission_floor", 0.55),
        "hard_recall_mask_prob_floor": getattr(args, "hard_recall_mask_prob_floor", 0.45),
        "hard_recall_deployment_floor": getattr(args, "hard_recall_deployment_floor", 0.22),
        "hard_recall_quality_target_floor": getattr(args, "hard_recall_quality_target_floor", 0.20),
        "route_classification_loss_weight": getattr(args, "route_classification_loss_weight", 0.0),
        "deployment_min_query_score": getattr(
            args, "deployment_min_query_score", DEFAULT_MIN_QUERY_SCORE
        ),
        "deployment_mask_threshold": getattr(
            args, "deployment_mask_threshold", PANOPTIC_QUALITY_MASK_THRESHOLD
        ),
        "sq_rq_base_semantic_loss_weight": getattr(args, "sq_rq_base_semantic_loss_weight", 0.25),
        "identity_loss_weight": args.identity_loss_weight,
        "identity_temperature": args.identity_temperature,
        "identity_negative_margin": args.identity_negative_margin,
            "router_load_balance_loss_weight": getattr(args, "router_load_balance_loss_weight", 0.01),
        "precision_phase_require_healthy_admission_epochs": args.precision_phase_require_healthy_admission_epochs,
        "precision_phase_min_object_recall": args.precision_phase_min_object_recall,
        "precision_phase_max_negative_margin_rate": args.precision_phase_max_negative_margin_rate,
        "precision_phase_min_sq_proxy": args.precision_phase_min_sq_proxy,
        "precision_phase_transition_epochs": args.precision_phase_transition_epochs,
        "objectness_warmup_positive_margin_floor_loss_weight": args.objectness_warmup_positive_margin_floor_loss_weight,
        "objectness_warmup_negative_margin_ceiling_loss_weight": args.objectness_warmup_negative_margin_ceiling_loss_weight,
        "active_query_objectness_loss_weight": schedule["query_objectness_loss_weight"],
        "active_query_objectness_positive_weight": schedule["query_objectness_positive_weight"],
        "active_query_objectness_negative_weight": schedule["query_objectness_negative_weight"],
        "active_query_objectness_positive_margin_floor_loss_weight": schedule["query_objectness_positive_margin_floor_loss_weight"],
        "active_query_objectness_positive_margin_floor": schedule["query_objectness_positive_margin_floor"],
        "active_query_objectness_negative_margin_ceiling_loss_weight": schedule["query_objectness_negative_margin_ceiling_loss_weight"],
        "active_query_objectness_negative_margin_ceiling": schedule["query_objectness_negative_margin_ceiling"],
        "objectness_warmup_epochs": args.objectness_warmup_epochs,
        "objectness_precision_phase_start_epoch": args.objectness_precision_phase_start_epoch,
        "objectness_precision_phase_loss_weight": args.objectness_precision_phase_loss_weight,
        "objectness_precision_phase_positive_weight": args.objectness_precision_phase_positive_weight,
        "objectness_precision_phase_negative_weight": args.objectness_precision_phase_negative_weight,
        "objectness_precision_phase_positive_margin_floor_loss_weight": args.objectness_precision_phase_positive_margin_floor_loss_weight,
        "objectness_precision_phase_negative_margin_ceiling_loss_weight": args.objectness_precision_phase_negative_margin_ceiling_loss_weight,
        "objectness_positive_margin_floor": args.objectness_positive_margin_floor,
        "objectness_negative_margin_ceiling": args.objectness_negative_margin_ceiling,
        "min_val_object_recall_for_checkpoint": args.min_val_object_recall_for_checkpoint,
        "min_val_mask_recall_for_checkpoint": args.min_val_mask_recall_for_checkpoint,
        "min_val_mask_precision_for_checkpoint": args.min_val_mask_precision_for_checkpoint,
        "max_val_positive_rate_ratio_for_checkpoint": args.max_val_positive_rate_ratio_for_checkpoint,
        "min_val_positive_object_margin_rate_for_checkpoint": args.min_val_positive_object_margin_rate_for_checkpoint,
        "min_val_positive_object_margin_mean_for_checkpoint": args.min_val_positive_object_margin_mean_for_checkpoint,
        "max_val_negative_object_margin_rate_for_checkpoint": args.max_val_negative_object_margin_rate_for_checkpoint,
        "min_val_rq_proxy_for_checkpoint": args.min_val_rq_proxy_for_checkpoint,
        "min_val_sq_proxy_for_checkpoint": args.min_val_sq_proxy_for_checkpoint,
        "require_final_instance_gate_for_best": args.require_final_instance_gate_for_best,
        "select_best_by_final_instance_pq": args.select_best_by_final_instance_pq,
        "final_instance_gate_report": rel(args.final_instance_gate_report),
        "final_instance_gate_command_template": args.final_instance_gate_command_template,
        "final_instance_gate_protocol": args.final_instance_gate_protocol,
        "final_instance_gate_timeout_seconds": args.final_instance_gate_timeout_seconds,
        "final_instance_gate_interval_epochs": args.final_instance_gate_interval_epochs,
        "final_instance_gate_checkpoint": rel(args.final_instance_gate_checkpoint),
        "min_final_instance_tp_for_best": args.min_final_instance_tp_for_best,
        "min_final_instance_rq_for_best": args.min_final_instance_rq_for_best,
        "min_final_instance_sq_for_best": args.min_final_instance_sq_for_best,
        "min_final_instance_pq_for_best": getattr(args, "min_final_instance_pq_for_best", 0.0),
        "max_final_instance_fp_for_best": args.max_final_instance_fp_for_best,
        "small_component_size": args.small_component_size,
        "target_selection_policy": "bottleneck_small_component_round_robin_when_components_exceed_queries",
        "bottleneck_profile": bottleneck_profile,
        "query_class_weights": [float(value) for value in query_class_weights.detach().cpu().tolist()],
        "semantic_class_weights": (
            None if semantic_class_weights_value is None
            else [float(value) for value in semantic_class_weights_value.detach().cpu().tolist()]
        ),
        "effective_argv": list(sys.argv),
        "claim_boundary": "Diagnostic/training checkpoint. Same-format PQ claim requires locked evaluator and comparable_for_matrix=true.",
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
        payload["optimizer_parameter_groups"] = [
            {
                "name": str(group.get("name", f"group_{index}")),
                "lr": float(group["lr"]),
                "weight_decay": float(group.get("weight_decay", 0.0)),
                "parameter_count": len(group["params"]),
            }
            for index, group in enumerate(optimizer.param_groups)
        ]
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
        payload["lr_scheduler_config"] = learning_rate_scheduler_config(args)
    if history is not None:
        payload["history"] = history
    if best_score is not None:
        payload["best_selection_score"] = best_score
    if best_val_semantic is not None:
        payload["best_val_semantic_token_accuracy"] = best_val_semantic
    if best_checkpoint_written is not None:
        payload["best_checkpoint_written"] = bool(best_checkpoint_written)
    if rng_state is not None:
        payload["rng_state"] = rng_state
    if training_rng_state is not None:
        payload["training_rng_state"] = training_rng_state
    if intra_epoch_progress is not None:
        payload["intra_epoch_progress"] = intra_epoch_progress
    return payload


def architecture_signature_from_args(args: argparse.Namespace) -> dict[str, Any]:
    geometry_config = None
    if getattr(args, "geometry_decoder_mode", "legacy_debug") == "geometry_v2":
        geometry_config = geometry_decoder_config(
            hidden_dim=args.hidden_dim, heads=args.heads, num_queries=args.num_queries,
            decoder_layers=args.query_decoder_layers, identity_dim=PANOPTIC_IDENTITY_DIM,
            num_stuff_queries=getattr(args, "num_stuff_queries", 32),
            local_neighbors=getattr(args, "geometry_local_neighbors", 4),
            coarse_grid_size=getattr(args, "geometry_coarse_grid_size", 4),
            typed_stuff_slots=getattr(args, "typed_stuff_slots", False),
            tensor_ring_rank=getattr(args, "tensor_ring_rank", 0),
            geometry_attention_tile_size=getattr(args, "geometry_attention_tile_size", 0),
        )
    input_protocol = input_protocol_from_args(args)
    active_feature_names = feature_names_for_input_protocol(input_protocol)
    return {
        "feature_dim": len(active_feature_names),
        "hidden_dim": args.hidden_dim,
        "layers": args.layers,
        "heads": args.heads,
        "num_queries": args.num_queries,
        "query_decoder_layers": args.query_decoder_layers,
        "num_labels": 36,
        "dropout": args.dropout,
        "position_encoding_version": POSITION_ENCODING_VERSION,
        "quality_head": PANOPTIC_QUALITY_HEAD_VERSION,
        "identity_head_version": PANOPTIC_IDENTITY_HEAD_VERSION,
        "identity_dim": PANOPTIC_IDENTITY_DIM,
        "geometry_decoder_mode": getattr(args, "geometry_decoder_mode", "legacy_debug"),
        "geometry_config": geometry_config,
        "sq_rq_config": sq_rq_config(
            enabled=bool(getattr(args, "sq_rq_enabled", False)), hidden_dim=args.hidden_dim,
            heads=args.heads, num_labels=36, gradient_scale=getattr(args, "sq_rq_gradient_scale", 0.0),
            query_confidence_threshold=getattr(args, "sq_rq_query_confidence_threshold", 0.6),
            token_membership_threshold=getattr(args, "sq_rq_token_membership_threshold", 0.5),
            training_membership_temperature=getattr(args, "sq_rq_training_membership_temperature", 0.1),
            semantic_query_residual_enabled=getattr(args, "semantic_query_residual_enabled", False),
        ) if getattr(args, "sq_rq_enabled", False) else None,
        "sparse_router_config": sparse_router_config(
            enabled=True, hidden_dim=args.hidden_dim, num_experts=args.router_num_experts,
            top_k=args.router_top_k, temperature=args.router_temperature,
            typed_branch_routers=getattr(args, "typed_branch_routers", False),
            branch_num_experts=getattr(args, "branch_num_experts", 2),
            branch_top_k=getattr(args, "branch_top_k", 1),
            branch_capacity_factor=getattr(args, "branch_capacity_factor", 1.25),
            branch_dropless=getattr(args, "branch_dropless", False),
        ) if getattr(args, "learned_sparse_router", False) else None,
        "explicit_route_classifier": bool(getattr(args, "explicit_route_classifier", False)),
        "route_conditioning_residual_scale": float(getattr(args, "route_conditioning_residual_scale", 0.10)),
        "dense_attention_feature_adapter": bool(getattr(args, "dense_attention_feature_adapter", False)),
        "dense_attention_window_size": int(getattr(args, "dense_attention_window_size", 128)),
        "dense_attention_adapter_residual_scale": float(getattr(args, "dense_attention_adapter_residual_scale", 0.10)),
        "input_protocol": input_protocol,
    }


def architecture_signature_from_checkpoint(ckpt: dict[str, Any]) -> dict[str, Any]:
    return {
        "feature_dim": len(ckpt.get("feature_names", [])),
        "hidden_dim": int(ckpt.get("hidden_dim", -1)),
        "layers": int(ckpt.get("layers", -1)),
        "heads": int(ckpt.get("heads", -1)),
        "num_queries": int(ckpt.get("num_queries", -1)),
        "query_decoder_layers": int(ckpt.get("query_decoder_layers", -1)),
        "num_labels": int(ckpt.get("num_labels", -1)),
        "dropout": float(ckpt.get("dropout", 0.1)),
        "position_encoding_version": ckpt.get("position_encoding_version"),
        "quality_head": ckpt.get("quality_head"),
        "identity_head_version": ckpt.get("identity_head_version"),
        "identity_dim": int(ckpt.get("identity_dim", -1)),
        "geometry_decoder_mode": ckpt.get("geometry_decoder_mode", "legacy_debug"),
        "geometry_config": ckpt.get("geometry_config"),
        "sq_rq_config": ckpt.get("sq_rq_config"),
        "sparse_router_config": ckpt.get("sparse_router_config"),
        "explicit_route_classifier": bool(ckpt.get("explicit_route_classifier", False)),
        "route_conditioning_residual_scale": float(ckpt.get("route_conditioning_residual_scale", 0.10)),
        "dense_attention_feature_adapter": bool(ckpt.get("dense_attention_feature_adapter", False)),
        "dense_attention_window_size": int(ckpt.get("dense_attention_window_size", 128)),
        "dense_attention_adapter_residual_scale": float(ckpt.get("dense_attention_adapter_residual_scale", 0.10)),
        "input_protocol": ckpt.get("input_protocol"),
    }


def objective_signature_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "checkpoint_metric": args.checkpoint_metric,
        "query_objectness_loss_weight": args.query_objectness_loss_weight,
        "query_objectness_positive_weight": args.query_objectness_positive_weight,
        "query_objectness_negative_weight": args.query_objectness_negative_weight,
        "mask_loss_weight": args.mask_loss_weight,
        "mask_tversky_loss_weight": args.mask_tversky_loss_weight,
        "rq_sq_quality_calibration_loss_weight": args.rq_sq_quality_calibration_loss_weight,
        "rq_sq_quality_ranking_weight": getattr(args, "rq_sq_quality_ranking_weight", 0.0),
        "rq_sq_quality_ranking_margin": getattr(args, "rq_sq_quality_ranking_margin", 0.0),
        "rq_sq_quality_ranking_top_k": getattr(args, "rq_sq_quality_ranking_top_k", 1),
        "rq_sq_quality_hard_negative_weight": getattr(
            args, "rq_sq_quality_hard_negative_weight", 0.0
        ),
        "rq_sq_quality_unmatched_ceiling_weight": getattr(
            args, "rq_sq_quality_unmatched_ceiling_weight", 0.0
        ),
        "rq_sq_quality_unmatched_ceiling_probability": getattr(
            args, "rq_sq_quality_unmatched_ceiling_probability", 0.05
        ),
        "family_recall_focus": getattr(args, "family_recall_focus", ""),
        "family_recall_loss_weight": getattr(args, "family_recall_loss_weight", 0.0),
        "active_loss_experts": getattr(args, "active_loss_experts", "joint_routed"),
        "hard_recall_labels": getattr(args, "hard_recall_labels", ""),
        "hard_recall_families": getattr(args, "hard_recall_families", ""),
        "component_seed_loss_weight": getattr(args, "component_seed_loss_weight", 0.0),
        "rq_admission_expert_weight": getattr(args, "rq_admission_expert_weight", 0.0),
        "mask_recall_expert_weight": getattr(args, "mask_recall_expert_weight", 0.0),
        "quality_deployment_expert_weight": getattr(args, "quality_deployment_expert_weight", 0.0),
        "hard_recall_deployment_floor": getattr(args, "hard_recall_deployment_floor", 0.22),
        "hard_recall_quality_target_floor": getattr(args, "hard_recall_quality_target_floor", 0.20),
        "route_classification_loss_weight": getattr(args, "route_classification_loss_weight", 0.0),
        "route_conditioning_enable_after_epoch": getattr(args, "route_conditioning_enable_after_epoch", 2),
        "route_conditioning_warmup_epochs": getattr(args, "route_conditioning_warmup_epochs", 3),
        "dense_attention_adapter_enable_after_epoch": getattr(args, "dense_attention_adapter_enable_after_epoch", 2),
        "dense_attention_adapter_warmup_epochs": getattr(args, "dense_attention_adapter_warmup_epochs", 3),
        "identity_loss_weight": args.identity_loss_weight,
        "identity_temperature": args.identity_temperature,
        "identity_negative_margin": args.identity_negative_margin,
        "router_load_balance_loss_weight": getattr(args, "router_load_balance_loss_weight", 0.01),
    }


def objective_signature_from_checkpoint(ckpt: dict[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint_metric": ckpt.get("checkpoint_metric"),
        "query_objectness_loss_weight": ckpt.get("query_objectness_loss_weight"),
        "query_objectness_positive_weight": ckpt.get("query_objectness_positive_weight"),
        "query_objectness_negative_weight": ckpt.get("query_objectness_negative_weight"),
        "mask_loss_weight": ckpt.get("mask_loss_weight"),
        "mask_tversky_loss_weight": ckpt.get("mask_tversky_loss_weight"),
        "rq_sq_quality_calibration_loss_weight": ckpt.get("rq_sq_quality_calibration_loss_weight", 0.0),
        "rq_sq_quality_ranking_weight": ckpt.get("rq_sq_quality_ranking_weight", 0.0),
        "rq_sq_quality_ranking_margin": ckpt.get("rq_sq_quality_ranking_margin", 0.0),
        "rq_sq_quality_ranking_top_k": ckpt.get("rq_sq_quality_ranking_top_k", 1),
        "rq_sq_quality_hard_negative_weight": ckpt.get(
            "rq_sq_quality_hard_negative_weight", 0.0
        ),
        "rq_sq_quality_unmatched_ceiling_weight": ckpt.get(
            "rq_sq_quality_unmatched_ceiling_weight", 0.0
        ),
        "rq_sq_quality_unmatched_ceiling_probability": ckpt.get(
            "rq_sq_quality_unmatched_ceiling_probability", 0.05
        ),
        "family_recall_focus": ckpt.get("family_recall_focus", ""),
        "family_recall_loss_weight": ckpt.get("family_recall_loss_weight", 0.0),
        "active_loss_experts": ckpt.get("active_loss_experts", "joint_routed"),
        "hard_recall_labels": ckpt.get("hard_recall_labels", ""),
        "hard_recall_families": ckpt.get("hard_recall_families", ""),
        "component_seed_loss_weight": ckpt.get("component_seed_loss_weight", 0.0),
        "rq_admission_expert_weight": ckpt.get("rq_admission_expert_weight", 0.0),
        "mask_recall_expert_weight": ckpt.get("mask_recall_expert_weight", 0.0),
        "quality_deployment_expert_weight": ckpt.get("quality_deployment_expert_weight", 0.0),
        "hard_recall_deployment_floor": ckpt.get("hard_recall_deployment_floor", 0.22),
        "hard_recall_quality_target_floor": ckpt.get("hard_recall_quality_target_floor", 0.20),
        "route_classification_loss_weight": ckpt.get("route_classification_loss_weight", 0.0),
        "route_conditioning_enable_after_epoch": ckpt.get("route_conditioning_enable_after_epoch", 2),
        "route_conditioning_warmup_epochs": ckpt.get("route_conditioning_warmup_epochs", 3),
        "dense_attention_adapter_enable_after_epoch": ckpt.get("dense_attention_adapter_enable_after_epoch", 2),
        "dense_attention_adapter_warmup_epochs": ckpt.get("dense_attention_adapter_warmup_epochs", 3),
        "sq_rq_base_semantic_loss_weight": ckpt.get("sq_rq_base_semantic_loss_weight", 0.25),
        "identity_loss_weight": ckpt.get("identity_loss_weight"),
        "identity_temperature": ckpt.get("identity_temperature"),
        "identity_negative_margin": ckpt.get("identity_negative_margin"),
        "router_load_balance_loss_weight": ckpt.get("router_load_balance_loss_weight", 0.0),
    }


OBJECTIVE_CONFIG_EXCLUDED_ARGS = {
    "allow_optimizer_objective_mismatch",
    "checkpoint_archive_dir",
    "checkpoint_archive_keep",
    "compile_mode",
    "compile_model",
    "device",
    "enable_cudnn_benchmark",
    "disable_tf32",
    "init_checkpoint",
    "last_model_output",
    "model_output",
    "progress_checkpoint_records",
    "progress_checkpoint_seconds",
    "progress_status_records",
    "progress_status_seconds",
    "report",
    "resume_checkpoint",
    "resume_optimizer",
    "train_prefetch_records",
    "train_prefetch_workers",
    "_training_provenance",
    "_sq_rq_auto_fused",
    "_sq_rq_auto_fuse_reason",
}


def normalize_objective_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): normalize_objective_value(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [normalize_objective_value(item) for item in value]
    if isinstance(value, set):
        return sorted((normalize_objective_value(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return float(format(value, ".17g"))
    if value is None or isinstance(value, (bool, int, str)):
        return value
    return str(value)


def objective_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    config = {
        key: normalize_objective_value(value)
        for key, value in sorted(vars(args).items())
        if key not in OBJECTIVE_CONFIG_EXCLUDED_ARGS
    }
    config["quality_objective_version"] = PANOPTIC_QUALITY_OBJECTIVE_VERSION
    config["quality_objective_config"] = quality_objective_contract(
        mask_threshold=getattr(args, "deployment_mask_threshold", PANOPTIC_QUALITY_MASK_THRESHOLD)
    )
    return config


def objective_config_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(normalize_objective_value(config), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def learning_rate_scheduler_config(args: argparse.Namespace) -> dict[str, float | int | str]:
    return {
        "name": "warmup_cosine_v1" if int(args.lr_decay_steps) > 0 else "constant_v1",
        "warmup_steps": int(args.lr_warmup_steps),
        "decay_steps": int(args.lr_decay_steps),
        "min_scale": float(args.lr_min_scale),
        "backbone_lr": float(args.lr),
        "router_lr_scale": float(args.router_lr_scale),
        "head_lr_scale": float(args.head_lr_scale),
    }


def optimizer_parameter_groups(model: Any, args: argparse.Namespace) -> list[dict[str, Any]]:
    runtime_model = getattr(model, "_orig_mod", model)
    grouped: dict[str, list[Any]] = {"backbone": [], "router": [], "heads": []}
    router_prefixes = (
        "sparse_router.", "sparse_experts.", "sparse_router_norm.", "branch_routers.",
        "route_family_head.", "route_family_embed.", "route_token_gate.",
        "route_token_norm.", "route_residual_logit_gate",
    )
    head_prefixes = (
        "semantic_head.", "query_", "token_ownership_head.", "null_ownership_head.",
        "sq_rq_cross_attention.", "sq_mask_residual_projection.",
        "family_seed_head.", "component_seed_head.", "thing_seed_score.",
    )
    gate_names = {
        "bridge_gate", "semantic_query_residual_gate", "sq_mask_residual_gate",
        "sq_ownership_residual_gate", "ownership_residual_gate",
    }
    for name, parameter in runtime_model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(router_prefixes):
            grouped["router"].append(parameter)
        elif name in gate_names or name.startswith(head_prefixes):
            grouped["heads"].append(parameter)
        else:
            grouped["backbone"].append(parameter)
    expected = [parameter for parameter in runtime_model.parameters() if parameter.requires_grad]
    assigned = [parameter for values in grouped.values() for parameter in values]
    if len(assigned) != len(expected) or {id(parameter) for parameter in assigned} != {id(parameter) for parameter in expected}:
        raise RuntimeError("optimizer parameter groups are incomplete or overlapping")
    scales = {"backbone": 1.0, "router": float(args.router_lr_scale), "heads": float(args.head_lr_scale)}
    return [
        {
            "name": name,
            "params": parameters,
            "lr": float(args.lr) * scales[name],
            "weight_decay": 0.01,
        }
        for name, parameters in grouped.items()
        if parameters
    ]


def configure_quality_calibration_scope(model: Any, enabled: bool) -> dict[str, Any]:
    runtime_model = getattr(model, "_orig_mod", model)
    if not enabled:
        return {
            "enabled": False,
            "trainable_parameter_names": [
                name for name, parameter in runtime_model.named_parameters() if parameter.requires_grad
            ],
        }
    trainable_parameter_names = []
    frozen_parameter_count = 0
    for name, parameter in runtime_model.named_parameters():
        trainable = name.startswith("query_quality_head.")
        parameter.requires_grad_(trainable)
        if trainable:
            trainable_parameter_names.append(name)
        else:
            frozen_parameter_count += int(parameter.numel())
    if not trainable_parameter_names:
        raise RuntimeError("quality calibration requires query_quality_head parameters")
    return {
        "enabled": True,
        "policy": "freeze_all_except_query_quality_head_v1",
        "trainable_parameter_names": trainable_parameter_names,
        "trainable_parameter_count": sum(
            int(parameter.numel())
            for name, parameter in runtime_model.named_parameters()
            if name in trainable_parameter_names
        ),
        "frozen_parameter_count": frozen_parameter_count,
    }


def build_optimizer_and_scheduler(torch: Any, model: Any, args: argparse.Namespace) -> tuple[Any, Any | None]:
    optimizer = torch.optim.AdamW(optimizer_parameter_groups(model, args))
    config = learning_rate_scheduler_config(args)
    if int(config["decay_steps"]) <= 0:
        return optimizer, None
    warmup_steps = int(config["warmup_steps"])
    decay_steps = max(int(config["decay_steps"]), warmup_steps + 1)
    min_scale = float(config["min_scale"])

    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = min(max((step - warmup_steps) / float(decay_steps - warmup_steps), 0.0), 1.0)
        return min_scale + (1.0 - min_scale) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)


def gradient_control_parameter_groups(model: Any) -> tuple[list[Any], dict[str, list[Any]]]:
    """Assign every trainable parameter to exactly one PCGrad ownership domain."""
    runtime_model = getattr(model, "_orig_mod", model)
    shared: list[Any] = []
    task_specific = {
        "semantic": [],
        "query_mask_quality": [],
        "teacher": [],
        "identity": [],
        "ownership": [],
    }
    shared_prefixes = (
        "input_proj.", "encoder.",
        "segment_encoder.", "segment_aggregate_encoder.", "segment_context_encoder.",
        "segment_pool_score.", "segment_fusion_norm.",
        "page_global_proj.", "layer_context_proj.", "context_fusion_norm.",
        "query_decoder.", "query_embed.", "thing_query_embed.", "stuff_query_embed.",
        "query_type_embed.", "query_anchor.", "local_graph_proj.", "coarse_proj.",
        "sparse_router.", "sparse_experts.", "sparse_router_norm.", "branch_routers.",
        "sq_rq_cross_attention.",
        "dense_adapter_attn.", "dense_adapter_ffn.",
        "dense_adapter_norm_attn.", "dense_adapter_norm_ffn.",
        "dense_adapter_residual_logit_gate",
    )
    shared_names = {"bridge_gate", "segment_fusion_gate", "context_fusion_gate"}
    for name, parameter in runtime_model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name in shared_names or name.startswith(shared_prefixes):
            shared.append(parameter)
        elif name.startswith("semantic_head."):
            task_specific["semantic"].append(parameter)
        elif name.startswith((
            "route_family_head.", "route_family_embed.", "route_token_gate.",
            "route_token_norm.", "route_residual_logit_gate",
        )):
            task_specific.setdefault("router", []).append(parameter)
        elif name == "semantic_query_residual_gate":
            task_specific["query_mask_quality"].append(parameter)
        elif name.startswith("query_identity_head."):
            task_specific["identity"].append(parameter)
        elif name in {"ownership_residual_gate", "sq_ownership_residual_gate"} or name.startswith(("query_ownership_head.", "token_ownership_head.", "null_ownership_head.")):
            task_specific["ownership"].append(parameter)
        else:
            task_specific["query_mask_quality"].append(parameter)
    owned = shared + [parameter for values in task_specific.values() for parameter in values]
    expected = [parameter for parameter in runtime_model.parameters() if parameter.requires_grad]
    if {id(parameter) for parameter in owned} != {id(parameter) for parameter in expected} or len(owned) != len(expected):
        raise RuntimeError("gradient-control parameter ownership is incomplete or overlapping")
    return shared, task_specific


def production_gradient_step(
    torch: Any,
    model: Any,
    optimizer: Any,
    task_losses: dict[str, Any | None],
    *,
    scheduler: Any | None = None,
    mode: str,
    task_specific_losses: dict[str, Any | None] | None = None,
    max_grad_norm: float = 1.0,
) -> dict[str, Any]:
    """Run one optimizer step with legacy sum or production PCGrad semantics."""
    optimizer.zero_grad(set_to_none=True)
    active = [loss for loss in task_losses.values() if loss is not None and bool(getattr(loss, "requires_grad", False))]
    if not active:
        raise ValueError("gradient step requires at least one differentiable task loss")
    if mode == "sum":
        torch.stack(active).sum().backward()
        report = {
            "schema_version": "floorplancad_multitask_gradient_control_report_v1",
            "mode": "sum",
            "active_tasks": [task for task, loss in task_losses.items() if loss is not None and bool(getattr(loss, "requires_grad", False))],
            "amp_scale": 1.0,
            "projection_count": 0,
            "raw_pairwise": {"pairs": {}},
            "projected_pairwise": {"pairs": {}},
        }
    elif mode == "pcgrad":
        shared, task_specific = gradient_control_parameter_groups(model)
        report = assign_multitask_gradients(
            torch,
            task_losses,
            shared,
            task_specific,
            task_specific_losses=task_specific_losses,
            amp_scale=1.0,
        )
        report["mode"] = "pcgrad"
    else:
        raise ValueError("gradient-control mode must be sum or pcgrad")
    clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
    if not bool(torch.isfinite(torch.as_tensor(clip_norm)).item()):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("non-finite gradient norm after gradient control")
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    report["clip_grad_norm"] = float(torch.as_tensor(clip_norm).item())
    report["learning_rates"] = {
        str(group.get("name", f"group_{index}")): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }
    return report


def summarize_gradient_control_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    pair_cosines: dict[str, list[float]] = {}
    projected_pair_cosines: dict[str, list[float]] = {}
    for report in reports:
        for name, values in report.get("raw_pairwise", {}).get("pairs", {}).items():
            pair_cosines.setdefault(name, []).append(float(values["cosine"]))
        for name, values in report.get("projected_pairwise", {}).get("pairs", {}).items():
            projected_pair_cosines.setdefault(name, []).append(float(values["cosine"]))
    return {
        "mode": reports[0].get("mode") if reports else None,
        "optimizer_steps_audited": len(reports),
        "conflicting_pair_observations": sum(value < 0.0 for values in pair_cosines.values() for value in values),
        "projection_count": sum(int(report.get("projection_count", 0)) for report in reports),
        "raw_pairwise_mean_cosines": {name: sum(values) / len(values) for name, values in sorted(pair_cosines.items())},
        "projected_pairwise_mean_cosines": {name: sum(values) / len(values) for name, values in sorted(projected_pair_cosines.items())},
    }


def load_resume_checkpoint(
    torch: Any,
    model: Any,
    optimizer: Any,
    path: Path | None,
    args: argparse.Namespace,
    device: Any,
    scheduler: Any | None = None,
) -> dict[str, Any]:
    if path is None:
        return {
            "enabled": False,
            "path": None,
            "start_epoch": 1,
            "history": [],
            "best_score": -float("inf"),
            "best_val_semantic": -1.0,
            "best_checkpoint_written": False,
            "run_id": None,
        }
    if not path.exists():
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    checkpoint_abi = validate_checkpoint_abi(ckpt)
    expected = architecture_signature_from_args(args)
    actual = architecture_signature_from_checkpoint(ckpt)
    if actual != expected:
        raise ValueError(f"resume checkpoint architecture mismatch: expected={expected}, actual={actual}, path={rel(path)}")
    state_dict = dict(ckpt["state_dict"])
    initialized_owner_residual_gate = False
    model_state = model.state_dict() if hasattr(model, "state_dict") else {}
    if "ownership_residual_gate" in model_state and "ownership_residual_gate" not in state_dict:
        state_dict["ownership_residual_gate"] = torch.zeros_like(model_state["ownership_residual_gate"])
        initialized_owner_residual_gate = True
    migrating_family_seed_head = (
        getattr(args, "geometry_decoder_mode", "legacy_debug") == "geometry_v2"
        and (bool(getattr(args, "content_seeded_queries", False)) or bool(getattr(args, "component_seeded_queries", False)))
        and "family_seed_head.weight" in model_state
        and "family_seed_head.weight" not in state_dict
    )
    migrating_component_seed_head = (
        bool(getattr(args, "component_seeded_queries", False))
        and "component_seed_head.0.weight" in model_state
        and "component_seed_head.0.weight" not in state_dict
    )
    if migrating_family_seed_head or migrating_component_seed_head:
        incompatible = model.load_state_dict(state_dict, strict=False)
        allowed_prefixes = ()
        if migrating_family_seed_head:
            allowed_prefixes += ("family_seed_head.",)
        if migrating_component_seed_head:
            allowed_prefixes += ("component_seed_head.",)
        forbidden_missing = [key for key in incompatible.missing_keys if not key.startswith(allowed_prefixes)]
        if incompatible.unexpected_keys or forbidden_missing:
            raise ValueError(
                "resume checkpoint migration has incompatible state: "
                f"missing={forbidden_missing}, unexpected={list(incompatible.unexpected_keys)}"
            )
    else:
        model.load_state_dict(state_dict, strict=True)
    optimizer_loaded = False
    scheduler_loaded = False
    expected_objective = objective_signature_from_args(args)
    actual_objective = objective_signature_from_checkpoint(ckpt)
    expected_objective_config = objective_config_from_args(args)
    expected_objective_hash = objective_config_hash(expected_objective_config)
    actual_objective_hash = ckpt.get("objective_config_hash")
    objective_match = isinstance(actual_objective_hash, str) and actual_objective_hash == expected_objective_hash
    if (
        args.resume_optimizer
        and ckpt.get("optimizer_state_dict") is not None
        and not objective_match
        and args.training_preset == "production"
    ):
        raise ValueError(
            "production optimizer resume requires an exact objective match; "
            "use --init-checkpoint for weights-only initialization"
        )
    if args.resume_optimizer and ckpt.get("optimizer_state_dict") is not None and not migrating_family_seed_head:
        allow_objective_override = bool(args.allow_optimizer_objective_mismatch and args.training_preset != "production")
        if objective_match or allow_objective_override:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            optimizer_loaded = True
            if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                scheduler_loaded = True
    runtime_model = getattr(model, "_orig_mod", model)
    source_epoch = int(ckpt.get("epoch", 0) or 0)
    routed_parameter_prefixes = (
        "sparse_router.", "sparse_experts.", "sparse_router_norm.", "branch_routers.",
        "bridge_gate", "semantic_query_residual_gate", "sq_mask_residual_gate",
        "sq_mask_residual_projection.", "sq_ownership_residual_gate", "ownership_residual_gate",
        "family_seed_head.", "component_seed_head.", "thing_seed_score.",
        "route_family_head.", "route_family_embed.", "route_token_gate.",
        "route_token_norm.", "route_residual_logit_gate",
    )
    route_residual_prefixes = ("route_token_gate.", "route_token_norm.", "route_residual_logit_gate")
    route_residual_may_be_inactive = source_epoch < int(getattr(args, "route_conditioning_enable_after_epoch", 2))
    named_parameters = list(runtime_model.named_parameters()) if hasattr(runtime_model, "named_parameters") else []
    router_parameter_items = [
        (name, parameter) for name, parameter in named_parameters
        if parameter.requires_grad and name.startswith(routed_parameter_prefixes)
    ]
    missing_router_optimizer_state = [
        name for name, parameter in router_parameter_items
        if not bool(optimizer.state.get(parameter))
        and not (route_residual_may_be_inactive and name.startswith(route_residual_prefixes))
    ]
    router_optimizer_states = len(router_parameter_items) - len(missing_router_optimizer_state)
    if optimizer_loaded and missing_router_optimizer_state:
        raise ValueError(
            "ABI-v7 optimizer resume lacks router/expert state; optimizer resume lacks routed/gated state: "
            f"{router_optimizer_states}/{len(router_parameter_items)} missing={missing_router_optimizer_state[:8]}"
        )
    training_state_compatible = bool(objective_match and not migrating_family_seed_head and not migrating_component_seed_head)
    history = (
        ckpt.get("history")
        if training_state_compatible and isinstance(ckpt.get("history"), list)
        else []
    )
    intra_epoch_progress = (
        ckpt.get("intra_epoch_progress")
        if training_state_compatible and isinstance(ckpt.get("intra_epoch_progress"), dict)
        else None
    )
    resume_intra_epoch = bool(
        training_state_compatible
        and intra_epoch_progress
        and ckpt.get("checkpoint_boundary") == "mid_epoch_progress_checkpoint"
    )
    start_epoch = (source_epoch if resume_intra_epoch else source_epoch + 1) if training_state_compatible else 1
    return {
        "enabled": True,
        "path": rel(path),
        "schema_version": ckpt.get("schema_version"),
        "checkpoint_boundary": ckpt.get("checkpoint_boundary"),
        "source_epoch": source_epoch,
        "start_epoch": start_epoch,
        "resume_intra_epoch": resume_intra_epoch,
        "resume_skip_records": int((intra_epoch_progress or {}).get("records_completed", 0) or 0),
        "intra_epoch_progress": intra_epoch_progress,
        "rng_state": ckpt.get("rng_state") if training_state_compatible else None,
        "training_rng_state": ckpt.get("training_rng_state") if training_state_compatible else None,
        "migration": "legacy_weights_plus_new_random_family_seed_head" if migrating_family_seed_head else (
            "legacy_weights_plus_new_random_component_seed_head" if migrating_component_seed_head else
            "legacy_ownership_weights_plus_zero_owner_residual_gate" if initialized_owner_residual_gate else None
        ),
        "history": history,
        "best_score": (
            float(ckpt.get("best_selection_score", ckpt.get("selection_score", -float("inf"))))
            if training_state_compatible else -float("inf")
        ),
        "best_val_semantic": (
            float(ckpt.get("best_val_semantic_token_accuracy", -1.0))
            if training_state_compatible else -1.0
        ),
        "best_checkpoint_written": bool(ckpt.get("best_checkpoint_written", False)) if training_state_compatible else False,
        "run_id": ckpt.get("run_id") if training_state_compatible else None,
        "sq_rq_auto_fused": bool(
            training_state_compatible
            and (checkpoint_abi.get("sq_rq_deployment") or {}).get("auto_fused", False)
        ),
        "sq_rq_auto_fuse_reason": (
            (checkpoint_abi.get("sq_rq_deployment") or {}).get("auto_fuse_reason")
            if training_state_compatible else None
        ),
        "optimizer_loaded": optimizer_loaded,
        "scheduler_loaded": scheduler_loaded,
        "scheduler_resume_missing": bool(optimizer_loaded and scheduler is not None and not scheduler_loaded),
        "router_optimizer_parameter_count": len(router_parameter_items),
        "router_optimizer_state_count": router_optimizer_states,
        "router_optimizer_state_complete": bool(router_parameter_items) and not missing_router_optimizer_state,
        "router_optimizer_state_missing_allowed_by_warmup": bool(route_residual_may_be_inactive),
        "optimizer_objective_match": objective_match,
        "training_state_objective_match": training_state_compatible,
        "training_state_reset_for_objective_mismatch": not training_state_compatible,
        "optimizer_objective_mismatch_blocked": bool(
            args.resume_optimizer
            and not objective_match
            and not (args.allow_optimizer_objective_mismatch and args.training_preset != "production")
        ),
        "checkpoint_objective_signature": actual_objective,
        "requested_objective_signature": expected_objective,
        "checkpoint_objective_config_hash": actual_objective_hash,
        "requested_objective_config_hash": expected_objective_hash,
        "legacy_checkpoint_missing_objective_hash": not isinstance(actual_objective_hash, str),
        "strict_architecture_match": True,
    }


def bounded_sq_rq_gradient_scale(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 0.1:
        raise argparse.ArgumentTypeError("SQ<-RQ gradient scale must be in [0, 0.1]")
    return parsed


def scheduled_auxiliary_scale(epoch: int, enable_after_epoch: int, warmup_epochs: int) -> float:
    start_epoch = max(1, int(enable_after_epoch))
    if int(epoch) < start_epoch:
        return 0.0
    warmup = max(1, int(warmup_epochs))
    return min(1.0, max(0.0, (int(epoch) - start_epoch + 1) / float(warmup)))


def style_feature_dropout_mask(torch: Any, batch_size: int, feature_dim: int, probability: float, device: Any) -> Any:
    """Sample one shared style-channel mask for primitive and segment inputs."""
    if not 0.0 <= float(probability) < 1.0:
        raise ValueError("style_feature_dropout must be in [0, 1)")
    mask = torch.ones((batch_size, 1, feature_dim), dtype=torch.float32, device=device)
    if probability <= 0.0:
        return mask
    style_indices = [index for index in (11, 12, 13, 14, 15) if index < feature_dim]
    if style_indices:
        keep = torch.rand((batch_size, len(style_indices)), device=device) >= float(probability)
        mask[:, 0, style_indices] = keep.to(mask.dtype)
    return mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--last-model-output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--resume-optimizer", action="store_true", help="Restore optimizer state from --resume-checkpoint when present.")
    parser.add_argument("--allow-optimizer-objective-mismatch", action="store_true", help="Dangerous override: restore optimizer even when the checkpoint objective differs.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--training-preset", choices=["custom", "dev", "production"], default="custom")
    parser.add_argument("--gradient-control", choices=["sum", "pcgrad"], default="sum", help="Shared-encoder multi-task gradient aggregation. Production forces deterministic PCGrad.")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--num-queries", type=int, default=96)
    parser.add_argument("--query-decoder-layers", type=int, default=1)
    parser.add_argument("--geometry-decoder-mode", choices=["legacy_debug", "geometry_v2"], default="legacy_debug")
    parser.add_argument("--num-stuff-queries", type=int, default=32)
    parser.add_argument("--typed-stuff-slots", action=argparse.BooleanOptionalAction, default=False,
                        help="Use one hard class-specific query slot for each FloorPlanCAD stuff label 30..34.")
    parser.add_argument("--geometry-local-neighbors", type=int, default=4)
    parser.add_argument("--geometry-coarse-grid-size", type=int, default=4)
    parser.add_argument("--geometry-attention-tile-size", type=int, default=0,
                        help="Exact streaming memory tile for geometry-v2 cross-attention; 0 keeps dense attention.")
    parser.add_argument("--tensor-ring-rank", type=int, default=0,
                        help="Enable tensor-ring factorized geometry projections with this ring rank; 0 keeps dense Linear.")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False,
                        help="Checkpoint encoder layers during training to bound activation memory.")
    parser.add_argument("--sq-rq-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sq-rq-gradient-scale", type=bounded_sq_rq_gradient_scale, default=0.0, help="RQ gradient into SQ cross-attention; must remain in [0, 0.1].")
    parser.add_argument("--sq-rq-query-confidence-threshold", type=float, default=0.6)
    parser.add_argument("--sq-rq-token-membership-threshold", type=float, default=0.5)
    parser.add_argument(
        "--sq-rq-training-membership-temperature", type=float, default=0.1,
        help="Soft membership temperature used only while training; 0 restores hard train/eval gating.",
    )
    parser.add_argument("--sq-rq-coverage-warmup-epochs", type=int, default=0)
    parser.add_argument("--sq-rq-warmup-query-confidence-threshold", type=float, default=0.2)
    parser.add_argument("--sq-rq-warmup-token-membership-threshold", type=float, default=0.2)
    parser.add_argument("--sq-rq-min-admitted-query-coverage", type=float, default=0.0)
    parser.add_argument("--sq-rq-min-context-edge-coverage", type=float, default=0.0)
    parser.add_argument("--learned-sparse-router", action=argparse.BooleanOptionalAction, default=False, help="Enable ABI-v7 prediction-only top-k token routing. Disabled preserves v6 architecture.")
    parser.add_argument("--router-num-experts", type=int, default=4)
    parser.add_argument("--router-top-k", type=int, default=2)
    parser.add_argument("--router-temperature", type=float, default=1.0)
    parser.add_argument("--typed-branch-routers", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable ABI-v8 semantic/RQ/SQ/bridge routers; disabled preserves shared-router behavior.")
    parser.add_argument("--branch-num-experts", type=int, default=2)
    parser.add_argument("--branch-top-k", type=int, default=1)
    parser.add_argument("--branch-capacity-factor", type=float, default=1.25)
    parser.add_argument("--branch-dropless", action=argparse.BooleanOptionalAction, default=False,
                        help="Do not discard branch-expert assignments when a capacity estimate is exceeded.")
    parser.add_argument("--router-load-balance-loss-weight", type=float, default=0.01)
    parser.add_argument("--router-z-loss-weight", type=float, default=0.001)
    parser.add_argument("--router-collapse-warmup-epochs", type=int, default=2)
    parser.add_argument("--router-max-dominant-expert-probability", type=float, default=0.80)
    parser.add_argument("--router-min-expert-assignment-fraction", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--style-feature-dropout", type=float, default=0.0,
                        help="Drop non-geometric style channels consistently from primitive and segment inputs during training.")
    parser.add_argument("--semantic-label-smoothing", type=float, default=0.0)
    parser.add_argument("--query-label-smoothing", type=float, default=0.0)
    parser.add_argument("--max-tokens-per-record", type=int, default=2048)
    parser.add_argument("--bottleneck-ledger", type=Path, default=DEFAULT_BOTTLENECK_LEDGER)
    parser.add_argument("--disable-bottleneck-weights", action="store_true")
    parser.add_argument("--no-object-query-weight", type=float, default=0.1)
    parser.add_argument("--recall-class-weight", type=float, default=2.5)
    parser.add_argument("--precision-class-weight", type=float, default=0.8)
    parser.add_argument("--grouping-class-weight", type=float, default=1.5)
    parser.add_argument("--extra-recall-labels", default="")
    parser.add_argument("--extra-grouping-labels", default="")
    parser.add_argument("--extra-precision-labels", default="")
    parser.add_argument("--small-component-size", type=int, default=3)
    parser.add_argument("--semantic-loss-weight", type=float, default=1.0)
    parser.add_argument("--query-loss-weight", type=float, default=1.0)
    parser.add_argument("--query-objectness-loss-weight", type=float, default=0.25)
    parser.add_argument("--query-objectness-positive-weight", type=float, default=2.0)
    parser.add_argument("--query-objectness-negative-weight", type=float, default=1.0)
    parser.add_argument("--query-objectness-positive-margin-floor-loss-weight", type=float, default=0.0)
    parser.add_argument("--query-objectness-negative-margin-ceiling-loss-weight", type=float, default=0.0)
    parser.add_argument("--rq-sq-quality-calibration-loss-weight", type=float, default=0.0)
    parser.add_argument("--rq-sq-quality-ranking-weight", type=float, default=0.25)
    parser.add_argument("--rq-sq-quality-ranking-margin", type=float, default=0.05)
    parser.add_argument("--rq-sq-quality-ranking-top-k", type=int, default=1)
    parser.add_argument("--rq-sq-quality-hard-negative-weight", type=float, default=0.1)
    parser.add_argument("--rq-sq-quality-unmatched-ceiling-weight", type=float, default=0.0)
    parser.add_argument("--rq-sq-quality-unmatched-ceiling-probability", type=float, default=0.05)
    parser.add_argument(
        "--family-recall-focus",
        choices=["", *FAMILY_NAMES],
        default="",
        help="Optional family-specific recall curriculum; use furniture only after high-RQ baseline is stable.",
    )
    parser.add_argument("--family-recall-loss-weight", type=float, default=0.0)
    parser.add_argument("--family-recall-admission-floor", type=float, default=0.25)
    parser.add_argument("--family-recall-mask-prob-floor", type=float, default=0.35)
    parser.add_argument("--family-recall-quality-floor", type=float, default=0.10)
    parser.add_argument(
        "--active-loss-experts",
        default="joint_routed",
        help="Comma-separated loss experts to optimize, or joint_routed for all: "
             "sq_semantic,rq_admission,mask_shape,quality_deployment,route_control,topology_merge,teacher_aux.",
    )
    parser.add_argument("--hard-recall-labels", default="")
    parser.add_argument(
        "--hard-recall-families",
        default="",
        help="Comma-separated families whose labels are added to --hard-recall-labels, e.g. furniture.",
    )
    parser.add_argument("--rq-admission-expert-weight", type=float, default=0.0)
    parser.add_argument("--mask-recall-expert-weight", type=float, default=0.0)
    parser.add_argument("--quality-deployment-expert-weight", type=float, default=0.0)
    parser.add_argument("--hard-recall-admission-floor", type=float, default=0.55)
    parser.add_argument("--hard-recall-mask-prob-floor", type=float, default=0.45)
    parser.add_argument("--hard-recall-deployment-floor", type=float, default=0.22)
    parser.add_argument("--hard-recall-quality-target-floor", type=float, default=0.20)
    parser.add_argument(
        "--quality-calibration-only",
        action="store_true",
        help="Freeze every parameter except query_quality_head for weights-only post-training calibration.",
    )
    parser.add_argument(
        "--deployment-min-query-score",
        type=float,
        default=DEFAULT_MIN_QUERY_SCORE,
        help="Class×quality admission threshold shared with the production apply decoder.",
    )
    parser.add_argument(
        "--deployment-mask-threshold",
        type=float,
        default=PANOPTIC_QUALITY_MASK_THRESHOLD,
        help="Mask probability threshold shared by quality targets, validation proxy, and apply decoding.",
    )
    parser.add_argument("--stuff-overlap-union-loss-weight", type=float, default=0.1)
    parser.add_argument("--semantic-query-residual-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--content-seeded-queries",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Experimental hard content seeding; disabled because ranking-only seed scores lack dense supervision.",
    )
    parser.add_argument("--component-seeded-queries", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--repeated-group-fusion", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--relation-bias-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--offset-vote-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--candidate-aware-queries", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--candidate-feature-dim", type=int, default=0)
    parser.add_argument("--candidate-proposals", type=Path, default=None)
    parser.add_argument("--val-candidate-proposals", type=Path, default=None)
    parser.add_argument("--max-candidate-queries", type=int, default=0)
    parser.add_argument("--candidate-mask-prior-logit", type=float, default=0.0)
    parser.add_argument("--candidate-mask-prior-loss-weight", type=float, default=0.0)
    parser.add_argument("--candidate-ablation-tag", default="")
    parser.add_argument("--weak-family-feature-fusion", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--explicit-route-classifier",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add supervised token-family route classification and route-conditioned token residuals.",
    )
    parser.add_argument("--route-classification-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--route-conditioning-residual-scale",
        type=float,
        default=0.10,
        help="Maximum residual scale for route-conditioned token features before learned gate and epoch warmup.",
    )
    parser.add_argument("--route-conditioning-enable-after-epoch", type=int, default=2)
    parser.add_argument("--route-conditioning-warmup-epochs", type=int, default=3)
    parser.add_argument(
        "--dense-attention-feature-adapter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply local-window multi-head self-attention before the main encoder for dense CAD feature mixing.",
    )
    parser.add_argument("--dense-attention-window-size", type=int, default=128)
    parser.add_argument(
        "--dense-attention-adapter-residual-scale",
        type=float,
        default=0.10,
        help="Maximum residual scale for the dense feature adapter before learned gate and epoch warmup.",
    )
    parser.add_argument("--dense-attention-adapter-enable-after-epoch", type=int, default=2)
    parser.add_argument("--dense-attention-adapter-warmup-epochs", type=int, default=3)
    parser.add_argument(
        "--quality-query-gradient-scale",
        type=float,
        default=0.0,
        help="Scale quality/ranking gradients into query representations; 0 preserves detached quality-head behavior.",
    )
    parser.add_argument("--component-seed-loss-weight", type=float, default=0.0)
    parser.add_argument("--offset-vote-loss-weight", type=float, default=0.0)
    parser.add_argument("--affinity-loss-weight", type=float, default=0.0)
    parser.add_argument("--allow-sq-rq-never-enabled-smoke", action="store_true")
    parser.add_argument("--quality-soft-target-weight", type=float, default=0.5)
    parser.add_argument(
        "--sq-rq-enable-after-epoch", type=int, default=5,
        help="Keep SQ-RQ cross-attention out of the main prediction path until this one-based epoch.",
    )
    parser.add_argument("--sq-rq-base-semantic-loss-weight", type=float, default=0.25)
    parser.add_argument("--sq-rq-auto-fuse", action=argparse.BooleanOptionalAction, default=True,
                        help="Fall back to the base semantic branch after validated SQ-RQ semantic regression.")
    parser.add_argument("--sq-rq-max-semantic-loss-regression", type=float, default=0.001,
                        help="Maximum validation semantic loss increase over the SQ base before auto-fuse.")
    parser.add_argument("--identity-loss-weight", type=float, default=0.25)
    parser.add_argument("--ownership-loss-weight", type=float, default=1.0)
    parser.add_argument("--ownership-mask-consistency-loss-weight", type=float, default=0.25)
    parser.add_argument("--identity-temperature", type=float, default=0.1)
    parser.add_argument("--identity-negative-margin", type=float, default=0.25)
    parser.add_argument("--objectness-positive-margin-floor", type=float, default=0.75)
    parser.add_argument("--objectness-negative-margin-ceiling", type=float, default=-0.25)
    parser.add_argument("--objectness-warmup-epochs", type=int, default=0)
    parser.add_argument("--objectness-warmup-loss-multiplier", type=float, default=4.0)
    parser.add_argument("--objectness-warmup-positive-multiplier", type=float, default=4.0)
    parser.add_argument("--objectness-warmup-negative-multiplier", type=float, default=0.25)
    parser.add_argument("--objectness-warmup-positive-margin-floor-loss-weight", type=float, default=0.0)
    parser.add_argument("--objectness-warmup-negative-margin-ceiling-loss-weight", type=float, default=0.0)
    parser.add_argument("--objectness-precision-phase-start-epoch", type=int, default=0)
    parser.add_argument("--objectness-precision-phase-loss-weight", type=float, default=0.35)
    parser.add_argument("--objectness-precision-phase-positive-weight", type=float, default=1.25)
    parser.add_argument("--objectness-precision-phase-negative-weight", type=float, default=2.0)
    parser.add_argument("--objectness-precision-phase-positive-margin-floor-loss-weight", type=float, default=0.0)
    parser.add_argument("--objectness-precision-phase-negative-margin-ceiling-loss-weight", type=float, default=0.0)
    parser.add_argument("--precision-phase-require-healthy-admission-epochs", type=int, default=0)
    parser.add_argument("--precision-phase-min-object-recall", type=float, default=0.20)
    parser.add_argument("--precision-phase-max-negative-margin-rate", type=float, default=0.20)
    parser.add_argument("--precision-phase-min-sq-proxy", type=float, default=0.20)
    parser.add_argument("--precision-phase-min-calibrated-rq", type=float, default=0.05)
    parser.add_argument(
        "--precision-phase-min-calibrated-proposal-coverage", type=float, default=0.005
    )
    parser.add_argument(
        "--precision-phase-transition-epochs",
        type=int,
        default=8,
        help="Blend precision objectives in or out over this many epochs instead of switching loss weights abruptly.",
    )
    parser.add_argument("--zero-admission-patience-epochs", type=int, default=0)
    parser.add_argument("--zero-admission-min-epoch", type=int, default=5)
    parser.add_argument("--min-val-object-recall-for-checkpoint", type=float, default=0.0)
    parser.add_argument("--min-val-mask-recall-for-checkpoint", type=float, default=0.0)
    parser.add_argument("--min-val-mask-precision-for-checkpoint", type=float, default=0.0)
    parser.add_argument("--max-val-positive-rate-ratio-for-checkpoint", type=float, default=0.0)
    parser.add_argument("--min-val-positive-object-margin-rate-for-checkpoint", type=float, default=0.0)
    parser.add_argument("--min-val-positive-object-margin-mean-for-checkpoint", type=float, default=-float("inf"))
    parser.add_argument("--max-val-negative-object-margin-rate-for-checkpoint", type=float, default=0.0)
    parser.add_argument("--min-val-rq-proxy-for-checkpoint", type=float, default=0.0)
    parser.add_argument("--min-val-sq-proxy-for-checkpoint", type=float, default=0.0)
    parser.add_argument("--min-val-instance-tp-for-checkpoint", type=int, default=0)
    parser.add_argument("--min-val-calibrated-proposal-coverage-for-checkpoint", type=float, default=0.0)
    parser.add_argument(
        "--max-val-unmatched-quality-for-checkpoint",
        type=float,
        default=1.0,
        help="Maximum mean unmatched class×quality deployment score for checkpoint promotion.",
    )
    parser.add_argument("--max-val-quality-ranking-violation-rate-for-checkpoint", type=float, default=1.0)
    parser.add_argument(
        "--require-final-instance-gate-for-best",
        action="store_true",
        help="Require an external eval-mode final-instance gate report to pass before writing --model-output.",
    )
    parser.add_argument(
        "--select-best-by-final-instance-pq",
        action="store_true",
        help="Rank gate-passing checkpoints by exact final-instance PQ instead of an internal proxy or loss.",
    )
    parser.add_argument("--final-instance-gate-report", type=Path, default=None)
    parser.add_argument(
        "--final-instance-gate-command-template",
        default=None,
        help="Per-epoch shell command with required {checkpoint}, {epoch}, and {report} placeholders.",
    )
    parser.add_argument("--final-instance-gate-protocol", default="floorplancad_official_line_json_primitive_index_v1")
    parser.add_argument("--final-instance-gate-timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--final-instance-gate-interval-epochs",
        type=int,
        default=1,
        help="Run the expensive exact final-instance gate every N epochs and on the final epoch.",
    )
    parser.add_argument("--final-instance-gate-checkpoint", type=Path, default=None)
    parser.add_argument("--min-final-instance-tp-for-best", type=int, default=1)
    parser.add_argument("--min-final-instance-rq-for-best", type=float, default=1e-9)
    parser.add_argument("--min-final-instance-sq-for-best", type=float, default=0.0)
    parser.add_argument("--min-final-instance-pq-for-best", type=float, default=0.0)
    parser.add_argument("--max-final-instance-fp-for-best", type=int, default=-1)
    parser.add_argument("--mask-loss-weight", type=float, default=2.0)
    parser.add_argument("--mask-geometry-connectivity-loss-weight", type=float, default=0.05)
    parser.add_argument(
        "--content-anchor-loss-weight",
        type=float,
        default=0.10,
        help="Auxiliary loss for content-seeded query family presence so RQ seeds use semantic evidence.",
    )
    parser.add_argument("--moe-branch-specialization-loss-weight", type=float, default=0.01)
    parser.add_argument(
        "--partial-component-policy",
        choices=["exclude", "window_visible"],
        default="exclude",
        help="Keep legacy exclusion or supervise page-partial components with their exact visible window mask.",
    )
    parser.add_argument(
        "--partial-component-min-tokens",
        type=int,
        default=2,
        help="Minimum visible tokens before a page-partial component is used with window_visible supervision.",
    )
    parser.add_argument("--mask-positive-weight", type=float, default=8.0)
    parser.add_argument("--mask-negative-weight", type=float, default=1.0)
    parser.add_argument("--mask-focal-gamma", type=float, default=1.5)
    parser.add_argument("--mask-area-ratio-loss-weight", type=float, default=0.0)
    parser.add_argument("--mask-area-overcoverage-weight", type=float, default=2.0)
    parser.add_argument("--mask-tversky-loss-weight", type=float, default=0.0)
    parser.add_argument("--mask-tversky-alpha", type=float, default=0.35)
    parser.add_argument("--mask-tversky-beta", type=float, default=0.65)
    parser.add_argument("--mask-positive-prob-floor-loss-weight", type=float, default=0.0)
    parser.add_argument("--mask-positive-prob-floor", type=float, default=0.45)
    parser.add_argument("--unmatched-mask-negative-loss-weight", type=float, default=0.10)
    parser.add_argument("--unmatched-mask-negative-top-k", type=int, default=16)
    parser.add_argument("--geometry-augmentation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--geometry-augmentation-scale-min", type=float, default=0.9)
    parser.add_argument("--geometry-augmentation-scale-max", type=float, default=1.1)
    parser.add_argument("--geometry-augmentation-translation", type=float, default=0.05)
    parser.add_argument("--mask-precision-phase-start-epoch", type=int, default=0)
    parser.add_argument("--mask-precision-phase-positive-weight", type=float, default=1.0)
    parser.add_argument("--mask-precision-phase-negative-weight", type=float, default=1.5)
    parser.add_argument("--mask-precision-phase-area-ratio-loss-weight", type=float, default=0.15)
    parser.add_argument("--mask-precision-phase-area-overcoverage-weight", type=float, default=2.0)
    parser.add_argument("--mask-precision-phase-tversky-loss-weight", type=float, default=0.25)
    parser.add_argument("--mask-precision-phase-positive-prob-floor-loss-weight", type=float, default=0.10)
    parser.add_argument("--teacher-proposals", type=Path, default=None)
    parser.add_argument("--teacher-loss-weight", type=float, default=0.0)
    parser.add_argument("--teacher-mask-loss-weight", type=float, default=1.0)
    parser.add_argument("--teacher-query-loss-weight", type=float, default=0.5)
    parser.add_argument("--teacher-min-gt-iou", type=float, default=0.5)
    parser.add_argument("--teacher-allow-unmatched", action="store_true", help="Use teacher proposals even when the builder did not mark them as GT-positive.")
    parser.add_argument(
        "--checkpoint-metric",
        choices=[
            "neg_loss",
            "semantic_token_accuracy",
            "component_proxy_score",
            "recall_gated_component_proxy",
            "pq_aware_component_proxy",
            "admission_aware_component_proxy",
            "joint_rq_sq_proxy",
        ],
        default="neg_loss",
    )
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--lr-decay-steps", type=int, default=0,
                        help="Optimizer-step budget for warm-up plus cosine decay; 0 keeps legacy constant LR diagnostic mode.")
    parser.add_argument("--lr-min-scale", type=float, default=0.1)
    parser.add_argument("--router-lr-scale", type=float, default=0.5)
    parser.add_argument("--head-lr-scale", type=float, default=1.0)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--batch-records", type=int, default=1, help="Accumulate this many records before one backward/optimizer step.")
    parser.add_argument(
        "--auto-throughput-profile",
        choices=["off", "gpu_32gb_safe", "gpu0_high_memory", "gpu0_full_memory"],
        default="off",
        help="Apply a CUDA memory profile: cap for 32 GiB devices or raise throughput on large-memory devices.",
    )
    parser.add_argument("--auto-profile-min-total-mib", type=int, default=90000)
    parser.add_argument("--auto-profile-batch-records", type=int, default=24)
    parser.add_argument("--auto-profile-prefetch-records", type=int, default=128)
    parser.add_argument("--auto-profile-prefetch-workers", type=int, default=10)
    parser.add_argument("--auto-profile-progress-checkpoint-records", type=int, default=1024)
    parser.add_argument("--auto-profile-full-free-mib", type=int, default=90000)
    parser.add_argument("--auto-profile-full-batch-records", type=int, default=96)
    parser.add_argument("--auto-profile-full-prefetch-records", type=int, default=768)
    parser.add_argument("--auto-profile-full-prefetch-workers", type=int, default=24)
    parser.add_argument("--auto-profile-full-progress-checkpoint-records", type=int, default=4096)
    parser.add_argument("--auto-profile-32gb-max-batch-records", type=int, default=1)
    parser.add_argument("--auto-profile-32gb-min-free-mib", type=int, default=30000,
                        help="Fail closed before model construction when a 24–40 GiB GPU lacks this much free memory.")
    parser.add_argument("--auto-profile-32gb-prefetch-records", type=int, default=32)
    parser.add_argument("--auto-profile-32gb-prefetch-workers", type=int, default=4)
    parser.add_argument("--auto-profile-32gb-geometry-attention-tile-size", type=int, default=128)
    parser.add_argument(
        "--train-prefetch-records",
        type=int,
        default=0,
        help="CPU records to parse ahead of the GPU training loop. Set 0 to disable prefetching.",
    )
    parser.add_argument(
        "--train-prefetch-workers",
        type=int,
        default=1,
        help="Parallel CPU parser workers used when --train-prefetch-records is enabled. Output order stays deterministic.",
    )
    parser.add_argument(
        "--amp",
        choices=["off", "bf16", "fp16"],
        default="off",
        help="CUDA mixed precision mode. bf16 is preferred on RTX 50/Blackwell; off preserves FP32 behavior.",
    )
    parser.add_argument("--disable-tf32", action="store_true", help="Disable CUDA TF32 matmul/cudnn acceleration.")
    parser.add_argument(
        "--enable-cudnn-benchmark",
        action="store_true",
        help="Enable cudnn.benchmark for long fixed-shape-ish CUDA runs after warmup.",
    )
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="Wrap the model with torch.compile for subsequent long runs. Resume checkpoints remain normal state_dict checkpoints.",
    )
    parser.add_argument("--compile-mode", default="reduce-overhead", help="torch.compile mode used when --compile-model is set.")
    parser.add_argument(
        "--progress-checkpoint-records",
        type=int,
        default=512,
        help="Write --last-model-output every N processed records inside an epoch. Set 0 to disable mid-epoch checkpoints.",
    )
    parser.add_argument(
        "--progress-checkpoint-seconds",
        type=int,
        default=900,
        help="Write --last-model-output at least every N seconds inside an epoch. Set 0 to disable time-based checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-archive-dir",
        type=Path,
        default=None,
        help="Optional directory for timestamped mid-epoch checkpoint copies in addition to --last-model-output.",
    )
    parser.add_argument(
        "--checkpoint-archive-keep",
        type=int,
        default=0,
        help="Keep the newest K archived mid-epoch checkpoints. Set 0 to disable archive retention.",
    )
    parser.add_argument(
        "--diagnostic-checkpoint-dir",
        type=Path,
        default=None,
        help="Directory for top-K epoch checkpoints ranked before promotion gates.",
    )
    parser.add_argument("--diagnostic-checkpoint-top-k", type=int, default=3)
    parser.add_argument(
        "--progress-status-records",
        type=int,
        default=128,
        help="Rewrite --report every N processed records inside an epoch so watchdogs can distinguish slow epochs from stalls. Set 0 to disable.",
    )
    parser.add_argument(
        "--progress-status-seconds",
        type=int,
        default=120,
        help="Rewrite --report at least every N seconds inside an epoch. Set 0 to disable time-based status refresh.",
    )
    parser.add_argument(
        "--component-matching",
        choices=["hungarian_cpu", "greedy_gpu", "greedy_gpu_train"],
        default="hungarian_cpu",
        help="Component query assignment. greedy_gpu is fail-closed unless its per-batch cost gap versus exact Hungarian is below 1%%.",
    )
    parser.add_argument(
        "--train-component-matching",
        choices=["hungarian_cpu", "greedy_gpu", "greedy_gpu_train"],
        default=None,
        help="Optional training-only matcher; gpu_32gb_safe defaults to greedy_gpu_train while validation keeps exact --component-matching.",
    )
    parser.add_argument("--train-matching-exact-audit-interval", type=int, default=64)
    parser.add_argument("--train-matching-max-assignment-churn", type=float, default=0.15)
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument(
        "--val-limit-records",
        type=int,
        default=0,
        help="Validation window cap; 0 evaluates the full validation cache. Nonzero caps are diagnostic only.",
    )
    parser.add_argument(
        "--val-limit-pages",
        type=int,
        default=0,
        help="Deterministic page/thing/stuff-stratified validation subset; 0 evaluates all pages. Diagnostic only.",
    )
    parser.add_argument(
        "--require-target-schema-v3", action=argparse.BooleanOptionalAction, default=True,
        help="Require lossless segment v3 targets for new training; disable only for read-only legacy diagnosis.",
    )
    parser.add_argument(
        "--input-feature-schema",
        choices=["v3", "v4", "v5", "v6"],
        default="v3",
        help="Select the locked primitive/segment feature contract used by training and checkpoint ABI.",
    )
    parser.add_argument(
        "--require-target-schema-v4", action=argparse.BooleanOptionalAction, default=False,
        help="Require raw-semantic V4 targets for new training.",
    )
    parser.add_argument(
        "--require-target-schema-v5", action=argparse.BooleanOptionalAction, default=False,
        help="Require structural V5 targets for new training.",
    )
    parser.add_argument(
        "--require-target-schema-v6", action=argparse.BooleanOptionalAction, default=False,
        help="Require structural-relation V6 targets for new training.",
    )
    parser.add_argument(
        "--allow-output-protocol-name-mismatch",
        action="store_true",
        help="Dangerous legacy-audit override: allow output/report paths to claim a newer protocol than --input-feature-schema.",
    )
    parser.add_argument("--seed", type=int, default=20260630)
    return parser.parse_args()


def apply_training_preset(args: argparse.Namespace) -> dict[str, Any]:
    if args.training_preset == "custom":
        return {"name": "custom", "fail_closed": False, "diagnostic_only": True, "overrides": {}}
    if args.training_preset not in {"dev", "production"}:
        raise ValueError(f"unsupported training preset: {args.training_preset}")
    if int(getattr(args, "limit_records", 0)) > 0:
        raise ValueError(
            "dev/production presets forbid deterministic JSONL prefixes; materialize a page-stratified "
            "development subset and train it with --limit-records 0"
        )
    input_feature_schema = input_feature_schema_from_args(args)
    required = {
        "geometry_decoder_mode": "geometry_v2",
        "input_feature_schema": input_feature_schema,
        "require_target_schema_v3": input_feature_schema == "v3",
        "require_target_schema_v4": input_feature_schema == "v4",
        "require_target_schema_v5": input_feature_schema == "v5",
        "require_target_schema_v6": input_feature_schema == "v6",
        "hidden_dim": max(int(getattr(args, "hidden_dim", 256)), 256),
        "layers": max(int(getattr(args, "layers", 4)), 4),
        "heads": max(int(getattr(args, "heads", 8)), 8),
        "num_queries": max(int(getattr(args, "num_queries", 0)), 256),
        "query_decoder_layers": 1,
        "num_stuff_queries": len(STUFF_LABELS),
        "typed_stuff_slots": True,
        "learned_sparse_router": True,
        "typed_branch_routers": False,
        "moe_branch_specialization_loss_weight": 0.0,
        "semantic_query_residual_enabled": True,
        "content_seeded_queries": False,
        "component_seeded_queries": False,
        "repeated_group_fusion": False,
        "relation_bias_enabled": False,
        "offset_vote_enabled": False,
        "candidate_aware_queries": False,
        "candidate_feature_dim": 0,
        "candidate_proposals": None,
        "val_candidate_proposals": None,
        "max_candidate_queries": 0,
        "candidate_mask_prior_logit": 0.0,
        "candidate_mask_prior_loss_weight": 0.0,
        "candidate_ablation_tag": "",
        "weak_family_feature_fusion": False,
        "quality_query_gradient_scale": 0.0,
        "explicit_route_classifier": bool(getattr(args, "explicit_route_classifier", False)),
        "route_classification_loss_weight": float(getattr(args, "route_classification_loss_weight", 0.0)),
        "route_conditioning_residual_scale": float(getattr(args, "route_conditioning_residual_scale", 0.10)),
        "route_conditioning_enable_after_epoch": max(int(getattr(args, "route_conditioning_enable_after_epoch", 2)), 1),
        "route_conditioning_warmup_epochs": max(int(getattr(args, "route_conditioning_warmup_epochs", 3)), 1),
        "dense_attention_feature_adapter": bool(getattr(args, "dense_attention_feature_adapter", False)),
        "dense_attention_window_size": max(int(getattr(args, "dense_attention_window_size", 128)), 1),
        "dense_attention_adapter_residual_scale": float(getattr(args, "dense_attention_adapter_residual_scale", 0.10)),
        "dense_attention_adapter_enable_after_epoch": max(int(getattr(args, "dense_attention_adapter_enable_after_epoch", 2)), 1),
        "dense_attention_adapter_warmup_epochs": max(int(getattr(args, "dense_attention_adapter_warmup_epochs", 3)), 1),
        "component_seed_loss_weight": 0.0,
        "offset_vote_loss_weight": 0.0,
        "affinity_loss_weight": 0.0,
        "quality_soft_target_weight": min(float(getattr(args, "quality_soft_target_weight", 0.5)), 0.5),
        "geometry_augmentation": False,
        "tensor_ring_rank": 0,
        "component_matching": "hungarian_cpu",
        "train_component_matching": "hungarian_cpu" if args.training_preset == "dev" else "greedy_gpu_train",
        "checkpoint_metric": "joint_rq_sq_proxy",
        "min_val_rq_proxy_for_checkpoint": max(float(args.min_val_rq_proxy_for_checkpoint), 0.10),
        "min_val_sq_proxy_for_checkpoint": max(float(args.min_val_sq_proxy_for_checkpoint), 0.20),
        "max_val_negative_object_margin_rate_for_checkpoint": (
            min(float(args.max_val_negative_object_margin_rate_for_checkpoint), 0.20)
            if float(args.max_val_negative_object_margin_rate_for_checkpoint) > 0.0
            else 0.20
        ),
        "precision_phase_require_healthy_admission_epochs": max(
            int(args.precision_phase_require_healthy_admission_epochs), 2
        ),
        "precision_phase_min_sq_proxy": max(float(getattr(args, "precision_phase_min_sq_proxy", 0.20)), 0.20),
        "precision_phase_min_calibrated_rq": max(
            float(getattr(args, "precision_phase_min_calibrated_rq", 0.05)), 0.05
        ),
        "precision_phase_min_calibrated_proposal_coverage": max(
            float(
                getattr(args, "precision_phase_min_calibrated_proposal_coverage", 0.005)
            ),
            0.005,
        ),
        "precision_phase_transition_epochs": max(int(getattr(args, "precision_phase_transition_epochs", 8)), 8),
        "objectness_precision_phase_start_epoch": max(int(args.objectness_precision_phase_start_epoch), 0),
        "mask_precision_phase_start_epoch": max(int(args.mask_precision_phase_start_epoch), 0),
        "mask_precision_phase_positive_weight": max(
            float(getattr(args, "mask_precision_phase_positive_weight", 1.0)), 4.0
        ),
        "mask_tversky_alpha": min(float(getattr(args, "mask_tversky_alpha", 0.35)), 0.35),
        "mask_tversky_beta": max(float(getattr(args, "mask_tversky_beta", 0.65)), 0.65),
        "unmatched_mask_negative_loss_weight": max(
            float(getattr(args, "unmatched_mask_negative_loss_weight", 0.0)), 0.10
        ),
        "unmatched_mask_negative_top_k": max(int(getattr(args, "unmatched_mask_negative_top_k", 0)), 16),
        "require_final_instance_gate_for_best": args.training_preset == "production",
        "min_final_instance_tp_for_best": max(int(args.min_final_instance_tp_for_best), 8),
        "min_final_instance_rq_for_best": max(float(args.min_final_instance_rq_for_best), 5.0),
        "min_final_instance_sq_for_best": max(float(getattr(args, "min_final_instance_sq_for_best", 0.0)), 70.0),
        "min_final_instance_pq_for_best": max(float(getattr(args, "min_final_instance_pq_for_best", 0.0)), 1.0),
        "max_final_instance_fp_for_best": (
            min(int(getattr(args, "max_final_instance_fp_for_best", -1)), 512)
            if int(getattr(args, "max_final_instance_fp_for_best", -1)) >= 0
            else 512
        ),
        "allow_optimizer_objective_mismatch": False,
        "batch_records": max(int(args.batch_records), 2),
        "identity_loss_weight": max(float(args.identity_loss_weight), 0.25),
        "rq_sq_quality_calibration_loss_weight": max(
            float(getattr(args, "rq_sq_quality_calibration_loss_weight", 0.0)), 0.25
        ),
        "rq_sq_quality_ranking_weight": max(float(getattr(args, "rq_sq_quality_ranking_weight", 0.0)), 0.25),
        "rq_sq_quality_ranking_margin": max(float(getattr(args, "rq_sq_quality_ranking_margin", 0.0)), 0.05),
        "rq_sq_quality_ranking_top_k": max(int(getattr(args, "rq_sq_quality_ranking_top_k", 1)), 4),
        "rq_sq_quality_hard_negative_weight": min(
            max(float(getattr(args, "rq_sq_quality_hard_negative_weight", 0.1)), 0.0), 0.1
        ),
        "rq_sq_quality_unmatched_ceiling_weight": min(
            max(float(getattr(args, "rq_sq_quality_unmatched_ceiling_weight", 0.0)), 0.0), 0.1
        ),
        "rq_sq_quality_unmatched_ceiling_probability": min(
            float(getattr(args, "rq_sq_quality_unmatched_ceiling_probability", 0.05)), 0.05
        ),
        "max_val_unmatched_quality_for_checkpoint": min(
            float(getattr(args, "max_val_unmatched_quality_for_checkpoint", 1.0)), 0.05
        ),
        "max_val_quality_ranking_violation_rate_for_checkpoint": min(
            float(getattr(args, "max_val_quality_ranking_violation_rate_for_checkpoint", 1.0)), 0.50
        ),
        "deployment_min_query_score": DEFAULT_MIN_QUERY_SCORE,
        "sq_rq_query_confidence_threshold": min(max(float(getattr(args, "sq_rq_query_confidence_threshold", 0.6)), 0.0), 1.0),
        "sq_rq_token_membership_threshold": min(max(float(getattr(args, "sq_rq_token_membership_threshold", 0.5)), 0.0), 1.0),
        "sq_rq_min_admitted_query_coverage": max(float(getattr(args, "sq_rq_min_admitted_query_coverage", 0.0)), 0.01),
        "sq_rq_min_context_edge_coverage": max(float(getattr(args, "sq_rq_min_context_edge_coverage", 0.0)), 0.002),
        "sq_rq_coverage_warmup_epochs": (
            0
            if bool(getattr(args, "quality_calibration_only", False))
            else max(int(getattr(args, "sq_rq_coverage_warmup_epochs", 0)), 3)
        ),
        "sq_rq_warmup_query_confidence_threshold": min(max(float(getattr(args, "sq_rq_warmup_query_confidence_threshold", 0.2)), 0.0), 0.4),
        "sq_rq_warmup_token_membership_threshold": min(max(float(getattr(args, "sq_rq_warmup_token_membership_threshold", 0.2)), 0.0), 0.4),
        "partial_component_policy": "exclude",
        "stuff_overlap_union_loss_weight": max(float(getattr(args, "stuff_overlap_union_loss_weight", 0.0)), 0.1),
        "content_anchor_loss_weight": 0.0,
        "sq_rq_enabled": True,
        "sq_rq_enable_after_epoch": 1,
        "sq_rq_auto_fuse": (
            False
            if bool(getattr(args, "quality_calibration_only", False))
            else bool(getattr(args, "sq_rq_auto_fuse", True))
        ),
        "sq_rq_max_semantic_loss_regression": min(
            max(float(getattr(args, "sq_rq_max_semantic_loss_regression", 0.001)), 0.0), 0.001
        ),
        "content_seeded_queries": False,
        "query_label_smoothing": 0.0,
        "sq_rq_gradient_scale": 0.0,
        "offset_vote_enabled": False,
        "offset_vote_loss_weight": 0.0,
        "affinity_loss_weight": 0.0,
        "objectness_warmup_epochs": max(int(getattr(args, "objectness_warmup_epochs", 0)), 3),
        "teacher_allow_unmatched": False,
        "gradient_control": "sum" if bool(getattr(args, "quality_calibration_only", False)) else "pcgrad",
        "lr_warmup_steps": max(int(getattr(args, "lr_warmup_steps", 0)), 50),
        "lr_min_scale": min(max(float(getattr(args, "lr_min_scale", 0.1)), 0.0), 0.5),
        "router_lr_scale": min(max(float(getattr(args, "router_lr_scale", 0.5)), 0.05), 1.0),
        "head_lr_scale": min(max(float(getattr(args, "head_lr_scale", 1.0)), 0.1), 2.0),
    }
    if args.training_preset == "production":
        required.update({
            "branch_dropless": False,
            "router_load_balance_loss_weight": max(float(getattr(args, "router_load_balance_loss_weight", 0.01)), 0.05),
            "router_z_loss_weight": max(float(getattr(args, "router_z_loss_weight", 0.001)), 0.001),
            "router_collapse_warmup_epochs": max(int(getattr(args, "router_collapse_warmup_epochs", 0)), 2),
            "router_max_dominant_expert_probability": min(float(getattr(args, "router_max_dominant_expert_probability", 0.80)), 0.80),
        "router_min_expert_assignment_fraction": max(float(getattr(args, "router_min_expert_assignment_fraction", 0.05)), 0.05),
        "style_feature_dropout": max(float(getattr(args, "style_feature_dropout", 0.0)), 0.10),
        "semantic_label_smoothing": max(float(getattr(args, "semantic_label_smoothing", 0.0)), 0.05),
        })
    overrides = {}
    for key, value in required.items():
        previous = getattr(args, key, None)
        setattr(args, key, value)
        if previous != value:
            overrides[key] = {"requested": previous, "effective": value}
    return {
        "name": args.training_preset,
        "fail_closed": True,
        "diagnostic_only": args.training_preset == "dev",
        "overrides": overrides,
    }


def resolve_training_step_budget(
    args: argparse.Namespace,
    training_preset: dict[str, Any],
) -> dict[str, Any]:
    steps_per_epoch = planned_page_aware_optimizer_steps(
        args.train,
        batch_records=int(args.batch_records),
        limit_records=int(args.limit_records),
    )
    planned_total_steps = steps_per_epoch * max(int(args.epochs), 0)
    requested_decay_steps = int(args.lr_decay_steps)
    auto_resolved = training_preset.get("name") in {"dev", "production"} and requested_decay_steps == 0
    if auto_resolved:
        args.lr_decay_steps = max(planned_total_steps, int(args.lr_warmup_steps) + 1)
        training_preset.setdefault("overrides", {})["lr_decay_steps"] = {
            "requested": requested_decay_steps,
            "effective": int(args.lr_decay_steps),
            "source": "page_aware_planned_optimizer_steps",
        }
    budget = {
        "source": "page_aware_jsonl_scan_v1",
        "records": sum(1 for _ in iter_jsonl(args.train, int(args.limit_records) or None)),
        "batch_records": int(args.batch_records),
        "steps_per_epoch": steps_per_epoch,
        "epochs": int(args.epochs),
        "planned_total_steps": planned_total_steps,
        "requested_lr_decay_steps": requested_decay_steps,
        "effective_lr_decay_steps": int(args.lr_decay_steps),
        "auto_resolved": auto_resolved,
    }
    training_preset["optimizer_step_budget"] = budget
    return budget


def main() -> int:
    args = parse_args()
    training_preset = apply_training_preset(args)
    if not 0.0 <= float(args.sq_rq_gradient_scale) <= 0.1:
        raise ValueError("--sq-rq-gradient-scale must be in [0, 0.1]")
    for option in (
        "sq_rq_query_confidence_threshold", "sq_rq_token_membership_threshold",
        "sq_rq_warmup_query_confidence_threshold", "sq_rq_warmup_token_membership_threshold",
        "sq_rq_training_membership_temperature",
    ):
        if not 0.0 <= float(getattr(args, option)) <= 1.0:
            raise ValueError(f"--{option.replace('_', '-')} must be in [0, 1]")
    if int(args.sq_rq_coverage_warmup_epochs) < 0:
        raise ValueError("--sq-rq-coverage-warmup-epochs must be non-negative")
    if int(args.sq_rq_enable_after_epoch) < 1:
        raise ValueError("--sq-rq-enable-after-epoch must be positive")
    if float(args.sq_rq_max_semantic_loss_regression) < 0.0:
        raise ValueError("--sq-rq-max-semantic-loss-regression must be non-negative")
    for option in ("sq_rq_min_admitted_query_coverage", "sq_rq_min_context_edge_coverage"):
        if not 0.0 <= float(getattr(args, option)) <= 1.0:
            raise ValueError(f"--{option.replace('_', '-')} must be in [0, 1]")
    for option in (
        "precision_phase_min_calibrated_rq",
        "precision_phase_min_calibrated_proposal_coverage",
    ):
        if not 0.0 <= float(getattr(args, option)) <= 1.0:
            raise ValueError(f"--{option.replace('_', '-')} must be in [0, 1]")
    if (
        float(args.rq_sq_quality_ranking_weight) < 0.0
        or float(args.rq_sq_quality_ranking_margin) < 0.0
        or int(args.rq_sq_quality_ranking_top_k) < 1
        or float(args.rq_sq_quality_hard_negative_weight) < 0.0
        or float(args.rq_sq_quality_unmatched_ceiling_weight) < 0.0
    ):
        raise ValueError("quality ranking weight/margin must be non-negative and --rq-sq-quality-ranking-top-k must be positive")
    if not 0.0 < float(args.rq_sq_quality_unmatched_ceiling_probability) < 1.0:
        raise ValueError("--rq-sq-quality-unmatched-ceiling-probability must be in (0, 1)")
    for option in (
        "min_val_calibrated_proposal_coverage_for_checkpoint",
        "max_val_unmatched_quality_for_checkpoint",
        "max_val_quality_ranking_violation_rate_for_checkpoint",
    ):
        if not 0.0 <= float(getattr(args, option)) <= 1.0:
            raise ValueError(f"--{option.replace('_', '-')} must be in [0, 1]")
    if int(args.min_val_instance_tp_for_checkpoint) < 0:
        raise ValueError("--min-val-instance-tp-for-checkpoint must be non-negative")
    if not 0.0 <= float(args.deployment_min_query_score) <= 1.0:
        raise ValueError("--deployment-min-query-score must be in [0, 1]")
    if not 0.0 <= float(args.deployment_mask_threshold) <= 1.0:
        raise ValueError("--deployment-mask-threshold must be in [0, 1]")
    if float(args.router_load_balance_loss_weight) < 0.0 or float(args.router_z_loss_weight) < 0.0:
        raise ValueError("router loss weights must be non-negative")
    for option in (
        "component_seed_loss_weight",
        "offset_vote_loss_weight",
        "affinity_loss_weight",
        "quality_soft_target_weight",
        "candidate_mask_prior_loss_weight",
        "family_recall_loss_weight",
        "rq_admission_expert_weight",
        "mask_recall_expert_weight",
        "quality_deployment_expert_weight",
        "route_classification_loss_weight",
    ):
        if float(getattr(args, option)) < 0.0:
            raise ValueError(f"--{option.replace('_', '-')} must be non-negative")
    if int(args.dense_attention_window_size) < 1:
        raise ValueError("--dense-attention-window-size must be positive")
    if float(args.candidate_mask_prior_logit) < 0.0:
        raise ValueError("--candidate-mask-prior-logit must be non-negative")
    if not 0.0 <= float(args.quality_soft_target_weight) <= 1.0:
        raise ValueError("--quality-soft-target-weight must be in [0, 1]")
    for option in ("family_recall_admission_floor", "family_recall_mask_prob_floor", "family_recall_quality_floor"):
        if not 0.0 < float(getattr(args, option)) < 1.0:
            raise ValueError(f"--{option.replace('_', '-')} must be in (0, 1)")
    for option in (
        "hard_recall_admission_floor",
        "hard_recall_mask_prob_floor",
        "hard_recall_deployment_floor",
        "hard_recall_quality_target_floor",
    ):
        if not 0.0 < float(getattr(args, option)) < 1.0:
            raise ValueError(f"--{option.replace('_', '-')} must be in (0, 1)")
    args._hard_recall_family_set = parse_family_set_csv(args.hard_recall_families)
    args._hard_recall_label_set = parse_int_set_csv(args.hard_recall_labels)
    for family in sorted(args._hard_recall_family_set):
        args._hard_recall_label_set.update(int(label) for label in FAMILY_LABELS[family] if int(label) < IGNORE_LABEL)
    invalid_hard_labels = [label for label in args._hard_recall_label_set if label < 0 or label >= IGNORE_LABEL]
    if invalid_hard_labels:
        raise ValueError(f"--hard-recall-labels contains invalid foreground labels: {invalid_hard_labels}")
    args._active_loss_experts = active_loss_experts_from_args(args)
    if bool(args.candidate_aware_queries) and (
        float(args.candidate_mask_prior_logit) > 0.0 or float(args.candidate_mask_prior_loss_weight) > 0.0
    ) and not str(args.candidate_ablation_tag).strip():
        raise ValueError("candidate mask prior is ablation-only; set --candidate-ablation-tag to make this explicit")
    if int(args.lr_warmup_steps) < 0 or int(args.lr_decay_steps) < 0:
        raise ValueError("learning-rate warmup/decay steps must be non-negative")
    if int(args.lr_decay_steps) and int(args.lr_decay_steps) <= int(args.lr_warmup_steps):
        raise ValueError("--lr-decay-steps must exceed --lr-warmup-steps when decay is enabled")
    if not 0.0 <= float(args.lr_min_scale) <= 1.0:
        raise ValueError("--lr-min-scale must be in [0, 1]")
    if float(args.router_lr_scale) <= 0.0 or float(args.head_lr_scale) <= 0.0:
        raise ValueError("router/head learning-rate scales must be positive")
    if int(args.router_collapse_warmup_epochs) < 0:
        raise ValueError("--router-collapse-warmup-epochs must be non-negative")
    if not 0.0 < float(args.router_max_dominant_expert_probability) <= 1.0:
        raise ValueError("--router-max-dominant-expert-probability must be in (0, 1]")
    if not 0.0 <= float(args.router_min_expert_assignment_fraction) <= 1.0:
        raise ValueError("--router-min-expert-assignment-fraction must be in [0, 1]")
    if int(args.final_instance_gate_interval_epochs) < 1:
        raise ValueError("--final-instance-gate-interval-epochs must be positive")
    if int(args.val_limit_pages) < 0:
        raise ValueError("--val-limit-pages must be non-negative")
    if not 0.0 < float(args.geometry_augmentation_scale_min) <= float(args.geometry_augmentation_scale_max):
        raise ValueError("geometry augmentation scale range is invalid")
    if float(args.geometry_augmentation_translation) < 0.0:
        raise ValueError("geometry augmentation translation must be non-negative")
    if args.geometry_augmentation and (
        float(args.geometry_augmentation_scale_min) != 1.0
        or float(args.geometry_augmentation_scale_max) != 1.0
        or float(args.geometry_augmentation_translation) != 0.0
    ):
        raise ValueError(
            "protocol-safe geometry augmentation only supports flips/quarter-turns; "
            "use scale-min=scale-max=1 and translation=0"
        )
    if float(args.unmatched_mask_negative_loss_weight) < 0.0 or int(args.unmatched_mask_negative_top_k) < 0:
        raise ValueError("unmatched-mask negative loss weight/top-k must be non-negative")
    if float(args.mask_precision_phase_positive_weight) <= 0.0:
        raise ValueError("--mask-precision-phase-positive-weight must be positive")
    if int(args.precision_phase_transition_epochs) <= 0:
        raise ValueError("--precision-phase-transition-epochs must be positive")
    if float(args.mask_tversky_alpha) < 0.0 or float(args.mask_tversky_beta) < 0.0:
        raise ValueError("mask Tversky alpha/beta must be non-negative")
    if float(args.mask_tversky_alpha) + float(args.mask_tversky_beta) <= 0.0:
        raise ValueError("mask Tversky alpha/beta cannot both be zero")
    for option in ("style_feature_dropout", "semantic_label_smoothing", "query_label_smoothing"):
        if not 0.0 <= float(getattr(args, option)) < 1.0:
            raise ValueError(f"--{option.replace('_', '-')} must be in [0, 1)")
    if args.sq_rq_enabled and args.geometry_decoder_mode != "geometry_v2":
        raise ValueError("SQ<-RQ requires --geometry-decoder-mode geometry_v2")
    if args.final_instance_gate_checkpoint is None:
        args.final_instance_gate_checkpoint = args.model_output.with_name(f".{args.model_output.stem}.per_epoch_final_gate.pt")
    if args.diagnostic_checkpoint_dir is None:
        args.diagnostic_checkpoint_dir = args.model_output.with_name(f"{args.model_output.stem}_diagnostic_topk")
    if int(args.diagnostic_checkpoint_top_k) < 0:
        raise ValueError("--diagnostic-checkpoint-top-k must be non-negative")
    input_feature_schema = input_feature_schema_from_args(args)
    protocol_claim_blockers = output_protocol_claim_blockers(args, input_feature_schema)
    if protocol_claim_blockers:
        payload = {
            "schema_version": "floorplancad_line_token_panoptic_moe_train_v1",
            "created_utc": utc_now(),
            "status": "blocked_protocol_claim_mismatch",
            "input_feature_schema": input_feature_schema,
            "blockers": protocol_claim_blockers,
            "override": "--allow-output-protocol-name-mismatch",
        }
        write_json(args.report, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2
    if args.require_final_instance_gate_for_best:
        required_placeholders = ("{checkpoint}", "{epoch}", "{report}")
        configuration_blockers = []
        if args.final_instance_gate_report is None:
            configuration_blockers.append("final_instance_gate_report_not_configured")
        if not args.final_instance_gate_command_template:
            configuration_blockers.append("final_instance_gate_command_template_not_configured")
        elif any(value not in args.final_instance_gate_command_template for value in required_placeholders):
            configuration_blockers.append("final_instance_gate_command_template_missing_required_placeholders")
        if configuration_blockers:
            payload = {
                "schema_version": "floorplancad_line_token_panoptic_moe_train_v1",
                "created_utc": utc_now(),
                "status": "blocked_configuration",
                "training_preset": training_preset,
                "blockers": configuration_blockers,
            }
            write_json(args.report, payload)
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 2
    missing = [path for path in [args.train, args.val] if not path.exists()]
    if args.init_checkpoint is not None and not args.init_checkpoint.exists():
        missing.append(args.init_checkpoint)
    if args.resume_checkpoint is not None and not args.resume_checkpoint.exists():
        missing.append(args.resume_checkpoint)
    if args.teacher_proposals is not None and not args.teacher_proposals.exists():
        missing.append(args.teacher_proposals)
    if args.candidate_proposals is not None and not args.candidate_proposals.exists():
        missing.append(args.candidate_proposals)
    if args.val_candidate_proposals is not None and not args.val_candidate_proposals.exists():
        missing.append(args.val_candidate_proposals)
    if missing:
        payload = {"schema_version": "floorplancad_line_token_panoptic_moe_train_v1", "created_utc": utc_now(), "status": "blocked", "blockers": [f"missing input: {rel(path)}" for path in missing]}
        write_json(args.report, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1
    if args.quality_calibration_only and args.init_checkpoint is None and args.resume_checkpoint is None:
        raise ValueError("--quality-calibration-only requires --init-checkpoint or --resume-checkpoint")
    input_protocol = input_protocol_from_args(args)
    active_feature_names = feature_names_for_input_protocol(input_protocol)
    required_target_schema = input_protocol["target_schema_version"]
    target_schema_blockers = []
    for split_name, path in (("train", args.train), ("val", args.val)):
        first_record = next(iter(iter_jsonl(path, 1)), None)
        if first_record is None:
            target_schema_blockers.append(f"{split_name}_target_cache_empty")
            continue
        try:
            if first_record.get("target_schema_version") != required_target_schema:
                raise ValueError(f"new training requires {required_target_schema}")
            if first_record.get("input_schema_version") != input_protocol["input_schema_version"]:
                raise ValueError(f"new training requires input schema {input_protocol['input_schema_version']}")
            load_panoptic_target_arrays(first_record, args.max_tokens_per_record, training=True, num_queries=args.num_queries)
        except ValueError as exc:
            target_schema_blockers.append(f"{split_name}_target_schema_invalid:{exc}")
    if target_schema_blockers:
        payload = {
            "schema_version": "floorplancad_line_token_panoptic_moe_train_v1",
            "created_utc": utc_now(),
            "status": "blocked_target_schema",
            "required_target_schema": required_target_schema,
            "required_input_schema": input_protocol["input_schema_version"],
            "legacy_diagnostic_only": True,
            "blockers": target_schema_blockers,
        }
        write_json(args.report, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2
    feature_ingress_reports = {
        split_name: feature_ingress_audit(
            path,
            max_tokens=args.max_tokens_per_record,
            num_queries=args.num_queries,
            feature_names=active_feature_names,
            input_protocol=input_protocol,
            limit_records=4,
            fail_closed=True,
        )
        for split_name, path in (("train", args.train), ("val", args.val))
    }
    feature_ingress_blockers = [
        f"{split_name}:{blocker}"
        for split_name, report in feature_ingress_reports.items()
        for blocker in report.get("blockers", [])
    ]
    if feature_ingress_blockers:
        payload = {
            "schema_version": "floorplancad_line_token_panoptic_moe_train_v1",
            "created_utc": utc_now(),
            "status": "blocked_feature_ingress",
            "feature_ingress": feature_ingress_reports,
            "blockers": feature_ingress_blockers,
        }
        write_json(args.report, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2
    training_step_budget = resolve_training_step_budget(args, training_preset)
    args._training_provenance = training_provenance(args)
    pack = import_torch()
    torch = pack["torch"]
    nn = pack["nn"]
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    official_sdp_enabled = False
    if device.type == "cuda" and hasattr(torch.backends, "cuda"):
        # Let PyTorch dispatch supported attention calls to FlashAttention or
        # the memory-efficient CUDA kernel; no project-specific CUDA extension.
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        official_sdp_enabled = True
    auto_throughput_profile = apply_auto_throughput_profile(args, torch, device)
    if auto_throughput_profile["launch_blocked"]:
        payload = {
            "schema_version": "floorplancad_line_token_panoptic_moe_train_v1",
            "created_utc": utc_now(),
            "status": "blocked_gpu_memory",
            "device": args.device,
            "auto_throughput_profile": auto_throughput_profile,
            "required_next_action": "wait for the selected GPU to become idle or lower --auto-profile-32gb-min-free-mib explicitly",
        }
        write_json(args.report, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2
    tf32_enabled = bool(args.device.startswith("cuda") and not args.disable_tf32)
    amp_dtype = amp_dtype_from_arg(torch, device, args.amp)
    cudnn_benchmark_enabled = bool(device.type == "cuda" and args.enable_cudnn_benchmark)
    init_payload = {
        "schema_version": "floorplancad_line_token_panoptic_moe_train_v1",
        "created_utc": utc_now(),
        "updated_utc": utc_now(),
        "status": "initializing",
        "attention_backend": "pytorch_sdpa_flash_or_mem_efficient" if official_sdp_enabled else "eager",
        "pid": os.getpid(),
        "inputs": {
            "train": rel(args.train),
            "val": rel(args.val),
            "init_checkpoint": rel(args.init_checkpoint),
            "resume_checkpoint": rel(args.resume_checkpoint),
            "teacher_proposals": rel(args.teacher_proposals),
            "candidate_proposals": rel(args.candidate_proposals),
            "val_candidate_proposals": rel(args.val_candidate_proposals),
        },
        "training_provenance": args._training_provenance,
        "feature_ingress": feature_ingress_reports,
        "outputs": {
            "model": rel(args.model_output),
            "last_model": rel(args.last_model_output),
            "report": rel(args.report),
        },
        "config": {
            "training_preset": training_preset,
            "optimizer_step_budget": training_step_budget,
            "device": args.device,
            "epochs": args.epochs,
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
            "heads": args.heads,
            "num_queries": args.num_queries,
            "query_decoder_layers": args.query_decoder_layers,
            "weak_family_feature_fusion": bool(args.weak_family_feature_fusion),
            "explicit_route_classifier": bool(args.explicit_route_classifier),
            "route_classification_loss_weight": float(args.route_classification_loss_weight),
            "route_conditioning_residual_scale": float(args.route_conditioning_residual_scale),
            "route_conditioning_enable_after_epoch": int(args.route_conditioning_enable_after_epoch),
            "route_conditioning_warmup_epochs": int(args.route_conditioning_warmup_epochs),
            "dense_attention_feature_adapter": bool(args.dense_attention_feature_adapter),
            "dense_attention_window_size": int(args.dense_attention_window_size),
            "dense_attention_adapter_residual_scale": float(args.dense_attention_adapter_residual_scale),
            "dense_attention_adapter_enable_after_epoch": int(args.dense_attention_adapter_enable_after_epoch),
            "dense_attention_adapter_warmup_epochs": int(args.dense_attention_adapter_warmup_epochs),
            "dropout": args.dropout,
            "position_encoding_version": POSITION_ENCODING_VERSION,
            "position_max_frequency_log2": POSITION_MAX_FREQUENCY_LOG2,
            "quality_head": PANOPTIC_QUALITY_HEAD_VERSION,
            "quality_objective": quality_objective_contract(mask_threshold=args.deployment_mask_threshold),
            "batch_records": args.batch_records,
            "auto_throughput_profile": auto_throughput_profile,
            "train_prefetch_records": args.train_prefetch_records,
            "train_prefetch_workers": args.train_prefetch_workers,
            "amp": args.amp,
            "tf32_enabled": tf32_enabled,
            "cudnn_benchmark_enabled": cudnn_benchmark_enabled,
            "compile_model": bool(args.compile_model),
            "compile_mode": args.compile_mode,
            "progress_checkpoint_records": args.progress_checkpoint_records,
            "progress_checkpoint_seconds": args.progress_checkpoint_seconds,
            "checkpoint_archive_dir": rel(args.checkpoint_archive_dir),
            "checkpoint_archive_keep": int(args.checkpoint_archive_keep),
            "graceful_signal_checkpoint": True,
            "component_matching": args.component_matching,
            "checkpoint_metric": args.checkpoint_metric,
            "limit_records": args.limit_records,
            "val_limit_records": args.val_limit_records,
        },
        "claim_boundary": "Initialization status only; no epoch has completed.",
        "comparable_for_matrix": False,
    }
    write_json(args.report, init_payload)
    print(json.dumps({"status": "initializing", "report": rel(args.report), "pid": os.getpid()}, ensure_ascii=False), flush=True)
    if tf32_enabled:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    if cudnn_benchmark_enabled:
        torch.backends.cudnn.benchmark = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    model = make_panoptic_model(
        nn, torch, len(active_feature_names), args.hidden_dim, args.layers, args.heads, args.num_queries,
        query_decoder_layers=args.query_decoder_layers, dropout=args.dropout,
        geometry_decoder_mode=args.geometry_decoder_mode, num_stuff_queries=args.num_stuff_queries,
        geometry_local_neighbors=args.geometry_local_neighbors, geometry_coarse_grid_size=args.geometry_coarse_grid_size,
        sq_rq_enabled=args.sq_rq_enabled, sq_rq_gradient_scale=args.sq_rq_gradient_scale,
        sq_rq_query_confidence_threshold=args.sq_rq_query_confidence_threshold,
        sq_rq_token_membership_threshold=args.sq_rq_token_membership_threshold,
        sq_rq_training_membership_temperature=args.sq_rq_training_membership_temperature,
        semantic_query_residual_enabled=args.semantic_query_residual_enabled,
        ownership_enabled=bool(args.geometry_decoder_mode == "geometry_v2" and args.sq_rq_enabled),
        learned_sparse_router=args.learned_sparse_router, router_num_experts=args.router_num_experts,
        router_top_k=args.router_top_k, router_temperature=args.router_temperature,
        typed_branch_routers=getattr(args, "typed_branch_routers", False),
        branch_num_experts=getattr(args, "branch_num_experts", 2),
        branch_top_k=getattr(args, "branch_top_k", 1),
        branch_capacity_factor=getattr(args, "branch_capacity_factor", 1.25),
        branch_dropless=getattr(args, "branch_dropless", False),
        typed_stuff_slots=getattr(args, "typed_stuff_slots", False),
        geometry_attention_tile_size=getattr(args, "geometry_attention_tile_size", 0),
        tensor_ring_rank=getattr(args, "tensor_ring_rank", 0),
        gradient_checkpointing=getattr(args, "gradient_checkpointing", False),
        content_seeded_queries=getattr(args, "content_seeded_queries", False),
        repeated_group_fusion=getattr(args, "repeated_group_fusion", False),
        relation_bias_enabled=getattr(args, "relation_bias_enabled", False),
        component_seeded_queries=getattr(args, "component_seeded_queries", False),
        offset_vote_enabled=getattr(args, "offset_vote_enabled", False),
        candidate_aware_queries=getattr(args, "candidate_aware_queries", False),
        candidate_feature_dim=getattr(args, "candidate_feature_dim", 0),
        candidate_mask_prior_logit=getattr(args, "candidate_mask_prior_logit", 0.0),
        weak_family_feature_fusion=getattr(args, "weak_family_feature_fusion", False),
        quality_query_gradient_scale=getattr(args, "quality_query_gradient_scale", 0.0),
        explicit_route_classifier=getattr(args, "explicit_route_classifier", False),
        dense_attention_feature_adapter=getattr(args, "dense_attention_feature_adapter", False),
        dense_attention_window_size=getattr(args, "dense_attention_window_size", 128),
        route_conditioning_residual_scale=getattr(args, "route_conditioning_residual_scale", 0.10),
        dense_attention_adapter_residual_scale=getattr(args, "dense_attention_adapter_residual_scale", 0.10),
    ).to(device)
    init_checkpoint_report = None
    if args.resume_checkpoint is None:
        init_checkpoint_report = load_init_checkpoint(torch, model, args.init_checkpoint, args, device)
    if args.quality_calibration_only and init_checkpoint_report is not None:
        source_deployment = init_checkpoint_report.get("sq_rq_deployment") or {}
        threshold_pairs = (
            (
                "query_confidence_threshold",
                float(args.sq_rq_query_confidence_threshold),
            ),
            (
                "token_membership_threshold",
                float(args.sq_rq_token_membership_threshold),
            ),
        )
        mismatches = {
            name: {"source": source_deployment.get(name), "requested": requested}
            for name, requested in threshold_pairs
            if name in source_deployment
            and not math.isclose(
                float(source_deployment[name]), requested, rel_tol=0.0, abs_tol=1e-12
            )
        }
        if mismatches:
            raise ValueError(
                "quality calibration must preserve source SQ-RQ deployment thresholds: "
                f"{mismatches}"
            )
    quality_calibration_scope = configure_quality_calibration_scope(
        model,
        bool(args.quality_calibration_only),
    )
    optimizer, scheduler = build_optimizer_and_scheduler(torch, model, args)
    resume_report = load_resume_checkpoint(
        torch, model, optimizer, args.resume_checkpoint, args, device, scheduler=scheduler,
    )
    compile_report: dict[str, Any] = {
        "requested": bool(args.compile_model),
        "enabled": False,
        "mode": args.compile_mode,
        "status": "not_requested",
    }
    if args.compile_model:
        if not hasattr(torch, "compile"):
            compile_report["status"] = "torch_compile_unavailable"
        else:
            try:
                model = torch.compile(model, mode=args.compile_mode)
                compile_report["enabled"] = True
                compile_report["status"] = "enabled"
            except Exception as exc:  # noqa: BLE001 - compile is an acceleration option, not a correctness requirement.
                compile_report["status"] = "failed_fell_back_to_eager"
                compile_report["error_type"] = type(exc).__name__
                compile_report["error"] = str(exc)
    restored_full_rng_state = restore_training_rng_state(torch, np, rng, resume_report.get("training_rng_state"))
    if not restored_full_rng_state and resume_report.get("rng_state") is not None:
        rng.setstate(resume_report["rng_state"])
    resume_report["full_rng_state_restored"] = bool(restored_full_rng_state)
    resume_report["legacy_local_rng_only"] = bool(
        resume_report.get("enabled") and not restored_full_rng_state and resume_report.get("rng_state") is not None
    )
    resume_report = reportable_resume_checkpoint(resume_report)
    sensitivity_smoke = feature_sensitivity_smoke(
        model,
        torch,
        args.train,
        device,
        max_tokens=args.max_tokens_per_record,
        num_queries=args.num_queries,
        feature_names=active_feature_names,
    )
    if not sensitivity_smoke.get("passed", False):
        payload = {
            "schema_version": "floorplancad_line_token_panoptic_moe_train_v1",
            "created_utc": utc_now(),
            "status": "blocked_feature_sensitivity",
            "feature_ingress": feature_ingress_reports,
            "feature_sensitivity": sensitivity_smoke,
            "blockers": sensitivity_smoke.get("blockers", []),
        }
        write_json(args.report, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2
    bottleneck_profile = load_bottleneck_profile(None if args.disable_bottleneck_weights else args.bottleneck_ledger)
    bottleneck_profile = apply_label_overrides(
        bottleneck_profile,
        extra_recall=parse_label_list(args.extra_recall_labels),
        extra_grouping=parse_label_list(args.extra_grouping_labels),
        extra_precision=parse_label_list(args.extra_precision_labels),
    )
    query_class_weights = build_query_class_weights(
        torch,
        bottleneck_profile,
        no_object_weight=args.no_object_query_weight,
        recall_class_weight=args.recall_class_weight,
        precision_class_weight=args.precision_class_weight,
        grouping_class_weight=args.grouping_class_weight,
    ).to(device)
    weight_limit = args.limit_records if args.limit_records > 0 else None
    class_weights = None if args.no_class_weights else unique_page_semantic_class_weights(
        torch, args.train, limit=weight_limit,
    ).to(device)
    ce_sem = nn.CrossEntropyLoss(
        ignore_index=IGNORE_LABEL, weight=class_weights,
        label_smoothing=float(args.semantic_label_smoothing),
    )
    ce_query = nn.CrossEntropyLoss(
        weight=query_class_weights, label_smoothing=float(args.query_label_smoothing),
    )
    teacher_by_record, teacher_report = load_teacher_proposals(
        args.teacher_proposals,
        positive_only=not args.teacher_allow_unmatched,
        min_gt_iou=args.teacher_min_gt_iou,
    )
    candidate_by_record, candidate_report = load_candidate_proposals(
        args.candidate_proposals,
        max_candidates=args.max_candidate_queries,
        feature_dim=args.candidate_feature_dim,
    )
    val_candidate_by_record, val_candidate_report = load_candidate_proposals(
        args.val_candidate_proposals if args.val_candidate_proposals is not None else args.candidate_proposals,
        max_candidates=args.max_candidate_queries,
        feature_dim=args.candidate_feature_dim,
    )
    train_limit = args.limit_records if args.limit_records > 0 else None
    val_limit = args.val_limit_records if args.val_limit_records > 0 else None
    if val_limit is not None and args.val_limit_pages > 0:
        raise ValueError("--val-limit-records and --val-limit-pages are mutually exclusive")
    selected_val_pages = deterministic_stratified_validation_pages(
        args.val, args.val_limit_pages, args.seed,
    )
    validation_is_full = val_limit is None and selected_val_pages is None
    train_candidate_coverage = candidate_record_coverage(
        args.train,
        limit=train_limit,
        record_id_allowlist=None,
        candidate_by_record=candidate_by_record,
        max_candidates=args.max_candidate_queries,
        feature_dim=args.candidate_feature_dim,
    ) if args.candidate_aware_queries and candidate_by_record else {"records": 0, "records_with_candidates": 0, "record_coverage": 0.0, "candidates": 0, "tokens_in_candidate_masks": 0}
    val_candidate_coverage = candidate_record_coverage(
        args.val,
        limit=val_limit,
        record_id_allowlist=selected_val_pages,
        candidate_by_record=val_candidate_by_record,
        max_candidates=args.max_candidate_queries,
        feature_dim=args.candidate_feature_dim,
    ) if args.candidate_aware_queries and val_candidate_by_record else {"records": 0, "records_with_candidates": 0, "record_coverage": 0.0, "candidates": 0, "tokens_in_candidate_masks": 0}
    if (
        args.candidate_aware_queries
        and int(args.max_candidate_queries) > 0
        and int(args.candidate_feature_dim) > 0
        and val_candidate_coverage["records"] > 0
        and val_candidate_coverage["records_with_candidates"] == 0
    ):
        payload = {
            "schema_version": "floorplancad_line_token_panoptic_moe_train_v1",
            "created_utc": utc_now(),
            "status": "blocked_candidate_validation_coverage",
            "blockers": ["candidate-aware validation candidate coverage is zero"],
            "candidate_report": candidate_report,
            "val_candidate_report": val_candidate_report,
            "train_candidate_coverage": train_candidate_coverage,
            "val_candidate_coverage": val_candidate_coverage,
        }
        write_json(args.report, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2
    if args.sq_rq_enabled and int(args.epochs) < int(args.sq_rq_enable_after_epoch) and not args.allow_sq_rq_never_enabled_smoke:
        payload = {
            "schema_version": "floorplancad_line_token_panoptic_moe_train_v1",
            "created_utc": utc_now(),
            "status": "blocked_sqrq_never_enabled",
            "blockers": ["epochs is smaller than sq_rq_enable_after_epoch"],
            "epochs": int(args.epochs),
            "sq_rq_enable_after_epoch": int(args.sq_rq_enable_after_epoch),
        }
        write_json(args.report, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2
    history = list(resume_report.get("history") or [])
    best_score = float(resume_report.get("best_score", -float("inf")))
    best_val_semantic = float(resume_report.get("best_val_semantic", -1.0))
    calibration_source_deployment = (
        (init_checkpoint_report or {}).get("sq_rq_deployment")
        if args.quality_calibration_only
        else None
    ) or {}
    sq_rq_auto_fused = bool(
        resume_report.get(
            "sq_rq_auto_fused",
            calibration_source_deployment.get("auto_fused", False),
        )
    )
    sq_rq_auto_fuse_reason = resume_report.get(
        "sq_rq_auto_fuse_reason",
        calibration_source_deployment.get("auto_fuse_reason"),
    )
    args._sq_rq_auto_fused = sq_rq_auto_fused
    args._sq_rq_auto_fuse_reason = sq_rq_auto_fuse_reason
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    run_id = str(resume_report.get("run_id") or f"panoptic_moe_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{os.getpid()}")
    start_epoch = int(resume_report.get("start_epoch", 1))
    resume_skip_records = int(resume_report.get("resume_skip_records", 0) or 0)
    startup_payload = {
        "schema_version": "floorplancad_line_token_panoptic_moe_train_v1",
        "created_utc": utc_now(),
        "updated_utc": utc_now(),
        "status": "running",
        "run_id": run_id,
        "pid": os.getpid(),
        "current_epoch": 0,
        "target_epochs": args.epochs,
        "latest_epoch": {},
        "history": [],
        "inputs": {
            "train": rel(args.train),
            "val": rel(args.val),
            "bottleneck_ledger": rel(args.bottleneck_ledger),
            "init_checkpoint": rel(args.init_checkpoint),
            "resume_checkpoint": rel(args.resume_checkpoint),
            "teacher_proposals": rel(args.teacher_proposals),
        },
        "outputs": {"model": rel(args.model_output), "report": rel(args.report)},
        "init_checkpoint": init_checkpoint_report,
        "resume_checkpoint": resume_report,
        "quality_calibration_scope": quality_calibration_scope,
        "feature_ingress": feature_ingress_reports,
        "feature_sensitivity": sensitivity_smoke,
        "config": {
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
            "heads": args.heads,
            "num_queries": args.num_queries,
            "query_decoder_layers": args.query_decoder_layers,
            "dropout": args.dropout,
            "style_feature_dropout": args.style_feature_dropout,
            "semantic_label_smoothing": args.semantic_label_smoothing,
            "query_label_smoothing": args.query_label_smoothing,
            "max_tokens_per_record": args.max_tokens_per_record,
            "checkpoint_metric": args.checkpoint_metric,
            "lr": args.lr,
            "batch_records": args.batch_records,
            "auto_throughput_profile": auto_throughput_profile,
            "train_prefetch_records": args.train_prefetch_records,
            "train_prefetch_workers": args.train_prefetch_workers,
            "amp": args.amp,
            "amp_dtype": str(amp_dtype).replace("torch.", "") if amp_dtype is not None else None,
            "tf32_enabled": tf32_enabled,
            "cudnn_benchmark_enabled": cudnn_benchmark_enabled,
            "compile_model": compile_report,
            "progress_checkpoint_records": args.progress_checkpoint_records,
            "progress_checkpoint_seconds": args.progress_checkpoint_seconds,
            "progress_status_records": args.progress_status_records,
            "progress_status_seconds": args.progress_status_seconds,
            "checkpoint_archive_dir": rel(args.checkpoint_archive_dir),
            "checkpoint_archive_keep": int(args.checkpoint_archive_keep),
            "graceful_signal_checkpoint": True,
            "component_matching": args.component_matching,
            "semantic_loss_weight": args.semantic_loss_weight,
            "query_loss_weight": args.query_loss_weight,
            "query_objectness_loss_weight": args.query_objectness_loss_weight,
            "query_objectness_positive_weight": args.query_objectness_positive_weight,
            "query_objectness_negative_weight": args.query_objectness_negative_weight,
            "query_objectness_positive_margin_floor_loss_weight": args.query_objectness_positive_margin_floor_loss_weight,
            "query_objectness_negative_margin_ceiling_loss_weight": args.query_objectness_negative_margin_ceiling_loss_weight,
            "rq_sq_quality_calibration_loss_weight": args.rq_sq_quality_calibration_loss_weight,
            "rq_sq_quality_ranking_weight": args.rq_sq_quality_ranking_weight,
            "rq_sq_quality_ranking_margin": args.rq_sq_quality_ranking_margin,
            "rq_sq_quality_ranking_top_k": args.rq_sq_quality_ranking_top_k,
            "rq_sq_quality_hard_negative_weight": args.rq_sq_quality_hard_negative_weight,
            "rq_sq_quality_unmatched_ceiling_weight": args.rq_sq_quality_unmatched_ceiling_weight,
            "rq_sq_quality_unmatched_ceiling_probability": args.rq_sq_quality_unmatched_ceiling_probability,
            "family_recall_focus": args.family_recall_focus,
            "family_recall_loss_weight": args.family_recall_loss_weight,
            "family_recall_admission_floor": args.family_recall_admission_floor,
            "family_recall_mask_prob_floor": args.family_recall_mask_prob_floor,
            "family_recall_quality_floor": args.family_recall_quality_floor,
            "quality_calibration_only": bool(args.quality_calibration_only),
            "lr_scheduler": learning_rate_scheduler_config(args),
            "precision_phase_require_healthy_admission_epochs": args.precision_phase_require_healthy_admission_epochs,
            "precision_phase_min_object_recall": args.precision_phase_min_object_recall,
            "precision_phase_max_negative_margin_rate": args.precision_phase_max_negative_margin_rate,
            "precision_phase_min_sq_proxy": args.precision_phase_min_sq_proxy,
            "precision_phase_transition_epochs": args.precision_phase_transition_epochs,
            "objectness_warmup_positive_margin_floor_loss_weight": args.objectness_warmup_positive_margin_floor_loss_weight,
            "objectness_warmup_negative_margin_ceiling_loss_weight": args.objectness_warmup_negative_margin_ceiling_loss_weight,
            "objectness_warmup_epochs": args.objectness_warmup_epochs,
            "objectness_warmup_loss_multiplier": args.objectness_warmup_loss_multiplier,
            "objectness_warmup_positive_multiplier": args.objectness_warmup_positive_multiplier,
            "objectness_warmup_negative_multiplier": args.objectness_warmup_negative_multiplier,
            "objectness_precision_phase_start_epoch": args.objectness_precision_phase_start_epoch,
            "objectness_precision_phase_loss_weight": args.objectness_precision_phase_loss_weight,
            "objectness_precision_phase_positive_weight": args.objectness_precision_phase_positive_weight,
            "objectness_precision_phase_negative_weight": args.objectness_precision_phase_negative_weight,
            "objectness_precision_phase_positive_margin_floor_loss_weight": args.objectness_precision_phase_positive_margin_floor_loss_weight,
            "objectness_precision_phase_negative_margin_ceiling_loss_weight": args.objectness_precision_phase_negative_margin_ceiling_loss_weight,
            "objectness_positive_margin_floor": args.objectness_positive_margin_floor,
            "objectness_negative_margin_ceiling": args.objectness_negative_margin_ceiling,
            "zero_admission_patience_epochs": args.zero_admission_patience_epochs,
            "zero_admission_min_epoch": args.zero_admission_min_epoch,
            "min_val_object_recall_for_checkpoint": args.min_val_object_recall_for_checkpoint,
            "min_val_mask_recall_for_checkpoint": args.min_val_mask_recall_for_checkpoint,
            "min_val_mask_precision_for_checkpoint": args.min_val_mask_precision_for_checkpoint,
            "max_val_positive_rate_ratio_for_checkpoint": args.max_val_positive_rate_ratio_for_checkpoint,
            "min_val_positive_object_margin_rate_for_checkpoint": args.min_val_positive_object_margin_rate_for_checkpoint,
            "min_val_positive_object_margin_mean_for_checkpoint": args.min_val_positive_object_margin_mean_for_checkpoint,
            "max_val_negative_object_margin_rate_for_checkpoint": args.max_val_negative_object_margin_rate_for_checkpoint,
            "min_val_rq_proxy_for_checkpoint": args.min_val_rq_proxy_for_checkpoint,
            "min_val_sq_proxy_for_checkpoint": args.min_val_sq_proxy_for_checkpoint,
            "min_val_instance_tp_for_checkpoint": args.min_val_instance_tp_for_checkpoint,
            "min_val_calibrated_proposal_coverage_for_checkpoint": args.min_val_calibrated_proposal_coverage_for_checkpoint,
            "max_val_unmatched_quality_for_checkpoint": args.max_val_unmatched_quality_for_checkpoint,
            "max_val_quality_ranking_violation_rate_for_checkpoint": args.max_val_quality_ranking_violation_rate_for_checkpoint,
            "require_final_instance_gate_for_best": args.require_final_instance_gate_for_best,
            "final_instance_gate_report": rel(args.final_instance_gate_report),
            "final_instance_gate_command_template": args.final_instance_gate_command_template,
            "final_instance_gate_protocol": args.final_instance_gate_protocol,
            "final_instance_gate_checkpoint": rel(args.final_instance_gate_checkpoint),
            "min_final_instance_tp_for_best": args.min_final_instance_tp_for_best,
            "min_final_instance_rq_for_best": args.min_final_instance_rq_for_best,
            "min_final_instance_sq_for_best": args.min_final_instance_sq_for_best,
            "min_final_instance_pq_for_best": getattr(args, "min_final_instance_pq_for_best", 0.0),
            "max_final_instance_fp_for_best": args.max_final_instance_fp_for_best,
            "mask_loss_weight": args.mask_loss_weight,
            "mask_positive_weight": args.mask_positive_weight,
            "mask_negative_weight": args.mask_negative_weight,
            "mask_focal_gamma": args.mask_focal_gamma,
            "mask_area_ratio_loss_weight": args.mask_area_ratio_loss_weight,
            "mask_area_overcoverage_weight": args.mask_area_overcoverage_weight,
            "mask_tversky_loss_weight": args.mask_tversky_loss_weight,
            "mask_tversky_alpha": args.mask_tversky_alpha,
            "mask_tversky_beta": args.mask_tversky_beta,
            "mask_positive_prob_floor_loss_weight": args.mask_positive_prob_floor_loss_weight,
            "mask_positive_prob_floor": args.mask_positive_prob_floor,
            "mask_precision_phase_start_epoch": args.mask_precision_phase_start_epoch,
            "mask_precision_phase_positive_weight": args.mask_precision_phase_positive_weight,
            "mask_precision_phase_negative_weight": args.mask_precision_phase_negative_weight,
            "mask_precision_phase_area_ratio_loss_weight": args.mask_precision_phase_area_ratio_loss_weight,
            "mask_precision_phase_area_overcoverage_weight": args.mask_precision_phase_area_overcoverage_weight,
            "mask_precision_phase_tversky_loss_weight": args.mask_precision_phase_tversky_loss_weight,
            "mask_precision_phase_positive_prob_floor_loss_weight": args.mask_precision_phase_positive_prob_floor_loss_weight,
            "teacher_loss_weight": args.teacher_loss_weight,
            "teacher_mask_loss_weight": args.teacher_mask_loss_weight,
            "teacher_query_loss_weight": args.teacher_query_loss_weight,
            "teacher_min_gt_iou": args.teacher_min_gt_iou,
            "teacher_allow_unmatched": args.teacher_allow_unmatched,
            "teacher_report": teacher_report,
            "candidate_report": candidate_report,
            "val_candidate_report": val_candidate_report,
            "candidate_ablation_only": bool(args.candidate_aware_queries),
            "candidate_ablation_tag": str(args.candidate_ablation_tag),
            "candidate_mask_prior_logit": float(args.candidate_mask_prior_logit),
            "train_candidate_coverage": train_candidate_coverage,
            "val_candidate_coverage": val_candidate_coverage,
            "family_recall_focus": args.family_recall_focus,
            "family_recall_loss_weight": args.family_recall_loss_weight,
            "family_recall_admission_floor": args.family_recall_admission_floor,
            "family_recall_mask_prob_floor": args.family_recall_mask_prob_floor,
            "family_recall_quality_floor": args.family_recall_quality_floor,
            "active_loss_experts": args.active_loss_experts,
            "loss_expert_groups": {key: list(value) for key, value in LOSS_EXPERT_GROUPS.items()},
            "hard_recall_labels": args.hard_recall_labels,
            "hard_recall_families": args.hard_recall_families,
            "effective_hard_recall_labels": sorted(args._hard_recall_label_set),
            "rq_admission_expert_weight": args.rq_admission_expert_weight,
            "mask_recall_expert_weight": args.mask_recall_expert_weight,
            "quality_deployment_expert_weight": args.quality_deployment_expert_weight,
            "hard_recall_admission_floor": args.hard_recall_admission_floor,
            "hard_recall_mask_prob_floor": args.hard_recall_mask_prob_floor,
            "hard_recall_deployment_floor": args.hard_recall_deployment_floor,
            "hard_recall_quality_target_floor": args.hard_recall_quality_target_floor,
            "val_limit_records": args.val_limit_records,
            "val_limit_pages": args.val_limit_pages,
            "selected_val_page_count": None if selected_val_pages is None else len(selected_val_pages),
            "selected_val_pages_sha256": (
                None if selected_val_pages is None
                else canonical_json_sha256(list(selected_val_pages))
            ),
            "seed": args.seed,
        },
        "claim_boundary": "Startup status only until at least one epoch is written.",
    }
    write_json(args.report, startup_payload)

    payload = startup_payload
    final_status = "trained"
    early_stop_payload = None
    best_checkpoint_claimed_by_resume = bool(resume_report.get("best_checkpoint_written", False))
    best_checkpoint_file_exists = args.model_output.exists()
    best_checkpoint_written = bool(best_checkpoint_claimed_by_resume and best_checkpoint_file_exists)
    throughput_history: list[dict[str, Any]] = []
    stop_requested: dict[str, Any] = {"requested": False, "signal": None, "at_utc": None}

    def request_graceful_stop(signum: int, _frame: Any) -> None:
        stop_requested["requested"] = True
        stop_requested["signal"] = int(signum)
        stop_requested["at_utc"] = utc_now()

    previous_handlers = {
        signal.SIGTERM: signal.getsignal(signal.SIGTERM),
        signal.SIGINT: signal.getsignal(signal.SIGINT),
    }
    signal.signal(signal.SIGTERM, request_graceful_stop)
    signal.signal(signal.SIGINT, request_graceful_stop)
    for epoch in range(start_epoch, args.epochs + 1):
        runtime_model = getattr(model, "_orig_mod", model)
        if args.quality_calibration_only and calibration_source_deployment:
            runtime_model.sq_rq_runtime_enabled = bool(
                calibration_source_deployment.get("enabled", False)
                and not sq_rq_auto_fused
            )
        else:
            runtime_model.sq_rq_runtime_enabled = bool(
                args.sq_rq_enabled
                and epoch >= args.sq_rq_enable_after_epoch
                and not sq_rq_auto_fused
            )
        runtime_model.route_conditioning_runtime_scale = scheduled_auxiliary_scale(
            epoch,
            getattr(args, "route_conditioning_enable_after_epoch", 2),
            getattr(args, "route_conditioning_warmup_epochs", 3),
        )
        runtime_model.dense_attention_adapter_runtime_scale = scheduled_auxiliary_scale(
            epoch,
            getattr(args, "dense_attention_adapter_enable_after_epoch", 2),
            getattr(args, "dense_attention_adapter_warmup_epochs", 3),
        )
        epoch_started = time.perf_counter()
        precision_phase_allowed = precision_phase_admission_ready(
            history,
            args.precision_phase_require_healthy_admission_epochs,
            args.precision_phase_min_object_recall,
            args.precision_phase_max_negative_margin_rate,
            args.precision_phase_min_sq_proxy,
            args.precision_phase_min_calibrated_rq,
            args.precision_phase_min_calibrated_proposal_coverage,
        )
        previous_objectness_progress = float(
            (((history[-1].get("objectness_schedule") or {}).get("precision_phase_progress", 0.0)) if history else 0.0)
            or 0.0
        )
        previous_mask_progress = float(
            (((history[-1].get("mask_loss_schedule") or {}).get("precision_phase_progress", 0.0)) if history else 0.0)
            or 0.0
        )
        schedule = objectness_schedule(
            args,
            epoch,
            precision_phase_allowed=precision_phase_allowed,
            previous_precision_phase_progress=previous_objectness_progress,
        )
        mask_schedule = mask_loss_schedule(
            args,
            epoch,
            precision_phase_allowed=precision_phase_allowed,
            previous_precision_phase_progress=previous_mask_progress,
        )
        sq_rq_train_thresholds = sq_rq_training_threshold_schedule(args, epoch)
        if args.quality_calibration_only:
            sq_rq_train_thresholds = {
                **sq_rq_train_thresholds,
                "query_confidence_threshold": float(
                    calibration_source_deployment.get(
                        "query_confidence_threshold", args.sq_rq_query_confidence_threshold
                    )
                ),
                "token_membership_threshold": float(
                    calibration_source_deployment.get(
                        "token_membership_threshold", args.sq_rq_token_membership_threshold
                    )
                ),
                "training_membership_temperature": 0.0,
                "warmup_epochs": 0,
                "phase": "frozen_hard_thresholds",
                "calibration_runtime_frozen": True,
            }
        set_sq_rq_runtime_thresholds(model, sq_rq_train_thresholds)
        model.train()
        if args.quality_calibration_only:
            model.eval()
            getattr(model, "_orig_mod", model).query_quality_head.train()
        resumed_epoch_progress = (
            resume_report.get("intra_epoch_progress")
            if epoch == start_epoch and resume_report.get("resume_intra_epoch")
            else None
        )
        saved_epoch_aggregate = (
            resumed_epoch_progress.get("epoch_aggregate")
            if isinstance(resumed_epoch_progress, dict) and isinstance(resumed_epoch_progress.get("epoch_aggregate"), dict)
            else {}
        )
        counters = Counter(saved_epoch_aggregate.get("counters") or {})
        gradient_control_reports: list[dict[str, Any]] = list(saved_epoch_aggregate.get("gradient_control_reports") or [])
        loss_sum = float(saved_epoch_aggregate.get("loss_sum", 0.0) or 0.0)
        batch_size = max(1, int(args.batch_records))
        records_seen_this_epoch = resume_skip_records if epoch == start_epoch else 0
        segment_start_completed_records = records_seen_this_epoch
        segment_start_counters = Counter(counters)
        next_progress_checkpoint = next_progress_threshold(records_seen_this_epoch, args.progress_checkpoint_records)
        next_status_records = next_progress_threshold(records_seen_this_epoch, args.progress_status_records)
        last_status_write = records_seen_this_epoch
        last_progress_checkpoint_time = epoch_started
        last_status_time = epoch_started
        pin_host_batch = device.type == "cuda"

        def throughput_payload(records_completed: int) -> dict[str, Any]:
            elapsed = max(time.perf_counter() - epoch_started, 1e-9)
            completed_this_run = max(int(records_completed) - segment_start_completed_records, 0)
            trained_this_run = max(int(counters["records"]) - int(segment_start_counters["records"]), 0)
            tokens_this_run = max(int(counters["tokens"]) - int(segment_start_counters["tokens"]), 0)
            return {
                "epoch": epoch,
                "records_completed_total": int(records_completed),
                "records_completed_this_run": completed_this_run,
                "records_seen_this_epoch": int(records_completed),
                "train_records_used_total": int(counters["records"]),
                "train_records_used_this_run": trained_this_run,
                "train_records_used": int(counters["records"]),
                "optimizer_steps_total": int(counters["optimizer_steps"]),
                "optimizer_steps_this_run": max(int(counters["optimizer_steps"]) - int(segment_start_counters["optimizer_steps"]), 0),
                "optimizer_steps": int(counters["optimizer_steps"]),
                "elapsed_seconds": elapsed,
                "records_per_second_completed_this_run": float(completed_this_run) / elapsed,
                "records_per_second_trained_this_run": float(trained_this_run) / elapsed,
                "tokens_per_second_this_run": float(tokens_this_run) / elapsed,
                "batch_records": batch_size,
                "train_prefetch_records": int(args.train_prefetch_records),
                "train_prefetch_workers": int(args.train_prefetch_workers),
                "amp": args.amp,
                "cuda_memory": cuda_memory_summary(torch, device),
            }

        def write_running_status(records_completed: int, *, reason: str) -> None:
            status_payload = {
                "schema_version": "floorplancad_line_token_panoptic_moe_train_v1",
                "created_utc": startup_payload["created_utc"],
                "updated_utc": utc_now(),
                "status": "running_epoch_training",
                "run_id": run_id,
                "pid": os.getpid(),
                "current_epoch": epoch,
                "target_epochs": args.epochs,
                "model_output": rel(args.model_output),
                "last_model_output": rel(args.last_model_output),
                "inputs": {
                    "train": rel(args.train),
                    "val": rel(args.val),
                    "init_checkpoint": rel(args.init_checkpoint),
                    "resume_checkpoint": rel(args.resume_checkpoint),
                    "teacher_proposals": rel(args.teacher_proposals),
                    "candidate_proposals": rel(args.candidate_proposals),
                    "val_candidate_proposals": rel(args.val_candidate_proposals),
                },
                "config": {
                    "batch_records": args.batch_records,
                    "auto_throughput_profile": auto_throughput_profile,
                    "train_prefetch_records": args.train_prefetch_records,
                    "train_prefetch_workers": args.train_prefetch_workers,
                    "amp": args.amp,
                    "amp_dtype": str(amp_dtype).replace("torch.", "") if amp_dtype is not None else None,
                    "tf32_enabled": tf32_enabled,
                    "cudnn_benchmark_enabled": cudnn_benchmark_enabled,
                    "compile_model": compile_report,
                    "progress_checkpoint_records": args.progress_checkpoint_records,
                    "progress_checkpoint_seconds": args.progress_checkpoint_seconds,
                    "progress_status_records": args.progress_status_records,
                    "progress_status_seconds": args.progress_status_seconds,
                    "checkpoint_archive_dir": rel(args.checkpoint_archive_dir),
                    "checkpoint_archive_keep": int(args.checkpoint_archive_keep),
                    "graceful_signal_checkpoint": True,
                    "component_matching": args.component_matching,
                    "resume_checkpoint": rel(args.resume_checkpoint),
                    "resume_optimizer": args.resume_optimizer,
                },
                "resume_checkpoint": resume_report,
                "best_checkpoint_written": best_checkpoint_written,
                "best_checkpoint_claimed_by_resume": best_checkpoint_claimed_by_resume,
                "best_checkpoint_file_exists": best_checkpoint_file_exists,
                "best_selection_score": best_score,
                "intra_epoch_status": {
                    **throughput_payload(records_completed),
                    "reason": reason,
                    "progress_checkpoint_records": int(args.progress_checkpoint_records),
                    "progress_checkpoint_seconds": int(args.progress_checkpoint_seconds),
                    "progress_status_seconds": int(args.progress_status_seconds),
                    "last_status_write_records": int(records_completed),
                },
                "history": history,
                "claim_boundary": "Intra-epoch training status only; no checkpoint promotion or matrix score.",
                "comparable_for_matrix": False,
            }
            write_json(args.report, status_payload)

        def write_progress_checkpoint(records_completed: int, *, reason: str, local_rng_state_override: Any | None = None) -> None:
            if args.last_model_output is None or (args.progress_checkpoint_records <= 0 and args.progress_checkpoint_seconds <= 0):
                return
            args.last_model_output.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_rng_state = capture_training_rng_state(torch, np, rng)
            if local_rng_state_override is not None:
                checkpoint_rng_state["local_record_rng"] = local_rng_state_override
            progress_payload = checkpoint_payload(
                model,
                args,
                run_id=run_id,
                pid=os.getpid(),
                epoch=epoch,
                selection_score=best_score,
                selection_gate={"passed": False, "reason": reason},
                schedule=schedule,
                mask_schedule=mask_schedule,
                bottleneck_profile=bottleneck_profile,
                query_class_weights=query_class_weights,
                boundary="mid_epoch_progress_checkpoint",
                optimizer=optimizer,
                scheduler=scheduler,
                semantic_class_weights_value=class_weights,
                history=history,
                best_score=best_score,
                best_val_semantic=best_val_semantic,
                best_checkpoint_written=best_checkpoint_written,
                rng_state=checkpoint_rng_state["local_record_rng"],
                training_rng_state=checkpoint_rng_state,
                intra_epoch_progress={
                    "epoch": epoch,
                    "records_completed": int(records_completed),
                    "train_records_used": int(counters["records"]),
                    "optimizer_steps": int(counters["optimizer_steps"]),
                    "batch_records": batch_size,
                    "reason": reason,
                    "checkpoint_written_utc": utc_now(),
                    "throughput": throughput_payload(records_completed),
                    "stop_requested": dict(stop_requested),
                    "epoch_aggregate": {
                        "scope": "epoch_cumulative_including_resumed_segments",
                        "counters": dict(counters),
                        "loss_sum": float(loss_sum),
                        "gradient_control_reports": list(gradient_control_reports),
                    },
                },
            )
            atomic_torch_save(
                torch,
                progress_payload,
                args.last_model_output,
            )
            if args.checkpoint_archive_dir is not None and args.checkpoint_archive_keep > 0:
                args.checkpoint_archive_dir.mkdir(parents=True, exist_ok=True)
                archive_name = f"epoch{epoch:04d}_records{int(records_completed):06d}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.pt"
                archive_path = args.checkpoint_archive_dir / archive_name
                atomic_torch_save(torch, progress_payload, archive_path)
                prune_checkpoint_archives(args.checkpoint_archive_dir, int(args.checkpoint_archive_keep))

        def train_batch(batch: list[tuple[Any, ...]]) -> None:
            nonlocal loss_sum
            if not batch:
                return
            thing_query_count = typed_thing_query_count(
                model, args.num_queries, getattr(args, "num_stuff_queries", 0),
            )
            max_len = max(int(item[1].shape[0]) for item in batch)
            feat_dim = int(batch[0][1].shape[1])
            x_cpu = maybe_pin_full(torch, (len(batch), max_len, feat_dim), 0.0, dtype=torch.float32, pin=pin_host_batch)
            y_cpu = maybe_pin_full(torch, (len(batch), max_len), IGNORE_LABEL, dtype=torch.long, pin=pin_host_batch)
            inst_cpu = maybe_pin_full(torch, (len(batch), max_len), -1, dtype=torch.long, pin=pin_host_batch)
            semantic_weight_cpu = maybe_pin_full(torch, (len(batch), max_len), 0.0, dtype=torch.float32, pin=pin_host_batch)
            length_weight_cpu = maybe_pin_full(torch, (len(batch), max_len), 0.0, dtype=torch.float32, pin=pin_host_batch)
            mask_valid_cpu = maybe_pin_full(torch, (len(batch), max_len), False, dtype=torch.bool, pin=pin_host_batch)
            token_padding_mask_cpu = maybe_pin_full(torch, (len(batch), max_len), True, dtype=torch.bool, pin=pin_host_batch)
            token_counts = []
            segment_batch_enabled = any(item[9] is not None for item in batch)
            segment_cpu = None
            segment_padding_cpu = None
            max_segments = 0
            candidate_batch_enabled = bool(candidate_by_record and args.max_candidate_queries > 0 and args.candidate_feature_dim > 0)
            candidate_cpu = None
            candidate_padding_cpu = None
            candidate_mask_cpu = None
            if candidate_batch_enabled:
                candidate_cpu = maybe_pin_full(
                    torch, (len(batch), int(args.max_candidate_queries), int(args.candidate_feature_dim)), 0.0,
                    dtype=torch.float32, pin=pin_host_batch,
                )
                candidate_padding_cpu = maybe_pin_full(
                    torch, (len(batch), int(args.max_candidate_queries)), True,
                    dtype=torch.bool, pin=pin_host_batch,
                )
                candidate_mask_cpu = maybe_pin_full(
                    torch, (len(batch), int(args.max_candidate_queries), max_len), 0.0,
                    dtype=torch.float32, pin=pin_host_batch,
                )
            if segment_batch_enabled:
                max_segments = max(int(item[9].shape[1]) for item in batch if item[9] is not None)
                segment_cpu = maybe_pin_full(
                    torch, (len(batch), max_len, max_segments, feat_dim), 0.0,
                    dtype=torch.float32, pin=pin_host_batch,
                )
                segment_padding_cpu = maybe_pin_full(
                    torch, (len(batch), max_len, max_segments), True,
                    dtype=torch.bool, pin=pin_host_batch,
                )
            for batch_idx, (_record, x_np, y_np, inst_np, _prim_np, semantic_weights_np, length_weights_np, mask_valid_np, _page_instance_ids, segment_features_np, segment_padding_np) in enumerate(batch):
                x_tensor = torch.from_numpy(x_np)
                y_tensor = torch.from_numpy(y_np)
                inst_tensor = torch.from_numpy(inst_np)
                semantic_weight_tensor = torch.from_numpy(semantic_weights_np)
                length_weight_tensor = torch.from_numpy(length_weights_np)
                mask_valid_tensor = torch.from_numpy(mask_valid_np)
                token_count = int(y_tensor.numel())
                token_counts.append(token_count)
                x_cpu[batch_idx, : x_tensor.shape[0], :] = x_tensor
                y_cpu[batch_idx, :token_count] = y_tensor
                inst_cpu[batch_idx, :token_count] = inst_tensor
                semantic_weight_cpu[batch_idx, :token_count] = semantic_weight_tensor
                length_weight_cpu[batch_idx, :token_count] = length_weight_tensor
                mask_valid_cpu[batch_idx, :token_count] = mask_valid_tensor
                token_padding_mask_cpu[batch_idx, :token_count] = False
                if segment_cpu is not None and segment_features_np is not None and segment_padding_np is not None:
                    segment_tensor = torch.from_numpy(segment_features_np)
                    segment_padding_tensor = torch.from_numpy(segment_padding_np)
                    segment_cpu[batch_idx, :token_count, :segment_tensor.shape[1], :] = segment_tensor
                    segment_padding_cpu[batch_idx, :token_count, :segment_padding_tensor.shape[1]] = segment_padding_tensor
                if candidate_cpu is not None and candidate_padding_cpu is not None and candidate_mask_cpu is not None:
                    candidate_np, candidate_padding_np, candidate_mask_np = candidate_arrays_for_record(
                        _record,
                        candidate_by_record,
                        max_candidates=args.max_candidate_queries,
                        feature_dim=args.candidate_feature_dim,
                    )
                    if candidate_np is not None and candidate_padding_np is not None:
                        candidate_cpu[batch_idx] = torch.from_numpy(candidate_np)
                        candidate_padding_cpu[batch_idx] = torch.from_numpy(candidate_padding_np)
                        if candidate_mask_np is not None:
                            candidate_mask_cpu[batch_idx, :, : candidate_mask_np.shape[1]] = torch.from_numpy(candidate_mask_np)
            x_batch = x_cpu.to(device, non_blocking=pin_host_batch)
            y_batch = y_cpu.to(device, non_blocking=pin_host_batch)
            inst_batch = inst_cpu.to(device, non_blocking=pin_host_batch)
            semantic_weight_batch = semantic_weight_cpu.to(device, non_blocking=pin_host_batch)
            length_weight_batch = length_weight_cpu.to(device, non_blocking=pin_host_batch)
            mask_valid_batch = mask_valid_cpu.to(device, non_blocking=pin_host_batch)
            token_padding_mask = token_padding_mask_cpu.to(device, non_blocking=pin_host_batch)
            segment_batch = None if segment_cpu is None else segment_cpu.to(device, non_blocking=pin_host_batch)
            segment_padding_batch = None if segment_padding_cpu is None else segment_padding_cpu.to(device, non_blocking=pin_host_batch)
            candidate_batch = None if candidate_cpu is None else candidate_cpu.to(device, non_blocking=pin_host_batch)
            candidate_padding_batch = None if candidate_padding_cpu is None else candidate_padding_cpu.to(device, non_blocking=pin_host_batch)
            candidate_mask_batch = None if candidate_mask_cpu is None else candidate_mask_cpu.to(device, non_blocking=pin_host_batch)
            if args.geometry_augmentation:
                geometry_parameters = {
                    "flip_x": torch.rand((x_batch.shape[0], 1, 1), device=device) < 0.5,
                    "flip_y": torch.rand((x_batch.shape[0], 1, 1), device=device) < 0.5,
                    "rotations": torch.randint(0, 4, (x_batch.shape[0], 1, 1), device=device),
                    "scale": torch.empty((x_batch.shape[0], 1, 1), device=device).uniform_(
                        args.geometry_augmentation_scale_min, args.geometry_augmentation_scale_max,
                    ),
                    "shift": torch.empty((x_batch.shape[0], 1, 1, 2), device=device).uniform_(
                        -args.geometry_augmentation_translation, args.geometry_augmentation_translation,
                    ),
                }
                x_batch = augment_geometry_features(
                    torch, x_batch,
                    scale_min=args.geometry_augmentation_scale_min,
                    scale_max=args.geometry_augmentation_scale_max,
                    translation=args.geometry_augmentation_translation,
                    parameters=geometry_parameters,
                )
                if segment_batch is not None:
                    segment_batch = augment_geometry_features(
                        torch, segment_batch,
                        scale_min=args.geometry_augmentation_scale_min,
                        scale_max=args.geometry_augmentation_scale_max,
                        translation=args.geometry_augmentation_translation,
                        parameters=geometry_parameters,
                    )
                if candidate_batch is not None:
                    candidate_batch = augment_candidate_descriptor_features(torch, candidate_batch, geometry_parameters)
            style_mask = style_feature_dropout_mask(
                torch, x_batch.shape[0], x_batch.shape[-1], args.style_feature_dropout, device,
            ).to(x_batch.dtype)
            x_batch = x_batch * style_mask
            if segment_batch is not None:
                segment_batch = segment_batch * style_mask.unsqueeze(2).to(segment_batch.dtype)

            with autocast_context(torch, device, amp_dtype):
                semantic_logits_b, query_logits_b, mask_logits_b, quality_logits_b, identity_embeddings_b = model(
                    x_batch, token_padding_mask=token_padding_mask,
                    segment_features=segment_batch, segment_padding_mask=segment_padding_batch,
                    candidate_features=candidate_batch, candidate_padding_mask=candidate_padding_batch,
                    candidate_token_masks=candidate_mask_batch,
                    return_quality=True, return_identity=True,
                )
            runtime_model = getattr(model, "_orig_mod", model)
            ownership_logits_b = runtime_model.last_ownership_logits
            query_admission_logits_b = runtime_model.last_query_admission_logits
            sq_rq_outputs_b = runtime_model.last_sq_rq_outputs
            router_diagnostics_b = runtime_model.last_router_diagnostics
            query_seed_diagnostics_b = getattr(runtime_model, "last_query_seed_diagnostics", None)
            family_seed_logits_b = getattr(runtime_model, "last_family_seed_logits", None)
            token_offsets_b = getattr(runtime_model, "last_token_offsets", None)
            token_affinity_b = getattr(runtime_model, "last_token_affinity_embeddings", None)
            route_logits_b = getattr(runtime_model, "last_explicit_route_logits", None)
            geometry_aux_outputs = runtime_model.last_aux_outputs if args.geometry_decoder_mode == "geometry_v2" else []
            losses = []
            task_loss_rows: dict[str, list[Any]] = {
                "semantic": [],
                "query_mask_quality": [],
                "teacher": [],
                "identity": [],
                "ownership": [],
                "router": [],
            }
            if router_diagnostics_b is not None:
                router_load_balance = router_diagnostics_b["load_balance_cv_squared"]
                router_switch_balance = router_diagnostics_b["switch_load_balance_loss"]
                router_z_loss = router_diagnostics_b["router_z_loss"]
                router_balance_weight = auxiliary_loss_weight_for_active_experts(
                    args.router_load_balance_loss_weight,
                    args._active_loss_experts,
                    set(LOSS_EXPERT_GROUPS),
                    joint_only=True,
                )
                router_z_weight = auxiliary_loss_weight_for_active_experts(
                    args.router_z_loss_weight,
                    args._active_loss_experts,
                    set(LOSS_EXPERT_GROUPS),
                    joint_only=True,
                )
                if router_balance_weight > 0.0 or router_z_weight > 0.0:
                    task_loss_rows["router"].append(
                        router_balance_weight * router_switch_balance
                        + router_z_weight * router_z_loss
                    )
                counters["router_load_balance_loss_sum"] += float(router_load_balance.detach().item())
                counters["router_switch_load_balance_loss_sum"] += float(router_switch_balance.detach().item())
                counters["router_z_loss_sum"] += float(router_z_loss.detach().item())
                counters["router_diagnostic_batches"] += 1
                counters["router_routing_entropy_sum"] += float(router_diagnostics_b["routing_entropy"].detach().item())
                for expert_index, probability in enumerate(router_diagnostics_b["mean_expert_probability"].detach().tolist()):
                    counters[f"router_expert_probability_{expert_index}"] += float(probability)
                for expert_index, fraction in enumerate(router_diagnostics_b["assignment_fraction"].detach().tolist()):
                    counters[f"router_expert_assignment_fraction_{expert_index}"] += float(fraction)
            branch_router_load_balance = query_logits_b.sum() * 0.0
            branch_diagnostics = getattr(runtime_model, "last_branch_router_diagnostics", {}) or {}
            enabled_branch_diagnostics = [
                value for value in branch_diagnostics.values() if value.get("enabled") and "load_balance_cv_squared" in value
            ]
            if enabled_branch_diagnostics:
                branch_router_load_balance = torch.stack(
                    [value["load_balance_cv_squared"] for value in enabled_branch_diagnostics]
                ).mean()
                branch_router_switch_balance = torch.stack(
                    [value["switch_load_balance_loss"] for value in enabled_branch_diagnostics]
                ).mean()
                branch_router_z_loss = torch.stack(
                    [value["router_z_loss"] for value in enabled_branch_diagnostics]
                ).mean()
                router_balance_weight = auxiliary_loss_weight_for_active_experts(
                    args.router_load_balance_loss_weight,
                    args._active_loss_experts,
                    set(LOSS_EXPERT_GROUPS),
                    joint_only=True,
                )
                router_z_weight = auxiliary_loss_weight_for_active_experts(
                    args.router_z_loss_weight,
                    args._active_loss_experts,
                    set(LOSS_EXPERT_GROUPS),
                    joint_only=True,
                )
                if router_balance_weight > 0.0 or router_z_weight > 0.0:
                    task_loss_rows["router"].append(
                        router_balance_weight * branch_router_switch_balance
                        + router_z_weight * branch_router_z_loss
                    )
                counters["branch_router_diagnostic_batches"] += 1
                counters["branch_router_load_balance_loss_sum"] += float(branch_router_load_balance.detach().item())
                counters["branch_router_switch_load_balance_loss_sum"] += float(branch_router_switch_balance.detach().item())
                counters["branch_router_z_loss_sum"] += float(branch_router_z_loss.detach().item())
                overflow = torch.stack([value["overflow_assignments"] for value in enabled_branch_diagnostics]).sum()
                usage_gate = torch.stack([value["usage_gate_passed"] for value in enabled_branch_diagnostics]).all()
                counters["branch_router_overflow_assignments"] += int(overflow.detach().item())
                counters["branch_router_usage_gate_pass_batches"] += int(usage_gate.detach().item())
            identity_rows: list[Any] = []
            identity_ids: list[list[str | None]] = []
            identity_valid_rows: list[Any] = []
            identity_window_indices: list[int] = []
            identity_page_ids: list[str] = []
            stuff_union_rows: list[tuple[str, int, Any, Any]] = []
            batch_records_used = 0
            for batch_idx, (record, _x_np, _y_np, _inst_np, _prim_np, _semantic_weights_np, _length_weights_np, _mask_valid_np, page_instance_ids, _segment_features_np, _segment_padding_np) in enumerate(batch):
                token_count = int(token_counts[batch_idx])
                availability = task_availability(record)
                y = y_batch[batch_idx, :token_count]
                inst = inst_batch[batch_idx, :token_count]
                semantic_weights = semantic_weight_batch[batch_idx, :token_count]
                length_weights = length_weight_batch[batch_idx, :token_count]
                mask_valid = mask_valid_batch[batch_idx, :token_count]
                semantic_logits = semantic_logits_b[batch_idx, :token_count, :].float()
                query_logits = query_logits_b[batch_idx].float()
                mask_logits = mask_logits_b[batch_idx, :, :token_count].float()
                quality_logits = quality_logits_b[batch_idx].float()
                admission_logits = query_admission_logits_b[batch_idx].float()
                family_seed_logits = None if family_seed_logits_b is None else family_seed_logits_b[batch_idx, :token_count].float()
                token_offsets = None if token_offsets_b is None else token_offsets_b[batch_idx, :token_count].float()
                token_affinity = None if token_affinity_b is None else token_affinity_b[batch_idx, :token_count].float()
                route_logits = None if route_logits_b is None else route_logits_b[batch_idx : batch_idx + 1, :token_count].float()
                if candidate_padding_batch is not None and candidate_mask_batch is not None:
                    valid_candidates = ~candidate_padding_batch[batch_idx].bool()
                    if bool(valid_candidates.any().item()):
                        counters["candidate_records_with_candidates"] += 1
                        counters["candidate_valid_total"] += int(valid_candidates.sum().item())
                        counters["candidate_mask_token_total"] += int(candidate_mask_batch[batch_idx, valid_candidates, :token_count].sum().item())
                target_labels, target_masks, target_weights, positives, target_diag = component_targets_schema_v2(
                    torch, y, page_instance_ids, mask_valid, length_weights, args.num_queries,
                    partial_component_policy=args.partial_component_policy,
                    partial_component_min_tokens=args.partial_component_min_tokens,
                )
                update_target_diagnostics(counters, target_diag)
                selected_primitive_indices = query_selected_primitive_indices(
                    torch,
                    query_seed_diagnostics_b,
                    batch_index=batch_idx,
                    num_queries=args.num_queries,
                    token_count=token_count,
                    device=device,
                )
                if availability["rq"]:
                    q_labels, q_masks, positives, matched = match_component_queries(
                        torch, query_logits, mask_logits, target_labels, target_masks, args.num_queries,
                        matching=(args.train_component_matching or args.component_matching), primitive_weights=target_weights,
                        thing_query_count=thing_query_count, typed_stuff_slots=bool(getattr(runtime_model, "typed_stuff_slots", False)),
                        selected_primitive_indices=selected_primitive_indices,
                    )
                    if (
                        (args.train_component_matching or args.component_matching) == "greedy_gpu_train"
                        and int(args.train_matching_exact_audit_interval) > 0
                        and int(counters["records"]) % int(args.train_matching_exact_audit_interval) == 0
                    ):
                        reference_labels, _reference_masks, _positive, _matched = match_component_queries(
                            torch, query_logits, mask_logits, target_labels, target_masks, args.num_queries,
                            matching="hungarian_cpu", primitive_weights=target_weights,
                            thing_query_count=thing_query_count, typed_stuff_slots=bool(getattr(runtime_model, "typed_stuff_slots", False)),
                            selected_primitive_indices=selected_primitive_indices,
                        )
                        churn = float(matching_assignment_churn(torch, q_labels, reference_labels).item())
                        counters["train_matching_exact_audits"] += 1
                        counters["train_matching_assignment_churn_sum"] += churn
                        if churn > float(args.train_matching_max_assignment_churn):
                            raise RuntimeError(f"GPU matcher exact audit failed: churn={churn:.6f}")
                else:
                    q_labels = torch.full((args.num_queries,), IGNORE_LABEL, dtype=torch.long, device=device)
                    q_masks = torch.zeros((args.num_queries, token_count), dtype=torch.float32, device=device)
                    positives = matched = 0
                matched_ids, matched_valid = matched_query_page_instance_ids(
                    torch, q_labels, q_masks, page_instance_ids, mask_valid
                )
                if availability["identity"]:
                    identity_rows.append(identity_embeddings_b[batch_idx])
                    identity_ids.append(matched_ids)
                    identity_valid_rows.append(matched_valid)
                    identity_window_indices.append(parse_int(record.get("window_index"), parse_int(record.get("window_start"), 0)))
                    identity_page_ids.append(str(record.get("original_record_id") or record.get("record_id")))
                if bool(getattr(runtime_model, "typed_stuff_slots", False)):
                    stuff_union_rows.append((
                        str(record.get("original_record_id") or record.get("record_id")),
                        parse_int(record.get("window_index"), 0),
                        torch.from_numpy(_prim_np).to(device),
                        mask_logits[int(thing_query_count):].clone(),
                    ))
                update_supervised_component_proxy(
                    torch,
                    counters,
                    query_logits,
                    mask_logits,
                    q_labels,
                    q_masks,
                    quality_logits if availability["quality"] else None,
                    primitive_weights=target_weights,
                    rq_available=availability["rq"],
                    ownership_logits=(
                        None
                        if ownership_logits_b is None
                        else ownership_logits_b[batch_idx, :token_count].float()
                    ),
                    deployment_min_query_score=args.deployment_min_query_score,
                    deployment_mask_threshold=args.deployment_mask_threshold,
                )
                update_family_seed_proxy(torch, counters, family_seed_logits, y, mask_valid)
                pos = q_labels != IGNORE_LABEL
                semantic = weighted_semantic_loss_schema_v2(
                    torch, semantic_logits, y, semantic_weights, ce_sem.weight,
                    args.semantic_label_smoothing,
                ) if availability["semantic"] else None
                query, objectness = rq_query_supervision_losses(
                    torch,
                    ce_query,
                    query_logits,
                    q_labels,
                    rq_available=availability["rq"],
                    admission_logits=admission_logits,
                    positive_weight=schedule["query_objectness_positive_weight"],
                    negative_weight=schedule["query_objectness_negative_weight"],
                    positive_margin_floor_loss_weight=schedule["query_objectness_positive_margin_floor_loss_weight"],
                    positive_margin_floor=schedule["query_objectness_positive_margin_floor"],
                    negative_margin_ceiling_loss_weight=schedule["query_objectness_negative_margin_ceiling_loss_weight"],
                    negative_margin_ceiling=schedule["query_objectness_negative_margin_ceiling"],
                )
                rq_sq_quality = rq_sq_quality_calibration_loss(
                    torch, quality_logits, mask_logits, q_labels, q_masks, target_weights,
                    ranking_weight=args.rq_sq_quality_ranking_weight,
                    ranking_margin=args.rq_sq_quality_ranking_margin,
                    ranking_top_k=args.rq_sq_quality_ranking_top_k,
                    hard_negative_weight=args.rq_sq_quality_hard_negative_weight,
                    unmatched_ceiling_weight=args.rq_sq_quality_unmatched_ceiling_weight,
                    unmatched_ceiling_probability=args.rq_sq_quality_unmatched_ceiling_probability,
                    foreground_scores=rq_sq_quality_deployment_scores(
                        torch, query_logits, quality_logits, mask_logits
                    )[1],
                    query_logits=query_logits,
                    positive_quality_floor_labels=args._hard_recall_label_set,
                    positive_quality_floor=(
                        args.hard_recall_quality_target_floor
                        if args.quality_deployment_expert_weight > 0.0
                        else 0.0
                    ),
                    ownership_logits=(
                        None
                        if ownership_logits_b is None
                        else ownership_logits_b[batch_idx, :token_count].float()
                    ),
                    mask_threshold=args.deployment_mask_threshold,
                    soft_target_weight=args.quality_soft_target_weight,
                ) if availability["quality"] else None
                family_recall_focus = (
                    family_recall_focus_loss(
                        torch,
                        query_logits,
                        admission_logits,
                        quality_logits if availability["quality"] else None,
                        mask_logits,
                        q_labels,
                        q_masks,
                        family=args.family_recall_focus,
                        admission_floor=args.family_recall_admission_floor,
                        mask_positive_prob_floor=args.family_recall_mask_prob_floor,
                        quality_floor=args.family_recall_quality_floor,
                        primitive_weights=target_weights,
                    )
                    if availability["rq"] and args.family_recall_focus and args.family_recall_loss_weight > 0.0
                    else None
                )
                rq_admission_hard_recall = (
                    hard_recall_admission_margin_loss(
                        torch,
                        admission_logits,
                        q_labels,
                        args._hard_recall_label_set,
                        probability_floor=args.hard_recall_admission_floor,
                    )
                    if availability["rq"] and args.rq_admission_expert_weight > 0.0
                    else None
                )
                mask_hard_recall_floor = (
                    hard_recall_mask_floor_loss(
                        torch,
                        mask_logits,
                        q_labels,
                        q_masks,
                        args._hard_recall_label_set,
                        probability_floor=args.hard_recall_mask_prob_floor,
                        primitive_weights=target_weights,
                    )
                    if availability["rq"] and args.mask_recall_expert_weight > 0.0
                    else None
                )
                quality_deployment_floor = (
                    hard_recall_quality_deployment_loss(
                        torch,
                        query_logits,
                        quality_logits if availability["quality"] else None,
                        q_labels,
                        args._hard_recall_label_set,
                        deployment_floor=args.hard_recall_deployment_floor,
                    )
                    if availability["quality"] and args.quality_deployment_expert_weight > 0.0
                    else None
                )
                if ownership_logits_b is None or not availability["ownership"]:
                    ownership_loss = None
                else:
                    owner_target = ownership_targets(torch, q_masks, q_labels, mask_valid)
                    ownership_ce = ownership_cross_entropy(torch, ownership_logits_b[batch_idx, :token_count].float(), owner_target, length_weights)
                    ownership_consistency = ownership_mask_consistency_loss(
                        torch, ownership_logits_b[batch_idx, :token_count].float(), mask_logits, q_labels, mask_valid,
                        no_object_label=IGNORE_LABEL, primitive_weights=length_weights,
                    )
                    ownership_loss = ownership_ce + args.ownership_mask_consistency_loss_weight * ownership_consistency
                if not availability["rq"]:
                    mask_loss = None
                elif positives == 0:
                    mask_loss = mask_logits.sum() * 0.0
                else:
                    mask = mask_logits[pos]
                    target = q_masks[pos]
                    mask_labels = q_labels[pos]
                    mask_loss = weighted_mask_loss(
                        torch,
                        mask,
                        target,
                        mask_labels,
                        query_class_weights,
                        positive_weight=mask_schedule["mask_positive_weight"],
                        negative_weight=mask_schedule["mask_negative_weight"],
                        focal_gamma=args.mask_focal_gamma,
                        area_ratio_loss_weight=mask_schedule["mask_area_ratio_loss_weight"],
                        area_overcoverage_weight=mask_schedule["mask_area_overcoverage_weight"],
                        tversky_loss_weight=mask_schedule["mask_tversky_loss_weight"],
                        tversky_alpha=args.mask_tversky_alpha,
                        tversky_beta=args.mask_tversky_beta,
                        positive_prob_floor_loss_weight=mask_schedule["mask_positive_prob_floor_loss_weight"],
                        positive_prob_floor=args.mask_positive_prob_floor,
                        primitive_weights=target_weights,
                    )
                    cached_neighbor_indices = getattr(runtime_model, "last_geometry_neighbor_indices", None)
                    cached_neighbor_valid = getattr(runtime_model, "last_geometry_neighbor_valid", None)
                    mask_loss = mask_loss + args.mask_geometry_connectivity_loss_weight * geometry_mask_connectivity_loss(
                        torch,
                        mask,
                        x_batch[batch_idx, :token_count],
                        target,
                        neighbor_indices=(None if cached_neighbor_indices is None else cached_neighbor_indices[batch_idx, :token_count]),
                        neighbor_valid=(None if cached_neighbor_valid is None else cached_neighbor_valid[batch_idx, :token_count]),
                    )
                if mask_loss is not None and args.unmatched_mask_negative_loss_weight > 0.0:
                    hard_negative_mask = unmatched_query_empty_mask_loss(
                        torch,
                        mask_logits,
                        q_labels,
                        admission_logits,
                        top_k=args.unmatched_mask_negative_top_k,
                        primitive_weights=target_weights,
                    )
                    mask_loss = mask_loss + args.unmatched_mask_negative_loss_weight * hard_negative_mask
                    counters["unmatched_mask_negative_loss_sum"] += float(hard_negative_mask.detach().item())
                teacher_loss = query_logits.sum() * 0.0
                teacher_pos_count = 0
                if availability["teacher"] and teacher_by_record and args.teacher_loss_weight > 0.0:
                    t_labels, t_masks, teacher_pos_count, t_diag = window_teacher_targets(torch, record, teacher_by_record, y.numel(), args.num_queries, device)
                    update_teacher_diagnostics(counters, t_diag, prefix="teacher")
                    if int(t_diag.get("teacher_components_kept", 0)) > 0:
                        tq_labels, tq_masks, identity_diag = align_teacher_to_gt_queries(torch, q_labels, q_masks, t_labels, t_masks)
                        counters.update(identity_diag)
                        tpos = tq_labels != IGNORE_LABEL
                        update_teacher_match_conflicts(torch, counters, q_labels, q_masks, tq_labels, tq_masks)
                        teacher_negative, teacher_negative_count = teacher_hard_negative_objectness_loss(
                            torch, ce_query, query_logits, mask_logits, q_labels, t_labels, t_masks
                        )
                        counters["teacher_hard_negative_objectness_queries"] += teacher_negative_count
                        if int(tpos.sum().item()) > 0:
                            teacher_query = teacher_positive_query_loss(torch, ce_query, query_logits, tq_labels)
                            teacher_mask = weighted_mask_loss(
                                torch,
                                mask_logits[tpos],
                                tq_masks[tpos],
                                tq_labels[tpos],
                                query_class_weights,
                                positive_weight=mask_schedule["mask_positive_weight"],
                                negative_weight=mask_schedule["mask_negative_weight"],
                                focal_gamma=args.mask_focal_gamma,
                                area_ratio_loss_weight=mask_schedule["mask_area_ratio_loss_weight"],
                                area_overcoverage_weight=mask_schedule["mask_area_overcoverage_weight"],
                                tversky_loss_weight=mask_schedule["mask_tversky_loss_weight"],
                                tversky_alpha=args.mask_tversky_alpha,
                                tversky_beta=args.mask_tversky_beta,
                                positive_prob_floor_loss_weight=mask_schedule["mask_positive_prob_floor_loss_weight"],
                                positive_prob_floor=args.mask_positive_prob_floor,
                                primitive_weights=target_weights,
                            )
                            teacher_loss = args.teacher_query_loss_weight * (teacher_query + teacher_negative) + args.teacher_mask_loss_weight * teacher_mask
                        else:
                            teacher_loss = args.teacher_query_loss_weight * teacher_negative
                geometry_aux_loss = (semantic if semantic is not None else query_logits).sum() * 0.0 if availability["rq"] else None
                sq_base_loss = None
                if sq_rq_outputs_b is not None and availability["sq"]:
                    sq_base_loss = weighted_semantic_loss_schema_v2(
                        torch,
                        sq_rq_outputs_b["semantic_base_logits"][batch_idx, :token_count].float(),
                        y,
                        semantic_weights,
                        ce_sem.weight,
                        args.semantic_label_smoothing,
                    )
                    if semantic is not None:
                        semantic = semantic + float(getattr(args, "sq_rq_base_semantic_loss_weight", 0.25)) * sq_base_loss
                if availability["rq"] and len(geometry_aux_outputs) > 1:
                    record_aux = [
                        {
                            key: value[batch_idx : batch_idx + 1, ..., :token_count] if key == "mask_logits" else value[batch_idx : batch_idx + 1]
                            for key, value in layer.items() if key != "layer_index" and value is not None
                        }
                        for layer in geometry_aux_outputs[:-1]
                    ]
                    geometry_aux_loss, _geometry_aux_diag = geometry_v2_auxiliary_loss(
                        torch, record_aux, target_labels, target_masks,
                        num_queries=args.num_queries, primitive_weights=target_weights, query_class_weights=query_class_weights,
                        matching=(args.train_component_matching or args.component_matching),
                        thing_query_count=thing_query_count, typed_stuff_slots=bool(getattr(runtime_model, "typed_stuff_slots", False)),
                        selected_primitive_indices=selected_primitive_indices,
                    )
                content_anchor_loss = (
                    family_seed_loss(torch, family_seed_logits, y, mask_valid, length_weights)
                    if availability["rq"] and family_seed_logits is not None
                    else None
                )
                component_seed = (
                    component_seed_loss(torch, component_seed_logits, y, mask_valid, length_weights)
                    if availability["rq"] and component_seed_logits is not None
                    else None
                )
                offset_vote = (
                    token_offset_vote_loss(torch, token_offsets, x_batch[batch_idx, :token_count], q_labels, q_masks, length_weights)
                    if availability["rq"]
                    else None
                )
                affinity = (
                    token_affinity_component_loss(torch, token_affinity, q_labels, q_masks, length_weights)
                    if availability["rq"]
                    else None
                )
                candidate_prior_loss = candidate_mask_prior_loss(
                    torch,
                    mask_logits,
                    None if candidate_mask_batch is None else candidate_mask_batch[batch_idx, :, :token_count],
                    None if candidate_padding_batch is None else candidate_padding_batch[batch_idx],
                    thing_query_count=thing_query_count,
                    primitive_weights=length_weights,
                ) if availability["rq"] else None
                route_classification = (
                    explicit_route_classification_loss(
                        torch,
                        route_logits,
                        y.unsqueeze(0),
                        mask_valid.unsqueeze(0),
                    )
                    if float(args.route_classification_loss_weight) > 0.0
                    else None
                )
                loss_terms = {
                        "semantic": semantic,
                        "query": query,
                        "query_objectness": objectness,
                        "rq_admission_hard_recall": rq_admission_hard_recall,
                        "quality_calibration": rq_sq_quality,
                        "quality_deployment_floor": quality_deployment_floor,
                        "family_recall_focus": family_recall_focus,
                        "mask": mask_loss,
                        "mask_hard_recall_floor": mask_hard_recall_floor,
                        "ownership": ownership_loss,
                        "teacher": teacher_loss,
                        "geometry_aux": geometry_aux_loss,
                        "content_anchor": content_anchor_loss,
                        "component_seed": component_seed,
                        "offset_vote": offset_vote,
                        "affinity": affinity,
                        "candidate_mask_prior": candidate_prior_loss,
                        "route_classification": route_classification,
                    }
                loss_weights = {
                        "semantic": args.semantic_loss_weight,
                        "query": args.query_loss_weight,
                        "query_objectness": schedule["query_objectness_loss_weight"],
                        "rq_admission_hard_recall": args.rq_admission_expert_weight,
                        "quality_calibration": args.rq_sq_quality_calibration_loss_weight,
                        "quality_deployment_floor": args.quality_deployment_expert_weight,
                        "family_recall_focus": args.family_recall_loss_weight,
                        "mask": args.mask_loss_weight,
                        "mask_hard_recall_floor": args.mask_recall_expert_weight,
                        "ownership": args.ownership_loss_weight,
                        "teacher": args.teacher_loss_weight,
                        "geometry_aux": 0.5,
                        "content_anchor": args.content_anchor_loss_weight,
                        "component_seed": args.component_seed_loss_weight,
                        "offset_vote": args.offset_vote_loss_weight,
                        "affinity": args.affinity_loss_weight,
                        "candidate_mask_prior": args.candidate_mask_prior_loss_weight,
                        "route_classification": args.route_classification_loss_weight,
                    }
                loss_weights = loss_weights_for_active_experts(loss_weights, args._active_loss_experts)
                update_loss_expert_counters(counters, loss_terms, loss_weights, args._active_loss_experts)
                loss, _loss_diagnostic = compose_task_loss_diagnostic(
                    loss_terms,
                    loss_weights,
                    pcgrad=args.gradient_control == "pcgrad",
                )
                if not torch.isfinite(loss):
                    counters["nonfinite_loss_skipped"] += 1
                    continue
                losses.append(loss)
                if semantic is not None and loss_weights["semantic"] > 0.0:
                    task_loss_rows["semantic"].append(loss_weights["semantic"] * semantic)
                query_mask_quality_terms = [
                    coefficient * value
                    for coefficient, value in (
                        (loss_weights["query"], query),
                        (loss_weights["query_objectness"], objectness),
                        (loss_weights["quality_calibration"], rq_sq_quality),
                        (loss_weights["rq_admission_hard_recall"], rq_admission_hard_recall),
                        (loss_weights["quality_deployment_floor"], quality_deployment_floor),
                        (loss_weights["family_recall_focus"], family_recall_focus),
                        (loss_weights["mask"], mask_loss),
                        (loss_weights["mask_hard_recall_floor"], mask_hard_recall_floor),
                        (loss_weights["geometry_aux"], geometry_aux_loss),
                        (loss_weights["content_anchor"], content_anchor_loss),
                        (loss_weights["component_seed"], component_seed),
                        (loss_weights["offset_vote"], offset_vote),
                        (loss_weights["affinity"], affinity),
                    )
                    if value is not None and coefficient > 0.0
                ]
                if query_mask_quality_terms:
                    task_loss_rows["query_mask_quality"].append(sum(query_mask_quality_terms))
                teacher_aux_terms = [
                    coefficient * value
                    for coefficient, value in (
                        (loss_weights["teacher"], teacher_loss if availability["teacher"] and teacher_by_record else None),
                        (loss_weights["candidate_mask_prior"], candidate_prior_loss),
                    )
                    if value is not None and coefficient > 0.0
                ]
                if teacher_aux_terms:
                    task_loss_rows["teacher"].append(sum(teacher_aux_terms))
                if ownership_loss is not None and loss_weights["ownership"] > 0.0:
                    task_loss_rows["ownership"].append(loss_weights["ownership"] * ownership_loss)
                if route_classification is not None and loss_weights["route_classification"] > 0.0:
                    task_loss_rows["router"].append(loss_weights["route_classification"] * route_classification)
                batch_records_used += 1
                counters["records"] += 1
                counters["tokens"] += token_count
                counters["labeled_tokens"] += int((y != IGNORE_LABEL).sum().item())
                counters["target_components"] += int(positives)
                counters["matched_components"] += int(matched)
                counters["semantic_loss_sum"] += float(semantic.detach().item()) if semantic is not None else 0.0
                counters["query_loss_sum"] += float(query.detach().item()) if query is not None else 0.0
                counters["query_objectness_loss_sum"] += float(objectness.detach().item()) if objectness is not None else 0.0
                counters["rq_sq_quality_calibration_loss_sum"] += float(rq_sq_quality.detach().item()) if rq_sq_quality is not None else 0.0
                counters["family_recall_focus_loss_sum"] += float(family_recall_focus.detach().item()) if family_recall_focus is not None else 0.0
                counters["rq_admission_hard_recall_loss_sum"] += float(rq_admission_hard_recall.detach().item()) if rq_admission_hard_recall is not None else 0.0
                counters["mask_hard_recall_floor_loss_sum"] += float(mask_hard_recall_floor.detach().item()) if mask_hard_recall_floor is not None else 0.0
                counters["quality_deployment_floor_loss_sum"] += float(quality_deployment_floor.detach().item()) if quality_deployment_floor is not None else 0.0
                counters["mask_loss_sum"] += float(mask_loss.detach().item()) if mask_loss is not None else 0.0
                counters["teacher_loss_sum"] += float(teacher_loss.detach().item())
                counters["geometry_aux_loss_sum"] += float(geometry_aux_loss.detach().item()) if geometry_aux_loss is not None else 0.0
                counters["content_anchor_loss_sum"] += float(content_anchor_loss.detach().item()) if content_anchor_loss is not None else 0.0
                counters["component_seed_loss_sum"] += float(component_seed.detach().item()) if component_seed is not None else 0.0
                counters["offset_vote_loss_sum"] += float(offset_vote.detach().item()) if offset_vote is not None else 0.0
                counters["affinity_loss_sum"] += float(affinity.detach().item()) if affinity is not None else 0.0
                counters["candidate_mask_prior_loss_sum"] += float(candidate_prior_loss.detach().item()) if candidate_prior_loss is not None else 0.0
                counters["route_classification_loss_sum"] += float(route_classification.detach().item()) if route_classification is not None else 0.0
                counters["ownership_loss_sum"] += float(ownership_loss.detach().item()) if ownership_loss is not None else 0.0
                counters["teacher_positive_components"] += int(teacher_pos_count)
                loss_sum += float(loss.detach().item())

            if losses:
                identity_aux_weight = auxiliary_loss_weight_for_active_experts(
                    args.identity_loss_weight, args._active_loss_experts, {"topology_merge"}
                )
                if identity_aux_weight > 0.0 and identity_rows:
                    identity_loss, identity_diag = adjacent_window_identity_loss(
                        torch.stack(identity_rows), identity_ids, torch.stack(identity_valid_rows),
                        identity_window_indices, identity_page_ids,
                        temperature=args.identity_temperature, negative_margin=args.identity_negative_margin,
                    )
                    task_loss_rows["identity"].append(identity_aux_weight * identity_loss)
                    counters["identity_positive_assignment_directions"] += int(identity_diag["positive_assignment_directions"])
                    counters["identity_negative_pairs"] += int(identity_diag["negative_pairs"])
                    counters["identity_loss_sum"] += float(identity_loss.detach().item())
                    counters["identity_batches"] += 1
                specialization = moe_branch_specialization_loss(torch, branch_diagnostics)
                specialization_weight = auxiliary_loss_weight_for_active_experts(
                    args.moe_branch_specialization_loss_weight,
                    args._active_loss_experts,
                    set(LOSS_EXPERT_GROUPS),
                    joint_only=True,
                )
                if specialization is not None and specialization_weight > 0.0:
                    task_loss_rows["router"].append(specialization_weight * specialization)
                    counters["moe_branch_specialization_loss_sum"] += float(specialization.detach().item())
                stuff_union_loss, stuff_union_pairs = stuff_overlap_union_consistency_loss(torch, stuff_union_rows)
                stuff_union_weight = auxiliary_loss_weight_for_active_experts(
                    args.stuff_overlap_union_loss_weight, args._active_loss_experts, {"sq_semantic"}
                )
                if stuff_union_loss is not None and stuff_union_weight > 0.0:
                    task_loss_rows["query_mask_quality"].append(stuff_union_weight * stuff_union_loss)
                    counters["stuff_overlap_union_loss_sum"] += float(stuff_union_loss.detach().item())
                    counters["stuff_overlap_union_pairs"] += int(stuff_union_pairs)
                batch_task_losses = {
                    task: torch.stack(values).mean() if values else None
                    for task, values in task_loss_rows.items()
                }
                task_specific_losses = dict(batch_task_losses)
                query_specific = batch_task_losses.get("query_mask_quality")
                teacher_specific = batch_task_losses.get("teacher")
                if teacher_specific is not None:
                    task_specific_losses["query_mask_quality"] = (
                        teacher_specific if query_specific is None else query_specific + teacher_specific
                    )
                gradient_control_reports.append(production_gradient_step(
                    torch,
                    model,
                    optimizer,
                    batch_task_losses,
                    scheduler=scheduler,
                    mode=args.gradient_control,
                    task_specific_losses=task_specific_losses,
                ))
                # Do not keep the previous forward graph alive through model-side
                # diagnostics until the next batch starts.
                runtime_model.last_aux_outputs = []
                runtime_model.last_sq_rq_outputs = None
                runtime_model.last_ownership_logits = None
                runtime_model.last_router_diagnostics = None
                runtime_model.last_typed_outputs = None
                runtime_model.last_family_seed_logits = None
            counters["optimizer_steps"] += int(bool(losses))
            counters["records_batched"] += batch_records_used

        pending_batch = []
        pending_page_id: str | None = None
        skipped_for_resume = 0
        train_records_source = iter_jsonl(args.train, train_limit)
        for record in train_records_source:
            if epoch == start_epoch and skipped_for_resume < resume_skip_records:
                skipped_for_resume += 1
                continue
            train_records_source = itertools.chain([record], train_records_source)
            break
        else:
            train_records_source = iter(())
        train_arrays_iter = prefetch_training_arrays(
            train_records_source,
            max_prefetch=args.train_prefetch_records,
            workers=args.train_prefetch_workers,
            max_tokens=args.max_tokens_per_record,
            rng=rng,
            input_level="primitive",
            num_queries=args.num_queries,
        )
        for record, arrays, rng_state_after_record, parse_error_type, parse_error_message in train_arrays_iter:
            rng_state_before_record = rng.getstate()
            records_seen_this_epoch += 1
            if parse_error_type is not None:
                raise RuntimeError(f"training record parse failed: {parse_error_type}: {parse_error_message}")
            rng.setstate(rng_state_after_record)
            if arrays is None:
                continue
            x_np, y_np, inst_np, prim_np, semantic_weights_np, length_weights_np, mask_valid_np, page_instance_ids, segment_features_np, segment_padding_np = arrays
            current_page_id = teacher_record_key(record)
            flushed_page_batch = False
            if should_flush_page_aware_batch(len(pending_batch), pending_page_id, current_page_id, batch_size):
                train_batch(pending_batch)
                pending_batch = []
                flushed_page_batch = True
            if flushed_page_batch:
                completed_records = page_aware_checkpoint_completed_records(records_seen_this_epoch, current_record_pending=True)
                checkpoint_due_by_records = args.progress_checkpoint_records > 0 and completed_records >= next_progress_checkpoint
                checkpoint_due_by_time = args.progress_checkpoint_seconds > 0 and (time.perf_counter() - last_progress_checkpoint_time) >= int(args.progress_checkpoint_seconds)
                if checkpoint_due_by_records or checkpoint_due_by_time:
                    write_progress_checkpoint(
                        completed_records,
                        reason="mid_epoch_progress_checkpoint_by_records" if checkpoint_due_by_records else "mid_epoch_progress_checkpoint_by_seconds",
                        local_rng_state_override=rng_state_before_record,
                    )
                    last_progress_checkpoint_time = time.perf_counter()
                    if args.progress_checkpoint_records > 0:
                        while next_progress_checkpoint <= completed_records:
                            next_progress_checkpoint += int(args.progress_checkpoint_records)
                status_due_by_records = args.progress_status_records > 0 and completed_records >= next_status_records
                status_due_by_time = args.progress_status_seconds > 0 and (time.perf_counter() - last_status_time) >= int(args.progress_status_seconds)
                if status_due_by_records or status_due_by_time:
                    write_running_status(
                        completed_records,
                        reason="periodic_intra_epoch_progress_by_records" if status_due_by_records else "periodic_intra_epoch_progress_by_seconds",
                    )
                    last_status_write = completed_records
                    last_status_time = time.perf_counter()
                    next_status_records = next_progress_threshold(completed_records, args.progress_status_records)
                if stop_requested["requested"]:
                    write_progress_checkpoint(
                        completed_records,
                        reason="graceful_stop_requested_mid_epoch_checkpoint",
                        local_rng_state_override=rng_state_before_record,
                    )
                    write_running_status(completed_records, reason="graceful_stop_requested_mid_epoch_checkpoint_written")
                    final_status = "stopped_gracefully_checkpointed"
                    payload = read_json_file(args.report)
                    payload["status"] = final_status
                    payload["stop_requested"] = dict(stop_requested)
                    payload["required_next_action"] = "resume from last_model_output with --resume-checkpoint and --resume-optimizer"
                    write_json(args.report, payload)
                    break
            pending_batch.append((record, x_np, y_np, inst_np, prim_np, semantic_weights_np, length_weights_np, mask_valid_np, page_instance_ids, segment_features_np, segment_padding_np))
            pending_page_id = current_page_id
        if stop_requested["requested"]:
            break
        train_batch(pending_batch)
        write_progress_checkpoint(records_seen_this_epoch, reason="epoch_train_scan_complete_before_validation")
        write_running_status(records_seen_this_epoch, reason="epoch_train_scan_complete_before_validation")
        if stop_requested["requested"]:
            final_status = "stopped_gracefully_checkpointed"
            payload = read_json_file(args.report)
            payload["status"] = final_status
            payload["stop_requested"] = dict(stop_requested)
            payload["required_next_action"] = "resume from last_model_output with --resume-checkpoint and --resume-optimizer"
            write_json(args.report, payload)
            break
        epoch_train_elapsed = max(time.perf_counter() - epoch_started, 1e-9)
        epoch_throughput = throughput_payload(records_seen_this_epoch)
        sq_rq_validation_thresholds = sq_rq_deployment_thresholds(args, epoch)
        if args.quality_calibration_only and calibration_source_deployment:
            sq_rq_validation_thresholds = {
                **sq_rq_validation_thresholds,
                "query_confidence_threshold": float(
                    calibration_source_deployment.get(
                        "query_confidence_threshold", args.sq_rq_query_confidence_threshold
                    )
                ),
                "token_membership_threshold": float(
                    calibration_source_deployment.get(
                        "token_membership_threshold", args.sq_rq_token_membership_threshold
                    )
                ),
                "calibration_runtime_frozen": True,
            }
        set_sq_rq_runtime_thresholds(model, sq_rq_validation_thresholds)
        val = evaluate(
            model,
            pack,
            args.val,
            device,
            args.max_tokens_per_record,
            val_limit,
            args.num_queries,
            query_class_weights,
            bottleneck_profile,
            args.small_component_size,
            mask_schedule["mask_positive_weight"],
            mask_schedule["mask_negative_weight"],
            args.mask_focal_gamma,
            mask_schedule["mask_area_ratio_loss_weight"],
            mask_schedule["mask_area_overcoverage_weight"],
            mask_schedule["mask_tversky_loss_weight"],
            args.mask_tversky_alpha,
            args.mask_tversky_beta,
            mask_schedule["mask_positive_prob_floor_loss_weight"],
            args.mask_positive_prob_floor,
            schedule["query_objectness_loss_weight"],
            schedule["query_objectness_positive_weight"],
            schedule["query_objectness_negative_weight"],
            schedule["query_objectness_positive_margin_floor_loss_weight"],
            schedule["query_objectness_positive_margin_floor"],
            schedule["query_objectness_negative_margin_ceiling_loss_weight"],
            schedule["query_objectness_negative_margin_ceiling"],
            amp_dtype=amp_dtype,
            teacher_by_record=teacher_by_record,
            teacher_loss_weight=args.teacher_loss_weight,
            teacher_mask_loss_weight=args.teacher_mask_loss_weight,
            teacher_query_loss_weight=args.teacher_query_loss_weight,
            matching=args.component_matching,
            semantic_loss_weight=args.semantic_loss_weight,
            query_loss_weight=args.query_loss_weight,
            mask_loss_weight=args.mask_loss_weight,
            rq_sq_quality_calibration_loss_weight=args.rq_sq_quality_calibration_loss_weight,
            rq_sq_quality_ranking_weight=args.rq_sq_quality_ranking_weight,
            rq_sq_quality_ranking_margin=args.rq_sq_quality_ranking_margin,
            rq_sq_quality_ranking_top_k=args.rq_sq_quality_ranking_top_k,
            rq_sq_quality_hard_negative_weight=args.rq_sq_quality_hard_negative_weight,
            rq_sq_quality_unmatched_ceiling_weight=args.rq_sq_quality_unmatched_ceiling_weight,
            rq_sq_quality_unmatched_ceiling_probability=args.rq_sq_quality_unmatched_ceiling_probability,
            quality_soft_target_weight=args.quality_soft_target_weight,
            positive_quality_floor_labels=args._hard_recall_label_set,
            positive_quality_floor=(
                args.hard_recall_quality_target_floor
                if args.quality_deployment_expert_weight > 0.0
                else 0.0
            ),
            ownership_loss_weight=args.ownership_loss_weight,
            ownership_mask_consistency_loss_weight=args.ownership_mask_consistency_loss_weight,
            identity_loss_weight=args.identity_loss_weight,
            identity_temperature=args.identity_temperature,
            identity_negative_margin=args.identity_negative_margin,
            geometry_aux_loss_weight=0.5,
            content_anchor_loss_weight=args.content_anchor_loss_weight,
            offset_vote_loss_weight=args.offset_vote_loss_weight,
            affinity_loss_weight=args.affinity_loss_weight,
            geometry_decoder_mode=args.geometry_decoder_mode,
            thing_query_count=(
                int(args.num_queries) - int(args.num_stuff_queries)
                if bool(getattr(runtime_model, "typed_stuff_slots", False))
                else None
            ),
            pcgrad_diagnostic=args.gradient_control == "pcgrad",
            router_load_balance_loss_weight=args.router_load_balance_loss_weight,
            semantic_class_weights_value=class_weights,
            partial_component_policy=args.partial_component_policy,
            partial_component_min_tokens=args.partial_component_min_tokens,
            semantic_label_smoothing=args.semantic_label_smoothing,
            query_label_smoothing=args.query_label_smoothing,
            unmatched_mask_negative_loss_weight=args.unmatched_mask_negative_loss_weight,
            unmatched_mask_negative_top_k=args.unmatched_mask_negative_top_k,
            deployment_min_query_score=args.deployment_min_query_score,
            deployment_mask_threshold=args.deployment_mask_threshold,
            record_id_allowlist=None if selected_val_pages is None else set(selected_val_pages),
            candidate_by_record=val_candidate_by_record,
            max_candidate_queries=args.max_candidate_queries,
            candidate_feature_dim=args.candidate_feature_dim,
        )
        sq_rq_fuse_event = None
        if (
            args.sq_rq_enabled
            and runtime_model.sq_rq_runtime_enabled
            and args.sq_rq_auto_fuse
            and not args.quality_calibration_only
        ):
            sq_rq_proxy = val.get("sq_rq_proxy") or {}
            semantic_regression = float(sq_rq_proxy.get("semantic_context_minus_base_loss", 0.0))
            if semantic_regression > float(args.sq_rq_max_semantic_loss_regression):
                sq_rq_auto_fused = True
                sq_rq_auto_fuse_reason = (
                    f"semantic_context_minus_base_loss={semantic_regression:.6f}>"
                    f"{float(args.sq_rq_max_semantic_loss_regression):.6f}"
                )
                args._sq_rq_auto_fused = True
                args._sq_rq_auto_fuse_reason = sq_rq_auto_fuse_reason
                runtime_model.sq_rq_runtime_enabled = False
                sq_rq_fuse_event = {
                    "triggered": True,
                    "reason": sq_rq_auto_fuse_reason,
                    "semantic_regression": semantic_regression,
                    "threshold": float(args.sq_rq_max_semantic_loss_regression),
                    "fallback": "base_semantic_branch_from_next_epoch",
                }
        selection_gate = {"passed": True, "reason": "metric_not_recall_gated"}
        if args.checkpoint_metric == "neg_loss":
            raw_loss = float(val["loss"])
            selection_score = -raw_loss if math.isfinite(raw_loss) else -float("inf")
        elif args.checkpoint_metric == "component_proxy_score":
            selection_score = float((val.get("component_proxy") or {}).get("component_proxy_score", 0.0))
        elif args.checkpoint_metric == "recall_gated_component_proxy":
            selection_score, selection_gate = recall_gated_selection_score(
                val,
                args.min_val_object_recall_for_checkpoint,
                args.min_val_mask_recall_for_checkpoint,
            )
        elif args.checkpoint_metric == "pq_aware_component_proxy":
            selection_score, selection_gate = pq_aware_selection_score(
                val,
                args.min_val_object_recall_for_checkpoint,
                args.min_val_mask_recall_for_checkpoint,
                args.min_val_mask_precision_for_checkpoint,
                args.max_val_positive_rate_ratio_for_checkpoint,
            )
        elif args.checkpoint_metric == "admission_aware_component_proxy":
            selection_score, selection_gate = admission_aware_selection_score(
                val,
                args.min_val_object_recall_for_checkpoint,
                args.min_val_mask_recall_for_checkpoint,
                args.min_val_mask_precision_for_checkpoint,
                args.max_val_positive_rate_ratio_for_checkpoint,
                args.min_val_positive_object_margin_rate_for_checkpoint,
                args.min_val_positive_object_margin_mean_for_checkpoint,
                args.max_val_negative_object_margin_rate_for_checkpoint,
            )
        elif args.checkpoint_metric == "joint_rq_sq_proxy":
            selection_score, selection_gate = joint_rq_sq_selection_score(
                val,
                args.min_val_rq_proxy_for_checkpoint,
                args.min_val_sq_proxy_for_checkpoint,
                args.max_val_negative_object_margin_rate_for_checkpoint,
                args.min_val_mask_precision_for_checkpoint,
                args.min_val_mask_recall_for_checkpoint,
                args.min_val_instance_tp_for_checkpoint,
                args.min_val_calibrated_proposal_coverage_for_checkpoint,
            )
        elif args.checkpoint_metric == "semantic_token_accuracy":
            selection_score = float(val["semantic_token_accuracy"])
        else:
            raise ValueError(f"unsupported checkpoint metric: {args.checkpoint_metric}")
        quality_gate = quality_checkpoint_selection_gate(
            val,
            max_unmatched_quality=args.max_val_unmatched_quality_for_checkpoint,
            max_ranking_violation_rate=args.max_val_quality_ranking_violation_rate_for_checkpoint,
        )
        if not quality_gate["passed"]:
            selection_score = -float("inf")
            selection_gate = {
                **selection_gate,
                "passed": False,
                "quality_calibration": quality_gate,
                "reason": f"{selection_gate.get('reason', 'selection_gate')};{quality_gate['reason']}",
            }
        else:
            selection_gate = {**selection_gate, "quality_calibration": quality_gate}
        diagnostic_component = val.get("component_proxy") or {}
        diagnostic_selection_score = (
            float(selection_score)
            if math.isfinite(selection_score)
            else float(
                diagnostic_component.get(
                    "calibrated_instance_proxy_pq",
                    diagnostic_component.get("component_proxy_score", 0.0),
                )
            )
        )
        if sq_rq_fuse_event is not None:
            selection_score = -float("inf")
            selection_gate = {
                **selection_gate,
                "passed": False,
                "sq_rq_auto_fuse": sq_rq_fuse_event,
                "reason": f"{selection_gate.get('reason', 'selection_gate')};sq_rq_auto_fused",
            }
        if args.sq_rq_enabled:
            coverage_gate_active = bool(runtime_model.sq_rq_runtime_enabled)
            if coverage_gate_active:
                sq_rq_coverage_gate = sq_rq_coverage_selection_gate(
                    val,
                    args.sq_rq_min_admitted_query_coverage,
                    args.sq_rq_min_context_edge_coverage,
                )
                if not sq_rq_coverage_gate["passed"]:
                    selection_score = -float("inf")
                    selection_gate = {
                        **selection_gate,
                        "passed": False,
                        "sq_rq_coverage": sq_rq_coverage_gate,
                        "reason": f"{selection_gate.get('reason', 'selection_gate')};{sq_rq_coverage_gate['reason']}",
                    }
                else:
                    selection_gate = {**selection_gate, "sq_rq_coverage": sq_rq_coverage_gate}
            else:
                selection_gate = {
                    **selection_gate,
                    "sq_rq_coverage": {
                        "passed": True,
                        "deferred": True,
                        "reason": "deferred_until_sq_rq_runtime_enabled",
                        "runtime_enabled": bool(runtime_model.sq_rq_runtime_enabled),
                        "threshold_phase": sq_rq_train_thresholds.get("phase"),
                    },
                }
            if not sq_rq_checkpoint_promotion_ready(
                bool(runtime_model.sq_rq_runtime_enabled),
                sq_rq_train_thresholds,
                auto_fused=bool(sq_rq_auto_fused),
            ):
                selection_score = -float("inf")
                selection_gate = {
                    **selection_gate,
                    "passed": False,
                    "sq_rq_warmup_not_promotable": True,
                    "training_threshold_phase": sq_rq_train_thresholds.get("phase"),
                    "validation_threshold_phase": sq_rq_validation_thresholds.get("phase"),
                    "reason": f"{selection_gate.get('reason', 'selection_gate')};sq_rq_warmup_checkpoint_not_promotable",
                }
        if args.learned_sparse_router and epoch > int(args.router_collapse_warmup_epochs):
            router_gate = router_usage_selection_gate(
                val,
                args.router_max_dominant_expert_probability,
                args.router_min_expert_assignment_fraction,
            )
            if not router_gate["passed"]:
                selection_score = -float("inf")
                selection_gate = {
                    **selection_gate,
                    "passed": False,
                    "router_usage": router_gate,
                    "reason": f"{selection_gate.get('reason', 'selection_gate')};{router_gate['reason']}",
                }
            else:
                selection_gate = {**selection_gate, "router_usage": router_gate}
        should_run_final_gate = (
            not args.require_final_instance_gate_for_best
            or epoch == int(args.epochs)
            or epoch % int(args.final_instance_gate_interval_epochs) == 0
        )
        if args.require_final_instance_gate_for_best and should_run_final_gate:
            atomic_torch_save(
                torch,
                checkpoint_payload(
                    model,
                    args,
                    run_id=run_id,
                    pid=os.getpid(),
                    epoch=epoch,
                    selection_score=selection_score,
                    selection_gate=selection_gate,
                    schedule=schedule,
                    mask_schedule=mask_schedule,
                    bottleneck_profile=bottleneck_profile,
                    query_class_weights=query_class_weights,
                    boundary="per_epoch_final_gate_candidate",
                    optimizer=optimizer,
                    scheduler=scheduler,
                    semantic_class_weights_value=class_weights,
                    history=history,
                    best_score=best_score,
                    best_val_semantic=best_val_semantic,
                    best_checkpoint_written=best_checkpoint_written,
                    training_rng_state=capture_training_rng_state(torch, np, rng),
                ),
                args.final_instance_gate_checkpoint,
            )
        final_gate = (
            run_per_epoch_final_gate(
                args,
                checkpoint_path=args.final_instance_gate_checkpoint,
                epoch=epoch,
            )
            if should_run_final_gate
            else {
                "required": True,
                "passed": False,
                "reason": "deferred_until_exact_gate_interval",
                "next_gate_epoch": min(
                    int(args.epochs),
                    ((epoch // int(args.final_instance_gate_interval_epochs)) + 1)
                    * int(args.final_instance_gate_interval_epochs),
                ),
            }
        )
        if args.require_final_instance_gate_for_best and not final_gate.get("passed"):
            selection_gate = {
                **selection_gate,
                "passed": False,
                "final_instance_gate_required": True,
                "final_instance_gate": final_gate,
                "reason": f"{selection_gate.get('reason', 'selection_gate')};final_instance_gate_not_passed",
            }
            selection_score = -float("inf")
        elif args.require_final_instance_gate_for_best and args.select_best_by_final_instance_pq:
            exact_pq = float((final_gate.get("metrics") or {}).get("PQ", float("nan")))
            if not math.isfinite(exact_pq):
                selection_gate = {
                    **selection_gate,
                    "passed": False,
                    "final_instance_gate_required": True,
                    "final_instance_gate": final_gate,
                    "reason": f"{selection_gate.get('reason', 'selection_gate')};exact_pq_missing",
                }
                selection_score = -float("inf")
            else:
                selection_score = exact_pq
                selection_gate = {
                    **selection_gate,
                    "passed": True,
                    "selection_source": "exact_final_instance_pq",
                    "final_instance_gate": final_gate,
                }
        exact_full_selection = bool(
            args.require_final_instance_gate_for_best
            and final_gate.get("passed")
            and args.select_best_by_final_instance_pq
        )
        if not validation_is_full and not exact_full_selection:
            selection_score = -float("inf")
            selection_gate = {
                **selection_gate,
                "passed": False,
                "validation_scope": "limited_diagnostic_only",
                "reason": f"{selection_gate.get('reason', 'selection_gate')};nonfull_validation_cannot_promote",
            }
        row = {
            "epoch": epoch,
            "validation_scope": (
                "full" if validation_is_full
                else ("stratified_page_diagnostic_only" if selected_val_pages is not None else "limited_diagnostic_only")
            ),
            "validation_page_count": None if selected_val_pages is None else len(selected_val_pages),
            "sq_rq_runtime": {
                "enabled": bool(runtime_model.sq_rq_runtime_enabled),
                "auto_fused": bool(sq_rq_auto_fused),
                "auto_fuse_reason": sq_rq_auto_fuse_reason,
                "event": sq_rq_fuse_event,
            },
            "sq_rq_train_thresholds": sq_rq_train_thresholds,
            "sq_rq_validation_thresholds": sq_rq_validation_thresholds,
            "lr_scheduler": {
                **learning_rate_scheduler_config(args),
                "last_epoch": None if scheduler is None else int(scheduler.last_epoch),
                "learning_rates": {
                    str(group.get("name", f"group_{index}")): float(group["lr"])
                    for index, group in enumerate(optimizer.param_groups)
                },
            },
            "resume_skip_records_applied": skipped_for_resume,
            "train_records": counters["records"],
            "train_tokens": counters["tokens"],
            "train_labeled_tokens": counters["labeled_tokens"],
            "train_target_components": counters["target_components"],
            "train_matched_components": counters["matched_components"],
            "train_target_selection": target_diagnostic_payload(counters),
            "train_candidate_proxy": {
                "records_with_candidates": counters["candidate_records_with_candidates"],
                "candidate_valid_total": counters["candidate_valid_total"],
                "candidate_mask_token_total": counters["candidate_mask_token_total"],
                "candidate_record_coverage": counters["candidate_records_with_candidates"] / max(counters["records"], 1),
            },
            "train_component_proxy": component_proxy_payload(counters),
            "train_content_anchor_proxy": family_seed_proxy_payload(counters),
            "train_loss_experts": loss_expert_payload(counters),
            "train_loss_breakdown": {
                "semantic": counters["semantic_loss_sum"] / max(counters["records"], 1),
                "query": counters["query_loss_sum"] / max(counters["records"], 1),
                "query_objectness": counters["query_objectness_loss_sum"] / max(counters["records"], 1),
                "rq_sq_quality_calibration": counters["rq_sq_quality_calibration_loss_sum"] / max(counters["records"], 1),
                "family_recall_focus": counters["family_recall_focus_loss_sum"] / max(counters["records"], 1),
                "rq_admission_hard_recall": counters["rq_admission_hard_recall_loss_sum"] / max(counters["records"], 1),
                "mask_hard_recall_floor": counters["mask_hard_recall_floor_loss_sum"] / max(counters["records"], 1),
                "quality_deployment_floor": counters["quality_deployment_floor_loss_sum"] / max(counters["records"], 1),
                "route_classification": counters["route_classification_loss_sum"] / max(counters["records"], 1),
                "route_classification_weight": args.route_classification_loss_weight,
                "explicit_route_classifier": bool(args.explicit_route_classifier),
                "route_conditioning_residual_scale": float(args.route_conditioning_residual_scale),
                "route_conditioning_runtime_scale": float(getattr(runtime_model, "route_conditioning_runtime_scale", 1.0)),
                "dense_attention_feature_adapter": bool(args.dense_attention_feature_adapter),
                "dense_attention_window_size": int(args.dense_attention_window_size),
                "dense_attention_adapter_residual_scale": float(args.dense_attention_adapter_residual_scale),
                "dense_attention_adapter_runtime_scale": float(getattr(runtime_model, "dense_attention_adapter_runtime_scale", 1.0)),
                "router_load_balance": counters["router_load_balance_loss_sum"] / max(counters["router_diagnostic_batches"], 1),
                "router_switch_load_balance": counters["router_switch_load_balance_loss_sum"] / max(counters["router_diagnostic_batches"], 1),
                "router_z_loss": counters["router_z_loss_sum"] / max(counters["router_diagnostic_batches"], 1),
                "router_routing_entropy": counters["router_routing_entropy_sum"] / max(counters["router_diagnostic_batches"], 1),
                "branch_router_load_balance": counters["branch_router_load_balance_loss_sum"] / max(counters["branch_router_diagnostic_batches"], 1),
                "branch_router_switch_load_balance": counters["branch_router_switch_load_balance_loss_sum"] / max(counters["branch_router_diagnostic_batches"], 1),
                "branch_router_z_loss": counters["branch_router_z_loss_sum"] / max(counters["branch_router_diagnostic_batches"], 1),
                "router_load_balance_weight": args.router_load_balance_loss_weight,
                "mask": counters["mask_loss_sum"] / max(counters["records"], 1),
                "unmatched_mask_negative": counters["unmatched_mask_negative_loss_sum"] / max(counters["records"], 1),
                "ownership": counters["ownership_loss_sum"] / max(counters["records"], 1),
                "geometry_aux": counters["geometry_aux_loss_sum"] / max(counters["records"], 1),
                "content_anchor": counters["content_anchor_loss_sum"] / max(counters["records"], 1),
                "component_seed": counters["component_seed_loss_sum"] / max(counters["records"], 1),
                "component_seed_weight": args.component_seed_loss_weight,
                "component_seeded_queries": bool(args.component_seeded_queries),
                "candidate_mask_prior": counters["candidate_mask_prior_loss_sum"] / max(counters["records"], 1),
                "identity": counters["identity_loss_sum"] / max(counters["identity_batches"], 1),
                "teacher": counters["teacher_loss_sum"] / max(counters["records"], 1),
                "teacher_loss_weight": args.teacher_loss_weight,
                "teacher_mask_loss_weight": args.teacher_mask_loss_weight,
                "teacher_query_loss_weight": args.teacher_query_loss_weight,
                "query_objectness_weight": schedule["query_objectness_loss_weight"],
                "query_objectness_positive_weight": schedule["query_objectness_positive_weight"],
                "query_objectness_negative_weight": schedule["query_objectness_negative_weight"],
                "query_objectness_positive_margin_floor_loss_weight": schedule["query_objectness_positive_margin_floor_loss_weight"],
                "query_objectness_positive_margin_floor": schedule["query_objectness_positive_margin_floor"],
                "query_objectness_negative_margin_ceiling_loss_weight": schedule["query_objectness_negative_margin_ceiling_loss_weight"],
                "query_objectness_negative_margin_ceiling": schedule["query_objectness_negative_margin_ceiling"],
                "family_recall_focus_weight": args.family_recall_loss_weight,
                "family_recall_focus_family": args.family_recall_focus,
                "active_loss_experts": args.active_loss_experts,
                "hard_recall_labels": args.hard_recall_labels,
                "hard_recall_families": args.hard_recall_families,
                "effective_hard_recall_labels": sorted(args._hard_recall_label_set),
                "rq_admission_expert_weight": args.rq_admission_expert_weight,
                "mask_recall_expert_weight": args.mask_recall_expert_weight,
                "quality_deployment_expert_weight": args.quality_deployment_expert_weight,
                "hard_recall_admission_floor": args.hard_recall_admission_floor,
                "hard_recall_mask_prob_floor": args.hard_recall_mask_prob_floor,
                "hard_recall_deployment_floor": args.hard_recall_deployment_floor,
                "hard_recall_quality_target_floor": args.hard_recall_quality_target_floor,
                "content_anchor_weight": args.content_anchor_loss_weight,
                "candidate_mask_prior_weight": args.candidate_mask_prior_loss_weight,
            },
            "train_router_proxy": {
                "enabled": counters["router_diagnostic_batches"] > 0,
                "load_balance_cv_squared": counters["router_load_balance_loss_sum"] / max(counters["router_diagnostic_batches"], 1),
                "branch_load_balance_cv_squared": counters["branch_router_load_balance_loss_sum"] / max(counters["branch_router_diagnostic_batches"], 1),
                "branch_router_diagnostic_batches": counters["branch_router_diagnostic_batches"],
                "branch_router_overflow_assignments": counters["branch_router_overflow_assignments"],
                "branch_router_usage_gate_pass_rate": counters["branch_router_usage_gate_pass_batches"] / max(counters["branch_router_diagnostic_batches"], 1),
                "mean_expert_probability": [
                    counters[f"router_expert_probability_{index}"] / max(counters["router_diagnostic_batches"], 1)
                    for index in range(args.router_num_experts if args.learned_sparse_router else 0)
                ],
                "pcgrad_objective_owner": "router" if args.learned_sparse_router else None,
            },
            "train_teacher_proxy": {
                "records_with_teacher": counters["teacher_records_with_teacher"],
                "teacher_components_total": counters["teacher_components_total"],
                "teacher_components_kept": counters["teacher_components_kept"],
                "teacher_components_dropped": counters["teacher_components_dropped"],
                "teacher_positive_components": counters["teacher_positive_components"],
                "teacher_matched_positive_queries": counters["teacher_matched_positive_queries"],
                "teacher_supervision_overlap_queries": counters["teacher_supervision_overlap_queries"],
                "teacher_label_conflict_queries": counters["teacher_label_conflict_queries"],
                "teacher_mask_conflict_queries": counters["teacher_mask_conflict_queries"],
                "teacher_identity_aligned": counters["teacher_identity_aligned"],
                "teacher_identity_unaligned": counters["teacher_identity_unaligned"],
                "gt_positive_teacher_negative_conflicts": counters["gt_positive_teacher_negative_conflicts"],
                "teacher_hard_negative_objectness_queries": counters["teacher_hard_negative_objectness_queries"],
            },
            "objectness_schedule": schedule,
            "mask_loss_schedule": mask_schedule,
            "precision_phase_allowed": precision_phase_allowed,
            "train_loss": loss_sum / max(counters["records"], 1),
            "gradient_control": summarize_gradient_control_reports(gradient_control_reports),
            "train_throughput": {
                **epoch_throughput,
                "epoch_train_elapsed_seconds": epoch_train_elapsed,
            },
            "nonfinite_loss_skipped": counters["nonfinite_loss_skipped"],
            "val": val,
            "selection_score": selection_score,
            "selection_gate": selection_gate,
            "final_instance_gate": final_gate,
            "checkpoint_metric": args.checkpoint_metric,
        }
        history.append(row)
        diagnostic_checkpoint_paths = save_top_k_diagnostic_checkpoint(
            torch,
            checkpoint_payload(
                model,
                args,
                run_id=run_id,
                pid=os.getpid(),
                epoch=epoch,
                selection_score=selection_score,
                selection_gate=selection_gate,
                schedule=schedule,
                mask_schedule=mask_schedule,
                bottleneck_profile=bottleneck_profile,
                query_class_weights=query_class_weights,
                boundary="top_k_diagnostic_checkpoint",
                optimizer=optimizer,
                scheduler=scheduler,
                semantic_class_weights_value=class_weights,
                history=history,
                best_score=best_score,
                best_val_semantic=best_val_semantic,
                best_checkpoint_written=best_checkpoint_written,
                rng_state=rng.getstate(),
                training_rng_state=capture_training_rng_state(torch, np, rng),
            ),
            args.diagnostic_checkpoint_dir,
            score=diagnostic_selection_score,
            keep=args.diagnostic_checkpoint_top_k,
        )
        row["diagnostic_selection_score"] = diagnostic_selection_score
        row["diagnostic_checkpoint_top_k"] = diagnostic_checkpoint_paths
        throughput_history.append(row["train_throughput"])
        best_val_semantic = max(best_val_semantic, float(val["semantic_token_accuracy"]))
        promotion_allowed = bool(training_preset.get("fail_closed", False))
        if (
            selection_gate.get("passed") is True
            and math.isfinite(selection_score)
            and selection_score > best_score
        ):
            best_score = selection_score
            if promotion_allowed:
                best_checkpoint_written = True
                best_checkpoint_file_exists = True
                atomic_torch_save(
                    torch,
                    checkpoint_payload(
                        model,
                        args,
                        run_id=run_id,
                        pid=os.getpid(),
                        epoch=epoch,
                        selection_score=selection_score,
                        selection_gate=selection_gate,
                        schedule=schedule,
                        mask_schedule=mask_schedule,
                        bottleneck_profile=bottleneck_profile,
                        query_class_weights=query_class_weights,
                        boundary="best_selection_checkpoint",
                        optimizer=optimizer,
                        scheduler=scheduler,
                        semantic_class_weights_value=class_weights,
                        history=history,
                        best_score=best_score,
                        best_val_semantic=best_val_semantic,
                        best_checkpoint_written=best_checkpoint_written,
                        rng_state=rng.getstate(),
                        training_rng_state=capture_training_rng_state(torch, np, rng),
                    ),
                    args.model_output,
                )
        if args.last_model_output is not None:
            args.last_model_output.parent.mkdir(parents=True, exist_ok=True)
            atomic_torch_save(
                torch,
                checkpoint_payload(
                    model,
                    args,
                    run_id=run_id,
                    pid=os.getpid(),
                    epoch=epoch,
                    selection_score=selection_score,
                    selection_gate=selection_gate,
                    schedule=schedule,
                    mask_schedule=mask_schedule,
                    bottleneck_profile=bottleneck_profile,
                    query_class_weights=query_class_weights,
                    boundary="last_epoch_diagnostic_checkpoint",
                    optimizer=optimizer,
                    scheduler=scheduler,
                    semantic_class_weights_value=class_weights,
                    history=history,
                    best_score=best_score,
                    best_val_semantic=best_val_semantic,
                    best_checkpoint_written=best_checkpoint_written,
                    rng_state=rng.getstate(),
                    training_rng_state=capture_training_rng_state(torch, np, rng),
                ),
                args.last_model_output,
            )
        payload = {
            "schema_version": "floorplancad_line_token_panoptic_moe_train_v1",
            "created_utc": utc_now(),
            "updated_utc": utc_now(),
            "status": "running",
            "run_id": run_id,
            "pid": os.getpid(),
            "model_output": rel(args.model_output),
            "last_model_output": rel(args.last_model_output),
            "inputs": {
                "train": rel(args.train),
                "val": rel(args.val),
                "init_checkpoint": rel(args.init_checkpoint),
                "resume_checkpoint": rel(args.resume_checkpoint),
                "teacher_proposals": rel(args.teacher_proposals),
                "candidate_proposals": rel(args.candidate_proposals),
                "val_candidate_proposals": rel(args.val_candidate_proposals),
            },
            "init_checkpoint": init_checkpoint_report,
            "resume_checkpoint": resume_report,
            "quality_calibration_scope": quality_calibration_scope,
            "architecture": {
                "family": "line_token_component",
                "model": "transformer_encoder_plus_component_query_mask_decoder",
                "hidden_dim": args.hidden_dim,
                "layers": args.layers,
                "heads": args.heads,
                "num_queries": args.num_queries,
                "query_decoder_layers": args.query_decoder_layers,
                "dropout": args.dropout,
                "position_encoding_version": POSITION_ENCODING_VERSION,
                "position_max_frequency_log2": POSITION_MAX_FREQUENCY_LOG2,
                "quality_head": PANOPTIC_QUALITY_HEAD_VERSION,
                "quality_objective": quality_objective_contract(mask_threshold=args.deployment_mask_threshold),
                "training_preset": training_preset,
                "optimizer_step_budget": training_step_budget,
                "losses": [
                    "semantic_ce",
                    "hungarian_component_query_ce",
                    "query_objectness_bce",
                    "hungarian_component_mask_bce",
                    "hungarian_component_mask_dice",
                    "mask_area_ratio_regularizer",
                    "mask_tversky_recall_regularizer",
                    "mask_positive_probability_floor",
                ],
                "matching": "class_plus_focal_mask_plus_dice_exact_hungarian_v1",
                "query_supervision": {
                    "no_object_query_weight": args.no_object_query_weight,
            "recall_class_weight": args.recall_class_weight,
            "precision_class_weight": args.precision_class_weight,
            "grouping_class_weight": args.grouping_class_weight,
            "extra_recall_labels": parse_label_list(args.extra_recall_labels),
            "extra_grouping_labels": parse_label_list(args.extra_grouping_labels),
            "extra_precision_labels": parse_label_list(args.extra_precision_labels),
            "small_component_size": args.small_component_size,
                    "target_selection_policy": "bottleneck_small_component_round_robin_when_components_exceed_queries",
                    "mask_positive_weight": mask_schedule["mask_positive_weight"],
                    "mask_negative_weight": mask_schedule["mask_negative_weight"],
                    "mask_focal_gamma": args.mask_focal_gamma,
                    "mask_area_ratio_loss_weight": mask_schedule["mask_area_ratio_loss_weight"],
                    "mask_area_overcoverage_weight": mask_schedule["mask_area_overcoverage_weight"],
                    "mask_tversky_loss_weight": mask_schedule["mask_tversky_loss_weight"],
                    "mask_tversky_alpha": args.mask_tversky_alpha,
                    "mask_tversky_beta": args.mask_tversky_beta,
                    "mask_positive_prob_floor_loss_weight": mask_schedule["mask_positive_prob_floor_loss_weight"],
                    "mask_positive_prob_floor": args.mask_positive_prob_floor,
                    "mask_loss_schedule": mask_schedule,
                    "mask_precision_phase_start_epoch": args.mask_precision_phase_start_epoch,
                    "mask_precision_phase_positive_weight": args.mask_precision_phase_positive_weight,
                    "mask_precision_phase_negative_weight": args.mask_precision_phase_negative_weight,
                    "mask_precision_phase_area_ratio_loss_weight": args.mask_precision_phase_area_ratio_loss_weight,
                    "mask_precision_phase_area_overcoverage_weight": args.mask_precision_phase_area_overcoverage_weight,
                    "mask_precision_phase_tversky_loss_weight": args.mask_precision_phase_tversky_loss_weight,
                    "mask_precision_phase_positive_prob_floor_loss_weight": args.mask_precision_phase_positive_prob_floor_loss_weight,
                    "query_objectness_loss_weight": args.query_objectness_loss_weight,
                    "query_objectness_positive_weight": args.query_objectness_positive_weight,
                    "query_objectness_negative_weight": args.query_objectness_negative_weight,
                    "query_objectness_positive_margin_floor_loss_weight": args.query_objectness_positive_margin_floor_loss_weight,
                    "query_objectness_negative_margin_ceiling_loss_weight": args.query_objectness_negative_margin_ceiling_loss_weight,
                    "objectness_schedule": schedule,
                    "objectness_warmup_epochs": args.objectness_warmup_epochs,
                    "objectness_warmup_loss_multiplier": args.objectness_warmup_loss_multiplier,
                    "objectness_warmup_positive_multiplier": args.objectness_warmup_positive_multiplier,
                    "objectness_warmup_negative_multiplier": args.objectness_warmup_negative_multiplier,
                    "objectness_warmup_positive_margin_floor_loss_weight": args.objectness_warmup_positive_margin_floor_loss_weight,
                    "objectness_warmup_negative_margin_ceiling_loss_weight": args.objectness_warmup_negative_margin_ceiling_loss_weight,
                    "objectness_precision_phase_start_epoch": args.objectness_precision_phase_start_epoch,
                    "objectness_precision_phase_loss_weight": args.objectness_precision_phase_loss_weight,
                    "objectness_precision_phase_positive_weight": args.objectness_precision_phase_positive_weight,
                    "objectness_precision_phase_negative_weight": args.objectness_precision_phase_negative_weight,
                    "objectness_precision_phase_positive_margin_floor_loss_weight": args.objectness_precision_phase_positive_margin_floor_loss_weight,
                    "objectness_precision_phase_negative_margin_ceiling_loss_weight": args.objectness_precision_phase_negative_margin_ceiling_loss_weight,
                    "precision_phase_transition_epochs": args.precision_phase_transition_epochs,
                    "objectness_positive_margin_floor": args.objectness_positive_margin_floor,
                    "objectness_negative_margin_ceiling": args.objectness_negative_margin_ceiling,
                    "zero_admission_patience_epochs": args.zero_admission_patience_epochs,
                    "zero_admission_min_epoch": args.zero_admission_min_epoch,
                    "min_val_object_recall_for_checkpoint": args.min_val_object_recall_for_checkpoint,
                    "min_val_mask_recall_for_checkpoint": args.min_val_mask_recall_for_checkpoint,
                    "min_val_mask_precision_for_checkpoint": args.min_val_mask_precision_for_checkpoint,
                    "max_val_positive_rate_ratio_for_checkpoint": args.max_val_positive_rate_ratio_for_checkpoint,
                    "min_val_positive_object_margin_rate_for_checkpoint": args.min_val_positive_object_margin_rate_for_checkpoint,
                    "min_val_positive_object_margin_mean_for_checkpoint": args.min_val_positive_object_margin_mean_for_checkpoint,
                    "max_val_negative_object_margin_rate_for_checkpoint": args.max_val_negative_object_margin_rate_for_checkpoint,
                    "require_final_instance_gate_for_best": args.require_final_instance_gate_for_best,
                    "final_instance_gate_report": rel(args.final_instance_gate_report),
                    "final_instance_gate_command_template": args.final_instance_gate_command_template,
                    "final_instance_gate_protocol": args.final_instance_gate_protocol,
                    "final_instance_gate_checkpoint": rel(args.final_instance_gate_checkpoint),
                    "min_final_instance_tp_for_best": args.min_final_instance_tp_for_best,
                    "min_final_instance_rq_for_best": args.min_final_instance_rq_for_best,
                    "min_final_instance_sq_for_best": args.min_final_instance_sq_for_best,
                    "min_final_instance_pq_for_best": getattr(args, "min_final_instance_pq_for_best", 0.0),
                    "max_final_instance_fp_for_best": args.max_final_instance_fp_for_best,
                    "bottleneck_profile": bottleneck_profile,
                },
            },
            "architecture_signature": architecture_signature_from_args(args),
            "objective_config": objective_config_from_args(args),
            "objective_config_hash": objective_config_hash(objective_config_from_args(args)),
            "semantic_class_weights": (
                None if class_weights is None else [float(value) for value in class_weights.detach().cpu().tolist()]
            ),
            "effective_argv": list(sys.argv),
            "training_scope": {
                "limit_records": args.limit_records,
                "val_limit_records": args.val_limit_records,
                "max_tokens_per_record": args.max_tokens_per_record,
                "batch_records": batch_size,
                "auto_throughput_profile": auto_throughput_profile,
                "train_prefetch_records": args.train_prefetch_records,
                "train_prefetch_workers": args.train_prefetch_workers,
                "amp": args.amp,
                "amp_dtype": str(amp_dtype).replace("torch.", "") if amp_dtype is not None else None,
                "tf32_enabled": tf32_enabled,
                "cudnn_benchmark_enabled": cudnn_benchmark_enabled,
                "compile_model": compile_report,
                "progress_checkpoint_records": args.progress_checkpoint_records,
                "progress_checkpoint_seconds": args.progress_checkpoint_seconds,
                "progress_status_records": args.progress_status_records,
                "progress_status_seconds": args.progress_status_seconds,
                "checkpoint_archive_dir": rel(args.checkpoint_archive_dir),
                "checkpoint_archive_keep": int(args.checkpoint_archive_keep),
                "graceful_signal_checkpoint": True,
                "component_matching": args.component_matching,
                "optimizer_steps": counters["optimizer_steps"],
                "paper_metric": False,
                "paper_metric_eligible": False,
                "metric_scope": "training_diagnostics_only",
                "paper_metric_requirement": "run the external stitched page-level official primitive-set evaluator",
            },
            "config": {
                "hidden_dim": args.hidden_dim,
                "layers": args.layers,
                "heads": args.heads,
                "num_queries": args.num_queries,
                "query_decoder_layers": args.query_decoder_layers,
                "dropout": args.dropout,
                "max_tokens_per_record": args.max_tokens_per_record,
                "checkpoint_metric": args.checkpoint_metric,
                "lr": args.lr,
                "batch_records": args.batch_records,
                "auto_throughput_profile": auto_throughput_profile,
                "train_prefetch_records": args.train_prefetch_records,
                "train_prefetch_workers": args.train_prefetch_workers,
                "amp": args.amp,
                "amp_dtype": str(amp_dtype).replace("torch.", "") if amp_dtype is not None else None,
                "tf32_enabled": tf32_enabled,
                "cudnn_benchmark_enabled": cudnn_benchmark_enabled,
                "compile_model": compile_report,
                "progress_checkpoint_records": args.progress_checkpoint_records,
                "progress_checkpoint_seconds": args.progress_checkpoint_seconds,
                "progress_status_records": args.progress_status_records,
                "progress_status_seconds": args.progress_status_seconds,
                "checkpoint_archive_dir": rel(args.checkpoint_archive_dir),
                "checkpoint_archive_keep": int(args.checkpoint_archive_keep),
                "graceful_signal_checkpoint": True,
                "resume_checkpoint": rel(args.resume_checkpoint),
                "resume_optimizer": args.resume_optimizer,
                "component_matching": args.component_matching,
            },
            "current_epoch": epoch,
            "target_epochs": args.epochs,
            "checkpoint_metric": args.checkpoint_metric,
            "best_selection_score": best_score,
            "best_checkpoint_written": best_checkpoint_written,
            "best_checkpoint_claimed_by_resume": best_checkpoint_claimed_by_resume,
            "best_checkpoint_file_exists": args.model_output.exists(),
            "best_val_semantic_token_accuracy": best_val_semantic,
            "latest_epoch": row,
            "throughput_history": throughput_history,
            "history": history,
            "claim_boundary": "Training artifact only; PQ/RQ/SQ requires panoptic apply/export and primitive-set evaluator.",
            "comparable_for_matrix": False,
        }
        write_json(args.report, payload)
        print(json.dumps(row, ensure_ascii=False))

        if args.zero_admission_patience_epochs > 0 and epoch >= args.zero_admission_min_epoch:
            recent = history[-args.zero_admission_patience_epochs :]
            zero_recent = (
                len(recent) == args.zero_admission_patience_epochs
                and all(
                    int(((item.get("val") or {}).get("component_proxy") or {}).get("query_predicted_object_total", 0)) == 0
                    for item in recent
                )
            )
            if zero_recent:
                final_status = "early_stopped_zero_admission"
                early_stop_payload = {
                    "reason": "validation_query_predicted_object_total_zero",
                    "epoch": epoch,
                    "patience_epochs": args.zero_admission_patience_epochs,
                    "zero_admission_min_epoch": args.zero_admission_min_epoch,
                    "recent_epochs": [int(item.get("epoch", 0)) for item in recent],
                    "required_next_action": "change object admission/router/mask objective before long training",
                }
                payload["status"] = final_status
                payload["early_stop"] = early_stop_payload
                write_json(args.report, payload)
                print(json.dumps({"status": final_status, "early_stop": early_stop_payload}, ensure_ascii=False))
                break

    payload["status"] = final_status
    best_checkpoint_file_exists = args.model_output.exists()
    best_checkpoint_written = bool(best_checkpoint_written and best_checkpoint_file_exists)
    payload["best_checkpoint_written"] = best_checkpoint_written
    payload["best_checkpoint_claimed_by_resume"] = best_checkpoint_claimed_by_resume
    payload["best_checkpoint_file_exists"] = best_checkpoint_file_exists
    if early_stop_payload is not None:
        payload["early_stop"] = early_stop_payload
    write_json(args.report, payload)
    for signum, previous in previous_handlers.items():
        signal.signal(signum, previous)
    print(json.dumps({"status": final_status, "report": rel(args.report), "model": rel(args.model_output), "checkpoint_metric": args.checkpoint_metric, "best_selection_score": best_score, "best_checkpoint_written": best_checkpoint_written, "best_checkpoint_file_exists": best_checkpoint_file_exists, "best_val_semantic_token_accuracy": best_val_semantic}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
