"""Strict loader for the V4 raw-semantic FloorPlanCAD segment cache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from experiments.floorplancad_target_schema_v2 import TargetSchemaV2Batch, load_target_schema_v2
from experiments.floorplancad_target_schema_v3 import V3_FEATURE_NAMES

SCHEMA_VERSION = "floorplancad_page_window_target_v4_raw_semantic_segments"
INPUT_SCHEMA_VERSION = "floorplancad_line_json_primitive_cache_v4_raw_semantic_segments"
SCHEMA_VERSION_V5 = "floorplancad_page_window_target_v5_structural_segments"
INPUT_SCHEMA_VERSION_V5 = "floorplancad_line_json_primitive_cache_v5_structural_segments"
SCHEMA_VERSION_V6 = "floorplancad_page_window_target_v6_structural_relation_segments"
INPUT_SCHEMA_VERSION_V6 = "floorplancad_line_json_primitive_cache_v6_structural_relation_segments"
V4_EXTRA_FEATURE_NAMES = (
    "primitive_kind_path",
    "primitive_kind_circle",
    "primitive_kind_ellipse",
    "primitive_kind_other",
    "path_cmd_line",
    "path_cmd_curve",
    "path_cmd_arc",
    "path_cmd_close",
    "segment_order_norm",
    "segment_arclength_start",
    "segment_arclength_mid",
    "segment_arclength_end",
    "tangent_dx",
    "tangent_dy",
    "turn_angle_norm",
    "primitive_closed",
    "rgb_r_norm",
    "rgb_g_norm",
    "rgb_b_norm",
    "layer_has_peers",
    "same_layer_fraction",
    "page_primitive_count_log1p",
)
V4_FEATURE_NAMES = (*V3_FEATURE_NAMES, *V4_EXTRA_FEATURE_NAMES)
V5_EXTRA_FEATURE_NAMES = (
    "bbox_width",
    "bbox_height",
    "bbox_area",
    "bbox_aspect_log",
    "bbox_compactness",
    "endpoint_closure_residual",
    "endpoint_span",
    "horizontal_vertical_balance",
    "turn_abs",
    "turn_squared",
    "closed_or_near_closed",
    "segment_density_log1p",
)
V5_FEATURE_NAMES = (*V4_FEATURE_NAMES, *V5_EXTRA_FEATURE_NAMES)
V6_EXTRA_FEATURE_NAMES = (
    "endpoint_start_degree_norm",
    "endpoint_end_degree_norm",
    "endpoint_degree_sum_norm",
    "junction_t_score",
    "junction_l_score",
    "junction_x_score",
    "nearest_long_straight_gap",
    "parallel_neighbor_fraction",
    "perpendicular_neighbor_fraction",
    "containment_depth_norm",
    "contains_neighbor_fraction",
    "repetition_spacing_score",
    "repetition_direction_score",
    "local_direction_hist_0",
    "local_direction_hist_45",
    "local_direction_hist_90",
    "local_direction_hist_135",
    "same_layer_structural_neighbor_fraction",
)
V6_FEATURE_NAMES = (*V5_FEATURE_NAMES, *V6_EXTRA_FEATURE_NAMES)
FORBIDDEN_INPUT_FIELDS = frozenset({"semanticId", "instanceId", "semantic_id", "instance_id"})
RAW_METADATA_ALIGNMENT = "vecformer_sampling_exact_v2_official_raw"
RAW_SVG_COORDINATE_PROTOCOL = "floorplancad_official_140_v1"


@dataclass(frozen=True)
class TargetSchemaV4Batch:
    """v2 targets plus V4 segment features and feature lineage metadata."""

    base: TargetSchemaV2Batch
    segment_features: tuple[np.ndarray, ...]
    primitive_feature_names: tuple[str, ...]
    feature_groups: dict[str, tuple[str, ...]]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)


def _validate_feature_lineage(record: dict[str, Any]) -> None:
    lineage = record.get("input_feature_lineage")
    if not isinstance(lineage, dict):
        raise ValueError("input_feature_lineage must be present for v4")
    leaked = sorted(FORBIDDEN_INPUT_FIELDS & set(lineage.get("consumed_input_fields", [])))
    if leaked:
        raise ValueError(f"label fields are forbidden v4 inputs: {leaked}")
    if lineage.get("raw_metadata_alignment") != RAW_METADATA_ALIGNMENT:
        raise ValueError(
            f"input_feature_lineage.raw_metadata_alignment must be {RAW_METADATA_ALIGNMENT}; "
            "legacy V4 caches must be rebuilt"
        )
    if lineage.get("raw_svg_coordinate_protocol") != RAW_SVG_COORDINATE_PROTOCOL:
        raise ValueError("v4 input features must come from the official FloorPlanCAD 140-coordinate SVGs")
    for field in ("primitive_kind", "path_commands", "segment_order_arclength", "tangent_curvature", "layer_identity"):
        if field not in lineage:
            raise ValueError(f"input_feature_lineage missing {field}")


def load_target_schema_v4(record: dict[str, Any]) -> TargetSchemaV4Batch:
    target_schema = record.get("target_schema_version")
    input_schema = record.get("input_schema_version")
    if target_schema == SCHEMA_VERSION and input_schema == INPUT_SCHEMA_VERSION:
        expected_features = V4_FEATURE_NAMES
        feature_schema_name = "v4"
    elif target_schema == SCHEMA_VERSION_V5 and input_schema == INPUT_SCHEMA_VERSION_V5:
        expected_features = V5_FEATURE_NAMES
        feature_schema_name = "v5"
    elif target_schema == SCHEMA_VERSION_V6 and input_schema == INPUT_SCHEMA_VERSION_V6:
        expected_features = V6_FEATURE_NAMES
        feature_schema_name = "v6"
    else:
        raise ValueError(
            f"target/input schema must be either ({SCHEMA_VERSION}, {INPUT_SCHEMA_VERSION}) "
            f"or ({SCHEMA_VERSION_V5}, {INPUT_SCHEMA_VERSION_V5}) "
            f"or ({SCHEMA_VERSION_V6}, {INPUT_SCHEMA_VERSION_V6})"
        )
    feature_names = record.get("primitive_feature_names")
    if tuple(feature_names or ()) != expected_features:
        raise ValueError(f"primitive_feature_names must match the locked {feature_schema_name} feature contract")
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("primitive_feature_names must be unique")
    _validate_feature_lineage(record)

    proxy = dict(record)
    proxy["target_schema_version"] = "floorplancad_page_window_target_v2"
    base = load_target_schema_v2(proxy)
    rows = record["primitive_rows"]
    segments: list[np.ndarray] = []
    for index, row in enumerate(rows):
        raw_segments = row.get("segment_features")
        if not isinstance(raw_segments, list) or not raw_segments:
            raise ValueError(f"primitive_rows[{index}].segment_features must be non-empty")
        if int(row.get("segment_count", -1)) != len(raw_segments):
            raise ValueError(f"primitive_rows[{index}].segment_count must match segment_features")
        array = np.asarray(raw_segments, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != len(expected_features):
            raise ValueError(f"primitive_rows[{index}].segment_features width mismatch")
        if not np.isfinite(array).all():
            raise ValueError(f"primitive_rows[{index}].segment_features must be finite")
        representative = np.asarray(row["features"], dtype=np.float32)
        if representative.shape != (len(expected_features),):
            raise ValueError(f"primitive_rows[{index}].features width mismatch")
        if not np.any(np.all(np.isclose(array, representative[None, :], atol=1e-7), axis=1)):
            raise ValueError(f"primitive_rows[{index}].features must be one retained segment")
        segments.append(array)
    return TargetSchemaV4Batch(
        base=base,
        segment_features=tuple(segments),
        primitive_feature_names=tuple(str(name) for name in feature_names),
        feature_groups={
            "v3_geometry": tuple(V3_FEATURE_NAMES),
            "raw_semantic": tuple(V4_EXTRA_FEATURE_NAMES[:16]),
            "style": tuple(V4_EXTRA_FEATURE_NAMES[16:19]),
            "layer": (
                "layer_has_peers",
                "same_layer_fraction",
                "page_primitive_count_log1p",
            ),
            "structural_v5": tuple(V5_EXTRA_FEATURE_NAMES) if expected_features in {V5_FEATURE_NAMES, V6_FEATURE_NAMES} else (),
            "structural_relation_v6": tuple(V6_EXTRA_FEATURE_NAMES) if expected_features == V6_FEATURE_NAMES else (),
        },
    )
