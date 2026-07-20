"""Strict loader for the lossless sampled-segment FloorPlanCAD target cache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from experiments.floorplancad_target_schema_v2 import TargetSchemaV2Batch, load_target_schema_v2


SCHEMA_VERSION = "floorplancad_page_window_target_v3_segments"
INPUT_SCHEMA_VERSION = "floorplancad_line_json_primitive_cache_v3_segments"
V3_FEATURE_NAMES = (
    "x1_norm", "y1_norm", "x2_norm", "y2_norm", "cx_norm", "cy_norm",
    "length_norm", "log_length_norm", "orientation", "horizontal", "vertical",
    "stroke_width_norm", "stroke_width_raw", "layer_id_norm",
    "segment_relative_length", "segment_count_log1p",
)


@dataclass(frozen=True)
class TargetSchemaV3Batch:
    """v2 targets plus an unpadded sampled-segment sequence per primitive."""

    base: TargetSchemaV2Batch
    segment_features: tuple[np.ndarray, ...]
    primitive_feature_names: tuple[str, ...]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)


def load_target_schema_v3(record: dict[str, Any]) -> TargetSchemaV3Batch:
    """Load v3 records and fail closed on lossy or order-leaking inputs."""
    if record.get("target_schema_version") != SCHEMA_VERSION:
        raise ValueError(f"target_schema_version must be {SCHEMA_VERSION}")
    if record.get("input_schema_version") != INPUT_SCHEMA_VERSION:
        raise ValueError(f"input_schema_version must be {INPUT_SCHEMA_VERSION}")
    feature_names = record.get("primitive_feature_names")
    if not isinstance(feature_names, list) or not feature_names:
        raise ValueError("primitive_feature_names must be a non-empty list")
    if "primitive_id_norm" in feature_names:
        raise ValueError("primitive_id_norm is forbidden in v3 model features")
    if "color_hash_norm" in feature_names:
        raise ValueError("color_hash_norm is forbidden in v3 model features")
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("primitive_feature_names must be unique")
    if tuple(feature_names) != V3_FEATURE_NAMES:
        raise ValueError("primitive_feature_names must match the locked v3 feature contract")

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
        if array.ndim != 2 or array.shape[1] != len(feature_names):
            raise ValueError(f"primitive_rows[{index}].segment_features width mismatch")
        if not np.isfinite(array).all():
            raise ValueError(f"primitive_rows[{index}].segment_features must be finite")
        representative = np.asarray(row["features"], dtype=np.float32)
        if representative.shape != (len(feature_names),):
            raise ValueError(f"primitive_rows[{index}].features width mismatch")
        if not np.any(np.all(np.isclose(array, representative[None, :], atol=1e-7), axis=1)):
            raise ValueError(f"primitive_rows[{index}].features must be one retained segment")
        segments.append(array)
    return TargetSchemaV3Batch(
        base=base,
        segment_features=tuple(segments),
        primitive_feature_names=tuple(str(name) for name in feature_names),
    )
