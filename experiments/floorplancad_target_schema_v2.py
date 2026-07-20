#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


SCHEMA_VERSION = "floorplancad_page_window_target_v2"
IGNORE_LABEL = 35
TASK_AVAILABILITY_KEYS = ("semantic", "rq", "sq", "quality", "ownership", "identity", "teacher")
REQUIRED_ROW_FIELDS = {
    "features",
    "semantic_id",
    "instance_id",
    "primitive_id",
    "page_instance_id",
    "mask_loss_valid",
    "inverse_exposure_weight",
    "log1p_primitive_length",
    "visible_fraction",
}


@dataclass(frozen=True)
class TargetSchemaV2Batch:
    features: np.ndarray
    labels: np.ndarray
    instances: np.ndarray
    primitive_ids: np.ndarray
    mask_loss_valid: np.ndarray
    inverse_exposure: np.ndarray
    log1p_length: np.ndarray
    visible_fraction: np.ndarray
    page_instance_ids: tuple[str | None, ...]
    record_id: str
    original_record_id: str
    window_index: int
    task_availability: dict[str, bool]


def task_availability(record: dict[str, Any]) -> dict[str, bool]:
    """Return explicit task supervision availability, failing closed on malformed flags."""
    raw = record.get("task_availability")
    if raw is None:
        return {key: True for key in TASK_AVAILABILITY_KEYS}
    if not isinstance(raw, dict):
        raise ValueError("task_availability must be an object when present")
    unknown = set(raw).difference(TASK_AVAILABILITY_KEYS)
    if unknown:
        raise ValueError(f"unknown task_availability keys: {sorted(unknown)}")
    result = {key: True for key in TASK_AVAILABILITY_KEYS}
    for key, value in raw.items():
        if not isinstance(value, bool):
            raise ValueError(f"task_availability[{key!r}] must be boolean")
        result[key] = value
    if not result["rq"]:
        result["quality"] = False
        result["ownership"] = False
        result["identity"] = False
    if not result["semantic"]:
        result["sq"] = False
    if not result["semantic"] and not result["rq"]:
        raise ValueError("task_availability must leave semantic or rq supervision enabled")
    return result


def _finite_float(value: Any, field: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{field} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{field} must be <= {maximum}")
    return result


def load_target_schema_v2(record: dict[str, Any]) -> TargetSchemaV2Batch:
    if record.get("target_schema_version") != SCHEMA_VERSION:
        raise ValueError(f"target_schema_version must be {SCHEMA_VERSION}; v1/fallback records are forbidden")
    if bool(record.get("query_overflow")):
        raise ValueError("query_overflow=true is forbidden for target-schema-v2 training consumption")
    overflow_count = int(record.get("query_overflow_component_count", 0))
    if overflow_count != 0:
        raise ValueError("query_overflow_component_count must be zero")
    target_count = int(record.get("query_target_count", -1))
    capacity = int(record.get("query_target_capacity", -1))
    if target_count < 0 or capacity <= 0 or target_count > capacity:
        raise ValueError("invalid query target count/capacity")
    rows = record.get("primitive_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("primitive_rows must be a non-empty list")
    feature_width = None
    features = []
    labels = []
    instances = []
    primitive_ids = []
    mask_loss_valid = []
    inverse_exposure = []
    log1p_length = []
    visible_fraction = []
    page_instance_ids = []
    seen_primitives = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"primitive_rows[{index}] must be an object")
        missing = sorted(REQUIRED_ROW_FIELDS - set(row))
        if missing:
            raise ValueError(f"primitive_rows[{index}] missing required fields: {missing}")
        row_features = row["features"]
        if not isinstance(row_features, list) or not row_features:
            raise ValueError(f"primitive_rows[{index}].features must be non-empty")
        if feature_width is None:
            feature_width = len(row_features)
        if len(row_features) != feature_width:
            raise ValueError("feature width mismatch")
        feature_values = [_finite_float(value, "features") for value in row_features]
        label = int(row["semantic_id"])
        instance = int(row["instance_id"])
        primitive_id = int(row["primitive_id"])
        if not 0 <= label <= IGNORE_LABEL:
            raise ValueError(f"semantic_id outside [0,{IGNORE_LABEL}]")
        if primitive_id < 0 or primitive_id in seen_primitives:
            raise ValueError("primitive_id must be unique and non-negative within a window")
        seen_primitives.add(primitive_id)
        identity = row["page_instance_id"]
        if identity is not None and not isinstance(identity, str):
            raise ValueError("page_instance_id must be string or null")
        valid = row["mask_loss_valid"]
        if not isinstance(valid, bool):
            raise ValueError("mask_loss_valid must be boolean")
        inverse = _finite_float(row["inverse_exposure_weight"], "inverse_exposure_weight", minimum=0.0, maximum=1.0)
        length = _finite_float(row["log1p_primitive_length"], "log1p_primitive_length", minimum=0.0)
        fraction = _finite_float(row["visible_fraction"], "visible_fraction", minimum=0.0, maximum=1.0)
        if identity is None and valid:
            raise ValueError("rows without page_instance_id cannot be valid mask targets")
        if fraction < 1.0 and valid:
            raise ValueError("partial boundary rows cannot be marked mask_loss_valid")
        features.append(feature_values)
        labels.append(label)
        instances.append(instance)
        primitive_ids.append(primitive_id)
        mask_loss_valid.append(valid)
        inverse_exposure.append(inverse)
        log1p_length.append(length)
        visible_fraction.append(fraction)
        page_instance_ids.append(identity)
    return TargetSchemaV2Batch(
        features=np.asarray(features, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        instances=np.asarray(instances, dtype=np.int64),
        primitive_ids=np.asarray(primitive_ids, dtype=np.int64),
        mask_loss_valid=np.asarray(mask_loss_valid, dtype=bool),
        inverse_exposure=np.asarray(inverse_exposure, dtype=np.float32),
        log1p_length=np.asarray(log1p_length, dtype=np.float32),
        visible_fraction=np.asarray(visible_fraction, dtype=np.float32),
        page_instance_ids=tuple(page_instance_ids),
        record_id=str(record.get("record_id", "")),
        original_record_id=str(record.get("original_record_id", record.get("record_id", ""))),
        window_index=int(record.get("window_index", 0)),
        task_availability=task_availability(record),
    )


def semantic_primitive_weights(batch: TargetSchemaV2Batch, *, normalize: bool = True) -> np.ndarray:
    weights = batch.inverse_exposure.astype(np.float64) * batch.log1p_length.astype(np.float64)
    weights = np.where(batch.labels == IGNORE_LABEL, 0.0, weights)
    if normalize:
        positive = weights > 0.0
        if np.any(positive):
            weights[positive] /= weights[positive].mean()
    return weights.astype(np.float32)


def component_mask_primitive_weights(
    batch: TargetSchemaV2Batch,
    target_page_instance_id: str,
    *,
    normalize: bool = True,
) -> np.ndarray:
    target_members = np.asarray([identity == target_page_instance_id for identity in batch.page_instance_ids], dtype=bool)
    if not np.any(target_members):
        raise ValueError(f"target page instance is absent: {target_page_instance_id}")
    if not np.all(batch.mask_loss_valid[target_members]):
        return np.zeros(len(batch.labels), dtype=np.float32)
    weights = batch.inverse_exposure.astype(np.float64) * batch.log1p_length.astype(np.float64)
    if normalize:
        positive = weights > 0.0
        if np.any(positive):
            weights[positive] /= weights[positive].mean()
    return weights.astype(np.float32)


def weighted_binary_mask_iou(
    prediction: np.ndarray,
    target: np.ndarray,
    primitive_weights: np.ndarray,
) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    weights = np.asarray(primitive_weights, dtype=np.float64)
    if prediction.shape != target.shape or prediction.shape != weights.shape:
        raise ValueError("prediction, target, and primitive_weights must have identical shapes")
    union = prediction | target
    if not np.any(union):
        return 0.0
    return float(weights[prediction & target].sum() / max(weights[union].sum(), 1e-12))
