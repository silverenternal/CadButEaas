"""Shared proposal scoring utilities for FloorPlanCAD panoptic decoding."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.floorplancad_panoptic_protocol import (
    DEFAULT_NO_OBJECT_LABEL,
    PANOPTIC_OBJECT_NORMALIZATION_THRESHOLD,
)


def sigmoid_np(logits: np.ndarray, *, clip: float = 40.0) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -float(clip), float(clip))))


def calibrated_proposal_score(class_score: float, quality_score: float) -> float:
    """Score an already-admitted query without multiplying admission twice."""
    return float(class_score) * float(quality_score)


def mask_objectness_scores_np(
    mask_logits: np.ndarray,
    *,
    object_normalization_threshold: float = PANOPTIC_OBJECT_NORMALIZATION_THRESHOLD,
) -> np.ndarray:
    if mask_logits.ndim != 2:
        raise ValueError("mask logits must have shape [queries, tokens]")
    if not 0.0 <= float(object_normalization_threshold) <= 1.0:
        raise ValueError("object normalization threshold must be in [0, 1]")
    probabilities = sigmoid_np(mask_logits)
    support = probabilities > float(object_normalization_threshold)
    support_count = np.maximum(support.sum(axis=1), 1)
    return np.clip((probabilities * support).sum(axis=1) / support_count, 0.0, 1.0).astype(np.float32)


def mask_objectness_score_np(
    mask_probabilities: np.ndarray,
    *,
    object_normalization_threshold: float = PANOPTIC_OBJECT_NORMALIZATION_THRESHOLD,
) -> float:
    if not 0.0 <= float(object_normalization_threshold) <= 1.0:
        raise ValueError("object normalization threshold must be in [0, 1]")
    support = mask_probabilities > float(object_normalization_threshold)
    return float(np.clip((mask_probabilities * support).sum() / max(int(support.sum()), 1), 0.0, 1.0))


def query_mask_objectness_scores(
    torch: Any,
    mask_logits: Any,
    *,
    object_normalization_threshold: float = PANOPTIC_OBJECT_NORMALIZATION_THRESHOLD,
) -> Any:
    if mask_logits.ndim < 2:
        raise ValueError("mask_logits must end with a token dimension")
    if not 0.0 <= float(object_normalization_threshold) <= 1.0:
        raise ValueError("object normalization threshold must be in [0, 1]")
    probability = torch.sigmoid(mask_logits.float())
    support = probability > float(object_normalization_threshold)
    support_count = support.to(probability.dtype).sum(dim=-1).clamp_min(1.0)
    score = (probability * support.to(probability.dtype)).sum(dim=-1) / support_count
    return score.clamp(min=0.0, max=1.0)


def rq_sq_quality_deployment_scores(
    torch: Any,
    query_logits: Any,
    quality_logits: Any,
    mask_logits: Any | None = None,
    *,
    no_object_label: int = DEFAULT_NO_OBJECT_LABEL,
) -> tuple[Any, Any, Any]:
    if query_logits.shape[:-1] != quality_logits.shape:
        raise ValueError("query_logits and quality_logits must have matching query dimensions")
    if int(query_logits.shape[-1]) <= int(no_object_label):
        raise ValueError("query_logits must include foreground and no-object labels")
    foreground_probability = (
        query_logits.float().softmax(dim=-1)[..., : int(no_object_label)].max(dim=-1).values.detach()
    )
    if mask_logits is not None:
        if mask_logits.shape[:-1] != quality_logits.shape:
            raise ValueError("mask_logits must match query dimensions")
        foreground_probability = foreground_probability * query_mask_objectness_scores(torch, mask_logits)
    quality_probability = torch.sigmoid(quality_logits.float())
    deployment_score = foreground_probability * quality_probability
    return quality_probability, foreground_probability, deployment_score
