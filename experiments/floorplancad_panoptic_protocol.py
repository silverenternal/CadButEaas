"""Shared FloorPlanCAD panoptic protocol constants.

Keep train, apply, and ownership decoding on one scored-output contract. This
module must stay dependency-light so it can be imported by launchers and tests.
"""

from __future__ import annotations

from experiments.floorplancad_panoptic_runtime_config import (
    PANOPTIC_TRAIN_CACHE_V3_R2,
    PANOPTIC_VAL_CACHE_V3_R2,
)


DEFAULT_NO_OBJECT_LABEL = 35
THING_LABELS = tuple(range(30))
STUFF_LABELS = tuple(range(30, 35))

PANOPTIC_QUALITY_OBJECTIVE_VERSION = "deployment_score_balanced_decoded_hard_mask_iou_v6"
PANOPTIC_QUALITY_MASK_THRESHOLD = 0.5
PANOPTIC_OBJECT_NORMALIZATION_THRESHOLD = 0.01
