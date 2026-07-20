"""Shared FloorPlanCAD panoptic protocol constants.

Keep train, apply, and ownership decoding on one scored-output contract. This
module must stay dependency-light so it can be imported by launchers and tests.
"""

from __future__ import annotations


DEFAULT_NO_OBJECT_LABEL = 35
THING_LABELS = tuple(range(30))
STUFF_LABELS = tuple(range(30, 35))

PANOPTIC_QUALITY_OBJECTIVE_VERSION = "deployment_score_balanced_decoded_hard_mask_iou_v6"
PANOPTIC_QUALITY_MASK_THRESHOLD = 0.5
PANOPTIC_OBJECT_NORMALIZATION_THRESHOLD = 0.01

PANOPTIC_TRAIN_CACHE_V3_R2 = "reports/vlm/floorplancad_line_json_primitive_cache_windowed_2048_s1536_v3_r2/train_windowed_primitive_cache.jsonl"
PANOPTIC_VAL_CACHE_V3_R2 = "reports/vlm/floorplancad_line_json_primitive_cache_windowed_2048_s1536_v3_r2/val_windowed_primitive_cache.jsonl"
