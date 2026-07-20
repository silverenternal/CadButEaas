"""Runtime profiles for FloorPlanCAD panoptic training and apply entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PANOPTIC_TRAIN_CACHE_V3_R2 = "reports/vlm/floorplancad_line_json_primitive_cache_windowed_2048_s1536_v3_r2/train_windowed_primitive_cache.jsonl"
PANOPTIC_VAL_CACHE_V3_R2 = "reports/vlm/floorplancad_line_json_primitive_cache_windowed_2048_s1536_v3_r2/val_windowed_primitive_cache.jsonl"
PANOPTIC_SAME_SET_V4_CACHE = "datasets/floorplancad_v4_same_set_overfit_32_v1/same_set_windowed_primitive_cache.jsonl"
PANOPTIC_SAME_SET_V4_CHECKPOINT = "reports/vlm/floorplancad_v4_same_set_overfit_32_v4_protocol_replay/best.pt"
PANOPTIC_FORBIDDEN_V6_TEST_CHECKPOINT = "reports/vlm/floorplancad_v6_test_train/last.pt"


@dataclass(frozen=True)
class PanopticRuntimeProfile:
    name: str
    input_feature_schema: str
    train_cache: str
    val_cache: str
    required_schema_flag: str
    training_preset: str = "custom"
    init_checkpoint: str | None = None
    diagnostic_only: bool = False

    def train_path(self, root: Path) -> Path:
        return root / self.train_cache

    def val_path(self, root: Path) -> Path:
        return root / self.val_cache

    def cli_schema_args(self) -> tuple[str, ...]:
        return ("--input-feature-schema", self.input_feature_schema, self.required_schema_flag)


RUNTIME_PROFILES: dict[str, PanopticRuntimeProfile] = {
    "v3_r2_default": PanopticRuntimeProfile(
        name="v3_r2_default",
        input_feature_schema="v3",
        train_cache=PANOPTIC_TRAIN_CACHE_V3_R2,
        val_cache=PANOPTIC_VAL_CACHE_V3_R2,
        required_schema_flag="--require-target-schema-v3",
    ),
    "same_set_v4_dev": PanopticRuntimeProfile(
        name="same_set_v4_dev",
        input_feature_schema="v4",
        train_cache=PANOPTIC_SAME_SET_V4_CACHE,
        val_cache=PANOPTIC_SAME_SET_V4_CACHE,
        required_schema_flag="--require-target-schema-v4",
        training_preset="dev",
        init_checkpoint=PANOPTIC_SAME_SET_V4_CHECKPOINT,
        diagnostic_only=True,
    ),
}


def runtime_profile(name: str = "v3_r2_default") -> PanopticRuntimeProfile:
    try:
        return RUNTIME_PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(RUNTIME_PROFILES))
        raise ValueError(f"unknown FloorPlanCAD panoptic runtime profile {name!r}; choices: {choices}") from exc


DEFAULT_RUNTIME_PROFILE = runtime_profile("v3_r2_default")
DEV_V4_RUNTIME_PROFILE = runtime_profile("same_set_v4_dev")
