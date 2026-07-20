from pathlib import Path

from experiments import floorplancad_apply_line_token_panoptic_moe as apply_script
from experiments import floorplancad_launch_panoptic_moe_training as launcher
from experiments import floorplancad_train_line_token_panoptic_moe as train
from experiments.floorplancad_panoptic_runtime_config import (
    DEFAULT_RUNTIME_PROFILE,
    DEV_V4_RUNTIME_PROFILE,
    PANOPTIC_TRAIN_CACHE_V3_R2,
    PANOPTIC_VAL_CACHE_V3_R2,
    runtime_profile,
)


ROOT = Path(__file__).parents[1]


def test_python_entrypoints_share_the_default_runtime_profile() -> None:
    assert train.DEFAULT_TRAIN == DEFAULT_RUNTIME_PROFILE.train_path(ROOT)
    assert train.DEFAULT_VAL == DEFAULT_RUNTIME_PROFILE.val_path(ROOT)
    assert apply_script.DEFAULT_CACHE == DEFAULT_RUNTIME_PROFILE.val_path(ROOT)
    assert launcher.DEFAULT_TRAIN == DEFAULT_RUNTIME_PROFILE.train_path(ROOT)
    assert launcher.DEFAULT_VAL == DEFAULT_RUNTIME_PROFILE.val_path(ROOT)


def test_default_runtime_profile_is_v3_r2_not_legacy_v2() -> None:
    profile = runtime_profile("v3_r2_default")
    assert profile.train_cache == PANOPTIC_TRAIN_CACHE_V3_R2
    assert profile.val_cache == PANOPTIC_VAL_CACHE_V3_R2
    assert profile.cli_schema_args() == ("--input-feature-schema", "v3", "--require-target-schema-v3")


def test_evalstable_script_uses_explicit_default_profile_cache() -> None:
    text = (ROOT / "scripts/run_floorplancad_true_moe_l_evalstable_gpu0.sh").read_text()
    assert "floorplancad_line_json_primitive_cache_windowed_2048_s1536_v2" not in text
    assert PANOPTIC_TRAIN_CACHE_V3_R2 in text
    assert PANOPTIC_VAL_CACHE_V3_R2 in text
    assert "--input-feature-schema v3" in text
    assert "--require-target-schema-v3" in text


def test_high_rq_dev_script_is_explicitly_v4_same_set_profile() -> None:
    text = (ROOT / "scripts/launch_floorplancad_high_rq_contract_dev.sh").read_text()
    assert 'RUNTIME_PROFILE="same_set_v4_dev"' in text
    assert DEV_V4_RUNTIME_PROFILE.train_cache in text
    assert DEV_V4_RUNTIME_PROFILE.init_checkpoint in text
    assert "--input-feature-schema v4" in text
    assert "--require-target-schema-v4" in text
