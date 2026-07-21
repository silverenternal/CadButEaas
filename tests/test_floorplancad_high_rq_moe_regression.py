import json
import sys
from pathlib import Path

import pytest

from experiments import floorplancad_train_line_token_panoptic_moe as train


REPLAY_REPORT = Path("results/floorplancad_v4_same_set_overfit_32_v4_protocol_replay_train.json")
LAUNCH_SCRIPT = Path("scripts/launch_floorplancad_high_rq_contract_dev.sh")


def _load_replay_report() -> dict:
    assert REPLAY_REPORT.is_file(), f"missing high-RQ replay report: {REPLAY_REPORT}"
    return json.loads(REPLAY_REPORT.read_text())


def test_high_rq_moe_replay_contract_is_preserved():
    report = _load_replay_report()
    config = report["config"]
    architecture = report["architecture_signature"]
    input_protocol = architecture["input_protocol"]
    router_config = architecture["sparse_router_config"]

    assert config["hidden_dim"] == 256
    assert config["layers"] == 4
    assert config["heads"] == 8
    assert config["query_decoder_layers"] == 1
    assert config["lr"] == 0.0
    assert input_protocol["content_seeded_queries"] is False
    assert router_config["enabled"] is True
    assert router_config["num_experts"] == 4
    assert router_config["top_k"] == 2


def test_high_rq_moe_replay_metrics_remain_the_regression_floor():
    report = _load_replay_report()
    epoch = report["history"][-1]
    component_proxy = epoch["val"]["component_proxy"]
    selection_gate = epoch["selection_gate"]
    target_selection = epoch["val"]["target_selection"]

    assert component_proxy["calibrated_instance_proxy_rq"] >= 0.38
    assert component_proxy["calibrated_instance_proxy_sq"] >= 0.80
    assert component_proxy["calibrated_instance_proxy_pq"] >= 0.30
    assert component_proxy["query_label_accuracy"] >= 0.90
    assert component_proxy["query_positive_object_recall"] >= 0.99
    assert selection_gate["passed"] is True
    assert target_selection["partial_mask_components_excluded"] > 0
    assert target_selection["partial_mask_components_kept_window_visible"] == 0


def test_dev_preset_rejects_bad_v6_defaults_and_promotes_safe_route_dense_mainline(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train"])
    args = train.parse_args()
    args.training_preset = "dev"
    args.hidden_dim = 128
    args.layers = 2
    args.heads = 4
    args.query_decoder_layers = 3
    args.learned_sparse_router = False
    args.content_seeded_queries = True
    args.component_seeded_queries = True
    args.repeated_group_fusion = True
    args.relation_bias_enabled = True
    args.offset_vote_enabled = True
    args.candidate_aware_queries = True
    args.candidate_feature_dim = 20
    args.weak_family_feature_fusion = True
    args.quality_query_gradient_scale = 0.5
    args.explicit_route_classifier = False
    args.route_classification_loss_weight = 0.0
    args.dense_attention_feature_adapter = False
    args.partial_component_policy = "window_visible"
    args.content_anchor_loss_weight = 0.10

    train.apply_training_preset(args)

    assert (args.hidden_dim, args.layers, args.heads, args.query_decoder_layers) == (256, 4, 8, 1)
    assert args.learned_sparse_router is True
    assert args.typed_branch_routers is True
    assert args.branch_dropless is True
    assert args.content_seeded_queries is True
    assert args.component_seeded_queries is False
    assert args.repeated_group_fusion is False
    assert args.relation_bias_enabled is False
    assert args.offset_vote_enabled is True
    assert args.candidate_aware_queries is False
    assert args.candidate_feature_dim == 0
    assert args.weak_family_feature_fusion is False
    assert args.quality_query_gradient_scale == 0.0
    assert args.explicit_route_classifier is True
    assert args.route_classification_loss_weight >= 0.05
    assert args.dense_attention_feature_adapter is True
    assert args.offset_vote_loss_weight >= 0.05
    assert args.affinity_loss_weight >= 0.02
    assert args.partial_component_policy == "exclude"
    assert args.content_anchor_loss_weight >= 0.05


def test_high_rq_contract_model_state_dict_includes_mainline_route_dense_only(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(sys, "argv", ["train"])
    args = train.parse_args()
    args.training_preset = "dev"
    train.apply_training_preset(args)

    model = train.make_panoptic_model(
        torch.nn,
        torch,
        feature_dim=len(train.model_feature_names_for_schema("v4")),
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        heads=args.heads,
        num_queries=args.num_queries,
        num_labels=36,
        query_decoder_layers=args.query_decoder_layers,
        dropout=args.dropout,
        geometry_decoder_mode=args.geometry_decoder_mode,
        num_stuff_queries=args.num_stuff_queries,
        typed_stuff_slots=args.typed_stuff_slots,
        ownership_enabled=True,
        sq_rq_enabled=args.sq_rq_enabled,
        learned_sparse_router=args.learned_sparse_router,
        typed_branch_routers=args.typed_branch_routers,
        branch_dropless=args.branch_dropless,
        router_num_experts=args.router_num_experts,
        router_top_k=args.router_top_k,
        router_temperature=args.router_temperature,
        content_seeded_queries=args.content_seeded_queries,
        repeated_group_fusion=args.repeated_group_fusion,
        relation_bias_enabled=args.relation_bias_enabled,
        component_seeded_queries=args.component_seeded_queries,
        offset_vote_enabled=args.offset_vote_enabled,
        candidate_aware_queries=args.candidate_aware_queries,
        candidate_feature_dim=args.candidate_feature_dim,
        weak_family_feature_fusion=args.weak_family_feature_fusion,
        quality_query_gradient_scale=args.quality_query_gradient_scale,
        explicit_route_classifier=args.explicit_route_classifier,
        dense_attention_feature_adapter=args.dense_attention_feature_adapter,
        dense_attention_window_size=args.dense_attention_window_size,
    )
    keys = set(model.state_dict())

    assert not any(key.startswith("family_seed_head.") for key in keys)
    assert not any(key.startswith("component_seed_head.") for key in keys)
    assert not any(key.startswith("relation_feature_proj.") for key in keys)
    assert any(key.startswith("token_offset_head.") for key in keys)
    assert any(key.startswith("token_affinity_head.") for key in keys)
    assert not any(key.startswith("candidate_feature_proj.") for key in keys)
    assert not any(key.startswith("weak_family_feature_proj.") for key in keys)
    assert "offset_ownership_gate" in keys
    assert any(key.startswith("sparse_router.") for key in keys)
    assert any(key.startswith("branch_routers.") for key in keys)
    assert any(key.startswith("branch_shared_experts.") for key in keys)
    assert any(key.startswith("route_family_head.") for key in keys)
    assert any(key.startswith("dense_adapter_attn.") for key in keys)


def test_high_rq_contract_launcher_uses_safe_checkpoint_and_gate_tests():
    text = LAUNCH_SCRIPT.read_text()
    command_block = text.split("command=(")[1].split("exec 9>")[0]

    assert "floorplancad_v4_same_set_overfit_32_v4_protocol_replay/best.pt" in text
    assert "floorplancad_v6_test_train/last.pt" not in command_block
    assert "tests/test_floorplancad_high_rq_moe_regression.py" in text
    assert "--training-preset dev" in text
    assert "--input-feature-schema v4" in text
    assert "--require-target-schema-v4" in text
    assert "--epochs 2" in text
    assert "--hidden-dim 256" in text
    assert "--layers 4" in text
    assert "--heads 8" in text
    assert "--query-decoder-layers 1" in text
    assert "--init-checkpoint \"$CHECKPOINT\"" in text
