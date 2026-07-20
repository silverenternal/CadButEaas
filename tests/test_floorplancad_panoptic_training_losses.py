from collections import Counter
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from experiments.floorplancad_train_line_token_panoptic_moe import (
    BASE_FEATURES,
    FAMILY_NAMES,
    IGNORE_LABEL,
    MODEL_FEATURE_NAMES,
    PANOPTIC_SEGMENT_INPUT_PROTOCOL_VERSION,
    PANOPTIC_QUALITY_OBJECTIVE_VERSION,
    TARGET_SCHEMA_V3,
    active_loss_experts_from_args,
    apply_training_preset,
    auxiliary_loss_weight_for_active_experts,
    checkpoint_abi_metadata,
    component_proxy_payload,
    component_seed_loss,
    configure_quality_calibration_scope,
    deployment_decoded_query_masks,
    explicit_route_classification_loss,
    family_recall_focus_loss,
    family_seed_loss,
    focused_family_positive_query_mask,
    geometry_mask_connectivity_loss,
    hard_recall_admission_margin_loss,
    hard_recall_label_mask,
    hard_recall_mask_floor_loss,
    hard_recall_quality_deployment_loss,
    parse_family_set_csv,
    loss_expert_payload,
    loss_weights_for_active_experts,
    joint_rq_sq_selection_score,
    load_resume_checkpoint,
    make_panoptic_model,
    mask_loss_schedule,
    moe_branch_specialization_loss,
    objectness_schedule,
    objective_config_from_args,
    objective_config_hash,
    output_protocol_claim_blockers,
    precision_phase_admission_ready,
    query_mask_objectness_scores,
    query_selected_primitive_indices,
    query_objectness_loss,
    rq_query_supervision_losses,
    rq_sq_quality_calibration_loss,
    rq_sq_quality_deployment_scores,
    rq_sq_quality_targets,
    scheduled_auxiliary_scale,
    sparse_endpoint_neighbor_graph,
    sq_rq_coverage_selection_gate,
    stuff_overlap_union_consistency_loss,
    teacher_positive_query_loss,
    token_affinity_component_loss,
    token_offset_vote_loss,
    tversky_loss_per_item,
    update_supervised_component_proxy,
    update_loss_expert_counters,
    update_teacher_match_conflicts,
    unmatched_query_empty_mask_loss,
)


def precision_schedule_args(**overrides):
    values = {
        "query_objectness_loss_weight": 0.25,
        "query_objectness_positive_weight": 2.0,
        "query_objectness_negative_weight": 1.0,
        "query_objectness_positive_margin_floor_loss_weight": 0.0,
        "query_objectness_negative_margin_ceiling_loss_weight": 0.0,
        "objectness_positive_margin_floor": 0.75,
        "objectness_negative_margin_ceiling": -0.25,
        "objectness_warmup_epochs": 3,
        "objectness_warmup_loss_multiplier": 4.0,
        "objectness_warmup_positive_multiplier": 4.0,
        "objectness_warmup_negative_multiplier": 0.25,
        "objectness_warmup_positive_margin_floor_loss_weight": 0.0,
        "objectness_warmup_negative_margin_ceiling_loss_weight": 0.0,
        "objectness_precision_phase_start_epoch": 1,
        "objectness_precision_phase_loss_weight": 0.35,
        "objectness_precision_phase_positive_weight": 1.25,
        "objectness_precision_phase_negative_weight": 2.0,
        "objectness_precision_phase_positive_margin_floor_loss_weight": 0.0,
        "objectness_precision_phase_negative_margin_ceiling_loss_weight": 0.0,
        "mask_positive_weight": 8.0,
        "mask_negative_weight": 1.0,
        "mask_area_ratio_loss_weight": 0.0,
        "mask_area_overcoverage_weight": 2.0,
        "mask_tversky_loss_weight": 0.0,
        "mask_positive_prob_floor_loss_weight": 0.0,
        "mask_precision_phase_start_epoch": 1,
        "mask_precision_phase_positive_weight": 4.0,
        "mask_precision_phase_negative_weight": 1.5,
        "mask_precision_phase_area_ratio_loss_weight": 0.15,
        "mask_precision_phase_area_overcoverage_weight": 2.0,
        "mask_precision_phase_tversky_loss_weight": 0.25,
        "mask_precision_phase_positive_prob_floor_loss_weight": 0.1,
        "precision_phase_transition_epochs": 8,
    }
    values.update(overrides)
    return Namespace(**values)


def optimizer_resume_args(**overrides):
    values = {
        "hidden_dim": 4,
        "layers": 1,
        "heads": 1,
        "num_queries": 2,
        "query_decoder_layers": 1,
        "dropout": 0.0,
        "position_encoding_version": "continuous_fourier_logspace_v2",
        "quality_head": "independent_query_soft_iou_v1",
        "identity_head_version": "normalized_query_identity_embedding_v1",
        "identity_dim": 32,
        "checkpoint_metric": "joint_rq_sq_proxy",
        "query_objectness_loss_weight": 0.25,
        "query_objectness_positive_weight": 2.0,
        "query_objectness_negative_weight": 1.0,
        "mask_loss_weight": 2.0,
        "mask_tversky_loss_weight": 0.25,
        "rq_sq_quality_calibration_loss_weight": 0.1,
        "identity_loss_weight": 0.25,
        "identity_temperature": 0.1,
        "identity_negative_margin": 0.2,
        "teacher_loss_weight": 0.5,
        "teacher_query_loss_weight": 0.5,
        "component_matching": "hungarian_cpu",
        "recall_class_weight": 2.5,
        "objectness_precision_phase_start_epoch": 20,
        "min_val_rq_proxy_for_checkpoint": 0.2,
        "resume_optimizer": True,
        "allow_optimizer_objective_mismatch": False,
        "training_preset": "default",
    }
    values.update(overrides)
    return Namespace(**values)


def optimizer_checkpoint(args):
    model = make_panoptic_model(torch.nn, torch, len(BASE_FEATURES), 4, 1, 1, 2, query_decoder_layers=1, dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters())
    input_protocol = {
        "version": PANOPTIC_SEGMENT_INPUT_PROTOCOL_VERSION,
        "target_schema_version": TARGET_SCHEMA_V3,
        "input_schema_version": "floorplancad_line_json_primitive_cache_v3_segments",
        "segment_features": True,
        "content_seeded_queries": False,
    }
    abi = checkpoint_abi_metadata(128, input_protocol=input_protocol)
    objective_config = objective_config_from_args(args)
    return model, optimizer, {
        "schema_version": "floorplancad_line_token_panoptic_moe_checkpoint_v2",
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "feature_names": list(MODEL_FEATURE_NAMES),
        "hidden_dim": 4,
        "layers": 1,
        "heads": 1,
        "num_queries": 2,
        "query_decoder_layers": 1,
        "num_labels": 36,
        "dropout": 0.0,
        "position_encoding_version": abi["position_encoding_version"],
        "quality_head": abi["quality_head"],
        "identity_head_version": abi["identity_head_version"],
        "identity_dim": abi["identity_dim"],
        "checkpoint_abi": abi,
        "feature_schema_sha256": abi["feature_schema_sha256"],
        "ontology_sha256": abi["ontology_sha256"],
        "window_contract_sha256": abi["window_contract_sha256"],
        "input_protocol": input_protocol,
        "objective_config": objective_config,
        "objective_config_hash": objective_config_hash(objective_config),
    }


def test_quality_objective_version_is_part_of_resume_hash_contract():
    config = objective_config_from_args(Namespace())

    assert config["quality_objective_version"] == PANOPTIC_QUALITY_OBJECTIVE_VERSION


def test_runtime_sq_rq_fuse_state_does_not_pollute_objective_hash():
    baseline = Namespace(mask_loss_weight=2.0)
    runtime = Namespace(
        mask_loss_weight=2.0,
        _sq_rq_auto_fused=True,
        _sq_rq_auto_fuse_reason="validation_regression",
    )

    assert objective_config_from_args(runtime) == objective_config_from_args(baseline)
    assert objective_config_hash(objective_config_from_args(runtime)) == objective_config_hash(
        objective_config_from_args(baseline)
    )


def test_output_protocol_claim_blocks_newer_path_than_input_schema():
    args = Namespace(
        model_output=Path("reports/vlm/floorplancad_v6_bad/best.pt"),
        last_model_output=None,
        report=Path("results/floorplancad_v6_bad.json"),
        diagnostic_checkpoint_dir=Path("reports/vlm/floorplancad_v6_bad/topk"),
        checkpoint_archive_dir=None,
        final_instance_gate_checkpoint=None,
        final_instance_gate_report=None,
        allow_output_protocol_name_mismatch=False,
    )

    blockers = output_protocol_claim_blockers(args, "v4")

    assert any("model_output_claims_v6_with_input_v4" in blocker for blocker in blockers)
    assert any("report_claims_v6_with_input_v4" in blocker for blocker in blockers)
    assert output_protocol_claim_blockers(args, "v6") == []
    args.allow_output_protocol_name_mismatch = True
    assert output_protocol_claim_blockers(args, "v4") == []


def test_teacher_query_loss_does_not_train_unmatched_queries_as_no_object():
    logits = torch.zeros((3, 36), requires_grad=True)
    labels = torch.tensor([2, IGNORE_LABEL, IGNORE_LABEL])
    loss = teacher_positive_query_loss(torch, torch.nn.CrossEntropyLoss(), logits, labels)

    loss.backward()

    assert torch.count_nonzero(logits.grad[0]).item() > 0
    assert torch.count_nonzero(logits.grad[1:]).item() == 0


def test_family_recall_focus_selects_only_requested_family():
    labels = torch.tensor([12, 17, IGNORE_LABEL])

    selected = focused_family_positive_query_mask(torch, labels, "furniture")

    assert selected.tolist() == [True, False, False]


def test_token_offset_vote_loss_trains_component_centers():
    features = torch.zeros((3, len(BASE_FEATURES)))
    features[:, 4:6] = torch.tensor([[0.1, 0.1], [0.3, 0.1], [0.8, 0.8]])
    offsets = torch.zeros((3, 2), requires_grad=True)
    labels = torch.tensor([12, IGNORE_LABEL])
    masks = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])

    loss = token_offset_vote_loss(torch, offsets, features, labels, masks, torch.ones(3))
    loss.backward()

    assert loss.item() > 0.0
    assert offsets.grad[:2, 0].abs().sum().item() > 0.0
    assert offsets.grad[2].abs().sum().item() == pytest.approx(0.0)


def test_token_affinity_component_loss_has_supervised_gradient():
    embeddings = torch.randn((4, 8), generator=torch.Generator().manual_seed(7), requires_grad=True)
    labels = torch.tensor([12, 17, IGNORE_LABEL])
    masks = torch.tensor([
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
    ])

    loss = token_affinity_component_loss(torch, embeddings, labels, masks, torch.ones(4))
    loss.backward()

    assert loss.item() >= 0.0
    assert embeddings.grad.abs().sum().item() > 0.0


def test_quality_deployment_score_includes_mask_objectness_when_available():
    query_logits = torch.zeros((2, IGNORE_LABEL + 1))
    query_logits[:, 2] = 5.0
    quality_logits = torch.zeros(2)
    mask_logits = torch.tensor([[5.0, 5.0, 5.0], [0.0, 0.0, 0.0]])

    quality, foreground, deployment = rq_sq_quality_deployment_scores(
        torch, query_logits, quality_logits, mask_logits
    )
    mask_scores = query_mask_objectness_scores(torch, mask_logits)

    assert quality.tolist() == pytest.approx([0.5, 0.5])
    assert foreground[0].item() > foreground[1].item()
    assert deployment.tolist() == pytest.approx((query_logits.softmax(-1)[:, 2] * 0.5 * mask_scores).tolist())


def test_family_recall_focus_loss_pushes_admission_quality_and_mask():
    labels = torch.tensor([12, 17, IGNORE_LABEL])
    query_logits = torch.zeros((3, 36), requires_grad=True)
    admission_logits = torch.tensor([-2.0, -2.0, -2.0], requires_grad=True)
    quality_logits = torch.tensor([-2.0, -2.0, -2.0], requires_grad=True)
    mask_logits = torch.tensor(
        [[-2.0, -2.0, 1.0], [-2.0, 1.0, -2.0], [-2.0, -2.0, -2.0]],
        requires_grad=True,
    )
    masks = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])

    loss = family_recall_focus_loss(
        torch,
        query_logits,
        admission_logits,
        quality_logits,
        mask_logits,
        labels,
        masks,
        family="furniture",
        admission_floor=0.25,
        mask_positive_prob_floor=0.45,
        quality_floor=0.10,
    )
    loss.backward()

    assert loss.item() > 0.0
    assert admission_logits.grad[0].item() < 0.0
    assert quality_logits.grad[0].item() < 0.0
    assert mask_logits.grad[0, :2].mean().item() < 0.0
    assert admission_logits.grad[1].item() == pytest.approx(0.0)


def test_hard_recall_losses_route_only_selected_labels():
    labels = torch.tensor([12, 13, 17, IGNORE_LABEL])
    selected = hard_recall_label_mask(torch, labels, {12, 13})
    assert selected.tolist() == [True, True, False, False]

    query_logits = torch.zeros((4, 36), requires_grad=True)
    admission_logits = torch.full((4,), -2.0, requires_grad=True)
    quality_logits = torch.full((4,), -2.0, requires_grad=True)
    mask_logits = torch.full((4, 3), -2.0, requires_grad=True)
    masks = torch.tensor([
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    loss = (
        hard_recall_admission_margin_loss(
            torch, admission_logits, labels, {12, 13}, probability_floor=0.55
        )
        + hard_recall_mask_floor_loss(
            torch, mask_logits, labels, masks, {12, 13}, probability_floor=0.45
        )
        + hard_recall_quality_deployment_loss(
            torch, query_logits, quality_logits, labels, {12, 13}, deployment_floor=0.22
        )
    )
    loss.backward()

    assert admission_logits.grad[:2].mean().item() < 0.0
    assert quality_logits.grad[:2].mean().item() < 0.0
    assert mask_logits.grad[:2][masks[:2].bool()].mean().item() < 0.0
    assert admission_logits.grad[2:].abs().max().item() == pytest.approx(0.0)


def test_loss_expert_accounting_respects_active_expert_weights():
    terms = {
        "query": torch.tensor(2.0),
        "query_objectness": torch.tensor(1.0),
        "rq_admission_hard_recall": torch.tensor(3.0),
        "mask": torch.tensor(5.0),
    }
    weights = {
        "query": 1.0,
        "query_objectness": 0.5,
        "rq_admission_hard_recall": 2.0,
        "mask": 4.0,
    }
    routed = loss_weights_for_active_experts(weights, {"rq_admission"})
    counters = Counter()

    update_loss_expert_counters(counters, terms, routed, {"rq_admission"})
    payload = loss_expert_payload(counters)

    assert routed["mask"] == 0.0
    assert payload["groups"]["rq_admission"]["weighted_total"] == pytest.approx(8.5)
    assert payload["groups"]["mask_shape"]["weighted_total"] == pytest.approx(0.0)


def test_active_loss_experts_support_independent_mode_aliases():
    assert active_loss_experts_from_args(Namespace(active_loss_experts="joint_routed")) >= {
        "rq_admission",
        "mask_shape",
        "quality_deployment",
    }
    assert active_loss_experts_from_args(Namespace(active_loss_experts="rq_admission_only")) == {
        "rq_admission"
    }
    assert active_loss_experts_from_args(Namespace(active_loss_experts="quality_deployment_only")) == {
        "quality_deployment"
    }
    assert active_loss_experts_from_args(Namespace(active_loss_experts="rq_admission,mask_shape")) == {
        "rq_admission",
        "mask_shape",
    }


def test_hard_recall_family_parser_expands_known_families():
    families = parse_family_set_csv("furniture, appliances_plumbing")

    assert families == {"furniture", "appliances_plumbing"}
    with pytest.raises(ValueError):
        parse_family_set_csv("unknown_family")


def test_component_seed_loss_trains_family_logits():
    logits = torch.zeros((4, len(FAMILY_NAMES) + 4), requires_grad=True)
    labels = torch.tensor([12, 18, 30, IGNORE_LABEL])
    valid = torch.ones_like(labels, dtype=torch.bool)

    loss = component_seed_loss(torch, logits, labels, valid)
    loss.backward()

    assert loss.item() > 0.0
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[:, :len(FAMILY_NAMES)]).item() > 0


def test_active_loss_experts_are_part_of_objective_hash_contract():
    base = objective_config_hash(objective_config_from_args(optimizer_resume_args()))
    changed = objective_config_hash(
        objective_config_from_args(optimizer_resume_args(active_loss_experts="quality_deployment_only"))
    )

    assert changed != base


def test_auxiliary_loss_weights_are_fail_closed_in_expert_only_mode():
    active = active_loss_experts_from_args(
        Namespace(active_loss_experts="sq_semantic,rq_admission,quality_deployment")
    )

    assert auxiliary_loss_weight_for_active_experts(0.25, active, {"topology_merge"}) == 0.0
    assert auxiliary_loss_weight_for_active_experts(0.05, active, set(), joint_only=True) == 0.0
    assert auxiliary_loss_weight_for_active_experts(0.10, active, {"sq_semantic"}) == pytest.approx(0.10)


def test_quality_head_predicts_detached_mask_iou_without_objectness_gradient():
    labels = torch.tensor([2])
    targets = torch.tensor([[1.0, 1.0]])
    mask_logits = torch.tensor([[4.0, 4.0]], requires_grad=True)
    quality_logits = torch.tensor([0.0], requires_grad=True)

    loss = rq_sq_quality_calibration_loss(torch, quality_logits, mask_logits, labels, targets)
    loss.backward()

    assert loss.item() > 0.0
    assert quality_logits.grad.item() < 0.0
    assert mask_logits.grad is None or torch.count_nonzero(mask_logits.grad).item() == 0


def test_quality_target_matches_deployment_hard_length_weighted_mask_iou():
    labels = torch.tensor([2])
    targets = torch.tensor([[1.0, 0.0, 1.0]])
    mask_logits = torch.tensor([[0.1, 0.1, -0.1]])

    uniform, positive = rq_sq_quality_targets(
        torch, mask_logits, labels, targets, torch.ones(3)
    )
    length_weighted, _ = rq_sq_quality_targets(
        torch, mask_logits, labels, targets, torch.tensor([1.0, 20.0, 1.0])
    )

    assert positive.tolist() == [True]
    assert uniform.item() == pytest.approx(1.0 / 3.0)
    assert length_weighted.item() == pytest.approx(1.0 / 22.0)


def test_quality_target_respects_ownership_null_competition():
    mask_logits = torch.full((2, 3), 4.0)
    ownership_logits = torch.tensor([
        [5.0, 1.0, 0.0],
        [1.0, 2.0, 4.0],
        [1.0, 5.0, 0.0],
    ])
    decoded = deployment_decoded_query_masks(
        torch, mask_logits, ownership_logits
    )

    assert decoded.tolist() == [[True, False, False], [False, False, True]]

    quality_target, _ = rq_sq_quality_targets(
        torch,
        mask_logits,
        torch.tensor([2, 3]),
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]),
        ownership_logits=ownership_logits,
    )
    assert quality_target.tolist() == pytest.approx([1.0, 0.5])


def test_unmatched_query_cannot_steal_positive_quality_target_ownership():
    mask_logits = torch.full((2, 1), 4.0)
    ownership_logits = torch.tensor([[1.0, 5.0, 0.0]])
    quality_target, positive = rq_sq_quality_targets(
        torch,
        mask_logits,
        torch.tensor([2, IGNORE_LABEL]),
        torch.tensor([[1.0], [0.0]]),
        ownership_logits=ownership_logits,
    )

    assert positive.tolist() == [True, False]
    assert quality_target.tolist() == pytest.approx([1.0, 0.0])


def test_quality_head_trains_unmatched_queries_toward_zero_quality():
    labels = torch.tensor([2, IGNORE_LABEL])
    targets = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    mask_logits = torch.tensor([[4.0, 4.0], [4.0, 4.0]], requires_grad=True)
    quality_logits = torch.tensor([0.0, 4.0], requires_grad=True)

    loss = rq_sq_quality_calibration_loss(torch, quality_logits, mask_logits, labels, targets)
    loss.backward()

    assert quality_logits.grad[0] < 0.0
    assert quality_logits.grad[1] > 0.0
    assert mask_logits.grad is None or torch.count_nonzero(mask_logits.grad).item() == 0


def test_quality_calibration_uses_non_saturating_logit_gradient_for_unmatched_queries():
    labels = torch.tensor([IGNORE_LABEL])
    targets = torch.zeros((1, 2))
    mask_logits = torch.zeros((1, 2), requires_grad=True)
    quality_logits = torch.tensor([-4.0], requires_grad=True)

    loss = rq_sq_quality_calibration_loss(
        torch,
        quality_logits,
        mask_logits,
        labels,
        targets,
        ranking_weight=0.0,
        hard_negative_weight=0.0,
    )
    loss.backward()

    assert quality_logits.grad.item() == pytest.approx(torch.sigmoid(quality_logits.detach()).item())
    assert mask_logits.grad is None or torch.count_nonzero(mask_logits.grad).item() == 0


def test_quality_calibration_balances_positive_gradient_across_negative_counts():
    def positive_gradient(negative_count):
        labels = torch.tensor([2] + [IGNORE_LABEL] * negative_count)
        targets = torch.ones((negative_count + 1, 2))
        mask_logits = torch.full((negative_count + 1, 2), 4.0)
        quality_logits = torch.zeros(negative_count + 1, requires_grad=True)
        loss = rq_sq_quality_calibration_loss(
            torch, quality_logits, mask_logits, labels, targets, ranking_weight=0.0
        )
        loss.backward()
        return quality_logits.grad[0].item()

    assert positive_gradient(2) == pytest.approx(positive_gradient(32), abs=1e-8)


def test_quality_hard_negative_gradient_is_not_diluted_by_easy_negatives():
    easy_negative_count = 128
    labels = torch.tensor([2, IGNORE_LABEL] + [IGNORE_LABEL] * easy_negative_count)
    targets = torch.zeros((easy_negative_count + 2, 2))
    targets[0] = 1.0
    mask_logits = torch.zeros_like(targets)
    mask_logits[0] = 4.0
    quality_logits = torch.tensor(
        [0.0, 2.0] + [-8.0] * easy_negative_count, requires_grad=True
    )

    loss = rq_sq_quality_calibration_loss(
        torch,
        quality_logits,
        mask_logits,
        labels,
        targets,
        ranking_weight=0.0,
        ranking_top_k=1,
    )
    loss.backward()

    hard_negative_gradient_floor = 0.05 * torch.sigmoid(torch.tensor(2.0)).item()
    assert quality_logits.grad[1].item() >= hard_negative_gradient_floor
    assert quality_logits.grad[1].item() > 50.0 * quality_logits.grad[2].item()


def test_quality_hard_negative_mining_uses_deployment_composite_score():
    labels = torch.tensor([2, IGNORE_LABEL, IGNORE_LABEL])
    targets = torch.tensor([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    mask_logits = torch.tensor([[4.0, 4.0], [0.0, 0.0], [0.0, 0.0]])
    quality_logits = torch.logit(torch.tensor([0.5, 0.9, 0.5]), eps=1e-6).requires_grad_()
    foreground_scores = torch.tensor([0.9, 0.01, 0.9])

    loss = rq_sq_quality_calibration_loss(
        torch,
        quality_logits,
        mask_logits,
        labels,
        targets,
        foreground_scores=foreground_scores,
        ranking_weight=0.0,
        ranking_top_k=1,
        hard_negative_weight=0.1,
    )
    loss.backward()

    assert quality_logits.grad[2].item() > 10.0 * quality_logits.grad[1].item()


def test_quality_deployment_scores_match_apply_admission_product():
    query_logits = torch.tensor([
        [4.0] + [0.0] * (IGNORE_LABEL - 1) + [-4.0],
        [-4.0] + [0.0] * (IGNORE_LABEL - 1) + [4.0],
    ])
    quality_logits = torch.logit(torch.tensor([0.5, 0.9]), eps=1e-6)

    quality, foreground, deployment = rq_sq_quality_deployment_scores(
        torch, query_logits, quality_logits
    )

    assert torch.allclose(deployment, foreground * quality)
    assert deployment[0].item() > deployment[1].item()


def test_quality_unmatched_ceiling_adds_direct_hard_negative_gradient():
    labels = torch.tensor([2, IGNORE_LABEL, IGNORE_LABEL])
    targets = torch.tensor([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    mask_logits = torch.tensor([[4.0, 4.0], [0.0, 0.0], [0.0, 0.0]])

    def gradients(ceiling_weight):
        quality_logits = torch.tensor([0.0, 1.0, -8.0], requires_grad=True)
        loss = rq_sq_quality_calibration_loss(
            torch,
            quality_logits,
            mask_logits,
            labels,
            targets,
            ranking_weight=0.0,
            ranking_top_k=1,
            unmatched_ceiling_weight=ceiling_weight,
            unmatched_ceiling_probability=0.05,
        )
        loss.backward()
        return quality_logits.grad

    baseline = gradients(0.0)
    penalized = gradients(0.75)

    hard_probability = torch.sigmoid(torch.tensor(1.0)).item()
    expected_ceiling_gradient = (
        0.75
        * 2.0
        * (hard_probability - 0.05)
        * hard_probability
        * (1.0 - hard_probability)
    )
    assert penalized[1].item() == pytest.approx(
        baseline[1].item() + expected_ceiling_gradient
    )
    assert penalized[0].item() == pytest.approx(baseline[0].item())
    assert penalized[2].item() == pytest.approx(baseline[2].item())


def test_quality_ranking_supports_configurable_hard_negative_top_k():
    labels = torch.tensor([2, IGNORE_LABEL, IGNORE_LABEL, IGNORE_LABEL])
    targets = torch.ones((4, 2))
    mask_logits = torch.full((4, 2), 4.0)
    quality_logits = torch.tensor([0.0, 3.0, 2.0, -2.0], requires_grad=True)

    top_one = rq_sq_quality_calibration_loss(
        torch, quality_logits, mask_logits, labels, targets, ranking_top_k=1
    )
    top_two = rq_sq_quality_calibration_loss(
        torch, quality_logits, mask_logits, labels, targets, ranking_top_k=2
    )
    top_two.backward()

    assert top_two.item() < top_one.item()
    assert torch.count_nonzero(quality_logits.grad).item() > 0
    with pytest.raises(ValueError, match="ranking_top_k"):
        rq_sq_quality_calibration_loss(torch, quality_logits, mask_logits, labels, targets, ranking_top_k=0)


def test_quality_ranking_uses_non_saturating_logit_surrogate():
    labels = torch.tensor([2, IGNORE_LABEL])
    targets = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    mask_logits = torch.tensor([[4.0, 4.0], [0.0, 0.0]])

    calibration_logits = torch.tensor([-8.0, -6.0], requires_grad=True)
    calibration = rq_sq_quality_calibration_loss(
        torch,
        calibration_logits,
        mask_logits,
        labels,
        targets,
        ranking_weight=0.0,
    )
    calibration.backward()
    calibration_gradient = calibration_logits.grad.detach().clone()

    ranked_logits = torch.tensor([-8.0, -6.0], requires_grad=True)
    ranked = rq_sq_quality_calibration_loss(
        torch,
        ranked_logits,
        mask_logits,
        labels,
        targets,
        ranking_weight=1.0,
        ranking_margin=0.05,
        ranking_top_k=1,
    )
    ranked.backward()

    assert ranked_logits.grad[0].item() == pytest.approx(calibration_gradient[0].item() - 1.0)
    assert ranked_logits.grad[1].item() == pytest.approx(calibration_gradient[1].item())


def test_quality_logit_ranking_surrogate_respects_probability_margin():
    labels = torch.tensor([2, IGNORE_LABEL])
    targets = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    mask_logits = torch.tensor([[0.0, -20.0], [0.0, 0.0]])
    quality_logits = torch.logit(torch.tensor([0.20, 0.10]), eps=1e-6)

    calibration = rq_sq_quality_calibration_loss(
        torch,
        quality_logits,
        mask_logits,
        labels,
        targets,
        ranking_weight=0.0,
        ranking_margin=0.05,
    )
    ranked = rq_sq_quality_calibration_loss(
        torch,
        quality_logits,
        mask_logits,
        labels,
        targets,
        ranking_weight=1.0,
        ranking_margin=0.05,
    )

    assert ranked.item() == pytest.approx(calibration.item())


def test_stuff_overlap_union_uses_symmetric_stop_gradient_consistency():
    first = torch.tensor([[0.0, 1.0]], requires_grad=True)
    second = torch.tensor([[2.0, -1.0]], requires_grad=True)
    rows = [
        ("page", 0, torch.tensor([10, 11]), first),
        ("page", 1, torch.tensor([11, 12]), second),
    ]

    loss, pairs = stuff_overlap_union_consistency_loss(torch, rows)
    assert pairs == 1
    assert loss is not None
    loss.backward()

    assert torch.count_nonzero(first.grad).item() > 0
    assert torch.count_nonzero(second.grad).item() > 0


def test_sq_rq_coverage_gate_blocks_empty_prediction_context():
    gate = sq_rq_coverage_selection_gate(
        {"sq_rq_proxy": {"admitted_query_coverage": 0.0, "context_edge_coverage": 0.001}},
        min_admitted_query_coverage=0.01,
        min_context_edge_coverage=0.002,
    )

    assert gate["passed"] is False
    assert set(gate["blockers"]) == {
        "sq_rq_admitted_query_coverage_below_minimum",
        "sq_rq_context_edge_coverage_below_minimum",
    }


def test_family_seed_loss_has_gradients_for_supported_families():
    seed_logits = torch.zeros((4, 5), requires_grad=True)
    labels = torch.tensor([0, 10, IGNORE_LABEL, 31])
    valid = torch.tensor([True, True, True, True])

    loss = family_seed_loss(torch, seed_logits, labels, valid)

    assert loss is not None
    loss.backward()
    assert seed_logits.grad is not None
    assert torch.count_nonzero(seed_logits.grad).item() > 0
    assert torch.count_nonzero(seed_logits.grad[2]).item() == 0


def test_query_seed_diagnostics_exports_selected_primitive_indices_for_matching():
    selected = query_selected_primitive_indices(
        torch,
        {"seed_indices": torch.tensor([[2, 0, 99]])},
        batch_index=0,
        num_queries=5,
        token_count=4,
        device=torch.device("cpu"),
    )

    assert selected.tolist() == [2, 0, -1, -1, -1]


def test_joint_proxy_gate_rejects_nonconserved_component_proxy():
    counters = Counter()
    query_logits = torch.full((2, IGNORE_LABEL + 1), -4.0)
    query_logits[0, 0] = 4.0
    query_logits[1, IGNORE_LABEL] = 4.0
    mask_logits = torch.tensor([[4.0, 4.0], [-4.0, -4.0]])
    q_labels = torch.tensor([0, IGNORE_LABEL])
    q_masks = torch.tensor([[1.0, 1.0], [0.0, 0.0]])

    update_supervised_component_proxy(
        torch,
        counters,
        query_logits,
        mask_logits,
        q_labels,
        q_masks,
        quality_logits=None,
        rq_available=True,
    )
    component = component_proxy_payload(counters)
    component["proxy_conservation"]["ok"] = False

    score, gate = joint_rq_sq_selection_score(
        {"component_proxy": component},
        min_rq_proxy=0.0,
        min_sq_proxy=0.0,
        max_negative_margin_rate=1.0,
    )

    assert score == -float("inf")
    assert gate["passed"] is False
    assert gate["proxy_conservation_ok"] is False


def test_missing_rq_supervision_masks_no_object_losses_and_proxy_metrics():
    query_logits = torch.zeros((2, IGNORE_LABEL + 1), requires_grad=True)
    q_labels = torch.full((2,), IGNORE_LABEL, dtype=torch.long)
    kwargs = {
        "positive_weight": 2.0,
        "negative_weight": 1.0,
        "positive_margin_floor_loss_weight": 0.0,
        "positive_margin_floor": 0.0,
        "negative_margin_ceiling_loss_weight": 0.0,
        "negative_margin_ceiling": 0.0,
    }

    query_loss, objectness_loss = rq_query_supervision_losses(
        torch,
        torch.nn.CrossEntropyLoss(),
        query_logits,
        q_labels,
        rq_available=False,
        **kwargs,
    )
    counters = Counter()
    update_supervised_component_proxy(
        torch,
        counters,
        query_logits.detach(),
        torch.zeros((2, 3)),
        q_labels,
        torch.zeros((2, 3)),
        rq_available=False,
    )

    assert query_loss is None
    assert objectness_loss is None
    assert query_logits.grad is None
    assert counters["rq_supervision_missing_records"] == 1
    assert counters["query_total"] == 0


def test_available_rq_supervision_still_trains_valid_no_object_targets():
    query_logits = torch.zeros((2, IGNORE_LABEL + 1), requires_grad=True)
    q_labels = torch.full((2,), IGNORE_LABEL, dtype=torch.long)

    query_loss, objectness_loss = rq_query_supervision_losses(
        torch,
        torch.nn.CrossEntropyLoss(),
        query_logits,
        q_labels,
        rq_available=True,
        positive_weight=2.0,
        negative_weight=1.0,
        positive_margin_floor_loss_weight=0.0,
        positive_margin_floor=0.0,
        negative_margin_ceiling_loss_weight=0.0,
        negative_margin_ceiling=0.0,
    )
    (query_loss + objectness_loss).backward()

    assert torch.count_nonzero(query_logits.grad).item() > 0


def test_factorized_objectness_trains_raw_admission_logits():
    query_logits = torch.zeros((2, IGNORE_LABEL + 1), requires_grad=True)
    admission_logits = torch.zeros((2,), requires_grad=True)
    q_labels = torch.tensor([4, IGNORE_LABEL])
    query_loss, objectness_loss = rq_query_supervision_losses(
        torch,
        torch.nn.CrossEntropyLoss(),
        query_logits,
        q_labels,
        rq_available=True,
        admission_logits=admission_logits,
        positive_weight=1.0,
        negative_weight=1.0,
        positive_margin_floor_loss_weight=0.0,
        positive_margin_floor=0.0,
        negative_margin_ceiling_loss_weight=0.0,
        negative_margin_ceiling=0.0,
    )
    (query_loss + objectness_loss).backward()
    assert admission_logits.grad is not None
    assert admission_logits.grad.tolist() == pytest.approx([-0.25, 0.25])


def test_factorized_objectness_balances_positive_and_negative_query_groups():
    admission_logits = torch.zeros((256,), requires_grad=True)
    q_labels = torch.full((256,), IGNORE_LABEL, dtype=torch.long)
    q_labels[:5] = 4
    objectness = query_objectness_loss(
        torch,
        torch.zeros((256, IGNORE_LABEL + 1)),
        q_labels,
        admission_logits=admission_logits,
        positive_weight=2.0,
        negative_weight=1.0,
        positive_margin_floor_loss_weight=0.0,
        positive_margin_floor=0.0,
        negative_margin_ceiling_loss_weight=0.0,
        negative_margin_ceiling=0.0,
    )
    objectness.backward()
    positive_gradient = admission_logits.grad[:5].abs().sum()
    negative_gradient = admission_logits.grad[5:].abs().sum()
    assert (positive_gradient / negative_gradient).item() == pytest.approx(2.0)


def test_unmatched_mask_loss_targets_only_high_admission_negatives():
    mask_logits = torch.tensor([[3.0, 3.0], [1.0, 1.0], [-3.0, -3.0]], requires_grad=True)
    q_labels = torch.full((3,), IGNORE_LABEL, dtype=torch.long)
    admission = torch.tensor([4.0, 2.0, -2.0])
    loss = unmatched_query_empty_mask_loss(
        torch, mask_logits, q_labels, admission, top_k=1,
    )
    loss.backward()
    assert mask_logits.grad[0].abs().sum() > 0
    assert mask_logits.grad[1:].count_nonzero() == 0


def test_precision_phase_gate_fails_closed_on_missing_or_nonfinite_metrics():
    healthy_component = {
        "query_predicted_object_total": 4,
        "query_target_positive_total": 4,
        "target_positive_tokens": 16,
        "query_positive_object_recall": 0.8,
        "query_negative_object_margin_positive_rate": 0.1,
        "mask_token_precision": 0.6,
        "mask_token_recall": 0.7,
        "calibrated_instance_proxy_rq": 0.25,
        "calibrated_query_proposal_coverage": 0.02,
    }

    def history(component):
        return [{"val": {"component_proxy": component}}]

    assert precision_phase_admission_ready(history(healthy_component), 1, 0.2, 0.2, 0.2)
    assert precision_phase_admission_ready(
        history(healthy_component), 1, 0.2, 0.2, 0.2, 0.2, 0.01
    )
    missing_calibrated = dict(healthy_component)
    missing_calibrated.pop("calibrated_instance_proxy_rq")
    assert not precision_phase_admission_ready(
        history(missing_calibrated), 1, 0.2, 0.2, 0.2, 0.2, 0.01
    )
    for key in (
        "query_positive_object_recall",
        "query_negative_object_margin_positive_rate",
        "mask_token_precision",
        "mask_token_recall",
    ):
        malformed = dict(healthy_component)
        malformed[key] = float("nan")
        assert not precision_phase_admission_ready(history(malformed), 1, 0.2, 0.2, 0.2)
        malformed.pop(key)
        assert not precision_phase_admission_ready(history(malformed), 1, 0.2, 0.2, 0.2)


def test_precision_phase_blends_after_warmup_and_backs_off_continuously():
    args = precision_schedule_args()

    warmup_objectness = objectness_schedule(args, 3, precision_phase_allowed=True)
    warmup_mask = mask_loss_schedule(args, 3, precision_phase_allowed=True)
    assert warmup_objectness["query_objectness_positive_weight"] == pytest.approx(8.0)
    assert warmup_objectness["precision_phase_progress"] == 0.0
    assert warmup_mask["mask_positive_weight"] == pytest.approx(8.0)
    assert warmup_mask["precision_phase_progress"] == 0.0

    first_objectness = objectness_schedule(args, 4, precision_phase_allowed=True)
    first_mask = mask_loss_schedule(args, 4, precision_phase_allowed=True)
    assert first_objectness["precision_phase_progress"] == pytest.approx(0.125)
    assert first_objectness["query_objectness_positive_weight"] == pytest.approx(1.90625)
    assert first_mask["precision_phase_progress"] == pytest.approx(0.125)
    assert first_mask["mask_positive_weight"] == pytest.approx(7.5)
    assert first_mask["mask_negative_weight"] == pytest.approx(1.0625)

    second_mask = mask_loss_schedule(
        args,
        5,
        precision_phase_allowed=True,
        previous_precision_phase_progress=first_mask["precision_phase_progress"],
    )
    backed_off = mask_loss_schedule(
        args,
        6,
        precision_phase_allowed=False,
        previous_precision_phase_progress=second_mask["precision_phase_progress"],
    )
    assert second_mask["precision_phase_progress"] == pytest.approx(0.25)
    assert backed_off["precision_phase_progress"] == pytest.approx(0.125)
    assert backed_off["mask_positive_weight"] == pytest.approx(first_mask["mask_positive_weight"])


def test_disabled_precision_phase_preserves_base_objectness_and_mask_objectives():
    args = precision_schedule_args(
        objectness_precision_phase_start_epoch=-1,
        mask_precision_phase_start_epoch=-1,
    )

    objectness = objectness_schedule(args, 20, precision_phase_allowed=True)
    mask = mask_loss_schedule(args, 20, precision_phase_allowed=True)

    assert objectness["precision_phase_progress"] == 0.0
    assert objectness["query_objectness_loss_weight"] == pytest.approx(0.25)
    assert objectness["query_objectness_positive_weight"] == pytest.approx(2.0)
    assert objectness["query_objectness_negative_weight"] == pytest.approx(1.0)
    assert mask["precision_phase_progress"] == 0.0
    assert mask["mask_positive_weight"] == pytest.approx(8.0)
    assert mask["mask_negative_weight"] == pytest.approx(1.0)


def test_zero_precision_phase_start_epoch_means_immediate_after_warmup():
    args = precision_schedule_args(
        objectness_warmup_epochs=0,
        objectness_precision_phase_start_epoch=0,
        mask_precision_phase_start_epoch=0,
    )

    objectness = objectness_schedule(args, 0, precision_phase_allowed=True)
    mask = mask_loss_schedule(args, 0, precision_phase_allowed=True)

    assert objectness["precision_phase_progress"] == pytest.approx(0.125)
    assert mask["precision_phase_progress"] == pytest.approx(0.125)


def test_tversky_beta_is_false_negative_weight():
    targets = torch.tensor([[1.0, 0.0]])
    false_negative = tversky_loss_per_item(
        torch, torch.tensor([[0.0, 0.0]]), targets, alpha=0.35, beta=0.65,
    )
    false_positive = tversky_loss_per_item(
        torch, torch.tensor([[1.0, 1.0]]), targets, alpha=0.35, beta=0.65,
    )

    assert false_negative > false_positive


def test_teacher_match_conflicts_are_reported():
    counters = Counter()
    supervised_labels = torch.tensor([2, 3, IGNORE_LABEL])
    teacher_labels = torch.tensor([4, 3, 5])
    supervised_masks = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    teacher_masks = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    update_teacher_match_conflicts(
        torch,
        counters,
        supervised_labels,
        supervised_masks,
        teacher_labels,
        teacher_masks,
    )

    assert counters["teacher_matched_positive_queries"] == 3
    assert counters["teacher_supervision_overlap_queries"] == 2
    assert counters["teacher_label_conflict_queries"] == 1
    assert counters["teacher_mask_conflict_queries"] == 1


@pytest.mark.parametrize(
    ("key", "changed_value"),
    [
        ("mask_loss_weight", 3.0),
        ("teacher_loss_weight", 0.75),
        ("component_matching", "greedy_gpu"),
        ("recall_class_weight", 3.0),
        ("objectness_precision_phase_start_epoch", 30),
        ("min_val_rq_proxy_for_checkpoint", 0.3),
    ],
)
def test_optimizer_resume_is_blocked_when_full_objective_changes(tmp_path, key, changed_value):
    source_args = optimizer_resume_args()
    model, optimizer, checkpoint = optimizer_checkpoint(source_args)
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)

    requested_args = optimizer_resume_args(**{key: changed_value})
    report = load_resume_checkpoint(torch, model, optimizer, path, requested_args, torch.device("cpu"))

    assert report["optimizer_objective_match"] is False
    assert report["optimizer_objective_mismatch_blocked"] is True
    assert report["optimizer_loaded"] is False
    assert report["training_state_reset_for_objective_mismatch"] is True
    assert report["start_epoch"] == 1
    assert report["history"] == []
    assert report["best_score"] == -float("inf")


def test_production_optimizer_resume_rejects_objective_mismatch(tmp_path):
    source_args = optimizer_resume_args()
    model, optimizer, checkpoint = optimizer_checkpoint(source_args)
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)
    requested_args = optimizer_resume_args(
        training_preset="production",
        mask_loss_weight=3.0,
    )

    with pytest.raises(ValueError, match="production optimizer resume requires an exact objective match"):
        load_resume_checkpoint(
            torch, model, optimizer, path, requested_args, torch.device("cpu")
        )


def test_legacy_checkpoint_without_objective_hash_blocks_optimizer_resume(tmp_path):
    args = optimizer_resume_args()
    model, optimizer, checkpoint = optimizer_checkpoint(args)
    checkpoint.pop("objective_config")
    checkpoint.pop("objective_config_hash")
    path = tmp_path / "legacy.pt"
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="objective config/hash missing"):
        load_resume_checkpoint(torch, model, optimizer, path, args, torch.device("cpu"))


def test_production_preset_forces_joint_fail_closed_gates():
    args = Namespace(
        training_preset="production",
        checkpoint_metric="neg_loss",
        min_val_rq_proxy_for_checkpoint=0.0,
        min_val_sq_proxy_for_checkpoint=0.0,
        max_val_negative_object_margin_rate_for_checkpoint=0.0,
        precision_phase_require_healthy_admission_epochs=0,
        objectness_precision_phase_start_epoch=0,
        mask_precision_phase_start_epoch=0,
        require_final_instance_gate_for_best=False,
        min_final_instance_tp_for_best=0,
        min_final_instance_rq_for_best=0.0,
        min_final_instance_sq_for_best=0.0,
        max_final_instance_fp_for_best=-1,
        num_queries=96,
        component_matching="greedy_gpu",
        allow_optimizer_objective_mismatch=True,
        batch_records=1,
        identity_loss_weight=0.0,
    )

    report = apply_training_preset(args)

    assert report["fail_closed"] is True
    assert args.checkpoint_metric == "joint_rq_sq_proxy"
    assert args.min_val_rq_proxy_for_checkpoint > 0.0
    assert args.min_val_sq_proxy_for_checkpoint > 0.0
    assert args.precision_phase_require_healthy_admission_epochs >= 2
    assert args.precision_phase_min_sq_proxy >= 0.20
    assert args.precision_phase_transition_epochs >= 8
    assert args.objectness_precision_phase_start_epoch == 0
    assert args.mask_precision_phase_start_epoch == 0
    assert args.mask_precision_phase_positive_weight >= 4.0
    assert args.mask_tversky_alpha <= 0.35
    assert args.mask_tversky_beta >= 0.65
    assert args.require_final_instance_gate_for_best is True
    assert args.allow_optimizer_objective_mismatch is False
    assert args.rq_sq_quality_calibration_loss_weight >= 0.25
    assert 0.0 <= args.rq_sq_quality_hard_negative_weight <= 0.1
    assert 0.0 <= args.rq_sq_quality_unmatched_ceiling_weight <= 0.1
    assert args.partial_component_policy == "exclude"


def test_panoptic_quality_head_is_independent_and_optional_at_inference():
    model = make_panoptic_model(torch.nn, torch, 16, 8, 1, 1, 3, query_decoder_layers=1, dropout=0.0)
    features = torch.zeros((1, 4, 16))

    legacy_outputs = model(features)
    quality_outputs = model(features, return_quality=True)

    assert len(legacy_outputs) == 3
    assert len(quality_outputs) == 4
    assert quality_outputs[3].shape == (1, 3)
    assert any(name.startswith("query_quality_head") for name, _parameter in model.named_parameters())


def test_quality_forward_gradient_isolated_to_quality_head():
    model = make_panoptic_model(
        torch.nn, torch, 16, 8, 1, 1, 3, query_decoder_layers=1, dropout=0.0
    )
    quality_logits = model(torch.zeros((1, 4, 16)), return_quality=True)[3]

    quality_logits.square().mean().backward()

    quality_parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if name.startswith("query_quality_head.")
    }
    assert quality_parameters
    assert all(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in quality_parameters.values()
    )
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad).item() == 0
        for name, parameter in model.named_parameters()
        if name not in quality_parameters
    )


def test_quality_calibration_scope_unfreezes_only_quality_head():
    model = make_panoptic_model(
        torch.nn, torch, 16, 8, 1, 1, 3, query_decoder_layers=1, dropout=0.0
    )

    report = configure_quality_calibration_scope(model, enabled=True)
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }

    assert report["enabled"] is True
    assert report["policy"] == "freeze_all_except_query_quality_head_v1"
    assert trainable == {
        "query_quality_head.weight",
        "query_quality_head.bias",
    }
    assert set(report["trainable_parameter_names"]) == trainable
    assert report["frozen_parameter_count"] > 0


def test_explicit_route_classification_loss_maps_semantic_labels_to_families():
    logits = torch.zeros((1, 4, len(FAMILY_NAMES)), requires_grad=True)
    labels = torch.tensor([[12, 30, 18, IGNORE_LABEL]])
    valid = torch.ones_like(labels, dtype=torch.bool)

    loss = explicit_route_classification_loss(torch, logits, labels, valid)
    loss.backward()

    assert loss.item() > 0.0
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad).item() > 0


def test_route_classifier_and_dense_adapter_forward_are_optional():
    model = make_panoptic_model(
        torch.nn,
        torch,
        len(BASE_FEATURES),
        16,
        1,
        2,
        8,
        geometry_decoder_mode="geometry_v2",
        num_stuff_queries=2,
        query_decoder_layers=1,
        dropout=0.0,
        explicit_route_classifier=True,
        dense_attention_feature_adapter=True,
        dense_attention_window_size=4,
    )
    features = torch.zeros((1, 6, len(BASE_FEATURES)))

    outputs = model(features, return_quality=True, return_identity=True)

    assert len(outputs) == 5
    assert model.last_explicit_route_logits.shape == (1, 6, len(FAMILY_NAMES))
    assert model.last_explicit_route_diagnostics["enabled"] is True
    assert model.last_explicit_route_diagnostics["residual_scale"].item() < 0.01
    assert model.last_dense_attention_adapter_diagnostics["window_size"] == 4
    assert model.last_dense_attention_adapter_diagnostics["residual_scale"].item() < 0.01


def test_route_and_dense_adapter_runtime_scale_can_disable_feature_perturbation():
    assert scheduled_auxiliary_scale(epoch=1, enable_after_epoch=2, warmup_epochs=3) == 0.0
    assert scheduled_auxiliary_scale(epoch=2, enable_after_epoch=2, warmup_epochs=3) == pytest.approx(1.0 / 3.0)
    assert scheduled_auxiliary_scale(epoch=4, enable_after_epoch=2, warmup_epochs=3) == 1.0

    model = make_panoptic_model(
        torch.nn,
        torch,
        len(BASE_FEATURES),
        16,
        1,
        2,
        8,
        geometry_decoder_mode="geometry_v2",
        num_stuff_queries=2,
        query_decoder_layers=1,
        dropout=0.0,
        explicit_route_classifier=True,
        dense_attention_feature_adapter=True,
        dense_attention_window_size=4,
    )
    model.route_conditioning_runtime_scale = 0.0
    model.dense_attention_adapter_runtime_scale = 0.0
    features = torch.zeros((1, 6, len(BASE_FEATURES)))

    outputs = model(features, return_quality=True, return_identity=True)

    assert len(outputs) == 5
    assert model.last_explicit_route_diagnostics["residual_scale"].item() == 0.0
    assert model.last_dense_attention_adapter_diagnostics["residual_scale"].item() == 0.0


def test_geometry_connectivity_separates_target_boundaries_and_branch_specialization_is_finite():
    masks = torch.tensor([[2.0, 2.0, -2.0]])
    features = torch.zeros((3, 16))
    features[:, 4:6] = torch.tensor([[0.0, 0.0], [0.1, 0.0], [0.9, 0.9]])
    targets = torch.tensor([[1.0, 1.0, 0.0]])
    connectivity = geometry_mask_connectivity_loss(torch, masks, features, targets)
    blurred = geometry_mask_connectivity_loss(torch, torch.full_like(masks, 2.0), features, targets)
    diagnostics = {
        "semantic": {"enabled": True, "routing_entropy": torch.tensor(0.2)},
        "rq": {"enabled": True, "routing_entropy": torch.tensor(0.5)},
    }
    specialization = moe_branch_specialization_loss(torch, diagnostics)
    assert torch.isfinite(connectivity) and specialization is not None and torch.isfinite(specialization)
    assert connectivity < blurred


def test_geometry_connectivity_ignores_out_of_range_invalid_padding_neighbors():
    masks = torch.tensor([[1.0, -1.0, 0.5]], requires_grad=True)
    features = torch.zeros((3, 16))
    targets = torch.tensor([[1.0, 0.0, 1.0]])
    indices = torch.tensor([[1, 7], [0, 7], [1, 7]])
    valid = torch.tensor([[True, False], [True, False], [True, False]])

    loss = geometry_mask_connectivity_loss(
        torch, masks, features, targets, neighbor_indices=indices, neighbor_valid=valid
    )

    assert torch.isfinite(loss)
    loss.backward()
    assert masks.grad is not None


def test_geometry_connectivity_rejects_out_of_range_valid_neighbors():
    masks = torch.tensor([[1.0, -1.0, 0.5]])
    features = torch.zeros((3, 16))
    indices = torch.tensor([[1], [3], [1]])
    valid = torch.ones_like(indices, dtype=torch.bool)

    with pytest.raises(ValueError, match="outside the local mask token range"):
        geometry_mask_connectivity_loss(
            torch, masks, features, neighbor_indices=indices, neighbor_valid=valid
        )


def test_connectivity_uses_sparse_endpoint_neighbors_without_cdist(monkeypatch):
    features = torch.zeros((8, 16))
    features[:, :4] = torch.arange(8, dtype=torch.float32).view(-1, 1).repeat(1, 4) / 10.0
    features[:, 4:6] = torch.stack([torch.arange(8, dtype=torch.float32) / 10.0, torch.zeros(8)], dim=-1)
    indices, valid, _distance = sparse_endpoint_neighbor_graph(torch, features.unsqueeze(0), neighbors=3)
    monkeypatch.setattr(torch, "cdist", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("cdist is forbidden")))
    loss = geometry_mask_connectivity_loss(
        torch,
        torch.zeros((1, 8), requires_grad=True),
        features,
        torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]),
        neighbor_indices=indices.squeeze(0),
        neighbor_valid=valid.squeeze(0),
    )
    assert torch.isfinite(loss)
