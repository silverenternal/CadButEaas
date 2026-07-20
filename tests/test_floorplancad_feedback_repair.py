from argparse import Namespace
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]

from experiments.floorplancad_sq_rq_cross_attention import SqRqCrossAttention
from experiments.floorplancad_apply_line_token_panoptic_moe import instances_from_windows
from experiments.floorplancad_v4_training_gate import (
    RAW_METADATA_ALIGNMENT,
    RAW_SVG_COORDINATE_PROTOCOL,
    preflight,
    validate_window_cache_lineage,
    validate_window_report,
)
from experiments import floorplancad_build_line_json_primitive_cache as CACHE_BUILDER
from experiments.floorplancad_train_line_token_panoptic_moe import (
    IGNORE_LABEL,
    build_optimizer_and_scheduler,
    component_proxy_payload,
    make_panoptic_model,
    joint_rq_sq_selection_score,
    quality_checkpoint_selection_gate,
    rq_sq_quality_ranking_pairs,
    router_usage_selection_gate,
    sq_rq_checkpoint_promotion_ready,
    sq_rq_deployment_thresholds,
    sq_rq_training_threshold_schedule,
    target_diagnostic_payload,
    update_component_proxy,
    update_target_diagnostics,
)


def _proxy_for(logits, mask_logits):
    counters = Counter()
    update_component_proxy(
        torch,
        counters,
        torch.tensor([logits], dtype=torch.float32),
        torch.tensor([mask_logits], dtype=torch.float32),
        torch.tensor([2]),
        torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
    )
    return component_proxy_payload(counters)


def test_instance_proxy_requires_admission_class_and_iou():
    no_object = [0.0] * (IGNORE_LABEL + 1)
    no_object[IGNORE_LABEL] = 4.0
    wrong_class = [0.0] * (IGNORE_LABEL + 1)
    wrong_class[3] = 4.0
    right_class = [0.0] * (IGNORE_LABEL + 1)
    right_class[2] = 4.0

    for payload in (
        _proxy_for(no_object, [4.0, 4.0, -4.0, -4.0]),
        _proxy_for(wrong_class, [4.0, 4.0, -4.0, -4.0]),
        _proxy_for(right_class, [-4.0, -4.0, 4.0, 4.0]),
    ):
        assert payload["instance_proxy_tp"] == 0
        assert payload["instance_proxy_rq"] == 0.0
        assert payload["instance_proxy_sq"] == 0.0
        assert payload["instance_proxy_pq"] == 0.0

    payload = _proxy_for(right_class, [4.0, 4.0, -4.0, -4.0])
    assert payload["instance_proxy_tp"] == 1
    assert payload["instance_proxy_fp"] == 0
    assert payload["instance_proxy_fn"] == 0
    assert payload["instance_proxy_rq"] == pytest.approx(1.0)
    assert payload["instance_proxy_sq"] == pytest.approx(1.0)


def test_instance_proxy_counts_false_positives_on_empty_target_windows():
    logits = torch.zeros((2, IGNORE_LABEL + 1))
    logits[:, 4] = 3.0
    counters = Counter()
    update_component_proxy(
        torch,
        counters,
        logits,
        torch.zeros((2, 3)),
        torch.full((2,), IGNORE_LABEL, dtype=torch.long),
        torch.zeros((2, 3)),
    )
    payload = component_proxy_payload(counters)
    assert payload["instance_proxy_tp"] == 0
    assert payload["instance_proxy_fp"] == 2


def test_quality_calibrated_training_proxy_matches_apply_admission():
    query_logits = torch.zeros((2, IGNORE_LABEL + 1))
    query_logits[:, 2] = 4.0
    mask_logits = torch.tensor([[4.0, 4.0], [4.0, 4.0]])
    quality_logits = torch.tensor([4.0, -4.0])
    q_labels = torch.tensor([2, IGNORE_LABEL])
    q_masks = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    counters = Counter()

    update_component_proxy(
        torch,
        counters,
        query_logits,
        mask_logits,
        q_labels,
        q_masks,
        quality_logits,
        deployment_min_query_score=0.2,
    )
    payload = component_proxy_payload(counters)
    instances, diagnostics = instances_from_windows(
        [[10, 11]],
        [query_logits.numpy()],
        [mask_logits.numpy()],
        quality_logits_rows=[quality_logits.numpy()],
        query_admission_policy="respect_no_object",
        min_query_score=0.2,
        min_object_margin=0.0,
        mask_threshold=0.5,
        merge_iou_threshold=0.25,
        merge_overlap_threshold=0.5,
        max_instances=8,
    )

    assert payload["calibrated_query_admitted_total"] == diagnostics["admitted_object_queries"] == 1
    assert payload["calibrated_query_rejected_low_score_total"] == diagnostics["rejected_low_score_queries"] == 1
    assert payload["calibrated_instance_proxy_tp"] == 1
    assert len(instances) == 1


def test_component_proxy_splits_low_score_and_empty_mask_for_furniture():
    query_logits = torch.full((3, IGNORE_LABEL + 1), -4.0)
    query_logits[:, 12] = 4.0
    mask_logits = torch.tensor([
        [4.0, 4.0, -4.0, -4.0],
        [4.0, 4.0, -4.0, -4.0],
        [-4.0, -4.0, -4.0, -4.0],
    ])
    quality_logits = torch.tensor([4.0, -4.0, 4.0])
    q_labels = torch.tensor([12, 12, 12])
    q_masks = torch.tensor([
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
    ])
    counters = Counter()

    update_component_proxy(
        torch,
        counters,
        query_logits,
        mask_logits,
        q_labels,
        q_masks,
        quality_logits,
        deployment_min_query_score=0.2,
    )
    payload = component_proxy_payload(counters)
    furniture = next(row for row in payload["per_family"] if row["family"] == "furniture")

    assert furniture["TP"] == 1
    assert furniture["fn_attribution"]["low_score"] == 2
    assert furniture["fn_attribution"].get("empty_mask", 0) == 0
    assert furniture["fn_attribution"]["low_score_or_empty_mask"] == 2
    assert furniture["score_diagnostics"]["positive_empty_mask_rate"] == pytest.approx(0.0)
    assert furniture["score_diagnostics"]["positive_mask_objectness_score_p10"] is not None
    assert payload["furniture_threshold_sweep"][0]["target"] == 3


def test_legacy_admission_rq_is_explicitly_separated_from_instance_rq():
    counters = Counter({
        "query_target_positive_total": 10,
        "query_positive_predicted_object": 8,
        "query_target_negative_total": 20,
        "query_negative_predicted_no_object": 18,
    })

    payload = component_proxy_payload(counters)

    assert payload["legacy_admission_balanced_rq"] == pytest.approx(2.0 * 0.8 * 0.9 / (0.8 + 0.9))
    assert payload["instance_proxy_rq"] == 0.0
    assert payload["legacy_admission_balanced_rq_protocol"].endswith("_not_instance_rq")
    assert payload["instance_proxy_protocol"] == "query_class_and_length_weighted_binary_mask_iou_0p5_tp_fp_fn_v2"


def test_joint_gate_reports_actual_query_admission_recall():
    _score, gate = joint_rq_sq_selection_score(
        {"component_proxy": {
            "query_positive_object_recall": 0.55,
            "instance_proxy_recall": 0.01,
            "query_predicted_object_total": 2,
            "calibrated_query_admitted_total": 1,
            "calibrated_query_proposal_total": 1,
            "calibrated_query_admitted_coverage": 0.5,
            "query_negative_object_margin_positive_rate": 0.0,
            "calibrated_instance_proxy_rq": 0.2,
            "calibrated_instance_proxy_sq": 0.6,
            "calibrated_instance_proxy_pq": 0.12,
            "calibrated_instance_proxy_precision": 1.0,
            "calibrated_instance_proxy_tp": 1,
            "calibrated_instance_proxy_fp": 0,
            "calibrated_instance_proxy_fn": 4,
        }},
        min_rq_proxy=0.1,
        min_sq_proxy=0.2,
        max_negative_margin_rate=0.2,
    )
    assert gate["query_positive_object_recall"] == pytest.approx(0.55)
    assert gate["selection_protocol"] == "quality_calibrated_deployment_admission_v1"


def test_quality_checkpoint_gate_fails_closed_without_support():
    gate = quality_checkpoint_selection_gate(
        {"quality_proxy": {
            "items": 0,
            "unmatched_items": 0,
            "ranking_pairs": 0,
            "unmatched_predicted_mean": 0.0,
            "ranking_violation_rate": 0.0,
        }},
        max_unmatched_quality=0.05,
        max_ranking_violation_rate=0.50,
    )

    assert gate["passed"] is False
    assert set(gate["blockers"]) == {
        "quality_support_missing",
        "quality_ranking_support_missing",
        "unmatched_deployment_score_support_missing",
    }


@pytest.mark.parametrize(
    ("unmatched_quality", "ranking_violation", "expected_blocker"),
    [
        (0.050001, 0.50, "unmatched_deployment_score_above_maximum"),
        (0.05, 0.500001, "quality_ranking_violation_above_maximum"),
    ],
)
def test_quality_checkpoint_gate_rejects_threshold_violations(
    unmatched_quality,
    ranking_violation,
    expected_blocker,
):
    gate = quality_checkpoint_selection_gate(
        {"quality_proxy": {
            "items": 32,
            "unmatched_items": 256,
            "ranking_pairs": 128,
            "unmatched_predicted_mean": unmatched_quality,
            "unmatched_deployment_score_mean": unmatched_quality,
            "ranking_violation_rate": ranking_violation,
        }},
        max_unmatched_quality=0.05,
        max_ranking_violation_rate=0.50,
    )

    assert gate["passed"] is False
    assert gate["blockers"] == [expected_blocker]


def test_quality_checkpoint_gate_accepts_inclusive_boundaries():
    gate = quality_checkpoint_selection_gate(
        {"quality_proxy": {
            "items": 32,
            "unmatched_items": 256,
            "ranking_pairs": 128,
            "unmatched_predicted_mean": 0.05,
            "unmatched_deployment_score_mean": 0.05,
            "ranking_violation_rate": 0.50,
        }},
        max_unmatched_quality=0.05,
        max_ranking_violation_rate=0.50,
    )

    assert gate["passed"] is True
    assert gate["blockers"] == []


def _joint_gate_validation_proxy(**component_overrides):
    component = {
        "query_positive_object_recall": 0.9,
        "query_predicted_object_total": 64,
        "calibrated_query_admitted_total": 48,
        "calibrated_query_proposal_total": 40,
        "calibrated_query_admitted_coverage": 0.08,
        "calibrated_query_proposal_coverage": 0.01,
        "query_negative_object_margin_positive_rate": 0.1,
        "mask_token_precision": 0.35,
        "mask_token_recall": 0.40,
        "calibrated_instance_proxy_rq": 0.20,
        "calibrated_instance_proxy_sq": 0.58,
        "calibrated_instance_proxy_pq": 0.116,
        "calibrated_instance_proxy_precision": 0.6,
        "calibrated_instance_proxy_tp": 32,
        "calibrated_instance_proxy_fp": 16,
        "calibrated_instance_proxy_fn": 32,
        "calibrated_instance_proxy_protocol": "respect_no_object_class_times_quality_threshold_then_length_weighted_binary_mask_iou_0p5_tp_fp_fn_v2",
        "proxy_conservation": {"ok": True},
    }
    component.update(component_overrides)
    return {"component_proxy": component}


@pytest.mark.parametrize(
    ("component_overrides", "reported_key"),
    [
        ({"mask_token_precision": 0.349999}, "mask_token_precision"),
        ({"mask_token_recall": 0.399999}, "mask_token_recall"),
        ({"calibrated_instance_proxy_tp": 31}, "instance_proxy_tp"),
        ({"calibrated_query_proposal_coverage": 0.009999}, "calibrated_query_proposal_coverage"),
    ],
)
def test_joint_gate_rejects_mask_tp_and_proposal_shortfalls(component_overrides, reported_key):
    score, gate = joint_rq_sq_selection_score(
        _joint_gate_validation_proxy(**component_overrides),
        min_rq_proxy=0.20,
        min_sq_proxy=0.58,
        max_negative_margin_rate=0.20,
        min_mask_precision=0.35,
        min_mask_recall=0.40,
        min_instance_tp=32,
        min_proposal_coverage=0.01,
    )

    assert score == -float("inf")
    assert gate["passed"] is False
    assert reported_key in gate


def test_joint_gate_accepts_mask_tp_and_proposal_boundaries():
    score, gate = joint_rq_sq_selection_score(
        _joint_gate_validation_proxy(),
        min_rq_proxy=0.20,
        min_sq_proxy=0.58,
        max_negative_margin_rate=0.20,
        min_mask_precision=0.35,
        min_mask_recall=0.40,
        min_instance_tp=32,
        min_proposal_coverage=0.01,
    )

    assert score == pytest.approx(0.116)
    assert gate["passed"] is True
    assert gate["instance_proxy_tp"] == 32
    assert gate["calibrated_query_proposal_coverage"] == pytest.approx(0.01)


def test_v4_launcher_rejects_legacy_unaligned_cache(tmp_path):
    cache = tmp_path / "train.jsonl"
    cache.write_text('{"input_feature_lineage":{"primitive_kind":"raw"}}\n', encoding="utf-8")
    rejected = validate_window_cache_lineage(cache)
    assert rejected["valid"] is False
    cache.write_text(
        json.dumps({"input_feature_lineage": {
            "raw_metadata_alignment": RAW_METADATA_ALIGNMENT,
            "raw_svg_coordinate_protocol": RAW_SVG_COORDINATE_PROTOCOL,
        }}) + "\n",
        encoding="utf-8",
    )
    assert validate_window_cache_lineage(cache)["valid"] is True
    with cache.open("a", encoding="utf-8") as handle:
        handle.write('{"input_feature_lineage":{"primitive_kind":"legacy_tail"}}\n')
    mixed = validate_window_cache_lineage(cache)
    assert mixed["valid"] is False
    assert mixed["last_raw_metadata_alignment"] is None


@pytest.mark.parametrize(
    "script",
    [
        "experiments/floorplancad_build_windowed_line_token_cache.py",
        "experiments/floorplancad_make_query_safe_window_cache.py",
    ],
)
def test_v4_cache_stage_scripts_are_directly_executable(script):
    completed = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def _write_aligned_cache(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"input_feature_lineage": {
            "raw_metadata_alignment": RAW_METADATA_ALIGNMENT,
            "raw_svg_coordinate_protocol": RAW_SVG_COORDINATE_PROTOCOL,
        }}) + "\n",
        encoding="utf-8",
    )


def _gate_args(tmp_path):
    return Namespace(
        primitive_cache_dir=tmp_path / "primitive",
        raw_window_cache_dir=tmp_path / "raw_window",
        window_cache_dir=tmp_path / "query_safe",
        launch_dir=tmp_path / "launch",
        raw_window_report=tmp_path / "raw_window.json",
        window_report=tmp_path / "query_safe.json",
        train_report=tmp_path / "train.json",
        query_capacity=256,
        allow_partial_cache=True,
    )


def test_v4_preflight_orders_primitive_raw_and_query_safe_rebuilds(tmp_path):
    args = _gate_args(tmp_path)
    gate = preflight(args)
    assert "mode=build-primitive" in gate["next_actions"][0]

    for split in ("train", "val"):
        _write_aligned_cache(args.primitive_cache_dir / f"{split}_primitive_cache.jsonl")
    gate = preflight(args)
    assert gate["next_actions"] == ["run mode=build-raw-window after train/val primitive cache completes"]

    for split in ("train", "val"):
        _write_aligned_cache(args.raw_window_cache_dir / f"{split}_windowed_primitive_cache.jsonl")
    args.raw_window_report.write_text(json.dumps({
        "status": "exported",
        "cache_dir": str(args.primitive_cache_dir),
        "output_dir": str(args.raw_window_cache_dir),
        "splits": {"train": {}, "val": {}},
    }), encoding="utf-8")
    gate = preflight(args)
    assert gate["next_actions"] == ["run mode=build-window after raw train/val window cache completes"]

    for split in ("train", "val"):
        _write_aligned_cache(args.window_cache_dir / f"{split}_windowed_primitive_cache.jsonl")
    args.window_report.write_text(json.dumps({
        "status": "exported",
        "source_dir": str(args.raw_window_cache_dir),
        "output_dir": str(args.window_cache_dir),
        "query_capacity": 256,
        "training_safe": True,
        "splits": {
            "train": {"input_windows": 2, "output_windows": 2},
            "val": {"input_windows": 1, "output_windows": 1},
        },
    }), encoding="utf-8")
    gate = preflight(args)
    assert gate["status"] == "ready"
    assert gate["next_actions"] == ["run mode=launch-train after all cache lineage and reports pass"]


def test_query_safe_report_must_match_paths_capacity_and_zero_drop(tmp_path):
    report = tmp_path / "report.json"
    report.write_text(json.dumps({
        "status": "exported",
        "source_dir": str(tmp_path / "source"),
        "output_dir": str(tmp_path / "output"),
        "query_capacity": 256,
        "training_safe": True,
        "splits": {
            "train": {"input_windows": 4, "output_windows": 3},
            "val": {"input_windows": 2, "output_windows": 2},
        },
    }), encoding="utf-8")
    result = validate_window_report(
        report,
        expected_paths={"source_dir": tmp_path / "source", "output_dir": tmp_path / "output"},
        expected_query_capacity=256,
        require_training_safe=True,
    )
    assert result["valid"] is False
    assert "dropped records" in result["blocker"]


def test_primitive_cache_write_is_atomic_on_failure(tmp_path):
    output = tmp_path / "cache.jsonl"
    output.write_text("previous\n", encoding="utf-8")

    def broken_rows():
        yield {"record_id": "partial"}
        raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        CACHE_BUILDER.write_jsonl(output, broken_rows())
    assert output.read_text(encoding="utf-8") == "previous\n"
    assert not list(tmp_path.glob("*.tmp"))


def test_target_aware_quality_pairs_cap_low_iou_margin():
    prediction = torch.tensor([0.01, 0.4])
    target = torch.tensor([0.02, 0.0])
    positive = torch.tensor([True, False])
    _positive, _negative, margin = rq_sq_quality_ranking_pairs(
        torch, prediction, target, positive, ranking_margin=0.05, ranking_top_k=1,
    )
    assert margin.tolist() == pytest.approx([0.02])


def test_top_one_branch_router_receives_task_gradient():
    torch.manual_seed(17)
    model = make_panoptic_model(
        torch.nn, torch, feature_dim=12, hidden_dim=16, layers=1, heads=4,
        num_queries=8, num_labels=6, query_decoder_layers=1, dropout=0.0,
        geometry_decoder_mode="legacy_debug", ownership_enabled=False,
        learned_sparse_router=True, typed_branch_routers=True,
        branch_num_experts=2, branch_top_k=1, branch_dropless=True,
    ).train()
    features = torch.randn(2, 9, 16)
    for branch_name, router in model.branch_routers.items():
        model.zero_grad(set_to_none=True)
        routed, _diagnostics = model._route_typed_branch(features, branch_name, torch.ones(2, 9, dtype=torch.bool))
        routed.square().mean().backward()
        gradients = [parameter.grad for parameter in router.parameters()]
        assert any(gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0 for gradient in gradients)


def test_router_gate_rejects_collapsed_usage():
    gate = router_usage_selection_gate(
        {"router_proxy": {"mean_expert_probability": [1.0, 0.0, 0.0, 0.0], "assignment_fraction": [1.0, 0.0, 0.0, 0.0]}},
        max_dominant_probability=0.8,
        min_assignment_fraction=0.05,
    )
    assert gate["passed"] is False
    assert len(gate["blockers"]) == 2


def test_sq_rq_default_path_is_forward_identity_with_adapter_gradient():
    torch.manual_seed(19)
    module = SqRqCrossAttention(
        primitive_dim=8, rq_query_dim=8, hidden_dim=8, num_classes=4, heads=2,
        query_confidence_threshold=0.1, token_membership_threshold=0.1,
    ).train()
    primitive = torch.randn(1, 4, 8)
    outputs = module(
        primitive,
        torch.randn(1, 2, 8),
        torch.full((1, 2, 4), 4.0),
        torch.tensor([[[5.0, -3.0, -3.0, -3.0], [5.0, -3.0, -3.0, -3.0]]]),
    )
    assert torch.equal(outputs["sq_tokens"], primitive)
    outputs["sq_tokens"].square().mean().backward()
    final_projection = module.sq_adapter[-1]
    assert final_projection.weight.grad is not None
    assert final_projection.weight.grad.abs().sum() > 0


def test_sq_rq_schedule_warms_training_but_not_final_thresholds():
    args = Namespace(
        sq_rq_coverage_warmup_epochs=4,
        sq_rq_enable_after_epoch=5,
        sq_rq_query_confidence_threshold=0.6,
        sq_rq_token_membership_threshold=0.5,
        sq_rq_warmup_query_confidence_threshold=0.2,
        sq_rq_warmup_token_membership_threshold=0.1,
        sq_rq_training_membership_temperature=0.1,
    )
    inactive = sq_rq_training_threshold_schedule(args, 4)
    first = sq_rq_training_threshold_schedule(args, 5)
    final = sq_rq_training_threshold_schedule(args, 9)
    deployment = sq_rq_deployment_thresholds(args, 5)
    assert inactive["phase"] == "inactive_before_enable"
    assert first["phase"] == "scheduled_soft_coverage"
    assert first["query_confidence_threshold"] < final["query_confidence_threshold"]
    assert final["phase"] == "frozen_hard_thresholds"
    assert first["training_membership_temperature"] > 0.0
    assert final["training_membership_temperature"] == 0.0
    assert deployment["phase"] == "deployment_hard_thresholds"
    assert deployment["query_confidence_threshold"] == final["query_confidence_threshold"]
    assert deployment["token_membership_threshold"] == final["token_membership_threshold"]
    assert deployment["training_membership_temperature"] == 0.0
    assert not sq_rq_checkpoint_promotion_ready(True, first)
    assert sq_rq_checkpoint_promotion_ready(True, final)
    assert not sq_rq_checkpoint_promotion_ready(False, first)
    assert sq_rq_checkpoint_promotion_ready(False, first, auto_fused=True)


def test_target_accounting_separates_policy_exclusions_from_capacity_drops():
    counters = Counter()
    update_target_diagnostics(counters, {
        "raw_target_components_total": 3,
        "policy_eligible_target_components": 2,
        "policy_excluded_target_components": 1,
        "capacity_target_components_total": 2,
        "capacity_target_components_kept": 2,
        "capacity_target_components_dropped": 0,
        "target_components_total": 3,
        "target_components_kept": 2,
        "target_components_dropped": 1,
        "partial_mask_components_too_small": 1,
    })
    payload = target_diagnostic_payload(counters)
    assert payload["capacity_target_keep_rate"] == 1.0
    assert payload["partial_mask_components_too_small"] == 1


def test_scheduler_uses_distinct_router_parameter_group_and_steps():
    model = make_panoptic_model(
        torch.nn, torch, feature_dim=12, hidden_dim=16, layers=1, heads=4,
        num_queries=8, num_labels=6, query_decoder_layers=1, dropout=0.0,
        geometry_decoder_mode="legacy_debug", ownership_enabled=False,
        learned_sparse_router=True,
    )
    args = Namespace(lr=1e-3, lr_warmup_steps=2, lr_decay_steps=8, lr_min_scale=0.1, router_lr_scale=0.5, head_lr_scale=1.0)
    optimizer, scheduler = build_optimizer_and_scheduler(torch, model, args)
    assert scheduler is not None
    by_name = {group["name"]: group["lr"] for group in optimizer.param_groups}
    assert by_name["router"] < by_name["backbone"]
    optimizer.zero_grad(set_to_none=True)
    next(model.parameters()).sum().backward()
    optimizer.step()
    scheduler.step()
    assert scheduler.last_epoch >= 1
