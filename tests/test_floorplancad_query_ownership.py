import inspect
import json

import numpy as np
import torch
import pytest

from experiments.floorplancad_apply_line_token_panoptic_moe import (
    branch_route_trace_payload,
    calibrated_proposal_score,
    instances_from_windows,
    primitive_reuse_diagnostics,
)
from experiments.floorplancad_query_ownership import ownership_cross_entropy, ownership_mask_consistency_loss, ownership_targets, select_global_owners
from experiments import floorplancad_train_line_token_panoptic_moe as train


def test_targets_include_matched_queries_null_and_exclude_partial() -> None:
    q_masks = torch.tensor([[1, 1, 0, 0, 0], [0, 0, 1, 0, 0]], dtype=torch.float32)
    q_labels = torch.tensor([3, 3])
    valid = torch.tensor([True, True, True, True, False])
    target = ownership_targets(torch, q_masks, q_labels, valid)
    assert target.tolist() == [0, 0, 1, 2, -100]
    logits = torch.randn(5, 3, requires_grad=True)
    loss = ownership_cross_entropy(torch, logits, target)
    loss.backward()
    assert torch.isfinite(loss) and logits.grad is not None


def test_ownership_consistency_tracks_mask_membership_and_ignores_partial_tokens() -> None:
    mask_logits = torch.tensor([[5.0, -5.0, 0.0], [-5.0, 5.0, 0.0]])
    query_labels = torch.tensor([3, 4])
    valid = torch.tensor([True, True, False])
    aligned = torch.tensor([[5.0, -5.0, -3.0], [-5.0, 5.0, -3.0], [0.0, 0.0, 0.0]], requires_grad=True)
    misaligned = -aligned.detach().clone()
    aligned_loss = ownership_mask_consistency_loss(torch, aligned, mask_logits, query_labels, valid, no_object_label=35)
    misaligned_loss = ownership_mask_consistency_loss(torch, misaligned, mask_logits, query_labels, valid, no_object_label=35)
    assert aligned_loss < misaligned_loss
    aligned_loss.backward()
    assert aligned.grad is not None and torch.isfinite(aligned.grad).all()


def test_global_owner_masks_are_unique_across_same_and_cross_label_queries() -> None:
    query_logits = np.full((3, 36), -5.0, dtype=np.float32)
    query_logits[0, 4] = query_logits[1, 4] = query_logits[2, 7] = 5.0
    query_logits[:, 35] = -3.0
    mask_logits = np.full((3, 5), 5.0, dtype=np.float32)
    owner_logits = np.full((5, 4), -4.0, dtype=np.float32)
    owner_logits[0:2, 0] = 4.0
    owner_logits[2, 1] = 4.0
    owner_logits[3, 2] = 4.0
    owner_logits[4, 3] = 4.0
    owners = select_global_owners(owner_logits, np.array([True, True, True]))
    assert owners.tolist() == [0, 0, 1, 2, 3]
    instances, diagnostics = instances_from_windows(
        [[10, 11, 12, 13, 14]], [query_logits], [mask_logits],
        ownership_logits_rows=[owner_logits], identity_embedding_rows=[np.eye(3, dtype=np.float32)],
        quality_logits_rows=[np.zeros(3, dtype=np.float32)], query_admission_policy="respect_no_object",
        min_query_score=0.0, mask_threshold=0.5, merge_iou_threshold=0.5,
        merge_overlap_threshold=0.5, max_instances=8,
    )
    reuse = primitive_reuse_diagnostics(instances)
    assert reuse["reused_primitives"] == 0
    assert reuse["cross_label_reused_primitives"] == 0
    assert diagnostics["null_owned_primitives"] == 1
    assert sorted(tuple(row["primitive_ids"]) for row in instances) == [(10, 11), (12,), (13,)]
    assert all(row["ownership_before_mask"] for row in instances)
    assert all(
        abs(row["score"] - row["class_score"] * row["quality_score"] * row["mask_objectness_score"]) < 1e-7
        for row in instances
    )


def test_calibrated_proposal_score_does_not_apply_admission_twice() -> None:
    assert calibrated_proposal_score(0.8, 0.5) == pytest.approx(0.4)


def test_instance_semantic_remask_removes_wrong_label_primitives() -> None:
    query_logits = np.full((1, 36), -10.0, dtype=np.float32)
    query_logits[0, 4] = 10.0
    mask_logits = np.asarray([[10.0, 10.0, 10.0]], dtype=np.float32)
    semantic_logits = np.full((3, 35), -10.0, dtype=np.float32)
    semantic_logits[0, 4] = 10.0
    semantic_logits[1, 8] = 10.0
    semantic_logits[2, 4] = 10.0

    instances, diagnostics = instances_from_windows(
        [[101, 102, 103]],
        [query_logits],
        [mask_logits],
        semantic_logits_rows=[semantic_logits],
        query_admission_policy="respect_no_object",
        min_query_score=0.0,
        min_object_margin=0.0,
        mask_threshold=0.5,
        merge_iou_threshold=0.25,
        merge_overlap_threshold=0.5,
        max_instances=8,
        instance_semantic_remask_policy="label_consistent_nonempty",
    )

    assert instances[0]["primitive_ids"] == [101, 103]
    assert instances[0]["semantic_remask"]["removed_count"] == 1
    assert diagnostics["semantic_remask_removed_primitives"] == 1


def test_instance_semantic_remask_falls_back_when_all_primitives_disagree() -> None:
    query_logits = np.full((1, 36), -10.0, dtype=np.float32)
    query_logits[0, 4] = 10.0
    mask_logits = np.asarray([[10.0, 10.0]], dtype=np.float32)
    semantic_logits = np.full((2, 35), -10.0, dtype=np.float32)
    semantic_logits[:, 8] = 10.0

    instances, diagnostics = instances_from_windows(
        [[101, 102]],
        [query_logits],
        [mask_logits],
        semantic_logits_rows=[semantic_logits],
        query_admission_policy="respect_no_object",
        min_query_score=0.0,
        min_object_margin=0.0,
        mask_threshold=0.5,
        merge_iou_threshold=0.25,
        merge_overlap_threshold=0.5,
        max_instances=8,
        instance_semantic_remask_policy="label_consistent_nonempty",
    )

    assert instances[0]["primitive_ids"] == [101, 102]
    assert instances[0]["semantic_remask"]["fallback_used"] is True
    assert diagnostics["semantic_remask_fallback_instances"] == 1


@pytest.mark.parametrize("ownership_enabled", [False, True])
def test_quality_calibrated_score_controls_query_admission(ownership_enabled: bool) -> None:
    query_logits = np.full((1, 36), -5.0, dtype=np.float32)
    query_logits[0, 4] = 5.0
    query_logits[0, 35] = -3.0
    kwargs = {}
    if ownership_enabled:
        kwargs = {
            "ownership_logits_rows": [np.array([[5.0, -5.0], [5.0, -5.0]], dtype=np.float32)],
            "identity_embedding_rows": [np.array([[1.0, 0.0]], dtype=np.float32)],
        }
    low_quality, low_diagnostics = instances_from_windows(
        [[10, 11]],
        [query_logits],
        [np.array([[5.0, 5.0]], dtype=np.float32)],
        quality_logits_rows=[np.array([-5.0], dtype=np.float32)],
        query_admission_policy="respect_no_object",
        min_query_score=0.2,
        mask_threshold=0.5,
        merge_iou_threshold=0.5,
        merge_overlap_threshold=0.5,
        max_instances=8,
        **kwargs,
    )
    high_quality, high_diagnostics = instances_from_windows(
        [[10, 11]],
        [query_logits],
        [np.array([[5.0, 5.0]], dtype=np.float32)],
        quality_logits_rows=[np.array([5.0], dtype=np.float32)],
        query_admission_policy="respect_no_object",
        min_query_score=0.2,
        mask_threshold=0.5,
        merge_iou_threshold=0.5,
        merge_overlap_threshold=0.5,
        max_instances=8,
        **kwargs,
    )
    assert low_quality == []
    assert low_diagnostics["rejected_low_score_queries"] == 1
    assert len(high_quality) == 1
    assert high_diagnostics["admitted_object_queries"] == 1
    assert high_diagnostics["query_score_distributions"]["calibrated_score"]["count"] == 1


def test_dropless_route_trace_keeps_null_capacity() -> None:
    payload = branch_route_trace_payload({
        "enabled": True,
        "capacity": None,
        "overflow_assignments": torch.tensor(0),
        "usage_gate_passed": torch.tensor(True),
        "assignment_fraction": torch.tensor([0.5, 0.5]),
    })
    assert payload["capacity"] is None
    assert payload["assignment_fraction"] == [0.5, 0.5]


def test_mask_guided_ownership_cannot_assign_primitives_outside_query_masks() -> None:
    query_logits = np.full((2, 36), -5.0, dtype=np.float32)
    query_logits[:, 4] = 5.0
    query_logits[:, 35] = -3.0
    mask_logits = np.array([[5.0, -5.0, -5.0], [-5.0, 5.0, -5.0]], dtype=np.float32)
    owner_logits = np.array([[5.0, 1.0, -2.0], [5.0, 1.0, -2.0], [5.0, 1.0, -2.0]], dtype=np.float32)
    instances, diagnostics = instances_from_windows(
        [[10, 11, 12]], [query_logits], [mask_logits],
        ownership_logits_rows=[owner_logits], identity_embedding_rows=[np.eye(2, dtype=np.float32)],
        quality_logits_rows=[np.zeros(2, dtype=np.float32)], query_admission_policy="respect_no_object",
        min_query_score=0.0, mask_threshold=0.5, merge_iou_threshold=0.5,
        merge_overlap_threshold=0.5, max_instances=8, ownership_membership_threshold=0.5,
    )
    assert sorted(tuple(row["primitive_ids"]) for row in instances) == [(10,), (11,)]
    assert diagnostics["ownership_membership_filtered_assignments"] == 4
    assert diagnostics["null_owned_primitives"] == 1


def test_membership_gate_does_not_remove_null_ownership_competition() -> None:
    query_logits = np.full((1, 36), -5.0, dtype=np.float32)
    query_logits[0, 4] = 5.0
    query_logits[0, 35] = -3.0
    instances, diagnostics = instances_from_windows(
        [[10]], [query_logits], [np.array([[5.0]], dtype=np.float32)],
        ownership_logits_rows=[np.array([[1.0, 5.0]], dtype=np.float32)],
        identity_embedding_rows=[np.array([[1.0, 0.0]], dtype=np.float32)],
        quality_logits_rows=[np.zeros(1, dtype=np.float32)],
        query_admission_policy="respect_no_object", min_query_score=0.0, mask_threshold=0.5,
        merge_iou_threshold=0.5, merge_overlap_threshold=0.5, max_instances=8,
        ownership_membership_threshold=0.5,
    )
    assert instances == []
    assert diagnostics["ownership_membership_considered_assignments"] == 1
    assert diagnostics["ownership_membership_filtered_assignments"] == 0
    assert diagnostics["null_owned_primitives"] == 1


def test_overlapping_windows_assign_a_primitive_once_after_identity_tracking() -> None:
    query_logits = np.full((1, 36), -4.0, dtype=np.float32)
    query_logits[0, 6] = 4.0
    query_logits[0, 35] = -3.0
    instances, diagnostics = instances_from_windows(
        [[10, 11], [11, 12]], [query_logits, query_logits],
        [np.array([[5.0, 5.0]], dtype=np.float32), np.array([[5.0, 5.0]], dtype=np.float32)],
        ownership_logits_rows=[
            np.array([[4.0, -2.0], [3.0, -2.0]], dtype=np.float32),
            np.array([[6.0, -2.0], [4.0, -2.0]], dtype=np.float32),
        ],
        identity_embedding_rows=[np.array([[1.0, 0.0]], dtype=np.float32)] * 2,
        quality_logits_rows=[np.zeros(1, dtype=np.float32)] * 2,
        query_admission_policy="respect_no_object", min_query_score=0.0, mask_threshold=0.5,
        merge_iou_threshold=0.5, merge_overlap_threshold=0.5, max_instances=8,
    )
    assert diagnostics["tracks"] == 1
    assert len(instances) == 1 and instances[0]["primitive_ids"] == [10, 11, 12]
    assert primitive_reuse_diagnostics(instances)["reused_primitives"] == 0


def test_cross_window_ownership_is_invariant_to_per_window_logit_shifts() -> None:
    query_logits = np.full((1, 36), -4.0, dtype=np.float32)
    query_logits[0, 6] = 4.0
    query_logits[0, 35] = -3.0

    def decode(first_owner_logits):
        return instances_from_windows(
            [[10], [10]],
            [query_logits, query_logits],
            [np.array([[5.0]], dtype=np.float32)] * 2,
            ownership_logits_rows=[
                np.asarray(first_owner_logits, dtype=np.float32),
                np.array([[0.0, 2.0]], dtype=np.float32),
            ],
            identity_embedding_rows=[np.array([[1.0, 0.0]], dtype=np.float32)] * 2,
            quality_logits_rows=[np.zeros(1, dtype=np.float32)] * 2,
            query_admission_policy="respect_no_object",
            min_query_score=0.0,
            mask_threshold=0.5,
            merge_iou_threshold=0.5,
            merge_overlap_threshold=0.5,
            max_instances=8,
            ownership_membership_threshold=0.5,
        )[0]

    baseline = decode([[1.0, 0.0]])
    shifted = decode([[101.0, 100.0]])

    assert baseline == shifted


def test_window_without_membership_does_not_veto_strong_overlapping_owner() -> None:
    query_logits = np.full((1, 36), -4.0, dtype=np.float32)
    query_logits[0, 6] = 4.0
    query_logits[0, 35] = -3.0
    instances, diagnostics = instances_from_windows(
        [[10], [10]],
        [query_logits, query_logits],
        [np.array([[-5.0]], dtype=np.float32), np.array([[5.0]], dtype=np.float32)],
        ownership_logits_rows=[
            np.array([[0.0, 0.0]], dtype=np.float32),
            np.array([[10.0, 0.0]], dtype=np.float32),
        ],
        identity_embedding_rows=[np.array([[1.0, 0.0]], dtype=np.float32)] * 2,
        quality_logits_rows=[np.zeros(1, dtype=np.float32)] * 2,
        query_admission_policy="respect_no_object",
        min_query_score=0.0,
        mask_threshold=0.5,
        merge_iou_threshold=0.5,
        merge_overlap_threshold=0.5,
        max_instances=8,
        ownership_membership_threshold=0.5,
    )

    assert len(instances) == 1
    assert instances[0]["primitive_ids"] == [10]
    assert diagnostics["ownership_membership_filtered_assignments"] == 1


def test_owner_selection_is_gt_free_and_iou_before_after_is_computable() -> None:
    parameters = set(inspect.signature(select_global_owners).parameters)
    assert parameters == {"ownership_logits", "admitted_queries"}
    gt = {1, 2}
    before = {1, 2, 3}
    after = {1, 2}
    iou = lambda pred: len(pred & gt) / len(pred | gt)
    assert iou(after) >= iou(before)


def test_teacher_window_uses_explicit_noncontiguous_primitive_ids() -> None:
    record = {
        "record_id": "page::window0", "original_record_id": "page",
        "window_start": 100, "window_end": 103,
        "primitive_rows": [{"primitive_id": 900}, {"primitive_id": 42}, {"primitive_id": 701}],
    }
    teacher = {"page": [{"label": 4, "primitive_positions": [1, 999], "primitive_ids": [42, 701], "score": 1.0}]}
    labels, masks, count, _ = train.window_teacher_targets(torch, record, teacher, 3, 4, torch.device("cpu"))
    assert count == 1 and labels.tolist() == [4]
    assert masks[0].tolist() == [0.0, 1.0, 1.0]


def test_positive_only_filters_hard_negatives_and_unmatched_mode_keeps_them(tmp_path, monkeypatch) -> None:
    path = tmp_path / "teacher.jsonl"
    rows = [{
        "record_id": "p", "artifact_provenance": {"token": "ok"},
        "teacher_proposals": [
            {"label": 4, "primitive_positions": [0], "primitive_ids": [100], "gt_match": {"positive": True, "gt_iou": 0.9}},
            {"label": 7, "primitive_positions": [1], "primitive_ids": [200], "gt_match": {"positive": False}},
        ],
    }]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    monkeypatch.setattr(train, "validate_teacher_provenance", lambda _: {"token": "ok"})
    monkeypatch.setattr(train, "TEACHER_PROVENANCE_FIELDS", ("token",))
    positive, positive_report = train.load_teacher_proposals(path, positive_only=True, min_gt_iou=0.5)
    all_rows, all_report = train.load_teacher_proposals(path, positive_only=False, min_gt_iou=0.5)
    assert [row["label"] for row in positive["p"]] == [4]
    assert [row["label"] for row in all_rows["p"]] == [4, train.IGNORE_LABEL]
    assert positive_report["filtered_nonpositive"] == 1
    assert all_report["hard_negative_objectness_proposals"] == 1


def test_page_aware_batching_keeps_all_three_windows_until_boundary_and_flushes_epoch_tail() -> None:
    pages = ["a", "a", "a", "b", "c"]
    pending: list[str] = []
    pending_page = None
    batches: list[list[str]] = []
    for page in pages:
        if train.should_flush_page_aware_batch(len(pending), pending_page, page, 2):
            batches.append(pending)
            pending = []
        pending.append(page)
        pending_page = page
    if pending:
        batches.append(pending)
    assert batches == [["a", "a", "a"], ["b", "c"]]
    assert sum(1 for left, right in zip(batches[0], batches[0][1:]) if left == right) == 2
    assert all(left != right for left, right in zip(batches[1], batches[1][1:]))


def test_page_boundary_checkpoint_does_not_skip_current_untrained_record() -> None:
    assert train.page_aware_checkpoint_completed_records(4, current_record_pending=True) == 3
    assert train.page_aware_checkpoint_completed_records(4, current_record_pending=False) == 4
    with pytest.raises(ValueError, match="cannot be negative"):
        train.page_aware_checkpoint_completed_records(0, current_record_pending=True)


def test_v5_abi_binds_ownership_and_pre_v5_is_diagnostic_only() -> None:
    model = train.make_panoptic_model(
        torch.nn, torch, len(train.BASE_FEATURES), 16, 1, 4, 8, 36, 2, 0.0,
        geometry_decoder_mode="geometry_v2", num_stuff_queries=2,
        sq_rq_enabled=True, ownership_enabled=True,
    )
    state = model.state_dict()
    geometry = train.geometry_decoder_config(
        hidden_dim=16, heads=4, num_queries=8, decoder_layers=2,
        identity_dim=train.PANOPTIC_IDENTITY_DIM, num_stuff_queries=2,
    )
    sq_rq = train.sq_rq_config(enabled=True, hidden_dim=16, heads=4, num_labels=36, gradient_scale=0.0)
    deployment = train.sq_rq_deployment_config(
        enabled=True,
        query_confidence_threshold=sq_rq["query_confidence_threshold"],
        token_membership_threshold=sq_rq["token_membership_threshold"],
        auto_fused=False,
        auto_fuse_reason=None,
    )
    ownership = train.ownership_config(hidden_dim=16, num_queries=8)
    abi = train.checkpoint_abi_metadata(
        2048, geometry_config=geometry, geometry_state_schema_sha256=train.model_state_schema_sha256(state),
        sq_rq=sq_rq, sq_rq_deployment=deployment, ownership=ownership,
    )
    objective = {
        "geometry_decoder_mode": "geometry_v2",
        "sq_rq_enabled": True,
        "ownership_loss_weight": 1.0,
        "quality_objective_version": train.PANOPTIC_QUALITY_OBJECTIVE_VERSION,
        "quality_objective_config": train.quality_objective_contract(),
    }
    checkpoint = {
        "schema_version": "floorplancad_line_token_panoptic_moe_checkpoint_v5_geometry_v2_sq_rq_ownership",
        "state_dict": state, "feature_names": train.BASE_FEATURES,
        "position_encoding_version": train.POSITION_ENCODING_VERSION, "quality_head": train.PANOPTIC_QUALITY_HEAD_VERSION,
        "identity_head_version": train.PANOPTIC_IDENTITY_HEAD_VERSION, "identity_dim": train.PANOPTIC_IDENTITY_DIM,
        "geometry_decoder_mode": "geometry_v2", "geometry_config": geometry, "sq_rq_config": sq_rq,
        "sq_rq_deployment": deployment,
        "sq_rq_auto_fused": False,
        "sq_rq_auto_fuse_reason": None,
        "ownership_config": ownership, "checkpoint_abi": abi,
        "feature_schema_sha256": train.feature_schema_sha256(), "ontology_sha256": train.ontology_sha256(),
        "window_contract_sha256": train.canonical_json_sha256(train.window_contract(2048)),
        "objective_config": objective, "objective_config_hash": train.objective_config_hash(objective),
    }
    validated = train.validate_checkpoint_abi(checkpoint)
    assert validated["production_compatible"] and validated["ownership_enabled"]
    tampered = dict(checkpoint)
    tampered["checkpoint_abi"] = dict(abi)
    tampered["checkpoint_abi"]["ownership_config"] = {**ownership, "decode_order": "mask_then_owner"}
    with pytest.raises(ValueError, match="ownership config hash"):
        train.validate_checkpoint_abi(tampered)
    old = dict(checkpoint)
    old["checkpoint_abi"] = {key: value for key, value in abi.items() if not key.startswith("ownership")}
    old["checkpoint_abi"]["abi_version"] = train.PANOPTIC_SQ_RQ_CHECKPOINT_ABI_VERSION
    old["schema_version"] = "floorplancad_line_token_panoptic_moe_checkpoint_v4_geometry_v2_sq_rq"
    old["ownership_config"] = None
    old["state_dict"] = {
        key: value for key, value in state.items()
        if "ownership_head" not in key and key != "ownership_residual_gate"
    }
    old["checkpoint_abi"]["geometry_state_schema_sha256"] = train.model_state_schema_sha256(old["state_dict"])
    with pytest.raises(ValueError, match="pre-v5"):
        train.validate_checkpoint_abi(old)
    assert not train.validate_checkpoint_abi(old, legacy_position_compat=True)["production_compatible"]
