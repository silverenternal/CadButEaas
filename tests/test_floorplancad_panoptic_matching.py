import pytest

from experiments.floorplancad_train_line_token_panoptic_moe import (
    IGNORE_LABEL,
    align_teacher_to_gt_queries,
    assignment_cost_diagnostics,
    component_assignment_cost,
    greedy_assignment_gpu,
    linear_sum_assignment_fallback,
    match_component_queries,
    teacher_hard_negative_objectness_loss,
)


torch = pytest.importorskip("torch")
pytest.importorskip("scipy")


def test_production_cost_combines_class_focal_mask_and_dice():
    query_logits = torch.tensor([[0.0, 5.0, 0.0], [0.0, 0.0, 5.0]])
    mask_logits = torch.tensor([[5.0, 5.0, -5.0, -5.0], [-5.0, -5.0, 5.0, 5.0]])
    target_labels = torch.tensor([1, 2])
    target_masks = torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])

    cost = component_assignment_cost(torch, query_logits, mask_logits, target_labels, target_masks)

    assert cost.shape == (2, 2)
    assert cost[0, 0] < cost[0, 1]
    assert cost[1, 1] < cost[1, 0]


def test_assignment_cost_is_positive_and_scale_normalized_like_vecformer():
    query_logits = torch.tensor([[0.0, 3.0, 0.0], [0.0, 0.0, 3.0]])
    mask_logits = torch.tensor([[4.0, 4.0, -4.0, -4.0], [-4.0, -4.0, 4.0, 4.0]])
    target_labels = torch.tensor([1, 2])
    target_masks = torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])

    cost = component_assignment_cost(torch, query_logits, mask_logits, target_labels, target_masks)

    assert torch.isfinite(cost).all()
    assert float(cost.min()) >= 0.0
    assert cost[0, 0] < cost[0, 1]
    assert cost[1, 1] < cost[1, 0]


def test_greedy_diagnostics_report_cost_gap_and_assignment_churn():
    cost = torch.tensor([[1.0, 2.0], [2.0, 100.0]])
    greedy_rows, greedy_cols = greedy_assignment_gpu(torch, cost)
    exact_rows, exact_cols = linear_sum_assignment_fallback(cost)

    diagnostics = assignment_cost_diagnostics(cost, greedy_rows, greedy_cols, exact_rows, exact_cols)

    assert diagnostics["candidate_cost"] == 101.0
    assert diagnostics["reference_cost"] == 4.0
    assert diagnostics["relative_gap"] > 0.01
    assert diagnostics["assignment_churn"] == 1.0


def test_greedy_matching_rejects_gap_at_or_above_one_percent(monkeypatch):
    monkeypatch.setattr(
        "experiments.floorplancad_train_line_token_panoptic_moe.component_assignment_cost",
        lambda *_args, **_kwargs: torch.tensor([[1.0, 2.0], [2.0, 100.0]]),
    )
    with pytest.raises(RuntimeError, match="not below the required 1%"):
        match_component_queries(
            torch,
            torch.zeros((2, 3)),
            torch.zeros((2, 2)),
            torch.tensor([1, 2]),
            torch.eye(2),
            2,
            matching="greedy_gpu",
        )


def test_greedy_matching_allows_audited_sub_one_percent_gap(monkeypatch):
    monkeypatch.setattr(
        "experiments.floorplancad_train_line_token_panoptic_moe.component_assignment_cost",
        lambda *_args, **_kwargs: torch.tensor([[1.0, 4.0], [4.0, 1.0]]),
    )
    labels, masks, positives, matched = match_component_queries(
        torch,
        torch.zeros((2, 3)),
        torch.zeros((2, 2)),
        torch.tensor([1, 2]),
        torch.eye(2),
        2,
        matching="greedy_gpu",
    )
    assert labels.tolist() == [1, 2]
    assert masks.tolist() == torch.eye(2).tolist()
    assert (positives, matched) == (2, 2)


def test_training_greedy_matching_stays_on_tensor_assignment_path(monkeypatch):
    monkeypatch.setattr(
        "experiments.floorplancad_train_line_token_panoptic_moe.component_assignment_cost",
        lambda *_args, **_kwargs: torch.tensor([[1.0, 4.0], [4.0, 1.0]]),
    )
    labels, masks, positives, matched = match_component_queries(
        torch,
        torch.zeros((2, 3)),
        torch.zeros((2, 2)),
        torch.tensor([1, 2]),
        torch.eye(2),
        2,
        matching="greedy_gpu_train",
    )
    assert labels.tolist() == [1, 2]
    assert masks.tolist() == torch.eye(2).tolist()
    assert (positives, matched) == (2, 2)


def test_teacher_distillation_reuses_gt_identity_query_without_conflict():
    gt_query_labels = torch.tensor([2, 3, IGNORE_LABEL])
    gt_query_masks = torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    teacher_labels = torch.tensor([3, 2, 4])
    teacher_masks = torch.tensor([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0]])

    labels, masks, diagnostics = align_teacher_to_gt_queries(
        torch, gt_query_labels, gt_query_masks, teacher_labels, teacher_masks
    )

    assert labels.tolist() == [2, 3, IGNORE_LABEL]
    assert masks.tolist()[:2] == gt_query_masks.tolist()[:2]
    assert diagnostics["teacher_identity_aligned"] == 2
    assert diagnostics["teacher_identity_unaligned"] == 1
    assert diagnostics["gt_positive_teacher_negative_conflicts"] == 0


def test_teacher_hard_negative_only_supervises_gt_unmatched_query_objectness():
    query_logits = torch.zeros((3, 36), requires_grad=True)
    mask_logits = torch.tensor([[5.0, 5.0], [-5.0, -5.0], [4.0, 4.0]])
    gt_query_labels = torch.tensor([2, IGNORE_LABEL, IGNORE_LABEL])
    teacher_labels = torch.tensor([IGNORE_LABEL])
    teacher_masks = torch.tensor([[1.0, 1.0]])

    loss, selected = teacher_hard_negative_objectness_loss(
        torch,
        torch.nn.CrossEntropyLoss(),
        query_logits,
        mask_logits,
        gt_query_labels,
        teacher_labels,
        teacher_masks,
    )
    loss.backward()

    assert selected == 1
    assert torch.count_nonzero(query_logits.grad[0]).item() == 0
    assert torch.count_nonzero(query_logits.grad[1]).item() == 0
    assert torch.count_nonzero(query_logits.grad[2]).item() > 0
