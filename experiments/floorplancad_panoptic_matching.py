"""Shared component matching costs for FloorPlanCAD panoptic training."""

from __future__ import annotations

import math
from typing import Any

from experiments.floorplancad_panoptic_protocol import DEFAULT_NO_OBJECT_LABEL, STUFF_LABELS


def linear_sum_assignment_fallback(cost: Any) -> tuple[list[int], list[int]]:
    """Run exact Hungarian assignment and fail closed when SciPy is unavailable."""
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise RuntimeError("exact Hungarian component matching requires scipy") from exc
    rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
    return rows.tolist(), cols.tolist()


def greedy_assignment_gpu(torch: Any, cost: Any) -> tuple[list[int], list[int]]:
    rows: list[int] = []
    cols: list[int] = []
    if int(cost.numel()) <= 0:
        return rows, cols
    work = cost.detach().clone()
    q_count, t_count = int(work.shape[0]), int(work.shape[1])
    inf = torch.tensor(float("inf"), dtype=work.dtype, device=work.device)
    for _ in range(min(q_count, t_count)):
        flat_idx = int(torch.argmin(work).item())
        row = flat_idx // t_count
        col = flat_idx % t_count
        value = float(work[row, col].detach().item())
        if not math.isfinite(value):
            break
        rows.append(row)
        cols.append(col)
        work[row, :] = inf
        work[:, col] = inf
    return rows, cols


def greedy_assignment_gpu_tensor(torch: Any, cost: Any) -> tuple[Any, Any]:
    """GPU-only greedy assignment for the training hot path."""
    count = min(int(cost.shape[0]), int(cost.shape[1]))
    if count == 0:
        empty = torch.empty((0,), dtype=torch.long, device=cost.device)
        return empty, empty
    work = cost.detach().clone()
    rows: list[Any] = []
    cols: list[Any] = []
    infinity = torch.tensor(float("inf"), dtype=work.dtype, device=work.device)
    width = int(work.shape[1])
    for _ in range(count):
        flat_index = torch.argmin(work)
        row = torch.div(flat_index, width, rounding_mode="floor")
        col = torch.remainder(flat_index, width)
        rows.append(row)
        cols.append(col)
        work[row, :] = infinity
        work[:, col] = infinity
    return torch.stack(rows).long(), torch.stack(cols).long()


def component_assignment_cost(
    torch: Any,
    query_logits: Any,
    mask_logits: Any,
    target_labels: Any,
    target_masks: Any,
    *,
    class_weight: float = 0.5,
    focal_mask_weight: float = 1.0,
    dice_weight: float = 1.0,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    primitive_weights: Any | None = None,
) -> Any:
    """Build VecFormer-aligned class + sigmoid-focal-mask + Dice assignment cost."""
    query_logits = query_logits.float()
    mask_logits = mask_logits.float()
    target_masks = target_masks.float()
    class_prob = torch.softmax(query_logits, dim=-1)
    class_cost = 1.0 - class_prob[:, target_labels]

    mask_prob = torch.sigmoid(mask_logits)
    token_weights = torch.ones_like(mask_prob[0]) if primitive_weights is None else primitive_weights.float().to(mask_prob.device)
    token_weights = token_weights.clamp_min(0.0)
    eps = 1e-6
    positive_focal = -float(focal_alpha) * torch.pow(1.0 - mask_prob, float(focal_gamma)) * torch.log(mask_prob.clamp_min(eps))
    negative_focal = -(1.0 - float(focal_alpha)) * torch.pow(mask_prob, float(focal_gamma)) * torch.log((1.0 - mask_prob).clamp_min(eps))
    focal_cost = (
        torch.einsum("qn,tn,n->qt", positive_focal, target_masks, token_weights)
        + torch.einsum("qn,tn,n->qt", negative_focal, 1.0 - target_masks, token_weights)
    ) / token_weights.sum().clamp_min(1e-6)

    intersection = torch.einsum("qn,tn,n->qt", mask_prob, target_masks, token_weights)
    denominator = torch.einsum("qn,n->q", mask_prob, token_weights).unsqueeze(1) + torch.einsum("tn,n->t", target_masks, token_weights).unsqueeze(0)
    dice_cost = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    eps = torch.finfo(torch.float32).eps
    class_cost = class_cost / class_cost.detach().mean().clamp_min(eps)
    focal_cost = focal_cost / focal_cost.detach().mean().clamp_min(eps)
    dice_cost = dice_cost / dice_cost.detach().mean().clamp_min(eps)
    return float(class_weight) * class_cost + float(focal_mask_weight) * focal_cost + float(dice_weight) * dice_cost


def assignment_cost_diagnostics(cost: Any, rows: list[int], cols: list[int], reference_rows: list[int], reference_cols: list[int]) -> dict[str, float]:
    candidate = sum(float(cost[row, col].detach().item()) for row, col in zip(rows, cols, strict=True))
    reference = sum(float(cost[row, col].detach().item()) for row, col in zip(reference_rows, reference_cols, strict=True))
    relative_gap = max(candidate - reference, 0.0) / max(abs(reference), 1e-8)
    candidate_pairs = set(zip(rows, cols, strict=True))
    reference_pairs = set(zip(reference_rows, reference_cols, strict=True))
    pair_count = max(len(reference_pairs), 1)
    churn = 1.0 - len(candidate_pairs & reference_pairs) / float(pair_count)
    return {"candidate_cost": candidate, "reference_cost": reference, "relative_gap": relative_gap, "assignment_churn": churn}


def match_component_queries(
    torch: Any,
    query_logits: Any,
    mask_logits: Any,
    target_labels: Any,
    target_masks: Any,
    num_queries: int,
    matching: str = "hungarian_cpu",
    primitive_weights: Any | None = None,
    thing_query_count: int | None = None,
    typed_stuff_slots: bool = False,
    *,
    no_object_label: int = DEFAULT_NO_OBJECT_LABEL,
    stuff_labels: tuple[int, ...] = STUFF_LABELS,
) -> tuple[Any, Any, int, int]:
    device = query_logits.device
    query_labels = torch.full((num_queries,), int(no_object_label), dtype=torch.long, device=device)
    query_masks = torch.zeros((num_queries, mask_logits.shape[-1]), dtype=torch.float32, device=device)
    positives = int(target_labels.numel())
    if positives == 0:
        return query_labels, query_masks, 0, 0

    with torch.no_grad():
        cost = component_assignment_cost(torch, query_logits, mask_logits, target_labels, target_masks, primitive_weights=primitive_weights)
        assignments: list[tuple[int, int]] = []
        tensor_assignments: list[tuple[Any, Any]] = []
        query_ranges = [(0, num_queries)]
        target_groups = [torch.arange(positives, device=device)]
        if thing_query_count is not None:
            thing_query_count = max(0, min(int(thing_query_count), num_queries))
            stuff_targets = (target_labels >= min(stuff_labels)) & (target_labels <= max(stuff_labels))
            target_groups = [torch.nonzero(~stuff_targets, as_tuple=False).flatten(), torch.nonzero(stuff_targets, as_tuple=False).flatten()]
            query_ranges = [(0, thing_query_count), (thing_query_count, num_queries)]
        if typed_stuff_slots:
            if thing_query_count is None or num_queries - thing_query_count != len(stuff_labels):
                raise ValueError("typed stuff slots require five dedicated stuff query positions")
            thing_targets = torch.nonzero((target_labels < min(stuff_labels)) | (target_labels > max(stuff_labels)), as_tuple=False).flatten()
            if thing_targets.numel():
                local_cost = cost[:thing_query_count][:, thing_targets]
                if matching == "greedy_gpu_train":
                    local_rows, local_cols = greedy_assignment_gpu_tensor(torch, local_cost)
                    tensor_assignments.append((local_rows, thing_targets.index_select(0, local_cols)))
                else:
                    local_rows, local_cols = (greedy_assignment_gpu(torch, local_cost) if matching == "greedy_gpu" else linear_sum_assignment_fallback(local_cost))
                    assignments.extend((int(row), int(thing_targets[int(col)])) for row, col in zip(local_rows, local_cols, strict=True))
            for label in stuff_labels:
                targets = torch.nonzero(target_labels == label, as_tuple=False).flatten()
                if targets.numel() > 1:
                    raise ValueError(f"stuff label {label} must be a page/window union target, not multiple instances")
                if targets.numel() == 1:
                    assignments.append((thing_query_count + label - min(stuff_labels), int(targets.item())))
            if matching == "greedy_gpu_train":
                rows_tensor = torch.cat([rows for rows, _cols in tensor_assignments]) if tensor_assignments else torch.empty((0,), dtype=torch.long, device=device)
                cols_tensor = torch.cat([cols for _rows, cols in tensor_assignments]) if tensor_assignments else torch.empty((0,), dtype=torch.long, device=device)
                for label in stuff_labels:
                    targets = torch.nonzero(target_labels == label, as_tuple=False).flatten()
                    if targets.numel() == 1:
                        rows_tensor = torch.cat([rows_tensor, torch.tensor([thing_query_count + label - min(stuff_labels)], device=device)])
                        cols_tensor = torch.cat([cols_tensor, targets])
                query_labels[rows_tensor] = target_labels[cols_tensor]
                query_masks[rows_tensor] = target_masks[cols_tensor]
                return query_labels, query_masks, positives, int(cols_tensor.numel())
            rows = [row for row, _ in assignments]
            cols = [col for _, col in assignments]
            if rows:
                query_rows = torch.as_tensor(rows, dtype=torch.long, device=device)
                target_cols = torch.as_tensor(cols, dtype=torch.long, device=device)
                query_labels[query_rows] = target_labels[target_cols]
                query_masks[query_rows] = target_masks[target_cols]
            return query_labels, query_masks, positives, len(cols)
        for (query_start, query_end), target_indices in zip(query_ranges, target_groups, strict=True):
            if target_indices.numel() == 0 or query_end <= query_start:
                continue
            local_cost = cost[query_start:query_end][:, target_indices]
            if matching == "greedy_gpu_train":
                local_rows, local_cols = greedy_assignment_gpu_tensor(torch, local_cost)
                tensor_assignments.append((local_rows + query_start, target_indices.index_select(0, local_cols)))
            elif matching == "greedy_gpu":
                local_rows, local_cols = greedy_assignment_gpu(torch, local_cost)
            else:
                local_rows, local_cols = linear_sum_assignment_fallback(local_cost)
            if matching != "greedy_gpu_train":
                assignments.extend((int(row) + query_start, int(target_indices[int(col)])) for row, col in zip(local_rows, local_cols, strict=True))
        if matching == "greedy_gpu_train":
            rows_tensor = torch.cat([rows for rows, _cols in tensor_assignments]) if tensor_assignments else torch.empty((0,), dtype=torch.long, device=device)
            cols_tensor = torch.cat([cols for _rows, cols in tensor_assignments]) if tensor_assignments else torch.empty((0,), dtype=torch.long, device=device)
            query_labels[rows_tensor] = target_labels[cols_tensor]
            query_masks[rows_tensor] = target_masks[cols_tensor]
            return query_labels, query_masks, positives, int(cols_tensor.numel())
        rows = [row for row, _ in assignments]
        cols = [col for _, col in assignments]
        if matching == "greedy_gpu":
            reference_rows, reference_cols = linear_sum_assignment_fallback(cost)
            diagnostics = assignment_cost_diagnostics(cost, rows, cols, reference_rows, reference_cols)
            if diagnostics["relative_gap"] >= 0.01:
                raise RuntimeError(
                    "greedy component assignment rejected: "
                    f"cost gap {diagnostics['relative_gap']:.6f} is not below the required 1%"
                )
        elif thing_query_count is None:
            rows, cols = linear_sum_assignment_fallback(cost)
    for qidx, tidx in zip(rows, cols, strict=True):
        query_labels[int(qidx)] = target_labels[int(tidx)]
        query_masks[int(qidx)] = target_masks[int(tidx)]
    return query_labels, query_masks, positives, len(cols)
