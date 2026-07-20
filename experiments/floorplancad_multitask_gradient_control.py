#!/usr/bin/env python3
"""Deterministic PCGrad control for FloorPlanCAD multi-task training.

The controller operates on already-computed task losses. Shared parameters receive
the mean of conflict-projected task gradients, while each task-specific parameter
receives only its own task gradient. When AMP-scaled losses are supplied,
``amp_scale`` must be the scale used for every loss; gradients are unscaled before
auditing, projection, finite checks, or assignment.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TASK_ORDER = (
    "semantic",
    "query_mask_quality",
    "teacher",
    "identity",
    "ownership",
    "router",
)
OBJECTIVE_ABI_FIELDS = {
    "multitask_gradient_control": "deterministic_pcgrad_v2_production_task_domains",
    "multitask_gradient_tasks": list(TASK_ORDER),
    "multitask_shared_aggregation": "mean_active_tasks",
    "multitask_projection_order": "canonical_then_lexical",
    "multitask_amp_contract": "unscale_before_audit_projection_assignment",
    "multitask_task_specific_contract": "own_task_gradient_only",
}

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/floorplancad_multitask_gradient_control_smoke.json"


def _ordered_tasks(tasks: Sequence[str]) -> list[str]:
    canonical = [task for task in TASK_ORDER if task in tasks]
    return canonical + sorted(set(tasks).difference(TASK_ORDER))


def _validate_parameters(parameters: Sequence[Any], label: str) -> list[Any]:
    values = list(parameters)
    identities = [id(parameter) for parameter in values]
    if len(identities) != len(set(identities)):
        raise ValueError(f"duplicate parameter in {label}")
    return values


def _flatten_gradients(torch: Any, gradients: Sequence[Any], parameters: Sequence[Any]) -> Any:
    chunks = []
    for gradient, parameter in zip(gradients, parameters, strict=True):
        if gradient is None:
            chunks.append(torch.zeros_like(parameter, memory_format=torch.preserve_format).reshape(-1))
        else:
            chunks.append(gradient.detach().reshape(-1))
    if not chunks:
        return torch.empty(0)
    return torch.cat(chunks)


def _unflatten_gradient(flat_gradient: Any, parameters: Sequence[Any]) -> list[Any]:
    gradients = []
    offset = 0
    for parameter in parameters:
        size = parameter.numel()
        gradients.append(flat_gradient[offset : offset + size].reshape_as(parameter).to(parameter.dtype))
        offset += size
    if offset != flat_gradient.numel():
        raise ValueError("flat gradient size does not match parameters")
    return gradients


def gradient_pairwise_audit(torch: Any, task_gradients: Mapping[str, Any]) -> dict[str, Any]:
    """Return norms, pairwise dot products, and cosine similarities."""

    tasks = _ordered_tasks(list(task_gradients))
    norms = {task: float(torch.linalg.vector_norm(task_gradients[task]).item()) for task in tasks}
    pairs: dict[str, dict[str, float]] = {}
    for left_index, left in enumerate(tasks):
        for right in tasks[left_index + 1 :]:
            dot = float(torch.dot(task_gradients[left], task_gradients[right]).item())
            denominator = norms[left] * norms[right]
            pairs[f"{left}__{right}"] = {
                "dot": dot,
                "cosine": dot / denominator if denominator > 0.0 else 0.0,
            }
    return {"task_order": tasks, "norms": norms, "pairs": pairs}


def deterministic_pcgrad(
    torch: Any,
    task_gradients: Mapping[str, Any],
    *,
    tolerance: float = 1e-10,
    max_passes: int = 256,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Project conflicting shared gradients using a deterministic PCGrad order.

    Pair sweeps are repeated because resolving one conflict can reintroduce another.
    The returned task gradients satisfy the pairwise non-negative-dot contract or
    the function fails closed instead of silently using conflicting gradients.
    """

    if tolerance < 0.0 or max_passes < 1:
        raise ValueError("tolerance must be non-negative and max_passes positive")
    tasks = _ordered_tasks(list(task_gradients))
    projected = {task: task_gradients[task].detach().clone() for task in tasks}
    passes = 0
    projection_count = 0
    for pass_index in range(max_passes):
        passes = pass_index + 1
        changed = False
        for left_index, left in enumerate(tasks):
            for right in tasks[left_index + 1 :]:
                left_gradient = projected[left]
                right_gradient = projected[right]
                dot = torch.dot(left_gradient, right_gradient)
                if float(dot.item()) >= -tolerance:
                    continue
                left_norm_sq = torch.dot(left_gradient, left_gradient)
                right_norm_sq = torch.dot(right_gradient, right_gradient)
                if float(left_norm_sq.item()) == 0.0 or float(right_norm_sq.item()) == 0.0:
                    continue
                old_left = left_gradient
                old_right = right_gradient
                projected[left] = old_left - dot / right_norm_sq * old_right
                projected[right] = old_right - dot / left_norm_sq * old_left
                projection_count += 1
                changed = True
        if not changed:
            break
    audit = gradient_pairwise_audit(torch, projected)
    minimum_dot = min((pair["dot"] for pair in audit["pairs"].values()), default=0.0)
    if minimum_dot < -tolerance:
        raise RuntimeError(
            f"PCGrad did not converge: minimum pairwise dot {minimum_dot:.6g} "
            f"after {passes} passes"
        )
    if not all(bool(torch.isfinite(gradient).all().item()) for gradient in projected.values()):
        raise FloatingPointError("non-finite projected gradient")
    return projected, {
        "passes": passes,
        "projection_count": projection_count,
        "minimum_pairwise_dot": minimum_dot,
        "pairwise": audit,
    }


def assign_multitask_gradients(
    torch: Any,
    losses: Mapping[str, Any | None],
    shared_parameters: Sequence[Any],
    task_specific_parameters: Mapping[str, Sequence[Any]],
    *,
    task_specific_losses: Mapping[str, Any | None] | None = None,
    amp_scale: float | None = None,
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Compute, project, and assign gradients without calling ``backward``.

    Integration sequence: zero optimizer gradients, create one scalar loss per
    active task, optionally apply one explicit/static loss scale and pass it as
    ``amp_scale``, call this function, then perform clipping/``optimizer.step``.
    Assigned gradients are already unscaled, so ``GradScaler.unscale_`` or
    ``GradScaler.step`` would incorrectly unscale them a second time. Autocast with
    unscaled losses (``amp_scale=None``) is the preferred mixed-precision path.
    """

    shared = _validate_parameters(shared_parameters, "shared_parameters")
    task_parameters = {
        task: _validate_parameters(parameters, f"task_specific_parameters[{task!r}]")
        for task, parameters in task_specific_parameters.items()
    }
    owner_by_parameter: dict[int, str] = {id(parameter): "shared" for parameter in shared}
    for task, parameters in task_parameters.items():
        for parameter in parameters:
            owner = owner_by_parameter.setdefault(id(parameter), task)
            if owner != task:
                raise ValueError(f"parameter ownership overlaps between {owner!r} and {task!r}")

    active_losses = {
        task: loss
        for task, loss in losses.items()
        if loss is not None and bool(getattr(loss, "requires_grad", False))
    }
    tasks = _ordered_tasks(list(active_losses))
    if not tasks:
        raise ValueError("at least one differentiable task loss is required")
    specific_loss_map = losses if task_specific_losses is None else task_specific_losses
    unknown_adapters = set(task_parameters).difference(specific_loss_map)
    if unknown_adapters:
        raise ValueError(f"task-specific parameters have no declared loss: {sorted(unknown_adapters)}")
    scale = 1.0 if amp_scale is None else float(amp_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("amp_scale must be finite and positive")

    per_task_shared: dict[str, Any] = {}
    per_task_specific: dict[str, list[Any]] = {}
    for task in tasks:
        differentiated = shared
        gradients = torch.autograd.grad(
            active_losses[task],
            differentiated,
            retain_graph=True,
            allow_unused=True,
        )
        unscaled = [None if gradient is None else gradient.detach() / scale for gradient in gradients]
        flat_shared = _flatten_gradients(torch, unscaled, shared)
        if not bool(torch.isfinite(flat_shared).all().item()):
            raise FloatingPointError(f"non-finite shared gradient for task {task!r}")
        per_task_shared[task] = flat_shared

    for task, parameters in task_parameters.items():
        if not parameters:
            per_task_specific[task] = []
            continue
        specific_loss = specific_loss_map.get(task)
        if specific_loss is None or not bool(getattr(specific_loss, "requires_grad", False)):
            per_task_specific[task] = [None] * len(parameters)
            continue
        gradients = torch.autograd.grad(specific_loss, parameters, retain_graph=True, allow_unused=True)
        own_gradients = [None if gradient is None else gradient.detach() / scale for gradient in gradients]
        if any(gradient is not None and not bool(torch.isfinite(gradient).all().item()) for gradient in own_gradients):
            raise FloatingPointError(f"non-finite task-specific gradient for task {task!r}")
        per_task_specific[task] = own_gradients

    raw_audit = gradient_pairwise_audit(torch, per_task_shared)
    projected, projection = deterministic_pcgrad(
        torch,
        per_task_shared,
        tolerance=tolerance,
    )
    aggregate = torch.stack([projected[task] for task in tasks]).mean(dim=0)
    for parameter, gradient in zip(shared, _unflatten_gradient(aggregate, shared), strict=True):
        parameter.grad = gradient.clone()

    for task, parameters in task_parameters.items():
        gradients = per_task_specific.get(task, [None] * len(parameters))
        for parameter, gradient in zip(parameters, gradients, strict=True):
            parameter.grad = None if gradient is None else gradient.to(parameter.dtype).clone()

    return {
        "schema_version": "floorplancad_multitask_gradient_control_report_v1",
        "active_tasks": tasks,
        "missing_or_inactive_tasks": _ordered_tasks(
            [task for task, loss in losses.items() if loss is None or task not in active_losses]
        ),
        "amp_scale": scale,
        "gradients_are_unscaled": True,
        "raw_pairwise": raw_audit,
        "projected_pairwise": projection["pairwise"],
        "projection_passes": projection["passes"],
        "projection_count": projection["projection_count"],
        "minimum_projected_pairwise_dot": projection["minimum_pairwise_dot"],
        "shared_aggregation": "mean_active_tasks",
        "task_specific_contract": "explicit_owner_loss_gradient_only",
        "objective_abi_fields": OBJECTIVE_ABI_FIELDS,
    }


def _smoke_payload(torch: Any) -> dict[str, Any]:
    torch.manual_seed(20260711)
    shared = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    semantic_adapter = torch.nn.Parameter(torch.tensor([0.5]))
    mask_adapter = torch.nn.Parameter(torch.tensor([-0.25]))
    losses = {
        "semantic": shared[0] - shared[1] + semantic_adapter.square().sum(),
        "mask": -shared[0] + 0.25 * shared[1] + mask_adapter.square().sum(),
        "objectness": None,
    }
    report = assign_multitask_gradients(
        torch,
        losses,
        [shared],
        {"semantic": [semantic_adapter], "mask": [mask_adapter]},
    )
    checks = {
        "conflict_detected": any(pair["dot"] < 0.0 for pair in report["raw_pairwise"]["pairs"].values()),
        "projected_pairwise_nonnegative": report["minimum_projected_pairwise_dot"] >= -1e-10,
        "shared_gradient_finite": bool(torch.isfinite(shared.grad).all().item()),
        "semantic_adapter_own_gradient": bool(torch.allclose(semantic_adapter.grad, torch.tensor([1.0]))),
        "mask_adapter_own_gradient": bool(torch.allclose(mask_adapter.grad, torch.tensor([-0.5]))),
    }
    return {
        "schema_version": "floorplancad_multitask_gradient_control_smoke_v1",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": "passed" if all(checks.values()) else "blocked",
        "checks": checks,
        "report": report,
        "train_integration_interface": {
            "call": "assign_multitask_gradients(torch, losses, shared_parameters, task_specific_parameters, amp_scale=explicit_static_scale_or_none)",
            "loss_keys": list(TASK_ORDER),
            "optimizer_sequence": [
                "optimizer.zero_grad(set_to_none=True)",
                "construct scalar per-task losses without fixed-weight summation",
                "optionally scale every loss using one explicit/static AMP scale",
                "assign_multitask_gradients",
                "clip already-unscaled gradients if configured",
                "optimizer.step; never call GradScaler.unscale_ or GradScaler.step on assigned gradients",
            ],
        },
        "objective_abi_fields": OBJECTIVE_ABI_FIELDS,
        "claim_boundary": "Synthetic gradient-control contract only; no training or model-quality claim.",
    }


def main() -> int:
    import torch

    payload = _smoke_payload(torch)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
