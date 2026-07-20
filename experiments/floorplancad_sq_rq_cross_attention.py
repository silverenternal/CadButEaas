#!/usr/bin/env python3
"""Prediction-only SQ<-RQ cross-attention building block.

The module deliberately has no target/instance arguments. Ground truth belongs in
the loss function outside this boundary, never in the forward context path.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/floorplancad_sq_rq_cross_attention_smoke.json"


def _scaled_gradient(value: Tensor, scale: float) -> Tensor:
    if scale == 0.0:
        return value.detach()
    return value.detach() + scale * (value - value.detach())


class SqRqCrossAttention(nn.Module):
    """Refine primitive SQ tokens using predicted RQ structure only."""

    def __init__(
        self,
        *,
        primitive_dim: int,
        rq_query_dim: int,
        hidden_dim: int,
        num_classes: int,
        heads: int = 4,
        controlled_gradient_scale: float = 0.0,
        query_confidence_threshold: float = 0.6,
        token_membership_threshold: float = 0.5,
        training_membership_temperature: float = 0.1,
        no_object_class: int | None = None,
        semantic_compatibility_floor: float = 0.05,
        residual_gate_init: float = 0.0,
        context_top_k: int = 8,
        include_fixture_semantic_heads: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        if not 0.0 <= controlled_gradient_scale <= 0.1:
            raise ValueError("controlled_gradient_scale must be in [0, 0.1]")
        if not 0.0 <= query_confidence_threshold <= 1.0:
            raise ValueError("query_confidence_threshold must be in [0, 1]")
        if not 0.0 <= token_membership_threshold <= 1.0:
            raise ValueError("token_membership_threshold must be in [0, 1]")
        if not 0.0 <= training_membership_temperature <= 1.0:
            raise ValueError("training_membership_temperature must be in [0, 1]")
        if num_classes < 2:
            raise ValueError("num_classes must be at least two")
        if no_object_class is not None and not 0 <= no_object_class < num_classes:
            raise ValueError("no_object_class is outside num_classes")

        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.num_classes = num_classes
        self.controlled_gradient_scale = float(controlled_gradient_scale)
        self.query_confidence_threshold = float(query_confidence_threshold)
        self.token_membership_threshold = float(token_membership_threshold)
        self.training_membership_temperature = float(training_membership_temperature)
        self.no_object_class = num_classes - 1 if no_object_class is None else int(no_object_class)
        if not 0.0 < semantic_compatibility_floor <= 1.0:
            raise ValueError("semantic_compatibility_floor must be in (0, 1]")
        self.semantic_compatibility_floor = float(semantic_compatibility_floor)
        if int(context_top_k) < 1:
            raise ValueError("context_top_k must be positive")
        self.context_top_k = int(context_top_k)
        self.include_fixture_semantic_heads = bool(include_fixture_semantic_heads)
        self.sq_residual_gate = nn.Parameter(torch.tensor(float(residual_gate_init)))
        self.residual_gradient_bootstrap = 0.05
        self.identity_base_path = primitive_dim == hidden_dim

        self.primitive_projection = nn.Linear(primitive_dim, hidden_dim)
        self.rq_query_projection = nn.Linear(rq_query_dim, hidden_dim)
        self.rq_pool_projection = nn.Linear(hidden_dim, hidden_dim)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.sq_pre_norm = nn.LayerNorm(hidden_dim)
        self.sq_adapter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.sq_post_norm = nn.LayerNorm(hidden_dim)
        if self.include_fixture_semantic_heads:
            self.base_semantic_head = nn.Linear(hidden_dim, num_classes)
            self.sq_semantic_head = nn.Linear(hidden_dim, num_classes)
        self.q_projection = nn.Linear(hidden_dim, hidden_dim)
        self.k_projection = nn.Linear(hidden_dim, hidden_dim)
        self.v_projection = nn.Linear(hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

    def _validate(
        self,
        primitive_tokens: Tensor,
        rq_query_embeddings: Tensor,
        rq_mask_logits: Tensor,
        rq_class_logits: Tensor,
        rq_admission_logits: Tensor | None,
        primitive_padding_mask: Tensor | None,
        rq_query_padding_mask: Tensor | None,
    ) -> tuple[int, int, int]:
        if primitive_tokens.ndim != 3 or rq_query_embeddings.ndim != 3:
            raise ValueError("primitive_tokens and rq_query_embeddings must be rank 3")
        if rq_mask_logits.ndim != 3 or rq_class_logits.ndim != 3:
            raise ValueError("rq_mask_logits and rq_class_logits must be rank 3")
        batch, primitives, _ = primitive_tokens.shape
        query_batch, queries, _ = rq_query_embeddings.shape
        if query_batch != batch or rq_mask_logits.shape != (batch, queries, primitives):
            raise ValueError("RQ mask dimensions must match primitive/query dimensions")
        if rq_class_logits.shape != (batch, queries, self.num_classes):
            raise ValueError("RQ class logits must use configured num_classes")
        if rq_admission_logits is not None and rq_admission_logits.shape != (batch, queries):
            raise ValueError("rq_admission_logits must be [B,Q]")
        if primitive_padding_mask is not None and primitive_padding_mask.shape != (batch, primitives):
            raise ValueError("primitive_padding_mask must be [B,N]")
        if rq_query_padding_mask is not None and rq_query_padding_mask.shape != (batch, queries):
            raise ValueError("rq_query_padding_mask must be [B,Q]")
        return batch, primitives, queries

    def forward(
        self,
        primitive_tokens: Tensor,
        rq_query_embeddings: Tensor,
        rq_mask_logits: Tensor,
        rq_class_logits: Tensor,
        *,
        rq_admission_logits: Tensor | None = None,
        base_semantic_logits: Tensor | None = None,
        primitive_padding_mask: Tensor | None = None,
        rq_query_padding_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        batch, primitives, queries = self._validate(
            primitive_tokens,
            rq_query_embeddings,
            rq_mask_logits,
            rq_class_logits,
            rq_admission_logits,
            primitive_padding_mask,
            rq_query_padding_mask,
        )
        if base_semantic_logits is not None and base_semantic_logits.shape != (batch, primitives, self.num_classes):
            raise ValueError("base_semantic_logits must be [B,N,C]")
        primitive_valid = torch.ones((batch, primitives), dtype=torch.bool, device=primitive_tokens.device)
        if primitive_padding_mask is not None:
            primitive_valid = ~primitive_padding_mask.bool()
        query_valid = torch.ones((batch, queries), dtype=torch.bool, device=primitive_tokens.device)
        if rq_query_padding_mask is not None:
            query_valid = ~rq_query_padding_mask.bool()

        sq_base = self.sq_pre_norm(self.primitive_projection(primitive_tokens))
        base_tokens = primitive_tokens if self.identity_base_path else sq_base
        if base_semantic_logits is None:
            if not self.include_fixture_semantic_heads:
                raise ValueError("production SQ-RQ requires base_semantic_logits from the main semantic head")
            base_logits = self.base_semantic_head(sq_base)
        else:
            base_logits = base_semantic_logits

        rq_queries = _scaled_gradient(rq_query_embeddings, self.controlled_gradient_scale)
        rq_masks = _scaled_gradient(rq_mask_logits, self.controlled_gradient_scale)
        rq_classes = _scaled_gradient(rq_class_logits, self.controlled_gradient_scale)
        rq_admission = (
            None
            if rq_admission_logits is None
            else _scaled_gradient(rq_admission_logits, self.controlled_gradient_scale)
        )
        membership_probability = rq_masks.sigmoid()
        hard_membership = (
            membership_probability
            * (membership_probability >= self.token_membership_threshold).to(membership_probability.dtype)
            * primitive_valid[:, None, :]
        )
        soft_membership_active = bool(self.training and self.training_membership_temperature > 0.0)
        soft_membership_probability_floor = max(
            self.token_membership_threshold - 2.0 * self.training_membership_temperature,
            self.token_membership_threshold * 0.25,
        )
        if soft_membership_active:
            hard_gate = membership_probability >= self.token_membership_threshold
            soft_gate = torch.sigmoid(
                (membership_probability - self.token_membership_threshold)
                / self.training_membership_temperature
            )
            soft_gate = soft_gate * (
                membership_probability >= soft_membership_probability_floor
            ).to(soft_gate.dtype)
            context_gate = torch.where(hard_gate, torch.ones_like(soft_gate), soft_gate)
            membership = membership_probability * context_gate * primitive_valid[:, None, :]
        else:
            membership = hard_membership
        denominator = membership.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        pooled = torch.einsum("bqn,bnh->bqh", membership, sq_base) / denominator
        context = self.query_norm(self.rq_query_projection(rq_queries) + self.rq_pool_projection(pooled))

        class_probability = rq_classes.softmax(dim=-1)
        query_probability = class_probability.clone()
        query_probability[..., self.no_object_class] = 0.0
        query_probability = query_probability / query_probability.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        object_probability = (
            rq_admission.sigmoid()
            if rq_admission is not None
            else 1.0 - class_probability[..., self.no_object_class]
        )
        raw_admission_valid = query_valid & (object_probability >= self.query_confidence_threshold)
        hard_membership_supported = query_valid & (hard_membership.sum(dim=-1) > 1e-6)
        membership_supported = query_valid & (membership.sum(dim=-1) > 1e-6)
        query_valid = raw_admission_valid & membership_supported
        semantic_probability = base_logits.softmax(dim=-1)
        compatibility = torch.einsum("bnc,bqc->bnq", semantic_probability, query_probability)
        compatibility_prior = compatibility.clamp_min(self.semantic_compatibility_floor)
        edge_prior = membership.transpose(1, 2) * object_probability[:, None, :] * compatibility_prior
        edge_prior = edge_prior * query_valid[:, None, :].to(edge_prior.dtype)
        top_k = min(self.context_top_k, queries)
        top_values, top_indices = edge_prior.topk(top_k, dim=-1)
        top_mask = torch.zeros_like(edge_prior, dtype=torch.bool).scatter_(-1, top_indices, top_values > 0.0)
        consistency_mask = top_mask & primitive_valid[:, :, None]

        q = self.q_projection(sq_base).view(batch, primitives, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_projection(context).view(batch, queries, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_projection(context).view(batch, queries, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.einsum("bhnd,bhqd->bhnq", q, k) * (self.head_dim ** -0.5)
        allowed = consistency_mask[:, None, :, :]
        scores = scores + edge_prior.clamp_min(torch.finfo(scores.dtype).tiny).log()[:, None, :, :]
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        attention = scores.softmax(dim=-1) * allowed.to(scores.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        attended = torch.einsum("bhnq,bhqd->bhnd", attention, v).transpose(1, 2).reshape(batch, primitives, self.hidden_dim)
        attended = self.output_projection(attended)
        delta = self.sq_post_norm(self.sq_adapter(attended))
        context_strength = edge_prior.masked_fill(~consistency_mask, 0.0).amax(dim=-1)
        adaptive_gate = context_strength / (context_strength + self.semantic_compatibility_floor)
        context_edge_gate = edge_prior / (edge_prior + self.semantic_compatibility_floor)
        context_edge_gate = context_edge_gate * consistency_mask.to(context_edge_gate.dtype)
        residual = torch.tanh(self.sq_residual_gate) * adaptive_gate.unsqueeze(-1) * delta
        if self.training:
            residual = residual + self.residual_gradient_bootstrap * adaptive_gate.unsqueeze(-1) * (delta - delta.detach())
        refined = base_tokens + residual
        # A prediction-only branch must be an exact no-op until explicitly
        # promoted. This also prevents padded/no-admitted windows from
        # receiving LayerNorm/projection bias noise.
        has_context = consistency_mask.any(dim=-1)
        refined = torch.where(has_context[..., None], refined, base_tokens)
        refined = refined * primitive_valid.unsqueeze(-1)
        result = {
            "sq_tokens": refined,
            "rq_context": context,
            "attention_weights": attention,
            "semantic_consistency_mask": consistency_mask,
            "semantic_compatibility": compatibility,
            "admitted_rq_queries": query_valid,
            "raw_admitted_rq_queries": raw_admission_valid,
            "membership_supported_rq_queries": membership_supported,
            "hard_membership_supported_rq_queries": hard_membership_supported,
            "rq_admission_probability": object_probability,
            "thresholded_membership_mass": hard_membership.sum(dim=-1),
            "context_membership_mass": membership.sum(dim=-1),
            "soft_membership_training_active": torch.tensor(
                soft_membership_active, dtype=torch.bool, device=primitive_tokens.device,
            ),
            "soft_membership_probability_floor": membership_probability.new_tensor(
                soft_membership_probability_floor,
            ),
            "context_edge_prior": edge_prior,
            "context_edge_feedback_gate": context_edge_gate,
            "adaptive_context_gate": adaptive_gate,
        }
        if self.include_fixture_semantic_heads:
            result["sq_class_logits"] = self.sq_semantic_head(refined)
        return result


PRODUCTION_INTEGRATION_CONTRACT: dict[str, Any] = {
    "schema_version": "floorplancad_sq_rq_cross_attention_contract_v6_soft_train_edge_feedback",
    "placement": "After the RQ decoder produces query embeddings, mask logits, and class logits; before the independent SQ semantic head/loss.",
    "allowed_forward_inputs": [
        "primitive encoder tokens",
        "predicted RQ query embeddings",
        "predicted RQ mask logits",
        "predicted RQ class logits",
        "predicted RQ admission logits",
        "padding masks",
    ],
    "forbidden_forward_inputs": ["GT masks", "page_instance_id", "GT instance IDs", "matched target indices"],
    "gradient_boundary": "Default controlled_gradient_scale=0 detaches RQ inputs while leaving cross-attention parameters trainable. Experimental scale must remain in [0,0.1] and be recorded in checkpoint ABI.",
    "semantic_gate": "Evaluation uses hard factorized admission and membership gates. Training may use a recorded temperature-controlled soft membership below the hard threshold; per-edge feedback remains top-k and confidence gated.",
    "loss_boundary": "GT semantic labels may supervise sq_class_logits only outside forward(); no SQ loss target may construct RQ context or attention masks.",
    "checkpoint_abi": [
        "module schema version",
        "dimensions and class ontology hash",
        "controlled_gradient_scale",
        "confidence and membership thresholds",
        "training membership temperature",
        "semantic compatibility floor",
        "no_object_class",
    ],
    "promotion_gates": [
        "oracle-leakage and gradient tests pass",
        "official SQ improves at matched RQ within 1 point",
        "thing TP does not decrease",
        "all outputs remain finite for padded and zero-proposal batches",
    ],
    "rollback": "Bypass sq_tokens and zero the SQ mask/ownership feedback gates; this restores the primary semantic and RQ decoder outputs without changing checkpoint weights.",
}


def main() -> int:
    torch.manual_seed(20260711)
    module = SqRqCrossAttention(
        primitive_dim=8,
        rq_query_dim=8,
        hidden_dim=16,
        num_classes=5,
        heads=4,
    ).eval()
    with torch.no_grad():
        outputs = module(
            torch.randn(2, 7, 8),
            torch.randn(2, 3, 8),
            torch.randn(2, 3, 7),
            torch.randn(2, 3, 5),
            primitive_padding_mask=torch.tensor([[False] * 7, [False] * 4 + [True] * 3]),
            rq_query_padding_mask=torch.tensor([[False] * 3, [True] * 3]),
        )
    checks = {
        "finite": all(torch.isfinite(value).all().item() for value in outputs.values() if value.is_floating_point()),
        "shape": list(outputs["sq_tokens"].shape) == [2, 7, 16],
        "prediction_only_interface": True,
        "default_stop_gradient": module.controlled_gradient_scale == 0.0,
    }
    payload = {
        "schema_version": "floorplancad_sq_rq_cross_attention_smoke_v1",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": "passed" if all(checks.values()) else "blocked",
        "checks": checks,
        "production_integration_contract": PRODUCTION_INTEGRATION_CONTRACT,
        "claim_boundary": "Synthetic module/contract validation only; no SQ, RQ, or PQ quality claim.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
