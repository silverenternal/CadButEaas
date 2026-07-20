#!/usr/bin/env python3
"""Learned cross-window identity embeddings and GT-free reciprocal tracking."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/floorplancad_window_identity_embedding_smoke.json"
FORBIDDEN_INFERENCE_FIELDS = frozenset({"page_instance_id", "gt_instance_id", "gt_label", "target_id"})


@dataclass(frozen=True)
class WindowIdentityConfig:
    hidden_dim: int = 64
    identity_dim: int = 32
    temperature: float = 0.1
    negative_margin: float = 0.25
    embedding_weight: float = 0.65
    overlap_weight: float = 0.35
    assignment_threshold: float = 0.55
    thing_label_max: int = 29


class QueryIdentityHead(nn.Module):
    """Project decoder queries into a normalized metric identity space."""

    def __init__(self, hidden_dim: int, identity_dim: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, identity_dim),
        )

    def forward(self, query_embeddings: torch.Tensor) -> torch.Tensor:
        if query_embeddings.ndim != 3:
            raise ValueError("query_embeddings must have shape [windows, queries, hidden_dim]")
        return F.normalize(self.projection(query_embeddings), dim=-1, eps=1e-6)


def adjacent_window_identity_loss(
    identity_embeddings: torch.Tensor,
    page_instance_ids: Sequence[Sequence[str | None]],
    mask_loss_valid: torch.Tensor,
    window_indices: Sequence[int],
    page_ids: Sequence[str],
    *,
    temperature: float = 0.1,
    negative_margin: float = 0.25,
) -> tuple[torch.Tensor, dict[str, int | float]]:
    """Symmetric adjacent-window assignment plus negative separation.

    ``page_instance_ids`` is supervision only. Partial targets are removed from both
    positive and negative sets so clipped masks cannot teach false identities.
    """

    if identity_embeddings.ndim != 3 or mask_loss_valid.shape != identity_embeddings.shape[:2]:
        raise ValueError("identity embeddings and mask_loss_valid shapes disagree")
    windows, queries, _ = identity_embeddings.shape
    if not (len(page_instance_ids) == len(window_indices) == len(page_ids) == windows):
        raise ValueError("window metadata length mismatch")
    if any(len(row) != queries for row in page_instance_ids):
        raise ValueError("page_instance_ids query dimension mismatch")

    assignment_terms: list[torch.Tensor] = []
    negative_terms: list[torch.Tensor] = []
    positive_pairs = 0
    negative_pairs = 0
    for left in range(windows):
        for right in range(left + 1, windows):
            if page_ids[left] != page_ids[right] or abs(int(window_indices[left]) - int(window_indices[right])) != 1:
                continue
            left_valid = [index for index in range(queries) if bool(mask_loss_valid[left, index]) and page_instance_ids[left][index] is not None]
            right_valid = [index for index in range(queries) if bool(mask_loss_valid[right, index]) and page_instance_ids[right][index] is not None]
            if not left_valid or not right_valid:
                continue
            similarity = identity_embeddings[left, left_valid] @ identity_embeddings[right, right_valid].transpose(0, 1)
            for transpose in (False, True):
                scores = similarity.transpose(0, 1) if transpose else similarity
                source = right_valid if transpose else left_valid
                candidates = left_valid if transpose else right_valid
                source_window = right if transpose else left
                candidate_window = left if transpose else right
                for row_index, source_index in enumerate(source):
                    identity = page_instance_ids[source_window][source_index]
                    matches = [column for column, candidate in enumerate(candidates) if page_instance_ids[candidate_window][candidate] == identity]
                    if len(matches) == 1:
                        target = torch.tensor([matches[0]], device=scores.device)
                        assignment_terms.append(F.cross_entropy(scores[row_index : row_index + 1] / temperature, target))
                        positive_pairs += 1
            different = torch.tensor(
                [[page_instance_ids[left][left_index] != page_instance_ids[right][right_index] for right_index in right_valid] for left_index in left_valid],
                dtype=torch.bool,
                device=identity_embeddings.device,
            )
            if different.any():
                negative_terms.append(F.relu(similarity[different] - negative_margin).mean())
                negative_pairs += int(different.sum().item())

    zero = identity_embeddings.sum() * 0.0
    assignment_loss = torch.stack(assignment_terms).mean() if assignment_terms else zero
    negative_loss = torch.stack(negative_terms).mean() if negative_terms else zero
    total = assignment_loss + negative_loss
    return total, {
        "positive_assignment_directions": positive_pairs,
        "negative_pairs": negative_pairs,
        "partial_or_unmatched_excluded": int((~mask_loss_valid).sum().item()),
        "assignment_loss": float(assignment_loss.detach()),
        "negative_loss": float(negative_loss.detach()),
    }


def _overlap_score(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_ids = set(map(int, left["primitive_ids"]))
    right_ids = set(map(int, right["primitive_ids"]))
    return len(left_ids & right_ids) / max(min(len(left_ids), len(right_ids)), 1)


def reciprocal_embedding_tracks(
    proposals: Sequence[dict[str, Any]],
    *,
    embedding_weight: float = 0.65,
    overlap_weight: float = 0.35,
    assignment_threshold: float = 0.55,
    thing_label_max: int = 29,
) -> list[dict[str, Any]]:
    """Create tracks using predictions only; GT identity is forbidden at inference."""

    if abs(embedding_weight + overlap_weight - 1.0) > 1e-6:
        raise ValueError("embedding_weight and overlap_weight must sum to one")
    for proposal in proposals:
        forbidden = FORBIDDEN_INFERENCE_FIELDS.intersection(proposal)
        if forbidden:
            raise ValueError(f"GT-only inference fields are forbidden: {sorted(forbidden)}")
        required = {"label", "window_index", "primitive_ids", "identity_embedding"}
        if missing := required.difference(proposal):
            raise ValueError(f"proposal missing fields: {sorted(missing)}")

    parents = list(range(len(proposals)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    windows = sorted({int(proposal["window_index"]) for proposal in proposals})
    for left_window, right_window in zip(windows, windows[1:]):
        if right_window - left_window != 1:
            continue
        left_indices = [index for index, proposal in enumerate(proposals) if int(proposal["window_index"]) == left_window and int(proposal["label"]) <= thing_label_max]
        right_indices = [index for index, proposal in enumerate(proposals) if int(proposal["window_index"]) == right_window and int(proposal["label"]) <= thing_label_max]
        scores: dict[tuple[int, int], float] = {}
        for left in left_indices:
            for right in right_indices:
                if int(proposals[left]["label"]) != int(proposals[right]["label"]):
                    continue
                left_embedding = F.normalize(torch.as_tensor(proposals[left]["identity_embedding"], dtype=torch.float32), dim=0)
                right_embedding = F.normalize(torch.as_tensor(proposals[right]["identity_embedding"], dtype=torch.float32), dim=0)
                cosine = float(torch.dot(left_embedding, right_embedding).clamp(-1, 1))
                scores[left, right] = embedding_weight * ((cosine + 1.0) / 2.0) + overlap_weight * _overlap_score(proposals[left], proposals[right])
        left_best = {left: max((pair for pair in scores if pair[0] == left), key=scores.get, default=None) for left in left_indices}
        right_best = {right: max((pair for pair in scores if pair[1] == right), key=scores.get, default=None) for right in right_indices}
        for pair, score in scores.items():
            if score >= assignment_threshold and left_best.get(pair[0]) == pair and right_best.get(pair[1]) == pair:
                union(*pair)

    groups: dict[tuple[str, int], list[int]] = {}
    for index, proposal in enumerate(proposals):
        label = int(proposal["label"])
        key = ("thing", find(index)) if label <= thing_label_max else ("stuff", label)
        groups.setdefault(key, []).append(index)
    merged = []
    for indices in groups.values():
        fragments = [proposals[index] for index in indices]
        merged.append({
            "label": int(fragments[0]["label"]),
            "primitive_ids": sorted({int(value) for fragment in fragments for value in fragment["primitive_ids"]}),
            "window_indices": sorted({int(fragment["window_index"]) for fragment in fragments}),
            "merged_fragments": len(fragments),
            "score": max(float(fragment.get("score", 0.0)) for fragment in fragments),
            "decoder": "reciprocal_embedding_overlap_identity_tracks_v1",
        })
    return sorted(merged, key=lambda row: (row["label"], row["primitive_ids"]))


def build_smoke() -> dict[str, Any]:
    torch.manual_seed(20260711)
    config = WindowIdentityConfig(hidden_dim=16, identity_dim=8)
    head = QueryIdentityHead(config.hidden_dim, config.identity_dim)
    queries = torch.randn(3, 3, config.hidden_dim, requires_grad=True)
    embeddings = head(queries)
    valid = torch.tensor([[True, True, False], [True, True, False], [True, True, True]])
    identities = [["a", "b", "partial"], ["a", "b", None], ["a", "b", "c"]]
    loss, diagnostics = adjacent_window_identity_loss(embeddings, identities, valid, [0, 1, 2], ["p", "p", "p"])
    loss.backward()
    proposals = [
        {"label": 4, "window_index": 0, "primitive_ids": [1, 2], "identity_embedding": [1.0, 0.0], "score": 0.9},
        {"label": 4, "window_index": 1, "primitive_ids": [2, 3], "identity_embedding": [0.99, 0.01], "score": 0.8},
        {"label": 4, "window_index": 2, "primitive_ids": [3, 4], "identity_embedding": [1.0, 0.0], "score": 0.7},
        {"label": 4, "window_index": 1, "primitive_ids": [8, 9], "identity_embedding": [0.0, 1.0], "score": 0.6},
    ]
    tracks = reciprocal_embedding_tracks(proposals)
    checks = {
        "normalized_embeddings": bool(torch.allclose(embeddings.norm(dim=-1), torch.ones_like(embeddings[..., 0]), atol=1e-5)),
        "finite_loss_and_gradients": bool(torch.isfinite(loss) and all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in head.parameters())),
        "three_window_track": any(row["primitive_ids"] == [1, 2, 3, 4] and row["merged_fragments"] == 3 for row in tracks),
        "same_label_distinct_identity_not_merged": len(tracks) == 2,
        "inference_inputs_are_prediction_only": all(not FORBIDDEN_INFERENCE_FIELDS.intersection(proposal) for proposal in proposals),
    }
    return {
        "schema_version": "floorplancad_window_identity_embedding_smoke_v1",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": "passed_no_training_module" if all(checks.values()) else "failed",
        "training_performed": False,
        "config": asdict(config),
        "checks": checks,
        "loss_diagnostics": diagnostics,
        "track_fixture": tracks,
        "production_interfaces": {
            "train_model": "Instantiate QueryIdentityHead(hidden_dim, identity_dim); apply to every decoder layer query embedding [B,Q,H].",
            "train_targets": "After bank-local Hungarian matching, pass matched page_instance_id and mask_loss_valid. Batch windows from the same page in adjacency order; call adjacent_window_identity_loss per layer and add identity_weight*loss.",
            "partial_policy": "Set mask_loss_valid=false entries invalid before the identity loss; they create neither positive nor negative pairs.",
            "checkpoint_abi": "Persist identity_head state, identity_dim, temperature, negative_margin, and identity objective hash; reject absent/mismatched fields in production.",
            "apply_model": "Export normalized identity_embedding for admitted thing queries only. Build proposal dictionaries without any target field and call reciprocal_embedding_tracks before unique primitive ownership.",
            "inference_contract": "Uses predicted label, primitive mask membership, window index, calibrated score, and identity embedding only; forbidden GT keys fail closed.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    artifact = build_smoke()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": artifact["status"], "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
