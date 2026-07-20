"""Global query ownership primitives for ownership-before-mask decoding."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.floorplancad_panoptic_protocol import PANOPTIC_OBJECT_NORMALIZATION_THRESHOLD
from experiments.floorplancad_panoptic_scoring import (
    calibrated_proposal_score,
    mask_objectness_score_np as mask_objectness_score,
)

OWNERSHIP_VERSION = "global_query_token_categorical_with_null_v2_window_normalized"
DEFAULT_MIN_QUERY_SCORE = 0.2
OBJECT_NORMALIZATION_THRESHOLD = PANOPTIC_OBJECT_NORMALIZATION_THRESHOLD


def ownership_config(*, hidden_dim: int, num_queries: int) -> dict[str, Any]:
    return {
        "version": OWNERSHIP_VERSION,
        "hidden_dim": int(hidden_dim),
        "num_queries": int(num_queries),
        "categories": "all_queries_plus_null",
        "decode_order": "admit_queries_then_mask_guided_global_owner",
        "owner_parameterization": "mask_membership_plus_admission_margin_plus_zero_initialized_residual",
        "cross_window_evidence": "max_window_log_posterior_margin_to_null",
        "posthoc_conflict_policy": "assert_only",
        "forbidden_inputs": ["gt_masks", "page_instance_id", "matched_target_indices"],
    }


def ownership_targets(torch: Any, q_masks: Any, q_labels: Any, mask_valid: Any) -> Any:
    """Build one owner per token from Hungarian-aligned full-component targets."""
    query_count, token_count = q_masks.shape
    target = torch.full((token_count,), query_count, dtype=torch.long, device=q_masks.device)
    valid_queries = q_labels >= 0
    assignments = (q_masks[valid_queries] > 0.5)
    if assignments.numel():
        if bool((assignments.sum(0) > 1).any().item()):
            raise ValueError("GT ownership must be globally unique across matched queries")
        matched_indices = torch.nonzero(valid_queries, as_tuple=False).flatten()
        owned = assignments.any(0)
        local_owner = assignments.to(torch.int64).argmax(0)
        target[owned] = matched_indices[local_owner[owned]]
    target[~mask_valid.bool()] = -100
    return target


def ownership_cross_entropy(torch: Any, ownership_logits: Any, target: Any, primitive_weights: Any | None = None) -> Any:
    losses = torch.nn.functional.cross_entropy(ownership_logits, target, ignore_index=-100, reduction="none")
    valid = target != -100
    if not bool(valid.any().item()):
        return ownership_logits.sum() * 0.0
    weights = torch.ones_like(losses) if primitive_weights is None else primitive_weights.to(losses.dtype)
    return (losses[valid] * weights[valid]).sum() / weights[valid].sum().clamp_min(1e-8)


def ownership_mask_consistency_loss(
    torch: Any,
    ownership_logits: Any,
    mask_logits: Any,
    query_labels: Any,
    mask_valid: Any,
    *,
    no_object_label: int,
    primitive_weights: Any | None = None,
) -> Any:
    """Keep categorical ownership aligned to detached mask/admission evidence."""
    if ownership_logits.shape != (mask_logits.shape[1], mask_logits.shape[0] + 1):
        raise ValueError("ownership logits must be token-by-query-plus-null")
    admitted = (query_labels != int(no_object_label)).to(mask_logits.dtype)
    membership = torch.sigmoid(mask_logits).transpose(0, 1) * admitted.unsqueeze(0)
    null_membership = (1.0 - membership.max(dim=-1).values).clamp_min(0.0).unsqueeze(-1)
    target = torch.cat([membership, null_membership], dim=-1).detach()
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    token_loss = -(target * torch.log_softmax(ownership_logits.float(), dim=-1)).sum(dim=-1)
    valid = mask_valid.bool()
    if not bool(valid.any().item()):
        return ownership_logits.sum() * 0.0
    weights = torch.ones_like(token_loss) if primitive_weights is None else primitive_weights.to(token_loss.dtype)
    return (token_loss[valid] * weights[valid]).sum() / weights[valid].sum().clamp_min(1e-8)


def select_global_owners(ownership_logits: Any, admitted_queries: Any) -> Any:
    """Select one admitted query or null for every token without GT inputs."""
    query_count = ownership_logits.shape[-1] - 1
    if admitted_queries.shape[-1] != query_count:
        raise ValueError("admitted query mask does not match ownership categories")
    masked = ownership_logits.copy()
    masked[..., :query_count][..., ~admitted_queries.astype(bool)] = -float("inf")
    return masked.argmax(axis=-1)


def decode_page_global_track_ownership(
    primitive_id_rows: list[list[int]],
    query_logits_rows: list[np.ndarray],
    mask_logits_rows: list[np.ndarray],
    ownership_logits_rows: list[np.ndarray],
    identity_rows: list[np.ndarray],
    quality_rows: list[np.ndarray] | None,
    admitted_rows: list[np.ndarray],
    *,
    ignore_label: int,
    soft_support_threshold: float = 0.25,
    track_threshold: float = 0.55,
    ownership_membership_threshold: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    candidates: list[dict[str, Any]] = []
    for window, (primitive_ids, class_logits, mask_logits, identities, admitted) in enumerate(zip(
        primitive_id_rows, query_logits_rows, mask_logits_rows, identity_rows, admitted_rows, strict=True
    )):
        shifted = class_logits - class_logits.max(-1, keepdims=True)
        probabilities = np.exp(shifted) / np.exp(shifted).sum(-1, keepdims=True)
        mask_probabilities = 1.0 / (1.0 + np.exp(-np.clip(mask_logits, -60.0, 60.0)))
        for query in np.flatnonzero(admitted):
            label = int(probabilities[query, :ignore_label].argmax())
            support = {
                int(primitive_ids[index]): float(mask_probabilities[query, index])
                for index in range(len(primitive_ids)) if mask_probabilities[query, index] >= soft_support_threshold
            }
            candidates.append({
                "window": window, "query": int(query), "label": label, "support": support,
                "identity": np.asarray(identities[query], dtype=np.float64),
                "class_score": float(probabilities[query, label]),
                "objectness_score": float(1.0 - probabilities[query, ignore_label]),
                "quality_score": (
                    float(1.0 / (1.0 + np.exp(-quality_rows[window][query])))
                    if quality_rows is not None
                    else 1.0
                ),
                "mask_objectness_score": mask_objectness_score(mask_probabilities[query]),
            })
    parents = list(range(len(candidates)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parents[max(left, right)] = min(left, right)

    for window in range(max((row["window"] for row in candidates), default=-1)):
        left = [index for index, row in enumerate(candidates) if row["window"] == window and row["label"] < 30]
        right = [index for index, row in enumerate(candidates) if row["window"] == window + 1 and row["label"] < 30]
        scores: dict[tuple[int, int], float] = {}
        for left_index in left:
            for right_index in right:
                left_row, right_row = candidates[left_index], candidates[right_index]
                if left_row["label"] != right_row["label"]:
                    continue
                left_identity, right_identity = left_row["identity"], right_row["identity"]
                cosine = float(np.dot(left_identity, right_identity) / max(np.linalg.norm(left_identity) * np.linalg.norm(right_identity), 1e-8))
                shared = set(left_row["support"]) & set(right_row["support"])
                overlap = sum(min(left_row["support"][pid], right_row["support"][pid]) for pid in shared)
                denominator = max(min(sum(left_row["support"].values()), sum(right_row["support"].values())), 1e-8)
                scores[left_index, right_index] = 0.65 * ((cosine + 1.0) / 2.0) + 0.35 * overlap / denominator
        left_best = {index: max((pair for pair in scores if pair[0] == index), key=scores.get, default=None) for index in left}
        right_best = {index: max((pair for pair in scores if pair[1] == index), key=scores.get, default=None) for index in right}
        for pair, score in scores.items():
            if score >= track_threshold and left_best[pair[0]] == pair and right_best[pair[1]] == pair:
                union(*pair)
    track_for_candidate: dict[int, tuple[str, int]] = {}
    for index, row in enumerate(candidates):
        track_for_candidate[index] = ("stuff", row["label"]) if row["label"] in range(30, 35) else ("thing", find(index))
    candidate_lookup = {(row["window"], row["query"]): index for index, row in enumerate(candidates)}
    ownership_evidence: dict[int, dict[tuple[str, int] | None, float]] = {}
    membership_filtered_assignments = 0
    membership_considered_assignments = 0
    for window, (primitive_ids, logits, mask_logits, admitted) in enumerate(zip(
        primitive_id_rows, ownership_logits_rows, mask_logits_rows, admitted_rows, strict=True
    )):
        null_index = logits.shape[1] - 1
        mask_probabilities = 1.0 / (1.0 + np.exp(-np.clip(mask_logits, -60.0, 60.0)))
        for local, primitive_id in enumerate(primitive_ids):
            evidence = ownership_evidence.setdefault(int(primitive_id), {})
            eligible: list[tuple[int, tuple[str, int]]] = []
            for query in np.flatnonzero(admitted):
                candidate = candidate_lookup.get((window, int(query)))
                if candidate is not None:
                    membership_considered_assignments += 1
                    if (
                        ownership_membership_threshold is not None
                        and float(mask_probabilities[query, local]) < ownership_membership_threshold
                    ):
                        membership_filtered_assignments += 1
                        continue
                    track = track_for_candidate[candidate]
                    eligible.append((int(query), track))
            category_indices = [query for query, _track in eligible] + [null_index]
            category_logits = np.asarray(logits[local, category_indices], dtype=np.float64)
            maximum = float(category_logits.max())
            log_normalizer = maximum + float(np.log(np.exp(category_logits - maximum).sum()))
            log_probabilities = category_logits - log_normalizer
            null_log_probability = float(log_probabilities[-1])
            evidence[None] = 0.0
            for (_query, track), log_probability in zip(eligible, log_probabilities[:-1], strict=True):
                log_posterior_margin = float(log_probability) - null_log_probability
                evidence[track] = max(evidence.get(track, -float("inf")), log_posterior_margin)
    owned: dict[tuple[str, int], list[int]] = {}
    null_count = 0
    for primitive_id, evidence in ownership_evidence.items():
        if not evidence:
            null_count += 1
            continue
        winner = max(evidence, key=evidence.get)
        if winner is None:
            null_count += 1
        else:
            owned.setdefault(winner, []).append(primitive_id)
    proposals = []
    for track, primitive_ids in owned.items():
        members = [row for index, row in enumerate(candidates) if track_for_candidate[index] == track]
        if not members:
            continue
        winner = max(
            members,
            key=lambda row: calibrated_proposal_score(row["class_score"], row["quality_score"])
            * row["mask_objectness_score"],
        )
        proposals.append({
            "label": int(winner["label"]), "primitive_ids": sorted(primitive_ids),
            "score": calibrated_proposal_score(winner["class_score"], winner["quality_score"]) * winner["mask_objectness_score"],
            "class_score": winner["class_score"], "objectness_score": winner["objectness_score"],
            "quality_score": winner["quality_score"], "mask_objectness_score": winner["mask_objectness_score"],
            "merged_fragments": len(members),
            "window_indices": sorted({row["window"] for row in members}),
            "ownership_before_mask": True, "decoder": "identity_soft_track_then_window_null_margin_page_global_ownership_v2",
        })
    return sorted(proposals, key=lambda row: row["score"], reverse=True), {
        "global_primitives": len(ownership_evidence), "null_owned_primitives": null_count,
        "tracks": len(set(track_for_candidate.values())), "instances": len(proposals),
        "ownership_membership_threshold": ownership_membership_threshold,
        "ownership_membership_considered_assignments": membership_considered_assignments,
        "ownership_membership_filtered_assignments": membership_filtered_assignments,
        "ownership_evidence_aggregation": "max_window_log_posterior_margin_to_null",
    }
