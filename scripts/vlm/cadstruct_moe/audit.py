"""Reusable audit summaries for CadStruct MoE expert execution."""

from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any

from .schema import ExpertPrediction, RoutedCandidate
from .experts.base import BaseExpert
from .experts.registry import describe_experts


def summarize_confidences(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "max": None}
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "min": round(ordered[0], 6),
        "mean": round(mean(ordered), 6),
        "max": round(ordered[-1], 6),
    }


def summarize_expert_execution(
    expert: BaseExpert,
    candidates: list[RoutedCandidate],
    predictions: list[ExpertPrediction],
) -> dict[str, Any]:
    fallback_predictions = [
        prediction for prediction in predictions
        if bool((prediction.metadata or {}).get("fallback"))
    ]
    routed_abstains = [
        candidate for candidate in candidates
        if bool((candidate.route_trace or {}).get("abstain"))
    ]
    missing_predictions = sorted(
        set(candidate.candidate_id for candidate in candidates)
        - set(prediction.candidate_id for prediction in predictions)
    )
    extra_predictions = sorted(
        set(prediction.candidate_id for prediction in predictions)
        - set(candidate.candidate_id for candidate in candidates)
    )
    load_error = getattr(expert, "_load_error", None)
    return {
        "expert": expert.name,
        "family": expert.family,
        "loaded": expert.is_loaded(),
        "checkpoint_hint": expert.checkpoint_hint,
        "label_space": list(expert.label_space),
        "candidate_count": len(candidates),
        "prediction_count": len(predictions),
        "missing_prediction_count": len(missing_predictions),
        "missing_prediction_ids_sample": missing_predictions[:20],
        "extra_prediction_count": len(extra_predictions),
        "extra_prediction_ids_sample": extra_predictions[:20],
        "fallback_prediction_count": len(fallback_predictions),
        "fallback_prediction_ratio": round(len(fallback_predictions) / max(len(predictions), 1), 6),
        "routed_abstain_count": len(routed_abstains),
        "routed_abstain_ratio": round(len(routed_abstains) / max(len(candidates), 1), 6),
        "candidate_confidence": summarize_confidences([candidate.confidence for candidate in candidates]),
        "prediction_confidence": summarize_confidences([prediction.confidence for prediction in predictions]),
        "candidate_type_counts": dict(Counter(candidate.candidate_type for candidate in candidates).most_common()),
        "prediction_label_counts": dict(Counter(prediction.label for prediction in predictions).most_common()),
        "prediction_source_counts": dict(Counter(prediction.source for prediction in predictions).most_common()),
        "load_error": str(load_error) if load_error else None,
        "status": "ok" if not load_error and len(missing_predictions) == 0 else "needs_attention",
    }


def summarize_moe_execution(
    experts: dict[str, BaseExpert],
    candidates_by_family: dict[str, list[RoutedCandidate]],
    predictions_by_family: dict[str, list[ExpertPrediction]],
) -> dict[str, Any]:
    expert_registry = describe_experts(experts)
    family_audit = {
        family: summarize_expert_execution(
            expert,
            candidates_by_family.get(family, []),
            predictions_by_family.get(family, []),
        )
        for family, expert in experts.items()
    }
    attention = [
        family for family, audit in family_audit.items()
        if audit.get("status") != "ok"
        or audit.get("fallback_prediction_count", 0) > 0
        or audit.get("missing_prediction_count", 0) > 0
    ]
    return {
        "version": "cadstruct_moe_expert_execution_audit_v1",
        "expert_registry": expert_registry,
        "families": family_audit,
        "needs_attention": sorted(attention),
        "totals": {
            "candidates": sum(len(items) for items in candidates_by_family.values()),
            "predictions": sum(len(items) for items in predictions_by_family.values()),
            "fallback_predictions": sum(int(audit.get("fallback_prediction_count", 0)) for audit in family_audit.values()),
            "missing_predictions": sum(int(audit.get("missing_prediction_count", 0)) for audit in family_audit.values()),
        },
    }
