"""Lightweight CadStruct MoE scaffolding.

This package intentionally avoids torch/model dependencies. It defines the
stable routing, schema, and fusion surface that model-specific experts can
implement later.
"""

from .router import DeterministicRouter, route_record
from .audit import summarize_expert_execution, summarize_moe_execution
from .schema import ExpertPrediction, FusionResult, RoutedCandidate, load_ontology
from .experts import DEFAULT_EXPERT_SPECS, ExpertSpec, build_default_experts, build_expert, describe_experts

__all__ = [
    "DeterministicRouter",
    "DEFAULT_EXPERT_SPECS",
    "ExpertPrediction",
    "ExpertSpec",
    "FusionResult",
    "build_default_experts",
    "build_expert",
    "describe_experts",
    "summarize_expert_execution",
    "summarize_moe_execution",
    "RoutedCandidate",
    "load_ontology",
    "route_record",
]
