"""Lightweight CadStruct MoE scaffolding.

This package intentionally avoids torch/model dependencies. It defines the
stable routing, schema, and fusion surface that model-specific experts can
implement later.
"""

from .router import DeterministicRouter, route_record
from .schema import ExpertPrediction, FusionResult, RoutedCandidate, load_ontology

__all__ = [
    "DeterministicRouter",
    "ExpertPrediction",
    "FusionResult",
    "RoutedCandidate",
    "load_ontology",
    "route_record",
]
