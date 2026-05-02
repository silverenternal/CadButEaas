"""Shared schema helpers for CadStruct MoE experts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_ONTOLOGY_PATH = Path("configs/vlm/cadstruct_ontology.json")


@dataclass(frozen=True)
class RoutedCandidate:
    """A model-agnostic item routed to one expert family."""

    candidate_id: str
    expert: str
    family: str
    candidate_type: str
    confidence: float
    bbox: list[float] | None = None
    source: str = "deterministic_router"
    payload: dict[str, Any] = field(default_factory=dict)
    route_trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        # Flatten route_trace for easier auditing
        if self.route_trace:
            result["matched_hint"] = self.route_trace.get("matched_hint")
            result["routing_confidence"] = self.route_trace.get("routing_confidence", self.confidence)
            result["abstain"] = self.route_trace.get("abstain", False)
        return result


@dataclass(frozen=True)
class ExpertPrediction:
    """A normalized expert output before cross-expert fusion."""

    candidate_id: str
    expert: str
    family: str
    label: str
    confidence: float
    bbox: list[float] | None = None
    geometry: dict[str, Any] = field(default_factory=dict)
    relations: list[dict[str, Any]] = field(default_factory=list)
    source: str = "expert"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FusionResult:
    """Integrated scene graph candidates plus audit warnings."""

    predictions: list[ExpertPrediction]
    scene_graph: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "predictions": [item.to_dict() for item in self.predictions],
            "scene_graph": self.scene_graph,
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }


def load_ontology(path: str | Path = DEFAULT_ONTOLOGY_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def family_to_expert(ontology: dict[str, Any]) -> dict[str, str]:
    families = ontology.get("families") or {}
    return {
        str(family): str(config.get("primary_expert"))
        for family, config in families.items()
        if isinstance(config, dict) and config.get("primary_expert")
    }


def label_to_family(ontology: dict[str, Any]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for family, config in (ontology.get("families") or {}).items():
        for label in config.get("labels") or []:
            labels[str(label)] = str(family)
    return labels
