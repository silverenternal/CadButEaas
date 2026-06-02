"""Base expert interfaces for the CadStruct MoE scaffold."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..schema import ExpertPrediction, RoutedCandidate


@dataclass
class BaseExpert:
    name: str
    family: str
    label_space: tuple[str, ...] = field(default_factory=tuple)
    checkpoint_hint: str | None = None

    def predict(self, candidates: list[RoutedCandidate]) -> list[ExpertPrediction]:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "label_space": list(self.label_space),
            "checkpoint_hint": self.checkpoint_hint,
            "loaded": self.is_loaded(),
            "status": "custom" if type(self) is not BaseExpert else "base",
        }

    def is_loaded(self) -> bool:
        return False


class PassthroughExpert(BaseExpert):
    """Non-learning placeholder used for integration and schema tests."""

    default_label: str = "unknown"

    def __init__(self, name: str, family: str, label_space: tuple[str, ...] = (), checkpoint_hint: str | None = None) -> None:
        super().__init__(name=name, family=family, label_space=label_space, checkpoint_hint=checkpoint_hint)

    def predict(self, candidates: list[RoutedCandidate]) -> list[ExpertPrediction]:
        predictions: list[ExpertPrediction] = []
        for candidate in candidates:
            predictions.append(
                ExpertPrediction(
                    candidate_id=candidate.candidate_id,
                    expert=self.name,
                    family=self.family,
                    label=self.default_label,
                    confidence=candidate.confidence,
                    bbox=candidate.bbox,
                    source=f"{self.name}_passthrough",
                    metadata={"candidate_type": candidate.candidate_type},
                )
            )
        return predictions

    def is_loaded(self) -> bool:
        return False
