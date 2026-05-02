"""Base expert interfaces for the CadStruct MoE scaffold."""

from __future__ import annotations

from dataclasses import dataclass

from ..schema import ExpertPrediction, RoutedCandidate


@dataclass
class BaseExpert:
    name: str
    family: str

    def predict(self, candidates: list[RoutedCandidate]) -> list[ExpertPrediction]:
        raise NotImplementedError


class PassthroughExpert(BaseExpert):
    """Non-learning placeholder used for integration and schema tests."""

    default_label: str = "unknown"

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
