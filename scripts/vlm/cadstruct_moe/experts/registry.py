"""Expert registry and factories for the raster-only CadStruct MoE stack."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .base import BaseExpert
from .room_space import RoomSpaceExpert
from .sheet_layout import SheetLayoutExpert
from .symbol_fixture import SymbolFixtureExpert
from .text_dimension import TextDimensionExpert
from .wall_opening import WallOpeningExpert


ExpertFactory = Callable[[], BaseExpert]


@dataclass(frozen=True)
class ExpertSpec:
    """Static metadata for one independently replaceable expert module."""

    family: str
    name: str
    factory: ExpertFactory
    responsibility: str
    model_kind: str
    production_gate: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "name": self.name,
            "responsibility": self.responsibility,
            "model_kind": self.model_kind,
            "production_gate": self.production_gate,
        }


DEFAULT_EXPERT_SPECS: tuple[ExpertSpec, ...] = (
    ExpertSpec(
        family="boundary",
        name="wall_opening",
        factory=WallOpeningExpert,
        responsibility="Detect and type walls, doors, windows, openings, and boundary fragments.",
        model_kind="graph-node crop GNN with passthrough fallback",
        production_gate="boundary precision/recall/F1 >= 0.98 and source_integrity violations == 0",
    ),
    ExpertSpec(
        family="space",
        name="room_space",
        factory=RoomSpaceExpert,
        responsibility="Classify room/space proposals and preserve geometry needed for topology.",
        model_kind="room context sklearn classifier with passthrough fallback",
        production_gate="space precision/recall/F1 >= 0.98 and bounded_by support available",
    ),
    ExpertSpec(
        family="symbol",
        name="symbol_fixture",
        factory=SymbolFixtureExpert,
        responsibility="Classify symbol proposals separately from generic symbol objectness.",
        model_kind="symbol fixture classifier with abstain/pass-through fallback",
        production_gate="symbol objectness recall >= 0.98 and typed symbol precision/recall/F1 >= 0.98",
    ),
    ExpertSpec(
        family="text",
        name="text_dimension",
        factory=TextDimensionExpert,
        responsibility="Classify text/dimension candidates after OCR/text localization.",
        model_kind="text-dimension classifier with OCR feature inputs",
        production_gate="text localization precision/recall >= 0.98 and OCR normalized accuracy >= 0.98",
    ),
    ExpertSpec(
        family="sheet",
        name="sheet_layout",
        factory=SheetLayoutExpert,
        responsibility="Identify title blocks, legends, schedules, stamps, and other sheet regions.",
        model_kind="rule-based layout classifier until a trained model is available",
        production_gate="sheet region precision/recall/F1 >= 0.98 before it gates main-graph semantics",
    ),
)


def default_expert_specs() -> dict[str, ExpertSpec]:
    return {spec.family: spec for spec in DEFAULT_EXPERT_SPECS}


def build_default_experts(families: list[str] | tuple[str, ...] | None = None) -> dict[str, BaseExpert]:
    specs = default_expert_specs()
    selected = list(families) if families is not None else list(specs)
    unknown = [family for family in selected if family not in specs]
    if unknown:
        raise KeyError(f"unknown expert families: {', '.join(sorted(unknown))}")
    return {family: specs[family].factory() for family in selected}


def build_expert(family: str) -> BaseExpert:
    return build_default_experts([family])[family]


def describe_experts(experts: dict[str, BaseExpert] | None = None) -> dict[str, dict[str, Any]]:
    specs = default_expert_specs()
    live = experts or {}
    descriptions: dict[str, dict[str, Any]] = {}
    for family, spec in specs.items():
        item = spec.to_dict()
        expert = live.get(family)
        if expert is not None:
            item["runtime"] = expert.describe()
            item["loaded"] = expert.is_loaded()
        descriptions[family] = item
    return descriptions
