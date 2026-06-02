"""Sheet/Layout expert wrapper.

Uses rule-based heuristics to classify sheet layout elements into:
  title_block, legend, schedule, stamp, notes

No trained model is available (training data only had "notes" class),
so this uses position cues and keyword matching from the training script.
"""

from __future__ import annotations

import math
import re
from typing import Any

from ..schema import ExpertPrediction, RoutedCandidate
from .base import BaseExpert, PassthroughExpert

LAYOUT_LABELS = ["title_block", "legend", "schedule", "stamp", "notes"]


def _normalize_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.strip().lower()
    text = re.sub(r"[\s_\-]+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _aspect_ratio(w: float, h: float) -> float:
    if h == 0:
        return float("inf")
    return max(w / h, h / w)


def _rule_based_detect(
    bbox: list[float], sheet_w: float, sheet_h: float, text: str
) -> tuple[str, float]:
    """Heuristic layout detection based on position + text cues."""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    bottom_frac = (sheet_h - y2) / max(sheet_h, 1.0)
    right_frac = (sheet_w - x2) / max(sheet_w, 1.0)
    left_frac = x1 / max(sheet_w, 1.0)
    top_frac = y1 / max(sheet_h, 1.0)
    area_ratio = (w * h) / max(sheet_w * sheet_h, 1.0)

    text_lower = text.lower()

    # Title block: bottom-right corner, large area, keywords
    if bottom_frac < 0.15 and right_frac < 0.35 and area_ratio > 0.01:
        if any(kw in text_lower for kw in ["title", "project", "drawing", "scale", "date", "sheet"]):
            return "title_block", 0.92

    # Schedule/table: rectangular grid pattern, typically right or center
    if any(kw in text_lower for kw in ["schedule", "room", "door", "area", "finish", "qty", "type"]):
        if area_ratio > 0.02 and _aspect_ratio(w, h) < 3.0:
            return "schedule", 0.88

    # Legend: right margin, symbol descriptions
    if right_frac < 0.20 and (y1 + y2) / 2 > sheet_h * 0.3:
        if any(kw in text_lower for kw in ["legend", "symbol", "key", "note"]):
            return "legend", 0.85

    # Stamp: small rectangular box, typically top-right or bottom-right
    if area_ratio < 0.02 and _aspect_ratio(w, h) < 2.0:
        if any(kw in text_lower for kw in ["revision", "approval", "stamp", "date", "checked"]):
            return "stamp", 0.82

    # Notes: text blocks outside main geometry area
    if left_frac < 0.10 and top_frac > 0.05:
        if any(kw in text_lower for kw in ["general note", "specification", "contractor", "verify"]):
            return "notes", 0.80

    # Default: notes for unmapped text blocks
    return "notes", 0.50


class SheetLayoutExpert(PassthroughExpert):
    """Sheet/Layout expert using rule-based heuristics.

    No trained model is available (training data only had "notes" class).
    Uses position cues (bottom-right=title_block, right margin=legend,
    small box=stamp) + keyword matching.
    """

    def __init__(self) -> None:
        super().__init__(
            name="sheet_layout",
            family="sheet",
            label_space=tuple(LAYOUT_LABELS),
            checkpoint_hint=None,
        )
        self.default_label = "title_block"

    def _get_page_size(self, candidates: list[RoutedCandidate]) -> tuple[float, float]:
        """Extract page size from candidate payloads."""
        for c in candidates:
            meta = c.payload.get("_page_metadata") or {}
            if meta.get("width") and meta.get("height"):
                return float(meta["width"]), float(meta["height"])
        return 2000.0, 2000.0  # default fallback

    def predict(self, candidates: list[RoutedCandidate]) -> list[ExpertPrediction]:
        sheet_w, sheet_h = self._get_page_size(candidates)
        predictions: list[ExpertPrediction] = []

        for candidate in candidates:
            bbox = _normalize_bbox(candidate.bbox)
            if bbox is None:
                predictions.append(
                    ExpertPrediction(
                        candidate_id=candidate.candidate_id,
                        expert=self.name,
                        family=self.family,
                        label=self.default_label,
                        confidence=candidate.confidence,
                        bbox=candidate.bbox,
                        source=f"{self.name}_passthrough",
                        metadata={"candidate_type": candidate.candidate_type, "fallback": True},
                    )
                )
                continue

            text = candidate.payload.get("raw_text") or candidate.payload.get("text") or ""
            label, confidence = _rule_based_detect(bbox, sheet_w, sheet_h, text)

            # Scale confidence slightly since this is heuristic
            confidence = min(confidence * 0.85, 0.95)

            predictions.append(
                ExpertPrediction(
                    candidate_id=candidate.candidate_id,
                    expert=self.name,
                    family=self.family,
                    label=label,
                    confidence=confidence,
                    bbox=candidate.bbox,
                    source=f"{self.name}_rule_based",
                    metadata={
                        "candidate_type": candidate.candidate_type,
                        "page_size": [sheet_w, sheet_h],
                    },
                )
            )

        return predictions

    def is_loaded(self) -> bool:
        return True
