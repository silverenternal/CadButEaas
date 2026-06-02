"""Text/Dimension expert wrapper.

Loads the trained v5-calibrated ExtraTrees model (29 features: bbox geometry +
layout/role + OCR text patterns + page context) and classifies text candidates into:
  dimension_line, dimension_text, leader_line, note_text, room_label
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from ..schema import ExpertPrediction, RoutedCandidate
from .base import BaseExpert, PassthroughExpert

FEATURE_NAMES = [
    "cx", "cy", "width", "height", "area", "log_aspect",
    "y_bucket_neg", "y_bucket_origin", "is_narrow_vertical", "is_horizontal_bar",
    "has_raw_text", "text_len", "word_count", "is_numeric", "has_x_separator",
    "has_dimension_unit", "has_foot_inch", "is_alpha_only", "has_digit_and_alpha",
    "is_short_label",
    "page_norm_cx", "page_norm_cy", "page_norm_w", "page_norm_h",
    "score_dimension_line", "score_dimension_text", "score_leader_line",
    "score_note_text", "score_room_label",
]

_MODEL_DIR = Path(__file__).resolve().parents[4] / "checkpoints" / "text_dimension_expert_v4_aug2"
_NOTE_TEXT_THRESHOLD = 0.69


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.strip().lower()
    text = re.sub(r"[\s_\-]+", " ", text)
    text = re.sub(r"[^a-z0-9\s.]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _compute_text_pattern_scores(raw: str, normalized: str, w: float, h: float) -> dict[str, float]:
    scores = {
        "score_dimension_line": 0.0,
        "score_dimension_text": 0.0,
        "score_leader_line": 0.0,
        "score_note_text": 0.0,
        "score_room_label": 0.0,
    }
    if not raw or not normalized:
        if 9.0 <= w <= 15.0 and 14.0 <= h <= 25.0:
            scores["score_leader_line"] = 0.6
        return scores

    if "x" in normalized.lower() or "\u00d7" in raw:
        scores["score_dimension_text"] = 0.95
        return scores

    cleaned = normalized.replace(" ", "").replace(".", "")
    is_alpha_only = cleaned.isalpha() and len(cleaned) > 0

    if is_alpha_only and len(cleaned) <= 25:
        scores["score_room_label"] = 0.9

    has_digit_and_alpha = bool(re.search(r"\d", raw)) and bool(re.search(r"[a-zA-Z]", raw))
    if has_digit_and_alpha and "x" not in normalized.lower():
        scores["score_note_text"] = 0.8

    if is_alpha_only and len(cleaned) > 5:
        scores["score_note_text"] = max(scores["score_note_text"], 0.4)

    if is_alpha_only and 2 <= len(cleaned) <= 4:
        scores["score_room_label"] = max(scores["score_room_label"], 0.6)
        scores["score_note_text"] = max(scores["score_note_text"], 0.3)

    return scores


def _extract_features(payload: dict[str, Any]) -> list[float] | None:
    """Extract 29 features from a text candidate payload."""
    bbox = payload.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None

    x1, y1, x2, y2 = [float(v) for v in bbox]
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    area = w * h
    log_aspect = math.log((w + 1.0) / max(h + 1.0, 1e-6))

    y_bucket_neg = float(y1 <= -9.5 and abs(y2) <= 0.5)
    y_bucket_origin = float(y1 < 0.0 and y2 <= 5.0 and w <= 15.0)
    is_narrow_vertical = float(9.0 <= w <= 15.0 and 14.0 <= h <= 25.0)
    is_horizontal_bar = float(y1 <= -4.5 and y2 >= -0.5 and abs(h) <= 11.0 and w <= 15.0)

    raw = payload.get("raw_text") or payload.get("text") or ""
    normalized = _normalize_text(raw)
    text_len = len(normalized)
    word_count = len(normalized.split()) if normalized else 0
    is_numeric = float(bool(re.match(r"^[\d.]+$", normalized.replace(" ", ""))))
    has_x_separator = float("x" in normalized.lower() or "\u00d7" in raw)
    has_dimension_unit = float(bool(re.search(r"\d\s*(mm|cm|m|in|ft)", raw, re.I)))
    has_foot_inch = float(bool(re.search(r"\d'[\d\"]*", raw)))
    cleaned = normalized.replace(" ", "").replace(".", "")
    is_alpha_only = float(cleaned.isalpha() and len(cleaned) > 0)
    has_digit_and_alpha = float(bool(re.search(r"\d", raw)) and bool(re.search(r"[a-zA-Z]", raw)))
    is_short_label = float(len(cleaned) <= 4 and cleaned.isalpha())

    meta = payload.get("_page_metadata") or {}
    img_w = float(meta.get("width", 0) or 1.0)
    img_h = float(meta.get("height", 0) or 1.0)
    page_norm_cx = cx / max(img_w, 1.0) if img_w > 1.0 else 0.5
    page_norm_cy = cy / max(img_h, 1.0) if img_h > 1.0 else 0.5
    page_norm_w = w / max(img_w, 1.0) if img_w > 1.0 else 0.01
    page_norm_h = h / max(img_h, 1.0) if img_h > 1.0 else 0.01

    pattern_scores = _compute_text_pattern_scores(raw, normalized, w, h)

    return [
        cx, cy, w, h, area, log_aspect,
        y_bucket_neg, y_bucket_origin, is_narrow_vertical, is_horizontal_bar,
        float(bool(raw)), text_len, word_count, is_numeric, has_x_separator,
        has_dimension_unit, has_foot_inch, is_alpha_only, has_digit_and_alpha,
        is_short_label,
        page_norm_cx, page_norm_cy, page_norm_w, page_norm_h,
        pattern_scores["score_dimension_line"], pattern_scores["score_dimension_text"],
        pattern_scores["score_leader_line"], pattern_scores["score_note_text"],
        pattern_scores["score_room_label"],
    ]


class TextDimensionExpert(PassthroughExpert):
    """Text/Dimension expert using trained v5-calibrated ExtraTrees model.

    Falls back to passthrough if model checkpoint is not found.
    """

    def __init__(self) -> None:
        super().__init__(
            name="text_dimension",
            family="text",
            label_space=("dimension_line", "dimension_text", "leader_line", "note_text", "room_label"),
            checkpoint_hint=str(_MODEL_DIR),
        )
        self.default_label = "dimension_text"
        self._model: Any = None
        self._encoder: Any = None
        self._class_names: list[str] = []
        self._load_model()

    def _load_model(self) -> None:
        model_path = _MODEL_DIR / "model_v4.joblib"
        if not model_path.exists():
            return
        try:
            import joblib
            data = joblib.load(str(model_path))
            self._model = data.get("classifier")
            self._encoder = data.get("encoder") or data.get("label_encoder")
            self._class_names = list(data.get("feature_names", FEATURE_NAMES))
        except Exception:
            self._model = None

    def is_loaded(self) -> bool:
        return self._model is not None

    def predict(self, candidates: list[RoutedCandidate]) -> list[ExpertPrediction]:
        if self._model is None:
            return super().predict(candidates)

        predictions: list[ExpertPrediction] = []
        feature_rows: list[tuple[RoutedCandidate, list[float]]] = []

        for candidate in candidates:
            # Ensure bbox is available in payload for feature extraction
            payload = dict(candidate.payload)
            if "bbox" not in payload and candidate.bbox is not None:
                payload["bbox"] = candidate.bbox
            feats = _extract_features(payload)
            if feats is None:
                # Fallback for candidates without valid bbox
                predictions.append(
                    ExpertPrediction(
                        candidate_id=candidate.candidate_id,
                        expert=self.name,
                        family=self.family,
                        label=self.default_label,
                        confidence=candidate.confidence,
                        bbox=candidate.bbox,
                        source=f"{self.name}_no_features",
                        metadata={"candidate_type": candidate.candidate_type, "fallback": True},
                    )
                )
                continue
            feature_rows.append((candidate, feats))

        if feature_rows:
            X = [fr[1] for fr in feature_rows]
            y_pred_idx = self._model.predict(X)
            y_pred_proba = self._model.predict_proba(X)

            for (candidate, _feats), pred_idx, proba in zip(feature_rows, y_pred_idx, y_pred_proba):
                label = self._encoder.inverse_transform([int(pred_idx)])[0] if self._encoder else str(pred_idx)
                if self._encoder is not None and label == "note_text":
                    classes = list(self._encoder.classes_)
                    if "note_text" in classes:
                        note_idx = classes.index("note_text")
                        if float(proba[note_idx]) < _NOTE_TEXT_THRESHOLD:
                            best_non_note = max(
                                (idx for idx, class_name in enumerate(classes) if class_name != "note_text"),
                                key=lambda idx: float(proba[idx]),
                            )
                            label = self._encoder.inverse_transform([int(best_non_note)])[0]
                            pred_idx = best_non_note
                confidence = float(max(proba))
                predictions.append(
                    ExpertPrediction(
                        candidate_id=candidate.candidate_id,
                        expert=self.name,
                        family=self.family,
                        label=label,
                        confidence=confidence,
                        bbox=candidate.bbox,
                        source=f"{self.name}_v5_calibrated_note_gate",
                        metadata={
                            "candidate_type": candidate.candidate_type,
                            "note_text_threshold": _NOTE_TEXT_THRESHOLD,
                            "all_probs": {
                                self._encoder.classes_[i]: round(float(p), 4)
                                for i, p in enumerate(proba)
                            } if self._encoder else {},
                        },
                    )
                )

        return predictions
