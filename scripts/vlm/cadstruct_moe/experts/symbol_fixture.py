"""Symbol/Fixture expert wrapper.

Loads the trained v9 ExtraTrees model (13 features: bbox + room context + neighbor)
and classifies symbols into 9 classes:
  appliance, bathtub, column, equipment, generic_symbol, shower, sink, stair, table

v9 improvements over v8:
- Uses class_weight='balanced' for better rare-class handling
- generic_symbol F1: 0.065 → 0.476, bathtub F1: 0.76 → 1.00
- Overall dev F1: 0.755 → 0.872

Falls back to passthrough ("generic_symbol") if model checkpoint is not found.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from ..schema import ExpertPrediction, RoutedCandidate
from .base import BaseExpert, PassthroughExpert

FEATURE_NAMES = [
    "cx", "cy", "width", "height", "area_norm", "log_aspect_ratio",
    "room_wet", "room_living", "room_service", "room_outdoor",
    "neighbor_count", "neighbor_avg_area_log", "neighbor_area_ratio_log",
]

_MODEL_DIR = Path(__file__).resolve().parents[4] / "checkpoints" / "symbol_fixture_expert_v9"


def _normalize_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def _extract_room_context(sym_bbox: list[float], rooms: list[dict]) -> list[float]:
    """Extract 4D room context features for a symbol."""
    wet, living, service, outdoor = 0.0, 0.0, 0.0, 0.0
    sym_cx = (sym_bbox[0] + sym_bbox[2]) / 2
    sym_cy = (sym_bbox[1] + sym_bbox[3]) / 2

    for room in rooms:
        room_bbox = _normalize_bbox(room.get("bbox"))
        if room_bbox and _room_contains(room_bbox, sym_cx, sym_cy):
            room_type = str(room.get("room_type", ""))
            if room_type in ("bathroom", "toilet", "shower_room"):
                wet = 1.0
            elif room_type in ("bedroom", "living_room", "kitchen", "corridor"):
                living = 1.0
            elif room_type in ("closet", "storage", "office"):
                service = 1.0
            elif room_type == "balcony":
                outdoor = 1.0
            break
    return [wet, living, service, outdoor]


def _room_contains(bbox: list[float], x: float, y: float) -> bool:
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def _extract_features(
    symbol_payload: dict[str, Any],
    rooms: list[dict],
    all_symbol_areas: list[float],
    mean_neighbor_area: float,
) -> list[float] | None:
    """Extract 13 features for a symbol candidate."""
    bbox = _normalize_bbox(symbol_payload.get("bbox"))
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    area = w * h
    log_aspect_ratio = math.log((w + 1.0) / (h + 1.0))

    # Bbox features (6) — normalized by page size if available
    meta = symbol_payload.get("_page_metadata") or {}
    img_w = float(meta.get("width", 0) or 2000.0)
    img_h = float(meta.get("height", 0) or 2000.0)
    bbox_feats = [
        cx / max(img_w, 1.0),
        cy / max(img_h, 1.0),
        w / max(img_w, 1.0),
        h / max(img_h, 1.0),
        area / max(img_w * img_h, 1.0),
        log_aspect_ratio,
    ]

    # Room context (4)
    room_feats = _extract_room_context(bbox, rooms)

    # Neighbor features (3)
    sym_area = area
    neighbor_count = max(0, len(all_symbol_areas) - 1)
    area_ratio = sym_area / max(mean_neighbor_area, 1.0)
    neighbor_feats = [
        min(neighbor_count, 100.0),
        math.log(mean_neighbor_area + 1.0),
        math.log(area_ratio + 1.0),
    ]

    return bbox_feats + room_feats + neighbor_feats


class SymbolFixtureExpert(PassthroughExpert):
    """Symbol/Fixture expert using trained v8 ExtraTrees model.

    Falls back to passthrough ("generic_symbol") if model checkpoint is not found.
    """

    def __init__(self) -> None:
        super().__init__(name="symbol_fixture", family="symbol")
        self.default_label = "generic_symbol"
        self._model: Any = None
        self._scaler: Any = None
        self._class_names: list[str] = []
        self._load_model()

    def _load_model(self) -> None:
        model_path = _MODEL_DIR / "model_v9.joblib"
        if not model_path.exists():
            return
        try:
            import joblib
            data = joblib.load(str(model_path))
            self._model = data.get("classifier")
            self._scaler = data.get("scaler")
            self._class_names = data.get("class_names", [])
        except Exception:
            self._model = None

    def predict(self, candidates: list[RoutedCandidate]) -> list[ExpertPrediction]:
        if self._model is None:
            return super().predict(candidates)

        # Build room list and symbol areas from candidates for context features
        rooms: list[dict] = []
        all_bboxes: list[list[float]] = []
        for candidate in candidates:
            bbox = candidate.bbox
            if bbox and len(bbox) == 4:
                all_bboxes.append([float(v) for v in bbox])
            # Collect rooms from payload if available
            room_list = candidate.payload.get("rooms", [])
            if room_list:
                rooms = room_list

        all_symbol_areas = [
            max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) for b in all_bboxes
        ]
        mean_neighbor_area = float(np.mean(all_symbol_areas)) if all_symbol_areas else 0.0

        predictions: list[ExpertPrediction] = []
        feature_rows: list[tuple[RoutedCandidate, list[float]]] = []

        for candidate in candidates:
            payload = dict(candidate.payload)
            if candidate.bbox is not None and "bbox" not in payload:
                payload["bbox"] = list(candidate.bbox)

            feats = _extract_features(payload, rooms, all_symbol_areas, mean_neighbor_area)
            if feats is None:
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
            X = np.array([fr[1] for fr in feature_rows], dtype=np.float64)
            X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)
            if self._scaler is not None:
                X = self._scaler.transform(X)

            y_pred = self._model.predict(X)
            y_pred_proba = self._model.predict_proba(X)

            for (candidate, _feats), pred_idx, proba in zip(feature_rows, y_pred, y_pred_proba):
                if self._class_names and pred_idx < len(self._class_names):
                    label = self._class_names[pred_idx]
                else:
                    label = str(pred_idx)
                confidence = float(max(proba))
                predictions.append(
                    ExpertPrediction(
                        candidate_id=candidate.candidate_id,
                        expert=self.name,
                        family=self.family,
                        label=label,
                        confidence=confidence,
                        bbox=candidate.bbox,
                        source=f"{self.name}_v9_extra_trees",
                        metadata={
                            "candidate_type": candidate.candidate_type,
                            "all_probs": {
                                self._class_names[i]: round(float(p), 4)
                                for i, p in enumerate(proba)
                            } if self._class_names else {},
                        },
                    )
                )

        return predictions
