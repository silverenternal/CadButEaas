"""Room/Space expert wrapper.

Loads the trained sklearn context model (enhanced 116 features: geometry +
contained symbols + boundaries + adjacency + text lexicon) and classifies
room candidates into types:
  room, bedroom, living_room, kitchen, bathroom, toilet, corridor, balcony,
  closet, office, storage, unknown_room

Falls back to passthrough ("room") if model checkpoint is not found.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ..schema import ExpertPrediction, RoutedCandidate
from .base import BaseExpert, PassthroughExpert

# Base feature names from context MLP (52D)
_BASE_FEATURE_NAMES = [
    "cx", "cy", "width", "height", "area", "aspect",
    "adjacency_degree", "contained_symbol_count", "contained_symbol_density",
    "room_label_count",
]

SYMBOL_TYPES = ["appliance", "bathtub", "column", "equipment", "generic_symbol", "shower", "sink", "stair", "table"]
BOUNDARY_TYPES = ["door", "hard_wall", "opening", "partition_wall", "window"]

# Room text lexicon
ROOM_TEXT_LABELS = [
    "balcony", "bathroom", "bedroom", "closet", "corridor",
    "kitchen", "living_room", "office", "storage", "toilet",
]

ROOM_TEXT_KEYWORDS = {
    "balcony": ["balcony", "terrace", "terassi", "kuisti", "parveke", "vilpola", "veranta", "patio", "lasikuisti", "avoterassi", "kattoterassi"],
    "bathroom": ["bath", "bathroom", "shower", "ph", "kh", "pesuh", "pesuhuone", "pesu", "kph", "psh", "sh", "suihku", "sauna", "pe"],
    "bedroom": ["bed", "br", "bedroom", "mh", "makuuhuone"],
    "closet": ["closet", "wardrobe", "cl", "vh", "vaatehuone", "pukuh", "pukuhuone", "pkh", "puku"],
    "corridor": ["hall", "corridor", "entry", "entrance", "et", "tk", "aula", "eteinen", "kaytava", "käytävä", "halli", "yla aula", "ylä aula"],
    "kitchen": ["kit", "kitchen", "k", "keittio", "keit", "kk", "tupak", "tupakeittio", "apuk", "apukeittio", "avok"],
    "living_room": ["living", "liv", "lounge", "oh", "olohuone", "rt", "r", "ruok", "ruokailu", "rh"],
    "office": ["office", "study", "tyoh", "tyohuone", "tyohuone", "th", "kirjasto", "toimisto", "arkisto"],
    "storage": ["storage", "store", "utility", "laundry", "khh", "var", "tekn", "varasto", "kodinhoito", "autotalli", "at", "var", "pannuh", "vaja", "ljh", "aitta", "kellari"],
    "toilet": ["wc", "toilet", "w c"],
}


def _normalize_room_text(value: str) -> str:
    text = value.lower().replace("ö", "o").replace("ä", "a").replace("å", "a")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _room_text_keyword_matches(label: str, text: str) -> bool:
    normalized = _normalize_room_text(text)
    tokens = set(normalized.split())
    for keyword in ROOM_TEXT_KEYWORDS.get(label, []):
        nk = _normalize_room_text(keyword)
        if len(nk) <= 3:
            if nk in tokens:
                return True
        elif nk in normalized:
            return True
    return False


def _room_text_match_vector(texts: list[str]) -> dict[str, float]:
    scores = {label: 0.0 for label in ROOM_TEXT_LABELS}
    for text in texts:
        for label in ROOM_TEXT_LABELS:
            if _room_text_keyword_matches(label, text):
                scores[label] += 1.0
    return scores


_MODEL_DIR = Path(__file__).resolve().parents[4] / "checkpoints" / "cadstruct_moe_room_space_hierarchical_sklearn_v5_t046"


# --- Bbox geometry helpers ---

def _bbox_area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _bbox_contains(left: list[float], right: list[float]) -> bool:
    return left[0] <= right[0] and left[1] <= right[1] and left[2] >= right[2] and left[3] >= right[3]


def _bbox_intersects(left: list[float], right: list[float]) -> bool:
    return not (left[2] < right[0] or right[2] < left[0] or left[3] < right[1] or right[3] < left[1])


def _overlap_length(left_min: float, left_max: float, right_min: float, right_max: float) -> float:
    return max(0.0, min(left_max, right_min) - max(left_min, right_min))  # BUG FIX: should be min(left_max, right_max)


def _overlap_length_fixed(left_min: float, left_max: float, right_min: float, right_max: float) -> float:
    return max(0.0, min(left_max, right_max) - max(left_min, right_min))


def _intersection_area(left: list[float], right: list[float]) -> float:
    return _overlap_length_fixed(left[0], left[2], right[0], right[2]) * _overlap_length_fixed(left[1], left[3], right[1], right[3])


def _bbox_gap(left: list[float], right: list[float]) -> float:
    dx = max(left[0] - right[2], right[0] - left[2], 0.0)
    dy = max(left[1] - right[3], right[1] - left[3], 0.0)
    return math.hypot(dx, dy)


def _bbox_center_inside(left: list[float], right: list[float]) -> bool:
    cx = (right[0] + right[2]) / 2.0
    cy = (right[1] + right[3]) / 2.0
    return left[0] <= cx <= left[2] and left[1] <= cy <= left[3]


def _adjacent(left: list[float], right: list[float]) -> bool:
    if _bbox_contains(left, right) or _bbox_contains(right, left):
        return False
    horizontal_gap = max(left[0] - right[2], right[0] - left[2], 0.0)
    vertical_gap = max(left[1] - right[3], right[1] - left[3], 0.0)
    if horizontal_gap > 2.0 or vertical_gap > 2.0:
        return False
    x_overlap = _overlap_length_fixed(left[0], left[2], right[0], right[2])
    y_overlap = _overlap_length_fixed(left[1], left[3], right[1], right[3])
    min_side = max(min(left[2] - left[0], left[3] - left[1], right[2] - right[0], right[3] - right[1]), 1.0)
    return max(x_overlap, y_overlap) / min_side >= 0.03


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _extract_room_features(
    room: dict[str, Any],
    context: dict[str, Any],
) -> list[float] | None:
    """Extract 116 enhanced features for a room candidate."""
    bbox = room["bbox"]
    width = float(context["width"])
    height = float(context["height"])
    x1, y1, x2, y2 = bbox
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    area = _bbox_area(bbox)
    page_area = max(width * height, 1.0)
    page_diag = max(math.hypot(width, height), 1.0)

    # --- Base features (52D) ---
    # Contained symbols
    symbol_counts: dict[str, float] = {label: 0.0 for label in SYMBOL_TYPES}
    symbol_areas: dict[str, float] = {label: 0.0 for label in SYMBOL_TYPES}
    contained_symbol_count = 0.0
    for symbol in context["symbols"]:
        if _bbox_contains(bbox, symbol["bbox"]):
            label = symbol["symbol_type"] if symbol["symbol_type"] in symbol_counts else "generic_symbol"
            contained_symbol_count += 1.0
            symbol_counts[label] += 1.0
            symbol_areas[label] += _bbox_area(symbol["bbox"]) / max(area, 1.0)

    # Boundary touch
    boundary_touch: dict[str, float] = {label: 0.0 for label in BOUNDARY_TYPES}
    for boundary in context["boundaries"]:
        if _bbox_intersects(bbox, boundary["bbox"]):
            label = boundary["semantic_type"]
            if label in boundary_touch:
                boundary_touch[label] += 1.0

    # Room label count
    room_label_count = sum(
        1.0 for text in context["texts"]
        if text["text_type"] == "room_label" and _bbox_contains(bbox, text["bbox"])
    )

    # Adjacency degree
    adjacency_degree = float(context["adjacency"].get(room["id"], 0))

    base_features = [
        ((x1 + x2) / 2.0) / max(width, 1.0),        # cx
        ((y1 + y2) / 2.0) / max(height, 1.0),        # cy
        w / max(width, 1.0),                          # width
        h / max(height, 1.0),                         # height
        area / page_area,                             # area
        math.log((w + 1.0) / (h + 1.0)),              # aspect
        adjacency_degree / 16.0,                      # adjacency_degree
        contained_symbol_count / 32.0,                # contained_symbol_count
        contained_symbol_count / max(area / 10000.0, 1.0),  # contained_symbol_density
        room_label_count / 4.0,                       # room_label_count
        *[symbol_counts[label] / 16.0 for label in SYMBOL_TYPES],
        *[symbol_areas[label] for label in SYMBOL_TYPES],
        *[boundary_touch[label] / 32.0 for label in BOUNDARY_TYPES],
    ]

    # --- Enhanced features (64D) ---
    margins = [x1 / max(width, 1.0), y1 / max(height, 1.0),
               (width - x2) / max(width, 1.0), (height - y2) / max(height, 1.0)]

    areas_all = sorted([_bbox_area(item["bbox"]) for item in context["rooms"]], reverse=True)
    area_rank = float(areas_all.index(area) if area in areas_all else len(areas_all)) / max(len(areas_all) - 1, 1)
    area_percentile = sum(1 for v in areas_all if v <= area) / max(len(areas_all), 1)

    same_row = 0.0
    same_col = 0.0
    adj_areas_list = []
    adj_widths = []
    adj_heights = []
    contained_room_count = 0.0
    inside_other_room_count = 0.0
    overlap_room_count = 0.0
    nearest_room_gap = page_diag

    for other in context["rooms"]:
        if other["id"] == room["id"]:
            continue
        other_bbox = other["bbox"]
        other_area = _bbox_area(other_bbox)
        if _bbox_contains(bbox, other_bbox):
            contained_room_count += 1.0
        if _bbox_contains(other_bbox, bbox):
            inside_other_room_count += 1.0
        if _bbox_intersects(bbox, other_bbox):
            inter_area = _intersection_area(bbox, other_bbox)
            if inter_area > 0.0:
                overlap_room_count += 1.0
        if _adjacent(bbox, other_bbox):
            adj_areas_list.append(other_area / page_area)
            adj_widths.append(max(0.0, other_bbox[2] - other_bbox[0]) / max(width, 1.0))
            adj_heights.append(max(0.0, other_bbox[3] - other_bbox[1]) / max(height, 1.0))
        if _overlap_length_fixed(y1, y2, other_bbox[1], other_bbox[3]) / max(min(h, other_bbox[3] - other_bbox[1]), 1.0) > 0.35:
            same_row += 1.0
        if _overlap_length_fixed(x1, x2, other_bbox[0], other_bbox[2]) / max(min(w, other_bbox[2] - other_bbox[0]), 1.0) > 0.35:
            same_col += 1.0
        nearest_room_gap = min(nearest_room_gap, _bbox_gap(bbox, other_bbox))

    # Symbol proximity
    symbol_center_counts: dict[str, float] = {label: 0.0 for label in SYMBOL_TYPES}
    symbol_overlap_counts: dict[str, float] = {label: 0.0 for label in SYMBOL_TYPES}
    symbol_near_counts: dict[str, float] = {label: 0.0 for label in SYMBOL_TYPES}
    nearest_symbol_gap = page_diag
    near_threshold = max(math.sqrt(max(area, 1.0)) * 0.25, 24.0)

    for symbol in context["symbols"]:
        label = symbol["symbol_type"] if symbol["symbol_type"] in symbol_center_counts else "generic_symbol"
        s_bbox = symbol["bbox"]
        if _bbox_center_inside(bbox, s_bbox):
            symbol_center_counts[label] += 1.0
        if _intersection_area(bbox, s_bbox) > 0.0:
            symbol_overlap_counts[label] += 1.0
        gap = _bbox_gap(bbox, s_bbox)
        nearest_symbol_gap = min(nearest_symbol_gap, gap)
        if gap <= near_threshold:
            symbol_near_counts[label] += 1.0

    # Text features
    rl_center_count = 0.0
    rl_overlap_count = 0.0
    dim_text_overlap_count = 0.0
    linked_room_texts = []
    linked_room_text_area = 0.0

    for text in context["texts"]:
        t_type = text["text_type"]
        t_bbox = text["bbox"]
        if t_type == "room_label" and _bbox_center_inside(bbox, t_bbox):
            rl_center_count += 1.0
            text_value = str(text.get("text") or "").strip()
            if text_value:
                linked_room_texts.append(text_value)
                linked_room_text_area += _bbox_area(t_bbox)
        if t_type == "room_label" and _intersection_area(bbox, t_bbox) > 0.0:
            rl_overlap_count += 1.0
        if t_type == "dimension_text" and _intersection_area(bbox, t_bbox) > 0.0:
            dim_text_overlap_count += 1.0

    # Boundary intersection
    bi_areas: dict[str, float] = {label: 0.0 for label in BOUNDARY_TYPES}
    bi_center_touch: dict[str, float] = {label: 0.0 for label in BOUNDARY_TYPES}
    boundary_band = max(6.0, min(w, h) * 0.03)
    expanded = [x1 - boundary_band, y1 - boundary_band, x2 + boundary_band, y2 + boundary_band]

    for boundary in context["boundaries"]:
        label = boundary["semantic_type"]
        if label not in bi_areas:
            continue
        bi_areas[label] += _intersection_area(bbox, boundary["bbox"]) / max(area, 1.0)
        if _bbox_intersects(expanded, boundary["bbox"]):
            bi_center_touch[label] += 1.0

    # Shape features
    shape = room.get("shape_features") or {}

    # Text lexicon matching
    text_match_scores = _room_text_match_vector(linked_room_texts)
    normalized_texts = [_normalize_room_text(text) for text in linked_room_texts]
    text_token_count = sum(len(t.split()) for t in normalized_texts)
    text_exact_unknown = sum(1 for t in normalized_texts if t in {"undefined", "ulko", "ulkotila", "autokatos"})

    enhanced = [
        x1 / max(width, 1.0),
        y1 / max(height, 1.0),
        x2 / max(width, 1.0),
        y2 / max(height, 1.0),
        min(margins),
        max(margins),
        float(x1 <= boundary_band),
        float(y1 <= boundary_band),
        float(width - x2 <= boundary_band),
        float(height - y2 <= boundary_band),
        area_rank,
        area_percentile,
        len(context["rooms"]) / 128.0,
        same_row / 32.0,
        same_col / 32.0,
        _mean(adj_areas_list),
        max(adj_areas_list) if adj_areas_list else 0.0,
        _mean(adj_widths),
        _mean(adj_heights),
        contained_room_count / 8.0,
        inside_other_room_count / 8.0,
        overlap_room_count / 16.0,
        nearest_room_gap / page_diag,
        nearest_symbol_gap / page_diag,
        rl_center_count / 4.0,
        rl_overlap_count / 4.0,
        dim_text_overlap_count / 16.0,
        *[symbol_center_counts[label] / 16.0 for label in SYMBOL_TYPES],
        *[symbol_overlap_counts[label] / 16.0 for label in SYMBOL_TYPES],
        *[symbol_near_counts[label] / 16.0 for label in SYMBOL_TYPES],
        *[bi_areas[label] for label in BOUNDARY_TYPES],
        *[bi_center_touch[label] / 32.0 for label in BOUNDARY_TYPES],
        float(shape.get("point_count") or 0.0) / 32.0,
        float(shape.get("polygon_area") or 0.0) / page_area,
        float(shape.get("polygon_perimeter") or 0.0) / page_diag,
        float(shape.get("bbox_fill_ratio") or 0.0),
        float(shape.get("compactness") or 0.0),
        len(linked_room_texts) / 4.0,
        text_token_count / 16.0,
        linked_room_text_area / max(area, 1.0),
        text_exact_unknown / 4.0,
        *[text_match_scores[label] / 4.0 for label in ROOM_TEXT_LABELS],
    ]

    return base_features + enhanced


def _build_context_from_candidates(candidates: list[RoutedCandidate]) -> dict[str, Any]:
    """Build a page-level context from all routed candidates.

    This extracts rooms, symbols, texts, and boundaries from candidate payloads.
    Used when running the expert on candidates without full page context.
    """
    rooms = []
    symbols = []
    texts = []
    boundaries = []
    page_w = 2000.0
    page_h = 2000.0

    for c in candidates:
        meta = c.payload.get("_page_metadata") or {}
        if meta.get("width"):
            page_w = float(meta["width"])
        if meta.get("height"):
            page_h = float(meta["height"])

        if c.family == "space" and c.bbox:
            rooms.append({
                "id": c.candidate_id,
                "room_type": "room",
                "bbox": [float(v) for v in c.bbox],
                "shape_features": c.payload.get("shape_features", {}),
            })
        elif c.family == "symbol" and c.bbox:
            symbols.append({
                "id": c.candidate_id,
                "symbol_type": c.payload.get("symbol_type", "generic_symbol"),
                "bbox": [float(v) for v in c.bbox],
            })
        elif c.family == "text" and c.bbox:
            texts.append({
                "id": c.candidate_id,
                "text_type": c.payload.get("text_type", "note_text"),
                "text": c.payload.get("raw_text", c.payload.get("text", "")),
                "bbox": [float(v) for v in c.bbox],
            })

    # Compute adjacency
    degrees: dict[str, int] = {r["id"]: 0 for r in rooms}
    for i, left in enumerate(rooms):
        for right in rooms[i + 1:]:
            if _adjacent(left["bbox"], right["bbox"]):
                degrees[left["id"]] += 1
                degrees[right["id"]] += 1

    return {
        "width": page_w,
        "height": page_h,
        "rooms": rooms,
        "symbols": symbols,
        "texts": texts,
        "boundaries": boundaries,
        "adjacency": degrees,
    }


def _context_from_payload(candidate: RoutedCandidate, fallback: dict[str, Any]) -> dict[str, Any]:
    context = candidate.payload.get("page_context")
    if not isinstance(context, dict):
        return fallback
    return {
        "width": float(context.get("width") or fallback.get("width") or 2000.0),
        "height": float(context.get("height") or fallback.get("height") or 2000.0),
        "rooms": list(context.get("rooms") or []),
        "symbols": list(context.get("symbols") or []),
        "texts": list(context.get("texts") or []),
        "boundaries": list(context.get("boundaries") or []),
        "adjacency": dict(context.get("adjacency") or {}),
    }


def _predict_hierarchical(
    gate_model: Any,
    typed_model: Any,
    typed_encoder: Any,
    room_threshold: float,
    X: np.ndarray,
) -> tuple[list[str], list[float], list[dict[str, float]]]:
    gate_probs = gate_model.predict_proba(X)
    typed_indices = typed_model.predict(X)
    typed_probs = typed_model.predict_proba(X)
    typed_labels = typed_encoder.inverse_transform(typed_indices)

    labels: list[str] = []
    confidences: list[float] = []
    all_probs: list[dict[str, float]] = []
    typed_classes = [str(item) for item in typed_encoder.classes_]
    for gate_prob, typed_label, typed_prob in zip(gate_probs, typed_labels, typed_probs):
        room_probability = float(gate_prob[1])
        if room_probability >= room_threshold:
            label = "room"
            confidence = room_probability
        else:
            label = str(typed_label)
            confidence = (1.0 - room_probability) * float(max(typed_prob))
        labels.append(label)
        confidences.append(float(confidence))
        probs = {"room": round(room_probability, 4)}
        probs.update({typed_classes[i]: round(float(value), 4) for i, value in enumerate(typed_prob)})
        all_probs.append(probs)
    return labels, confidences, all_probs


class RoomSpaceExpert(PassthroughExpert):
    """Room/Space expert using trained sklearn context model.

    Falls back to passthrough ("room") if model checkpoint is not found.
    """

    def __init__(self) -> None:
        super().__init__(name="room_space", family="space")
        self.default_label = "room"
        self._model: Any = None
        self._label_encoder: Any = None
        self._gate_model: Any = None
        self._typed_model: Any = None
        self._typed_label_encoder: Any = None
        self._room_threshold: float = 0.5
        self._model_type: str = "context"
        self._feature_set: str = "enhanced"
        self._load_model()

    def _load_model(self) -> None:
        model_path = _MODEL_DIR / "model.joblib"
        if not model_path.exists():
            return
        try:
            import joblib
            data = joblib.load(str(model_path))
            if data.get("gate_model") is not None and data.get("typed_model") is not None:
                self._model_type = "hierarchical"
                self._gate_model = data.get("gate_model")
                self._typed_model = data.get("typed_model")
                self._typed_label_encoder = data.get("typed_label_encoder")
                self._room_threshold = float(data.get("room_threshold", 0.5))
                self._model = self._gate_model
            else:
                self._model_type = "context"
                self._model = data.get("model")
                self._label_encoder = data.get("label_encoder")
                self._feature_set = data.get("feature_set", "enhanced")
        except Exception:
            self._model = None

    def predict(self, candidates: list[RoutedCandidate]) -> list[ExpertPrediction]:
        if self._model is None:
            return super().predict(candidates)

        # Build a fallback context from the supplied batch. Prefer per-record
        # page_context payloads because end-to-end batches can contain many
        # records; mixing all rooms into one global context corrupts adjacency,
        # area rank, and text/symbol containment features.
        fallback_context = _build_context_from_candidates(candidates)

        # Extract features for room candidates only
        predictions: list[ExpertPrediction] = []
        feature_rows: list[tuple[RoutedCandidate, list[float]]] = []

        for candidate in candidates:
            if candidate.family != "space":
                continue

            room = {
                "id": candidate.candidate_id,
                "room_type": "room",
                "bbox": [float(v) for v in candidate.bbox] if candidate.bbox else [0, 0, 0, 0],
                "shape_features": candidate.payload.get("shape_features", {}),
            }

            context = _context_from_payload(candidate, fallback_context)
            feats = _extract_room_features(room, context)
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

            if self._model_type == "hierarchical":
                labels, confidences, all_probs = _predict_hierarchical(
                    self._gate_model,
                    self._typed_model,
                    self._typed_label_encoder,
                    self._room_threshold,
                    X,
                )
            else:
                y_pred = self._model.predict(X)
                y_pred_proba = self._model.predict_proba(X) if hasattr(self._model, "predict_proba") else None
                labels = [
                    str(self._label_encoder.inverse_transform([int(pred_idx)])[0]) if self._label_encoder else str(pred_idx)
                    for pred_idx in y_pred
                ]
                confidences = [
                    float(max(proba)) if proba is not None else feature_rows[index][0].confidence
                    for index, proba in enumerate(y_pred_proba if y_pred_proba is not None else [None] * len(feature_rows))
                ]
                all_probs = [
                    {
                        str(self._label_encoder.classes_[i]): round(float(p), 4)
                        for i, p in enumerate(proba)
                    } if self._label_encoder and proba is not None else {}
                    for proba in (y_pred_proba if y_pred_proba is not None else [None] * len(feature_rows))
                ]

            for (candidate, _feats), label, confidence, probs in zip(feature_rows, labels, confidences, all_probs):
                predictions.append(
                    ExpertPrediction(
                        candidate_id=candidate.candidate_id,
                        expert=self.name,
                        family=self.family,
                        label=label,
                        confidence=confidence,
                        bbox=candidate.bbox,
                        source=f"{self.name}_sklearn_context",
                        metadata={
                            "candidate_type": candidate.candidate_type,
                            "model_type": self._model_type,
                            "all_probs": probs,
                        },
                    )
                )

        return predictions
