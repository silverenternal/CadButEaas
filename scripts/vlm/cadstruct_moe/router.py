"""Auditable deterministic router for CadStruct MoE candidates."""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .schema import RoutedCandidate, family_to_expert, load_ontology

# Fair features: only geometry + page context (no type code leakage)
FAIR_FEATURE_NAMES = [
    "bbox_area", "bbox_aspect", "bbox_center_x", "bbox_center_y",
    "bbox_width_norm", "bbox_height_norm", "bbox_area_norm",
    "page_aspect", "page_area",
    "n_candidates_same_family",
]


BOUNDARY_HINTS = {"wall", "door", "window", "opening", "line", "polyline", "boundary"}
TEXT_HINTS = {"text", "ocr", "dimension", "leader", "callout", "note", "label"}
SYMBOL_HINTS = {"symbol", "fixture", "furniture", "stair", "column", "equipment", "icon"}
SHEET_HINTS = {"title", "table", "schedule", "legend", "stamp", "sheet"}
SPACE_HINTS = {"room", "space", "area", "region", "polygon", "closed"}


class DeterministicRouter:
    """Route candidates by geometry/type hints before training a learned router."""

    def __init__(self, ontology_path: str | Path = "configs/vlm/cadstruct_ontology.json") -> None:
        self.ontology = load_ontology(ontology_path)
        self.family_experts = family_to_expert(self.ontology)

    def route_record(self, record: dict[str, Any]) -> list[RoutedCandidate]:
        routed: list[RoutedCandidate] = []
        graph = ((record.get("request_hints") or {}).get("primitive_graph") or {})

        for node in graph.get("nodes") or []:
            routed.append(self.route_primitive_node(node))

        for index, candidate in enumerate(iter_candidates(record, "text_candidates")):
            routed.append(self.route_external_candidate(candidate, index, default_family="text"))

        for index, candidate in enumerate(iter_candidates(record, "symbol_candidates")):
            routed.append(self.route_external_candidate(candidate, index, default_family="symbol"))

        for index, candidate in enumerate(iter_candidates(record, "semantic_regions")):
            routed.append(self.route_external_candidate(candidate, index, default_family="space"))

        for index, candidate in enumerate(iter_candidates(record, "layout_regions")):
            routed.append(self.route_external_candidate(candidate, index, default_family="sheet"))

        return routed

    def route_primitive_node(self, node: dict[str, Any]) -> RoutedCandidate:
        node_id = str(node.get("id") or node.get("target_id") or f"primitive_{id(node)}")
        node_type = normalize_hint(node.get("type") or node.get("semantic_type") or node.get("label") or "primitive")
        bbox = normalize_bbox(node.get("bbox"))
        family, matched_hint, abstain = infer_family_with_trace(node_type, bbox=None, default_family="boundary")
        expert = self.family_experts.get(family, f"{family}_expert")
        routing_confidence = 1.0 if not abstain else 0.3
        return RoutedCandidate(
            candidate_id=node_id,
            expert=expert,
            family=family,
            candidate_type=node_type,
            confidence=routing_confidence,
            bbox=bbox,
            payload={"node": node},
            route_trace={
                "matched_hint": matched_hint,
                "routing_confidence": routing_confidence,
                "abstain": abstain,
                "routing_method": "primitive_hint_match",
            },
        )

    def route_external_candidate(
        self,
        candidate: dict[str, Any],
        index: int,
        default_family: str,
    ) -> RoutedCandidate:
        hint = normalize_hint(
            candidate.get("type")
            or candidate.get("semantic_type")
            or candidate.get("symbol_type")
            or candidate.get("layout_type")
            or candidate.get("text_type")
            or default_family
        )
        bbox = normalize_bbox(candidate.get("bbox"))
        family, matched_hint, abstain = infer_family_with_trace(hint, bbox=bbox, default_family=default_family)
        expert = self.family_experts.get(family, f"{family}_expert")
        input_confidence = safe_float(candidate.get("confidence"), 1.0)
        routing_confidence = input_confidence if not abstain else 0.3
        candidate_id = str(candidate.get("id") or candidate.get("target_id") or f"{family}_{index}")
        return RoutedCandidate(
            candidate_id=candidate_id,
            expert=expert,
            family=family,
            candidate_type=hint,
            confidence=routing_confidence,
            bbox=bbox,
            payload={"candidate": candidate},
            route_trace={
                "matched_hint": matched_hint,
                "routing_confidence": routing_confidence,
                "abstain": abstain,
                "routing_method": "external_hint_match",
                "input_confidence": input_confidence,
            },
        )


def route_record(record: dict[str, Any], ontology_path: str | Path = "configs/vlm/cadstruct_ontology.json") -> list[dict[str, Any]]:
    return [item.to_dict() for item in DeterministicRouter(ontology_path).route_record(record)]


class LearnedRouter:
    """Route candidates using a trained sklearn model + geometry features."""

    def __init__(
        self,
        model_path: str | Path = "checkpoints/moe_router_v2/model_v2_fair.joblib",
        ontology_path: str | Path = "configs/vlm/cadstruct_ontology.json",
    ) -> None:
        self.ontology = load_ontology(ontology_path)
        self.family_experts = family_to_expert(self.ontology)
        self._model = None
        self._scaler = None
        self._label_encoder = None
        self._feature_names = []
        self._load_model(model_path)

    def _load_model(self, model_path: str | Path) -> None:
        path = Path(model_path)
        if not path.exists():
            return
        try:
            import joblib
            data = joblib.load(str(path))
            self._model = data.get("model")
            self._scaler = data.get("scaler")
            self._label_encoder = data.get("label_encoder")
            self._feature_names = data.get("feature_names", FAIR_FEATURE_NAMES)
        except Exception:
            self._model = None

    def _extract_features(self, candidate: dict, page_meta: dict, all_candidates: list) -> list[float] | None:
        if self._model is None:
            return None
        pw = page_meta.get("width") or 1000
        ph = page_meta.get("height") or 1000
        page_area = pw * ph
        page_aspect = pw / max(1, ph)

        bbox = candidate.get("bbox")
        if bbox is None or len(bbox) != 4:
            return None
        x1, y1, x2, y2 = bbox
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        area = w * h
        aspect = w / max(1, h)

        fam = candidate.get("_family", "")
        n_same = sum(1 for c in all_candidates if c.get("_family") == fam)

        return [
            float(area),
            float(aspect),
            float((x1 + x2) / 2 / max(1, pw)),
            float((y1 + y2) / 2 / max(1, ph)),
            float(w / max(1, pw)),
            float(h / max(1, ph)),
            float(area / max(1, page_area)),
            float(page_aspect),
            float(page_area),
            float(n_same),
        ]

    def route_record(self, record: dict[str, Any]) -> list[RoutedCandidate]:
        if self._model is None:
            # Fallback to deterministic
            return DeterministicRouter().route_record(record)

        # First pass: collect all candidates for sibling count feature
        det = DeterministicRouter()
        pre_routed: list[RoutedCandidate] = []
        all_candidates_with_family: list[dict] = []

        # Build pre-routed candidates and track family for feature extraction
        graph = ((record.get("request_hints") or {}).get("primitive_graph") or {})
        for node in graph.get("nodes") or []:
            pre_routed.append(det.route_primitive_node(node))

        for index, candidate in enumerate(iter_candidates(record, "text_candidates")):
            routed = det.route_external_candidate(candidate, index, default_family="text")
            pre_routed.append(routed)
            all_candidates_with_family.append({
                "bbox": list(routed.bbox) if routed.bbox else None,
                "_family": "text_dimension",
            })

        for index, candidate in enumerate(iter_candidates(record, "symbol_candidates")):
            routed = det.route_external_candidate(candidate, index, default_family="symbol")
            pre_routed.append(routed)
            all_candidates_with_family.append({
                "bbox": list(routed.bbox) if routed.bbox else None,
                "_family": "symbol_fixture",
            })

        for index, candidate in enumerate(iter_candidates(record, "semantic_regions")):
            routed = det.route_external_candidate(candidate, index, default_family="space")
            pre_routed.append(routed)
            all_candidates_with_family.append({
                "bbox": list(routed.bbox) if routed.bbox else None,
                "_family": "room_space",
            })

        for index, candidate in enumerate(iter_candidates(record, "layout_regions")):
            routed = det.route_external_candidate(candidate, index, default_family="sheet")
            pre_routed.append(routed)
            all_candidates_with_family.append({
                "bbox": list(routed.bbox) if routed.bbox else None,
                "_family": "sheet_layout",
            })

        # Extract features for each candidate
        page_meta = (record.get("metadata") or {})
        feature_rows = []
        for rc in pre_routed:
            candidate_dict = {"bbox": list(rc.bbox) if rc.bbox else None, "_family": rc.expert}
            feats = self._extract_features(candidate_dict, page_meta, all_candidates_with_family)
            if feats is not None:
                feature_rows.append((rc, feats))

        if not feature_rows:
            return pre_routed

        X = [[f for f in feats] for _, feats in feature_rows]
        import numpy as np
        X = np.array(X, dtype=np.float64)
        if self._scaler is not None:
            X = self._scaler.transform(X)

        y_pred = self._model.predict(X)
        y_pred_proba = self._model.predict_proba(X)

        # Map predictions back to families
        idx_to_family = {}
        if self._label_encoder is not None:
            for idx, cls_name in enumerate(self._label_encoder.classes_):
                idx_to_family[idx] = cls_name

        routed: list[RoutedCandidate] = []
        for (rc, _feats), pred_idx, proba in zip(feature_rows, y_pred, y_pred_proba):
            pred_family = idx_to_family.get(int(pred_idx), rc.family)
            pred_expert = self.family_experts.get(pred_family, f"{pred_family}_expert")
            confidence = float(max(proba))
            is_correct = pred_family == rc.family

            routed.append(RoutedCandidate(
                candidate_id=rc.candidate_id,
                expert=pred_expert,
                family=pred_family,
                candidate_type=rc.candidate_type,
                confidence=confidence,
                bbox=rc.bbox,
                payload=rc.payload,
                route_trace={
                    "matched_hint": rc.route_trace.get("matched_hint"),
                    "routing_confidence": confidence,
                    "abstain": False,
                    "routing_method": "learned_fair",
                    "predicted_family": pred_family,
                    "deterministic_family": rc.family,
                    "family_changed": not is_correct,
                },
            ))

        return routed


def route_record_learned(record: dict[str, Any], model_path: str | Path = "checkpoints/moe_router_v2/model_v2_fair.joblib", ontology_path: str | Path = "configs/vlm/cadstruct_ontology.json") -> list[dict[str, Any]]:
    return [item.to_dict() for item in LearnedRouter(model_path, ontology_path).route_record(record)]


def iter_candidates(record: dict[str, Any], key: str) -> Iterable[dict[str, Any]]:
    hints = record.get("request_hints") or {}
    value = record.get(key)
    if value is None:
        value = hints.get(key)
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                yield item


def infer_family_with_trace(hint: str, bbox: list[float] | None, default_family: str) -> tuple[str, str | None, bool]:
    """Infer family with routing trace: returns (family, matched_hint, abstain)."""
    tokens = set(hint.replace("-", "_").split("_"))

    for hint_set, family in [
        (TEXT_HINTS, "text"),
        (SHEET_HINTS, "sheet"),
        (SYMBOL_HINTS, "symbol"),
        (SPACE_HINTS, "space"),
        (BOUNDARY_HINTS, "boundary"),
    ]:
        matched = tokens & hint_set
        if matched:
            return family, sorted(matched)[0], False

    # Fallback: large region → space
    if bbox and is_large_region(bbox):
        return "space", "large_region_bbox", False

    # No hint matched — abstain
    return default_family, None, True


def infer_family(hint: str, bbox: list[float] | None, default_family: str) -> str:
    """Legacy function — delegates to infer_family_with_trace for consistency."""
    family, _, _ = infer_family_with_trace(hint, bbox, default_family)
    return family


def is_large_region(bbox: list[float]) -> bool:
    if len(bbox) != 4:
        return False
    width = max(0.0, bbox[2] - bbox[0])
    height = max(0.0, bbox[3] - bbox[1])
    return width * height >= 0.08


def normalize_hint(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    return text.replace(" ", "_")


def normalize_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def safe_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return min(1.0, max(0.0, number))
