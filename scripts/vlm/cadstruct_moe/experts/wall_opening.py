"""Wall/opening expert wrapper.

Loads the production graph-node crop GNN checkpoint when available and predicts
hard_wall/door/window labels. Falls back to the deterministic passthrough
expert only when the checkpoint or runtime dependencies are unavailable.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

from ..schema import ExpertPrediction, RoutedCandidate
from .base import PassthroughExpert

ROOT = Path(__file__).resolve().parents[4]
VLM_DIR = ROOT / "scripts" / "vlm"
GNN_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "cadstruct_graph_node_crop_gnn_h1024_c32_ms3_l2_floor_target_doorw150_e120"
    / "model_best.pt"
)
ARBITRATION_CHECKPOINT = ROOT / "checkpoints" / "boundary_label_arbitration_v1" / "model.joblib"
ARBITRATION_ORIENTATIONS = ["horizontal", "vertical", "diagonal", "rectangular", "unknown"]


class WallOpeningExpert(PassthroughExpert):
    def __init__(self) -> None:
        super().__init__(
            name="wall_opening",
            family="boundary",
            label_space=("hard_wall", "partition_wall", "door", "window", "opening", "curtain_wall"),
            checkpoint_hint=str(GNN_CHECKPOINT),
        )
        self.default_label = "hard_wall"
        self._model: Any = None
        self._checkpoint: dict[str, Any] = {}
        self._feature_spec: Any = None
        self._labels: list[str] = []
        self._device: Any = None
        self._arbiter: Any = None
        self._arbiter_labels: list[str] = []
        self._load_error: str | None = None
        self._load_gnn_model()
        self._load_arbitration_model()

    def _load_gnn_model(self) -> None:
        if not GNN_CHECKPOINT.exists():
            self._load_error = f"missing checkpoint: {GNN_CHECKPOINT}"
            return

        try:
            if str(VLM_DIR) not in sys.path:
                sys.path.insert(0, str(VLM_DIR))

            import torch
            from graph_node_model import FeatureSpec
            from train_graph_node_crop_gnn_classifier import load_checkpoint

            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model, self._checkpoint = load_checkpoint(GNN_CHECKPOINT, self._device)
            self._feature_spec = FeatureSpec(**self._checkpoint["feature_spec"])
            self._labels = list(self._feature_spec.labels)
        except Exception as exc:  # pragma: no cover - depends on optional torch/checkpoint runtime
            self._model = None
            self._load_error = f"{type(exc).__name__}: {exc}"

    def _load_arbitration_model(self) -> None:
        if not ARBITRATION_CHECKPOINT.exists():
            return
        try:
            import joblib
            data = joblib.load(str(ARBITRATION_CHECKPOINT))
            self._arbiter = data.get("model")
            self._arbiter_labels = [str(item) for item in data.get("labels") or []]
        except Exception:
            self._arbiter = None
            self._arbiter_labels = []

    def is_loaded(self) -> bool:
        return self._model is not None and self._feature_spec is not None and bool(self._labels)

    def predict(self, candidates: list[RoutedCandidate]) -> list[ExpertPrediction]:
        if self._model is None or self._feature_spec is None or not self._labels:
            predictions = super().predict(candidates)
            if self._load_error:
                predictions = [
                    _with_metadata(prediction, {"fallback": True, "load_error": self._load_error})
                    for prediction in predictions
                ]
            return predictions

        try:
            probs = self._predict_probabilities(candidates)
        except Exception as exc:  # pragma: no cover - defensive fallback for malformed upstream payloads
            return [
                ExpertPrediction(
                    candidate_id=candidate.candidate_id,
                    expert=self.name,
                    family=self.family,
                    label=self.default_label,
                    confidence=candidate.confidence,
                    bbox=candidate.bbox,
                    source=f"{self.name}_gnn_fallback",
                    metadata={
                        "candidate_type": candidate.candidate_type,
                        "fallback": True,
                        "inference_error": f"{type(exc).__name__}: {exc}",
                    },
                )
                for candidate in candidates
            ]

        arbitration = self._predict_arbitration(candidates)
        predictions: list[ExpertPrediction] = []
        for index, (candidate, prob) in enumerate(zip(candidates, probs, strict=True)):
            pred_id = int(prob.argmax())
            gnn_label = self._labels[pred_id]
            gnn_confidence = float(prob[pred_id])
            arb = arbitration[index] if arbitration and index < len(arbitration) else None
            label = str(arb["label"]) if arb else gnn_label
            confidence = float(arb["confidence"]) if arb else gnn_confidence
            predictions.append(
                ExpertPrediction(
                    candidate_id=candidate.candidate_id,
                    expert=self.name,
                    family=self.family,
                    label=label,
                    confidence=confidence,
                    bbox=candidate.bbox,
                    source=f"{self.name}_label_arbitration_v1" if arb else f"{self.name}_crop_gnn",
                    metadata={
                        "candidate_type": candidate.candidate_type,
                        "checkpoint": str(GNN_CHECKPOINT),
                        "arbitration_checkpoint": str(ARBITRATION_CHECKPOINT) if arb else None,
                        "gnn_label": gnn_label,
                        "gnn_confidence": round(gnn_confidence, 6),
                        "all_probs": {
                            self._labels[index]: round(float(value), 6)
                            for index, value in enumerate(prob.tolist())
                        },
                        "arbitration_probs": arb["all_probs"] if arb else {},
                    },
                )
            )
        return predictions

    def _predict_probabilities(self, candidates: list[RoutedCandidate]) -> Any:
        if str(VLM_DIR) not in sys.path:
            sys.path.insert(0, str(VLM_DIR))

        from train_graph_node_crop_gnn_classifier import build_split, predict_all

        label_to_id = {label: index for index, label in enumerate(self._labels)}
        config = self._checkpoint.get("model_config") or {}
        sample = _candidates_to_graph_sample(candidates, self._labels[0])
        split = build_split(
            [sample],
            self._feature_spec,
            label_to_id,
            int(config.get("crop_size", 32)),
            [float(value) for value in config.get("crop_pad_scales", [0.15, 0.35, 0.8])],
            float(config.get("min_pad", 8.0)),
            False,
        )
        return predict_all(self._model, split, self._labels, batch_samples=1, device=self._device)

    def _predict_arbitration(self, candidates: list[RoutedCandidate]) -> list[dict[str, Any]] | None:
        if self._arbiter is None:
            return None
        rows = [_boundary_arbitration_features(candidate) for candidate in candidates]
        if not rows:
            return None
        probs = self._arbiter.predict_proba(rows)
        classes = [str(item) for item in getattr(self._arbiter, "classes_", self._arbiter_labels)]
        result = []
        for prob in probs:
            best_idx = max(range(len(classes)), key=lambda idx: float(prob[idx]))
            result.append(
                {
                    "label": classes[best_idx],
                    "confidence": float(prob[best_idx]),
                    "all_probs": {classes[idx]: round(float(prob[idx]), 6) for idx in range(len(classes))},
                }
            )
        return result


def _with_metadata(prediction: ExpertPrediction, extra: dict[str, Any]) -> ExpertPrediction:
    metadata = dict(prediction.metadata)
    metadata.update(extra)
    return ExpertPrediction(
        candidate_id=prediction.candidate_id,
        expert=prediction.expert,
        family=prediction.family,
        label=prediction.label,
        confidence=prediction.confidence,
        bbox=prediction.bbox,
        geometry=prediction.geometry,
        relations=prediction.relations,
        source=prediction.source,
        metadata=metadata,
    )


def _boundary_arbitration_features(candidate: RoutedCandidate) -> list[float]:
    features = _candidate_features(candidate)
    bbox = _normalize_bbox(features.get("bbox") or candidate.bbox or candidate.payload.get("bbox"))
    x1, y1, x2, y2 = bbox
    width = max(x2 - x1, 1e-6)
    height = max(y2 - y1, 1e-6)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    page_bbox = _candidate_page_bounds(candidate)
    page_w = max(page_bbox[2] - page_bbox[0], 1e-6)
    page_h = max(page_bbox[3] - page_bbox[1], 1e-6)
    length = float(features.get("length") or max(width, height))
    area = width * height
    group_count = float(candidate.payload.get("bbox_group_count") or 1.0)
    orientation = str(features.get("orientation") or "unknown")
    orient = [1.0 if orientation == item else 0.0 for item in ARBITRATION_ORIENTATIONS]
    return [
        x1,
        y1,
        x2,
        y2,
        width,
        height,
        area,
        math.log((width + 1.0) / (height + 1.0)),
        length,
        length / max(max(page_w, page_h), 1e-6),
        width / page_w,
        height / page_h,
        area / max(page_w * page_h, 1e-6),
        (cx - page_bbox[0]) / page_w,
        (cy - page_bbox[1]) / page_h,
        group_count,
        float(group_count >= 2.0),
        *orient,
    ]


def _candidate_page_bounds(candidate: RoutedCandidate) -> list[float]:
    payload = candidate.payload
    page_bbox = payload.get("page_bbox")
    if isinstance(page_bbox, list) and len(page_bbox) >= 4:
        return _normalize_bbox(page_bbox)
    meta = payload.get("_page_metadata") or {}
    width = float(meta.get("width") or 0.0)
    height = float(meta.get("height") or 0.0)
    if width > 0.0 and height > 0.0:
        return [0.0, 0.0, width, height]
    return [0.0, 0.0, 2000.0, 2000.0]


def _candidates_to_graph_sample(candidates: list[RoutedCandidate], dummy_label: str) -> dict[str, Any]:
    nodes = []
    id_to_local: dict[str, int] = {}
    for index, candidate in enumerate(candidates):
        node_id = index
        id_to_local[candidate.candidate_id] = node_id
        features = _candidate_features(candidate)
        nodes.append(
            {
                "id": node_id,
                "label": dummy_label,
                "features": features,
            }
        )

    edges = []
    for candidate in candidates:
        for edge in candidate.payload.get("edges") or []:
            if not isinstance(edge, dict):
                continue
            source = id_to_local.get(str(edge.get("source")))
            target = id_to_local.get(str(edge.get("target")))
            if source is None or target is None:
                continue
            edges.append({"source": source, "target": target, "relation": str(edge.get("relation") or "unknown")})

    image = None
    source_dataset = "unknown"
    for candidate in candidates:
        image = candidate.payload.get("image") or candidate.payload.get("raster_path") or image
        source_dataset = str(candidate.payload.get("source_dataset") or source_dataset)

    return {"image": image, "source_dataset": source_dataset, "nodes": nodes, "edges": edges}


def _candidate_features(candidate: RoutedCandidate) -> dict[str, Any]:
    payload_features = candidate.payload.get("features")
    features = dict(payload_features) if isinstance(payload_features, dict) else {}

    bbox = _normalize_bbox(features.get("bbox") or candidate.bbox or candidate.payload.get("bbox"))
    centroid = features.get("centroid")
    if not isinstance(centroid, list) or len(centroid) < 2:
        centroid = [(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5]

    width = max(0.0, bbox[2] - bbox[0])
    height = max(0.0, bbox[3] - bbox[1])
    angle = float(features.get("angle_degrees", candidate.payload.get("angle_degrees", 0.0)) or 0.0)
    features.update(
        {
            "primitive_type": str(features.get("primitive_type") or candidate.payload.get("primitive_type") or "bbox"),
            "bbox": bbox,
            "centroid": [float(centroid[0]), float(centroid[1])],
            "length": float(features.get("length", candidate.payload.get("length", max(width, height))) or 0.0),
            "angle_degrees": angle,
            "orientation": str(features.get("orientation") or candidate.payload.get("orientation") or _orientation(width, height, angle)),
        }
    )
    return features


def _normalize_bbox(value: Any) -> list[float]:
    if isinstance(value, list) and len(value) >= 4:
        try:
            x1, y1, x2, y2 = [float(item or 0.0) for item in value[:4]]
            return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        except (TypeError, ValueError):
            pass
    return [0.0, 0.0, 0.0, 0.0]


def _orientation(width: float, height: float, angle_degrees: float) -> str:
    if width > height * 2.0:
        return "horizontal"
    if height > width * 2.0:
        return "vertical"
    if math.isfinite(angle_degrees) and abs(angle_degrees) % 90.0 > 10.0:
        return "diagonal"
    return "rectangular"
