"""Cross-expert fusion helpers for CadStruct MoE outputs."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .schema import ExpertPrediction, FusionResult


BOUNDARY_LABELS = {"hard_wall", "partition_wall", "door", "window", "opening", "curtain_wall"}
OPENING_LABELS = {"door", "window", "opening"}
ROOM_LABELS = {
    "room",
    "bedroom",
    "living_room",
    "kitchen",
    "bathroom",
    "toilet",
    "corridor",
    "balcony",
    "closet",
    "office",
    "storage",
    "unknown_room",
}


def fuse_predictions(predictions: list[ExpertPrediction]) -> FusionResult:
    warnings: list[str] = []
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    by_id: dict[str, ExpertPrediction] = {}

    for prediction in predictions:
        if prediction.candidate_id in by_id:
            warnings.append(f"duplicate_candidate_id:{prediction.candidate_id}")
        by_id[prediction.candidate_id] = prediction
        geometry = dict(prediction.geometry or {})
        bbox = prediction.bbox or []
        geometry.setdefault("bbox", bbox)
        nodes.append(
            {
                "id": prediction.candidate_id,
                "semantic_type": prediction.label,
                "expert": prediction.expert,
                "family": prediction.family,
                "confidence": prediction.confidence,
                "source_expert": prediction.source,
                "geometry": geometry,
                "audit_trace": {
                    "origin": "cadstruct_moe_fusion",
                    "stage": "rule_fusion",
                    "family": prediction.family,
                },
                "metadata": dict(prediction.metadata or {}),
            }
        )
        edges.extend(prediction.relations)

    warnings.extend(check_architectural_constraints(predictions, edges))
    return FusionResult(
        predictions=predictions,
        scene_graph={"nodes": nodes, "edges": edges},
        warnings=sorted(set(warnings)),
        metadata={"fusion_policy": "rule_constraints_v0"},
    )


def check_architectural_constraints(predictions: list[ExpertPrediction], edges: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    labels = {item.candidate_id: item.label for item in predictions}
    relation_targets: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        relation = str(edge.get("relation") or "")
        if source and target and relation:
            relation_targets[source].add(relation)
            relation_targets[target].add(relation)

    for item in predictions:
        relations = relation_targets.get(item.candidate_id, set())
        if item.label in OPENING_LABELS and not (relations & {"attached_to", "interrupted_by", "touches"}):
            warnings.append(f"opening_without_wall_relation:{item.candidate_id}")
        if item.label in ROOM_LABELS and not (relations & {"bounded_by", "bounds", "contains", "inside"}):
            warnings.append(f"room_without_boundary_relation:{item.candidate_id}")
        if item.family == "text" and item.label == "dimension_text" and not (relations & {"labels", "dimension_of", "attached_to"}):
            warnings.append(f"dimension_text_without_link:{item.candidate_id}")

    for edge in edges:
        source_label = labels.get(str(edge.get("source") or ""))
        target_label = labels.get(str(edge.get("target") or ""))
        relation = str(edge.get("relation") or "")
        if relation == "interrupted_by" and source_label not in BOUNDARY_LABELS and target_label not in OPENING_LABELS:
            warnings.append("invalid_interrupted_by_relation")
    return warnings
