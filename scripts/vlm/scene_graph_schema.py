"""Auditable scene-graph schema and contract checks for MoE fusion outputs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_ONTOLOGY_PATH = Path("configs/vlm/cadstruct_ontology.json")
SCENE_GRAPH_SCHEMA_VERSION = "cadstruct-moe-scene-graph-v1"
DEFAULT_RELATION_TYPES = {
    "touches",
    "contains",
    "contained_in",
    "bounds",
    "interrupted_by",
    "attached_to",
    "inside",
    "labels",
    "dimension_of",
    "adjacent_to",
}


def load_ontology(path: str | Path = DEFAULT_ONTOLOGY_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ontology_label_to_family(ontology_path: str | Path = DEFAULT_ONTOLOGY_PATH) -> dict[str, str]:
    ontology = load_ontology(ontology_path)
    mapping: dict[str, str] = {}
    for family, cfg in (ontology.get("families") or {}).items():
        labels = cfg.get("labels") or []
        for label in labels:
            mapping[str(label)] = str(family)
    return mapping


def ontology_families_and_relations(path: str | Path = DEFAULT_ONTOLOGY_PATH) -> tuple[set[str], set[str]]:
    ontology = load_ontology(path)
    families = set(str(key) for key in (ontology.get("families") or {}).keys())
    relations = set(str(x) for x in ontology.get("relation_types") or DEFAULT_RELATION_TYPES)
    return families, relations


def stable_manifest_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SceneGraphNode:
    """Scene graph node in the unified contract."""

    id: str
    semantic_type: str
    family: str
    source_expert: str
    confidence: float
    geometry: dict[str, Any]
    audit_trace: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SceneGraphEdge:
    """Scene graph edge in the unified contract."""

    source: str
    target: str
    relation: str
    source_expert: str
    confidence: float
    geometry: dict[str, Any]
    audit_trace: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SceneGraph:
    """Contract-ready graph object."""

    nodes: list[SceneGraphNode]
    edges: list[SceneGraphEdge]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": SCENE_GRAPH_SCHEMA_VERSION,
            "nodes": [item.to_dict() for item in self.nodes],
            "edges": [item.to_dict() for item in self.edges],
            "metadata": dict(self.metadata),
        }


def _coerce_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return min(1.0, max(0.0, number))


def _coerce_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    result = []
    for item in value:
        try:
            result.append(float(item))
        except (TypeError, ValueError):
            return None
    return result


def _is_finite_bbox(bbox: list[float]) -> bool:
    return all(isinstance(value, float | int) and value == value and abs(value) != float("inf") for value in bbox)


def _normalize_source(value: Any) -> str:
    text = str(value or "").strip()
    return text or "unknown"

def _normalize_str(value: Any) -> str:
    return str(value or "").strip()


def prediction_to_node(prediction: Any, ontology_path: str | Path = DEFAULT_ONTOLOGY_PATH) -> SceneGraphNode:
    prediction_dict = _as_mapping(prediction)
    candidate_id = str(prediction_dict.get("candidate_id") or "")
    if not candidate_id:
        raise ValueError("scene graph node missing candidate_id")

    semantic_type = _normalize_str(prediction_dict.get("label") or prediction_dict.get("semantic_type"))
    family = _normalize_str(
        prediction_dict.get("family")
        or ontology_label_to_family(ontology_path).get(semantic_type)
        or "unknown"
    )
    source_expert = _normalize_source(prediction_dict.get("source_expert") or prediction_dict.get("source") or prediction_dict.get("expert"))
    confidence = _coerce_float(prediction_dict.get("confidence"), 0.5)
    bbox = _coerce_bbox(prediction_dict.get("bbox") or (prediction_dict.get("geometry", {}) or {}).get("bbox"))
    geometry = dict(prediction_dict.get("geometry") or {})
    geometry.setdefault("bbox", bbox)
    metadata = dict(prediction_dict.get("metadata") or {})
    audit_trace = dict(prediction_dict.get("audit_trace") or {})
    audit_trace.setdefault("origin", "cadstruct_moe_expert")
    return SceneGraphNode(
        id=candidate_id,
        semantic_type=semantic_type,
        family=family,
        source_expert=source_expert,
        confidence=confidence,
        geometry=geometry,
        audit_trace=audit_trace,
        metadata=metadata,
    )


def prediction_to_edges(prediction: Any, ontology_path: str | Path = DEFAULT_ONTOLOGY_PATH) -> list[SceneGraphEdge]:
    del ontology_path
    prediction_dict = _as_mapping(prediction)
    source = str(prediction_dict.get("candidate_id") or "")
    relations = prediction_dict.get("relations") or []
    if not isinstance(relations, list):
        return []

    source_expert = _normalize_source(
        prediction_dict.get("source_expert") or prediction_dict.get("source") or prediction_dict.get("expert")
    )
    out: list[SceneGraphEdge] = []
    for relation in relations:
        relation_item = relation if isinstance(relation, dict) else {}
        target = _normalize_str(relation_item.get("target"))
        rel = _normalize_str(relation_item.get("relation"))
        if not source or not target or not rel:
            continue
        confidence = _coerce_float(relation_item.get("confidence"), prediction_dict.get("confidence") or 1.0)
        geometry = dict(relation_item.get("geometry") or {})
        audit_trace = dict(relation_item.get("audit_trace") or {})
        audit_trace.setdefault("origin", "cadstruct_moe_expert_relation")
        metadata = dict(relation_item.get("metadata") or {})
        out.append(
            SceneGraphEdge(
                source=source,
                target=target,
                relation=rel,
                source_expert=source_expert,
                confidence=confidence,
                geometry=geometry,
                audit_trace=audit_trace,
                metadata=metadata,
            )
        )
    return out


def convert_predictions_to_scene_graph(
    predictions: list[Any],
    ontology_path: str | Path = DEFAULT_ONTOLOGY_PATH,
) -> SceneGraph:
    nodes = []
    edges = []
    for prediction in predictions:
        node = prediction_to_node(prediction, ontology_path=ontology_path)
        nodes.append(node)
        edges.extend(prediction_to_edges(prediction, ontology_path=ontology_path))
    node_ids = [node.id for node in nodes]
    return SceneGraph(
        nodes=nodes,
        edges=edges,
        metadata={
            "schema_version": SCENE_GRAPH_SCHEMA_VERSION,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "unique_node_ids": len(set(node_ids)),
        },
    )


def validate_scene_graph(
    scene_graph: Any,
    ontology_path: str | Path = DEFAULT_ONTOLOGY_PATH,
) -> tuple[bool, list[str]]:
    allowed_families, allowed_relations = ontology_families_and_relations(ontology_path)
    allowed_labels = ontology_label_to_family(ontology_path)
    errors: list[str] = []
    graph = scene_graph if isinstance(scene_graph, dict) else {}
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    if not isinstance(nodes, list):
        errors.append("nodes_not_list")
        nodes = []
    if not isinstance(edges, list):
        errors.append("edges_not_list")
        edges = []

    node_ids: set[str] = set()
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            errors.append(f"node_not_object:{index}")
            continue
        identifier = _normalize_str(node.get("id"))
        if not identifier:
            errors.append(f"node_missing_id:{index}")
        if identifier in node_ids:
            errors.append(f"node_duplicate_id:{identifier}")
        else:
            node_ids.add(identifier)
        semantic_type = _normalize_str(node.get("semantic_type"))
        if not semantic_type:
            errors.append(f"node_missing_label:{identifier}")
        elif semantic_type not in allowed_labels:
            errors.append(f"unknown_node_label:{semantic_type}")
        family = _normalize_str(node.get("family"))
        if family and family not in allowed_families and family != "unknown":
            errors.append(f"unknown_node_family:{identifier}:{family}")
        source_expert = _normalize_source(node.get("source_expert"))
        if not source_expert:
            errors.append(f"node_missing_source_expert:{identifier}")
        confidence = node.get("confidence")
        if not isinstance(confidence, (int, float)):
            errors.append(f"node_invalid_confidence:{identifier}")
        geometry = node.get("geometry")
        if not isinstance(geometry, dict):
            errors.append(f"node_missing_geometry:{identifier}")
            continue
        bbox = _coerce_bbox(geometry.get("bbox"))
        if bbox is None:
            errors.append(f"node_invalid_bbox:{identifier}")
        else:
            if not _is_finite_bbox(bbox):
                errors.append(f"node_nonfinite_bbox:{identifier}")
            x1, y1, x2, y2 = bbox
            if x2 <= x1 or y2 <= y1:
                errors.append(f"node_bbox_area_zero_or_negative:{identifier}:{x1},{y1},{x2},{y2}")
        if not isinstance(node.get("audit_trace"), dict):
            errors.append(f"node_missing_audit_trace:{identifier}")

    for index, edge in enumerate(edges):
        if not isinstance(edge, dict):
            errors.append(f"edge_not_object:{index}")
            continue
        source = _normalize_str(edge.get("source"))
        target = _normalize_str(edge.get("target"))
        relation = _normalize_str(edge.get("relation"))
        if not source:
            errors.append(f"edge_missing_source:{index}")
        if not target:
            errors.append(f"edge_missing_target:{index}")
        if source and source not in node_ids:
            errors.append(f"edge_source_unknown_node:{index}:{source}")
        if target and target not in node_ids:
            errors.append(f"edge_target_unknown_node:{index}:{target}")
        if not relation:
            errors.append(f"edge_missing_relation:{index}")
        elif relation not in allowed_relations:
            errors.append(f"unknown_edge_relation:{relation}:{index}")
        if not isinstance(edge.get("source_expert"), str) or not edge.get("source_expert").strip():
            errors.append(f"edge_missing_source_expert:{index}")
        if not isinstance(edge.get("confidence"), (int, float)):
            errors.append(f"edge_invalid_confidence:{index}")
        if not isinstance(edge.get("geometry"), dict):
            errors.append(f"edge_missing_geometry:{index}")
        if not isinstance(edge.get("audit_trace"), dict):
            errors.append(f"edge_missing_audit_trace:{index}")

    return len(errors) == 0, errors


def assert_scene_graph_contract(graph: Any, ontology_path: str | Path = DEFAULT_ONTOLOGY_PATH) -> dict[str, Any]:
    valid, errors = validate_scene_graph(graph, ontology_path=ontology_path)
    if not valid:
        raise ValueError(f"scene_graph_contract_invalid: {errors}")
    return graph


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "__dict__"):
        return value.__dict__
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    raise TypeError(f"cannot read prediction-like object: {type(value)!r}")

