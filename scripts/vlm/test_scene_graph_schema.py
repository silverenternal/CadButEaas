from pathlib import Path

from . import scene_graph_schema as sgs


def _node(
    node_id: str = "n0",
    label: str = "hard_wall",
    family: str = "boundary",
    geometry=None,
    expert: str = "wall_opening",
    bbox=None,
) -> dict:
    return {
        "id": node_id,
        "semantic_type": label,
        "family": family,
        "source_expert": expert,
        "confidence": 0.91,
        "geometry": geometry
        if geometry is not None
        else ({"bbox": bbox if bbox is not None else [0.0, 0.0, 0.1, 0.1]}),
        "audit_trace": {"origin": "unit"},
    }


def _edge(source: str = "n0", target: str = "n1", relation: str = "contains", expert: str = "wall_opening") -> dict:
    return {
        "source": source,
        "target": target,
        "relation": relation,
        "source_expert": expert,
        "confidence": 0.82,
        "geometry": {},
        "audit_trace": {"origin": "unit"},
    }


def test_schema_validation_ok() -> None:
    graph = {
        "version": sgs.SCENE_GRAPH_SCHEMA_VERSION,
        "nodes": [
            _node("n0", "hard_wall", "boundary"),
            _node("n1", "room", "space"),
        ],
        "edges": [
            _edge("n0", "n1", "contains"),
        ],
    }
    valid, errors = sgs.validate_scene_graph(graph, ontology_path=Path("configs/vlm/cadstruct_ontology.json"))
    assert valid, errors


def test_schema_validation_missing_required_field() -> None:
    bad = {
        "nodes": [
            {
                "id": "n0",
                "semantic_type": "hard_wall",
                "family": "boundary",
                "source_expert": "wall_opening",
                "confidence": 0.7,
                "geometry": {"bbox": [0.0, 0.0, 0.1, 0.1]},
                "audit_trace": {"origin": "unit"},
            },
            {
                "id": "n1",
                "family": "space",
                "source_expert": "room_space",
                "confidence": 0.7,
                "geometry": {"bbox": [0.0, 0.0, 0.1, 0.1]},
                "audit_trace": {"origin": "unit"},
            },
        ],
        "edges": [],
    }
    valid, errors = sgs.validate_scene_graph(bad, ontology_path=Path("configs/vlm/cadstruct_ontology.json"))
    assert not valid
    assert any(error.startswith("node_missing_label") for error in errors)


def test_schema_validation_illegal_bbox_fails() -> None:
    bad = {
        "nodes": [
            _node("n0", bbox=[0.2, 0.2, 0.1, 0.3]),
            _node("n1", label="room", family="space", bbox=[0, 0, 0.1, 0.1]),
        ],
        "edges": [_edge("n0", "n1")],
    }
    valid, errors = sgs.validate_scene_graph(bad, ontology_path=Path("configs/vlm/cadstruct_ontology.json"))
    assert not valid
    assert any("node_bbox_area_zero_or_negative" in error for error in errors)


def test_schema_validation_duplicate_node_id() -> None:
    bad = {
        "nodes": [_node("dup"), _node("dup", label="room", family="space"), _node("n2", label="room_label", family="text")],
        "edges": [],
    }
    valid, errors = sgs.validate_scene_graph(bad, ontology_path=Path("configs/vlm/cadstruct_ontology.json"))
    assert not valid
    assert any("node_duplicate_id:dup" == error for error in errors)


def test_schema_validation_unknown_relation_label() -> None:
    bad = {
        "nodes": [_node("n0"), _node("n1", label="room", family="space")],
        "edges": [_edge("n0", "n1", relation="unknown_relation")],
    }
    valid, errors = sgs.validate_scene_graph(bad, ontology_path=Path("configs/vlm/cadstruct_ontology.json"))
    assert not valid
    assert any("unknown_edge_relation" in error for error in errors)
