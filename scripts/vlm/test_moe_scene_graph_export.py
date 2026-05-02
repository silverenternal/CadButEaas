import json
from pathlib import Path

from scripts.vlm.audit_moe_scene_graph import audit
from scripts.vlm.export_moe_scene_graph import export_record, summarize


def test_export_moe_scene_graph_smoke() -> None:
    record = {
        "image_path": "sample.png",
        "annotation_path": "sample.svg",
        "source_dataset": "synthetic",
        "request_hints": {
            "primitive_graph": {
                "nodes": [
                    {"id": 0, "bbox": [0, 0, 100, 5]},
                    {"id": 1, "bbox": [40, 0, 50, 8]},
                ],
                "edges": [],
            }
        },
        "expected_json": {
            "semantic_candidates": [
                {"target_id": 0, "semantic_type": "hard_wall", "confidence": 1.0},
                {"target_id": 1, "semantic_type": "door", "confidence": 1.0},
            ],
            "room_candidates": [
                {"id": "room0", "room_type": "bedroom", "bbox": [0, 0, 100, 100], "confidence": 1.0}
            ],
            "symbol_candidates": [
                {"id": "sink0", "symbol_type": "sink", "bbox": [20, 20, 30, 30], "confidence": 1.0}
            ],
        },
    }
    exported = export_record(record, "expected_json")
    graph = exported["fusion"]["scene_graph"]
    assert len(graph["nodes"]) == 4
    assert any(edge["relation"] == "contains" for edge in graph["edges"])
    assert summarize([exported])["records"] == 1
    assert audit([exported], "memory")["family_counts"]["space"] == 1
    json.dumps(exported)
