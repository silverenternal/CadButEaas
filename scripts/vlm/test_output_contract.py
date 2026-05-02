#!/usr/bin/env python3
"""Fast regression checks for Raster VLM output parsing."""

from __future__ import annotations

from output_contract import normalize_output, parse_model_json


def main() -> None:
    test_fenced_json_and_alias()
    test_scene_graph_semantic_repair()
    test_partial_semantic_recovery()
    print("output_contract regression checks passed")


def test_fenced_json_and_alias() -> None:
    parsed, warnings = parse_model_json(
        """```json
        {"semantic_candidates":[{"target_id":"4","semantic_type":"wall","confidence":"0.9"}],"warnings":[]}
        ```"""
    )
    output = normalize_output(parsed, "test")
    assert not warnings
    assert output["semantic_candidates"][0]["target_id"] == 4
    assert output["semantic_candidates"][0]["semantic_type"] == "hard_wall"


def test_scene_graph_semantic_repair() -> None:
    output = normalize_output(
        {
            "scene_graph": {
                "nodes": [{"id": "2", "semantic_type": "door", "primitive_id": "7"}],
                "edges": [],
            }
        },
        "test",
    )
    assert output["semantic_candidates"] == [
        {
            "target_id": 7,
            "semantic_type": "door",
            "confidence": 0.5,
            "source": "test_scene_graph_repair",
        }
    ]
    assert "semantic_candidates_repaired_from_scene_graph" in output["warnings"]


def test_partial_semantic_recovery() -> None:
    raw = """
    {
      "semantic_candidates": [
        {"target_id": 0, "semantic_type": "hard_wall", "confidence": 0.9, "source": "x"},
        {"target_id": 1, "semantic_type": "door", "confidence": 0.8, "source": "x"},
        {"target_id": 2, "semantic_type": "
    """
    parsed, warnings = parse_model_json(raw)
    output = normalize_output(parsed, "test")
    assert len(output["semantic_candidates"]) == 2
    assert output["semantic_candidates"][1]["semantic_type"] == "door"
    assert "partial_json_recovered" in warnings


if __name__ == "__main__":
    main()
