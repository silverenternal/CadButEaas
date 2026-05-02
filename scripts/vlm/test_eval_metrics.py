#!/usr/bin/env python3
"""Fast regression checks for CadStruct evaluation metrics."""

from __future__ import annotations

from eval_metrics import dimension_hit, geometry_consistency, relation_f1, semantic_exact_f1, semantic_hit


def main() -> None:
    test_basic_hits()
    test_relation_f1()
    test_geometry_consistency()
    print("eval_metrics regression checks passed")


def test_basic_hits() -> None:
    expected = {
        "dimension_candidates": [{"nominal_value": 80}],
        "semantic_candidates": [{"target_id": 1, "semantic_type": "door"}],
    }
    actual = {
        "dimension_candidates": [{"nominal_value": "80.0"}],
        "semantic_candidates": [{"target_id": 1, "semantic_type": "door"}],
    }
    assert dimension_hit(expected, actual)
    assert semantic_hit(expected, actual)
    assert semantic_exact_f1(expected, actual) == 1.0
    assert semantic_exact_f1(expected, {"semantic_candidates": [{"target_id": 99, "semantic_type": "door"}]}) == 0.0


def test_relation_f1() -> None:
    expected = {"scene_graph": {"edges": [{"source": 1, "target": 2, "relation": "adjacent_to"}]}}
    actual = {"scene_graph": {"edges": [{"source": 2, "target": 1, "relation": "adjacent_to"}]}}
    assert relation_f1(expected, actual) == 1.0


def test_geometry_consistency() -> None:
    sample = {
        "request_hints": {
            "primitive_graph": {
                "nodes": [{"id": 3, "length": 100.0, "orientation": "horizontal"}],
                "edges": [],
            }
        }
    }
    actual = {"semantic_candidates": [{"target_id": 3, "semantic_type": "hard_wall"}]}
    assert geometry_consistency(sample, actual) == 1.0
    actual = {"semantic_candidates": [{"target_id": 99, "semantic_type": "hard_wall"}]}
    assert geometry_consistency(sample, actual) == 0.0


if __name__ == "__main__":
    main()
