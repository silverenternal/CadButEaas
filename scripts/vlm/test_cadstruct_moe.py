from scripts.vlm.cadstruct_moe import DeterministicRouter, build_default_experts, describe_experts, summarize_expert_execution
from scripts.vlm.cadstruct_moe.fusion import fuse_predictions


def test_deterministic_router_routes_core_families() -> None:
    record = {
        "request_hints": {
            "primitive_graph": {
                "nodes": [
                    {"id": "n0", "type": "polyline", "bbox": [0.0, 0.0, 1.0, 0.1]},
                    {"id": "n1", "type": "door", "bbox": [0.4, 0.0, 0.5, 0.1]},
                ],
                "edges": [],
            },
            "text_candidates": [{"id": "t0", "type": "dimension_text", "bbox": [0.1, 0.1, 0.2, 0.2]}],
            "symbol_candidates": [{"id": "s0", "type": "stair", "bbox": [0.3, 0.3, 0.4, 0.4]}],
            "semantic_regions": [{"id": "r0", "type": "room_polygon", "bbox": [0.0, 0.0, 0.8, 0.8]}],
        }
    }

    routed = DeterministicRouter().route_record(record)
    by_id = {item.candidate_id: item for item in routed}
    assert by_id["n0"].expert == "wall_opening"
    assert by_id["n1"].family == "boundary"
    assert by_id["t0"].expert == "text_dimension"
    assert by_id["s0"].expert == "symbol_fixture"
    assert by_id["r0"].expert == "room_space"


def test_passthrough_experts_and_fusion_warnings() -> None:
    experts = build_default_experts(["boundary", "space"])
    router = DeterministicRouter()
    routed = router.route_record(
        {
            "request_hints": {
                "primitive_graph": {"nodes": [{"id": "wall0", "type": "wall"}], "edges": []},
                "semantic_regions": [{"id": "room0", "type": "room_polygon", "bbox": [0.0, 0.0, 0.8, 0.8]}],
            }
        }
    )
    wall_predictions = experts["boundary"].predict([item for item in routed if item.family == "boundary"])
    room_predictions = experts["space"].predict([item for item in routed if item.family == "space"])
    fused = fuse_predictions(wall_predictions + room_predictions)
    assert len(fused.scene_graph["nodes"]) == 2
    assert "room_without_boundary_relation:room0" in fused.warnings


def test_expert_registry_is_descriptive() -> None:
    expert_bundle = build_default_experts()
    registry = describe_experts(expert_bundle)
    assert set(registry) == {"boundary", "space", "symbol", "text", "sheet"}
    assert registry["boundary"]["name"] == "wall_opening"


def test_expert_audit_summary_tracks_basic_counts() -> None:
    expert = build_default_experts(["sheet"])["sheet"]
    from scripts.vlm.cadstruct_moe.schema import RoutedCandidate
    routed = [
        RoutedCandidate(
            candidate_id="c0",
            expert="sheet_layout",
            family="sheet",
            candidate_type="title_block",
            confidence=0.6,
            bbox=[0.0, 0.0, 10.0, 10.0],
            source="deterministic_router",
            payload={},
            route_trace={},
        )
    ]
    predictions = expert.predict(routed)
    summary = summarize_expert_execution(expert, routed, predictions)
    assert summary["candidate_count"] == 1
    assert summary["prediction_count"] == 1
    assert summary["fallback_prediction_count"] == 0
