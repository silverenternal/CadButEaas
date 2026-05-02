from scripts.vlm.cadstruct_moe import DeterministicRouter
from scripts.vlm.cadstruct_moe.experts import RoomSpaceExpert, WallOpeningExpert
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
    router = DeterministicRouter()
    routed = router.route_record(
        {
            "request_hints": {
                "primitive_graph": {"nodes": [{"id": "wall0", "type": "wall"}], "edges": []},
                "semantic_regions": [{"id": "room0", "type": "room_polygon", "bbox": [0.0, 0.0, 0.8, 0.8]}],
            }
        }
    )
    wall_predictions = WallOpeningExpert().predict([item for item in routed if item.family == "boundary"])
    room_predictions = RoomSpaceExpert().predict([item for item in routed if item.family == "space"])
    fused = fuse_predictions(wall_predictions + room_predictions)
    assert len(fused.scene_graph["nodes"]) == 2
    assert "room_without_boundary_relation:room0" in fused.warnings
