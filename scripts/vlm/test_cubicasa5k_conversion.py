from pathlib import Path
import json

from scripts.vlm.convert_cubicasa5k_svg import convert_dataset, find_image_for_svg, path_bbox, text_bbox
from scripts.vlm.export_moe_scene_graph import bbox_hosts_symbol
from scripts.vlm.render_scene_graph_visual_demo import canvas_to_image_transform, map_bbox
from scripts.vlm.prepare_room_space_dataset import to_room_sample
from scripts.vlm.apply_node_quality_gate import apply_quality_gate
from scripts.vlm.apply_roomspace_link_gate_v3 import process_row
from scripts.vlm.audit_visual_scene_graph_defects import audit_sample
from scripts.vlm.roomspace_geometry import best_room_for_label


def test_convert_cubicasa5k_svg_smoke(tmp_path: Path) -> None:
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    (sample_dir / "model.svg").write_text(
        """
        <svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
          <rect class="Wall" x="0" y="0" width="100" height="5" />
          <polygon class="Bedroom" points="10,10 90,10 90,90 10,90" />
          <rect class="Door" x="45" y="0" width="10" height="8" />
          <rect class="Sink" x="20" y="20" width="10" height="10" />
        </svg>
        """,
        encoding="utf-8",
    )

    records = convert_dataset(tmp_path, limit=None, min_bbox_area=1.0)
    assert len(records) == 1
    record = records[0]
    expected = record["expected_json"]
    labels = {item["semantic_type"] for item in expected["semantic_candidates"]}
    rooms = {item["room_type"] for item in expected["room_candidates"]}
    symbols = {item["symbol_type"] for item in expected["symbol_candidates"]}
    assert labels == {"hard_wall", "door"}
    assert rooms == {"bedroom"}
    assert symbols == {"sink"}

    room_sample = to_room_sample(record, min_room_area=1.0)
    assert room_sample is not None
    assert room_sample["rooms"][0]["room_type"] == "bedroom"


def test_converted_record_is_json_serializable(tmp_path: Path) -> None:
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    (sample_dir / "model.svg").write_text(
        """
        <svg xmlns="http://www.w3.org/2000/svg">
          <path id="LivingRoom" d="M 1 1 L 20 1 L 20 20 L 1 20 Z" />
        </svg>
        """,
        encoding="utf-8",
    )
    records = convert_dataset(tmp_path, limit=None, min_bbox_area=1.0)
    assert json.loads(json.dumps(records[0]))["metadata"]["room_count"] == 1


def test_svg_relative_path_bbox_does_not_expand_to_origin() -> None:
    bbox = path_bbox("M2391.82,684.80 q-87.40,0.00 -87.40,-87.40 l87.40,0.00Z")
    assert bbox == [2304.42, 597.4, 2391.82, 684.8]


def test_convert_cubicasa5k_svg_relative_door_path(tmp_path: Path) -> None:
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    (sample_dir / "model.svg").write_text(
        """
        <svg xmlns="http://www.w3.org/2000/svg" width="3000" height="1600" viewBox="0 0 3000 1600">
          <g class="Door">
            <path d="M2391.82,684.80 q-87.40,0.00 -87.40,-87.40 l87.40,0.00Z" />
          </g>
        </svg>
        """,
        encoding="utf-8",
    )
    records = convert_dataset(tmp_path, limit=None, min_bbox_area=1.0)
    node = records[0]["request_hints"]["primitive_graph"]["nodes"][0]
    assert node["semantic_type"] == "door"
    assert node["bbox"] == [2304.42, 597.4, 2391.82, 684.8]


def test_hidden_svg_elements_are_not_converted(tmp_path: Path) -> None:
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    (sample_dir / "model.svg").write_text(
        """
        <svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
          <g class="Dimension" style="display: none;">
            <polygon class="DimensionMark" points="0,0 10,0 10,10 0,10" />
            <text class="TextLabel DimensionMeasureLabel" x="20" y="20">123</text>
          </g>
          <g class="Name">
            <text x="30" y="30">Kitchen</text>
          </g>
        </svg>
        """,
        encoding="utf-8",
    )
    records = convert_dataset(tmp_path, limit=None, min_bbox_area=1.0)
    texts = records[0]["request_hints"]["text_candidates"]
    assert len(texts) == 1
    assert texts[0]["text_type"] == "room_label"
    assert texts[0]["text"] == "Kitchen"


def test_text_bbox_honors_anchor_and_inherited_font_size() -> None:
    import xml.etree.ElementTree as ET

    element = ET.fromstring('<text x="100" y="50" text-anchor="middle">1234</text>')
    bbox = text_bbox(element, inherited_font_size=20.0)
    assert bbox == [78.0, 30.0, 122.0, 50.0]


def test_visible_numeric_text_is_exported_as_dimension_text(tmp_path: Path) -> None:
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    (sample_dir / "model.svg").write_text(
        """
        <svg xmlns="http://www.w3.org/2000/svg" width="200" height="100">
          <g class="TextLabel" font-size="12">
            <text x="50" y="50" text-anchor="middle">1200</text>
          </g>
        </svg>
        """,
        encoding="utf-8",
    )
    records = convert_dataset(tmp_path, limit=None, min_bbox_area=1.0)
    texts = records[0]["expected_json"]["text_candidates"]
    assert texts[0]["text"] == "1200"
    assert texts[0]["text_type"] == "dimension_text"
    assert texts[0]["font_size"] == 12.0


def test_electricity_sign_is_generic_symbol_not_equipment(tmp_path: Path) -> None:
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    (sample_dir / "model.svg").write_text(
        """
        <svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
          <path class="electricitySign" d="M 10 10 L 20 10 L 15 20 Z" />
        </svg>
        """,
        encoding="utf-8",
    )
    records = convert_dataset(tmp_path, limit=None, min_bbox_area=1.0)
    symbols = records[0]["expected_json"]["symbol_candidates"]
    assert symbols[0]["raw_label"] == "electricitySign"
    assert symbols[0]["symbol_type"] == "generic_symbol"


def test_room_hosting_allows_small_edge_fixture_overlap() -> None:
    assert bbox_hosts_symbol([0.0, 0.0, 100.0, 100.0], [95.0, 40.0, 105.0, 50.0])
    assert not bbox_hosts_symbol([0.0, 0.0, 100.0, 100.0], [110.0, 40.0, 120.0, 50.0])


def test_source_geometry_is_preserved_for_room_polygon(tmp_path: Path) -> None:
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    (sample_dir / "model.svg").write_text(
        """
        <svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
          <polygon class="Bedroom" points="10,10 90,10 90,90 10,90" />
        </svg>
        """,
        encoding="utf-8",
    )
    records = convert_dataset(tmp_path, limit=None, min_bbox_area=1.0)
    room = records[0]["expected_json"]["room_candidates"][0]
    assert room["geometry"]["type"] == "polygon"
    assert room["geometry"]["points"] == [[10.0, 10.0], [90.0, 10.0], [90.0, 90.0], [10.0, 90.0]]


def test_cubicasa_image_selection_prefers_scaled_floorplan(tmp_path: Path) -> None:
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    svg_path = sample_dir / "model.svg"
    svg_path.write_text('<svg xmlns="http://www.w3.org/2000/svg" />', encoding="utf-8")
    original = sample_dir / "F1_original.png"
    scaled = sample_dir / "F1_scaled.png"
    original.write_bytes(b"original")
    scaled.write_bytes(b"scaled")
    assert find_image_for_svg(svg_path) == scaled


def test_canvas_to_image_transform_uses_meet_fit_with_padding() -> None:
    transform = canvas_to_image_transform(
        [0.0, 0.0, 100.0, 200.0],
        image_w=300,
        image_h=300,
        config={"canvas_policy": {"svg_image_fit": "meet"}},
    )
    assert transform["mode"] == "meet"
    assert transform["scale_x"] == transform["scale_y"] == 1.5
    assert transform["offset_x"] == 75.0
    assert transform["offset_y"] == 0.0
    assert transform["content_bbox"] == [75.0, 0.0, 225.0, 300.0]
    assert map_bbox([0.0, 0.0, 100.0, 200.0], transform) == [75.0, 0.0, 150.0, 300.0]


def test_canvas_to_image_transform_uses_stretch_fit_without_padding() -> None:
    transform = canvas_to_image_transform(
        [0.0, 0.0, 100.0, 200.0],
        image_w=300,
        image_h=300,
        config={"canvas_policy": {"svg_image_fit": "stretch"}},
    )
    assert transform["mode"] == "stretch"
    assert transform["scale_x"] == 3.0
    assert transform["scale_y"] == 1.5
    assert transform["offset_x"] == 0.0
    assert transform["offset_y"] == 0.0
    assert transform["content_bbox"] == [0.0, 0.0, 300.0, 300.0]
    assert map_bbox([0.0, 0.0, 100.0, 200.0], transform) == [0.0, 0.0, 300.0, 300.0]


def test_quality_gate_uses_svg_viewbox_and_records_clipping_metadata(tmp_path: Path) -> None:
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    svg_path = sample_dir / "model.svg"
    svg_path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100" />',
        encoding="utf-8",
    )
    image_path = sample_dir / "F1_original.png"
    image_path.write_bytes(b"not-a-real-png")
    row = {
        "image": str(image_path),
        "annotation": str(svg_path),
        "scene_graph": {
            "nodes": [
                {
                    "id": "boundary_1",
                    "family": "boundary",
                    "semantic_type": "door",
                    "confidence": 0.99,
                    "geometry": {"bbox": [-10.0, -5.0, 20.0, 30.0]},
                    "metadata": {"raw_label": "door"},
                }
            ],
            "edges": [],
        },
    }
    converted = {"annotation_path": str(svg_path), "metadata": {"width": 999, "height": 999}}
    gated, events = apply_quality_gate(row, converted, thresholds={"canvas_tolerance": 0.5, "drop_outside_ratio": 0.9, "review_outside_ratio": 0.01, "large_boundary_area_ratio": 0.9, "large_opening_aspect_ratio": 50.0, "large_opening_area_ratio": 0.9, "symbol_ink_ratio_min": 0.006, "tiny_room_area": 1.0, "tiny_room_side": 1.0, "low_confidence_review": 0.0}, mode="review")
    node = gated["scene_graph"]["nodes"][0]
    assert node["geometry"]["bbox"] == [0.0, 0.0, 20.0, 30.0]
    assert node["metadata"]["source_canvas_bbox"] == [0.0, 0.0, 100.0, 100.0]
    assert node["metadata"]["unclipped_bbox"] == [-10.0, -5.0, 20.0, 30.0]
    assert node["metadata"]["was_clipped_to_canvas"] is True
    assert events[0]["decision"] == "review"


def test_roomspace_polygon_contains_center_beats_bbox_boundary() -> None:
    room = {
        "id": "room_poly",
        "family": "space",
        "geometry": {
            "bbox": [0, 0, 100, 100],
            "source_geometry": {"type": "polygon", "points": [[0, 0], [100, 0], [100, 100], [0, 100]]},
        },
    }
    label = {"id": "label", "family": "text", "semantic_type": "room_label", "geometry": {"bbox": [94, 45, 104, 55]}}
    best, relation = best_room_for_label(label, [room], [0, 0, 120, 120])
    assert best["id"] == "room_poly"
    assert relation["contains"] is True
    assert relation["method"] == "polygon_contains_center"


def test_roomspace_adaptive_margin_links_near_boundary_label() -> None:
    room = {"id": "room", "family": "space", "geometry": {"bbox": [0, 0, 100, 100]}}
    label = {"id": "label", "family": "text", "semantic_type": "room_label", "geometry": {"bbox": [100.5, 40, 105.5, 50]}}
    best, relation = best_room_for_label(label, [room], [0, 0, 500, 500])
    assert best["id"] == "room"
    assert relation["contains"] is True
    assert relation["method"] == "nearest_with_adaptive_margin"


def test_roomspace_gate_marks_small_unlabeled_room_review_without_deleting() -> None:
    row = {
        "image": "datasets/external/cubicasa5k_zenodo/unpacked/cubicasa5k/high_quality_architectural/test/F1_original.png",
        "scene_graph": {
            "nodes": [
                {"id": "room_small", "family": "space", "semantic_type": "toilet", "confidence": 0.9, "geometry": {"bbox": [10, 10, 40, 40]}, "metadata": {}},
                {"id": "boundary_1", "family": "boundary", "semantic_type": "hard_wall", "confidence": 0.9, "geometry": {"bbox": [8, 8, 42, 12]}, "metadata": {}},
            ],
            "edges": [],
        },
        "warnings": [],
        "route_trace": {},
    }
    converted = {"metadata": {"width": 200, "height": 200}}
    updated, _links, gates = process_row(row, converted)
    nodes = updated["scene_graph"]["nodes"]
    assert [node["id"] for node in nodes] == ["room_small", "boundary_1"]
    assert "needs_review_room_label_missing" in nodes[0]["quality_flags"]
    assert gates and gates[0]["decision"] == "review"


def test_roomspace_audit_accepts_explicit_label_edge_for_narrow_room() -> None:
    row = {
        "image": "datasets/external/cubicasa5k_zenodo/unpacked/cubicasa5k/high_quality_architectural/test/F1_original.png",
        "scene_graph": {
            "nodes": [
                {"id": "room_narrow", "family": "space", "semantic_type": "room", "confidence": 0.9, "geometry": {"bbox": [0, 0, 37, 100]}, "metadata": {}},
                {"id": "label_narrow", "family": "text", "semantic_type": "room_label", "confidence": 0.9, "geometry": {"bbox": [18, 8, 68, 18]}, "metadata": {"text": "UNDEFINED"}},
            ],
            "edges": [{"source": "label_narrow", "target": "room_narrow", "relation": "labels"}],
        },
        "warnings": [],
        "route_trace": {"source_mode": "real_upstream_saved_model_predictions"},
    }
    converted = {
        "metadata": {"width": 100, "height": 120},
        "expected_json": {"text_candidates": []},
    }
    sample = audit_sample(row, {row["image"]: converted}, {}, argparse_namespace())
    assert sample["defect_counts"].get("room_without_label", 0) == 0
    assert sample["defect_counts"].get("label_without_room", 0) == 0


def argparse_namespace():
    class Args:
        canvas_tolerance = 0.5
        symbol_ink_ratio_threshold = 0.006
        large_boundary_area_ratio = 0.08
        tiny_room_area_threshold = 5000.0
        tiny_room_side_threshold = 80.0

    return Args()
