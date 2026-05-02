from pathlib import Path
import json

from scripts.vlm.convert_cubicasa5k_svg import convert_dataset
from scripts.vlm.prepare_room_space_dataset import to_room_sample


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
