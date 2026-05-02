from scripts.vlm.train_symbol_fixture_expert import evaluate_predictions as evaluate_symbol_predictions
from scripts.vlm.train_symbol_fixture_expert import predict_rows as predict_symbol_rows
from scripts.vlm.train_symbol_fixture_expert import train_model as train_symbol_model
from scripts.vlm.train_text_dimension_expert import evaluate_predictions as evaluate_text_predictions
from scripts.vlm.train_text_dimension_expert import predict_rows as predict_text_rows
from scripts.vlm.train_text_dimension_expert import train_model as train_text_model


def test_symbol_fixture_baseline_smoke() -> None:
    rows = [
        {
            "metadata": {"width": 100, "height": 100},
            "rooms": [{"id": "room0", "room_type": "bathroom", "bbox": [0, 0, 80, 80]}],
            "symbols": [
                {"id": "sink0", "symbol_type": "sink", "bbox": [10, 10, 20, 20]},
                {"id": "stair0", "symbol_type": "stair", "bbox": [70, 70, 95, 95]},
            ],
            "host_links": [{"source": "room0", "target": "sink0", "relation": "contains"}],
        }
    ]
    model = train_symbol_model(rows)
    predictions = predict_symbol_rows(rows, model)
    report = evaluate_symbol_predictions(predictions)
    assert report["symbols"] == 2
    assert report["host_link"]["matched"] == 1


def test_text_dimension_baseline_smoke() -> None:
    rows = [
        {
            "metadata": {"width": 100, "height": 100},
            "text_candidates": [
                {"id": "line0", "text_type": "dimension_line", "bbox": [0, 10, 90, 12]},
                {"id": "text0", "text_type": "dimension_text", "bbox": [40, 0, 50, 8]},
            ],
            "dimension_links": [{"source": "text0", "target": "line0", "relation": "dimension_of"}],
        }
    ]
    model = train_text_model(rows)
    predictions = predict_text_rows(rows, model)
    report = evaluate_text_predictions(predictions)
    assert report["text_candidates"] == 2
    assert report["dimension_link"]["matched"] == 1
