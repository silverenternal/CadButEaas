#!/usr/bin/env python3
"""Evaluate RoomSpace context MLP with predicted upstream expert outputs."""

from __future__ import annotations

import argparse
import json
import resource
from collections import Counter
from pathlib import Path
from typing import Any

import torch

try:
    from train_room_space_context_mlp import ContextMLP, load_jsonl, row_context, room_feature
    from train_room_space_expert import evaluate_predictions, write_jsonl
except ImportError:
    from scripts.vlm.train_room_space_context_mlp import ContextMLP, load_jsonl, row_context, room_feature
    from scripts.vlm.train_room_space_expert import evaluate_predictions, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="datasets/cadstruct_cubicasa5k_moe/dev.jsonl")
    parser.add_argument("--model", default="checkpoints/cadstruct_moe_room_space_context_mlp_streamed/model.pt")
    parser.add_argument("--metadata", default="checkpoints/cadstruct_moe_room_space_context_mlp_streamed/model_metadata.json")
    parser.add_argument("--symbol-predictions", default="checkpoints/cadstruct_moe_symbol_fixture_crop_mlp/dev_predictions.jsonl")
    parser.add_argument("--text-predictions", default="checkpoints/cadstruct_moe_text_dimension_crop_mlp/dev_predictions.jsonl")
    parser.add_argument("--boundary-predictions", default="")
    parser.add_argument("--boundary-mode", choices=["gold", "none", "predicted"], default="gold")
    parser.add_argument("--symbol-confidence-threshold", type=float, default=0.0)
    parser.add_argument("--text-confidence-threshold", type=float, default=0.0)
    parser.add_argument("--boundary-confidence-threshold", type=float, default=0.0)
    parser.add_argument("--output", default="reports/vlm/moe/room_space_context_predicted_upstream_dev.json")
    parser.add_argument(
        "--predictions-output",
        default="reports/vlm/moe/room_space_context_predicted_upstream_dev_predictions.jsonl",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    labels = [str(label) for label in metadata["labels"]]
    model = ContextMLP(
        input_dim=len(metadata["feature_names"]),
        hidden_dim=int(metadata["hidden_dim"]),
        output_dim=len(labels),
        dropout=float(metadata.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))

    rows = load_jsonl(Path(args.input))
    symbol_index = load_symbol_predictions(Path(args.symbol_predictions), args.symbol_confidence_threshold)
    text_index = load_text_predictions(Path(args.text_predictions), args.text_confidence_threshold)
    boundary_index = (
        load_boundary_predictions(Path(args.boundary_predictions), args.boundary_confidence_threshold)
        if args.boundary_mode == "predicted" and args.boundary_predictions
        else {}
    )

    predictions, upstream_audit = predict_rows_with_upstream(rows, model, labels, device, symbol_index, text_index, boundary_index, args.boundary_mode)
    write_jsonl(Path(args.predictions_output), predictions)
    report = evaluate_predictions(predictions)
    report["input"] = args.input
    report["model"] = args.model
    report["metadata"] = args.metadata
    report["predictions_output"] = args.predictions_output
    report["upstream_audit"] = upstream_audit
    report["memory_audit"] = memory_audit(device)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def predict_rows_with_upstream(
    rows: list[dict[str, Any]],
    model: ContextMLP,
    labels: list[str],
    device: torch.device,
    symbol_index: dict[str, list[dict[str, Any]]],
    text_index: dict[str, list[dict[str, Any]]],
    boundary_index: dict[str, list[dict[str, Any]]],
    boundary_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions = []
    audit = Counter()
    model.eval()
    with torch.no_grad():
        for row in rows:
            context = row_context(row)
            key = row_key(row)
            if key in symbol_index:
                context["symbols"] = symbol_index[key]
                audit["records_with_predicted_symbols"] += 1
            if key in text_index:
                context["texts"] = text_index[key]
                audit["records_with_predicted_texts"] += 1
            if boundary_mode == "none":
                context["boundaries"] = []
                audit["records_with_boundary_removed"] += 1
            elif boundary_mode == "predicted":
                context["boundaries"] = boundary_index.get(key, [])
                audit["records_with_predicted_boundaries"] += int(key in boundary_index)
            audit["records"] += 1
            audit["symbols_used"] += len(context["symbols"])
            audit["texts_used"] += len(context["texts"])
            audit["boundaries_used"] += len(context["boundaries"])
            room_predictions = []
            for room in context["rooms"]:
                feature = room_feature(room, context)
                if feature is None:
                    continue
                logits = model(torch.tensor([feature], dtype=torch.float32, device=device))
                probs = torch.softmax(logits, dim=1)[0]
                pred_index = int(probs.argmax().detach().cpu())
                room_predictions.append(
                    {
                        "id": room["id"],
                        "gold": room["room_type"],
                        "prediction": labels[pred_index],
                        "confidence": float(probs[pred_index].detach().cpu()),
                        "bbox": room["bbox"],
                        "iou": 1.0,
                    }
                )
            predictions.append(
                {
                    "image": row.get("image_path"),
                    "annotation": row.get("annotation_path"),
                    "source_dataset": row.get("source_dataset"),
                    "rooms": room_predictions,
                }
            )
    return predictions, dict(audit)


def load_symbol_predictions(path: Path, threshold: float) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for row in load_jsonl(path):
        symbols = []
        for item in row.get("symbols") or []:
            if float(item.get("confidence") or 0.0) < threshold:
                continue
            bbox = normalize_bbox(item.get("bbox"))
            if bbox is None:
                continue
            symbols.append(
                {
                    "id": str(item.get("id") or f"symbol_{len(symbols)}"),
                    "symbol_type": str(item.get("prediction") or "generic_symbol"),
                    "bbox": bbox,
                }
            )
        index[row_key(row)] = symbols
    return index


def load_text_predictions(path: Path, threshold: float) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for row in load_jsonl(path):
        texts = []
        for item in row.get("text_candidates") or []:
            if float(item.get("confidence") or 0.0) < threshold:
                continue
            bbox = normalize_bbox(item.get("bbox"))
            if bbox is None:
                continue
            texts.append(
                {
                    "id": str(item.get("id") or f"text_{len(texts)}"),
                    "text_type": str(item.get("prediction") or "note_text"),
                    "bbox": bbox,
                }
            )
        index[row_key(row)] = texts
    return index


def load_boundary_predictions(path: Path, threshold: float) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for row in load_jsonl(path):
        boundaries = []
        candidates = row.get("boundaries") or row.get("boundary_nodes") or row.get("nodes") or []
        for item in candidates:
            if float(item.get("confidence") or 1.0) < threshold:
                continue
            bbox = normalize_bbox(item.get("bbox"))
            if bbox is None:
                continue
            boundaries.append(
                {
                    "semantic_type": str(item.get("prediction") or item.get("semantic_type") or "unknown"),
                    "bbox": bbox,
                }
            )
        index[row_key(row)] = boundaries
    return index


def row_key(row: dict[str, Any]) -> str:
    value = row.get("image_path") or row.get("image") or row.get("image_file") or row.get("annotation_path") or row.get("annotation")
    return str(value or "")


def normalize_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def memory_audit(device: torch.device) -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    audit: dict[str, Any] = {"max_rss_kb": int(usage.ru_maxrss)}
    if device.type == "cuda":
        audit["cuda_peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        audit["cuda_peak_reserved_mb"] = torch.cuda.max_memory_reserved(device) / (1024 * 1024)
    return audit


if __name__ == "__main__":
    main()
