#!/usr/bin/env python3
"""Train a context-feature MLP for RoomSpaceExpert from integrated MoE records."""

from __future__ import annotations

import argparse
import json
import math
import random
import resource
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from PIL import Image

try:
    from train_room_space_expert import evaluate_predictions, write_jsonl
except ImportError:
    from scripts.vlm.train_room_space_expert import evaluate_predictions, write_jsonl


SYMBOL_TYPES = ["appliance", "bathtub", "column", "equipment", "generic_symbol", "shower", "sink", "stair", "table"]
BOUNDARY_TYPES = ["door", "hard_wall", "opening", "partition_wall", "window"]
FEATURE_NAMES = [
    "cx",
    "cy",
    "width",
    "height",
    "area",
    "aspect",
    "adjacency_degree",
    "contained_symbol_count",
    "contained_symbol_density",
    "room_label_count",
    *[f"symbol_count_{label}" for label in SYMBOL_TYPES],
    *[f"symbol_area_{label}" for label in SYMBOL_TYPES],
    *[f"boundary_touch_{label}" for label in BOUNDARY_TYPES],
]


class ContextMLP(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="datasets/cadstruct_cubicasa5k_moe")
    parser.add_argument("--output-dir", default="checkpoints/cadstruct_moe_room_space_context_mlp")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--class-weight-mode", choices=["balanced", "sqrt", "none"], default="sqrt")
    parser.add_argument("--no-feature-cache", action="store_true", help="Do not write streamed training features to output_dir.")
    parser.add_argument("--seed", type=int, default=20260430)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_feature_cache = None if args.no_feature_cache else output_dir / "train_features.jsonl"
    train_items, train_stream_audit = collect_items_from_jsonl(input_dir / "train.jsonl", train_feature_cache)
    labels = sorted({item["label"] for item in train_items})
    label_to_index = {label: index for index, label in enumerate(labels)}
    train_x, train_y = tensorize_items(train_items, label_to_index)

    model = ContextMLP(len(FEATURE_NAMES), args.hidden_dim, len(labels), args.dropout).to(device)
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    weights = class_weight_tensor(train_y, len(labels), args.class_weight_mode).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    epoch_log = train_loop(model, train_x, train_y, optimizer, loss_fn, args.epochs, args.batch_size, args.seed)

    model_path = output_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    metadata = {
        "model_type": "room_space_context_mlp",
        "labels": labels,
        "feature_names": FEATURE_NAMES,
        "symbol_types": SYMBOL_TYPES,
        "boundary_types": BOUNDARY_TYPES,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "class_weight_mode": args.class_weight_mode,
        "train_items": len(train_items),
        "train_stream_audit": train_stream_audit,
        "train_feature_cache": str(train_feature_cache) if train_feature_cache else None,
        "device": str(device),
        "notes": "Context MLP over geometry, contained symbols, boundaries, and adjacency. Uses gold candidate boxes; not a detector.",
    }
    (output_dir / "model_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary: dict[str, Any] = {
        "input_dir": str(input_dir),
        "model": str(model_path),
        "metadata": str(output_dir / "model_metadata.json"),
        "model_type": "room_space_context_mlp",
        "epoch_log": epoch_log,
        "train_item_counts": dict(Counter(item["label"] for item in train_items)),
        "train_stream_audit": train_stream_audit,
        "splits": {},
    }
    for split in ("dev", "smoke"):
        path = input_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        rows = load_jsonl(path)
        predictions = predict_rows(rows, model, labels, device)
        write_jsonl(output_dir / f"{split}_predictions.jsonl", predictions)
        summary["splits"][split] = evaluate_predictions(predictions)
        summary["splits"][split]["context_audit"] = context_audit(rows)

    summary["memory_audit"] = memory_audit("after_evaluation", device)
    (output_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def collect_items(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = []
    for row in rows:
        context = row_context(row)
        for room in context["rooms"]:
            feature = room_feature(room, context)
            if feature is not None:
                items.append({"id": room["id"], "label": room["room_type"], "feature": feature})
    return items


def collect_items_from_jsonl(path: Path, feature_cache: Path | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Stream integrated MoE rows and keep only room feature vectors for training."""
    items = []
    rows = 0
    rooms = 0
    skipped_rooms = 0
    feature_handle = feature_cache.open("w", encoding="utf-8") if feature_cache else None
    try:
        for row in iter_jsonl(path):
            rows += 1
            context = row_context(row)
            for room in context["rooms"]:
                feature = room_feature(room, context)
                if feature is None:
                    skipped_rooms += 1
                    continue
                item = {"id": room["id"], "label": room["room_type"], "feature": feature}
                items.append(item)
                rooms += 1
                if feature_handle:
                    feature_handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    finally:
        if feature_handle:
            feature_handle.close()
    return items, {
        "source": str(path),
        "rows": rows,
        "items": len(items),
        "rooms": rooms,
        "skipped_rooms": skipped_rooms,
        "feature_cache": str(feature_cache) if feature_cache else None,
    }


def predict_rows(
    rows: list[dict[str, Any]],
    model: ContextMLP,
    labels: list[str],
    device: torch.device,
) -> list[dict[str, Any]]:
    predictions = []
    model.eval()
    with torch.no_grad():
        for row in rows:
            context = row_context(row)
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
    return predictions


def row_context(row: dict[str, Any]) -> dict[str, Any]:
    expected = row.get("expected_json") or {}
    width, height = page_size(row)
    rooms = [
        {
            "id": str(item.get("id") or f"room_{index}"),
            "room_type": str(item.get("room_type") or "room"),
            "bbox": normalize_bbox(item.get("bbox")) or [0.0, 0.0, 0.0, 0.0],
            "shape_features": item.get("shape_features") if isinstance(item.get("shape_features"), dict) else {},
        }
        for index, item in enumerate(expected.get("room_candidates") or [])
        if isinstance(item, dict) and normalize_bbox(item.get("bbox")) is not None
    ]
    symbols = [
        {
            "id": str(item.get("id") or f"symbol_{index}"),
            "symbol_type": str(item.get("symbol_type") or "generic_symbol"),
            "bbox": normalize_bbox(item.get("bbox")) or [0.0, 0.0, 0.0, 0.0],
        }
        for index, item in enumerate(expected.get("symbol_candidates") or [])
        if isinstance(item, dict) and normalize_bbox(item.get("bbox")) is not None
    ]
    texts = [
        {
            "id": str(item.get("id") or f"text_{index}"),
            "text_type": str(item.get("text_type") or "note_text"),
            "text": str(item.get("text") or ""),
            "font_size": item.get("font_size"),
            "bbox": normalize_bbox(item.get("bbox")) or [0.0, 0.0, 0.0, 0.0],
        }
        for index, item in enumerate(expected.get("text_candidates") or [])
        if isinstance(item, dict) and normalize_bbox(item.get("bbox")) is not None
    ]
    graph = ((row.get("request_hints") or {}).get("primitive_graph") or {})
    boundaries = [
        {
            "semantic_type": str(node.get("semantic_type") or "unknown"),
            "bbox": normalize_bbox(node.get("bbox")) or [0.0, 0.0, 0.0, 0.0],
        }
        for node in graph.get("nodes") or []
        if isinstance(node, dict) and normalize_bbox(node.get("bbox")) is not None
    ]
    adjacency = room_adjacency(rooms)
    return {
        "width": width,
        "height": height,
        "rooms": rooms,
        "symbols": symbols,
        "texts": texts,
        "boundaries": boundaries,
        "adjacency": adjacency,
    }


def room_feature(room: dict[str, Any], context: dict[str, Any]) -> list[float] | None:
    bbox = room["bbox"]
    width = float(context["width"])
    height = float(context["height"])
    x1, y1, x2, y2 = bbox
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    area = bbox_area(bbox)
    page_area = max(width * height, 1.0)
    symbol_counts = {label: 0.0 for label in SYMBOL_TYPES}
    symbol_areas = {label: 0.0 for label in SYMBOL_TYPES}
    contained_symbol_count = 0.0
    for symbol in context["symbols"]:
        if bbox_contains(bbox, symbol["bbox"]):
            label = symbol["symbol_type"] if symbol["symbol_type"] in symbol_counts else "generic_symbol"
            contained_symbol_count += 1.0
            symbol_counts[label] += 1.0
            symbol_areas[label] += bbox_area(symbol["bbox"]) / max(area, 1.0)
    boundary_touch = {label: 0.0 for label in BOUNDARY_TYPES}
    for boundary in context["boundaries"]:
        if bbox_intersects(bbox, boundary["bbox"]):
            label = boundary["semantic_type"]
            if label in boundary_touch:
                boundary_touch[label] += 1.0
    room_label_count = sum(1.0 for text in context["texts"] if text["text_type"] == "room_label" and bbox_contains(bbox, text["bbox"]))
    adjacency_degree = float(context["adjacency"].get(room["id"], 0))
    return [
        ((x1 + x2) / 2.0) / max(width, 1.0),
        ((y1 + y2) / 2.0) / max(height, 1.0),
        w / max(width, 1.0),
        h / max(height, 1.0),
        area / page_area,
        math.log((w + 1.0) / (h + 1.0)),
        adjacency_degree / 16.0,
        contained_symbol_count / 32.0,
        contained_symbol_count / max(area / 10000.0, 1.0),
        room_label_count / 4.0,
        *[symbol_counts[label] / 16.0 for label in SYMBOL_TYPES],
        *[symbol_areas[label] for label in SYMBOL_TYPES],
        *[boundary_touch[label] / 32.0 for label in BOUNDARY_TYPES],
    ]


def room_adjacency(rooms: list[dict[str, Any]]) -> dict[str, int]:
    degrees = {room["id"]: 0 for room in rooms}
    for left_index, left in enumerate(rooms):
        for right in rooms[left_index + 1 :]:
            if adjacent(left["bbox"], right["bbox"]):
                degrees[left["id"]] += 1
                degrees[right["id"]] += 1
    return degrees


def adjacent(left: list[float], right: list[float]) -> bool:
    if bbox_contains(left, right) or bbox_contains(right, left):
        return False
    horizontal_gap = max(left[0] - right[2], right[0] - left[2], 0.0)
    vertical_gap = max(left[1] - right[3], right[1] - left[3], 0.0)
    if horizontal_gap > 2.0 or vertical_gap > 2.0:
        return False
    x_overlap = overlap_length(left[0], left[2], right[0], right[2])
    y_overlap = overlap_length(left[1], left[3], right[1], right[3])
    min_side = max(min(left[2] - left[0], left[3] - left[1], right[2] - right[0], right[3] - right[1]), 1.0)
    return max(x_overlap, y_overlap) / min_side >= 0.03


def train_loop(
    model: ContextMLP,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    batch_size: int,
    seed: int,
) -> list[dict[str, float]]:
    log = []
    generator = torch.Generator(device=train_x.device)
    generator.manual_seed(seed)
    for epoch in range(1, epochs + 1):
        order = torch.randperm(train_x.shape[0], generator=generator, device=train_x.device)
        total_loss = 0.0
        correct = 0
        total = 0
        model.train()
        for start in range(0, train_x.shape[0], batch_size):
            batch_index = order[start : start + batch_size]
            xb = train_x[batch_index]
            yb = train_y[batch_index]
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * int(yb.numel())
            correct += int((logits.argmax(dim=1) == yb).sum().detach().cpu())
            total += int(yb.numel())
        log.append({"epoch": epoch, "loss": total_loss / max(total, 1), "accuracy": correct / max(total, 1)})
    return log


def tensorize_items(items: list[dict[str, Any]], label_to_index: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor([item["feature"] for item in items], dtype=torch.float32)
    y = torch.tensor([label_to_index[item["label"]] for item in items], dtype=torch.long)
    return x, y


def class_weight_tensor(y: torch.Tensor, classes: int, mode: str) -> torch.Tensor:
    if mode == "none":
        return torch.ones(classes, dtype=torch.float32)
    counts = torch.bincount(y.cpu(), minlength=classes).float()
    weights = counts.sum() / torch.clamp(counts, min=1.0)
    if mode == "sqrt":
        weights = torch.sqrt(weights)
    return weights / weights.mean()


def context_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    room_counts = []
    symbol_counts = []
    for row in rows:
        context = row_context(row)
        room_counts.append(len(context["rooms"]))
        symbol_counts.append(len(context["symbols"]))
    return {
        "rows": len(rows),
        "rooms": sum(room_counts),
        "symbols": sum(symbol_counts),
        "max_rooms_per_record": max(room_counts) if room_counts else 0,
        "max_symbols_per_record": max(symbol_counts) if symbol_counts else 0,
    }


def page_size(row: dict[str, Any]) -> tuple[float, float]:
    metadata = row.get("metadata") or {}
    width = metadata.get("width")
    height = metadata.get("height")
    if width and height:
        return float(width), float(height)
    image_path = row.get("image_path")
    if image_path:
        try:
            with Image.open(str(image_path)) as image:
                return float(image.size[0]), float(image.size[1])
        except OSError:
            pass
    return 1.0, 1.0


def normalize_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def bbox_area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def bbox_contains(left: list[float], right: list[float]) -> bool:
    return left[0] <= right[0] and left[1] <= right[1] and left[2] >= right[2] and left[3] >= right[3]


def bbox_intersects(left: list[float], right: list[float]) -> bool:
    return not (left[2] < right[0] or right[2] < left[0] or left[3] < right[1] or right[3] < left[1])


def overlap_length(left_min: float, left_max: float, right_min: float, right_max: float) -> float:
    return max(0.0, min(left_max, right_max) - max(left_min, right_min))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def memory_audit(stage: str, device: torch.device) -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    audit: dict[str, Any] = {"stage": stage, "max_rss_kb": int(usage.ru_maxrss)}
    if device.type == "cuda":
        audit["cuda_peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        audit["cuda_peak_reserved_mb"] = torch.cuda.max_memory_reserved(device) / (1024 * 1024)
    return audit


if __name__ == "__main__":
    main()
