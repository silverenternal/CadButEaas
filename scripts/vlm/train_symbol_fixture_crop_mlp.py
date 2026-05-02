#!/usr/bin/env python3
"""Train a small crop+geometry MLP for SymbolFixtureExpert."""

from __future__ import annotations

import argparse
import json
import math
import random
import resource
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageStat

try:
    from train_symbol_fixture_expert import evaluate_predictions, load_jsonl, predict_host_links, write_jsonl
except ImportError:
    from scripts.vlm.train_symbol_fixture_expert import evaluate_predictions, load_jsonl, predict_host_links, write_jsonl


FEATURE_NAMES = [
    "cx",
    "cy",
    "width",
    "height",
    "area",
    "aspect",
    "crop_mean",
    "crop_std",
    "crop_dark_ratio",
    "crop_light_ratio",
    "crop_width_norm",
    "crop_height_norm",
]


class SymbolMLP(torch.nn.Module):
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
    parser.add_argument("--config", help="Optional expert training config JSON using configs/vlm/expert_training_schema.json.")
    parser.add_argument("--dataset-dir", default="datasets/cadstruct_symbols_v1")
    parser.add_argument("--output-dir", default="checkpoints/cadstruct_moe_symbol_fixture_crop_mlp")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-train-items", type=int, default=100000)
    parser.add_argument("--max-eval-items", type=int, default=0)
    parser.add_argument("--max-per-label", type=int, default=20000)
    parser.add_argument("--image-cache-size", type=int, default=8)
    parser.add_argument("--class-weight-mode", choices=["balanced", "sqrt", "none"], default="sqrt")
    parser.add_argument("--seed", type=int, default=20260430)
    args = parser.parse_args()
    args = apply_training_config(args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_jsonl(dataset_dir / "train.jsonl")
    train_items = collect_items(train_rows, args.max_train_items, args.max_per_label, args.seed, args.image_cache_size)
    labels = sorted({item["label"] for item in train_items})
    label_to_index = {label: index for index, label in enumerate(labels)}
    train_x, train_y = tensorize_items(train_items, label_to_index)

    class_weights = class_weight_tensor(train_y, len(labels), args.class_weight_mode).to(device)
    model = SymbolMLP(len(FEATURE_NAMES), args.hidden_dim, len(labels), args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    train_x = train_x.to(device)
    train_y = train_y.to(device)
    epoch_log = train_loop(model, train_x, train_y, optimizer, loss_fn, args.epochs, args.batch_size, args.seed)

    model_path = output_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    metadata = {
        "model_type": "symbol_fixture_crop_mlp",
        "labels": labels,
        "feature_names": FEATURE_NAMES,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "train_items": len(train_items),
        "class_weight_mode": args.class_weight_mode,
        "image_cache_size": args.image_cache_size,
        "device": str(device),
        "notes": "Crop-statistics MLP baseline; replace with detector/crop encoder for paper metrics.",
    }
    (output_dir / "model_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "model": str(model_path),
        "metadata": str(output_dir / "model_metadata.json"),
        "model_type": "symbol_fixture_crop_mlp",
        "epoch_log": epoch_log,
        "train_item_counts": dict(Counter(item["label"] for item in train_items)),
        "splits": {},
    }
    for split in ("dev", "smoke"):
        path = dataset_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        rows = load_jsonl(path)
        predictions = predict_rows(rows, model, labels, device, args.max_eval_items, args.image_cache_size)
        write_jsonl(output_dir / f"{split}_predictions.jsonl", predictions)
        summary["splits"][split] = evaluate_predictions(predictions)

    summary["memory_audit"] = memory_audit("after_evaluation", device)
    (output_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def collect_items(
    rows: list[dict[str, Any]],
    max_items: int,
    max_per_label: int,
    seed: int,
    image_cache_size: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    image_cache: OrderedDict[str, Image.Image | None] = OrderedDict()
    for row in rows:
        width = float((row.get("metadata") or {}).get("width") or 1.0)
        height = float((row.get("metadata") or {}).get("height") or 1.0)
        image = load_image(row.get("image"), image_cache, image_cache_size)
        for item in row.get("symbols") or []:
            label = str(item.get("symbol_type") or "generic_symbol")
            feature = feature_vector(item.get("bbox"), width, height, image)
            if feature is None:
                continue
            grouped[label].append({"feature": feature, "label": label})
    rng = random.Random(seed)
    selected = []
    for _label, items in grouped.items():
        rng.shuffle(items)
        selected.extend(items[:max_per_label])
    rng.shuffle(selected)
    return selected[:max_items] if max_items > 0 else selected


def tensorize_items(items: list[dict[str, Any]], label_to_index: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor([item["feature"] for item in items], dtype=torch.float32)
    y = torch.tensor([label_to_index[item["label"]] for item in items], dtype=torch.long)
    return x, y


def train_loop(
    model: SymbolMLP,
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


def predict_rows(
    rows: list[dict[str, Any]],
    model: SymbolMLP,
    labels: list[str],
    device: torch.device,
    max_eval_items: int,
    image_cache_size: int,
) -> list[dict[str, Any]]:
    predictions = []
    image_cache: OrderedDict[str, Image.Image | None] = OrderedDict()
    seen = 0
    model.eval()
    with torch.no_grad():
        for row in rows:
            width = float((row.get("metadata") or {}).get("width") or 1.0)
            height = float((row.get("metadata") or {}).get("height") or 1.0)
            image = load_image(row.get("image"), image_cache, image_cache_size)
            symbols = []
            for item in row.get("symbols") or []:
                if max_eval_items > 0 and seen >= max_eval_items:
                    break
                feature = feature_vector(item.get("bbox"), width, height, image)
                if feature is None:
                    continue
                logits = model(torch.tensor([feature], dtype=torch.float32, device=device))
                probs = torch.softmax(logits, dim=1)[0]
                pred_index = int(probs.argmax().detach().cpu())
                symbols.append(
                    {
                        "id": item.get("id"),
                        "gold": item.get("symbol_type"),
                        "prediction": labels[pred_index],
                        "confidence": float(probs[pred_index].detach().cpu()),
                        "bbox": item.get("bbox"),
                        "iou": 1.0,
                    }
                )
                seen += 1
            predictions.append(
                {
                    "image": row.get("image"),
                    "annotation": row.get("annotation"),
                    "source_dataset": row.get("source_dataset"),
                    "symbols": symbols,
                    "host_links_gold": row.get("host_links") or [],
                    "host_links_pred": predict_host_links(symbols, row.get("rooms") or []),
                }
            )
            if max_eval_items > 0 and seen >= max_eval_items:
                break
    return predictions


def feature_vector(value: Any, width: float, height: float, image: Image.Image | None) -> list[float] | None:
    bbox = normalize_bbox(value)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    stats = crop_stats(image, bbox)
    return [
        ((x1 + x2) / 2.0) / max(width, 1.0),
        ((y1 + y2) / 2.0) / max(height, 1.0),
        w / max(width, 1.0),
        h / max(height, 1.0),
        (w * h) / max(width * height, 1.0),
        math.log((w + 1.0) / (h + 1.0)),
        *stats,
    ]


def crop_stats(image: Image.Image | None, bbox: list[float]) -> list[float]:
    if image is None:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    width, height = image.size
    x1 = max(0, min(width - 1, int(math.floor(bbox[0]))))
    y1 = max(0, min(height - 1, int(math.floor(bbox[1]))))
    x2 = max(x1 + 1, min(width, int(math.ceil(bbox[2]))))
    y2 = max(y1 + 1, min(height, int(math.ceil(bbox[3]))))
    crop = image.crop((x1, y1, x2, y2)).resize((32, 32))
    stat = ImageStat.Stat(crop)
    mean = float(stat.mean[0]) / 255.0
    std = float(stat.stddev[0]) / 255.0
    if hasattr(crop, "get_flattened_data"):
        pixels = list(crop.get_flattened_data())
    else:
        pixels = list(crop.getdata())
    dark_ratio = sum(1 for value in pixels if int(value) < 96) / max(len(pixels), 1)
    light_ratio = sum(1 for value in pixels if int(value) > 224) / max(len(pixels), 1)
    return [mean, std, dark_ratio, light_ratio, (x2 - x1) / max(width, 1), (y2 - y1) / max(height, 1)]


def load_image(path: Any, cache: OrderedDict[str, Image.Image | None], cache_size: int) -> Image.Image | None:
    if not path:
        return None
    key = str(path)
    if key in cache:
        image = cache.pop(key)
        cache[key] = image
        return image
    try:
        image = Image.open(key).convert("L")
    except OSError:
        image = None
    if cache_size > 0:
        cache[key] = image
        while len(cache) > cache_size:
            _old_key, old_image = cache.popitem(last=False)
            if old_image is not None:
                old_image.close()
    return image


def class_weight_tensor(y: torch.Tensor, classes: int, mode: str) -> torch.Tensor:
    if mode == "none":
        return torch.ones(classes, dtype=torch.float32)
    counts = torch.bincount(y.cpu(), minlength=classes).float()
    weights = counts.sum() / torch.clamp(counts, min=1.0)
    if mode == "sqrt":
        weights = torch.sqrt(weights)
    return weights / weights.mean()


def apply_training_config(args: argparse.Namespace) -> argparse.Namespace:
    if not getattr(args, "config", None):
        return args
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    mapping = {
        "dataset_dir": "dataset_dir",
        "output_dir": "output_dir",
        "epochs": "epochs",
        "batch_size": "batch_size",
        "hidden_dim": "hidden_dim",
        "dropout": "dropout",
        "lr": "learning_rate",
        "max_train_items": "max_train_items",
        "max_eval_items": "max_eval_items",
        "max_per_label": "max_per_label",
        "image_cache_size": "image_cache_size",
        "class_weight_mode": "class_weight_mode",
        "seed": "seed",
    }
    for attr, key in mapping.items():
        if key in config:
            setattr(args, attr, config[key])
    return args


def normalize_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def memory_audit(stage: str, device: torch.device) -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    audit: dict[str, Any] = {"stage": stage, "max_rss_kb": int(usage.ru_maxrss)}
    if device.type == "cuda":
        audit["cuda_peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        audit["cuda_peak_reserved_mb"] = torch.cuda.max_memory_reserved(device) / (1024 * 1024)
    return audit


if __name__ == "__main__":
    main()
