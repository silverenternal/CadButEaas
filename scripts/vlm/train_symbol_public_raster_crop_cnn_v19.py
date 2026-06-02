#!/usr/bin/env python3
"""Train a light raster crop CNN for symbol type under oracle localization."""

from __future__ import annotations

import argparse
import json
import random
import resource
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets/symbol_expert_public_raster_v19"
REPORT = ROOT / "reports/vlm"
CHECKPOINT = ROOT / "checkpoints/symbol_public_raster_v19_crop_cnn"
LABELS = ["appliance", "bathtub", "column", "equipment", "generic_symbol", "shower", "sink", "stair", "table"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def flatten_items(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        image = str(row.get("image") or "")
        for index, target in enumerate((row.get("targets") or {}).get("boxes") or []):
            label = str(target.get("label") or "")
            bbox = target.get("bbox")
            if label not in LABELS or not isinstance(bbox, list) or len(bbox) != 4:
                continue
            out.append(
                {
                    "id": f"{row.get('id')}_symbol_{index}",
                    "row_id": row.get("id"),
                    "split": split,
                    "image": image,
                    "bbox": [int(v) for v in bbox],
                    "label": label,
                }
            )
    return out


def sample_balanced(items: list[dict[str, Any]], max_per_label: int | None, seed: int) -> list[dict[str, Any]]:
    if not max_per_label:
        return list(items)
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[str(item["label"])].append(item)
    out: list[dict[str, Any]] = []
    for label in LABELS:
        rows = grouped.get(label, [])
        rng.shuffle(rows)
        out.extend(rows[:max_per_label])
    rng.shuffle(out)
    return out


class CropDataset(Dataset):
    def __init__(self, items: list[dict[str, Any]], size: int, pad: int, label_to_id: dict[str, int]) -> None:
        self.items = items
        self.size = size
        self.pad = pad
        self.label_to_id = label_to_id

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.items[index]
        arr = load_crop(item["image"], item["bbox"], self.size, self.pad)
        x = torch.from_numpy(arr[None, :, :])
        y = torch.tensor(self.label_to_id[str(item["label"])], dtype=torch.long)
        return x, y


def load_crop(image_path: str, bbox: list[int], size: int, pad: int) -> np.ndarray:
    path = Path(image_path)
    with Image.open(path if path.is_absolute() else ROOT / path) as image:
        gray = image.convert("L")
        width, height = gray.size
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(width, x2 + pad)
        y2 = min(height, y2 + pad)
        if x2 <= x1 or y2 <= y1:
            crop = Image.new("L", (size, size), 255)
        else:
            crop = gray.crop((x1, y1, x2, y2))
            crop = ImageOps.autocontrast(crop)
        crop = crop.resize((size, size), Image.Resampling.BICUBIC)
    arr = 1.0 - (np.asarray(crop, dtype=np.float32) / 255.0)
    return arr


class SymbolCropCNN(torch.nn.Module):
    def __init__(self, classes: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Dropout(0.15),
            torch.nn.Linear(128, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def class_weights(items: list[dict[str, Any]], device: torch.device) -> torch.Tensor:
    counts = Counter(str(item["label"]) for item in items)
    values = []
    total = sum(counts.values())
    for label in LABELS:
        values.append((total / max(counts.get(label, 0), 1)) ** 0.5)
    weights = torch.tensor(values, dtype=torch.float32, device=device)
    return weights / weights.mean()


def train_epoch(model: torch.nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module, device: torch.device) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * int(y.numel())
        correct += int((logits.argmax(dim=1) == y).sum().detach().cpu())
        total += int(y.numel())
    return {"loss": round(total_loss / max(total, 1), 6), "accuracy": round(correct / max(total, 1), 6)}


def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[str], list[str]]:
    model.eval()
    gold: list[str] = []
    pred: list[str] = []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device, non_blocking=True))
            pred.extend(LABELS[int(i)] for i in logits.argmax(dim=1).detach().cpu().tolist())
            gold.extend(LABELS[int(i)] for i in y.tolist())
    return gold, pred


def metrics(gold: list[str], pred: list[str]) -> dict[str, Any]:
    confusion = {label: Counter() for label in LABELS}
    correct = 0
    for g, p in zip(gold, pred, strict=True):
        confusion[g][p] += 1
        correct += int(g == p)
    per_label = {}
    f1s = []
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in LABELS if other != label)
        fn = sum(v for k, v in confusion[label].items() if k != label)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_label[label] = {"precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6), "support": sum(confusion[label].values())}
        f1s.append(f1)
    return {
        "accuracy": round(correct / max(len(gold), 1), 6),
        "macro_f1": round(sum(f1s) / max(len(f1s), 1), 6),
        "per_label": per_label,
        "confusion": {label: dict(confusion[label]) for label in LABELS},
    }


def memory_audit(device: torch.device) -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    audit: dict[str, Any] = {"max_rss_kb": int(usage.ru_maxrss)}
    if device.type == "cuda":
        audit["cuda_peak_allocated_mb"] = round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 3)
        audit["cuda_peak_reserved_mb"] = round(torch.cuda.max_memory_reserved(device) / (1024 * 1024), 3)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA))
    parser.add_argument("--checkpoint-dir", default=str(CHECKPOINT))
    parser.add_argument("--eval-output", default=str(REPORT / "symbol_public_raster_v19_crop_cnn_type_eval.json"))
    parser.add_argument("--max-train-per-label", type=int, default=2500)
    parser.add_argument("--limit-dev", type=int, default=None)
    parser.add_argument("--limit-locked", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--size", type=int, default=48)
    parser.add_argument("--pad", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    data = Path(args.data)
    train_items = sample_balanced(flatten_items(load_jsonl(data / "train.jsonl"), "train"), args.max_train_per_label, args.seed)
    dev_items = flatten_items(load_jsonl(data / "dev.jsonl"), "dev")
    locked_items = flatten_items(load_jsonl(data / "locked.jsonl"), "locked")
    if args.limit_dev:
        dev_items = dev_items[: args.limit_dev]
    if args.limit_locked:
        locked_items = locked_items[: args.limit_locked]

    label_to_id = {label: i for i, label in enumerate(LABELS)}
    train_loader = DataLoader(CropDataset(train_items, args.size, args.pad, label_to_id), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    dev_loader = DataLoader(CropDataset(dev_items, args.size, args.pad, label_to_id), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    locked_loader = DataLoader(CropDataset(locked_items, args.size, args.pad, label_to_id), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    model = SymbolCropCNN(len(LABELS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights(train_items, device))
    epoch_log = []
    for epoch in range(1, args.epochs + 1):
        row = train_epoch(model, train_loader, optimizer, loss_fn, device)
        row["epoch"] = epoch
        epoch_log.append(row)

    dev_gold, dev_pred = predict(model, dev_loader, device)
    locked_gold, locked_pred = predict(model, locked_loader, device)
    dev_metrics = metrics(dev_gold, dev_pred)
    locked_metrics = metrics(locked_gold, locked_pred)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / "model.pt")
    metadata = {
        "labels": LABELS,
        "model_type": "symbol_public_raster_crop_cnn_v19",
        "runtime_boundary": "oracle_localization_type_head_audit_not_full_detector",
        "size": args.size,
        "pad": args.pad,
    }
    write_json(checkpoint_dir / "model_metadata.json", metadata)

    report = {
        "version": "symbol_public_raster_v19_crop_cnn_type_eval",
        "run_mode": "oracle_gold_box_raster_crop_cnn_type_classifier",
        "source_integrity": {
            "model_input": "raster_crop_only",
            "gold_bbox_use": "crop_oracle_for_type_head_audit",
            "raw_label_or_semantic_type_used_as_runtime_feature": False,
            "locked_gold_use": "crop_oracle_and_evaluation_only",
        },
        "dataset": str(data.relative_to(ROOT) if data.is_relative_to(ROOT) else data),
        "train_sampling": {
            "sampled_train_items": len(train_items),
            "max_train_per_label": args.max_train_per_label,
            "sampled_label_counts": dict(Counter(str(item["label"]) for item in train_items).most_common()),
        },
        "model": {"epochs": args.epochs, "batch_size": args.batch_size, "size": args.size, "pad": args.pad, "device": str(device)},
        "epoch_log": epoch_log,
        "dev_metrics": dev_metrics,
        "locked_metrics": locked_metrics,
        "memory_audit": memory_audit(device),
        "adopted": False,
        "adoption_note": "Oracle-localization type-head audit only; not a full detector.",
    }
    write_json(Path(args.eval_output), report)
    print(json.dumps({"dev_macro_f1": dev_metrics["macro_f1"], "locked_macro_f1": locked_metrics["macro_f1"], "checkpoint": str(checkpoint_dir / "model.pt")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
