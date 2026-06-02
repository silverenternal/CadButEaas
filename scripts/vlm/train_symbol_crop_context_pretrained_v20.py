#!/usr/bin/env python3
"""Train a pretrained shared crop/context encoder for symbol typing."""

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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets/symbol_crop_context_cache_v20"
CHECKPOINT = ROOT / "checkpoints/symbol_crop_context_pretrained_v20"
REPORT = ROOT / "reports/vlm/symbol_crop_context_pretrained_v20_eval.json"
LABELS = ["appliance", "bathtub", "column", "equipment", "generic_symbol", "shower", "sink", "stair", "table"]
CROP_VIEWS = ["tight", "padded", "context"]
FORBIDDEN_RUNTIME_FIELDS = ["raw_label", "semantic_type", "expected_json", "annotation_path", "svg_geometry"]
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def source_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def sample_balanced(items: list[dict[str, Any]], max_per_label: int | None, seed: int) -> list[dict[str, Any]]:
    if not max_per_label:
        return list(items)
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[str(item.get("label"))].append(item)
    out: list[dict[str, Any]] = []
    for label in LABELS:
        rows = grouped.get(label, [])
        rng.shuffle(rows)
        out.extend(rows[:max_per_label])
    rng.shuffle(out)
    return out


def limit_items(items: list[dict[str, Any]], limit: int | None, seed: int) -> list[dict[str, Any]]:
    if not limit or len(items) <= limit:
        return list(items)
    rng = random.Random(seed)
    rows = list(items)
    rng.shuffle(rows)
    return rows[:limit]


def geom_tensor(item: dict[str, Any]) -> torch.Tensor:
    geom = item.get("geometry") or {}
    bbox_norm = geom.get("bbox_norm") or [0.0, 0.0, 0.0, 0.0]
    center_norm = geom.get("center_norm") or [0.0, 0.0]
    values = [
        *bbox_norm[:4],
        *center_norm[:2],
        geom.get("width_norm", 0.0),
        geom.get("height_norm", 0.0),
        geom.get("area_norm", 0.0),
        geom.get("aspect_log", 0.0),
    ]
    return torch.tensor([float(v) for v in values], dtype=torch.float32)


def load_crop(path: str, size: int, augment: bool, rng: random.Random) -> np.ndarray:
    with Image.open(source_path(path)) as opened:
        crop = opened.convert("RGB")
    crop = ImageOps.autocontrast(crop)
    if augment:
        if rng.random() < 0.5:
            crop = ImageOps.mirror(crop)
        angle = rng.uniform(-3.0, 3.0)
        crop = crop.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=(255, 255, 255))
    crop = crop.resize((size, size), Image.Resampling.BICUBIC)
    arr = np.asarray(crop, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.transpose(2, 0, 1)


class CropContextDataset(Dataset):
    def __init__(self, items: list[dict[str, Any]], size: int, label_to_id: dict[str, int], augment: bool, seed: int) -> None:
        self.items = items
        self.size = size
        self.label_to_id = label_to_id
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        item = self.items[index]
        rng = random.Random(self.seed + index)
        crops = item.get("crops") or {}
        views = []
        for view in CROP_VIEWS:
            path = str((crops.get(view) or {}).get("path") or "")
            views.append(load_crop(path, self.size, self.augment, rng))
        x = torch.from_numpy(np.stack(views).astype(np.float32))
        y = torch.tensor(self.label_to_id[str(item.get("label"))], dtype=torch.long)
        return x, geom_tensor(item), y, str(item.get("id"))


def build_encoder(name: str, weights: str) -> tuple[torch.nn.Module, int]:
    use_weights = weights == "imagenet"
    if name == "convnext_tiny":
        weight_enum = models.ConvNeXt_Tiny_Weights.DEFAULT if use_weights else None
        model = models.convnext_tiny(weights=weight_enum)
        dim = int(model.classifier[-1].in_features)
        model.classifier = torch.nn.Identity()
        return model, dim
    weight_enum = models.ResNet18_Weights.DEFAULT if use_weights else None
    model = models.resnet18(weights=weight_enum)
    dim = int(model.fc.in_features)
    model.fc = torch.nn.Identity()
    return model, dim


class SharedPretrainedTypeHead(torch.nn.Module):
    def __init__(self, encoder_name: str, weights: str, classes: int, freeze_encoder: bool) -> None:
        super().__init__()
        self.encoder, dim = build_encoder(encoder_name, weights)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.geom = torch.nn.Sequential(torch.nn.Linear(10, 32), torch.nn.SiLU(), torch.nn.LayerNorm(32))
        self.head = torch.nn.Sequential(
            torch.nn.Dropout(0.25),
            torch.nn.Linear(dim * len(CROP_VIEWS) + 32, 512),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.15),
            torch.nn.Linear(512, 128),
            torch.nn.SiLU(),
            torch.nn.Linear(128, classes),
        )

    def forward(self, crops: torch.Tensor, geom: torch.Tensor) -> torch.Tensor:
        batch, views, channels, height, width = crops.shape
        encoded = self.encoder(crops.reshape(batch * views, channels, height, width))
        if encoded.ndim > 2:
            encoded = torch.flatten(encoded, 1)
        encoded = encoded.reshape(batch, views * encoded.shape[-1])
        return self.head(torch.cat([encoded, self.geom(geom)], dim=1))


def class_weights(items: list[dict[str, Any]], device: torch.device) -> torch.Tensor:
    counts = Counter(str(item.get("label")) for item in items)
    total = sum(counts.values())
    values = [(total / max(counts.get(label, 0), 1)) ** 0.3 for label in LABELS]
    weights = torch.tensor(values, dtype=torch.float32, device=device)
    return weights / weights.mean()


def balanced_sampler(items: list[dict[str, Any]]) -> WeightedRandomSampler:
    counts = Counter(str(item.get("label")) for item in items)
    weights = [1.0 / max(counts.get(str(item.get("label")), 0), 1) for item in items]
    return WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), num_samples=len(items), replacement=True)


def train_epoch(model: torch.nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module, device: torch.device) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0
    for crops, geom, y, _ids in loader:
        crops = crops.to(device, non_blocking=True)
        geom = geom.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(crops, geom)
        loss = loss_fn(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * int(y.numel())
        correct += int((logits.argmax(dim=1) == y).sum().detach().cpu())
        total += int(y.numel())
    return {"loss": round(total_loss / max(total, 1), 6), "accuracy": round(correct / max(total, 1), 6)}


def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[str], list[str], list[float]]:
    model.eval()
    gold: list[str] = []
    pred: list[str] = []
    confs: list[float] = []
    with torch.no_grad():
        for crops, geom, y, _ids in loader:
            logits = model(crops.to(device, non_blocking=True), geom.to(device, non_blocking=True))
            probs = torch.softmax(logits, dim=1)
            conf, index = probs.max(dim=1)
            pred.extend(LABELS[int(i)] for i in index.detach().cpu().tolist())
            confs.extend(float(v) for v in conf.detach().cpu().tolist())
            gold.extend(LABELS[int(i)] for i in y.tolist())
    return gold, pred, confs


def metrics(gold: list[str], pred: list[str]) -> dict[str, Any]:
    confusion = {label: Counter() for label in LABELS}
    correct = 0
    for g, p in zip(gold, pred, strict=True):
        confusion[g][p] += 1
        correct += int(g == p)
    per_label: dict[str, Any] = {}
    f1s = []
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in LABELS if other != label)
        fn = sum(value for key, value in confusion[label].items() if key != label)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_label[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": sum(confusion[label].values()),
        }
        f1s.append(f1)
    return {
        "accuracy": round(correct / max(len(gold), 1), 6),
        "macro_f1": round(sum(f1s) / max(len(f1s), 1), 6),
        "per_label": per_label,
        "confusion": {label: dict(confusion[label]) for label in LABELS},
    }


def abstain_metrics(gold: list[str], pred: list[str], confs: list[float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for threshold in [0.5, 0.7, 0.85, 0.95]:
        keep = [idx for idx, conf in enumerate(confs) if conf >= threshold]
        row = metrics([gold[idx] for idx in keep], [pred[idx] for idx in keep]) if keep else {"accuracy": 0.0, "macro_f1": 0.0}
        row.update({"threshold": threshold, "kept": len(keep), "total": len(gold), "abstain_rate": round(1.0 - len(keep) / max(len(gold), 1), 6)})
        rows.append(row)
    return rows


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
    parser.add_argument("--eval-output", default=str(REPORT))
    parser.add_argument("--max-train-per-label", type=int, default=3000)
    parser.add_argument("--limit-dev", type=int, default=None)
    parser.add_argument("--limit-locked", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument("--encoder", choices=["resnet18", "convnext_tiny"], default="resnet18")
    parser.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--unfreeze-encoder", action="store_true")
    parser.add_argument("--encoder-lr-mult", type=float, default=0.05)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    data_dir = Path(args.data)
    train_items = sample_balanced(load_jsonl(data_dir / "train.jsonl"), args.max_train_per_label, args.seed)
    dev_items = limit_items(load_jsonl(data_dir / "dev.jsonl"), args.limit_dev, args.seed + 1)
    locked_items = limit_items(load_jsonl(data_dir / "locked.jsonl"), args.limit_locked, args.seed + 2)

    label_to_id = {label: index for index, label in enumerate(LABELS)}
    train_loader = DataLoader(
        CropContextDataset(train_items, args.size, label_to_id, augment=not args.no_augment, seed=args.seed),
        batch_size=args.batch_size,
        sampler=balanced_sampler(train_items),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    dev_loader = DataLoader(
        CropContextDataset(dev_items, args.size, label_to_id, augment=False, seed=args.seed),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    locked_loader = DataLoader(
        CropContextDataset(locked_items, args.size, label_to_id, augment=False, seed=args.seed),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = SharedPretrainedTypeHead(args.encoder, args.weights, len(LABELS), freeze_encoder=not args.unfreeze_encoder).to(device)
    if args.init_checkpoint:
        state = torch.load(source_path(args.init_checkpoint), map_location="cpu")
        model.load_state_dict(state)
    if args.unfreeze_encoder:
        encoder_params = [param for param in model.encoder.parameters() if param.requires_grad]
        head_params = [
            param
            for name, param in model.named_parameters()
            if param.requires_grad and not name.startswith("encoder.")
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": args.lr * args.encoder_lr_mult},
                {"params": head_params, "lr": args.lr},
            ],
            weight_decay=1e-4,
        )
    else:
        optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=args.lr, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights(train_items, device), label_smoothing=0.03)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05)

    epoch_log: list[dict[str, Any]] = []
    best_dev = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, args.epochs + 1):
        row = train_epoch(model, train_loader, optimizer, loss_fn, device)
        scheduler.step()
        dev_gold, dev_pred, _dev_confs = predict(model, dev_loader, device)
        dev_metrics = metrics(dev_gold, dev_pred)
        row.update({"epoch": epoch, "dev_macro_f1": dev_metrics["macro_f1"], "dev_accuracy": dev_metrics["accuracy"]})
        epoch_log.append(row)
        if float(dev_metrics["macro_f1"]) > best_dev:
            best_dev = float(dev_metrics["macro_f1"])
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    dev_gold, dev_pred, dev_confs = predict(model, dev_loader, device)
    locked_gold, locked_pred, locked_confs = predict(model, locked_loader, device)
    dev_metrics = metrics(dev_gold, dev_pred)
    locked_metrics = metrics(locked_gold, locked_pred)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / "model.pt")
    write_json(
        checkpoint_dir / "model_metadata.json",
        {
            "model_type": "symbol_crop_context_pretrained_v20",
            "encoder": args.encoder,
            "weights": args.weights,
            "freeze_encoder": not args.unfreeze_encoder,
            "labels": LABELS,
            "crop_views": CROP_VIEWS,
            "size": args.size,
            "claim_boundary": "Oracle gold-box type-head audit only; not full detector performance.",
            "runtime_contract": {
                "allowed_model_inputs": ["crops.tight", "crops.padded", "crops.context", "geometry"],
                "forbidden_runtime_features": FORBIDDEN_RUNTIME_FIELDS,
            },
        },
    )

    report = {
        "version": "symbol_crop_context_pretrained_v20_eval",
        "claim_boundary": "Oracle gold-box type-head audit only. This is not full symbol detection performance.",
        "source_integrity": {
            "model_input": "cached_raster_crops_plus_bbox_geometry",
            "raw_label_or_semantic_type_used_as_runtime_feature": False,
            "forbidden_runtime_features": FORBIDDEN_RUNTIME_FIELDS,
            "locked_gold_use": "crop_oracle_and_evaluation_only",
        },
        "dataset": rel(data_dir),
        "checkpoint": rel(checkpoint_dir / "model.pt"),
        "config": vars(args) | {"device": str(device)},
        "counts": {
            "train": len(train_items),
            "dev": len(dev_items),
            "locked": len(locked_items),
            "train_label_counts": dict(Counter(str(item.get("label")) for item in train_items).most_common()),
            "dev_label_counts": dict(Counter(str(item.get("label")) for item in dev_items).most_common()),
            "locked_label_counts": dict(Counter(str(item.get("label")) for item in locked_items).most_common()),
        },
        "epoch_log": epoch_log,
        "dev": dev_metrics,
        "locked": locked_metrics,
        "abstain_sweeps": {"dev": abstain_metrics(dev_gold, dev_pred, dev_confs), "locked": abstain_metrics(locked_gold, locked_pred, locked_confs)},
        "baseline_comparison": {
            "previous_cached_tree_locked_macro_f1": 0.61441,
            "previous_from_scratch_cnn_locked_macro_f1": 0.47511,
            "previous_handcrafted_locked_macro_f1": 0.436258,
            "delta_vs_cached_tree": round(float(locked_metrics["macro_f1"]) - 0.61441, 6),
        },
        "gate": {
            "stage_1_min_type_macro_f1_0_65": float(locked_metrics["macro_f1"]) >= 0.65,
            "beats_cached_tree_baseline": float(locked_metrics["macro_f1"]) > 0.61441,
            "beats_from_scratch_cnn": float(locked_metrics["macro_f1"]) > 0.47511,
        },
        "memory_audit": memory_audit(device),
    }
    write_json(Path(args.eval_output), report)
    print(json.dumps({"dev_macro_f1": dev_metrics["macro_f1"], "locked_macro_f1": locked_metrics["macro_f1"], "checkpoint": rel(checkpoint_dir / "model.pt")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
