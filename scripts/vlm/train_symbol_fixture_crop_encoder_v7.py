#!/usr/bin/env python3
"""Train a lightweight CNN crop encoder for symbol fixture classification (R3-T2 v7).

Uses actual crop pixels from CubiCasa5K PNG images instead of hand-crafted features.
Architecture: Lightweight ResNet-style CNN + geometry features → classification.

This replaces the MLP v6 (F1=0.702) with a visual CNN to reach the 0.90 target.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import resource
import sys
import time
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
except ImportError:
    print("ERROR: torch and torchvision are required. Activate .venv-vlm.")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow is required.")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = ROOT / "reports" / "vlm"
CHECKPOINTS_DIR = ROOT / "checkpoints" / "symbol_fixture_crop_encoder_v7"

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def memory_audit(stage: str) -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {"stage": stage, "max_rss_kb": int(usage.ru_maxrss)}


# ---------------------------------------------------------------------------
# Image loading with cache
# ---------------------------------------------------------------------------
class ImageCache:
    def __init__(self, max_size: int = 64):
        self._cache: OrderedDict[str, Image.Image | None] = OrderedDict()
        self._max_size = max_size

    def get(self, path: str) -> Image.Image | None:
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        try:
            img = Image.open(path).convert("RGB")
        except (OSError, FileNotFoundError):
            img = None
        self._cache[path] = img
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)
        return img


# ---------------------------------------------------------------------------
# Geometry features (concatenated with CNN features)
# ---------------------------------------------------------------------------
def geometry_features(row: dict[str, Any]) -> list[float]:
    box = normalize_bbox(row.get("bbox"))
    if box is None:
        return [0.0] * 6
    x1, y1, x2, y2 = box
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    area = w * h
    aspect = max(w, h) / max(min(w, h), 1e-6)
    room_type = str(row.get("room_context") or "unknown_room")
    room_hash = hash(room_type) % 1000 / 1000.0
    return [
        math.log1p(w) / 10.0,
        math.log1p(h) / 10.0,
        math.log1p(area) / 10.0,
        math.log1p(aspect) / 5.0,
        room_hash,
        1.0 if (abs(x1) < 1e-6 and abs(y1) < 1e-6) else 0.0,
    ]


# ---------------------------------------------------------------------------
# Lightweight CNN for crop classification
# ---------------------------------------------------------------------------
class LightweightCropCNN(nn.Module):
    """Lightweight CNN for symbol crop classification.
    
    Architecture: 4 conv blocks + global average pooling + FC.
    Designed for small crops (64x64) with limited compute.
    """
    def __init__(self, num_classes: int, geom_dim: int = 6):
        super().__init__()
        # Conv blocks: 3→32→64→128→256
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64→32
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32→16
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16→8
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # 8→1
        )
        
        # Feature fusion: CNN (256) + geometry (6)
        self.classifier = nn.Sequential(
            nn.Linear(256 + geom_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, crops: torch.Tensor, geom: torch.Tensor) -> torch.Tensor:
        # crops: (B, 3, 64, 64)
        # geom: (B, 6)
        x = self.conv1(crops)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)  # (B, 256, 1, 1)
        cnn_features = x.squeeze(-1).squeeze(-1)  # (B, 256)
        
        # Concatenate with geometry
        fused = torch.cat([cnn_features, geom], dim=1)
        return self.classifier(fused)


class SymbolCropDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], label_to_idx: dict[str, int],
                 cache: ImageCache, crop_size: int = 64, augment: bool = False):
        self.rows = rows
        self.label_to_idx = label_to_idx
        self.cache = cache
        self.crop_size = crop_size
        self.augment = augment
        
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def __len__(self) -> int:
        return len(self.rows)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows[idx]
        
        # Load and crop image
        image_path = row.get("image", "")
        bbox = normalize_bbox(row.get("bbox"))
        
        crop = self._extract_crop(image_path, bbox)
        
        # Convert to tensor
        crop_tensor = self.normalize(transforms.ToTensor()(crop))
        
        # Geometry features
        geom = torch.tensor(geometry_features(row), dtype=torch.float32)
        
        # Label
        label = self.label_to_idx[row["group_class"]]
        
        return crop_tensor, geom, torch.tensor(label, dtype=torch.long)
    
    def _extract_crop(self, image_path: str, bbox: list[float] | None) -> Image.Image:
        """Extract crop from image, with padding if bbox is out of bounds."""
        if bbox is None:
            return Image.new("RGB", (self.crop_size, self.crop_size), (128, 128, 128))
        
        # Resolve path
        path = Path(image_path)
        if not path.is_absolute():
            path = ROOT / image_path
        
        img = self.cache.get(str(path))
        if img is None:
            return Image.new("RGB", (self.crop_size, self.crop_size), (128, 128, 128))
        
        img_w, img_h = img.size
        x1 = max(0, min(img_w - 1, int(math.floor(bbox[0]))))
        y1 = max(0, min(img_h - 1, int(math.floor(bbox[1]))))
        x2 = max(x1 + 1, min(img_w, int(math.ceil(bbox[2]))))
        y2 = max(y1 + 1, min(img_h, int(math.ceil(bbox[3]))))
        
        crop = img.crop((x1, y1, x2, y2))
        
        # Add padding if crop is very small
        if crop.size[0] < 8 or crop.size[1] < 8:
            padded = Image.new("RGB", (max(crop.size[0], 16), max(crop.size[1], 16)), (128, 128, 128))
            padded.paste(crop, (0, 0))
            crop = padded
        
        # Resize to target size
        crop = crop.resize((self.crop_size, self.crop_size), Image.BILINEAR)
        
        # Data augmentation
        if self.augment:
            if random.random() > 0.5:
                crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                crop = crop.rotate(angle, resample=Image.BILINEAR, fillcolor=(128, 128, 128))
        
        return crop


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                optimizer: torch.optim.Optimizer, device: torch.device) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for crops, geom, labels in loader:
        crops, geom, labels = crops.to(device), geom.to(device), labels.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        logits = model(crops, geom)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += float(loss.detach().cpu()) * labels.shape[0]
        correct += int((logits.argmax(dim=1) == labels).sum().detach().cpu())
        total += labels.shape[0]
    
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             label_set: list[str]) -> dict[str, Any]:
    model.eval()
    labels = sorted(label_set)
    confusion = {l: Counter() for l in labels}
    total = 0
    correct = 0
    
    all_preds = []
    all_confs = []
    all_golds = []
    
    for crops, geom, label_ids in loader:
        crops, geom = crops.to(device), geom.to(device)
        logits = model(crops, geom)
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1).cpu().numpy()
        confs = torch.max(probs, dim=1).values.cpu().numpy()
        golds = label_ids.numpy()
        
        for p, c, g in zip(preds, confs, golds):
            pred_label = labels[int(p)]
            gold_label = labels[int(g)]
            confusion.setdefault(gold_label, Counter())
            confusion[gold_label][pred_label] += 1
            total += 1
            correct += int(pred_label == gold_label)
            all_preds.append(pred_label)
            all_confs.append(float(c))
            all_golds.append(gold_label)
    
    per_label, macro_f1 = _classification_report(labels, confusion)
    
    return {
        "symbols": total,
        "accuracy": correct / max(total, 1),
        "macro_f1": macro_f1,
        "per_label": per_label,
        "confusion": {l: dict(c) for l, c in confusion.items()},
        "predictions": all_preds,
        "confidences": all_confs,
        "golds": all_golds,
    }


def _classification_report(labels: list[str], confusion: dict[str, Counter]) -> tuple[dict[str, Any], float]:
    per_label = {}
    f1s = []
    for label in labels:
        tp = confusion.get(label, Counter()).get(label, 0)
        fp = sum(confusion.get(o, Counter()).get(label, 0) for o in labels if o != label)
        fn = sum(c for p, c in confusion.get(label, Counter()).items() if p != label)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        f1s.append(f1)
        per_label[label] = {"precision": prec, "recall": rec, "f1": f1, "support": sum(confusion[label].values())}
    return per_label, sum(f1s) / max(len(f1s), 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 70)
    print("R3-T2 v7: Train Symbol Fixture CNN Crop Encoder")
    print("=" * 70)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=str(ROOT / "datasets" / "symbol_fixture_detector_v1"))
    parser.add_argument("--output-dir", default=str(CHECKPOINTS_DIR))
    parser.add_argument("--report", default=str(REPORTS_DIR / "symbol_fixture_crop_encoder_v7_eval.json"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-train", type=int, default=0, help="0 = all")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n1. Device: {device}")

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Load data
    print("\n2. Loading detector_v1 dataset...")
    train_rows = load_jsonl(dataset_dir / "train.jsonl")
    dev_rows = load_jsonl(dataset_dir / "dev.jsonl")
    locked_rows = load_jsonl(dataset_dir / "locked.jsonl")
    print(f"   train={len(train_rows)}, dev={len(dev_rows)}, locked={len(locked_rows)}")

    # Filter hard negatives and unknowns from training
    train_rows = [r for r in train_rows if not r.get("is_hard_negative") and not r.get("is_open_set_unknown")]

    label_set = sorted(set(r["group_class"] for r in train_rows + dev_rows + locked_rows))
    label_to_idx = {l: i for i, l in enumerate(label_set)}
    num_classes = len(label_set)
    print(f"   Classes ({num_classes}): {label_set}")

    if args.max_train > 0:
        rng = random.Random(args.seed)
        indices = list(range(len(train_rows)))
        rng.shuffle(indices)
        train_rows = [train_rows[i] for i in indices[:args.max_train]]

    # 3. Create datasets and loaders
    print("\n3. Creating datasets and loaders...")
    cache = ImageCache(max_size=128)
    
    train_ds = SymbolCropDataset(train_rows, label_to_idx, cache, args.crop_size, augment=True)
    dev_ds = SymbolCropDataset(dev_rows, label_to_idx, cache, args.crop_size, augment=False)
    locked_ds = SymbolCropDataset(locked_rows, label_to_idx, cache, args.crop_size, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    locked_loader = DataLoader(locked_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    print(f"   train={len(train_ds)}, dev={len(dev_ds)}, locked={len(locked_ds)}")

    # 4. Class weights
    counts = np.zeros(num_classes, dtype=np.float32)
    for r in train_rows:
        counts[label_to_idx[r["group_class"]]] += 1
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = np.sqrt(weights)
    weights = weights / weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    # 5. Create model
    print(f"\n4. Creating CNN model...")
    model = LightweightCropCNN(num_classes, geom_dim=6).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")

    # 6. Train
    print(f"\n5. Training ({args.epochs} epochs)...")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_dev_f1 = 0.0
    epoch_log = []
    t0 = time.time()
    
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        
        # Evaluate on dev
        dev_metrics = evaluate(model, dev_loader, device, label_set)
        
        log_entry = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 4),
            "train_acc": round(train_metrics["accuracy"], 4),
            "dev_f1": round(dev_metrics["macro_f1"], 4),
            "dev_acc": round(dev_metrics["accuracy"], 4),
        }
        epoch_log.append(log_entry)
        
        if dev_metrics["macro_f1"] > best_dev_f1:
            best_dev_f1 = dev_metrics["macro_f1"]
            torch.save(model.state_dict(), output_dir / "model_best.pt")
        
        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch}/{args.epochs} ({elapsed:.0f}s): "
                  f"train_acc={train_metrics['accuracy']:.4f}, dev_f1={dev_metrics['macro_f1']:.4f}, "
                  f"best_dev_f1={best_dev_f1:.4f}")
    
    # Load best model
    model.load_state_dict(torch.load(output_dir / "model_best.pt", weights_only=True))

    # 7. Final evaluation
    print("\n6. Final evaluation...")
    summary = {
        "task_id": "R3-T2",
        "status": "attempted",
        "model_type": "symbol_fixture_crop_encoder_v7_cnn",
        "dataset_dir": str(dataset_dir),
        "checkpoint_dir": str(output_dir),
        "model": str(output_dir / "model_best.pt"),
        "total_params": total_params,
        "target_dev_macro_f1": 0.90,
        "v6_mlp_baseline": "dev F1=0.702",
        "epoch_log": epoch_log,
        "splits": {},
    }
    
    for split_name, loader in [("dev", dev_loader), ("locked", locked_loader), ("train", train_loader)]:
        print(f"\n   --- {split_name} ---")
        metrics = evaluate(model, loader, device, label_set)
        
        # Add top errors
        errors = Counter()
        for pred, gold in zip(metrics["predictions"], metrics["golds"]):
            if pred != gold:
                errors[f"{gold}->{pred}"] += 1
        metrics["top_errors"] = [{"pair": p, "count": c} for p, c in errors.most_common(10)]
        
        # Remove large arrays from summary
        del metrics["predictions"]
        del metrics["confidences"]
        del metrics["golds"]
        
        summary["splits"][split_name] = metrics
        print(f"   macro_f1={metrics['macro_f1']:.4f}, accuracy={metrics['accuracy']:.4f}")

    # 8. Acceptance criteria
    dev_f1 = summary.get("splits", {}).get("dev", {}).get("macro_f1", 0.0)
    locked_f1 = summary.get("splits", {}).get("locked", {}).get("macro_f1", 0.0)
    
    acceptance = {
        "dev_macro_f1_ge_0_90": dev_f1 >= 0.90,
        "dev_macro_f1": round(dev_f1, 4),
        "v6_mlp_baseline": 0.702,
        "dev_improves_over_v6": dev_f1 > 0.702,
        "locked_macro_f1": round(locked_f1, 4),
        "done_when_passed": dev_f1 >= 0.90 and dev_f1 > 0.702,
    }
    summary["acceptance"] = acceptance
    summary["status"] = "passed" if acceptance["done_when_passed"] else "attempted"
    summary["memory_audit"] = memory_audit("after_evaluation")

    print("\n" + "=" * 70)
    print(f"7. Acceptance: done_when_passed={acceptance['done_when_passed']}")
    print(f"   dev macro F1: {dev_f1:.4f} (target >= 0.90, v6_mlp=0.702)")
    print(f"   locked macro F1: {locked_f1:.4f}")
    print("=" * 70)

    # 9. Write outputs
    write_json(Path(args.report), summary)
    write_json(output_dir / "train_summary.json", summary)
    
    # Write metadata
    metadata = {
        "model_type": "symbol_fixture_crop_encoder_v7_cnn",
        "labels": label_set,
        "crop_size": args.crop_size,
        "geom_dim": 6,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "device": str(device),
        "seed": args.seed,
        "feature_description": "Lightweight CNN (3→32→64→128→256) + 6 geometry features",
    }
    write_json(output_dir / "model_metadata.json", metadata)

    print(f"\nOutputs written:")
    print(f"  {output_dir / 'train_summary.json'}")
    print(f"  {args.report}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
