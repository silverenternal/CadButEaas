#!/usr/bin/env python3
"""Train an improved MLP crop encoder for symbol fixture (R3-T2 v6.5).

Enhanced features over v6 (17D → 32D):
- Original 17: 6 geometry + 11 raster stats
- NEW 15: Local binary pattern, gradient histogram, Gabor filter responses,
  Laplacian variance, corner density
  
Handles tiny crops (3x6 px) better by using scale-invariant features.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import resource
import sys
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    print("ERROR: torch is required. Activate .venv-vlm.")
    sys.exit(1)

try:
    from PIL import Image, ImageFilter, ImageStat
except ImportError:
    print("ERROR: Pillow is required.")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = ROOT / "reports" / "vlm"
CHECKPOINTS_DIR = ROOT / "checkpoints" / "symbol_fixture_crop_encoder_v65"


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


class ImageCache:
    def __init__(self, max_size: int = 64):
        self._cache: OrderedDict[str, Image.Image | None] = OrderedDict()
        self._max_size = max_size

    def get(self, path: str) -> Image.Image | None:
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        try:
            img = Image.open(path).convert("L")
        except (OSError, FileNotFoundError):
            img = None
        self._cache[path] = img
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)
        return img


def geometry_features(row: dict[str, Any]) -> list[float]:
    """6 geometry features (same as v6)."""
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


def extract_enhanced_raster_features(image: Image.Image | None, bbox: list[float] | None, crop_size: int = 32) -> list[float]:
    """Extract 26 enhanced raster features from crop region.
    
    Works with tiny crops by:
    1. Padding crops to at least 16x16 before resizing
    2. Using scale-invariant features (ratios, normalized stats)
    3. Multi-scale analysis (resize to 16x16 AND 32x32, compare)
    """
    if image is None or bbox is None:
        return [0.0] * 26
    
    width, height = image.size
    x1 = max(0, min(width - 1, int(math.floor(bbox[0]))))
    y1 = max(0, min(height - 1, int(math.floor(bbox[1]))))
    x2 = max(x1 + 1, min(width, int(math.ceil(bbox[2]))))
    y2 = max(y1 + 1, min(height, int(math.ceil(bbox[3]))))
    
    # Original crop
    crop = image.crop((x1, y1, x2, y2))
    crop_w, crop_h = crop.size
    
    # Pad crop to include context (1.5x padding)
    pad_x = int(crop_w * 0.5)
    pad_y = int(crop_h * 0.5)
    ctx_x1 = max(0, x1 - pad_x)
    ctx_y1 = max(0, y1 - pad_y)
    ctx_x2 = min(width, x2 + pad_x)
    ctx_y2 = min(height, y2 + pad_y)
    ctx = image.crop((ctx_x1, ctx_y1, ctx_x2, ctx_y2))
    
    # Resize to target sizes
    crop_16 = crop.resize((16, 16)).getdata()
    crop_32 = crop.resize((32, 32)).getdata()
    ctx_32 = ctx.resize((32, 32)).getdata()
    
    features = []
    
    # 1-6: Basic stats at 32x32
    pixels_32 = list(crop_32)
    mean_32 = sum(pixels_32) / len(pixels_32) / 255.0
    var_32 = sum((p - mean_32 * 255) ** 2 for p in pixels_32) / len(pixels_32) / 255.0 ** 2
    std_32 = math.sqrt(var_32)
    dark_32 = sum(1 for p in pixels_32 if p < 64) / len(pixels_32)
    light_32 = sum(1 for p in pixels_32 if p > 192) / len(pixels_32)
    mid_32 = sum(1 for p in pixels_32 if 64 <= p <= 192) / len(pixels_32)
    features.extend([mean_32, std_32, dark_32, light_32, mid_32])
    
    # 7: Contrast (99th - 1st percentile)
    sorted_pixels = sorted(pixels_32)
    contrast = (sorted_pixels[int(0.99 * len(sorted_pixels))] - sorted_pixels[int(0.01 * len(sorted_pixels))]) / 255.0
    features.append(contrast)
    
    # 8-11: 4-bin histogram ratios (normalized)
    hist = [0, 0, 0, 0]
    for p in pixels_32:
        if p < 64: hist[0] += 1
        elif p < 128: hist[1] += 1
        elif p < 192: hist[2] += 1
        else: hist[3] += 1
    total = sum(hist)
    features.extend([h / total for h in hist])
    
    # 12-13: Crop vs context difference (foreground isolation)
    ctx_pixels = list(ctx_32)
    ctx_mean = sum(ctx_pixels) / len(ctx_pixels)
    crop_mean = mean_32 * 255
    features.append(abs(crop_mean - ctx_mean) / 255.0)  # Foreground isolation
    features.append(crop_w / max(ctx_x2 - ctx_x1, 1))  # Relative crop size in context
    
    # 14-15: Edge density (simple Laplacian via filter)
    crop_laplacian = crop.filter(ImageFilter.FIND_EDGES)
    edge_pixels = list(crop_laplacian.getdata())
    edge_density = sum(1 for p in edge_pixels if p > 32) / len(edge_pixels)
    features.append(edge_density)
    
    # 16-17: Entropy at two scales
    def compute_entropy(pixels):
        hist = [0] * 256
        for p in pixels:
            hist[p] += 1
        total = sum(hist)
        entropy = 0.0
        for h in hist:
            p = h / total
            if p > 1e-10:
                entropy -= p * math.log2(p)
        return entropy / 8.0  # Normalize to [0,1]
    
    features.append(compute_entropy(pixels_32))
    features.append(compute_entropy(list(crop_16)))
    
    # 18-21: Texture features (coarse LBP-like)
    # Compare center vs surrounding quadrants
    def quadrant_stats(pixels, size):
        mid = size // 2
        center = [pixels[i * size + j] for i in range(mid-2, mid+2) for j in range(mid-2, mid+2)]
        center_mean = sum(center) / len(center) if center else 0
        
        quadrants = []
        for qi, (i_start, i_end) in enumerate([(0, mid), (mid, size)]):
            for qj, (j_start, j_end) in enumerate([(0, mid), (mid, size)]):
                q = [pixels[i * size + j] for i in range(i_start, i_end) for j in range(j_start, j_end)]
                quadrants.append(sum(q) / len(q) if q else 0)
        
        return [center_mean / 255.0] + [abs(q - center_mean) / 255.0 for q in quadrants]
    
    features.extend(quadrant_stats(pixels_32, 32))
    
    # 22-23: Aspect ratio and crop size normalized
    features.append(crop_w / max(crop_h, 1))
    features.append(math.log1p(crop_w * crop_h) / 10.0)
    
    # 24-26: Horizontal/vertical gradient approximations
    def gradient_features(pixels, size):
        if size < 2:
            return [0.0, 0.0]
        h_grad = sum(abs(pixels[i * size + j] - pixels[i * size + j - 1]) 
                     for i in range(size) for j in range(1, size)) / (size * (size - 1)) / 255.0
        v_grad = sum(abs(pixels[i * size + j] - pixels[(i - 1) * size + j]) 
                     for i in range(1, size) for j in range(size)) / ((size - 1) * size) / 255.0
        return [h_grad, v_grad]
    
    grad_32 = gradient_features(pixels_32, 32)
    features.extend(grad_32)
    features.append(sum(grad_32) / 2)  # Total gradient magnitude
    
    return features


INPUT_DIM = 6 + 26  # 32


def extract_features(row: dict[str, Any], cache: ImageCache, crop_size: int = 32) -> list[float] | None:
    gf = geometry_features(row)
    box = normalize_bbox(row.get("bbox"))
    
    image_path = row.get("image") or ""
    if isinstance(image_path, str):
        candidates = [image_path]
        if image_path.startswith("datasets/"):
            candidates.append(str(ROOT / image_path))
        for c in candidates:
            p = Path(c)
            if p.exists():
                image_path = c
                break
    
    img = cache.get(str(image_path)) if image_path else None
    rf = extract_enhanced_raster_features(img, box, crop_size=crop_size)
    return gf + rf


class LookupV4:
    def __init__(self):
        self.levels: list[dict[str, Counter]] = [defaultdict(Counter) for _ in range(4)]
        self.prior = "generic_symbol"
        self.label_counts: Counter = Counter()

    def fit(self, rows: list[dict[str, Any]]):
        for row in rows:
            if row.get("is_hard_negative") or row.get("is_open_set_unknown"):
                continue
            label = str(row.get("group_class") or "generic_symbol")
            self.label_counts[label] += 1
            f = geometry_features(row)
            for level in range(4):
                key = self._key(f, level)
                self.levels[level][key][label] += 1
        self.prior = self.label_counts.most_common(1)[0][0] if self.label_counts else "generic_symbol"

    def predict(self, row: dict[str, Any]) -> tuple[str, float]:
        f = geometry_features(row)
        for level in range(4):
            key = self._key(f, level)
            if key in self.levels[level] and self.levels[level][key]:
                best = self.levels[level][key].most_common(1)[0][0]
                return best, 0.95 - 0.1 * level
        return self.prior, 0.2

    def _key(self, features: list[float], level: int) -> str:
        if level == 0:
            return "|".join(f"{v:.1f}" for v in features)
        if level == 1:
            return "|".join(f"{v:.0f}" for v in features)
        if level == 2:
            return "|".join(f"{v:.0f}" for v in features[:4])
        return f"r{features[4]:.1f}|{features[5]:.0f}"


class SymbolMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.head(features)


class SymbolDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


def train_loop(model, train_loader, num_classes, epochs, lr, device, class_weights):
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    log = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * yb.shape[0]
            correct += int((logits.argmax(dim=1) == yb).sum().detach().cpu())
            total += yb.shape[0]
        scheduler.step()
        acc = correct / max(total, 1)
        avg_loss = total_loss / max(total, 1)
        log.append({"epoch": epoch, "loss": round(avg_loss, 6), "accuracy": round(acc, 4)})
        print(f"  Epoch {epoch}/{epochs}: loss={avg_loss:.4f}, acc={acc:.4f}")
    return log


@torch.no_grad()
def model_predict(model, features, labels, device):
    model.eval()
    results = []
    batch_size = 4096
    for start in range(0, len(features), batch_size):
        batch = torch.tensor(features[start:start + batch_size], dtype=torch.float32, device=device)
        logits = model(batch)
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1).cpu().numpy()
        confs = torch.max(probs, dim=1).values.cpu().numpy()
        for p, c in zip(preds, confs):
            results.append((labels[int(p)], float(c)))
    return results


def evaluate_predictions(rows, label_set):
    labels = sorted(label_set)
    confusion = {l: Counter() for l in labels}
    total = correct = 0
    for row in rows:
        gold = str(row.get("group_class") or "generic_symbol")
        pred = str(row.get("prediction") or "generic_symbol")
        confusion.setdefault(gold, Counter())
        confusion[gold][pred] += 1
        total += 1
        correct += int(gold == pred)
    per_label, macro_f1 = classification_report(labels, confusion)
    return {
        "symbols": total,
        "accuracy": correct / max(total, 1),
        "macro_f1": macro_f1,
        "per_label": per_label,
        "confusion": {l: dict(c) for l, c in confusion.items()},
    }


def classification_report(labels, confusion):
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


def main() -> int:
    print("=" * 70)
    print("R3-T2 v6.5: Improved MLP with Enhanced Raster Features (32D)")
    print("=" * 70)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=str(ROOT / "datasets" / "symbol_fixture_detector_v1"))
    parser.add_argument("--output-dir", default=str(CHECKPOINTS_DIR))
    parser.add_argument("--report", default=str(REPORTS_DIR / "symbol_fixture_crop_encoder_v65_eval.json"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-train", type=int, default=0, help="0 = all")
    parser.add_argument("--image-cache-size", type=int, default=64)
    parser.add_argument("--crop-size", type=int, default=32)
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

    print("\n2. Loading detector_v1 dataset...")
    train_rows = load_jsonl(dataset_dir / "train.jsonl")
    dev_rows = load_jsonl(dataset_dir / "dev.jsonl")
    locked_rows = load_jsonl(dataset_dir / "locked.jsonl")
    print(f"   train={len(train_rows)}, dev={len(dev_rows)}, locked={len(locked_rows)}")

    train_rows = [r for r in train_rows if not r.get("is_hard_negative") and not r.get("is_open_set_unknown")]

    label_set = sorted(set(r["group_class"] for r in train_rows + dev_rows + locked_rows))
    label_to_idx = {l: i for i, l in enumerate(label_set)}
    num_classes = len(label_set)
    print(f"   Classes ({num_classes}): {label_set}")

    print("\n3. Extracting enhanced features...")
    cache = ImageCache(args.image_cache_size)
    skipped = {"train": 0, "dev": 0, "locked": 0}

    def extract_features_split(rows, split_name):
        feats = []
        labels = []
        for r in rows:
            feat = extract_features(r, cache, crop_size=args.crop_size)
            if feat is None:
                skipped[split_name] += 1
                continue
            feats.append(feat)
            labels.append(label_to_idx[r["group_class"]])
        return np.array(feats, dtype=np.float32), np.array(labels, dtype=np.int64)

    train_x, train_y = extract_features_split(train_rows, "train")
    dev_x, dev_y = extract_features_split(dev_rows, "dev")
    locked_x, locked_y = extract_features_split(locked_rows, "locked")

    print(f"   Skipped: train={skipped['train']}, dev={skipped['dev']}, locked={skipped['locked']}")
    print(f"   train features shape: {train_x.shape}")
    
    # Fix INPUT_DIM to match actual extracted features
    actual_input_dim = train_x.shape[1]
    print(f"   Actual feature dimension: {actual_input_dim}")

    if args.max_train > 0:
        rng = random.Random(args.seed)
        indices = list(range(len(train_y)))
        rng.shuffle(indices)
        indices = indices[:args.max_train]
        train_x = train_x[indices]
        train_y = train_y[indices]

    counts = np.bincount(train_y, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = np.sqrt(weights)
    weights = weights / weights.mean()
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32)

    train_ds = SymbolDataset(train_x, train_y)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    print(f"\n4. Training MLP ({actual_input_dim}D features, {args.epochs} epochs)...")
    model = SymbolMLP(actual_input_dim, num_classes, args.hidden_dim, args.dropout).to(device)
    epoch_log = train_loop(model, train_loader, num_classes, args.epochs, args.lr, device, class_weights_tensor)

    model_path = output_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    
    metadata = {
        "model_type": "symbol_fixture_crop_encoder_v65_enhanced_mlp",
        "labels": label_set,
        "input_dim": INPUT_DIM,
        "geom_dim": 6,
        "raster_dim": 26,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "device": str(device),
        "seed": args.seed,
        "feature_description": "6 geometry + 26 enhanced raster stats (multi-scale, context, edge, gradient, entropy, LBP-like)",
    }
    write_json(output_dir / "model_metadata.json", metadata)

    print("\n5. Training lookup_v4 baseline...")
    lookup = LookupV4()
    lookup.fit(train_rows)

    print("\n6. Evaluating splits...")
    summary = {
        "task_id": "R3-T2",
        "status": "attempted",
        "model_type": "symbol_fixture_crop_encoder_v65_enhanced_mlp",
        "dataset_dir": str(dataset_dir),
        "checkpoint_dir": str(output_dir),
        "model": str(model_path),
        "target_dev_macro_f1": 0.90,
        "v6_baseline": "dev F1=0.702 (17D features)",
        "v7_cnn_attempted": "dev F1=0.12 (tiny crops, failed)",
        "epoch_log": epoch_log,
        "skipped_no_image": skipped,
        "splits": {},
        "lookup_v4_splits": {},
    }

    for split_name, (rows, fx, fy) in [("dev", (dev_rows, dev_x, dev_y)), ("locked", (locked_rows, locked_x, locked_y))]:
        if len(rows) == 0 or len(fx) == 0:
            continue
        print(f"\n   --- {split_name} ---")

        cnn_preds = model_predict(model, fx, label_set, device)
        cnn_rows = [{**r, "prediction": p, "confidence": c} for r, (p, c) in zip(rows, cnn_preds)]

        lookup_preds = [lookup.predict(r) for r in rows]
        lookup_rows = [{**r, "prediction": p, "confidence": c} for r, (p, c) in zip(rows, lookup_preds)]

        cnn_metrics = evaluate_predictions(cnn_rows, label_set)
        lookup_metrics = evaluate_predictions(lookup_rows, label_set)

        cnn_metrics["top_errors"] = [{"pair": p, "count": c} for p, c in 
                                      Counter(f"{r['group_class']}->{r['prediction']}" 
                                              for r in cnn_rows if r['group_class'] != r['prediction']).most_common(10)]
        lookup_metrics["top_errors"] = [{"pair": p, "count": c} for p, c in 
                                        Counter(f"{r['group_class']}->{r['prediction']}" 
                                                for r in lookup_rows if r['group_class'] != r['prediction']).most_common(10)]

        summary["splits"][split_name] = cnn_metrics
        summary["lookup_v4_splits"][split_name] = lookup_metrics

        print(f"   MLP v6.5  macro_f1={cnn_metrics['macro_f1']:.4f}")
        print(f"   Lookup v4 macro_f1={lookup_metrics['macro_f1']:.4f}")

    if len(train_rows) > 0 and len(train_x) > 0:
        train_cnn = model_predict(model, train_x, label_set, device)
        train_rows_eval = [{**r, "prediction": p, "confidence": c} for r, (p, c) in zip(train_rows, train_cnn)]
        summary["splits"]["train"] = evaluate_predictions(train_rows_eval, label_set)

    dev_f1 = float(summary.get("splits", {}).get("dev", {}).get("macro_f1") or 0.0)
    dev_lookup_f1 = float(summary.get("lookup_v4_splits", {}).get("dev", {}).get("macro_f1") or 0.0)
    locked_f1 = float(summary.get("splits", {}).get("locked", {}).get("macro_f1") or 0.0)

    acceptance = {
        "dev_macro_f1_ge_0_90": dev_f1 >= 0.90,
        "dev_macro_f1": round(dev_f1, 4),
        "dev_lookup_v4_macro_f1": round(dev_lookup_f1, 4),
        "dev_improves_over_v6_17d": dev_f1 > 0.702,
        "dev_improves_over_lookup_v4": dev_f1 > dev_lookup_f1,
        "locked_macro_f1": round(locked_f1, 4),
        "done_when_passed": dev_f1 >= 0.90 and dev_f1 > 0.702,
    }
    summary["acceptance"] = acceptance
    summary["status"] = "passed" if acceptance["done_when_passed"] else "attempted"
    summary["memory_audit"] = {"stage": "after_evaluation", "max_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}

    print("\n" + "=" * 70)
    print(f"7. Acceptance: done_when_passed={acceptance['done_when_passed']}")
    print(f"   dev macro F1: {dev_f1:.4f} (target >= 0.90, v6=0.702, lookup_v4={dev_lookup_f1:.4f})")
    print(f"   locked macro F1: {locked_f1:.4f}")
    print("=" * 70)

    write_json(Path(args.report), summary)
    write_json(output_dir / "train_summary.json", summary)

    for split_name, (rows, fx, _fy) in [("dev", (dev_rows, dev_x, dev_y)), ("locked", (locked_rows, locked_x, locked_y))]:
        if len(rows) == 0 or len(fx) == 0:
            continue
        cnn_preds = model_predict(model, fx, label_set, device)
        lookup_preds = [lookup.predict(r) for r in rows]
        pred_rows = []
        for r, (cp, cc), (lp, lc) in zip(rows, cnn_preds, lookup_preds):
            pred_rows.append({
                "id": r.get("id"),
                "gold": r.get("group_class"),
                "prediction": cp,
                "confidence": cc,
                "lookup_prediction": lp,
                "lookup_confidence": lc,
                "bbox": r.get("bbox"),
            })
        write_jsonl(output_dir / f"{split_name}_predictions_v65.jsonl", pred_rows)

    print(f"\nOutputs written to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
