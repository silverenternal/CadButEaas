#!/usr/bin/env python3
"""Train a lightweight crop encoder for symbol fixture detection with REAL crop images (R3-T2 v6).

Replaces simulated crop features with actual raster statistics from PNG floorplan images.
Uses the symbol_fixture_detector_v1 dataset which references CubiCasa5K PNG images.

Architecture: MLP with real crop raster stats + geometry features.
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
    print("ERROR: torch is required. Activate .venv-vlm: source .venv-vlm/bin/activate")
    sys.exit(1)

try:
    from PIL import Image, ImageStat
except ImportError:
    print("ERROR: Pillow is required. pip install Pillow")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = ROOT / "reports" / "vlm"
CHECKPOINTS_DIR = ROOT / "checkpoints" / "symbol_fixture_crop_encoder_v6"

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
    return {"stage": stage, "max_rss_kb": int(usage.ru_maxrss), "note": "ru_maxrss is KiB on Linux."}


# ---------------------------------------------------------------------------
# Image loading with cache
# ---------------------------------------------------------------------------
class ImageCache:
    def __init__(self, max_size: int = 16):
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


# ---------------------------------------------------------------------------
# Real crop raster features
# ---------------------------------------------------------------------------
def crop_raster_stats(image: Image.Image | None, bbox: list[float], crop_size: int = 64) -> list[float]:
    """Extract 12 raster statistics from a cropped bbox region.

    Features: mean, std, dark_ratio, light_ratio, w_norm, h_norm,
    + entropy, edge_density, 4-bin intensity histogram ratios.
    """
    if image is None or bbox is None:
        return [0.0] * 12
    width, height = image.size
    x1 = max(0, min(width - 1, int(math.floor(bbox[0]))))
    y1 = max(0, min(height - 1, int(math.floor(bbox[1]))))
    x2 = max(x1 + 1, min(width, int(math.ceil(bbox[2]))))
    y2 = max(y1 + 1, min(height, int(math.ceil(bbox[3]))))
    crop = image.crop((x1, y1, x2, y2)).resize((crop_size, crop_size))
    stat = ImageStat.Stat(crop)
    mean_val = float(stat.mean[0]) / 255.0
    std_val = float(stat.stddev[0]) / max(255.0, 1e-6)
    pixels = list(crop.getdata())
    dark_ratio = sum(1 for p in pixels if int(p) < 96) / max(len(pixels), 1)
    light_ratio = sum(1 for p in pixels if int(p) > 224) / max(len(pixels), 1)
    w_norm = (x2 - x1) / max(width, 1)
    h_norm = (y2 - y1) / max(height, 1)

    # Entropy
    hist = crop.histogram()
    total = sum(hist)
    entropy = 0.0
    for h_val in hist:
        p = h_val / max(total, 1)
        if p > 1e-10:
            entropy -= p * math.log2(p)
    entropy = entropy / max(math.log2(256), 1e-6)  # normalize to [0,1]

    # 4-bin intensity histogram ratios
    n_bins = len(hist)
    q1 = sum(hist[:n_bins // 4]) / max(total, 1)
    q2 = sum(hist[n_bins // 4:n_bins // 2]) / max(total, 1)
    q3 = sum(hist[n_bins // 2:3 * n_bins // 4]) / max(total, 1)
    q4 = sum(hist[3 * n_bins // 4:]) / max(total, 1)

    return [mean_val, std_val, dark_ratio, light_ratio, w_norm, h_norm,
            entropy, q1, q2, q3, q4]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
GEOM_DIM = 6
RASTER_DIM = 11  # mean, std, dark, light, w_norm, h_norm, entropy, 4 hist bins
INPUT_DIM = GEOM_DIM + RASTER_DIM  # 17


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


def extract_features(row: dict[str, Any], cache: ImageCache, crop_size: int = 64) -> list[float] | None:
    """Extract 18D feature vector: 6 geometry + 12 real raster crop stats."""
    gf = geometry_features(row)
    box = normalize_bbox(row.get("bbox"))

    # Resolve image path
    image_path = row.get("image") or ""
    if isinstance(image_path, str):
        # Try multiple path conventions
        candidates = [image_path]
        # If path starts with datasets/external, try relative to ROOT
        if image_path.startswith("datasets/"):
            candidates.append(str(ROOT / image_path))
        # Try direct path
        for c in candidates:
            p = Path(c)
            if p.exists():
                image_path = c
                break

    img = cache.get(str(image_path)) if image_path else None
    rf = crop_raster_stats(img, box, crop_size=crop_size)
    return gf + rf


# ---------------------------------------------------------------------------
# Lookup-v4 baseline
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# MLP model
# ---------------------------------------------------------------------------
class SymbolCropMLP(nn.Module):
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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_loop(
    model: SymbolCropMLP,
    train_loader: DataLoader,
    num_classes: int,
    epochs: int,
    lr: float,
    device: torch.device,
    class_weights: torch.Tensor | None,
) -> list[dict[str, float]]:
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
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
        acc = correct / max(total, 1)
        avg_loss = total_loss / max(total, 1)
        log.append({"epoch": epoch, "loss": round(avg_loss, 6), "accuracy": round(acc, 4)})
        print(f"  Epoch {epoch}/{epochs}: loss={avg_loss:.4f}, acc={acc:.4f}")
    return log


@torch.no_grad()
def model_predict(model: SymbolCropMLP, features: np.ndarray, labels: list[str], device: torch.device) -> list[tuple[str, float]]:
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


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_predictions(rows: list[dict[str, Any]], label_set: list[str]) -> dict[str, Any]:
    labels = sorted(label_set)
    confusion = {l: Counter() for l in labels}
    total = correct = 0
    for row in rows:
        gold = str(row.get("group_class") or "generic_symbol")
        pred = str(row.get("prediction") or "generic_symbol")
        confusion.setdefault(gold, Counter())
        confusion.setdefault(pred, Counter())
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


def classification_report(labels: list[str], confusion: dict[str, Counter]) -> tuple[dict[str, Any], float]:
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


def error_pair_counts(rows: list[dict[str, Any]]) -> Counter:
    pairs = Counter()
    for row in rows:
        gold = str(row.get("group_class") or "generic_symbol")
        pred = str(row.get("prediction") or "generic_symbol")
        if gold != pred:
            pairs[f"{gold}->{pred}"] += 1
    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 70)
    print("R3-T2 v6: Train Symbol Fixture Crop Encoder with REAL raster crop features")
    print("=" * 70)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=str(ROOT / "datasets" / "symbol_fixture_detector_v1"))
    parser.add_argument("--output-dir", default=str(CHECKPOINTS_DIR))
    parser.add_argument("--report", default=str(REPORTS_DIR / "symbol_fixture_crop_encoder_v6_eval.json"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-train", type=int, default=0, help="0 = all")
    parser.add_argument("--image-cache-size", type=int, default=64)
    parser.add_argument("--crop-size", type=int, default=64, help="Crop resolution for raster stats")
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

    # Filter negatives
    train_rows = [r for r in train_rows if not r.get("is_hard_negative") and not r.get("is_open_set_unknown")]

    label_set = sorted(set(r["group_class"] for r in train_rows + dev_rows + locked_rows))
    label_to_idx = {l: i for i, l in enumerate(label_set)}
    num_classes = len(label_set)
    print(f"   Classes ({num_classes}): {label_set}")

    # 3. Extract features with REAL crop images
    print("\n3. Extracting features with REAL raster crop stats...")
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

    print(f"   Skipped (no image): train={skipped['train']}, dev={skipped['dev']}, locked={skipped['locked']}")
    print(f"   train features shape: {train_x.shape}")

    if args.max_train > 0:
        rng = random.Random(args.seed)
        indices = list(range(len(train_y)))
        rng.shuffle(indices)
        indices = indices[:args.max_train]
        train_x = train_x[indices]
        train_y = train_y[indices]

    # 4. Class weights
    counts = np.bincount(train_y, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = np.sqrt(weights)
    weights = weights / weights.mean()
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32)

    # 5. Data loaders
    train_ds = SymbolDataset(train_x, train_y)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    # 6. Train MLP
    print(f"\n4. Training MLP ({args.epochs} epochs, hidden={args.hidden_dim}, dropout={args.dropout})...")
    model = SymbolCropMLP(INPUT_DIM, num_classes, args.hidden_dim, args.dropout).to(device)
    epoch_log = train_loop(model, train_loader, num_classes, args.epochs, args.lr, device, class_weights_tensor)

    model_path = output_dir / "model.pt"
    torch.save(model.state_dict(), model_path)

    metadata = {
        "model_type": "symbol_fixture_crop_encoder_v6_real_raster",
        "labels": label_set,
        "input_dim": INPUT_DIM,
        "geom_dim": GEOM_DIM,
        "raster_dim": RASTER_DIM,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "device": str(device),
        "seed": args.seed,
        "feature_description": f"6 geometry + 12 real raster crop stats (mean, std, dark_ratio, light_ratio, w_norm, h_norm, entropy, 4-bin hist), crop_size={args.crop_size}",
    }
    write_json(output_dir / "model_metadata.json", metadata)

    # 7. Train lookup_v4 baseline
    print("\n5. Training lookup_v4 baseline...")
    lookup = LookupV4()
    lookup.fit(train_rows)

    # 8. Evaluate splits
    print("\n6. Evaluating splits...")
    summary: dict[str, Any] = {
        "task_id": "R3-T2",
        "status": "attempted",
        "model_type": "symbol_fixture_crop_encoder_v6_real_raster",
        "dataset_dir": str(dataset_dir),
        "checkpoint_dir": str(output_dir),
        "model": str(model_path),
        "metadata": str(output_dir / "model_metadata.json"),
        "target_dev_macro_f1": 0.90,
        "baseline_v4_reference": "lookup_v4 (geometry-context quantized lookup)",
        "v5_baseline": "simulated_crop_features, dev F1=0.697",
        "epoch_log": epoch_log,
        "skipped_no_image": skipped,
        "splits": {},
        "lookup_v4_splits": {},
    }

    for split_name, (rows, fx, fy) in [("dev", (dev_rows, dev_x, dev_y)), ("locked", (locked_rows, locked_x, locked_y))]:
        if len(rows) == 0 or len(fx) == 0:
            continue
        print(f"\n   --- {split_name} ---")

        # MLP predictions
        cnn_preds = model_predict(model, fx, label_set, device)
        cnn_rows = [{**r, "prediction": p, "confidence": c} for r, (p, c) in zip(rows, cnn_preds)]

        # Lookup-v4 predictions
        lookup_preds = [lookup.predict(r) for r in rows]
        lookup_rows = [{**r, "prediction": p, "confidence": c} for r, (p, c) in zip(rows, lookup_preds)]

        cnn_metrics = evaluate_predictions(cnn_rows, label_set)
        lookup_metrics = evaluate_predictions(lookup_rows, label_set)

        cnn_metrics["top_errors"] = [{"pair": p, "count": c} for p, c in error_pair_counts(cnn_rows).most_common(10)]
        lookup_metrics["top_errors"] = [{"pair": p, "count": c} for p, c in error_pair_counts(lookup_rows).most_common(10)]

        summary["splits"][split_name] = cnn_metrics
        summary["lookup_v4_splits"][split_name] = lookup_metrics

        print(f"   MLP v6    macro_f1={cnn_metrics['macro_f1']:.4f}")
        print(f"   Lookup v4 macro_f1={lookup_metrics['macro_f1']:.4f}")

    # Train eval
    if len(train_rows) > 0 and len(train_x) > 0:
        train_cnn = model_predict(model, train_x, label_set, device)
        train_rows_eval = [{**r, "prediction": p, "confidence": c} for r, (p, c) in zip(train_rows, train_cnn)]
        summary["splits"]["train"] = evaluate_predictions(train_rows_eval, label_set)

    # 9. Acceptance criteria
    dev_f1 = float(summary.get("splits", {}).get("dev", {}).get("macro_f1") or 0.0)
    dev_lookup_f1 = float(summary.get("lookup_v4_splits", {}).get("dev", {}).get("macro_f1") or 0.0)
    locked_f1 = float(summary.get("splits", {}).get("locked", {}).get("macro_f1") or 0.0)
    locked_lookup_f1 = float(summary.get("lookup_v4_splits", {}).get("locked", {}).get("macro_f1") or 0.0)

    acceptance = {
        "dev_macro_f1_ge_0_90": dev_f1 >= 0.90,
        "dev_macro_f1": round(dev_f1, 4),
        "dev_lookup_v4_macro_f1": round(dev_lookup_f1, 4),
        "dev_improves_over_lookup_v4": dev_f1 > dev_lookup_f1,
        "dev_improves_over_v5_simulated": dev_f1 > 0.697,
        "locked_macro_f1": round(locked_f1, 4),
        "locked_lookup_v4_macro_f1": round(locked_lookup_f1, 4),
        "locked_improves_over_lookup_v4": locked_f1 > locked_lookup_f1,
        "done_when_passed": dev_f1 >= 0.90 and dev_f1 > dev_lookup_f1 and locked_f1 > locked_lookup_f1,
    }
    summary["acceptance"] = acceptance
    summary["status"] = "passed" if acceptance["done_when_passed"] else "attempted_not_passed"
    summary["memory_audit"] = memory_audit("after_evaluation")

    print("\n" + "=" * 70)
    print(f"7. Acceptance: done_when_passed={acceptance['done_when_passed']}")
    print(f"   dev macro F1: {dev_f1:.4f} (target >= 0.90, lookup_v4={dev_lookup_f1:.4f}, v5_simulated=0.697)")
    print(f"   locked macro F1: {locked_f1:.4f} (lookup_v4={locked_lookup_f1:.4f})")
    print("=" * 70)

    # 10. Write outputs
    write_json(Path(args.report), summary)
    write_json(output_dir / "train_summary.json", summary)

    # Write predictions
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
        write_jsonl(output_dir / f"{split_name}_predictions_v6.jsonl", pred_rows)

    print(f"\nOutputs written:")
    print(f"  {output_dir / 'train_summary.json'}")
    print(f"  {args.report}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
