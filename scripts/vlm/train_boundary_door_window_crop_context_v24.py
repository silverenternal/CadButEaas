#!/usr/bin/env python3
"""Train a raster crop/context boundary type head and evaluate locked50."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_boundary_type_fusion_v24 import bbox, center_covered, gold_by_row, iou, load_jsonl  # noqa: E402


LABELS = ["hard_wall", "door", "window", "background"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}


class CropDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, rows: list[dict[str, str]], root: Path, augment: bool) -> None:
        self.rows = rows
        self.root = root
        aug = [
            transforms.RandomApply([transforms.ColorJitter(brightness=0.12, contrast=0.12)], p=0.35),
            transforms.RandomAffine(degrees=1.5, translate=(0.015, 0.015), scale=(0.96, 1.04), fill=255),
        ]
        self.transform = transforms.Compose(
            [
                *(aug if augment else []),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.92, 0.92, 0.92], std=[0.18, 0.18, 0.18]),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.rows[idx]
        with Image.open(self.root / row["crop_path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, LABEL_TO_ID[row["label"]]


class SmallBoundaryCNN(nn.Module):
    def __init__(self, num_classes: int = len(LABELS)) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 192, 3, padding=1),
            nn.BatchNorm2d(192),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.18), nn.Linear(192, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(backbone: str, pretrained: bool) -> nn.Module:
    if backbone == "small_cnn":
        return SmallBoundaryCNN()
    if backbone == "resnet18":
        weights = None
        if pretrained:
            try:
                weights = models.ResNet18_Weights.DEFAULT
            except Exception:
                weights = None
        model = models.resnet18(weights=weights)
        model.fc = nn.Sequential(nn.Dropout(0.18), nn.Linear(model.fc.in_features, len(LABELS)))
        return model
    raise ValueError(f"Unsupported backbone: {backbone}")


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_by_row(rows: list[dict[str, str]], val_ratio: float, seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    row_ids = sorted({row["row_id"] for row in rows})
    rng = random.Random(seed)
    rng.shuffle(row_ids)
    val_count = max(1, int(round(len(row_ids) * val_ratio)))
    val_ids = set(row_ids[:val_count])
    train = [row for row in rows if row["row_id"] not in val_ids]
    val = [row for row in rows if row["row_id"] in val_ids]
    return train, val


def class_weights(rows: list[dict[str, str]], device: torch.device) -> torch.Tensor:
    counts = Counter(row["label"] for row in rows)
    total = sum(counts.values())
    values = []
    for label in LABELS:
        values.append(total / max(counts[label], 1))
    weights = torch.tensor(values, dtype=torch.float32, device=device)
    return weights / weights.mean()


def run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * int(y.numel())
            total += int(y.numel())
            correct += int((logits.argmax(dim=1) == y).sum().item())
    return {"loss": round(total_loss / max(total, 1), 6), "accuracy": round(correct / max(total, 1), 6)}


@torch.no_grad()
def predict_manifest(
    model: nn.Module,
    rows: list[dict[str, str]],
    root: Path,
    batch_size: int,
    device: torch.device,
) -> dict[tuple[str, str], dict[str, Any]]:
    dataset = CropDataset(rows, root, augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()
    output: dict[tuple[str, str], dict[str, Any]] = {}
    offset = 0
    for x, _y in loader:
        probs = torch.softmax(model(x.to(device)), dim=1).cpu().numpy()
        for row, prob in zip(rows[offset : offset + len(probs)], probs, strict=True):
            pred_idx = int(np.argmax(prob))
            output[(row["row_id"], row["candidate_id"])] = {
                "visual_prediction": LABELS[pred_idx],
                "visual_confidence": round(float(prob[pred_idx]), 6),
                "visual_probabilities": {label: round(float(prob[idx]), 6) for idx, label in enumerate(LABELS)},
            }
        offset += len(probs)
    return output


def base_label(candidate: dict[str, Any]) -> str:
    value = str(candidate.get("fusion_prediction") or candidate.get("prediction") or "")
    return value if value in {"hard_wall", "door", "window"} else "hard_wall"


def fused_label(
    candidate: dict[str, Any],
    visual: dict[str, Any] | None,
    threshold: float,
    mode: str,
    door_threshold: float,
    window_threshold: float,
) -> tuple[str, dict[str, Any]]:
    base = base_label(candidate)
    trace = {"base_prediction": base, "visual_applied": False, "integration_mode": mode}
    if visual is None:
        return base, trace
    trace.update(visual)
    pred = str(visual["visual_prediction"])
    conf = float(visual["visual_confidence"])
    if mode == "overwrite" and pred in {"hard_wall", "door", "window"} and conf >= threshold:
        trace["visual_applied"] = True
        return pred, trace
    if mode == "fail_closed":
        class_threshold = door_threshold if pred == "door" else window_threshold if pred == "window" else threshold
        if base == "hard_wall" and pred in {"door", "window"} and conf >= class_threshold:
            trace["visual_applied"] = True
            trace["override_rule"] = "hard_wall_to_opening_only"
            return pred, trace
    return base, trace


def evaluate_page_level(
    prediction_rows: list[dict[str, Any]],
    visual_by_candidate: dict[tuple[str, str], dict[str, Any]],
    gold_path: Path,
    limit: int | None,
    cap: int,
    threshold: float,
    mode: str,
    door_threshold: float,
    window_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gold = gold_by_row(gold_path, limit)
    enriched_rows = []
    total = proposal_hit = classified_hit = predicted = 0
    per_label: dict[str, Counter[str]] = defaultdict(Counter)
    wrong_pairs = Counter()
    for row in prediction_rows[: limit or len(prediction_rows)]:
        row_id = str(row.get("id"))
        candidates = []
        pred_cache = []
        for candidate in (row.get("candidate_stream") or [])[:cap]:
            item = dict(candidate)
            key = (row_id, str(item.get("candidate_id")))
            label, trace = fused_label(item, visual_by_candidate.get(key), threshold, mode, door_threshold, window_threshold)
            item["visual_crop_context_trace"] = trace
            item["prediction"] = label
            candidates.append(item)
            pred_cache.append(label)
        copied = dict(row)
        copied["candidate_stream"] = candidates
        enriched_rows.append(copied)
        predicted += len(candidates)
        for gold_item in gold.get(row_id, []):
            total += 1
            label = gold_item["label"]
            per_label[label]["gold"] += 1
            matches = []
            for idx, candidate in enumerate(candidates):
                cb = bbox(candidate.get("bbox"))
                if cb is not None and (center_covered(cb, gold_item["bbox"]) or iou(cb, gold_item["bbox"]) >= 0.30):
                    matches.append(idx)
            if matches:
                proposal_hit += 1
                per_label[label]["proposal_matched"] += 1
            if any(pred_cache[idx] == label for idx in matches):
                classified_hit += 1
                per_label[label]["classified_matched"] += 1
            elif matches:
                wrong_pairs[f"{label}->{pred_cache[matches[0]]}"] += 1
    metrics = {
        "gold": total,
        "predicted": predicted,
        "candidate_inflation": round(predicted / max(total, 1), 6),
        "proposal_recall": round(proposal_hit / max(total, 1), 6),
        "classified_recall": round(classified_hit / max(total, 1), 6),
        "classified_precision_proxy": round(classified_hit / max(predicted, 1), 6),
        "per_label": {
            label: {
                "gold": counts["gold"],
                "proposal_matched": counts["proposal_matched"],
                "classified_matched": counts["classified_matched"],
                "proposal_recall": round(counts["proposal_matched"] / max(counts["gold"], 1), 6),
                "classified_recall": round(counts["classified_matched"] / max(counts["gold"], 1), 6),
            }
            for label, counts in sorted(per_label.items())
        },
        "wrong_pairs": dict(wrong_pairs),
    }
    return metrics, enriched_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop-root", default="datasets/boundary_door_window_crop_context_v24")
    parser.add_argument("--locked-predictions", default="reports/vlm/boundary_type_fusion_v24_locked50_predictions.jsonl")
    parser.add_argument("--dataset", default="datasets/boundary_expert_public_raster_v19")
    parser.add_argument("--output-dir", default="checkpoints/boundary_door_window_crop_context_v24")
    parser.add_argument("--eval-output", default="reports/vlm/boundary_door_window_crop_context_v24_locked50_eval.json")
    parser.add_argument("--predictions-output", default="reports/vlm/boundary_door_window_crop_context_v24_locked50_predictions.jsonl")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--backbone", choices=["small_cnn", "resnet18"], default="small_cnn")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--integration-mode", choices=["fail_closed", "overwrite"], default="fail_closed")
    parser.add_argument("--door-threshold", type=float, default=0.85)
    parser.add_argument("--window-threshold", type=float, default=0.85)
    parser.add_argument("--locked-limit", type=int, default=50)
    parser.add_argument("--cap", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20260511)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crop_root = ROOT / args.crop_root
    rows = read_manifest(crop_root / "dev50_train" / "manifest.csv")
    train_rows, val_rows = split_by_row(rows, args.val_ratio, args.seed)
    train_loader = DataLoader(CropDataset(train_rows, crop_root, augment=True), batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(CropDataset(val_rows, crop_root, augment=False), batch_size=args.batch_size, shuffle=False, num_workers=2)
    model = build_model(args.backbone, args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_rows, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    history = []
    best_state = None
    best_val = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer, criterion)
        val_metrics = run_epoch(model, val_loader, device, None, criterion)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        if val_metrics["accuracy"] > best_val:
            best_val = val_metrics["accuracy"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "labels": LABELS,
            "threshold": args.threshold,
            "backbone": args.backbone,
            "pretrained": bool(args.pretrained),
            "history": history,
            "source": str(crop_root / "dev50_train" / "manifest.csv"),
        },
        model_path,
    )
    locked_manifest = read_manifest(crop_root / "locked50_eval" / "manifest.csv")
    visual = predict_manifest(model, locked_manifest, crop_root, args.batch_size, device)
    locked_rows = load_jsonl(ROOT / args.locked_predictions)
    metrics, enriched = evaluate_page_level(
        locked_rows,
        visual,
        ROOT / args.dataset / "locked.jsonl",
        args.locked_limit,
        args.cap,
        args.threshold,
        args.integration_mode,
        args.door_threshold,
        args.window_threshold,
    )
    base_metrics, _base_rows = evaluate_page_level(
        locked_rows,
        {},
        ROOT / args.dataset / "locked.jsonl",
        args.locked_limit,
        args.cap,
        args.threshold,
        args.integration_mode,
        args.door_threshold,
        args.window_threshold,
    )
    write_jsonl(ROOT / args.predictions_output, enriched)
    report = {
        "version": "boundary_door_window_crop_context_v24_locked50_eval",
        "claim_boundary": "Raster crop/context boundary type head trained on the selected train manifest only; locked50 gold used only for final evaluation.",
        "model": str(model_path),
        "device": str(device),
        "backbone": args.backbone,
        "pretrained": bool(args.pretrained),
        "threshold": args.threshold,
        "integration_mode": args.integration_mode,
        "door_threshold": args.door_threshold,
        "window_threshold": args.window_threshold,
        "history": history,
        "baseline_without_visual": base_metrics,
        "locked_eval": metrics,
        "success_gate": {
            "classified_recall_min": 0.95,
            "door_recall_min": 0.9,
            "window_recall_min": 0.9,
            "no_drop_vs_baseline": True,
            "baseline_classified_recall": base_metrics["classified_recall"],
            "baseline_door_recall": base_metrics["per_label"]["door"]["classified_recall"],
            "baseline_window_recall": base_metrics["per_label"]["window"]["classified_recall"],
            "locked_classified_recall": metrics["classified_recall"],
            "locked_door_recall": metrics["per_label"]["door"]["classified_recall"],
            "locked_window_recall": metrics["per_label"]["window"]["classified_recall"],
            "passed": metrics["classified_recall"] >= max(0.95, base_metrics["classified_recall"])
            and metrics["per_label"]["door"]["classified_recall"] >= 0.9
            and metrics["per_label"]["window"]["classified_recall"] >= 0.9
            and metrics["per_label"]["door"]["classified_recall"] >= base_metrics["per_label"]["door"]["classified_recall"]
            and metrics["per_label"]["window"]["classified_recall"] >= base_metrics["per_label"]["window"]["classified_recall"],
        },
    }
    write_json(ROOT / args.eval_output, report)
    print(json.dumps({"history": history, "locked_eval": metrics, "success_gate": report["success_gate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
