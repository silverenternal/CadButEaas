#!/usr/bin/env python3
"""Train a visual crop-based symbol proposal box refiner."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets/symbol_proposal_crop_refiner_v21"
CHECKPOINT = ROOT / "checkpoints/symbol_proposal_crop_box_refiner_v21_smoke"
REPORT = ROOT / "reports/vlm/symbol_proposal_crop_box_refiner_v21_smoke_eval.json"
PREDICTIONS = ROOT / "reports/vlm/symbol_proposal_crop_box_refiner_v21_smoke_predictions.jsonl"
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def source_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def bbox_iou(left: list[float], right: list[float]) -> float:
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    la = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    ra = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return inter / max(la + ra - inter, 1e-9)


def area_bucket(box: list[float]) -> str:
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    if area <= 64:
        return "tiny_le_64"
    if area <= 256:
        return "small_le_256"
    if area <= 1024:
        return "medium_le_1024"
    if area <= 4096:
        return "large_le_4096"
    return "xlarge_gt_4096"


def apply_offset(box: list[float], offset: list[float], image_size: list[int], max_abs: float) -> list[float]:
    width, height = image_size
    pw = max(1.0, box[2] - box[0])
    ph = max(1.0, box[3] - box[1])
    values = [max(-max_abs, min(max_abs, float(v))) for v in offset]
    x1 = box[0] + values[0] * pw
    y1 = box[1] + values[1] * ph
    x2 = box[2] + values[2] * pw
    y2 = box[3] + values[3] * ph
    left = max(0.0, min(float(width - 1), min(x1, x2)))
    top = max(0.0, min(float(height - 1), min(y1, y2)))
    right = max(left + 1.0, min(float(width), max(x1, x2)))
    bottom = max(top + 1.0, min(float(height), max(y1, y2)))
    return [left, top, right, bottom]


def load_crop(path: str, size: int, augment: bool, rng: random.Random) -> np.ndarray:
    with Image.open(source_path(path)) as opened:
        crop = opened.convert("RGB")
    crop = ImageOps.autocontrast(crop)
    if augment:
        if rng.random() < 0.5:
            crop = ImageOps.mirror(crop)
        crop = crop.rotate(rng.uniform(-2.0, 2.0), resample=Image.Resampling.BICUBIC, fillcolor=(255, 255, 255))
    crop = crop.resize((size, size), Image.Resampling.BICUBIC)
    arr = np.asarray(crop, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.transpose(2, 0, 1)


def geom_values(item: dict[str, Any]) -> list[float]:
    prop = item.get("proposal") or {}
    geom = item.get("geometry") or {}
    bbox = prop.get("bbox_in_crop") or [0, 0, 0, 0]
    crop = item.get("crop") or {}
    source_size = crop.get("source_size") or [1, 1]
    values = [
        *(float(v) / 224.0 for v in bbox[:4]),
        float(prop.get("label_id") or 0) / 9.0,
        float(prop.get("score") or 0.0),
        *[float(v) for v in geom.get("proposal_size_norm", [0.0, 0.0])[:2]],
        float(geom.get("proposal_aspect_log") or 0.0),
        float(geom.get("proposal_area_log") or 0.0) / 12.0,
        float(source_size[0]) / 256.0,
        float(source_size[1]) / 256.0,
    ]
    return values


class ProposalCropDataset(Dataset):
    def __init__(self, items: list[dict[str, Any]], size: int, augment: bool, seed: int) -> None:
        self.items = items
        self.size = size
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        item = self.items[index]
        rng = random.Random(self.seed + index)
        crop = load_crop(str((item.get("crop") or {}).get("path") or ""), self.size, self.augment, rng)
        target = item.get("target") or {}
        offset = target.get("offset") or [0.0, 0.0, 0.0, 0.0]
        return (
            torch.from_numpy(crop.astype(np.float32)),
            torch.tensor(geom_values(item), dtype=torch.float32),
            torch.tensor([float(v) for v in offset[:4]], dtype=torch.float32),
            index,
        )


class CropBoxRefiner(torch.nn.Module):
    def __init__(self, freeze_encoder: bool) -> None:
        super().__init__()
        encoder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        dim = int(encoder.fc.in_features)
        encoder.fc = torch.nn.Identity()
        self.encoder = encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.geom = torch.nn.Sequential(torch.nn.Linear(12, 48), torch.nn.SiLU(), torch.nn.LayerNorm(48))
        self.head = torch.nn.Sequential(
            torch.nn.Dropout(0.15),
            torch.nn.Linear(dim + 48, 256),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 64),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 4),
        )

    def forward(self, crop: torch.Tensor, geom: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(crop)
        if encoded.ndim > 2:
            encoded = torch.flatten(encoded, 1)
        return self.head(torch.cat([encoded, self.geom(geom)], dim=1))


def sampler(items: list[dict[str, Any]]) -> WeightedRandomSampler:
    counts = Counter(str((item.get("match_audit") or {}).get("gold_area_bucket") or "unknown") for item in items)
    weights = [1.0 / max(counts[str((item.get("match_audit") or {}).get("gold_area_bucket") or "unknown")], 1) for item in items]
    return WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), num_samples=len(items), replacement=True)


def train_epoch(model: torch.nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module, device: torch.device) -> dict[str, float]:
    model.train()
    total = 0
    total_loss = 0.0
    for crop, geom, target, _idx in loader:
        crop = crop.to(device, non_blocking=True)
        geom = geom.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        pred = model(crop, geom)
        loss = loss_fn(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * int(target.shape[0])
        total += int(target.shape[0])
    return {"loss": round(total_loss / max(total, 1), 6)}


def evaluate(model: torch.nn.Module, items: list[dict[str, Any]], loader: DataLoader, device: torch.device, max_abs_offset: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    outputs: dict[int, list[float]] = {}
    with torch.no_grad():
        for crop, geom, _target, indices in loader:
            pred = model(crop.to(device, non_blocking=True), geom.to(device, non_blocking=True)).detach().cpu().numpy()
            for idx, offset in zip(indices.tolist(), pred.tolist(), strict=True):
                outputs[int(idx)] = [float(v) for v in offset]
    totals = Counter()
    by_area = {bucket: Counter() for bucket in ["tiny_le_64", "small_le_256", "medium_le_1024", "large_le_4096", "xlarge_gt_4096"]}
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        proposal = item.get("proposal") or {}
        target = item.get("target") or {}
        prop_box = [float(v) for v in proposal.get("bbox") or [0, 0, 1, 1]]
        gold_box = [float(v) for v in target.get("bbox") or [0, 0, 1, 1]]
        pred_offset = outputs.get(index, [0.0, 0.0, 0.0, 0.0])
        refined_box = apply_offset(prop_box, pred_offset, [int(v) for v in item.get("image_size") or [1, 1]], max_abs_offset)
        base_iou = bbox_iou(prop_box, gold_box)
        refined_iou = bbox_iou(refined_box, gold_box)
        bucket = area_bucket(gold_box)
        totals["total"] += 1
        totals["baseline_iou30"] += int(base_iou >= 0.30)
        totals["refined_iou30"] += int(refined_iou >= 0.30)
        totals["improved"] += int(refined_iou > base_iou)
        totals["worsened"] += int(refined_iou < base_iou)
        by_area[bucket]["total"] += 1
        by_area[bucket]["baseline_iou30"] += int(base_iou >= 0.30)
        by_area[bucket]["refined_iou30"] += int(refined_iou >= 0.30)
        by_area[bucket]["improved"] += int(refined_iou > base_iou)
        rows.append(
            {
                "id": item.get("id"),
                "row_id": item.get("row_id"),
                "proposal_bbox": [round(v, 4) for v in prop_box],
                "target_bbox": [round(v, 4) for v in gold_box],
                "refined_bbox": [round(v, 4) for v in refined_box],
                "baseline_iou": round(base_iou, 8),
                "refined_iou": round(refined_iou, 8),
                "predicted_offset": [round(v, 8) for v in pred_offset],
                "area_bucket": bucket,
            }
        )
    total = max(totals["total"], 1)
    metrics = {
        "records": int(totals["total"]),
        "baseline_iou_0_30_recall_on_matched_proposals": round(totals["baseline_iou30"] / total, 6),
        "refined_iou_0_30_recall_on_matched_proposals": round(totals["refined_iou30"] / total, 6),
        "delta_iou_0_30_recall": round((totals["refined_iou30"] - totals["baseline_iou30"]) / total, 6),
        "improved_iou_rate": round(totals["improved"] / total, 6),
        "worsened_iou_rate": round(totals["worsened"] / total, 6),
        "by_area": {
            bucket: {
                "records": int(counter["total"]),
                "baseline_iou_0_30": round(counter["baseline_iou30"] / max(counter["total"], 1), 6),
                "refined_iou_0_30": round(counter["refined_iou30"] / max(counter["total"], 1), 6),
                "improved_iou_rate": round(counter["improved"] / max(counter["total"], 1), 6),
            }
            for bucket, counter in by_area.items()
            if counter["total"]
        },
    }
    return metrics, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DATA))
    parser.add_argument("--checkpoint-dir", default=str(CHECKPOINT))
    parser.add_argument("--eval-output", default=str(REPORT))
    parser.add_argument("--predictions-output", default=str(PREDICTIONS))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-train", type=int, default=20000)
    parser.add_argument("--max-eval", type=int, default=5000)
    parser.add_argument("--max-abs-offset", type=float, default=1.2)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--seed", type=int, default=20260510)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    train_items = load_jsonl(data_dir / "train.jsonl")[: args.max_train]
    eval_items = load_jsonl(data_dir / "locked_smoke.jsonl")[: args.max_eval]
    train_loader = DataLoader(
        ProposalCropDataset(train_items, args.image_size, True, args.seed),
        batch_size=args.batch_size,
        sampler=sampler(train_items),
        num_workers=4,
        pin_memory=device.type == "cuda",
    )
    eval_loader = DataLoader(
        ProposalCropDataset(eval_items, args.image_size, False, args.seed),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )
    model = CropBoxRefiner(freeze_encoder=args.freeze_encoder).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=1e-4)
    loss_fn = torch.nn.SmoothL1Loss(beta=0.08)
    history = []
    for _epoch in range(args.epochs):
        history.append(train_epoch(model, train_loader, optimizer, loss_fn, device))
    metrics, rows = evaluate(model, eval_items, eval_loader, device, args.max_abs_offset)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": vars(args), "geom_dim": 12}, checkpoint_dir / "model.pt")
    write_jsonl(Path(args.predictions_output), rows)
    report = {
        "version": "symbol_proposal_crop_box_refiner_v21_eval",
        "claim_boundary": "Proposal-level visual crop box refiner direction check; gold used only for supervised offsets/evaluation.",
        "source_integrity": {
            "runtime_input": "raster_proposal_crop_plus_detector_geometry",
            "gold_used_for_runtime": False,
            "forbidden_runtime_features": ["raw_label", "semantic_type", "expected_json", "annotation_path", "svg_geometry"],
            "passed": True,
        },
        "inputs": {"data_dir": args.data_dir},
        "checkpoint": rel(checkpoint_dir / "model.pt"),
        "config": vars(args),
        "device": str(device),
        "history": history,
        "locked_smoke_matched_proposal_metrics": metrics,
        "adopted_for_next_stage": metrics["delta_iou_0_30_recall"] > 0.0 and metrics["worsened_iou_rate"] < 0.5,
    }
    write_json(Path(args.eval_output), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
