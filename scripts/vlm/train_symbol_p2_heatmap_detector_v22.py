#!/usr/bin/env python3
"""Train/evaluate a P2-style heatmap detector for tiny raster symbols."""

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
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import nms

from train_symbol_tile_detector_v20 import (
    FORBIDDEN_RUNTIME_FIELDS,
    ID_TO_LABEL,
    LABELS,
    LABEL_TO_ID,
    area_bucket,
    bbox_iou,
    center_covered,
    load_jsonl,
    nwd_similarity,
    rel,
    source_path,
    target_area_buckets,
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21"
CHECKPOINT = ROOT / "checkpoints/symbol_p2_heatmap_detector_v22"
REPORT = ROOT / "reports/vlm/symbol_p2_heatmap_detector_v22_eval.json"
EPS = 1e-6


def sample_tiles_area_aware(rows: list[dict[str, Any]], limit: int | None, seed: int, positive_ratio: float, small_positive_ratio: float) -> list[dict[str, Any]]:
    if not limit or len(rows) <= limit:
        return list(rows)
    rng = random.Random(seed)
    positives = [row for row in rows if int((row.get("target_counts") or {}).get("symbols") or 0) > 0]
    empties = [row for row in rows if int((row.get("target_counts") or {}).get("symbols") or 0) == 0]
    small_positive = [row for row in positives if target_area_buckets(row) & {"tiny_le_64", "small_le_256"}]
    small_ids = {id(row) for row in small_positive}
    other_positive = [row for row in positives if id(row) not in small_ids]
    for group in (small_positive, other_positive, empties):
        rng.shuffle(group)
    pos_n = min(len(positives), int(limit * positive_ratio))
    small_n = min(len(small_positive), int(pos_n * small_positive_ratio))
    other_n = min(len(other_positive), pos_n - small_n)
    selected = small_positive[:small_n] + other_positive[:other_n]
    if len(selected) < pos_n:
        selected.extend(small_positive[small_n : small_n + (pos_n - len(selected))])
    if len(selected) < pos_n:
        selected.extend(other_positive[other_n : other_n + (pos_n - len(selected))])
    selected.extend(empties[: max(0, limit - len(selected))])
    if len(selected) < limit:
        used = {id(row) for row in selected}
        leftovers = [row for row in rows if id(row) not in used]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: limit - len(selected)])
    rng.shuffle(selected)
    return selected[:limit]


class SymbolP2Dataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], input_size: int, stride: int, augment: bool, seed: int) -> None:
        self.rows = rows
        self.input_size = input_size
        self.stride = stride
        self.grid_size = input_size // stride
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        row = self.rows[index]
        tile = row.get("tile") or {}
        x1, y1, x2, y2 = [int(v) for v in tile.get("bbox") or [0, 0, 1, 1]]
        tile_w = max(1, x2 - x1)
        tile_h = max(1, y2 - y1)
        with Image.open(source_path(str(row.get("image") or ""))) as opened:
            crop = opened.convert("RGB").crop((x1, y1, x2, y2))
        crop = ImageOps.autocontrast(crop).resize((self.input_size, self.input_size), Image.Resampling.BILINEAR)
        rng = random.Random(self.seed + index)
        flip = self.augment and rng.random() < 0.5
        if flip:
            crop = ImageOps.mirror(crop)
        image = torch.from_numpy((np.asarray(crop, dtype=np.float32) / 255.0).transpose(2, 0, 1)).float()

        heatmap = torch.zeros((len(LABELS), self.grid_size, self.grid_size), dtype=torch.float32)
        size = torch.zeros((2, self.grid_size, self.grid_size), dtype=torch.float32)
        offset = torch.zeros((2, self.grid_size, self.grid_size), dtype=torch.float32)
        mask = torch.zeros((1, self.grid_size, self.grid_size), dtype=torch.float32)
        gold: list[dict[str, Any]] = []
        for target in ((row.get("targets") or {}).get("boxes") or []):
            box = [float(v) for v in target.get("bbox") or []]
            if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
                continue
            sx = self.input_size / tile_w
            sy = self.input_size / tile_h
            scaled = [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]
            if flip:
                scaled = [self.input_size - scaled[2], scaled[1], self.input_size - scaled[0], scaled[3]]
            cx = (scaled[0] + scaled[2]) / 2.0 / self.stride
            cy = (scaled[1] + scaled[3]) / 2.0 / self.stride
            gx = min(self.grid_size - 1, max(0, int(cx)))
            gy = min(self.grid_size - 1, max(0, int(cy)))
            label = str(target.get("label") or "generic_symbol")
            label_index = LABEL_TO_ID.get(label, 5) - 1
            radius = 1 if max(scaled[2] - scaled[0], scaled[3] - scaled[1]) <= 24 else 2
            for yy in range(max(0, gy - radius), min(self.grid_size, gy + radius + 1)):
                for xx in range(max(0, gx - radius), min(self.grid_size, gx + radius + 1)):
                    dist2 = (yy - gy) ** 2 + (xx - gx) ** 2
                    heatmap[label_index, yy, xx] = max(float(heatmap[label_index, yy, xx]), float(np.exp(-dist2 / max(radius, 1))))
            size[0, gy, gx] = np.log(max(1.0, scaled[2] - scaled[0]) / self.input_size + EPS)
            size[1, gy, gx] = np.log(max(1.0, scaled[3] - scaled[1]) / self.input_size + EPS)
            offset[0, gy, gx] = cx - gx
            offset[1, gy, gx] = cy - gy
            mask[0, gy, gx] = 1.0
            gold.append(target)

        meta = {"id": row.get("id"), "row_id": row.get("row_id"), "tile_bbox": [x1, y1, x2, y2], "tile_size": [tile_w, tile_h], "gold": gold}
        return image, {"heatmap": heatmap, "size": size, "offset": offset, "mask": mask}, meta


def collate(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]]) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[dict[str, Any]]]:
    images, targets, metas = zip(*batch, strict=True)
    return torch.stack(list(images)), {key: torch.stack([target[key] for target in targets]) for key in targets[0]}, list(metas)


class P2HeatmapDetector(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
        )
        self.body = nn.Sequential(
            nn.Conv2d(64, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
        )
        self.heatmap = nn.Conv2d(96, num_classes, 1)
        self.size = nn.Conv2d(96, 2, 1)
        self.offset = nn.Conv2d(96, 2, 1)
        nn.init.constant_(self.heatmap.bias, -4.6)
        nn.init.constant_(self.size.bias, -3.5)
        nn.init.constant_(self.offset.bias, 0.0)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.body(self.stem(images))
        return {"heatmap": self.heatmap(feat), "size": self.size(feat), "offset": torch.sigmoid(self.offset(feat))}


def focal_bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = torch.where(targets > 0, probs, 1.0 - probs)
    alpha = torch.where(targets > 0, torch.full_like(targets, 0.75), torch.full_like(targets, 0.25))
    return (alpha * (1.0 - pt).pow(2.0) * bce).mean()


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> dict[str, float]:
    model.train()
    totals = Counter()
    seen = 0
    for images, targets, _metas in loader:
        images = images.to(device, non_blocking=True)
        targets = {key: value.to(device, non_blocking=True) for key, value in targets.items()}
        outputs = model(images)
        mask = targets["mask"]
        pos = mask.sum().clamp_min(1.0)
        heatmap_loss = focal_bce_loss(outputs["heatmap"], targets["heatmap"])
        size_loss = (F.smooth_l1_loss(outputs["size"] * mask, targets["size"] * mask, reduction="sum") / pos)
        offset_loss = (F.smooth_l1_loss(outputs["offset"] * mask, targets["offset"] * mask, reduction="sum") / pos)
        loss = heatmap_loss + 3.0 * size_loss + offset_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        batch = int(images.shape[0])
        seen += batch
        totals["loss"] += float(loss.detach().cpu()) * batch
        totals["heatmap_loss"] += float(heatmap_loss.detach().cpu()) * batch
        totals["size_loss"] += float(size_loss.detach().cpu()) * batch
        totals["offset_loss"] += float(offset_loss.detach().cpu()) * batch
    return {key: round(value / max(seen, 1), 6) for key, value in totals.items()}


def decode_outputs(outputs: dict[str, torch.Tensor], metas: list[dict[str, Any]], input_size: int, stride: int, score_threshold: float, topk: int) -> dict[str, list[dict[str, Any]]]:
    probs = torch.sigmoid(outputs["heatmap"]).detach().cpu()
    sizes = outputs["size"].detach().cpu()
    offsets = outputs["offset"].detach().cpu()
    page_preds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    batch, classes, grid_h, grid_w = probs.shape
    for item in range(batch):
        meta = metas[item]
        left, top, _right, _bottom = [float(v) for v in meta["tile_bbox"]]
        tile_w, tile_h = [float(v) for v in meta["tile_size"]]
        flat_scores, flat_indices = torch.topk(probs[item].reshape(-1), k=min(topk, probs[item].numel()))
        for score, flat_index in zip(flat_scores.tolist(), flat_indices.tolist(), strict=True):
            if score < score_threshold:
                continue
            class_index = flat_index // (grid_h * grid_w)
            cell = flat_index % (grid_h * grid_w)
            gy = cell // grid_w
            gx = cell % grid_w
            ox = float(offsets[item, 0, gy, gx])
            oy = float(offsets[item, 1, gy, gx])
            bw = float(torch.exp(sizes[item, 0, gy, gx]).clamp(1.0 / input_size, 1.0)) * input_size
            bh = float(torch.exp(sizes[item, 1, gy, gx]).clamp(1.0 / input_size, 1.0)) * input_size
            cx = (gx + ox) * stride
            cy = (gy + oy) * stride
            x1 = max(0.0, cx - bw / 2.0) / input_size * tile_w + left
            y1 = max(0.0, cy - bh / 2.0) / input_size * tile_h + top
            x2 = min(float(input_size), cx + bw / 2.0) / input_size * tile_w + left
            y2 = min(float(input_size), cy + bh / 2.0) / input_size * tile_h + top
            label_id = int(class_index) + 1
            page_preds[str(meta["row_id"])].append(
                {"bbox": [x1, y1, x2, y2], "label_id": label_id, "label": ID_TO_LABEL[label_id], "score": float(score), "tile_id": meta["id"]}
            )
    return page_preds


def collect_predictions(model: nn.Module, rows: list[dict[str, Any]], device: torch.device, args: argparse.Namespace) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, dict[str, Any]]]]:
    dataset = SymbolP2Dataset(rows, args.input_size, args.stride, augment=False, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda", collate_fn=collate)
    page_preds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    page_golds: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    model.eval()
    with torch.no_grad():
        for images, _targets, metas in loader:
            outputs = model(images.to(device, non_blocking=True))
            batch_preds = decode_outputs(outputs, metas, args.input_size, args.stride, args.decode_score_threshold, args.topk_per_tile)
            for row_id, preds in batch_preds.items():
                page_preds[row_id].extend(preds)
            for meta in metas:
                row_id = str(meta["row_id"])
                for gold in meta["gold"]:
                    target_id = str(gold.get("target_id") or f"{row_id}_{len(page_golds[row_id])}")
                    page_golds[row_id][target_id] = {
                        "target_id": target_id,
                        "bbox": [float(v) for v in gold.get("page_bbox") or gold.get("bbox")],
                        "label": str(gold.get("label") or "generic_symbol"),
                    }
    return page_preds, page_golds


def merge_predictions(preds: list[dict[str, Any]], score_threshold: float, nms_threshold: float, max_per_page: int) -> list[dict[str, Any]]:
    filtered = [pred for pred in preds if float(pred["score"]) >= score_threshold]
    if not filtered:
        return []
    boxes = torch.tensor([pred["bbox"] for pred in filtered], dtype=torch.float32)
    scores = torch.tensor([float(pred["score"]) for pred in filtered], dtype=torch.float32)
    labels = [int(pred["label_id"]) for pred in filtered]
    keep_indices: list[int] = []
    for label in sorted(set(labels)):
        idx = torch.tensor([i for i, current in enumerate(labels) if current == label], dtype=torch.long)
        keep = nms(boxes[idx], scores[idx], nms_threshold)
        keep_indices.extend(int(idx[int(i)]) for i in keep.tolist())
    keep_indices.sort(key=lambda i: float(filtered[i]["score"]), reverse=True)
    return [filtered[i] for i in keep_indices[:max_per_page]]


def score_predictions(
    page_preds: dict[str, list[dict[str, Any]]],
    page_golds: dict[str, dict[str, dict[str, Any]]],
    score_threshold: float,
    nms_threshold: float,
    max_per_page: int,
    tile_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    totals = Counter()
    by_label = Counter()
    by_label_center = Counter()
    by_label_iou = Counter()
    by_area = Counter()
    by_area_center = Counter()
    by_area_iou = Counter()
    by_area_nwd_070 = Counter()
    typed_correct = 0
    for row_id, gold_map in page_golds.items():
        merged = merge_predictions(page_preds.get(row_id, []), score_threshold, nms_threshold, max_per_page)
        used_iou: set[int] = set()
        used_center: set[int] = set()
        for gold in gold_map.values():
            gold_box = [float(v) for v in gold["bbox"]]
            label = str(gold["label"])
            bucket = area_bucket(gold_box)
            by_label[label] += 1
            by_area[bucket] += 1
            best_iou = 0.0
            best_iou_index: int | None = None
            best_nwd = 0.0
            center_index: int | None = None
            for pred_index, pred in enumerate(merged):
                pred_box = [float(v) for v in pred["bbox"]]
                iou = bbox_iou(pred_box, gold_box)
                if iou > best_iou:
                    best_iou = iou
                    best_iou_index = pred_index
                best_nwd = max(best_nwd, nwd_similarity(pred_box, gold_box))
                if center_index is None and pred_index not in used_center and center_covered(pred_box, gold_box):
                    center_index = pred_index
            if best_nwd >= 0.70:
                by_area_nwd_070[bucket] += 1
            if best_iou_index is not None and best_iou >= 0.30 and best_iou_index not in used_iou:
                used_iou.add(best_iou_index)
                totals["matched_iou"] += 1
                by_label_iou[label] += 1
                by_area_iou[bucket] += 1
                if merged[best_iou_index]["label"] == label:
                    typed_correct += 1
            if center_index is not None:
                used_center.add(center_index)
                totals["matched_center"] += 1
                by_label_center[label] += 1
                by_area_center[bucket] += 1
        totals["gold"] += len(gold_map)
        totals["predicted"] += len(merged)
        predictions.append({"row_id": row_id, "predicted_symbols": merged, "gold_symbol_count": len(gold_map)})
    precision = totals["matched_iou"] / max(totals["predicted"], 1)
    recall = totals["matched_iou"] / max(totals["gold"], 1)
    report = {
        "rows": len(page_golds),
        "tiles": tile_count,
        "symbol_bbox_iou_0_30": {
            "matched": int(totals["matched_iou"]),
            "predicted": int(totals["predicted"]),
            "gold": int(totals["gold"]),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall), 6),
        },
        "symbol_bbox_center_recall": round(totals["matched_center"] / max(totals["gold"], 1), 6),
        "candidate_inflation": round(totals["predicted"] / max(totals["gold"], 1), 6),
        "typed_accuracy_on_iou_matches": round(typed_correct / max(totals["matched_iou"], 1), 6),
        "type_center_recall": {label: round(by_label_center[label] / max(by_label[label], 1), 6) for label in sorted(by_label)},
        "type_iou_recall": {label: round(by_label_iou[label] / max(by_label[label], 1), 6) for label in sorted(by_label)},
        "area_center_recall": {bucket: round(by_area_center[bucket] / max(by_area[bucket], 1), 6) for bucket in sorted(by_area)},
        "area_iou_recall": {bucket: round(by_area_iou[bucket] / max(by_area[bucket], 1), 6) for bucket in sorted(by_area)},
        "nwd_tiny_box_audit": {
            "area_recall_at_0_70": {bucket: round(by_area_nwd_070[bucket] / max(by_area[bucket], 1), 6) for bucket in sorted(by_area)}
        },
    }
    return report, predictions


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
    parser.add_argument("--predictions-output", default=str(ROOT / "reports/vlm/symbol_p2_heatmap_detector_v22_predictions.jsonl"))
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--limit-train-tiles", type=int, default=8000)
    parser.add_argument("--limit-dev-tiles", type=int, default=1000)
    parser.add_argument("--limit-locked-tiles", type=int, default=1000)
    parser.add_argument("--train-positive-ratio", type=float, default=0.95)
    parser.add_argument("--train-small-positive-ratio", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--input-size", type=int, default=384)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--decode-score-threshold", type=float, default=0.01)
    parser.add_argument("--score-threshold-grid", default="0.05,0.10,0.15,0.20")
    parser.add_argument("--nms-threshold-grid", default="0.35,0.45,0.55")
    parser.add_argument("--max-per-page", type=int, default=500)
    parser.add_argument("--topk-per-tile", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260510)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    data_dir = Path(args.data)
    train_rows = sample_tiles_area_aware(load_jsonl(data_dir / "train.jsonl"), args.limit_train_tiles, args.seed, args.train_positive_ratio, args.train_small_positive_ratio)
    dev_rows = sample_tiles_area_aware(load_jsonl(data_dir / "dev.jsonl"), args.limit_dev_tiles, args.seed + 1, 0.85, 0.75)
    locked_rows = sample_tiles_area_aware(load_jsonl(data_dir / "locked.jsonl"), args.limit_locked_tiles, args.seed + 2, 0.85, 0.75)

    model = P2HeatmapDetector(num_classes=len(LABELS)).to(device)
    if args.init_checkpoint:
        model.load_state_dict(torch.load(source_path(args.init_checkpoint), map_location="cpu"))
    epoch_log: list[dict[str, Any]] = []
    if not args.eval_only:
        loader = DataLoader(
            SymbolP2Dataset(train_rows, args.input_size, args.stride, augment=True, seed=args.seed),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=collate,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.1)
        for epoch in range(1, args.epochs + 1):
            row = train_epoch(model, loader, optimizer, device)
            scheduler.step()
            row["epoch"] = epoch
            epoch_log.append(row)

    dev_preds, dev_golds = collect_predictions(model, dev_rows, device, args)
    locked_preds, locked_golds = collect_predictions(model, locked_rows, device, args)
    score_grid = [float(item) for item in args.score_threshold_grid.split(",") if item.strip()]
    nms_grid = [float(item) for item in args.nms_threshold_grid.split(",") if item.strip()]
    grid_reports = []
    for score_threshold in score_grid:
        for nms_threshold in nms_grid:
            dev_eval, _ = score_predictions(dev_preds, dev_golds, score_threshold, nms_threshold, args.max_per_page, len(dev_rows))
            grid_reports.append({"score_threshold": score_threshold, "nms_threshold": nms_threshold, "dev": dev_eval})
    grid_reports.sort(
        key=lambda row: (
            row["dev"]["symbol_bbox_center_recall"],
            row["dev"]["symbol_bbox_iou_0_30"]["recall"],
            -row["dev"]["candidate_inflation"],
        ),
        reverse=True,
    )
    selected = grid_reports[0]
    locked_eval, locked_predictions = score_predictions(
        locked_preds, locked_golds, float(selected["score_threshold"]), float(selected["nms_threshold"]), args.max_per_page, len(locked_rows)
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / "model.pt")
    write_json(
        checkpoint_dir / "model_metadata.json",
        {
            "model_type": "symbol_p2_heatmap_detector_v22",
            "labels": LABELS,
            "input_size": args.input_size,
            "stride": args.stride,
            "runtime_contract": {"model_input_features": ["image_tile_pixels"], "forbidden_runtime_features": FORBIDDEN_RUNTIME_FIELDS},
        },
    )
    report = {
        "version": "symbol_p2_heatmap_detector_v22_eval",
        "claim_boundary": "P2-style high-resolution heatmap detector prototype for raster symbol body localization.",
        "source_integrity": {
            "model_input": "raster_tile_pixels_only",
            "raw_label_or_semantic_type_used_as_runtime_feature": False,
            "forbidden_runtime_features": FORBIDDEN_RUNTIME_FIELDS,
            "locked_gold_use": "evaluation_only",
        },
        "baseline_to_beat": {
            "scaled_faster_rcnn_center_recall": 0.851394,
            "scaled_faster_rcnn_tiny_iou_recall": 0.393013,
            "scaled_faster_rcnn_candidate_inflation": 7.919152,
        },
        "dataset": rel(data_dir),
        "checkpoint": rel(checkpoint_dir / "model.pt"),
        "config": vars(args) | {"device": str(device)},
        "counts": {
            "train_tiles": len(train_rows),
            "dev_tiles": len(dev_rows),
            "locked_tiles": len(locked_rows),
            "train_positive_tiles": sum(1 for row in train_rows if int((row.get("target_counts") or {}).get("symbols") or 0) > 0),
            "train_small_or_tiny_positive_tiles": sum(1 for row in train_rows if target_area_buckets(row) & {"tiny_le_64", "small_le_256"}),
        },
        "epoch_log": epoch_log,
        "threshold_grid": grid_reports,
        "selected_thresholds": {"score_threshold": float(selected["score_threshold"]), "nms_threshold": float(selected["nms_threshold"])},
        "dev": selected["dev"],
        "locked": locked_eval,
        "gate": {
            "beats_scaled_faster_rcnn_center_0_851394": locked_eval["symbol_bbox_center_recall"] > 0.851394,
            "beats_scaled_faster_rcnn_tiny_iou_0_393013": locked_eval["area_iou_recall"].get("tiny_le_64", 0.0) > 0.393013,
            "candidate_inflation_lte_7_919152": locked_eval["candidate_inflation"] <= 7.919152,
        },
        "memory_audit": memory_audit(device),
    }
    report["gate"]["passed"] = all(bool(value) for value in report["gate"].values())
    write_json(Path(args.eval_output), report)
    write_jsonl(Path(args.predictions_output), locked_predictions)
    print(json.dumps({"locked": locked_eval, "gate": report["gate"], "checkpoint": rel(checkpoint_dir / "model.pt")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
