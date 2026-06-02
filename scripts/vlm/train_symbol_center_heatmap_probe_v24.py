#!/usr/bin/env python3
"""Train/evaluate a raster-only symbol center heatmap proposal route.

This is intentionally a proposal frontend, not a symbol type model.  It learns
class-agnostic symbol centers from raster tiles, then audits whether the route
can recover centers missed by the current YOLO symbol detector.
"""

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
    area_bucket,
    bbox_iou,
    center_covered,
    load_jsonl,
    rel,
    source_path,
    target_area_buckets,
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21"
CHECKPOINT = ROOT / "checkpoints/symbol_center_heatmap_probe_v24"
REPORT = ROOT / "reports/vlm/symbol_center_heatmap_probe_v24_eval.json"
PREDICTIONS = ROOT / "reports/vlm/symbol_center_heatmap_probe_v24_predictions.jsonl"
YOLO_BASELINE_REPORT = ROOT / "reports/vlm/symbol_yolov8n_pretrained_v22_dedup_hi640_probe_page_eval.json"
YOLO_BASELINE_PREDICTIONS = ROOT / "reports/vlm/symbol_yolov8n_pretrained_v22_dedup_hi640_probe_page_predictions.jsonl"
YOLO_DIR = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21_yolo_v22"


def sample_tiles_area_aware(rows: list[dict[str, Any]], limit: int | None, seed: int, positive_ratio: float, small_positive_ratio: float) -> list[dict[str, Any]]:
    if not limit or len(rows) <= limit:
        return list(rows)
    rng = random.Random(seed)
    positives = [row for row in rows if int((row.get("target_counts") or {}).get("symbols") or 0) > 0]
    empties = [row for row in rows if int((row.get("target_counts") or {}).get("symbols") or 0) == 0]
    small = [row for row in positives if target_area_buckets(row) & {"tiny_le_64", "small_le_256"}]
    small_ids = {id(row) for row in small}
    other = [row for row in positives if id(row) not in small_ids]
    for group in (small, other, empties):
        rng.shuffle(group)
    positive_n = min(len(positives), int(limit * positive_ratio))
    small_n = min(len(small), int(positive_n * small_positive_ratio))
    selected = small[:small_n] + other[: max(0, positive_n - small_n)]
    if len(selected) < positive_n:
        selected.extend(small[small_n : small_n + positive_n - len(selected)])
    selected.extend(empties[: max(0, limit - len(selected))])
    if len(selected) < limit:
        used = {id(row) for row in selected}
        leftovers = [row for row in rows if id(row) not in used]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: limit - len(selected)])
    rng.shuffle(selected)
    return selected[:limit]


def gaussian_2d(radius: int) -> np.ndarray:
    diameter = radius * 2 + 1
    yy, xx = np.ogrid[:diameter, :diameter]
    center = radius
    sigma = max(float(diameter) / 6.0, 1e-6)
    return np.exp(-((xx - center) ** 2 + (yy - center) ** 2) / (2 * sigma * sigma)).astype(np.float32)


class SymbolCenterDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], input_size: int, stride: int, augment: bool, seed: int) -> None:
        self.rows = rows
        self.input_size = input_size
        self.stride = stride
        self.grid_size = input_size // stride
        self.augment = augment
        self.seed = seed
        self.kernels = {1: gaussian_2d(1), 2: gaussian_2d(2), 3: gaussian_2d(3)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        row = self.rows[index]
        tile = row.get("tile") or {}
        left, top, right, bottom = [int(v) for v in tile.get("bbox") or [0, 0, 1, 1]]
        tile_w = max(1, right - left)
        tile_h = max(1, bottom - top)
        with Image.open(source_path(str(row.get("image") or ""))) as opened:
            crop = opened.convert("RGB").crop((left, top, right, bottom))
        crop = ImageOps.autocontrast(crop).resize((self.input_size, self.input_size), Image.Resampling.BILINEAR)
        rng = random.Random(self.seed + index)
        flip = self.augment and rng.random() < 0.5
        if flip:
            crop = ImageOps.mirror(crop)
        image = torch.from_numpy((np.asarray(crop, dtype=np.float32) / 255.0).transpose(2, 0, 1)).float()

        heatmap = np.zeros((1, self.grid_size, self.grid_size), dtype=np.float32)
        gold: list[dict[str, Any]] = []
        sx = self.input_size / tile_w
        sy = self.input_size / tile_h
        for target in ((row.get("targets") or {}).get("boxes") or []):
            box = [float(v) for v in target.get("bbox") or []]
            page_box = [float(v) for v in target.get("page_bbox") or target.get("bbox") or []]
            if len(box) != 4 or len(page_box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
                continue
            scaled = [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]
            if flip:
                scaled = [self.input_size - scaled[2], scaled[1], self.input_size - scaled[0], scaled[3]]
            cx = (scaled[0] + scaled[2]) / 2.0 / self.stride
            cy = (scaled[1] + scaled[3]) / 2.0 / self.stride
            gx = min(self.grid_size - 1, max(0, int(round(cx))))
            gy = min(self.grid_size - 1, max(0, int(round(cy))))
            max_side = max(scaled[2] - scaled[0], scaled[3] - scaled[1])
            radius = 1 if max_side <= 16 else 2 if max_side <= 48 else 3
            kernel = self.kernels[radius]
            y1 = max(0, gy - radius)
            y2 = min(self.grid_size, gy + radius + 1)
            x1 = max(0, gx - radius)
            x2 = min(self.grid_size, gx + radius + 1)
            ky1 = y1 - (gy - radius)
            ky2 = ky1 + (y2 - y1)
            kx1 = x1 - (gx - radius)
            kx2 = kx1 + (x2 - x1)
            heatmap[0, y1:y2, x1:x2] = np.maximum(heatmap[0, y1:y2, x1:x2], kernel[ky1:ky2, kx1:kx2])
            gold.append(target)

        meta = {
            "id": row.get("id"),
            "row_id": row.get("row_id"),
            "tile_bbox": [left, top, right, bottom],
            "tile_size": [tile_w, tile_h],
            "gold": gold,
        }
        return image, torch.from_numpy(heatmap), meta


def collate(batch: list[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]]) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    images, heatmaps, metas = zip(*batch, strict=True)
    return torch.stack(list(images)), torch.stack(list(heatmaps)), list(metas)


class CenterHeatmapNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            nn.Conv2d(96, 1, 1),
        )
        nn.init.constant_(self.net[-1].bias, -5.0)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


def center_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    positive = targets.gt(0.05)
    weights = torch.where(positive, torch.full_like(targets, 20.0), torch.full_like(targets, 1.0))
    focal = torch.where(positive, (1.0 - probs).pow(2.0), probs.pow(2.0))
    return (weights * focal * bce).mean()


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> dict[str, float]:
    model.train()
    totals = Counter()
    seen = 0
    for images, heatmaps, _metas in loader:
        images = images.to(device, non_blocking=True)
        heatmaps = heatmaps.to(device, non_blocking=True)
        logits = model(images)
        loss = center_loss(logits, heatmaps)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        batch = int(images.shape[0])
        seen += batch
        totals["loss"] += float(loss.detach().cpu()) * batch
    return {key: round(value / max(seen, 1), 6) for key, value in totals.items()}


def decode_centers(logits: torch.Tensor, metas: list[dict[str, Any]], input_size: int, stride: int, score_threshold: float, topk: int, box_size: int) -> dict[str, list[dict[str, Any]]]:
    probs = torch.sigmoid(logits).detach().cpu()
    pooled = F.max_pool2d(probs, kernel_size=3, stride=1, padding=1)
    peaks = probs * probs.eq(pooled)
    page_preds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    batch, _channels, grid_h, grid_w = peaks.shape
    for item in range(batch):
        meta = metas[item]
        left, top, _right, _bottom = [float(v) for v in meta["tile_bbox"]]
        tile_w, tile_h = [float(v) for v in meta["tile_size"]]
        scores, indices = torch.topk(peaks[item, 0].reshape(-1), k=min(topk, grid_h * grid_w))
        for score, flat_index in zip(scores.tolist(), indices.tolist(), strict=True):
            if score < score_threshold:
                continue
            gy = int(flat_index) // grid_w
            gx = int(flat_index) % grid_w
            cx = (gx + 0.5) * stride / input_size * tile_w + left
            cy = (gy + 0.5) * stride / input_size * tile_h + top
            half = float(box_size) / 2.0
            page_preds[str(meta["row_id"])].append(
                {
                    "bbox": [cx - half, cy - half, cx + half, cy + half],
                    "label_id": 5,
                    "label": "generic_symbol",
                    "score": float(score),
                    "tile_id": meta["id"],
                    "proposal_source": "center_heatmap_v24",
                }
            )
    return page_preds


def collect_predictions(model: nn.Module, rows: list[dict[str, Any]], device: torch.device, args: argparse.Namespace) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, dict[str, Any]]]]:
    dataset = SymbolCenterDataset(rows, args.input_size, args.stride, augment=False, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda", collate_fn=collate)
    page_preds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    page_golds: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    model.eval()
    with torch.no_grad():
        for images, _heatmaps, metas in loader:
            batch_preds = decode_centers(model(images.to(device, non_blocking=True)), metas, args.input_size, args.stride, args.decode_score_threshold, args.topk_per_tile, args.box_size)
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
    filtered = [pred for pred in preds if float(pred.get("score", 0.0)) >= score_threshold]
    if not filtered:
        return []
    boxes = torch.tensor([pred["bbox"] for pred in filtered], dtype=torch.float32)
    scores = torch.tensor([float(pred.get("score", 0.0)) for pred in filtered], dtype=torch.float32)
    labels = [int(pred.get("label_id") or 5) for pred in filtered]
    keep_indices: list[int] = []
    for label in sorted(set(labels)):
        idx = torch.tensor([i for i, current in enumerate(labels) if current == label], dtype=torch.long)
        keep = nms(boxes[idx], scores[idx], nms_threshold)
        keep_indices.extend(int(idx[int(i)]) for i in keep.tolist())
    keep_indices.sort(key=lambda i: float(filtered[i].get("score", 0.0)), reverse=True)
    return [filtered[i] for i in keep_indices[:max_per_page]]


def score_predictions(
    page_preds: dict[str, list[dict[str, Any]]],
    page_golds: dict[str, dict[str, dict[str, Any]]],
    score_threshold: float,
    nms_threshold: float,
    max_per_page: int,
    tile_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    totals = Counter()
    by_label = Counter()
    by_label_center = Counter()
    by_label_iou = Counter()
    by_area = Counter()
    by_area_center = Counter()
    by_area_iou = Counter()
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
            center_index: int | None = None
            for pred_index, pred in enumerate(merged):
                pred_box = [float(v) for v in pred["bbox"]]
                iou = bbox_iou(pred_box, gold_box)
                if iou > best_iou:
                    best_iou = iou
                    best_iou_index = pred_index
                if center_index is None and pred_index not in used_center and center_covered(pred_box, gold_box):
                    center_index = pred_index
            if best_iou_index is not None and best_iou >= 0.30 and best_iou_index not in used_iou:
                used_iou.add(best_iou_index)
                totals["matched_iou"] += 1
                by_label_iou[label] += 1
                by_area_iou[bucket] += 1
                if merged[best_iou_index].get("label") == label:
                    typed_correct += 1
            if center_index is not None:
                used_center.add(center_index)
                totals["matched_center"] += 1
                by_label_center[label] += 1
                by_area_center[bucket] += 1
            else:
                errors.append({"row_id": row_id, "target_id": gold["target_id"], "label": label, "area_bucket": bucket, "gold_bbox": gold_box, "best_iou": round(best_iou, 6), "error": "missed_center"})
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
    }
    return report, predictions, errors


def load_page_predictions(path: Path) -> dict[str, list[dict[str, Any]]]:
    rows = load_jsonl(path)
    return {str(row["row_id"]): list(row.get("predicted_symbols") or []) for row in rows}


def filter_rows_to_page_ids(rows: list[dict[str, Any]], page_ids: set[str]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("row_id")) in page_ids]


def image_path_for_yolo_tile(yolo_dir: Path, split: str, row: dict[str, Any]) -> Path:
    yolo_split = "val" if split == "dev" else split
    return yolo_dir / "images" / yolo_split / f"{row['id']}.jpg"


def filter_rows_with_exported_yolo_images(rows: list[dict[str, Any]], split: str, yolo_dir: Path) -> list[dict[str, Any]]:
    return [row for row in rows if image_path_for_yolo_tile(yolo_dir, split, row).exists()]


def union_predictions(left: dict[str, list[dict[str, Any]]], right: dict[str, list[dict[str, Any]]], heatmap_score_scale: float) -> dict[str, list[dict[str, Any]]]:
    keys = set(left) | set(right)
    out: dict[str, list[dict[str, Any]]] = {}
    for key in keys:
        baseline = []
        for pred in left.get(key, []):
            item = dict(pred)
            # Preserve current best YOLO output in mixed-source NMS/cap.  The
            # original score is kept for audit while the rank score is used only
            # inside this no-drop union evaluation.
            item["original_score"] = float(item.get("score", 0.0))
            item["score"] = 1.0 + float(item.get("score", 0.0))
            item["proposal_source"] = item.get("proposal_source") or "yolo_baseline"
            baseline.append(item)
        heatmap = []
        for pred in right.get(key, []):
            item = dict(pred)
            item["original_score"] = float(item.get("score", 0.0))
            item["score"] = float(item.get("score", 0.0)) * heatmap_score_scale
            item["proposal_source"] = item.get("proposal_source") or "center_heatmap_v24"
            heatmap.append(item)
        out[key] = baseline + heatmap
    return out


def selection_key(row: dict[str, Any]) -> tuple[float, float, float]:
    metrics = row["metrics"]
    return (
        float(metrics["symbol_bbox_center_recall"]),
        float(metrics["symbol_bbox_iou_0_30"]["recall"]),
        -float(metrics["candidate_inflation"]),
    )


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
    parser.add_argument("--predictions-output", default=str(PREDICTIONS))
    parser.add_argument("--yolo-baseline-report", default=str(YOLO_BASELINE_REPORT))
    parser.add_argument("--yolo-baseline-predictions", default=str(YOLO_BASELINE_PREDICTIONS))
    parser.add_argument("--yolo-dir", default=str(YOLO_DIR))
    parser.add_argument("--align-locked-to-yolo-baseline-pages", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--filter-to-exported-yolo-tiles", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--limit-train-tiles", type=int, default=30000)
    parser.add_argument("--limit-dev-tiles", type=int, default=2000)
    parser.add_argument("--limit-locked-tiles", type=int, default=2000)
    parser.add_argument("--train-positive-ratio", type=float, default=0.95)
    parser.add_argument("--train-small-positive-ratio", type=float, default=0.85)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--input-size", type=int, default=384)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--decode-score-threshold", type=float, default=0.001)
    parser.add_argument("--score-threshold-grid", default="0.005,0.01,0.02,0.03,0.05")
    parser.add_argument("--nms-threshold-grid", default="0.15,0.25,0.35")
    parser.add_argument("--topk-per-tile", type=int, default=160)
    parser.add_argument("--box-size", type=int, default=18)
    parser.add_argument("--max-per-page", type=int, default=1200)
    parser.add_argument("--union-heatmap-score-scale", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260510)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    data_dir = Path(args.data)
    train_source = load_jsonl(data_dir / "train.jsonl")
    dev_source = load_jsonl(data_dir / "dev.jsonl")
    locked_source = load_jsonl(data_dir / "locked.jsonl")
    if args.filter_to_exported_yolo_tiles:
        yolo_dir = Path(args.yolo_dir)
        train_source = filter_rows_with_exported_yolo_images(train_source, "train", yolo_dir)
        dev_source = filter_rows_with_exported_yolo_images(dev_source, "dev", yolo_dir)
        locked_source = filter_rows_with_exported_yolo_images(locked_source, "locked", yolo_dir)
    train_rows = sample_tiles_area_aware(train_source, args.limit_train_tiles, args.seed, args.train_positive_ratio, args.train_small_positive_ratio)
    dev_rows = sample_tiles_area_aware(dev_source, args.limit_dev_tiles, args.seed + 1, 0.85, 0.75)
    locked_rows = sample_tiles_area_aware(locked_source, args.limit_locked_tiles, args.seed + 2, 0.85, 0.75)
    baseline_page_ids: set[str] = set()
    baseline_predictions_path = Path(args.yolo_baseline_predictions)
    if args.align_locked_to_yolo_baseline_pages and baseline_predictions_path.exists():
        baseline_page_ids = set(load_page_predictions(baseline_predictions_path))
        aligned_locked_rows = filter_rows_to_page_ids(locked_rows, baseline_page_ids)
        if aligned_locked_rows:
            locked_rows = aligned_locked_rows

    model = CenterHeatmapNet().to(device)
    if args.init_checkpoint:
        model.load_state_dict(torch.load(source_path(args.init_checkpoint), map_location="cpu"))
    epoch_log: list[dict[str, Any]] = []
    if not args.eval_only:
        loader = DataLoader(
            SymbolCenterDataset(train_rows, args.input_size, args.stride, augment=True, seed=args.seed),
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
            metrics, _preds, _errors = score_predictions(dev_preds, dev_golds, score_threshold, nms_threshold, args.max_per_page, len(dev_rows))
            grid_reports.append({"score_threshold": score_threshold, "nms_threshold": nms_threshold, "metrics": metrics})
    grid_reports.sort(key=selection_key, reverse=True)
    selected = grid_reports[0]
    locked_eval, locked_predictions, locked_errors = score_predictions(
        locked_preds,
        locked_golds,
        float(selected["score_threshold"]),
        float(selected["nms_threshold"]),
        args.max_per_page,
        len(locked_rows),
    )

    baseline_report_path = Path(args.yolo_baseline_report)
    yolo_locked = None
    union_eval = None
    if baseline_report_path.exists() and baseline_predictions_path.exists():
        baseline_report = json.loads(baseline_report_path.read_text(encoding="utf-8"))
        yolo_locked = baseline_report.get("locked") or baseline_report.get("dev")
        union_preds = union_predictions(load_page_predictions(baseline_predictions_path), locked_preds, args.union_heatmap_score_scale)
        union_eval, _union_predictions, _union_errors = score_predictions(
            union_preds,
            locked_golds,
            0.0,
            float(selected["nms_threshold"]),
            args.max_per_page,
            len(locked_rows),
        )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / "model.pt")
    write_json(
        checkpoint_dir / "model_metadata.json",
        {
            "model_type": "symbol_center_heatmap_probe_v24",
            "input_size": args.input_size,
            "stride": args.stride,
            "runtime_contract": {"model_input_features": ["image_tile_pixels"], "forbidden_runtime_features": FORBIDDEN_RUNTIME_FIELDS},
        },
    )
    error_buckets = Counter(f"{row['error']}:{row['label']}:{row['area_bucket']}" for row in locked_errors)
    report = {
        "version": "symbol_center_heatmap_probe_v24_eval",
        "claim_boundary": "Raster-only class-agnostic symbol center heatmap route for recall recovery; classification remains delegated to downstream type adapter.",
        "source_integrity": {
            "model_input": "raster_tile_pixels_only",
            "raw_label_or_semantic_type_used_as_runtime_feature": False,
            "forbidden_runtime_features": FORBIDDEN_RUNTIME_FIELDS,
            "locked_gold_use": "evaluation_only",
            "metric_scope": "page-level locked tile sample",
        },
        "baseline_to_beat": {
            "current_best_yolo_center_recall": 0.911595,
            "current_best_yolo_iou_0_30_recall": 0.719572,
            "current_best_yolo_precision": 0.096685,
        },
        "dataset": rel(data_dir),
        "checkpoint": rel(checkpoint_dir / "model.pt"),
        "config": vars(args) | {"device": str(device)},
        "counts": {
            "train_tiles": len(train_rows),
            "dev_tiles": len(dev_rows),
            "locked_tiles": len(locked_rows),
            "locked_pages": len({str(row.get("row_id")) for row in locked_rows}),
            "aligned_to_yolo_baseline_pages": bool(args.align_locked_to_yolo_baseline_pages and baseline_page_ids),
            "yolo_baseline_pages": len(baseline_page_ids),
            "filtered_to_exported_yolo_tiles": bool(args.filter_to_exported_yolo_tiles),
            "exported_yolo_train_source_tiles": len(train_source),
            "exported_yolo_dev_source_tiles": len(dev_source),
            "exported_yolo_locked_source_tiles": len(locked_source),
            "train_positive_tiles": sum(1 for row in train_rows if int((row.get("target_counts") or {}).get("symbols") or 0) > 0),
            "train_small_or_tiny_positive_tiles": sum(1 for row in train_rows if target_area_buckets(row) & {"tiny_le_64", "small_le_256"}),
        },
        "epoch_log": epoch_log,
        "threshold_grid": grid_reports,
        "selected_thresholds": {"score_threshold": float(selected["score_threshold"]), "nms_threshold": float(selected["nms_threshold"])},
        "locked_heatmap": locked_eval,
        "locked_yolo_baseline": yolo_locked,
        "locked_yolo_heatmap_union": union_eval,
        "error_buckets_top20": [{"bucket": key, "count": int(value)} for key, value in error_buckets.most_common(20)],
        "success_gate": {
            "stage_1_center_recall_min_0_94": locked_eval["symbol_bbox_center_recall"] >= 0.94 or bool(union_eval and union_eval["symbol_bbox_center_recall"] >= 0.94),
            "stage_1_iou_0_30_recall_min_0_78": locked_eval["symbol_bbox_iou_0_30"]["recall"] >= 0.78 or bool(union_eval and union_eval["symbol_bbox_iou_0_30"]["recall"] >= 0.78),
            "must_not_drop_center_recall_below_0_911595": locked_eval["symbol_bbox_center_recall"] >= 0.911595 or bool(union_eval and union_eval["symbol_bbox_center_recall"] >= 0.911595),
            "precision_improves_over_0_096685": locked_eval["symbol_bbox_iou_0_30"]["precision"] > 0.096685 or bool(union_eval and union_eval["symbol_bbox_iou_0_30"]["precision"] > 0.096685),
        },
        "memory_audit": memory_audit(device),
    }
    report["success_gate"]["passed"] = all(bool(value) for value in report["success_gate"].values())
    report["decision"] = "promote_center_route" if report["success_gate"]["passed"] else "keep_as_diagnostic_and_continue_center_model_work"
    write_json(Path(args.eval_output), report)
    write_jsonl(Path(args.predictions_output), locked_predictions)
    write_json(Path(args.eval_output).with_name("symbol_center_heatmap_probe_v24_error_buckets.json"), locked_errors[:2000])
    print(json.dumps({"locked_heatmap": locked_eval, "locked_union": union_eval, "gate": report["success_gate"], "decision": report["decision"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
