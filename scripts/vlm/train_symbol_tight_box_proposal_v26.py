#!/usr/bin/env python3
"""Train a raster-only dense tight-box symbol proposal head.

v25 proved that center recovery alone is not enough.  This v26 probe predicts
center heat, l/t/r/b distances, and a local IoU-quality score so decoding can
prefer tight boxes instead of flooding the page with fixed or poorly sized
boxes.  It is still a proposal frontend; symbol typing remains downstream.
"""

from __future__ import annotations

import argparse
import json
import random
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

from train_symbol_center_heatmap_probe_v24 import (
    YOLO_BASELINE_PREDICTIONS,
    YOLO_BASELINE_REPORT,
    YOLO_DIR,
    filter_rows_to_page_ids,
    filter_rows_with_exported_yolo_images,
    gaussian_2d,
    load_page_predictions,
    sample_tiles_area_aware,
    score_predictions,
    selection_key,
    union_predictions,
)
from train_symbol_tile_detector_v20 import FORBIDDEN_RUNTIME_FIELDS, bbox_iou, load_jsonl, rel, source_path, target_area_buckets, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21"
CHECKPOINT = ROOT / "checkpoints/symbol_tight_box_proposal_v26"
REPORT = ROOT / "reports/vlm/symbol_tight_box_proposal_v26_eval.json"
PREDICTIONS = ROOT / "reports/vlm/symbol_tight_box_proposal_v26_predictions.jsonl"


def clamp_box(box: list[float], width: float, height: float) -> list[float]:
    x1 = max(0.0, min(width, box[0]))
    y1 = max(0.0, min(height, box[1]))
    x2 = max(0.0, min(width, box[2]))
    y2 = max(0.0, min(height, box[3]))
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


class TightBoxDataset(Dataset):
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

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
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

        heat = np.zeros((1, self.grid_size, self.grid_size), dtype=np.float32)
        dist = np.zeros((4, self.grid_size, self.grid_size), dtype=np.float32)
        quality = np.zeros((1, self.grid_size, self.grid_size), dtype=np.float32)
        mask = np.zeros((1, self.grid_size, self.grid_size), dtype=np.float32)
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
            center_x = (gx + 0.5) * self.stride
            center_y = (gy + 0.5) * self.stride
            ltrb = [
                max(0.5, center_x - scaled[0]),
                max(0.5, center_y - scaled[1]),
                max(0.5, scaled[2] - center_x),
                max(0.5, scaled[3] - center_y),
            ]
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
            heat[0, y1:y2, x1:x2] = np.maximum(heat[0, y1:y2, x1:x2], kernel[ky1:ky2, kx1:kx2])
            dist[:, gy, gx] = np.log1p(np.asarray(ltrb, dtype=np.float32))
            mask[0, gy, gx] = 1.0
            pred_scaled = [center_x - ltrb[0], center_y - ltrb[1], center_x + ltrb[2], center_y + ltrb[3]]
            quality[0, gy, gx] = max(0.0, min(1.0, bbox_iou(pred_scaled, scaled)))
            gold.append(target)

        meta = {
            "id": row.get("id"),
            "row_id": row.get("row_id"),
            "tile_bbox": [left, top, right, bottom],
            "tile_size": [tile_w, tile_h],
            "gold": gold,
        }
        return image, torch.from_numpy(heat), torch.from_numpy(dist), torch.from_numpy(quality), torch.from_numpy(mask), meta


def collate(batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    images, heat, dist, quality, mask, metas = zip(*batch, strict=True)
    return torch.stack(list(images)), torch.stack(list(heat)), torch.stack(list(dist)), torch.stack(list(quality)), torch.stack(list(mask)), list(metas)


class TightBoxNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
        )
        self.obj = nn.Conv2d(128, 1, 1)
        self.dist = nn.Conv2d(128, 4, 1)
        self.quality = nn.Conv2d(128, 1, 1)
        nn.init.constant_(self.obj.bias, -5.0)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.backbone(images)
        return self.obj(feat), self.dist(feat), self.quality(feat)


def center_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    positive = targets.gt(0.05)
    weights = torch.where(positive, torch.full_like(targets, 28.0), torch.full_like(targets, 1.0))
    focal = torch.where(positive, (1.0 - probs).pow(2.0), probs.pow(2.0))
    return (weights * focal * bce).mean()


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> dict[str, float]:
    model.train()
    totals = Counter()
    seen = 0
    for images, heat, dist, quality, mask, _metas in loader:
        images = images.to(device, non_blocking=True)
        heat = heat.to(device, non_blocking=True)
        dist = dist.to(device, non_blocking=True)
        quality = quality.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        obj_logits, pred_dist, pred_quality = model(images)
        obj_loss = center_loss(obj_logits, heat)
        denom = mask.sum().clamp_min(1.0)
        dist_loss = (F.smooth_l1_loss(pred_dist, dist, reduction="none") * mask).sum() / denom
        quality_loss = (F.binary_cross_entropy_with_logits(pred_quality, quality, reduction="none") * mask).sum() / denom
        loss = obj_loss + 0.20 * dist_loss + 0.10 * quality_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        batch = int(images.shape[0])
        seen += batch
        totals["loss"] += float(loss.detach().cpu()) * batch
        totals["obj_loss"] += float(obj_loss.detach().cpu()) * batch
        totals["dist_loss"] += float(dist_loss.detach().cpu()) * batch
        totals["quality_loss"] += float(quality_loss.detach().cpu()) * batch
    return {key: round(value / max(seen, 1), 6) for key, value in totals.items()}


def decode(obj_logits: torch.Tensor, dist_logits: torch.Tensor, quality_logits: torch.Tensor, metas: list[dict[str, Any]], input_size: int, stride: int, score_threshold: float, topk: int, min_side: float, max_side: float) -> dict[str, list[dict[str, Any]]]:
    obj = torch.sigmoid(obj_logits).detach().cpu()
    quality = torch.sigmoid(quality_logits).detach().cpu()
    dist = torch.expm1(dist_logits.detach().cpu()).clamp(min=min_side / 2.0, max=max_side)
    score_map = obj * quality.clamp_min(0.05)
    pooled = F.max_pool2d(score_map, kernel_size=3, stride=1, padding=1)
    peaks = score_map * score_map.eq(pooled)
    page_preds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    batch, _channels, grid_h, grid_w = peaks.shape
    for item in range(batch):
        meta = metas[item]
        left, top, _right, _bottom = [float(v) for v in meta["tile_bbox"]]
        tile_w, tile_h = [float(v) for v in meta["tile_size"]]
        sx = tile_w / input_size
        sy = tile_h / input_size
        scores, indices = torch.topk(peaks[item, 0].reshape(-1), k=min(topk, grid_h * grid_w))
        for score, flat_index in zip(scores.tolist(), indices.tolist(), strict=True):
            if score < score_threshold:
                continue
            gy = int(flat_index) // grid_w
            gx = int(flat_index) % grid_w
            cx = (gx + 0.5) * stride
            cy = (gy + 0.5) * stride
            l, t, r, b = [float(v) for v in dist[item, :, gy, gx].tolist()]
            box_tile = [(cx - l) * sx, (cy - t) * sy, (cx + r) * sx, (cy + b) * sy]
            box_tile = clamp_box(box_tile, tile_w, tile_h)
            bw = box_tile[2] - box_tile[0]
            bh = box_tile[3] - box_tile[1]
            if bw < min_side or bh < min_side or bw > max_side or bh > max_side:
                continue
            page_preds[str(meta["row_id"])].append(
                {
                    "bbox": [box_tile[0] + left, box_tile[1] + top, box_tile[2] + left, box_tile[3] + top],
                    "label_id": 5,
                    "label": "generic_symbol",
                    "score": float(score),
                    "tile_id": meta["id"],
                    "proposal_source": "tight_box_v26",
                }
            )
    return page_preds


def collect(model: nn.Module, rows: list[dict[str, Any]], device: torch.device, args: argparse.Namespace) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, dict[str, Any]]]]:
    dataset = TightBoxDataset(rows, args.input_size, args.stride, augment=False, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda", collate_fn=collate)
    page_preds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    page_golds: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    model.eval()
    with torch.no_grad():
        for images, _heat, _dist, _quality, _mask, metas in loader:
            obj_logits, dist_logits, quality_logits = model(images.to(device, non_blocking=True))
            batch_preds = decode(obj_logits, dist_logits, quality_logits, metas, args.input_size, args.stride, args.decode_score_threshold, args.topk_per_tile, args.min_box_side, args.max_box_side)
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


def merge_page_predictions(preds: dict[str, list[dict[str, Any]]], score_threshold: float, nms_threshold: float, max_per_page: int) -> dict[str, list[dict[str, Any]]]:
    return {row_id: merge_one(items, score_threshold, nms_threshold, max_per_page) for row_id, items in preds.items()}


def merge_one(preds: list[dict[str, Any]], score_threshold: float, nms_threshold: float, max_per_page: int) -> list[dict[str, Any]]:
    filtered = [p for p in preds if float(p.get("score", 0.0)) >= score_threshold]
    if not filtered:
        return []
    boxes = torch.tensor([p["bbox"] for p in filtered], dtype=torch.float32)
    scores = torch.tensor([float(p.get("score", 0.0)) for p in filtered], dtype=torch.float32)
    keep = nms(boxes, scores, nms_threshold)
    keep_indices = sorted([int(i) for i in keep.tolist()], key=lambda i: float(filtered[i].get("score", 0.0)), reverse=True)
    return [filtered[i] for i in keep_indices[:max_per_page]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA))
    parser.add_argument("--checkpoint-dir", default=str(CHECKPOINT))
    parser.add_argument("--eval-output", default=str(REPORT))
    parser.add_argument("--predictions-output", default=str(PREDICTIONS))
    parser.add_argument("--yolo-baseline-report", default=str(YOLO_BASELINE_REPORT))
    parser.add_argument("--yolo-baseline-predictions", default=str(YOLO_BASELINE_PREDICTIONS))
    parser.add_argument("--yolo-dir", default=str(YOLO_DIR))
    parser.add_argument("--limit-train-tiles", type=int, default=12000)
    parser.add_argument("--limit-dev-tiles", type=int, default=1600)
    parser.add_argument("--limit-locked-tiles", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--input-size", type=int, default=384)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--decode-score-threshold", type=float, default=0.001)
    parser.add_argument("--score-threshold-grid", default="0.003,0.005,0.01,0.02,0.03,0.05")
    parser.add_argument("--nms-threshold-grid", default="0.15,0.25,0.35,0.45")
    parser.add_argument("--topk-per-tile", type=int, default=120)
    parser.add_argument("--max-per-page", type=int, default=900)
    parser.add_argument("--min-box-side", type=float, default=2.0)
    parser.add_argument("--max-box-side", type=float, default=128.0)
    parser.add_argument("--union-score-scale", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260510)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = Path(args.data)
    train_source = filter_rows_with_exported_yolo_images(load_jsonl(data_dir / "train.jsonl"), "train", Path(args.yolo_dir))
    dev_source = filter_rows_with_exported_yolo_images(load_jsonl(data_dir / "dev.jsonl"), "dev", Path(args.yolo_dir))
    locked_source = filter_rows_with_exported_yolo_images(load_jsonl(data_dir / "locked.jsonl"), "locked", Path(args.yolo_dir))
    train_rows = sample_tiles_area_aware(train_source, args.limit_train_tiles, args.seed, 0.95, 0.85)
    dev_rows = sample_tiles_area_aware(dev_source, args.limit_dev_tiles, args.seed + 1, 0.85, 0.75)
    locked_rows = sample_tiles_area_aware(locked_source, args.limit_locked_tiles, args.seed + 2, 0.85, 0.75)
    baseline_predictions_path = Path(args.yolo_baseline_predictions)
    baseline_page_ids = set(load_page_predictions(baseline_predictions_path)) if baseline_predictions_path.exists() else set()
    if baseline_page_ids:
        aligned = filter_rows_to_page_ids(locked_rows, baseline_page_ids)
        if aligned:
            locked_rows = aligned

    model = TightBoxNet().to(device)
    loader = DataLoader(
        TightBoxDataset(train_rows, args.input_size, args.stride, augment=True, seed=args.seed),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.1)
    epoch_log: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        row = train_epoch(model, loader, optimizer, device)
        scheduler.step()
        row["epoch"] = epoch
        epoch_log.append(row)

    dev_preds, dev_golds = collect(model, dev_rows, device, args)
    locked_preds, locked_golds = collect(model, locked_rows, device, args)
    grid_reports = []
    for score_threshold in [float(item) for item in args.score_threshold_grid.split(",") if item.strip()]:
        for nms_threshold in [float(item) for item in args.nms_threshold_grid.split(",") if item.strip()]:
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

    yolo_locked = None
    union_eval = None
    if Path(args.yolo_baseline_report).exists() and baseline_predictions_path.exists():
        baseline_report = json.loads(Path(args.yolo_baseline_report).read_text(encoding="utf-8"))
        yolo_locked = baseline_report.get("locked") or baseline_report.get("dev")
        union_preds = union_predictions(load_page_predictions(baseline_predictions_path), locked_preds, args.union_score_scale)
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
            "model_type": "symbol_tight_box_proposal_v26",
            "input_size": args.input_size,
            "stride": args.stride,
            "runtime_contract": {"model_input_features": ["image_tile_pixels"], "forbidden_runtime_features": FORBIDDEN_RUNTIME_FIELDS},
        },
    )
    report = {
        "version": "symbol_tight_box_proposal_v26_eval",
        "claim_boundary": "Raster-only dense tight-box proposal head; symbol typing remains downstream.",
        "source_integrity": {
            "model_input": "raster_tile_pixels_only",
            "raw_label_or_semantic_type_used_as_runtime_feature": False,
            "forbidden_runtime_features": FORBIDDEN_RUNTIME_FIELDS,
            "locked_gold_use": "evaluation_only",
            "metric_scope": "page-level locked tile sample",
        },
        "dataset": rel(data_dir),
        "checkpoint": rel(checkpoint_dir / "model.pt"),
        "config": vars(args) | {"device": str(device)},
        "counts": {
            "train_tiles": len(train_rows),
            "dev_tiles": len(dev_rows),
            "locked_tiles": len(locked_rows),
            "locked_pages": len({str(row.get("row_id")) for row in locked_rows}),
            "aligned_to_yolo_baseline_pages": bool(baseline_page_ids),
            "train_positive_tiles": sum(1 for row in train_rows if int((row.get("target_counts") or {}).get("symbols") or 0) > 0),
            "train_small_or_tiny_positive_tiles": sum(1 for row in train_rows if target_area_buckets(row) & {"tiny_le_64", "small_le_256"}),
        },
        "epoch_log": epoch_log,
        "threshold_grid": grid_reports,
        "selected_thresholds": {"score_threshold": float(selected["score_threshold"]), "nms_threshold": float(selected["nms_threshold"])},
        "locked_tight_box": locked_eval,
        "locked_yolo_baseline": yolo_locked,
        "locked_yolo_tight_box_union": union_eval,
        "error_buckets_top20": [{"bucket": key, "count": int(value)} for key, value in Counter(f"{row['error']}:{row['label']}:{row['area_bucket']}" for row in locked_errors).most_common(20)],
        "stage_gate": {
            "center_recall_min_0_94": locked_eval["symbol_bbox_center_recall"] >= 0.94 or bool(union_eval and union_eval["symbol_bbox_center_recall"] >= 0.94),
            "iou_0_30_recall_min_0_82": locked_eval["symbol_bbox_iou_0_30"]["recall"] >= 0.82 or bool(union_eval and union_eval["symbol_bbox_iou_0_30"]["recall"] >= 0.82),
            "precision_min_0_12": locked_eval["symbol_bbox_iou_0_30"]["precision"] >= 0.12 or bool(union_eval and union_eval["symbol_bbox_iou_0_30"]["precision"] >= 0.12),
            "candidate_inflation_max_7": locked_eval["candidate_inflation"] <= 7.0 or bool(union_eval and union_eval["candidate_inflation"] <= 7.0),
        },
    }
    report["stage_gate"]["passed"] = all(bool(value) for value in report["stage_gate"].values())
    report["decision"] = "promote_tight_box_route" if report["stage_gate"]["passed"] else "failed_gate_continue_instance_mask_or_stronger_backbone"
    write_json(Path(args.eval_output), report)
    write_jsonl(Path(args.predictions_output), locked_predictions)
    write_json(Path(args.eval_output).with_name("symbol_tight_box_proposal_v26_error_buckets.json"), locked_errors[:2000])
    print(json.dumps({"locked_tight_box": locked_eval, "locked_union": union_eval, "gate": report["stage_gate"], "decision": report["decision"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
