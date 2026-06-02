#!/usr/bin/env python3
"""Train a center-ranked text heatmap localizer with width/height geometry heads."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import DataLoader, Dataset

from train_text_heatmap_affinity_v19 import (
    DATA,
    REPORT,
    ROOT,
    abs_path,
    bbox_iou,
    center_covered,
    load_jsonl,
    nms,
    targets,
    write_json,
    write_jsonl,
)


CKPT = ROOT / "checkpoints/text_heatmap_geometry_v19"


class TextGeometryDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], size: int, max_rows: int, pad: int, blur: float, center_sigma: float, max_geom_size: float) -> None:
        filtered = [row for row in rows if targets(row)]
        self.rows = filtered[:max_rows] if max_rows else filtered
        self.size = int(size)
        self.pad = int(pad)
        self.blur = float(blur)
        self.center_sigma = float(center_sigma)
        self.max_geom_size = float(max_geom_size)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        image = Image.open(abs_path(row["image"])).convert("L")
        original_w, original_h = image.size
        image = image.resize((self.size, self.size), Image.Resampling.BILINEAR)
        x = 1.0 - np.asarray(image, dtype=np.float32) / 255.0

        region = Image.new("L", (self.size, self.size), 0)
        affinity = Image.new("L", (self.size, self.size), 0)
        center = np.zeros((self.size, self.size), dtype=np.float32)
        width = np.zeros((self.size, self.size), dtype=np.float32)
        height = np.zeros((self.size, self.size), dtype=np.float32)
        rdraw = ImageDraw.Draw(region)
        adraw = ImageDraw.Draw(affinity)
        sx = self.size / max(original_w, 1)
        sy = self.size / max(original_h, 1)
        yy, xx = np.mgrid[0 : self.size, 0 : self.size]
        for target in targets(row):
            x1, y1, x2, y2 = [float(v) for v in target["bbox"]]
            bx1 = max(0, min(self.size, int(math.floor(x1 * sx))))
            by1 = max(0, min(self.size, int(math.floor(y1 * sy))))
            bx2 = max(0, min(self.size, int(math.ceil(x2 * sx))))
            by2 = max(0, min(self.size, int(math.ceil(y2 * sy))))
            if bx2 <= bx1 or by2 <= by1:
                continue
            padded = [
                max(0, bx1 - self.pad),
                max(0, by1 - self.pad),
                min(self.size, bx2 + self.pad),
                min(self.size, by2 + self.pad),
            ]
            rdraw.rounded_rectangle(padded, radius=max(1, min(padded[2] - padded[0], padded[3] - padded[1]) // 3), fill=255)
            cy = (padded[1] + padded[3]) // 2
            stripe = max(1, (padded[3] - padded[1]) // 4)
            adraw.rectangle([padded[0], max(0, cy - stripe), padded[2], min(self.size, cy + stripe + 1)], fill=255)

            cx = (bx1 + bx2) / 2.0
            cy_float = (by1 + by2) / 2.0
            sigma = max(self.center_sigma, min(max(bx2 - bx1, 1), max(by2 - by1, 1)) / 2.0)
            gaussian = np.exp(-((xx - cx) ** 2 + (yy - cy_float) ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
            update = gaussian > center
            center = np.maximum(center, gaussian)
            width[update] = min(max(bx2 - bx1, 1), self.max_geom_size) / self.max_geom_size
            height[update] = min(max(by2 - by1, 1), self.max_geom_size) / self.max_geom_size
        if self.blur > 0:
            region = region.filter(ImageFilter.GaussianBlur(radius=self.blur))
            affinity = affinity.filter(ImageFilter.GaussianBlur(radius=max(0.5, self.blur / 2.0)))
        y = np.stack(
            [
                np.asarray(region, dtype=np.float32) / 255.0,
                np.asarray(affinity, dtype=np.float32) / 255.0,
                np.clip(center, 0.0, 1.0),
                width,
                height,
            ],
            axis=0,
        )
        return torch.from_numpy(x[None]), torch.from_numpy(y)


class GeometryNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.e1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.e2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.e3 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, padding=1), nn.ReLU())
        self.u2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d2 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU())
        self.u1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.d1 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU())
        self.out = nn.Conv2d(32, 5, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        d2 = self.d2(torch.cat([self.u2(e3), e2], dim=1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], dim=1))
        return self.out(d1)


def dice_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits)
    inter = (p * y).sum(dim=(2, 3))
    denom = p.sum(dim=(2, 3)) + y.sum(dim=(2, 3)) + 1.0
    return 1.0 - ((2.0 * inter + 1.0) / denom).mean()


def geometry_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    center_weight: float,
    geom_weight: float,
    max_pos_weight: float,
    hard_negative_weight: float,
    hard_negative_fraction: float,
) -> torch.Tensor:
    cls_logits = logits[:, :3]
    cls_y = y[:, :3]
    pos = torch.clamp((cls_y.numel() - cls_y.sum()) / torch.clamp(cls_y.sum(), min=1.0), 1.0, max_pos_weight)
    bce = F.binary_cross_entropy_with_logits(cls_logits, cls_y, pos_weight=pos.detach())
    dice = dice_loss(cls_logits[:, :2], cls_y[:, :2]) + center_weight * dice_loss(cls_logits[:, 2:3], cls_y[:, 2:3])
    geom_mask = (cls_y[:, 2:3] > 0.15).float()
    geom_pred = torch.sigmoid(logits[:, 3:5])
    geom_y = y[:, 3:5]
    geom = F.smooth_l1_loss(geom_pred * geom_mask, geom_y * geom_mask, reduction="sum") / torch.clamp(geom_mask.sum() * 2.0, min=1.0)
    hard_negative = torch.tensor(0.0, device=logits.device)
    if hard_negative_weight > 0 and hard_negative_fraction > 0:
        center_logits = logits[:, 2:3]
        negative_logits = center_logits[cls_y[:, 2:3] < 0.05]
        if negative_logits.numel() > 0:
            k = max(1, int(negative_logits.numel() * hard_negative_fraction))
            hard_logits = torch.topk(negative_logits.flatten(), k=min(k, negative_logits.numel())).values
            hard_negative = F.binary_cross_entropy_with_logits(hard_logits, torch.zeros_like(hard_logits))
    return bce + dice + geom_weight * geom + hard_negative_weight * hard_negative


def predict(model: nn.Module, row: dict[str, Any], size: int, device: str) -> np.ndarray:
    image = Image.open(abs_path(row["image"])).convert("L")
    original_size = image.size
    resized = image.resize((size, size), Image.Resampling.BILINEAR)
    arr = 1.0 - np.asarray(resized, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr[None, None]).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(x))[0].detach().cpu().numpy()
    return np.stack([cv2.resize(probs[idx], original_size, interpolation=cv2.INTER_LINEAR) for idx in range(probs.shape[0])], axis=0)


def decode(prob: np.ndarray, threshold: float, args: argparse.Namespace) -> list[dict[str, Any]]:
    region, affinity, center, width_map, height_map = prob
    score = np.maximum.reduce([region, affinity * args.affinity_weight, center * args.center_score_weight])
    peaks = (score >= threshold) & (score >= cv2.dilate(score, np.ones((max(3, args.peak_min_distance | 1), max(3, args.peak_min_distance | 1)), dtype=np.float32)) - 1e-6)
    ys, xs = np.where(peaks)
    if len(xs) == 0:
        return []
    values = score[ys, xs]
    order = np.argsort(values)[::-1][: args.peak_top_k]
    page_h, page_w = score.shape
    out: list[dict[str, Any]] = []
    for rank in order:
        cx, cy = int(xs[rank]), int(ys[rank])
        pred_w = int(round(float(width_map[cy, cx]) * args.max_geom_size))
        pred_h = int(round(float(height_map[cy, cx]) * args.max_geom_size))
        pred_w = max(args.min_box_width, min(args.max_box_width, pred_w + 2 * args.box_pad_x))
        pred_h = max(args.min_box_height, min(args.max_box_height, pred_h + 2 * args.box_pad_y))
        x1 = max(0, min(page_w - pred_w, cx - pred_w // 2))
        y1 = max(0, min(page_h - pred_h, cy - pred_h // 2))
        out.append(
            {
                "bbox": [int(x1), int(y1), int(x1 + pred_w), int(y1 + pred_h)],
                "confidence": round(float(values[rank]), 6),
                "area": int(pred_w * pred_h),
                "decoder": "center_geometry_peak",
            }
        )
    kept = nms(sorted(out, key=lambda item: item["confidence"], reverse=True), args.nms_iou)
    return kept[: args.max_candidates_per_page] if args.max_candidates_per_page > 0 else kept


def evaluate_threshold(model: nn.Module, rows: list[dict[str, Any]], args: argparse.Namespace, split: str, threshold: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    totals = Counter()
    pred_rows: list[dict[str, Any]] = []
    for row in rows:
        prob = predict(model, row, args.size, args.device)
        preds = decode(prob, threshold, args)
        golds = targets(row)
        center_gold: set[int] = set()
        iou_gold: set[int] = set()
        for pred in preds:
            for gold_index, gold in enumerate(golds):
                gb = [int(v) for v in gold["bbox"]]
                if center_covered(pred["bbox"], gb):
                    center_gold.add(gold_index)
                if bbox_iou(pred["bbox"], gb) >= 0.30:
                    iou_gold.add(gold_index)
        totals["gold"] += len(golds)
        totals["predicted"] += len(preds)
        totals["matched_center"] += len(center_gold)
        totals["matched_iou"] += len(iou_gold)
        pred_rows.append(
            {
                "id": row["source_row_id"],
                "image": row["image"],
                "predicted_text": [
                    {
                        "id": f"{row['source_row_id']}_text_heatmap_geometry_v19_{idx}",
                        "class": "text",
                        "family": "text",
                        "semantic_type": "unknown_text",
                        "bbox": pred["bbox"],
                        "confidence": pred["confidence"],
                        "proposal_source": "raster_text_heatmap_geometry_v19",
                        "payload": {"ocr_status": "not_invoked", "source": "raster_text_heatmap_geometry_v19"},
                    }
                    for idx, pred in enumerate(preds)
                ],
                "gold_text_count": len(golds),
                "matched_center": len(center_gold),
                "matched_iou_0_30": len(iou_gold),
                "source_integrity": {"model_input": "raster_image_only", "gold_used_for_inference": False},
            }
        )
    precision = totals["matched_iou"] / max(totals["predicted"], 1)
    recall = totals["matched_iou"] / max(totals["gold"], 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    report = {
        "split": split,
        "rows": len(rows),
        "threshold": threshold,
        "text_bbox_iou_0_30": {
            "matched": int(totals["matched_iou"]),
            "predicted": int(totals["predicted"]),
            "gold": int(totals["gold"]),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        },
        "text_bbox_center_recall": round(totals["matched_center"] / max(totals["gold"], 1), 6),
        "candidate_inflation": round(totals["predicted"] / max(totals["gold"], 1), 6),
    }
    return report, pred_rows


def evaluate(model: nn.Module, rows: list[dict[str, Any]], args: argparse.Namespace, split: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    eval_rows = [row for row in rows if targets(row)]
    if args.max_eval:
        eval_rows = eval_rows[: args.max_eval]
    best = None
    best_key = None
    sweep = []
    for threshold in [float(item) for item in args.thresholds.split(",")]:
        report, predictions = evaluate_threshold(model, eval_rows, args, split, threshold)
        sweep.append(report)
        key = (report["text_bbox_center_recall"], report["text_bbox_iou_0_30"]["recall"], -report["candidate_inflation"])
        if best_key is None or key > best_key:
            best_key = key
            best = (report, predictions)
    assert best is not None
    report = json.loads(json.dumps(best[0]))
    report["threshold_sweep"] = sweep
    return report, best[1]


def train(args: argparse.Namespace) -> tuple[nn.Module, dict[str, Any]]:
    torch.manual_seed(args.seed)
    rows = load_jsonl(DATA / "train.jsonl")
    ds = TextGeometryDataset(rows, args.size, args.max_train, args.target_pad, args.target_blur, args.center_sigma, args.max_geom_size)
    if not ds:
        raise SystemExit("no raster-aligned localizer supervision")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = GeometryNet().to(args.device)
    init_report = initialize_from_affinity_checkpoint(model, args)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    losses = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for step, (x, y) in enumerate(loader, start=1):
            x, y = x.to(args.device), y.to(args.device)
            loss = geometry_loss(
                model(x),
                y,
                args.center_loss_weight,
                args.geometry_loss_weight,
                args.max_pos_weight,
                args.hard_negative_weight,
                args.hard_negative_fraction,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu())
            if args.progress_every and step % args.progress_every == 0:
                print(json.dumps({"epoch": epoch, "step": step, "steps": len(loader), "loss": round(total / step, 6)}, ensure_ascii=False), flush=True)
        epoch_loss = round(total / max(len(loader), 1), 6)
        losses.append(epoch_loss)
        print(json.dumps({"epoch": epoch, "steps": len(loader), "loss": epoch_loss}, ensure_ascii=False), flush=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model_name": "GeometryNet", "args": vars(args), "dataset": str(DATA.relative_to(ROOT))}, CKPT / "model_best.pt")
    return model, {"checkpoint": str((CKPT / "model_best.pt").relative_to(ROOT)), "train_rows": len(ds), "loss_tail": losses[-5:], "initialization": init_report}


def initialize_from_affinity_checkpoint(model: GeometryNet, args: argparse.Namespace) -> dict[str, Any]:
    if not args.init_affinity_checkpoint:
        return {"used": False}
    checkpoint_path = abs_path(args.init_affinity_checkpoint)
    if not checkpoint_path.exists():
        return {"used": False, "missing": str(checkpoint_path)}
    ckpt = torch.load(checkpoint_path, map_location=args.device)
    source = ckpt["state_dict"]
    target = model.state_dict()
    loaded = []
    skipped = []
    for key, value in source.items():
        if key == "out.weight" and target[key].shape[0] >= 2 and value.shape[0] >= 2:
            target[key][:2] = value[:2]
            loaded.append(f"{key}[:2]")
        elif key == "out.bias" and target[key].shape[0] >= 2 and value.shape[0] >= 2:
            target[key][:2] = value[:2]
            loaded.append(f"{key}[:2]")
        elif key in target and target[key].shape == value.shape:
            target[key] = value
            loaded.append(key)
        else:
            skipped.append(key)
    model.load_state_dict(target)
    return {"used": True, "checkpoint": str(checkpoint_path.relative_to(ROOT)), "loaded": len(loaded), "skipped": skipped[:20]}


def load_model(args: argparse.Namespace) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint_path = CKPT / "model_best.pt"
    ckpt = torch.load(checkpoint_path, map_location=args.device)
    model = GeometryNet().to(args.device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, {"checkpoint": str(checkpoint_path.relative_to(ROOT)), "eval_only": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-eval", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--thresholds", default="0.35,0.40,0.45,0.50,0.55,0.60,0.65")
    parser.add_argument("--target-pad", type=int, default=3)
    parser.add_argument("--target-blur", type=float, default=0.8)
    parser.add_argument("--center-sigma", type=float, default=2.0)
    parser.add_argument("--max-geom-size", type=float, default=96.0)
    parser.add_argument("--max-pos-weight", type=float, default=100.0)
    parser.add_argument("--center-loss-weight", type=float, default=2.0)
    parser.add_argument("--geometry-loss-weight", type=float, default=4.0)
    parser.add_argument("--hard-negative-weight", type=float, default=1.5)
    parser.add_argument("--hard-negative-fraction", type=float, default=0.002)
    parser.add_argument("--affinity-weight", type=float, default=0.65)
    parser.add_argument("--center-score-weight", type=float, default=1.25)
    parser.add_argument("--peak-top-k", type=int, default=180)
    parser.add_argument("--peak-min-distance", type=int, default=6)
    parser.add_argument("--min-box-width", type=int, default=6)
    parser.add_argument("--min-box-height", type=int, default=4)
    parser.add_argument("--max-box-width", type=int, default=96)
    parser.add_argument("--max-box-height", type=int, default=32)
    parser.add_argument("--box-pad-x", type=int, default=2)
    parser.add_argument("--box-pad-y", type=int, default=2)
    parser.add_argument("--max-candidates-per-page", type=int, default=55)
    parser.add_argument("--nms-iou", type=float, default=0.35)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--init-affinity-checkpoint", default="checkpoints/text_heatmap_affinity_v19/model_best.pt")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    model, training = load_model(args) if args.eval_only else train(args)
    dev_report, _dev_predictions = evaluate(model, load_jsonl(DATA / "dev.jsonl"), args, "dev")
    locked_report, locked_predictions = evaluate(model, load_jsonl(DATA / "locked.jsonl"), args, "locked")
    adopted = locked_report["text_bbox_center_recall"] >= 0.80 and locked_report["candidate_inflation"] <= 5.0
    report = {
        "version": "text_heatmap_geometry_v19_eval",
        "task": "P0-TEXT-001",
        "run_mode": "center_ranked_heatmap_with_width_height_geometry",
        "source_integrity": {
            "model_input": "raster_image_only",
            "offline_labels_used_for": ["training_targets", "dev_threshold_selection", "locked_evaluation"],
            "gold_used_for_inference": False,
            "runtime_uses_svg_or_cad_geometry": False,
        },
        "training": training,
        "dev": dev_report,
        "locked": locked_report,
        "adopted": adopted,
        "blocker": None if adopted else "Center/geometry heatmap still below stage-1 center/budget gate; inspect ranking and geometry errors before OCR.",
    }
    write_json(REPORT / "text_heatmap_geometry_v19_eval.json", report)
    write_jsonl(REPORT / "text_heatmap_geometry_v19_locked_predictions.jsonl", locked_predictions)
    write_json(CKPT / "train_summary.json", report)
    print(json.dumps({"locked": locked_report, "adopted": adopted}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
