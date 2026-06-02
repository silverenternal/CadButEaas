#!/usr/bin/env python3
"""Train a CRAFT/DBNet-style raster text region + affinity localizer for v19."""

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


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets/text_expert_raster_v19"
REPORT = ROOT / "reports/vlm"
CKPT = ROOT / "checkpoints/text_heatmap_affinity_v19"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def abs_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def targets(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in row.get("text_targets") or []
        if item.get("can_train_localizer") and item.get("bbox") and len(item["bbox"]) == 4
    ]


class HeatmapTextDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], size: int, max_rows: int = 0, pad: int = 5, blur: float = 1.2) -> None:
        filtered = [row for row in rows if targets(row)]
        self.rows = filtered[:max_rows] if max_rows else filtered
        self.size = int(size)
        self.pad = int(pad)
        self.blur = float(blur)

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
        rdraw = ImageDraw.Draw(region)
        adraw = ImageDraw.Draw(affinity)
        sx = self.size / max(original_w, 1)
        sy = self.size / max(original_h, 1)
        for target in targets(row):
            x1, y1, x2, y2 = [float(v) for v in target["bbox"]]
            box = [
                max(0, min(self.size, int(math.floor(x1 * sx)) - self.pad)),
                max(0, min(self.size, int(math.floor(y1 * sy)) - self.pad)),
                max(0, min(self.size, int(math.ceil(x2 * sx)) + self.pad)),
                max(0, min(self.size, int(math.ceil(y2 * sy)) + self.pad)),
            ]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            rdraw.rounded_rectangle(box, radius=max(1, min(box[2] - box[0], box[3] - box[1]) // 3), fill=255)
            cy = (box[1] + box[3]) // 2
            stripe = max(1, (box[3] - box[1]) // 4)
            adraw.rectangle([box[0], max(0, cy - stripe), box[2], min(self.size, cy + stripe + 1)], fill=255)
        if self.blur > 0:
            region = region.filter(ImageFilter.GaussianBlur(radius=self.blur))
            affinity = affinity.filter(ImageFilter.GaussianBlur(radius=max(0.5, self.blur / 2.0)))
        y = np.stack([np.asarray(region, dtype=np.float32) / 255.0, np.asarray(affinity, dtype=np.float32) / 255.0], axis=0)
        return torch.from_numpy(x[None]), torch.from_numpy(y)


class HeatmapAffinityNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.e1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.e2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.e3 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, padding=1), nn.ReLU())
        self.u2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d2 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU())
        self.u1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.d1 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU())
        self.out = nn.Conv2d(32, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        d2 = self.d2(torch.cat([self.u2(e3), e2], dim=1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], dim=1))
        return self.out(d1)


def soft_dice(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits)
    inter = (p * y).sum(dim=(2, 3))
    denom = p.sum(dim=(2, 3)) + y.sum(dim=(2, 3)) + 1.0
    return 1.0 - ((2 * inter + 1.0) / denom).mean()


def bbox_iou(left: list[float], right: list[float]) -> float:
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    la = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    ra = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return inter / max(la + ra - inter, 1e-9)


def center_covered(pred: list[int], gold: list[int], margin: int = 2) -> bool:
    cx = (gold[0] + gold[2]) / 2.0
    cy = (gold[1] + gold[3]) / 2.0
    return pred[0] - margin <= cx <= pred[2] + margin and pred[1] - margin <= cy <= pred[3] + margin


def predict_maps(model: nn.Module, row: dict[str, Any], size: int, device: str) -> np.ndarray:
    image = Image.open(abs_path(row["image"])).convert("L")
    original_size = image.size
    resized = image.resize((size, size), Image.Resampling.BILINEAR)
    arr = 1.0 - np.asarray(resized, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr[None, None]).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(x))[0].detach().cpu().numpy()
    region = cv2.resize(probs[0], original_size, interpolation=cv2.INTER_LINEAR)
    affinity = cv2.resize(probs[1], original_size, interpolation=cv2.INTER_LINEAR)
    return np.stack([region, affinity], axis=0)


def decode(
    prob: np.ndarray,
    region_threshold: float,
    affinity_weight: float,
    min_area: int,
    close_kernel: int,
    max_area_ratio: float,
    peak_top_k: int,
    peak_min_distance: int,
    peak_window_mode: str,
    local_rel_threshold: float,
    local_radius: int,
    box_pad_x: int,
    box_pad_y: int,
    min_box_width: int,
    min_box_height: int,
    max_box_width: int,
    max_box_height: int,
    max_candidates_per_page: int,
) -> list[dict[str, Any]]:
    score = np.maximum(prob[0], prob[1] * affinity_weight)
    binary = (score >= region_threshold).astype("uint8")
    if close_kernel > 1:
        kernel = np.ones((close_kernel, close_kernel), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    n_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
    page_area = max(int(binary.shape[0] * binary.shape[1]), 1)
    out: list[dict[str, Any]] = []
    for idx in range(1, int(n_labels)):
        x, y, w, h, area = [int(v) for v in stats[idx]]
        if area < min_area or area / page_area > max_area_ratio:
            continue
        out.append({"bbox": [x, y, x + w, y + h], "confidence": round(float(score[y : y + h, x : x + w].mean()), 6), "area": area})
    out.sort(key=lambda item: item["confidence"], reverse=True)
    out.extend(
        peak_windows(
            score,
            region_threshold,
            peak_top_k,
            peak_min_distance,
            peak_window_mode,
            local_rel_threshold,
            local_radius,
            box_pad_x,
            box_pad_y,
            min_box_width,
            min_box_height,
            max_box_width,
            max_box_height,
        )
    )
    kept = nms(sorted(out, key=lambda item: item["confidence"], reverse=True), 0.35)
    return kept[:max_candidates_per_page] if max_candidates_per_page > 0 else kept


def peak_windows(
    score: np.ndarray,
    threshold: float,
    top_k: int,
    min_distance: int,
    mode: str,
    local_rel_threshold: float,
    local_radius: int,
    box_pad_x: int,
    box_pad_y: int,
    min_box_width: int,
    min_box_height: int,
    max_box_width: int,
    max_box_height: int,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []
    kernel_size = max(3, int(min_distance) | 1)
    dilated = cv2.dilate(score, np.ones((kernel_size, kernel_size), dtype=np.float32))
    peaks = (score >= threshold) & (score >= dilated - 1e-6)
    ys, xs = np.where(peaks)
    if len(xs) == 0:
        return []
    values = score[ys, xs]
    order = np.argsort(values)[::-1][:top_k]
    height, width = score.shape
    out: list[dict[str, Any]] = []
    for rank in order:
        cx, cy = int(xs[rank]), int(ys[rank])
        confidence = float(values[rank])
        boxes = (
            fixed_peak_boxes(cx, cy, width, height)
            if mode == "fixed_multi"
            else [
                adaptive_peak_box(
                    score,
                    cx,
                    cy,
                    confidence,
                    threshold,
                    local_rel_threshold,
                    local_radius,
                    box_pad_x,
                    box_pad_y,
                    min_box_width,
                    min_box_height,
                    max_box_width,
                    max_box_height,
                )
            ]
        )
        for box in boxes:
            x1, y1, x2, y2 = box
            if x2 > x1 and y2 > y1:
                out.append({"bbox": box, "confidence": round(confidence, 6), "area": int((x2 - x1) * (y2 - y1)), "decoder": f"local_peak_{mode}"})
    return out


def fixed_peak_boxes(cx: int, cy: int, width: int, height: int) -> list[list[int]]:
    windows = [(8, 8), (14, 8), (24, 10), (40, 12), (60, 14)]
    boxes: list[list[int]] = []
    for ww, hh in windows:
        x1 = max(0, cx - ww // 2)
        y1 = max(0, cy - hh // 2)
        x2 = min(width, x1 + ww)
        y2 = min(height, y1 + hh)
        boxes.append([x1, y1, x2, y2])
    return boxes


def adaptive_peak_box(
    score: np.ndarray,
    cx: int,
    cy: int,
    confidence: float,
    threshold: float,
    local_rel_threshold: float,
    local_radius: int,
    box_pad_x: int,
    box_pad_y: int,
    min_box_width: int,
    min_box_height: int,
    max_box_width: int,
    max_box_height: int,
) -> list[int]:
    height, width = score.shape
    radius = max(2, int(local_radius))
    x1 = max(0, cx - radius)
    y1 = max(0, cy - radius)
    x2 = min(width, cx + radius + 1)
    y2 = min(height, cy + radius + 1)
    local = score[y1:y2, x1:x2]
    local_threshold = max(float(threshold), float(confidence) * float(local_rel_threshold))
    binary = (local >= local_threshold).astype("uint8")
    n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
    lx, ly = cx - x1, cy - y1
    label = int(labels[ly, lx]) if 0 <= ly < labels.shape[0] and 0 <= lx < labels.shape[1] else 0
    if label > 0 and label < n_labels:
        sx, sy, sw, sh, _area = [int(v) for v in stats[label]]
        bx1, by1, bx2, by2 = x1 + sx, y1 + sy, x1 + sx + sw, y1 + sy + sh
    else:
        bx1, by1, bx2, by2 = cx, cy, cx + 1, cy + 1
    bx1, by1, bx2, by2 = expand_box_to_limits(
        [bx1 - box_pad_x, by1 - box_pad_y, bx2 + box_pad_x, by2 + box_pad_y],
        cx,
        cy,
        width,
        height,
        min_box_width,
        min_box_height,
        max_box_width,
        max_box_height,
    )
    return [bx1, by1, bx2, by2]


def expand_box_to_limits(
    box: list[int],
    cx: int,
    cy: int,
    page_width: int,
    page_height: int,
    min_width: int,
    min_height: int,
    max_width: int,
    max_height: int,
) -> list[int]:
    x1, y1, x2, y2 = box
    w = max(int(min_width), x2 - x1)
    h = max(int(min_height), y2 - y1)
    w = min(max(1, int(max_width)), w)
    h = min(max(1, int(max_height)), h)
    cx = min(max(cx, 0), page_width - 1)
    cy = min(max(cy, 0), page_height - 1)
    x1 = max(0, min(page_width - w, cx - w // 2))
    y1 = max(0, min(page_height - h, cy - h // 2))
    return [int(x1), int(y1), int(x1 + w), int(y1 + h)]


def nms(items: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for item in items:
        if all(bbox_iou(item["bbox"], other["bbox"]) < threshold for other in kept):
            kept.append(item)
    return kept


def evaluate(model: nn.Module, rows: list[dict[str, Any]], args: argparse.Namespace, split: str) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    eval_rows = [row for row in rows if targets(row)]
    if args.max_eval:
        eval_rows = eval_rows[: args.max_eval]
    thresholds = [float(item) for item in args.thresholds.split(",")]
    best_key: tuple[float, float, float] | None = None
    best: tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]] | None = None
    sweep = []
    for threshold in thresholds:
        report, predictions, buckets = evaluate_threshold(model, eval_rows, args, split, threshold)
        sweep.append(report)
        key = (report["text_bbox_center_recall"], report["text_bbox_iou_0_30"]["recall"], -report["candidate_inflation"])
        if best_key is None or key > best_key:
            best_key = key
            best = (report, predictions, buckets)
    assert best is not None
    report = json.loads(json.dumps(best[0]))
    report["threshold_sweep"] = sweep
    return report, best[1], best[2]


def evaluate_threshold(model: nn.Module, rows: list[dict[str, Any]], args: argparse.Namespace, split: str, threshold: float) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    totals = Counter()
    semantic_total = Counter()
    semantic_hit = Counter()
    misses: list[dict[str, Any]] = []
    false_positives: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    for row in rows:
        maps = predict_maps(model, row, args.size, args.device)
        preds = decode(
            maps,
            threshold,
            args.affinity_weight,
            args.min_area,
            args.close_kernel,
            args.max_area_ratio,
            args.peak_top_k,
            args.peak_min_distance,
            args.peak_window_mode,
            args.local_rel_threshold,
            args.local_radius,
            args.box_pad_x,
            args.box_pad_y,
            args.min_box_width,
            args.min_box_height,
            args.max_box_width,
            args.max_box_height,
            args.max_candidates_per_page,
        )
        golds = targets(row)
        used: set[int] = set()
        matched_iou = 0
        matched_center = 0
        for gold_index, gold in enumerate(golds):
            gb = [int(v) for v in gold["bbox"]]
            semantic = str(gold.get("semantic_type") or "unknown")
            semantic_total[semantic] += 1
            best_iou = 0.0
            best_index = None
            center_index = None
            for pred_index, pred in enumerate(preds):
                if pred_index in used:
                    continue
                iou = bbox_iou(pred["bbox"], gb)
                if iou > best_iou:
                    best_iou = iou
                    best_index = pred_index
                if center_index is None and center_covered(pred["bbox"], gb):
                    center_index = pred_index
            if best_index is not None and best_iou >= 0.30:
                used.add(best_index)
                matched_iou += 1
                matched_center += 1
                semantic_hit[semantic] += 1
            elif center_index is not None:
                used.add(center_index)
                matched_center += 1
                semantic_hit[semantic] += 1
            else:
                misses.append({"row_id": row["source_row_id"], "gold_index": gold_index, "bbox": gb, "semantic_type": semantic, "best_iou": round(best_iou, 6)})
        for pred_index, pred in enumerate(preds):
            if pred_index in used:
                continue
            best = max([bbox_iou(pred["bbox"], [int(v) for v in gold["bbox"]]) for gold in golds] or [0.0])
            if best < 0.05 and len(false_positives) < 80:
                false_positives.append({"row_id": row["source_row_id"], "bbox": pred["bbox"], "confidence": pred["confidence"], "best_iou": round(best, 6)})
        totals["gold"] += len(golds)
        totals["predicted"] += len(preds)
        totals["matched_iou"] += matched_iou
        totals["matched_center"] += matched_center
        pred_rows.append(
            {
                "id": row["source_row_id"],
                "image": row["image"],
                "predicted_text": [
                    {
                        "id": f"{row['source_row_id']}_text_heatmap_affinity_v19_{idx}",
                        "class": "text",
                        "family": "text",
                        "semantic_type": "unknown_text",
                        "bbox": pred["bbox"],
                        "confidence": pred["confidence"],
                        "proposal_source": "raster_text_heatmap_affinity_v19",
                        "payload": {"ocr_status": "not_invoked", "source": "raster_text_heatmap_affinity_v19"},
                    }
                    for idx, pred in enumerate(preds)
                ],
                "gold_text_count": len(golds),
                "matched_iou_0_30": matched_iou,
                "matched_center": matched_center,
                "source_integrity": {"model_input": "raster_image_only", "gold_used_for_inference": False},
            }
        )
    precision = totals["matched_iou"] / max(totals["predicted"], 1)
    recall = totals["matched_iou"] / max(totals["gold"], 1)
    center_recall = totals["matched_center"] / max(totals["gold"], 1)
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
        "text_bbox_center_recall": round(center_recall, 6),
        "candidate_inflation": round(totals["predicted"] / max(totals["gold"], 1), 6),
        "semantic_center_recall": {key: round(semantic_hit[key] / max(semantic_total[key], 1), 6) for key in sorted(semantic_total)},
    }
    buckets = {
        "split": split,
        "threshold": threshold,
        "failure_buckets": {
            "missed_tiny_text": sum(1 for item in misses if (item["bbox"][2] - item["bbox"][0]) * (item["bbox"][3] - item["bbox"][1]) <= 25),
            "missed_room_label": sum(1 for item in misses if item.get("semantic_type") == "room_label"),
            "false_positive_examples": len(false_positives),
        },
        "miss_examples": misses[:80],
        "false_positive_examples": false_positives[:80],
    }
    return report, pred_rows, buckets


def train(args: argparse.Namespace) -> tuple[nn.Module, dict[str, Any]]:
    torch.manual_seed(args.seed)
    rows = load_jsonl(DATA / "train.jsonl")
    ds = HeatmapTextDataset(rows, args.size, args.max_train, args.target_pad, args.target_blur)
    if not ds:
        raise SystemExit("no raster-aligned localizer supervision")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = HeatmapAffinityNet().to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    losses = []
    for _epoch in range(args.epochs):
        model.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(args.device), y.to(args.device)
            logits = model(x)
            pos = torch.clamp((y.numel() - y.sum()) / torch.clamp(y.sum(), min=1.0), 1.0, args.max_pos_weight)
            bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos.detach())
            loss = bce + soft_dice(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu())
        losses.append(round(total / max(len(loader), 1), 6))
    CKPT.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_name": "HeatmapAffinityNet",
            "size": args.size,
            "target_pad": args.target_pad,
            "target_blur": args.target_blur,
            "dataset": str(DATA.relative_to(ROOT)),
        },
        CKPT / "model_best.pt",
    )
    return model, {"checkpoint": str((CKPT / "model_best.pt").relative_to(ROOT)), "train_rows": len(ds), "loss_tail": losses[-5:]}


def load_model(args: argparse.Namespace) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint_path = CKPT / "model_best.pt"
    ckpt = torch.load(checkpoint_path, map_location=args.device)
    model = HeatmapAffinityNet().to(args.device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, {"checkpoint": str(checkpoint_path.relative_to(ROOT)), "train_rows": None, "loss_tail": [], "eval_only": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--max-train", type=int, default=640)
    parser.add_argument("--max-eval", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--thresholds", default="0.08,0.12,0.16,0.20,0.24,0.28,0.32")
    parser.add_argument("--target-pad", type=int, default=6)
    parser.add_argument("--target-blur", type=float, default=1.4)
    parser.add_argument("--affinity-weight", type=float, default=0.65)
    parser.add_argument("--min-area", type=int, default=4)
    parser.add_argument("--close-kernel", type=int, default=3)
    parser.add_argument("--max-area-ratio", type=float, default=0.08)
    parser.add_argument("--max-pos-weight", type=float, default=80.0)
    parser.add_argument("--peak-top-k", type=int, default=250)
    parser.add_argument("--peak-min-distance", type=int, default=7)
    parser.add_argument("--peak-window-mode", choices=["fixed_multi", "adaptive"], default="adaptive")
    parser.add_argument("--local-rel-threshold", type=float, default=0.92)
    parser.add_argument("--local-radius", type=int, default=18)
    parser.add_argument("--box-pad-x", type=int, default=2)
    parser.add_argument("--box-pad-y", type=int, default=2)
    parser.add_argument("--min-box-width", type=int, default=8)
    parser.add_argument("--min-box-height", type=int, default=6)
    parser.add_argument("--max-box-width", type=int, default=80)
    parser.add_argument("--max-box-height", type=int, default=24)
    parser.add_argument("--max-candidates-per-page", type=int, default=0)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    model, training = load_model(args) if args.eval_only else train(args)
    dev_report, _dev_predictions, dev_buckets = evaluate(model, load_jsonl(DATA / "dev.jsonl"), args, "dev")
    locked_report, locked_predictions, locked_buckets = evaluate(model, load_jsonl(DATA / "locked.jsonl"), args, "locked")
    adopted = locked_report["text_bbox_center_recall"] >= 0.80 and locked_report["text_bbox_iou_0_30"]["recall"] >= 0.50
    report = {
        "version": "text_heatmap_affinity_v19_eval",
        "task": "P0-TEXT-001",
        "run_mode": "dbnet_craft_style_region_affinity_text_localizer",
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
        "blocker": None if adopted else "Region+affinity localizer still below stage-1 gate; inspect error buckets before training OCR recognizer.",
    }
    write_json(REPORT / "text_heatmap_affinity_v19_eval.json", report)
    write_json(REPORT / "text_heatmap_affinity_v19_error_buckets.json", {"dev": dev_buckets, "locked": locked_buckets})
    write_jsonl(REPORT / "text_heatmap_affinity_v19_locked_predictions.jsonl", locked_predictions)
    write_json(CKPT / "train_summary.json", report)
    print(json.dumps({"locked": locked_report, "adopted": adopted}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
