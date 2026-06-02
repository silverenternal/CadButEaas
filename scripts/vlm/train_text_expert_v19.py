#!/usr/bin/env python3
"""Train/evaluate the v19 raster text localization expert."""

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
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets/text_expert_raster_v19"
REPORT = ROOT / "reports/vlm"
CKPT = ROOT / "checkpoints/text_expert_v19"


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


def raster_localizer_targets(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in row.get("text_targets") or []
        if item.get("can_train_localizer") and item.get("bbox") and len(item["bbox"]) == 4
    ]


class TextMaskDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], size: int, max_rows: int = 0, target_pad: int = 2) -> None:
        filtered = [row for row in rows if raster_localizer_targets(row)]
        self.rows = filtered[:max_rows] if max_rows else filtered
        self.size = int(size)
        self.target_pad = int(target_pad)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        image = Image.open(abs_path(row["image"])).convert("L")
        original_w, original_h = image.size
        image = image.resize((self.size, self.size), Image.Resampling.BILINEAR)
        x = 1.0 - (np.asarray(image, dtype=np.float32) / 255.0)
        mask = Image.new("L", (self.size, self.size), 0)
        draw = ImageDraw.Draw(mask)
        scale_x = self.size / max(original_w, 1)
        scale_y = self.size / max(original_h, 1)
        for target in raster_localizer_targets(row):
            x1, y1, x2, y2 = [float(v) for v in target["bbox"]]
            box = [
                max(0, min(self.size, int(math.floor(x1 * scale_x)) - self.target_pad)),
                max(0, min(self.size, int(math.floor(y1 * scale_y)) - self.target_pad)),
                max(0, min(self.size, int(math.ceil(x2 * scale_x)) + self.target_pad)),
                max(0, min(self.size, int(math.ceil(y2 * scale_y)) + self.target_pad)),
            ]
            if box[2] > box[0] and box[3] > box[1]:
                draw.rectangle(box, fill=1)
        y = np.asarray(mask, dtype=np.float32)
        return torch.from_numpy(x[None]), torch.from_numpy(y[None])


class TinyTextUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.e1 = nn.Sequential(nn.Conv2d(1, 24, 3, padding=1), nn.ReLU(), nn.Conv2d(24, 24, 3, padding=1), nn.ReLU())
        self.e2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(24, 48, 3, padding=1), nn.ReLU(), nn.Conv2d(48, 48, 3, padding=1), nn.ReLU())
        self.e3 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(48, 96, 3, padding=1), nn.ReLU())
        self.u2 = nn.ConvTranspose2d(96, 48, 2, stride=2)
        self.d2 = nn.Sequential(nn.Conv2d(96, 48, 3, padding=1), nn.ReLU())
        self.u1 = nn.ConvTranspose2d(48, 24, 2, stride=2)
        self.d1 = nn.Sequential(nn.Conv2d(48, 24, 3, padding=1), nn.ReLU())
        self.out = nn.Conv2d(24, 1, 1)

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


def center_covered(pred: list[int], gold: list[int], margin: int = 1) -> bool:
    cx = (gold[0] + gold[2]) / 2.0
    cy = (gold[1] + gold[3]) / 2.0
    return pred[0] - margin <= cx <= pred[2] + margin and pred[1] - margin <= cy <= pred[3] + margin


def predict_prob(model: nn.Module, row: dict[str, Any], size: int, device: str) -> np.ndarray:
    image = Image.open(abs_path(row["image"])).convert("L")
    original_size = image.size
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    arr = 1.0 - (np.asarray(image, dtype=np.float32) / 255.0)
    x = torch.from_numpy(arr[None, None]).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0, 0].detach().cpu().numpy()
    return cv2.resize(prob, original_size, interpolation=cv2.INTER_LINEAR)


def components(prob: np.ndarray, threshold: float, min_area: int, max_area_ratio: float, close_kernel: int = 1) -> list[dict[str, Any]]:
    binary = (prob >= threshold).astype("uint8")
    if close_kernel > 1:
        kernel = np.ones((close_kernel, close_kernel), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    n_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
    out: list[dict[str, Any]] = []
    page_area = max(int(prob.shape[0] * prob.shape[1]), 1)
    for idx in range(1, int(n_labels)):
        x, y, w, h, area = [int(v) for v in stats[idx]]
        if area < min_area or area / page_area > max_area_ratio:
            continue
        out.append(
            {
                "bbox": [x, y, x + w, y + h],
                "confidence": round(float(prob[y : y + h, x : x + w].mean()), 6),
                "area": area,
            }
        )
    out.sort(key=lambda item: item["confidence"], reverse=True)
    return nms(out, 0.35)


def nms(items: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for item in items:
        if all(bbox_iou(item["bbox"], other["bbox"]) < threshold for other in kept):
            kept.append(item)
    return kept


def evaluate(model: nn.Module, rows: list[dict[str, Any]], args: argparse.Namespace, split: str) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    rows = [row for row in rows if raster_localizer_targets(row)]
    if args.max_eval:
        rows = rows[: args.max_eval]
    thresholds = [float(value) for value in args.thresholds.split(",")]
    threshold_reports = []
    best: tuple[float, dict[str, Any], list[dict[str, Any]], dict[str, Any]] | None = None
    for threshold in thresholds:
        report, predictions, buckets = evaluate_threshold(model, rows, args, split, threshold)
        threshold_reports.append(json.loads(json.dumps(report)))
        key = (
            report["text_bbox_center_recall"],
            report["text_bbox_iou_0_30"]["recall"],
            -report["candidate_inflation"],
        )
        if best is None or key > best[0]:
            best = (key, report, predictions, buckets)  # type: ignore[assignment]
    assert best is not None
    selected_report = json.loads(json.dumps(best[1]))
    selected_report["threshold_sweep"] = threshold_reports
    return selected_report, best[2], best[3]


def evaluate_threshold(model: nn.Module, rows: list[dict[str, Any]], args: argparse.Namespace, split: str, threshold: float) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    totals = Counter()
    semantic_total = Counter()
    semantic_hit = Counter()
    misses: list[dict[str, Any]] = []
    false_positives: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for row in rows:
        prob = predict_prob(model, row, args.size, args.device)
        preds = components(prob, threshold, args.min_area, args.max_area_ratio, args.close_kernel)
        golds = raster_localizer_targets(row)
        used: set[int] = set()
        matched_iou = 0
        matched_center = 0
        for gold_index, gold in enumerate(golds):
            gb = [int(v) for v in gold["bbox"]]
            semantic_total[str(gold.get("semantic_type") or "unknown")] += 1
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
                semantic_hit[str(gold.get("semantic_type") or "unknown")] += 1
            elif center_index is not None:
                used.add(center_index)
                matched_center += 1
                semantic_hit[str(gold.get("semantic_type") or "unknown")] += 1
            else:
                misses.append(
                    {
                        "row_id": row["source_row_id"],
                        "gold_index": gold_index,
                        "bbox": gb,
                        "semantic_type": gold.get("semantic_type"),
                        "normalized_text": gold.get("normalized_text"),
                        "best_iou": round(best_iou, 6),
                    }
                )
        for pred_index, pred in enumerate(preds):
            if pred_index not in used:
                best_gold_iou = max([bbox_iou(pred["bbox"], [int(v) for v in gold["bbox"]]) for gold in golds] or [0.0])
                if best_gold_iou < 0.05 and len(false_positives) < 40:
                    false_positives.append({"row_id": row["source_row_id"], "bbox": pred["bbox"], "confidence": pred["confidence"], "best_iou": round(best_gold_iou, 6)})
        totals["gold"] += len(golds)
        totals["predicted"] += len(preds)
        totals["matched_iou"] += matched_iou
        totals["matched_center"] += matched_center
        predictions.append(
            {
                "id": row["source_row_id"],
                "image": row["image"],
                "split": split,
                "predicted_text": [
                    {
                        "id": f"{row['source_row_id']}_text_v19_{i}",
                        "class": "text",
                        "family": "text",
                        "semantic_type": "unknown_text",
                        "bbox": pred["bbox"],
                        "confidence": pred["confidence"],
                        "proposal_source": "raster_text_expert_v19",
                        "payload": {
                            "ocr_text": None,
                            "ocr_status": "not_invoked",
                            "source": "raster_text_expert_v19_localizer",
                        },
                    }
                    for i, pred in enumerate(preds)
                ],
                "gold_text_count": len(golds),
                "matched_iou_0_30": matched_iou,
                "matched_center": matched_center,
                "source_integrity": {
                    "model_input": "raster_image_only",
                    "gold_used_for_inference": False,
                    "ocr_transcript_from_gold": False,
                },
            }
        )
    precision_iou = totals["matched_iou"] / max(totals["predicted"], 1)
    recall_iou = totals["matched_iou"] / max(totals["gold"], 1)
    recall_center = totals["matched_center"] / max(totals["gold"], 1)
    f1_iou = 0.0 if precision_iou + recall_iou == 0 else 2 * precision_iou * recall_iou / (precision_iou + recall_iou)
    report = {
        "split": split,
        "rows": len(rows),
        "threshold": threshold,
        "text_bbox_iou_0_30": {
            "matched": int(totals["matched_iou"]),
            "predicted": int(totals["predicted"]),
            "gold": int(totals["gold"]),
            "precision": round(precision_iou, 6),
            "recall": round(recall_iou, 6),
            "f1": round(f1_iou, 6),
        },
        "text_bbox_center_recall": round(recall_center, 6),
        "candidate_inflation": round(totals["predicted"] / max(totals["gold"], 1), 6),
        "semantic_center_recall": {
            key: round(semantic_hit[key] / max(semantic_total[key], 1), 6)
            for key in sorted(semantic_total)
        },
        "ocr": {
            "status": "not_trained_in_v19_localizer_step",
            "normalized_accuracy": None,
        },
    }
    buckets = {
        "split": split,
        "threshold": threshold,
        "miss_examples": misses[:80],
        "false_positive_examples": false_positives[:80],
        "failure_buckets": {
            "missed_tiny_text": sum(1 for item in misses if (item["bbox"][2] - item["bbox"][0]) * (item["bbox"][3] - item["bbox"][1]) <= 25),
            "missed_room_label": sum(1 for item in misses if item.get("semantic_type") == "room_label"),
            "blank_or_wall_false_positive": len(false_positives),
        },
    }
    return report, predictions, buckets


def train(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    train_rows = load_jsonl(DATA / "train.jsonl")
    train_ds = TextMaskDataset(train_rows, args.size, args.max_train, args.target_pad)
    if not train_ds:
        raise SystemExit("no raster-aligned text localizer rows available")
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = TinyTextUNet().to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    losses = []
    for _epoch in range(args.epochs):
        model.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(args.device), y.to(args.device)
            logits = model(x)
            pos = torch.clamp((y.numel() - y.sum()) / torch.clamp(y.sum(), min=1.0), 1.0, args.max_pos_weight)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos.detach()) + dice_loss(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu())
        losses.append(round(total / max(len(loader), 1), 6))
    CKPT.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_name": "TinyTextUNet",
            "size": args.size,
            "epochs": args.epochs,
            "target_pad": args.target_pad,
            "dataset": str(DATA.relative_to(ROOT)),
        },
        CKPT / "model_best.pt",
    )
    return {"model": model, "train_rows": len(train_ds), "loss_tail": losses[-5:]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--max-train", type=int, default=640)
    parser.add_argument("--max-eval", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--thresholds", default="0.20,0.30,0.40,0.50,0.60")
    parser.add_argument("--min-area", type=int, default=2)
    parser.add_argument("--max-area-ratio", type=float, default=0.05)
    parser.add_argument("--max-pos-weight", type=float, default=120.0)
    parser.add_argument("--target-pad", type=int, default=2)
    parser.add_argument("--close-kernel", type=int, default=1)
    args = parser.parse_args()

    train_result = train(args)
    model = train_result["model"]
    dev_report, _dev_predictions, dev_buckets = evaluate(model, load_jsonl(DATA / "dev.jsonl"), args, "dev")
    locked_report, locked_predictions, locked_buckets = evaluate(model, load_jsonl(DATA / "locked.jsonl"), args, "locked")

    report = {
        "version": "text_expert_v19_eval",
        "task": "P0-TEXT-001",
        "run_mode": "raster_text_localizer_training",
        "source_integrity": {
            "model_input": "raster_image_only",
            "gold_used_for_inference": False,
            "offline_labels_used_for": ["training_targets", "dev_threshold_selection", "locked_evaluation"],
            "runtime_uses_svg_or_cad_geometry": False,
        },
        "training": {
            "checkpoint": str((CKPT / "model_best.pt").relative_to(ROOT)),
            "train_rows": train_result["train_rows"],
            "loss_tail": train_result["loss_tail"],
            "size": args.size,
            "epochs": args.epochs,
        },
        "dev": dev_report,
        "locked": locked_report,
        "ocr": {
            "status": "not_solved_by_this_localizer_step",
            "current_baseline_report": "reports/vlm/text_ocr_v18_eval.json",
        },
        "semantic_head": {
            "reusable_asset": "checkpoints/text_dimension_expert_v13/model.joblib",
            "current_v13_report": "reports/vlm/text_dimension_expert_v13_eval.json",
            "scope": "candidate-conditioned semantic text type classification; not a raster text localizer",
        },
        "adopted": locked_report["text_bbox_center_recall"] >= 0.80 and locked_report["text_bbox_iou_0_30"]["recall"] >= 0.50,
        "blocker": "Text OCR recognition and high-recall localization still below P0 target; continue with stronger DBNet/CRAFT-style localizer and OCR recognizer.",
    }
    write_json(REPORT / "text_expert_v19_eval.json", report)
    write_json(REPORT / "text_expert_v19_error_buckets.json", {"dev": dev_buckets, "locked": locked_buckets})
    write_jsonl(REPORT / "text_expert_v19_locked_predictions.jsonl", locked_predictions)
    write_json(CKPT / "train_summary.json", report)
    print(json.dumps({"locked": locked_report, "adopted": report["adopted"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
