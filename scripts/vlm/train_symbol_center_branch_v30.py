#!/usr/bin/env python3
"""Train the v30 raster-only symbol center branch and smoke-evaluate it."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

from train_symbol_center_size_proposal_v25 import CenterSizeNet, center_loss, decode
from train_symbol_center_heatmap_probe_v24 import gaussian_2d, score_predictions
from train_symbol_tile_detector_v20 import FORBIDDEN_RUNTIME_FIELDS, load_jsonl, rel, source_path, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "datasets/symbol_center_branch_v30/manifest.json"
OUTPUT = ROOT / "checkpoints/symbol_center_branch_v30/model.pt"
SMOKE_REPORT = ROOT / "reports/vlm/symbol_center_branch_v30_smoke_eval.json"
SMOKE_PREDICTIONS = ROOT / "reports/vlm/symbol_center_branch_v30_smoke_predictions.jsonl"


def group_targets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for target in rows:
        tile_id = str(target.get("tile_id"))
        if tile_id not in grouped:
            grouped[tile_id] = {
                "id": tile_id,
                "row_id": str(target.get("row_id")),
                "image": target.get("image"),
                "image_size": target.get("image_size"),
                "tile_bbox": target.get("tile_bbox"),
                "targets": [],
            }
        grouped[tile_id]["targets"].append(target)
    return list(grouped.values())


class CenterBranchDataset(Dataset):
    def __init__(self, tiles: list[dict[str, Any]], input_size: int, stride: int, augment: bool, seed: int) -> None:
        self.tiles = tiles
        self.input_size = input_size
        self.stride = stride
        self.grid_size = input_size // stride
        self.augment = augment
        self.seed = seed
        self.kernels = {1: gaussian_2d(1), 2: gaussian_2d(2), 3: gaussian_2d(3)}

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        tile = self.tiles[index]
        left, top, right, bottom = [float(v) for v in tile.get("tile_bbox") or [0, 0, 1, 1]]
        tile_w = max(1.0, right - left)
        tile_h = max(1.0, bottom - top)
        with Image.open(source_path(str(tile.get("image") or ""))) as opened:
            crop = opened.convert("RGB").crop((int(left), int(top), int(right), int(bottom)))
        crop = ImageOps.autocontrast(crop).resize((self.input_size, self.input_size), Image.Resampling.BILINEAR)
        rng = random.Random(self.seed + index)
        flip = self.augment and rng.random() < 0.5
        if flip:
            crop = ImageOps.mirror(crop)
        image = torch.from_numpy((np.asarray(crop, dtype=np.float32) / 255.0).transpose(2, 0, 1)).float()

        heatmap = np.zeros((1, self.grid_size, self.grid_size), dtype=np.float32)
        size = np.zeros((2, self.grid_size, self.grid_size), dtype=np.float32)
        size_mask = np.zeros((1, self.grid_size, self.grid_size), dtype=np.float32)
        for target in tile.get("targets") or []:
            tx, ty = [float(v) for v in target.get("tile_center") or [0, 0]]
            tw, th = [float(v) for v in target.get("page_size") or [1, 1]]
            if flip:
                tx = tile_w - tx
            sx = self.input_size / tile_w
            sy = self.input_size / tile_h
            gx = min(self.grid_size - 1, max(0, int(round(tx * sx / self.stride))))
            gy = min(self.grid_size - 1, max(0, int(round(ty * sy / self.stride))))
            max_side = max(tw * sx, th * sy)
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
            size[:, gy, gx] = [np.log1p(max(1.0, tw)), np.log1p(max(1.0, th))]
            size_mask[0, gy, gx] = 1.0

        meta = {
            "id": tile.get("id"),
            "row_id": tile.get("row_id"),
            "tile_bbox": [left, top, right, bottom],
            "tile_size": [tile_w, tile_h],
            "gold": [
                {
                    "target_id": target.get("target_id"),
                    "bbox": target.get("page_bbox"),
                    "page_bbox": target.get("page_bbox"),
                    "label": target.get("label"),
                }
                for target in tile.get("targets") or []
            ],
        }
        return image, torch.from_numpy(heatmap), torch.from_numpy(size), torch.from_numpy(size_mask), meta


def collate(batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    images, heatmaps, sizes, size_masks, metas = zip(*batch, strict=True)
    return torch.stack(list(images)), torch.stack(list(heatmaps)), torch.stack(list(sizes)), torch.stack(list(size_masks)), list(metas)


def sample_tiles(tiles: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if not limit or len(tiles) <= limit:
        return list(tiles)
    rng = random.Random(seed)
    out = list(tiles)
    rng.shuffle(out)
    return out[:limit]


def train_epoch(model: torch.nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> dict[str, float]:
    import torch.nn.functional as F

    model.train()
    totals = Counter()
    seen = 0
    for images, heatmaps, sizes, size_masks, _metas in loader:
        images = images.to(device, non_blocking=True)
        heatmaps = heatmaps.to(device, non_blocking=True)
        sizes = sizes.to(device, non_blocking=True)
        size_masks = size_masks.to(device, non_blocking=True)
        obj_logits, pred_sizes = model(images)
        obj_loss = center_loss(obj_logits, heatmaps)
        size_loss = (F.smooth_l1_loss(pred_sizes, sizes, reduction="none") * size_masks).sum() / size_masks.sum().clamp_min(1.0)
        loss = obj_loss + 0.15 * size_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        batch = int(images.shape[0])
        seen += batch
        totals["loss"] += float(loss.detach().cpu()) * batch
        totals["obj_loss"] += float(obj_loss.detach().cpu()) * batch
        totals["size_loss"] += float(size_loss.detach().cpu()) * batch
    return {key: round(value / max(seen, 1), 6) for key, value in totals.items()}


def collect(model: torch.nn.Module, tiles: list[dict[str, Any]], args: argparse.Namespace, device: torch.device) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, dict[str, Any]]]]:
    dataset = CenterBranchDataset(tiles, args.input_size, args.stride, augment=False, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda", collate_fn=collate)
    page_preds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    page_golds: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    model.eval()
    with torch.no_grad():
        for images, _heatmaps, _sizes, _size_masks, metas in loader:
            obj_logits, size_logits = model(images.to(device, non_blocking=True))
            decoded = decode(obj_logits, size_logits, metas, args.input_size, args.stride, args.decode_score_threshold, args.topk_per_tile, args.min_box_size, args.max_box_size)
            for row_id, preds in decoded.items():
                for pred in preds:
                    pred["proposal_source"] = "center_branch_v30"
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--smoke-eval-output", default=str(SMOKE_REPORT))
    parser.add_argument("--smoke-predictions-output", default=str(SMOKE_PREDICTIONS))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--limit-train-tiles", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--input-size", type=int, default=384)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--decode-score-threshold", type=float, default=0.001)
    parser.add_argument("--score-threshold-grid", default="0.001,0.003,0.005,0.01")
    parser.add_argument("--nms-threshold-grid", default="0.25,0.35,0.45,0.55")
    parser.add_argument("--topk-per-tile", type=int, default=120)
    parser.add_argument("--max-per-page", type=int, default=1200)
    parser.add_argument("--min-box-size", type=float, default=2.0)
    parser.add_argument("--max-box-size", type=float, default=160.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260511)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))

    manifest_path = Path(args.dataset)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = manifest_path.parent
    train_targets = load_jsonl(ROOT / manifest["outputs"]["train_center_targets"])
    smoke_targets = load_jsonl(ROOT / manifest["outputs"]["smoke_center_targets"])
    train_tiles = sample_tiles(group_targets(train_targets), args.limit_train_tiles, args.seed)
    smoke_tiles = group_targets(smoke_targets)

    model = CenterSizeNet().to(device)
    loader = DataLoader(
        CenterBranchDataset(train_tiles, args.input_size, args.stride, augment=True, seed=args.seed),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    epoch_log = []
    for epoch in range(1, args.epochs + 1):
        row = train_epoch(model, loader, optimizer, device)
        row["epoch"] = epoch
        epoch_log.append(row)

    smoke_preds, smoke_golds = collect(model, smoke_tiles, args, device)
    grid = []
    for score_threshold in [float(item) for item in args.score_threshold_grid.split(",") if item.strip()]:
        for nms_threshold in [float(item) for item in args.nms_threshold_grid.split(",") if item.strip()]:
            metrics, predictions, errors = score_predictions(smoke_preds, smoke_golds, score_threshold, nms_threshold, args.max_per_page, len(smoke_tiles))
            grid.append({"score_threshold": score_threshold, "nms_threshold": nms_threshold, "metrics": metrics, "error_count": len(errors), "predictions": predictions})
    grid.sort(key=lambda row: (row["metrics"]["symbol_bbox_center_recall"], row["metrics"]["symbol_bbox_iou_0_30"]["recall"], -row["metrics"]["candidate_inflation"]), reverse=True)
    selected = grid[0]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)
    write_json(
        output.with_name("model_metadata.json"),
        {
            "model_type": "symbol_center_branch_v30",
            "dataset": rel(manifest_path),
            "input_size": args.input_size,
            "stride": args.stride,
            "runtime_contract": {"model_input_features": ["image_tile_pixels", "tile.bbox"], "forbidden_runtime_features": FORBIDDEN_RUNTIME_FIELDS},
        },
    )
    report = {
        "version": "symbol_center_branch_v30_smoke_eval",
        "metric_mode": "smoke",
        "claim_boundary": "Raster-only center discovery branch; tight boxes/masks and final duplicate compression are downstream.",
        "dataset": rel(manifest_path),
        "checkpoint": rel(output),
        "config": vars(args) | {"device": str(device)},
        "counts": {"train_tiles": len(train_tiles), "smoke_tiles": len(smoke_tiles), "smoke_targets": len(smoke_targets)},
        "epoch_log": epoch_log,
        "threshold_grid": [{k: v for k, v in row.items() if k != "predictions"} for row in grid],
        "selected_thresholds": {"score_threshold": selected["score_threshold"], "nms_threshold": selected["nms_threshold"]},
        "smoke_center_branch": selected["metrics"],
        "stage_gate": {
            "center_recall_min_0_94": selected["metrics"]["symbol_bbox_center_recall"] >= 0.94,
            "candidate_inflation_max_7": selected["metrics"]["candidate_inflation"] <= 7.0,
        },
    }
    report["stage_gate"]["passed"] = all(report["stage_gate"].values())
    write_json(Path(args.smoke_eval_output), report)
    write_jsonl(Path(args.smoke_predictions_output), selected["predictions"])
    print(json.dumps({"smoke_center_branch": selected["metrics"], "stage_gate": report["stage_gate"], "checkpoint": rel(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
