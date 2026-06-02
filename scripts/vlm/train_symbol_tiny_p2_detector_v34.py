#!/usr/bin/env python3
"""Smoke-first pixel-level tiny symbol detector for v34 proposal source."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_symbol_p2_heatmap_detector_v22 import (
    LABELS,
    P2HeatmapDetector,
    ROOT,
    SymbolP2Dataset,
    collate,
    collect_predictions,
    memory_audit,
    sample_tiles_area_aware,
    score_predictions,
    target_area_buckets,
    train_epoch,
)
from train_symbol_tile_detector_v20 import FORBIDDEN_RUNTIME_FIELDS, load_jsonl, rel, write_json, write_jsonl


def resolve_data_dir(data_arg: str) -> Path:
    path = Path(data_arg)
    if path.suffix == ".json":
        manifest = json.loads((path if path.is_absolute() else ROOT / path).read_text(encoding="utf-8"))
        split_path = Path((manifest.get("splits") or {}).get("train") or "")
        return (ROOT / split_path).parent if not split_path.is_absolute() else split_path.parent
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/symbol_tile_detector_tiny_sahi_v21/manifest.json")
    parser.add_argument("--output-dir", default="checkpoints/symbol_tiny_p2_detector_v34")
    parser.add_argument("--eval-output", default="reports/vlm/symbol_tiny_p2_detector_v34_smoke_eval.json")
    parser.add_argument("--predictions-output", default="reports/vlm/symbol_tiny_p2_detector_v34_smoke_predictions.jsonl")
    parser.add_argument("--limit-train-tiles", type=int, default=3000)
    parser.add_argument("--limit-dev-tiles", type=int, default=400)
    parser.add_argument("--limit-smoke-tiles", type=int, default=200)
    parser.add_argument("--train-positive-ratio", type=float, default=0.95)
    parser.add_argument("--train-small-positive-ratio", type=float, default=0.85)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--input-size", type=int, default=384)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--decode-score-threshold", type=float, default=0.0)
    parser.add_argument("--score-threshold-grid", default="0.0,0.002,0.005,0.01,0.02,0.05")
    parser.add_argument("--nms-threshold-grid", default="0.45,0.55,0.65,0.75")
    parser.add_argument("--max-per-page", type=int, default=900)
    parser.add_argument("--topk-per-tile", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260512)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    data_dir = resolve_data_dir(args.data)
    train_rows = sample_tiles_area_aware(
        load_jsonl(data_dir / "train.jsonl"),
        args.limit_train_tiles,
        args.seed,
        args.train_positive_ratio,
        args.train_small_positive_ratio,
    )
    dev_rows = sample_tiles_area_aware(load_jsonl(data_dir / "dev.jsonl"), args.limit_dev_tiles, args.seed + 1, 0.9, 0.8)
    smoke_name = "smoke_v30.jsonl" if (data_dir / "smoke_v30.jsonl").exists() else "smoke.jsonl"
    smoke_rows = sample_tiles_area_aware(load_jsonl(data_dir / smoke_name), args.limit_smoke_tiles, args.seed + 2, 0.9, 0.8)
    if not train_rows or not smoke_rows:
        raise SystemExit("missing train or smoke rows for v34 detector")

    model = P2HeatmapDetector(num_classes=len(LABELS)).to(device)
    epoch_log: list[dict[str, Any]] = []
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
    smoke_preds, smoke_golds = collect_predictions(model, smoke_rows, device, args)

    grid_reports: list[dict[str, Any]] = []
    for score_threshold in [float(item) for item in args.score_threshold_grid.split(",") if item.strip()]:
        for nms_threshold in [float(item) for item in args.nms_threshold_grid.split(",") if item.strip()]:
            dev_eval, _ = score_predictions(dev_preds, dev_golds, score_threshold, nms_threshold, args.max_per_page, len(dev_rows))
            grid_reports.append({"score_threshold": score_threshold, "nms_threshold": nms_threshold, "dev": dev_eval})
    grid_reports.sort(
        key=lambda row: (
            row["dev"]["symbol_bbox_center_recall"],
            row["dev"]["symbol_bbox_iou_0_30"]["recall"],
            row["dev"]["area_iou_recall"].get("tiny_le_64", 0.0),
            -row["dev"]["candidate_inflation"],
        ),
        reverse=True,
    )
    selected = grid_reports[0]
    smoke_eval, smoke_predictions = score_predictions(
        smoke_preds,
        smoke_golds,
        float(selected["score_threshold"]),
        float(selected["nms_threshold"]),
        args.max_per_page,
        len(smoke_rows),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")
    write_json(
        output_dir / "model_metadata.json",
        {
            "model_type": "symbol_tiny_p2_detector_v34",
            "labels": LABELS,
            "input_size": args.input_size,
            "stride": args.stride,
            "runtime_contract": {
                "model_input_features": ["image_tile_pixels", "tile.bbox"],
                "forbidden_runtime_features": FORBIDDEN_RUNTIME_FIELDS,
            },
        },
    )
    write_jsonl(Path(args.predictions_output), smoke_predictions)
    report = {
        "version": "symbol_tiny_p2_detector_v34_smoke_eval",
        "task": "P1-02-real-pixel-level-tiny-detector-source-v34",
        "claim_boundary": "Smoke-first pixel-level P2 heatmap detector trained from raster tile pixels; no SVG/parser geometry at runtime.",
        "dataset": rel(data_dir),
        "smoke_split": smoke_name,
        "checkpoint": rel(output_dir / "model.pt"),
        "predictions": rel(Path(args.predictions_output)),
        "config": vars(args) | {"device": str(device)},
        "counts": {
            "train_tiles": len(train_rows),
            "dev_tiles": len(dev_rows),
            "smoke_tiles": len(smoke_rows),
            "train_positive_tiles": sum(1 for row in train_rows if int((row.get("target_counts") or {}).get("symbols") or 0) > 0),
            "train_tiny_or_small_positive_tiles": sum(1 for row in train_rows if target_area_buckets(row) & {"tiny_le_64", "small_le_256"}),
        },
        "epoch_log": epoch_log,
        "threshold_grid": grid_reports,
        "selected_thresholds": {"score_threshold": float(selected["score_threshold"]), "nms_threshold": float(selected["nms_threshold"])},
        "dev": selected["dev"],
        "smoke": smoke_eval,
        "gate": {
            "smoke_center_recall_min_0_60": smoke_eval["symbol_bbox_center_recall"] >= 0.60,
            "smoke_iou_0_30_recall_min_0_35": smoke_eval["symbol_bbox_iou_0_30"]["recall"] >= 0.35,
            "smoke_tiny_iou_recall_min_0_30": smoke_eval["area_iou_recall"].get("tiny_le_64", 0.0) >= 0.30,
        },
        "memory_audit": memory_audit(device),
    }
    report["gate"]["passed"] = all(bool(value) for value in report["gate"].values())
    write_json(Path(args.eval_output), report)
    print(
        json.dumps(
            {
                "smoke": smoke_eval,
                "gate": report["gate"],
                "checkpoint": rel(output_dir / "model.pt"),
                "predictions": rel(Path(args.predictions_output)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
