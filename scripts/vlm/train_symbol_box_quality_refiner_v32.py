#!/usr/bin/env python3
"""Train/evaluate a lightweight v32 symbol box-quality refiner."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_symbol_tile_detector_v20 import bbox_iou, rel, write_json


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "datasets/symbol_box_quality_refiner_v32/manifest.json"
DEFAULT_OUTPUT_DIR = ROOT / "checkpoints/symbol_box_quality_refiner_v32"
DEFAULT_EVAL_OUTPUT = ROOT / "reports/vlm/symbol_box_quality_refiner_v32_smoke_eval.json"


def source_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def apply_delta(box: list[float], delta: np.ndarray, max_delta: float) -> list[float]:
    dx, dy, dw, dh = [float(np.clip(v, -max_delta, max_delta)) for v in delta.tolist()]
    x1, y1, x2, y2 = [float(v) for v in box]
    width = max(x2 - x1, 1e-6)
    height = max(y2 - y1, 1e-6)
    cx = (x1 + x2) / 2.0 + dx * width
    cy = (y1 + y2) / 2.0 + dy * height
    new_w = max(1.0, width * (1.0 + dw))
    new_h = max(1.0, height * (1.0 + dh))
    return [cx - new_w / 2.0, cy - new_h / 2.0, cx + new_w / 2.0, cy + new_h / 2.0]


class BoxRefiner(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def rows_to_arrays(rows: list[dict[str, Any]], feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([[float((row.get("features") or {}).get(name, 0.0) or 0.0) for name in feature_names] for row in rows], dtype=np.float32)
    y = np.asarray([row.get("delta_target") or [0.0, 0.0, 0.0, 0.0] for row in rows], dtype=np.float32)
    return x, y


def normalize(train_x: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-6] = 1.0
    return (train_x - mean) / std, [(x - mean) / std for x in others], mean, std


def evaluate_rows(
    model: BoxRefiner,
    rows: list[dict[str, Any]],
    x: np.ndarray,
    max_delta: float,
    protect_area_buckets: set[str] | None = None,
) -> dict[str, Any]:
    if not rows:
        return {}
    model.eval()
    with torch.no_grad():
        pred_delta = model(torch.from_numpy(x).float()).cpu().numpy()
    counts = Counter()
    by_area = defaultdict(Counter)
    by_label = defaultdict(Counter)
    examples = []
    for row, delta in zip(rows, pred_delta, strict=True):
        candidate_box = [float(v) for v in row["candidate_bbox"]]
        target_box = [float(v) for v in row["target_bbox"]]
        bucket = str(row.get("area_bucket") or "unknown")
        label = str(row.get("target_label") or "unknown")
        input_iou = bbox_iou(candidate_box, target_box)
        if protect_area_buckets and bucket in protect_area_buckets:
            refined_box = candidate_box
        else:
            refined_box = apply_delta(candidate_box, delta, max_delta)
        refined_iou = bbox_iou(refined_box, target_box)
        counts["rows"] += 1
        counts["input_iou_ge_0_30"] += int(input_iou >= 0.30)
        counts["refined_iou_ge_0_30"] += int(refined_iou >= 0.30)
        counts["input_iou_ge_0_50"] += int(input_iou >= 0.50)
        counts["refined_iou_ge_0_50"] += int(refined_iou >= 0.50)
        counts["refined_iou_better"] += int(refined_iou > input_iou + 1e-6)
        counts["refined_iou_worse"] += int(refined_iou + 1e-6 < input_iou)
        for key, value in {
            "input_iou_sum": input_iou,
            "refined_iou_sum": refined_iou,
            "iou_delta_sum": refined_iou - input_iou,
        }.items():
            counts[key] += value
            by_area[bucket][key] += value
            by_label[label][key] += value
        by_area[bucket]["rows"] += 1
        by_area[bucket]["input_iou_ge_0_30"] += int(input_iou >= 0.30)
        by_area[bucket]["refined_iou_ge_0_30"] += int(refined_iou >= 0.30)
        by_label[label]["rows"] += 1
        by_label[label]["input_iou_ge_0_30"] += int(input_iou >= 0.30)
        by_label[label]["refined_iou_ge_0_30"] += int(refined_iou >= 0.30)
        if len(examples) < 200 and (input_iou < 0.30 <= refined_iou or refined_iou + 0.10 < input_iou):
            examples.append(
                {
                    "row_id": row["row_id"],
                    "candidate_index": row["candidate_index"],
                    "label": label,
                    "area_bucket": bucket,
                    "input_iou": round(input_iou, 6),
                    "refined_iou": round(refined_iou, 6),
                    "candidate_bbox": candidate_box,
                    "refined_bbox": [round(v, 3) for v in refined_box],
                    "target_bbox": target_box,
                }
            )
    total = max(int(counts["rows"]), 1)

    def summarize(counter: Counter) -> dict[str, float]:
        n = max(int(counter["rows"]), 1)
        return {
            "rows": int(counter["rows"]),
            "input_iou_mean": round(float(counter["input_iou_sum"]) / n, 6),
            "refined_iou_mean": round(float(counter["refined_iou_sum"]) / n, 6),
            "mean_iou_delta": round(float(counter["iou_delta_sum"]) / n, 6),
            "input_iou_0_30_recall": round(float(counter["input_iou_ge_0_30"]) / n, 6),
            "refined_iou_0_30_recall": round(float(counter["refined_iou_ge_0_30"]) / n, 6),
        }

    return {
        "rows": total,
        "input_iou_mean": round(float(counts["input_iou_sum"]) / total, 6),
        "refined_iou_mean": round(float(counts["refined_iou_sum"]) / total, 6),
        "mean_iou_delta": round(float(counts["iou_delta_sum"]) / total, 6),
        "input_iou_0_30_recall": round(float(counts["input_iou_ge_0_30"]) / total, 6),
        "refined_iou_0_30_recall": round(float(counts["refined_iou_ge_0_30"]) / total, 6),
        "input_iou_0_50_recall": round(float(counts["input_iou_ge_0_50"]) / total, 6),
        "refined_iou_0_50_recall": round(float(counts["refined_iou_ge_0_50"]) / total, 6),
        "refined_iou_better_rate": round(float(counts["refined_iou_better"]) / total, 6),
        "refined_iou_worse_rate": round(float(counts["refined_iou_worse"]) / total, 6),
        "by_area": {key: summarize(value) for key, value in sorted(by_area.items())},
        "by_label": {key: summarize(value) for key, value in sorted(by_label.items())},
        "examples": examples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--eval-output", type=Path, default=DEFAULT_EVAL_OUTPUT)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-delta", type=float, default=2.0)
    parser.add_argument("--protect-area-buckets", default="tiny_le_64")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    manifest_path = source_path(args.data)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    outputs = manifest.get("outputs") or {}
    train_rows = load_jsonl(source_path(outputs["train"]))
    dev_rows = load_jsonl(source_path(outputs.get("dev") or outputs["train"]))
    eval_rows = load_jsonl(source_path(outputs.get("smoke_eval") or outputs.get("dev") or outputs["train"]))
    feature_names = sorted((train_rows[0].get("features") or {}).keys())
    train_x_raw, train_y = rows_to_arrays(train_rows, feature_names)
    dev_x_raw, _ = rows_to_arrays(dev_rows, feature_names)
    eval_x_raw, _ = rows_to_arrays(eval_rows, feature_names)
    train_x, normalized, mean, std = normalize(train_x_raw, dev_x_raw, eval_x_raw)
    dev_x, eval_x = normalized

    model = BoxRefiner(train_x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(np.clip(train_y, -args.max_delta, args.max_delta)).float()),
        batch_size=args.batch_size,
        shuffle=True,
    )
    history = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * batch_x.shape[0]
            total_rows += batch_x.shape[0]
        if epoch in {0, args.epochs - 1} or (epoch + 1) % 5 == 0:
            history.append({"epoch": epoch + 1, "train_loss": round(total_loss / max(total_rows, 1), 6)})

    protect_area_buckets = {value.strip() for value in args.protect_area_buckets.split(",") if value.strip()}
    dev_eval = evaluate_rows(model, dev_rows, dev_x, args.max_delta, protect_area_buckets)
    smoke_eval = evaluate_rows(model, eval_rows, eval_x, args.max_delta, protect_area_buckets)
    output_dir = source_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "feature_names": feature_names,
            "feature_mean": mean.tolist(),
            "feature_std": std.tolist(),
            "max_delta": args.max_delta,
            "protect_area_buckets": sorted(protect_area_buckets),
            "model_type": "mlp_box_delta_refiner_v32",
        },
        model_path,
    )
    metadata_path = output_dir / "model_metadata.json"
    write_json(
        metadata_path,
        {
            "version": "symbol_box_quality_refiner_v32",
            "model": rel(model_path),
            "feature_names": feature_names,
            "data": rel(manifest_path),
            "epochs": args.epochs,
            "max_delta": args.max_delta,
            "protect_area_buckets": sorted(protect_area_buckets),
            "history": history,
        },
    )
    report = {
        "version": "symbol_box_quality_refiner_v32_smoke_eval",
        "metric_mode": "smoke_probe",
        "claim_boundary": "Candidate-level box-quality/refiner smoke probe. Offline gold boxes train/evaluate deltas; runtime model inputs are raster-derived candidate fields.",
        "inputs": {
            "data": rel(manifest_path),
            "train_rows": len(train_rows),
            "dev_rows": len(dev_rows),
            "smoke_eval_rows": len(eval_rows),
        },
        "artifacts": {
            "model": rel(model_path),
            "metadata": rel(metadata_path),
        },
        "application_policy": {
            "protect_area_buckets": sorted(protect_area_buckets),
            "reason": "Tiny boxes are not yet reliably improved by the learned delta refiner; leave them unchanged until a tiny-specialized refiner is trained.",
        },
        "feature_names": feature_names,
        "history": history,
        "dev_eval": dev_eval,
        "smoke_eval": smoke_eval,
        "stage_gate": {
            "tiny_iou_recall_improves": smoke_eval.get("by_area", {}).get("tiny_le_64", {}).get("refined_iou_0_30_recall", 0.0)
            > smoke_eval.get("by_area", {}).get("tiny_le_64", {}).get("input_iou_0_30_recall", 0.0),
            "overall_iou_0_30_recall_not_drop": smoke_eval.get("refined_iou_0_30_recall", 0.0) >= smoke_eval.get("input_iou_0_30_recall", 0.0),
            "mean_iou_improves": smoke_eval.get("refined_iou_mean", 0.0) > smoke_eval.get("input_iou_mean", 0.0),
            "runtime_fields_raster_only": True,
        },
        "source_integrity": {
            "model_input": "raster-derived candidate bbox/score/source/type fields",
            "offline_gold_used_as_training_label": True,
            "svg_or_parser_geometry_used_at_runtime": False,
            "expected_json_used_at_runtime": False,
        },
    }
    report["stage_gate"]["passed"] = all(report["stage_gate"].values())
    eval_output = source_path(args.eval_output)
    write_json(eval_output, report)
    print(json.dumps({"model": rel(model_path), "smoke_eval": smoke_eval, "stage_gate": report["stage_gate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
