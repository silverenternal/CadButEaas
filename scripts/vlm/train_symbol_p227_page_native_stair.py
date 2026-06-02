#!/usr/bin/env python3
"""Train/evaluate P227 page-native one-class stair specialist YOLO detector."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets/symbol_p227_page_native_stair_yolo/data.yaml"
OUT = ROOT / "checkpoints/symbol_p227_page_native_stair_yolo"
REPORT = ROOT / "reports/vlm/symbol_p227_page_native_stair_train_eval.json"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def yolo_metrics(metrics: Any) -> dict[str, Any]:
    box = getattr(metrics, "box", None)
    names = getattr(metrics, "names", {}) or {}
    maps = getattr(box, "maps", []) if box is not None else []
    per_class = {str(names.get(index, index)): round(float(value), 6) for index, value in enumerate(maps)}
    return {
        "map50_95": round(float(getattr(box, "map", 0.0) if box is not None else 0.0), 6),
        "map50": round(float(getattr(box, "map50", 0.0) if box is not None else 0.0), 6),
        "map75": round(float(getattr(box, "map75", 0.0) if box is not None else 0.0), 6),
        "per_class_map50_95": per_class,
    }


def train(args: argparse.Namespace) -> Path:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    results = model.train(
        data=str(Path(args.data).resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str((out / "ultralytics_runs").resolve()),
        name=args.run_name,
        exist_ok=True,
        pretrained=True,
        seed=args.seed,
        patience=args.patience,
        val=not args.no_train_val,
        cos_lr=True,
        close_mosaic=args.close_mosaic,
        degrees=2.0,
        translate=0.08,
        scale=0.70,
        shear=0.5,
        mosaic=1.0,
        mixup=0.08,
        copy_paste=0.0,
        hsv_h=0.01,
        hsv_s=0.20,
        hsv_v=0.20,
        verbose=False,
    )
    run_dir = Path(getattr(results, "save_dir", out / "ultralytics_runs" / args.run_name))
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    source = best if best.exists() else last
    target = out / "model.pt"
    if source.exists():
        shutil.copy2(source, target)
    return target if target.exists() else source


def evaluate(weights: Path, args: argparse.Namespace) -> dict[str, Any]:
    model = YOLO(str(weights))
    out = {}
    for split in args.eval_splits.split(","):
        split = split.strip()
        if not split:
            continue
        metrics = model.val(
            data=str(Path(args.data).resolve()),
            split=split,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
            plots=False,
            verbose=False,
        )
        out[split] = yolo_metrics(metrics)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA))
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--report", default=str(REPORT))
    parser.add_argument("--model", default="yolov8m.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=227)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--eval-splits", default="val,test")
    parser.add_argument("--run-name", default="train")
    parser.add_argument("--no-train-val", action="store_true")
    parser.add_argument("--eval-only-weights", default="")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    weights = Path(args.eval_only_weights) if args.eval_only_weights else train(args)
    eval_report = evaluate(weights, args)
    report = {
        "id": "P227_page_native_stair_train_eval",
        "claim_boundary": "Page-native one-class stair YOLO internal-probe metric only; page-level sliced inference and fusion required before promotion.",
        "data": rel(data_path),
        "weights": rel(weights),
        "config": vars(args),
        "tile_eval": eval_report,
        "next_required": "Run P227 page inference on locked P101 rows, then compare/fuse with P224a and P226.",
    }
    write_json(Path(args.report), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
