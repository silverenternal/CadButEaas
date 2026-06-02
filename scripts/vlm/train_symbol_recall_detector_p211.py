#!/usr/bin/env python3
"""Train/evaluate P211 high-recall YOLO detector smoke."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def train(args) -> Path:
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    results = model.train(
        data=str((ROOT / args.data).resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str((out / "runs").resolve()),
        name="train",
        exist_ok=True,
        pretrained=True,
        val=not args.no_train_val,
        seed=args.seed,
        patience=args.patience,
    )
    save_dir = Path(getattr(results, "save_dir", out / "runs" / "train"))
    best = save_dir / "weights" / "best.pt"
    last = save_dir / "weights" / "last.pt"
    weights = best if best.exists() else last
    final = out / "model.pt"
    shutil.copy2(weights, final)
    return final


def eval_model(weights: Path, args) -> dict:
    model = YOLO(str(weights))
    metrics = model.val(
        data=str((ROOT / args.data).resolve()),
        split=args.eval_split,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        plots=False,
        verbose=False,
    )
    box = metrics.box
    names = getattr(metrics, "names", {}) or {}
    maps = getattr(box, "maps", [])
    per_class = {}
    for idx, value in enumerate(maps):
        per_class[str(names.get(idx, idx))] = round(float(value), 6)
    return {
        "map50_95": round(float(box.map), 6),
        "map50": round(float(box.map50), 6),
        "map75": round(float(box.map75), 6),
        "per_class_map50_95": per_class,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/symbol_recall_detector_p211_yolo_smoke_server/data.yaml")
    ap.add_argument("--output-dir", default="checkpoints/symbol_recall_detector_p211_smoke")
    ap.add_argument("--report", default="reports/vlm/symbol_recall_detector_p211_smoke_eval.json")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--imgsz", type=int, default=384)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="0")
    ap.add_argument("--seed", type=int, default=211)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--no-train-val", action="store_true")
    ap.add_argument("--eval-only-weights", default="")
    args = ap.parse_args()
    weights = ROOT / args.eval_only_weights if args.eval_only_weights else train(args)
    eval_report = eval_model(weights, args)
    report = {
        "id": "P211_symbol_recall_detector_smoke",
        "claim_boundary": "Detector smoke metric only on P211 oversampled tile data; not page-level symbol F1.",
        "data": args.data,
        "weights": rel(weights),
        "config": vars(args),
        "tile_eval": eval_report,
    }
    write_json(ROOT / args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
