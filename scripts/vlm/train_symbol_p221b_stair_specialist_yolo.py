#!/usr/bin/env python3
"""Train/evaluate P221b stair-only YOLO specialist."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets/symbol_p221b_stair_specialist_yolo/data.yaml"
OUT = ROOT / "checkpoints/symbol_p221b_stair_specialist_yolo"
REPORT = ROOT / "reports/vlm/symbol_p221b_stair_specialist_train_eval.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--report", default=str(REPORT))
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="0")
    ap.add_argument("--seed", type=int, default=2215)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(out / "ultralytics_runs"),
        name="train",
        exist_ok=True,
        seed=args.seed,
        patience=20,
        workers=4,
        verbose=False,
    )
    run_dir = Path(results.save_dir)
    weights = run_dir / "weights" / "best.pt"
    target_weights = out / "model.pt"
    if weights.exists():
        shutil.copy2(weights, target_weights)
    eval_model = YOLO(str(target_weights if target_weights.exists() else weights))
    val = eval_model.val(data=str(args.data), split="test", imgsz=args.imgsz, batch=args.batch, device=args.device, verbose=False)
    box = getattr(val, "box", None)
    report = {
        "id": "P221b_stair_specialist_yolo_train_eval",
        "data": str(Path(args.data)),
        "weights": str(target_weights),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "metrics": {
            "map50_95": float(getattr(box, "map", 0.0) if box is not None else 0.0),
            "map50": float(getattr(box, "map50", 0.0) if box is not None else 0.0),
            "map75": float(getattr(box, "map75", 0.0) if box is not None else 0.0),
        },
        "claim_boundary": "Crop-level stair specialist locked split eval; page-level deployment/fusion vs P222 required before symbol metric promotion.",
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
