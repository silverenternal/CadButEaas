#!/usr/bin/env python3
"""Build P200 crop verifier dataset from current-best symbol candidates."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

import sweep_symbol_disagreement_backfill_p165 as p165

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OVERLAY = ROOT / "reports/vlm/symbol_box_refiner_p197b_over_p196c_best_overlay.jsonl"
DEFAULT_OUT = ROOT / "datasets/symbol_crop_verifier_p200"
LABELS = ["false_positive", "appliance", "bathtub", "column", "equipment", "generic_symbol", "shower", "sink", "stair", "table"]
LABEL_TO_ID = {name: idx for idx, name in enumerate(LABELS)}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def target_symbols(row: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for idx, item in enumerate((row.get("targets") or {}).get("symbol") or []):
        box = p165.bbox4(item.get("bbox"))
        if box is None:
            continue
        label = str(item.get("semantic_type") or item.get("symbol_type") or item.get("label") or "generic_symbol")
        out.append({"id": str(item.get("target_id") or idx), "bbox": box, "label": label, "bucket": p165.bucket(box)})
    return out


def best_gold(pred: dict[str, Any], golds: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, float, float]:
    best = None
    best_iou = 0.0
    best_dist = 1e9
    for gold in golds:
        overlap = p165.iou(pred["bbox"], gold["bbox"])
        dist = p165.center_distance(pred["bbox"], gold["bbox"])
        if overlap > best_iou or (overlap == best_iou and dist < best_dist):
            best = gold
            best_iou = overlap
            best_dist = dist
    return best, best_iou, best_dist


def crop_box(box: list[float], image_size: tuple[int, int], scale: float, min_size: int) -> tuple[int, int, int, int]:
    width, height = image_size
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    bw, bh = max(1.0, box[2] - box[0]), max(1.0, box[3] - box[1])
    side = max(min_size, bw * scale, bh * scale)
    x1 = max(0, int(round(cx - side / 2)))
    y1 = max(0, int(round(cy - side / 2)))
    x2 = min(width, int(round(cx + side / 2)))
    y2 = min(height, int(round(cy + side / 2)))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlay", default=str(DEFAULT_OVERLAY))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--crop-size", type=int, default=160)
    parser.add_argument("--crop-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=20260519)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    rows = load_jsonl(Path(args.overlay))
    out = Path(args.out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    records = []
    stats = Counter()
    by_row = defaultdict(Counter)
    for row in rows:
        row_id = str(row.get("row_id") or row.get("id"))
        image_path = Path(str(row.get("image") or row.get("image_path") or ""))
        if not image_path.is_absolute():
            image_path = ROOT / image_path
        if not image_path.exists():
            continue
        golds = target_symbols(row)
        preds = p165.normalized(row.get("symbol_candidates") or [], "p200_candidate")
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_size = image.size
            for idx, pred in enumerate(preds):
                gold, overlap, dist = best_gold(pred, golds)
                is_tp = bool(gold is not None and overlap >= 0.30)
                center_hit = bool(gold is not None and p165.center_covered(pred["bbox"], gold["bbox"]))
                if is_tp:
                    class_name = str(gold["label"])
                else:
                    class_name = "false_positive"
                if class_name not in LABEL_TO_ID:
                    class_name = "generic_symbol" if is_tp else "false_positive"
                x1, y1, x2, y2 = crop_box(pred["bbox"], image_size, args.crop_scale, args.crop_size)
                rel_path = Path("images") / f"{row_id}_{idx:04d}_{class_name}.jpg"
                crop = image.crop((x1, y1, x2, y2)).resize((args.crop_size, args.crop_size))
                crop.save(out / rel_path, quality=92)
                rec = {
                    "row_id": row_id,
                    "candidate_index": idx,
                    "image": str((out / rel_path).resolve()),
                    "label": class_name,
                    "label_id": LABEL_TO_ID[class_name],
                    "is_true_positive": is_tp,
                    "pred_label": pred["label"],
                    "pred_bucket": pred["bucket"],
                    "pred_score": round(float(pred["score"]), 6),
                    "pred_bbox": pred["bbox"],
                    "gold_label": None if gold is None else gold["label"],
                    "best_iou": round(overlap, 6),
                    "best_center_distance": round(dist, 3),
                    "center_hit": center_hit,
                    "crop_bbox": [x1, y1, x2, y2],
                    "source_overlay": str(Path(args.overlay)),
                    "claim_boundary": "Gold-derived labels are for supervised training/evaluation only; runtime verifier consumes raster crops and candidate metadata only.",
                }
                records.append(rec)
                stats[f"label_{class_name}"] += 1
                stats[f"pred_{pred['label']}|{pred['bucket']}"] += 1
                by_row[row_id][class_name] += 1
    rng.shuffle(records)
    n = len(records)
    train_n = int(n * 0.70)
    val_n = int(n * 0.15)
    splits = {"train": records[:train_n], "val": records[train_n:train_n + val_n], "test": records[train_n + val_n:]}
    for split, items in splits.items():
        (out / f"{split}.jsonl").write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in items) + "\n")
    manifest = out / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n")
    report = {
        "id": "P200_symbol_crop_verifier_dataset",
        "source_overlay": args.overlay,
        "labels": LABELS,
        "counts": {"total": n, "train": len(splits["train"]), "val": len(splits["val"]), "test": len(splits["test"])},
        "stats_top": stats.most_common(60),
        "outputs": {"manifest": str(manifest), "train": str(out / "train.jsonl"), "val": str(out / "val.jsonl"), "test": str(out / "test.jsonl"), "images": str(img_dir)},
        "claim_boundary": "Crop labels are derived offline from gold for supervised verifier training; no gold/vector data is available to runtime verifier."
    }
    (out / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
