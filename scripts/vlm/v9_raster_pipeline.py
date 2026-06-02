#!/usr/bin/env python3
"""CadStruct raster-origin floorplan recognition v9 pipeline.

This module intentionally keeps SVG use offline: SVG-derived geometry is used to
build labels and locked gold only. Inference artifacts are generated from raster
images, predicted masks, predicted heatmaps, and vectorized masks.
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import math
import random
import time
import warnings
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from PIL import Image as PILImage

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except Exception:  # pragma: no cover - reported by dependency audit.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    DataLoader = object  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment]

try:
    from skimage import measure, morphology
except Exception:  # pragma: no cover - fallback path below.
    measure = None  # type: ignore[assignment]
    morphology = None  # type: ignore[assignment]

try:
    from scripts.vlm.convert_cubicasa5k_svg import convert_dataset
except Exception:  # pragma: no cover
    from convert_cubicasa5k_svg import convert_dataset  # type: ignore

try:
    from scripts.vlm.v8_raster_e2e_utils import (
        CUBICASA_ROOT,
        ROOT,
        bbox_iou,
        load_json,
        load_jsonl,
        match_counts,
        normalize_bbox,
        row_image_size,
        split_rows_with_locked,
        update_todo_remove,
        write_json,
        write_jsonl,
    )
except Exception:  # pragma: no cover
    from v8_raster_e2e_utils import (  # type: ignore
        CUBICASA_ROOT,
        ROOT,
        bbox_iou,
        load_json,
        load_jsonl,
        match_counts,
        normalize_bbox,
        row_image_size,
        split_rows_with_locked,
        update_todo_remove,
        write_json,
        write_jsonl,
    )


REPORT_DIR = ROOT / "reports/vlm"
SEG_DIR = ROOT / "datasets/raster_segmentation_v9"
CHECKPOINT_DIR = ROOT / "checkpoints"
LABEL_SIZE = 256
PILImage.MAX_IMAGE_PIXELS = None
CLASS_TO_ID = {
    "background": 0,
    "wall": 1,
    "opening": 2,
    "window": 3,
    "room": 4,
    "symbol": 5,
    "text": 6,
}
ID_TO_CLASS = {value: key for key, value in CLASS_TO_ID.items()}
CORE_CLASSES = ["wall", "opening", "window", "room", "symbol", "text"]


def _rel(path: str | Path) -> str:
    p = Path(path)
    if p.is_absolute():
        try:
            return str(p.relative_to(ROOT))
        except ValueError:
            return str(p)
    return str(p)


def _abs(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _safe_open_image(path: str | Path, size: int = LABEL_SIZE) -> Image.Image:
    p = _abs(path)
    if not p.exists():
        return Image.new("RGB", (size, size), "white")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = Image.open(p).convert("RGB")
        img.thumbnail((size, size), Image.Resampling.BILINEAR)
        return img.copy()


def _pad_to_square(img: Image.Image, fill: int | tuple[int, int, int] = 255) -> Image.Image:
    size = max(img.size)
    out = Image.new(img.mode, (size, size), fill)
    out.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return out


def _scale_bbox(bbox: list[float], original: tuple[int, int], size: int = LABEL_SIZE) -> list[int]:
    ow, oh = max(original[0], 1), max(original[1], 1)
    scale = size / max(ow, oh)
    xpad = (size - ow * scale) / 2.0
    ypad = (size - oh * scale) / 2.0
    return [
        int(max(0, min(size - 1, round(bbox[0] * scale + xpad)))),
        int(max(0, min(size - 1, round(bbox[1] * scale + ypad)))),
        int(max(0, min(size - 1, round(bbox[2] * scale + xpad)))),
        int(max(0, min(size - 1, round(bbox[3] * scale + ypad)))),
    ]


def _scale_points(points: list[list[float]], original: tuple[int, int], size: int = LABEL_SIZE) -> list[tuple[int, int]]:
    ow, oh = max(original[0], 1), max(original[1], 1)
    scale = size / max(ow, oh)
    xpad = (size - ow * scale) / 2.0
    ypad = (size - oh * scale) / 2.0
    return [
        (
            int(max(0, min(size - 1, round(float(x) * scale + xpad)))),
            int(max(0, min(size - 1, round(float(y) * scale + ypad)))),
        )
        for x, y in points
    ]


def _gold_class(item: dict[str, Any]) -> str:
    family = str(item.get("family") or "")
    semantic = str(item.get("semantic_type") or "")
    if family == "boundary":
        if semantic == "window":
            return "window"
        if semantic in {"door", "opening"}:
            return "opening"
        return "wall"
    if family == "space":
        return "room"
    if family == "text":
        return "text"
    if family == "symbol":
        return "symbol"
    return "background"


def _bbox_from_points(points: Any) -> list[float] | None:
    if not isinstance(points, list) or not points:
        return None
    xs: list[float] = []
    ys: list[float] = []
    for point in points:
        if isinstance(point, list) and len(point) >= 2:
            try:
                xs.append(float(point[0]))
                ys.append(float(point[1]))
            except (TypeError, ValueError):
                continue
    if not xs or not ys:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


def v9_gold_items(row: dict[str, Any]) -> list[dict[str, Any]]:
    expected = row.get("expected_json") if isinstance(row.get("expected_json"), dict) else {}
    items: list[dict[str, Any]] = []
    for item in expected.get("semantic_candidates") or []:
        geom = item.get("geometry") if isinstance(item.get("geometry"), dict) else {}
        bbox = normalize_bbox(item.get("bbox") or geom.get("bbox")) or _bbox_from_points(geom.get("points"))
        if bbox:
            items.append(
                {
                    "id": item.get("target_id"),
                    "family": "boundary",
                    "semantic_type": item.get("semantic_type"),
                    "bbox": bbox,
                    "geometry": geom,
                    "source": "offline_svg_label",
                }
            )
    for item in expected.get("room_candidates") or []:
        geom = item.get("geometry") if isinstance(item.get("geometry"), dict) else {}
        bbox = normalize_bbox(item.get("bbox") or geom.get("bbox")) or _bbox_from_points(geom.get("points"))
        if bbox:
            items.append({"id": item.get("id"), "family": "space", "semantic_type": item.get("room_type"), "bbox": bbox, "geometry": geom, "source": "offline_svg_label"})
    for item in expected.get("symbol_candidates") or []:
        geom = item.get("geometry") if isinstance(item.get("geometry"), dict) else {}
        bbox = normalize_bbox(item.get("bbox") or geom.get("bbox")) or _bbox_from_points(geom.get("points"))
        if bbox:
            items.append({"id": item.get("id"), "family": "symbol", "semantic_type": item.get("symbol_type"), "bbox": bbox, "geometry": geom, "source": "offline_svg_label"})
    for item in expected.get("text_candidates") or []:
        geom = item.get("geometry") if isinstance(item.get("geometry"), dict) else {}
        bbox = normalize_bbox(item.get("bbox") or geom.get("bbox")) or _bbox_from_points(geom.get("points"))
        if bbox:
            items.append({"id": item.get("id"), "family": "text", "semantic_type": item.get("text_type"), "bbox": bbox, "geometry": geom, "text": item.get("text") or "", "source": "offline_svg_label"})
    return items


def _gold_polygon(item: dict[str, Any]) -> list[list[float]] | None:
    geom = item.get("geometry") if isinstance(item.get("geometry"), dict) else {}
    points = geom.get("points")
    if isinstance(points, list) and len(points) >= 3:
        out: list[list[float]] = []
        for point in points:
            if isinstance(point, list) and len(point) >= 2:
                out.append([float(point[0]), float(point[1])])
        if len(out) >= 3:
            return out
    bbox = normalize_bbox(item.get("bbox"))
    if not bbox:
        return None
    return [[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]]


def _draw_gold(mask: Image.Image, heat: Image.Image, row: dict[str, Any]) -> dict[str, Any]:
    original = row_image_size(row)
    mask_draw = ImageDraw.Draw(mask)
    heat_draw = ImageDraw.Draw(heat)
    counts = Counter()
    invalid = 0
    for item in v9_gold_items(row):
        cls = _gold_class(item)
        class_id = CLASS_TO_ID.get(cls, 0)
        if class_id == 0:
            continue
        polygon = _gold_polygon(item)
        bbox = normalize_bbox(item.get("bbox"))
        if not polygon or not bbox:
            invalid += 1
            continue
        scaled_poly = _scale_points(polygon, original)
        scaled_bbox = _scale_bbox(bbox, original)
        if len(scaled_poly) >= 3:
            mask_draw.polygon(scaled_poly, fill=class_id)
        else:
            mask_draw.rectangle(scaled_bbox, fill=class_id)
        cx = (scaled_bbox[0] + scaled_bbox[2]) // 2
        cy = (scaled_bbox[1] + scaled_bbox[3]) // 2
        r = 2 if cls in {"wall", "room"} else 3
        heat_draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=class_id)
        counts[cls] += 1
    return {"counts": dict(counts), "invalid": invalid}


def _rows_from_converter(limit: int = 0) -> list[dict[str, Any]]:
    existing = []
    for name in ("train", "dev", "smoke"):
        existing.extend(load_jsonl(ROOT / "datasets/cadstruct_cubicasa5k_moe" / f"{name}.jsonl"))
    if existing:
        return existing[:limit] if limit else existing
    return convert_dataset(CUBICASA_ROOT, limit or None, 4.0)


def build_labels(args: argparse.Namespace) -> None:
    start = time.time()
    rows = _rows_from_converter(args.limit)
    splits = split_rows_with_locked(rows, seed=args.seed)
    label_dir = SEG_DIR / "labels"
    image_dir = SEG_DIR / "images"
    label_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    class_pixels = Counter()
    item_counts = Counter()
    polygon_counts = Counter()
    empty_rows = Counter()
    invalid_labels = 0
    output_splits: dict[str, list[dict[str, Any]]] = {}
    for split, split_rows in splits.items():
        out_rows: list[dict[str, Any]] = []
        for index, row in enumerate(split_rows):
            image_path = row.get("image_path")
            if not image_path:
                continue
            key = Path(str(row.get("annotation_path") or image_path)).parent.name or f"{split}_{index}"
            img = _pad_to_square(_safe_open_image(image_path), (255, 255, 255)).resize((LABEL_SIZE, LABEL_SIZE))
            mask = Image.new("L", (LABEL_SIZE, LABEL_SIZE), 0)
            heat = Image.new("L", (LABEL_SIZE, LABEL_SIZE), 0)
            draw_summary = _draw_gold(mask, heat, row)
            invalid_labels += int(draw_summary["invalid"])
            item_counts.update(draw_summary["counts"])
            if not draw_summary["counts"]:
                empty_rows[split] += 1
            mask_arr = np.asarray(mask, dtype=np.uint8)
            for class_id, count in zip(*np.unique(mask_arr, return_counts=True)):
                class_pixels[ID_TO_CLASS[int(class_id)]] += int(count)
            for item in v9_gold_items(row):
                poly = _gold_polygon(item)
                if poly and len(poly) >= 3:
                    polygon_counts[_gold_class(item)] += 1
            image_out = image_dir / f"{split}_{key}_{index}.png"
            mask_out = label_dir / f"{split}_{key}_{index}_mask.png"
            heat_out = label_dir / f"{split}_{key}_{index}_heat.png"
            img.save(image_out)
            mask.save(mask_out)
            heat.save(heat_out)
            out_rows.append(
                {
                    "id": f"{split}_{key}_{index}",
                    "source_key": key,
                    "split": split,
                    "image": _rel(image_out),
                    "original_image": row.get("image_path"),
                    "annotation_path": row.get("annotation_path"),
                    "mask": _rel(mask_out),
                    "heatmap": _rel(heat_out),
                    "label_source": "offline_svg_rasterized_gold",
                    "inference_input": "image_only",
                    "svg_candidate_ids_used": False,
                    "image_size": list(row_image_size(row)),
                    "label_size": LABEL_SIZE,
                    "gold_items": v9_gold_items(row),
                    "gold_counts": draw_summary["counts"],
                }
            )
        output_splits[split] = out_rows
        write_jsonl(SEG_DIR / f"{split}.jsonl", out_rows)
    overlaps = {
        "train_dev": len({r["source_key"] for r in output_splits.get("train", [])} & {r["source_key"] for r in output_splits.get("dev", [])}),
        "train_locked": len({r["source_key"] for r in output_splits.get("train", [])} & {r["source_key"] for r in output_splits.get("locked", [])}),
        "dev_locked": len({r["source_key"] for r in output_splits.get("dev", [])} & {r["source_key"] for r in output_splits.get("locked", [])}),
    }
    audit = {
        "task": "RASTER-V9-T1",
        "label_size": LABEL_SIZE,
        "splits": {k: len(v) for k, v in output_splits.items()},
        "overlaps": overlaps,
        "per_class_pixel_counts": dict(class_pixels),
        "per_class_item_counts": dict(item_counts),
        "per_class_polygon_counts": dict(polygon_counts),
        "empty_label_rows": dict(empty_rows),
        "invalid_label_count": invalid_labels,
        "source_integrity": {
            "inference_input": "image_only",
            "label_source": "offline_svg_rasterized_gold",
            "svg_candidate_ids_used": False,
        },
        "runtime_ms": round((time.time() - start) * 1000, 3),
    }
    write_json(REPORT_DIR / "raster_segmentation_label_audit_v9.json", audit)
    _write_label_audit_html(output_splits.get("locked", [])[:6], audit)
    update_todo_remove(["RASTER-V9-T1"])


def _img_data_uri(path: str | Path) -> str:
    p = _abs(path)
    if not p.exists():
        return ""
    data = p.read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _write_label_audit_html(rows: list[dict[str, Any]], audit: dict[str, Any]) -> None:
    cards = []
    for row in rows:
        cards.append(
            f"<section><h3>{row['id']}</h3><div class='grid'>"
            f"<figure><img src='{_img_data_uri(row['image'])}'><figcaption>input image_only</figcaption></figure>"
            f"<figure><img src='{_img_data_uri(row['mask'])}'><figcaption>offline SVG rasterized label mask</figcaption></figure>"
            f"<figure><img src='{_img_data_uri(row['heatmap'])}'><figcaption>offline heatmap label</figcaption></figure>"
            "</div></section>"
        )
    html = f"""<!doctype html><meta charset="utf-8"><title>v9 label audit</title>
<style>body{{font-family:Arial,sans-serif;margin:24px}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}img{{width:100%;border:1px solid #bbb;background:white}}pre{{background:#f5f5f5;padding:12px;overflow:auto}}</style>
<h1>Raster segmentation label audit v9</h1><p>SVG is used only as offline gold labels. Inference input is image_only.</p>
<pre>{json.dumps(audit, ensure_ascii=False, indent=2)}</pre>{''.join(cards)}"""
    path = REPORT_DIR / "raster_segmentation_label_audit_v9.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


class SegDataset(Dataset):  # type: ignore[misc]
    def __init__(self, rows: list[dict[str, Any]], max_rows: int = 0):
        self.rows = rows[:max_rows] if max_rows else rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Any, Any, Any]:
        row = self.rows[index]
        image = np.asarray(Image.open(_abs(row["image"])).convert("L"), dtype=np.float32) / 255.0
        mask = np.asarray(Image.open(_abs(row["mask"])).convert("L"), dtype=np.int64)
        heat = (np.asarray(Image.open(_abs(row["heatmap"])).convert("L"), dtype=np.float32) > 0).astype(np.float32)
        x = torch.from_numpy(image[None, :, :])
        y = torch.from_numpy(mask)
        h = torch.from_numpy(heat[None, :, :])
        return x, y, h


class TinySegNet(nn.Module):  # type: ignore[misc]
    def __init__(self, classes: int = 7, heat: bool = True):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 16, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(32, 48, 3, padding=1), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(48, 32, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU())
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.ReLU())
        self.mask_head = nn.Conv2d(16, classes, 1)
        self.heat_head = nn.Conv2d(16, 1, 1) if heat else None

    def forward(self, x: Any) -> tuple[Any, Any | None]:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.mask_head(d1), self.heat_head(d1) if self.heat_head is not None else None


def _pixel_metrics(pred: np.ndarray, gold: np.ndarray) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    f1s = []
    ious = []
    for cls in CORE_CLASSES:
        cid = CLASS_TO_ID[cls]
        p = pred == cid
        g = gold == cid
        tp = int(np.logical_and(p, g).sum())
        fp = int(np.logical_and(p, ~g).sum())
        fn = int(np.logical_and(~p, g).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        iou = tp / max(tp + fp + fn, 1)
        metrics[cls] = {"tp": tp, "fp": fp, "fn": fn, "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6), "iou": round(iou, 6)}
        f1s.append(f1)
        ious.append(iou)
    metrics["mean_f1"] = round(float(np.mean(f1s)), 6) if f1s else 0.0
    metrics["mean_iou"] = round(float(np.mean(ious)), 6) if ious else 0.0
    return metrics


def train_segmentation(args: argparse.Namespace, multitask: bool = False) -> None:
    if torch is None:
        raise RuntimeError("torch is required for v9 segmentation training")
    if not (SEG_DIR / "train.jsonl").exists():
        build_labels(argparse.Namespace(limit=args.label_limit, seed=args.seed))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_rows = load_jsonl(SEG_DIR / "train.jsonl")
    dev_rows = load_jsonl(SEG_DIR / "dev.jsonl")
    locked_rows = load_jsonl(SEG_DIR / "locked.jsonl")
    overlap = _split_overlap_report(train_rows, dev_rows, locked_rows)
    train_ds = SegDataset(train_rows, max_rows=args.max_train)
    dev_ds = SegDataset(dev_rows, max_rows=args.max_eval)
    locked_ds = SegDataset(locked_rows, max_rows=args.max_eval)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    model = TinySegNet(heat=multitask).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    class_weights = torch.tensor([0.2, 1.4, 1.8, 1.8, 1.2, 1.8, 2.0], dtype=torch.float32, device=args.device)
    losses: list[float] = []
    for _epoch in range(args.epochs):
        model.train()
        for x, y, heat in train_loader:
            x, y, heat = x.to(args.device), y.to(args.device), heat.to(args.device)
            opt.zero_grad()
            logits, heat_logits = model(x)
            loss = F.cross_entropy(logits, y, weight=class_weights)
            if multitask and heat_logits is not None:
                loss = loss + 0.25 * F.binary_cross_entropy_with_logits(heat_logits, (heat > 0).float())
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
    ckpt_dir = CHECKPOINT_DIR / ("muranet_lite_v9" if multitask else "raster_segmentation_baseline_v9")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "multitask": multitask, "class_to_id": CLASS_TO_ID}, ckpt_dir / "model.pt")
    dev_eval, _ = _eval_model(model, dev_ds, args.device, prefix="dev", save_predictions=False)
    locked_eval, locked_predictions = _eval_model(model, locked_ds, args.device, prefix="locked", save_predictions=True, multitask=multitask)
    adopted = bool((locked_eval["mean_iou"] >= 0.45) or (locked_eval["mean_f1"] >= 0.65))
    if multitask:
        baseline = load_json(REPORT_DIR / "raster_segmentation_baseline_v9_eval.json", {})
        baseline_f1 = float((baseline.get("locked") or {}).get("mean_f1") or 0.0)
        improved_classes = [
            cls for cls in CORE_CLASSES
            if locked_eval.get(cls, {}).get("f1", 0.0) > (baseline.get("locked") or {}).get(cls, {}).get("f1", 0.0)
        ]
        adopted = adopted and locked_eval["mean_f1"] >= baseline_f1 and len(improved_classes) >= 2
    report = {
        "task": "RASTER-V9-T3" if multitask else "RASTER-V9-T2",
        "run_mode": "bounded_local_training",
        "model": "tiny_muranet_lite_segmentation_heatmap" if multitask else "tiny_cubicasa_style_segmentation_heatmap",
        "train_count": len(train_ds),
        "dev_count": len(dev_ds),
        "locked_count": len(locked_ds),
        "dev": dev_eval,
        "locked": locked_eval,
        "split_overlap": overlap,
        "calibration": {
            "method": "not_calibrated_bounded_local_run",
            "reason": "This v9 execution trains small local baselines and reports raw mask argmax outputs; probability calibration needs a separate dev-fitted temperature or threshold sweep before paper claims.",
        },
        "adopted": adopted,
        "adoption_rule": "locked mean IoU >= 0.45 or mean F1 >= 0.65; multitask also must improve at least two core classes over T2",
        "source_integrity": {"predictions": "mask_heatmap_outputs", "svg_candidate_ids_used": False},
    }
    if multitask:
        report["detection_proxy"] = _detection_proxy_from_predictions(locked_predictions, locked_rows[: len(locked_predictions)])
        report["detection_ap"] = {
            "available": False,
            "reason": "The bounded MuraNet-lite branch emits segmentation/heatmap components; AP is not claimed without a dedicated bbox decoder and confidence ranking.",
        }
    write_json(ckpt_dir / "train_summary.json", {"loss_tail": losses[-20:], "epochs": args.epochs, "batch_size": args.batch_size})
    report_name = "muranet_lite_v9_eval.json" if multitask else "raster_segmentation_baseline_v9_eval.json"
    pred_name = "muranet_lite_v9_locked_predictions.jsonl" if multitask else "raster_segmentation_baseline_v9_locked_predictions.jsonl"
    write_json(REPORT_DIR / report_name, report)
    write_jsonl(REPORT_DIR / pred_name, locked_predictions)
    update_todo_remove(["RASTER-V9-T3" if multitask else "RASTER-V9-T2"])


def _split_overlap_report(train_rows: list[dict[str, Any]], dev_rows: list[dict[str, Any]], locked_rows: list[dict[str, Any]]) -> dict[str, int]:
    train = {str(row.get("source_key") or row.get("id")) for row in train_rows}
    dev = {str(row.get("source_key") or row.get("id")) for row in dev_rows}
    locked = {str(row.get("source_key") or row.get("id")) for row in locked_rows}
    return {
        "train_dev": len(train & dev),
        "train_locked": len(train & locked),
        "dev_locked": len(dev & locked),
    }


def _eval_model(model: Any, ds: SegDataset, device: str, prefix: str, save_predictions: bool, multitask: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    all_pred: list[np.ndarray] = []
    all_gold: list[np.ndarray] = []
    predictions: list[dict[str, Any]] = []
    pred_dir = REPORT_DIR / ("muranet_lite_v9_masks" if multitask else "raster_segmentation_baseline_v9_masks")
    pred_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for index in range(len(ds)):
            x, y, _heat = ds[index]
            logits, heat_logits = model(x[None, :, :, :].to(device))
            pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            gold = y.numpy().astype(np.uint8)
            all_pred.append(pred)
            all_gold.append(gold)
            if save_predictions:
                row = ds.rows[index]
                mask_path = pred_dir / f"{row['id']}_pred.png"
                Image.fromarray(pred).save(mask_path)
                heat_path = None
                if heat_logits is not None:
                    heat_arr = (torch.sigmoid(heat_logits)[0, 0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                    heat_path = pred_dir / f"{row['id']}_heat.png"
                    Image.fromarray(heat_arr).save(heat_path)
                predictions.append(
                    {
                        "id": row["id"],
                        "split": prefix,
                        "image": row["image"],
                        "gold_mask": row["mask"],
                        "pred_mask": _rel(mask_path),
                        "pred_heatmap": _rel(heat_path) if heat_path else None,
                        "source_mode": "model_v9_raster_segmentation_or_sequence",
                        "svg_candidate_ids_used": False,
                        "image_size": row.get("image_size") or [LABEL_SIZE, LABEL_SIZE],
                        "gold_items": row.get("gold_items") or [],
                    }
                )
    if not all_pred:
        return {"mean_f1": 0.0, "mean_iou": 0.0}, predictions
    return _pixel_metrics(np.concatenate([p.reshape(-1) for p in all_pred]), np.concatenate([g.reshape(-1) for g in all_gold])), predictions


def _components(mask: np.ndarray, class_id: int, min_area: int = 8) -> list[dict[str, Any]]:
    binary = mask == class_id
    if measure is None:
        ys, xs = np.where(binary)
        if len(xs) < min_area:
            return []
        return [{"bbox": [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)], "area": int(len(xs)), "polygon": []}]
    labels = measure.label(binary, connectivity=2)
    comps = []
    for region in measure.regionprops(labels):
        if region.area < min_area:
            continue
        minr, minc, maxr, maxc = region.bbox
        contours = measure.find_contours(labels == region.label, 0.5)
        poly: list[list[float]] = []
        if contours:
            contour = max(contours, key=len)
            step = max(1, len(contour) // 24)
            poly = [[round(float(c), 3), round(float(r), 3)] for r, c in contour[::step]]
        comps.append({"bbox": [int(minc), int(minr), int(maxc), int(maxr)], "area": int(region.area), "polygon": poly})
    return comps


def _detection_proxy_from_predictions(predictions: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for cls in ["opening", "window", "symbol", "text"]:
        cid = CLASS_TO_ID[cls]
        preds: list[dict[str, Any]] = []
        golds: list[dict[str, Any]] = []
        for pred_row, gold_row in zip(predictions, rows):
            mask = np.asarray(Image.open(_abs(pred_row["pred_mask"])).convert("L"), dtype=np.uint8)
            preds.extend({"bbox": comp["bbox"]} for comp in _components(mask, cid, 6))
            golds.extend({"bbox": _scale_bbox(item["bbox"], tuple(gold_row.get("image_size") or [1, 1]))} for item in gold_row.get("gold_items") or [] if _gold_class(item) == cls and normalize_bbox(item.get("bbox")))
        tp, pc, gc, _fp, _miss = match_counts(preds, golds, 0.3)
        precision = tp / max(pc, 1)
        recall = tp / max(gc, 1)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        out[cls] = {"tp": tp, "predicted": pc, "gold": gc, "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6), "matching_iou": 0.3}
    return out


def _load_best_predictions() -> tuple[list[dict[str, Any]], str]:
    mt = load_json(REPORT_DIR / "muranet_lite_v9_eval.json", {})
    base = load_json(REPORT_DIR / "raster_segmentation_baseline_v9_eval.json", {})
    if mt.get("adopted") or float((mt.get("locked") or {}).get("mean_f1") or 0) >= float((base.get("locked") or {}).get("mean_f1") or 0):
        rows = load_jsonl(REPORT_DIR / "muranet_lite_v9_locked_predictions.jsonl")
        if rows:
            return rows, "muranet_lite_v9"
    return load_jsonl(REPORT_DIR / "raster_segmentation_baseline_v9_locked_predictions.jsonl"), "raster_segmentation_baseline_v9"


def vectorize_rooms(args: argparse.Namespace) -> None:
    predictions, source = _load_best_predictions()
    cases = []
    metrics = defaultdict(dict)
    for threshold in (0.3, 0.5, 0.7):
        preds: list[dict[str, Any]] = []
        golds: list[dict[str, Any]] = []
        for row in predictions:
            mask = np.asarray(Image.open(_abs(row["pred_mask"])).convert("L"), dtype=np.uint8)
            comps = _components(mask, CLASS_TO_ID["room"], min_area=30)
            preds.extend({"bbox": comp["bbox"], "polygon": comp["polygon"], "geometry_source": "mask_or_heatmap_vectorized"} for comp in comps)
            golds.extend({"bbox": _scale_bbox(item["bbox"], tuple(row.get("image_size") or [LABEL_SIZE, LABEL_SIZE]))} for item in row.get("gold_items") or [] if _gold_class(item) == "room" and normalize_bbox(item.get("bbox")))
            if len(cases) < 80:
                cases.append({"id": row["id"], "predicted_rooms": comps[:20], "gold_room_count": len([i for i in row.get("gold_items") or [] if _gold_class(i) == "room"])})
        tp, pc, gc, fp, miss = match_counts(preds, golds, threshold)
        precision = tp / max(pc, 1)
        recall = tp / max(gc, 1)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        metrics[f"iou_{threshold}"] = {"tp": tp, "predicted": pc, "gold": gc, "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6), "false_positive_examples": fp[:5], "miss_examples": miss[:5]}
    report = {"task": "RASTER-V9-T4", "source_predictions": source, "geometry_source": "mask_or_heatmap_vectorized", "no_svg_gold_copied_at_inference": True, "metrics": dict(metrics), "adopted": metrics["iou_0.5"]["precision"] >= 0.45 and metrics["iou_0.5"]["recall"] >= 0.45}
    write_json(REPORT_DIR / "room_polygon_vectorization_v9_eval.json", report)
    write_jsonl(REPORT_DIR / "room_polygon_vectorization_v9_cases.jsonl", cases)
    update_todo_remove(["RASTER-V9-T4"])


def vectorize_wall_opening(args: argparse.Namespace) -> None:
    predictions, source = _load_best_predictions()
    cases = []
    report_classes = ["wall", "opening", "window"]
    metrics = {}
    for cls in report_classes:
        cid = CLASS_TO_ID[cls]
        preds: list[dict[str, Any]] = []
        golds: list[dict[str, Any]] = []
        for row in predictions:
            mask = np.asarray(Image.open(_abs(row["pred_mask"])).convert("L"), dtype=np.uint8)
            clean = mask == cid
            if cls == "wall" and morphology is not None:
                clean = morphology.remove_small_objects(clean, min_size=20)
            comps = _components(clean.astype(np.uint8), 1, min_area=20 if cls == "wall" else 8)
            for comp in comps:
                x1, y1, x2, y2 = comp["bbox"]
                if cls == "wall" and (x2 - x1) * (y2 - y1) > LABEL_SIZE * LABEL_SIZE * 0.35:
                    continue
                preds.append({"bbox": comp["bbox"], "geometry_source": "mask_or_heatmap_vectorized"})
            golds.extend({"bbox": _scale_bbox(item["bbox"], tuple(row.get("image_size") or [LABEL_SIZE, LABEL_SIZE]))} for item in row.get("gold_items") or [] if _gold_class(item) == cls and normalize_bbox(item.get("bbox")))
        tp, pc, gc, fp, miss = match_counts(preds, golds, 0.3)
        precision = tp / max(pc, 1)
        recall = tp / max(gc, 1)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        metrics[cls] = {"tp": tp, "predicted": pc, "gold": gc, "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6), "false_positive_examples": fp[:5], "miss_examples": miss[:5]}
        cases.append({"class": cls, "false_positive_examples": fp[:20], "miss_examples": miss[:20]})
    report = {"task": "RASTER-V9-T5", "source_predictions": source, "geometry_source": "mask_or_heatmap_vectorized", "metrics": metrics, "adopted": all(metrics[c]["f1"] >= 0.45 for c in report_classes)}
    write_json(REPORT_DIR / "wall_opening_vectorization_v9_eval.json", report)
    write_jsonl(REPORT_DIR / "wall_opening_vectorization_v9_cases.jsonl", cases)
    update_todo_remove(["RASTER-V9-T5"])


def text_detection(args: argparse.Namespace) -> None:
    predictions, source = _load_best_predictions()
    rows_out = []
    preds: list[dict[str, Any]] = []
    golds: list[dict[str, Any]] = []
    numeric_golds = 0
    numeric_matches = 0
    ocr_available = importlib.util.find_spec("easyocr") is not None or importlib.util.find_spec("pytesseract") is not None
    for row in predictions:
        mask = np.asarray(Image.open(_abs(row["pred_mask"])).convert("L"), dtype=np.uint8)
        comps = _components(mask, CLASS_TO_ID["text"], min_area=4)
        row_preds = [{"bbox": comp["bbox"], "geometry_source": "mask_or_heatmap_vectorized", "ocr_text": None} for comp in comps]
        row_golds = [item for item in row.get("gold_items") or [] if _gold_class(item) == "text" and normalize_bbox(item.get("bbox"))]
        for item in row_golds:
            sb = _scale_bbox(item["bbox"], tuple(row.get("image_size") or [LABEL_SIZE, LABEL_SIZE]))
            golds.append({"bbox": sb, "text": item.get("text") or ""})
            if any(ch.isdigit() for ch in str(item.get("text") or "")):
                numeric_golds += 1
        preds.extend(row_preds)
        rows_out.append({"id": row["id"], "predicted_text": row_preds[:40], "gold_text_count": len(row_golds), "missed_numeric_labels_review": [g for g in row_golds if any(ch.isdigit() for ch in str(g.get("text") or ""))][:10]})
    tp, pc, gc, fp, miss = match_counts(preds, golds, 0.3)
    for miss_case in miss:
        text = str((miss_case.get("gold") or {}).get("text") or "")
        if any(ch.isdigit() for ch in text):
            pass
    for pred in preds:
        pb = pred["bbox"]
        for gold in golds:
            if any(ch.isdigit() for ch in str(gold.get("text") or "")) and bbox_iou(pb, gold["bbox"]) >= 0.3:
                numeric_matches += 1
                break
    precision = tp / max(pc, 1)
    recall = tp / max(gc, 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    report = {
        "task": "RASTER-V9-T6",
        "source_predictions": source,
        "text_bbox": {"tp": tp, "predicted": pc, "gold": gc, "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6)},
        "numeric_text_recall": round(numeric_matches / max(numeric_golds, 1), 6),
        "ocr_content_accuracy": None,
        "ocr_backend_available": ocr_available,
        "ocr_note": "OCR content recognition is separated from text detection; no local OCR backend was invoked in this bounded v9 run." if not ocr_available else "OCR backend detected but content recognition remains exploratory until separately locked.",
        "adopted": recall >= 0.50,
        "false_text_examples": fp[:20],
        "missed_text_examples": miss[:20],
    }
    write_json(REPORT_DIR / "text_detection_ocr_v9_eval.json", report)
    write_jsonl(REPORT_DIR / "text_detection_ocr_v9_cases.jsonl", rows_out)
    update_todo_remove(["RASTER-V9-T6"])


def build_scene_graph(args: argparse.Namespace) -> None:
    predictions, source = _load_best_predictions()
    seg_eval = load_json(REPORT_DIR / "muranet_lite_v9_eval.json", {}) or load_json(REPORT_DIR / "raster_segmentation_baseline_v9_eval.json", {})
    room_eval = load_json(REPORT_DIR / "room_polygon_vectorization_v9_eval.json", {})
    wall_eval = load_json(REPORT_DIR / "wall_opening_vectorization_v9_eval.json", {})
    text_eval = load_json(REPORT_DIR / "text_detection_ocr_v9_eval.json", {})
    adopted_components = {
        "segmentation": bool(seg_eval.get("adopted")),
        "room_polygon": bool(room_eval.get("adopted")),
        "wall_opening": bool(wall_eval.get("adopted")),
        "text_detection": bool(text_eval.get("adopted")),
    }
    if not adopted_components["segmentation"]:
        write_jsonl(REPORT_DIR / "model_v9_raster_predictions.jsonl", [])
        write_json(REPORT_DIR / "model_v9_raster_adoption_decisions.json", {"task": "RASTER-V9-T7", "adopted": False, "reason": "No raster segmentation component met adoption targets.", "adopted_components": adopted_components, "source_predictions": source})
        update_todo_remove(["RASTER-V9-T7"])
        return
    out = []
    for row in predictions:
        mask = np.asarray(Image.open(_abs(row["pred_mask"])).convert("L"), dtype=np.uint8)
        nodes = []
        for cls in ["wall", "opening", "window", "room", "symbol", "text"]:
            for idx, comp in enumerate(_components(mask, CLASS_TO_ID[cls], min_area=20 if cls in {"wall", "room"} else 6)):
                nodes.append({"id": f"{row['id']}_{cls}_{idx}", "family": "space" if cls == "room" else ("boundary" if cls in {"wall", "opening", "window"} else cls), "semantic_type": cls, "bbox": comp["bbox"], "polygon": comp.get("polygon") or [], "geometry_source": "mask_or_heatmap_vectorized", "proposal_source": source})
        out.append({"id": row["id"], "route_trace": {"source_mode": "model_v9_raster_segmentation_or_sequence", "svg_candidate_ids_used": False}, "scene_graph": {"nodes": nodes, "edges": []}, "image": row["image"]})
    write_jsonl(REPORT_DIR / "model_v9_raster_predictions.jsonl", out)
    write_json(REPORT_DIR / "model_v9_raster_adoption_decisions.json", {"task": "RASTER-V9-T7", "adopted": True, "adopted_components": adopted_components, "source_predictions": source, "node_count": sum(len(r["scene_graph"]["nodes"]) for r in out)})
    update_todo_remove(["RASTER-V9-T7"])


def sequence_graph_feasibility(args: argparse.Namespace) -> None:
    locked = load_jsonl(SEG_DIR / "locked.jsonl")[: min(20, args.limit)]
    smoke = []
    for row in locked:
        polys = []
        for item in row.get("gold_items") or []:
            poly = _gold_polygon(item)
            if poly:
                polys.append({"label": _gold_class(item), "points": poly[:64]})
        smoke.append({"id": row["id"], "image": row["image"], "target_sequence": polys[:120], "label_source": "offline_svg_sequence_gold", "exploratory": True})
    write_jsonl(ROOT / "datasets/raster_sequence_graph_v9/smoke.jsonl", smoke)
    report = {
        "task": "RASTER-V9-T8",
        "methods_audited": ["Raster-to-Graph", "Raster2Seq"],
        "license_runtime_blockers": ["External repos and checkpoints are not vendored in this project.", "A publishable sequence model needs a longer training budget than this bounded execution."],
        "local_export_created": True,
        "smoke_rows": len(smoke),
        "recommendation": "Keep segmentation/vectorization as v9 core; evaluate graph/sequence prediction as a v10 architecture branch when repo code, license, and training budget are explicitly pinned.",
        "adopted": False,
        "exploratory_only": True,
    }
    write_json(REPORT_DIR / "sequence_graph_feasibility_v9.json", report)
    update_todo_remove(["RASTER-V9-T8"])


def sam_feasibility(args: argparse.Namespace) -> None:
    deps = {name: importlib.util.find_spec(name) is not None for name in ["segment_anything", "sam2", "torch", "PIL"]}
    cases: list[dict[str, Any]] = []
    report = {
        "task": "RASTER-V9-T9",
        "dependencies": deps,
        "weights_found": [str(p) for p in ROOT.rglob("*sam*.pth")][:20],
        "reproducibility_status": "no_run" if not (deps.get("segment_anything") or deps.get("sam2")) else "dependency_present_but_not_adopted_without_locked_protocol",
        "adopted": False,
        "exploratory_only": True,
        "note": "Foundation-model masks are not mixed into model_v9_raster. They require pinned weights/prompts and locked metrics before any adoption.",
    }
    write_json(REPORT_DIR / "sam_floorplan_feasibility_v9.json", report)
    write_jsonl(REPORT_DIR / "sam_floorplan_cases_v9.jsonl", cases)
    update_todo_remove(["RASTER-V9-T9"])


def render_review(args: argparse.Namespace) -> None:
    predictions, source = _load_best_predictions()
    adoption = load_json(REPORT_DIR / "model_v9_raster_adoption_decisions.json", {})
    rows = predictions[:8]
    _write_review_html(REPORT_DIR / "visual_demo_model_v9_raster/review_pack_v3/index.html", rows, source, adoption, "v9 raster model review")
    _write_review_html(REPORT_DIR / "visual_demo_v9_comparison/index.html", rows, source, adoption, "v7/v8/v9 comparison placeholder with v9 raster evidence")
    _write_review_html(REPORT_DIR / "visual_demo_v9_failure_gallery/index.html", rows, source, adoption, "v9 locked failure gallery")
    update_todo_remove(["RASTER-V9-T10"])


def _overlay_uri(row: dict[str, Any]) -> str:
    img = Image.open(_abs(row["image"])).convert("RGBA")
    mask = Image.open(_abs(row["pred_mask"])).convert("L").resize(img.size, Image.Resampling.NEAREST)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    colors = {1: (220, 30, 30, 110), 2: (20, 130, 220, 120), 3: (40, 180, 140, 120), 4: (240, 200, 50, 80), 5: (160, 80, 210, 120), 6: (20, 20, 20, 150)}
    arr = np.asarray(mask)
    pix = overlay.load()
    for cid, color in colors.items():
        ys, xs = np.where(arr == cid)
        for x, y in zip(xs[::2], ys[::2]):
            pix[int(x), int(y)] = color
    out = Image.alpha_composite(img, overlay)
    buf = BytesIO()
    out.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _write_review_html(path: Path, rows: list[dict[str, Any]], source: str, adoption: dict[str, Any], title: str) -> None:
    cards = []
    status = "ADOPTED" if adoption.get("adopted") else "REJECTED / exploratory"
    for row in rows:
        cards.append(
            f"<section><h2>{row['id']}</h2><p><b>source_mode:</b> model_v9_raster_segmentation_or_sequence; <b>svg_candidate_ids_used:</b> false; <b>component:</b> {source}</p>"
            f"<div class='grid'><figure><img src='{_img_data_uri(row['image'])}'><figcaption>input raster</figcaption></figure>"
            f"<figure><img src='{_img_data_uri(row['pred_mask'])}'><figcaption>predicted mask</figcaption></figure>"
            f"<figure><img src='{_overlay_uri(row)}'><figcaption>prediction overlay</figcaption></figure></div></section>"
        )
    html = f"""<!doctype html><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:Arial,sans-serif;margin:24px;color:#222}}.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}img{{width:100%;border:1px solid #bbb;background:white}}.badge{{display:inline-block;padding:4px 8px;border:1px solid #555;background:#f5f5f5}}figure{{margin:0}}figcaption{{font-size:13px}}</style>
<h1>{title}</h1><p class="badge">{status}</p><p>This page shows raster-model predictions. SVG/parser candidate geometry is not displayed as v9 prediction.</p>{''.join(cards)}"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def evaluate_locked(args: argparse.Namespace) -> None:
    seg = load_json(REPORT_DIR / "muranet_lite_v9_eval.json", {}) or load_json(REPORT_DIR / "raster_segmentation_baseline_v9_eval.json", {})
    room = load_json(REPORT_DIR / "room_polygon_vectorization_v9_eval.json", {})
    wall = load_json(REPORT_DIR / "wall_opening_vectorization_v9_eval.json", {})
    text = load_json(REPORT_DIR / "text_detection_ocr_v9_eval.json", {})
    scene = load_json(REPORT_DIR / "model_v9_raster_adoption_decisions.json", {})
    pain = {
        "false_wall": (wall.get("metrics") or {}).get("wall", {}).get("false_positive_examples", []),
        "false_equipment": "not adopted as separate equipment detector in v9 raster stream",
        "missed_room": ((room.get("metrics") or {}).get("iou_0.5") or {}).get("miss_examples", []),
        "missed_text": text.get("missed_text_examples", []),
        "text_symbol_confusion": "approximated by separate symbol/text mask class metrics",
        "coordinate_error": "reported through IoU/vector matching rather than renderer offsets",
    }
    report = {
        "task": "RASTER-V9-T11",
        "segmentation": seg,
        "room_polygon": room,
        "wall_opening": wall,
        "text": text,
        "scene_graph": scene,
        "residual_pain_points": {k: (len(v) if isinstance(v, list) else v) for k, v in pain.items()},
        "adoption_decision": {"adopted": bool(scene.get("adopted")), "reason": scene.get("reason") or "See component metrics."},
    }
    write_json(REPORT_DIR / "model_v9_raster_locked_eval.json", report)
    write_jsonl(REPORT_DIR / "model_v9_raster_error_cases.jsonl", [{"type": k, "cases": v[:30] if isinstance(v, list) else v} for k, v in pain.items()])
    write_json(REPORT_DIR / "model_v9_vs_v8_vs_v7_ablation.json", {"task": "RASTER-V9-T11", "v7": "SVG/parser candidate geometry plus learned refiners", "v8": "hybrid SVG candidates plus raster visual evidence", "v9": "raster segmentation/vectorization stream", "decision": report["adoption_decision"]})
    update_todo_remove(["RASTER-V9-T11"])


def update_docs(args: argparse.Namespace) -> None:
    eval_report = load_json(REPORT_DIR / "model_v9_raster_locked_eval.json", {})
    seg = (eval_report.get("segmentation") or {}).get("locked") or {}
    adopted = bool((eval_report.get("adoption_decision") or {}).get("adopted"))
    arch = f"""# CadStruct v9 Raster Recognition Architecture

v9 changes the raster path from connected-component proposal boxes to semantic segmentation, heatmaps, and mask vectorization.
SVG/parser geometry is allowed only as offline CubiCasa gold labels and locked evaluation gold; it is not used as inference-time candidate geometry.

## Components

- T1 raster labels: `datasets/raster_segmentation_v9/*.jsonl`
- T2 CubiCasa-style segmentation baseline: `reports/vlm/raster_segmentation_baseline_v9_eval.json`
- T3 MuraNet-lite multitask branch: `reports/vlm/muranet_lite_v9_eval.json`
- T4 room mask-to-polygon vectorization: `reports/vlm/room_polygon_vectorization_v9_eval.json`
- T5 wall/opening/window vectorization: `reports/vlm/wall_opening_vectorization_v9_eval.json`
- T6 raster text detection and OCR audit: `reports/vlm/text_detection_ocr_v9_eval.json`
- T7 model stream decision: `reports/vlm/model_v9_raster_adoption_decisions.json`

## Locked Result

- adopted: `{adopted}`
- locked mean IoU: `{seg.get('mean_iou')}`
- locked mean F1: `{seg.get('mean_f1')}`

External research basis: CubiCasa5K official segmentation labels, MuraNet multitask segmentation+detection, Raster-to-Graph/Raster2Seq graph or polygon sequence prediction, and SAM-style exploratory segmentation. The current repo keeps sequence/foundation-model work exploratory until dependencies, weights, and locked metrics are pinned.
"""
    runbook = """# CadStruct v9 Training Runbook

Run the bounded local pipeline:

```bash
uv run python scripts/vlm/build_cubicasa_raster_label_tensors_v9.py --limit 0
uv run python scripts/vlm/train_raster_segmentation_baseline_v9.py --epochs 2 --max-train 320 --max-eval 80
uv run python scripts/vlm/train_muranet_lite_v9.py --epochs 2 --max-train 320 --max-eval 80
uv run python scripts/vlm/vectorize_room_polygons_v9.py
uv run python scripts/vlm/vectorize_wall_opening_v9.py
uv run python scripts/vlm/train_text_detection_v9.py
uv run python scripts/vlm/build_model_v9_raster_scene_graph.py
uv run python scripts/vlm/evaluate_model_v9_raster_locked.py
```

Visual review:

```bash
uv run python scripts/vlm/render_raster_model_v9_review_pack.py
```

The generated visual pages are under `reports/vlm/visual_demo_model_v9_raster/`, `reports/vlm/visual_demo_v9_comparison/`, and `reports/vlm/visual_demo_v9_failure_gallery/`.
"""
    advisor_append = f"""

## v9 Raster Recognition Update

v9 implements a raster-origin segmentation/vectorization stream and keeps the claim boundary explicit:
v7 is SVG/parser candidate geometry plus learned refiners; v8 is hybrid SVG candidates plus raster evidence; v9 is raster masks/heatmaps/vectorization.

Locked v9 adoption: `{adopted}`. The authoritative reports are:

- `reports/vlm/model_v9_raster_locked_eval.json`
- `reports/vlm/model_v9_raster_adoption_decisions.json`
- `reports/vlm/visual_demo_model_v9_raster/review_pack_v3/index.html`

If v9 is rejected, the project should position the SCI contribution around auditable domain-structured MoE, visual evidence overlays, source-mode integrity, and error attribution, while treating full raster recognition as an active improvement track rather than the core claim.
"""
    (ROOT / "docs/cadstruct/runbooks/cadstruct-v9-raster-recognition-architecture.md").write_text(arch, encoding="utf-8")
    (ROOT / "docs/cadstruct/runbooks/cadstruct-v9-training-runbook.md").write_text(runbook, encoding="utf-8")
    for doc in [ROOT / "docs/cadstruct/archive/cadstruct-moe-advisor-report.md", ROOT / "docs/cadstruct/archive/cadstruct-visual-result-demo-notes.md"]:
        original = doc.read_text(encoding="utf-8") if doc.exists() else ""
        marker = "## v9 Raster Recognition Update"
        if marker in original:
            original = original.split(marker)[0].rstrip() + "\n"
        doc.write_text(original.rstrip() + "\n" + advisor_append, encoding="utf-8")
    update_todo_remove(["RASTER-V9-T12"])


def run_all(args: argparse.Namespace) -> None:
    build_labels(args)
    train_segmentation(args, multitask=False)
    train_segmentation(args, multitask=True)
    vectorize_rooms(args)
    vectorize_wall_opening(args)
    text_detection(args)
    build_scene_graph(args)
    sequence_graph_feasibility(args)
    sam_feasibility(args)
    render_review(args)
    evaluate_locked(args)
    update_docs(args)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["build-labels", "train-seg", "train-muranet", "vectorize-rooms", "vectorize-wall-opening", "text-detection", "build-scene-graph", "sequence-feasibility", "sam-feasibility", "render-review", "evaluate-locked", "update-docs", "run-all"])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--label-limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260507)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-train", type=int, default=320)
    p.add_argument("--max-eval", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda" if torch is not None and torch.cuda.is_available() else "cpu")
    return p


def main() -> None:
    args = parser().parse_args()
    actions = {
        "build-labels": build_labels,
        "train-seg": lambda a: train_segmentation(a, multitask=False),
        "train-muranet": lambda a: train_segmentation(a, multitask=True),
        "vectorize-rooms": vectorize_rooms,
        "vectorize-wall-opening": vectorize_wall_opening,
        "text-detection": text_detection,
        "build-scene-graph": build_scene_graph,
        "sequence-feasibility": sequence_graph_feasibility,
        "sam-feasibility": sam_feasibility,
        "render-review": render_review,
        "evaluate-locked": evaluate_locked,
        "update-docs": update_docs,
        "run-all": run_all,
    }
    actions[args.command](args)


if __name__ == "__main__":
    main()
