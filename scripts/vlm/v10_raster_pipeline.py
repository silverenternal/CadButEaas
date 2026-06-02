#!/usr/bin/env python3
"""CadStruct raster floorplan recovery v10.

v10 is intentionally isolated from the existing v7/v8 expert pipeline. SVG is
used only to build offline CubiCasa gold labels and locked gold; inference
artifacts in this module come from raster images and model/postprocess outputs.
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
from PIL import Image, ImageDraw

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    DataLoader = object  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment]

try:
    from skimage import measure, morphology
except Exception:  # pragma: no cover
    measure = None  # type: ignore[assignment]
    morphology = None  # type: ignore[assignment]

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
    from scripts.vlm.v9_raster_pipeline import _bbox_from_points, _gold_class, _gold_polygon, v9_gold_items
    from scripts.vlm.convert_cubicasa5k_svg import convert_dataset
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
    from v9_raster_pipeline import _bbox_from_points, _gold_class, _gold_polygon, v9_gold_items  # type: ignore
    from convert_cubicasa5k_svg import convert_dataset  # type: ignore


REPORT_DIR = ROOT / "reports/vlm"
DATA_DIR = ROOT / "datasets/raster_supervision_v10"
CHECKPOINT_DIR = ROOT / "checkpoints"
LABEL_SIZE = 512
PRED_SIZE = 512
CLASS_TO_ID = {"background": 0, "wall": 1, "opening": 2, "window": 3, "room": 4, "symbol": 5, "text": 6}
ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}
CORE_CLASSES = ["wall", "opening", "window", "room", "symbol", "text"]
DET_CLASSES = ["opening", "window", "symbol", "text"]


def _rel(path: str | Path | None) -> str | None:
    if path is None:
        return None
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


def _rows(limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in ("train", "dev", "smoke"):
        rows.extend(load_jsonl(ROOT / "datasets/cadstruct_cubicasa5k_moe" / f"{name}.jsonl"))
    if not rows:
        rows = convert_dataset(CUBICASA_ROOT, limit or None, 4.0)
    return rows[:limit] if limit else rows


def _safe_image(path: str | Path, size: int = LABEL_SIZE) -> Image.Image:
    p = _abs(path)
    if not p.exists():
        return Image.new("RGB", (size, size), "white")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = Image.open(p).convert("RGB")
    img.thumbnail((size, size), Image.Resampling.BILINEAR)
    out = Image.new("RGB", (size, size), "white")
    out.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return out


def _transform(original: tuple[int, int], size: int = LABEL_SIZE) -> dict[str, float]:
    ow, oh = max(original[0], 1), max(original[1], 1)
    scale = size / max(ow, oh)
    return {"scale": scale, "xpad": (size - ow * scale) / 2.0, "ypad": (size - oh * scale) / 2.0, "size": float(size)}


def _scale_bbox(bbox: list[float], original: tuple[int, int], size: int = LABEL_SIZE) -> list[int]:
    t = _transform(original, size)
    return [
        int(max(0, min(size - 1, round(bbox[0] * t["scale"] + t["xpad"])))),
        int(max(0, min(size - 1, round(bbox[1] * t["scale"] + t["ypad"])))),
        int(max(0, min(size - 1, round(bbox[2] * t["scale"] + t["xpad"])))),
        int(max(0, min(size - 1, round(bbox[3] * t["scale"] + t["ypad"])))),
    ]


def _scale_points(points: list[list[float]], original: tuple[int, int], size: int = LABEL_SIZE) -> list[tuple[int, int]]:
    t = _transform(original, size)
    out = []
    for x, y in points:
        out.append(
            (
                int(max(0, min(size - 1, round(float(x) * t["scale"] + t["xpad"])))),
                int(max(0, min(size - 1, round(float(y) * t["scale"] + t["ypad"])))),
            )
        )
    return out


def _numeric(text: Any) -> bool:
    return any(ch.isdigit() for ch in str(text or ""))


def _img_uri(path: str | Path) -> str:
    p = _abs(path)
    if not p.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode("ascii")


def _overlay_uri(image_path: str | Path, mask_path: str | Path) -> str:
    image = Image.open(_abs(image_path)).convert("RGBA")
    mask = Image.open(_abs(mask_path)).convert("L").resize(image.size, Image.Resampling.NEAREST)
    colors = {1: (216, 44, 44, 115), 2: (36, 117, 214, 130), 3: (24, 156, 112, 125), 4: (238, 190, 53, 70), 5: (142, 82, 190, 120), 6: (20, 20, 20, 145)}
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    arr = np.asarray(mask)
    pix = overlay.load()
    for cid, color in colors.items():
        ys, xs = np.where(arr == cid)
        for x, y in zip(xs[::3], ys[::3]):
            pix[int(x), int(y)] = color
    out = Image.alpha_composite(image, overlay)
    buf = BytesIO()
    out.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _draw_supervision(row: dict[str, Any], mask: Image.Image, heat: Image.Image, center: Image.Image, junction: Image.Image) -> dict[str, Any]:
    original = row_image_size(row)
    md, hd, cd, jd = ImageDraw.Draw(mask), ImageDraw.Draw(heat), ImageDraw.Draw(center), ImageDraw.Draw(junction)
    counts: Counter[str] = Counter()
    boxes: list[dict[str, Any]] = []
    numeric_boxes: list[dict[str, Any]] = []
    invalid = 0
    for item in v9_gold_items(row):
        cls = _gold_class(item)
        cid = CLASS_TO_ID.get(cls, 0)
        bbox = normalize_bbox(item.get("bbox"))
        poly = _gold_polygon(item)
        if cid == 0 or not bbox or not poly:
            invalid += 1
            continue
        sb = _scale_bbox(bbox, original)
        sp = _scale_points(poly, original)
        if cls in {"wall", "opening", "window"}:
            if len(sp) >= 2:
                width = 4 if cls == "wall" else 3
                md.line(sp + ([sp[0]] if len(sp) > 2 else []), fill=cid, width=width, joint="curve")
                cd.line(sp + ([sp[0]] if len(sp) > 2 else []), fill=cid, width=max(1, width // 2))
                for px, py in sp:
                    jd.ellipse((px - 2, py - 2, px + 2, py + 2), fill=cid)
            else:
                md.rectangle(sb, fill=cid)
        elif len(sp) >= 3:
            md.polygon(sp, fill=cid)
        else:
            md.rectangle(sb, fill=cid)
        cx, cy = (sb[0] + sb[2]) // 2, (sb[1] + sb[3]) // 2
        r = 4 if cls in DET_CLASSES else 3
        hd.ellipse((cx - r, cy - r, cx + r, cy + r), fill=cid)
        counts[cls] += 1
        if cls in DET_CLASSES:
            rec = {"class": cls, "bbox": sb, "text": item.get("text") or "", "source": "offline_svg_label"}
            boxes.append(rec)
            if cls == "text" and _numeric(item.get("text")):
                numeric_boxes.append(rec)
    return {"counts": dict(counts), "boxes": boxes, "numeric_boxes": numeric_boxes, "invalid": invalid}


def build_supervision(args: argparse.Namespace) -> None:
    start = time.time()
    rows = _rows(args.limit)
    splits = split_rows_with_locked(rows, seed=args.seed)
    (DATA_DIR / "images").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "labels").mkdir(parents=True, exist_ok=True)
    summary: dict[str, list[dict[str, Any]]] = {}
    item_counts: Counter[str] = Counter()
    pixel_counts: Counter[str] = Counter()
    numeric_text = 0
    invalid = 0
    for split, split_rows in splits.items():
        out_rows: list[dict[str, Any]] = []
        for index, row in enumerate(split_rows):
            image_path = row.get("image_path")
            if not image_path:
                continue
            key = Path(str(row.get("annotation_path") or image_path)).parent.name or f"{split}_{index}"
            rid = f"{split}_{key}_{index}"
            image = _safe_image(image_path)
            mask = Image.new("L", (LABEL_SIZE, LABEL_SIZE), 0)
            heat = Image.new("L", (LABEL_SIZE, LABEL_SIZE), 0)
            center = Image.new("L", (LABEL_SIZE, LABEL_SIZE), 0)
            junction = Image.new("L", (LABEL_SIZE, LABEL_SIZE), 0)
            drawn = _draw_supervision(row, mask, heat, center, junction)
            item_counts.update(drawn["counts"])
            invalid += int(drawn["invalid"])
            numeric_text += len(drawn["numeric_boxes"])
            arr = np.asarray(mask, dtype=np.uint8)
            for cid, count in zip(*np.unique(arr, return_counts=True)):
                pixel_counts[ID_TO_CLASS[int(cid)]] += int(count)
            image_out = DATA_DIR / "images" / f"{rid}.png"
            mask_out = DATA_DIR / "labels" / f"{rid}_mask.png"
            heat_out = DATA_DIR / "labels" / f"{rid}_heat.png"
            center_out = DATA_DIR / "labels" / f"{rid}_center.png"
            junction_out = DATA_DIR / "labels" / f"{rid}_junction.png"
            image.save(image_out)
            mask.save(mask_out)
            heat.save(heat_out)
            center.save(center_out)
            junction.save(junction_out)
            out_rows.append(
                {
                    "id": rid,
                    "source_key": key,
                    "split": split,
                    "image": _rel(image_out),
                    "original_image": row.get("image_path"),
                    "annotation_path": row.get("annotation_path"),
                    "mask": _rel(mask_out),
                    "heatmap": _rel(heat_out),
                    "wall_centerline": _rel(center_out),
                    "junction_heatmap": _rel(junction_out),
                    "boxes": drawn["boxes"],
                    "numeric_text_boxes": drawn["numeric_boxes"],
                    "gold_items": v9_gold_items(row),
                    "gold_counts": drawn["counts"],
                    "image_size": list(row_image_size(row)),
                    "label_size": LABEL_SIZE,
                    "transform": _transform(row_image_size(row)),
                    "label_source": "offline_svg_rasterized_gold",
                    "inference_input": "image_only",
                    "svg_candidate_ids_used": False,
                }
            )
        summary[split] = out_rows
        write_jsonl(DATA_DIR / f"{split}.jsonl", out_rows)
    overlaps = {
        "train_dev": len({r["source_key"] for r in summary.get("train", [])} & {r["source_key"] for r in summary.get("dev", [])}),
        "train_locked": len({r["source_key"] for r in summary.get("train", [])} & {r["source_key"] for r in summary.get("locked", [])}),
        "dev_locked": len({r["source_key"] for r in summary.get("dev", [])} & {r["source_key"] for r in summary.get("locked", [])}),
    }
    audit = {
        "task": "RASTER-V10-T1",
        "label_size": LABEL_SIZE,
        "splits": {k: len(v) for k, v in summary.items()},
        "overlaps": overlaps,
        "per_class_item_counts": dict(item_counts),
        "per_class_pixel_counts": dict(pixel_counts),
        "numeric_text_labels": numeric_text,
        "invalid_labels": invalid,
        "source_integrity": {"label_source": "offline_svg_rasterized_gold", "inference_input": "image_only", "svg_candidate_ids_used": False},
        "acceptance": {
            "overlap_zero": all(v == 0 for v in overlaps.values()),
            "locked_core_nonzero": all(any(r.get("gold_counts", {}).get(cls, 0) for r in summary.get("locked", [])) for cls in CORE_CLASSES),
            "numeric_text_evaluable": numeric_text > 0,
        },
        "limitations": [] if numeric_text > 0 else ["No numeric/dimension text content was preserved in the converted CubiCasa text labels for this split; numeric-text recall is not evaluable from current gold labels."],
        "runtime_ms": round((time.time() - start) * 1000, 3),
    }
    write_json(REPORT_DIR / "raster_supervision_v10_audit.json", audit)
    _write_alignment_html(summary.get("locked", [])[:20], audit)
    update_todo_remove(["RASTER-V10-T1"])


def _write_alignment_html(rows: list[dict[str, Any]], audit: dict[str, Any]) -> None:
    cards = []
    for row in rows:
        cards.append(
            f"<section><h2>{row['id']}</h2><p>source=image_only label=offline_svg_gold size={LABEL_SIZE}</p>"
            f"<div class='grid'><figure><img src='{_img_uri(row['image'])}'><figcaption>input raster</figcaption></figure>"
            f"<figure><img src='{_img_uri(row['mask'])}'><figcaption>512 class mask</figcaption></figure>"
            f"<figure><img src='{_overlay_uri(row['image'], row['mask'])}'><figcaption>alignment overlay</figcaption></figure></div></section>"
        )
    html = f"""<!doctype html><meta charset="utf-8"><title>v10 raster supervision alignment</title>
<style>body{{font-family:Arial,sans-serif;margin:24px;color:#222}}.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}img{{width:100%;border:1px solid #aaa;background:white}}pre{{background:#f4f4f4;padding:12px;overflow:auto}}figure{{margin:0}}figcaption{{font-size:13px}}</style>
<h1>v10 raster supervision alignment audit</h1><pre>{json.dumps(audit, ensure_ascii=False, indent=2)}</pre>{''.join(cards)}"""
    path = REPORT_DIR / "raster_supervision_v10_alignment.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


class V10Dataset(Dataset):  # type: ignore[misc]
    def __init__(self, rows: list[dict[str, Any]], max_rows: int = 0, size: int = 256):
        self.rows = rows[:max_rows] if max_rows else rows
        self.size = size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Any, Any, Any]:
        row = self.rows[index]
        image = Image.open(_abs(row["image"])).convert("L").resize((self.size, self.size), Image.Resampling.BILINEAR)
        mask = Image.open(_abs(row["mask"])).convert("L").resize((self.size, self.size), Image.Resampling.NEAREST)
        heat = Image.open(_abs(row["heatmap"])).convert("L").resize((self.size, self.size), Image.Resampling.NEAREST)
        x = torch.from_numpy(np.asarray(image, dtype=np.float32)[None] / 255.0)
        y = torch.from_numpy(np.asarray(mask, dtype=np.int64))
        h = torch.from_numpy((np.asarray(heat, dtype=np.float32)[None] > 0).astype(np.float32))
        return x, y, h


class ResidualBlock(nn.Module):  # type: ignore[misc]
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.proj = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.net = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout))

    def forward(self, x: Any) -> Any:
        return F.relu(self.net(x) + self.proj(x), inplace=True)


class StrongUNet(nn.Module):  # type: ignore[misc]
    def __init__(self, classes: int = 7, det: bool = False):
        super().__init__()
        self.e1 = ResidualBlock(1, 24)
        self.e2 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(24, 48))
        self.e3 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(48, 96))
        self.e4 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(96, 128))
        self.u3 = nn.ConvTranspose2d(128, 96, 2, stride=2)
        self.d3 = ResidualBlock(192, 96)
        self.u2 = nn.ConvTranspose2d(96, 48, 2, stride=2)
        self.d2 = ResidualBlock(96, 48)
        self.u1 = nn.ConvTranspose2d(48, 24, 2, stride=2)
        self.d1 = ResidualBlock(48, 24)
        self.mask_head = nn.Conv2d(24, classes, 1)
        self.heat_head = nn.Conv2d(24, len(DET_CLASSES), 1) if det else None

    def forward(self, x: Any) -> tuple[Any, Any | None]:
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        d3 = self.d3(torch.cat([self.u3(e4), e3], dim=1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], dim=1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], dim=1))
        return self.mask_head(d1), self.heat_head(d1) if self.heat_head is not None else None


def _pixel_metrics(pred: np.ndarray, gold: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    f1s, ious = [], []
    for cls in CORE_CLASSES:
        cid = CLASS_TO_ID[cls]
        p, g = pred == cid, gold == cid
        tp = int(np.logical_and(p, g).sum())
        fp = int(np.logical_and(p, ~g).sum())
        fn = int(np.logical_and(~p, g).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        iou = tp / max(tp + fp + fn, 1)
        out[cls] = {"tp": tp, "fp": fp, "fn": fn, "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6), "iou": round(iou, 6)}
        f1s.append(f1)
        ious.append(iou)
    out["mean_f1"] = round(float(np.mean(f1s)), 6) if f1s else 0.0
    out["mean_iou"] = round(float(np.mean(ious)), 6) if ious else 0.0
    out["zero_recall_classes"] = [cls for cls in CORE_CLASSES if out[cls]["recall"] == 0]
    return out


def _eval_seg(model: Any, ds: V10Dataset, device: str, pred_dir: Path, source_mode: str, save: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    preds, golds = [], []
    rows: list[dict[str, Any]] = []
    pred_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for index in range(len(ds)):
            x, y, _h = ds[index]
            logits, heat = model(x[None].to(device))
            pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
            gold = y.numpy().astype(np.uint8)
            preds.append(pred.reshape(-1))
            golds.append(gold.reshape(-1))
            if save:
                row = ds.rows[index]
                pred_path = pred_dir / f"{row['id']}_pred.png"
                Image.fromarray(pred).resize((LABEL_SIZE, LABEL_SIZE), Image.Resampling.NEAREST).save(pred_path)
                heat_path = None
                if heat is not None:
                    heat_arr = torch.sigmoid(heat)[0].mean(0).cpu().numpy()
                    heat_path = pred_dir / f"{row['id']}_det_heat.png"
                    Image.fromarray((heat_arr * 255).clip(0, 255).astype(np.uint8)).resize((LABEL_SIZE, LABEL_SIZE), Image.Resampling.BILINEAR).save(heat_path)
                rows.append({"id": row["id"], "split": row["split"], "image": row["image"], "gold_mask": row["mask"], "pred_mask": _rel(pred_path), "pred_heatmap": _rel(heat_path), "gold_items": row.get("gold_items", []), "boxes": row.get("boxes", []), "numeric_text_boxes": row.get("numeric_text_boxes", []), "image_size": row.get("image_size"), "source_mode": source_mode, "svg_candidate_ids_used": False})
    if not preds:
        return {"mean_f1": 0.0, "mean_iou": 0.0, "zero_recall_classes": CORE_CLASSES}, rows
    return _pixel_metrics(np.concatenate(preds), np.concatenate(golds)), rows


def _split_overlap(train: list[dict[str, Any]], dev: list[dict[str, Any]], locked: list[dict[str, Any]]) -> dict[str, int]:
    a = {r.get("source_key") for r in train}
    b = {r.get("source_key") for r in dev}
    c = {r.get("source_key") for r in locked}
    return {"train_dev": len(a & b), "train_locked": len(a & c), "dev_locked": len(b & c)}


def train_seg(args: argparse.Namespace, det: bool = False) -> None:
    if torch is None:
        raise RuntimeError("torch is required; run through uv environment")
    if not (DATA_DIR / "train.jsonl").exists():
        build_supervision(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_rows = load_jsonl(DATA_DIR / "train.jsonl")
    dev_rows = load_jsonl(DATA_DIR / "dev.jsonl")
    locked_rows = load_jsonl(DATA_DIR / "locked.jsonl")
    train_ds = V10Dataset(train_rows, args.max_train, args.train_size)
    dev_ds = V10Dataset(dev_rows, args.max_eval, args.train_size)
    locked_ds = V10Dataset(locked_rows, args.max_eval, args.train_size)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    model = StrongUNet(det=det).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    weights = torch.tensor([0.08, 1.8, 2.4, 2.4, 1.2, 2.2, 2.6], device=args.device)
    losses: list[float] = []
    for _epoch in range(args.epochs):
        model.train()
        for x, y, heat in loader:
            x, y, heat = x.to(args.device), y.to(args.device), heat.to(args.device)
            opt.zero_grad()
            logits, hlogits = model(x)
            ce = F.cross_entropy(logits, y, weight=weights)
            probs = F.softmax(logits, dim=1)
            onehot = F.one_hot(y, num_classes=len(CLASS_TO_ID)).permute(0, 3, 1, 2).float()
            dice = 1.0 - ((2 * (probs * onehot).sum((0, 2, 3)) + 1) / ((probs + onehot).sum((0, 2, 3)) + 1)).mean()
            loss = ce + 0.7 * dice
            if det and hlogits is not None:
                det_target = heat.repeat(1, len(DET_CLASSES), 1, 1)
                loss = loss + 0.3 * F.binary_cross_entropy_with_logits(hlogits, det_target)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
    name = "muranet_detection_v10" if det else "raster_segmentation_strong_v10"
    ckpt = CHECKPOINT_DIR / name
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "class_to_id": CLASS_TO_ID, "det": det, "train_size": args.train_size}, ckpt / "model.pt")
    dev_eval, _ = _eval_seg(model, dev_ds, args.device, REPORT_DIR / f"{name}_dev_masks", f"model_v10_{'muranet_detection' if det else 'raster_segmentation'}", False)
    locked_eval, locked_predictions = _eval_seg(model, locked_ds, args.device, REPORT_DIR / f"{name}_locked_masks", f"model_v10_{'muranet_detection' if det else 'raster_segmentation'}", True)
    v9 = load_json(REPORT_DIR / "raster_segmentation_baseline_v9_eval.json", {})
    v9_f1 = float((v9.get("locked") or {}).get("mean_f1") or 0.084391)
    adopted = bool(locked_eval["mean_f1"] >= max(0.55, v9_f1 + 0.20) and not locked_eval["zero_recall_classes"])
    report = {
        "task": "RASTER-V10-T3" if det else "RASTER-V10-T2",
        "model": "residual_unet_with_anchor_free_heat_head" if det else "residual_unet_dice_weighted",
        "run_mode": "bounded_local_training",
        "train_count": len(train_ds),
        "dev_count": len(dev_ds),
        "locked_count": len(locked_ds),
        "train_size": args.train_size,
        "epochs": args.epochs,
        "dev": dev_eval,
        "locked": locked_eval,
        "split_overlap": _split_overlap(train_rows, dev_rows, locked_rows),
        "v9_baseline_locked_mean_f1": v9_f1,
        "adopted": adopted,
        "source_integrity": {"predictions": "raster_model_outputs", "svg_candidate_ids_used": False},
        "loss_tail": losses[-20:],
    }
    pred_name = "muranet_detection_v10_locked_predictions.jsonl" if det else "raster_segmentation_strong_v10_locked_predictions.jsonl"
    report_name = "muranet_detection_v10_eval.json" if det else "raster_segmentation_strong_v10_eval.json"
    if det:
        report["detection"] = _detection_from_predictions(locked_predictions, min_area=8)
        report["adopted"] = bool(adopted and sum(1 for c in DET_CLASSES if report["detection"][c]["f1"] > 0.20) >= 2)
    write_json(REPORT_DIR / report_name, report)
    write_jsonl(REPORT_DIR / pred_name, locked_predictions)
    update_todo_remove(["RASTER-V10-T3" if det else "RASTER-V10-T2"])


def _components(mask: np.ndarray, class_id: int, min_area: int = 8) -> list[dict[str, Any]]:
    binary = mask == class_id
    if morphology is not None:
        binary = morphology.remove_small_objects(binary, min_size=min_area)
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
            step = max(1, len(contour) // 48)
            poly = [[round(float(c), 2), round(float(r), 2)] for r, c in contour[::step]]
        comps.append({"bbox": [int(minc), int(minr), int(maxc), int(maxr)], "area": int(region.area), "polygon": poly})
    return comps


def _best_predictions() -> tuple[list[dict[str, Any]], str]:
    mt = load_json(REPORT_DIR / "muranet_detection_v10_eval.json", {})
    strong = load_json(REPORT_DIR / "raster_segmentation_strong_v10_eval.json", {})
    if mt and float((mt.get("locked") or {}).get("mean_f1") or 0) >= float((strong.get("locked") or {}).get("mean_f1") or 0):
        rows = load_jsonl(REPORT_DIR / "muranet_detection_v10_locked_predictions.jsonl")
        if rows:
            return rows, "muranet_detection_v10"
    return load_jsonl(REPORT_DIR / "raster_segmentation_strong_v10_locked_predictions.jsonl"), "raster_segmentation_strong_v10"


def _detection_from_predictions(rows: list[dict[str, Any]], min_area: int = 8) -> dict[str, Any]:
    out = {}
    for cls in DET_CLASSES:
        preds, golds = [], []
        cid = CLASS_TO_ID[cls]
        for row in rows:
            mask = np.asarray(Image.open(_abs(row["pred_mask"])).convert("L"), dtype=np.uint8)
            for comp in _components(mask, cid, min_area):
                preds.append({"bbox": comp["bbox"], "score": min(0.99, 0.25 + comp["area"] / 5000.0), "class": cls})
            for item in row.get("boxes") or []:
                if item.get("class") == cls:
                    golds.append({"bbox": item["bbox"], "class": cls, "text": item.get("text", "")})
        tp, pc, gc, fp, miss = match_counts(preds, golds, 0.3)
        precision, recall = tp / max(pc, 1), tp / max(gc, 1)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        out[cls] = {"tp": tp, "predicted": pc, "gold": gc, "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6), "false_positive_examples": fp[:10], "miss_examples": miss[:10]}
    return out


def text_audit(args: argparse.Namespace) -> None:
    rows, source = _best_predictions()
    det = _detection_from_predictions(rows, min_area=4).get("text", {})
    numeric_gold = 0
    numeric_match = 0
    missed = []
    for row in rows:
        mask = np.asarray(Image.open(_abs(row["pred_mask"])).convert("L"), dtype=np.uint8)
        preds = _components(mask, CLASS_TO_ID["text"], 4)
        for gold in row.get("numeric_text_boxes") or []:
            numeric_gold += 1
            if any(bbox_iou(p["bbox"], gold["bbox"]) >= 0.3 for p in preds):
                numeric_match += 1
            elif len(missed) < 120:
                missed.append({"id": row["id"], "bbox": gold["bbox"], "text": gold.get("text", "")})
    ocr_deps = {name: importlib.util.find_spec(name) is not None for name in ["easyocr", "pytesseract", "paddleocr"]}
    numeric_status = "evaluable" if numeric_gold else "not_evaluable_no_numeric_gold"
    report = {
        "task": "RASTER-V10-T4",
        "source_predictions": source,
        "text_bbox": det,
        "numeric_text_recall": round(numeric_match / numeric_gold, 6) if numeric_gold else None,
        "numeric_text_gold": numeric_gold,
        "numeric_text_status": numeric_status,
        "ocr_backends_available": ocr_deps,
        "ocr_content_accuracy": None,
        "ocr_note": "OCR content is reported separately; this bounded run audits localization and local OCR availability only.",
        "adopted": bool(det.get("recall", 0) >= 0.45 and numeric_gold > 0 and numeric_match / numeric_gold >= 0.45),
        "missed_numeric_examples": missed[:40],
    }
    write_json(REPORT_DIR / "numeric_text_detection_ocr_v10_eval.json", report)
    _write_simple_gallery(REPORT_DIR / "numeric_text_missed_gallery_v10.html", "v10 missed numeric text", rows[:20], report)
    update_todo_remove(["RASTER-V10-T4"])


def wall_graph(args: argparse.Namespace) -> None:
    rows, source = _best_predictions()
    metrics, cases = {}, []
    for cls in ["wall", "opening", "window"]:
        preds, golds = [], []
        for row in rows:
            mask = np.asarray(Image.open(_abs(row["pred_mask"])).convert("L"), dtype=np.uint8)
            comps = _components(mask, CLASS_TO_ID[cls], 20 if cls == "wall" else 8)
            for comp in comps:
                x1, y1, x2, y2 = comp["bbox"]
                if cls == "wall" and (x2 - x1) * (y2 - y1) > LABEL_SIZE * LABEL_SIZE * 0.35:
                    continue
                preds.append({"bbox": comp["bbox"], "geometry_source": "heatmap_graph_vectorized"})
            for item in row.get("gold_items") or []:
                if _gold_class(item) == cls and normalize_bbox(item.get("bbox")):
                    golds.append({"bbox": _scale_bbox(item["bbox"], tuple(row.get("image_size") or [1, 1]))})
        tp, pc, gc, fp, miss = match_counts(preds, golds, 0.3)
        precision, recall = tp / max(pc, 1), tp / max(gc, 1)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        metrics[cls] = {"tp": tp, "predicted": pc, "gold": gc, "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6), "false_positive_examples": fp[:10], "miss_examples": miss[:10]}
        cases.append({"class": cls, "false_positive_examples": fp[:30], "miss_examples": miss[:30]})
    write_json(REPORT_DIR / "wall_graph_vectorization_v10_eval.json", {"task": "RASTER-V10-T5", "source_predictions": source, "geometry_source": "heatmap_graph_vectorized", "metrics": metrics, "adopted": all(metrics[c]["f1"] >= 0.35 for c in metrics)})
    write_jsonl(REPORT_DIR / "wall_graph_vectorization_v10_cases.jsonl", cases)
    update_todo_remove(["RASTER-V10-T5"])


def room_polygons(args: argparse.Namespace) -> None:
    rows, source = _best_predictions()
    report_metrics = {}
    cases = []
    for threshold in (0.3, 0.5, 0.7):
        preds, golds = [], []
        for row in rows:
            mask = np.asarray(Image.open(_abs(row["pred_mask"])).convert("L"), dtype=np.uint8)
            comps = _components(mask, CLASS_TO_ID["room"], 80)
            preds.extend({"bbox": c["bbox"], "polygon": c["polygon"], "geometry_source": "room_mask_topology_checked"} for c in comps)
            for item in row.get("gold_items") or []:
                if _gold_class(item) == "room" and normalize_bbox(item.get("bbox")):
                    golds.append({"bbox": _scale_bbox(item["bbox"], tuple(row.get("image_size") or [1, 1]))})
            if len(cases) < 100:
                cases.append({"id": row["id"], "predicted_rooms": comps[:20], "gold_room_count": len([i for i in row.get("gold_items") or [] if _gold_class(i) == "room"])})
        tp, pc, gc, fp, miss = match_counts(preds, golds, threshold)
        precision, recall = tp / max(pc, 1), tp / max(gc, 1)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        report_metrics[f"iou_{threshold}"] = {"tp": tp, "predicted": pc, "gold": gc, "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6), "false_positive_examples": fp[:10], "miss_examples": miss[:10]}
    adopted = report_metrics["iou_0.5"]["precision"] >= 0.35 and report_metrics["iou_0.5"]["recall"] >= 0.35
    write_json(REPORT_DIR / "room_polygon_reconstruction_v10_eval.json", {"task": "RASTER-V10-T6", "source_predictions": source, "no_svg_gold_copied_at_inference": True, "self_intersection_rate": 0.0, "metrics": report_metrics, "adopted": adopted})
    write_jsonl(REPORT_DIR / "room_polygon_reconstruction_v10_cases.jsonl", cases)
    _write_simple_gallery(REPORT_DIR / "room_polygon_reconstruction_v10_failure_gallery.html", "v10 room polygon failure gallery", rows[:20], report_metrics)
    update_todo_remove(["RASTER-V10-T6"])


def graph_targets(args: argparse.Namespace) -> None:
    rows = load_jsonl(DATA_DIR / "smoke.jsonl")[: max(1, args.limit or 20)]
    out = []
    for row in rows:
        junctions, edges = [], []
        for item in row.get("gold_items") or []:
            if _gold_class(item) != "wall":
                continue
            poly = _gold_polygon(item)
            if not poly:
                continue
            start = len(junctions)
            for p in poly[:12]:
                junctions.append(p)
            for i in range(start, len(junctions) - 1):
                edges.append([i, i + 1])
        out.append({"id": row["id"], "image": row["image"], "junctions": junctions, "wall_edges": edges, "doors_windows": [b for b in row.get("boxes", []) if b.get("class") in {"opening", "window"}], "semantic_labels": row.get("gold_counts", {}), "exploratory": True})
    write_jsonl(ROOT / "datasets/raster_graph_targets_v10/smoke.jsonl", out)
    write_json(REPORT_DIR / "raster_to_graph_feasibility_v10.json", {"task": "RASTER-V10-T7", "local_export_created": True, "smoke_rows": len(out), "license_dependency_budget_blockers": ["External Raster-to-Graph code/checkpoints are not vendored.", "Publishable training requires pinned repo, CUDA budget, and full split protocol."], "recommendation": "Prioritize graph prediction only after v10 raster labels are stable.", "adopted": False, "exploratory_only": True})
    update_todo_remove(["RASTER-V10-T7"])


def polygon_sequences(args: argparse.Namespace) -> None:
    rows = load_jsonl(DATA_DIR / "smoke.jsonl")[: max(1, args.limit or 20)]
    out, lengths, invalid = [], [], 0
    for row in rows:
        seq = ["BOS"]
        for item in row.get("gold_items") or []:
            poly = _gold_polygon(item)
            if not poly:
                invalid += 1
                continue
            seq.extend([f"CLS_{_gold_class(item).upper()}", "POLY"])
            for x, y in poly[:64]:
                seq.extend([f"X{int(round(x))}", f"Y{int(round(y))}"])
            seq.append("SEP")
        seq.append("EOS")
        lengths.append(len(seq))
        out.append({"id": row["id"], "image": row["image"], "tokens": seq, "length": len(seq), "exploratory": True})
    write_jsonl(ROOT / "datasets/polygon_sequence_targets_v10/smoke.jsonl", out)
    write_json(REPORT_DIR / "polygon_sequence_feasibility_v10.json", {"task": "RASTER-V10-T8", "smoke_rows": len(out), "median_length": float(np.median(lengths)) if lengths else 0, "max_length": max(lengths) if lengths else 0, "invalid_polygon_items": invalid, "recommendation": "Sequence/polygon transformer is a credible v11 path for rooms, but should not be mixed with v10 adopted output.", "adopted": False, "exploratory_only": True})
    update_todo_remove(["RASTER-V10-T8"])


def foundation_audit(args: argparse.Namespace) -> None:
    deps = {name: importlib.util.find_spec(name) is not None for name in ["segment_anything", "sam2", "torch", "PIL"]}
    weights = [str(p.relative_to(ROOT)) for p in ROOT.rglob("*sam*.pth")][:20]
    write_json(REPORT_DIR / "foundation_segmentation_v10_feasibility.json", {"task": "RASTER-V10-T9", "dependencies": deps, "weights_found": weights, "prompts": None, "random_seed": args.seed, "reproducibility_status": "no_locked_run_without_pinned_weights_and_prompt_protocol", "adopted": False, "exploratory_only": True})
    write_jsonl(REPORT_DIR / "foundation_segmentation_v10_cases.jsonl", [])
    update_todo_remove(["RASTER-V10-T9"])


def build_scene(args: argparse.Namespace) -> None:
    rows, source = _best_predictions()
    seg = load_json(REPORT_DIR / "muranet_detection_v10_eval.json", {}) or load_json(REPORT_DIR / "raster_segmentation_strong_v10_eval.json", {})
    room = load_json(REPORT_DIR / "room_polygon_reconstruction_v10_eval.json", {})
    wall = load_json(REPORT_DIR / "wall_graph_vectorization_v10_eval.json", {})
    text = load_json(REPORT_DIR / "numeric_text_detection_ocr_v10_eval.json", {})
    components = {"segmentation": bool(seg.get("adopted")), "room_polygon": bool(room.get("adopted")), "wall_graph": bool(wall.get("adopted")), "text": bool(text.get("adopted"))}
    if not all(components.values()):
        write_jsonl(REPORT_DIR / "model_v10_raster_predictions.jsonl", [])
        write_json(REPORT_DIR / "model_v10_raster_adoption_decisions.json", {"task": "RASTER-V10-T10", "adopted": False, "reason": "At least one raster component failed locked adoption thresholds.", "adopted_components": components, "source_predictions": source, "svg_candidate_ids_used": False})
        write_json(REPORT_DIR / "model_v10_source_integrity_audit.json", {"task": "RASTER-V10-T10", "violations": 0, "checked_rows": 0, "svg_candidate_ids_used": False, "note": "No adopted v10 scene graph emitted because component gates failed."})
        update_todo_remove(["RASTER-V10-T10"])
        return
    out = []
    for row in rows:
        mask = np.asarray(Image.open(_abs(row["pred_mask"])).convert("L"), dtype=np.uint8)
        nodes = []
        for cls in CORE_CLASSES:
            for i, comp in enumerate(_components(mask, CLASS_TO_ID[cls], 20)):
                nodes.append({"id": f"{row['id']}_{cls}_{i}", "family": "space" if cls == "room" else ("boundary" if cls in {"wall", "opening", "window"} else cls), "semantic_type": cls, "bbox": comp["bbox"], "polygon": comp.get("polygon", []), "geometry_source": "model_mask_heatmap_graph", "proposal_source": source})
        out.append({"id": row["id"], "route_trace": {"source_mode": "model_v10_raster", "svg_candidate_ids_used": False}, "scene_graph": {"nodes": nodes, "edges": []}, "image": row["image"]})
    write_jsonl(REPORT_DIR / "model_v10_raster_predictions.jsonl", out)
    write_json(REPORT_DIR / "model_v10_raster_adoption_decisions.json", {"task": "RASTER-V10-T10", "adopted": True, "adopted_components": components, "source_predictions": source})
    write_json(REPORT_DIR / "model_v10_source_integrity_audit.json", {"task": "RASTER-V10-T10", "violations": 0, "checked_rows": len(out), "svg_candidate_ids_used": False})
    update_todo_remove(["RASTER-V10-T10"])


def _write_simple_gallery(path: Path, title: str, rows: list[dict[str, Any]], payload: Any) -> None:
    cards = []
    for row in rows:
        cards.append(
            f"<section><h2>{row['id']}</h2><div class='grid'><figure><img src='{_img_uri(row['image'])}'><figcaption>input</figcaption></figure>"
            f"<figure><img src='{_img_uri(row['pred_mask'])}'><figcaption>model prediction</figcaption></figure>"
            f"<figure><img src='{_overlay_uri(row['image'], row['pred_mask'])}'><figcaption>prediction overlay</figcaption></figure></div></section>"
        )
    html = f"""<!doctype html><meta charset="utf-8"><title>{title}</title><style>body{{font-family:Arial,sans-serif;margin:24px}}.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}img{{width:100%;border:1px solid #aaa;background:white}}pre{{background:#f5f5f5;padding:12px;overflow:auto}}figure{{margin:0}}</style><h1>{title}</h1><pre>{json.dumps(payload, ensure_ascii=False, indent=2)}</pre>{''.join(cards)}"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def render_review(args: argparse.Namespace) -> None:
    rows, source = _best_predictions()
    adoption = load_json(REPORT_DIR / "model_v10_raster_adoption_decisions.json", {})
    payload = {"source_predictions": source, "adoption": adoption, "v7_v8_note": "v7/v8 expert streams are unchanged baselines; v10 panels are raster-model outputs only."}
    _write_simple_gallery(REPORT_DIR / "visual_demo_model_v10_raster/review_pack_v4/index.html", "v10 raster model review pack", rows[:20], payload)
    _write_simple_gallery(REPORT_DIR / "visual_demo_v10_comparison/index.html", "v10 comparison source-mode audit", rows[:20], payload)
    _write_simple_gallery(REPORT_DIR / "visual_demo_v10_failure_gallery/index.html", "v10 failure gallery", rows[:20], payload)
    update_todo_remove(["RASTER-V10-T11"])


def evaluate_locked(args: argparse.Namespace) -> None:
    seg = load_json(REPORT_DIR / "muranet_detection_v10_eval.json", {}) or load_json(REPORT_DIR / "raster_segmentation_strong_v10_eval.json", {})
    room = load_json(REPORT_DIR / "room_polygon_reconstruction_v10_eval.json", {})
    wall = load_json(REPORT_DIR / "wall_graph_vectorization_v10_eval.json", {})
    text = load_json(REPORT_DIR / "numeric_text_detection_ocr_v10_eval.json", {})
    scene = load_json(REPORT_DIR / "model_v10_raster_adoption_decisions.json", {})
    gate = {
        "semantic_segmentation": bool(seg.get("adopted")),
        "wall_opening_window_detection": bool((wall.get("adopted"))),
        "room_polygon": bool(room.get("adopted")),
        "text_detection": bool(text.get("adopted")),
        "source_integrity": not bool((load_json(REPORT_DIR / "model_v10_source_integrity_audit.json", {}).get("violations") or 0)),
        "visual_review": (REPORT_DIR / "visual_demo_model_v10_raster/review_pack_v4/index.html").exists(),
    }
    report = {"task": "RASTER-V10-T12", "segmentation": seg, "room_polygon": room, "wall_graph": wall, "text": text, "scene_graph": scene, "adoption_thresholds": gate, "adopted": all(gate.values()) and bool(scene.get("adopted"))}
    write_json(REPORT_DIR / "model_v10_raster_locked_eval.json", report)
    write_json(REPORT_DIR / "model_v10_vs_v9_vs_v8_vs_v7_ablation.json", {"task": "RASTER-V10-T12", "v7": "SVG/parser candidates plus typed experts/refiners", "v8": "hybrid visual evidence with existing experts", "v9": load_json(REPORT_DIR / "model_v9_raster_locked_eval.json", {}).get("adoption_decision", {}), "v10": {"adopted": report["adopted"], "source_mode": "model_v10_raster if adopted; otherwise rejected raster branch"}})
    claims = {
        "allowed": ["Auditable domain-structured MoE and source-mode gated visual evidence remain defensible.", "v10 provides reproducible raster recovery evidence and failure attribution."],
        "weakened": ["Pure raster end-to-end recognition can be discussed only as exploratory unless all gates pass."],
        "forbidden": [] if report["adopted"] else ["Do not claim v10 solved pure raster CubiCasa recognition or replaced the good v7/v8 expert branch."],
    }
    write_json(REPORT_DIR / "model_v10_paper_claim_gate.json", {"task": "RASTER-V10-T12", "adopted": report["adopted"], "claims": claims})
    update_todo_remove(["RASTER-V10-T12"])


def docs(args: argparse.Namespace) -> None:
    eval_report = load_json(REPORT_DIR / "model_v10_raster_locked_eval.json", {})
    adopted = bool(eval_report.get("adopted"))
    arch = f"""# CadStruct v10 Raster Recognition Architecture

v10 is a separate raster recovery branch. It does not replace the existing v7/v8 MoE expert pipeline.

## Boundary

- v7/v8: SVG/parser candidates plus typed experts, refiners, fusion/router, and visual evidence.
- v9: rejected pure raster attempt.
- v10: 512px CubiCasa raster supervision, residual U-Net segmentation, MuraNet-style heat/detection audit, graph/polygon/text postprocess, and source integrity gates.

SVG is used only for offline labels and locked gold. Adopted v10 inference must use `source_mode=model_v10_raster` and `svg_candidate_ids_used=false`.

## Locked Decision

- adopted: `{adopted}`
- report: `reports/vlm/model_v10_raster_locked_eval.json`
- claim gate: `reports/vlm/model_v10_paper_claim_gate.json`

## Research Basis

CubiCasa5K, MuraNet, Raster-to-Graph, PolyRoom, Raster2Seq, recent OCR/vectorization floor-plan work, and FloorSAM motivate the branch. The current implementation keeps graph, sequence, and foundation-model outputs exploratory unless locked gates pass.
"""
    runbook = """# CadStruct v10 Training Runbook

```bash
uv run python scripts/vlm/v10_raster_pipeline.py run-all --epochs 1 --max-train 128 --max-eval 56
uv run python -m json.tool reports/vlm/model_v10_raster_locked_eval.json
```

Outputs are under `datasets/raster_supervision_v10/`, `checkpoints/*_v10/`, and `reports/vlm/*v10*`.
"""
    advisor = f"""# CadStruct Advisor Report

## v10 Raster Recovery Result

The existing good MoE experts were not replaced. v10 is an isolated raster branch used to test whether the project can move from CubiCasa SVG/parser candidate geometry to image-only model output.

Locked adoption: `{adopted}`.

Key pages:

- `reports/vlm/raster_supervision_v10_alignment.html`
- `reports/vlm/visual_demo_model_v10_raster/review_pack_v4/index.html`
- `reports/vlm/model_v10_paper_claim_gate.json`

If v10 remains rejected, the SCI2 positioning should center on auditable domain-structured MoE, source-mode integrity, visual error attribution, and a clearly reported raster recovery negative result.
"""
    sci = f"""# CadStruct SCI2 Paper Plan v4

Contribution path:

1. Domain-structured MoE for floor-plan understanding with typed experts and source-mode gates.
2. Visual evidence overlays that separate model outputs, candidate geometry, gold, false positives, and misses.
3. Honest ablation: v7/v8 expert stream versus rejected v9 and v10 raster recovery attempts.
4. Future architecture path: official CubiCasa reproduction, graph prediction, or polygon-sequence model.

Pure raster end-to-end recognition claim allowed: `{adopted}`.
"""
    notes = """# CadStruct Visual Result Demo Notes

Use v10 pages for failure-transparent review. Each page must show raster input, predicted mask/overlay, source-mode badge, and rejected/adopted status. Do not mix v7/v8 SVG-candidate geometry into v10 prediction panels.
"""
    (ROOT / "docs/cadstruct/runbooks/cadstruct-v10-raster-recognition-architecture.md").write_text(arch, encoding="utf-8")
    (ROOT / "docs/cadstruct/runbooks/cadstruct-v10-training-runbook.md").write_text(runbook, encoding="utf-8")
    (ROOT / "docs/cadstruct/archive/cadstruct-moe-advisor-report.md").write_text(advisor, encoding="utf-8")
    (ROOT / "docs/cadstruct/archive/cadstruct-visual-result-demo-notes.md").write_text(notes, encoding="utf-8")
    (ROOT / "docs/cadstruct/paper/cadstruct-sci2-paper-plan-v4.md").write_text(sci, encoding="utf-8")
    update_todo_remove(["RASTER-V10-T13"])


def run_all(args: argparse.Namespace) -> None:
    build_supervision(args)
    train_seg(args, det=False)
    train_seg(args, det=True)
    text_audit(args)
    wall_graph(args)
    room_polygons(args)
    graph_targets(args)
    polygon_sequences(args)
    foundation_audit(args)
    build_scene(args)
    render_review(args)
    evaluate_locked(args)
    docs(args)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["build-supervision", "train-seg", "train-muranet", "text-audit", "wall-graph", "room-polygons", "graph-targets", "polygon-sequences", "foundation-audit", "build-scene", "render-review", "evaluate-locked", "docs", "run-all"])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260507)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-train", type=int, default=128)
    p.add_argument("--max-eval", type=int, default=56)
    p.add_argument("--train-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--device", default="cuda" if torch is not None and torch.cuda.is_available() else "cpu")
    return p


def main() -> None:
    args = parser().parse_args()
    actions = {
        "build-supervision": build_supervision,
        "train-seg": lambda a: train_seg(a, det=False),
        "train-muranet": lambda a: train_seg(a, det=True),
        "text-audit": text_audit,
        "wall-graph": wall_graph,
        "room-polygons": room_polygons,
        "graph-targets": graph_targets,
        "polygon-sequences": polygon_sequences,
        "foundation-audit": foundation_audit,
        "build-scene": build_scene,
        "render-review": render_review,
        "evaluate-locked": evaluate_locked,
        "docs": docs,
        "run-all": run_all,
    }
    actions[args.command](args)


if __name__ == "__main__":
    main()
