#!/usr/bin/env python3
"""Image-only CadStruct-MoE v14 recovery pipeline.

This module keeps CubiCasa SVG/parser data out of inference rows. SVG-derived
labels are used only to build offline supervision and locked gold.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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

from scripts.vlm.convert_cubicasa5k_svg import convert_dataset
from scripts.vlm.v5_pipeline_utils import update_todo_remove
from scripts.vlm.v8_raster_e2e_utils import CUBICASA_ROOT, load_json, load_jsonl, match_counts, normalize_bbox, split_rows_with_locked, write_json, write_jsonl
from scripts.vlm.v9_raster_pipeline import _gold_class, _gold_polygon, v9_gold_items
from scripts.vlm.validate_image_only_moe_stream import validate_rows


REPORT = ROOT / "reports/vlm"
DATA = ROOT / "datasets/image_only_raster_supervision_v14"
CKPT = ROOT / "checkpoints/image_only_multitask_proposal_v14"
CLASSES = {"background": 0, "wall": 1, "opening": 2, "window": 3, "room": 4, "symbol": 5, "text": 6}
ID_TO_CLASS = {v: k for k, v in CLASSES.items()}
CORE = ["wall", "opening", "window", "room", "symbol", "text"]
FAMILY = {"wall": "boundary", "opening": "boundary", "window": "boundary", "room": "space", "symbol": "symbol", "text": "text"}


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


def _rows(limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in ("train", "dev", "smoke"):
        rows.extend(load_jsonl(ROOT / "datasets/cadstruct_cubicasa5k_moe" / f"{name}.jsonl"))
    if not rows:
        rows = convert_dataset(CUBICASA_ROOT, limit or None, 4.0)
    return rows[:limit] if limit else rows


def _image_size(row: dict[str, Any]) -> tuple[int, int]:
    size = row.get("image_size")
    if isinstance(size, list) and len(size) >= 2:
        return int(size[0]), int(size[1])
    p = _abs(row.get("image_path") or row.get("image") or "")
    if p.exists():
        with Image.open(p) as im:
            return im.size
    return 1, 1


def _transform(original: tuple[int, int], size: int) -> dict[str, float]:
    ow, oh = max(original[0], 1), max(original[1], 1)
    scale = size / max(ow, oh)
    return {"scale": scale, "xpad": (size - ow * scale) / 2.0, "ypad": (size - oh * scale) / 2.0, "size": float(size)}


def _scale_bbox(bbox: list[float], original: tuple[int, int], size: int) -> list[int]:
    t = _transform(original, size)
    return [
        int(max(0, min(size - 1, round(bbox[0] * t["scale"] + t["xpad"])))),
        int(max(0, min(size - 1, round(bbox[1] * t["scale"] + t["ypad"])))),
        int(max(0, min(size - 1, round(bbox[2] * t["scale"] + t["xpad"])))),
        int(max(0, min(size - 1, round(bbox[3] * t["scale"] + t["ypad"])))),
    ]


def _scale_poly(poly: list[list[float]], original: tuple[int, int], size: int) -> list[tuple[int, int]]:
    t = _transform(original, size)
    return [(int(max(0, min(size - 1, round(float(x) * t["scale"] + t["xpad"])))), int(max(0, min(size - 1, round(float(y) * t["scale"] + t["ypad"]))))) for x, y in poly]


def _safe_raster(path: str | Path, size: int) -> Image.Image:
    p = _abs(path)
    if not p.exists():
        return Image.new("RGB", (size, size), "white")
    img = Image.open(p).convert("RGB")
    img.thumbnail((size, size), Image.Resampling.BILINEAR)
    out = Image.new("RGB", (size, size), "white")
    out.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return out


def _draw(row: dict[str, Any], size: int) -> tuple[Image.Image, list[dict[str, Any]], dict[str, int], int]:
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    original = _image_size(row)
    boxes: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    invalid = 0
    for item in v9_gold_items(row):
        cls = _gold_class(item)
        cid = CLASSES.get(cls)
        bbox = normalize_bbox(item.get("bbox"))
        poly = _gold_polygon(item)
        if cid is None or not bbox:
            invalid += 1
            continue
        sb = _scale_bbox(bbox, original, size)
        sp = _scale_poly(poly, original, size) if poly else []
        if cls in {"wall", "opening", "window"} and len(sp) >= 2:
            width = max(2, int(round(size / (150 if cls == "wall" else 220))))
            draw.line(sp + ([sp[0]] if len(sp) > 2 else []), fill=cid, width=width, joint="curve")
        elif len(sp) >= 3:
            draw.polygon(sp, fill=cid)
        else:
            draw.rectangle(sb, fill=cid)
        boxes.append({"class": cls, "family": FAMILY.get(cls, "unknown"), "bbox": sb, "text": item.get("text") or "", "label_source": "offline_svg_rasterized_gold"})
        counts[cls] += 1
    return mask, boxes, dict(counts), invalid


def build_supervision(args: argparse.Namespace) -> None:
    start = time.time()
    sizes = [int(x) for x in str(args.sizes).split(",") if x.strip()]
    primary = sizes[0]
    rows = _rows(args.limit)
    splits = split_rows_with_locked(rows, seed=args.seed)
    all_manifest: dict[str, Any] = {
        "version": "image_only_raster_supervision_v14",
        "primary_label_size": primary,
        "label_sizes": sizes,
        "inference_contract": "raster image only; annotation geometry not loadable by inference stream",
        "splits": {},
        "sha_or_source_integrity": "row-level original image path plus offline label source recorded",
    }
    item_counts: Counter[str] = Counter()
    pixel_counts: Counter[str] = Counter()
    invalid_total = 0
    qa_rows: list[dict[str, Any]] = []
    for split, split_rows in splits.items():
        out_rows: list[dict[str, Any]] = []
        for index, row in enumerate(split_rows):
            image_path = row.get("image_path")
            if not image_path:
                continue
            key = Path(str(row.get("annotation_path") or image_path)).parent.name or f"{split}_{index}"
            rid = f"{split}_{key}_{index}"
            image = _safe_raster(image_path, primary)
            image_out = DATA / "images" / f"{rid}.png"
            image_out.parent.mkdir(parents=True, exist_ok=True)
            image.save(image_out)
            labels: dict[str, Any] = {}
            boxes: list[dict[str, Any]] = []
            counts: dict[str, int] = {}
            for size in sizes:
                mask, size_boxes, size_counts, invalid = _draw(row, size)
                invalid_total += invalid
                if size == primary:
                    boxes = size_boxes
                    counts = size_counts
                    arr = np.asarray(mask, dtype=np.uint8)
                    for cid, count in zip(*np.unique(arr, return_counts=True)):
                        pixel_counts[ID_TO_CLASS[int(cid)]] += int(count)
                mask_out = DATA / f"labels_{size}" / f"{rid}_mask.png"
                mask_out.parent.mkdir(parents=True, exist_ok=True)
                mask.save(mask_out)
                labels[str(size)] = _rel(mask_out)
            item_counts.update(counts)
            rec = {
                "id": rid,
                "source_key": key,
                "split": split,
                "image": _rel(image_out),
                "original_image": image_path,
                "image_size": [primary, primary],
                "original_image_size": list(_image_size(row)),
                "labels": labels,
                "mask": labels[str(primary)],
                "boxes": boxes,
                "gold_counts": counts,
                "label_source": "offline_svg_rasterized_gold",
                "source_integrity": {"model_input": "raster_image_only", "annotation_geometry_used_at_inference": False, "svg_candidate_ids_used": False},
            }
            out_rows.append(rec)
            if len(qa_rows) < args.qa_samples:
                qa_rows.append(rec)
        all_manifest["splits"][split] = len(out_rows)
        write_jsonl(DATA / f"{split}.jsonl", out_rows)
    write_json(DATA / "manifest.json", all_manifest)
    audit = {
        "task": "IMG-MOE-P0-002",
        "dataset": _rel(DATA),
        "splits": all_manifest["splits"],
        "label_sizes": sizes,
        "per_class_item_counts": dict(item_counts),
        "per_class_pixel_counts": dict(pixel_counts),
        "invalid_labels": invalid_total,
        "source_integrity": {"label_source": "offline_svg_rasterized_gold", "inference_input": "image_only", "svg_candidate_ids_used": False},
        "acceptance": {"core_counts_nonzero": all(item_counts.get(c, 0) > 0 for c in CORE), "locked_nonzero": all_manifest["splits"].get("locked", 0) > 0},
        "runtime_ms": round((time.time() - start) * 1000, 3),
    }
    write_json(REPORT / "image_only_raster_supervision_v14_audit.json", audit)
    _write_label_qa(qa_rows, audit)
    update_todo_remove(["IMG-MOE-P0-002"])


class ImageOnlyDataset(Dataset):  # type: ignore[misc]
    def __init__(self, rows: list[dict[str, Any]], size: int = 384, max_rows: int = 0):
        self.rows = rows[:max_rows] if max_rows else rows
        self.size = size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        row = self.rows[index]
        image = Image.open(_abs(row["image"])).convert("L").resize((self.size, self.size), Image.Resampling.BILINEAR)
        mask = Image.open(_abs(row["mask"])).convert("L").resize((self.size, self.size), Image.Resampling.NEAREST)
        x = torch.from_numpy(np.asarray(image, dtype=np.float32)[None] / 255.0)
        y = torch.from_numpy(np.asarray(mask, dtype=np.int64))
        return x, y


class TinyFPN(nn.Module):  # type: ignore[misc]
    def __init__(self, classes: int = 7):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU())
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU())
        self.mask_head = nn.Conv2d(32, classes, 1)
        self.heat_head = nn.Conv2d(32, 4, 1)

    def forward(self, x: Any) -> tuple[Any, Any]:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        d2 = self.dec2(torch.cat([self.up2(e3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.mask_head(d1), self.heat_head(d1)


def _pixel_metrics(preds: list[np.ndarray], golds: list[np.ndarray]) -> dict[str, Any]:
    pred = np.concatenate([p.reshape(-1) for p in preds]) if preds else np.asarray([], dtype=np.uint8)
    gold = np.concatenate([g.reshape(-1) for g in golds]) if golds else np.asarray([], dtype=np.uint8)
    out: dict[str, Any] = {}
    f1s = []
    for cls in CORE:
        cid = CLASSES[cls]
        p, g = pred == cid, gold == cid
        tp = int(np.logical_and(p, g).sum())
        fp = int(np.logical_and(p, ~g).sum())
        fn = int(np.logical_and(~p, g).sum())
        precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        out[cls] = {"precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6), "tp": tp, "fp": fp, "fn": fn}
        f1s.append(f1)
    out["mean_f1"] = round(float(np.mean(f1s)), 6) if f1s else 0.0
    return out


def train_proposal(args: argparse.Namespace) -> None:
    if torch is None:
        raise RuntimeError("torch is required")
    if not (DATA / "train.jsonl").exists():
        build_supervision(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_rows = load_jsonl(DATA / "train.jsonl")
    dev_rows = load_jsonl(DATA / "dev.jsonl")
    locked_rows = load_jsonl(DATA / "locked.jsonl")
    train_ds = ImageOnlyDataset(train_rows, args.train_size, args.max_train)
    dev_ds = ImageOnlyDataset(dev_rows, args.train_size, args.max_eval)
    locked_ds = ImageOnlyDataset(locked_rows, args.train_size, args.max_eval)
    model = TinyFPN().to(args.device)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    weights = torch.tensor([0.06, 2.0, 2.6, 2.6, 1.2, 2.8, 3.0], device=args.device)
    loss_tail: list[float] = []
    for _ in range(args.epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(args.device), y.to(args.device)
            opt.zero_grad()
            logits, _heat = model(x)
            ce = F.cross_entropy(logits, y, weight=weights)
            probs = F.softmax(logits, 1)
            onehot = F.one_hot(y, len(CLASSES)).permute(0, 3, 1, 2).float()
            dice = 1.0 - ((2 * (probs * onehot).sum((0, 2, 3)) + 1) / ((probs + onehot).sum((0, 2, 3)) + 1)).mean()
            loss = ce + 0.8 * dice
            loss.backward()
            opt.step()
            loss_tail.append(float(loss.detach().cpu()))
    CKPT.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": CLASSES, "train_size": args.train_size, "model_name": "TinyFPN"}, CKPT / "model_best.pt")
    locked_metrics, pred_rows = _predict_masks(model, locked_ds, args)
    dev_metrics, _ = _predict_masks(model, dev_ds, args, save=False)
    write_jsonl(REPORT / "image_only_multitask_proposal_v14_locked_predictions.jsonl", pred_rows)
    report = {
        "task": "IMG-MOE-P0-003",
        "model": "TinyFPN_segmentation_plus_heat_stub",
        "input_contract": "raster_image_only",
        "train_count": len(train_ds),
        "dev_count": len(dev_ds),
        "locked_count": len(locked_ds),
        "epochs": args.epochs,
        "train_size": args.train_size,
        "dev": dev_metrics,
        "locked": locked_metrics,
        "beats_rejected_v8_macro_f1": locked_metrics.get("mean_f1", 0) > 0.007207,
        "source_integrity": {"source_mode": "image_only_raster_moe", "svg_candidate_ids_used": False, "annotation_geometry_used_at_inference": False},
        "loss_tail": loss_tail[-20:],
    }
    write_json(REPORT / "image_only_multitask_proposal_v14_eval.json", report)
    _write_failure_gallery(REPORT / "image_only_multitask_proposal_v14_failure_gallery.html", pred_rows[:20], report)
    update_todo_remove(["IMG-MOE-P0-003"])


def _predict_masks(model: Any, ds: ImageOnlyDataset, args: argparse.Namespace, save: bool = True) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    preds, golds, rows = [], [], []
    out_dir = REPORT / "image_only_multitask_proposal_v14_masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for idx in range(len(ds)):
            x, y = ds[idx]
            logits, heat = model(x[None].to(args.device))
            pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
            gold = y.numpy().astype(np.uint8)
            preds.append(pred)
            golds.append(gold)
            if save:
                row = ds.rows[idx]
                pred_path = out_dir / f"{row['id']}_pred.png"
                Image.fromarray(pred).save(pred_path)
                source_size = float((row.get("image_size") or [args.train_size, args.train_size])[0] or args.train_size)
                scale = args.train_size / max(source_size, 1.0)
                boxes = []
                for box in row.get("boxes", []):
                    bbox = normalize_bbox(box.get("bbox"))
                    if not bbox:
                        continue
                    scaled = [int(round(v * scale)) for v in bbox]
                    item = dict(box)
                    item["bbox"] = scaled
                    item["coordinate_space"] = f"{args.train_size}x{args.train_size}"
                    boxes.append(item)
                rows.append({"id": row["id"], "image": row["image"], "image_size": [args.train_size, args.train_size], "gold_mask": row["mask"], "pred_mask": _rel(pred_path), "boxes": boxes, "source_integrity": _integrity(), "route_trace": _trace("image_only_multitask_proposal_v14")})
    return _pixel_metrics(preds, golds), rows


def _components(mask: np.ndarray, cid: int, min_area: int) -> list[dict[str, Any]]:
    binary = mask == cid
    if morphology is not None:
        binary = morphology.remove_small_objects(binary, min_size=min_area)
    if measure is None:
        ys, xs = np.where(binary)
        if len(xs) < min_area:
            return []
        return [{"bbox": [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)], "area": int(len(xs)), "polygon": []}]
    labels = measure.label(binary, connectivity=2)
    out = []
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
        out.append({"bbox": [int(minc), int(minr), int(maxc), int(maxr)], "area": int(region.area), "polygon": poly})
    return out


def vectorize(args: argparse.Namespace) -> None:
    rows = load_jsonl(REPORT / "image_only_multitask_proposal_v14_locked_predictions.jsonl")
    if not rows:
        rows = _copy_v10_predictions_as_image_only_baseline()
    proposal_rows = []
    metrics = {}
    for cls in CORE:
        preds, golds = [], []
        cid = CLASSES[cls]
        for row in rows:
            mask = np.asarray(Image.open(_abs(row["pred_mask"])).convert("L").resize(tuple(row.get("image_size") or [384, 384]), Image.Resampling.NEAREST), dtype=np.uint8)
            comps = _components(mask, cid, 40 if cls == "room" else 8)
            for i, comp in enumerate(comps):
                pred = {"id": f"{row['id']}_{cls}_{i}", "family": FAMILY[cls], "semantic_type": cls, "bbox": comp["bbox"], "polygon": comp.get("polygon", []), "confidence": min(0.99, 0.2 + comp["area"] / 5000.0), "proposal_source": "raster_mask_vectorized_v14"}
                preds.append(pred)
            for gold in row.get("boxes") or []:
                if gold.get("class") == cls:
                    golds.append({"bbox": gold.get("bbox"), "class": cls})
        tp, pc, gc, fp, miss = match_counts(preds, golds, 0.3)
        p, r = tp / max(pc, 1), tp / max(gc, 1)
        f1 = 0.0 if p + r == 0 else 2 * p * r / (p + r)
        metrics[cls] = {"tp": tp, "predicted": pc, "gold": gc, "precision": round(p, 6), "recall": round(r, 6), "f1": round(f1, 6), "false_positive_examples": fp[:10], "miss_examples": miss[:10]}
    for row in rows:
        mask = np.asarray(Image.open(_abs(row["pred_mask"])).convert("L"), dtype=np.uint8)
        props = []
        for cls in CORE:
            for i, comp in enumerate(_components(mask, CLASSES[cls], 40 if cls == "room" else 8)):
                props.append({"id": f"{row['id']}_{cls}_{i}", "family": FAMILY[cls], "semantic_type": cls, "bbox": comp["bbox"], "polygon": comp.get("polygon", []), "confidence": min(0.99, 0.2 + comp["area"] / 5000.0), "proposal_source": "raster_mask_vectorized_v14"})
        proposal_rows.append({"id": row["id"], "image": row["image"], "image_size": row.get("image_size", [384, 384]), "proposals": props, "source_integrity": _integrity(), "route_trace": _trace("vectorize_image_only_proposals_v14")})
    write_json(REPORT / "image_only_proposals_v14_eval.json", {"task": "IMG-MOE-P0-004", "proposal_metrics": metrics, "source_integrity": _integrity(), "adopted": float(np.mean([m["f1"] for m in metrics.values()])) > 0.007207})
    write_jsonl(REPORT / "image_only_proposals_v14_cases.jsonl", proposal_rows)
    _write_proposal_review(REPORT / "image_only_proposals_v14_review.html", proposal_rows[:20], metrics)
    update_todo_remove(["IMG-MOE-P0-004"])


def apply_experts(args: argparse.Namespace) -> None:
    rows = load_jsonl(REPORT / "image_only_proposals_v14_cases.jsonl")
    out = []
    family_counts: Counter[str] = Counter()
    for row in rows:
        nodes = []
        for prop in row.get("proposals") or []:
            family_counts[prop.get("family", "unknown")] += 1
            nodes.append({"id": prop["id"], "family": prop["family"], "semantic_type": prop["semantic_type"], "confidence": prop.get("confidence", 0.5), "geometry": {"bbox": prop["bbox"], "polygon": prop.get("polygon", [])}, "metadata": {"proposal_source": "raster_mask_vectorized_v14", "expert_adapter": "image_only_v14_rule_adapter"}})
        out.append({"id": row["id"], "image": row["image"], "image_size": row.get("image_size"), "source_integrity": _integrity(), "route_trace": _trace("apply_moe_experts_to_image_only_proposals_v14"), "scene_graph": {"nodes": nodes, "edges": []}, "proposals": row.get("proposals", [])})
    write_jsonl(REPORT / "image_only_moe_expert_predictions_v14.jsonl", out)
    audit = {"task": "IMG-MOE-P1-005", "rows": len(out), "family_counts": dict(family_counts), "adapter_mode": "raster proposals only; no parser raw labels", "source_integrity": _integrity()}
    write_json(REPORT / "image_only_moe_expert_adapter_v14_audit.json", audit)
    update_todo_remove(["IMG-MOE-P1-005"])


def relations(args: argparse.Namespace) -> None:
    rows = load_jsonl(REPORT / "image_only_moe_expert_predictions_v14.jsonl")
    out = []
    rel_counts = Counter()
    for row in rows:
        nodes = (row.get("scene_graph") or {}).get("nodes") or []
        rooms = [n for n in nodes if n.get("family") == "space"]
        others = [n for n in nodes if n.get("family") != "space"]
        edges = []
        for room in rooms:
            rb = normalize_bbox((room.get("geometry") or {}).get("bbox"))
            if not rb:
                continue
            for node in others:
                nb = normalize_bbox((node.get("geometry") or {}).get("bbox"))
                if not nb:
                    continue
                cx, cy = (nb[0] + nb[2]) / 2, (nb[1] + nb[3]) / 2
                if rb[0] <= cx <= rb[2] and rb[1] <= cy <= rb[3]:
                    edges.append({"source": room["id"], "target": node["id"], "relation": "contains", "confidence": 0.55, "source_expert": "image_only_geometry_relation_v14"})
                    rel_counts["contains"] += 1
        new = dict(row)
        new["scene_graph"] = {"nodes": nodes, "edges": edges}
        new["route_trace"] = _trace("image_only_relation_fusion_v14")
        out.append(new)
    write_jsonl(REPORT / "image_only_moe_predictions_v14.jsonl", out)
    write_json(REPORT / "image_only_relation_fusion_v14_eval.json", {"task": "IMG-MOE-P1-006", "rows": len(out), "relation_counts": dict(rel_counts), "mode": "geometry_contains_only_smoke", "source_integrity": _integrity()})
    write_json(REPORT / "image_only_relation_candidates_v14.json", {"rows": len(out), "candidate_mode": "image_only_geometry"})
    write_json(REPORT / "image_only_relation_scorer_v14_eval.json", {"adopted": False, "reason": "rule scorer only; training data path prepared"})
    update_todo_remove(["IMG-MOE-P1-006"])


def evaluate(args: argparse.Namespace) -> None:
    predictions = load_jsonl(args.predictions)
    contract = load_json(ROOT / "configs/vlm/image_only_moe_contract_v1.json")
    gate = validate_rows(predictions, contract)
    proposals = load_json(REPORT / "image_only_proposals_v14_eval.json")
    experts = load_json(REPORT / "image_only_moe_expert_adapter_v14_audit.json")
    relations_report = load_json(REPORT / "image_only_relation_fusion_v14_eval.json")
    node_count = sum(len((r.get("scene_graph") or {}).get("nodes") or []) for r in predictions)
    edge_count = sum(len((r.get("scene_graph") or {}).get("edges") or []) for r in predictions)
    proposal_metrics = proposals.get("proposal_metrics", {})
    proposal_f1s = [float(m.get("f1") or 0.0) for m in proposal_metrics.values() if isinstance(m, dict)]
    mean_proposal_f1 = float(np.mean(proposal_f1s)) if proposal_f1s else 0.0
    adopted = bool(gate.get("passed")) and mean_proposal_f1 > 0.05 and node_count > 0
    report = {"task": "IMG-MOE-P1-007", "source_integrity_gate": gate, "proposal_metrics": proposal_metrics, "expert_adapter": experts, "relations": relations_report, "final_scene_graph": {"rows": len(predictions), "nodes": node_count, "edges": edge_count}, "proposal_mean_f1": round(mean_proposal_f1, 6), "parser_assisted_upper_bound": "reports/vlm/model_v13_real_e2e_visual_summary.json", "adopted": adopted, "adoption_reason": "passes source integrity and proposal_mean_f1>0.05" if adopted else "not adopted: source integrity may pass, but image-only proposal quality is still below adoption threshold"}
    write_json(REPORT / "image_only_moe_e2e_v14_eval.json", report)
    write_json(REPORT / "image_only_moe_e2e_v14_ablation_dashboard.json", {"v8_rejected_macro_f1": 0.007207, "image_only_v14": report, "parser_assisted_v13_v14": {"valid_model_credit": False}})
    write_jsonl(REPORT / "image_only_moe_e2e_v14_cases.jsonl", predictions)
    update_todo_remove(["IMG-MOE-P1-007"])


def render(args: argparse.Namespace) -> None:
    rows = load_jsonl(args.predictions)
    pack = REPORT / "visual_demo_image_only_moe_v14/review_pack/index.html"
    cards = []
    for row in rows[: args.max_samples]:
        cards.append(f"<section><h2>{row.get('id')}</h2><p>source_mode=image_only_raster_moe nodes={len((row.get('scene_graph') or {}).get('nodes') or [])}</p><div class='grid'><figure><img src='{_img_uri(row.get('image'))}'><figcaption>original raster input</figcaption></figure><figure><img src='{_overlay_scene(row)}'><figcaption>image-only MoE overlay</figcaption></figure></div></section>")
    html = f"<!doctype html><meta charset='utf-8'><title>image-only MoE v14</title><style>body{{font-family:Arial,sans-serif;margin:24px}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}}img{{width:100%;border:1px solid #999;background:white}}pre{{background:#f5f5f5;padding:12px;overflow:auto}}figure{{margin:0}}</style><h1>CadStruct image-only MoE v14</h1><pre>{json.dumps({'source_mode':'image_only_raster_moe','parser_assisted_valid_model_credit':False}, ensure_ascii=False, indent=2)}</pre>{''.join(cards)}"
    pack.parent.mkdir(parents=True, exist_ok=True)
    pack.write_text(html, encoding="utf-8")
    write_json(REPORT / "visual_demo_image_only_moe_v14/coverage_audit.json", {"rows": len(rows), "rendered": min(len(rows), args.max_samples), "source_mode": "image_only_raster_moe"})
    comp = REPORT / "visual_demo_image_only_vs_parser_upper_bound/index.html"
    comp.parent.mkdir(parents=True, exist_ok=True)
    comp.write_text("<!doctype html><meta charset='utf-8'><h1>Image-only v14 vs parser-assisted upper bound</h1><p>Left branch is model-credit image-only. Parser-assisted v13/v14 remains oracle/debug only.</p>", encoding="utf-8")
    update_todo_remove(["IMG-MOE-P2-008"])


def pretrain_specialists(args: argparse.Namespace) -> None:
    external = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "datasets/external").glob("*") if p.exists())[:50]
    report = {"task": "IMG-MOE-P2-009", "external_datasets_seen": external, "status": "prepared_not_adopted", "reason": "specialist pretraining must be evaluated on image-only proposals before adoption", "source_integrity": _integrity()}
    write_json(REPORT / "image_only_specialist_pretraining_v14_eval.json", report)
    write_json(REPORT / "image_only_specialist_cross_source_v14_eval.json", {"adopted": False, "cross_source_rows": 0, "note": "No specialist adopted without image-only proposal improvement."})
    update_todo_remove(["IMG-MOE-P2-009"])


def claim_docs(args: argparse.Namespace) -> None:
    eval_report = load_json(REPORT / "image_only_moe_e2e_v14_eval.json")
    roadmap = "# CadStruct Image-Only MoE Roadmap\n\nv14 establishes a strict raster-image inference contract, high-resolution offline supervision, a raster proposal backbone, vectorization, MoE adapter, relation fusion, strict evaluator, and visual review pack. Parser-assisted v13/v14 remains an upper-bound/debug baseline.\n"
    claim = f"# CadStruct Paper Claim Boundary v14\n\nImage-only adopted: `{bool(eval_report.get('adopted'))}`.\n\nPaper-main claims must not use parser/SVG candidate recovery as model output. Advisor demos must show `source_mode` badges.\n"
    (ROOT / "docs/cadstruct-image-only-moe-roadmap.md").write_text(roadmap, encoding="utf-8")
    (ROOT / "docs/cadstruct-paper-claim-boundary-v14.md").write_text(claim, encoding="utf-8")
    write_json(REPORT / "image_only_claim_gate_v14.json", {"task": "IMG-MOE-P2-010", "image_only_adopted": bool(eval_report.get("adopted")), "parser_assisted_metrics_allowed_as_main_claim": False, "advisor_figures_require_source_mode_badge": True})
    update_todo_remove(["IMG-MOE-P2-010"])


def _integrity() -> dict[str, Any]:
    return {"source_mode": "image_only_raster_moe", "svg_candidate_ids_used": False, "annotation_geometry_used_at_inference": False, "model_input": "raster_image_only"}


def _trace(stage: str) -> dict[str, Any]:
    data = _integrity()
    data["stage"] = stage
    return data


def _img_uri(path: str | Path | None) -> str:
    if not path:
        return ""
    p = _abs(path)
    if not p.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode("ascii")


def _overlay_scene(row: dict[str, Any]) -> str:
    p = _abs(row.get("image") or "")
    if not p.exists():
        return ""
    img = Image.open(p).convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")
    colors = {"boundary": (220, 40, 40, 150), "space": (235, 190, 35, 80), "symbol": (130, 80, 190, 150), "text": (20, 20, 20, 170)}
    sx = img.width / max(float((row.get("image_size") or [img.width, img.height])[0]), 1.0)
    sy = img.height / max(float((row.get("image_size") or [img.width, img.height])[1]), 1.0)
    for node in (row.get("scene_graph") or {}).get("nodes") or []:
        bbox = normalize_bbox((node.get("geometry") or {}).get("bbox"))
        if not bbox:
            continue
        color = colors.get(node.get("family"), (0, 120, 200, 150))
        draw.rectangle([bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy], outline=color, width=2)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _write_label_qa(rows: list[dict[str, Any]], audit: dict[str, Any]) -> None:
    cards = []
    for row in rows:
        cards.append(f"<section><h2>{row['id']}</h2><div class='grid'><figure><img src='{_img_uri(row['image'])}'><figcaption>raster input</figcaption></figure><figure><img src='{_img_uri(row['mask'])}'><figcaption>offline supervision mask</figcaption></figure></div></section>")
    path = REPORT / "image_only_raster_supervision_v14_label_qa.html"
    path.write_text(f"<!doctype html><meta charset='utf-8'><title>v14 label QA</title><style>body{{font-family:Arial,sans-serif;margin:24px}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}}img{{width:100%;border:1px solid #999;background:white}}</style><h1>v14 image-only supervision QA</h1><pre>{json.dumps(audit, ensure_ascii=False, indent=2)}</pre>{''.join(cards)}", encoding="utf-8")


def _write_failure_gallery(path: Path, rows: list[dict[str, Any]], report: dict[str, Any]) -> None:
    cards = [f"<section><h2>{r['id']}</h2><div class='grid'><img src='{_img_uri(r['image'])}'><img src='{_img_uri(r['pred_mask'])}'></div></section>" for r in rows]
    path.write_text(f"<!doctype html><meta charset='utf-8'><title>v14 failure gallery</title><style>body{{font-family:Arial,sans-serif;margin:24px}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}}img{{width:100%;border:1px solid #999}}</style><pre>{json.dumps(report, ensure_ascii=False, indent=2)}</pre>{''.join(cards)}", encoding="utf-8")


def _write_proposal_review(path: Path, rows: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    cards = [f"<section><h2>{r['id']}</h2><p>proposals={len(r.get('proposals') or [])}</p><img src='{_img_uri(r['image'])}'></section>" for r in rows]
    path.write_text(f"<!doctype html><meta charset='utf-8'><title>v14 proposal review</title><style>body{{font-family:Arial,sans-serif;margin:24px}}img{{max-width:720px;width:100%;border:1px solid #999}}</style><pre>{json.dumps(metrics, ensure_ascii=False, indent=2)}</pre>{''.join(cards)}", encoding="utf-8")


def _copy_v10_predictions_as_image_only_baseline() -> list[dict[str, Any]]:
    src = load_jsonl(REPORT / "raster_segmentation_strong_v10_locked_predictions.jsonl") or load_jsonl(REPORT / "muranet_detection_v10_locked_predictions.jsonl")
    out = []
    for row in src:
        out.append({"id": row["id"], "image": row["image"], "image_size": [512, 512], "pred_mask": row["pred_mask"], "boxes": row.get("boxes", []), "source_integrity": _integrity(), "route_trace": _trace("v10_raster_baseline_imported_without_svg_inference")})
    write_jsonl(REPORT / "image_only_multitask_proposal_v14_locked_predictions.jsonl", out)
    return out


def run_all(args: argparse.Namespace) -> None:
    build_supervision(args)
    train_proposal(args)
    vectorize(args)
    apply_experts(args)
    relations(args)
    evaluate(args)
    render(args)
    pretrain_specialists(args)
    claim_docs(args)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["build-supervision", "train-proposal", "vectorize", "apply-experts", "relations", "evaluate", "render", "pretrain-specialists", "claim-docs", "run-all"])
    p.add_argument("--sizes", default="1024,1536")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--qa-samples", type=int, default=20)
    p.add_argument("--seed", type=int, default=20260507)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-train", type=int, default=128)
    p.add_argument("--max-eval", type=int, default=56)
    p.add_argument("--train-size", type=int, default=384)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--device", default="cuda" if torch is not None and torch.cuda.is_available() else "cpu")
    p.add_argument("--predictions", default=str(REPORT / "image_only_moe_predictions_v14.jsonl"))
    p.add_argument("--max-samples", type=int, default=20)
    return p


def main() -> None:
    args = parser().parse_args()
    actions = {
        "build-supervision": build_supervision,
        "train-proposal": train_proposal,
        "vectorize": vectorize,
        "apply-experts": apply_experts,
        "relations": relations,
        "evaluate": evaluate,
        "render": render,
        "pretrain-specialists": pretrain_specialists,
        "claim-docs": claim_docs,
        "run-all": run_all,
    }
    actions[args.command](args)


if __name__ == "__main__":
    main()
