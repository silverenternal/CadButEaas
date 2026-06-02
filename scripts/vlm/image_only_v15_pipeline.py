#!/usr/bin/env python3
"""Strict raster-image CadStruct-MoE v15 proposal recovery pipeline."""

from __future__ import annotations

import argparse
import base64
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

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
from scripts.vlm.v8_raster_e2e_utils import CUBICASA_ROOT, bbox_area, bbox_iou, load_json, load_jsonl, match_counts, normalize_bbox, sample_key, write_json, write_jsonl
from scripts.vlm.v9_raster_pipeline import _gold_class, _gold_polygon, v9_gold_items
from scripts.vlm.validate_image_only_moe_stream import validate_rows


REPORT = ROOT / "reports/vlm"
DATA = ROOT / "datasets/image_only_raster_supervision_v15"
CKPT = ROOT / "checkpoints/image_only_multitask_proposal_v15"
HEADS = ["wall", "opening", "window", "room", "room_boundary", "symbol", "text"]
CORE = ["wall", "opening", "window", "room", "symbol", "text"]
HEAD_INDEX = {name: index for index, name in enumerate(HEADS)}
FAMILY = {"wall": "boundary", "opening": "boundary", "window": "boundary", "room": "space", "symbol": "symbol", "text": "text"}
TODO = [
    "IMG-MOE-V15-P0-001",
    "IMG-MOE-V15-P0-002",
    "IMG-MOE-V15-P0-003",
    "IMG-MOE-V15-P0-004",
    "IMG-MOE-V15-P0-005",
    "IMG-MOE-V15-P1-006",
    "IMG-MOE-V15-P1-007",
    "IMG-MOE-V15-P1-008",
    "IMG-MOE-V15-P2-009",
    "IMG-MOE-V15-P2-010",
]


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


def _integrity() -> dict[str, Any]:
    return {
        "source_mode": "image_only_raster_moe",
        "svg_candidate_ids_used": False,
        "annotation_geometry_used_at_inference": False,
        "model_input": "raster_image_only",
    }


def _trace(stage: str) -> dict[str, Any]:
    data = _integrity()
    data["stage"] = stage
    return data


def _rows(limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in ("train", "dev", "smoke"):
        rows.extend(load_jsonl(ROOT / "datasets/cadstruct_cubicasa5k_moe" / f"{name}.jsonl"))
    if not rows:
        rows = convert_dataset(CUBICASA_ROOT, limit or None, 4.0)
    seen = set()
    deduped = []
    for row in rows:
        key = str(row.get("annotation_path") or row.get("image_path") or len(deduped))
        if key not in seen and row.get("image_path") and row.get("annotation_path"):
            seen.add(key)
            deduped.append(row)
    return deduped[:limit] if limit else deduped


def _split_rows_v15(rows: list[dict[str, Any]], seed: int, min_train: int = 64) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    n = len(rows)
    locked_count = 0 if n < 8 else max(4, min(64, int(round(n * 0.12))))
    dev_count = 0 if n < 12 else max(4, min(64, int(round(n * 0.12))))
    while n - locked_count - dev_count < min(min_train, max(1, int(n * 0.5))) and (locked_count > 4 or dev_count > 4):
        if dev_count >= locked_count and dev_count > 4:
            dev_count -= 1
        elif locked_count > 4:
            locked_count -= 1
        else:
            break
    locked = rows[:locked_count]
    dev = rows[locked_count : locked_count + dev_count]
    train = rows[locked_count + dev_count :]
    smoke = locked[:5] or dev[:5] or train[:5]
    source_sets = {name: {sample_key(row.get("image_path")) for row in part} for name, part in {"train": train, "dev": dev, "locked": locked}.items()}
    overlaps = {
        "train_dev": len(source_sets["train"] & source_sets["dev"]),
        "train_locked": len(source_sets["train"] & source_sets["locked"]),
        "dev_locked": len(source_sets["dev"] & source_sets["locked"]),
    }
    split_ok = len(train) >= max(64, 2 * max(len(dev), 1)) if n >= 128 else len(train) > len(dev) + len(locked)
    return {"train": train, "dev": dev, "locked": locked, "smoke": smoke}, {
        "available_rows": n,
        "train_count": len(train),
        "dev_count": len(dev),
        "locked_count": len(locked),
        "overlaps": overlaps,
        "split_sanity_passed": bool(split_ok and not any(overlaps.values())),
    }


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
    return {"scale": scale, "xpad": (size - ow * scale) / 2.0, "ypad": (size - oh * scale) / 2.0}


def _scale_bbox(bbox: list[float], original: tuple[int, int], size: int) -> list[int]:
    t = _transform(original, size)
    x1 = int(max(0, min(size - 1, round(bbox[0] * t["scale"] + t["xpad"]))))
    y1 = int(max(0, min(size - 1, round(bbox[1] * t["scale"] + t["ypad"]))))
    x2 = int(max(0, min(size - 1, round(bbox[2] * t["scale"] + t["xpad"]))))
    y2 = int(max(0, min(size - 1, round(bbox[3] * t["scale"] + t["ypad"]))))
    return [min(x1, x2), min(y1, y2), max(x1 + 1, x2), max(y1 + 1, y2)]


def _scale_poly(poly: list[list[float]], original: tuple[int, int], size: int) -> list[tuple[int, int]]:
    t = _transform(original, size)
    return [
        (int(max(0, min(size - 1, round(float(x) * t["scale"] + t["xpad"])))), int(max(0, min(size - 1, round(float(y) * t["scale"] + t["ypad"])))))
        for x, y in poly
    ]


def _safe_raster(path: str | Path, size: int) -> Image.Image:
    p = _abs(path)
    if not p.exists():
        return Image.new("RGB", (size, size), "white")
    img = Image.open(p).convert("RGB")
    img.thumbnail((size, size), Image.Resampling.BILINEAR)
    out = Image.new("RGB", (size, size), "white")
    out.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return out


def _draw_targets(row: dict[str, Any], size: int) -> tuple[dict[str, Image.Image], list[dict[str, Any]], dict[str, int], int]:
    masks = {head: Image.new("L", (size, size), 0) for head in HEADS}
    draws = {head: ImageDraw.Draw(masks[head]) for head in HEADS}
    original = _image_size(row)
    boxes: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    invalid = 0
    for item in v9_gold_items(row):
        cls = _gold_class(item)
        if cls not in CORE:
            invalid += 1
            continue
        bbox = normalize_bbox(item.get("bbox"))
        if not bbox:
            invalid += 1
            continue
        poly = _gold_polygon(item)
        sb = _scale_bbox(bbox, original, size)
        sp = _scale_poly(poly, original, size) if poly else []
        if cls == "wall":
            width = max(2, int(round(size / 120)))
            if len(sp) >= 2:
                draws["wall"].line(sp + ([sp[0]] if len(sp) > 2 else []), fill=255, width=width, joint="curve")
            else:
                draws["wall"].rectangle(sb, fill=255)
        elif cls in {"opening", "window"}:
            width = max(2, int(round(size / 180)))
            if len(sp) >= 2:
                draws[cls].line(sp, fill=255, width=width)
            else:
                draws[cls].rectangle(sb, fill=255)
        elif cls == "room":
            if len(sp) >= 3:
                draws["room"].polygon(sp, fill=255)
                draws["room_boundary"].line(sp + [sp[0]], fill=255, width=max(2, size // 220), joint="curve")
            else:
                draws["room"].rectangle(sb, fill=255)
                draws["room_boundary"].rectangle(sb, outline=255, width=max(2, size // 220))
        else:
            pad = max(1, int(round(size / 512)))
            box = [max(0, sb[0] - pad), max(0, sb[1] - pad), min(size - 1, sb[2] + pad), min(size - 1, sb[3] + pad)]
            draws[cls].rectangle(box, fill=255)
        boxes.append({"class": cls, "family": FAMILY.get(cls, "unknown"), "bbox": sb, "text": item.get("text") or "", "label_source": "offline_svg_rasterized_gold"})
        counts[cls] += 1
    for head in ("opening", "window", "symbol", "text"):
        masks[head] = masks[head].filter(ImageFilter.MaxFilter(3))
    return masks, boxes, dict(counts), invalid


def _head_overlay_uri(row: dict[str, Any]) -> str:
    img = Image.open(_abs(row["image"])).convert("RGBA")
    colors = {
        "wall": (220, 40, 40, 135),
        "opening": (40, 150, 220, 150),
        "window": (25, 180, 160, 150),
        "room": (235, 190, 35, 75),
        "room_boundary": (240, 120, 0, 130),
        "symbol": (130, 80, 190, 150),
        "text": (20, 20, 20, 170),
    }
    for head, rel_path in (row.get("targets") or {}).items():
        p = _abs(rel_path)
        if not p.exists():
            continue
        mask = Image.open(p).convert("L").resize(img.size, Image.Resampling.NEAREST)
        color = Image.new("RGBA", img.size, colors.get(head, (0, 120, 200, 120)))
        img.alpha_composite(Image.composite(color, Image.new("RGBA", img.size, (0, 0, 0, 0)), mask))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _img_uri(path: str | Path | None) -> str:
    if not path:
        return ""
    p = _abs(path)
    if not p.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode("ascii")


def build_supervision(args: argparse.Namespace) -> None:
    start = time.time()
    sizes = [int(x) for x in str(args.sizes).split(",") if x.strip()]
    primary = sizes[0]
    rows = _rows(args.limit)
    splits, split_report = _split_rows_v15(rows, args.seed)
    all_manifest: dict[str, Any] = {
        "version": "image_only_raster_supervision_v15",
        "primary_label_size": primary,
        "label_sizes": sizes,
        "heads": HEADS,
        "inference_contract": "raster image only; annotation geometry is offline supervision only",
        "splits": {},
        "split_report": split_report,
    }
    item_counts: Counter[str] = Counter()
    positive_pixels: dict[str, Counter[str]] = {head: Counter() for head in HEADS}
    invalid_total = 0
    qa_rows: list[dict[str, Any]] = []
    DATA.mkdir(parents=True, exist_ok=True)
    for split, split_rows in splits.items():
        out_rows: list[dict[str, Any]] = []
        for index, row in enumerate(split_rows):
            image_path = row.get("image_path")
            if not image_path:
                continue
            key = sample_key(image_path) or Path(str(row.get("annotation_path") or image_path)).parent.name or f"{split}_{index}"
            rid = f"{split}_{key}_{index}"
            image = _safe_raster(image_path, primary)
            image_out = DATA / "images" / f"{rid}.png"
            image_out.parent.mkdir(parents=True, exist_ok=True)
            image.save(image_out)
            targets_by_size: dict[str, dict[str, str]] = {}
            boxes: list[dict[str, Any]] = []
            counts: dict[str, int] = {}
            for size in sizes:
                masks, size_boxes, size_counts, invalid = _draw_targets(row, size)
                invalid_total += invalid
                targets_by_size[str(size)] = {}
                if size == primary:
                    boxes = size_boxes
                    counts = size_counts
                for head, mask in masks.items():
                    out = DATA / f"targets_{size}" / head / f"{rid}.png"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    mask.save(out)
                    targets_by_size[str(size)][head] = _rel(out)
                    if size == primary:
                        positive_pixels[head][split] += int(np.asarray(mask, dtype=np.uint8).astype(bool).sum())
            item_counts.update(counts)
            rec = {
                "id": rid,
                "source_key": key,
                "split": split,
                "image": _rel(image_out),
                "original_image": image_path,
                "image_size": [primary, primary],
                "original_image_size": list(_image_size(row)),
                "targets_by_size": targets_by_size,
                "targets": targets_by_size[str(primary)],
                "boxes": boxes,
                "gold_counts": counts,
                "label_source": "offline_svg_rasterized_gold",
                "source_integrity": _integrity(),
            }
            out_rows.append(rec)
            if len(qa_rows) < args.qa_samples and split != "smoke":
                qa_rows.append(rec)
        all_manifest["splits"][split] = len(out_rows)
        write_jsonl(DATA / f"{split}.jsonl", out_rows)
    write_json(DATA / "manifest.json", all_manifest)
    head_stats = {}
    for head in HEADS:
        head_stats[head] = {}
        for split, count in all_manifest["splits"].items():
            denom = max(count * primary * primary, 1)
            ratio = positive_pixels[head][split] / denom
            head_stats[head][split] = {"positive_pixels": int(positive_pixels[head][split]), "positive_ratio": round(ratio, 8)}
    warnings = []
    for head, splits_stats in head_stats.items():
        train_ratio = float((splits_stats.get("train") or {}).get("positive_ratio") or 0)
        if train_ratio <= 0:
            warnings.append({"head": head, "reason": "no_train_positive_pixels"})
        if train_ratio > 0.65:
            warnings.append({"head": head, "reason": "near_all_image_positive_target", "positive_ratio": train_ratio})
    audit = {
        "task": ["IMG-MOE-V15-P0-001", "IMG-MOE-V15-P0-002"],
        "dataset": _rel(DATA),
        "splits": all_manifest["splits"],
        "split_report": split_report,
        "label_sizes": sizes,
        "per_class_item_counts": dict(item_counts),
        "per_head_pixel_stats": head_stats,
        "invalid_labels": invalid_total,
        "warnings": warnings,
        "source_integrity": _integrity() | {"label_source": "offline_svg_rasterized_gold"},
        "acceptance": {
            "split_overlaps_zero": not any(split_report["overlaps"].values()),
            "split_sanity_passed": split_report["split_sanity_passed"],
            "core_counts_nonzero": all(item_counts.get(c, 0) > 0 for c in CORE),
            "heads_nonzero": all((head_stats[h].get("train") or {}).get("positive_pixels", 0) > 0 for h in HEADS),
            "no_all_image_targets": not any(w.get("reason") == "near_all_image_positive_target" for w in warnings),
        },
        "runtime_ms": round((time.time() - start) * 1000, 3),
    }
    write_json(REPORT / "image_only_raster_supervision_v15_audit.json", audit)
    write_json(REPORT / "image_only_multitask_targets_v15_audit.json", audit)
    _write_label_qa(REPORT / "image_only_multitask_targets_v15_qa.html", qa_rows, audit)
    update_todo_remove(["IMG-MOE-V15-P0-001", "IMG-MOE-V15-P0-002"])


def audit_geometry(args: argparse.Namespace) -> dict[str, Any]:
    if not (DATA / "manifest.json").exists():
        build_supervision(args)
    bad: list[dict[str, Any]] = []
    checked = 0
    full_image_boxes = 0
    tiny_boxes = 0
    for split in ("train", "dev", "locked"):
        for row in load_jsonl(DATA / f"{split}.jsonl"):
            size = row.get("image_size") or [args.train_size, args.train_size]
            w, h = int(size[0]), int(size[1])
            for box in row.get("boxes") or []:
                checked += 1
                bbox = normalize_bbox(box.get("bbox"))
                if not bbox:
                    bad.append({"id": row.get("id"), "split": split, "class": box.get("class"), "reason": "invalid_bbox", "bbox": box.get("bbox")})
                    continue
                area_ratio = bbox_area(bbox) / max(w * h, 1)
                if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > w or bbox[3] > h:
                    bad.append({"id": row.get("id"), "split": split, "class": box.get("class"), "reason": "off_canvas", "bbox": bbox, "image_size": size})
                if area_ratio > 0.9:
                    full_image_boxes += 1
                    bad.append({"id": row.get("id"), "split": split, "class": box.get("class"), "reason": "near_full_image_box", "bbox": bbox})
                if box.get("class") in {"symbol", "text"} and bbox_area(bbox) < 4:
                    tiny_boxes += 1
                    bad.append({"id": row.get("id"), "split": split, "class": box.get("class"), "reason": "tiny_symbol_or_text_box", "bbox": bbox})
            for head, rel_path in (row.get("targets") or {}).items():
                p = _abs(rel_path)
                if not p.exists():
                    bad.append({"id": row.get("id"), "split": split, "class": head, "reason": "missing_target", "path": rel_path})
                    continue
                with Image.open(p) as im:
                    if im.size != (w, h):
                        bad.append({"id": row.get("id"), "split": split, "class": head, "reason": "target_size_mismatch", "target_size": im.size, "image_size": size})
    report = {
        "task": "IMG-MOE-V15-P0-003",
        "checked_boxes": checked,
        "bad_cases": len(bad),
        "full_image_boxes": full_image_boxes,
        "tiny_symbol_or_text_boxes": tiny_boxes,
        "coordinate_space_consistent": not any(c["reason"] in {"off_canvas", "target_size_mismatch"} for c in bad),
        "passes_training_gate": not any(c["reason"] in {"off_canvas", "target_size_mismatch", "near_full_image_box"} for c in bad),
        "source_integrity": _integrity(),
    }
    write_json(REPORT / "image_only_label_geometry_v15_audit.json", report)
    write_jsonl(REPORT / "image_only_label_geometry_v15_bad_cases.jsonl", bad[:1000])
    update_todo_remove(["IMG-MOE-V15-P0-003"])
    return report


class MultiHeadDataset(Dataset):  # type: ignore[misc]
    def __init__(self, rows: list[dict[str, Any]], size: int = 384, max_rows: int = 0):
        self.rows = rows[:max_rows] if max_rows else rows
        self.size = size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        row = self.rows[index]
        image = Image.open(_abs(row["image"])).convert("L").resize((self.size, self.size), Image.Resampling.BILINEAR)
        x = torch.from_numpy(np.asarray(image, dtype=np.float32)[None] / 255.0)
        ys = []
        for head in HEADS:
            mask = Image.open(_abs(row["targets"][head])).convert("L").resize((self.size, self.size), Image.Resampling.NEAREST)
            ys.append((np.asarray(mask, dtype=np.float32) > 0).astype(np.float32))
        y = torch.from_numpy(np.stack(ys, axis=0))
        return x, y


class SmallUNet(nn.Module):  # type: ignore[misc]
    def __init__(self, out_channels: int = len(HEADS)):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU())
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU())
        self.head = nn.Conv2d(32, out_channels, 1)

    def forward(self, x: Any) -> Any:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        d2 = self.dec2(torch.cat([self.up2(e3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)


def _dice_loss(logits: Any, target: Any) -> Any:
    probs = torch.sigmoid(logits)
    dims = (0, 2, 3)
    inter = (probs * target).sum(dims)
    denom = probs.sum(dims) + target.sum(dims)
    return 1.0 - ((2.0 * inter + 1.0) / (denom + 1.0)).mean()


def _head_metrics(preds: list[np.ndarray], golds: list[np.ndarray]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    f1s = []
    for head in HEADS:
        idx = HEAD_INDEX[head]
        p = np.concatenate([(x[idx] > 0).reshape(-1) for x in preds]) if preds else np.asarray([], dtype=bool)
        g = np.concatenate([(x[idx] > 0).reshape(-1) for x in golds]) if golds else np.asarray([], dtype=bool)
        tp = int(np.logical_and(p, g).sum())
        fp = int(np.logical_and(p, ~g).sum())
        fn = int(np.logical_and(~p, g).sum())
        precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        out[head] = {"precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6), "tp": tp, "fp": fp, "fn": fn}
        if head in CORE:
            f1s.append(f1)
    out["mean_f1"] = round(float(np.mean(f1s)), 6) if f1s else 0.0
    return out


def _predict(model: Any, ds: MultiHeadDataset, args: argparse.Namespace, prefix: str, save: bool = True) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    preds, golds, rows = [], [], []
    out_dir = REPORT / f"image_only_multitask_proposal_v15_{prefix}_masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for idx in range(len(ds)):
            x, y = ds[idx]
            logits = model(x[None].to(args.device))
            prob = torch.sigmoid(logits)[0].cpu().numpy().astype(np.float32)
            pred = (prob >= args.threshold).astype(np.uint8)
            gold = y.numpy().astype(np.uint8)
            preds.append(pred)
            golds.append(gold)
            if save:
                row = ds.rows[idx]
                head_paths = {}
                prob_paths = {}
                for head in HEADS:
                    hidx = HEAD_INDEX[head]
                    mask_path = out_dir / f"{row['id']}_{head}.png"
                    prob_path = out_dir / f"{row['id']}_{head}_prob.png"
                    Image.fromarray((pred[hidx] * 255).astype(np.uint8)).save(mask_path)
                    Image.fromarray(np.clip(prob[hidx] * 255, 0, 255).astype(np.uint8)).save(prob_path)
                    head_paths[head] = _rel(mask_path)
                    prob_paths[head] = _rel(prob_path)
                source_size = float((row.get("image_size") or [args.train_size, args.train_size])[0] or args.train_size)
                scale = args.train_size / max(source_size, 1.0)
                boxes = []
                for box in row.get("boxes", []):
                    bbox = normalize_bbox(box.get("bbox"))
                    if bbox:
                        item = dict(box)
                        item["bbox"] = [int(round(v * scale)) for v in bbox]
                        item["coordinate_space"] = f"{args.train_size}x{args.train_size}"
                        boxes.append(item)
                rows.append({
                    "id": row["id"],
                    "image": row["image"],
                    "image_size": [args.train_size, args.train_size],
                    "pred_masks": head_paths,
                    "pred_probs": prob_paths,
                    "target_masks": row.get("targets"),
                    "boxes": boxes,
                    "source_integrity": _integrity(),
                    "route_trace": _trace("image_only_multitask_proposal_v15"),
                })
    return _head_metrics(preds, golds), rows


def train_proposal(args: argparse.Namespace) -> None:
    if torch is None:
        raise RuntimeError("torch is required")
    if not (DATA / "train.jsonl").exists():
        build_supervision(args)
    geom = audit_geometry(args)
    if not geom.get("passes_training_gate") and not args.allow_known_label_warnings:
        raise RuntimeError("image-only v15 label geometry audit failed; see reports/vlm/image_only_label_geometry_v15_audit.json")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_rows = load_jsonl(DATA / "train.jsonl")
    dev_rows = load_jsonl(DATA / "dev.jsonl")
    locked_rows = load_jsonl(DATA / "locked.jsonl")
    if args.overfit:
        train_rows = train_rows[: max(1, min(args.overfit_samples, len(train_rows)))]
        dev_rows = train_rows
        locked_rows = train_rows
    manifest = load_json(DATA / "manifest.json")
    split_report = manifest.get("split_report") or {}
    if not args.overfit and not bool(split_report.get("split_sanity_passed")) and not args.allow_small_train:
        raise RuntimeError(f"v15 split sanity failed: {split_report}")
    train_ds = MultiHeadDataset(train_rows, args.train_size, args.max_train)
    dev_ds = MultiHeadDataset(dev_rows, args.train_size, args.max_eval)
    locked_ds = MultiHeadDataset(locked_rows, args.train_size, args.max_eval)
    model = SmallUNet().to(args.device)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    pos_weights = torch.tensor([3.0, 6.0, 6.0, 1.3, 2.2, 8.0, 8.0], device=args.device).view(1, -1, 1, 1)
    loss_tail: list[float] = []
    best_dev = -1.0
    CKPT.mkdir(parents=True, exist_ok=True)
    for _epoch in range(args.epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(args.device), y.to(args.device)
            opt.zero_grad()
            logits = model(x)
            bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weights)
            loss = bce + 0.9 * _dice_loss(logits, y)
            loss.backward()
            opt.step()
            loss_tail.append(float(loss.detach().cpu()))
        dev_metrics, _ = _predict(model, dev_ds, args, "dev", save=False)
        dev_score = float(dev_metrics.get("mean_f1") or 0)
        if dev_score >= best_dev:
            best_dev = dev_score
            torch.save({"state_dict": model.state_dict(), "heads": HEADS, "train_size": args.train_size, "model_name": "SmallUNet_multihead_v15"}, CKPT / "model_best.pt")
    locked_metrics, pred_rows = _predict(model, locked_ds, args, "locked")
    dev_metrics, _ = _predict(model, dev_ds, args, "dev", save=False)
    write_jsonl(REPORT / "image_only_multitask_proposal_v15_locked_predictions.jsonl", pred_rows)
    report = {
        "task": "IMG-MOE-V15-P0-004" if args.overfit else "IMG-MOE-V15-P0-005",
        "model": "SmallUNet_multihead_bce_dice_v15",
        "input_contract": "raster_image_only",
        "train_count": len(train_ds),
        "dev_count": len(dev_ds),
        "locked_count": len(locked_ds),
        "epochs": args.epochs,
        "train_size": args.train_size,
        "overfit": bool(args.overfit),
        "dev": dev_metrics,
        "locked": locked_metrics,
        "best_dev_mean_f1": round(best_dev, 6),
        "beats_v14_proposal_mean_f1": locked_metrics.get("mean_f1", 0) > 0,
        "beats_rejected_v8_macro_f1": locked_metrics.get("mean_f1", 0) > 0.007207,
        "source_integrity": _integrity(),
        "loss_tail": loss_tail[-20:],
        "adopted": False,
        "adoption_reason": "proposal model must pass vectorized E2E gates before adoption",
    }
    if args.overfit:
        write_json(REPORT / "image_only_multitask_proposal_v15_overfit.json", report)
        _write_failure_gallery(REPORT / "image_only_multitask_proposal_v15_overfit_gallery.html", pred_rows[:20], report)
        update_todo_remove(["IMG-MOE-V15-P0-004"])
    else:
        write_json(REPORT / "image_only_multitask_proposal_v15_eval.json", report)
        _write_failure_gallery(REPORT / "image_only_multitask_proposal_v15_failure_gallery.html", pred_rows[:20], report)
        update_todo_remove(["IMG-MOE-V15-P0-005"])


def _components(binary: np.ndarray, min_area: int, max_area_ratio: float = 0.6) -> list[dict[str, Any]]:
    if morphology is not None:
        binary = morphology.remove_small_objects(binary.astype(bool), min_size=min_area)
    if measure is None:
        ys, xs = np.where(binary)
        if len(xs) < min_area:
            return []
        return [{"bbox": [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)], "area": int(len(xs)), "polygon": []}]
    labels = measure.label(binary, connectivity=2)
    out = []
    image_area = max(binary.shape[0] * binary.shape[1], 1)
    for region in measure.regionprops(labels):
        if region.area < min_area or region.area / image_area > max_area_ratio:
            continue
        minr, minc, maxr, maxc = region.bbox
        contours = measure.find_contours(labels == region.label, 0.5)
        poly: list[list[float]] = []
        if contours:
            contour = max(contours, key=len)
            step = max(1, len(contour) // 64)
            poly = [[round(float(c), 2), round(float(r), 2)] for r, c in contour[::step]]
        out.append({"bbox": [int(minc), int(minr), int(maxc), int(maxr)], "area": int(region.area), "polygon": poly})
    return out


def _proposal_components(row: dict[str, Any], cls: str) -> list[dict[str, Any]]:
    prob_path = (row.get("pred_probs") or {}).get(cls)
    mask_path = (row.get("pred_masks") or {}).get(cls)
    path = prob_path or mask_path
    if not path:
        return []
    arr = np.asarray(Image.open(_abs(path)).convert("L"), dtype=np.uint8)
    if prob_path:
        thresholds = {"wall": 70, "room": 95, "room_boundary": 90, "opening": 80, "window": 80, "symbol": 120, "text": 120}
        binary = arr >= thresholds.get(cls, 100)
    else:
        binary = arr > 0
    if cls == "wall":
        min_area, max_ratio = 18, 0.35
    elif cls == "room":
        min_area, max_ratio = 150, 0.75
    elif cls in {"symbol", "text"}:
        min_area, max_ratio = 8, 0.08
    else:
        min_area, max_ratio = 8, 0.18
    return _components(binary, min_area, max_ratio)


def _nms(props: list[dict[str, Any]], threshold: float = 0.5) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for prop in sorted(props, key=lambda p: float(p.get("confidence") or 0), reverse=True):
        bbox = normalize_bbox(prop.get("bbox"))
        if not bbox:
            continue
        if all(bbox_iou(bbox, normalize_bbox(k.get("bbox")) or [0, 0, 0, 0]) < threshold for k in kept):
            kept.append(prop)
    return kept


def vectorize(args: argparse.Namespace) -> None:
    rows = load_jsonl(REPORT / "image_only_multitask_proposal_v15_locked_predictions.jsonl")
    if not rows:
        raise RuntimeError("missing v15 locked predictions; run train first")
    proposal_rows = []
    metrics = {}
    full_image_suppressed = Counter()
    for row in rows:
        props = []
        iw, ih = row.get("image_size") or [args.train_size, args.train_size]
        image_area = max(iw * ih, 1)
        for cls in CORE:
            cls_props = []
            for i, comp in enumerate(_proposal_components(row, cls)):
                area_ratio = bbox_area(comp["bbox"]) / image_area
                if cls in {"symbol", "text"} and area_ratio > 0.08:
                    full_image_suppressed[cls] += 1
                    continue
                if cls in {"opening", "window"} and area_ratio > 0.25:
                    full_image_suppressed[cls] += 1
                    continue
                confidence = min(0.99, 0.25 + comp["area"] / max(image_area * (0.08 if cls in {"symbol", "text"} else 0.25), 1))
                cls_props.append({
                    "id": f"{row['id']}_{cls}_{i}",
                    "family": FAMILY[cls],
                    "semantic_type": cls,
                    "bbox": comp["bbox"],
                    "polygon": comp.get("polygon", []),
                    "confidence": round(float(confidence), 4),
                    "proposal_source": "raster_multihead_vectorized_v15",
                })
            props.extend(_nms(cls_props, 0.35 if cls in {"symbol", "text"} else 0.55))
        proposal_rows.append({"id": row["id"], "image": row["image"], "image_size": row.get("image_size", [args.train_size, args.train_size]), "proposals": props, "boxes": row.get("boxes", []), "source_integrity": _integrity(), "route_trace": _trace("vectorize_image_only_proposals_v15")})
    for cls in CORE:
        preds, golds = [], []
        for row in proposal_rows:
            preds.extend([p for p in row.get("proposals") or [] if p.get("semantic_type") == cls])
            golds.extend([{"bbox": g.get("bbox"), "class": cls} for g in row.get("boxes") or [] if g.get("class") == cls])
        tp, pc, gc, fp, miss = match_counts(preds, golds, 0.25 if cls in {"wall", "opening", "window"} else 0.3)
        precision, recall = tp / max(pc, 1), tp / max(gc, 1)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        metrics[cls] = {"tp": tp, "predicted": pc, "gold": gc, "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6), "false_positive_examples": fp[:10], "miss_examples": miss[:10]}
    mean_f1 = float(np.mean([m["f1"] for m in metrics.values()])) if metrics else 0.0
    report = {
        "task": "IMG-MOE-V15-P1-006",
        "proposal_metrics": metrics,
        "proposal_mean_f1": round(mean_f1, 6),
        "full_image_component_suppressed": dict(full_image_suppressed),
        "source_integrity": _integrity(),
        "adopted": mean_f1 > 0.05 and all(metrics.get(c, {}).get("f1", 0) > 0 for c in ["wall", "room", "symbol", "text"]),
    }
    write_json(REPORT / "image_only_proposals_v15_eval.json", report)
    write_jsonl(REPORT / "image_only_proposals_v15_cases.jsonl", proposal_rows)
    _write_proposal_review(REPORT / "image_only_proposals_v15_review.html", proposal_rows[:30], metrics)
    update_todo_remove(["IMG-MOE-V15-P1-006"])


def apply_experts(args: argparse.Namespace) -> None:
    rows = load_jsonl(REPORT / "image_only_proposals_v15_cases.jsonl")
    proposal_eval = load_json(REPORT / "image_only_proposals_v15_eval.json")
    family_enabled = {cls: (proposal_eval.get("proposal_metrics") or {}).get(cls, {}).get("f1", 0) > 0 for cls in CORE}
    out = []
    family_counts: Counter[str] = Counter()
    disabled_counts: Counter[str] = Counter()
    for row in rows:
        nodes = []
        for prop in row.get("proposals") or []:
            cls = prop.get("semantic_type")
            if not family_enabled.get(cls, False):
                disabled_counts[str(cls)] += 1
                continue
            family_counts[prop.get("family", "unknown")] += 1
            nodes.append({
                "id": prop["id"],
                "family": prop["family"],
                "semantic_type": prop["semantic_type"],
                "confidence": prop.get("confidence", 0.5),
                "geometry": {"bbox": prop["bbox"], "polygon": prop.get("polygon", [])},
                "metadata": {"proposal_source": "raster_multihead_vectorized_v15", "expert_adapter": "image_only_v15_proposal_gate_adapter"},
            })
        out.append({"id": row["id"], "image": row["image"], "image_size": row.get("image_size"), "source_integrity": _integrity(), "route_trace": _trace("apply_moe_experts_to_image_only_proposals_v15"), "scene_graph": {"nodes": nodes, "edges": []}, "proposals": row.get("proposals", [])})
    write_jsonl(REPORT / "image_only_moe_expert_predictions_v15.jsonl", out)
    audit = {
        "task": "IMG-MOE-V15-P1-007",
        "rows": len(out),
        "family_counts": dict(family_counts),
        "disabled_proposal_families": dict(disabled_counts),
        "adapter_mode": "raster proposals only; no parser raw labels",
        "source_integrity": _integrity(),
    }
    write_json(REPORT / "image_only_moe_expert_adapter_v15_audit.json", audit)
    update_todo_remove(["IMG-MOE-V15-P1-007"])


def relations(args: argparse.Namespace) -> None:
    rows = load_jsonl(REPORT / "image_only_moe_expert_predictions_v15.jsonl")
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
                    edges.append({"source": room["id"], "target": node["id"], "relation": "contains", "confidence": 0.55, "source_expert": "image_only_geometry_relation_v15"})
                    rel_counts["contains"] += 1
        new = dict(row)
        new["scene_graph"] = {"nodes": nodes, "edges": edges}
        new["route_trace"] = _trace("image_only_relation_fusion_v15")
        out.append(new)
    write_jsonl(REPORT / "image_only_moe_predictions_v15.jsonl", out)
    write_json(REPORT / "image_only_relation_fusion_v15_eval.json", {"rows": len(out), "relation_counts": dict(rel_counts), "mode": "image_only_geometry_contains_v15", "source_integrity": _integrity()})


def evaluate(args: argparse.Namespace) -> None:
    predictions = load_jsonl(args.predictions)
    if not predictions and Path(args.predictions).name != "image_only_moe_predictions_v15.jsonl":
        predictions = load_jsonl(REPORT / "image_only_moe_predictions_v15.jsonl")
    contract = load_json(ROOT / "configs/vlm/image_only_moe_contract_v1.json")
    gate = validate_rows(predictions, contract)
    proposals = load_json(REPORT / "image_only_proposals_v15_eval.json")
    experts = load_json(REPORT / "image_only_moe_expert_adapter_v15_audit.json")
    train_eval = load_json(REPORT / "image_only_multitask_proposal_v15_eval.json")
    overfit = load_json(REPORT / "image_only_multitask_proposal_v15_overfit.json")
    node_count = sum(len((r.get("scene_graph") or {}).get("nodes") or []) for r in predictions)
    edge_count = sum(len((r.get("scene_graph") or {}).get("edges") or []) for r in predictions)
    metrics = proposals.get("proposal_metrics") or {}
    mean_f1 = float(proposals.get("proposal_mean_f1") or 0.0)
    floors = {cls: float((metrics.get(cls) or {}).get("f1") or 0.0) > 0.0 for cls in ["wall", "room", "symbol", "text"]}
    adopted = bool(gate.get("passed")) and mean_f1 > 0.05 and all(floors.values()) and node_count > 0
    report = {
        "task": "IMG-MOE-V15-P1-008",
        "source_integrity_gate": gate,
        "split_sanity": (load_json(DATA / "manifest.json").get("split_report") or {}),
        "label_sanity": load_json(REPORT / "image_only_label_geometry_v15_audit.json"),
        "overfit_proof": overfit,
        "proposal_training": train_eval,
        "proposal_metrics": metrics,
        "expert_adapter": experts,
        "final_scene_graph": {"rows": len(predictions), "nodes": node_count, "edges": edge_count},
        "proposal_mean_f1": round(mean_f1, 6),
        "proposal_floors": floors,
        "baselines": {"v14_proposal_mean_f1": 0.0, "v8_rejected_macro_f1": 0.007207, "parser_assisted_v13_v14_is_oracle_only": True},
        "adopted": adopted,
        "adoption_reason": "passes strict source and proposal gates" if adopted else "not adopted: strict image-only proposal quality remains below adoption gate",
    }
    write_json(REPORT / "image_only_moe_e2e_v15_eval.json", report)
    write_json(REPORT / "image_only_moe_e2e_v15_ablation_dashboard.json", report)
    write_jsonl(REPORT / "image_only_moe_e2e_v15_cases.jsonl", predictions)
    update_todo_remove(["IMG-MOE-V15-P1-008"])


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


def render(args: argparse.Namespace) -> None:
    rows = load_jsonl(args.predictions) or load_jsonl(REPORT / "image_only_moe_predictions_v15.jsonl")
    proposal_rows = {r["id"]: r for r in load_jsonl(REPORT / "image_only_proposals_v15_cases.jsonl") if r.get("id")}
    eval_report = load_json(REPORT / "image_only_moe_e2e_v15_eval.json")
    cards = []
    for row in rows[: args.max_samples]:
        proposal = proposal_rows.get(row.get("id"), {})
        cards.append(
            f"<section><h2>{row.get('id')}</h2><p>source_mode=image_only_raster_moe adopted={bool(eval_report.get('adopted'))} nodes={len((row.get('scene_graph') or {}).get('nodes') or [])}</p>"
            f"<div class='grid'><figure><img src='{_img_uri(row.get('image'))}'><figcaption>original raster input</figcaption></figure>"
            f"<figure><img src='{_overlay_scene(row)}'><figcaption>final image-only MoE nodes</figcaption></figure>"
            f"<figure><img src='{_overlay_proposals(proposal)}'><figcaption>vectorized proposals</figcaption></figure>"
            f"<figure><pre>{json.dumps((eval_report.get('proposal_metrics') or {}), ensure_ascii=False, indent=2)[:4000]}</pre></figure></div></section>"
        )
    html = f"<!doctype html><meta charset='utf-8'><title>image-only MoE v15</title><style>body{{font-family:Arial,sans-serif;margin:24px}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}}img{{width:100%;border:1px solid #999;background:white}}pre{{background:#f5f5f5;padding:12px;overflow:auto;white-space:pre-wrap}}figure{{margin:0}}section{{border-top:1px solid #ddd;padding-top:18px}}</style><h1>CadStruct image-only MoE v15</h1><pre>{json.dumps(eval_report, ensure_ascii=False, indent=2)[:8000]}</pre>{''.join(cards)}"
    pack = REPORT / "visual_demo_image_only_moe_v15/review_pack/index.html"
    pack.parent.mkdir(parents=True, exist_ok=True)
    pack.write_text(html, encoding="utf-8")
    fail = REPORT / "visual_demo_image_only_moe_v15/failure_gallery.html"
    fail.write_text(html.replace("CadStruct image-only MoE v15", "CadStruct image-only MoE v15 failure-first gallery"), encoding="utf-8")
    write_json(REPORT / "visual_demo_image_only_moe_v15/coverage_audit.json", {"task": "IMG-MOE-V15-P2-009", "rows": len(rows), "rendered": min(len(rows), args.max_samples), "original_raster_rendered": all(bool(_img_uri(r.get("image"))) for r in rows[: args.max_samples]), "source_mode": "image_only_raster_moe", "adopted": bool(eval_report.get("adopted"))})
    update_todo_remove(["IMG-MOE-V15-P2-009"])


def _overlay_proposals(row: dict[str, Any]) -> str:
    p = _abs(row.get("image") or "")
    if not p.exists():
        return ""
    img = Image.open(p).convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")
    colors = {"boundary": (220, 40, 40, 160), "space": (235, 190, 35, 95), "symbol": (130, 80, 190, 160), "text": (20, 20, 20, 170)}
    sx = img.width / max(float((row.get("image_size") or [img.width, img.height])[0]), 1.0)
    sy = img.height / max(float((row.get("image_size") or [img.width, img.height])[1]), 1.0)
    for prop in row.get("proposals") or []:
        bbox = normalize_bbox(prop.get("bbox"))
        if bbox:
            color = colors.get(prop.get("family"), (0, 120, 200, 150))
            draw.rectangle([bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy], outline=color, width=2)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def claim_docs(args: argparse.Namespace) -> None:
    eval_report = load_json(REPORT / "image_only_moe_e2e_v15_eval.json")
    proposal = load_json(REPORT / "image_only_proposals_v15_eval.json")
    roadmap = f"""# CadStruct Image-Only MoE Roadmap v15

v15 enforces the real project contract: the MoE front end receives only a raster floorplan image. CubiCasa SVG/parser geometry is used only as offline supervision and locked evaluation gold.

Current status:
- Image-only adopted: `{bool(eval_report.get('adopted'))}`
- Proposal mean F1: `{eval_report.get('proposal_mean_f1')}`
- Parser-assisted v13/v14 remains oracle/debug only, not model-credit evidence.

The main remaining bottleneck is raster proposal generation. Expert models can only improve labels after credible wall/room/symbol/text proposals exist.
"""
    claim = f"""# CadStruct Paper Claim Boundary v15

Paper-main image-only claims must use `reports/vlm/image_only_moe_e2e_v15_eval.json` or later streams that pass `configs/vlm/image_only_moe_contract_v1.json`.

v15 adoption decision: `{bool(eval_report.get('adopted'))}`.

Rejected or limited results must be reported as recovery experiments:
- v14 collapse: single mutually exclusive mask and tiny train split.
- v15 correction: adaptive split, multi-head targets, geometry audit, overfit-first training, and family-specific vectorization.
- Current proposal metrics: `{json.dumps(proposal.get('proposal_metrics') or {}, ensure_ascii=False)[:3000]}`
"""
    (ROOT / "docs/cadstruct-image-only-moe-roadmap-v15.md").write_text(roadmap, encoding="utf-8")
    (ROOT / "docs/cadstruct-paper-claim-boundary-v15.md").write_text(claim, encoding="utf-8")
    write_json(REPORT / "image_only_claim_gate_v15.json", {"task": "IMG-MOE-V15-P2-010", "image_only_adopted": bool(eval_report.get("adopted")), "proposal_mean_f1": eval_report.get("proposal_mean_f1"), "parser_assisted_metrics_allowed_as_main_claim": False, "advisor_figures_require_source_mode_badge": True})
    update_todo_remove(["IMG-MOE-V15-P2-010"])


def _write_label_qa(path: Path, rows: list[dict[str, Any]], audit: dict[str, Any]) -> None:
    cards = []
    for row in rows:
        cards.append(f"<section><h2>{row['id']}</h2><div class='grid'><figure><img src='{_img_uri(row['image'])}'><figcaption>raster input</figcaption></figure><figure><img src='{_head_overlay_uri(row)}'><figcaption>multi-head offline supervision overlay</figcaption></figure></div></section>")
    path.write_text(f"<!doctype html><meta charset='utf-8'><title>v15 target QA</title><style>body{{font-family:Arial,sans-serif;margin:24px}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}}img{{width:100%;border:1px solid #999;background:white}}pre{{white-space:pre-wrap;background:#f5f5f5;padding:12px}}</style><h1>v15 image-only supervision QA</h1><pre>{json.dumps(audit, ensure_ascii=False, indent=2)}</pre>{''.join(cards)}", encoding="utf-8")


def _write_failure_gallery(path: Path, rows: list[dict[str, Any]], report: dict[str, Any]) -> None:
    cards = []
    for row in rows:
        imgs = "".join(f"<figure><img src='{_img_uri(path)}'><figcaption>{head}</figcaption></figure>" for head, path in (row.get("pred_masks") or {}).items())
        cards.append(f"<section><h2>{row['id']}</h2><div class='grid'><figure><img src='{_img_uri(row['image'])}'><figcaption>raster input</figcaption></figure>{imgs}</div></section>")
    path.write_text(f"<!doctype html><meta charset='utf-8'><title>v15 proposal gallery</title><style>body{{font-family:Arial,sans-serif;margin:24px}}.grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}}img{{width:100%;border:1px solid #999;background:white}}pre{{white-space:pre-wrap;background:#f5f5f5;padding:12px}}figure{{margin:0}}</style><pre>{json.dumps(report, ensure_ascii=False, indent=2)}</pre>{''.join(cards)}", encoding="utf-8")


def _write_proposal_review(path: Path, rows: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    cards = [f"<section><h2>{r['id']}</h2><p>proposals={len(r.get('proposals') or [])}</p><img src='{_overlay_proposals(r)}'></section>" for r in rows]
    path.write_text(f"<!doctype html><meta charset='utf-8'><title>v15 proposal review</title><style>body{{font-family:Arial,sans-serif;margin:24px}}img{{max-width:720px;width:100%;border:1px solid #999;background:white}}pre{{white-space:pre-wrap;background:#f5f5f5;padding:12px}}</style><pre>{json.dumps(metrics, ensure_ascii=False, indent=2)}</pre>{''.join(cards)}", encoding="utf-8")


def run_all(args: argparse.Namespace) -> None:
    build_supervision(args)
    audit_geometry(args)
    overfit_args = argparse.Namespace(**vars(args))
    overfit_args.overfit = True
    overfit_args.epochs = max(args.overfit_epochs, args.epochs)
    overfit_args.max_train = min(args.max_train, args.overfit_samples)
    train_proposal(overfit_args)
    full_args = argparse.Namespace(**vars(args))
    full_args.overfit = False
    train_proposal(full_args)
    vectorize(args)
    apply_experts(args)
    relations(args)
    evaluate(args)
    render(args)
    claim_docs(args)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["build-supervision", "build-targets", "audit-geometry", "train-proposal", "vectorize", "apply-experts", "relations", "evaluate", "render", "claim-docs", "run-all"])
    p.add_argument("--sizes", default="512,1024")
    p.add_argument("--limit", type=int, default=512)
    p.add_argument("--qa-samples", type=int, default=24)
    p.add_argument("--seed", type=int, default=20260507)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--overfit-epochs", type=int, default=12)
    p.add_argument("--overfit", action="store_true")
    p.add_argument("--overfit-samples", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-train", type=int, default=256)
    p.add_argument("--max-eval", type=int, default=64)
    p.add_argument("--train-size", type=int, default=256)
    p.add_argument("--threshold", type=float, default=0.45)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--allow-small-train", action="store_true")
    p.add_argument("--allow-known-label-warnings", action="store_true")
    p.add_argument("--device", default="cuda" if torch is not None and torch.cuda.is_available() else "cpu")
    p.add_argument("--predictions", default=str(REPORT / "image_only_moe_predictions_v15.jsonl"))
    p.add_argument("--max-samples", type=int, default=24)
    return p


def main() -> None:
    args = parser().parse_args()
    actions = {
        "build-supervision": build_supervision,
        "build-targets": build_supervision,
        "audit-geometry": audit_geometry,
        "train-proposal": train_proposal,
        "vectorize": vectorize,
        "apply-experts": apply_experts,
        "relations": relations,
        "evaluate": evaluate,
        "render": render,
        "claim-docs": claim_docs,
        "run-all": run_all,
    }
    actions[args.command](args)


if __name__ == "__main__":
    main()
