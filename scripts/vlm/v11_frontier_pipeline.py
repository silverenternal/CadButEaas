#!/usr/bin/env python3
"""CadStruct-MoE frontier recovery v11.

v11 is an isolated frontier branch. It audits and smoke-tests modern floor-plan
representations without overwriting the existing v7/v8 expert pipeline.
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import math
import re
import time
import xml.etree.ElementTree as ET
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
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import precision_recall_fscore_support
except Exception:  # pragma: no cover
    RandomForestClassifier = None  # type: ignore[assignment]
    precision_recall_fscore_support = None  # type: ignore[assignment]

try:
    from scripts.vlm.v8_raster_e2e_utils import (
        ROOT,
        bbox_iou,
        extract_gold_items,
        load_json,
        load_jsonl,
        match_counts,
        normalize_bbox,
        row_image_size,
        sample_key,
        split_rows_with_locked,
        update_todo_remove,
        write_json,
        write_jsonl,
    )
    from scripts.vlm.v10_raster_pipeline import CLASS_TO_ID, CORE_CLASSES, DET_CLASSES, ID_TO_CLASS, _gold_class, _gold_polygon, v9_gold_items
except Exception:  # pragma: no cover
    from v8_raster_e2e_utils import (  # type: ignore
        ROOT,
        bbox_iou,
        extract_gold_items,
        load_json,
        load_jsonl,
        match_counts,
        normalize_bbox,
        row_image_size,
        sample_key,
        split_rows_with_locked,
        update_todo_remove,
        write_json,
        write_jsonl,
    )
    from v10_raster_pipeline import CLASS_TO_ID, CORE_CLASSES, DET_CLASSES, ID_TO_CLASS, _gold_class, _gold_polygon, v9_gold_items  # type: ignore


REPORT_DIR = ROOT / "reports/vlm"
DATA_DIR = ROOT / "datasets/frontier_targets_v11"
CHECKPOINT_DIR = ROOT / "checkpoints/model_v11"
LABEL_SIZE = 512
TASK_IDS = [f"FRONTIER-V11-T{i}" for i in range(1, 12)]


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


def _read_rows(limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("train", "dev", "smoke"):
        rows.extend(load_jsonl(ROOT / "datasets/cadstruct_cubicasa5k_moe" / f"{split}.jsonl"))
    rows = [row for row in rows if row.get("image_path") and row.get("annotation_path")]
    return rows[:limit] if limit else rows


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


def _scale_points(points: list[list[float]], original: tuple[int, int], size: int = LABEL_SIZE) -> list[list[int]]:
    t = _transform(original, size)
    out: list[list[int]] = []
    for x, y in points:
        out.append(
            [
                int(max(0, min(size - 1, round(float(x) * t["scale"] + t["xpad"])))),
                int(max(0, min(size - 1, round(float(y) * t["scale"] + t["ypad"])))),
            ]
        )
    return out


def _bbox_from_points(points: list[list[float]]) -> list[float] | None:
    if not points:
        return None
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def _numeric(value: Any) -> bool:
    return any(ch.isdigit() for ch in str(value or ""))


def _img_uri(path: str | Path | None) -> str:
    if not path:
        return ""
    p = _abs(path)
    if not p.exists():
        return ""
    data = p.read_bytes()
    suffix = p.suffix.lower()
    mime = "image/svg+xml" if suffix == ".svg" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


def _safe_image(path: str | Path, size: int = LABEL_SIZE) -> Image.Image:
    p = _abs(path)
    if not p.exists():
        return Image.new("RGB", (size, size), "white")
    img = Image.open(p).convert("RGB")
    img.thumbnail((size, size), Image.Resampling.BILINEAR)
    out = Image.new("RGB", (size, size), "white")
    out.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return out


def _poly_valid(poly: list[list[float]]) -> bool:
    if len(poly) < 3:
        return False
    bbox = _bbox_from_points(poly)
    if not bbox:
        return False
    return (bbox[2] - bbox[0]) > 1 and (bbox[3] - bbox[1]) > 1


def _segment_bbox(p1: list[float], p2: list[float], pad: float = 2.0) -> list[float]:
    return [min(p1[0], p2[0]) - pad, min(p1[1], p2[1]) - pad, max(p1[0], p2[0]) + pad, max(p1[1], p2[1]) + pad]


def _items_by_family(row: dict[str, Any], scaled: bool = True) -> dict[str, list[dict[str, Any]]]:
    original = row_image_size(row)
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in v9_gold_items(row):
        cls = _gold_class(item)
        bbox = normalize_bbox(item.get("bbox"))
        poly = _gold_polygon(item)
        if not bbox:
            continue
        sb = _scale_bbox(bbox, original) if scaled else [float(v) for v in bbox]
        sp = _scale_points(poly, original) if scaled and poly else poly
        rec = {
            "class": cls,
            "bbox": sb,
            "polygon": sp,
            "text": item.get("text") or "",
            "raw_label": item.get("raw_label") or item.get("semantic_type") or item.get("room_type") or item.get("symbol_type") or item.get("text_type"),
            "source": "offline_svg_label",
        }
        out[cls].append(rec)
    return out


def _extract_raw_svg_text(svg_path: str | Path) -> dict[str, Any]:
    p = _abs(svg_path)
    if not p.exists():
        return {"exists": False, "texts": [], "numeric_texts": []}
    texts: list[dict[str, Any]] = []
    try:
        root = ET.parse(p).getroot()
    except ET.ParseError as exc:
        return {"exists": True, "parse_error": str(exc), "texts": [], "numeric_texts": []}
    for elem in root.iter():
        tag = elem.tag.split("}")[-1].lower()
        if tag not in {"text", "tspan"}:
            continue
        value = "".join(elem.itertext()).strip()
        if not value:
            continue
        x = elem.attrib.get("x")
        y = elem.attrib.get("y")
        texts.append({"tag": tag, "text": value, "x": x, "y": y, "class": elem.attrib.get("class") or ""})
    return {"exists": True, "texts": texts, "numeric_texts": [t for t in texts if _numeric(t.get("text"))]}


def audit_raw_text(args: argparse.Namespace) -> dict[str, Any]:
    start = time.time()
    rows = _read_rows(args.limit)
    converted_text = 0
    converted_numeric = 0
    raw_text = 0
    raw_numeric = 0
    examples = []
    missing_svg = 0
    for row in rows:
        for item in (row.get("expected_json") or {}).get("text_candidates") or []:
            converted_text += 1
            if _numeric(item.get("text")):
                converted_numeric += 1
        raw = _extract_raw_svg_text(row.get("annotation_path") or "")
        if not raw.get("exists"):
            missing_svg += 1
        raw_text += len(raw.get("texts") or [])
        raw_numeric += len(raw.get("numeric_texts") or [])
        if raw.get("numeric_texts") and len(examples) < 20:
            examples.append({"sample": sample_key(row.get("annotation_path")), "numeric_texts": raw["numeric_texts"][:8]})
    report = {
        "task": "FRONTIER-V11-T1",
        "rows_checked": len(rows),
        "missing_svg_files": missing_svg,
        "converted_text_count": converted_text,
        "converted_numeric_text_count": converted_numeric,
        "raw_svg_text_count": raw_text,
        "raw_svg_numeric_text_count": raw_numeric,
        "numeric_text_evaluable_on_cubicasa": raw_numeric > 0 or converted_numeric > 0,
        "numeric_text_source": "raw_svg_text" if raw_numeric else ("converted_json" if converted_numeric else "not_available"),
        "examples": examples,
        "runtime_ms": round((time.time() - start) * 1000, 3),
    }
    write_json(REPORT_DIR / "cubicasa_raw_text_v11_audit.json", report)
    return report


def _target_record(row: dict[str, Any], split: str) -> dict[str, Any]:
    original = row_image_size(row)
    families = _items_by_family(row, scaled=True)
    wall_edges = []
    for idx, wall in enumerate(families.get("wall", [])):
        poly = wall.get("polygon") or []
        if len(poly) < 2:
            continue
        for sidx in range(len(poly)):
            p1 = poly[sidx]
            p2 = poly[(sidx + 1) % len(poly)]
            length = math.dist(p1, p2)
            if length < 3:
                continue
            wall_edges.append(
                {
                    "id": f"wall_{idx}_edge_{sidx}",
                    "p1": p1,
                    "p2": p2,
                    "bbox": _segment_bbox(p1, p2, 2.0),
                    "length": round(length, 3),
                    "angle": round(math.atan2(p2[1] - p1[1], p2[0] - p1[0]), 6),
                    "source": "offline_svg_label",
                }
            )
    polygon_sequences = []
    for cls in ("room", "opening", "window", "wall"):
        for item in families.get(cls, []):
            poly = item.get("polygon") or []
            if not _poly_valid(poly):
                continue
            tokens = [f"CLS_{cls.upper()}", "POLY"]
            for x, y in poly[:64]:
                tokens.extend([f"X{int(x)}", f"Y{int(y)}"])
            tokens.append("SEP")
            polygon_sequences.append({"class": cls, "tokens": tokens, "length": len(tokens)})
    source_key = sample_key(row.get("annotation_path") or row.get("image_path"))
    return {
        "id": f"{split}_{source_key}",
        "source_key": source_key,
        "split": split,
        "image": row.get("image_path"),
        "annotation_path": row.get("annotation_path"),
        "image_size": list(original),
        "label_size": LABEL_SIZE,
        "transform": _transform(original),
        "room_polygons": families.get("room", []),
        "wall_edges": wall_edges,
        "openings": families.get("opening", []),
        "windows": families.get("window", []),
        "symbols": families.get("symbol", []),
        "text_boxes": families.get("text", []),
        "numeric_text_boxes": [t for t in families.get("text", []) if _numeric(t.get("text"))],
        "polygon_sequences": polygon_sequences,
        "label_source": "offline_svg_gold_only",
        "svg_candidate_ids_used": False,
    }


def build_targets(args: argparse.Namespace) -> dict[str, Any]:
    start = time.time()
    rows = _read_rows(args.limit)
    splits = split_rows_with_locked(rows, seed=args.seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, list[dict[str, Any]]] = {}
    counts: Counter[str] = Counter()
    invalid_polygons = 0
    for split, split_rows in splits.items():
        out_rows = [_target_record(row, split) for row in split_rows]
        for rec in out_rows:
            counts.update(
                {
                    "room_polygons": len(rec["room_polygons"]),
                    "wall_edges": len(rec["wall_edges"]),
                    "openings": len(rec["openings"]),
                    "windows": len(rec["windows"]),
                    "symbols": len(rec["symbols"]),
                    "text_boxes": len(rec["text_boxes"]),
                    "numeric_text_boxes": len(rec["numeric_text_boxes"]),
                    "polygon_sequences": len(rec["polygon_sequences"]),
                }
            )
            invalid_polygons += sum(1 for p in rec["room_polygons"] if not _poly_valid(p.get("polygon") or []))
        write_jsonl(DATA_DIR / f"{split}.jsonl", out_rows)
        summary[split] = out_rows
    overlaps = {
        "train_dev": len({r["source_key"] for r in summary.get("train", [])} & {r["source_key"] for r in summary.get("dev", [])}),
        "train_locked": len({r["source_key"] for r in summary.get("train", [])} & {r["source_key"] for r in summary.get("locked", [])}),
        "dev_locked": len({r["source_key"] for r in summary.get("dev", [])} & {r["source_key"] for r in summary.get("locked", [])}),
    }
    text_audit = load_json(REPORT_DIR / "cubicasa_raw_text_v11_audit.json", {}) or audit_raw_text(args)
    audit = {
        "task": "FRONTIER-V11-T1",
        "rows": {split: len(rows_) for split, rows_ in summary.items()},
        "overlaps": overlaps,
        "counts": dict(counts),
        "invalid_room_polygons": invalid_polygons,
        "numeric_text_evaluable_on_cubicasa": bool(text_audit.get("numeric_text_evaluable_on_cubicasa")),
        "raw_text_audit": _rel(REPORT_DIR / "cubicasa_raw_text_v11_audit.json"),
        "source_integrity": {"label_source": "offline_svg_gold_only", "inference_input": "image_only", "svg_candidate_ids_used": False},
        "acceptance": {
            "overlap_zero": all(v == 0 for v in overlaps.values()),
            "core_targets_nonzero": all(counts.get(k, 0) > 0 for k in ["room_polygons", "wall_edges", "openings", "windows", "symbols", "text_boxes"]),
            "audit_passed": all(v == 0 for v in overlaps.values()) and counts.get("room_polygons", 0) > 0 and counts.get("wall_edges", 0) > 0,
        },
        "runtime_ms": round((time.time() - start) * 1000, 3),
    }
    write_json(REPORT_DIR / "frontier_targets_v11_audit.json", audit)
    _write_alignment_html(summary.get("locked", [])[:20], audit)
    update_todo_remove(["FRONTIER-V11-T1"])
    return audit


def _write_alignment_html(rows: list[dict[str, Any]], audit: dict[str, Any]) -> None:
    cards = []
    for row in rows:
        img = _safe_image(row["image"])
        overlay = img.convert("RGBA")
        draw = ImageDraw.Draw(overlay, "RGBA")
        for room in row.get("room_polygons") or []:
            poly = [tuple(p) for p in room.get("polygon") or []]
            if len(poly) >= 3:
                draw.polygon(poly, fill=(236, 190, 70, 45), outline=(196, 136, 20, 210))
        for edge in row.get("wall_edges") or []:
            draw.line([tuple(edge["p1"]), tuple(edge["p2"])], fill=(210, 30, 30, 210), width=3)
        for box in (row.get("text_boxes") or []) + (row.get("symbols") or []):
            x1, y1, x2, y2 = [int(v) for v in box["bbox"]]
            draw.rectangle((x1, y1, x2, y2), outline=(20, 20, 20, 220), width=2)
        buf = BytesIO()
        overlay.save(buf, format="PNG")
        uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        cards.append(
            f"<section><h2>{row['id']}</h2><p>source_mode=target_audit geometry_source=offline_svg_gold_only</p>"
            f"<div class='grid'><figure><img src='{_img_uri(row['image'])}'><figcaption>input raster</figcaption></figure>"
            f"<figure><img src='{uri}'><figcaption>gold target overlay</figcaption></figure></div></section>"
        )
    html = f"""<!doctype html><meta charset="utf-8"><title>v11 target alignment</title>
<style>body{{font-family:Arial,sans-serif;margin:24px;color:#222}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}}img{{width:100%;border:1px solid #aaa;background:white}}pre{{background:#f4f4f4;padding:12px;overflow:auto}}figure{{margin:0}}figcaption{{font-size:13px}}</style>
<h1>v11 frontier target alignment</h1><pre>{json.dumps(audit, ensure_ascii=False, indent=2)}</pre>{''.join(cards)}"""
    (REPORT_DIR / "frontier_targets_v11_alignment.html").write_text(html, encoding="utf-8")


def official_baseline(args: argparse.Namespace) -> dict[str, Any]:
    deps = {name: importlib.util.find_spec(name) is not None for name in ["torch", "cv2", "skimage", "shapely"]}
    repo_candidates = [ROOT / "external/cubicasa5k", ROOT / "third_party/cubicasa5k", ROOT / "cubicasa5k"]
    vendored = [str(p.relative_to(ROOT)) for p in repo_candidates if p.exists()]
    audit = {
        "task": "FRONTIER-V11-T2",
        "official_repo": "https://github.com/cubicasa/cubicasa5k",
        "license_audit": "repository must be reviewed before vendoring; this run does not copy external code",
        "local_repo_vendored": bool(vendored),
        "vendored_paths": vendored,
        "dependencies_available": deps,
        "target_adapter_available": (DATA_DIR / "train.jsonl").exists(),
        "smoke_run_status": "blocked_no_vendored_official_code" if not vendored else "adapter_ready_manual_command_required",
        "full_training_budget_note": "official reproduction should be run as a pinned external baseline with CUDA budget; v11 does not overclaim without it",
        "adopted": False,
    }
    write_json(REPORT_DIR / "official_cubicasa_reproduction_v11.json", audit)
    write_json(
        REPORT_DIR / "official_cubicasa_baseline_v11_eval.json",
        {
            "task": "FRONTIER-V11-T2",
            "locked_metrics": None,
            "blocked": not bool(vendored),
            "blocker": None if vendored else "Official CubiCasa training code is not vendored locally; no fair reproduction was executed.",
            "adopted": False,
        },
    )
    update_todo_remove(["FRONTIER-V11-T2"])
    return audit


def _edge_targets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    targets = []
    for row in rows:
        targets.extend({"id": row["id"], **edge} for edge in row.get("wall_edges") or [])
    return targets


def _hough_edges(row: dict[str, Any], max_edges: int = 160) -> list[dict[str, Any]]:
    if cv2 is None:
        return []
    img = np.asarray(_safe_image(row["image"]).convert("L"))
    edges = cv2.Canny(img, 80, 180)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=45, minLineLength=25, maxLineGap=8)
    out: list[dict[str, Any]] = []
    if lines is None:
        return out
    for idx, line in enumerate(lines[:max_edges]):
        x1, y1, x2, y2 = [int(v) for v in line[0]]
        if math.dist((x1, y1), (x2, y2)) < 15:
            continue
        out.append(
            {
                "id": f"{row['id']}_edge_pred_{idx}",
                "p1": [x1, y1],
                "p2": [x2, y2],
                "bbox": _segment_bbox([x1, y1], [x2, y2], 3),
                "geometry_source": "edge_graph_model_v11_hough_smoke",
                "score": 0.35,
            }
        )
    return out


def export_edge_graph_targets(args: argparse.Namespace) -> dict[str, Any]:
    if not (DATA_DIR / "train.jsonl").exists():
        build_targets(args)
    out_dir = ROOT / "datasets/edge_graph_targets_v11"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in ("train", "dev", "locked"):
        rows = load_jsonl(DATA_DIR / f"{split}.jsonl")
        records = []
        for row in rows:
            records.append({"id": row["id"], "image": row["image"], "wall_edges": row.get("wall_edges") or [], "openings": row.get("openings") or [], "windows": row.get("windows") or []})
        write_jsonl(out_dir / f"{split}.jsonl", records)
        summary[split] = {"rows": len(records), "wall_edges": sum(len(r["wall_edges"]) for r in records)}
    report = {"task": "FRONTIER-V11-T3", "target_dir": _rel(out_dir), "summary": summary}
    write_json(REPORT_DIR / "edge_graph_targets_v11_audit.json", report)
    return report


def evaluate_edge_graph(args: argparse.Namespace) -> dict[str, Any]:
    export_edge_graph_targets(args)
    rows = load_jsonl(DATA_DIR / "locked.jsonl")[: args.max_eval]
    preds, golds, cases = [], [], []
    full_image_edges = 0
    for row in rows:
        row_preds = _hough_edges(row)
        row_golds = row.get("wall_edges") or []
        for pred in row_preds:
            x1, y1, x2, y2 = pred["bbox"]
            if (x2 - x1) * (y2 - y1) > LABEL_SIZE * LABEL_SIZE * 0.35:
                full_image_edges += 1
                continue
            preds.append(pred)
        golds.extend(row_golds)
        if len(cases) < 120:
            cases.append({"id": row["id"], "predicted_edges": row_preds[:30], "gold_edges": row_golds[:30]})
    tp, pc, gc, fp, miss = match_counts(preds, golds, 0.25)
    precision, recall = tp / max(pc, 1), tp / max(gc, 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    report = {
        "task": "FRONTIER-V11-T3",
        "run_mode": "hough_edge_proposal_smoke_no_svg_at_inference",
        "locked_rows": len(rows),
        "metrics": {"tp": tp, "predicted": pc, "gold": gc, "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6)},
        "false_full_image_wall_outputs": full_image_edges,
        "adopted": bool(f1 >= 0.35 and full_image_edges == 0),
        "failure_taxonomy": {"false_edge": fp[:30], "missing_edge": miss[:30], "broken_continuity": [], "wrong_attachment": []},
        "geometry_source": "edge_graph_model_v11_hough_smoke",
    }
    write_json(REPORT_DIR / "edge_graph_model_v11_eval.json", report)
    write_jsonl(REPORT_DIR / "edge_graph_model_v11_cases.jsonl", cases)
    update_todo_remove(["FRONTIER-V11-T3"])
    return report


class TinySegDataset(Dataset):  # type: ignore[misc]
    def __init__(self, rows: list[dict[str, Any]], size: int):
        self.rows = rows
        self.size = size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        row = self.rows[index]
        image = _safe_image(row["image"], self.size).convert("L").resize((self.size, self.size), Image.Resampling.BILINEAR)
        mask = Image.new("L", (LABEL_SIZE, LABEL_SIZE), 0)
        draw = ImageDraw.Draw(mask)
        for room in row.get("room_polygons") or []:
            poly = [tuple(p) for p in room.get("polygon") or []]
            if len(poly) >= 3:
                draw.polygon(poly, fill=1)
        mask = mask.resize((self.size, self.size), Image.Resampling.NEAREST)
        x = torch.from_numpy(np.asarray(image, dtype=np.float32)[None] / 255.0)
        y = torch.from_numpy(np.asarray(mask, dtype=np.float32)[None])
        return x, y


class TinyRoomNet(nn.Module):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 12, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(12, 24, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(24, 12, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(12, 1, 1),
        )

    def forward(self, x: Any) -> Any:
        return self.net(x)


def polygon_sequence_branch(args: argparse.Namespace) -> dict[str, Any]:
    if not (DATA_DIR / "train.jsonl").exists():
        build_targets(args)
    out_dir = ROOT / "datasets/polygon_sequence_targets_v11"
    out_dir.mkdir(parents=True, exist_ok=True)
    lengths: list[int] = []
    invalid = 0
    for split in ("train", "dev", "locked"):
        rows = load_jsonl(DATA_DIR / f"{split}.jsonl")
        records = []
        for row in rows:
            seq = ["BOS"]
            for item in row.get("polygon_sequences") or []:
                seq.extend(item["tokens"])
            seq.append("EOS")
            lengths.append(len(seq))
            invalid += sum(1 for item in row.get("room_polygons") or [] if not _poly_valid(item.get("polygon") or []))
            records.append({"id": row["id"], "image": row["image"], "tokens": seq, "length": len(seq), "label_source": "offline_svg_gold_only"})
        write_jsonl(out_dir / f"{split}.jsonl", records)
    locked = load_jsonl(DATA_DIR / "locked.jsonl")[: args.max_eval]
    train = load_jsonl(DATA_DIR / "train.jsonl")[: args.max_train]
    model_status = "not_run_torch_unavailable"
    room_metrics = {"iou_0.3": {}, "iou_0.5": {}, "iou_0.7": {}}
    if torch is not None and train and locked:
        device = args.device
        ds = TinySegDataset(train, args.train_size)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
        model = TinyRoomNet().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        for _ in range(args.epochs):
            model.train()
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = F.binary_cross_entropy_with_logits(model(x), y)
                loss.backward()
                opt.step()
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "train_size": args.train_size}, CHECKPOINT_DIR / "polygon_sequence_room_smoke.pt")
        model_status = "bounded_tiny_room_mask_encoder_smoke"
        model.eval()
        pred_boxes, gold_boxes = [], []
        with torch.no_grad():
            for row in locked:
                x, _ = TinySegDataset([row], args.train_size)[0]
                mask = torch.sigmoid(model(x[None].to(device)))[0, 0].cpu().numpy() > 0.5
                ys, xs = np.where(mask)
                if len(xs) > 20:
                    scale = LABEL_SIZE / args.train_size
                    pred_boxes.append({"bbox": [int(xs.min() * scale), int(ys.min() * scale), int((xs.max() + 1) * scale), int((ys.max() + 1) * scale)], "geometry_source": "polygon_sequence_model_v11_tiny_smoke"})
                for room in row.get("room_polygons") or []:
                    bbox = normalize_bbox(room.get("bbox"))
                    if bbox:
                        gold_boxes.append({"bbox": bbox})
        for thr in (0.3, 0.5, 0.7):
            tp, pc, gc, fp, miss = match_counts(pred_boxes, gold_boxes, thr)
            precision, recall = tp / max(pc, 1), tp / max(gc, 1)
            f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
            room_metrics[f"iou_{thr}"] = {"tp": tp, "predicted": pc, "gold": gc, "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6), "false_positive_examples": fp[:10], "miss_examples": miss[:10]}
    adopted = room_metrics["iou_0.5"].get("precision", 0) >= 0.35 and room_metrics["iou_0.5"].get("recall", 0) >= 0.35
    report = {
        "task": "FRONTIER-V11-T4",
        "target_dir": _rel(out_dir),
        "median_token_length": float(np.median(lengths)) if lengths else 0.0,
        "max_token_length": max(lengths) if lengths else 0,
        "invalid_polygon_count": invalid,
        "model_status": model_status,
        "no_svg_copied_at_inference": True,
        "room_polygon_metrics": room_metrics,
        "invalid_polygon_rate": round(invalid / max(sum(1 for r in load_jsonl(DATA_DIR / "locked.jsonl") for _ in r.get("room_polygons", [])), 1), 6),
        "adopted": bool(adopted),
    }
    write_json(REPORT_DIR / "polygon_sequence_model_v11_eval.json", report)
    _write_failure_gallery(REPORT_DIR / "polygon_sequence_model_v11_failure_gallery.html", "v11 polygon sequence smoke", locked[:20], report)
    update_todo_remove(["FRONTIER-V11-T4"])
    return report


_IMAGE_FEATURE_CACHE: dict[str, Image.Image] = {}


def _feature_image(image_path: str | Path) -> Image.Image | None:
    key = str(image_path)
    if key in _IMAGE_FEATURE_CACHE:
        return _IMAGE_FEATURE_CACHE[key]
    p = _abs(image_path)
    if not p.exists():
        return None
    img = _safe_image(p, LABEL_SIZE).convert("L")
    if len(_IMAGE_FEATURE_CACHE) < 256:
        _IMAGE_FEATURE_CACHE[key] = img
    return img


def _crop_features(image_path: str | Path, bbox: list[float]) -> list[float]:
    img = _feature_image(image_path)
    if img is None:
        return [0.0] * 8
    x1, y1, x2, y2 = [int(max(0, v)) for v in bbox]
    x1, y1 = min(x1, LABEL_SIZE - 1), min(y1, LABEL_SIZE - 1)
    x2, y2 = min(max(x2, x1 + 1), LABEL_SIZE), min(max(y2, y1 + 1), LABEL_SIZE)
    x2 = max(x2, x1 + 1)
    y2 = max(y2, y1 + 1)
    crop = img.crop((x1, y1, x2, y2)).resize((24, 24), Image.Resampling.BILINEAR)
    arr = np.asarray(crop, dtype=np.float32) / 255.0
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    return [float(arr.mean()), float(arr.std()), float((arr < 0.5).mean()), float((arr < 0.82).mean()), float(w), float(h), float(w / max(h, 1)), float(w * h)]


def topology_refiner(args: argparse.Namespace) -> dict[str, Any]:
    if RandomForestClassifier is None:
        report = {"task": "FRONTIER-V11-T5", "blocked": True, "blocker": "sklearn unavailable", "adopted": False}
        write_json(REPORT_DIR / "topology_refiner_v11_eval.json", report)
        update_todo_remove(["FRONTIER-V11-T5"])
        return report
    if not (DATA_DIR / "train.jsonl").exists():
        build_targets(args)
    train = load_jsonl(DATA_DIR / "train.jsonl")[: args.max_train]
    locked = load_jsonl(DATA_DIR / "locked.jsonl")[: args.max_eval]
    x_train, y_train = [], []
    for row in train:
        for room in row.get("room_polygons") or []:
            bbox = normalize_bbox(room.get("bbox"))
            if bbox:
                x_train.append(_crop_features(row["image"], bbox))
                y_train.append(1)
                x1, y1, x2, y2 = bbox
                x_train.append(_crop_features(row["image"], [x1 + 18, y1 + 18, x2 + 18, y2 + 18]))
                y_train.append(0)
    model = RandomForestClassifier(n_estimators=48, random_state=args.seed, class_weight="balanced")
    if len(set(y_train)) >= 2:
        model.fit(x_train, y_train)
    before = Counter()
    after = Counter()
    for row in locked:
        rooms = [r for r in row.get("room_polygons") or [] if normalize_bbox(r.get("bbox"))]
        before["gold_rooms"] += len(rooms)
        kept = 0
        for room in rooms:
            prob = 1.0
            if len(set(y_train)) >= 2:
                prob = float(model.predict_proba([_crop_features(row["image"], room["bbox"])])[0][1])
            if prob >= 0.35:
                kept += 1
        after["kept_rooms"] += kept
        after["missed_room"] += max(0, len(rooms) - kept)
    report = {
        "task": "FRONTIER-V11-T5",
        "run_mode": "bounded_random_forest_room_candidate_refiner",
        "input_provenance": "v7/v8 candidate geometry remains protected; this smoke uses gold-shaped candidates only to validate refiner plumbing",
        "learned_correction_source": "topology_refiner_v11",
        "train_candidates": len(x_train),
        "locked_rows": len(locked),
        "before": dict(before),
        "after": dict(after),
        "adopted": False,
        "rejection_reason": "Not adopted because smoke used gold-shaped candidates; must be rerun on real v7/v8 predictions before replacing any output.",
    }
    write_json(REPORT_DIR / "topology_refiner_v11_eval.json", report)
    _write_failure_gallery(REPORT_DIR / "topology_refiner_v11_visual_delta.html", "v11 topology refiner delta", locked[:20], report)
    update_todo_remove(["FRONTIER-V11-T5"])
    return report


def small_object_text_detector(args: argparse.Namespace) -> dict[str, Any]:
    if RandomForestClassifier is None:
        report = {"task": "FRONTIER-V11-T6", "blocked": True, "blocker": "sklearn unavailable", "adopted": False}
        write_json(REPORT_DIR / "small_object_text_detector_v11_eval.json", report)
        update_todo_remove(["FRONTIER-V11-T6"])
        return report
    if not (DATA_DIR / "train.jsonl").exists():
        build_targets(args)
    train = load_jsonl(DATA_DIR / "train.jsonl")[: args.max_train]
    locked = load_jsonl(DATA_DIR / "locked.jsonl")[: args.max_eval]
    classes = ["opening", "window", "symbol", "text"]
    class_to_int = {c: i for i, c in enumerate(classes)}
    x_train, y_train = [], []
    for row in train:
        for cls in classes:
            for item in row.get(cls + "s" if cls in {"opening", "window", "symbol"} else "text_boxes") or []:
                bbox = normalize_bbox(item.get("bbox"))
                if bbox:
                    x_train.append(_crop_features(row["image"], bbox))
                    y_train.append(class_to_int[cls])
    clf = RandomForestClassifier(n_estimators=64, random_state=args.seed, class_weight="balanced")
    trained = len(set(y_train)) >= 2
    if trained:
        clf.fit(x_train, y_train)
    evals: dict[str, Any] = {}
    for cls in classes:
        preds, golds = [], []
        for row in locked:
            key = cls + "s" if cls in {"opening", "window", "symbol"} else "text_boxes"
            for item in row.get(key) or []:
                bbox = normalize_bbox(item.get("bbox"))
                if not bbox:
                    continue
                golds.append({"bbox": bbox, "class": cls, "text": item.get("text", "")})
                if trained:
                    pred_cls = classes[int(clf.predict([_crop_features(row["image"], bbox)])[0])]
                    if pred_cls == cls:
                        preds.append({"bbox": bbox, "class": cls, "geometry_source": "small_object_text_detector_v11_rf_smoke"})
        tp, pc, gc, fp, miss = match_counts(preds, golds, 0.5)
        precision, recall = tp / max(pc, 1), tp / max(gc, 1)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        evals[cls] = {"tp": tp, "predicted": pc, "gold": gc, "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6), "false_positive_examples": fp[:10], "miss_examples": miss[:10]}
    numeric_gold = sum(len(r.get("numeric_text_boxes") or []) for r in locked)
    ocr_deps = {name: importlib.util.find_spec(name) is not None for name in ["easyocr", "pytesseract", "paddleocr"]}
    report = {
        "task": "FRONTIER-V11-T6",
        "run_mode": "bounded_random_forest_crop_classifier_with_gold_candidate_smoke",
        "detector_dependencies": {"ultralytics": importlib.util.find_spec("ultralytics") is not None, "transformers": importlib.util.find_spec("transformers") is not None},
        "ocr_backends_available": ocr_deps,
        "per_class": evals,
        "text_bbox_recall": evals["text"]["recall"],
        "ocr_content_accuracy": None,
        "numeric_gold": numeric_gold,
        "numeric_recall": None if numeric_gold == 0 else evals["text"]["recall"],
        "adopted": False,
        "rejection_reason": "Smoke classifier is candidate-conditioned; it validates separation of localization/OCR metrics but is not an image-only detector.",
    }
    write_json(REPORT_DIR / "small_object_text_detector_v11_eval.json", report)
    write_json(REPORT_DIR / "text_ocr_protocol_v11_eval.json", {"task": "FRONTIER-V11-T6", "ocr_backends_available": ocr_deps, "numeric_gold": numeric_gold, "numeric_status": "evaluable" if numeric_gold else "not_evaluable_no_numeric_gold", "content_accuracy": None})
    _write_failure_gallery(REPORT_DIR / "text_symbol_ocr_failure_gallery_v11.html", "v11 text/symbol OCR failure gallery", locked[:20], report)
    update_todo_remove(["FRONTIER-V11-T6"])
    return report


def foundation_pseudolabel(args: argparse.Namespace) -> dict[str, Any]:
    deps = {name: importlib.util.find_spec(name) is not None for name in ["segment_anything", "sam2", "torch", "transformers"]}
    search_roots = [ROOT / "checkpoints", ROOT / "models", ROOT / "weights", ROOT / "datasets/external"]
    weights: list[str] = []
    for base in search_roots:
        if not base.exists():
            continue
        for p in base.rglob("*sam*"):
            if p.is_file() and p.suffix.lower() in {".pt", ".pth", ".safetensors"}:
                weights.append(str(p.relative_to(ROOT)))
                if len(weights) >= 30:
                    break
        if len(weights) >= 30:
            break
    report = {
        "task": "FRONTIER-V11-T7",
        "dependencies": deps,
        "weights_found": weights,
        "fixed_prompts": "not_defined_no_local_sam_weights",
        "random_seed": args.seed,
        "locked_reproducibility": False,
        "exploratory_only": True,
        "adopted": False,
        "blocker": "No pinned local SAM/SAM2 weights and prompt protocol were available.",
    }
    write_json(REPORT_DIR / "foundation_pseudolabel_v11_eval.json", report)
    write_jsonl(REPORT_DIR / "foundation_pseudolabel_v11_cases.jsonl", [])
    update_todo_remove(["FRONTIER-V11-T7"])
    return report


def vector_graph_dataset_audit(args: argparse.Namespace) -> dict[str, Any]:
    candidates = [ROOT / "datasets/external/resplan", ROOT / "datasets/resplan", ROOT / "datasets/external/ResPlan"]
    local = [str(p.relative_to(ROOT)) for p in candidates if p.exists()]
    report = {
        "task": "FRONTIER-V11-T8",
        "datasets": [
            {"name": "ResPlan", "source": "https://arxiv.org/abs/2508.14006", "local_available": bool(local), "local_paths": local, "recommendation": "pretraining_feasible_only_after_license_and_schema_review" if local else "blocked_no_local_dataset"},
            {"name": "CubiCasa5K", "source": "https://github.com/cubicasa/cubicasa5k", "local_available": True, "recommendation": "locked_benchmark_only_for_current_project"},
        ],
        "no_external_data_mixed_into_locked_eval": True,
        "pretraining_tasks": ["edge_graph_endpoint_prediction", "polygon_sequence_decoding", "topology_refiner_candidate_scoring"],
        "adopted": False,
    }
    write_json(REPORT_DIR / "vector_graph_dataset_feasibility_v11.json", report)
    write_json(ROOT / "datasets/resplan_adapter_v11_schema.json", {"status": "schema_placeholder", "requires_local_resplan": True, "fields": ["image", "vector_graph", "room_polygons", "wall_edges", "semantic_labels"]})
    update_todo_remove(["FRONTIER-V11-T8"])
    return report


def build_model_v11(args: argparse.Namespace) -> dict[str, Any]:
    reports = {
        "official": load_json(REPORT_DIR / "official_cubicasa_baseline_v11_eval.json", {}),
        "edge_graph": load_json(REPORT_DIR / "edge_graph_model_v11_eval.json", {}),
        "polygon_sequence": load_json(REPORT_DIR / "polygon_sequence_model_v11_eval.json", {}),
        "small_object_text": load_json(REPORT_DIR / "small_object_text_detector_v11_eval.json", {}),
        "topology_refiner": load_json(REPORT_DIR / "topology_refiner_v11_eval.json", {}),
        "v10": load_json(REPORT_DIR / "model_v10_raster_locked_eval.json", {}),
    }
    adopted_components = {name: bool(rep.get("adopted")) for name, rep in reports.items()}
    adopted = any(adopted_components.values())
    predictions: list[dict[str, Any]] = []
    if adopted:
        for row in load_jsonl(DATA_DIR / "locked.jsonl")[: args.max_eval]:
            nodes: list[dict[str, Any]] = []
            edges: list[dict[str, Any]] = []
            if adopted_components.get("edge_graph"):
                for idx, edge in enumerate(_hough_edges(row)):
                    x1, y1, x2, y2 = edge["bbox"]
                    if (x2 - x1) * (y2 - y1) > LABEL_SIZE * LABEL_SIZE * 0.35:
                        continue
                    node_id = f"{row['id']}_wall_edge_{idx}"
                    nodes.append(
                        {
                            "id": node_id,
                            "family": "boundary",
                            "semantic_type": "wall",
                            "bbox": [round(float(v), 3) for v in edge["bbox"]],
                            "geometry": {"type": "segment", "p1": edge["p1"], "p2": edge["p2"]},
                            "score": edge.get("score", 0.35),
                            "geometry_source": edge["geometry_source"],
                            "proposal_source": "image_only_hough_edge_proposal",
                            "svg_candidate_ids_used": False,
                        }
                    )
            predictions.append(
                {
                    "id": row["id"],
                    "route_trace": {"source_mode": "model_v11_frontier", "svg_candidate_ids_used": False},
                    "scene_graph": {"nodes": nodes, "edges": edges},
                    "image": row["image"],
                }
            )
    write_jsonl(REPORT_DIR / "model_v11_predictions.jsonl", predictions)
    decisions = {
        "task": "FRONTIER-V11-T9",
        "adopted": adopted,
        "adopted_components": adopted_components,
        "decision": "augment_v7_v8" if adopted else "do_not_replace_v7_v8",
        "reason": "No v11 branch passed source-gated locked adoption thresholds." if not adopted else "At least one component passed; manual integration review still required.",
        "rejected_branch_reports": {name: _rel(REPORT_DIR / fname) for name, fname in {
            "edge_graph": "edge_graph_model_v11_eval.json",
            "polygon_sequence": "polygon_sequence_model_v11_eval.json",
            "small_object_text": "small_object_text_detector_v11_eval.json",
            "topology_refiner": "topology_refiner_v11_eval.json",
        }.items()},
    }
    integrity = {
        "task": "FRONTIER-V11-T9",
        "checked_rows": len(predictions),
        "checked_nodes": sum(len((row.get("scene_graph") or {}).get("nodes") or []) for row in predictions),
        "violations": sum(
            1
            for row in predictions
            for node in (row.get("scene_graph") or {}).get("nodes") or []
            if node.get("svg_candidate_ids_used") or str(node.get("geometry_source") or "").startswith("offline_svg")
        ),
        "svg_candidate_ids_used": False,
        "note": "No adopted v11 scene graph emitted." if not predictions else "Adopted predictions contain no SVG candidate ids.",
    }
    ablation = {"task": "FRONTIER-V11-T9", "v7_v8_baseline": "protected_existing_moe_expert_pipeline", "v10": reports["v10"], "v11": decisions}
    write_json(REPORT_DIR / "model_v11_adoption_decisions.json", decisions)
    write_json(REPORT_DIR / "model_v11_source_integrity_audit.json", integrity)
    write_json(REPORT_DIR / "model_v11_vs_v10_vs_v8_vs_v7_ablation.json", ablation)
    update_todo_remove(["FRONTIER-V11-T9"])
    return decisions


def _write_failure_gallery(path: Path, title: str, rows: list[dict[str, Any]], payload: Any) -> None:
    cards = []
    for row in rows:
        img = _safe_image(row["image"])
        overlay = img.convert("RGBA")
        draw = ImageDraw.Draw(overlay, "RGBA")
        for room in row.get("room_polygons") or []:
            poly = [tuple(p) for p in room.get("polygon") or []]
            if len(poly) >= 3:
                draw.polygon(poly, fill=(236, 190, 70, 38), outline=(170, 120, 20, 200))
        for edge in row.get("wall_edges") or []:
            draw.line([tuple(edge["p1"]), tuple(edge["p2"])], fill=(205, 35, 35, 210), width=3)
        for box in row.get("text_boxes") or []:
            x1, y1, x2, y2 = [int(v) for v in box["bbox"]]
            draw.rectangle((x1, y1, x2, y2), outline=(35, 35, 35, 220), width=2)
        buf = BytesIO()
        overlay.save(buf, format="PNG")
        uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        cards.append(
            f"<section><h2>{row['id']}</h2><p>source_mode=v11_failure_review geometry_source=offline_gold_for_error_context</p>"
            f"<div class='grid'><figure><img src='{_img_uri(row['image'])}'><figcaption>input raster</figcaption></figure>"
            f"<figure><img src='{uri}'><figcaption>gold context / rejected branch review</figcaption></figure></div></section>"
        )
    html = f"""<!doctype html><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:Arial,sans-serif;margin:24px;color:#222}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}}img{{width:100%;border:1px solid #aaa;background:white}}pre{{background:#f5f5f5;padding:12px;overflow:auto}}figure{{margin:0}}figcaption{{font-size:13px}}</style>
<h1>{title}</h1><pre>{json.dumps(payload, ensure_ascii=False, indent=2)}</pre>{''.join(cards)}"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def render_review_pack(args: argparse.Namespace) -> dict[str, Any]:
    if not (DATA_DIR / "locked.jsonl").exists():
        build_targets(args)
    locked = load_jsonl(DATA_DIR / "locked.jsonl")[: max(20, args.max_eval)]
    decisions = load_json(REPORT_DIR / "model_v11_adoption_decisions.json", {})
    payload = {
        "task": "FRONTIER-V11-T10",
        "adoption": decisions,
        "source_mode_note": "Panels separate input raster, gold context, rejected v10/v11 branches, and protected v7/v8 baseline.",
        "shown_examples": len(locked),
    }
    base = REPORT_DIR / "visual_demo_model_v11_frontier"
    _write_failure_gallery(base / "review_pack/index.html", "v11 frontier review pack", locked, payload)
    _write_failure_gallery(base / "failure_gallery/index.html", "v11 failure gallery", locked, payload)
    advisor = f"""<!doctype html><meta charset="utf-8"><title>v11 advisor summary</title>
<style>body{{font-family:Arial,sans-serif;margin:24px;max-width:980px;color:#222}}pre{{background:#f5f5f5;padding:12px;overflow:auto}}a{{color:#0645ad}}</style>
<h1>CadStruct v11 advisor summary</h1>
<p>v7/v8 MoE experts remain the protected baseline. v11 frontier branches are shown with source-mode badges and are not presented as adopted unless locked gates pass.</p>
<ul><li><a href="visual_demo_model_v11_frontier/review_pack/index.html">review pack</a></li><li><a href="visual_demo_model_v11_frontier/failure_gallery/index.html">failure gallery</a></li></ul>
<pre>{json.dumps(payload, ensure_ascii=False, indent=2)}</pre>"""
    (REPORT_DIR / "model_v11_advisor_summary.html").write_text(advisor, encoding="utf-8")
    write_json(REPORT_DIR / "model_v11_visual_review_audit.json", payload)
    update_todo_remove(["FRONTIER-V11-T10"])
    return payload


def docs(args: argparse.Namespace) -> dict[str, Any]:
    decisions = load_json(REPORT_DIR / "model_v11_adoption_decisions.json", {})
    sources = [
        "CubiCasa5K official repository: https://github.com/cubicasa/cubicasa5k",
        "MuraNet: https://arxiv.org/abs/2309.00348",
        "PolyRoom: https://arxiv.org/abs/2407.10439",
        "Raster2Seq: https://arxiv.org/abs/2602.09016",
        "CAGE: https://arxiv.org/abs/2509.15459",
        "FloorSAM: https://arxiv.org/abs/2509.15750",
        "ResPlan: https://arxiv.org/abs/2508.14006",
    ]
    adopted = bool(decisions.get("adopted"))
    arch = f"""# CadStruct v11 Frontier Architecture

v11 is a frontier recovery branch around the existing CadStruct-MoE system. It does not overwrite or retrain the good v7/v8 expert models.

## Boundary

- v7/v8: protected MoE baseline with parser/SVG candidates, typed experts, refiners, fusion/router, and visual evidence.
- v10: rejected raster branch with locked evidence.
- v11: target repair, official baseline audit, edge graph proposals, polygon sequence smoke, topology refiner smoke, small-object/text protocol, foundation-model audit, source-gated assembly, and advisor visual evidence.

SVG/parser geometry is used only as offline gold. Adopted v11 inference must declare `svg_candidate_ids_used=false`.

## Research Basis

{chr(10).join(f'- {s}' for s in sources)}

## Locked Decision

Adopted: `{adopted}`. See `reports/vlm/model_v11_adoption_decisions.json`.
"""
    runbook = """# CadStruct v11 Training Runbook

```bash
uv run python scripts/vlm/v11_frontier_pipeline.py run-all --limit 520 --epochs 1 --max-train 96 --max-eval 40 --train-size 128 --batch-size 4
uv run python -m json.tool reports/vlm/model_v11_adoption_decisions.json
```

The bounded run is intended to validate the complete pipeline and produce honest locked evidence. Full adoption requires official CubiCasa reproduction or stronger edge/polygon/text branches.
"""
    sci = f"""# CadStruct SCI2 Paper Plan v5

## Allowed Claims

- Auditable domain-structured MoE with source-integrity gates.
- Visual failure attribution that separates input, gold context, candidate geometry, rejected raster branches, and adopted outputs.
- v11 frontier negative/positive evidence under locked gates.

## Forbidden Unless Gates Pass

- Do not claim pure raster end-to-end CubiCasa recognition is solved.
- Do not claim v11 replaces v7/v8 unless `model_v11_adoption_decisions.json` reports adoption.

Current v11 adopted: `{adopted}`.

## Research Basis

{chr(10).join(f'- {s}' for s in sources)}
"""
    advisor = f"""# CadStruct Advisor Report v11

The project still has publishable value around a protected MoE architecture, source-integrity visualization, and rigorous failure attribution. The v11 work adds a modern frontier audit but keeps failed branches visible.

Adopted v11 branch: `{adopted}`.

Key evidence:

- `reports/vlm/frontier_targets_v11_alignment.html`
- `reports/vlm/visual_demo_model_v11_frontier/review_pack/index.html`
- `reports/vlm/model_v11_advisor_summary.html`
- `reports/vlm/model_v11_adoption_decisions.json`
"""
    (ROOT / "docs/cadstruct/runbooks/cadstruct-v11-frontier-architecture.md").write_text(arch, encoding="utf-8")
    (ROOT / "docs/cadstruct/runbooks/cadstruct-v11-training-runbook.md").write_text(runbook, encoding="utf-8")
    (ROOT / "docs/cadstruct/paper/cadstruct-sci2-paper-plan-v5.md").write_text(sci, encoding="utf-8")
    (ROOT / "docs/cadstruct-advisor-report-v11.md").write_text(advisor, encoding="utf-8")
    report = {"task": "FRONTIER-V11-T11", "docs_written": ["docs/cadstruct/runbooks/cadstruct-v11-frontier-architecture.md", "docs/cadstruct/runbooks/cadstruct-v11-training-runbook.md", "docs/cadstruct/paper/cadstruct-sci2-paper-plan-v5.md", "docs/cadstruct-advisor-report-v11.md"], "adopted": adopted}
    write_json(REPORT_DIR / "model_v11_docs_audit.json", report)
    update_todo_remove(["FRONTIER-V11-T11"])
    return report


def run_all(args: argparse.Namespace) -> None:
    audit_raw_text(args)
    build_targets(args)
    official_baseline(args)
    evaluate_edge_graph(args)
    polygon_sequence_branch(args)
    topology_refiner(args)
    small_object_text_detector(args)
    foundation_pseudolabel(args)
    vector_graph_dataset_audit(args)
    build_model_v11(args)
    render_review_pack(args)
    docs(args)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "command",
        choices=[
            "audit-raw-text",
            "build-targets",
            "official-baseline",
            "edge-graph",
            "polygon-sequence",
            "topology-refiner",
            "small-object-text",
            "foundation-pseudolabel",
            "vector-datasets",
            "build-model",
            "render-review",
            "docs",
            "run-all",
        ],
    )
    p.add_argument("--limit", type=int, default=520)
    p.add_argument("--seed", type=int, default=20260507)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-train", type=int, default=96)
    p.add_argument("--max-eval", type=int, default=40)
    p.add_argument("--train-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--device", default="cuda" if torch is not None and torch.cuda.is_available() else "cpu")
    return p


def main() -> None:
    args = parser().parse_args()
    actions = {
        "audit-raw-text": audit_raw_text,
        "build-targets": build_targets,
        "official-baseline": official_baseline,
        "edge-graph": evaluate_edge_graph,
        "polygon-sequence": polygon_sequence_branch,
        "topology-refiner": topology_refiner,
        "small-object-text": small_object_text_detector,
        "foundation-pseudolabel": foundation_pseudolabel,
        "vector-datasets": vector_graph_dataset_audit,
        "build-model": build_model_v11,
        "render-review": render_review_pack,
        "docs": docs,
        "run-all": run_all,
    }
    actions[args.command](args)


if __name__ == "__main__":
    main()
