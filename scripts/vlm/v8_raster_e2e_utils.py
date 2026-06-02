"""Utilities for CadStruct-MoE raster end-to-end v8 experiments."""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parents[2]
CUBICASA_ROOT = ROOT / "datasets/external/cubicasa5k_zenodo/unpacked"
CONVERTED_DIR = ROOT / "datasets/cadstruct_cubicasa5k_moe"
DATASET_DIR = ROOT / "datasets/raster_e2e_detector_v8"
REPORT_DIR = ROOT / "reports/vlm"
CHECKPOINT_DIR = ROOT / "checkpoints"

FAMILIES = ["boundary", "space", "symbol", "text"]
DEMO_SAMPLE_HINTS = {"11563", "13277", "13624", "5039", "9351"}


def load_json(path: str | Path, default: Any | None = None) -> Any:
    p = ROOT / path if not Path(path).is_absolute() else Path(path)
    if not p.exists():
        return {} if default is None else default
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    p = ROOT / path if not Path(path).is_absolute() else Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = ROOT / path if not Path(path).is_absolute() else Path(path)
    rows: list[dict[str, Any]] = []
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = ROOT / path if not Path(path).is_absolute() else Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def update_todo_remove(task_ids: list[str]) -> None:
    path = ROOT / "todo.json"
    if not path.exists():
        return
    data = load_json(path, {})
    tasks = data.get("tasks") if isinstance(data.get("tasks"), list) else []
    before = len(tasks)
    remaining = [task for task in tasks if str(task.get("id")) not in set(task_ids)]
    removed = before - len(remaining)
    data["tasks"] = remaining
    status = data.setdefault("status", {})
    status["pending"] = len(remaining)
    status["completed_removed_this_run"] = int(status.get("completed_removed_this_run") or 0) + removed
    status["completed"] = int(status.get("completed") or 0) + removed
    status["in_progress"] = 0
    if not remaining:
        data["phase"] = "completed"
    write_json(path, data)


def sample_key(path: str | Path | None) -> str:
    if not path:
        return ""
    parts = Path(str(path)).parts
    if len(parts) >= 2 and Path(parts[-1]).suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"}:
        return parts[-2]
    return Path(str(path)).stem


def normalize_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(value[index]) for index in range(4)]
    except (TypeError, ValueError):
        return None
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def bbox_area(bbox: list[float] | None) -> float:
    if not bbox:
        return 0.0
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def bbox_iou(left: list[float], right: list[float]) -> float:
    ix1 = max(left[0], right[0])
    iy1 = max(left[1], right[1])
    ix2 = min(left[2], right[2])
    iy2 = min(left[3], right[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = bbox_area(left) + bbox_area(right) - inter
    return inter / max(union, 1e-9)


def match_counts(preds: list[dict[str, Any]], golds: list[dict[str, Any]], iou_threshold: float = 0.5) -> tuple[int, int, int, list[dict[str, Any]], list[dict[str, Any]]]:
    used: set[int] = set()
    matches = 0
    fp_cases: list[dict[str, Any]] = []
    for pred in preds:
        pb = normalize_bbox(pred.get("bbox"))
        best_iou = 0.0
        best_idx = -1
        if pb:
            for idx, gold in enumerate(golds):
                if idx in used:
                    continue
                gb = normalize_bbox(gold.get("bbox"))
                if not gb:
                    continue
                score = bbox_iou(pb, gb)
                if score > best_iou:
                    best_iou = score
                    best_idx = idx
        if best_iou >= iou_threshold and best_idx >= 0:
            matches += 1
            used.add(best_idx)
        else:
            fp_cases.append({"prediction": pred, "best_iou": round(best_iou, 6)})
    miss_cases = [{"gold": gold} for idx, gold in enumerate(golds) if idx not in used]
    return matches, len(preds), len(golds), fp_cases, miss_cases


def f1(tp: int, predicted: int, gold: int) -> dict[str, Any]:
    precision = tp / max(predicted, 1)
    recall = tp / max(gold, 1)
    score = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"tp": int(tp), "predicted": int(predicted), "gold": int(gold), "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(score, 6)}


def extract_gold_items(row: dict[str, Any]) -> list[dict[str, Any]]:
    expected = row.get("expected_json") if isinstance(row.get("expected_json"), dict) else {}
    items: list[dict[str, Any]] = []
    for item in expected.get("semantic_candidates") or []:
        bbox = normalize_bbox(item.get("bbox") or (item.get("geometry") or {}).get("bbox"))
        if bbox:
            items.append({"id": item.get("target_id"), "family": "boundary", "semantic_type": item.get("semantic_type"), "bbox": bbox, "source": "offline_svg_label"})
    for item in expected.get("room_candidates") or []:
        bbox = normalize_bbox(item.get("bbox"))
        if bbox:
            items.append({"id": item.get("id"), "family": "space", "semantic_type": item.get("room_type"), "bbox": bbox, "source": "offline_svg_label"})
    for item in expected.get("symbol_candidates") or []:
        bbox = normalize_bbox(item.get("bbox"))
        if bbox:
            items.append({"id": item.get("id"), "family": "symbol", "semantic_type": item.get("symbol_type"), "bbox": bbox, "source": "offline_svg_label"})
    for item in expected.get("text_candidates") or []:
        bbox = normalize_bbox(item.get("bbox"))
        if bbox:
            items.append({"id": item.get("id"), "family": "text", "semantic_type": item.get("text_type"), "bbox": bbox, "text": item.get("text") or "", "source": "offline_svg_label"})
    return items


def family_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(item.get("family") or "unknown") for item in items))


def split_rows_with_locked(rows: list[dict[str, Any]], seed: int = 20260507) -> dict[str, list[dict[str, Any]]]:
    import random

    rows = [row for row in rows if row.get("image_path") and row.get("annotation_path")]
    visual_locked = [row for row in rows if sample_key(row.get("image_path")) in DEMO_SAMPLE_HINTS]
    rest = [row for row in rows if sample_key(row.get("image_path")) not in DEMO_SAMPLE_HINTS]
    random.Random(seed).shuffle(rest)
    locked_target = max(50, min(240, int(len(rows) * 0.08))) if rows else 0
    locked = visual_locked + rest[: max(0, locked_target - len(visual_locked))]
    rest = rest[max(0, locked_target - len(visual_locked)) :]
    dev_count = max(40, min(240, int(len(rest) * 0.10))) if rest else 0
    dev = rest[:dev_count]
    train = rest[dev_count:]
    smoke = (locked[:5] or dev[:5] or train[:5])
    return {"train": train, "dev": dev, "locked": locked, "smoke": smoke}


def image_ink_features(image_path: str | Path, bbox: list[float] | None = None) -> dict[str, float]:
    p = ROOT / image_path if not Path(image_path).is_absolute() else Path(image_path)
    with Image.open(p).convert("L") as image:
        if bbox:
            x1, y1, x2, y2 = clamp_bbox_to_image(bbox, image.size)
            if x2 <= x1 or y2 <= y1:
                return {"dark_ratio": 0.0, "mean": 1.0, "std": 0.0, "area": 0.0, "width": 0.0, "height": 0.0}
            image = image.crop((x1, y1, x2, y2))
        stat = ImageStat.Stat(image)
        pixels = list(image.getdata())
        total = max(len(pixels), 1)
        dark = sum(1 for value in pixels if value < 210)
        very_dark = sum(1 for value in pixels if value < 80)
        w, h = image.size
        return {
            "dark_ratio": dark / total,
            "very_dark_ratio": very_dark / total,
            "mean": stat.mean[0] / 255.0,
            "std": stat.stddev[0] / 255.0,
            "area": float(w * h),
            "width": float(w),
            "height": float(h),
            "aspect": max(w, h) / max(min(w, h), 1),
        }


def clamp_bbox_to_image(bbox: list[float], size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = size
    x1 = max(0, min(width, int(math.floor(bbox[0]))))
    y1 = max(0, min(height, int(math.floor(bbox[1]))))
    x2 = max(0, min(width, int(math.ceil(bbox[2]))))
    y2 = max(0, min(height, int(math.ceil(bbox[3]))))
    return x1, y1, x2, y2


def connected_components_from_image(image_path: str | Path, dark_threshold: int = 205, stride: int = 2, min_pixels: int = 18, max_components: int = 900) -> list[dict[str, Any]]:
    p = ROOT / image_path if not Path(image_path).is_absolute() else Path(image_path)
    with Image.open(p).convert("L") as image:
        if stride > 1:
            small = image.resize((max(1, image.width // stride), max(1, image.height // stride)))
        else:
            small = image
        width, height = small.size
        data = small.load()
        visited = bytearray(width * height)
        components: list[dict[str, Any]] = []
        for y in range(height):
            for x in range(width):
                idx = y * width + x
                if visited[idx] or data[x, y] >= dark_threshold:
                    visited[idx] = 1
                    continue
                stack = [(x, y)]
                visited[idx] = 1
                xs: list[int] = []
                ys: list[int] = []
                while stack:
                    cx, cy = stack.pop()
                    xs.append(cx)
                    ys.append(cy)
                    for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                        if nx < 0 or ny < 0 or nx >= width or ny >= height:
                            continue
                        nidx = ny * width + nx
                        if visited[nidx]:
                            continue
                        visited[nidx] = 1
                        if data[nx, ny] < dark_threshold:
                            stack.append((nx, ny))
                if len(xs) < min_pixels:
                    continue
                bbox = [min(xs) * stride, min(ys) * stride, (max(xs) + 1) * stride, (max(ys) + 1) * stride]
                w = max(1.0, bbox[2] - bbox[0])
                h = max(1.0, bbox[3] - bbox[1])
                components.append({"bbox": bbox, "pixel_count": len(xs) * stride * stride, "width": w, "height": h, "aspect": max(w, h) / max(min(w, h), 1.0)})
        components.sort(key=lambda item: item["pixel_count"], reverse=True)
        return components[:max_components]


def classify_component(component: dict[str, Any], image_size: tuple[int, int]) -> str:
    w = float(component.get("width") or 0.0)
    h = float(component.get("height") or 0.0)
    area = w * h
    image_area = max(float(image_size[0] * image_size[1]), 1.0)
    aspect = max(w, h) / max(min(w, h), 1.0)
    if area / image_area > 0.03:
        return "space"
    if aspect > 5.5 or max(w, h) > max(image_size) * 0.18:
        return "boundary"
    if area < 1800 and max(w, h) < 80:
        return "text"
    return "symbol"


def summarize_manifest_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    boxes = Counter()
    for row in rows:
        counts.update(family_counts(row.get("gold_items") or {}))
        for item in row.get("gold_items") or []:
            if normalize_bbox(item.get("bbox")):
                boxes[str(item.get("family") or "unknown")] += 1
    return {"rows": len(rows), "family_counts": dict(counts), "bbox_counts": dict(boxes)}


def now_ms(start: float) -> float:
    return round((time.time() - start) * 1000.0, 3)


def row_image_size(row: dict[str, Any]) -> tuple[int, int]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    width = metadata.get("width")
    height = metadata.get("height")
    if width and height:
        return int(width), int(height)
    image = row.get("image") or row.get("image_path")
    if image:
        p = ROOT / str(image)
        if p.exists():
            with Image.open(p) as img:
                return img.size
    return 1, 1


def markdown_table(rows: list[list[Any]]) -> str:
    if not rows:
        return ""
    header = "| " + " | ".join(str(v) for v in rows[0]) + " |"
    sep = "| " + " | ".join("---" for _ in rows[0]) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in rows[1:]]
    return "\n".join([header, sep, *body])
