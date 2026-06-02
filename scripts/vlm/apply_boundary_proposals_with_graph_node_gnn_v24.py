#!/usr/bin/env python3
"""Apply raster boundary proposals with the existing graph-node crop-GNN.

This script implements the P0 boundary route in todo.json:
1. generate raster-only line/segment proposals from page pixels,
2. optionally add the existing weak YOLO probe as one proposal stream,
3. classify proposals with the already strong graph-node crop-GNN checkpoint,
4. evaluate proposal recall and classified page-level metrics separately.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader

from graph_node_model import FeatureSpec, tensorize
from train_graph_node_crop_classifier import build_crop_tensor
from train_graph_node_crop_gnn_classifier import collate_graph_batch, load_checkpoint


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "datasets/boundary_expert_public_raster_v19"
DEFAULT_CHECKPOINT = ROOT / "checkpoints/cadstruct_graph_node_crop_gnn_h384_c32_ms3_l2_e24/model_best.pt"
DEFAULT_YOLO_PREDICTIONS = ROOT / "runs/detect/runs/detect/runs/vlm/boundary_public_raster_v24_yolo_probe_locked_val/predictions.json"
DEFAULT_OUTPUT = ROOT / "reports/vlm/boundary_graph_node_gnn_v24_predictions.jsonl"
DEFAULT_EVAL = ROOT / "reports/vlm/boundary_graph_node_gnn_v24_eval.json"
DEFAULT_ERRORS = ROOT / "reports/vlm/boundary_graph_node_gnn_v24_error_buckets.json"

LABELS = ["hard_wall", "door", "window"]
BOUNDARY_TO_GRAPH_LABEL = {"wall": "hard_wall", "opening": "door", "door": "door", "window": "window"}
YOLO_CLASS_TO_LABEL = {1: "hard_wall", 2: "door", 3: "window", 0: "hard_wall"}
TILE_RE = re.compile(r"^(?P<row>.+)_t(?P<size>\d+)_(?P<x1>-?\d+)_(?P<y1>-?\d+)_(?P<x2>-?\d+)_(?P<y2>-?\d+)")
FUSION_FEATURE_NAMES = [
    "gnn_prob_hard_wall",
    "gnn_prob_door",
    "gnn_prob_window",
    "yolo_hint_hard_wall",
    "yolo_hint_door",
    "yolo_hint_window",
    "proposal_confidence",
    "bbox_width",
    "bbox_height",
    "bbox_area",
    "bbox_aspect_log",
    "bbox_length",
    "bbox_thickness",
    "orientation_horizontal",
    "orientation_vertical",
]


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def area(b: list[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def center(b: list[float]) -> tuple[float, float]:
    return (b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5


def iou(left: list[float] | None, right: list[float] | None) -> float:
    if left is None or right is None:
        return 0.0
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    return inter / max(area(left) + area(right) - inter, 1e-9)


def center_covered(pred: list[float], gold: list[float], margin: float = 3.0) -> bool:
    gx, gy = center(gold)
    return pred[0] - margin <= gx <= pred[2] + margin and pred[1] - margin <= gy <= pred[3] + margin


def orientation_for_box(b: list[float]) -> str:
    return "horizontal" if (b[2] - b[0]) >= (b[3] - b[1]) else "vertical"


def clip_box(b: list[float], width: int, height: int) -> list[float] | None:
    out = [max(0.0, b[0]), max(0.0, b[1]), min(float(width), b[2]), min(float(height), b[3])]
    return out if out[2] > out[0] and out[3] > out[1] else None


def extract_runs(mask: np.ndarray, axis: str, min_length: int, thickness_pad: int) -> list[list[float]]:
    proposals: list[list[float]] = []
    if axis == "horizontal":
        projection = mask.sum(axis=1)
        rows = np.where(projection > 0)[0]
        for group in contiguous_groups(rows):
            y1, y2 = int(group[0]), int(group[-1]) + 1
            cols = np.where(mask[y1:y2, :].sum(axis=0) > 0)[0]
            for cgroup in contiguous_groups(cols):
                x1, x2 = int(cgroup[0]), int(cgroup[-1]) + 1
                if x2 - x1 >= min_length:
                    proposals.append([x1, max(0, y1 - thickness_pad), x2, min(mask.shape[0], y2 + thickness_pad)])
    else:
        projection = mask.sum(axis=0)
        cols = np.where(projection > 0)[0]
        for group in contiguous_groups(cols):
            x1, x2 = int(group[0]), int(group[-1]) + 1
            rows = np.where(mask[:, x1:x2].sum(axis=1) > 0)[0]
            for rgroup in contiguous_groups(rows):
                y1, y2 = int(rgroup[0]), int(rgroup[-1]) + 1
                if y2 - y1 >= min_length:
                    proposals.append([max(0, x1 - thickness_pad), y1, min(mask.shape[1], x2 + thickness_pad), y2])
    return proposals


def contiguous_groups(values: np.ndarray, max_gap: int = 2) -> list[np.ndarray]:
    if values.size == 0:
        return []
    breaks = np.where(np.diff(values) > max_gap)[0] + 1
    return [group for group in np.split(values, breaks) if group.size]


def raster_line_proposals(image_path: Path, min_length: int, max_components: int) -> list[dict[str, Any]]:
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return []
    height, width = gray.shape[:2]
    _, inv = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    inv = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel_open)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, min_length // 2), 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(12, min_length // 2)))
    masks = {
        "horizontal": cv2.dilate(cv2.erode(inv, horizontal_kernel), horizontal_kernel),
        "vertical": cv2.dilate(cv2.erode(inv, vertical_kernel), vertical_kernel),
    }
    boxes: list[tuple[str, list[float]]] = []
    for orient, mask in masks.items():
        for box in extract_runs(mask, orient, min_length, thickness_pad=4):
            clipped = clip_box(box, width, height)
            if clipped is not None:
                boxes.append((orient, clipped))
    boxes.sort(key=lambda item: (item[1][2] - item[1][0]) * (item[1][3] - item[1][1]), reverse=True)
    output: list[dict[str, Any]] = []
    for index, (orient, box) in enumerate(deduplicate_boxes(boxes, iou_threshold=0.82)[:max_components]):
        output.append(make_candidate(f"raster_line_{index:05d}", box, "raster_line_morphology", orient, None))
    return output


def deduplicate_boxes(items: list[tuple[str, list[float]]], iou_threshold: float) -> list[tuple[str, list[float]]]:
    kept: list[tuple[str, list[float]]] = []
    for orient, box in items:
        if any(orient == other_orient and iou(box, other_box) >= iou_threshold for other_orient, other_box in kept):
            continue
        kept.append((orient, box))
    return kept


def load_yolo_predictions(path: Path, score_min: float, allowed_row_ids: set[str] | None = None) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    by_row: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in rows:
        score = float(item.get("score") or 0.0)
        if score < score_min:
            continue
        parsed = parse_tile_id(str(item.get("image_id") or Path(str(item.get("file_name") or "")).stem))
        if parsed is None:
            continue
        row_id, ox, oy = parsed
        if allowed_row_ids is not None and row_id not in allowed_row_ids:
            continue
        raw_box = item.get("bbox")
        if not isinstance(raw_box, list) or len(raw_box) != 4:
            continue
        x, y, w, h = [float(v) for v in raw_box]
        box = [ox + x, oy + y, ox + x + w, oy + y + h]
        class_id = int(item.get("category_id") or 0)
        label_hint = YOLO_CLASS_TO_LABEL.get(class_id, "hard_wall")
        by_row[row_id].append(make_candidate("", box, "boundary_yolo_v24_probe", orientation_for_box(box), label_hint, score))
    return by_row


def parse_tile_id(value: str) -> tuple[str, int, int] | None:
    match = TILE_RE.match(value)
    if not match:
        return None
    return match.group("row"), int(match.group("x1")), int(match.group("y1"))


def make_candidate(
    candidate_id: str,
    box: list[float],
    source: str,
    orient: str,
    label_hint: str | None,
    confidence: float | None = None,
) -> dict[str, Any]:
    x1, y1, x2, y2 = box
    return {
        "candidate_id": candidate_id,
        "bbox": [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)],
        "proposal_source": source,
        "label_hint": label_hint,
        "confidence": confidence,
        "orientation": orient,
    }


def merge_candidates(candidates: list[dict[str, Any]], width: int, height: int, cap: int) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for cand in candidates:
        b = clip_box(cand["bbox"], width, height)
        if b is None:
            continue
        if area(b) < 8.0:
            continue
        out = dict(cand)
        out["bbox"] = [round(v, 3) for v in b]
        cleaned.append(out)
    cleaned.sort(key=lambda c: (float(c.get("confidence") or 0.0), area(c["bbox"])), reverse=True)
    kept: list[dict[str, Any]] = []
    for cand in cleaned:
        b = cand["bbox"]
        if any(iou(b, other["bbox"]) >= 0.88 for other in kept):
            continue
        out = dict(cand)
        out["candidate_id"] = out.get("candidate_id") or f"boundary_prop_{len(kept):05d}"
        kept.append(out)
        if len(kept) >= cap:
            break
    return kept


def raster_stats(image: Image.Image, box: list[float]) -> dict[str, float]:
    gray = image.convert("L")
    edge = gray.filter(ImageFilter.FIND_EDGES)
    w, h = gray.size
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return {name: 0.0 for name in ["raster_mean", "raster_std", "raster_dark_density", "raster_edge_density", "raster_context_dark_density", "raster_dark_ratio"]}
    crop = np.asarray(gray.crop((x1, y1, x2, y2)), dtype=np.float32) / 255.0
    edge_crop = np.asarray(edge.crop((x1, y1, x2, y2)), dtype=np.float32) / 255.0
    pad = max(8, int(max(x2 - x1, y2 - y1) * 0.35))
    cx1, cy1, cx2, cy2 = max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)
    context = np.asarray(gray.crop((cx1, cy1, cx2, cy2)), dtype=np.float32) / 255.0
    dark = float((crop < 0.55).mean()) if crop.size else 0.0
    context_dark = float((context < 0.55).mean()) if context.size else 0.0
    return {
        "raster_mean": float(crop.mean()) if crop.size else 0.0,
        "raster_std": float(crop.std()) if crop.size else 0.0,
        "raster_dark_density": dark,
        "raster_edge_density": float((edge_crop > 0.18).mean()) if edge_crop.size else 0.0,
        "raster_context_dark_density": context_dark,
        "raster_dark_ratio": dark / max(context_dark, 1e-6),
    }


def node_features(cand: dict[str, Any], image: Image.Image, width: int, height: int) -> dict[str, Any]:
    b = cand["bbox"]
    x1, y1, x2, y2 = b
    bw, bh = x2 - x1, y2 - y1
    cx, cy = center(b)
    orient = str(cand.get("orientation") or orientation_for_box(b))
    length = bw if orient == "horizontal" else bh
    diag = max(math.hypot(width, height), 1.0)
    centered_x = (cx / max(width, 1)) - 0.5
    centered_y = (cy / max(height, 1)) - 0.5
    angle = 0.0 if orient == "horizontal" else 90.0
    features = {
        "primitive_type": "bbox",
        "bbox": b,
        "centroid": [cx, cy],
        "length": length,
        "angle_degrees": angle,
        "orientation": orient,
        "se2_cx": centered_x,
        "se2_cy": centered_y,
        "se2_width": bw / max(width, 1),
        "se2_height": bh / max(height, 1),
        "se2_area": area(b) / max(width * height, 1),
        "log_area_frac": math.log1p(area(b)) / math.log1p(max(width * height, 1)),
        "log_length_frac": math.log1p(length) / math.log1p(diag),
        "aspect_log": math.log(max(bw, 1.0) / max(bh, 1.0)),
        "radial_norm": math.hypot(centered_x, centered_y),
        "cos2_local_angle": math.cos(math.radians(2.0 * angle)),
        "sin2_local_angle": math.sin(math.radians(2.0 * angle)),
        "source_unknown": 1.0,
    }
    features.update(raster_stats(image, b))
    return features


def build_edges(nodes: list[dict[str, Any]], grid_size: float = 96.0) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    boxes = [node["features"]["bbox"] for node in nodes]
    centers = [center(box) for box in boxes]
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, (cx, cy) in enumerate(centers):
        grid[(int(cx // grid_size), int(cy // grid_size))].append(idx)
    seen: set[tuple[int, int]] = set()
    for i, left in enumerate(boxes):
        lc = centers[i]
        gx, gy = int(lc[0] // grid_size), int(lc[1] // grid_size)
        neighbor_indices = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neighbor_indices.extend(grid.get((gx + dx, gy + dy), []))
        for j in neighbor_indices:
            if j <= i:
                continue
            pair = (i, j)
            if pair in seen:
                continue
            seen.add(pair)
            right = boxes[j]
            rc = centers[j]
            dist = math.hypot(lc[0] - rc[0], lc[1] - rc[1])
            if dist <= 42.0 or iou(left, right) > 0.01:
                edges.append({"source": nodes[i]["id"], "target": nodes[j]["id"], "relation": "touches"})
    degree = Counter()
    for edge in edges:
        degree[int(edge["source"])] += 1
        degree[int(edge["target"])] += 1
    for node in nodes:
        deg = float(degree[int(node["id"])])
        node["features"]["graph_degree"] = deg
        node["features"]["graph_in_degree"] = deg
        node["features"]["graph_out_degree"] = deg
        node["features"]["relation_touches"] = deg
        node["features"]["relation_opens_in_wall"] = 0.0
        node["features"]["relation_window_in_wall"] = 0.0
        node["features"]["relation_contained_in"] = 0.0
        node["features"]["relation_contains"] = 0.0
    return edges


def build_sample(row: dict[str, Any], yolo_by_row: dict[str, list[dict[str, Any]]], args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    image_path = ROOT / str(row.get("image"))
    with Image.open(image_path) as opened:
        image = opened.convert("L")
    width, height = image.size
    line_candidates = [] if args.disable_line_proposals else raster_line_proposals(image_path, args.min_line_length, args.max_line_components)
    yolo_candidates = yolo_by_row.get(str(row.get("id")), []) if args.use_yolo_probe else []
    candidates = merge_candidates(line_candidates + yolo_candidates, width, height, args.max_candidates)
    nodes = []
    for idx, cand in enumerate(candidates):
        cand = dict(cand)
        cand["candidate_id"] = cand.get("candidate_id") or f"boundary_prop_{idx:05d}"
        nodes.append(
            {
                "id": idx,
                "candidate_id": cand["candidate_id"],
                "features": node_features(cand, image, width, height),
                "label": cand.get("label_hint") if cand.get("label_hint") in LABELS else "hard_wall",
                "proposal_source": cand.get("proposal_source"),
                "proposal_confidence": cand.get("confidence"),
            }
        )
    edges = build_edges(nodes)
    sample = {
        "id": row.get("id"),
        "image": row.get("image"),
        "image_size": [width, height],
        "source_dataset": row.get("source_dataset") or "cubicasa5k",
        "nodes": nodes,
        "edges": edges,
    }
    return sample, candidates


def build_split(samples: list[dict[str, Any]], feature_spec: FeatureSpec, crop_size: int, crop_pad_scales: list[float], min_pad: float) -> dict[str, Any]:
    rows = [{"features": node["features"], "label": node["label"]} for sample in samples for node in sample["nodes"]]
    label_to_id = {label: idx for idx, label in enumerate(LABELS)}
    x, y = tensorize(rows, feature_spec, label_to_id)
    crops = build_crop_tensor(samples, crop_size, crop_pad_scales, min_pad)
    sample_ranges = []
    edge_indices = []
    edge_types = []
    offset = 0
    for sample in samples:
        nodes = sample.get("nodes") or []
        node_id_to_local = {int(node["id"]): index for index, node in enumerate(nodes)}
        pairs = []
        types = []
        for edge in sample.get("edges") or []:
            s = node_id_to_local.get(int(edge.get("source", -1)))
            t = node_id_to_local.get(int(edge.get("target", -1)))
            if s is None or t is None or s == t:
                continue
            pairs.append((offset + s, offset + t))
            pairs.append((offset + t, offset + s))
            types.extend([0, 0])
        edge_indices.append(torch.tensor(pairs, dtype=torch.long).t() if pairs else torch.empty(2, 0, dtype=torch.long))
        edge_types.append(torch.tensor(types, dtype=torch.long) if types else torch.empty(0, dtype=torch.long))
        sample_ranges.append((offset, offset + len(nodes)))
        offset += len(nodes)
    return {
        "samples": samples,
        "rows": rows,
        "x": x,
        "crops": crops,
        "y": y,
        "row_weight": torch.ones(int(y.numel()), dtype=torch.float32),
        "sample_ranges": sample_ranges,
        "edge_indices": edge_indices,
        "edge_types": edge_types,
    }


def predict_samples(model: torch.nn.Module, split: dict[str, Any], labels: list[str], batch_samples: int, device: torch.device) -> torch.Tensor:
    if int(split["y"].shape[0]) == 0:
        return torch.empty(0, len(labels), dtype=torch.float32)
    probs_by_node = torch.empty(split["y"].shape[0], len(labels), dtype=torch.float32)
    loader = DataLoader(
        list(range(len(split["samples"]))),
        batch_size=batch_samples,
        shuffle=False,
        collate_fn=lambda batch: collate_graph_batch(split, batch),
    )
    offset = 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            count = int(batch["y"].shape[0])
            moved = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            probs = torch.softmax(model(moved["x"], moved["crops"], moved["edge_index"], moved.get("edge_type")), dim=-1).detach().cpu()
            probs_by_node[offset : offset + count] = probs
            offset += count
    return probs_by_node


def fusion_feature_row(candidate: dict[str, Any], prob: torch.Tensor) -> list[float]:
    raw_features = candidate.get("features") if isinstance(candidate.get("features"), dict) else {}
    box = bbox(candidate.get("bbox")) or bbox(raw_features.get("bbox")) or [0.0, 0.0, 1.0, 1.0]
    width = max(box[2] - box[0], 1e-6)
    height = max(box[3] - box[1], 1e-6)
    hint = str(candidate.get("label_hint") or candidate.get("label") or "")
    horizontal = 1.0 if width >= height else 0.0
    return [
        float(prob[0]),
        float(prob[1]),
        float(prob[2]),
        1.0 if hint == "hard_wall" else 0.0,
        1.0 if hint == "door" else 0.0,
        1.0 if hint == "window" else 0.0,
        float(candidate.get("proposal_confidence") or 0.0),
        width,
        height,
        width * height,
        math.log(width / height),
        max(width, height),
        min(width, height),
        horizontal,
        1.0 - horizontal,
    ]


def load_fusion_model(path: str) -> Any | None:
    if not path:
        return None
    bundle = joblib.load(ROOT / path if not Path(path).is_absolute() else path)
    return bundle.get("model") if isinstance(bundle, dict) else bundle


def assign_predictions(
    samples: list[dict[str, Any]],
    probs: torch.Tensor,
    label_hint_prior: float,
    fusion_model: Any | None,
    hint_override_thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    outputs = []
    offset = 0
    for sample in samples:
        stream = []
        sample_nodes = sample.get("nodes") or []
        sample_probs = probs[offset : offset + len(sample_nodes)]
        fusion_predictions: list[str | None]
        if fusion_model is not None and sample_nodes:
            feature_matrix = np.asarray(
                [fusion_feature_row(node, prob) for node, prob in zip(sample_nodes, sample_probs, strict=True)],
                dtype=np.float32,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                fusion_predictions = [str(item) for item in fusion_model.predict(feature_matrix).tolist()]
        else:
            fusion_predictions = [None] * len(sample_nodes)
        for node, fusion_prediction in zip(sample_nodes, fusion_predictions, strict=True):
            prob = probs[offset]
            gnn_pred_id = int(prob.argmax())
            fused = prob.clone()
            hint = node.get("label")
            if hint in LABELS and node.get("proposal_source") == "boundary_yolo_v24_probe" and label_hint_prior > 0.0:
                fused[LABELS.index(str(hint))] += float(label_hint_prior)
                fused = fused / fused.sum().clamp_min(1e-6)
            if fusion_prediction is not None:
                pred_id = LABELS.index(fusion_prediction) if fusion_prediction in LABELS else int(fused.argmax())
            else:
                pred_id = int(fused.argmax())
            override_label = None
            hint_threshold = hint_override_thresholds.get(str(hint))
            if hint_threshold is not None and hint in {"door", "window"} and float(node.get("proposal_confidence") or 0.0) >= hint_threshold:
                override_label = str(hint)
                pred_id = LABELS.index(override_label)
            b = node["features"]["bbox"]
            stream.append(
                {
                    "candidate_id": node["candidate_id"],
                    "bbox": b,
                    "prediction": LABELS[pred_id],
                    "gnn_prediction": LABELS[gnn_pred_id],
                    "fusion_prediction": fusion_prediction,
                    "hint_override_prediction": override_label,
                    "label_hint": hint if hint in LABELS else None,
                    "confidence": round(float(prob[pred_id]), 6),
                    "fused_confidence": round(float(fused[pred_id]), 6),
                    "probabilities": {label: round(float(prob[i]), 6) for i, label in enumerate(LABELS)},
                    "fused_probabilities": {label: round(float(fused[i]), 6) for i, label in enumerate(LABELS)},
                    "proposal_source": node.get("proposal_source"),
                    "proposal_confidence": node.get("proposal_confidence"),
                }
            )
            offset += 1
        outputs.append(
            {
                "id": sample.get("id"),
                "image": sample.get("image"),
                "image_size": sample.get("image_size"),
                "source_integrity": integrity(),
                "candidate_stream": sorted(stream, key=lambda item: item["confidence"], reverse=True),
            }
        )
    return outputs


def integrity() -> dict[str, Any]:
    return {
        "source_mode": "raster_only_boundary_proposal_with_reused_graph_node_crop_gnn",
        "model_input": "raster_page_pixels_and_raster_derived_candidate_boxes",
        "svg_candidate_ids_used": False,
        "annotation_geometry_used_at_inference": False,
        "gold_loaded_after_inference_for_evaluation_only": True,
    }


def gold_items(row: dict[str, Any]) -> list[dict[str, Any]]:
    items = []
    for target in (row.get("targets") or {}).get("boxes") or []:
        b = bbox(target.get("bbox"))
        label = BOUNDARY_TO_GRAPH_LABEL.get(str(target.get("label")), str(target.get("label")))
        if b is not None and label in LABELS:
            copied = dict(target)
            copied["bbox"] = b
            copied["graph_label"] = label
            items.append(copied)
    return items


def evaluate(outputs: list[dict[str, Any]], gold_rows: list[dict[str, Any]], caps: list[int]) -> tuple[dict[str, Any], dict[str, Any]]:
    gold_by_id = {str(row.get("id")): gold_items(row) for row in gold_rows}
    pred_by_id = {str(row.get("id")): row.get("candidate_stream") or [] for row in outputs}
    cap_reports = {str(cap): evaluate_cap(pred_by_id, gold_by_id, cap) for cap in caps}
    full = evaluate_cap(pred_by_id, gold_by_id, None)
    hard_cases = build_error_buckets(pred_by_id, gold_by_id)
    return {
        "full": full,
        "cap_sweep": cap_reports,
        "primary_cap": 800,
        "primary": cap_reports.get("800") or full,
    }, hard_cases


def evaluate_cap(pred_by_id: dict[str, list[dict[str, Any]]], gold_by_id: dict[str, list[dict[str, Any]]], cap: int | None) -> dict[str, Any]:
    proposal_hit = classified_hit = total = 0
    predicted = 0
    per_label: dict[str, Counter[str]] = defaultdict(Counter)
    for row_id, golds in gold_by_id.items():
        preds = pred_by_id.get(row_id, [])
        if cap is not None:
            preds = preds[:cap]
        predicted += len(preds)
        for gold in golds:
            total += 1
            label = gold["graph_label"]
            per_label[label]["gold"] += 1
            matches = [pred for pred in preds if center_covered(pred["bbox"], gold["bbox"]) or iou(pred["bbox"], gold["bbox"]) >= 0.30]
            if matches:
                proposal_hit += 1
                per_label[label]["proposal_matched"] += 1
            if any(pred.get("prediction") == label for pred in matches):
                classified_hit += 1
                per_label[label]["classified_matched"] += 1
    return {
        "gold": total,
        "predicted": predicted,
        "candidate_inflation": round(predicted / max(total, 1), 6),
        "proposal_recall": round(proposal_hit / max(total, 1), 6),
        "classified_recall": round(classified_hit / max(total, 1), 6),
        "classified_precision_proxy": round(classified_hit / max(predicted, 1), 6),
        "per_label": {
            label: {
                "gold": counts["gold"],
                "proposal_matched": counts["proposal_matched"],
                "classified_matched": counts["classified_matched"],
                "proposal_recall": round(counts["proposal_matched"] / max(counts["gold"], 1), 6),
                "classified_recall": round(counts["classified_matched"] / max(counts["gold"], 1), 6),
            }
            for label, counts in sorted(per_label.items())
        },
    }


def build_error_buckets(pred_by_id: dict[str, list[dict[str, Any]]], gold_by_id: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    missed = []
    wrong_type = []
    for row_id, golds in gold_by_id.items():
        preds = pred_by_id.get(row_id, [])[:800]
        for gold in golds:
            matches = [pred for pred in preds if center_covered(pred["bbox"], gold["bbox"]) or iou(pred["bbox"], gold["bbox"]) >= 0.30]
            if not matches:
                missed.append({"row_id": row_id, "target_id": gold.get("target_id"), "label": gold["graph_label"], "bbox": gold["bbox"]})
            elif not any(pred.get("prediction") == gold["graph_label"] for pred in matches):
                best = max(matches, key=lambda pred: iou(pred["bbox"], gold["bbox"]))
                wrong_type.append(
                    {
                        "row_id": row_id,
                        "target_id": gold.get("target_id"),
                        "gold_label": gold["graph_label"],
                        "pred_label": best.get("prediction"),
                        "bbox": gold["bbox"],
                        "pred_bbox": best.get("bbox"),
                    }
                )
    return {
        "summary": {
            "missed_at_cap800": len(missed),
            "wrong_type_at_cap800": len(wrong_type),
            "missed_by_label": dict(Counter(item["label"] for item in missed)),
            "wrong_type_pairs": dict(Counter(f"{item['gold_label']}->{item['pred_label']}" for item in wrong_type)),
        },
        "missed_examples": missed[:200],
        "wrong_type_examples": wrong_type[:200],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--split", default="locked")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--yolo-predictions", default=str(DEFAULT_YOLO_PREDICTIONS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--eval-output", default=str(DEFAULT_EVAL))
    parser.add_argument("--error-output", default=str(DEFAULT_ERRORS))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-samples", type=int, default=2)
    parser.add_argument("--min-line-length", type=int, default=28)
    parser.add_argument("--max-line-components", type=int, default=1800)
    parser.add_argument("--max-candidates", type=int, default=2500)
    parser.add_argument("--use-yolo-probe", action="store_true", default=True)
    parser.add_argument("--disable-line-proposals", action="store_true")
    parser.add_argument("--yolo-score-min", type=float, default=0.05)
    parser.add_argument("--label-hint-prior", type=float, default=0.75)
    parser.add_argument("--fusion-model", default="")
    parser.add_argument("--hint-override-door-threshold", type=float, default=-1.0)
    parser.add_argument("--hint-override-window-threshold", type=float, default=-1.0)
    parser.add_argument("--chunk-rows", type=int, default=0)
    args = parser.parse_args()

    limit = 5 if args.smoke else (args.limit or None)
    dataset_path = Path(args.dataset) / f"{args.split}.jsonl"
    rows = load_jsonl(dataset_path, limit)
    allowed_row_ids = {str(row.get("id")) for row in rows}
    yolo_by_row = load_yolo_predictions(Path(args.yolo_predictions), args.yolo_score_min, allowed_row_ids) if args.use_yolo_probe else {}
    device = torch.device(args.device)
    model, checkpoint = load_checkpoint(Path(args.checkpoint), device)
    config = checkpoint["model_config"]
    feature_spec = FeatureSpec(**checkpoint["feature_spec"])
    fusion_model = load_fusion_model(args.fusion_model)
    hint_override_thresholds = {}
    if args.hint_override_door_threshold >= 0.0:
        hint_override_thresholds["door"] = float(args.hint_override_door_threshold)
    if args.hint_override_window_threshold >= 0.0:
        hint_override_thresholds["window"] = float(args.hint_override_window_threshold)
    outputs: list[dict[str, Any]] = []
    chunk_rows = args.chunk_rows or len(rows)
    for start in range(0, len(rows), chunk_rows):
        chunk = rows[start : start + chunk_rows]
        samples = [build_sample(row, yolo_by_row, args)[0] for row in chunk]
        split = build_split(samples, feature_spec, int(config["crop_size"]), list(config["crop_pad_scales"]), float(config["min_pad"]))
        probs = predict_samples(model, split, LABELS, args.batch_samples, device)
        outputs.extend(assign_predictions(samples, probs, args.label_hint_prior, fusion_model, hint_override_thresholds))
        print(
            json.dumps(
                {
                    "progress": "boundary_graph_node_gnn_v24",
                    "rows_done": min(start + len(chunk), len(rows)),
                    "rows_total": len(rows),
                    "nodes_in_chunk": int(probs.shape[0]),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
            flush=True,
        )
    eval_metrics, errors = evaluate(outputs, rows, caps=[200, 400, 800, 1200, 2500])

    primary = eval_metrics["primary"]
    report = {
        "version": "boundary_graph_node_gnn_v24_eval",
        "task": "P0-BOUNDARY-PROPOSAL-002",
        "claim_boundary": "Raster-only proposal frontend plus reused graph-node crop-GNN. Gold boundary boxes are used only after inference for evaluation.",
        "dataset": str(dataset_path),
        "split": args.split,
        "rows": len(rows),
        "checkpoint": str(Path(args.checkpoint)),
        "predictions": str(Path(args.output)),
        "source_integrity": integrity(),
        "config": {
            "min_line_length": args.min_line_length,
            "max_line_components": args.max_line_components,
            "max_candidates": args.max_candidates,
            "use_yolo_probe": bool(args.use_yolo_probe),
            "disable_line_proposals": bool(args.disable_line_proposals),
            "yolo_score_min": args.yolo_score_min,
            "label_hint_prior": args.label_hint_prior,
            "fusion_model": args.fusion_model or None,
            "hint_override_thresholds": hint_override_thresholds,
            "device": args.device,
        },
        "metrics": eval_metrics,
        "success_gate": {
            "stage_1_after_cap_recall_min": 0.85,
            "stage_1_window_recall_min": 0.8,
            "candidate_inflation_max": 20.0,
            "proposal_recall_at_cap800": primary["proposal_recall"],
            "classified_recall_at_cap800": primary["classified_recall"],
            "window_proposal_recall_at_cap800": primary["per_label"].get("window", {}).get("proposal_recall", 0.0),
            "candidate_inflation_at_cap800": primary["candidate_inflation"],
            "passed": primary["proposal_recall"] >= 0.85
            and primary["per_label"].get("window", {}).get("proposal_recall", 0.0) >= 0.8
            and primary["candidate_inflation"] <= 20.0,
        },
    }
    write_jsonl(Path(args.output), outputs)
    write_json(Path(args.eval_output), report)
    write_json(Path(args.error_output), errors)
    print(json.dumps({"rows": len(rows), "primary": primary, "success_gate": report["success_gate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
