#!/usr/bin/env python3
"""Train/evaluate a tile-based raster symbol body detector."""

from __future__ import annotations

import argparse
import json
import random
import resource
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.ops import nms


Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets/symbol_tile_detector_v20"
CHECKPOINT = ROOT / "checkpoints/symbol_tile_detector_v20"
REPORT = ROOT / "reports/vlm/symbol_tile_detector_v20_eval.json"
LABELS = ["appliance", "bathtub", "column", "equipment", "generic_symbol", "shower", "sink", "stair", "table"]
LABEL_TO_ID = {label: index + 1 for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index + 1: label for index, label in enumerate(LABELS)}
FORBIDDEN_RUNTIME_FIELDS = ["raw_label", "semantic_type", "expected_json", "annotation_path", "svg_geometry"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def source_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def bbox_iou(left: list[float], right: list[float]) -> float:
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    la = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    ra = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return inter / max(la + ra - inter, 1e-9)


def nwd_similarity(left: list[float], right: list[float], normalizer: float = 32.0) -> float:
    """NWD-style box similarity for tiny-box audit; product gates still use IoU."""
    lcx = (left[0] + left[2]) / 2.0
    lcy = (left[1] + left[3]) / 2.0
    rcx = (right[0] + right[2]) / 2.0
    rcy = (right[1] + right[3]) / 2.0
    lw = max(0.0, left[2] - left[0])
    lh = max(0.0, left[3] - left[1])
    rw = max(0.0, right[2] - right[0])
    rh = max(0.0, right[3] - right[1])
    distance = ((lcx - rcx) ** 2 + (lcy - rcy) ** 2 + ((lw - rw) ** 2 + (lh - rh) ** 2) / 4.0) ** 0.5
    return float(np.exp(-distance / max(normalizer, 1e-6)))


def center_covered(pred: list[float], gold: list[float], margin: float = 2.0) -> bool:
    cx = (gold[0] + gold[2]) / 2.0
    cy = (gold[1] + gold[3]) / 2.0
    return pred[0] - margin <= cx <= pred[2] + margin and pred[1] - margin <= cy <= pred[3] + margin


def area_bucket(box: list[float]) -> str:
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    if area <= 64:
        return "tiny_le_64"
    if area <= 256:
        return "small_le_256"
    if area <= 1024:
        return "medium_le_1024"
    if area <= 4096:
        return "large_le_4096"
    return "xlarge_gt_4096"


def target_area_buckets(row: dict[str, Any]) -> set[str]:
    buckets: set[str] = set()
    for target in ((row.get("targets") or {}).get("boxes") or []):
        box = target.get("page_bbox") or target.get("bbox") or []
        if isinstance(box, list) and len(box) == 4:
            buckets.add(area_bucket([float(v) for v in box]))
    return buckets


def sample_tiles(rows: list[dict[str, Any]], limit: int | None, seed: int, positive_ratio: float) -> list[dict[str, Any]]:
    if not limit or len(rows) <= limit:
        return list(rows)
    rng = random.Random(seed)
    positives = [row for row in rows if int((row.get("target_counts") or {}).get("symbols") or 0) > 0]
    empties = [row for row in rows if int((row.get("target_counts") or {}).get("symbols") or 0) == 0]
    rng.shuffle(positives)
    rng.shuffle(empties)
    pos_n = min(len(positives), int(limit * positive_ratio))
    empty_n = min(len(empties), limit - pos_n)
    out = positives[:pos_n] + empties[:empty_n]
    if len(out) < limit:
        out.extend(positives[pos_n : pos_n + (limit - len(out))])
    rng.shuffle(out)
    return out[:limit]


def sample_tiles_area_aware(
    rows: list[dict[str, Any]],
    limit: int | None,
    seed: int,
    positive_ratio: float,
    small_positive_ratio: float,
) -> list[dict[str, Any]]:
    if not limit or len(rows) <= limit:
        return list(rows)
    rng = random.Random(seed)
    positives = [row for row in rows if int((row.get("target_counts") or {}).get("symbols") or 0) > 0]
    empties = [row for row in rows if int((row.get("target_counts") or {}).get("symbols") or 0) == 0]
    small_positive = [
        row
        for row in positives
        if target_area_buckets(row) & {"tiny_le_64", "small_le_256"}
    ]
    small_ids = {id(row) for row in small_positive}
    other_positive = [row for row in positives if id(row) not in small_ids]
    for group in (small_positive, other_positive, empties):
        rng.shuffle(group)
    pos_n = min(len(positives), int(limit * positive_ratio))
    small_n = min(len(small_positive), int(pos_n * small_positive_ratio))
    other_n = min(len(other_positive), pos_n - small_n)
    selected = small_positive[:small_n] + other_positive[:other_n]
    if len(selected) < pos_n:
        selected.extend(small_positive[small_n : small_n + (pos_n - len(selected))])
    if len(selected) < pos_n:
        selected.extend(other_positive[other_n : other_n + (pos_n - len(selected))])
    empty_n = min(len(empties), limit - len(selected))
    selected.extend(empties[:empty_n])
    if len(selected) < limit:
        leftovers = [row for row in rows if id(row) not in {id(item) for item in selected}]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: limit - len(selected)])
    rng.shuffle(selected)
    return selected[:limit]


class SymbolTileDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], augment: bool, seed: int) -> None:
        self.rows = rows
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        row = self.rows[index]
        tile = row.get("tile") or {}
        x1, y1, x2, y2 = [int(v) for v in tile.get("bbox") or [0, 0, 1, 1]]
        with Image.open(source_path(str(row.get("image") or ""))) as opened:
            crop = opened.convert("RGB").crop((x1, y1, x2, y2))
        crop = ImageOps.autocontrast(crop)
        rng = random.Random(self.seed + index)
        flip = self.augment and rng.random() < 0.5
        if flip:
            crop = ImageOps.mirror(crop)
        arr = np.asarray(crop, dtype=np.float32) / 255.0
        image = torch.from_numpy(arr.transpose(2, 0, 1)).float()
        width = crop.size[0]
        boxes: list[list[float]] = []
        labels: list[int] = []
        areas: list[float] = []
        for target in ((row.get("targets") or {}).get("boxes") or []):
            box = [float(v) for v in target.get("bbox") or []]
            if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
                continue
            if flip:
                box = [width - box[2], box[1], width - box[0], box[3]]
            label = str(target.get("label") or "")
            boxes.append(box)
            labels.append(LABEL_TO_ID.get(label, int(target.get("label_id") or 5)))
            areas.append(max(1.0, (box[2] - box[0]) * (box[3] - box[1])))
        target_tensor = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }
        meta = {
            "id": row.get("id"),
            "row_id": row.get("row_id"),
            "image": row.get("image"),
            "tile_bbox": [x1, y1, x2, y2],
            "gold": ((row.get("targets") or {}).get("boxes") or []),
        }
        return image, target_tensor, meta


def collate(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]]) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]], list[dict[str, Any]]]:
    images, targets, metas = zip(*batch, strict=True)
    return list(images), list(targets), list(metas)


def anchor_generator(profile: str) -> AnchorGenerator | None:
    if profile == "default":
        return None
    if profile == "small":
        sizes = ((8, 12, 16, 24), (16, 24, 32, 48), (32, 48, 64, 96))
        ratios = ((0.5, 0.75, 1.0, 1.5, 2.0),) * len(sizes)
        return AnchorGenerator(sizes=sizes, aspect_ratios=ratios)
    if profile == "tiny":
        sizes = ((4, 6, 8, 12, 16), (8, 12, 16, 24, 32), (16, 24, 32, 48, 64))
        ratios = ((0.5, 0.75, 1.0, 1.5, 2.0),) * len(sizes)
        return AnchorGenerator(sizes=sizes, aspect_ratios=ratios)
    raise ValueError(f"unknown anchor profile: {profile}")


def make_model(weights: str, num_classes: int, anchor_profile: str) -> torch.nn.Module:
    weight_enum = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if weights == "coco" else None
    kwargs: dict[str, Any] = {
        "weights": weight_enum,
        "weights_backbone": None,
        "rpn_pre_nms_top_n_train": 3000,
        "rpn_pre_nms_top_n_test": 3000,
        "rpn_post_nms_top_n_train": 2000,
        "rpn_post_nms_top_n_test": 1600,
        "box_detections_per_img": 900,
    }
    model = fasterrcnn_mobilenet_v3_large_fpn(**kwargs)
    anchors = anchor_generator(anchor_profile)
    if anchors is not None:
        model.rpn.anchor_generator = anchors
        out_channels = model.backbone.out_channels
        num_anchors = anchors.num_anchors_per_location()[0]
        model.rpn.head = RPNHead(out_channels, num_anchors)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_epoch(model: torch.nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> dict[str, float]:
    model.train()
    totals: Counter[str] = Counter()
    batches = 0
    for images, targets, _metas in loader:
        images = [image.to(device, non_blocking=True) for image in images]
        targets = [{key: value.to(device, non_blocking=True) for key, value in target.items()} for target in targets]
        optimizer.zero_grad(set_to_none=True)
        losses = model(images, targets)
        loss = sum(value for value in losses.values())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        totals["loss"] += float(loss.detach().cpu()) * len(images)
        for key, value in losses.items():
            totals[key] += float(value.detach().cpu()) * len(images)
        batches += len(images)
    return {key: round(value / max(batches, 1), 6) for key, value in totals.items()}


def weighted_box_fusion(label_preds: list[dict[str, Any]], iou_threshold: float) -> list[dict[str, Any]]:
    remaining = sorted(label_preds, key=lambda item: float(item["score"]), reverse=True)
    fused: list[dict[str, Any]] = []
    while remaining:
        seed = remaining.pop(0)
        seed_box = [float(v) for v in seed["bbox"]]
        cluster = [seed]
        leftovers: list[dict[str, Any]] = []
        for pred in remaining:
            if bbox_iou(seed_box, [float(v) for v in pred["bbox"]]) >= iou_threshold:
                cluster.append(pred)
            else:
                leftovers.append(pred)
        remaining = leftovers
        weights = [max(float(item["score"]), 1e-6) for item in cluster]
        total_weight = sum(weights)
        box = [
            sum(float(item["bbox"][index]) * weight for item, weight in zip(cluster, weights, strict=True)) / total_weight
            for index in range(4)
        ]
        best = max(cluster, key=lambda item: float(item["score"]))
        fused.append(
            {
                "bbox": box,
                "label_id": int(best["label_id"]),
                "label": str(best["label"]),
                "score": float(best["score"]),
                "tile_id": best.get("tile_id"),
                "merge_cluster_size": len(cluster),
            }
        )
    return fused


def merge_page_predictions(
    page_preds: list[dict[str, Any]],
    score_threshold: float,
    nms_threshold: float,
    max_per_page: int,
    merge_mode: str,
) -> list[dict[str, Any]]:
    filtered = [pred for pred in page_preds if float(pred["score"]) >= score_threshold]
    if not filtered:
        return []
    boxes = torch.tensor([pred["bbox"] for pred in filtered], dtype=torch.float32)
    scores = torch.tensor([float(pred["score"]) for pred in filtered], dtype=torch.float32)
    labels = [int(pred["label_id"]) for pred in filtered]
    if merge_mode == "nms":
        keep_indices: list[int] = []
        for label in sorted(set(labels)):
            idx = torch.tensor([i for i, current in enumerate(labels) if current == label], dtype=torch.long)
            keep = nms(boxes[idx], scores[idx], nms_threshold)
            keep_indices.extend(int(idx[int(i)]) for i in keep.tolist())
        keep_indices.sort(key=lambda i: float(filtered[i]["score"]), reverse=True)
        return [filtered[i] for i in keep_indices[:max_per_page]]
    if merge_mode == "wbf":
        fused: list[dict[str, Any]] = []
        for label in sorted(set(labels)):
            label_preds = [pred for pred in filtered if int(pred["label_id"]) == label]
            fused.extend(weighted_box_fusion(label_preds, nms_threshold))
        fused.sort(key=lambda item: float(item["score"]), reverse=True)
        return fused[:max_per_page]
    raise ValueError(f"unknown merge_mode: {merge_mode}")


def evaluate_model(
    model: torch.nn.Module,
    rows: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    score_threshold: float,
    nms_threshold: float,
    max_per_page: int,
    merge_mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    page_preds, page_golds = collect_page_predictions(model, rows, device, batch_size, num_workers)
    return score_page_predictions(page_preds, page_golds, score_threshold, nms_threshold, max_per_page, merge_mode, len(rows))


def collect_page_predictions(
    model: torch.nn.Module,
    rows: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, dict[str, Any]]]]:
    dataset = SymbolTileDataset(rows, augment=False, seed=0)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda", collate_fn=collate)
    model.eval()
    page_preds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    page_golds: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    with torch.no_grad():
        for images, _targets, metas in loader:
            outputs = model([image.to(device, non_blocking=True) for image in images])
            for output, meta in zip(outputs, metas, strict=True):
                left, top, _right, _bottom = [int(v) for v in meta["tile_bbox"]]
                row_id = str(meta["row_id"])
                for gold in meta["gold"]:
                    target_id = str(gold.get("target_id") or f"{row_id}_{len(page_golds[row_id])}")
                    page_golds[row_id][target_id] = {
                        "target_id": target_id,
                        "bbox": [float(v) for v in gold.get("page_bbox") or gold.get("bbox")],
                        "label": str(gold.get("label") or "generic_symbol"),
                    }
                boxes = output.get("boxes", torch.empty((0, 4))).detach().cpu().tolist()
                labels = output.get("labels", torch.empty((0,), dtype=torch.long)).detach().cpu().tolist()
                scores = output.get("scores", torch.empty((0,))).detach().cpu().tolist()
                for box, label_id, score in zip(boxes, labels, scores, strict=True):
                    page_preds[row_id].append(
                        {
                            "bbox": [float(box[0] + left), float(box[1] + top), float(box[2] + left), float(box[3] + top)],
                            "label_id": int(label_id),
                            "label": ID_TO_LABEL.get(int(label_id), "generic_symbol"),
                            "score": float(score),
                            "tile_id": meta["id"],
                        }
                    )
    return page_preds, page_golds


def score_page_predictions(
    page_preds: dict[str, list[dict[str, Any]]],
    page_golds: dict[str, dict[str, dict[str, Any]]],
    score_threshold: float,
    nms_threshold: float,
    max_per_page: int,
    merge_mode: str,
    tile_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    totals = Counter()
    by_label = Counter()
    by_label_center = Counter()
    by_label_iou = Counter()
    by_area = Counter()
    by_area_center = Counter()
    by_area_iou = Counter()
    by_area_nwd_sum = Counter()
    by_area_nwd_050 = Counter()
    by_area_nwd_070 = Counter()
    nwd_sum = 0.0
    nwd_050 = 0
    nwd_070 = 0
    typed_correct = 0
    for row_id, gold_map in page_golds.items():
        merged = merge_page_predictions(page_preds.get(row_id, []), score_threshold, nms_threshold, max_per_page, merge_mode)
        used_iou: set[int] = set()
        used_center: set[int] = set()
        for gold in gold_map.values():
            gold_box = [float(v) for v in gold["bbox"]]
            label = str(gold["label"])
            bucket = area_bucket(gold_box)
            by_label[label] += 1
            by_area[bucket] += 1
            best_iou = 0.0
            best_nwd = 0.0
            best_iou_index: int | None = None
            center_index: int | None = None
            for pred_index, pred in enumerate(merged):
                pred_box = [float(v) for v in pred["bbox"]]
                iou = bbox_iou(pred_box, gold_box)
                if iou > best_iou:
                    best_iou = iou
                    best_iou_index = pred_index
                best_nwd = max(best_nwd, nwd_similarity(pred_box, gold_box))
                if center_index is None and pred_index not in used_center and center_covered(pred_box, gold_box):
                    center_index = pred_index
            nwd_sum += best_nwd
            by_area_nwd_sum[bucket] += best_nwd
            if best_nwd >= 0.50:
                nwd_050 += 1
                by_area_nwd_050[bucket] += 1
            if best_nwd >= 0.70:
                nwd_070 += 1
                by_area_nwd_070[bucket] += 1
            if best_iou_index is not None and best_iou >= 0.30 and best_iou_index not in used_iou:
                used_iou.add(best_iou_index)
                totals["matched_iou"] += 1
                by_label_iou[label] += 1
                by_area_iou[bucket] += 1
                if merged[best_iou_index]["label"] == label:
                    typed_correct += 1
            if center_index is not None:
                used_center.add(center_index)
                totals["matched_center"] += 1
                by_label_center[label] += 1
                by_area_center[bucket] += 1
        totals["gold"] += len(gold_map)
        totals["predicted"] += len(merged)
        predictions.append({"row_id": row_id, "predicted_symbols": merged, "gold_symbol_count": len(gold_map)})

    precision = totals["matched_iou"] / max(totals["predicted"], 1)
    recall = totals["matched_iou"] / max(totals["gold"], 1)
    report = {
        "rows": len(page_golds),
        "tiles": tile_count,
        "symbol_bbox_iou_0_30": {
            "matched": int(totals["matched_iou"]),
            "predicted": int(totals["predicted"]),
            "gold": int(totals["gold"]),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall), 6),
        },
        "symbol_bbox_center_recall": round(totals["matched_center"] / max(totals["gold"], 1), 6),
        "candidate_inflation": round(totals["predicted"] / max(totals["gold"], 1), 6),
        "typed_accuracy_on_iou_matches": round(typed_correct / max(totals["matched_iou"], 1), 6),
        "type_center_recall": {label: round(by_label_center[label] / max(by_label[label], 1), 6) for label in sorted(by_label)},
        "type_iou_recall": {label: round(by_label_iou[label] / max(by_label[label], 1), 6) for label in sorted(by_label)},
        "area_center_recall": {bucket: round(by_area_center[bucket] / max(by_area[bucket], 1), 6) for bucket in sorted(by_area)},
        "area_iou_recall": {bucket: round(by_area_iou[bucket] / max(by_area[bucket], 1), 6) for bucket in sorted(by_area)},
        "nwd_tiny_box_audit": {
            "mean_best_similarity": round(nwd_sum / max(totals["gold"], 1), 6),
            "recall_at_0_50": round(nwd_050 / max(totals["gold"], 1), 6),
            "recall_at_0_70": round(nwd_070 / max(totals["gold"], 1), 6),
            "area_mean_best_similarity": {bucket: round(by_area_nwd_sum[bucket] / max(by_area[bucket], 1), 6) for bucket in sorted(by_area)},
            "area_recall_at_0_50": {bucket: round(by_area_nwd_050[bucket] / max(by_area[bucket], 1), 6) for bucket in sorted(by_area)},
            "area_recall_at_0_70": {bucket: round(by_area_nwd_070[bucket] / max(by_area[bucket], 1), 6) for bucket in sorted(by_area)},
        },
    }
    return report, predictions


def memory_audit(device: torch.device) -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    audit: dict[str, Any] = {"max_rss_kb": int(usage.ru_maxrss)}
    if device.type == "cuda":
        audit["cuda_peak_allocated_mb"] = round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 3)
        audit["cuda_peak_reserved_mb"] = round(torch.cuda.max_memory_reserved(device) / (1024 * 1024), 3)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA))
    parser.add_argument("--checkpoint-dir", default=str(CHECKPOINT))
    parser.add_argument("--eval-output", default=str(REPORT))
    parser.add_argument("--predictions-output", default=str(ROOT / "reports/vlm/symbol_tile_detector_v20_predictions.jsonl"))
    parser.add_argument("--weights", choices=["coco", "none"], default="coco")
    parser.add_argument("--anchor-profile", choices=["default", "small", "tiny"], default="default")
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--limit-train-tiles", type=int, default=6000)
    parser.add_argument("--dev-split", default="dev", choices=["train", "dev", "locked", "smoke"])
    parser.add_argument("--locked-split", default="locked", choices=["train", "dev", "locked", "smoke"])
    parser.add_argument("--limit-dev-tiles", type=int, default=2000)
    parser.add_argument("--limit-locked-tiles", type=int, default=2000)
    parser.add_argument("--train-positive-ratio", type=float, default=0.85)
    parser.add_argument("--train-small-positive-ratio", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--score-threshold-grid", default=None)
    parser.add_argument("--nms-threshold", type=float, default=0.45)
    parser.add_argument("--nms-threshold-grid", default=None)
    parser.add_argument("--merge-mode", choices=["nms", "wbf"], default="nms")
    parser.add_argument("--merge-mode-grid", default=None)
    parser.add_argument("--max-per-page", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260510)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    data_dir = Path(args.data)
    train_source_rows = load_jsonl(data_dir / "train.jsonl")
    if args.train_small_positive_ratio > 0:
        train_rows = sample_tiles_area_aware(
            train_source_rows,
            args.limit_train_tiles,
            args.seed,
            positive_ratio=args.train_positive_ratio,
            small_positive_ratio=args.train_small_positive_ratio,
        )
    else:
        train_rows = sample_tiles(train_source_rows, args.limit_train_tiles, args.seed, positive_ratio=args.train_positive_ratio)
    dev_rows = sample_tiles(load_jsonl(data_dir / f"{args.dev_split}.jsonl"), args.limit_dev_tiles, args.seed + 1, positive_ratio=0.85)
    locked_rows = sample_tiles(load_jsonl(data_dir / f"{args.locked_split}.jsonl"), args.limit_locked_tiles, args.seed + 2, positive_ratio=0.85)

    model = make_model(args.weights, num_classes=len(LABELS) + 1, anchor_profile=args.anchor_profile).to(device)
    if args.init_checkpoint:
        model.load_state_dict(torch.load(source_path(args.init_checkpoint), map_location="cpu"))
    epoch_log: list[dict[str, Any]] = []
    if not args.eval_only:
        train_loader = DataLoader(
            SymbolTileDataset(train_rows, augment=True, seed=args.seed),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=collate,
        )
        optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.1)
        for epoch in range(1, args.epochs + 1):
            row = train_epoch(model, train_loader, optimizer, device)
            scheduler.step()
            row["epoch"] = epoch
            epoch_log.append(row)

    score_grid = [args.score_threshold]
    nms_grid = [args.nms_threshold]
    if args.score_threshold_grid:
        score_grid = [float(item) for item in args.score_threshold_grid.split(",") if item.strip()]
    if args.nms_threshold_grid:
        nms_grid = [float(item) for item in args.nms_threshold_grid.split(",") if item.strip()]
    merge_grid = [args.merge_mode]
    if args.merge_mode_grid:
        merge_grid = [item.strip() for item in args.merge_mode_grid.split(",") if item.strip()]
    dev_page_preds, dev_page_golds = collect_page_predictions(model, dev_rows, device, args.batch_size, args.num_workers)
    locked_page_preds, locked_page_golds = collect_page_predictions(model, locked_rows, device, args.batch_size, args.num_workers)
    grid_reports: list[dict[str, Any]] = []
    for score_threshold in score_grid:
        for nms_threshold in nms_grid:
            for merge_mode in merge_grid:
                dev_eval_grid, _ = score_page_predictions(
                    dev_page_preds, dev_page_golds, score_threshold, nms_threshold, args.max_per_page, merge_mode, len(dev_rows)
                )
                grid_reports.append(
                    {"score_threshold": score_threshold, "nms_threshold": nms_threshold, "merge_mode": merge_mode, "dev": dev_eval_grid}
                )
    grid_reports.sort(
        key=lambda row: (
            row["dev"]["symbol_bbox_center_recall"],
            row["dev"]["symbol_bbox_iou_0_30"]["recall"],
            -row["dev"]["candidate_inflation"],
        ),
        reverse=True,
    )
    selected = grid_reports[0] if grid_reports else {"score_threshold": args.score_threshold, "nms_threshold": args.nms_threshold, "merge_mode": args.merge_mode}
    dev_eval = selected["dev"] if "dev" in selected else score_page_predictions(
        dev_page_preds,
        dev_page_golds,
        float(selected["score_threshold"]),
        float(selected["nms_threshold"]),
        args.max_per_page,
        str(selected.get("merge_mode") or args.merge_mode),
        len(dev_rows),
    )[0]
    locked_eval, locked_predictions = score_page_predictions(
        locked_page_preds,
        locked_page_golds,
        float(selected["score_threshold"]),
        float(selected["nms_threshold"]),
        args.max_per_page,
        str(selected.get("merge_mode") or args.merge_mode),
        len(locked_rows),
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / "model.pt")
    write_json(
        checkpoint_dir / "model_metadata.json",
        {
            "model_type": "symbol_tile_detector_v20_fasterrcnn_mobilenet_fpn",
            "labels": LABELS,
            "weights": args.weights,
            "anchor_profile": args.anchor_profile,
            "runtime_contract": {
                "model_input_features": ["image_tile_pixels"],
                "forbidden_runtime_features": FORBIDDEN_RUNTIME_FIELDS,
            },
        },
    )
    report = {
        "version": "symbol_tile_detector_v20_eval",
        "claim_boundary": "Tile detector body localization audit. Type-head integration is separate.",
        "source_integrity": {
            "model_input": "raster_tile_pixels_only",
            "raw_label_or_semantic_type_used_as_runtime_feature": False,
            "forbidden_runtime_features": FORBIDDEN_RUNTIME_FIELDS,
            "locked_gold_use": "evaluation_only",
        },
        "dataset": rel(data_dir),
        "checkpoint": rel(checkpoint_dir / "model.pt"),
        "config": vars(args) | {"device": str(device)},
        "counts": {
            "train_tiles": len(train_rows),
            "dev_tiles": len(dev_rows),
            "locked_tiles": len(locked_rows),
            "dev_split": args.dev_split,
            "locked_split": args.locked_split,
            "train_positive_tiles": sum(1 for row in train_rows if int((row.get("target_counts") or {}).get("symbols") or 0) > 0),
            "dev_positive_tiles": sum(1 for row in dev_rows if int((row.get("target_counts") or {}).get("symbols") or 0) > 0),
            "locked_positive_tiles": sum(1 for row in locked_rows if int((row.get("target_counts") or {}).get("symbols") or 0) > 0),
            "train_small_or_tiny_positive_tiles": sum(
                1
                for row in train_rows
                if target_area_buckets(row) & {"tiny_le_64", "small_le_256"}
            ),
        },
        "epoch_log": epoch_log,
        "threshold_grid": grid_reports,
        "selected_thresholds": {
            "score_threshold": float(selected["score_threshold"]),
            "nms_threshold": float(selected["nms_threshold"]),
            "merge_mode": str(selected.get("merge_mode") or args.merge_mode),
        },
        "dev": dev_eval,
        "locked": locked_eval,
        "gate": {
            "stage_1_center_recall_min_0_70": locked_eval["symbol_bbox_center_recall"] >= 0.70,
            "stage_1_candidate_inflation_max_30": locked_eval["candidate_inflation"] <= 30.0,
            "beats_legacy_center_recall_0_220663": locked_eval["symbol_bbox_center_recall"] > 0.220663,
        },
        "memory_audit": memory_audit(device),
    }
    write_json(Path(args.eval_output), report)
    write_jsonl(Path(args.predictions_output), locked_predictions)
    print(json.dumps({"locked": locked_eval, "checkpoint": rel(checkpoint_dir / "model.pt")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
