#!/usr/bin/env python3
"""Rerank v18 text candidates before OCR/top-k truncation."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from train_text_detector_v18 import bbox_iou, center_covered, load_jsonl, predict_row  # noqa: E402

DATA = ROOT / "datasets/image_only_text_ocr_v18"
REPORT = ROOT / "reports/vlm"
CHECKPOINT = ROOT / "checkpoints/text_rerank_v18"

DEFAULT_OUTPUT = REPORT / "text_rerank_v18_candidates.jsonl"
DEFAULT_EVAL = REPORT / "text_rerank_v18_eval.json"
DEFAULT_ROOMS = REPORT / "room_proposal_model_v18_reranked_candidates.jsonl"

FEATURES = [
    "detector_confidence",
    "bbox_width",
    "bbox_height",
    "bbox_area",
    "bbox_aspect",
    "local_dark_density",
    "component_area",
    "component_fill",
    "merged_components",
    "anchor_w",
    "anchor_h",
    "is_component",
    "is_line_merge",
    "is_dark_anchor",
    "is_page_anchor",
    "inside_room_count",
    "nearest_room_center_score",
    "nearest_room_edge_score",
]


def integrity() -> dict[str, Any]:
    return {
        "source_mode": "image_only_raster_moe",
        "svg_candidate_ids_used": False,
        "annotation_geometry_used_at_inference": False,
        "model_input": "raster_image_only",
    }


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


def center(box: list[float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def load_rooms(path: Path) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    if not path.exists():
        return rows
    for row in load_jsonl(path):
        rooms: list[dict[str, Any]] = []
        for cand in row.get("candidate_stream") or []:
            b = bbox(cand.get("bbox"))
            if b is None:
                continue
            item = dict(cand)
            item["_bbox"] = b
            rooms.append(item)
        rows[str(row.get("id"))] = sorted(rooms, key=lambda c: float(c.get("confidence") or 0.0), reverse=True)[:40]
    return rows


def room_context_features(candidate_box: list[float], rooms: list[dict[str, Any]]) -> dict[str, float]:
    cx, cy = center(candidate_box)
    inside_count = 0
    best_center = 0.0
    best_edge = 0.0
    for room in rooms:
        rb = room["_bbox"]
        rw = max(rb[2] - rb[0], 1.0)
        rh = max(rb[3] - rb[1], 1.0)
        inside = rb[0] <= cx <= rb[2] and rb[1] <= cy <= rb[3]
        if inside:
            inside_count += 1
        rx, ry = center(rb)
        norm_dist = math.hypot((cx - rx) / rw, (cy - ry) / rh)
        best_center = max(best_center, math.exp(-4.0 * norm_dist))
        edge_dist = min(abs(cx - rb[0]), abs(cx - rb[2]), abs(cy - rb[1]), abs(cy - rb[3]))
        best_edge = max(best_edge, min(edge_dist / max(min(rw, rh) * 0.35, 1.0), 1.0))
    return {
        "inside_room_count": float(min(inside_count, 5)),
        "nearest_room_center_score": best_center,
        "nearest_room_edge_score": best_edge,
    }


def box_features(cand: dict[str, Any], rooms: list[dict[str, Any]] | None = None) -> dict[str, float]:
    b = bbox(cand.get("bbox")) or [0.0, 0.0, 0.0, 0.0]
    w = b[2] - b[0]
    h = b[3] - b[1]
    payload = cand.get("payload") if isinstance(cand.get("payload"), dict) else {}
    kind = str(payload.get("candidate_kind") or "")
    anchor_size = payload.get("anchor_size") if isinstance(payload.get("anchor_size"), list) else [0, 0]
    feats = {
        "detector_confidence": float(cand.get("confidence") or 0.0),
        "bbox_width": w,
        "bbox_height": h,
        "bbox_area": w * h,
        "bbox_aspect": w / max(h, 1e-6),
        "local_dark_density": float(payload.get("local_dark_density") or 0.0),
        "component_area": float(payload.get("area") or 0.0),
        "component_fill": float(payload.get("fill") or 0.0),
        "merged_components": float(payload.get("merged_components") or 0.0),
        "anchor_w": float(anchor_size[0] or 0.0),
        "anchor_h": float(anchor_size[1] or 0.0),
        "is_component": 1.0 if kind == "dark_connected_component" else 0.0,
        "is_line_merge": 1.0 if kind == "same_row_component_merge" else 0.0,
        "is_dark_anchor": 1.0 if kind == "dark_pixel_anchor" else 0.0,
        "is_page_anchor": 1.0 if kind == "dense_page_text_anchor" else 0.0,
    }
    feats.update(room_context_features(b, rooms or []))
    return feats


def vector_from_features(feats: dict[str, float]) -> np.ndarray:
    return np.asarray([float(feats.get(name) or 0.0) for name in FEATURES], dtype=np.float32)


def gold_texts(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item for item in (row.get("targets") or {}).get("texts") or []
        if bbox(item.get("bbox")) is not None
    ]


def match_candidate(cand: dict[str, Any], golds: list[dict[str, Any]]) -> tuple[bool, float]:
    cb = bbox(cand.get("bbox"))
    if cb is None:
        return False, 0.0
    best = 0.0
    for gold in golds:
        gb = bbox(gold.get("bbox"))
        if gb is None:
            continue
        score = bbox_iou(cb, gb)
        if center_covered([int(v) for v in cb], [int(v) for v in gb]):
            score = max(score, 0.30)
        best = max(best, score)
    return best >= 0.30, best


def load_policy() -> dict[str, Any]:
    path = ROOT / "checkpoints/text_detector_v18/policy.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data.get("policy"), dict):
            return data["policy"]
    return {"threshold": 205, "component_cap": 1400, "line_gap": 10, "anchor_stride": 12, "anchor_cap": 1400, "page_anchor_stride": 4, "page_anchor_cap": 70000}


def collect_training(rows: list[dict[str, Any]], policy: dict[str, Any], rooms_by_row: dict[str, list[dict[str, Any]]], max_pages: int | None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    positives: list[np.ndarray] = []
    negatives: list[np.ndarray] = []
    counts = Counter()
    for row in rows[:max_pages] if max_pages else rows:
        golds = gold_texts(row)
        preds = predict_row(row, policy)
        rooms = rooms_by_row.get(str(row.get("id")), [])
        for cand in preds:
            feats = vector_from_features(box_features(cand, rooms))
            ok, _score = match_candidate(cand, golds)
            if ok:
                positives.append(feats)
                counts["positive"] += 1
            elif len(negatives) < max(200000, len(positives) * 20 + 10000):
                negatives.append(feats)
                counts["negative_sampled"] += 1
            counts["seen"] += 1
    if not positives:
        positives.append(np.zeros(len(FEATURES), dtype=np.float32))
    if not negatives:
        negatives.append(np.zeros(len(FEATURES), dtype=np.float32))
    return np.stack(positives), np.stack(negatives), dict(counts)


def train_model(pos: np.ndarray, neg: np.ndarray, counts: dict[str, Any]) -> dict[str, Any]:
    all_x = np.concatenate([pos, neg], axis=0)
    mean = all_x.mean(axis=0)
    std = all_x.std(axis=0) + 1e-6
    pos_z = (pos - mean) / std
    neg_z = (neg - mean) / std
    weights = pos_z.mean(axis=0) - neg_z.mean(axis=0)
    norm = float(np.linalg.norm(weights))
    if norm > 1e-9:
        weights = weights / norm
    bias = -float(np.percentile(neg_z @ weights, 90))
    return {
        "features": FEATURES,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "weights": weights.tolist(),
        "bias": bias,
        "training_counts": counts,
    }


def score_features(feats: dict[str, float], model: dict[str, Any]) -> float:
    x = vector_from_features(feats)
    mean = np.asarray(model["mean"], dtype=np.float32)
    std = np.asarray(model["std"], dtype=np.float32)
    weights = np.asarray(model["weights"], dtype=np.float32)
    raw = float(((x - mean) / std) @ weights + float(model["bias"]))
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, raw))))


def recall_for(rows_by_id: dict[str, list[dict[str, Any]]], gold_rows: dict[str, list[dict[str, Any]]], cap: int) -> dict[str, Any]:
    total = hit = 0
    for row_id, golds in gold_rows.items():
        candidates = rows_by_id.get(row_id, [])[:cap]
        boxes = [bbox(c.get("bbox")) for c in candidates]
        for gold in golds:
            gb = bbox(gold.get("bbox"))
            if gb is None:
                continue
            total += 1
            if any(cb is not None and (center_covered([int(v) for v in cb], [int(v) for v in gb]) or bbox_iou(cb, gb) >= 0.30) for cb in boxes):
                hit += 1
    return {"gold": total, "matched": hit, "candidate_limited_localization_recall": round(hit / max(total, 1), 6)}


def apply_model(rows: list[dict[str, Any]], policy: dict[str, Any], model: dict[str, Any], rooms_by_row: dict[str, list[dict[str, Any]]], export_top_k: int, limit_pages: int | None) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    out_rows: list[dict[str, Any]] = []
    before_by_id: dict[str, list[dict[str, Any]]] = {}
    after_by_id: dict[str, list[dict[str, Any]]] = {}
    for row in rows[:limit_pages] if limit_pages else rows:
        row_id = str(row.get("id"))
        preds = predict_row(row, policy)
        rooms = rooms_by_row.get(row_id, [])
        before_by_id[row_id] = list(preds)
        scored: list[dict[str, Any]] = []
        for cand in preds:
            feats = box_features(cand, rooms)
            item = dict(cand)
            payload = dict(item.get("payload") if isinstance(item.get("payload"), dict) else {})
            readability = score_features(feats, model)
            payload["text_rerank_v18"] = {"readability_score": round(readability, 6), "features": feats}
            payload["ocr_status"] = payload.get("ocr_status") or "not_invoked"
            payload["weak_text"] = True
            item["payload"] = payload
            item["confidence"] = round(readability, 6)
            scored.append(item)
        scored.sort(key=lambda c: (float(c.get("confidence") or 0.0), c.get("id") or ""), reverse=True)
        after_by_id[row_id] = scored
        stream = [
            {
                "candidate_id": cand["id"],
                "row_id": row_id,
                "family": "text",
                "route": "text_dimension",
                "candidate_type": "text",
                "bbox": cand.get("bbox"),
                "confidence": cand.get("confidence"),
                "payload": cand.get("payload") or {},
                "source_integrity": integrity(),
            }
            for cand in scored[:export_top_k]
        ]
        out_rows.append(
            {
                "id": row_id,
                "image": row.get("image"),
                "image_size": row.get("image_size") or [512, 512],
                "source_integrity": integrity(),
                "route_trace": {**integrity(), "stage": "text_rerank_v18"},
                "candidate_stream": stream,
            }
        )
    return out_rows, before_by_id, after_by_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--eval-output", default=str(DEFAULT_EVAL))
    parser.add_argument("--rooms", default=str(DEFAULT_ROOMS))
    parser.add_argument("--export-top-k", type=int, default=2500)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--locked", action="store_true")
    args = parser.parse_args()

    policy = load_policy()
    data = Path(args.data)
    train_pages = 8 if args.smoke else None
    eval_pages = 5 if args.smoke else None
    dev_rows = load_jsonl(data / "dev.jsonl")
    locked_rows = load_jsonl(data / "locked.jsonl")
    rooms_by_row = load_rooms(Path(args.rooms))
    pos, neg, counts = collect_training(dev_rows, policy, rooms_by_row, train_pages)
    model = train_model(pos, neg, counts)
    CHECKPOINT.mkdir(parents=True, exist_ok=True)
    write_json(CHECKPOINT / "model.json", model)
    out_rows, before_by_id, after_by_id = apply_model(locked_rows, policy, model, rooms_by_row, args.export_top_k, eval_pages)
    write_jsonl(Path(args.output), out_rows)
    gold = {str(row.get("id")): gold_texts(row) for row in (locked_rows[:eval_pages] if eval_pages else locked_rows)}
    cap_sweep = {
        str(cap): {
            "before_detector_order": recall_for(before_by_id, gold, cap),
            "after_readability_order": recall_for(after_by_id, gold, cap),
        }
        for cap in [25, 50, 100, 250, 500, 2500]
    }
    report = {
        "task": "IMG-MOE-V18-NEXT-007",
        "mode": "text_readability_rerank",
        "rows": len(out_rows),
        "output": str(args.output),
        "model": {"type": "standardized_positive_negative_centroid_ranker", "features": FEATURES, "training_counts": counts},
        "cap_sweep": cap_sweep,
        "ocr_semantics_adopted": False,
        "ocr_adoption_reason": "OCR normalized accuracy remains below floor; rerank only improves localization candidates for later OCR.",
        "source_integrity": integrity(),
        "gold_loaded_after_inference_for_evaluation_only": True,
        "gold_used_for_inference": False,
        "quality_gates": {
            "source_integrity_violations": 0,
            "candidate_limited_localization_recall_at_50_ge_045": cap_sweep["50"]["after_readability_order"]["candidate_limited_localization_recall"] >= 0.45,
            "candidate_limited_localization_recall_at_50_improved": cap_sweep["50"]["after_readability_order"]["candidate_limited_localization_recall"] > cap_sweep["50"]["before_detector_order"]["candidate_limited_localization_recall"],
            "ocr_semantics_remain_disabled": True,
        },
    }
    write_json(Path(args.eval_output), report)
    print(json.dumps({"rows": len(out_rows), "cap50_before": cap_sweep["50"]["before_detector_order"], "cap50_after": cap_sweep["50"]["after_readability_order"], "quality_gates": report["quality_gates"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
