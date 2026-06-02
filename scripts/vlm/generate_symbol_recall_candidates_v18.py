#!/usr/bin/env python3
"""Generate inference-time raster symbol recall candidates and score them."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/vlm"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_topology_relations_v18 import bbox, integrity, iou, write_json, write_jsonl  # noqa: E402
from nms_topology_relations_v18 import load_jsonl  # noqa: E402
from train_missing_symbol_recall_expert_v18 import infer_features, score  # noqa: E402

DEFAULT_INPUT = REPORT / "detector_adapter_v18_symbol_proposal_combined.jsonl"
DEFAULT_MODEL = ROOT / "checkpoints/missing_symbol_recall_expert_v18/model.json"
DEFAULT_OUTPUT = REPORT / "detector_adapter_v18_symbol_recall_expert.jsonl"
DEFAULT_AUDIT = REPORT / "detector_adapter_v18_symbol_recall_expert_audit.json"


def resolve_image(path: str | None) -> Path | None:
    if not path:
        return None
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def image_array(row: dict[str, Any], cache: dict[str, np.ndarray]) -> np.ndarray | None:
    image_path = resolve_image(str(row.get("image") or ""))
    if image_path is None or not image_path.exists():
        return None
    key = str(image_path)
    if key not in cache:
        cache[key] = np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)
    return cache[key]


def crop_stats(arr: np.ndarray, box: list[float]) -> dict[str, float]:
    height, width = arr.shape
    x1 = max(0, min(width - 1, int(math.floor(box[0]))))
    y1 = max(0, min(height - 1, int(math.floor(box[1]))))
    x2 = max(x1 + 1, min(width, int(math.ceil(box[2]))))
    y2 = max(y1 + 1, min(height, int(math.ceil(box[3]))))
    crop = arr[y1:y2, x1:x2]
    if crop.size == 0:
        return {
            "crop_dark_density_205": 0.0,
            "crop_dark_density_225": 0.0,
            "crop_mean_gray": 0.0,
            "crop_std_gray": 0.0,
            "crop_edge_touch_dark_ratio": 0.0,
        }
    border = np.concatenate([crop[0, :], crop[-1, :], crop[:, 0], crop[:, -1]])
    return {
        "crop_dark_density_205": round(float((crop <= 205).mean()), 6),
        "crop_dark_density_225": round(float((crop <= 225).mean()), 6),
        "crop_mean_gray": round(float(crop.mean()), 6),
        "crop_std_gray": round(float(crop.std()), 6),
        "crop_edge_touch_dark_ratio": round(float((border <= 205).mean()), 6) if border.size else 0.0,
    }


def box_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def clip_box(box: list[float], width: int, height: int) -> list[int] | None:
    x1 = max(0, min(width - 1, int(math.floor(box[0]))))
    y1 = max(0, min(height - 1, int(math.floor(box[1]))))
    x2 = max(x1 + 1, min(width, int(math.ceil(box[2]))))
    y2 = max(y1 + 1, min(height, int(math.ceil(box[3]))))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def candidate_stream(row: dict[str, Any]) -> list[dict[str, Any]]:
    return list(((row.get("scene_graph") or {}).get("candidate_stream") or []))


def stream_counts(stream: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(cand.get("family") or "unknown") for cand in stream))


def existing_symbol_boxes(stream: list[dict[str, Any]]) -> list[list[float]]:
    boxes: list[list[float]] = []
    for cand in stream:
        if cand.get("family") != "symbol":
            continue
        b = bbox(cand.get("bbox"))
        if b is not None:
            boxes.append(b)
    return boxes


def component_boxes(arr: np.ndarray, threshold: int, args: argparse.Namespace) -> list[dict[str, Any]]:
    import cv2

    mask = arr <= threshold
    n_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype("uint8"), 8)
    out: list[dict[str, Any]] = []
    for idx in range(1, int(n_labels)):
        x, y, w, h, area = [int(v) for v in stats[idx]]
        if area < args.min_component_area or area > args.max_component_area:
            continue
        if w < args.min_side or h < args.min_side or w > args.max_side or h > args.max_side:
            continue
        aspect = w / max(h, 1)
        if aspect < args.min_aspect or aspect > args.max_aspect:
            continue
        fill = area / max(w * h, 1)
        if fill < args.min_fill:
            continue
        for pad in args.component_pads:
            box = clip_box([x - pad, y - pad, x + w + pad, y + h + pad], arr.shape[1], arr.shape[0])
            if box is None:
                continue
            out.append(
                {
                    "bbox": box,
                    "proposal_kind": f"micro_component_t{threshold}_p{pad}",
                    "component_area": int(area),
                    "component_fill": round(float(fill), 6),
                    "threshold": int(threshold),
                    "pad": int(pad),
                }
            )
    return out


def anchor_boxes(arr: np.ndarray, threshold: int, args: argparse.Namespace) -> list[dict[str, Any]]:
    ys, xs = np.nonzero(arr <= threshold)
    if len(xs) == 0:
        return []
    stride = int(args.anchor_stride)
    buckets = Counter((int(x // stride) * stride + stride // 2, int(y // stride) * stride + stride // 2) for x, y in zip(xs, ys, strict=True))
    out: list[dict[str, Any]] = []
    for (cx, cy), count in buckets.most_common(int(args.anchor_bucket_cap)):
        local = arr[max(0, cy - 4) : min(arr.shape[0], cy + 5), max(0, cx - 4) : min(arr.shape[1], cx + 5)]
        density = float((local <= threshold).mean()) if local.size else 0.0
        if density < args.anchor_min_density:
            continue
        for side in args.anchor_sides:
            box = clip_box([cx - side / 2.0, cy - side / 2.0, cx + side / 2.0, cy + side / 2.0], arr.shape[1], arr.shape[0])
            if box is None:
                continue
            out.append(
                {
                    "bbox": box,
                    "proposal_kind": f"dark_micro_anchor_t{threshold}_s{side}",
                    "local_dark_density": round(density, 6),
                    "dark_bucket_count": int(count),
                    "threshold": int(threshold),
                    "anchor_side": int(side),
                }
            )
            if len(out) >= args.anchor_candidate_cap:
                return out
    return out


def proposal_features(row: dict[str, Any], arr: np.ndarray, stream: list[dict[str, Any]], box: list[float], raw: dict[str, Any]) -> dict[str, Any]:
    counts = stream_counts(stream)
    features = {
        "existing_symbol_candidate_count": counts.get("symbol", 0),
        "space_candidate_count": counts.get("space", 0),
        "boundary_candidate_count": counts.get("boundary", 0),
        "text_candidate_count": counts.get("text", 0),
        **crop_stats(arr, box),
        "component_area": float(raw.get("component_area") or 0.0),
        "component_fill": float(raw.get("component_fill") or 0.0),
        "local_dark_density": float(raw.get("local_dark_density") or 0.0),
        "dark_bucket_count": float(raw.get("dark_bucket_count") or 0.0),
    }
    return features


def score_proposal(
    row: dict[str, Any],
    arr: np.ndarray,
    stream: list[dict[str, Any]],
    raw: dict[str, Any],
    model: dict[str, Any],
    image_cache: dict[str, Image.Image],
) -> dict[str, Any]:
    box = raw["bbox"]
    record = {
        "id": f"{row.get('id')}|symbol_recall_window|{box[0]}_{box[1]}_{box[2]}_{box[3]}",
        "row_id": row.get("id"),
        "image": row.get("image"),
        "image_size": row.get("image_size") or [arr.shape[1], arr.shape[0]],
        "bbox": box,
        "features": proposal_features(row, arr, stream, box, raw),
    }
    return {
        **raw,
        "objectness_score": round(score(record, model, image_cache), 6),
        "model_features": infer_features(record),
    }


def dedupe_ranked(raw_items: list[dict[str, Any]], existing_boxes: list[list[float]], args: argparse.Namespace) -> list[dict[str, Any]]:
    ranked = sorted(raw_items, key=lambda item: (float(item.get("objectness_score") or 0.0), -box_area(item["bbox"])), reverse=True)
    kept: list[dict[str, Any]] = []
    for item in ranked:
        if float(item.get("objectness_score") or 0.0) < args.score_threshold:
            continue
        if any(iou(item["bbox"], existing) > args.existing_duplicate_iou for existing in existing_boxes):
            continue
        if any(iou(item["bbox"], prev["bbox"]) > args.new_duplicate_iou for prev in kept):
            continue
        kept.append(item)
        if len(kept) >= args.max_added_per_row:
            break
    return kept


def make_candidate(row: dict[str, Any], item: dict[str, Any], index: int) -> dict[str, Any]:
    row_id = str(row.get("id"))
    box = item["bbox"]
    candidate_id = f"{row_id}_symbol_recall_expert_v18_{index:03d}_{box[0]}_{box[1]}_{box[2]}_{box[3]}"
    payload = {
        "candidate_kind": "symbol_recall_expert_v18",
        "proposal_kind": item.get("proposal_kind"),
        "source": "symbol_recall_expert_v18",
        "symbol_type": "symbol",
        "typed_symbol_type": "symbol",
        "type_label_adopted": False,
        "objectness_score": item["objectness_score"],
        "abstain": False,
        "raster_threshold": item.get("threshold"),
        "component_area": item.get("component_area"),
        "component_fill": item.get("component_fill"),
        "local_dark_density": item.get("local_dark_density"),
        "model_features": item.get("model_features"),
    }
    return {
        "candidate_id": candidate_id,
        "candidate_contract_version": "detector_candidate_contract_v1",
        "row_id": row_id,
        "family": "symbol",
        "route": "symbol_fixture",
        "candidate_type": "symbol",
        "bbox": box,
        "confidence": item["objectness_score"],
        "payload": payload,
        "source_integrity": integrity(),
        "provenance": {
            "input_source": "symbol_recall_expert_v18",
            "raw_candidate_id": candidate_id,
            "row_id": row_id,
            "family": "symbol",
            "route": "symbol_fixture",
            "raster_only": True,
            "image": row.get("image"),
        },
        "audit_trace": {
            "stage": "symbol_recall_expert_v18",
            "proposal_kind": item.get("proposal_kind"),
            "bbox": box,
            "objectness_score": item["objectness_score"],
            "source_integrity": integrity(),
        },
    }


def generate_for_row(
    row: dict[str, Any],
    model: dict[str, Any],
    image_np_cache: dict[str, np.ndarray],
    image_pil_cache: dict[str, Image.Image],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Counter[str], list[dict[str, Any]]]:
    out = json.loads(json.dumps(row, ensure_ascii=False))
    arr = image_array(out, image_np_cache)
    stream = candidate_stream(out)
    counts: Counter[str] = Counter({"rows": 1, "input_candidates": len(stream)})
    if arr is None:
        counts["missing_image_rows"] += 1
        return out, counts, []

    raw: list[dict[str, Any]] = []
    for threshold in args.thresholds:
        raw.extend(component_boxes(arr, int(threshold), args))
        raw.extend(anchor_boxes(arr, int(threshold), args))
    counts["raw_windows"] = len(raw)
    scored = [score_proposal(out, arr, stream, item, model, image_pil_cache) for item in raw]
    selected = dedupe_ranked(scored, existing_symbol_boxes(stream), args)
    added = [make_candidate(out, item, index) for index, item in enumerate(selected)]
    new_stream = stream + added
    scene = dict(out.get("scene_graph") if isinstance(out.get("scene_graph"), dict) else {})
    scene["candidate_stream"] = new_stream
    scene["candidate_counts"] = stream_counts(new_stream)
    scene["symbol_recall_expert_v18"] = {
        "enabled": True,
        "added_candidates": len(added),
        "raw_windows": len(raw),
        "score_threshold": args.score_threshold,
        "max_added_per_row": args.max_added_per_row,
    }
    out["scene_graph"] = scene
    counts["added_symbol_recall_candidates"] = len(added)
    counts["output_candidates"] = len(new_stream)
    examples = [
        {
            "row_id": out.get("id"),
            "candidate_id": cand.get("candidate_id"),
            "bbox": cand.get("bbox"),
            "score": cand.get("confidence"),
            "proposal_kind": (cand.get("payload") or {}).get("proposal_kind"),
        }
        for cand in added[:10]
    ]
    return out, counts, examples


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def run(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model = json.loads(Path(args.model).read_text(encoding="utf-8"))
    rows = load_jsonl(Path(args.input))
    if args.limit:
        rows = rows[: args.limit]
    image_np_cache: dict[str, np.ndarray] = {}
    image_pil_cache: dict[str, Image.Image] = {}
    out_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    per_row_added: Counter[int] = Counter()
    examples: list[dict[str, Any]] = []

    for row in rows:
        out, row_counts, row_examples = generate_for_row(row, model, image_np_cache, image_pil_cache, args)
        counts.update(row_counts)
        per_row_added[int(row_counts.get("added_symbol_recall_candidates", 0))] += 1
        if len(examples) < 100:
            examples.extend(row_examples[: max(0, 100 - len(examples))])
        out_rows.append(out)

    audit = {
        "task": "IMG-MOE-V18-REBUILD-001.step_j3_generate_inference_time_symbol_recall_candidates",
        "input": str(args.input),
        "output": str(args.output),
        "model": str(args.model),
        "rows": len(out_rows),
        "counts": dict(counts),
        "added_per_row_histogram": dict(sorted(per_row_added.items())),
        "examples": examples,
        "params": {
            "thresholds": args.thresholds,
            "score_threshold": args.score_threshold,
            "max_added_per_row": args.max_added_per_row,
            "existing_duplicate_iou": args.existing_duplicate_iou,
            "new_duplicate_iou": args.new_duplicate_iou,
            "anchor_sides": args.anchor_sides,
            "component_pads": args.component_pads,
        },
        "source_integrity": integrity(),
        "gold_loaded": False,
        "gold_used_for_inference": False,
        "adopted": False,
    }
    return out_rows, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--thresholds", type=parse_csv_ints, default=parse_csv_ints("185,205,225"))
    parser.add_argument("--component-pads", type=parse_csv_ints, default=parse_csv_ints("1,3"))
    parser.add_argument("--anchor-sides", type=parse_csv_ints, default=parse_csv_ints("5,7,9,13"))
    parser.add_argument("--min-component-area", type=int, default=1)
    parser.add_argument("--max-component-area", type=int, default=220)
    parser.add_argument("--min-side", type=int, default=1)
    parser.add_argument("--max-side", type=int, default=36)
    parser.add_argument("--min-aspect", type=float, default=0.08)
    parser.add_argument("--max-aspect", type=float, default=12.0)
    parser.add_argument("--min-fill", type=float, default=0.02)
    parser.add_argument("--anchor-stride", type=int, default=8)
    parser.add_argument("--anchor-bucket-cap", type=int, default=240)
    parser.add_argument("--anchor-candidate-cap", type=int, default=480)
    parser.add_argument("--anchor-min-density", type=float, default=0.05)
    parser.add_argument("--score-threshold", type=float, default=0.564182)
    parser.add_argument("--max-added-per-row", type=int, default=64)
    parser.add_argument("--existing-duplicate-iou", type=float, default=0.82)
    parser.add_argument("--new-duplicate-iou", type=float, default=0.72)
    args = parser.parse_args()

    rows, audit = run(args)
    write_jsonl(Path(args.output), rows)
    write_json(Path(args.audit), audit)
    print(json.dumps({"rows": audit["rows"], "counts": audit["counts"], "added_per_row_histogram": audit["added_per_row_histogram"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
