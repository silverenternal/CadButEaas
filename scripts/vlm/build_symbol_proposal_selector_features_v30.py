#!/usr/bin/env python3
"""Build candidate-level features for the v30 symbol proposal selector."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from train_symbol_tile_detector_v20 import area_bucket, bbox_iou, center_covered, load_jsonl, rel, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]


def load_preds(path: Path) -> dict[str, list[dict[str, Any]]]:
    return {str(row.get("row_id")): list(row.get("predicted_symbols") or []) for row in load_jsonl(path)}


def load_golds(path: Path) -> dict[str, list[dict[str, Any]]]:
    pages: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in load_jsonl(path):
        row_id = str(row.get("row_id"))
        target_id = str(row.get("target_id") or f"{row_id}_{len(pages[row_id])}")
        pages[row_id][target_id] = {
            "target_id": target_id,
            "bbox": [float(v) for v in row.get("page_bbox") or []],
            "label": str(row.get("label") or "generic_symbol"),
            "area_bucket": str(row.get("area_bucket") or area_bucket([float(v) for v in row.get("page_bbox") or [0, 0, 1, 1]])),
        }
    return {key: list(value.values()) for key, value in pages.items()}


def box_features(box: list[float]) -> dict[str, float]:
    width = max(0.0, box[2] - box[0])
    height = max(0.0, box[3] - box[1])
    area = width * height
    aspect = width / max(height, 1e-6)
    return {"width": width, "height": height, "area": area, "aspect": aspect}


def center_distance(left: list[float], right: list[float]) -> float:
    lcx = (left[0] + left[2]) / 2.0
    lcy = (left[1] + left[3]) / 2.0
    rcx = (right[0] + right[2]) / 2.0
    rcy = (right[1] + right[3]) / 2.0
    return ((lcx - rcx) ** 2 + (lcy - rcy) ** 2) ** 0.5


def source_merge(mask: dict[str, list[dict[str, Any]]], center: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row_id in set(mask) | set(center):
        rows = []
        for pred in mask.get(row_id, []):
            item = dict(pred)
            item["proposal_source"] = "mask_v28"
            rows.append(item)
        for pred in center.get(row_id, []):
            item = dict(pred)
            item["proposal_source"] = "center_branch_v30"
            rows.append(item)
        out[row_id] = rows
    return out


def nearest_other_features(index: int, preds: list[dict[str, Any]]) -> dict[str, float]:
    box = [float(v) for v in preds[index].get("bbox") or [0, 0, 0, 0]]
    max_iou = 0.0
    overlap_count = 0
    near_center_count = 0
    for other_index, other in enumerate(preds):
        if other_index == index:
            continue
        other_box = [float(v) for v in other.get("bbox") or [0, 0, 0, 0]]
        iou = bbox_iou(box, other_box)
        max_iou = max(max_iou, iou)
        if iou >= 0.30:
            overlap_count += 1
        if center_distance(box, other_box) <= 12.0:
            near_center_count += 1
    return {"max_peer_iou": max_iou, "overlap_peer_count": float(overlap_count), "near_center_peer_count": float(near_center_count)}


def best_gold_label(pred_box: list[float], pred_label: str, golds: list[dict[str, Any]]) -> dict[str, Any]:
    best_iou = 0.0
    best_center = False
    best_gold: dict[str, Any] | None = None
    for gold in golds:
        gold_box = [float(v) for v in gold.get("bbox") or []]
        if len(gold_box) != 4:
            continue
        iou = bbox_iou(pred_box, gold_box)
        if iou > best_iou:
            best_iou = iou
            best_gold = gold
        if center_covered(pred_box, gold_box):
            best_center = True
    positive = bool(best_gold and best_iou >= 0.30)
    typed_positive = bool(positive and str(best_gold.get("label")) == pred_label)
    return {
        "target": int(positive),
        "typed_target": int(typed_positive),
        "best_iou": round(best_iou, 6),
        "center_covers_any_gold": int(best_center),
        "best_gold_label": None if best_gold is None else best_gold.get("label"),
        "best_gold_area_bucket": None if best_gold is None else best_gold.get("area_bucket"),
        "best_gold_target_id": None if best_gold is None else best_gold.get("target_id"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/symbol_center_branch_v30/manifest.json")
    parser.add_argument("--mask-predictions", default="reports/vlm/symbol_yolov8s_seg_rect_v28_smoke_v30_page_predictions.jsonl")
    parser.add_argument("--center-predictions", default="reports/vlm/symbol_center_branch_v30_smoke_predictions.jsonl")
    parser.add_argument("--output", default="datasets/symbol_proposal_selector_v30/smoke_features.jsonl")
    parser.add_argument("--manifest-output", default="datasets/symbol_proposal_selector_v30/manifest.json")
    args = parser.parse_args()

    manifest_path = Path(args.dataset)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    golds = load_golds(ROOT / manifest["outputs"]["smoke_center_targets"])
    candidates = source_merge(load_preds(Path(args.mask_predictions)), load_preds(Path(args.center_predictions)))

    rows: list[dict[str, Any]] = []
    counts = Counter()
    for row_id, preds in candidates.items():
        page_golds = golds.get(row_id, [])
        for index, pred in enumerate(preds):
            box = [float(v) for v in pred.get("bbox") or []]
            if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
                continue
            source = str(pred.get("proposal_source") or "unknown")
            label = str(pred.get("label") or "generic_symbol")
            features = {
                "score": float(pred.get("score", 0.0)),
                "is_mask_v28": float(source == "mask_v28"),
                "is_center_branch_v30": float(source == "center_branch_v30"),
                "label_id": float(pred.get("label_id") or 5),
            }
            features.update(box_features(box))
            features.update(nearest_other_features(index, preds))
            labels = best_gold_label(box, label, page_golds)
            row = {
                "row_id": row_id,
                "candidate_index": index,
                "proposal_source": source,
                "bbox": box,
                "label": label,
                "label_id": int(pred.get("label_id") or 5),
                "features": features,
                "labels": labels,
            }
            rows.append(row)
            counts[f"source:{source}"] += 1
            counts[f"target:{labels['target']}"] += 1
            if labels["target"]:
                counts[f"positive_source:{source}"] += 1

    write_jsonl(Path(args.output), rows)
    out_manifest = {
        "version": "symbol_proposal_selector_features_v30",
        "metric_mode": "smoke",
        "claim_boundary": "Offline selector feature table; gold labels are training/evaluation labels only.",
        "inputs": {
            "dataset": rel(manifest_path),
            "mask_predictions": rel(Path(args.mask_predictions)),
            "center_predictions": rel(Path(args.center_predictions)),
        },
        "outputs": {"features": rel(Path(args.output))},
        "counts": dict(counts) | {"rows": len(rows), "pages": len({row["row_id"] for row in rows})},
        "feature_names": sorted(rows[0]["features"]) if rows else [],
    }
    write_json(Path(args.manifest_output), out_manifest)
    print(json.dumps({"features": rel(Path(args.output)), "counts": out_manifest["counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
