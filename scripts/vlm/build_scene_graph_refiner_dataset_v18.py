#!/usr/bin/env python3
"""Build keep/drop training rows for the v18 scene-graph refiner."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/vlm"
DATASET = ROOT / "datasets/image_only_scene_graph_refiner_v18"

DEFAULT_CANDIDATES = REPORT / "detector_adapter_v18_routed_candidates.jsonl"
DEFAULT_RELATIONS = REPORT / "topology_relations_v18_rerank_features.jsonl"
DEFAULT_OUTPUT = DATASET / "locked.jsonl"
DEFAULT_MANIFEST = DATASET / "manifest.json"
GOLD_FILES = {
    "boundary": ROOT / "datasets/image_only_boundary_detector_v18/locked.jsonl",
    "space": ROOT / "datasets/image_only_room_polygon_v18/locked.jsonl",
    "symbol": ROOT / "datasets/image_only_symbol_detector_v18/locked.jsonl",
    "text": ROOT / "datasets/image_only_text_ocr_v18/locked.jsonl",
}
GOLD_KEYS = {"boundary": "boxes", "space": "rooms", "symbol": "symbols", "text": "texts"}


def integrity() -> dict[str, Any]:
    return {
        "source_mode": "image_only_raster_moe",
        "svg_candidate_ids_used": False,
        "annotation_geometry_used_at_inference": False,
        "model_input": "raster_image_only",
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def iou(left: list[float] | None, right: list[float] | None) -> float:
    if left is None or right is None:
        return 0.0
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    return inter / max(area(left) + area(right) - inter, 1e-9)


def center_covered(pred: list[float], gold: list[float], margin: float = 2.0) -> bool:
    gx, gy = center(gold)
    return pred[0] - margin <= gx <= pred[2] + margin and pred[1] - margin <= gy <= pred[3] + margin


def load_gold() -> dict[str, dict[str, list[dict[str, Any]]]]:
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for family, path in GOLD_FILES.items():
        rows: dict[str, list[dict[str, Any]]] = {}
        for row in load_jsonl(path):
            rows[row["id"]] = [
                item for item in (row.get("targets") or {}).get(GOLD_KEYS[family]) or []
                if bbox(item.get("bbox")) is not None
            ]
        out[family] = rows
    return out


def relation_support(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    support: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for row in load_jsonl(path):
        row_id = str(row.get("row_id"))
        score = float(row.get("confidence") or 0.0)
        rel = str(row.get("relation"))
        for key in ["source_candidate_id", "target_candidate_id"]:
            cid = row.get(key)
            if not cid:
                continue
            support[row_id][cid]["relation_support_count"] += 1.0
            support[row_id][cid]["relation_support_score_sum"] += score
            support[row_id][cid]["relation_support_score_max"] = max(support[row_id][cid]["relation_support_score_max"], score)
            support[row_id][cid][f"support_{rel}"] += 1.0
    return support


def best_match(candidate_box: list[float], golds: list[dict[str, Any]], family: str) -> tuple[dict[str, Any] | None, float, dict[str, Any]]:
    threshold = {"space": 0.50, "boundary": 0.30, "symbol": 0.25, "text": 0.25}.get(family, 0.30)
    best_item: dict[str, Any] | None = None
    best_score = 0.0
    best_center_item: dict[str, Any] | None = None
    best_center_score = 0.0
    for gold in golds:
        gb = bbox(gold.get("bbox"))
        if gb is None:
            continue
        score = iou(candidate_box, gb)
        center_hit = center_covered(candidate_box, gb)
        if center_hit:
            best_center_item = gold
            best_center_score = max(best_center_score, max(score, 0.30))
        if family != "space" and center_hit:
            score = max(score, threshold)
        if score > best_score:
            best_item = gold
            best_score = score
    if best_score >= threshold:
        return best_item, best_score, {"strict_iou50_keep": family == "space", "center_or_iou30_keep": True}
    if family == "space" and best_center_item is not None:
        return best_center_item, best_center_score, {"strict_iou50_keep": False, "center_or_iou30_keep": True}
    return None, best_score, {"strict_iou50_keep": False, "center_or_iou30_keep": False}


def numeric_features(candidate: dict[str, Any], support: dict[str, float]) -> dict[str, float]:
    b = bbox(candidate.get("bbox")) or [0.0, 0.0, 0.0, 0.0]
    width = max(0.0, b[2] - b[0])
    height = max(0.0, b[3] - b[1])
    payload = candidate.get("payload") if isinstance(candidate.get("payload"), dict) else {}
    return {
        "detector_confidence": float(candidate.get("confidence") or 0.0),
        "bbox_area": area(b),
        "bbox_width": width,
        "bbox_height": height,
        "bbox_aspect": width / max(height, 1e-9),
        "relation_support_count": float(support.get("relation_support_count", 0.0)),
        "relation_support_score_sum": float(support.get("relation_support_score_sum", 0.0)),
        "relation_support_score_max": float(support.get("relation_support_score_max", 0.0)),
        "support_bounded_by": float(support.get("support_bounded_by", 0.0)),
        "support_contains_symbol": float(support.get("support_contains_symbol", 0.0)),
        "support_labeled_by_text": float(support.get("support_labeled_by_text", 0.0)),
        "support_adjacent_to": float(support.get("support_adjacent_to", 0.0)),
        "ocr_confidence": float(payload.get("ocr_confidence") or 0.0),
        "type_confidence": float(payload.get("type_confidence") or 0.0),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    gold = load_gold()
    support = relation_support(Path(args.relation_features))
    counts = Counter()
    positives = Counter()
    DATASET.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", encoding="utf-8") as handle:
        for row in load_jsonl(Path(args.candidates)):
            row_id = str(row.get("id"))
            for cand in ((row.get("scene_graph") or {}).get("candidate_stream") or []):
                family = str(cand.get("family") or "unknown")
                cb = bbox(cand.get("bbox"))
                if family not in gold or cb is None:
                    continue
                match, score, match_flags = best_match(cb, gold[family].get(row_id, []), family)
                payload = cand.get("payload") if isinstance(cand.get("payload"), dict) else {}
                predicted_type = payload.get("typed_symbol_type") or payload.get("symbol_type") or cand.get("candidate_type")
                gold_type = None
                typed_correct = None
                if match and family == "symbol":
                    gold_type = match.get("symbol_type") or match.get("semantic_type")
                    typed_correct = str(predicted_type) == str(gold_type)
                item = {
                    "id": f"{row_id}_{cand.get('candidate_id')}",
                    "row_id": row_id,
                    "candidate_id": cand.get("candidate_id"),
                    "family": family,
                    "candidate_type": cand.get("candidate_type"),
                    "bbox": cand.get("bbox"),
                    "features": numeric_features(cand, support[row_id].get(cand.get("candidate_id"), {})),
                    "label_keep": bool(match),
                    "label_policy": "space_center_or_iou30_anchor" if family == "space" else "family_default_iou_or_center",
                    "match_flags": match_flags,
                    "match_score": round(score, 6),
                    "matched_target_id": match.get("target_id") if match else None,
                    "predicted_type": predicted_type,
                    "gold_type": gold_type,
                    "typed_correct": typed_correct,
                    "source_integrity": integrity(),
                }
                counts[family] += 1
                if item["label_keep"]:
                    positives[family] += 1
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    manifest = {
        "task": "IMG-MOE-V18-P1-010",
        "dataset": str(args.output),
        "rows": sum(counts.values()),
        "family_counts": dict(counts),
        "positive_counts": dict(positives),
        "positive_rates": {k: round(positives[k] / max(v, 1), 6) for k, v in counts.items()},
        "source_integrity": integrity(),
        "gold_loaded_after_inference_for_labeling_only": True,
        "gold_used_for_inference": False,
    }
    write_json(Path(args.manifest), manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--relation-features", default=str(DEFAULT_RELATIONS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
