#!/usr/bin/env python3
"""Build dev/locked hard-negative objectness rows for v18 symbol candidates."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from train_symbol_detector_v18 import PARAM_GRID, bbox_iou, center_covered, load_jsonl, predict_row  # noqa: E402

DATA = ROOT / "datasets/image_only_symbol_detector_v18"
OUT = ROOT / "datasets/image_only_symbol_objectness_v18"
REPORT = ROOT / "reports/vlm"


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


def load_detector_policy() -> dict[str, Any]:
    policy_path = ROOT / "checkpoints/symbol_detector_v18/policy.json"
    if policy_path.exists():
        data = json.loads(policy_path.read_text(encoding="utf-8"))
        policy = data.get("policy")
        if isinstance(policy, dict):
            return policy
    return PARAM_GRID[1]


def gold_symbols(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item for item in (row.get("targets") or {}).get("symbols") or []
        if bbox(item.get("bbox")) is not None
    ]


def match_candidate(candidate: dict[str, Any], golds: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, float, bool]:
    cb = bbox(candidate.get("bbox"))
    if cb is None:
        return None, 0.0, False
    best: tuple[float, dict[str, Any] | None, bool] = (0.0, None, False)
    for gold in golds:
        gb = bbox(gold.get("bbox"))
        if gb is None:
            continue
        score = bbox_iou(cb, gb)
        covered = center_covered([int(v) for v in cb], [int(v) for v in gb])
        if covered:
            score = max(score, 0.25)
        if score > best[0]:
            best = (score, gold, covered)
    if best[0] >= 0.25:
        return best[1], best[0], best[2]
    return None, best[0], best[2]


def features(candidate: dict[str, Any]) -> dict[str, float]:
    b = bbox(candidate.get("bbox")) or [0.0, 0.0, 0.0, 0.0]
    w = max(0.0, b[2] - b[0])
    h = max(0.0, b[3] - b[1])
    payload = candidate.get("payload") if isinstance(candidate.get("payload"), dict) else {}
    kind = str(payload.get("candidate_kind") or "unknown")
    anchor_size = payload.get("anchor_size") if isinstance(payload.get("anchor_size"), list) else [0, 0]
    return {
        "detector_confidence": float(candidate.get("confidence") or 0.0),
        "bbox_width": w,
        "bbox_height": h,
        "bbox_area": w * h,
        "bbox_aspect": w / max(h, 1e-6),
        "local_dark_density": float(payload.get("local_dark_density") or 0.0),
        "component_area": float(payload.get("area") or 0.0),
        "component_fill": float(payload.get("fill") or 0.0),
        "anchor_w": float(anchor_size[0] or 0.0),
        "anchor_h": float(anchor_size[1] or 0.0),
        "is_dark_pixel_anchor": 1.0 if kind == "dark_pixel_anchor" else 0.0,
        "is_dark_connected_component": 1.0 if kind == "dark_connected_component" else 0.0,
    }


def build_split(split: str, rows: list[dict[str, Any]], params: dict[str, Any], limit: int | None) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"{split}.jsonl"
    counts = Counter()
    by_type = Counter()
    selected = rows[:limit] if limit else rows
    with out_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            row_id = str(row.get("id"))
            preds = predict_row(row, params)
            golds = gold_symbols(row)
            for rank, cand in enumerate(preds):
                match, score, center_hit = match_candidate(cand, golds)
                label = bool(match)
                symbol_type = str(match.get("symbol_type") or match.get("semantic_type") or "symbol") if match else None
                item = {
                    "id": f"{row_id}_{cand.get('id')}",
                    "split": split,
                    "row_id": row_id,
                    "image": row.get("image"),
                    "candidate_id": cand.get("id"),
                    "bbox": cand.get("bbox"),
                    "rank": rank,
                    "features": features(cand),
                    "label_objectness": label,
                    "match_score": round(float(score), 6),
                    "center_hit": bool(center_hit),
                    "gold_type": symbol_type,
                    "payload": cand.get("payload") or {},
                    "source_integrity": integrity(),
                }
                counts["rows"] += 1
                counts["positive" if label else "negative"] += 1
                if symbol_type:
                    by_type[symbol_type] += 1
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return {
        "split": split,
        "path": str(out_path),
        "rows": counts["rows"],
        "positive": counts["positive"],
        "negative": counts["negative"],
        "positive_rate": round(counts["positive"] / max(counts["rows"], 1), 6),
        "positive_by_type": dict(sorted(by_type.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA))
    parser.add_argument("--limit-dev", type=int, default=None)
    parser.add_argument("--limit-locked", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.limit_dev = args.limit_dev or 5
        args.limit_locked = args.limit_locked or 5
    data = Path(args.data)
    params = load_detector_policy()
    manifest = {
        "task": "IMG-MOE-V18-NEXT-006",
        "dataset": str(OUT),
        "detector_policy": params,
        "splits": {
            "dev": build_split("dev", load_jsonl(data / "dev.jsonl"), params, args.limit_dev),
            "locked": build_split("locked", load_jsonl(data / "locked.jsonl"), params, args.limit_locked),
        },
        "source_integrity": integrity(),
        "gold_loaded_after_inference_for_labeling_only": True,
        "gold_used_for_inference": False,
    }
    write_json(OUT / "manifest.json", manifest)
    write_json(REPORT / "symbol_objectness_dataset_v18_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
