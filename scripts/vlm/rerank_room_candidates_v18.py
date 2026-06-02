#!/usr/bin/env python3
"""Rerank v18 room candidates to reduce dense-anchor false positives."""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vlm import train_room_proposal_model_v18 as room_model

REPORT = ROOT / "reports/vlm"
DEFAULT_OUT = REPORT / "room_proposal_model_v18_reranked_candidates.jsonl"
DEFAULT_EVAL = REPORT / "room_proposal_model_v18_rerank_eval.json"


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


def bbox_iou(left: list[float] | None, right: list[float] | None) -> float:
    return room_model.bbox_iou(left, right)


def center_covered(pred: list[float], gold: list[float], margin: float = 2.0) -> bool:
    cx = (gold[0] + gold[2]) / 2.0
    cy = (gold[1] + gold[3]) / 2.0
    return pred[0] - margin <= cx <= pred[2] + margin and pred[1] - margin <= cy <= pred[3] + margin


def area(box: list[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def nms_diverse(items: list[dict[str, Any]], cap: int, threshold: float) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for item in sorted(items, key=lambda x: float(x.get("rerank_score", x.get("confidence", 0.0))), reverse=True):
        if len(selected) >= cap:
            break
        if any(bbox_iou(item.get("bbox"), kept.get("bbox")) >= threshold for kept in selected):
            continue
        selected.append(item)
    if len(selected) < cap:
        seen = {tuple(x["bbox"]) for x in selected}
        for item in sorted(items, key=lambda x: float(x.get("rerank_score", x.get("confidence", 0.0))), reverse=True):
            key = tuple(item["bbox"])
            if key in seen:
                continue
            selected.append(item)
            seen.add(key)
            if len(selected) >= cap:
                break
    return selected


def score_candidate(item: dict[str, Any], source_bonus: float) -> float:
    sf = item.get("shape_features") if isinstance(item.get("shape_features"), dict) else {}
    box = item.get("bbox") or [0, 0, 0, 0]
    w = max(1.0, float(box[2]) - float(box[0]))
    h = max(1.0, float(box[3]) - float(box[1]))
    ar = max(w / h, h / w)
    size = min(1.0, area(box) / (512.0 * 512.0) * 8.0)
    border = float(sf.get("border_density") or 0.0)
    fill = float(sf.get("fill_ratio") or 0.0)
    base = float(item.get("confidence") or 0.0)
    aspect_penalty = max(0.0, ar - 4.0) * 0.04
    tiny_penalty = 0.18 if min(w, h) < 36 else 0.0
    return base + source_bonus + border * 0.45 + fill * 0.35 + size * 0.18 - aspect_penalty - tiny_penalty


def make_candidate_rows(prediction_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return room_model.routed_candidate_rows(prediction_rows)


def load_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, int]]]:
    train_rows = room_model.load_jsonl(room_model.DATA / "train.jsonl")
    dev_rows = room_model.load_jsonl(room_model.DATA / "dev.jsonl")
    locked_rows = room_model.load_jsonl(room_model.DATA / "locked.jsonl")
    priors = room_model.learn_anchor_priors(train_rows + dev_rows)
    return locked_rows, dev_rows, priors


def generate_pool(row: dict[str, Any], priors: list[dict[str, int]]) -> list[dict[str, Any]]:
    dense_params = dict(room_model.PARAM_GRID[0])
    precise_params = dict(room_model.PARAM_GRID[1])
    dense = room_model.detect_room_proposals(row, dense_params, priors)
    precise = room_model.detect_room_proposals(row, precise_params, priors)
    merged: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for source, bonus, items in [("dense", -0.08, dense), ("precise", 0.38, precise)]:
        for item in items:
            key = tuple(int(v) for v in item["bbox"])
            cand = dict(item)
            cand["rerank_source"] = source
            cand["rerank_score"] = round(score_candidate(cand, bonus), 6)
            old = merged.get(key)
            if old is None or cand["rerank_score"] > old["rerank_score"]:
                merged[key] = cand
    return list(merged.values())


def eval_recall(preds_by_row: dict[str, list[dict[str, Any]]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = Counter()
    for row in rows:
        row_id = str(row["id"])
        preds = preds_by_row.get(row_id, [])
        golds = (row.get("targets") or {}).get("rooms") or []
        used: set[int] = set()
        for gold in golds:
            totals["gold"] += 1
            gb = gold.get("bbox")
            best = None
            best_iou = 0.0
            for idx, pred in enumerate(preds):
                if idx in used:
                    continue
                score = bbox_iou(pred.get("bbox"), gb)
                if score > best_iou:
                    best_iou = score
                    best = idx
            if best is not None and best_iou >= 0.50:
                used.add(best)
                totals["matched_iou50"] += 1
            if any(center_covered(pred.get("bbox"), gb) or bbox_iou(pred.get("bbox"), gb) >= 0.30 for pred in preds):
                totals["matched_center_or_iou30"] += 1
        totals["predicted"] += len(preds)
    precision = totals["matched_iou50"] / max(totals["predicted"], 1)
    recall = totals["matched_iou50"] / max(totals["gold"], 1)
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return {
        "predicted": totals["predicted"],
        "gold": totals["gold"],
        "matched_iou50": totals["matched_iou50"],
        "precision_iou50": round(precision, 6),
        "recall_iou50": round(recall, 6),
        "f1_iou50": round(f1, 6),
        "center_or_iou30_recall": round(totals["matched_center_or_iou30"] / max(totals["gold"], 1), 6),
        "candidate_inflation": round(totals["predicted"] / max(totals["gold"], 1), 6),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    warnings.filterwarnings("ignore", category=FutureWarning)
    locked_rows, _dev_rows, priors = load_rows()
    if args.smoke:
        locked_rows = locked_rows[:5]
    pools: dict[str, list[dict[str, Any]]] = {}
    for row in locked_rows:
        pools[str(row["id"])] = generate_pool(row, priors)

    sweeps: dict[str, Any] = {}
    for cap in args.sweep_caps:
        selected = {row_id: nms_diverse(pool, cap, args.nms_iou) for row_id, pool in pools.items()}
        sweeps[str(cap)] = eval_recall(selected, locked_rows)

    cap = args.export_top_k
    selected = {row_id: nms_diverse(pool, cap, args.nms_iou) for row_id, pool in pools.items()}
    prediction_rows: list[dict[str, Any]] = []
    for row in locked_rows:
        row_id = str(row["id"])
        proposals = selected[row_id]
        prediction_rows.append({
            "id": row_id,
            "image": row.get("image"),
            "image_size": row.get("image_size") or [512, 512],
            "source_integrity": integrity(),
            "proposals": proposals,
            "proposal_count_before_export_cap": len(pools[row_id]),
            "gold_counts": row.get("target_counts"),
        })
    write_jsonl(Path(args.output), make_candidate_rows(prediction_rows))
    report = {
        "task": "IMG-MOE-V18-NEXT-001",
        "method": "hybrid precise+dense room rerank with IoU diversity",
        "smoke": bool(args.smoke),
        "export_top_k": cap,
        "nms_iou": args.nms_iou,
        "pool_counts": {
            "rows": len(pools),
            "total": sum(len(v) for v in pools.values()),
            "mean_per_page": round(sum(len(v) for v in pools.values()) / max(len(pools), 1), 3),
        },
        "sweep": sweeps,
        "selected_metric": sweeps[str(cap)],
        "source_integrity": integrity(),
        "gold_loaded_after_inference_for_evaluation_only": True,
        "gold_used_for_inference": False,
        "output": str(args.output),
    }
    write_json(Path(args.eval_output), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--eval-output", default=str(DEFAULT_EVAL))
    parser.add_argument("--export-top-k", type=int, default=100)
    parser.add_argument("--sweep-caps", type=int, nargs="+", default=[20, 40, 60, 80, 100, 150, 200, 300])
    parser.add_argument("--nms-iou", type=float, default=0.72)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    report = build(args)
    print(json.dumps({
        "selected_metric": report["selected_metric"],
        "pool_counts": report["pool_counts"],
        "output": report["output"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
