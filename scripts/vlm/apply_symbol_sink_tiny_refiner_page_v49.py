#!/usr/bin/env python3
"""Apply the sink/tiny box refiner to P2-transfer selected candidates and score page metrics."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from train_symbol_box_refiner_v38 import apply_delta
from train_symbol_support_suppression_v35 import load_jsonl, source_path, vector
from train_symbol_tile_detector_v20 import area_bucket, bbox_iou, center_covered, rel, write_json, write_jsonl


ROUTE_LABELS = {"sink", "equipment"}
ROUTE_AREAS = {"tiny_le_64", "small_le_256"}


def valid_box(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    box = [float(v) for v in value]
    return box if box[2] > box[0] and box[3] > box[1] else None


def load_gold(smoke_rows: Path) -> dict[str, dict[str, dict[str, Any]]]:
    by_page: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in load_jsonl(source_path(smoke_rows)):
        page_id = str(row.get("row_id") or "")
        for gold in ((row.get("targets") or {}).get("boxes") or []):
            target_id = str(gold.get("target_id") or "")
            box = valid_box(gold.get("page_bbox") or gold.get("bbox"))
            if page_id and target_id and box:
                by_page[page_id][target_id] = {
                    "target_id": target_id,
                    "bbox": box,
                    "label": str(gold.get("label") or "generic_symbol"),
                    "area_bucket": str(gold.get("area_bucket") or area_bucket(box)),
                }
    return by_page


def score_candidates(rows: list[dict[str, Any]], model_bundle: dict[str, Any]) -> list[dict[str, Any]]:
    names = list(model_bundle["feature_names"])
    model = model_bundle["model"]
    if not rows:
        return []
    x = np.asarray([vector(row, names) for row in rows], dtype=np.float32)
    probs = model.predict_proba(x)[:, 1]
    out: list[dict[str, Any]] = []
    for row, prob in zip(rows, probs, strict=True):
        item = dict(row)
        item["policy_score"] = float(prob)
        out.append(item)
    return out


def select_page(rows: list[dict[str, Any]], threshold: float, cluster_topk: int, max_per_page: int) -> list[dict[str, Any]]:
    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cluster[str(row.get("cluster_key") or row.get("cluster_id") or "")].append(row)
    selected_ids: set[str] = set()
    selected: list[dict[str, Any]] = []
    for items in by_cluster.values():
        items.sort(key=lambda item: (float(item.get("policy_score") or 0.0), float(item.get("score") or 0.0)), reverse=True)
        for rank, item in enumerate(items):
            if float(item.get("policy_score") or 0.0) >= threshold or rank < cluster_topk:
                cid = str(item.get("candidate_id") or "")
                if cid and cid not in selected_ids:
                    selected_ids.add(cid)
                    selected.append(item)
    selected.sort(key=lambda item: (float(item.get("policy_score") or 0.0), float(item.get("score") or 0.0)), reverse=True)
    return selected[:max_per_page]


def should_refine(row: dict[str, Any]) -> bool:
    box = valid_box(row.get("bbox"))
    if not box:
        return False
    return str(row.get("label") or "") in ROUTE_LABELS or area_bucket(box) in ROUTE_AREAS


def refine_selected(rows: list[dict[str, Any]], refiner_bundle: dict[str, Any], clip: float) -> tuple[list[dict[str, Any]], Counter]:
    names = list(refiner_bundle["feature_names"])
    model = refiner_bundle["model"]
    routed = [row for row in rows if should_refine(row)]
    audit = Counter()
    if not routed:
        return rows, audit
    x = np.asarray([vector(row, names) for row in routed], dtype=np.float32)
    deltas = model.predict(x)
    delta_by_id = {str(row.get("candidate_id")): list(delta) for row, delta in zip(routed, deltas, strict=True)}
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        cid = str(item.get("candidate_id") or "")
        box = valid_box(item.get("bbox"))
        if box and cid in delta_by_id:
            item["original_bbox"] = box
            item["bbox"] = apply_delta(box, delta_by_id[cid], clip)
            item["refined_by"] = "symbol_box_refiner_v38_sink_tiny_v49"
            audit["refined"] += 1
            audit[f"refined_label:{item.get('label')}"] += 1
            audit[f"refined_area:{area_bucket(box)}"] += 1
        out.append(item)
    return out, audit


def evaluate(selected_by_page: dict[str, list[dict[str, Any]]], gold_by_page: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    totals = Counter()
    misses_by_label = Counter()
    misses_by_area = Counter()
    for page_id, gold_map in gold_by_page.items():
        selected = selected_by_page.get(page_id, [])
        used_iou: set[int] = set()
        center_hits: set[str] = set()
        iou_hits: set[str] = set()
        typed_total = 0
        typed_correct = 0
        for target_id, gold in gold_map.items():
            gold_box = [float(v) for v in gold["bbox"]]
            best_iou = 0.0
            best_idx = None
            center_idx = None
            for idx, pred in enumerate(selected):
                box = valid_box(pred.get("bbox"))
                if not box:
                    continue
                iou = bbox_iou(box, gold_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
                if center_idx is None and center_covered(box, gold_box):
                    center_idx = idx
            if center_idx is not None:
                center_hits.add(target_id)
            if best_idx is not None and best_iou >= 0.30 and best_idx not in used_iou:
                used_iou.add(best_idx)
                iou_hits.add(target_id)
                typed_total += 1
                if str(selected[best_idx].get("label") or "") == str(gold.get("label") or ""):
                    typed_correct += 1
            else:
                misses_by_label[str(gold.get("label") or "unknown")] += 1
                misses_by_area[str(gold.get("area_bucket") or "unknown")] += 1
        totals["gold"] += len(gold_map)
        totals["selected"] += len(selected)
        totals["iou_hit"] += len(iou_hits)
        totals["center_hit"] += len(center_hits)
        totals["typed_total"] += typed_total
        totals["typed_correct"] += typed_correct
    precision = totals["iou_hit"] / max(totals["selected"], 1)
    recall = totals["iou_hit"] / max(totals["gold"], 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "pages": len(gold_by_page),
        "symbol_bbox_iou_0_30": {
            "matched": int(totals["iou_hit"]),
            "predicted": int(totals["selected"]),
            "gold": int(totals["gold"]),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        },
        "symbol_bbox_center_recall": round(totals["center_hit"] / max(totals["gold"], 1), 6),
        "candidate_inflation": round(totals["selected"] / max(totals["gold"], 1), 6),
        "typed_accuracy_on_iou_matches": round(totals["typed_correct"] / max(totals["typed_total"], 1), 6),
        "misses_by_label": dict(misses_by_label.most_common()),
        "misses_by_area": dict(misses_by_area.most_common()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="datasets/symbol_detector_listwise_recovery_v47_p2_transfer_smoke120/manifest.json")
    parser.add_argument("--smoke-rows", default="datasets/symbol_tile_detector_tiny_sahi_v21/smoke_v30.jsonl")
    parser.add_argument("--suppression-model", default="checkpoints/symbol_support_suppression_v35_p2_transfer_smoke120_t065_c1/model.joblib")
    parser.add_argument("--refiner-model", default="checkpoints/symbol_box_refiner_v38_sink_tiny_v49/model.joblib")
    parser.add_argument("--split", default="smoke_eval", choices=["smoke_eval", "dev", "train", "all"])
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--cluster-topk", type=int, default=1)
    parser.add_argument("--max-per-page", type=int, default=120)
    parser.add_argument("--clip", type=float, default=0.75)
    parser.add_argument("--output", default="reports/vlm/symbol_sink_tiny_refiner_page_v49_eval.json")
    parser.add_argument("--predictions-output", default="reports/vlm/symbol_sink_tiny_refiner_page_v49_predictions.jsonl")
    args = parser.parse_args()

    manifest = json.loads(source_path(args.data).read_text(encoding="utf-8"))
    rows_all = load_jsonl(source_path(manifest["outputs"]["rows"]))
    rows = [row for row in rows_all if args.split == "all" or str(row.get("split") or "") == args.split]
    scored = score_candidates(rows, joblib.load(source_path(args.suppression_model)))
    pages: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        pages[str(row["page_id"])].append(row)
    selected_raw = {
        page_id: select_page(page_rows, args.threshold, args.cluster_topk, args.max_per_page)
        for page_id, page_rows in pages.items()
    }
    refiner = joblib.load(source_path(args.refiner_model))
    selected_refined: dict[str, list[dict[str, Any]]] = {}
    route_audit = Counter()
    for page_id, selected in selected_raw.items():
        refined, audit = refine_selected(selected, refiner, args.clip)
        selected_refined[page_id] = refined
        route_audit.update(audit)
    gold_by_page_all = load_gold(source_path(args.smoke_rows))
    gold_by_page = {page_id: gold_by_page_all[page_id] for page_id in selected_refined.keys() if page_id in gold_by_page_all}
    baseline = evaluate(selected_raw, gold_by_page)
    refined = evaluate(selected_refined, gold_by_page)
    predictions = [
        {
            "page_id": page_id,
            "predicted_symbols": [
                {
                    "candidate_id": row.get("candidate_id"),
                    "bbox": row.get("bbox"),
                    "label": row.get("label"),
                    "confidence": round(float(row.get("policy_score") or 0.0), 6),
                    "refined_by": row.get("refined_by"),
                }
                for row in rows
            ],
            "source_integrity": {"gold_used_for_inference": False, "runtime_uses_svg_or_cad_geometry": False},
        }
        for page_id, rows in selected_refined.items()
    ]
    report = {
        "version": "symbol_sink_tiny_refiner_page_v49",
        "data": rel(source_path(args.data)),
        "suppression_model": rel(source_path(args.suppression_model)),
        "refiner_model": rel(source_path(args.refiner_model)),
        "split": args.split,
        "route": {"labels": sorted(ROUTE_LABELS), "areas": sorted(ROUTE_AREAS), "clip": args.clip},
        "selection_policy": {"threshold": args.threshold, "cluster_topk": args.cluster_topk, "max_per_page": args.max_per_page},
        "route_audit": dict(route_audit),
        "baseline_without_refiner": baseline,
        "refined": refined,
        "gate": {
            "precision_not_drop": refined["symbol_bbox_iou_0_30"]["precision"] >= baseline["symbol_bbox_iou_0_30"]["precision"],
            "recall_not_drop": refined["symbol_bbox_iou_0_30"]["recall"] >= baseline["symbol_bbox_iou_0_30"]["recall"],
            "sink_misses_reduce": refined["misses_by_label"].get("sink", 0) <= baseline["misses_by_label"].get("sink", 0),
            "tiny_misses_reduce": refined["misses_by_area"].get("tiny_le_64", 0) <= baseline["misses_by_area"].get("tiny_le_64", 0),
            "no_oracle_inference": True,
        },
    }
    report["gate"]["passed"] = all(report["gate"].values())
    write_json(source_path(args.output), report)
    write_jsonl(source_path(args.predictions_output), predictions)
    print(json.dumps({"baseline": baseline, "refined": refined, "route_audit": dict(route_audit), "gate": report["gate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
