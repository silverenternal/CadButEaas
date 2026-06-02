#!/usr/bin/env python3
"""Build detector-proposal distribution dataset for symbol v24 audits."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from train_symbol_tile_detector_v20 import LABELS, area_bucket, bbox_iou, center_covered, load_jsonl, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def page_gold_from_tiles(tile_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_target: dict[str, dict[str, Any]] = {}
    for row in tile_rows:
        for target in (row.get("targets") or {}).get("boxes") or []:
            target_id = str(target.get("target_id") or "")
            if not target_id or target_id in by_target:
                continue
            box = target.get("page_bbox") or target.get("bbox")
            if not isinstance(box, list) or len(box) != 4:
                continue
            by_target[target_id] = {
                "target_id": target_id,
                "bbox": [float(v) for v in box],
                "label": str(target.get("label") or "generic_symbol"),
                "area_bucket": target.get("area_bucket") or area_bucket([float(v) for v in box]),
            }
    return by_target


def load_golds(tile_path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    pages: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(tile_path):
        pages[str(row.get("row_id"))].append(row)
    return {row_id: page_gold_from_tiles(rows) for row_id, rows in pages.items()}


def load_predictions(path: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in load_jsonl(path):
        out[str(row.get("row_id"))] = list(row.get("predicted_symbols") or [])
    return out


def best_match(pred: dict[str, Any], golds: dict[str, dict[str, Any]]) -> dict[str, Any]:
    box = [float(v) for v in pred.get("bbox") or []]
    if len(box) != 4:
        return {"gold_label": "background", "matched_gold_id": "", "match_iou": 0.0, "center_hit": False, "area_bucket": "unknown"}
    best_id = ""
    best_gold: dict[str, Any] | None = None
    best_iou = 0.0
    best_center = False
    best_score = 0.0
    for gold_id, gold in golds.items():
        gold_box = [float(v) for v in gold["bbox"]]
        overlap = bbox_iou(box, gold_box)
        covered = center_covered(box, gold_box)
        score = max(overlap, 1.0 if covered else 0.0)
        if score > best_score:
            best_score = score
            best_id = gold_id
            best_gold = gold
            best_iou = overlap
            best_center = covered
    if best_gold is None or best_score <= 0.0:
        return {"gold_label": "background", "matched_gold_id": "", "match_iou": 0.0, "center_hit": False, "area_bucket": "unknown"}
    return {
        "gold_label": best_gold["label"],
        "matched_gold_id": best_id,
        "match_iou": round(best_iou, 6),
        "center_hit": bool(best_center),
        "area_bucket": best_gold.get("area_bucket") or area_bucket(best_gold["bbox"]),
    }


def build_rows(preds: dict[str, list[dict[str, Any]]], golds: dict[str, dict[str, dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    totals = Counter()
    matched_gold_ids: dict[str, set[str]] = defaultdict(set)
    for row_id, page_preds in sorted(preds.items()):
        page_golds = golds.get(row_id, {})
        totals["pages"] += 1
        totals["gold"] += len(page_golds)
        totals["predicted"] += len(page_preds)
        for index, pred in enumerate(page_preds):
            match = best_match(pred, page_golds)
            if match["matched_gold_id"]:
                matched_gold_ids[row_id].add(str(match["matched_gold_id"]))
            box = [float(v) for v in pred.get("bbox") or [0, 0, 1, 1]]
            width = max(0.0, box[2] - box[0])
            height = max(0.0, box[3] - box[1])
            label = str(pred.get("label") or "generic_symbol")
            rows.append(
                {
                    "row_id": row_id,
                    "proposal_id": str(pred.get("candidate_id") or pred.get("id") or f"{row_id}_symbol_yolo_{index:06d}"),
                    "bbox": box,
                    "proposal_label": label,
                    "proposal_score": float(pred.get("score") or pred.get("confidence") or 0.0),
                    "tile_id": pred.get("tile_id"),
                    "bbox_width": round(width, 6),
                    "bbox_height": round(height, 6),
                    "bbox_area": round(width * height, 6),
                    "bbox_aspect": round(width / max(height, 1e-6), 6),
                    "proposal_area_bucket": area_bucket(box),
                    "gold_label": match["gold_label"],
                    "matched_gold_id": match["matched_gold_id"],
                    "match_iou": match["match_iou"],
                    "center_hit": match["center_hit"],
                    "gold_area_bucket": match["area_bucket"],
                    "is_false_positive": match["gold_label"] == "background",
                    "type_correct_if_matched": match["gold_label"] == label if match["gold_label"] != "background" else False,
                    "offline_only_columns": ["gold_label", "matched_gold_id", "match_iou", "center_hit", "gold_area_bucket", "is_false_positive", "type_correct_if_matched"],
                }
            )
    class_counts = Counter(row["gold_label"] for row in rows)
    type_correct = sum(1 for row in rows if row["type_correct_if_matched"])
    matched_rows = sum(1 for row in rows if row["matched_gold_id"])
    page_metrics = page_level_metrics(preds, golds)
    audit = {
        "pages": totals["pages"],
        "gold": page_metrics["gold"],
        "predicted": page_metrics["predicted"],
        "candidate_inflation": page_metrics["candidate_inflation"],
        "center_recall": page_metrics["center_recall"],
        "iou_0_30_recall": page_metrics["iou_0_30_recall"],
        "proposal_precision_proxy": page_metrics["iou_0_30_precision"],
        "real_proposal_type_accuracy": page_metrics["typed_accuracy_on_iou_matches"],
        "proposal_row_match_rate": round(matched_rows / max(totals["predicted"], 1), 6),
        "proposal_row_type_accuracy_if_matched": round(type_correct / max(matched_rows, 1), 6),
        "proposal_gold_label_counts": dict(class_counts),
    }
    return rows, audit


def page_level_metrics(preds: dict[str, list[dict[str, Any]]], golds: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    totals = Counter()
    for row_id in sorted(preds):
        gold_map = golds.get(row_id, {})
        merged = preds.get(row_id, [])
        used_iou: set[int] = set()
        used_center: set[int] = set()
        for gold in gold_map.values():
            gold_box = [float(v) for v in gold["bbox"]]
            label = str(gold["label"])
            best_iou = 0.0
            best_iou_index: int | None = None
            center_index: int | None = None
            for pred_index, pred in enumerate(merged):
                pred_box = [float(v) for v in pred.get("bbox") or []]
                if len(pred_box) != 4:
                    continue
                overlap = bbox_iou(pred_box, gold_box)
                if overlap > best_iou:
                    best_iou = overlap
                    best_iou_index = pred_index
                if center_index is None and pred_index not in used_center and center_covered(pred_box, gold_box):
                    center_index = pred_index
            if best_iou_index is not None and best_iou >= 0.30 and best_iou_index not in used_iou:
                used_iou.add(best_iou_index)
                totals["matched_iou"] += 1
                if str(merged[best_iou_index].get("label")) == label:
                    totals["typed_correct_iou"] += 1
            if center_index is not None:
                used_center.add(center_index)
                totals["matched_center"] += 1
        totals["gold"] += len(gold_map)
        totals["predicted"] += len(merged)
    return {
        "gold": int(totals["gold"]),
        "predicted": int(totals["predicted"]),
        "candidate_inflation": round(totals["predicted"] / max(totals["gold"], 1), 6),
        "center_recall": round(totals["matched_center"] / max(totals["gold"], 1), 6),
        "iou_0_30_recall": round(totals["matched_iou"] / max(totals["gold"], 1), 6),
        "iou_0_30_precision": round(totals["matched_iou"] / max(totals["predicted"], 1), 6),
        "typed_accuracy_on_iou_matches": round(totals["typed_correct_iou"] / max(totals["matched_iou"], 1), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="reports/vlm/symbol_yolov8n_pretrained_v22_dedup_hi640_probe_page_predictions.jsonl")
    parser.add_argument("--source-eval", default="reports/vlm/symbol_yolov8n_pretrained_v22_dedup_hi640_probe_page_eval.json")
    parser.add_argument("--locked-tiles", default="datasets/symbol_tile_detector_tiny_sahi_v21/locked.jsonl")
    parser.add_argument("--output-dir", default="datasets/symbol_detector_proposal_type_calibration_v24")
    args = parser.parse_args()

    output_dir = resolve(args.output_dir)
    preds = load_predictions(resolve(args.predictions))
    golds = load_golds(resolve(args.locked_tiles))
    rows, audit = build_rows(preds, golds)
    source_eval_path = resolve(args.source_eval)
    if source_eval_path.exists():
        source_eval = json.loads(source_eval_path.read_text(encoding="utf-8"))
        locked_metrics = source_eval.get("locked") or source_eval.get("metrics") or {}
        if locked_metrics:
            audit.update(
                {
                    "gold": int((locked_metrics.get("symbol_bbox_iou_0_30") or {}).get("gold", audit["gold"])),
                    "predicted": int((locked_metrics.get("symbol_bbox_iou_0_30") or {}).get("predicted", audit["predicted"])),
                    "candidate_inflation": locked_metrics.get("candidate_inflation", audit["candidate_inflation"]),
                    "center_recall": locked_metrics.get("symbol_bbox_center_recall", audit["center_recall"]),
                    "iou_0_30_recall": (locked_metrics.get("symbol_bbox_iou_0_30") or {}).get("recall", audit["iou_0_30_recall"]),
                    "proposal_precision_proxy": (locked_metrics.get("symbol_bbox_iou_0_30") or {}).get("precision", audit["proposal_precision_proxy"]),
                    "real_proposal_type_accuracy": locked_metrics.get("typed_accuracy_on_iou_matches", audit["real_proposal_type_accuracy"]),
                    "page_metric_source": str(source_eval_path.relative_to(ROOT) if source_eval_path.is_relative_to(ROOT) else source_eval_path),
                }
            )
    output_path = output_dir / "locked_proposals.jsonl"
    write_jsonl(output_path, rows)
    manifest = {
        "version": "symbol_detector_proposal_type_calibration_v24",
        "task": "P0-03-symbol-proposal-and-type-adaptation",
        "claim_boundary": "Rows are real detector proposals. Offline gold matching is used only to train/audit proposal-distribution gate/type adapters.",
        "runtime_feature_columns": ["bbox", "proposal_label", "proposal_score", "tile_id", "bbox_width", "bbox_height", "bbox_area", "bbox_aspect", "proposal_area_bucket"],
        "offline_only_columns": ["gold_label", "matched_gold_id", "match_iou", "center_hit", "gold_area_bucket", "is_false_positive", "type_correct_if_matched"],
        "splits": {
            "locked": {
                "proposals": str(output_path.relative_to(ROOT)),
                "source_predictions": args.predictions,
                "source_tiles": args.locked_tiles,
                **audit,
            }
        },
        "gaps": {
            "dev_or_train_detector_predictions_available": False,
            "reason": "Current stored page-level YOLO proposal artifact is locked-only; training a detector-proposal type adapter needs dev/train proposal export or a re-run of YOLO over dev tiles.",
        },
        "success_gate": {
            "stage_1_center_recall_min": 0.94,
            "stage_1_iou_0_30_recall_min": 0.78,
            "stage_1_precision_must_improve_over": 0.096685,
            "must_not_drop_center_recall_below": 0.911595,
            "measured_center_recall": audit["center_recall"],
            "measured_iou_0_30_recall": audit["iou_0_30_recall"],
            "measured_precision_proxy": audit["proposal_precision_proxy"],
            "measured_type_accuracy": audit["real_proposal_type_accuracy"],
            "passed": audit["center_recall"] >= 0.94 and audit["iou_0_30_recall"] >= 0.78 and audit["proposal_precision_proxy"] > 0.096685,
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps({"manifest": str((output_dir / "manifest.json").relative_to(ROOT)), "success_gate": manifest["success_gate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
