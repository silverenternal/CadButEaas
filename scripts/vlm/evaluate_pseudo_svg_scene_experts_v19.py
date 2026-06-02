#!/usr/bin/env python3
"""Evaluate pseudo-SVG scene expert rows against locked raster gold."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def abs_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in value[:4]]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def bbox_iou(left: list[float], right: list[float]) -> float:
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    la = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    ra = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return inter / max(la + ra - inter, 1e-9)


def center_covered(pred: list[float], gold: list[float], margin: float = 2.0) -> bool:
    cx = (gold[0] + gold[2]) / 2.0
    cy = (gold[1] + gold[3]) / 2.0
    return pred[0] - margin <= cx <= pred[2] + margin and pred[1] - margin <= cy <= pred[3] + margin


def canonical(family: str, label: str) -> str:
    value = str(label or "").strip()
    if family == "boundary":
        return {"hard_wall": "wall", "partition_wall": "wall", "door": "opening", "opening": "opening", "window": "window"}.get(value, value or "wall")
    if family == "space":
        return value or "room"
    if family == "symbol":
        return value or "generic_symbol"
    if family == "text":
        return value or "room_label"
    return value or family


def load_gold(structured_path: Path, boundary_path: Path, text_path: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    out: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))

    for row in load_jsonl(boundary_path):
        row_id = str(row["id"])
        for item in (row.get("targets") or {}).get("boxes") or []:
            box = bbox(item.get("bbox"))
            if box:
                out[row_id]["boundary"].append({"bbox": box, "label": canonical("boundary", item.get("label")), "id": item.get("target_id")})

    for row in load_jsonl(structured_path):
        row_id = str(row["id"])
        structured = row.get("structured") or {}
        for item in structured.get("rooms") or []:
            box = bbox(item.get("bbox"))
            if box:
                out[row_id]["space"].append({"bbox": box, "label": canonical("space", item.get("semantic_type") or "room"), "id": item.get("id")})
        for item in structured.get("symbols") or []:
            box = bbox(item.get("bbox"))
            if box:
                out[row_id]["symbol"].append({"bbox": box, "label": canonical("symbol", item.get("semantic_type") or "generic_symbol"), "id": item.get("id")})

    for row in load_jsonl(text_path):
        row_id = str(row["id"])
        for item in (row.get("targets") or {}).get("texts") or []:
            box = bbox(item.get("bbox"))
            if box:
                out[row_id]["text"].append({"bbox": box, "label": canonical("text", item.get("semantic_type") or "room_label"), "id": item.get("target_id")})

    return {row_id: dict(families) for row_id, families in out.items()}


def load_predictions(path: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    out: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in load_jsonl(path):
        row_id = str(row["id"])
        for node in ((row.get("scene_graph") or {}).get("nodes") or []):
            family = str(node.get("family") or "")
            if family not in {"boundary", "space", "symbol", "text"}:
                continue
            box = bbox(node.get("bbox"))
            if not box:
                continue
            metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
            out[row_id][family].append(
                {
                    "bbox": box,
                    "label": canonical(family, node.get("semantic_type")),
                    "id": node.get("id"),
                    "confidence": float(node.get("confidence") or 0.0),
                    "source_expert": node.get("source_expert"),
                    "source": node.get("source"),
                    "proposal_source": metadata.get("proposal_source"),
                    "source_integrity": metadata.get("source_integrity"),
                }
            )
    return {row_id: dict(families) for row_id, families in out.items()}


def match_family(preds: list[dict[str, Any]], golds: list[dict[str, Any]]) -> dict[str, Any]:
    used_preds: set[int] = set()
    matches: list[dict[str, Any]] = []
    missed: list[dict[str, Any]] = []
    for gold_index, gold in enumerate(golds):
        best = None
        best_iou = 0.0
        for pred_index, pred in enumerate(preds):
            if pred_index in used_preds:
                continue
            overlap = bbox_iou(pred["bbox"], gold["bbox"])
            if center_covered(pred["bbox"], gold["bbox"]) or overlap >= 0.30:
                if overlap >= best_iou:
                    best = pred_index
                    best_iou = overlap
        if best is None:
            missed.append(gold)
            continue
        used_preds.add(best)
        pred = preds[best]
        matches.append({"gold": gold, "pred": pred, "iou": best_iou, "label_correct": gold["label"] == pred["label"]})

    false_positive = [pred for idx, pred in enumerate(preds) if idx not in used_preds]
    return {"matches": matches, "missed": missed, "false_positive": false_positive}


def per_label_report(matches: list[dict[str, Any]], missed: list[dict[str, Any]], false_positive: list[dict[str, Any]]) -> dict[str, Any]:
    labels = sorted({m["gold"]["label"] for m in matches} | {m["pred"]["label"] for m in matches} | {g["label"] for g in missed} | {p["label"] for p in false_positive})
    stats = {label: Counter() for label in labels}
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for match in matches:
        gold_label = match["gold"]["label"]
        pred_label = match["pred"]["label"]
        confusion[gold_label][pred_label] += 1
        if gold_label == pred_label:
            stats[gold_label]["tp"] += 1
        else:
            stats[gold_label]["fn"] += 1
            stats[pred_label]["fp"] += 1
    for item in missed:
        stats[item["label"]]["fn"] += 1
    for item in false_positive:
        stats[item["label"]]["fp"] += 1

    per_label = {}
    f1s = []
    for label in labels:
        tp = stats[label]["tp"]
        fp = stats[label]["fp"]
        fn = stats[label]["fn"]
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_label[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        }
        f1s.append(f1)
    return {
        "macro_f1": round(sum(f1s) / max(len(f1s), 1), 6),
        "per_label": per_label,
        "confusion": {key: dict(value) for key, value in sorted(confusion.items())},
    }


def audit_integrity(predictions: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    violations = Counter()
    checked = 0
    for families in predictions.values():
        for preds in families.values():
            for pred in preds:
                checked += 1
                integrity = pred.get("source_integrity") if isinstance(pred.get("source_integrity"), dict) else {}
                if integrity.get("annotation_geometry_used_at_inference"):
                    violations["annotation_geometry_used_at_inference"] += 1
                if integrity.get("runtime_uses_svg_or_cad_geometry"):
                    violations["runtime_uses_svg_or_cad_geometry"] += 1
                if integrity.get("svg_candidate_ids_used"):
                    violations["svg_candidate_ids_used"] += 1
    return {"checked_predictions": checked, "violations": dict(violations), "passes": not violations}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="reports/vlm/pseudo_svg_scene_expert_rows_v19.jsonl")
    parser.add_argument("--structured-gold", default="datasets/image_only_structured_targets_v16/locked.jsonl")
    parser.add_argument("--boundary-gold", default="datasets/image_only_boundary_detector_v18/locked.jsonl")
    parser.add_argument("--text-gold", default="datasets/image_only_text_ocr_v18/locked.jsonl")
    parser.add_argument("--audit", default="reports/vlm/pseudo_svg_scene_expert_eval_v19.json")
    args = parser.parse_args()

    gold = load_gold(abs_path(args.structured_gold), abs_path(args.boundary_gold), abs_path(args.text_gold))
    predictions = load_predictions(abs_path(args.predictions))
    families = ["boundary", "space", "symbol", "text"]
    by_family = {}
    aggregate_counts = Counter()

    for family in families:
        all_matches: list[dict[str, Any]] = []
        all_missed: list[dict[str, Any]] = []
        all_false_positive: list[dict[str, Any]] = []
        row_buckets: list[dict[str, Any]] = []
        for row_id in sorted(set(gold) | set(predictions)):
            result = match_family(predictions.get(row_id, {}).get(family, []), gold.get(row_id, {}).get(family, []))
            all_matches.extend(result["matches"])
            all_missed.extend(result["missed"])
            all_false_positive.extend(result["false_positive"])
            if result["missed"] or result["false_positive"]:
                row_buckets.append(
                    {
                        "row_id": row_id,
                        "gold": len(gold.get(row_id, {}).get(family, [])),
                        "predicted": len(predictions.get(row_id, {}).get(family, [])),
                        "matched": len(result["matches"]),
                        "missed": len(result["missed"]),
                        "false_positive": len(result["false_positive"]),
                    }
                )
        gold_count = len(all_matches) + len(all_missed)
        pred_count = len(all_matches) + len(all_false_positive)
        label_correct = sum(1 for item in all_matches if item["label_correct"])
        label_report = per_label_report(all_matches, all_missed, all_false_positive)
        by_family[family] = {
            "gold": gold_count,
            "predicted": pred_count,
            "matched": len(all_matches),
            "false_positive": len(all_false_positive),
            "missed": len(all_missed),
            "localization_precision": round(len(all_matches) / max(pred_count, 1), 6),
            "localization_recall": round(len(all_matches) / max(gold_count, 1), 6),
            "localization_f1": round(2 * len(all_matches) / max(pred_count + gold_count, 1), 6),
            "matched_label_accuracy": round(label_correct / max(len(all_matches), 1), 6),
            "label_macro_f1_detection_aware": label_report["macro_f1"],
            "per_label": label_report["per_label"],
            "confusion": label_report["confusion"],
            "worst_rows": sorted(row_buckets, key=lambda item: (item["missed"], item["false_positive"]), reverse=True)[:12],
            "missed_by_label": dict(Counter(item["label"] for item in all_missed).most_common()),
            "false_positive_by_label": dict(Counter(item["label"] for item in all_false_positive).most_common()),
            "false_positive_by_proposal_source": dict(Counter(str(item.get("proposal_source") or "unknown") for item in all_false_positive).most_common()),
        }
        aggregate_counts[f"{family}_gold"] = gold_count
        aggregate_counts[f"{family}_predicted"] = pred_count

    audit = {
        "version": "pseudo_svg_scene_expert_eval_v19",
        "task": "P0-PSEUDO-SVG-001",
        "purpose": "Evaluate whether connected SVG/scene-era experts actually improve raster-derived pseudo-SVG recognition under locked gold.",
        "inputs": {
            "predictions": args.predictions,
            "structured_gold": args.structured_gold,
            "boundary_gold": args.boundary_gold,
            "text_gold": args.text_gold,
        },
        "source_integrity": audit_integrity(predictions),
        "aggregate_counts": dict(aggregate_counts),
        "by_family": by_family,
        "decision": {
            "integration_success": True,
            "production_adopted": False,
            "reason": "Expert wrappers run, but locked localization and label metrics remain far below the 0.98 production target.",
        },
    }
    write_json(abs_path(args.audit), audit)
    print(json.dumps({"source_integrity": audit["source_integrity"], "by_family": {k: {m: v[m] for m in ["gold", "predicted", "matched", "localization_precision", "localization_recall", "matched_label_accuracy", "label_macro_f1_detection_aware"]} for k, v in by_family.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
