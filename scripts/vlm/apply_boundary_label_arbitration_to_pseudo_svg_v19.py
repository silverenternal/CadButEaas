#!/usr/bin/env python3
"""Apply/audit boundary_label_arbitration_v1 on pseudo-SVG boundary candidates."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/vlm"
MODEL = ROOT / "checkpoints/boundary_label_arbitration_v1/model.joblib"
LABELS = ["hard_wall", "door", "window"]
ORIENTATIONS = ["horizontal", "vertical", "diagonal", "rectangular", "unknown"]
FEATURE_CONTRACT = [
    "x1",
    "y1",
    "x2",
    "y2",
    "width",
    "height",
    "area",
    "log_width_height_ratio",
    "length",
    "length_page_norm",
    "width_page_norm",
    "height_page_norm",
    "area_page_norm",
    "cx_page_norm",
    "cy_page_norm",
    "group_count",
    "group_count_ge_2",
    *[f"orientation_{item}" for item in ORIENTATIONS],
]


def abs_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def source_integrity() -> dict[str, Any]:
    return {
        "source_mode": "image_only_raster_pseudo_svg_normalized",
        "svg_candidate_ids_used": False,
        "annotation_geometry_used_at_inference": False,
        "model_input": "raster_image_only",
        "runtime_uses_svg_or_cad_geometry": False,
        "strong_expert": "boundary_label_arbitration_v1",
    }


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


def gold_by_row(path: Path) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for row in load_jsonl(path):
        out[str(row["id"])] = [
            item
            for item in (row.get("targets") or {}).get("boxes") or []
            if item.get("bbox") and len(item["bbox"]) == 4
        ]
    return out


def page_bbox(row: dict[str, Any]) -> list[float]:
    size = row.get("image_size") or [512, 512]
    return [0.0, 0.0, float(size[0]), float(size[1])]


def feature_row(candidate: dict[str, Any], page: list[float]) -> list[float]:
    bbox = [float(v) for v in candidate["bbox"]]
    x1, y1, x2, y2 = bbox
    width = max(x2 - x1, 1e-6)
    height = max(y2 - y1, 1e-6)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    page_w = max(page[2] - page[0], 1e-6)
    page_h = max(page[3] - page[1], 1e-6)
    payload = candidate.get("payload") or {}
    features = payload.get("features") or {}
    length = float(features.get("length") or max(width, height))
    group_count = float(features.get("member_count") or len(payload.get("member_path_ids") or []) or 1)
    orientation = str(features.get("orientation") or "unknown")
    if orientation not in ORIENTATIONS:
        orientation = "unknown"
    orient = [1.0 if orientation == item else 0.0 for item in ORIENTATIONS]
    return [
        x1,
        y1,
        x2,
        y2,
        width,
        height,
        width * height,
        math.log((width + 1.0) / (height + 1.0)),
        length,
        length / max(max(page_w, page_h), 1e-6),
        width / page_w,
        height / page_h,
        (width * height) / max(page_w * page_h, 1e-6),
        (cx - page[0]) / page_w,
        (cy - page[1]) / page_h,
        group_count,
        float(group_count >= 2.0),
        *orient,
    ]


def map_model_label(label: str) -> str:
    if label == "hard_wall":
        return "wall"
    if label == "door":
        return "opening"
    return label


def best_gold(candidate: dict[str, Any], golds: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, float]:
    bbox = [float(v) for v in candidate["bbox"]]
    best = None
    best_iou = 0.0
    for gold in golds:
        gb = [float(v) for v in gold["bbox"]]
        iou = bbox_iou(bbox, gb)
        if center_covered(bbox, gb) or iou >= 0.30:
            if iou >= best_iou:
                best = gold
                best_iou = iou
    return best, best_iou


def evaluate_labeled(rows: list[dict[str, Any]], gold: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    totals = Counter()
    per_label = {key: Counter() for key in ["wall", "opening", "window"]}
    for row in rows:
        for candidate in row["scene_graph"]["candidate_stream"]:
            if candidate.get("family") != "boundary":
                continue
            totals["predicted_boundary"] += 1
            matched, iou = best_gold(candidate, gold.get(str(row["id"]), []))
            if matched is None:
                totals["unmatched"] += 1
                continue
            gold_label = str(matched.get("label"))
            pred_label = str(candidate.get("candidate_type") or "")
            totals["matched"] += 1
            totals[f"gold_{gold_label}"] += 1
            totals[f"pred_{pred_label}"] += 1
            if pred_label == gold_label:
                totals["label_correct"] += 1
                per_label.setdefault(gold_label, Counter())["tp"] += 1
            else:
                per_label.setdefault(gold_label, Counter())["fn"] += 1
                per_label.setdefault(pred_label, Counter())["fp"] += 1
    label_metrics = {}
    for label in ["wall", "opening", "window"]:
        tp = per_label[label]["tp"]
        fp = per_label[label]["fp"]
        fn = per_label[label]["fn"]
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        label_metrics[label] = {"precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6)}
    return {
        "predicted_boundary": int(totals["predicted_boundary"]),
        "matched_boundary": int(totals["matched"]),
        "unmatched_boundary": int(totals["unmatched"]),
        "matched_label_accuracy": round(totals["label_correct"] / max(totals["matched"], 1), 6),
        "per_label": label_metrics,
        "label_counts": {key: int(value) for key, value in totals.items() if key.startswith("pred_") or key.startswith("gold_")},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="datasets/pseudo_svg_vectorizer_v19_locked/pseudo_svg_normalized_candidates_recall.jsonl")
    parser.add_argument("--output", default="reports/vlm/pseudo_svg_boundary_label_arbitrated_v19_candidates.jsonl")
    parser.add_argument("--audit", default="reports/vlm/pseudo_svg_boundary_label_arbitration_v19_audit.json")
    parser.add_argument("--model", default=str(MODEL))
    parser.add_argument("--gold", default="datasets/image_only_boundary_detector_v18/locked.jsonl")
    args = parser.parse_args()

    bundle = joblib.load(abs_path(args.model))
    model = bundle["model"]
    rows = load_jsonl(abs_path(args.input))
    out_rows = []
    feature_examples = []
    totals = Counter()
    for row in rows:
        page = page_bbox(row)
        boundary_candidates = [candidate for candidate in row["scene_graph"]["candidate_stream"] if candidate.get("family") == "boundary"]
        if boundary_candidates:
            x = np.asarray([feature_row(candidate, page) for candidate in boundary_candidates], dtype=float)
            probs = model.predict_proba(x)
            classes = [str(item) for item in model.classes_]
            for candidate, prob in zip(boundary_candidates, probs):
                best_index = int(np.argmax(prob))
                model_label = classes[best_index]
                mapped = map_model_label(model_label)
                payload = dict(candidate.get("payload") or {})
                payload["boundary_label_arbitration_v1"] = {
                    "model_label": model_label,
                    "mapped_label": mapped,
                    "confidence": round(float(prob[best_index]), 6),
                    "probs": {classes[idx]: round(float(prob[idx]), 6) for idx in range(len(classes))},
                    "feature_contract": FEATURE_CONTRACT,
                }
                candidate["payload"] = payload
                candidate["candidate_type"] = mapped
                candidate["confidence"] = round(max(float(candidate.get("confidence") or 0.0), float(prob[best_index])), 6)
                candidate["source_integrity"] = source_integrity()
                totals[f"pred_{mapped}"] += 1
                if len(feature_examples) < 5:
                    feature_examples.append({name: round(value, 6) for name, value in zip(FEATURE_CONTRACT, feature_row(candidate, page))})
        out = dict(row)
        out["source_integrity"] = source_integrity()
        out["scene_graph"] = dict(row["scene_graph"])
        out["scene_graph"]["candidate_stream"] = row["scene_graph"]["candidate_stream"]
        out_rows.append(out)
    write_jsonl(abs_path(args.output), out_rows)
    eval_report = evaluate_labeled(out_rows, gold_by_row(abs_path(args.gold)))
    baseline_rows = json.loads(json.dumps(rows))
    for row in baseline_rows:
        for candidate in row["scene_graph"]["candidate_stream"]:
            if candidate.get("family") == "boundary":
                candidate["candidate_type"] = "wall"
    baseline_report = evaluate_labeled(baseline_rows, gold_by_row(abs_path(args.gold)))
    report = {
        "version": "pseudo_svg_boundary_label_arbitration_v19",
        "task": "P0-PSEUDO-SVG-001",
        "input": args.input,
        "output": args.output,
        "model": str(abs_path(args.model).relative_to(ROOT)),
        "source_integrity": source_integrity(),
        "feature_contract": {
            "compatible": True,
            "feature_names": FEATURE_CONTRACT,
            "model_labels": [str(item) for item in model.classes_],
            "label_mapping": {"hard_wall": "wall", "door": "opening", "window": "window"},
            "note": "Features are reconstructed from pseudo-SVG normalized line geometry; no offline SVG/CAD geometry is used at inference.",
            "examples": feature_examples,
        },
        "prediction_counts": {key: int(value) for key, value in totals.items()},
        "baseline_all_wall": baseline_report,
        "evaluation_against_locked_gold": eval_report,
        "adoption_gate": {
            "passes": eval_report["matched_label_accuracy"] >= baseline_report["matched_label_accuracy"],
            "baseline_matched_label_accuracy": baseline_report["matched_label_accuracy"],
            "arbitrated_matched_label_accuracy": eval_report["matched_label_accuracy"],
            "decision": "do_not_adopt_directly" if eval_report["matched_label_accuracy"] < baseline_report["matched_label_accuracy"] else "candidate_for_further_eval",
        },
        "adoption_note": "This audits label arbitration compatibility only. Direct adoption is blocked if it underperforms the all-wall baseline on pseudo-SVG matched candidates.",
    }
    write_json(abs_path(args.audit), report)
    print(json.dumps({"prediction_counts": report["prediction_counts"], "evaluation": eval_report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
