#!/usr/bin/env python3
"""Train/evaluate a lightweight image-only raster candidate detector baseline."""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from v8_raster_e2e_utils import (
    CHECKPOINT_DIR,
    DATASET_DIR,
    FAMILIES,
    ROOT,
    classify_component,
    connected_components_from_image,
    f1,
    load_jsonl,
    match_counts,
    normalize_bbox,
    sample_key,
    update_todo_remove,
    write_json,
    write_jsonl,
)


THRESHOLD_GRID = [190, 210, 225]
MIN_PIXELS_GRID = [8, 18]


def main() -> None:
    start = time.time()
    train = load_jsonl(DATASET_DIR / "train.jsonl")
    dev = load_jsonl(DATASET_DIR / "dev.jsonl")
    locked = load_jsonl(DATASET_DIR / "locked.jsonl")
    best = select_threshold(dev)
    locked_predictions, locked_eval, error_cases = evaluate_split(locked, best)
    train_summary = {
        "version": "raster_candidate_detector_v8",
        "run_mode": "full" if len(train) > 1000 and len(locked) >= 50 else "smoke",
        "algorithm": "PIL grayscale dark-pixel connected components with dev-selected thresholds",
        "inference_input": "image_only",
        "train_rows": len(train),
        "dev_rows": len(dev),
        "locked_rows": len(locked),
        "selected_thresholds": best,
        "elapsed_seconds": round(time.time() - start, 3),
        "claim_boundary": "This is a measured image-only baseline proposal generator; it does not use SVG candidate ids at inference.",
    }
    ckpt_dir = CHECKPOINT_DIR / "raster_candidate_detector_v8"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"version": "raster_candidate_detector_v8", "selected_thresholds": best, "families": FAMILIES}, ckpt_dir / "model.pt")
    write_json(ckpt_dir / "train_summary.json", train_summary)

    train_keys = {row.get("sample_id") for row in train}
    locked_keys = {row.get("sample_id") for row in locked}
    macro_f1 = sum(locked_eval["per_family"][family]["f1"] for family in FAMILIES) / len(FAMILIES)
    adopted = bool(macro_f1 >= 0.50 and min(locked_eval["per_family"][family]["recall"] for family in FAMILIES) >= 0.20)
    report = {
        "version": "raster_candidate_detector_v8_eval",
        "checkpoint": str((ckpt_dir / "model.pt").relative_to(ROOT)),
        "train_summary": train_summary,
        "selected_thresholds": best,
        "locked_eval": locked_eval,
        "macro_f1": round(macro_f1, 6),
        "adopted": adopted,
        "adoption_rule": "adopt only if macro_f1 >= 0.50 and every family locked recall >= 0.20; otherwise raster stream is exploratory/rejected",
        "leakage_check": {
            "train_locked_overlap": len(train_keys & locked_keys),
            "train_locked_overlap_examples": sorted(train_keys & locked_keys)[:10],
        },
        "false_positive_audit": locked_eval.get("false_positive_audit", {}),
        "claim_boundary": "Weak locked metrics must be reported as rejected/exploratory, not hidden behind SVG candidate geometry.",
    }
    write_json("reports/vlm/raster_candidate_detector_v8_eval.json", report)
    write_jsonl("reports/vlm/raster_candidate_detector_v8_locked_predictions.jsonl", locked_predictions)
    write_jsonl("reports/vlm/raster_candidate_detector_v8_error_cases.jsonl", error_cases[:1000])
    update_todo_remove(["RASTER-V8-T3"])
    print(json.dumps({"adopted": adopted, "macro_f1": report["macro_f1"], "locked_rows": len(locked_predictions)}, ensure_ascii=False, indent=2))


def select_threshold(rows: list[dict[str, Any]]) -> dict[str, Any]:
    best: dict[str, Any] = {}
    best_score = -1.0
    for threshold in THRESHOLD_GRID:
        for min_pixels in MIN_PIXELS_GRID:
            predictions, report, _cases = evaluate_split(rows[:40], {"dark_threshold": threshold, "min_pixels": min_pixels, "stride": 4})
            score = sum(report["per_family"][family]["f1"] for family in FAMILIES) / len(FAMILIES)
            inflation_penalty = min(float(report["candidate_inflation"].get("overall", 0.0)), 6.0) * 0.01
            score -= inflation_penalty
            if score > best_score:
                best_score = score
                best = {"dark_threshold": threshold, "min_pixels": min_pixels, "stride": 2, "dev_selection_score": round(score, 6)}
    return best


def evaluate_split(rows: list[dict[str, Any]], params: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    totals: dict[str, Counter[str]] = {family: Counter() for family in FAMILIES}
    predictions: list[dict[str, Any]] = []
    error_cases: list[dict[str, Any]] = []
    fp_counter = Counter()
    predicted_total = 0
    gold_total = 0
    for row in rows:
        pred_items = predict_row(row, params)
        gold_items = [item for item in row.get("gold_items") or [] if item.get("family") in FAMILIES]
        pred_by_family = group_by_family(pred_items)
        gold_by_family = group_by_family(gold_items)
        for family in FAMILIES:
            tp, pred_count, gold_count, fp_cases, miss_cases = match_counts(pred_by_family[family], gold_by_family[family], iou_threshold=0.5)
            totals[family].update({"tp": tp, "predicted": pred_count, "gold": gold_count})
            predicted_total += pred_count
            gold_total += gold_count
            fp_counter[family] += len(fp_cases)
            for case in fp_cases[:8]:
                error_cases.append({"sample_id": row.get("sample_id"), "family": family, "error_type": "detector_false_positive", **case})
            for case in miss_cases[:8]:
                error_cases.append({"sample_id": row.get("sample_id"), "family": family, "error_type": "detector_miss", **case})
        predictions.append(
            {
                "sample_id": row.get("sample_id"),
                "image": row.get("image"),
                "source_dataset": row.get("source_dataset"),
                "inference_input": "image_only",
                "proposal_source": "raster_candidate_detector_v8",
                "detector_params": params,
                "proposals": pred_items,
                "gold_family_counts": row.get("gold_family_counts"),
            }
        )
    per_family = {family: f1(totals[family]["tp"], totals[family]["predicted"], totals[family]["gold"]) for family in FAMILIES}
    report = {
        "rows": len(rows),
        "per_family": per_family,
        "candidate_inflation": {
            "overall": round(predicted_total / max(gold_total, 1), 6),
            **{family: round(totals[family]["predicted"] / max(totals[family]["gold"], 1), 6) for family in FAMILIES},
        },
        "false_positive_audit": dict(fp_counter),
    }
    return predictions, report, error_cases


def predict_row(row: dict[str, Any], params: dict[str, Any]) -> list[dict[str, Any]]:
    image = row.get("image")
    if not image:
        return []
    p = ROOT / str(image)
    if not p.exists():
        return []
    with Image.open(p) as img:
        image_size = img.size
    components = connected_components_from_image(
        image,
        dark_threshold=int(params.get("dark_threshold", 205)),
        stride=int(params.get("stride", 2)),
        min_pixels=int(params.get("min_pixels", 24)),
        max_components=260,
    )
    out: list[dict[str, Any]] = []
    for idx, component in enumerate(components):
        bbox = normalize_bbox(component.get("bbox"))
        if not bbox:
            continue
        family = classify_component(component, image_size)
        out.append(
            {
                "id": f"raster_{sample_key(image)}_{idx}",
                "family": family,
                "semantic_type": family,
                "bbox": [round(v, 3) for v in bbox],
                "confidence": confidence_for(component, family),
                "proposal_source": "raster_candidate_detector_v8",
                "features": {key: component[key] for key in ["pixel_count", "width", "height", "aspect"] if key in component},
            }
        )
    return out


def confidence_for(component: dict[str, Any], family: str) -> float:
    pix = float(component.get("pixel_count") or 0.0)
    aspect = float(component.get("aspect") or 1.0)
    base = min(0.95, 0.30 + pix / 5000.0)
    if family == "boundary":
        base += min(0.25, aspect / 40.0)
    return round(max(0.05, min(0.99, base)), 6)


def group_by_family(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out = {family: [] for family in FAMILIES}
    for item in items:
        family = str(item.get("family") or "")
        if family in out:
            out[family].append(item)
    return out


if __name__ == "__main__":
    main()
