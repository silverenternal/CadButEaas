#!/usr/bin/env python3
"""Build crop/context data for a visual boundary door/window specialist.

The runtime model must remain raster-only. Gold labels are used here only for
offline dev training and locked evaluation manifests.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_boundary_type_fusion_v24 import bbox, center_covered, gold_by_row, iou  # noqa: E402


LABELS = ["hard_wall", "door", "window", "background"]


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= limit:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def candidate_gold_match(candidate: dict[str, Any], gold_items: list[dict[str, Any]]) -> tuple[str, str | None, float]:
    cb = bbox(candidate.get("bbox"))
    if cb is None:
        return "background", None, 0.0
    best_label = "background"
    best_target_id = None
    best_score = 0.0
    for item in gold_items:
        score = max(iou(cb, item["bbox"]), 1.0 if center_covered(cb, item["bbox"]) else 0.0)
        if score > best_score:
            best_label = str(item["label"])
            best_target_id = str(item.get("target_id"))
            best_score = float(score)
    if best_score <= 0.0:
        return "background", None, 0.0
    return best_label, best_target_id, best_score


def expanded_crop_box(box: list[float], image_size: tuple[int, int], context_scale: float, min_context: int) -> tuple[int, int, int, int]:
    width, height = image_size
    x1, y1, x2, y2 = box
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    pad = max(bw, bh) * context_scale
    pad = max(pad, float(min_context))
    return (
        max(0, int(round(x1 - pad))),
        max(0, int(round(y1 - pad))),
        min(width, int(round(x2 + pad))),
        min(height, int(round(y2 + pad))),
    )


def keep_candidate(label: str, candidate: dict[str, Any], mode: str) -> bool:
    if label in {"door", "window"}:
        return True
    if mode == "eval":
        return True
    hint = str(candidate.get("label_hint") or "")
    pred = str(candidate.get("prediction") or candidate.get("gnn_prediction") or "")
    fusion = str(candidate.get("fusion_prediction") or "")
    if label == "hard_wall" and ({hint, pred, fusion} & {"door", "window"}):
        return True
    if label == "background" and hint in {"door", "window"}:
        return True
    return False


def crop_name(row_id: str, candidate_id: Any) -> str:
    clean_candidate = str(candidate_id).replace("/", "_").replace(" ", "_")
    return f"{row_id}__{clean_candidate}.png"


def build_split(
    *,
    predictions_path: Path,
    gold_path: Path,
    output_dir: Path,
    split: str,
    limit: int | None,
    cap: int,
    context_scale: float,
    min_context: int,
    image_size: int,
    mode: str,
    max_per_label: int,
    seed: int,
) -> dict[str, Any]:
    rows = load_jsonl(predictions_path, limit)
    gold = gold_by_row(gold_path, limit)
    rng = random.Random(seed)
    selected: dict[str, list[tuple[dict[str, Any], dict[str, Any], list[float], str, str | None, float]]] = {label: [] for label in LABELS}
    seen = 0
    for row in rows:
        row_id = str(row.get("id"))
        for candidate in (row.get("candidate_stream") or [])[:cap]:
            cb = bbox(candidate.get("bbox"))
            if cb is None:
                continue
            label, target_id, match_score = candidate_gold_match(candidate, gold.get(row_id, []))
            if not keep_candidate(label, candidate, mode):
                continue
            selected[label].append((row, candidate, cb, label, target_id, match_score))
            seen += 1
    eligible_counts = {label: len(items) for label, items in selected.items()}
    if mode == "train" and max_per_label > 0:
        for label, items in selected.items():
            if len(items) > max_per_label:
                selected[label] = rng.sample(items, max_per_label)
    selected_items = [item for label in LABELS for item in selected[label]]
    selected_items.sort(key=lambda item: (str(item[0].get("id")), str(item[1].get("candidate_id"))))
    crops_dir = output_dir / split / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / split / "manifest.csv"
    counts = {label: 0 for label in LABELS}
    skipped = 0
    records = 0
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "row_id",
                "candidate_id",
                "crop_path",
                "label",
                "target_id",
                "match_score",
                "bbox",
                "crop_box",
                "prediction",
                "gnn_prediction",
                "fusion_prediction",
                "label_hint",
                "proposal_source",
                "proposal_confidence",
                "source_image",
            ],
        )
        writer.writeheader()
        image_cache: dict[str, Image.Image] = {}
        try:
            for row, candidate, cb, label, target_id, match_score in selected_items:
                row_id = str(row.get("id"))
                image_rel = str(row.get("image") or "")
                if not image_rel:
                    skipped += 1
                    continue
                image_path = ROOT / image_rel
                if image_rel not in image_cache:
                    image_cache[image_rel] = Image.open(image_path).convert("RGB")
                image = image_cache[image_rel]
                crop_box = expanded_crop_box(cb, image.size, context_scale, min_context)
                if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
                    skipped += 1
                    continue
                resized = image.crop(crop_box).resize((image_size, image_size), Image.Resampling.BILINEAR)
                name = crop_name(row_id, candidate.get("candidate_id"))
                crop_rel = f"{split}/crops/{name}"
                resized.save(output_dir / crop_rel)
                counts[label] += 1
                records += 1
                writer.writerow(
                    {
                        "row_id": row_id,
                        "candidate_id": candidate.get("candidate_id"),
                        "crop_path": crop_rel,
                        "label": label,
                        "target_id": target_id or "",
                        "match_score": round(match_score, 6),
                        "bbox": json.dumps(cb),
                        "crop_box": json.dumps(crop_box),
                        "prediction": candidate.get("prediction"),
                        "gnn_prediction": candidate.get("gnn_prediction"),
                        "fusion_prediction": candidate.get("fusion_prediction"),
                        "label_hint": candidate.get("label_hint"),
                        "proposal_source": candidate.get("proposal_source"),
                        "proposal_confidence": candidate.get("proposal_confidence"),
                        "source_image": image_rel,
                    }
                )
        finally:
            for image in image_cache.values():
                image.close()
    return {
        "split": split,
        "mode": mode,
        "predictions": str(predictions_path),
        "gold": str(gold_path),
        "manifest": str(manifest_path),
        "records": records,
        "counts": counts,
        "eligible_records_before_sampling": seen,
        "eligible_counts_before_sampling": eligible_counts,
        "skipped": skipped,
        "cap": cap,
        "limit": limit,
        "max_per_label": max_per_label,
        "context_scale": context_scale,
        "min_context": min_context,
        "image_size": image_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-predictions", default="reports/vlm/boundary_graph_node_gnn_v24_dev50_predictions.jsonl")
    parser.add_argument("--locked-predictions", default="reports/vlm/boundary_type_fusion_v24_locked50_predictions.jsonl")
    parser.add_argument("--dataset", default="datasets/boundary_expert_public_raster_v19")
    parser.add_argument("--output-dir", default="datasets/boundary_door_window_crop_context_v24")
    parser.add_argument("--dev-limit", type=int, default=50)
    parser.add_argument("--locked-limit", type=int, default=50)
    parser.add_argument("--cap", type=int, default=800)
    parser.add_argument("--context-scale", type=float, default=2.0)
    parser.add_argument("--min-context", type=int, default=48)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--max-train-per-label", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=20260511)
    args = parser.parse_args()

    dataset = ROOT / args.dataset
    output_dir = ROOT / args.output_dir
    summary = {
        "version": "boundary_door_window_crop_context_v24",
        "claim_boundary": "Raster crops are generated from candidate boxes. Gold labels are used only for offline dev training and locked evaluation manifests.",
        "source_integrity": {
            "runtime_input": "raster_page_pixels_and_raster_derived_candidate_boxes",
            "svg_geometry_used_at_runtime": False,
            "locked_gold_used_for_training": False,
        },
        "splits": [
            build_split(
                predictions_path=ROOT / args.dev_predictions,
                gold_path=dataset / "dev.jsonl",
                output_dir=output_dir,
                split="dev50_train",
                limit=args.dev_limit,
                cap=args.cap,
                context_scale=args.context_scale,
                min_context=args.min_context,
                image_size=args.image_size,
                mode="train",
                max_per_label=args.max_train_per_label,
                seed=args.seed,
            ),
            build_split(
                predictions_path=ROOT / args.locked_predictions,
                gold_path=dataset / "locked.jsonl",
                output_dir=output_dir,
                split="locked50_eval",
                limit=args.locked_limit,
                cap=args.cap,
                context_scale=args.context_scale,
                min_context=args.min_context,
                image_size=args.image_size,
                mode="eval",
                max_per_label=0,
                seed=args.seed,
            ),
        ],
    }
    write_json(output_dir / "manifest.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
