#!/usr/bin/env python3
"""Build sink/tiny hard-case rows from P2 transfer recovery candidates."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from train_symbol_support_suppression_v35 import load_jsonl, source_path
from train_symbol_tile_detector_v20 import area_bucket, bbox_iou, rel, write_json, write_jsonl


FOCUS_LABELS = {"sink", "equipment"}
FOCUS_AREAS = {"tiny_le_64", "small_le_256"}


def valid_box(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    box = [float(v) for v in value]
    return box if box[2] > box[0] and box[3] > box[1] else None


def load_gold(smoke_rows: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    gold_by_id: dict[str, dict[str, Any]] = {}
    page_meta: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(source_path(smoke_rows)):
        page_id = str(row.get("row_id") or "")
        if page_id and page_id not in page_meta:
            page_meta[page_id] = {"image": row.get("image"), "image_size": row.get("image_size")}
        for gold in ((row.get("targets") or {}).get("boxes") or []):
            target_id = str(gold.get("target_id") or "")
            box = valid_box(gold.get("page_bbox") or gold.get("bbox"))
            if target_id and box:
                gold_by_id[target_id] = {
                    "target_id": target_id,
                    "page_id": page_id,
                    "bbox": box,
                    "label": str(gold.get("label") or "generic_symbol"),
                    "area_bucket": str(gold.get("area_bucket") or area_bucket(box)),
                }
    return gold_by_id, page_meta


def delta_target(pred: list[float], gold: list[float]) -> dict[str, float]:
    pw = max(pred[2] - pred[0], 1e-6)
    ph = max(pred[3] - pred[1], 1e-6)
    pcx = (pred[0] + pred[2]) * 0.5
    pcy = (pred[1] + pred[3]) * 0.5
    gw = max(gold[2] - gold[0], 1e-6)
    gh = max(gold[3] - gold[1], 1e-6)
    gcx = (gold[0] + gold[2]) * 0.5
    gcy = (gold[1] + gold[3]) * 0.5
    return {
        "dcx": (gcx - pcx) / pw,
        "dcy": (gcy - pcy) / ph,
        "dw": (gw - pw) / pw,
        "dh": (gh - ph) / ph,
    }


def choose_target(row: dict[str, Any], gold_by_id: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    labels = row.get("labels") or {}
    candidates = []
    best_id = str(labels.get("best_iou_target_id") or "")
    if best_id:
        candidates.append(best_id)
    candidates.extend(str(item) for item in labels.get("center_target_ids") or [])
    for target_id in candidates:
        gold = gold_by_id.get(target_id)
        if gold and (gold["label"] in FOCUS_LABELS or gold["area_bucket"] in FOCUS_AREAS):
            return gold
    return None


def build_rows(candidate_rows: list[dict[str, Any]], gold_by_id: dict[str, dict[str, Any]], page_meta: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out: list[dict[str, Any]] = []
    counts = Counter()
    split_counts = Counter()
    for row in candidate_rows:
        box = valid_box(row.get("bbox"))
        if not box:
            counts["skipped_invalid_box"] += 1
            continue
        gold = choose_target(row, gold_by_id)
        if gold is None:
            counts["skipped_not_focus"] += 1
            continue
        label = str(row.get("label") or "")
        labels = row.get("labels") or {}
        best_iou = float(labels.get("best_iou") or 0.0)
        actual_iou = bbox_iou(box, gold["bbox"])
        center_only = bool(labels.get("center_target_ids")) and actual_iou < 0.30
        sample_kind = "positive_iou" if actual_iou >= 0.30 else "center_only_low_iou" if center_only else "loose_low_iou"
        page_id = str(row.get("page_id") or gold["page_id"])
        features = dict(row.get("features") or {})
        features.update(
            {
                "pred_area_bucket_tiny": 1.0 if area_bucket(box) == "tiny_le_64" else 0.0,
                "target_area_bucket_tiny": 1.0 if gold["area_bucket"] == "tiny_le_64" else 0.0,
                "target_area_bucket_small": 1.0 if gold["area_bucket"] == "small_le_256" else 0.0,
                "target_label_sink": 1.0 if gold["label"] == "sink" else 0.0,
                "target_label_equipment": 1.0 if gold["label"] == "equipment" else 0.0,
                "input_iou": actual_iou,
            }
        )
        item = {
            "page_id": page_id,
            "split": str(row.get("split") or "train"),
            "candidate_id": row.get("candidate_id"),
            "candidate_bbox": box,
            "candidate_label": label,
            "candidate_score": float(row.get("score") or 0.0),
            "target_id": gold["target_id"],
            "target_bbox": gold["bbox"],
            "target_label": gold["label"],
            "target_area_bucket": gold["area_bucket"],
            "input_iou": round(actual_iou, 6),
            "original_best_iou": round(best_iou, 6),
            "sample_kind": sample_kind,
            "box_delta": delta_target(box, gold["bbox"]),
            "features": features,
            "image": (page_meta.get(page_id) or {}).get("image"),
            "image_size": (page_meta.get(page_id) or {}).get("image_size"),
            "source_integrity": {
                "runtime_input_allowed": ["raster crop pixels", "candidate bbox", "candidate score", "predicted type"],
                "gold_used_for_inference": False,
                "runtime_uses_svg_or_cad_geometry": False,
            },
        }
        out.append(item)
        counts["rows"] += 1
        counts[f"kind:{sample_kind}"] += 1
        counts[f"target_label:{gold['label']}"] += 1
        counts[f"target_area:{gold['area_bucket']}"] += 1
        split_counts[item["split"]] += 1
    manifest = {
        "version": "symbol_sink_tiny_hardcase_dataset_v49",
        "rows": len(out),
        "counts": dict(counts),
        "split_counts": dict(split_counts),
        "focus_labels": sorted(FOCUS_LABELS),
        "focus_areas": sorted(FOCUS_AREAS),
        "source_integrity": {
            "runtime_input_allowed": ["raster crop pixels", "candidate bbox", "candidate score", "predicted type"],
            "offline_labels_used_for": ["hard_case_mining", "box_refiner_training", "audit"],
            "gold_used_for_inference": False,
        },
    }
    return out, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="datasets/symbol_detector_listwise_recovery_v47_p2_transfer_smoke120/manifest.json")
    parser.add_argument("--smoke-rows", default="datasets/symbol_tile_detector_tiny_sahi_v21/smoke_v30.jsonl")
    parser.add_argument("--output-dir", default="datasets/symbol_sink_tiny_hardcases_v49")
    args = parser.parse_args()

    manifest = json.loads(source_path(args.data).read_text(encoding="utf-8"))
    candidate_rows = load_jsonl(source_path(manifest["outputs"]["rows"]))
    gold_by_id, page_meta = load_gold(source_path(args.smoke_rows))
    rows, out_manifest = build_rows(candidate_rows, gold_by_id, page_meta)
    output_dir = source_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "rows.jsonl", rows)
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_split[str(row.get("split") or "train")].append(row)
    outputs = {"rows": rel(output_dir / "rows.jsonl")}
    for split, split_rows in sorted(by_split.items()):
        path = output_dir / f"{split}.jsonl"
        write_jsonl(path, split_rows)
        outputs[split] = rel(path)
    out_manifest.update(
        {
            "inputs": {"candidate_manifest": rel(source_path(args.data)), "smoke_rows": rel(source_path(args.smoke_rows))},
            "outputs": outputs,
        }
    )
    write_json(output_dir / "manifest.json", out_manifest)
    print(json.dumps(out_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
