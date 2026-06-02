#!/usr/bin/env python3
"""Bridge detector page predictions into a suppression listwise training view."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from train_symbol_tile_detector_v20 import area_bucket, rel, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
import importlib.util

_support_path = ROOT / "scripts" / "vlm" / "build_symbol_support_suppression_dataset_v32.py"
_spec = importlib.util.spec_from_file_location("build_symbol_support_suppression_dataset_v32", _support_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"unable to load {_support_path}")
_support = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_support)
feature_vector = _support.feature_vector
valid_box = _support.valid_box


def source_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def split_for_page(row_id: str) -> str:
    score = sum(ord(ch) for ch in row_id)
    bucket = score % 10
    if bucket < 7:
        return "train"
    if bucket < 8:
        return "dev"
    return "smoke_eval"


def cluster_key(pred: dict[str, Any]) -> str:
    return f"{pred.get('row_id')}|{int(pred.get('cluster_id') or 0)}"


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def build_rows(
    page_predictions: list[dict[str, Any]],
    max_per_page: int,
    smoke_limit: int,
    detector_source: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counts = Counter()
    by_source = Counter()
    by_area = Counter()
    by_split = Counter()
    page_counter = 0
    for page in page_predictions:
        page_id = str(page.get("row_id") or page.get("page_id") or "")
        if not page_id:
            continue
        preds = list(page.get("predicted_symbols") or [])[:max_per_page]
        split = split_for_page(page_id)
        by_split[split] += 1
        cluster_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
        page_w = 1.0
        page_h = 1.0
        for pred in preds:
            cluster_map[cluster_key({"row_id": page_id, **pred})].append(pred)
            box = pred.get("bbox") or []
            if len(box) == 4:
                try:
                    page_w = max(page_w, float(box[2]))
                    page_h = max(page_h, float(box[3]))
                except (TypeError, ValueError):
                    pass
        page_candidate_count = float(len(preds))
        page_stats = {"page_candidate_count": page_candidate_count, "page_w": page_w, "page_h": page_h}
        for index, pred in enumerate(preds):
            box = [float(v) for v in pred.get("bbox") or []]
            if len(box) != 4 or not valid_box(box):
                continue
            source = str(pred.get("proposal_source") or pred.get("source") or detector_source)
            area = area_bucket(box)
            by_source[source] += 1
            by_area[area] += 1
            counts["rows"] += 1
            counts[f"source:{source}"] += 1
            counts[f"area:{area}"] += 1
            row = {
                "row_id": page_id,
                "page_id": page_id,
                "candidate_id": str(pred.get("candidate_id") or f"{page_id}_cand_{index}"),
                "candidate_index": index,
                "split": split,
                "cluster_id": int(pred.get("cluster_id") or 0),
                "cluster_key": cluster_key({"row_id": page_id, **pred}),
                "proposal_source": source,
                "detector_source": detector_source,
                "bbox": box,
                "score": safe_float(pred.get("score") or pred.get("confidence")),
                "selector_score": safe_float(pred.get("selector_score") or pred.get("score") or pred.get("confidence")),
                "pre_selector_score": safe_float(pred.get("pre_selector_score") or pred.get("score") or pred.get("confidence")),
                "label": pred.get("label"),
                "label_id": int(pred.get("label_id") or 5),
                "page_stats": page_stats,
            }
            rows.append(row)
        page_counter += 1
        if smoke_limit and page_counter >= smoke_limit:
            break
    manifest = {
        "version": "symbol_detector_to_suppression_bridge_v47",
        "rows": len(rows),
        "counts": dict(counts),
        "by_source": dict(by_source),
        "by_area": dict(by_area),
        "by_split": dict(by_split),
        "source_integrity": {
            "runtime_input": "raster prediction outputs only",
            "offline_supervision": "page prediction boxes and scores",
            "svg_or_cad_geometry_at_runtime": False,
        },
    }
    return rows, manifest


def attach_listwise_labels(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cluster_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cluster_map[str(row["cluster_key"])].append(row)
    for key, items in cluster_map.items():
        scores = [safe_float(item.get("selector_score") or item.get("score")) for item in items]
        cluster_size = float(len(items))
        cluster_score_max = max(scores) if scores else 0.0
        cluster_score_mean = sum(scores) / max(len(scores), 1)
        cluster_source_center_count = sum(1.0 for item in items if str(item.get("proposal_source")) == "center_branch_v30")
        cluster_mask_count = sum(1.0 for item in items if str(item.get("proposal_source")) == "mask_v28")
        cluster_stats = {
            "cluster_size": cluster_size,
            "cluster_score_max": cluster_score_max,
            "cluster_score_mean": cluster_score_mean,
            "cluster_source_center_count": cluster_source_center_count,
            "cluster_mask_count": cluster_mask_count,
        }
        for row in items:
            row["features"] = feature_vector(
                row,
                row,
                {"best_iou": float(row.get("input_best_iou", 0.0)), "center_target_ids": []},
                cluster_stats,
                row["page_stats"],
                [float(row["page_stats"].get("page_w", 1.0)), float(row["page_stats"].get("page_h", 1.0))],
            )
            row["labels"] = {
                "keep": bool(float(row.get("score", 0.0)) >= 0.5),
                "drop": bool(float(row.get("score", 0.0)) < 0.5),
                "suppression_reason": "bridge_positive" if float(row.get("score", 0.0)) >= 0.5 else "duplicate_support_negative",
                "bridge_positive": bool(float(row.get("score", 0.0)) >= 0.5),
                "duplicate_support_negative": bool(float(row.get("score", 0.0)) < 0.5),
            }
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page-predictions", default="reports/vlm/symbol_yolov8n_seg_rect_v27_page_predictions.jsonl")
    parser.add_argument("--output", default="datasets/symbol_detector_to_suppression_bridge_v47/train.jsonl")
    parser.add_argument("--manifest-output", default="datasets/symbol_detector_to_suppression_bridge_v47/manifest.json")
    parser.add_argument("--smoke-output", default="datasets/symbol_detector_to_suppression_bridge_v47_smoke.jsonl")
    parser.add_argument("--smoke-limit", type=int, default=250)
    parser.add_argument("--max-per-page", type=int, default=240)
    parser.add_argument("--detector-source", default="yolo_seg_rect_v27")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    page_predictions = load_jsonl(source_path(args.page_predictions))
    rows, manifest = build_rows(page_predictions, args.max_per_page, args.smoke_limit, args.detector_source)
    rows = attach_listwise_labels(rows)
    output_path = source_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, rows)
    smoke_rows = rows[: min(len(rows), args.smoke_limit)]
    write_jsonl(source_path(args.smoke_output), smoke_rows)
    split_rows = {"train": [], "dev": [], "locked": []}
    for row in rows:
        split = str(row.get("split") or "train")
        mapped_split = "locked" if split == "smoke_eval" else split
        item = dict(row)
        item["split"] = mapped_split
        split_rows.setdefault(mapped_split, []).append(item)
    train_rows_path = source_path("datasets/symbol_detector_to_suppression_bridge_v47/train_rows.jsonl")
    dev_rows_path = source_path("datasets/symbol_detector_to_suppression_bridge_v47/dev_rows.jsonl")
    locked_rows_path = source_path("datasets/symbol_detector_to_suppression_bridge_v47/locked_rows.jsonl")
    write_jsonl(train_rows_path, split_rows.get("train", []))
    write_jsonl(dev_rows_path, split_rows.get("dev", []))
    write_jsonl(locked_rows_path, split_rows.get("locked", []))
    manifest.update(
        {
            "version": "symbol_detector_to_suppression_bridge_v47",
            "outputs": {
                "rows": rel(output_path),
                "train_rows": rel(train_rows_path),
                "dev_rows": rel(dev_rows_path),
                "locked_rows": rel(locked_rows_path),
            },
            "smoke_output": rel(source_path(args.smoke_output)),
            "page_predictions": rel(source_path(args.page_predictions)),
            "split_counts": {key: len(value) for key, value in split_rows.items()},
        }
    )
    write_json(source_path(args.manifest_output), manifest)
    print(json.dumps({"manifest": manifest, "smoke_rows": len(smoke_rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
