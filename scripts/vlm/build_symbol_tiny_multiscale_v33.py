#!/usr/bin/env python3
"""Build a tiny/multiscale proposal dataset from the v31 proposal cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from train_symbol_tile_detector_v20 import area_bucket, bbox_iou, center_covered, load_jsonl, rel, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "datasets/symbol_tiny_multiscale_v33"
REPORT = ROOT / "reports/vlm/symbol_tiny_multiscale_v33_dataset_audit.json"

TINY_BUCKETS = {"tiny_le_64", "small_le_256"}
SPLIT_RATIO = 0.8


def page_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    pages: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = str(row.get("row_id"))
        if row_id not in pages:
            pages[row_id] = {
                "row_id": row_id,
                "image": row.get("image"),
                "image_size": row.get("image_size") or [0, 0],
                "label_counts": Counter(),
                "area_counts": Counter(),
                "targets": {},
            }
        pages[row_id]["label_counts"][str(row.get("label") or "generic_symbol")] += 1
        pages[row_id]["area_counts"][str(row.get("area_bucket") or "unknown")] += 1
        target_id = str(row.get("target_id") or "")
        if target_id:
            pages[row_id]["targets"][target_id] = {
                "target_id": target_id,
                "label": str(row.get("label") or "generic_symbol"),
                "area_bucket": str(row.get("area_bucket") or "unknown"),
                "bbox": [float(v) for v in row.get("page_bbox") or []],
            }
    return pages


def stable_split(row_id: str, ratio: float = SPLIT_RATIO) -> str:
    digest = hashlib.sha1(row_id.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return "train" if value < ratio else "smoke"


def valid_box(box: Any) -> list[float] | None:
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        values = [float(v) for v in box]
    except (TypeError, ValueError):
        return None
    if values[2] <= values[0] or values[3] <= values[1]:
        return None
    return values


class IntegralImageStats:
    def __init__(self, image_path: str) -> None:
        with Image.open(image_path if Path(image_path).is_absolute() else ROOT / image_path) as image:
            gray = np.asarray(image.convert("L"), dtype=np.float64)
        self.width = int(gray.shape[1])
        self.height = int(gray.shape[0])
        self.sum = self._integral(gray)
        self.sum_sq = self._integral(gray * gray)
        self.dark_210 = self._integral((gray < 210).astype(np.float64))
        self.dark_160 = self._integral((gray < 160).astype(np.float64))
        self.dark_80 = self._integral((gray < 80).astype(np.float64))

    @staticmethod
    def _integral(array: np.ndarray) -> np.ndarray:
        return np.pad(array.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)), mode="constant")

    @staticmethod
    def _sum(integral: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
        return float(integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1])

    def region(self, box: list[float], pad: int = 0) -> tuple[int, int, int, int]:
        x1 = max(0, min(self.width, int(round(box[0])) - pad))
        y1 = max(0, min(self.height, int(round(box[1])) - pad))
        x2 = max(0, min(self.width, int(round(box[2])) + pad))
        y2 = max(0, min(self.height, int(round(box[3])) + pad))
        return x1, y1, x2, y2

    def crop_features(self, box: list[float], pad: int = 1) -> dict[str, float]:
        x1, y1, x2, y2 = self.region(box, pad=pad)
        if x2 <= x1 or y2 <= y1:
            return {
                "crop_dark_ratio_210": 0.0,
                "crop_dark_ratio_160": 0.0,
                "crop_dark_ratio_80": 0.0,
                "crop_mean": 0.0,
                "crop_std": 0.0,
                "crop_inner_dark_ratio_210": 0.0,
                "crop_border_dark_ratio_210": 0.0,
                "crop_dark_balance_x": 0.0,
                "crop_dark_balance_y": 0.0,
            }
        total = max(float((x2 - x1) * (y2 - y1)), 1.0)
        raw_sum = self._sum(self.sum, x1, y1, x2, y2)
        raw_sum_sq = self._sum(self.sum_sq, x1, y1, x2, y2)
        mean = raw_sum / total
        std = float(np.sqrt(max(raw_sum_sq / total - mean * mean, 0.0)))
        dark_210 = self._sum(self.dark_210, x1, y1, x2, y2)
        dark_160 = self._sum(self.dark_160, x1, y1, x2, y2)
        dark_80 = self._sum(self.dark_80, x1, y1, x2, y2)

        ix1 = x1 + max(1, (x2 - x1) // 4)
        iy1 = y1 + max(1, (y2 - y1) // 4)
        ix2 = x2 - max(1, (x2 - x1) // 4)
        iy2 = y2 - max(1, (y2 - y1) // 4)
        inner_total = 0.0
        inner_area = 0.0
        if ix2 > ix1 and iy2 > iy1:
            inner_area = float((ix2 - ix1) * (iy2 - iy1))
            inner_total = self._sum(self.dark_210, ix1, iy1, ix2, iy2)
        border_area = max(total - inner_area, 1.0)
        border_total = max(dark_210 - inner_total, 0.0)

        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        left_dark = self._sum(self.dark_210, x1, y1, mid_x, y2) if mid_x > x1 else 0.0
        right_dark = self._sum(self.dark_210, mid_x, y1, x2, y2) if x2 > mid_x else 0.0
        top_dark = self._sum(self.dark_210, x1, y1, x2, mid_y) if mid_y > y1 else 0.0
        bottom_dark = self._sum(self.dark_210, x1, mid_y, x2, y2) if y2 > mid_y else 0.0

        return {
            "crop_dark_ratio_210": dark_210 / total,
            "crop_dark_ratio_160": dark_160 / total,
            "crop_dark_ratio_80": dark_80 / total,
            "crop_mean": mean / 255.0,
            "crop_std": std / 255.0,
            "crop_inner_dark_ratio_210": inner_total / max(inner_area, 1.0),
            "crop_border_dark_ratio_210": border_total / border_area,
            "crop_dark_balance_x": (left_dark - right_dark) / max(dark_210, 1.0),
            "crop_dark_balance_y": (top_dark - bottom_dark) / max(dark_210, 1.0),
        }


def candidate_features(
    page: dict[str, Any],
    candidate: dict[str, Any],
    match: dict[str, Any],
    stats: IntegralImageStats,
    index: int,
    total: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    box = valid_box(candidate.get("bbox"))
    if box is None:
        raise ValueError("invalid candidate box")
    width = max(1.0, box[2] - box[0])
    height = max(1.0, box[3] - box[1])
    page_w, page_h = [max(1.0, float(v)) for v in page.get("image_size") or [1, 1]]
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    best_iou = float(match.get("best_iou") or 0.0)
    center_hit = 1.0 if match.get("center_target_ids") else 0.0
    area = width * height
    bucket = area_bucket([int(round(v)) for v in box])
    features = {
        "score": float(candidate.get("score", 0.0)),
        "pre_selector_score": float(candidate.get("pre_selector_score", candidate.get("score", 0.0))),
        "selector_score": float(candidate.get("selector_score", candidate.get("score", 0.0))),
        "is_mask_v28": float(str(candidate.get("proposal_source") or "") == "mask_v28"),
        "is_center_branch_v30": float(str(candidate.get("proposal_source") or "") == "center_branch_v30"),
        "label_id": float(candidate.get("label_id") or 5),
        "cluster_id": float(candidate.get("cluster_id") or 0),
        "candidate_index": float(index),
        "candidate_count_page": float(total),
        "width": width,
        "height": height,
        "area": area,
        "aspect": width / max(height, 1e-6),
        "area_ratio_page": area / max(page_w * page_h, 1.0),
        "center_x_ratio": cx / max(page_w, 1.0),
        "center_y_ratio": cy / max(page_h, 1.0),
        "edge_distance_ratio": min(cx, page_w - cx, cy, page_h - cy) / max(min(page_w, page_h), 1.0),
        "best_iou_hint": best_iou,
        "center_hit_hint": center_hit,
        "tiny_hint": float(bucket in TINY_BUCKETS),
    }
    features.update(stats.crop_features(box, pad=1))
    features.update({f"pad2_{k}": v for k, v in stats.crop_features(box, pad=2).items()})
    features.update({f"pad4_{k}": v for k, v in stats.crop_features(box, pad=4).items()})
    best_target_id = str(match.get("best_iou_target_id") or "")
    gold_target = (page.get("targets") or {}).get(best_target_id) if best_target_id else None
    gold_bucket = str((gold_target or {}).get("area_bucket") or "unknown")
    labels = {
        "target_iou_0_30": int(best_iou >= 0.30),
        "target_center": int(center_hit > 0.0),
        "target_tiny_iou_0_30": int(best_iou >= 0.30 and gold_bucket in TINY_BUCKETS),
    }
    return features, labels, gold_bucket, best_target_id


def build_split(rows: list[dict[str, Any]], pages: dict[str, dict[str, Any]], split: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    page_rows: list[dict[str, Any]] = []
    counts = Counter()
    for row in rows:
        row_id = str(row.get("row_id"))
        if stable_split(row_id) != split:
            continue
        page = pages[row_id]
        page_rows.append(
            {
                "row_id": row_id,
                "image": page.get("image"),
                "image_size": page.get("image_size"),
                "gold_symbols": list((page.get("targets") or {}).values()),
            }
        )
        stats = IntegralImageStats(str(page.get("image") or ""))
        candidates = list(row.get("predicted_symbols") or [])
        matches = list(row.get("candidate_gold_matches") or [])
        total = len(candidates)
        for index, candidate in enumerate(candidates):
            box = valid_box(candidate.get("bbox"))
            match = matches[index] if index < len(matches) else {"best_iou": 0.0, "center_target_ids": []}
            if box is None:
                continue
            features, labels, gold_bucket, best_target_id = candidate_features(page, candidate, match, stats, index, total)
            item = {
                "row_id": row_id,
                "split": split,
                "image": page.get("image"),
                "image_size": page.get("image_size"),
                "page_candidate_count": total,
                "candidate_index": index,
                "candidate": candidate,
                "match": {
                    "best_iou": float(match.get("best_iou") or 0.0),
                    "center_target_ids": list(match.get("center_target_ids") or []),
                    "gold_area_bucket": gold_bucket,
                    "best_iou_target_id": best_target_id,
                },
                "features": features,
                "labels": labels,
            }
            out_rows.append(item)
            counts[f"candidate_source:{candidate.get('proposal_source') or 'unknown'}"] += 1
            counts[f"target_iou_0_30:{labels['target_iou_0_30']}"] += 1
            counts[f"target_center:{labels['target_center']}"] += 1
            counts[f"target_tiny_iou_0_30:{labels['target_tiny_iou_0_30']}"] += 1
            if labels["target_tiny_iou_0_30"]:
                counts["tiny_positive"] += 1
    return out_rows, page_rows, {"rows": len(out_rows), "counts": dict(counts), "pages": len({row["row_id"] for row in out_rows})}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-cache", default="reports/vlm/symbol_proposal_eval_v31_smoke_cache.jsonl")
    parser.add_argument("--page-targets", default="datasets/symbol_center_branch_v30/smoke_center_targets.jsonl")
    parser.add_argument("--output-dir", default=str(OUT))
    args = parser.parse_args()

    cache_rows = load_jsonl(Path(args.input_cache))
    page_rows = load_jsonl(Path(args.page_targets))
    if not cache_rows or not page_rows:
        raise SystemExit("missing proposal cache or page targets")

    pages = page_lookup(page_rows)
    train_rows, train_pages, train_audit = build_split(cache_rows, pages, "train")
    smoke_rows, smoke_pages, smoke_audit = build_split(cache_rows, pages, "smoke")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "smoke.jsonl", smoke_rows)
    write_jsonl(output_dir / "train_pages.jsonl", train_pages)
    write_jsonl(output_dir / "smoke_pages.jsonl", smoke_pages)

    feature_names = sorted(train_rows[0]["features"]) if train_rows else sorted(smoke_rows[0]["features"])
    manifest = {
        "version": "symbol_tiny_multiscale_v33",
        "claim_boundary": "Offline tiny/multiscale proposal training view built from cached raster proposals. Runtime features are candidate score, box geometry, and raster crop statistics only.",
        "inputs": {
            "proposal_cache": rel(Path(args.input_cache)),
            "page_targets": rel(Path(args.page_targets)),
        },
        "outputs": {
            "train": rel(output_dir / "train.jsonl"),
            "smoke": rel(output_dir / "smoke.jsonl"),
            "train_pages": rel(output_dir / "train_pages.jsonl"),
            "smoke_pages": rel(output_dir / "smoke_pages.jsonl"),
        },
        "runtime_contract": {
            "model_input_features": ["image_crop_pixels", "candidate_bbox", "proposal_scores"],
            "forbidden_runtime_features": ["svg_geometry", "annotation_path", "raw_label", "semantic_type"],
            "label_use": "offline_supervised_training_and_evaluation_only",
        },
        "feature_names": [name for name in feature_names if name not in {"best_iou_hint", "center_hit_hint"}],
        "feature_names_raw": feature_names,
        "splits": {"train": train_audit, "smoke": smoke_audit},
        "source_counts": {
            "cache_rows": len(cache_rows),
            "pages": len(page_rows),
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    audit = {
        "version": "symbol_tiny_multiscale_v33_dataset_audit",
        "outputs": {
            "manifest": rel(output_dir / "manifest.json"),
            "train": rel(output_dir / "train.jsonl"),
            "smoke": rel(output_dir / "smoke.jsonl"),
            "train_pages": rel(output_dir / "train_pages.jsonl"),
            "smoke_pages": rel(output_dir / "smoke_pages.jsonl"),
        },
        "counts": {
            "train_rows": len(train_rows),
            "smoke_rows": len(smoke_rows),
            "train_pages": train_audit["pages"],
            "smoke_pages": smoke_audit["pages"],
        },
        "split_audits": {"train": train_audit, "smoke": smoke_audit},
        "gate": {
            "smoke_has_tiny_positives": smoke_audit["counts"].get("tiny_positive", 0) > 0,
            "feature_contract_written": True,
            "split_disjoint_by_row_id": True,
        },
    }
    audit["gate"]["passed"] = all(bool(v) for v in audit["gate"].values())
    write_json(output_dir / "dataset_audit.json", audit)
    print(json.dumps({"manifest": rel(output_dir / "manifest.json"), "counts": audit["counts"], "gate": audit["gate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
