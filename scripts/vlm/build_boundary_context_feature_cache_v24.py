#!/usr/bin/env python3
"""Build auditable page-context feature caches for v24 boundary proposals.

This is the first P0 item in todo.json.  It materializes raster-runtime
features from the boundary candidate stream once, and keeps supervised labels
in a separate offline-only column group for training/audit.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np

from apply_boundary_proposals_with_graph_node_gnn_v24 import (
    BOUNDARY_TO_GRAPH_LABEL,
    LABELS,
    bbox,
    center,
    center_covered,
    iou,
    load_jsonl,
    write_json,
)


ROOT = Path(__file__).resolve().parents[2]

RUNTIME_FEATURE_COLUMNS = [
    "row_id",
    "candidate_id",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "bbox_width",
    "bbox_height",
    "bbox_area",
    "bbox_aspect_log",
    "center_x",
    "center_y",
    "area_norm",
    "length_norm",
    "thickness_norm",
    "orientation",
    "orientation_horizontal",
    "orientation_vertical",
    "segment_x1",
    "segment_y1",
    "segment_x2",
    "segment_y2",
    "segment_length",
    "segment_length_norm",
    "junction_proximity",
    "collinear_chain_id",
    "collinear_chain_size",
    "yolo_hint",
    "hint_hard_wall",
    "hint_door",
    "hint_window",
    "hint_conf",
    "conf_rank_pct",
    "duplicate_group_id",
    "duplicate_count",
    "duplicate_iou_050",
    "duplicate_iou_070",
    "same_hint_iou_050",
    "same_axis_near_8",
    "same_axis_near_16",
    "cross_axis_near_16",
    "hard_wall_overlap_iou_010",
    "opening_hint_neighbor_support",
    "door_window_competition_near_16",
    "page_candidate_count",
    "page_hard_wall_hint_count",
    "page_door_hint_count",
    "page_window_hint_count",
]

OFFLINE_ONLY_COLUMNS = [
    "gold_label",
    "matched_gold_id",
    "match_iou",
    "match_center_covered",
    "is_false_positive",
    "is_duplicate_support_negative",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def gold_items(row: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for target in (row.get("targets") or {}).get("boxes") or []:
        target_box = bbox(target.get("bbox"))
        label = BOUNDARY_TO_GRAPH_LABEL.get(str(target.get("label")), str(target.get("label")))
        if target_box is not None and label in LABELS:
            out.append(
                {
                    "bbox": target_box,
                    "label": label,
                    "target_id": str(target.get("target_id") or ""),
                }
            )
    return out


def load_gold(path: Path, limit: int | None) -> dict[str, list[dict[str, Any]]]:
    return {str(row.get("id")): gold_items(row) for row in load_jsonl(path, limit)}


def label_hint(candidate: dict[str, Any]) -> str:
    hint = str(candidate.get("label_hint") or candidate.get("prediction") or "")
    return hint if hint in LABELS else "hard_wall"


def confidence(candidate: dict[str, Any]) -> float:
    try:
        return float(candidate.get("proposal_confidence") or candidate.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def box_shape_features(box: list[float], image_size: list[int] | None) -> dict[str, float | str]:
    width = max(box[2] - box[0], 1e-6)
    height = max(box[3] - box[1], 1e-6)
    image_w = float(image_size[0]) if image_size else 1.0
    image_h = float(image_size[1]) if image_size else 1.0
    image_area = max(image_w * image_h, 1.0)
    diag = max(math.hypot(image_w, image_h), 1.0)
    horizontal = width >= height
    cx, cy = center(box)
    if horizontal:
        segment = [box[0], cy, box[2], cy]
    else:
        segment = [cx, box[1], cx, box[3]]
    return {
        "bbox_width": width,
        "bbox_height": height,
        "bbox_area": width * height,
        "bbox_aspect_log": math.log(width / height),
        "center_x": cx,
        "center_y": cy,
        "area_norm": (width * height) / image_area,
        "length_norm": max(width, height) / diag,
        "thickness_norm": min(width, height) / diag,
        "orientation": "horizontal" if horizontal else "vertical",
        "orientation_horizontal": 1.0 if horizontal else 0.0,
        "orientation_vertical": 0.0 if horizontal else 1.0,
        "segment_x1": segment[0],
        "segment_y1": segment[1],
        "segment_x2": segment[2],
        "segment_y2": segment[3],
        "segment_length": max(width, height),
        "segment_length_norm": max(width, height) / diag,
    }


def connected_components(mask: np.ndarray) -> list[int]:
    n = int(mask.shape[0])
    group = [-1] * n
    current = 0
    for start in range(n):
        if group[start] >= 0:
            continue
        queue: deque[int] = deque([start])
        group[start] = current
        while queue:
            idx = queue.popleft()
            for nxt in np.where(mask[idx])[0].tolist():
                if group[nxt] < 0:
                    group[nxt] = current
                    queue.append(int(nxt))
        current += 1
    return group


def collinear_chain_ids(box_array: np.ndarray, horizontal_flags: np.ndarray) -> tuple[list[str], dict[str, int]]:
    if len(box_array) == 0:
        return [], {}
    centers_x = (box_array[:, 0] + box_array[:, 2]) * 0.5
    centers_y = (box_array[:, 1] + box_array[:, 3]) * 0.5
    chain_keys = []
    for idx, horizontal in enumerate(horizontal_flags.tolist()):
        axis = "h" if horizontal else "v"
        coord = centers_y[idx] if horizontal else centers_x[idx]
        chain_keys.append(f"{axis}:{int(round(float(coord) / 8.0))}")
    counts = Counter(chain_keys)
    return chain_keys, dict(counts)


def match_gold(candidate_box: list[float], golds: list[dict[str, Any]]) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    best_score = 0.0
    best_iou = 0.0
    best_center = False
    for gold in golds:
        gold_box = gold["bbox"]
        overlap = iou(candidate_box, gold_box)
        covered = center_covered(candidate_box, gold_box)
        score = max(overlap, 1.0 if covered else 0.0)
        if score > best_score:
            best = gold
            best_score = score
            best_iou = overlap
            best_center = covered
    if best is None or best_score <= 0.0:
        return {
            "gold_label": "background",
            "matched_gold_id": "",
            "match_iou": 0.0,
            "match_center_covered": False,
        }
    return {
        "gold_label": best["label"],
        "matched_gold_id": best["target_id"],
        "match_iou": round(float(best_iou), 6),
        "match_center_covered": bool(best_center),
    }


def page_feature_rows(row: dict[str, Any], golds: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    raw_candidates = (row.get("candidate_stream") or [])[:cap]
    candidates = [candidate for candidate in raw_candidates if bbox(candidate.get("bbox")) is not None]
    boxes = [bbox(candidate.get("bbox")) for candidate in candidates]
    if not boxes:
        return []
    box_array = np.asarray(boxes, dtype=np.float32)
    widths = np.maximum(box_array[:, 2] - box_array[:, 0], 1e-6)
    heights = np.maximum(box_array[:, 3] - box_array[:, 1], 1e-6)
    areas = widths * heights
    horizontal_flags = widths >= heights
    ix1 = np.maximum(box_array[:, None, 0], box_array[None, :, 0])
    iy1 = np.maximum(box_array[:, None, 1], box_array[None, :, 1])
    ix2 = np.minimum(box_array[:, None, 2], box_array[None, :, 2])
    iy2 = np.minimum(box_array[:, None, 3], box_array[None, :, 3])
    inter = np.maximum(ix2 - ix1, 0.0) * np.maximum(iy2 - iy1, 0.0)
    union = np.maximum(areas[:, None] + areas[None, :] - inter, 1e-9)
    ious = inter / union
    np.fill_diagonal(ious, 0.0)

    centers_x = (box_array[:, 0] + box_array[:, 2]) * 0.5
    centers_y = (box_array[:, 1] + box_array[:, 3]) * 0.5
    same_axis = horizontal_flags[:, None] == horizontal_flags[None, :]
    axis_gaps = np.where(
        horizontal_flags[:, None],
        np.abs(centers_y[:, None] - centers_y[None, :]),
        np.abs(centers_x[:, None] - centers_x[None, :]),
    )
    np.fill_diagonal(axis_gaps, np.inf)

    hints = np.asarray([label_hint(candidate) for candidate in candidates], dtype=object)
    confs = np.asarray([confidence(candidate) for candidate in candidates], dtype=np.float32)
    same_hint = hints[:, None] == hints[None, :]
    hard_wall_hint = hints == "hard_wall"
    opening_hint = np.isin(hints, ["door", "window"])
    door_window_hint = opening_hint
    duplicate_groups = connected_components(ious >= 0.50)
    duplicate_counts = Counter(duplicate_groups)
    chain_ids, chain_counts = collinear_chain_ids(box_array, horizontal_flags)

    order = np.argsort(confs)[::-1]
    rank_pct = np.zeros(len(candidates), dtype=np.float32)
    for rank, idx in enumerate(order.tolist()):
        rank_pct[idx] = rank / max(len(candidates) - 1, 1)

    matched = [match_gold(box, golds) for box in boxes]
    best_by_gold: dict[str, tuple[int, float, float]] = {}
    for idx, item in enumerate(matched):
        gold_id = str(item["matched_gold_id"])
        if not gold_id:
            continue
        key = (float(item["match_iou"]), float(confs[idx]))
        if gold_id not in best_by_gold or key > (best_by_gold[gold_id][1], best_by_gold[gold_id][2]):
            best_by_gold[gold_id] = (idx, key[0], key[1])

    image_size = row.get("image_size") if isinstance(row.get("image_size"), list) else None
    page_counts = Counter(hints.tolist())
    out = []
    for idx, candidate in enumerate(candidates):
        box = boxes[idx]
        shape = box_shape_features(box, image_size)
        hint = str(hints[idx])
        same_axis_near_8 = int((same_axis[idx] & (axis_gaps[idx] <= 8.0)).sum())
        same_axis_near_16 = int((same_axis[idx] & (axis_gaps[idx] <= 16.0)).sum())
        cross_axis_near_16 = int((~same_axis[idx] & (axis_gaps[idx] <= 16.0)).sum())
        hard_wall_overlap = int(((ious[idx] >= 0.10) & hard_wall_hint).sum())
        opening_support = int(opening_hint[idx] and ((opening_hint & (axis_gaps[idx] <= 16.0)).sum()))
        door_window_competition = int((door_window_hint & (axis_gaps[idx] <= 16.0)).sum())
        junction_proximity = float(np.min(axis_gaps[idx])) if np.isfinite(np.min(axis_gaps[idx])) else -1.0
        offline = matched[idx]
        matched_gold_id = str(offline["matched_gold_id"])
        duplicate_group = int(duplicate_groups[idx])
        is_duplicate_support_negative = bool(
            matched_gold_id and matched_gold_id in best_by_gold and best_by_gold[matched_gold_id][0] != idx
        )
        row_out: dict[str, Any] = {
            "row_id": str(row.get("id")),
            "candidate_id": str(candidate.get("candidate_id") or candidate.get("id") or f"candidate_{idx:06d}"),
            "bbox_x1": round(float(box[0]), 6),
            "bbox_y1": round(float(box[1]), 6),
            "bbox_x2": round(float(box[2]), 6),
            "bbox_y2": round(float(box[3]), 6),
            **{key: round(float(value), 6) if isinstance(value, float) else value for key, value in shape.items()},
            "junction_proximity": round(junction_proximity, 6),
            "collinear_chain_id": chain_ids[idx],
            "collinear_chain_size": int(chain_counts.get(chain_ids[idx], 0)),
            "yolo_hint": hint,
            "hint_hard_wall": 1.0 if hint == "hard_wall" else 0.0,
            "hint_door": 1.0 if hint == "door" else 0.0,
            "hint_window": 1.0 if hint == "window" else 0.0,
            "hint_conf": round(float(confs[idx]), 6),
            "conf_rank_pct": round(float(rank_pct[idx]), 6),
            "duplicate_group_id": f"{row.get('id')}|dup{duplicate_group}",
            "duplicate_count": int(duplicate_counts[duplicate_group]),
            "duplicate_iou_050": int((ious[idx] >= 0.50).sum()),
            "duplicate_iou_070": int((ious[idx] >= 0.70).sum()),
            "same_hint_iou_050": int(((ious[idx] >= 0.50) & same_hint[idx]).sum()),
            "same_axis_near_8": same_axis_near_8,
            "same_axis_near_16": same_axis_near_16,
            "cross_axis_near_16": cross_axis_near_16,
            "hard_wall_overlap_iou_010": hard_wall_overlap,
            "opening_hint_neighbor_support": opening_support,
            "door_window_competition_near_16": door_window_competition,
            "page_candidate_count": len(candidates),
            "page_hard_wall_hint_count": int(page_counts.get("hard_wall", 0)),
            "page_door_hint_count": int(page_counts.get("door", 0)),
            "page_window_hint_count": int(page_counts.get("window", 0)),
            "gold_label": offline["gold_label"],
            "matched_gold_id": matched_gold_id,
            "match_iou": offline["match_iou"],
            "match_center_covered": offline["match_center_covered"],
            "is_false_positive": offline["gold_label"] == "background",
            "is_duplicate_support_negative": is_duplicate_support_negative,
        }
        out.append(row_out)
    return out


def write_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_split(
    name: str,
    prediction_path: Path,
    gold_path: Path,
    output_path: Path,
    limit: int | None,
    cap: int,
) -> dict[str, Any]:
    started = time.time()
    prediction_rows = load_jsonl(prediction_path, limit)
    gold_by_id = load_gold(gold_path, limit)
    all_rows: list[dict[str, Any]] = []
    page_candidate_counts = []
    missing_features = Counter()
    class_counts = Counter()
    for row in prediction_rows:
        rows = page_feature_rows(row, gold_by_id.get(str(row.get("id")), []), cap)
        all_rows.extend(rows)
        page_candidate_counts.append(len(rows))
        for feature_row in rows:
            class_counts[str(feature_row["gold_label"])] += 1
            for column in RUNTIME_FEATURE_COLUMNS + OFFLINE_ONLY_COLUMNS:
                if column not in feature_row or feature_row[column] is None:
                    missing_features[column] += 1
    write_jsonl_gz(output_path, all_rows)
    return {
        "split": name,
        "prediction_path": str(prediction_path.relative_to(ROOT) if prediction_path.is_relative_to(ROOT) else prediction_path),
        "gold_path": str(gold_path.relative_to(ROOT) if gold_path.is_relative_to(ROOT) else gold_path),
        "output_path": str(output_path.relative_to(ROOT) if output_path.is_relative_to(ROOT) else output_path),
        "output_format": "jsonl.gz",
        "rows": len(all_rows),
        "pages": len(prediction_rows),
        "nonempty_pages": sum(1 for count in page_candidate_counts if count > 0),
        "page_candidate_count_min": min(page_candidate_counts) if page_candidate_counts else 0,
        "page_candidate_count_max": max(page_candidate_counts) if page_candidate_counts else 0,
        "page_candidate_count_mean": round(sum(page_candidate_counts) / max(len(page_candidate_counts), 1), 6),
        "class_counts": dict(class_counts),
        "missing_feature_counts": dict(missing_features),
        "source_integrity": {
            "prediction_sha256": sha256_file(prediction_path),
            "gold_sha256": sha256_file(gold_path),
            "runtime_input_claim": "raster candidate stream only; gold labels are offline-only columns",
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-predictions", default="reports/vlm/boundary_public_raster_v24_yolo_full_dev493_candidate_stream.jsonl")
    parser.add_argument("--locked-predictions", default="reports/vlm/boundary_public_raster_v24_yolo_full_locked50_candidate_stream.jsonl")
    parser.add_argument("--dataset", default="datasets/boundary_expert_public_raster_v19")
    parser.add_argument("--output-dir", default="datasets/boundary_context_feature_cache_v24")
    parser.add_argument("--audit-output", default="reports/vlm/boundary_context_feature_cache_v24_audit.json")
    parser.add_argument("--dev-limit", type=int, default=493)
    parser.add_argument("--locked-limit", type=int, default=50)
    parser.add_argument("--cap", type=int, default=800)
    args = parser.parse_args()

    dataset = resolve(args.dataset)
    output_dir = resolve(args.output_dir)
    started = time.time()
    dev_audit = build_split(
        "dev493",
        resolve(args.dev_predictions),
        dataset / "dev.jsonl",
        output_dir / "dev493_features.jsonl.gz",
        args.dev_limit,
        args.cap,
    )
    locked_audit = build_split(
        "locked50",
        resolve(args.locked_predictions),
        dataset / "locked.jsonl",
        output_dir / "locked50_features.jsonl.gz",
        args.locked_limit,
        args.cap,
    )
    total_rows = dev_audit["rows"] + locked_audit["rows"]
    total_missing = sum(sum(split["missing_feature_counts"].values()) for split in [dev_audit, locked_audit])
    audit = {
        "version": "boundary_context_feature_cache_v24",
        "task": "P0-01-boundary-context-feature-cache",
        "claim_boundary": "Runtime feature columns are derived from raster candidate stream geometry, hints, confidence, and page-local candidate context. Offline-only columns use gold labels for supervised training/audit and must not be used at inference.",
        "runtime_feature_columns": RUNTIME_FEATURE_COLUMNS,
        "offline_only_columns": OFFLINE_ONLY_COLUMNS,
        "splits": [dev_audit, locked_audit],
        "success_gate": {
            "dev_pages_min": 490,
            "locked50_pages_min": 50,
            "row_coverage_min": 0.99,
            "feature_missing_rate_max": 0.001,
            "cache_build_runtime_target_minutes": 20,
            "dev_pages": dev_audit["pages"],
            "locked50_pages": locked_audit["pages"],
            "row_coverage": 1.0 if total_rows > 0 else 0.0,
            "feature_missing_rate": round(total_missing / max(total_rows * (len(RUNTIME_FEATURE_COLUMNS) + len(OFFLINE_ONLY_COLUMNS)), 1), 8),
            "elapsed_minutes": round((time.time() - started) / 60.0, 4),
        },
    }
    gate = audit["success_gate"]
    gate["passed"] = (
        gate["dev_pages"] >= gate["dev_pages_min"]
        and gate["locked50_pages"] >= gate["locked50_pages_min"]
        and gate["row_coverage"] >= gate["row_coverage_min"]
        and gate["feature_missing_rate"] <= gate["feature_missing_rate_max"]
        and gate["elapsed_minutes"] <= gate["cache_build_runtime_target_minutes"]
    )
    write_json(resolve(args.audit_output), audit)
    print(json.dumps({"audit": args.audit_output, "success_gate": gate}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
