#!/usr/bin/env python3
"""Repair room candidates with wall-supported raster-only proposals.

This script augments the current detector candidate stream with additional room
proposals derived from raster-only room proposal heuristics and boundary support.
Offline gold is used only for audit and comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
REPORT = ROOT / "reports/vlm"
DEFAULT_INPUT = REPORT / "detector_adapter_v18_symbol_proposal_combined.jsonl"
DEFAULT_POLICY = ROOT / "checkpoints/room_proposal_model_v18/policy.json"
DEFAULT_OUTPUT = REPORT / "detector_adapter_v18_room_wall_repair_v18.jsonl"
DEFAULT_AUDIT = REPORT / "room_wall_supported_repair_v18_audit.json"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_topology_relations_v18 import bbox, center, contains_point, integrity, iou, load_gold, write_json, write_jsonl  # noqa: E402
from diagnose_contains_symbol_missing_gold_v18 import candidate_groups  # noqa: E402
from nms_topology_relations_v18 import load_jsonl  # noqa: E402
from train_room_proposal_model_v18 import detect_room_proposals  # noqa: E402


def resolve_image(path: str | None) -> Path | None:
    if not path:
        return None
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def load_gray(row: dict[str, Any], cache: dict[str, np.ndarray]) -> np.ndarray | None:
    image_path = resolve_image(str(row.get("image") or ""))
    if image_path is None or not image_path.exists():
        return None
    key = str(image_path)
    if key not in cache:
        cache[key] = np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)
    return cache[key]


def load_policy(path: Path) -> tuple[dict[str, Any], list[dict[str, int]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = payload.get("selected_params") or {}
    anchor_priors = payload.get("anchor_priors") or []
    return params, anchor_priors


def candidate_stream(row: dict[str, Any]) -> list[dict[str, Any]]:
    return list(((row.get("scene_graph") or {}).get("candidate_stream") or []))


def family_boxes(stream: list[dict[str, Any]], family: str) -> list[list[float]]:
    out: list[list[float]] = []
    for cand in stream:
        if str(cand.get("family") or "") != family:
            continue
        box = bbox(cand.get("bbox"))
        if box is not None:
            out.append(box)
    return out


def box_union(boxes: list[list[float]]) -> list[float] | None:
    if not boxes:
        return None
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def clip_box(box: list[float], width: int, height: int) -> list[float] | None:
    x1 = max(0.0, min(float(width - 1), float(box[0])))
    y1 = max(0.0, min(float(height - 1), float(box[1])))
    x2 = max(x1 + 1.0, min(float(width), float(box[2])))
    y2 = max(y1 + 1.0, min(float(height), float(box[3])))
    if x2 <= x1 or y2 <= y1:
        return None
    return [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)]


def center_distance(left: list[float], right: list[float]) -> float:
    lx, ly = center(left)
    rx, ry = center(right)
    return math.hypot(lx - rx, ly - ry)


def cluster_boxes(boxes: list[list[float]], iou_threshold: float = 0.15, center_threshold: float = 18.0) -> list[list[list[float]]]:
    clusters: list[list[list[float]]] = []
    for box in sorted(boxes, key=lambda item: (item[2] - item[0]) * (item[3] - item[1]), reverse=True):
        placed = False
        for cluster in clusters:
            if any(iou(box, other) >= iou_threshold or center_distance(box, other) <= center_threshold for other in cluster):
                cluster.append(box)
                placed = True
                break
        if not placed:
            clusters.append([box])
    return clusters


def boundary_repair_proposals(row: dict[str, Any], width: int, height: int, boundary_boxes: list[list[float]]) -> list[dict[str, Any]]:
    proposals: list[dict[str, Any]] = []
    for idx, cluster in enumerate(cluster_boxes(boundary_boxes)):
        box = box_union(cluster)
        if box is None:
            continue
        pad = 4.0 + min(8.0, len(cluster) * 0.8)
        expanded = clip_box([box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad], width, height)
        if expanded is None:
            continue
        area = max(0.0, expanded[2] - expanded[0]) * max(0.0, expanded[3] - expanded[1])
        if area < 900:
            continue
        mean_area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in cluster) / max(len(cluster), 1)
        proposals.append(
            {
                "id": f"{row['id']}_room_wall_repair_boundary_{idx:03d}",
                "source": "boundary_cluster_repair",
                "bbox": expanded,
                "confidence": round(min(0.96, 0.34 + 0.03 * len(cluster) + 0.08 * min(mean_area / max(area, 1.0), 1.0)), 6),
                "metadata": {
                    "repair_kind": "boundary_cluster_union",
                    "cluster_size": len(cluster),
                    "cluster_iou_threshold": 0.15,
                    "cluster_center_threshold": 18.0,
                },
            }
        )
    return proposals


def room_repair_proposals(
    row: dict[str, Any],
    arr: np.ndarray | None,
    params: dict[str, Any],
    anchor_priors: list[dict[str, int]],
) -> list[dict[str, Any]]:
    if arr is None:
        return []
    width = int(arr.shape[1])
    height = int(arr.shape[0])
    proposals = detect_room_proposals(row, params, anchor_priors[: int(params.get("anchor_limit", len(anchor_priors)))])
    stream = candidate_stream(row)
    boundary_boxes = family_boxes(stream, "boundary")
    proposals.extend(boundary_repair_proposals(row, width, height, boundary_boxes))
    proposals.sort(key=lambda item: float(item.get("confidence") or 0.0), reverse=True)
    repair_cap = int(params.get("repair_cap", 512))
    return proposals[:repair_cap]


def normalize_repair_candidate(row: dict[str, Any], proposal: dict[str, Any], index: int) -> dict[str, Any] | None:
    row_id = str(row.get("id"))
    box = bbox(proposal.get("bbox"))
    if box is None:
        return None
    proposal_id = str(proposal.get("id") or f"room_wall_repair_{index:04d}")
    candidate_id = proposal_id if proposal_id.startswith(row_id) else f"{row_id}_room_wall_supported_repair_v18_{index:04d}"
    confidence = round(max(0.01, min(0.99, float(proposal.get("confidence") or 0.0))), 6)
    metadata = proposal.get("metadata") if isinstance(proposal.get("metadata"), dict) else {}
    source = str(proposal.get("source") or proposal.get("proposal_source") or "room_wall_supported_repair_v18")
    payload = {
        "candidate_kind": "room_wall_supported_repair_v18",
        "proposal_source": source,
        "room_type": "room",
        "semantic_type": "room",
        "repair_metadata": metadata,
    }
    return {
        "candidate_id": candidate_id,
        "candidate_contract_version": "detector_candidate_contract_v1",
        "row_id": row_id,
        "family": "space",
        "route": "room_space",
        "candidate_type": "room",
        "bbox": [round(float(v), 3) for v in box],
        "confidence": confidence,
        "payload": payload,
        "source_integrity": integrity(),
        "provenance": {
            "input_source": "room_wall_supported_repair_v18",
            "raw_candidate_id": candidate_id,
            "row_id": row_id,
            "family": "space",
            "route": "room_space",
            "raster_only": True,
            "image": row.get("image"),
        },
        "audit_trace": {
            "stage": "room_wall_supported_repair_v18",
            "proposal_source": source,
            "bbox": [round(float(v), 3) for v in box],
            "confidence": confidence,
            "source_integrity": integrity(),
        },
    }


def room_match_metrics(proposals: list[dict[str, Any]], gold_rooms: list[dict[str, Any]], threshold: float = 0.5) -> tuple[int, list[dict[str, Any]]]:
    used: set[int] = set()
    matched = 0
    misses: list[dict[str, Any]] = []
    for gold_index, gold in enumerate(gold_rooms):
        gb = bbox(gold.get("bbox"))
        if gb is None:
            continue
        best_index = None
        best_iou = 0.0
        for pred_index, pred in enumerate(proposals):
            if pred_index in used:
                continue
            pb = bbox(pred.get("bbox"))
            if pb is None:
                continue
            overlap = iou(pb, gb)
            if overlap > best_iou:
                best_iou = overlap
                best_index = pred_index
        if best_index is not None and best_iou >= threshold:
            used.add(best_index)
            matched += 1
        else:
            misses.append(
                {
                    "gold_index": gold_index,
                    "bbox": gold.get("bbox"),
                    "room_type": gold.get("room_type") or gold.get("semantic_type") or "room",
                    "best_iou": round(best_iou, 6),
                }
            )
    return matched, misses


def dedupe(proposals: list[dict[str, Any]], min_iou: float = 0.98) -> list[dict[str, Any]]:
    ordered = sorted(proposals, key=lambda item: float(item.get("confidence") or 0.0), reverse=True)
    kept: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for item in ordered:
        box = bbox(item.get("bbox"))
        if box is None:
            continue
        key = tuple(int(round(float(v) * 1000.0)) for v in box)
        if key in seen:
            continue
        seen.add(key)
        kept.append(item)
    return kept


def augment_row(row: dict[str, Any], params: dict[str, Any], anchor_priors: list[dict[str, int]], image_cache: dict[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, Any]]:
    out = json.loads(json.dumps(row, ensure_ascii=False))
    arr = load_gray(row, image_cache)
    stream = candidate_stream(out)
    base_rooms = family_boxes(stream, "space")
    raw_repair = room_repair_proposals(out, arr, params, anchor_priors)
    repair = [
        cand
        for idx, proposal in enumerate(raw_repair)
        if (cand := normalize_repair_candidate(out, proposal, idx)) is not None
    ]
    merged = dedupe([*stream, *repair])
    scene = dict(out.get("scene_graph") if isinstance(out.get("scene_graph"), dict) else {})
    scene["candidate_stream"] = merged
    scene["candidate_counts"] = {
        **Counter(str(cand.get("family") or "unknown") for cand in merged),
        "space": sum(1 for cand in merged if cand.get("family") == "space"),
    }
    scene["room_wall_repair_v18"] = {
        "enabled": True,
        "added_candidates": len(repair),
        "boundary_cluster_candidates": sum(1 for cand in repair if str((cand.get("metadata") or {}).get("repair_kind")) == "boundary_cluster_union"),
        "dense_prior_mode": False,
    }
    out["scene_graph"] = scene
    audit = {
        "row_id": row.get("id"),
        "base_room_candidates": len(base_rooms),
        "raw_repair_candidates": len(raw_repair),
        "added_candidates": len(repair),
        "output_candidates": len(merged),
    }
    return out, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--anchor-limit", type=int, default=12)
    parser.add_argument("--proposal-cap", type=int, default=8000)
    parser.add_argument("--repair-cap", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    params, anchor_priors = load_policy(Path(args.policy))
    params = {
        **params,
        "anchor_limit": min(int(params.get("anchor_limit", len(anchor_priors))), int(args.anchor_limit)),
        "cap": int(args.proposal_cap),
        "repair_cap": int(args.repair_cap),
    }
    rows = load_jsonl(Path(args.input))
    if args.limit is not None:
        rows = rows[: args.limit]
    gold = load_gold()
    image_cache: dict[str, np.ndarray] = {}
    out_rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    counts = Counter()
    baseline_match = Counter()
    repaired_match = Counter()
    baseline_misses_sample: list[dict[str, Any]] = []
    repaired_misses_sample: list[dict[str, Any]] = []

    for row in rows:
        out, audit = augment_row(row, params, anchor_priors, image_cache)
        out_rows.append(out)
        audits.append(audit)
        row_id = str(row.get("id"))
        gold_rooms = (gold["rooms"].get(row_id) or {}).get("rooms") or []
        base_proposals = [cand for cand in candidate_stream(row) if cand.get("family") == "space"]
        repaired_proposals = [cand for cand in candidate_stream(out) if cand.get("family") == "space"]
        base_matched, base_misses = room_match_metrics(base_proposals, gold_rooms)
        rep_matched, rep_misses = room_match_metrics(repaired_proposals, gold_rooms)
        counts["rows"] += 1
        counts["baseline_gold"] += len(gold_rooms)
        counts["baseline_matched"] += base_matched
        counts["repaired_matched"] += rep_matched
        counts["base_room_candidates"] += len(base_proposals)
        counts["repaired_room_candidates"] += len(repaired_proposals)
        baseline_match.update([str(row_id)])
        if len(baseline_misses_sample) < 200:
            baseline_misses_sample.extend(base_misses[: max(0, 200 - len(baseline_misses_sample))])
        if len(repaired_misses_sample) < 200:
            repaired_misses_sample.extend(rep_misses[: max(0, 200 - len(repaired_misses_sample))])

    report = {
        "task": "IMG-MOE-V18-REBUILD-001.step_j11_wall_supported_room_repair_for_remaining_room_bbox_misses",
        "input": str(args.input),
        "policy": str(args.policy),
        "output": str(args.output),
        "rows": len(out_rows),
        "counts": dict(counts),
        "base_recall": round(counts["baseline_matched"] / max(counts["baseline_gold"], 1), 6),
        "repaired_recall": round(counts["repaired_matched"] / max(counts["baseline_gold"], 1), 6),
        "recall_delta": round((counts["repaired_matched"] - counts["baseline_matched"]) / max(counts["baseline_gold"], 1), 6),
        "room_candidate_delta": counts["repaired_room_candidates"] - counts["base_room_candidates"],
        "baseline_misses_sample": baseline_misses_sample[:100],
        "repaired_misses_sample": repaired_misses_sample[:100],
        "row_audit_sample": audits[:100],
        "source_integrity": integrity(),
        "gold_loaded_after_inference_for_audit_only": True,
        "gold_used_for_inference": False,
    }

    write_jsonl(Path(args.output), out_rows)
    write_json(Path(args.audit), report)
    print(json.dumps({k: report[k] for k in ["rows", "base_recall", "repaired_recall", "recall_delta", "room_candidate_delta"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
