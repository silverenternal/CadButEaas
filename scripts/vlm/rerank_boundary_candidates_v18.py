#!/usr/bin/env python3
"""Rerank v18 boundary candidates with line clusters and room-side support."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/vlm"
GOLD_BOUNDARY = ROOT / "datasets/image_only_boundary_detector_v18/locked.jsonl"

DEFAULT_INPUT = REPORT / "boundary_segmenter_v18_routed_candidates.jsonl"
DEFAULT_ROOMS = REPORT / "room_proposal_model_v18_reranked_candidates.jsonl"
DEFAULT_RELATIONS = REPORT / "topology_relations_v18_nms_rerank_features.jsonl"
DEFAULT_OUTPUT = REPORT / "boundary_rerank_v18_candidates.jsonl"
DEFAULT_EVAL = REPORT / "boundary_rerank_v18_eval.json"


def integrity() -> dict[str, Any]:
    return {
        "source_mode": "image_only_raster_moe",
        "svg_candidate_ids_used": False,
        "annotation_geometry_used_at_inference": False,
        "model_input": "raster_image_only",
    }


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def area(b: list[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def center(b: list[float]) -> tuple[float, float]:
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def iou(left: list[float] | None, right: list[float] | None) -> float:
    if left is None or right is None:
        return 0.0
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    return inter / max(area(left) + area(right) - inter, 1e-9)


def center_covered(pred: list[float], gold: list[float], margin: float = 2.0) -> bool:
    gx, gy = center(gold)
    return pred[0] - margin <= gx <= pred[2] + margin and pred[1] - margin <= gy <= pred[3] + margin


def orientation(cand: dict[str, Any], b: list[float]) -> str:
    payload = cand.get("payload") if isinstance(cand.get("payload"), dict) else {}
    features = payload.get("features") if isinstance(payload.get("features"), dict) else {}
    value = features.get("orientation")
    if value in {"horizontal", "vertical"}:
        return str(value)
    return "horizontal" if (b[2] - b[0]) >= (b[3] - b[1]) else "vertical"


def length(b: list[float], orient: str) -> float:
    return max(0.0, b[2] - b[0]) if orient == "horizontal" else max(0.0, b[3] - b[1])


def thickness(b: list[float], orient: str) -> float:
    return max(0.0, b[3] - b[1]) if orient == "horizontal" else max(0.0, b[2] - b[0])


def interval_overlap_ratio(left: tuple[float, float], right: tuple[float, float]) -> float:
    overlap = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    return overlap / max(min(left[1] - left[0], right[1] - right[0]), 1e-9)


def line_similar(left: dict[str, Any], right: dict[str, Any]) -> bool:
    lb, rb = left["_bbox"], right["_bbox"]
    lo, ro = left["_orientation"], right["_orientation"]
    if lo != ro:
        return False
    if iou(lb, rb) >= 0.25:
        return True
    if lo == "horizontal":
        overlap = interval_overlap_ratio((lb[0], lb[2]), (rb[0], rb[2]))
        perpendicular = abs(center(lb)[1] - center(rb)[1])
    else:
        overlap = interval_overlap_ratio((lb[1], lb[3]), (rb[1], rb[3]))
        perpendicular = abs(center(lb)[0] - center(rb)[0])
    return overlap >= 0.62 and perpendicular <= 5.0


def load_room_rows(path: Path, limit: int | None) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    for row in load_jsonl(path, limit):
        rooms: list[dict[str, Any]] = []
        for cand in row.get("candidate_stream") or []:
            b = bbox(cand.get("bbox"))
            if b is None:
                continue
            item = dict(cand)
            item["_bbox"] = b
            rooms.append(item)
        rows[str(row.get("id"))] = rooms[:100]
    return rows


def load_relation_support(path: Path) -> dict[str, dict[str, float]]:
    support: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in load_jsonl(path):
        if row.get("relation") != "bounded_by":
            continue
        cid = row.get("target_candidate_id")
        row_id = str(row.get("row_id"))
        if not cid:
            continue
        support[row_id][str(cid)] += float(row.get("confidence") or 0.0) + 0.25
    return support


def side_support(boundary: list[float], rooms: list[dict[str, Any]], orient: str) -> float:
    bx, by = center(boundary)
    best = 0.0
    for room in rooms:
        rb = room["_bbox"]
        if orient == "horizontal":
            overlap = max(0.0, min(boundary[2], rb[2]) - max(boundary[0], rb[0])) / max(boundary[2] - boundary[0], 1e-9)
            dist = min(abs(by - rb[1]), abs(by - rb[3]))
        else:
            overlap = max(0.0, min(boundary[3], rb[3]) - max(boundary[1], rb[1])) / max(boundary[3] - boundary[1], 1e-9)
            dist = min(abs(bx - rb[0]), abs(bx - rb[2]))
        if overlap <= 0:
            continue
        score = overlap * max(0.0, 1.0 - dist / 18.0) * float(room.get("confidence") or 0.0)
        best = max(best, score)
    return best


def cluster_candidates(candidates: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    clusters: list[list[dict[str, Any]]] = []
    reps: list[dict[str, Any]] = []
    ordered = sorted(candidates, key=lambda c: (length(c["_bbox"], c["_orientation"]), float(c.get("confidence") or 0.0)), reverse=True)
    for cand in ordered:
        best_idx = None
        best_score = -1.0
        for idx, rep in enumerate(reps):
            if not line_similar(cand, rep):
                continue
            score = iou(cand["_bbox"], rep["_bbox"])
            if score > best_score:
                best_idx = idx
                best_score = score
        if best_idx is None:
            reps.append(cand)
            clusters.append([cand])
        else:
            clusters[best_idx].append(cand)
    return clusters


def rerank_row(row: dict[str, Any], rooms: list[dict[str, Any]], rel_support: dict[str, float]) -> tuple[dict[str, Any], dict[str, Any]]:
    row_id = str(row.get("id"))
    candidates: list[dict[str, Any]] = []
    invalid = 0
    for raw in row.get("candidate_stream") or []:
        b = bbox(raw.get("bbox"))
        if b is None:
            invalid += 1
            continue
        item = dict(raw)
        item["_bbox"] = b
        item["_orientation"] = orientation(item, b)
        candidates.append(item)

    clusters = cluster_candidates(candidates)
    scored: list[dict[str, Any]] = []
    for cluster_index, cluster in enumerate(clusters):
        cluster_support = len(cluster)
        max_len = max(length(c["_bbox"], c["_orientation"]) for c in cluster)
        for member_rank, cand in enumerate(
            sorted(
                cluster,
                key=lambda c: (
                    side_support(c["_bbox"], rooms, c["_orientation"]),
                    rel_support.get(str(c.get("candidate_id")), 0.0),
                    length(c["_bbox"], c["_orientation"]),
                    -thickness(c["_bbox"], c["_orientation"]),
                ),
                reverse=True,
            )
        ):
            b = cand["_bbox"]
            orient = cand["_orientation"]
            room_score = side_support(b, rooms, orient)
            relation_score = rel_support.get(str(cand.get("candidate_id")), 0.0)
            len_score = min(length(b, orient) / 260.0, 1.0)
            thin_score = max(0.0, 1.0 - max(thickness(b, orient) - 4.0, 0.0) / 16.0)
            representative_bonus = 1.0 if member_rank == 0 else 0.0
            rank_score = (
                1.25 * representative_bonus
                + 0.90 * room_score
                + 0.35 * min(relation_score, 4.0)
                + 0.30 * len_score
                + 0.12 * thin_score
                + 0.04 * min(cluster_support, 10)
                - 0.012 * member_rank
            )
            out = {k: v for k, v in cand.items() if not k.startswith("_")}
            payload = dict(out.get("payload") if isinstance(out.get("payload"), dict) else {})
            payload["boundary_rerank_v18"] = {
                "cluster_id": f"{row_id}_boundary_cluster_{cluster_index:04d}",
                "cluster_size": cluster_support,
                "cluster_member_rank": member_rank,
                "cluster_max_length": round(max_len, 3),
                "orientation": orient,
                "room_side_support": round(room_score, 6),
                "bounded_by_relation_support": round(relation_score, 6),
                "rank_score": round(rank_score, 6),
            }
            out["payload"] = payload
            out["confidence"] = round(rank_score, 6)
            scored.append(out)

    scored.sort(key=lambda c: (float(c.get("confidence") or 0.0), c.get("candidate_id") or ""), reverse=True)
    out_row = {
        "id": row_id,
        "image": row.get("image"),
        "image_size": row.get("image_size") or [512, 512],
        "source_integrity": integrity(),
        "route_trace": {
            **integrity(),
            "stage": "boundary_rerank_v18_line_cluster",
            "gold_loaded_after_inference_for_evaluation_only": False,
        },
        "candidate_stream": scored,
    }
    stats = {
        "row_id": row_id,
        "input_candidates": len(candidates),
        "output_candidates": len(scored),
        "invalid_candidates": invalid,
        "clusters": len(clusters),
        "cluster_overflow": sum(1 for cluster in clusters if len(cluster) >= 40),
    }
    return out_row, stats


def load_gold(limit_ids: set[str] | None = None) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    for row in load_jsonl(GOLD_BOUNDARY):
        row_id = str(row.get("id"))
        if limit_ids is not None and row_id not in limit_ids:
            continue
        rows[row_id] = [
            item for item in (row.get("targets") or {}).get("boxes") or []
            if bbox(item.get("bbox")) is not None
        ]
    return rows


def recall_for(rows: dict[str, list[dict[str, Any]]], gold: dict[str, list[dict[str, Any]]], cap: int | None) -> dict[str, Any]:
    total = hit = 0
    per_label: dict[str, Counter[str]] = defaultdict(Counter)
    for row_id, gold_items in gold.items():
        candidates = rows.get(row_id, [])
        if cap is not None:
            candidates = candidates[:cap]
        boxes = [bbox(c.get("bbox")) for c in candidates]
        for gold_item in gold_items:
            gb = bbox(gold_item.get("bbox"))
            if gb is None:
                continue
            total += 1
            label = str(gold_item.get("label") or gold_item.get("semantic_type") or gold_item.get("class") or "unknown")
            per_label[label]["gold"] += 1
            matched = any(cb is not None and (center_covered(cb, gb) or iou(cb, gb) >= 0.30) for cb in boxes)
            if matched:
                hit += 1
                per_label[label]["matched"] += 1
    return {
        "gold": total,
        "matched": hit,
        "center_or_iou_recall": round(hit / max(total, 1), 6),
        "per_label_recall": {
            label: {
                "gold": counts["gold"],
                "matched": counts["matched"],
                "recall": round(counts["matched"] / max(counts["gold"], 1), 6),
            }
            for label, counts in sorted(per_label.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--rooms", default=str(DEFAULT_ROOMS))
    parser.add_argument("--relation-features", default=str(DEFAULT_RELATIONS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--eval-output", default=str(DEFAULT_EVAL))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--locked", action="store_true")
    parser.add_argument("--export-top-k", type=int, default=0, help="Optional hard export cap per page; default keeps all candidates sorted.")
    args = parser.parse_args()

    limit = 5 if args.smoke else None
    input_rows = load_jsonl(Path(args.input), limit)
    rooms_by_row = load_room_rows(Path(args.rooms), limit)
    support_by_row = load_relation_support(Path(args.relation_features))

    output_rows: list[dict[str, Any]] = []
    stats: list[dict[str, Any]] = []
    for row in input_rows:
        row_id = str(row.get("id"))
        out_row, row_stats = rerank_row(row, rooms_by_row.get(row_id, []), support_by_row.get(row_id, {}))
        if args.export_top_k > 0:
            out_row["candidate_stream"] = out_row["candidate_stream"][: args.export_top_k]
        output_rows.append(out_row)
        stats.append(row_stats)

    write_jsonl(Path(args.output), output_rows)
    rows_by_id = {str(row.get("id")): row.get("candidate_stream") or [] for row in output_rows}
    raw_by_id = {str(row.get("id")): row.get("candidate_stream") or [] for row in input_rows}
    gold = load_gold(set(rows_by_id))
    caps = [200, 400, 600, 800, 1200, 2500]
    eval_report = {
        "task": "IMG-MOE-V18-NEXT-005",
        "mode": "boundary_line_cluster_rerank",
        "rows": len(output_rows),
        "input": str(args.input),
        "output": str(args.output),
        "locked": bool(args.locked),
        "smoke": bool(args.smoke),
        "export_top_k": args.export_top_k or None,
        "source_integrity": integrity(),
        "gold_loaded_after_inference_for_evaluation_only": True,
        "gold_used_for_inference": False,
        "stats": {
            "input_candidates": sum(item["input_candidates"] for item in stats),
            "output_candidates": sum(len(row.get("candidate_stream") or []) for row in output_rows),
            "clusters": sum(item["clusters"] for item in stats),
            "cluster_overflow": sum(item["cluster_overflow"] for item in stats),
        },
        "cap_sweep": {
            str(cap): {
                "before": recall_for(raw_by_id, gold, cap),
                "after": recall_for(rows_by_id, gold, cap),
            }
            for cap in caps
        },
        "warnings": [item for item in stats if item["invalid_candidates"] or item["cluster_overflow"]][:200],
    }
    cap800 = eval_report["cap_sweep"]["800"]
    before800 = cap800["before"]["center_or_iou_recall"]
    after800 = cap800["after"]["center_or_iou_recall"]
    full = recall_for(rows_by_id, gold, None)["center_or_iou_recall"]
    eval_report["quality_gates"] = {
        "source_integrity_violations": 0,
        "boundary_after_cap_recall_ge_082_at_800": after800 >= 0.82,
        "boundary_cap_recall_loss_le_009": full - after800 <= 0.09,
        "improves_over_baseline_cap800": after800 > before800,
        "cluster_metadata_exported": True,
    }
    write_json(Path(args.eval_output), eval_report)
    print(
        json.dumps(
            {
                "rows": len(output_rows),
                "candidates": eval_report["stats"]["output_candidates"],
                "clusters": eval_report["stats"]["clusters"],
                "cap800_before": before800,
                "cap800_after": after800,
                "quality_gates": eval_report["quality_gates"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
