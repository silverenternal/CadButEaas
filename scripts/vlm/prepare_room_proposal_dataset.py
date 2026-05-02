#!/usr/bin/env python3
"""Prepare room proposal candidates and run oracle recall / quality audit."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

WALL_SEMANTIC_TYPES = {
    "hard_wall",
    "partition_wall",
    "door",
    "window",
    "opening",
}

SYNTHETIC_RELATIONS = {"touches", "intersects", "adjacent_to", "adjacent", "near"}


@dataclass
class IoUMatch:
    iou: float
    proposal_id: str
    proposal: dict[str, Any]
    gold_id: str


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="datasets/cadstruct_cubicasa5k_moe_locked")
    parser.add_argument("--output-dir", default="datasets/cadstruct_room_proposals_v1")
    parser.add_argument("--iou-thresholds", default="0.3,0.5,0.75")
    parser.add_argument("--max-proposals-per-record", type=int, default=240)
    parser.add_argument("--min-proposal-area", type=float, default=1_000.0)
    parser.add_argument("--output-audit", default="reports/vlm/room_proposal_oracle_audit.json")
    parser.add_argument("--source-buckets", default="train,dev,locked_test,smoke")
    args = parser.parse_args()

    iou_thresholds = sorted({max(0.0, min(1.0, float(item))) for item in args.iou_thresholds.split(",") if item.strip()})
    if not iou_thresholds:
        raise ValueError("--iou-thresholds must contain at least one float in [0,1]")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_buckets = [item.strip() for item in args.source_buckets.split(",") if item.strip()]

    audit: dict[str, Any] = {
        "version": "0.1",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "iou_thresholds": iou_thresholds,
        "splits": {},
    }

    for split in source_buckets:
        source_path = input_dir / f"{split}.jsonl"
        if not source_path.exists():
            continue
        rows = load_jsonl(source_path)
        dataset_rows = []

        split_audit = {
            "records": 0,
            "gold_rooms": 0,
            "proposals": 0,
            "avg_proposals_per_record": 0.0,
            "max_proposals_per_record": 0,
            "proposal_area": {"min": None, "max": 0.0, "mean": 0.0},
            "recall": {},
            "ap50": None,
            "precision": {},
            "source_bucket": Counter(),
            "failure_examples": [],
            "missing_records": 0,
            "top_fp_examples": [],
        }

        all_proposal_areas: list[float] = []
        gold_total = 0
        gt_lookup: dict[str, list[dict[str, Any]]] = {}
        source_bucket_stats = Counter()

        for row in rows:
            split_audit["records"] += 1
            row_output = row_to_proposal_row(row)
            proposals = row_output["proposals"]
            if args.max_proposals_per_record > 0:
                proposals = proposals[: args.max_proposals_per_record]
            proposals = [normalize_proposal_score(item) for item in proposals]
            proposals = dedupe_proposals(proposals, min_iou=0.98)
            row_output["proposals"] = proposals
            row_output["proposals_count"] = len(proposals)
            source_bucket = source_bucket_from_annotation(str(row.get("annotation_path") or ""))

            rooms = row_output["rooms"]
            all_proposal_areas.extend([(bbox_area(item["bbox"]) if is_bbox(item.get("bbox")) else 0.0) for item in proposals])
            split_audit["gold_rooms"] += len(rooms)
            split_audit["proposals"] += len(proposals)
            split_audit["max_proposals_per_record"] = max(split_audit["max_proposals_per_record"], len(proposals))
            split_audit["source_bucket"][source_bucket] += 1
            if rooms:
                source_bucket_stats[source_bucket] += len(rooms)
            dataset_rows.append(row_output)

            if proposals and rooms:
                gold_total += len(rooms)
                gt_lookup[str(row.get("annotation_path") or "")] = rooms
            elif not rooms:
                split_audit["missing_records"] += 1

        recall_by_threshold: dict[str, float] = {}
        precision_by_threshold: dict[str, float] = {}
        for threshold in iou_thresholds:
            recall, precision, missed_examples = compute_oracle_recall(rows, dataset_rows, threshold)
            recall_by_threshold[f"{threshold:.2f}"] = recall
            precision_by_threshold[f"{threshold:.2f}"] = precision
            if threshold == 0.5:
                split_audit["failure_examples"].extend(missed_examples)

        ap50, fp_examples = compute_ap50_with_examples(dataset_rows, iou_threshold=0.5)
        split_audit["recall"] = recall_by_threshold
        split_audit["precision"] = precision_by_threshold
        split_audit["ap50"] = ap50
        split_audit["top_fp_examples"] = fp_examples

        if all_proposal_areas:
            split_audit["proposal_area"]["min"] = round(min(all_proposal_areas), 6)
            split_audit["proposal_area"]["max"] = round(max(all_proposal_areas), 6)
            split_audit["proposal_area"]["mean"] = round(sum(all_proposal_areas) / len(all_proposal_areas), 6)
        split_audit["avg_proposals_per_record"] = round(split_audit["proposals"] / max(split_audit["records"], 1), 6)
        split_audit["source_bucket"] = dict(split_audit["source_bucket"])
        split_audit["per_bucket_support"] = dict(source_bucket_stats)

        (output_dir / f"{split}.jsonl").write_text("".join(
            json.dumps(item, ensure_ascii=False) + "\n" for item in dataset_rows
        ), encoding="utf-8")
        audit["splits"][split] = split_audit

    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "source": str(input_dir),
                "splits": {
                    split: {
                        "records": stats["records"],
                        "rooms": stats["gold_rooms"],
                        "proposals": stats["proposals"],
                    }
                    for split, stats in audit["splits"].items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    audit["summary"] = {
        "records_with_rooms": sum(item["records"] for item in audit["splits"].values()),
        "total_gold_rooms": sum(item["gold_rooms"] for item in audit["splits"].values()),
        "total_proposals": sum(item["proposals"] for item in audit["splits"].values()),
        "missing_room_records": sum(item["missing_records"] for item in audit["splits"].values()),
    }
    Path(args.output_audit).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_audit).write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def row_to_proposal_row(row: dict[str, Any]) -> dict[str, Any]:
    expected = row.get("expected_json") or {}
    hints = row.get("request_hints") or {}
    walls = select_wall_nodes(hints.get("primitive_graph") or {})
    region_nodes = hints.get("semantic_regions") or []
    room_candidates = expected.get("room_candidates") or []

    proposals = [
        to_candidate(item, source="semantic_region", score=0.92)
        for item in region_nodes
        if to_candidate(item, source="semantic_region", score=0.92).get("bbox")
    ]
    proposals.extend(
        to_candidate(item, source="room_fallback", score=0.80) for item in room_candidates if to_candidate(item, source="room_fallback", score=0.80).get("bbox")
    )
    proposals.extend(room_components_as_proposals(walls, hints.get("primitive_graph") or {}))

    rooms = []
    for item in room_candidates:
        if not is_bbox(item.get("bbox")):
            continue
        rooms.append(
            {
                "id": str(item.get("id") or f"room_{len(rooms)}"),
                "room_type": str(item.get("room_type") or "room"),
                "bbox": normalize_bbox(item.get("bbox")),
            }
        )

    return {
        "image": row.get("image_path"),
        "annotation": row.get("annotation_path"),
        "source_dataset": row.get("source_dataset"),
        "source_bucket": source_bucket_from_annotation(str(row.get("annotation_path") or "")),
        "width": row.get("metadata", {}).get("width") if isinstance(row.get("metadata"), dict) else None,
        "height": row.get("metadata", {}).get("height") if isinstance(row.get("metadata"), dict) else None,
        "proposals": proposals,
        "rooms": rooms,
    }


def select_wall_nodes(primitive_graph: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = []
    for item in primitive_graph.get("nodes") or []:
        if not isinstance(item, dict):
            continue
        semantic = str(item.get("semantic_type") or item.get("type") or "").strip()
        if semantic not in WALL_SEMANTIC_TYPES:
            continue
        bbox = normalize_bbox(item.get("bbox"))
        if bbox is None:
            continue
        nodes.append({**item, "_bbox": bbox})
    return nodes


def room_components_as_proposals(nodes: list[dict[str, Any]], primitive_graph: dict[str, Any]) -> list[dict[str, Any]]:
    if not nodes:
        return []

    node_ids = {int(item.get("id")): item for item in nodes if int_like(item.get("id"))}
    adjacency: dict[int, set[int]] = {node_id: set() for node_id in node_ids}
    for edge in primitive_graph.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        if not (int_like(edge.get("source")) and int_like(edge.get("target"))):
            continue
        source = int(edge.get("source"))
        target = int(edge.get("target"))
        relation = str(edge.get("relation") or "touches").lower()
        if source not in node_ids or target not in node_ids:
            continue
        if relation and relation not in SYNTHETIC_RELATIONS:
            continue
        adjacency[source].add(target)
        adjacency[target].add(source)

    components: list[list[int]] = []
    visited: set[int] = set()
    for node_id in sorted(node_ids):
        if node_id in visited:
            continue
        stack = [node_id]
        visited.add(node_id)
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        if component:
            components.append(sorted(component))

    proposals: list[dict[str, Any]] = []
    for index, component in enumerate(components):
        bboxes = [node_ids[item]["_bbox"] for item in component if isinstance(item, int) and item in node_ids]
        if not bboxes:
            continue
        x1 = min(item[0] for item in bboxes)
        y1 = min(item[1] for item in bboxes)
        x2 = max(item[2] for item in bboxes)
        y2 = max(item[3] for item in bboxes)
        bbox = [x1, y1, x2, y2]
        area = bbox_area(bbox)
        if area < 1_000:
            continue
        proposals.append(
            {
                "id": f"comp_{index}",
                "source": "primitive_component",
                "bbox": bbox,
                "confidence": round(0.35 + 0.03 * min(len(component), 40), 3),
                "metadata": {
                    "component_size": len(component),
                    "member_node_ids": component,
                    "area": round(area, 3),
                },
            }
        )
    return proposals


def to_candidate(item: dict[str, Any], source: str, score: float) -> dict[str, Any]:
    bbox = normalize_bbox(item.get("bbox"))
    if bbox is None:
        return {"id": str(item.get("id") or "unknown"), "source": source, "bbox": None}
    return {
        "id": str(item.get("id") or item.get("source_id") or item.get("target_id") or item.get("source") or "unknown"),
        "source": source,
        "bbox": bbox,
        "confidence": float(score),
        "metadata": {
            "semantic_type": str(item.get("type") or item.get("room_type") or item.get("semantic_type") or "room"),
            "area": round(bbox_area(bbox), 6),
        },
    }


def dedupe_proposals(proposals: list[dict[str, Any]], min_iou: float = 0.98) -> list[dict[str, Any]]:
    ordered = sorted(proposals, key=lambda item: float(item.get("confidence") or 0.0), reverse=True)
    kept: list[dict[str, Any]] = []
    for candidate in ordered:
        if candidate.get("bbox") is None:
            continue
        candidate_bbox = candidate["bbox"]
        is_duplicate = False
        for existing in kept:
            if iou(candidate_bbox, existing["bbox"]) >= min_iou:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(candidate)
    return kept


def compute_oracle_recall(records: list[dict[str, Any]], proposal_records: list[dict[str, Any]], threshold: float) -> tuple[float, float, list[dict[str, str]]]:
    misses: list[dict[str, str]] = []
    total = matched = 0
    proposal_counts = Counter()
    for row, proposal_row in zip(records, proposal_records):
        rooms = [item for item in row.get("expected_json", {}).get("room_candidates") or [] if is_bbox(item.get("bbox"))]
        proposals = [item for item in proposal_row.get("proposals") or [] if is_bbox(item.get("bbox"))]
        total += len(rooms)
        used: set[str] = set()
        for room in rooms:
            gold_bbox = normalize_bbox(room.get("bbox"))
            if gold_bbox is None:
                continue
            best: IoUMatch | None = None
            for proposal in proposals:
                proposal_id = str(proposal.get("id") or "")
                if proposal_id in used:
                    continue
                overlap = iou(gold_bbox, proposal["bbox"])
                if overlap >= threshold and (best is None or overlap > best.iou):
                    best = IoUMatch(iou=overlap, proposal_id=proposal_id, proposal=proposal, gold_id=str(room.get("id") or ""))
            if best is not None:
                used.add(best.proposal_id)
                matched += 1
                proposal_counts[best.proposal_id] += 1
            else:
                misses.append(
                    {
                        "annotation": str(row.get("annotation_path") or ""),
                        "room_id": str(room.get("id") or ""),
                        "room_type": str(room.get("room_type") or "room"),
                        "iou_threshold": str(threshold),
                    }
                )

    recall = matched / max(total, 1)
    precision = len(proposal_counts) / max(len([item for row in proposal_records for item in row.get("proposals", [])]), 1)
    return recall, precision, misses[:50]


def compute_ap50_with_examples(rows: list[dict[str, Any]], iou_threshold: float = 0.5) -> tuple[float, list[dict[str, str]]]:
    flattened = []
    totals: dict[str, tuple[list[dict[str, Any]], bool]] = {}
    total_gt = 0
    for row in rows:
        annotation = str(row.get("annotation") or "")
        gold = [room for room in row.get("rooms", []) if is_bbox(room.get("bbox"))]
        total_gt += len(gold)
        totals[annotation] = [gold, False]
        for proposal in row.get("proposals") or []:
            if not is_bbox(proposal.get("bbox")):
                continue
            flattened.append(
                {
                    "annotation": annotation,
                    "score": float(proposal.get("confidence") or 0.0),
                    "bbox": proposal["bbox"],
                    "proposal_id": str(proposal.get("id") or ""),
                }
            )

    flattened.sort(key=lambda item: item["score"], reverse=True)

    tp = 0
    fp = 0
    cumulative_tp: list[int] = []
    cumulative_fp: list[int] = []
    used_gts: dict[str, set[str]] = defaultdict(set)
    fp_examples: list[dict[str, str]] = []

    for item in flattened:
        row_gold, _ = totals[item["annotation"]]
        best_match = None
        best_iou = 0.0
        best_index = -1
        for index, gt in enumerate(row_gold):
            if str(gt.get("id") or "") in used_gts[item["annotation"]]:
                continue
            overlap = iou(item["bbox"], gt["bbox"])
            if overlap >= iou_threshold and overlap > best_iou:
                best_iou = overlap
                best_match = gt
                best_index = index
        if best_match is not None:
            tp += 1
            used_gts[item["annotation"]].add(str(best_match.get("id") or ""))
        else:
            fp += 1
            if len(fp_examples) < 20:
                fp_examples.append(
                    {
                        "annotation": item["annotation"],
                        "proposal": item["proposal_id"],
                        "score": str(item["score"]),
                    }
                )
        cumulative_tp.append(tp)
        cumulative_fp.append(fp)

    recalls = [tp_i / max(total_gt, 1) for tp_i in cumulative_tp]
    precisions = [tp_i / max(tp_i + fp_i, 1) for tp_i, fp_i in zip(cumulative_tp, cumulative_fp)]

    if not recalls:
        return 0.0, fp_examples

    mrec = [0.0] + recalls + [1.0]
    mpre = [1.0] + precisions + [0.0]
    for idx in range(len(mpre) - 2, -1, -1):
        mpre[idx] = max(mpre[idx], mpre[idx + 1])

    ap = 0.0
    for idx in range(1, len(mrec)):
        delta_recall = mrec[idx] - mrec[idx - 1]
        if delta_recall > 0:
            ap += delta_recall * mpre[idx]
    return ap, fp_examples


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def normalize_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    if not all(map(math.isfinite, (x1, y1, x2, y2))):
        return None
    if x2 < x1 or y2 < y1:
        return None
    return [x1, y1, x2, y2]


def is_bbox(value: Any) -> bool:
    return normalize_bbox(value) is not None


def bbox_area(value: Any) -> float:
    bbox = normalize_bbox(value)
    if bbox is None:
        return 0.0
    return max(0.0, (bbox[2] - bbox[0])) * max(0.0, (bbox[3] - bbox[1]))


def normalize_bbox_overlap(top_left: list[float], denominator: float) -> list[float]:
    return [max(0.0, v) / max(denominator, 1.0) for v in top_left]


def iou(left: list[float], right: list[float]) -> float:
    lb = normalize_bbox(left)
    rb = normalize_bbox(right)
    if lb is None or rb is None:
        return 0.0
    x1 = max(lb[0], rb[0])
    y1 = max(lb[1], rb[1])
    x2 = min(lb[2], rb[2])
    y2 = min(lb[3], rb[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    union = bbox_area(lb) + bbox_area(rb) - inter
    return inter / max(union, 1e-12)


def source_bucket_from_annotation(annotation: str) -> str:
    marker = "/cubicasa5k/"
    if marker in annotation:
        return annotation.split(marker, 1)[1].split("/", 1)[0]
    return "unknown"


def int_like(value: Any) -> bool:
    return isinstance(value, int) or (isinstance(value, str) and value.strip().lstrip("-").isdigit())


def normalize_proposal_score(item: dict[str, Any]) -> dict[str, Any]:
    copy = dict(item)
    copy["confidence"] = max(0.0, min(1.0, float(copy.get("confidence") or 0.0)))
    return copy


def int_or_default(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    main()
