#!/usr/bin/env python3
"""Audit how CubiCasa assignment supervision can join to the v18 scored cache.

The join is intentionally offline-only. It checks whether host_link labels from
the supervision dataset overlap existing contains_symbol rows in the fixed
scored-row cache. It does not create runtime candidates or relation edges.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUPERVISION = ROOT / "datasets/external_supervision/cubicasa_contains_symbol_assignment_v18"
DEFAULT_IMAGE_ONLY = ROOT / "datasets/image_only_structured_targets_v16"
DEFAULT_CACHE = ROOT / "reports/vlm/relation_graph_scored_rows_cache_v18.jsonl"
DEFAULT_AUDIT = ROOT / "reports/vlm/cubicasa_to_v18_assignment_supervision_audit.json"

_BBOX_RE = re.compile(r"_(\d+)_(\d+)_(\d+)_(\d+)(?:_|$)")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(item) for item in value[:4]]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def bbox_from_candidate_id(candidate_id: str) -> list[float] | None:
    matches = list(_BBOX_RE.finditer(candidate_id))
    if not matches:
        return None
    match = matches[-1]
    x1, y1, x2, y2 = [float(item) for item in match.groups()]
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def iou(left: list[float] | None, right: list[float] | None) -> float:
    if left is None or right is None:
        return 0.0
    ix1 = max(left[0], right[0])
    iy1 = max(left[1], right[1])
    ix2 = min(left[2], right[2])
    iy2 = min(left[3], right[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return inter / max(area(left) + area(right) - inter, 1e-9)


def image_only_index(root: Path) -> tuple[dict[str, str], dict[str, str]]:
    row_to_source: dict[str, str] = {}
    source_to_row: dict[str, str] = {}
    for split in ["train", "dev", "locked", "smoke"]:
        for row in load_jsonl(root / f"{split}.jsonl"):
            row_id = str(row.get("id") or "")
            source_key = str(row.get("source_key") or "")
            if row_id and source_key:
                row_to_source[row_id] = source_key
                source_to_row[source_key] = row_id
    return row_to_source, source_to_row


def supervision_by_row(root: Path) -> tuple[dict[str, list[dict[str, Any]]], Counter]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counts = Counter()
    for split in ["train", "dev", "locked"]:
        for row in load_jsonl(root / f"{split}.jsonl"):
            counts["supervision_rows"] += 1
            counts[f"supervision_rows_{split}"] += 1
            if int(row.get("label") or 0) == 1:
                counts["supervision_positive_rows"] += 1
                counts[f"supervision_positive_rows_{split}"] += 1
            image_only_row_id = str(row.get("image_only_row_id") or "")
            if image_only_row_id:
                counts["supervision_rows_with_image_only_row_id"] += 1
                out[image_only_row_id].append(row)
    return out, counts


def cache_contains_by_row(cache_path: Path, row_to_source: dict[str, str]) -> tuple[dict[str, list[dict[str, Any]]], Counter]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counts = Counter()
    with cache_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            counts["cache_rows"] += 1
            if row.get("relation") != "contains_symbol":
                continue
            counts["cache_contains_symbol_rows"] += 1
            row_id = str(row.get("row_id") or "")
            if row_id in row_to_source:
                counts["cache_contains_symbol_rows_with_source_key"] += 1
                source_box = bbox_from_candidate_id(str(row.get("source_candidate_id") or ""))
                target_box = bbox_from_candidate_id(str(row.get("target_candidate_id") or ""))
                if source_box is not None:
                    counts["cache_contains_rows_with_source_bbox"] += 1
                if target_box is not None:
                    counts["cache_contains_rows_with_target_bbox"] += 1
                item = {
                    "row_id": row_id,
                    "relation_id": row.get("relation_id"),
                    "source_candidate_id": row.get("source_candidate_id"),
                    "target_candidate_id": row.get("target_candidate_id"),
                    "source_box": source_box,
                    "target_box": target_box,
                    "relation_score": row.get("relation_score"),
                    "assignment_score": row.get("assignment_score"),
                }
                out[row_id].append(item)
    counts["cache_row_ids"] = len(out)
    return out, counts


def audit_join(
    supervision: dict[str, list[dict[str, Any]]],
    cache: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    thresholds = [(0.05, 0.05), (0.10, 0.10), (0.25, 0.10), (0.25, 0.25), (0.50, 0.25)]
    counts = Counter()
    threshold_hits = {f"room_iou_ge_{rt}_symbol_iou_ge_{st}": 0 for rt, st in thresholds}
    examples: list[dict[str, Any]] = []
    rows_with_labels = set(supervision)
    rows_with_cache = set(cache)
    for row_id in sorted(rows_with_labels & rows_with_cache):
        labels = [row for row in supervision[row_id] if int(row.get("label") or 0) == 1]
        cache_rows = cache[row_id]
        counts["overlap_row_ids"] += 1
        counts["overlap_positive_labels"] += len(labels)
        counts["overlap_cache_contains_rows"] += len(cache_rows)
        for label in labels:
            room = label.get("room") if isinstance(label.get("room"), dict) else {}
            symbol = label.get("symbol") if isinstance(label.get("symbol"), dict) else {}
            room_box = bbox(room.get("bbox_512"))
            symbol_box = bbox(symbol.get("bbox_512"))
            best: dict[str, Any] | None = None
            for candidate in cache_rows:
                room_score = iou(room_box, candidate.get("source_box"))
                symbol_score = iou(symbol_box, candidate.get("target_box"))
                combined = room_score * symbol_score
                if best is None or combined > best["combined_iou"]:
                    best = {
                        "relation_id": candidate.get("relation_id"),
                        "room_iou": room_score,
                        "symbol_iou": symbol_score,
                        "combined_iou": combined,
                        "relation_score": candidate.get("relation_score"),
                        "assignment_score": candidate.get("assignment_score"),
                    }
            if not best:
                counts["positive_labels_without_cache_candidate"] += 1
                continue
            counts["positive_labels_with_any_cache_candidate"] += 1
            for room_threshold, symbol_threshold in thresholds:
                if best["room_iou"] >= room_threshold and best["symbol_iou"] >= symbol_threshold:
                    threshold_hits[f"room_iou_ge_{room_threshold}_symbol_iou_ge_{symbol_threshold}"] += 1
            if len(examples) < 50:
                examples.append(
                    {
                        "row_id": row_id,
                        "source_key": label.get("source_key"),
                        "relation_key": label.get("relation_key"),
                        "room_type": room.get("room_type"),
                        "symbol_type": symbol.get("symbol_type"),
                        "best_cache_match": {
                            **best,
                            "room_iou": round(float(best["room_iou"]), 6),
                            "symbol_iou": round(float(best["symbol_iou"]), 6),
                            "combined_iou": round(float(best["combined_iou"]), 6),
                        },
                    }
                )
    coverage = {
        name: {
            "matched_positive_labels": hits,
            "coverage": round(hits / max(counts["overlap_positive_labels"], 1), 6),
        }
        for name, hits in threshold_hits.items()
    }
    return {
        "counts": dict(counts),
        "row_id_coverage": {
            "supervision_row_ids": len(rows_with_labels),
            "cache_row_ids": len(rows_with_cache),
            "overlap_row_ids": len(rows_with_labels & rows_with_cache),
            "supervision_without_cache_row_ids": len(rows_with_labels - rows_with_cache),
            "cache_without_supervision_row_ids": len(rows_with_cache - rows_with_labels),
        },
        "positive_geometry_match_coverage": coverage,
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervision-root", default=str(DEFAULT_SUPERVISION))
    parser.add_argument("--image-only-root", default=str(DEFAULT_IMAGE_ONLY))
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--audit-output", default=str(DEFAULT_AUDIT))
    args = parser.parse_args()

    row_to_source, source_to_row = image_only_index(Path(args.image_only_root))
    supervision, supervision_counts = supervision_by_row(Path(args.supervision_root))
    cache, cache_counts = cache_contains_by_row(Path(args.cache), row_to_source)
    join = audit_join(supervision, cache)
    audit = {
        "task": "IMG-MOE-V18-REBUILD-006.step_schema_join_to_existing_scored_cache",
        "supervision_root": args.supervision_root,
        "image_only_root": args.image_only_root,
        "cache": args.cache,
        "image_only_rows": len(row_to_source),
        "image_only_sources": len(source_to_row),
        "supervision_counts": dict(supervision_counts),
        "cache_counts": dict(cache_counts),
        "join": join,
        "join_contract": {
            "method": "offline page/source_key plus 512-space bbox overlap; no direct detector candidate id join is assumed",
            "runtime_candidates_created": False,
            "new_relation_edges_created": False,
            "selection_source_must_remain_existing_scored_cache": True,
        },
        "source_integrity": {
            "source_mode": "offline_supervision_to_scored_cache_audit",
            "model_input": "audit_labels_only_not_runtime_input",
            "svg_candidate_ids_used": False,
            "annotation_geometry_used_at_inference": False,
            "gold_used_for_inference": False,
        },
    }
    write_json(Path(args.audit_output), audit)
    print(
        json.dumps(
            {
                "audit_output": args.audit_output,
                "row_id_coverage": join["row_id_coverage"],
                "positive_geometry_match_coverage": join["positive_geometry_match_coverage"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
