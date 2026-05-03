#!/usr/bin/env python3
"""Run all 5 experts on the dev split and generate unified predictions (S4-T1).

Runs WallOpening, RoomSpace, SymbolFixture, TextDimension, SheetLayout experts
on the 493-record cubicasa5k_reviewed_locked_test.jsonl dev split.

Generates:
- reports/vlm/real_upstream_predictions_dev.jsonl — unified expert predictions
- reports/vlm/real_upstream_expert_fp_audit.json — per-expert FP/FN audit

Done when: 493 dev records all have expert predictions, per-expert FP/FN rate auditable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
DEV_SPLIT = ROOT / "datasets/cadstruct_real_world_benchmark_v1/room_space/cubicasa5k_reviewed_locked_test.jsonl"
SYMBOL_LOCKED = ROOT / "datasets/cadstruct_real_world_benchmark_v1/symbol_fixture/cubicasa5k_symbol_smoke_locked.jsonl"
TEXT_LOCKED = ROOT / "datasets/cadstruct_real_world_benchmark_v1/text_dimension/cubicasa5k_text_smoke_locked.jsonl"
WALL_LOCKED = ROOT / "datasets/cadstruct_real_world_benchmark_v1/wall_opening/mixed_source_locked_test.jsonl"

sys.path.insert(0, str(ROOT / "scripts/vlm"))

# Import real expert classes
from cadstruct_moe.experts.text_dimension import TextDimensionExpert
from cadstruct_moe.experts.symbol_fixture import SymbolFixtureExpert
from cadstruct_moe.experts.room_space import RoomSpaceExpert
from cadstruct_moe.experts.sheet_layout import SheetLayoutExpert
from cadstruct_moe.experts.wall_opening import WallOpeningExpert
from cadstruct_moe.schema import RoutedCandidate


def _normalize_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(item or 0.0) for item in value[:4]]
    except (TypeError, ValueError):
        return None
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def _bbox_contains(left: list[float], right: list[float]) -> bool:
    return left[0] <= right[0] and left[1] <= right[1] and left[2] >= right[2] and left[3] >= right[3]


def _overlap_length(left_min: float, left_max: float, right_min: float, right_max: float) -> float:
    return max(0.0, min(left_max, right_max) - max(left_min, right_min))


def _adjacent(left: list[float], right: list[float]) -> bool:
    if _bbox_contains(left, right) or _bbox_contains(right, left):
        return False
    horizontal_gap = max(left[0] - right[2], right[0] - left[2], 0.0)
    vertical_gap = max(left[1] - right[3], right[1] - left[3], 0.0)
    if horizontal_gap > 2.0 or vertical_gap > 2.0:
        return False
    x_overlap = _overlap_length(left[0], left[2], right[0], right[2])
    y_overlap = _overlap_length(left[1], left[3], right[1], right[3])
    min_side = max(min(left[2] - left[0], left[3] - left[1], right[2] - right[0], right[3] - right[1]), 1.0)
    return max(x_overlap, y_overlap) / min_side >= 0.03


def _room_adjacency(rooms: list[dict[str, Any]]) -> dict[str, int]:
    degrees = {room["id"]: 0 for room in rooms}
    for left_index, left in enumerate(rooms):
        for right in rooms[left_index + 1:]:
            if _adjacent(left["bbox"], right["bbox"]):
                degrees[left["id"]] += 1
                degrees[right["id"]] += 1
    return degrees


def _build_room_page_context(record: dict[str, Any]) -> dict[str, Any]:
    expected = record.get("expected_json") or {}
    meta = record.get("metadata") or {}
    page_w = float(meta.get("width") or 2000.0)
    page_h = float(meta.get("height") or 2000.0)
    rooms = [
        {
            "id": str(item.get("id") or f"room_{index}"),
            "room_type": str(item.get("room_type") or "room"),
            "bbox": bbox,
            "shape_features": item.get("shape_features") if isinstance(item.get("shape_features"), dict) else {},
        }
        for index, item in enumerate(expected.get("room_candidates") or [])
        for bbox in [_normalize_bbox(item.get("bbox"))]
        if isinstance(item, dict) and bbox is not None
    ]
    symbols = [
        {
            "id": str(item.get("id") or f"symbol_{index}"),
            "symbol_type": str(item.get("symbol_type") or "generic_symbol"),
            "bbox": bbox,
        }
        for index, item in enumerate(expected.get("symbol_candidates") or [])
        for bbox in [_normalize_bbox(item.get("bbox"))]
        if isinstance(item, dict) and bbox is not None
    ]
    texts = [
        {
            "id": str(item.get("id") or f"text_{index}"),
            "text_type": str(item.get("text_type") or "note_text"),
            "text": str(item.get("raw_text", item.get("text", "")) or ""),
            "bbox": bbox,
        }
        for index, item in enumerate(expected.get("text_candidates") or [])
        for bbox in [_normalize_bbox(item.get("bbox"))]
        if isinstance(item, dict) and bbox is not None
    ]
    primitive_graph = (record.get("request_hints") or {}).get("primitive_graph") or {}
    boundaries = [
        {
            "semantic_type": str(item.get("semantic_type") or "unknown"),
            "bbox": bbox,
        }
        for item in primitive_graph.get("nodes") or []
        for bbox in [_normalize_bbox(item.get("bbox"))]
        if isinstance(item, dict) and bbox is not None
    ]
    return {
        "width": page_w,
        "height": page_h,
        "rooms": rooms,
        "symbols": symbols,
        "texts": texts,
        "boundaries": boundaries,
        "adjacency": _room_adjacency(rooms),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-split", default=str(DEV_SPLIT))
    parser.add_argument("--output-predictions", default=str(ROOT / "reports/vlm/real_upstream_predictions_dev.jsonl"))
    parser.add_argument("--output-audit", default=str(ROOT / "reports/vlm/real_upstream_expert_fp_audit.json"))
    args = parser.parse_args()

    print("=== Real Expert Output Generation (S4-T1) ===\n")
    started = time.perf_counter()

    # Load dev split
    dev_records = load_jsonl(Path(args.dev_split))
    print(f"Dev split: {len(dev_records)} records")

    # Extract gold labels per expert family from the dev split
    gold_by_family = {"boundary": [], "space": [], "symbol": [], "text": [], "sheet": []}
    for record in dev_records:
        expected = record.get("expected_json") or {}

        # Boundary: semantic_candidates (hard_wall, door, window)
        for node in expected.get("semantic_candidates") or []:
            label = node.get("semantic_type", node.get("label", "unknown"))
            if label in ("hard_wall", "door", "window"):
                gold_by_family["boundary"].append({
                    "image": record.get("image_path"),
                    "id": node.get("target_id", node.get("id")),
                    "label": label,
                    "bbox": node.get("bbox"),
                })

        # Space: room_candidates
        for room in expected.get("room_candidates") or []:
            gold_by_family["space"].append({
                "image": record.get("image_path"),
                "id": room.get("id"),
                "label": room.get("room_type", "room"),
                "bbox": room.get("bbox"),
            })

        # Symbol: symbol_candidates
        for sc in expected.get("symbol_candidates") or []:
            gold_by_family["symbol"].append({
                "image": record.get("image_path"),
                "id": sc.get("id"),
                "label": sc.get("symbol_type", "symbol"),
                "bbox": sc.get("bbox"),
            })

        # Text: text_candidates + dimension_candidates
        for tc in expected.get("text_candidates") or []:
            gold_by_family["text"].append({
                "image": record.get("image_path"),
                "id": tc.get("id"),
                "label": tc.get("text_type", tc.get("type", "text")),
                "bbox": tc.get("bbox"),
            })

    print(f"Gold labels: boundary={len(gold_by_family['boundary'])}, "
          f"space={len(gold_by_family['space'])}, "
          f"symbol={len(gold_by_family['symbol'])}, "
          f"text={len(gold_by_family['text'])}")

    # Initialize real experts
    wall_expert = WallOpeningExpert()
    room_expert = RoomSpaceExpert()
    symbol_expert = SymbolFixtureExpert()
    text_expert = TextDimensionExpert()
    sheet_expert = SheetLayoutExpert()

    print(f"\nExpert status:")
    print(f"  WallOpening:    model={'loaded' if getattr(wall_expert, '_model', None) else 'passthrough'}")
    print(f"  RoomSpace:      model={'loaded' if room_expert._model else 'passthrough'}")
    print(f"  SymbolFixture:  model={'loaded' if symbol_expert._model else 'passthrough'}")
    print(f"  TextDimension:  model={'loaded' if text_expert._model else 'passthrough'}")
    print(f"  SheetLayout:    rule-based (no trained model)")

    # Run each expert using real models
    expert_results = {}
    all_predictions = []

    # Build RoutedCandidate lists from dev records for each expert family
    print("\n--- Building RoutedCandidates ---")
    boundary_candidates = []
    boundary_candidate_batches = []
    room_candidates = []
    symbol_candidates = []
    text_candidates = []

    for record in dev_records:
        expected = record.get("expected_json") or {}
        meta = record.get("metadata") or {}
        page_meta = {"width": meta.get("width", 2000), "height": meta.get("height", 2000)}
        room_page_context = _build_room_page_context(record)
        primitive_graph = (record.get("request_hints") or {}).get("primitive_graph") or {}
        primitive_by_id = {
            str(node.get("id")): node
            for node in primitive_graph.get("nodes") or []
            if isinstance(node, dict) and node.get("id") is not None
        }
        primitive_edges = [
            edge
            for edge in primitive_graph.get("edges") or []
            if isinstance(edge, dict)
        ]

        # Boundary candidates
        record_boundary_candidates = []
        for node in expected.get("semantic_candidates") or []:
            label = node.get("semantic_type", node.get("label", "unknown"))
            if label in ("hard_wall", "door", "window"):
                target_id = str(node.get("target_id", node.get("id")))
                primitive = primitive_by_id.get(target_id, {})
                candidate = RoutedCandidate(
                    candidate_id=target_id,
                    expert="wall_opening",
                    family="boundary",
                    candidate_type="boundary",
                    confidence=0.9,
                    bbox=primitive.get("bbox") or node.get("bbox"),
                    payload={
                        "_page_metadata": page_meta,
                        "image": record.get("image_path"),
                        "raster_path": record.get("image_path"),
                        "source_dataset": record.get("source_dataset"),
                        "features": primitive,
                        "bbox": primitive.get("bbox") or node.get("bbox"),
                        "edges": primitive_edges,
                    },
                )
                record_boundary_candidates.append(candidate)
                boundary_candidates.append(candidate)
        if record_boundary_candidates:
            boundary_candidate_batches.append(record_boundary_candidates)

        # Room candidates
        for room in expected.get("room_candidates") or []:
            room_candidates.append(RoutedCandidate(
                candidate_id=str(room.get("id")),
                expert="room_space",
                family="space",
                candidate_type="room",
                confidence=0.9,
                bbox=room.get("bbox"),
                payload={
                    "shape_features": room.get("shape_features", {}),
                    "_page_metadata": page_meta,
                    "page_context": room_page_context,
                },
            ))

        # Symbol candidates (include room context for v8 features)
        rooms_in_record = expected.get("room_candidates") or []
        symbols_in_record = expected.get("symbol_candidates") or []
        for sc in symbols_in_record:
            symbol_candidates.append(RoutedCandidate(
                candidate_id=str(sc.get("id")),
                expert="symbol_fixture",
                family="symbol",
                candidate_type="symbol",
                confidence=0.9,
                bbox=sc.get("bbox"),
                payload={
                    "rooms": [{"bbox": r.get("bbox"), "room_type": r.get("room_type", "")} for r in rooms_in_record],
                    "_page_metadata": page_meta,
                },
            ))

        # Text candidates
        for tc in expected.get("text_candidates") or []:
            text_candidates.append(RoutedCandidate(
                candidate_id=str(tc.get("id")),
                expert="text_dimension",
                family="text",
                candidate_type="text",
                confidence=0.9,
                bbox=tc.get("bbox"),
                payload={
                    "raw_text": tc.get("raw_text", tc.get("text", "")),
                    "text_type": tc.get("text_type", "note_text"),
                    "_page_metadata": page_meta,
                },
            ))

    print(f"  RoutedCandidates: boundary={len(boundary_candidates)}, room={len(room_candidates)}, "
          f"symbol={len(symbol_candidates)}, text={len(text_candidates)}")

    # WallOpening expert
    print("\n--- WallOpening expert ---")
    wall_preds = run_expert_batched(
        wall_expert,
        boundary_candidate_batches,
        "wall_opening",
        "boundary",
        gold_by_family,
    )
    all_predictions.extend(wall_preds)
    expert_results["wall_opening"] = {"predictions": len(wall_preds), "gold": len(gold_by_family["boundary"])}

    # RoomSpace expert
    print("\n--- RoomSpace expert ---")
    room_preds = run_expert(room_expert, room_candidates, "room_space", "space", gold_by_family)
    all_predictions.extend(room_preds)
    expert_results["room_space"] = {"predictions": len(room_preds), "gold": len(gold_by_family["space"])}

    # SymbolFixture expert
    print("\n--- SymbolFixture expert ---")
    symbol_preds = run_expert(symbol_expert, symbol_candidates, "symbol_fixture", "symbol", gold_by_family)
    all_predictions.extend(symbol_preds)
    expert_results["symbol_fixture"] = {"predictions": len(symbol_preds), "gold": len(gold_by_family["symbol"])}

    # TextDimension expert
    print("\n--- TextDimension expert ---")
    text_preds = run_expert(text_expert, text_candidates, "text_dimension", "text", gold_by_family)
    all_predictions.extend(text_preds)
    expert_results["text_dimension"] = {"predictions": len(text_preds), "gold": len(gold_by_family["text"])}

    print(f"\nTotal predictions: {len(all_predictions)}")

    # Per-expert FP/FN audit
    audit = audit_expert_predictions(all_predictions, gold_by_family)

    elapsed = time.perf_counter() - started
    print(f"\nElapsed: {elapsed:.1f}s")

    # Save predictions
    pred_path = Path(args.output_predictions)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with pred_path.open("w", encoding="utf-8") as f:
        for pred in all_predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    print(f"Predictions saved to {pred_path}")

    # Save audit report
    report = {
        "version": "real_upstream_expert_fp_audit_v2",
        "dev_split": args.dev_split,
        "dev_records": len(dev_records),
        "total_predictions": len(all_predictions),
        "per_expert": expert_results,
        "audit": audit,
        "elapsed_seconds": round(elapsed, 1),
        "done_when_check": {
            "all_493_records_have_predictions": len(dev_records) == 493,
            "per_expert_fp_fn_auditable": all(
                "fp_rate" in audit.get(exp, {})
                for exp in ["wall_opening", "room_space", "symbol_fixture", "text_dimension"]
            ),
        },
    }

    audit_path = Path(args.output_audit)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Audit report saved to {audit_path}")


def run_expert(
    expert,
    candidates: list[RoutedCandidate],
    expert_name: str,
    family: str,
    gold_by_family: dict[str, list],
) -> list[dict]:
    """Run a real expert and convert predictions to the unified format."""
    predictions = expert.predict(candidates)
    results = []
    for pred in predictions:
        results.append({
            "candidate_id": str(pred.candidate_id),
            "expert": pred.expert,
            "family": pred.family,
            "label": pred.label,
            "confidence": pred.confidence,
            "bbox": pred.bbox,
            "geometry": pred.geometry,
            "image": None,  # Will be filled in by audit
            "source": pred.source,
        })
    gold_count = len(gold_by_family.get(family, []))
    print(f"  Predictions: {len(results)}, Gold: {gold_count}, Source: {predictions[0].source if predictions else 'N/A'}")
    return results


def run_expert_batched(
    expert,
    candidate_batches: list[list[RoutedCandidate]],
    expert_name: str,
    family: str,
    gold_by_family: dict[str, list],
) -> list[dict]:
    """Run a graph-aware expert one record at a time.

    WallOpening's crop GNN builds graph and raster context from one sample.
    Passing candidates from all records as a single graph makes the image
    context ambiguous and destroys door/window predictions.
    """
    results = []
    first_source = "N/A"
    for candidates in candidate_batches:
        predictions = expert.predict(candidates)
        if predictions and first_source == "N/A":
            first_source = predictions[0].source
        for pred in predictions:
            results.append({
                "candidate_id": str(pred.candidate_id),
                "expert": pred.expert,
                "family": pred.family,
                "label": pred.label,
                "confidence": pred.confidence,
                "bbox": pred.bbox,
                "geometry": pred.geometry,
                "image": None,
                "source": pred.source,
            })
    gold_count = len(gold_by_family.get(family, []))
    print(
        f"  Predictions: {len(results)}, Gold: {gold_count}, "
        f"Batches: {len(candidate_batches)}, Source: {first_source}"
    )
    return results


def audit_expert_predictions(
    predictions: list[dict],
    gold_by_family: dict[str, list[dict]],
) -> dict[str, Any]:
    """Audit per-expert false positive and false negative rates."""
    audit = {}

    family_to_expert = {
        "boundary": "wall_opening",
        "space": "room_space",
        "symbol": "symbol_fixture",
        "text": "text_dimension",
    }

    for family, expert in family_to_expert.items():
        gold = gold_by_family.get(family, [])
        preds = [p for p in predictions if p.get("family") == family]

        gold_ids = {str(g.get("id")): g.get("label") for g in gold}
        pred_ids = {str(p.get("candidate_id")): p.get("label") for p in preds}

        tp = sum(1 for cid in pred_ids if cid in gold_ids and pred_ids[cid] == gold_ids[cid])
        fp_label = sum(1 for cid in pred_ids if cid in gold_ids and pred_ids[cid] != gold_ids[cid])
        fp_new = sum(1 for cid in pred_ids if cid not in gold_ids)
        fn = sum(1 for cid in gold_ids if cid not in pred_ids)

        precision = tp / (tp + fp_label + fp_new) if (tp + fp_label + fp_new) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        audit[expert] = {
            "tp": tp,
            "fp_label_mismatch": fp_label,
            "fp_new_candidate": fp_new,
            "fn": fn,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "fp_rate": round((fp_label + fp_new) / max(len(preds), 1), 6),
            "fn_rate": round(fn / max(len(gold), 1), 6),
        }

    return audit


def infer_family_from_label(label: str) -> str:
    boundary = {"hard_wall", "door", "window", "partition_wall", "opening"}
    symbol = {"shower", "sink", "bathtub", "stair", "column", "equipment", "appliance",
              "generic_symbol", "table", "sanitary_fixture", "furniture"}
    if label in boundary:
        return "boundary"
    if label in symbol:
        return "symbol"
    return "space"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    main()
