#!/usr/bin/env python3
"""Audit symbol long-tail errors and conservative arbitration options."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT / "scripts" / "vlm") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "vlm"))

from fuse_real_upstream import (  # noqa: E402
    compute_invalid_graph_rate,
    evaluate_nodes,
    evaluate_relations,
    extract_gold,
    fuse_predictions_with_gold_id_space,
    load_jsonl,
)
from train_symbol_label_arbitration_v1 import (  # noqa: E402
    ADJUSTED_PREDICTIONS,
    BASE_PREDICTIONS,
    LABELS,
    LOCKED_SPLIT,
    apply_arbitration,
    extract_items,
)

ERROR_PACK = ROOT / "reports" / "vlm" / "symbol_long_tail_error_pack_v1.jsonl"
CONSERVATIVE_REPORT = ROOT / "reports" / "vlm" / "symbol_conservative_arbitration_v1.json"
CROSS_SOURCE_PACK = ROOT / "reports" / "vlm" / "symbol_cross_source_annotation_pack_v1.json"
CROSS_SOURCE_JSONL = ROOT / "reports" / "vlm" / "symbol_cross_source_annotation_pack_v1.pending.jsonl"
FLOORPLANCAD = ROOT / "datasets" / "external" / "floorplancad" / "samples.json"
WATCHED = {"generic_symbol", "bathtub", "equipment", "column"}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def symbol_prediction_rows(predictions: list[dict[str, Any]], locked_rows: list[dict[str, Any]], locked_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    symbol_preds = [pred for pred in predictions if str(pred.get("family")) == "symbol"]
    rows = []
    for item, pred in zip(locked_items, symbol_preds):
        metadata = pred.get("metadata") or {}
        probs = metadata.get("arbitration_probs") or {}
        confidence = float(pred.get("confidence") or 0.0)
        rows.append(
            {
                "record_index": int(item["record_index"]),
                "image": item.get("image"),
                "candidate_id": str(item.get("candidate_id")),
                "gold_label": str(item.get("label")),
                "pred_label": str(pred.get("label")),
                "base_label": str(metadata.get("base_label") or pred.get("label")),
                "confidence": confidence,
                "arbitration_probs": probs,
                "bbox": pred.get("bbox"),
            }
        )
    return rows


def build_error_pack(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    errors = []
    for row in rows:
        if row["gold_label"] in WATCHED or row["pred_label"] in WATCHED:
            if row["gold_label"] != row["pred_label"] or row["confidence"] >= 0.75:
                errors.append(row)
    errors.sort(key=lambda r: (r["gold_label"] == r["pred_label"], -float(r["confidence"])))
    return errors[:500]


def conservative_lookup(rows: list[dict[str, Any]], threshold: float, abstain_labels: set[str]) -> dict[tuple[int, str], tuple[str, float, dict[str, float]]]:
    lookup = {}
    for row in rows:
        probs = row.get("arbitration_probs") or {}
        pred = str(row["pred_label"])
        conf = float(row["confidence"])
        label = pred
        source = "symbol_label_arbitration_v1"
        if conf < threshold or pred in abstain_labels:
            label = str(row["base_label"])
            source = "base_prediction_fallback"
        lookup[(int(row["record_index"]), str(row["candidate_id"]))] = (label, conf, {**probs, "_conservative_source": source})
    return lookup


def eval_predictions(predictions: list[dict[str, Any]], locked_rows: list[dict[str, Any]]) -> dict[str, Any]:
    gold_nodes, gold_edges = extract_gold(locked_rows)
    nodes, edges = fuse_predictions_with_gold_id_space(predictions, locked_rows)
    return {
        "node_evaluation": evaluate_nodes(nodes, gold_nodes),
        "relation_evaluation": evaluate_relations(edges, gold_edges),
        "invalid_graph_rate": round(compute_invalid_graph_rate(nodes, edges), 6),
    }


def per_label(metrics: dict[str, Any], labels: list[str]) -> dict[str, Any]:
    src = (metrics.get("node_evaluation") or {}).get("per_label") or {}
    return {label: src.get(label) for label in labels}


def load_floorplancad() -> list[dict[str, Any]]:
    if not FLOORPLANCAD.exists():
        return []
    data = json.loads(FLOORPLANCAD.read_text(encoding="utf-8"))
    return list((data.get("samples") if isinstance(data, dict) else data) or [])


def build_cross_source_pack() -> dict[str, Any]:
    mapping = {
        "toilet": "generic_symbol",
        "sink": "sink",
        "shower": "shower",
        "bathtub": "bathtub",
        "column": "column",
        "stair": "stair",
        "sliding_door": None,
        "wall": None,
        "window": None,
        "door": None,
    }
    records = []
    source_counts = Counter()
    for sample in load_floorplancad():
        rel = sample.get("filepath")
        image_path = ROOT / "datasets" / "external" / "floorplancad" / str(rel)
        if not rel or not image_path.exists():
            continue
        detections = (((sample.get("ground_truth") or {}).get("detections")) or [])
        candidate_dets = []
        for idx, det in enumerate(detections):
            label = str(det.get("label") or "unknown")
            if label in mapping:
                source_counts[label] += 1
            target = mapping.get(label)
            if target:
                candidate_dets.append(
                    {
                        "det_id": idx,
                        "source_label": label,
                        "bbox_normalized_xywh": det.get("bounding_box"),
                        "suggested_9class_symbol_type": target,
                        "gold_9class_symbol_type": None,
                        "annotation_status": "pending_review",
                    }
                )
        if candidate_dets:
            records.append(
                {
                    "source_dataset": "floorplancad",
                    "image_path": str(Path("datasets") / "external" / "floorplancad" / str(rel)),
                    "symbol_annotations": candidate_dets[:80],
                    "annotation_status": "pending_review",
                }
            )
        if len(records) >= 50:
            break
    write_jsonl(CROSS_SOURCE_JSONL, records)
    return {
        "version": "symbol_cross_source_annotation_pack_v1",
        "created": "2026-05-03",
        "status": "pending_annotation",
        "source": str(FLOORPLANCAD.relative_to(ROOT)),
        "selected_drawings": len(records),
        "pending_jsonl": str(CROSS_SOURCE_JSONL.relative_to(ROOT)),
        "source_label_counts": dict(source_counts.most_common()),
        "target_9class_labels": LABELS,
        "annotation_fields": ["image_path", "bbox_normalized_xywh", "gold_9class_symbol_type", "occlusion/legibility", "review_notes"],
        "paper_boundary": "This is a candidate annotation pack, not a passed cross-source symbol smoke test.",
        "sample_preview": records[:3],
    }


def main() -> int:
    locked_rows = load_jsonl(LOCKED_SPLIT)
    locked_items = extract_items(locked_rows)
    adjusted_predictions = load_jsonl(ADJUSTED_PREDICTIONS)
    base_predictions = load_jsonl(BASE_PREDICTIONS)
    rows = symbol_prediction_rows(adjusted_predictions, locked_rows, locked_items)
    error_pack = build_error_pack(rows)
    write_jsonl(ERROR_PACK, error_pack)

    full_metrics = eval_predictions(adjusted_predictions, locked_rows)
    base_metrics = eval_predictions(base_predictions, locked_rows)
    variants = []
    for threshold in [0.0, 0.35, 0.5, 0.65, 0.8]:
        for abstain in [set(), {"generic_symbol"}, {"generic_symbol", "bathtub"}]:
            lookup = conservative_lookup(rows, threshold, abstain)
            preds, application = apply_arbitration(base_predictions, locked_rows, lookup)
            metrics = eval_predictions(preds, locked_rows)
            variants.append(
                {
                    "threshold": threshold,
                    "abstain_labels": sorted(abstain),
                    "application": application,
                    "node_macro_f1": metrics["node_evaluation"]["macro_f1"],
                    "relation_f1_repair_enabled_legacy": metrics["relation_evaluation"]["f1"],
                    "invalid_graph_rate": metrics["invalid_graph_rate"],
                    "long_tail_per_label": per_label(metrics, sorted(WATCHED)),
                }
            )
    best = max(variants, key=lambda row: (row["node_macro_f1"], row["relation_f1_repair_enabled_legacy"]))
    conservative_report = {
        "version": "symbol_conservative_arbitration_v1",
        "created": "2026-05-03",
        "error_pack": str(ERROR_PACK.relative_to(ROOT)),
        "baseline_base_predictions": {
            "node_macro_f1": base_metrics["node_evaluation"]["macro_f1"],
            "relation_f1_repair_enabled_legacy": base_metrics["relation_evaluation"]["f1"],
            "long_tail_per_label": per_label(base_metrics, sorted(WATCHED)),
        },
        "full_arbitration": {
            "node_macro_f1": full_metrics["node_evaluation"]["macro_f1"],
            "relation_f1_repair_enabled_legacy": full_metrics["relation_evaluation"]["f1"],
            "long_tail_per_label": per_label(full_metrics, sorted(WATCHED)),
        },
        "sweep_variants": variants,
        "best_by_node_macro_f1": best,
        "adoption_recommendation": "keep_full_arbitration" if best["node_macro_f1"] <= full_metrics["node_evaluation"]["macro_f1"] else "consider_conservative_appendix_only",
        "note": "Relation values here use the existing repair-enabled legacy fusion path only for symbol-arbitration comparison; paper-main relation remains the no-repair reconciliation source.",
        "status": "passed",
    }
    write_json(CONSERVATIVE_REPORT, conservative_report)
    write_json(CROSS_SOURCE_PACK, build_cross_source_pack())

    print(f"wrote {ERROR_PACK}")
    print(f"wrote {CONSERVATIVE_REPORT}")
    print(f"wrote {CROSS_SOURCE_PACK}")
    print(json.dumps({"error_rows": len(error_pack), "best_node_macro_f1": best["node_macro_f1"], "recommendation": conservative_report["adoption_recommendation"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
