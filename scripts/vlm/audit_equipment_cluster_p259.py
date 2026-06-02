#!/usr/bin/env python3
"""Audit equipment one-to-many and matching-policy failure modes after P256."""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "vlm"))

from fuse_symbol_p206g_with_p211_p212 import bbox_iou, center_covered, load_p206g  # noqa: E402


P256_PREDS = ROOT / "reports/vlm/p256_runtime_box_calibration_predictions.jsonl"
GOLD_OVERLAY = ROOT / "reports/vlm/symbol_p224a_column_frozen_overlay.jsonl"
OUT_JSON = ROOT / "reports/vlm/p259_equipment_cluster_audit.json"
OUT_MD = ROOT / "reports/vlm/p259_equipment_cluster_audit.md"

LABEL = "equipment"
IOU_STRICT = 0.30
GOLD_CLUSTER_IOU = 0.05
GOLD_CLUSTER_CENTER_FACTOR = 1.25


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def label_of(candidate: dict[str, Any]) -> str:
    return str(
        candidate.get("candidate_type")
        or candidate.get("label")
        or candidate.get("symbol_type")
        or (candidate.get("payload") or {}).get("symbol_type")
        or "generic_symbol"
    )


def score_of(candidate: dict[str, Any]) -> float:
    return float(candidate.get("confidence") or candidate.get("score") or (candidate.get("payload") or {}).get("score") or 1.0)


def load_predictions(path: Path) -> dict[str, list[dict[str, Any]]]:
    by_row: dict[str, list[dict[str, Any]]] = {}
    for row in load_jsonl(path):
        row_id = str(row.get("row_id") or row.get("id"))
        candidates = row.get("routed_candidates") or row.get("expert_predictions") or row.get("predicted_symbols") or []
        preds = []
        for index, candidate in enumerate(candidates):
            preds.append(
                {
                    "index": index,
                    "label": label_of(candidate),
                    "bbox": [float(v) for v in candidate["bbox"]],
                    "score": score_of(candidate),
                    "source": str(candidate.get("source") or (candidate.get("payload") or {}).get("source") or ""),
                    "candidate_id": str(candidate.get("candidate_id") or candidate.get("id") or f"{row_id}_pred_{index:05d}"),
                }
            )
        by_row[row_id] = preds
    return by_row


def width(box: list[float]) -> float:
    return max(0.0, box[2] - box[0])


def height(box: list[float]) -> float:
    return max(0.0, box[3] - box[1])


def area(box: list[float]) -> float:
    return width(box) * height(box)


def center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def diag(box: list[float]) -> float:
    return max(math.hypot(width(box), height(box)), 1e-6)


def center_distance(a: list[float], b: list[float]) -> float:
    ax, ay = center(a)
    bx, by = center(b)
    return math.hypot(ax - bx, ay - by)


def round6(value: float) -> float:
    return round(float(value), 6)


def metric(tp: int, pred_count: int, gold_count: int) -> dict[str, float | int]:
    precision = tp / pred_count if pred_count else 0.0
    recall = tp / gold_count if gold_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "pred": pred_count,
        "gold": gold_count,
        "precision": round6(precision),
        "recall": round6(recall),
        "f1": round6(f1),
    }


def greedy_typed_match_pairs(preds: list[dict[str, Any]], golds: list[dict[str, Any]]) -> tuple[dict[int, int], dict[int, int]]:
    candidates = []
    for pred_index, pred in enumerate(preds):
        if pred["label"] != LABEL:
            continue
        for gold_index, gold in enumerate(golds):
            if str(gold["label"]) != LABEL:
                continue
            iou = bbox_iou(pred["bbox"], [float(v) for v in gold["bbox"]])
            if iou >= IOU_STRICT:
                candidates.append((iou, pred_index, gold_index))
    candidates.sort(reverse=True)
    pred_to_gold: dict[int, int] = {}
    gold_to_pred: dict[int, int] = {}
    for _, pred_index, gold_index in candidates:
        if pred_index in pred_to_gold or gold_index in gold_to_pred:
            continue
        pred_to_gold[pred_index] = gold_index
        gold_to_pred[gold_index] = pred_index
    return pred_to_gold, gold_to_pred


def is_gold_cluster_neighbor(box_a: list[float], box_b: list[float]) -> bool:
    if bbox_iou(box_a, box_b) >= GOLD_CLUSTER_IOU:
        return True
    threshold = GOLD_CLUSTER_CENTER_FACTOR * max(diag(box_a), diag(box_b))
    return center_distance(box_a, box_b) <= threshold


def connected_components(golds: list[dict[str, Any]]) -> list[list[int]]:
    parent = list(range(len(golds)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    boxes = [[float(v) for v in gold["bbox"]] for gold in golds]
    for left in range(len(golds)):
        for right in range(left + 1, len(golds)):
            if is_gold_cluster_neighbor(boxes[left], boxes[right]):
                union(left, right)
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(golds)):
        groups[find(index)].append(index)
    return list(groups.values())


def main() -> None:
    preds_by_row = load_predictions(P256_PREDS)
    _, _, golds_by_row = load_p206g(GOLD_OVERLAY)

    totals = Counter()
    conflict_types = Counter()
    per_row_summaries: list[dict[str, Any]] = []
    unmatched_examples: list[dict[str, Any]] = []
    cluster_examples: list[dict[str, Any]] = []

    rows_with_equipment = 0
    row_count = len(golds_by_row)
    multi_match_upper_tp = 0
    center_cover_upper_tp = 0
    one_pred_multi_gold_extra_tp = 0
    cluster_total = 0
    cluster_gold_total = 0
    cluster_matched = 0
    cluster_all_gold_oracle_tp = 0
    official_tp = 0
    official_pred = 0
    official_gold = 0

    for row_id, gold_map in golds_by_row.items():
        all_preds = preds_by_row.get(row_id, [])
        all_golds = list(gold_map.values())
        equipment_preds = [pred for pred in all_preds if pred["label"] == LABEL]
        equipment_gold_indices = [index for index, gold in enumerate(all_golds) if str(gold["label"]) == LABEL]
        if not equipment_preds and not equipment_gold_indices:
            continue
        rows_with_equipment += 1

        pred_to_gold, gold_to_pred = greedy_typed_match_pairs(all_preds, all_golds)
        used_equipment_preds = {pred_index for pred_index, gold_index in pred_to_gold.items() if str(all_golds[gold_index]["label"]) == LABEL}
        equipment_gold_set = set(equipment_gold_indices)
        matched_equipment_gold = set(gold_to_pred).intersection(equipment_gold_set)
        unmatched_equipment_gold = equipment_gold_set - matched_equipment_gold

        official_tp += len(matched_equipment_gold)
        official_pred += len(equipment_preds)
        official_gold += len(equipment_gold_indices)

        row_conflicts = Counter()
        row_multi_extra = 0
        row_center_cover_extra = 0
        row_multi_oracle_extra = 0

        for gold_index in equipment_gold_indices:
            gold_box = [float(v) for v in all_golds[gold_index]["bbox"]]
            overlapping_preds = []
            center_cover_preds = []
            for pred_index, pred in enumerate(all_preds):
                if pred["label"] != LABEL:
                    continue
                iou = bbox_iou(pred["bbox"], gold_box)
                if iou >= IOU_STRICT:
                    overlapping_preds.append((pred_index, iou))
                if center_covered(pred["bbox"], gold_box):
                    center_cover_preds.append((pred_index, iou))
            if overlapping_preds:
                multi_match_upper_tp += 1
            if center_cover_preds:
                center_cover_upper_tp += 1

            if gold_index not in unmatched_equipment_gold:
                continue

            totals["official_unmatched_equipment_gold"] += 1
            matched_overlap = [(pred_index, iou) for pred_index, iou in overlapping_preds if pred_index in used_equipment_preds]
            unused_overlap = [(pred_index, iou) for pred_index, iou in overlapping_preds if pred_index not in used_equipment_preds]
            matched_center_cover = [(pred_index, iou) for pred_index, iou in center_cover_preds if pred_index in used_equipment_preds]
            unused_center_cover = [(pred_index, iou) for pred_index, iou in center_cover_preds if pred_index not in used_equipment_preds]

            if matched_overlap:
                conflict_type = "shares_already_matched_prediction_iou_ge_0_30"
                row_multi_extra += 1
                one_pred_multi_gold_extra_tp += 1
            elif unused_overlap:
                conflict_type = "has_unused_prediction_iou_ge_0_30"
                row_multi_extra += 1
            elif matched_center_cover:
                conflict_type = "shares_already_matched_prediction_center_cover"
                row_center_cover_extra += 1
            elif unused_center_cover:
                conflict_type = "has_unused_prediction_center_cover"
                row_center_cover_extra += 1
            else:
                conflict_type = "no_same_label_prediction_covers_gold"

            conflict_types[conflict_type] += 1
            row_conflicts[conflict_type] += 1
            best_overlap = max(overlapping_preds, key=lambda item: item[1], default=None)
            best_center = max(center_cover_preds, key=lambda item: item[1], default=None)
            if len(unmatched_examples) < 40:
                best_pred_index = None
                if best_overlap is not None:
                    best_pred_index = best_overlap[0]
                elif best_center is not None:
                    best_pred_index = best_center[0]
                best_pred = all_preds[best_pred_index] if best_pred_index is not None else None
                matched_gold_index = pred_to_gold.get(best_pred_index) if best_pred_index is not None else None
                unmatched_examples.append(
                    {
                        "row_id": row_id,
                        "conflict_type": conflict_type,
                        "gold_index": gold_index,
                        "gold_bbox": [round6(v) for v in gold_box],
                        "best_iou": round6(best_overlap[1]) if best_overlap else 0.0,
                        "best_center_cover_iou": round6(best_center[1]) if best_center else 0.0,
                        "best_pred_index": best_pred_index,
                        "best_pred_bbox": [round6(v) for v in best_pred["bbox"]] if best_pred else None,
                        "best_pred_score": round6(best_pred["score"]) if best_pred else None,
                        "best_pred_matched_gold_index": matched_gold_index,
                        "best_pred_matched_gold_bbox": [round6(v) for v in all_golds[matched_gold_index]["bbox"]]
                        if matched_gold_index is not None
                        else None,
                    }
                )

        row_equipment_golds = [all_golds[index] for index in equipment_gold_indices]
        clusters = connected_components(row_equipment_golds)
        row_cluster_stats = []
        for cluster in clusters:
            cluster_total += 1
            cluster_gold_total += len(cluster)
            absolute_gold_indices = [equipment_gold_indices[index] for index in cluster]
            cluster_gold_boxes = [[float(v) for v in all_golds[index]["bbox"]] for index in absolute_gold_indices]
            cluster_pred_indices = set()
            cluster_matched_gold_count = 0
            cluster_any_prediction = False
            for relative_index, absolute_gold_index in enumerate(absolute_gold_indices):
                gold_box = cluster_gold_boxes[relative_index]
                if absolute_gold_index in matched_equipment_gold:
                    cluster_matched_gold_count += 1
                for pred_index, pred in enumerate(all_preds):
                    if pred["label"] != LABEL:
                        continue
                    if bbox_iou(pred["bbox"], gold_box) >= IOU_STRICT or center_covered(pred["bbox"], gold_box):
                        cluster_pred_indices.add(pred_index)
                        cluster_any_prediction = True
            if cluster_any_prediction:
                cluster_matched += 1
                cluster_all_gold_oracle_tp += len(cluster)
                if cluster_matched_gold_count < len(cluster):
                    row_multi_oracle_extra += len(cluster) - cluster_matched_gold_count
            row_cluster_stats.append(
                {
                    "gold_count": len(cluster),
                    "pred_count": len(cluster_pred_indices),
                    "official_matched_gold_count": cluster_matched_gold_count,
                    "oracle_cluster_matched": cluster_any_prediction,
                }
            )
            if cluster_any_prediction and cluster_matched_gold_count < len(cluster) and len(cluster_examples) < 25:
                cluster_examples.append(
                    {
                        "row_id": row_id,
                        "gold_count": len(cluster),
                        "pred_count": len(cluster_pred_indices),
                        "official_matched_gold_count": cluster_matched_gold_count,
                        "gold_indices": absolute_gold_indices,
                        "gold_bboxes": [[round6(v) for v in box] for box in cluster_gold_boxes],
                        "pred_indices": sorted(cluster_pred_indices),
                        "pred_bboxes": [[round6(v) for v in all_preds[index]["bbox"]] for index in sorted(cluster_pred_indices)],
                    }
                )

        if row_conflicts or row_multi_oracle_extra:
            per_row_summaries.append(
                {
                    "row_id": row_id,
                    "equipment_pred": len(equipment_preds),
                    "equipment_gold": len(equipment_gold_indices),
                    "official_tp": len(matched_equipment_gold),
                    "official_fn": len(unmatched_equipment_gold),
                    "official_fp": max(0, len(equipment_preds) - len(matched_equipment_gold)),
                    "multi_match_oracle_extra_tp": row_multi_extra,
                    "center_cover_oracle_extra_tp": row_center_cover_extra,
                    "cluster_oracle_extra_tp": row_multi_oracle_extra,
                    "conflict_types": dict(row_conflicts),
                    "clusters": row_cluster_stats,
                }
            )

    official = metric(official_tp, official_pred, official_gold)
    multi_match = metric(multi_match_upper_tp, official_pred, official_gold)
    center_cover = metric(center_cover_upper_tp, official_pred, official_gold)
    cluster_oracle = metric(min(cluster_all_gold_oracle_tp, official_gold), official_pred, official_gold)

    per_row_summaries.sort(
        key=lambda row: (
            row["multi_match_oracle_extra_tp"] + row["center_cover_oracle_extra_tp"] + row["cluster_oracle_extra_tp"],
            row["official_fn"],
        ),
        reverse=True,
    )

    result = {
        "id": "p259_equipment_cluster_and_matching_policy_audit",
        "phase": "P259_equipment_cluster_matching_policy",
        "source_predictions": str(P256_PREDS.relative_to(ROOT)),
        "gold_overlay": str(GOLD_OVERLAY.relative_to(ROOT)),
        "scope": {
            "label": LABEL,
            "iou_threshold": IOU_STRICT,
            "rows_total": row_count,
            "rows_with_equipment": rows_with_equipment,
            "note": "All upper bounds are diagnostics only. They do not change runtime predictions or official one-prediction-to-one-gold evaluation.",
        },
        "official_equipment_metric": official,
        "diagnostic_upper_bounds": {
            "allow_one_prediction_to_match_multiple_golds_iou_ge_0_30": {
                **multi_match,
                "delta_f1_vs_official": round6(float(multi_match["f1"]) - float(official["f1"])),
                "extra_tp_vs_official": multi_match_upper_tp - official_tp,
            },
            "allow_center_cover_match_multiple_golds": {
                **center_cover,
                "delta_f1_vs_official": round6(float(center_cover["f1"]) - float(official["f1"])),
                "extra_tp_vs_official": center_cover_upper_tp - official_tp,
            },
            "cluster_matched_if_any_cluster_gold_has_prediction": {
                **cluster_oracle,
                "delta_f1_vs_official": round6(float(cluster_oracle["f1"]) - float(official["f1"])),
                "extra_tp_vs_official": cluster_all_gold_oracle_tp - official_tp,
                "cluster_count": cluster_total,
                "cluster_gold_total": cluster_gold_total,
                "cluster_matched_count": cluster_matched,
                "cluster_recall": round6(cluster_matched / cluster_total) if cluster_total else 0.0,
            },
        },
        "unmatched_gold_conflict_breakdown": dict(conflict_types),
        "one_pred_multi_gold_extra_tp": one_pred_multi_gold_extra_tp,
        "top_rows_by_conflict": per_row_summaries[:25],
        "unmatched_examples": unmatched_examples,
        "cluster_examples": cluster_examples,
        "interpretation": {
            "decision": "equipment_gap_is_mostly_matching_or_granularity_artifact"
            if one_pred_multi_gold_extra_tp >= 40
            else "equipment_gap_still_requires_runtime_prediction_policy",
            "recommended_next": "P260_equipment_granularity_and_reporting_policy"
            if one_pred_multi_gold_extra_tp >= 40
            else "P260_equipment_runtime_box_policy_search",
            "guardrail": "Do not promote diagnostic upper bounds as official metrics; use them to decide whether the paper should report equipment as annotation-granularity limited.",
        },
    }

    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = [
        "# P259 Equipment Cluster and Matching-Policy Audit",
        "",
        "## Scope",
        f"- Predictions: `{P256_PREDS.relative_to(ROOT)}`",
        f"- Gold overlay: `{GOLD_OVERLAY.relative_to(ROOT)}`",
        f"- Label: `{LABEL}`",
        f"- Official IoU threshold: `{IOU_STRICT}`",
        "- Upper bounds below are diagnostic only; they are not official promoted metrics.",
        "",
        "## Official Equipment Metric",
        f"- TP / pred / gold: {official['tp']} / {official['pred']} / {official['gold']}",
        f"- Precision / recall / F1: {official['precision']} / {official['recall']} / {official['f1']}",
        "",
        "## Diagnostic Upper Bounds",
        "| Diagnostic | TP | Precision | Recall | F1 | ΔF1 | Extra TP |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, row in result["diagnostic_upper_bounds"].items():
        md.append(
            f"| {key} | {row['tp']} | {row['precision']} | {row['recall']} | {row['f1']} | "
            f"{row['delta_f1_vs_official']} | {row['extra_tp_vs_official']} |"
        )
    md.extend(
        [
            "",
            "## Unmatched Gold Conflict Breakdown",
        ]
    )
    for key, value in conflict_types.most_common():
        md.append(f"- {key}: {value}")
    md.extend(
        [
            "",
            "## Interpretation",
            f"- Decision: `{result['interpretation']['decision']}`",
            f"- Recommended next: `{result['interpretation']['recommended_next']}`",
            "- This supports a bounded equipment annotation-granularity diagnostic before any additional detector fusion.",
            "",
            "## Top Conflict Rows",
            "| row_id | pred | gold | TP | FN | multi-extra | center-extra | cluster-extra |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in per_row_summaries[:15]:
        md.append(
            f"| {row['row_id']} | {row['equipment_pred']} | {row['equipment_gold']} | {row['official_tp']} | "
            f"{row['official_fn']} | {row['multi_match_oracle_extra_tp']} | "
            f"{row['center_cover_oracle_extra_tp']} | {row['cluster_oracle_extra_tp']} |"
        )
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
