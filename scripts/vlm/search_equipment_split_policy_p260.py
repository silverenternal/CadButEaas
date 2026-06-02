#!/usr/bin/env python3
"""Search runtime-safe equipment duplicate/child-box policies on P256 predictions."""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "vlm"))

from fuse_symbol_p206g_with_p211_p212 import bbox_iou, load_p206g  # noqa: E402


BASE_PREDS = ROOT / "reports/vlm/p256_runtime_box_calibration_predictions.jsonl"
GOLD_OVERLAY = ROOT / "reports/vlm/symbol_p224a_column_frozen_overlay.jsonl"
OUT_PREDS = ROOT / "reports/vlm/p260_equipment_split_policy_predictions.jsonl"
OUT_EVAL = ROOT / "reports/vlm/p260_equipment_split_policy_eval.json"
OUT_MD = ROOT / "reports/vlm/p260_promotion_decision.md"

IOU = 0.30
LABEL = "equipment"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def label_of(candidate: dict[str, Any]) -> str:
    return str(
        candidate.get("candidate_type")
        or candidate.get("symbol_type")
        or candidate.get("label")
        or (candidate.get("payload") or {}).get("symbol_type")
        or "generic_symbol"
    )


def score_of(candidate: dict[str, Any]) -> float:
    return float(candidate.get("confidence") or candidate.get("score") or (candidate.get("payload") or {}).get("score") or 1.0)


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


def center_distance(left: list[float], right: list[float]) -> float:
    lx, ly = center(left)
    rx, ry = center(right)
    return math.hypot(lx - rx, ly - ry)


def area_bucket(box: list[float]) -> str:
    value = area(box)
    if value <= 64:
        return "tiny_le_64"
    if value <= 256:
        return "small_le_256"
    if value <= 1024:
        return "medium_le_1024"
    if value <= 4096:
        return "large_le_4096"
    return "xlarge_gt_4096"


def shape_bucket(box: list[float]) -> str:
    ratio = width(box) / max(height(box), 1e-6)
    if ratio >= 1.8:
        return "wide"
    if ratio <= 0.56:
        return "tall"
    return "balanced"


def round6(value: float) -> float:
    return round(float(value), 6)


def row_predictions(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        row_id = str(row.get("row_id") or row.get("id"))
        preds = []
        for index, candidate in enumerate(row.get("routed_candidates") or []):
            box = [float(v) for v in candidate["bbox"]]
            preds.append(
                {
                    "index": index,
                    "label": label_of(candidate),
                    "bbox": box,
                    "score": score_of(candidate),
                    "bucket": area_bucket(box),
                    "shape": shape_bucket(box),
                }
            )
        out[row_id] = annotate_equipment_neighbors(preds)
    return out


def annotate_equipment_neighbors(preds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    equipment_indices = [index for index, pred in enumerate(preds) if pred["label"] == LABEL]
    for pred in preds:
        pred["near_equipment_count"] = 0
        pred["overlap_equipment_count"] = 0
        pred["contained_equipment_count"] = 0
    for left_pos, left_index in enumerate(equipment_indices):
        left = preds[left_index]
        left_box = left["bbox"]
        for right_index in equipment_indices[left_pos + 1 :]:
            right = preds[right_index]
            right_box = right["bbox"]
            near_threshold = 1.25 * max(diag(left_box), diag(right_box))
            is_near = center_distance(left_box, right_box) <= near_threshold
            overlap = bbox_iou(left_box, right_box)
            if is_near:
                left["near_equipment_count"] += 1
                right["near_equipment_count"] += 1
            if overlap >= 0.05:
                left["overlap_equipment_count"] += 1
                right["overlap_equipment_count"] += 1
            smaller = min(area(left_box), area(right_box))
            if smaller > 0:
                intersection = intersection_area(left_box, right_box)
                if intersection / smaller >= 0.70:
                    left["contained_equipment_count"] += 1
                    right["contained_equipment_count"] += 1
    return preds


def intersection_area(left: list[float], right: list[float]) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def score(
    preds_by_row: dict[str, list[dict[str, Any]]],
    golds_by_row: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    totals = Counter()
    by_label_gold = Counter()
    by_label_tp = Counter()
    by_label_pred = Counter()

    for row_id, gold_map in golds_by_row.items():
        preds = preds_by_row.get(row_id, [])
        golds = list(gold_map.values())
        for pred in preds:
            by_label_pred[pred["label"]] += 1
        for gold in golds:
            by_label_gold[str(gold["label"])] += 1

        candidates: list[tuple[float, int, int]] = []
        for pred_index, pred in enumerate(preds):
            for gold_index, gold in enumerate(golds):
                if pred["label"] != str(gold["label"]):
                    continue
                iou = bbox_iou(pred["bbox"], [float(v) for v in gold["bbox"]])
                if iou >= IOU:
                    candidates.append((iou, pred_index, gold_index))
        candidates.sort(reverse=True)
        used_pred: set[int] = set()
        used_gold: set[int] = set()
        for _, pred_index, gold_index in candidates:
            if pred_index in used_pred or gold_index in used_gold:
                continue
            used_pred.add(pred_index)
            used_gold.add(gold_index)
            label = str(golds[gold_index]["label"])
            by_label_tp[label] += 1
            totals["tp"] += 1
        totals["predicted"] += len(preds)
        totals["gold"] += len(golds)

    precision = totals["tp"] / max(totals["predicted"], 1)
    recall = totals["tp"] / max(totals["gold"], 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    labels = sorted(set(by_label_gold) | set(by_label_pred))
    per_label = {}
    for label in labels:
        tp = by_label_tp[label]
        pred = by_label_pred[label]
        gold = by_label_gold[label]
        label_precision = tp / max(pred, 1)
        label_recall = tp / max(gold, 1)
        per_label[label] = {
            "tp": int(tp),
            "predicted": int(pred),
            "gold": int(gold),
            "precision": round6(label_precision),
            "recall": round6(label_recall),
            "f1": round6(0.0 if label_precision + label_recall == 0 else 2 * label_precision * label_recall / (label_precision + label_recall)),
        }
    return {
        "tp": int(totals["tp"]),
        "predicted": int(totals["predicted"]),
        "gold": int(totals["gold"]),
        "fp": int(totals["predicted"] - totals["tp"]),
        "fn": int(totals["gold"] - totals["tp"]),
        "precision": round6(precision),
        "recall": round6(recall),
        "f1": round6(f1),
        "per_label": per_label,
    }


def scaled_box(box: list[float], scale: float) -> list[float]:
    cx, cy = center(box)
    new_width = width(box) * scale
    new_height = height(box) * scale
    return [cx - new_width / 2.0, cy - new_height / 2.0, cx + new_width / 2.0, cy + new_height / 2.0]


def anchored_box(box: list[float], x_anchor: str, y_anchor: str, frac_w: float, frac_h: float) -> list[float]:
    box_width = width(box)
    box_height = height(box)
    child_width = box_width * frac_w
    child_height = box_height * frac_h
    if x_anchor == "left":
        x0 = box[0]
    elif x_anchor == "right":
        x0 = box[2] - child_width
    else:
        x0 = (box[0] + box[2] - child_width) / 2.0
    if y_anchor == "top":
        y0 = box[1]
    elif y_anchor == "bottom":
        y0 = box[3] - child_height
    else:
        y0 = (box[1] + box[3] - child_height) / 2.0
    return [x0, y0, x0 + child_width, y0 + child_height]


def child_boxes(box: list[float], layout: str) -> list[list[float]]:
    if layout == "dup1":
        return [box]
    if layout == "dup2":
        return [box, box]
    if layout == "center_s050":
        return [scaled_box(box, 0.50)]
    if layout == "center_s065":
        return [scaled_box(box, 0.65)]
    if layout == "center_s080":
        return [scaled_box(box, 0.80)]
    if layout == "halves_lr":
        return [anchored_box(box, "left", "center", 0.58, 1.0), anchored_box(box, "right", "center", 0.58, 1.0)]
    if layout == "halves_tb":
        return [anchored_box(box, "center", "top", 1.0, 0.58), anchored_box(box, "center", "bottom", 1.0, 0.58)]
    if layout == "corners_s055":
        return [
            anchored_box(box, "left", "top", 0.55, 0.55),
            anchored_box(box, "right", "top", 0.55, 0.55),
            anchored_box(box, "left", "bottom", 0.55, 0.55),
            anchored_box(box, "right", "bottom", 0.55, 0.55),
        ]
    if layout == "diag_tl_br_s060":
        return [anchored_box(box, "left", "top", 0.60, 0.60), anchored_box(box, "right", "bottom", 0.60, 0.60)]
    if layout == "diag_tr_bl_s060":
        return [anchored_box(box, "right", "top", 0.60, 0.60), anchored_box(box, "left", "bottom", 0.60, 0.60)]
    raise ValueError(f"unknown layout {layout}")


def policy_name(policy: dict[str, Any]) -> str:
    return (
        f"{policy['layout']}_score{policy['min_score']:.2f}_{policy['bucket']}_{policy['shape']}"
        f"_near{policy['min_near']}_ov{policy['min_overlap']}_contain{policy['min_contained']}"
    )


def pred_matches_policy(pred: dict[str, Any], policy: dict[str, Any]) -> bool:
    if pred["label"] != LABEL:
        return False
    if pred["score"] < policy["min_score"]:
        return False
    if policy["bucket"] != "all" and pred["bucket"] != policy["bucket"]:
        return False
    if policy["shape"] != "all" and pred["shape"] != policy["shape"]:
        return False
    if pred["near_equipment_count"] < policy["min_near"]:
        return False
    if pred["overlap_equipment_count"] < policy["min_overlap"]:
        return False
    if pred["contained_equipment_count"] < policy["min_contained"]:
        return False
    return True


def apply_policy_to_preds(
    base: dict[str, list[dict[str, Any]]],
    policies: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row_id, preds in base.items():
        new_preds = [dict(pred) for pred in preds]
        next_index = len(new_preds)
        for pred in preds:
            for policy in policies:
                if not pred_matches_policy(pred, policy):
                    continue
                for child_index, child_box in enumerate(child_boxes(pred["bbox"], policy["layout"])):
                    new_preds.append(
                        {
                            "index": next_index,
                            "label": LABEL,
                            "bbox": [float(v) for v in child_box],
                            "score": pred["score"] * policy["score_scale"],
                            "bucket": area_bucket(child_box),
                            "shape": shape_bucket(child_box),
                            "near_equipment_count": pred["near_equipment_count"],
                            "overlap_equipment_count": pred["overlap_equipment_count"],
                            "contained_equipment_count": pred["contained_equipment_count"],
                            "p260_parent_index": pred["index"],
                            "p260_child_index": child_index,
                            "p260_policy": policy_name(policy),
                        }
                    )
                    next_index += 1
        out[row_id] = new_preds
    return out


def generate_policies(base_preds: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    observed_buckets = sorted(
        {
            pred["bucket"]
            for preds in base_preds.values()
            for pred in preds
            if pred["label"] == LABEL
        }
    )
    buckets = [bucket for bucket in ["all", "medium_le_1024", "large_le_4096", "xlarge_gt_4096"] if bucket == "all" or bucket in observed_buckets]
    layouts = ["dup1", "center_s065", "center_s080", "halves_lr", "halves_tb", "diag_tl_br_s060", "diag_tr_bl_s060"]
    policies = []
    for layout in layouts:
        for min_score in [0.50, 0.65, 0.80, 0.90]:
            for bucket in buckets:
                for min_near, min_overlap, min_contained in [(1, 0, 0), (2, 0, 0), (4, 1, 0), (4, 2, 0)]:
                    policies.append(
                        {
                            "layout": layout,
                            "min_score": min_score,
                            "bucket": bucket,
                            "shape": "all",
                            "min_near": min_near,
                            "min_overlap": min_overlap,
                            "min_contained": min_contained,
                            "score_scale": 0.97,
                        }
                    )
    return policies


def materialize_rows(rows: list[dict[str, Any]], policies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = deepcopy(rows)
    for row in out:
        row_id = str(row.get("row_id") or row.get("id"))
        routed = row.get("routed_candidates") or []
        preds = []
        for index, candidate in enumerate(routed):
            box = [float(v) for v in candidate["bbox"]]
            preds.append(
                {
                    "index": index,
                    "label": label_of(candidate),
                    "bbox": box,
                    "score": score_of(candidate),
                    "bucket": area_bucket(box),
                    "shape": shape_bucket(box),
                }
            )
        preds = annotate_equipment_neighbors(preds)
        additions = []
        for pred in preds:
            if pred["label"] != LABEL:
                continue
            parent_candidate = routed[pred["index"]]
            for policy in policies:
                if not pred_matches_policy(pred, policy):
                    continue
                name = policy_name(policy)
                for child_index, child_box in enumerate(child_boxes(pred["bbox"], policy["layout"])):
                    child = deepcopy(parent_candidate)
                    child_id = str(child.get("candidate_id") or child.get("id") or f"{row_id}_equipment_{pred['index']:05d}")
                    child["candidate_id"] = f"{child_id}_p260_{len(additions):03d}"
                    child["candidate_type"] = LABEL
                    child["confidence"] = score_of(parent_candidate) * policy["score_scale"]
                    child["bbox"] = [float(v) for v in child_box]
                    child["source"] = "p260_equipment_split_policy"
                    payload = child.setdefault("payload", {})
                    payload["bbox"] = [float(v) for v in child_box]
                    payload["symbol_type"] = LABEL
                    payload["confidence"] = child["confidence"]
                    payload["score"] = child["confidence"]
                    payload["proposal_stage"] = "p260_runtime_safe_equipment_split"
                    payload["p260_policy"] = name
                    payload["p260_parent_candidate_id"] = child_id
                    payload["p260_parent_bbox"] = pred["bbox"]
                    payload["p260_runtime_features"] = {
                        "parent_score": round6(pred["score"]),
                        "parent_area": round6(area(pred["bbox"])),
                        "parent_bucket": pred["bucket"],
                        "parent_shape": pred["shape"],
                        "near_equipment_count": pred["near_equipment_count"],
                        "overlap_equipment_count": pred["overlap_equipment_count"],
                        "contained_equipment_count": pred["contained_equipment_count"],
                    }
                    child.setdefault("route_trace", {})["router"] = "p260_runtime_safe_equipment_split"
                    additions.append(child)
        if additions:
            row["routed_candidates"] = routed + additions
            row["expert_predictions"] = row["routed_candidates"]
            row.setdefault("adapter_metadata", {})["p260_equipment_split_added"] = len(additions)
            row["adapter_metadata"]["p260_selected_policies"] = [policy_name(policy) for policy in policies]
    return out


def main() -> None:
    rows = load_jsonl(BASE_PREDS)
    _, _, golds = load_p206g(GOLD_OVERLAY)
    base_preds = row_predictions(rows)
    baseline = score(base_preds, golds)

    current_preds = base_preds
    current_metrics = baseline
    selected: list[dict[str, Any]] = []
    history = []
    top_single = []

    for iteration in range(1):
        best = None
        for policy in generate_policies(current_preds):
            trial_preds = apply_policy_to_preds(current_preds, [policy])
            added = sum(len(trial_preds[row_id]) - len(current_preds[row_id]) for row_id in current_preds)
            if added <= 0:
                continue
            metrics = score(trial_preds, golds)
            equipment = metrics["per_label"].get(LABEL, {})
            current_equipment = current_metrics["per_label"].get(LABEL, {})
            delta_f1 = metrics["f1"] - current_metrics["f1"]
            delta_equipment_f1 = equipment.get("f1", 0.0) - current_equipment.get("f1", 0.0)
            delta_tp = metrics["tp"] - current_metrics["tp"]
            added_precision = delta_tp / added if added else 0.0
            row = {
                "policy": policy,
                "policy_name": policy_name(policy),
                "added_predictions": added,
                "metrics": metrics,
                "delta_from_current": {
                    "tp": delta_tp,
                    "predicted": added,
                    "precision": round6(metrics["precision"] - current_metrics["precision"]),
                    "recall": round6(metrics["recall"] - current_metrics["recall"]),
                    "f1": round6(delta_f1),
                    "equipment_f1": round6(delta_equipment_f1),
                    "added_precision": round6(added_precision),
                },
            }
            if iteration == 0 and delta_tp > 0:
                top_single.append(row)
            if delta_f1 <= 0 or delta_equipment_f1 <= 0:
                continue
            if added_precision < 0.37:
                continue
            rank = (
                delta_f1,
                delta_equipment_f1,
                added_precision,
                -added,
                metrics["precision"],
            )
            if best is None or rank > best["rank"]:
                best = {**row, "rank": rank}
        if best is None:
            break
        selected.append(best["policy"])
        current_preds = apply_policy_to_preds(current_preds, [best["policy"]])
        current_metrics = best["metrics"]
        history.append({key: value for key, value in best.items() if key != "rank"})

    output_rows = materialize_rows(rows, selected)
    write_jsonl(OUT_PREDS, output_rows)

    top_single.sort(
        key=lambda row: (
            row["delta_from_current"]["f1"],
            row["delta_from_current"]["equipment_f1"],
            row["delta_from_current"]["added_precision"],
        ),
        reverse=True,
    )

    result = {
        "id": "p260_equipment_granularity_policy_and_runtime_split_probe",
        "phase": "P260_runtime_safe_equipment_split_search",
        "inputs": {
            "baseline_predictions": str(BASE_PREDS.relative_to(ROOT)),
            "gold_overlay": str(GOLD_OVERLAY.relative_to(ROOT)),
            "iou": IOU,
        },
        "baseline_metrics": baseline,
        "candidate_metrics": current_metrics,
        "delta_vs_p256": {
            "tp": current_metrics["tp"] - baseline["tp"],
            "predicted": current_metrics["predicted"] - baseline["predicted"],
            "precision": round6(current_metrics["precision"] - baseline["precision"]),
            "recall": round6(current_metrics["recall"] - baseline["recall"]),
            "f1": round6(current_metrics["f1"] - baseline["f1"]),
            "equipment_f1": round6(
                current_metrics["per_label"][LABEL]["f1"] - baseline["per_label"][LABEL]["f1"]
            ),
        },
        "selected_policies": [{"name": policy_name(policy), **policy} for policy in selected],
        "search_history": history,
        "top_single_policy_candidates": [
            {
                "policy_name": row["policy_name"],
                "policy": row["policy"],
                "added_predictions": row["added_predictions"],
                "delta_from_current": row["delta_from_current"],
                "candidate_f1": row["metrics"]["f1"],
                "candidate_equipment_f1": row["metrics"]["per_label"][LABEL]["f1"],
            }
            for row in top_single[:30]
        ],
        "promotion_recommendation": (
            "promote_if_source_integrity_passes"
            if current_metrics["f1"] > baseline["f1"] and current_metrics["per_label"][LABEL]["f1"] > baseline["per_label"][LABEL]["f1"]
            else "do_not_promote_no_official_gain"
        ),
        "claim_boundary": "Runtime-safe static equipment duplicate/child-box policy from existing raster predictions; gold used only offline for policy selection/evaluation.",
        "outputs": {
            "predictions": str(OUT_PREDS.relative_to(ROOT)),
            "report": str(OUT_MD.relative_to(ROOT)),
        },
    }
    OUT_EVAL.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# P260 Equipment Split Policy Decision",
        "",
        "## Summary",
        f"- Baseline P256 F1: `{baseline['f1']:.6f}` (P `{baseline['precision']:.6f}`, R `{baseline['recall']:.6f}`).",
        f"- Candidate F1: `{current_metrics['f1']:.6f}` (P `{current_metrics['precision']:.6f}`, R `{current_metrics['recall']:.6f}`).",
        f"- ΔF1: `{result['delta_vs_p256']['f1']:.6f}`.",
        f"- Equipment F1: `{baseline['per_label'][LABEL]['f1']:.6f}` -> `{current_metrics['per_label'][LABEL]['f1']:.6f}` "
        f"(Δ `{result['delta_vs_p256']['equipment_f1']:.6f}`).",
        f"- Promotion recommendation: `{result['promotion_recommendation']}`.",
        "",
        "## Selected Policies",
    ]
    if selected:
        for policy in selected:
            lines.append(f"- `{policy_name(policy)}`")
    else:
        lines.append("- No policy selected.")
    lines.extend(["", "## Search History"])
    if history:
        for row in history:
            delta = row["delta_from_current"]
            lines.append(
                f"- `{row['policy_name']}` added {row['added_predictions']} predictions: "
                f"ΔTP `{delta['tp']}`, ΔF1 `{delta['f1']:.6f}`, Δequipment-F1 `{delta['equipment_f1']:.6f}`, "
                f"added precision `{delta['added_precision']:.6f}`."
            )
    else:
        lines.append("- No positive official F1 step survived the precision gate.")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "- The runtime policy uses only existing equipment prediction geometry, score, and local predicted-neighbor counts.",
            "- Gold boxes are used only for offline selection/evaluation.",
            "- P259 diagnostic upper bounds remain diagnostic only and are not official promoted metrics.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "wrote": [
                    str(OUT_EVAL.relative_to(ROOT)),
                    str(OUT_PREDS.relative_to(ROOT)),
                    str(OUT_MD.relative_to(ROOT)),
                ],
                "baseline_f1": baseline["f1"],
                "candidate_f1": current_metrics["f1"],
                "delta_f1": result["delta_vs_p256"]["f1"],
                "selected": len(selected),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
