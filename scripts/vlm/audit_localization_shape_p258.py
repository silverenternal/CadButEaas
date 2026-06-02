#!/usr/bin/env python3
"""Audit localization and box-shape failure modes after P256/P257."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "vlm"))

from fuse_symbol_p206g_with_p211_p212 import bbox_iou, center_covered, load_p206g, nwd_similarity  # noqa: E402


P256_PREDS = ROOT / "reports/vlm/p256_runtime_box_calibration_predictions.jsonl"
GOLD_OVERLAY = ROOT / "reports/vlm/symbol_p224a_column_frozen_overlay.jsonl"
OUT_JSON = ROOT / "reports/vlm/p258_localization_shape_audit.json"
OUT_MD = ROOT / "reports/vlm/p258_localization_shape_plan.md"

TARGET_LABELS = ["equipment", "stair", "column"]
IOU_STRICT = 0.30


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
    return max((width(box) ** 2 + height(box) ** 2) ** 0.5, 1e-6)


def greedy_typed_matches(preds: list[dict[str, Any]], golds: list[dict[str, Any]]) -> set[int]:
    candidates = []
    for pred_index, pred in enumerate(preds):
        for gold_index, gold in enumerate(golds):
            if pred["label"] != str(gold["label"]):
                continue
            iou = bbox_iou(pred["bbox"], [float(v) for v in gold["bbox"]])
            if iou >= IOU_STRICT:
                candidates.append((iou, pred_index, gold_index))
    candidates.sort(reverse=True)
    used_pred: set[int] = set()
    matched_gold: set[int] = set()
    for _, pred_index, gold_index in candidates:
        if pred_index in used_pred or gold_index in matched_gold:
            continue
        used_pred.add(pred_index)
        matched_gold.add(gold_index)
    return matched_gold


def nearest_same_label(preds: list[dict[str, Any]], gold: dict[str, Any]) -> dict[str, Any] | None:
    label = str(gold["label"])
    gold_box = [float(v) for v in gold["bbox"]]
    best = None
    for pred in preds:
        if pred["label"] != label:
            continue
        pred_box = pred["bbox"]
        iou = bbox_iou(pred_box, gold_box)
        pc = center(pred_box)
        gc = center(gold_box)
        center_dist = ((pc[0] - gc[0]) ** 2 + (pc[1] - gc[1]) ** 2) ** 0.5
        row = {
            "pred": pred,
            "iou": iou,
            "center_covered": center_covered(pred_box, gold_box),
            "nwd": nwd_similarity(pred_box, gold_box),
            "center_dist_norm": center_dist / diag(gold_box),
            "width_ratio_pred_over_gold": width(pred_box) / max(width(gold_box), 1e-6),
            "height_ratio_pred_over_gold": height(pred_box) / max(height(gold_box), 1e-6),
            "area_ratio_pred_over_gold": area(pred_box) / max(area(gold_box), 1e-6),
        }
        rank = (row["iou"], row["nwd"], -row["center_dist_norm"])
        if best is None or rank > best["rank"]:
            best = {"rank": rank, **row}
    return best


def bucket_failure(best: dict[str, Any] | None) -> str:
    if best is None:
        return "no_same_label_prediction"
    if best["iou"] >= IOU_STRICT:
        return "duplicate_or_matching_conflict"
    if best["center_covered"] and best["iou"] < IOU_STRICT:
        if best["area_ratio_pred_over_gold"] < 0.65:
            return "center_hit_pred_too_small"
        if best["area_ratio_pred_over_gold"] > 1.55:
            return "center_hit_pred_too_large"
        if best["width_ratio_pred_over_gold"] < 0.65 or best["height_ratio_pred_over_gold"] < 0.65:
            return "center_hit_aspect_mismatch"
        return "center_hit_iou_low"
    if best["iou"] >= 0.10:
        return "near_iou_low_overlap"
    return "missing_or_far_same_label"


def median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return round(float(median(values)), 6)


def main() -> None:
    preds_by_row = load_predictions(P256_PREDS)
    _, _, golds_by_row = load_p206g(GOLD_OVERLAY)

    label_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    row_cluster_counts: dict[str, Counter] = defaultdict(Counter)

    for row_id, gold_map in golds_by_row.items():
        preds = preds_by_row.get(row_id, [])
        golds = list(gold_map.values())
        matched = greedy_typed_matches(preds, golds)
        for gold_index, gold in enumerate(golds):
            label = str(gold["label"])
            if label not in TARGET_LABELS or gold_index in matched:
                continue
            best = nearest_same_label(preds, gold)
            failure = bucket_failure(best)
            gold_box = [float(v) for v in gold["bbox"]]
            row = {
                "row_id": row_id,
                "label": label,
                "failure": failure,
                "gold_bbox": gold_box,
                "gold_area": round(area(gold_box), 6),
                "best_iou": round(best["iou"], 6) if best else 0.0,
                "center_covered": bool(best["center_covered"]) if best else False,
                "nwd": round(best["nwd"], 6) if best else 0.0,
                "center_dist_norm": round(best["center_dist_norm"], 6) if best else None,
                "width_ratio_pred_over_gold": round(best["width_ratio_pred_over_gold"], 6) if best else None,
                "height_ratio_pred_over_gold": round(best["height_ratio_pred_over_gold"], 6) if best else None,
                "area_ratio_pred_over_gold": round(best["area_ratio_pred_over_gold"], 6) if best else None,
                "pred_bbox": best["pred"]["bbox"] if best else None,
                "pred_score": round(best["pred"]["score"], 6) if best else None,
            }
            label_rows[label].append(row)
            row_cluster_counts[label][row_id] += 1
            if len(examples[label]) < 30:
                examples[label].append(row)

    summary = {}
    for label, rows in label_rows.items():
        failures = Counter(row["failure"] for row in rows)
        center_rows = [row for row in rows if row["center_covered"]]
        ratios = {
            "width_ratio_median": median_or_none([row["width_ratio_pred_over_gold"] for row in rows if row["width_ratio_pred_over_gold"] is not None]),
            "height_ratio_median": median_or_none([row["height_ratio_pred_over_gold"] for row in rows if row["height_ratio_pred_over_gold"] is not None]),
            "area_ratio_median": median_or_none([row["area_ratio_pred_over_gold"] for row in rows if row["area_ratio_pred_over_gold"] is not None]),
            "best_iou_median": median_or_none([row["best_iou"] for row in rows]),
            "center_hit_area_ratio_median": median_or_none([row["area_ratio_pred_over_gold"] for row in center_rows if row["area_ratio_pred_over_gold"] is not None]),
        }
        top_rows = row_cluster_counts[label].most_common(10)
        summary[label] = {
            "unmatched_count": len(rows),
            "failure_buckets": dict(failures),
            "ratio_summary": ratios,
            "top_rows_by_unmatched_count": [{"row_id": row_id, "count": count} for row_id, count in top_rows],
        }

    result = {
        "id": "p258_localization_shape_audit",
        "phase": "P258_localization_box_shape_modeling",
        "source_predictions": str(P256_PREDS.relative_to(ROOT)),
        "gold_overlay": str(GOLD_OVERLAY.relative_to(ROOT)),
        "summary": summary,
        "examples": examples,
        "recommendations": {
            "equipment": [
                "Primary issue is localization/shape around existing predictions, not proposal absence.",
                "Investigate one-to-many gold annotations around one predicted equipment object before adding proposals.",
                "Any transform must be instance-aware; global scaling already produced only +0.000753 F1 in P256.",
            ],
            "stair": [
                "Split sentinel placeholders from visual stair structures before another proposal model.",
                "Generic stair proposal fusion has already failed under P234/P257 precision gates.",
            ],
            "column": [
                "Many column misses are true missing proposals; use a dedicated tiny-square/vertical-column detector or rule only if source-safe.",
                "Avoid large broad detector proposal fusion because precision risk is high.",
            ],
        },
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# P258 Localization / Box-Shape Plan",
        "",
        "## Summary",
    ]
    for label, row in summary.items():
        lines.append(f"- `{label}`: unmatched `{row['unmatched_count']}`, buckets `{row['failure_buckets']}`, ratios `{row['ratio_summary']}`.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "- `equipment` is primarily a localization/shape and possible one-to-many annotation problem; adding proposals is not the right first move.",
            "- `stair` remains representation-mixed; generic proposal fusion has repeatedly failed precision gates.",
            "- `column` has genuine missing proposals and likely needs a constrained tiny-column specialist rather than broad proposal fusion.",
            "",
            "## Next Experiment Recommendation",
            "- P259 should audit one-to-many equipment clusters and test instance merging/suppression-aware scoring before more box scaling.",
            "- P260 should split stair sentinel placeholders from visual stair structures and evaluate them separately.",
            "- P261 should consider a tiny-column specialist only if it can be constrained to high precision.",
            "",
            "## Do Not Do",
            "- Do not repeat p211/p226/p228 stair fusion unchanged.",
            "- Do not use SVG/gold/parser fields at runtime.",
            "- Do not mix these raster adapter diagnostics with SVG/contract main paper metrics.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": [str(OUT_JSON.relative_to(ROOT)), str(OUT_MD.relative_to(ROOT))], "labels": list(summary)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
