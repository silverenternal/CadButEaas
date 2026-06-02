#!/usr/bin/env python3
"""Audit P232 runtime symbol errors into proposal, localization, and type buckets."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "vlm"))

from fuse_symbol_p206g_with_p211_p212 import bbox_iou, center_covered, load_p206g, nwd_similarity  # noqa: E402


P232_PREDS = ROOT / "reports/vlm/p232_repaired_contract_predictions.jsonl"
GOLD_OVERLAY = ROOT / "reports/vlm/symbol_p224a_column_frozen_overlay.jsonl"
OUT_JSON = ROOT / "reports/vlm/p246_symbol_error_upper_bounds.json"
OUT_MD = ROOT / "reports/vlm/p246_symbol_error_upper_bounds.md"

IOU_STRICT = 0.30
IOU_NEAR = 0.10
NWD_NEAR = 0.70


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_p232(path: Path) -> dict[str, list[dict[str, Any]]]:
    by_row: dict[str, list[dict[str, Any]]] = {}
    for row in load_jsonl(path):
        row_id = str(row.get("row_id") or row.get("id"))
        preds: list[dict[str, Any]] = []
        for candidate in row.get("routed_candidates") or row.get("predicted_symbols") or []:
            label = str(
                candidate.get("candidate_type")
                or candidate.get("symbol_type")
                or candidate.get("label")
                or (candidate.get("payload") or {}).get("symbol_type")
                or "generic_symbol"
            )
            preds.append(
                {
                    "bbox": [float(v) for v in candidate.get("bbox")],
                    "label": label,
                    "score": float(candidate.get("confidence") or candidate.get("score") or 1.0),
                }
            )
        by_row[row_id] = preds
    return by_row


def prf(tp: int, predicted: int, gold: int) -> dict[str, Any]:
    precision = tp / max(predicted, 1)
    recall = tp / max(gold, 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "tp": int(tp),
        "predicted": int(predicted),
        "gold": int(gold),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def greedy_match(
    preds: list[dict[str, Any]],
    golds: list[dict[str, Any]],
    *,
    iou_threshold: float,
    require_label: bool,
) -> list[tuple[int, int, float]]:
    candidates: list[tuple[float, int, int]] = []
    for pred_index, pred in enumerate(preds):
        for gold_index, gold in enumerate(golds):
            if require_label and str(pred["label"]) != str(gold["label"]):
                continue
            iou = bbox_iou(pred["bbox"], gold["bbox"])
            if iou >= iou_threshold:
                candidates.append((iou, pred_index, gold_index))
    candidates.sort(reverse=True)
    used_preds: set[int] = set()
    used_golds: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, pred_index, gold_index in candidates:
        if pred_index in used_preds or gold_index in used_golds:
            continue
        used_preds.add(pred_index)
        used_golds.add(gold_index)
        matches.append((pred_index, gold_index, iou))
    return matches


def best_overlap(preds: list[dict[str, Any]], gold: dict[str, Any]) -> dict[str, Any]:
    best = {"iou": 0.0, "nwd": 0.0, "center": False, "label": None}
    for pred in preds:
        iou = bbox_iou(pred["bbox"], gold["bbox"])
        nwd = nwd_similarity(pred["bbox"], gold["bbox"])
        center = center_covered(pred["bbox"], gold["bbox"])
        if (iou, nwd, center) > (best["iou"], best["nwd"], best["center"]):
            best = {"iou": iou, "nwd": nwd, "center": center, "label": pred["label"]}
    return best


def summarize_per_label(counter: Counter, total_by_label: Counter) -> dict[str, Any]:
    return {
        label: {
            "count": int(counter[label]),
            "gold": int(total_by_label[label]),
            "rate": round(counter[label] / max(total_by_label[label], 1), 6),
        }
        for label in sorted(total_by_label)
    }


def main() -> None:
    _, _, gold_by_row_map = load_p206g(GOLD_OVERLAY)
    pred_by_row = load_p232(P232_PREDS)
    row_ids = sorted(gold_by_row_map)

    totals = Counter()
    total_gold_by_label: Counter = Counter()
    typed_tp_by_label: Counter = Counter()
    object_tp_by_label: Counter = Counter()
    near_by_label: Counter = Counter()
    center_by_label: Counter = Counter()
    nwd_by_label: Counter = Counter()
    error_by_label: dict[str, Counter] = defaultdict(Counter)

    page_examples: list[dict[str, Any]] = []

    for row_id in row_ids:
        preds = pred_by_row.get(row_id, [])
        golds = list(gold_by_row_map[row_id].values())
        typed_matches = greedy_match(preds, golds, iou_threshold=IOU_STRICT, require_label=True)
        object_matches = greedy_match(preds, golds, iou_threshold=IOU_STRICT, require_label=False)

        typed_gold_indexes = {gold_index for _, gold_index, _ in typed_matches}
        object_gold_indexes = {gold_index for _, gold_index, _ in object_matches}
        totals["predicted"] += len(preds)
        totals["gold"] += len(golds)
        totals["typed_tp"] += len(typed_matches)
        totals["object_tp"] += len(object_matches)

        for _, gold_index, _ in typed_matches:
            typed_tp_by_label[str(golds[gold_index]["label"])] += 1
        for _, gold_index, _ in object_matches:
            object_tp_by_label[str(golds[gold_index]["label"])] += 1

        page_error = Counter()
        for gold_index, gold in enumerate(golds):
            label = str(gold["label"])
            total_gold_by_label[label] += 1
            overlap = best_overlap(preds, gold)
            if overlap["iou"] >= IOU_NEAR:
                near_by_label[label] += 1
            if overlap["center"]:
                center_by_label[label] += 1
            if overlap["nwd"] >= NWD_NEAR:
                nwd_by_label[label] += 1

            if gold_index in typed_gold_indexes:
                continue
            if gold_index in object_gold_indexes:
                bucket = "type_error_on_good_box"
            elif overlap["iou"] >= IOU_NEAR or overlap["center"] or overlap["nwd"] >= NWD_NEAR:
                bucket = "localization_or_box_shape_error"
            else:
                bucket = "missing_proposal"
            error_by_label[label][bucket] += 1
            page_error[bucket] += 1

        if page_error:
            page_examples.append(
                {
                    "row_id": row_id,
                    "gold": len(golds),
                    "predicted": len(preds),
                    "typed_tp": len(typed_matches),
                    "object_tp": len(object_matches),
                    "errors": dict(page_error),
                }
            )

    typed = prf(totals["typed_tp"], totals["predicted"], totals["gold"])
    objectness = prf(totals["object_tp"], totals["predicted"], totals["gold"])
    type_gain = objectness["f1"] - typed["f1"]

    result = {
        "id": "p246_symbol_error_upper_bounds",
        "purpose": "Decompose the P232 runtime raster symbol gap before doing more metric chasing.",
        "inputs": {
            "predictions": str(P232_PREDS.relative_to(ROOT)),
            "gold_overlay": str(GOLD_OVERLAY.relative_to(ROOT)),
            "strict_iou": IOU_STRICT,
            "near_iou": IOU_NEAR,
            "near_nwd": NWD_NEAR,
        },
        "headline": {
            "current_typed_detection": typed,
            "oracle_type_on_existing_good_boxes": objectness,
            "max_f1_gain_from_type_fix_only": round(type_gain, 6),
            "interpretation": (
                "If the objectness upper bound is still far below target, the dominant problem is proposal/localization, "
                "not MoE label arbitration alone."
            ),
        },
        "per_label": {
            label: {
                "gold": int(total_gold_by_label[label]),
                "typed_recall": round(typed_tp_by_label[label] / max(total_gold_by_label[label], 1), 6),
                "objectness_recall_iou_0_30": round(object_tp_by_label[label] / max(total_gold_by_label[label], 1), 6),
                "near_recall_iou_0_10": round(near_by_label[label] / max(total_gold_by_label[label], 1), 6),
                "center_recall": round(center_by_label[label] / max(total_gold_by_label[label], 1), 6),
                "nwd_recall_0_70": round(nwd_by_label[label] / max(total_gold_by_label[label], 1), 6),
                "error_buckets": dict(error_by_label[label]),
            }
            for label in sorted(total_gold_by_label)
        },
        "error_bucket_totals": {
            bucket: int(sum(label_counter[bucket] for label_counter in error_by_label.values()))
            for bucket in ["type_error_on_good_box", "localization_or_box_shape_error", "missing_proposal"]
        },
        "worst_pages": sorted(page_examples, key=lambda row: (sum(row["errors"].values()), row["gold"]), reverse=True)[:20],
        "recommended_route": [
            "Do not spend the next iteration on small geometry scaling rules; P232 already shows only +0.001884 F1.",
            "If oracle-type upper bound is high, prioritize runtime-safe crop/type-head replay on P232 boxes.",
            "If oracle-type upper bound remains low, prioritize proposal recall and localization: candidate generation, heatmap/objectness, and representation-aware stair handling.",
            "For stair, report aggregate honestly but train/evaluate separate visual-object, grouped-structure, sentinel, and tread buckets.",
        ],
    }

    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# P246 Symbol Error Upper-Bound Audit",
        "",
        "## Headline",
        f"- Current typed runtime detection F1: `{typed['f1']:.6f}` (P `{typed['precision']:.6f}`, R `{typed['recall']:.6f}`).",
        f"- Oracle type on existing IoU>=0.30 boxes F1: `{objectness['f1']:.6f}` (P `{objectness['precision']:.6f}`, R `{objectness['recall']:.6f}`).",
        f"- Maximum F1 gain from fixing labels only on already-good boxes: `{type_gain:.6f}`.",
        "- Reviewer implication: if this upper bound is materially below the paper target, the next useful work is candidate/localization, not another MoE-only label tweak.",
        "",
        "## Error Bucket Totals",
    ]
    for bucket, count in result["error_bucket_totals"].items():
        lines.append(f"- `{bucket}`: `{count}`")
    lines.extend(["", "## Per-Label Audit", ""])
    lines.append("| label | gold | typed R | object R | near R | center R | nwd R | main missed bucket |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for label, row in result["per_label"].items():
        buckets = row["error_buckets"]
        main_bucket = max(buckets.items(), key=lambda item: item[1])[0] if buckets else "none"
        lines.append(
            f"| {label} | {row['gold']} | {row['typed_recall']:.6f} | {row['objectness_recall_iou_0_30']:.6f} | "
            f"{row['near_recall_iou_0_10']:.6f} | {row['center_recall']:.6f} | {row['nwd_recall_0_70']:.6f} | {main_bucket} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "- P232 is not a strong final raster result by reviewer standards; it is a baseline plus source-safe repair.",
            "- The next experiment should be chosen by the upper bound: type-head replay if label errors dominate, proposal/localization if objectness upper bound remains low.",
            "- Stair should stop being treated as one homogeneous detection class; the current aggregate mixes visual objects, grouped structures, placeholders, and line-like treads.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": [str(OUT_JSON.relative_to(ROOT)), str(OUT_MD.relative_to(ROOT))], "typed_f1": typed["f1"], "object_f1": objectness["f1"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
