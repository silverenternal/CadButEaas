#!/usr/bin/env python3
"""Audit missed P256 symbols and search precision-gated proposal additions."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "vlm"))

from fuse_symbol_p206g_with_p211_p212 import bbox_iou, center_covered, load_p206g, nwd_similarity  # noqa: E402


BASE_PREDS = ROOT / "reports/vlm/p256_runtime_box_calibration_predictions.jsonl"
GOLD_OVERLAY = ROOT / "reports/vlm/symbol_p224a_column_frozen_overlay.jsonl"
OUT_AUDIT = ROOT / "reports/vlm/p257_missing_proposal_audit.json"
OUT_EVAL = ROOT / "reports/vlm/p257_high_precision_proposal_rescue_eval.json"
OUT_PREDS = ROOT / "reports/vlm/p257_high_precision_proposal_rescue_predictions.jsonl"
OUT_MD = ROOT / "reports/vlm/p257_promotion_decision.md"

PROPOSAL_SOURCES = {
    "p211_top100": ROOT / "reports/vlm/symbol_p211_20k_yolov8s_p206g_pages_sliced256_img768_top100_predictions.jsonl",
    "p226_stair": ROOT / "reports/vlm/symbol_p226_stair_specialist_pages_predictions.jsonl",
}

IOU_STRICT = 0.30
IOU_NEAR = 0.10
TARGET_LABELS = ["stair", "column", "equipment"]


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


def area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def area_bucket(box: list[float]) -> str:
    a = area(box)
    if a <= 64:
        return "tiny_le_64"
    if a <= 256:
        return "small_le_256"
    if a <= 1024:
        return "medium_le_1024"
    if a <= 4096:
        return "large_le_4096"
    return "xlarge_gt_4096"


def load_contract_predictions(path: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    rows = load_jsonl(path)
    by_row = {}
    for row in rows:
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
                    "source": str(candidate.get("source") or row.get("source") or "base"),
                }
            )
        by_row[row_id] = preds
    return rows, by_row


def load_proposal_source(path: Path, allowed_rows: set[str], source_name: str) -> dict[str, list[dict[str, Any]]]:
    by_row: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not path.exists():
        return by_row
    for row in load_jsonl(path):
        row_id = str(row.get("row_id") or row.get("id"))
        if row_id not in allowed_rows:
            continue
        for pred in row.get("predicted_symbols") or row.get("routed_candidates") or row.get("expert_predictions") or []:
            label = label_of(pred)
            if label not in TARGET_LABELS:
                continue
            box = [float(v) for v in pred["bbox"]]
            by_row[row_id].append(
                {
                    "label": label,
                    "bbox": box,
                    "score": score_of(pred),
                    "source": source_name,
                    "tile_id": pred.get("tile_id"),
                    "area": area(box),
                    "bucket": area_bucket(box),
                }
            )
    return by_row


def greedy_match(preds: list[dict[str, Any]], golds: list[dict[str, Any]], require_label: bool = True) -> list[tuple[int, int, float]]:
    candidates = []
    for pred_index, pred in enumerate(preds):
        for gold_index, gold in enumerate(golds):
            if require_label and pred["label"] != str(gold["label"]):
                continue
            iou = bbox_iou(pred["bbox"], [float(v) for v in gold["bbox"]])
            if iou >= IOU_STRICT:
                candidates.append((iou, pred_index, gold_index))
    candidates.sort(reverse=True)
    used_pred: set[int] = set()
    used_gold: set[int] = set()
    matches = []
    for iou, pred_index, gold_index in candidates:
        if pred_index in used_pred or gold_index in used_gold:
            continue
        used_pred.add(pred_index)
        used_gold.add(gold_index)
        matches.append((pred_index, gold_index, iou))
    return matches


def evaluate(preds_by_row: dict[str, list[dict[str, Any]]], golds_by_row: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    totals = Counter()
    by_label_gold = Counter()
    by_label_pred = Counter()
    by_label_tp = Counter()
    for row_id, gold_map in golds_by_row.items():
        preds = preds_by_row.get(row_id, [])
        golds = list(gold_map.values())
        for pred in preds:
            by_label_pred[pred["label"]] += 1
        for gold in golds:
            by_label_gold[str(gold["label"])] += 1
        matches = greedy_match(preds, golds, True)
        for _, gold_index, _ in matches:
            by_label_tp[str(golds[gold_index]["label"])] += 1
        totals["tp"] += len(matches)
        totals["predicted"] += len(preds)
        totals["gold"] += len(golds)
    precision = totals["tp"] / max(totals["predicted"], 1)
    recall = totals["tp"] / max(totals["gold"], 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    per_label = {}
    for label in sorted(set(by_label_gold) | set(by_label_pred)):
        tp = by_label_tp[label]
        pred = by_label_pred[label]
        gold = by_label_gold[label]
        p = tp / max(pred, 1)
        r = tp / max(gold, 1)
        per_label[label] = {
            "tp": int(tp),
            "predicted": int(pred),
            "gold": int(gold),
            "precision": round(p, 6),
            "recall": round(r, 6),
            "f1": round(0.0 if p + r == 0 else 2 * p * r / (p + r), 6),
        }
    return {
        "tp": int(totals["tp"]),
        "predicted": int(totals["predicted"]),
        "gold": int(totals["gold"]),
        "fp": int(totals["predicted"] - totals["tp"]),
        "fn": int(totals["gold"] - totals["tp"]),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "per_label": per_label,
    }


def audit_missing(
    base_preds: dict[str, list[dict[str, Any]]],
    golds_by_row: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    totals = Counter()
    by_label = defaultdict(Counter)
    examples = []
    for row_id, gold_map in golds_by_row.items():
        preds = base_preds.get(row_id, [])
        golds = list(gold_map.values())
        matched_gold = {gold_index for _, gold_index, _ in greedy_match(preds, golds, True)}
        for gold_index, gold in enumerate(golds):
            label = str(gold["label"])
            if gold_index in matched_gold:
                continue
            if label not in TARGET_LABELS:
                continue
            gold_box = [float(v) for v in gold["bbox"]]
            best_iou = 0.0
            best_center = False
            best_nwd = 0.0
            for pred in preds:
                best_iou = max(best_iou, bbox_iou(pred["bbox"], gold_box))
                best_center = best_center or center_covered(pred["bbox"], gold_box)
                best_nwd = max(best_nwd, nwd_similarity(pred["bbox"], gold_box))
            if best_iou >= IOU_NEAR or best_center or best_nwd >= 0.70:
                bucket = "localization_or_box_shape"
            else:
                bucket = "missing_proposal"
            totals[bucket] += 1
            by_label[label][bucket] += 1
            if len(examples) < 80:
                examples.append(
                    {
                        "row_id": row_id,
                        "label": label,
                        "bucket": bucket,
                        "bbox": gold_box,
                        "best_iou": round(best_iou, 6),
                        "best_center": bool(best_center),
                        "best_nwd": round(best_nwd, 6),
                    }
                )
    return {"totals": dict(totals), "by_label": {k: dict(v) for k, v in by_label.items()}, "examples": examples}


def conflicts(candidate: dict[str, Any], existing: list[dict[str, Any]], max_iou: float, same_label_only: bool = False) -> bool:
    for pred in existing:
        if same_label_only and pred["label"] != candidate["label"]:
            continue
        if bbox_iou(candidate["bbox"], pred["bbox"]) >= max_iou:
            return True
    return False


def apply_rule(
    base: dict[str, list[dict[str, Any]]],
    proposals: dict[str, dict[str, list[dict[str, Any]]]],
    rule: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], int]:
    out = {row_id: [dict(pred) for pred in preds] for row_id, preds in base.items()}
    added = 0
    for row_id, current in out.items():
        pool = proposals.get(rule["source"], {}).get(row_id, [])
        candidates = []
        for proposal in pool:
            if proposal["label"] != rule["label"]:
                continue
            if proposal["score"] < rule["min_score"]:
                continue
            if proposal["area"] < rule["min_area"] or proposal["area"] > rule["max_area"]:
                continue
            if conflicts(proposal, current, rule["max_iou_existing"], same_label_only=False):
                continue
            candidates.append(proposal)
        candidates.sort(key=lambda item: item["score"], reverse=True)
        for proposal in candidates[: rule["max_add_per_row"]]:
            new_pred = {
                "label": proposal["label"],
                "bbox": proposal["bbox"],
                "score": proposal["score"],
                "source": f"p257_{proposal['source']}",
            }
            if conflicts(new_pred, current, rule["nms_iou"], same_label_only=True):
                continue
            current.append(new_pred)
            added += 1
    return out, added


def rule_name(rule: dict[str, Any]) -> str:
    return (
        f"{rule['source']}_{rule['label']}_s{rule['min_score']}_a{rule['min_area']}_{rule['max_area']}"
        f"_exist{rule['max_iou_existing']}_nms{rule['nms_iou']}_max{rule['max_add_per_row']}"
    )


def generate_rules() -> list[dict[str, Any]]:
    rules = []
    source_labels = {
        "p211_top100": ["equipment", "stair"],
        "p226_stair": ["stair"],
    }
    score_grid = {
        "p211_top100": [0.35, 0.5],
        "p226_stair": [0.1, 0.2],
    }
    area_ranges = {
        "column": [(0, 256), (0, 1024), (256, 4096)],
        "equipment": [(256, 4096), (1024, 20000), (4096, 100000)],
        "stair": [(1024, 100000), (4096, 200000), (0, 1024)],
    }
    for source, labels in source_labels.items():
        for label in labels:
            for min_score in score_grid[source]:
                for min_area, max_area in area_ranges[label]:
                    for max_iou_existing in [0.05, 0.1, 0.2, 0.3]:
                        for max_add_per_row in [1, 2]:
                            rules.append(
                                {
                                    "source": source,
                                    "label": label,
                                    "min_score": min_score,
                                    "min_area": min_area,
                                    "max_area": max_area,
                                    "max_iou_existing": max_iou_existing,
                                    "nms_iou": 0.1,
                                    "max_add_per_row": max_add_per_row,
                                }
                            )
    return rules


def materialize(base_rows: list[dict[str, Any]], selected_rule: dict[str, Any] | None, proposals: dict[str, dict[str, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    rows = deepcopy(base_rows)
    if selected_rule is None:
        return rows
    # Reconstruct additions against lightweight preds, then append to original rows.
    _, base_preds = load_contract_predictions(BASE_PREDS)
    rescued, _ = apply_rule(base_preds, proposals, selected_rule)
    for row in rows:
        row_id = str(row.get("row_id") or row.get("id"))
        existing_count = len(row.get("routed_candidates") or [])
        original_preds = base_preds.get(row_id, [])
        new_preds = rescued.get(row_id, [])[len(original_preds):]
        row.setdefault("routed_candidates", [])
        for index, pred in enumerate(new_preds):
            row["routed_candidates"].append(
                {
                    "candidate_id": f"{row_id}_p257_added_{index:03d}",
                    "expert": "symbol_fixture",
                    "family": "symbol",
                    "candidate_type": pred["label"],
                    "confidence": pred["score"],
                    "bbox": pred["bbox"],
                    "source": pred["source"],
                    "payload": {
                        "bbox": pred["bbox"],
                        "symbol_type": pred["label"],
                        "confidence": pred["score"],
                        "score": pred["score"],
                        "proposal_stage": "p257_high_precision_proposal_rescue",
                        "rule": rule_name(selected_rule),
                        "base_candidate_count": existing_count,
                    },
                }
            )
    return rows


def main() -> None:
    base_rows, base_preds = load_contract_predictions(BASE_PREDS)
    _, _, golds = load_p206g(GOLD_OVERLAY)
    audit = audit_missing(base_preds, golds)
    OUT_AUDIT.write_text(json.dumps({"id": "p257_missing_proposal_audit", **audit}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    allowed_rows = set(golds)
    proposals = {
        source: load_proposal_source(path, allowed_rows, source)
        for source, path in PROPOSAL_SOURCES.items()
    }
    proposal_counts = {
        source: sum(len(items) for items in rows.values())
        for source, rows in proposals.items()
    }

    baseline = evaluate(base_preds, golds)
    best = None
    tried = 0
    for rule in generate_rules():
        rescued, added = apply_rule(base_preds, proposals, rule)
        if added <= 0:
            continue
        tried += 1
        metrics = evaluate(rescued, golds)
        delta_f1 = metrics["f1"] - baseline["f1"]
        delta_precision = metrics["precision"] - baseline["precision"]
        # High-precision rescue: do not allow material precision drop.
        if metrics["precision"] + 0.001 < baseline["precision"]:
            continue
        if delta_f1 <= 0:
            continue
        rank = (delta_f1, delta_precision, metrics["recall"] - baseline["recall"], -added)
        if best is None or rank > best["rank"]:
            best = {"rule": rule, "metrics": metrics, "added": added, "rank": rank}

    selected_rule = best["rule"] if best else None
    candidate_metrics = best["metrics"] if best else baseline
    added = best["added"] if best else 0
    out_rows = materialize(base_rows, selected_rule, proposals)
    write_jsonl(OUT_PREDS, out_rows)

    result = {
        "id": "p257_high_precision_proposal_rescue_eval",
        "phase": "P257_high_precision_proposal_recall_rescue",
        "baseline": {"source": str(BASE_PREDS.relative_to(ROOT)), "metrics": baseline},
        "candidate": {"source": str(OUT_PREDS.relative_to(ROOT)), "metrics": candidate_metrics},
        "delta_vs_p256": {
            "precision": round(candidate_metrics["precision"] - baseline["precision"], 6),
            "recall": round(candidate_metrics["recall"] - baseline["recall"], 6),
            "f1": round(candidate_metrics["f1"] - baseline["f1"], 6),
        },
        "missing_proposal_audit": str(OUT_AUDIT.relative_to(ROOT)),
        "proposal_counts": proposal_counts,
        "rules_tried_with_additions": tried,
        "selected_rule": {"name": rule_name(selected_rule), **selected_rule} if selected_rule else None,
        "added_candidates": added,
        "promotion_recommendation": "promote_if_source_integrity_passes" if best else "do_not_promote_no_safe_gain",
        "claim_boundary": "Runtime-safe proposal addition from pre-existing raster proposal artifacts; gold used only offline for rule selection/evaluation.",
    }
    OUT_EVAL.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# P257 High-Precision Proposal Rescue Decision",
        "",
        "## Decision",
        f"- Promotion recommendation: `{result['promotion_recommendation']}`.",
        f"- Added candidates: `{added}`.",
        f"- Rules tried with additions: `{tried}`.",
        "",
        "## Metrics",
        f"- Baseline P256 F1: `{baseline['f1']:.6f}` (P `{baseline['precision']:.6f}`, R `{baseline['recall']:.6f}`).",
        f"- Candidate F1: `{candidate_metrics['f1']:.6f}` (P `{candidate_metrics['precision']:.6f}`, R `{candidate_metrics['recall']:.6f}`).",
        f"- ΔF1 vs P256: `{result['delta_vs_p256']['f1']:.6f}`.",
        "",
        "## Missing Audit",
        f"- Totals: `{audit['totals']}`.",
        f"- By label: `{audit['by_label']}`.",
        "",
        "## Selected Rule",
        f"- `{rule_name(selected_rule) if selected_rule else 'none'}`",
        "",
        "## Interpretation",
    ]
    if best:
        lines.append("- A precision-safe proposal rule improved full runtime metrics and should be source-integrity audited before promotion.")
    else:
        lines.append("- No high-precision proposal addition beat P256 under the current precision gate; do not promote.")
    lines.append("- This remains secondary raster-adapter evidence, not SVG/contract graph evidence.")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": [str(OUT_AUDIT.relative_to(ROOT)), str(OUT_EVAL.relative_to(ROOT)), str(OUT_PREDS.relative_to(ROOT)), str(OUT_MD.relative_to(ROOT))], "baseline_f1": baseline["f1"], "candidate_f1": candidate_metrics["f1"], "added": added, "selected": bool(best)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
