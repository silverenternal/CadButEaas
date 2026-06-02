#!/usr/bin/env python3
"""Audit whether P205b detector candidates can recover P202 false negatives."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import sweep_symbol_disagreement_backfill_p165 as p165
from fuse_symbol_detector_with_p182_p186 import detector_by_row, load_jsonl, write_json


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("row_id") or row.get("id"))


def target_symbols_with_label(row: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for idx, item in enumerate((row.get("targets") or {}).get("symbol") or []):
        box = p165.bbox4(item.get("bbox"))
        if box is None:
            continue
        label = str(item.get("semantic_type") or item.get("symbol_type") or item.get("label") or item.get("raw_label") or "generic_symbol")
        out.append({"id": str(item.get("target_id") or idx), "bbox": box, "bucket": p165.bucket(box), "label": label})
    return out


def greedy_matched_gold_ids(golds: list[dict[str, Any]], preds: list[dict[str, Any]], threshold: float = 0.30) -> set[int]:
    used_pred: set[int] = set()
    matched: set[int] = set()
    for gi, gold in enumerate(golds):
        best_idx = None
        best_iou = 0.0
        for pi, pred in enumerate(preds):
            if pi in used_pred:
                continue
            overlap = p165.iou(pred["bbox"], gold["bbox"])
            if overlap > best_iou:
                best_iou = overlap
                best_idx = pi
        if best_idx is not None and best_iou >= threshold:
            used_pred.add(best_idx)
            matched.add(gi)
    return matched


def center_hit(pred: dict[str, Any], gold: dict[str, Any]) -> bool:
    return p165.center_covered(pred["bbox"], gold["bbox"])


def audit(rows: list[dict[str, Any]], detector: dict[str, list[dict[str, Any]]], thresholds: list[float]) -> dict[str, Any]:
    baseline_maps = {row_id(row): p165.normalized(row.get("symbol_candidates") or [], "p202") for row in rows}
    overall = {}
    examples = []
    for score_threshold in thresholds:
        totals = Counter()
        by_label = defaultdict(Counter)
        by_bucket = defaultdict(Counter)
        for row in rows:
            rid = row_id(row)
            golds = target_symbols_with_label(row)
            baseline = baseline_maps[rid]
            matched = greedy_matched_gold_ids(golds, baseline)
            fns = [gold for gi, gold in enumerate(golds) if gi not in matched]
            dets = [cand for cand in detector.get(rid, []) if float(cand.get("score") or 0.0) >= score_threshold]
            totals["baseline_fn"] += len(fns)
            totals["detector_candidates"] += len(dets)
            used_det_for_fn: set[int] = set()
            for gold in fns:
                totals["fn_gold"] += 1
                by_label[gold["label"]]["fn"] += 1
                by_bucket[gold["bucket"]]["fn"] += 1
                best_iou = 0.0
                best_idx = None
                best_center = False
                for idx, det in enumerate(dets):
                    overlap = p165.iou(det["bbox"], gold["bbox"])
                    if overlap > best_iou:
                        best_iou = overlap
                        best_idx = idx
                    best_center = best_center or center_hit(det, gold)
                if best_iou >= 0.30:
                    totals["recoverable_iou"] += 1
                    by_label[gold["label"]]["recoverable_iou"] += 1
                    by_bucket[gold["bucket"]]["recoverable_iou"] += 1
                    if best_idx is not None:
                        used_det_for_fn.add(best_idx)
                    if len(examples) < 30:
                        examples.append({"threshold": score_threshold, "row_id": rid, "label": gold["label"], "bucket": gold["bucket"], "best_iou": round(best_iou, 4), "gold_bbox": gold["bbox"], "det_bbox": dets[best_idx]["bbox"] if best_idx is not None else None, "det_label": dets[best_idx]["label"] if best_idx is not None else None, "det_score": round(float(dets[best_idx].get("score") or 0.0), 6) if best_idx is not None else None})
                if best_center:
                    totals["recoverable_center"] += 1
                    by_label[gold["label"]]["recoverable_center"] += 1
                    by_bucket[gold["bucket"]]["recoverable_center"] += 1
            for idx, det in enumerate(dets):
                if idx in used_det_for_fn:
                    continue
                if not any(p165.iou(det["bbox"], gold["bbox"]) >= 0.30 for gold in fns):
                    totals["non_recovering_candidates"] += 1
        overall[str(score_threshold)] = {
            "threshold": score_threshold,
            "totals": dict(totals),
            "recoverable_iou_rate": round(totals["recoverable_iou"] / max(totals["fn_gold"], 1), 6),
            "recoverable_center_rate": round(totals["recoverable_center"] / max(totals["fn_gold"], 1), 6),
            "non_recovering_per_iou_recovery": round(totals["non_recovering_candidates"] / max(totals["recoverable_iou"], 1), 6),
            "by_label": {k: dict(v) for k, v in sorted(by_label.items())},
            "by_bucket": {k: dict(v) for k, v in sorted(by_bucket.items())},
        }
    return {"thresholds": overall, "examples": examples}


def render(report: dict[str, Any]) -> str:
    lines = ["# P206b P205b Recall Oracle Audit", "", "## Threshold Summary", "", "| Detector Score | Baseline FN | Recoverable IoU | Recoverable Center | Detector Candidates | Non-Recovering / IoU Recovery |", "|---:|---:|---:|---:|---:|---:|"]
    for key, item in report["audit"]["thresholds"].items():
        t = item["totals"]
        lines.append(f"| {float(key):.3f} | {t.get('fn_gold', 0)} | {t.get('recoverable_iou', 0)} ({item['recoverable_iou_rate']:.3f}) | {t.get('recoverable_center', 0)} ({item['recoverable_center_rate']:.3f}) | {t.get('detector_candidates', 0)} | {item['non_recovering_per_iou_recovery']:.2f} |")
    lines += ["", "## Interpretation", ""]
    lines += [f"- {note}" for note in report["interpretation"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-overlay", default="reports/vlm/symbol_context_verifier_p202_overlay.jsonl")
    parser.add_argument("--detector-predictions", default="reports/vlm/symbol_tiled_recall_p205b_30k_p101_predictions.jsonl")
    parser.add_argument("--out-json", default="reports/vlm/symbol_p205b_recall_oracle_p206b.json")
    parser.add_argument("--out-md", default="reports/vlm/symbol_p205b_recall_oracle_p206b.md")
    args = parser.parse_args()
    rows = load_jsonl(Path(args.base_overlay))
    detector = detector_by_row(Path(args.detector_predictions))
    audit_result = audit(rows, detector, [0.003, 0.005, 0.01, 0.02, 0.05])
    report = {
        "id": "P206b_p205b_recall_oracle_audit",
        "claim_boundary": "Offline oracle audit only. Gold labels identify P202 false negatives and recoverable detector candidates; no runtime policy may use gold.",
        "inputs": {"base_overlay": args.base_overlay, "detector_predictions": args.detector_predictions},
        "audit": audit_result,
        "interpretation": [
            "P205b should only be promoted if enough P202 false negatives have high-quality detector candidates and the non-recovering candidate burden is manageable.",
            "A high non-recovering/recovery ratio explains why raw fusion and simple P200/P202 gating select max_add=0.",
            "If recoverable center is much higher than recoverable IoU, the next module should be a box refiner rather than another classifier gate.",
        ],
    }
    write_json(Path(args.out_json), report)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(render(report), encoding="utf-8")
    print(json.dumps({"outputs": {"json": args.out_json, "md": args.out_md}, "summary": audit_result["thresholds"]}, ensure_ascii=False, indent=2)[:8000])


if __name__ == "__main__":
    main()
