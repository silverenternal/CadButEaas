#!/usr/bin/env python3
"""Compact fusion of P205b tiled recall detector over current P202 overlay."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import sweep_symbol_disagreement_backfill_p165 as p165
from fuse_symbol_detector_with_p182_p186 import detector_by_row, load_jsonl, materialize, write_json, write_jsonl

TARGET_LABELS = {"sink", "shower", "equipment", "stair", "appliance"}
WET_LABELS = {"sink", "shower", "bathtub"}


def policies() -> list[dict[str, Any]]:
    out = []
    for labels in [WET_LABELS, TARGET_LABELS]:
        for min_score in [0.01, 0.02, 0.05]:
            for max_iou in [0.03, 0.08, 0.16]:
                for min_dist in [8, 16]:
                    for max_add in [0, 1, 2, 3, 5]:
                        for max_total in [64, 96]:
                            out.append({
                                "name": f"p205b_l{len(labels)}_s{min_score}_iou{max_iou}_d{min_dist}_a{max_add}_t{max_total}",
                                "labels": sorted(labels),
                                "min_score": min_score,
                                "max_iou_to_core": max_iou,
                                "min_center_dist_to_core": min_dist,
                                "max_add_per_row": max_add,
                                "max_total_per_row": max_total,
                                "detector_nms": 0.55,
                                "score_scale": 1.0,
                            })
    return out


def nms(preds: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    kept = []
    for pred in sorted(preds, key=lambda item: float(item.get("score") or 0.0), reverse=True):
        if all(p165.iou(pred["bbox"], old["bbox"]) < threshold for old in kept):
            kept.append(pred)
    return kept


def fuse_row(core: list[dict[str, Any]], detector: list[dict[str, Any]], policy: dict[str, Any], row_id: str) -> list[dict[str, Any]]:
    labels = set(policy["labels"])
    candidates = []
    for cand in nms(detector, float(policy["detector_nms"])):
        if cand["label"] not in labels:
            continue
        if float(cand.get("score") or 0.0) < float(policy["min_score"]):
            continue
        best_iou, best_dist = p165.best_overlap_to_core(cand, core)
        if best_iou >= float(policy["max_iou_to_core"]):
            continue
        if best_dist < float(policy["min_center_dist_to_core"]):
            continue
        item = copy.deepcopy(cand)
        item["row_id"] = row_id
        item["best_iou_to_core"] = best_iou
        item["min_center_dist_to_core"] = best_dist
        candidates.append(item)
    fused = sorted(core, key=lambda item: float(item.get("score") or 0.0), reverse=True)
    fused.extend(sorted(candidates, key=lambda item: float(item.get("score") or 0.0), reverse=True)[: int(policy["max_add_per_row"])])
    return sorted(fused, key=lambda item: float(item.get("score") or 0.0), reverse=True)[: int(policy["max_total_per_row"])]


def render(report: dict[str, Any]) -> str:
    lines = [
        "# P205b Tiled Recall Fusion",
        "",
        f"Decision: **{report['decision']}**",
        "",
        "| Variant | Precision | Recall | F1 | Center | Inflation |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    b = report["baseline_metrics"]
    m = report["best_metrics"]
    lines.append(f"| `P202_baseline` | {b['precision']:.6f} | {b['recall']:.6f} | {b['f1']:.6f} | {b['center_recall']:.6f} | {b['prediction_inflation']:.6f} |")
    lines.append(f"| `P205b_best` | {m['precision']:.6f} | {m['recall']:.6f} | {m['f1']:.6f} | {m['center_recall']:.6f} | {m['prediction_inflation']:.6f} |")
    lines += ["", "## Best Policy", "", "```json", json.dumps(report["best_policy"], ensure_ascii=False, indent=2), "```", "", "## Top Policies", ""]
    for item in report["top_candidates"][:20]:
        x = item["metrics"]
        lines.append(f"- `{item['policy']['name']}` F1 `{x['f1']:.6f}` P `{x['precision']:.6f}` R `{x['recall']:.6f}` center `{x['center_recall']:.6f}` inflation `{x['prediction_inflation']:.6f}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-overlay", default="reports/vlm/symbol_context_verifier_p202_overlay.jsonl")
    parser.add_argument("--detector-predictions", default="reports/vlm/symbol_tiled_recall_p205b_30k_p101_predictions.jsonl")
    parser.add_argument("--output-json", default="configs/vlm/symbol_tiled_recall_p205b_fusion.json")
    parser.add_argument("--output-md", default="reports/vlm/symbol_tiled_recall_p205b_fusion.md")
    parser.add_argument("--output-overlay", default="reports/vlm/symbol_tiled_recall_p205b_fusion_overlay.jsonl")
    args = parser.parse_args()
    rows = load_jsonl(Path(args.base_overlay))
    detector = detector_by_row(Path(args.detector_predictions))
    golds = {str(row.get("row_id") or row.get("id")): p165.target_symbols(row) for row in rows}
    core = {str(row.get("row_id") or row.get("id")): p165.normalized(row.get("symbol_candidates") or [], "p202_core") for row in rows}
    baseline = p165.evaluate(golds, core)
    scored = []
    for policy in policies():
        pred_map = {rid: fuse_row(core.get(rid, []), detector.get(rid, []), policy, rid) for rid in golds}
        scored.append({"policy": policy, "metrics": p165.evaluate(golds, pred_map)})
    scored.sort(key=lambda item: (item["metrics"]["f1"], item["metrics"]["recall"], item["metrics"]["center_recall"], -item["metrics"]["prediction_inflation"]), reverse=True)
    best = scored[0]
    best_map = {rid: fuse_row(core.get(rid, []), detector.get(rid, []), best["policy"], rid) for rid in golds}
    write_jsonl(Path(args.output_overlay), materialize(rows, best_map, best["policy"], Path(args.detector_predictions)))
    report = {
        "id": "P205b_tiled_recall_fusion",
        "claim_boundary": "P101-selected compact fusion of raster-only P205b detector predictions over P202 baseline; gold labels used only for offline evaluation/policy selection.",
        "inputs": {"base_overlay": args.base_overlay, "detector_predictions": args.detector_predictions},
        "detector_row_overlap": {"base_rows": len(rows), "detector_rows": len(detector), "matched_rows": sum(1 for row in rows if str(row.get("row_id") or row.get("id")) in detector)},
        "baseline_metrics": baseline,
        "best_policy": best["policy"],
        "best_metrics": best["metrics"],
        "delta_vs_baseline": p165.delta(best["metrics"], baseline),
        "decision": "promote_candidate" if best["metrics"]["f1"] > baseline["f1"] else "no_promotion_keep_P202",
        "searched_policy_count": len(scored),
        "top_candidates": scored[:50],
        "outputs": {"json": args.output_json, "md": args.output_md, "overlay": args.output_overlay},
    }
    write_json(Path(args.output_json), report)
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text(render(report), encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "baseline": baseline, "best_metrics": best["metrics"], "delta": report["delta_vs_baseline"], "outputs": report["outputs"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
