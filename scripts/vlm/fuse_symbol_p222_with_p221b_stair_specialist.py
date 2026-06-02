#!/usr/bin/env python3
"""Fuse P221b stair-specialist page proposals into the P222 frozen baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from freeze_symbol_p222_p221a_sink_tiny import bootstrap, metrics, score_rows
from fuse_symbol_p206g_with_p211_p212 import LABEL_TO_ID, bbox_iou, load_p206g, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports/vlm/symbol_p222_p221a_frozen_overlay.jsonl"
P221B = ROOT / "reports/vlm/symbol_p221b_stair_specialist_page_predictions.jsonl"
REPORT = ROOT / "reports/vlm/symbol_p221b_stair_specialist_fusion_eval.json"
OVERLAY = ROOT / "reports/vlm/symbol_p221b_stair_specialist_fusion_overlay.jsonl"
MD = ROOT / "reports/vlm/symbol_p221b_stair_specialist_fusion_eval.md"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_specialist(path: Path, allowed_rows: set[str], score_field: str = "score", bbox_field: str = "bbox") -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(path):
        rid = str(row.get("row_id"))
        if rid not in allowed_rows:
            continue
        preds = []
        for pred in row.get("predicted_symbols") or []:
            preds.append({
                "bbox": [float(v) for v in pred.get(bbox_field, pred["bbox"])],
                "label": "stair",
                "label_id": LABEL_TO_ID.get("stair", 8),
                "score": float(pred.get(score_field, pred.get("score", 0.0)) or 0.0),
                "raw_score": float(pred.get("score", 0.0) or 0.0),
                "verifier_score": float(pred.get("verifier_score", 0.0) or 0.0),
                "fused_score": float(pred.get("fused_score", pred.get("score", 0.0)) or 0.0),
                "source": "p221b_stair_specialist_added",
                "tile_id": pred.get("tile_id"),
            })
        out[rid] = preds
    return out


def center_dist(left: list[float], right: list[float]) -> float:
    lcx = (left[0] + left[2]) / 2.0
    lcy = (left[1] + left[3]) / 2.0
    rcx = (right[0] + right[2]) / 2.0
    rcy = (right[1] + right[3]) / 2.0
    return ((lcx - rcx) ** 2 + (lcy - rcy) ** 2) ** 0.5


def conflict(candidate: dict[str, Any], existing: list[dict[str, Any]], policy: dict[str, Any]) -> bool:
    cbox = [float(v) for v in candidate["bbox"]]
    same_label_only = bool(policy.get("same_label_only", False))
    for pred in existing:
        if same_label_only and str(pred.get("label")) != "stair":
            continue
        pbox = [float(v) for v in pred["bbox"]]
        if bbox_iou(cbox, pbox) >= float(policy["max_iou_to_existing"]):
            return True
        if center_dist(cbox, pbox) <= float(policy["min_center_dist_to_existing"]):
            return True
    return False


def fuse(core: dict[str, list[dict[str, Any]]], specialist: dict[str, list[dict[str, Any]]], policy: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    threshold = float(policy["score_threshold"])
    max_add = int(policy["max_add_per_row"])
    out: dict[str, list[dict[str, Any]]] = {}
    for rid, core_preds in core.items():
        merged = [dict(pred) for pred in core_preds]
        additions: list[dict[str, Any]] = []
        for pred in sorted(specialist.get(rid, []), key=lambda item: float(item.get("score", 0.0)), reverse=True):
            if float(pred.get("score", 0.0)) < threshold:
                continue
            if conflict(pred, merged + additions, policy):
                continue
            add = dict(pred)
            add["source"] = "p221b_stair_specialist_added"
            additions.append(add)
            if len(additions) >= max_add:
                break
        out[rid] = merged + additions
    return out


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("row_id"))


def build_overlay(rows: list[dict[str, Any]], fused: dict[str, list[dict[str, Any]]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        rid = row_id(row)
        new_row = dict(row)
        candidates = []
        for index, pred in enumerate(fused.get(rid, [])):
            candidates.append({
                "id": f"{rid}_p221b_symbol_{index:05d}",
                "target_id": f"{rid}_p221b_symbol_{index:05d}",
                "symbol_type": pred.get("label"),
                "bbox": pred.get("bbox"),
                "confidence": pred.get("score"),
                "source": pred.get("source"),
                "metadata": {"tile_id": pred.get("tile_id"), "fusion_policy": policy.get("name")},
            })
        new_row["symbol_candidates"] = candidates
        new_row["symbol_policy_overlay"] = {"policy_id": "p221b_stair_specialist_fusion", "policy": policy}
        output.append(new_row)
    return output


def policy_grid() -> list[dict[str, Any]]:
    policies = []
    for threshold in [0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70]:
        for max_add in [1, 2, 3, 5, 8, 12, 20]:
            for max_iou in [0.05, 0.10, 0.20, 0.35, 0.50]:
                for min_dist in [0.0, 4.0, 8.0, 16.0, 32.0]:
                    for same_label_only in [False, True]:
                        policies.append({
                            "name": f"p221b_stair_t{threshold:g}_a{max_add}_iou{max_iou:g}_d{min_dist:g}_same{int(same_label_only)}",
                            "score_threshold": threshold,
                            "max_add_per_row": max_add,
                            "max_iou_to_existing": max_iou,
                            "min_center_dist_to_existing": min_dist,
                            "same_label_only": same_label_only,
                        })
    return policies


def render(report: dict[str, Any]) -> str:
    bm = report["baseline_metrics"]
    sm = report["selected_metrics"]
    boot = report["bootstrap_vs_p222"]
    return "\n".join([
        "# P221b Stair Specialist Page Fusion",
        "",
        "## Metrics",
        "| Variant | F1 | Precision | Recall | TP | Pred | Gold |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| P222 frozen | {bm['f1']:.6f} | {bm['precision']:.6f} | {bm['recall']:.6f} | {bm['tp']} | {bm['predicted']} | {bm['gold']} |",
        f"| P221b fused | {sm['f1']:.6f} | {sm['precision']:.6f} | {sm['recall']:.6f} | {sm['tp']} | {sm['predicted']} | {sm['gold']} |",
        "",
        "## Selected Policy",
        f"- `{report['selected_policy']['name']}`",
        f"- Added predictions: `{report['selected_added_predictions']}`",
        "",
        "## Bootstrap vs P222",
        f"- ΔF1 mean/CI/P>0: `{boot['f1_delta']['mean']:.6f}` / `{boot['f1_delta']['ci95']}` / `{boot['f1_delta']['prob_positive']:.3f}`",
        f"- ΔPrecision mean/CI/P>0: `{boot['precision_delta']['mean']:.6f}` / `{boot['precision_delta']['ci95']}` / `{boot['precision_delta']['prob_positive']:.3f}`",
        f"- ΔRecall mean/CI/P>0: `{boot['recall_delta']['mean']:.6f}` / `{boot['recall_delta']['ci95']}` / `{boot['recall_delta']['prob_positive']:.3f}`",
        "",
        "## Claim Boundary",
        report["claim_boundary"],
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=str(BASE))
    parser.add_argument("--specialist", default=str(P221B))
    parser.add_argument("--report", default=str(REPORT))
    parser.add_argument("--overlay", default=str(OVERLAY))
    parser.add_argument("--md", default=str(MD))
    parser.add_argument("--score-field", default="score", choices=["score", "raw_score", "verifier_score", "fused_score"])
    parser.add_argument("--bbox-field", default="bbox", choices=["bbox", "verifier_bbox"])
    args = parser.parse_args()

    rows, core, golds = load_p206g(Path(args.base))
    ids = [row_id(row) for row in rows]
    specialist = load_specialist(Path(args.specialist), set(ids), args.score_field, args.bbox_field)
    base_per_row = score_rows(core, golds, ids)
    baseline = metrics(base_per_row)
    grid = []
    best = None
    for policy in policy_grid():
        fused = fuse(core, specialist, policy)
        per_row = score_rows(fused, golds, ids)
        m = metrics(per_row)
        added = sum(max(0, len(fused[rid]) - len(core.get(rid, []))) for rid in ids)
        item = {"policy": policy, "metrics": m, "added_predictions": added}
        grid.append(item)
        key = (m["f1"], m["precision"] >= baseline["precision"] - 0.002, m["recall"], -added, m["precision"])
        if best is None or key > best[0]:
            best = (key, item, fused, per_row)
    assert best is not None
    selected = best[1]
    selected_fused = best[2]
    selected_per_row = best[3]
    boot = bootstrap(base_per_row, selected_per_row, seed=221)
    overlay_rows = build_overlay(rows, selected_fused, selected["policy"])
    write_jsonl(Path(args.overlay), overlay_rows)
    grid.sort(key=lambda item: (item["metrics"]["f1"], item["metrics"]["precision"], item["metrics"]["recall"]), reverse=True)
    report = {
        "id": "P221b_stair_specialist_page_fusion",
        "base_overlay": str(Path(args.base).relative_to(ROOT) if Path(args.base).is_relative_to(ROOT) else args.base),
        "specialist_predictions": str(Path(args.specialist).relative_to(ROOT) if Path(args.specialist).is_relative_to(ROOT) else args.specialist),
        "specialist_score_field": args.score_field,
        "specialist_bbox_field": args.bbox_field,
        "baseline_metrics": baseline,
        "selected_policy": selected["policy"],
        "selected_metrics": selected["metrics"],
        "selected_added_predictions": selected["added_predictions"],
        "bootstrap_vs_p222": boot,
        "top_grid": grid[:25],
        "outputs": {
            "overlay": str(Path(args.overlay).relative_to(ROOT) if Path(args.overlay).is_relative_to(ROOT) else args.overlay),
            "markdown": str(Path(args.md).relative_to(ROOT) if Path(args.md).is_relative_to(ROOT) else args.md),
        },
        "promotion_decision": "promote_only_if_precision_CI_non_negative_and_source_audit_passes",
        "claim_boundary": "Internal P101 page-level raster-only fusion against P222; not independent-generalization evidence until held-out page validation is proven compatible.",
    }
    write_json(Path(args.report), report)
    Path(args.md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md).write_text(render(report), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["baseline_metrics", "selected_policy", "selected_metrics", "selected_added_predictions", "bootstrap_vs_p222"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
