#!/usr/bin/env python3
"""P186 bridge: fuse a GPU symbol detector page-prediction file into the P182 P101 overlay.

The detector predictions are produced by eval_symbol_yolo_tile_detector_v22.py or
similar scripts as JSONL rows: {row_id, predicted_symbols, gold_symbol_count}.
This script never uses gold labels as runtime features; gold is used only for the
offline P101 metric report and policy selection.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import sweep_symbol_disagreement_backfill_p165 as p165

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = ROOT / "reports/vlm/symbol_policy_moe_overlay_p182_best.jsonl"
DEFAULT_OUT_JSON = ROOT / "configs/vlm/symbol_detector_p182_fusion_p186.json"
DEFAULT_OUT_MD = ROOT / "reports/vlm/symbol_detector_p182_fusion_p186.md"
DEFAULT_OUT_OVERLAY = ROOT / "reports/vlm/symbol_policy_moe_overlay_p186_best.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def detector_by_row(path: Path) -> dict[str, list[dict[str, Any]]]:
    by_row: dict[str, list[dict[str, Any]]] = {}
    for row in load_jsonl(path):
        row_id = str(row.get("row_id") or row.get("id"))
        preds = []
        for idx, raw in enumerate(row.get("predicted_symbols") or row.get("symbol_candidates") or []):
            box = p165.bbox4(raw.get("bbox"))
            if box is None:
                continue
            label = str(raw.get("label") or raw.get("symbol_type") or raw.get("semantic_type") or "generic_symbol")
            try:
                score = float(raw.get("score") if raw.get("score") is not None else raw.get("confidence"))
            except (TypeError, ValueError):
                score = 0.0
            preds.append({
                "bbox": box,
                "label": label,
                "score": score,
                "bucket": p165.bucket(box),
                "raw": raw,
                "source_policy": "gpu_detector_p186",
                "source_index": idx,
            })
        by_row[row_id] = preds
    return by_row


def nms(preds: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for pred in sorted(preds, key=lambda item: float(item.get("score") or 0.0), reverse=True):
        if all(p165.iou(pred["bbox"], old["bbox"]) < threshold for old in kept):
            kept.append(pred)
    return kept


def add_detector_features(candidate: dict[str, Any], core: list[dict[str, Any]], row_id: str) -> dict[str, Any]:
    best_iou, best_dist = p165.best_overlap_to_core(candidate, core)
    item = copy.deepcopy(candidate)
    item["row_id"] = row_id
    item["best_iou_to_core"] = best_iou
    item["min_center_dist_to_core"] = best_dist
    return item


def fuse_row(core: list[dict[str, Any]], detector: list[dict[str, Any]], policy: dict[str, Any], row_id: str) -> list[dict[str, Any]]:
    blocked_labels = set(policy.get("blocked_labels") or [])
    blocked_label_buckets = set(policy.get("blocked_label_buckets") or [])
    min_score = float(policy["min_score"])
    max_iou = float(policy["max_iou_to_core"])
    min_dist = float(policy["min_center_dist_to_core"])
    max_add = int(policy["max_add_per_row"])
    max_total = int(policy["max_total_per_row"])
    det_nms = float(policy["detector_nms"])
    score_scale = float(policy.get("score_scale") or 1.0)

    enriched = [add_detector_features(cand, core, row_id) for cand in detector]
    candidates = []
    for cand in nms(enriched, det_nms):
        label = str(cand["label"])
        label_bucket = label + "|" + str(cand["bucket"])
        if label in blocked_labels or label_bucket in blocked_label_buckets:
            continue
        if float(cand["score"]) < min_score:
            continue
        if float(cand["best_iou_to_core"]) > max_iou:
            continue
        if float(cand["min_center_dist_to_core"]) < min_dist:
            continue
        cand = copy.deepcopy(cand)
        cand["score"] = min(1.0, float(cand["score"]) * score_scale)
        candidates.append(cand)
    fused = sorted(core, key=lambda item: float(item.get("score") or 0.0), reverse=True)
    fused.extend(sorted(candidates, key=lambda item: float(item.get("score") or 0.0), reverse=True)[:max_add])
    return sorted(fused, key=lambda item: float(item.get("score") or 0.0), reverse=True)[:max_total]


def policies() -> list[dict[str, Any]]:
    blocked_sets = [
        [],
        ["generic_symbol|medium", "appliance|small", "sink|medium"],
        ["generic_symbol|medium", "appliance|small", "sink|medium", "equipment|tiny", "column|xlarge"],
    ]
    out = []
    for min_score in [0.25, 0.30, 0.35, 0.45, 0.55, 0.65]:
        for max_iou in [0.05, 0.08, 0.12, 0.18]:
            for min_dist in [8, 16, 24, 32]:
                for max_add in [0, 1, 2, 3, 5, 8]:
                    for max_total in [32, 48, 64]:
                        for blocked in blocked_sets:
                            out.append({
                                "name": f"p186_s{min_score}_iou{max_iou}_d{min_dist}_a{max_add}_t{max_total}_blk{len(blocked)}",
                                "min_score": min_score,
                                "max_iou_to_core": max_iou,
                                "min_center_dist_to_core": min_dist,
                                "max_add_per_row": max_add,
                                "max_total_per_row": max_total,
                                "detector_nms": 0.70,
                                "score_scale": 1.0,
                                "blocked_labels": [],
                                "blocked_label_buckets": blocked,
                            })
    unique = {json.dumps(item, sort_keys=True): item for item in out}
    return list(unique.values())


def materialize(base_rows: list[dict[str, Any]], preds_by_row: dict[str, list[dict[str, Any]]], policy: dict[str, Any], detector_path: Path) -> list[dict[str, Any]]:
    rows = []
    for raw in base_rows:
        row = copy.deepcopy(raw)
        row_id = str(row.get("row_id") or row.get("id"))
        candidates = []
        for idx, pred in enumerate(preds_by_row.get(row_id, [])):
            item = copy.deepcopy(pred.get("raw") or {})
            item["bbox"] = pred["bbox"]
            item["symbol_type"] = pred["label"]
            item["confidence"] = float(pred["score"])
            item["id"] = f"{row_id}_p186_best_symbol_{idx:05d}"
            item["target_id"] = item["id"]
            item["source"] = "symbol_detector_p182_fusion_p186"
            metadata = item.setdefault("metadata", {})
            metadata["p186_policy"] = policy["name"]
            metadata["p186_detector_predictions"] = str(detector_path)
            metadata["p186_source_policy"] = pred.get("source_policy")
            candidates.append(item)
        row["symbol_candidates"] = candidates
        if isinstance(row.get("expected_json"), dict):
            row["expected_json"]["symbol_candidates"] = [copy.deepcopy(item) for item in candidates]
        row["symbol_policy_overlay"] = {
            "policy_id": "p186_best",
            "description": "P186 GPU detector bridge fused with P182 precision core",
            "policy": policy,
            "detector_predictions": str(detector_path),
        }
        rows.append(row)
    return rows


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# P186 GPU Detector → P182 Bridge",
        "",
        "Decision: **" + str(report["decision"]) + "**",
        "",
        "## Metrics",
        "",
        "| Variant | Precision | Recall | F1 | Center | Inflation |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant_name, metrics in report["baseline_metrics"].items():
        lines.append(
            "| `" + str(variant_name) + "` | "
            + f"{metrics['precision']:.6f} | {metrics['recall']:.6f} | {metrics['f1']:.6f} | "
            + f"{metrics['center_recall']:.6f} | {metrics['prediction_inflation']:.6f} |"
        )
    best_metrics = report["best_metrics"]
    lines.append(
        "| `p186_best` | "
        + f"{best_metrics['precision']:.6f} | {best_metrics['recall']:.6f} | {best_metrics['f1']:.6f} | "
        + f"{best_metrics['center_recall']:.6f} | {best_metrics['prediction_inflation']:.6f} |"
    )
    lines += [
        "",
        "## Best Policy",
        "",
        "- `" + str(report["best_policy"]["name"]) + "`",
        "- config: `" + json.dumps(report["best_policy"], ensure_ascii=False) + "`",
        "",
        "## Artifacts",
        "",
    ]
    for value in report["outputs"].values():
        lines.append("- `" + str(value) + "`")
    lines += ["", "## Top Candidates", ""]
    for item in report["top_candidates"][:20]:
        metrics = item["metrics"]
        policy_name = item["policy"]["name"]
        lines.append(
            "- `" + str(policy_name) + "` "
            + f"F1 `{metrics['f1']:.6f}`, P `{metrics['precision']:.6f}`, R `{metrics['recall']:.6f}`, "
            + f"inflation `{metrics['prediction_inflation']:.6f}`"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-overlay", default=str(DEFAULT_BASE))
    parser.add_argument("--detector-predictions", required=True)
    parser.add_argument("--output-json", default=str(DEFAULT_OUT_JSON))
    parser.add_argument("--output-md", default=str(DEFAULT_OUT_MD))
    parser.add_argument("--output-overlay", default=str(DEFAULT_OUT_OVERLAY))
    args = parser.parse_args()

    base_path = Path(args.base_overlay)
    detector_path = Path(args.detector_predictions)
    rows = load_jsonl(base_path)
    detector = detector_by_row(detector_path)
    golds_by_row = {str(row.get("row_id") or row.get("id")): p165.target_symbols(row) for row in rows}
    core_by_row = {str(row.get("row_id") or row.get("id")): p165.normalized(row.get("symbol_candidates") or [], "p182_core") for row in rows}
    baseline = p165.evaluate(golds_by_row, core_by_row)

    scored = []
    for policy in policies():
        fused = {row_id: fuse_row(core_by_row.get(row_id, []), detector.get(row_id, []), policy, row_id) for row_id in golds_by_row}
        metrics = p165.evaluate(golds_by_row, fused)
        scored.append({"policy": policy, "metrics": metrics})
    scored.sort(key=lambda item: (item["metrics"]["f1"], item["metrics"]["precision"], item["metrics"]["recall"], -item["metrics"]["prediction_inflation"]), reverse=True)
    best = scored[0]
    best_preds = {row_id: fuse_row(core_by_row.get(row_id, []), detector.get(row_id, []), best["policy"], row_id) for row_id in golds_by_row}
    write_jsonl(Path(args.output_overlay), materialize(rows, best_preds, best["policy"], detector_path))
    report = {
        "id": "SCI-P2-186-symbol-gpu-detector-p182-fusion",
        "created_on": "2026-05-17",
        "decision": "promote_candidate" if best["metrics"]["f1"] > baseline["f1"] else "no_promotion_keep_p182",
        "claim_boundary": "Offline P101 bridge selection; locked gold used for diagnostics only. Requires independent confirmation before paper claim if selected on the same 74-row overlay subset.",
        "inputs": {"base_overlay": str(base_path), "detector_predictions": str(detector_path)},
        "detector_row_overlap": {"base_rows": len(rows), "detector_rows": len(detector), "matched_rows": sum(1 for row in rows if str(row.get("row_id") or row.get("id")) in detector)},
        "baseline_metrics": {"p182_best": baseline},
        "searched_policy_count": len(scored),
        "best_policy": best["policy"],
        "best_metrics": best["metrics"],
        "delta_vs_p182": p165.delta(best["metrics"], baseline),
        "top_candidates": scored[:50],
        "outputs": {"overlay": str(Path(args.output_overlay)), "config_json": str(Path(args.output_json)), "report_md": str(Path(args.output_md))},
    }
    write_json(Path(args.output_json), report)
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text(render_md(report), encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "best_metrics": best["metrics"], "delta_vs_p182": report["delta_vs_p182"], "detector_row_overlap": report["detector_row_overlap"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
