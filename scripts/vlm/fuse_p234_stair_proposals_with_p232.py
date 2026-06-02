#!/usr/bin/env python3
"""P234 additive stair proposal fusion on top of promoted P232."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

import sys

sys.path.insert(0, str(ROOT / "scripts" / "vlm"))

from freeze_symbol_p222_p221a_sink_tiny import bbox_iou, bootstrap, metrics, score_rows  # noqa: E402
from fuse_symbol_p206g_with_p211_p212 import load_p206g  # noqa: E402


DEFAULT_BASE = ROOT / "reports" / "vlm" / "p232_repaired_contract_predictions.jsonl"
DEFAULT_OVERLAY = ROOT / "reports" / "vlm" / "symbol_p224a_column_frozen_overlay.jsonl"
DEFAULT_OUT = ROOT / "reports" / "vlm" / "p234_stair_fusion_predictions.jsonl"
DEFAULT_EVAL = ROOT / "reports" / "vlm" / "p234_stair_fusion_eval.json"
STAIR_SOURCES = {
    "p226": ROOT / "reports" / "vlm" / "symbol_p226_stair_specialist_pages_predictions.jsonl",
    "p228": ROOT / "reports" / "vlm" / "symbol_p228_merged_stair_pages_predictions.jsonl",
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def baseline_preds(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for row in rows:
        preds = []
        for item in row.get("expert_predictions") or []:
            preds.append({
                "id": str(item["candidate_id"]),
                "target_id": str(item["candidate_id"]),
                "label": str(item["label"]),
                "symbol_type": str(item["label"]),
                "bbox": [float(v) for v in item["bbox"]],
                "confidence": float(item.get("confidence") or 0.0),
                "score": float(item.get("confidence") or 0.0),
                "source": str(item.get("source") or "p232"),
            })
        out[str(row["row_id"])] = preds
    return out


def load_stair_source(path: Path, source_name: str) -> dict[str, list[dict[str, Any]]]:
    by_row: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(path):
        row_id = str(row.get("row_id") or row.get("id"))
        for index, pred in enumerate(row.get("predicted_symbols") or row.get("symbol_candidates") or []):
            label = str(pred.get("label") or pred.get("symbol_type") or "")
            if label != "stair":
                continue
            box = [float(v) for v in pred["bbox"]]
            by_row[row_id].append({
                "id": f"{row_id}_p234_{source_name}_{index:05d}",
                "target_id": f"{row_id}_p234_{source_name}_{index:05d}",
                "label": "stair",
                "symbol_type": "stair",
                "bbox": box,
                "confidence": float(pred.get("score", pred.get("confidence", 0.0)) or 0.0),
                "score": float(pred.get("score", pred.get("confidence", 0.0)) or 0.0),
                "source": f"p234_{source_name}_raster_stair_proposal",
                "tile_id": pred.get("tile_id"),
            })
    return by_row


def simple_nms(preds: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    kept = []
    for pred in sorted(preds, key=lambda item: float(item.get("score") or 0.0), reverse=True):
        box = [float(v) for v in pred["bbox"]]
        if any(bbox_iou(box, [float(v) for v in other["bbox"]]) >= threshold for other in kept):
            continue
        kept.append(pred)
    return kept


def overlaps_existing_stair(candidate: dict[str, Any], base_preds: list[dict[str, Any]], threshold: float) -> bool:
    box = [float(v) for v in candidate["bbox"]]
    for pred in base_preds:
        if pred["label"] != "stair":
            continue
        if bbox_iou(box, [float(v) for v in pred["bbox"]]) >= threshold:
            return True
    return False


def fuse(base: dict[str, list[dict[str, Any]]], stair_sources: dict[str, dict[str, list[dict[str, Any]]]], rule: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out = {}
    enabled = set(rule["sources"])
    for row_id, base_preds_row in base.items():
        extras = []
        for source_name, by_row in stair_sources.items():
            if source_name not in enabled:
                continue
            for pred in by_row.get(row_id, []):
                score = float(pred.get("score") or 0.0)
                box_area = area([float(v) for v in pred["bbox"]])
                if score < float(rule["min_score"]):
                    continue
                if box_area < float(rule["min_area"]):
                    continue
                if box_area > float(rule["max_area"]):
                    continue
                if overlaps_existing_stair(pred, base_preds_row, float(rule["max_iou_existing_stair"])):
                    continue
                extras.append(dict(pred))
        extras = simple_nms(extras, float(rule["nms_iou"]))[: int(rule["max_add_per_row"])]
        out[row_id] = list(base_preds_row) + extras
    return out


def evaluate(preds_by_row: dict[str, list[dict[str, Any]]], golds_by_row: dict[str, dict[str, dict[str, Any]]], row_ids: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_row = score_rows(preds_by_row, golds_by_row, row_ids)
    return metrics(per_row), per_row


def per_label_metrics(preds_by_row: dict[str, list[dict[str, Any]]], golds_by_row: dict[str, dict[str, dict[str, Any]]], row_ids: list[str]) -> dict[str, Any]:
    labels = sorted({gold["label"] for rid in row_ids for gold in golds_by_row[rid].values()} | {pred["label"] for preds in preds_by_row.values() for pred in preds})
    out = {}
    for label in labels:
        label_preds = defaultdict(list)
        label_golds = {}
        for rid in row_ids:
            label_preds[rid] = [pred for pred in preds_by_row.get(rid, []) if pred["label"] == label]
            label_golds[rid] = {gid: gold for gid, gold in golds_by_row[rid].items() if gold["label"] == label}
        out[label] = metrics(score_rows(label_preds, label_golds, row_ids))
    return out


def rule_grid() -> list[dict[str, Any]]:
    rules = []
    for sources in [["p226"], ["p228"], ["p226", "p228"]]:
        for min_score in [0.005, 0.01, 0.02, 0.04, 0.08]:
            for min_area in [0.0, 64.0]:
                for max_area in [1024.0, 4096.0, 20000.0]:
                    if min_area >= max_area:
                        continue
                    for nms_iou in [0.1, 0.3]:
                        for max_add in [1, 2, 4]:
                            rules.append({
                                "name": f"src{'-'.join(sources)}_s{min_score}_a{min_area}_{max_area}_nms{nms_iou}_max{max_add}",
                                "sources": sources,
                                "min_score": min_score,
                                "min_area": min_area,
                                "max_area": max_area,
                                "nms_iou": nms_iou,
                                "max_add_per_row": max_add,
                                "max_iou_existing_stair": 0.30,
                            })
    return rules


def write_runtime(base_rows: list[dict[str, Any]], fused_preds: dict[str, list[dict[str, Any]]], rule: dict[str, Any], output: Path) -> None:
    rows = []
    for row in base_rows:
        row_id = str(row["row_id"])
        expert_predictions = []
        added = 0
        for pred in fused_preds[row_id]:
            if pred["source"].startswith("p234_"):
                added += 1
            expert_predictions.append({
                "candidate_id": pred["id"],
                "expert": "symbol_fixture",
                "family": "symbol",
                "label": pred["label"],
                "confidence": pred["confidence"],
                "bbox": pred["bbox"],
                "geometry": {"bbox": pred["bbox"]},
                "relations": [],
                "source": pred["source"],
                "metadata": {"contract_version": "p234_stair_fusion_v0", "fusion_rule": rule["name"] if pred["source"].startswith("p234_") else "p232_baseline"},
            })
        rows.append({
            "row_id": row_id,
            "source": "p234_stair_fusion",
            "expert_predictions": expert_predictions,
            "adapter_metadata": {
                "contract_version": "p234_stair_fusion_v0",
                "added_stair_candidates": added,
                "selected_rule": rule["name"],
                "runtime_source_integrity": "raster_stair_predictions_and_p232_contract_only_no_svg_no_expected_json_no_offline_labels",
            },
        })
    write_jsonl(output, rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--eval-out", type=Path, default=DEFAULT_EVAL)
    args = parser.parse_args()

    base_rows = load_jsonl(args.base)
    base_preds = baseline_preds(base_rows)
    _rows, _overlay_preds, golds_by_row = load_p206g(args.overlay)
    row_ids = [str(row["row_id"]) for row in base_rows]
    stair_sources = {name: load_stair_source(path, name) for name, path in STAIR_SOURCES.items() if path.exists()}
    base_metrics, base_per_row = evaluate(base_preds, golds_by_row, row_ids)
    base_per_label = per_label_metrics(base_preds, golds_by_row, row_ids)

    best = None
    tried = 0
    for rule in rule_grid():
        tried += 1
        fused_preds = fuse(base_preds, stair_sources, rule)
        item_metrics, item_per_row = evaluate(fused_preds, golds_by_row, row_ids)
        item_per_label = per_label_metrics(fused_preds, golds_by_row, row_ids)
        delta = {key: round(item_metrics[key] - base_metrics[key], 6) for key in ["precision", "recall", "f1"]}
        stair_delta = {key: round(item_per_label["stair"][key] - base_per_label["stair"][key], 6) for key in ["precision", "recall", "f1"]}
        added = sum(1 for preds in fused_preds.values() for pred in preds if pred["source"].startswith("p234_"))
        item = {"rule": rule, "metrics": item_metrics, "per_row": item_per_row, "per_label": item_per_label, "delta": delta, "stair_delta": stair_delta, "added": added, "preds": fused_preds}
        key = (delta["precision"] >= 0, delta["f1"], stair_delta["f1"], stair_delta["recall"], -added)
        if best is None or key > (best["delta"]["precision"] >= 0, best["delta"]["f1"], best["stair_delta"]["f1"], best["stair_delta"]["recall"], -best["added"]):
            best = item
    assert best is not None
    write_runtime(base_rows, best["preds"], best["rule"], args.output)
    report = {
        "id": "p234_stair_fusion_eval",
        "phase": "P234_dedicated_raster_stair_proposal_source",
        "base_contract": str(args.base),
        "stair_sources": {name: str(path) for name, path in STAIR_SOURCES.items() if path.exists()},
        "output": str(args.output),
        "rules_tried": tried,
        "baseline_metrics_iou_0_30": base_metrics,
        "candidate_metrics_iou_0_30": best["metrics"],
        "delta_vs_p232": best["delta"],
        "baseline_stair_metrics": base_per_label["stair"],
        "candidate_stair_metrics": best["per_label"]["stair"],
        "stair_delta_vs_p232": best["stair_delta"],
        "bootstrap_vs_p232": bootstrap(base_per_row, best["per_row"], iterations=1000, seed=234),
        "selected_rule": best["rule"],
        "added_stair_candidates": best["added"],
        "per_label_metrics_iou_0_30": best["per_label"],
        "promotion_recommendation": "promote" if best["delta"]["f1"] > 0 and best["delta"]["precision"] >= 0 and best["stair_delta"]["f1"] > 0 else "do_not_promote",
        "claim_boundary": "Offline gate search over raster stair detector outputs. Runtime JSONL uses P232 predictions plus raster-derived stair predictions and constants only.",
    }
    write_json(args.eval_out, report)
    print(json.dumps({"eval": str(args.eval_out), "metrics": best["metrics"], "delta": best["delta"], "stair": best["per_label"]["stair"], "stair_delta": best["stair_delta"], "added": best["added"], "recommendation": report["promotion_recommendation"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
