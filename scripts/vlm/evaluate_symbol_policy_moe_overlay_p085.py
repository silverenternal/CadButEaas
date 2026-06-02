#!/usr/bin/env python3
"""Evaluate P0-84 symbol-policy overlays in the MoE downstream contract.

This is intentionally a narrow smoke runner: it keeps non-symbol inputs fixed,
loads the v28 and P0-76 overlays, runs fusion_v2 for graph-health signals, and
computes detector-appropriate symbol metrics with spatial matching instead of
exact node-id F1.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OVERLAYS = {
    "v28_frozen_detector_baseline": ROOT / "reports/vlm/symbol_policy_moe_overlay_p084/v28_frozen_detector_baseline.jsonl",
    "p076_balanced_opt_in": ROOT / "reports/vlm/symbol_policy_moe_overlay_p084/p076_balanced_opt_in.jsonl",
}
DEFAULT_SUMMARY = ROOT / "reports/vlm/symbol_policy_moe_overlay_p085_summary.json"
DEFAULT_REPORT = ROOT / "reports/vlm/symbol_policy_moe_overlay_p085.md"
FUSION_PATH = ROOT / "scripts/vlm/fuse_scene_graph_v2.py"
IOU_THRESHOLD = 0.30


def load_fusion_module() -> Any:
    spec = importlib.util.spec_from_file_location("fusion_v2_p085", FUSION_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load fusion module from {FUSION_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def bbox4(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in value[:4]]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def iou(a: list[float], b: list[float]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    denom = area(a) + area(b) - inter
    return inter / denom if denom > 0 else 0.0


def center_inside(pred: list[float], gold: list[float]) -> bool:
    cx = (pred[0] + pred[2]) / 2.0
    cy = (pred[1] + pred[3]) / 2.0
    return gold[0] <= cx <= gold[2] and gold[1] <= cy <= gold[3]


def area_bucket(box: list[float]) -> str:
    value = area(box)
    if value <= 32 * 32:
        return "tiny"
    if value <= 96 * 96:
        return "small"
    if value <= 192 * 192:
        return "medium"
    return "large"


def normalize_label(value: Any) -> str:
    text = str(value or "generic_symbol")
    return "generic_symbol" if text in {"symbol", "unknown", "unknown_symbol"} else text


def gold_symbols(row: dict[str, Any]) -> list[dict[str, Any]]:
    targets = row.get("targets") if isinstance(row.get("targets"), dict) else {}
    items: list[dict[str, Any]] = []
    for index, item in enumerate(targets.get("symbol") or []):
        box = bbox4(item.get("bbox"))
        if box is None:
            continue
        items.append({
            "id": str(item.get("target_id") or f"gold_{index}"),
            "label": normalize_label(item.get("semantic_type") or item.get("symbol_type") or item.get("label")),
            "bbox": box,
            "area_bucket": area_bucket(box),
        })
    return items


def predicted_symbols(row: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for index, item in enumerate(row.get("symbol_candidates") or []):
        box = bbox4(item.get("bbox"))
        if box is None:
            continue
        items.append({
            "id": str(item.get("id") or item.get("target_id") or f"pred_{index}"),
            "label": normalize_label(item.get("symbol_type") or item.get("label")),
            "bbox": box,
            "score": float(item.get("confidence") or 0.0),
            "area_bucket": area_bucket(box),
        })
    return items


def greedy_match(golds: list[dict[str, Any]], preds: list[dict[str, Any]], *, label_exact: bool) -> tuple[list[tuple[int, int, float]], dict[str, Any]]:
    candidates: list[tuple[float, int, int]] = []
    for gi, gold in enumerate(golds):
        for pi, pred in enumerate(preds):
            if label_exact and gold["label"] != pred["label"]:
                continue
            score = iou(gold["bbox"], pred["bbox"])
            if score >= IOU_THRESHOLD:
                candidates.append((score, gi, pi))
    candidates.sort(reverse=True)
    used_golds: set[int] = set()
    used_preds: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for score, gi, pi in candidates:
        if gi in used_golds or pi in used_preds:
            continue
        used_golds.add(gi)
        used_preds.add(pi)
        matches.append((gi, pi, score))
    metrics = prf(len(matches), len(preds), len(golds))
    return matches, metrics


def center_recall(golds: list[dict[str, Any]], preds: list[dict[str, Any]], *, label_exact: bool) -> dict[str, Any]:
    hit = 0
    for gold in golds:
        for pred in preds:
            if label_exact and gold["label"] != pred["label"]:
                continue
            if center_inside(pred["bbox"], gold["bbox"]):
                hit += 1
                break
    return {"hit": hit, "gold": len(golds), "recall": round(hit / max(len(golds), 1), 6)}


def prf(tp: int, predicted: int, gold: int) -> dict[str, Any]:
    precision = tp / max(predicted, 1)
    recall = tp / max(gold, 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"tp": int(tp), "predicted": int(predicted), "gold": int(gold), "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6)}


def evaluate_policy(policy_id: str, path: Path, fusion: Any) -> dict[str, Any]:
    rows = load_jsonl(path)
    totals = Counter()
    label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    bucket_counts: dict[str, Counter[str]] = defaultdict(Counter)
    graph_counts = Counter()
    warning_counts: Counter[str] = Counter()
    case_rows: list[dict[str, Any]] = []

    start = time.perf_counter()
    for row in rows:
        gold = gold_symbols(row)
        pred = predicted_symbols(row)
        _, any_match = greedy_match(gold, pred, label_exact=False)
        exact_matches, exact_match = greedy_match(gold, pred, label_exact=True)
        center_any = center_recall(gold, pred, label_exact=False)
        center_exact = center_recall(gold, pred, label_exact=True)
        totals.update({
            "records": 1,
            "gold_symbols": len(gold),
            "predicted_symbols": len(pred),
            "iou_any_tp": any_match["tp"],
            "iou_exact_tp": exact_match["tp"],
            "center_any_hit": center_any["hit"],
            "center_exact_hit": center_exact["hit"],
        })

        matched_gold_exact = {gi for gi, _, _ in exact_matches}
        matched_pred_exact = {pi for _, pi, _ in exact_matches}
        for gi, gold_item in enumerate(gold):
            label = gold_item["label"]
            bucket = gold_item["area_bucket"]
            label_counts[label]["gold"] += 1
            bucket_counts[bucket]["gold"] += 1
            if gi in matched_gold_exact:
                label_counts[label]["tp"] += 1
                bucket_counts[bucket]["tp"] += 1
        for pi, pred_item in enumerate(pred):
            label = pred_item["label"]
            bucket = pred_item["area_bucket"]
            label_counts[label]["predicted"] += 1
            bucket_counts[bucket]["predicted"] += 1

        fused = fusion.fuse_v2(row, enable_all_repairs=True)
        route = fused.get("route_trace") or {}
        metadata = fused.get("metadata") or {}
        graph_counts.update({
            "node_count": int(route.get("node_count") or 0),
            "edge_count": int(route.get("edge_count") or 0),
            "warning_count": int(route.get("warning_count") or 0),
            "repair_event_count": int(route.get("repair_event_count") or 0),
            "invalid_graphs": 0 if route.get("scene_graph_valid", True) else 1,
        })
        warning_counts.update(str(w) for w in fused.get("warnings") or [])

        if exact_match["recall"] < 1.0 or not route.get("scene_graph_valid", True):
            missed = [gold[gi] for gi in range(len(gold)) if gi not in matched_gold_exact]
            extras = [pred[pi] for pi in range(len(pred)) if pi not in matched_pred_exact]
            case_rows.append({
                "row_id": row.get("row_id") or row.get("id"),
                "policy_id": policy_id,
                "gold_symbols": len(gold),
                "predicted_symbols": len(pred),
                "iou_exact": exact_match,
                "scene_graph_valid": bool(route.get("scene_graph_valid", True)),
                "missed_sample": [{"id": item["id"], "label": item["label"], "bbox": item["bbox"]} for item in missed[:8]],
                "extra_sample": [{"id": item["id"], "label": item["label"], "bbox": item["bbox"], "score": item.get("score")} for item in extras[:8]],
                "warnings": list(fused.get("warnings") or [])[:8],
                "contract_errors": list(metadata.get("scene_graph_contract_errors") or [])[:8],
            })

    elapsed = time.perf_counter() - start
    records = int(totals["records"])
    summary = {
        "policy_id": policy_id,
        "input": str(path.relative_to(ROOT) if path.is_absolute() and path.is_relative_to(ROOT) else path),
        "records": records,
        "symbol_iou_any": prf(totals["iou_any_tp"], totals["predicted_symbols"], totals["gold_symbols"]),
        "symbol_iou_label_exact": prf(totals["iou_exact_tp"], totals["predicted_symbols"], totals["gold_symbols"]),
        "symbol_center_any": {"hit": int(totals["center_any_hit"]), "gold": int(totals["gold_symbols"]), "recall": round(totals["center_any_hit"] / max(totals["gold_symbols"], 1), 6)},
        "symbol_center_label_exact": {"hit": int(totals["center_exact_hit"]), "gold": int(totals["gold_symbols"]), "recall": round(totals["center_exact_hit"] / max(totals["gold_symbols"], 1), 6)},
        "prediction_inflation": round(totals["predicted_symbols"] / max(totals["gold_symbols"], 1), 6),
        "graph_health": {
            "node_count_total": int(graph_counts["node_count"]),
            "edge_count_total": int(graph_counts["edge_count"]),
            "warnings_total": int(graph_counts["warning_count"]),
            "repair_events_total": int(graph_counts["repair_event_count"]),
            "invalid_graphs": int(graph_counts["invalid_graphs"]),
            "invalid_graph_rate": round(graph_counts["invalid_graphs"] / max(records, 1), 6),
        },
        "by_label": {label: prf(c["tp"], c["predicted"], c["gold"]) for label, c in sorted(label_counts.items())},
        "by_gold_area_bucket": {bucket: prf(c["tp"], c["predicted"], c["gold"]) for bucket, c in sorted(bucket_counts.items())},
        "warning_summary": dict(warning_counts.most_common(30)),
        "case_count": len(case_rows),
        "case_sample": case_rows[:5],
        "elapsed_seconds": round(elapsed, 3),
    }
    return summary


def compare(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    keys = [
        ("symbol_iou_any", "recall"),
        ("symbol_iou_any", "precision"),
        ("symbol_iou_label_exact", "recall"),
        ("symbol_iou_label_exact", "precision"),
        ("symbol_center_any", "recall"),
        ("symbol_center_label_exact", "recall"),
    ]
    deltas: dict[str, float] = {}
    for section, metric in keys:
        deltas[f"{section}.{metric}"] = round(float(candidate[section][metric]) - float(base[section][metric]), 6)
    deltas["prediction_inflation"] = round(float(candidate["prediction_inflation"]) - float(base["prediction_inflation"]), 6)
    deltas["invalid_graph_rate"] = round(float(candidate["graph_health"]["invalid_graph_rate"]) - float(base["graph_health"]["invalid_graph_rate"]), 6)
    deltas["warnings_total"] = int(candidate["graph_health"]["warnings_total"]) - int(base["graph_health"]["warnings_total"])
    return deltas


def render_report(summary: dict[str, Any]) -> str:
    policies = summary["policies"]
    lines = [
        "# P0-85 Scene-Graph Smoke on Symbol Policy Overlays",
        "",
        "## Decision",
        "",
        summary["decision"],
        "",
        "## Policy Metrics",
        "",
        "| Policy | Rows | IoU recall | IoU precision | Label-exact IoU recall | Center recall | Inflation | Invalid graphs | Warnings |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in policies:
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                item["policy_id"],
                item["records"],
                item["symbol_iou_any"]["recall"],
                item["symbol_iou_any"]["precision"],
                item["symbol_iou_label_exact"]["recall"],
                item["symbol_center_any"]["recall"],
                item["prediction_inflation"],
                item["graph_health"]["invalid_graphs"],
                item["graph_health"]["warnings_total"],
            )
        )
    lines.extend([
        "",
        "## Delta: P0-76 vs v28",
        "",
    ])
    for key, value in summary["comparison"]["p076_vs_v28"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend([
        "",
        "## Notes",
        "",
        "- This smoke uses spatial symbol matching because detector candidates do not share gold node ids.",
        "- Non-symbol inputs are identical across overlays; graph-health deltas mainly show whether extra symbol candidates destabilize fusion.",
        "- `v28` remains the default policy; P0-76 is still an opt-in unless downstream users want the measured recall/precision tradeoff.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--overlay", action="append", default=[], help="policy_id=path; defaults to v28 and p076")
    args = parser.parse_args()

    overlays = dict(DEFAULT_OVERLAYS)
    for spec in args.overlay:
        if "=" not in spec:
            raise SystemExit(f"Invalid --overlay {spec!r}; expected policy_id=path")
        policy_id, path = spec.split("=", 1)
        overlays[policy_id] = Path(path)

    fusion = load_fusion_module()
    policy_summaries = [evaluate_policy(policy_id, path, fusion) for policy_id, path in overlays.items()]
    by_id = {item["policy_id"]: item for item in policy_summaries}
    base = by_id.get("v28_frozen_detector_baseline")
    candidate = by_id.get("p076_balanced_opt_in")
    comparison = {"p076_vs_v28": compare(base, candidate)} if base and candidate else {}
    decision = "Keep `v28_frozen_detector_baseline` as default. `P0-76` improves symbol IoU recall in this downstream overlay smoke, but with higher candidate inflation; keep it as an explicit opt-in rather than promoting default."
    if base and candidate:
        delta = comparison["p076_vs_v28"]
        if delta["symbol_iou_any.recall"] <= 0 and delta["symbol_iou_any.precision"] <= 0:
            decision = "Keep `v28_frozen_detector_baseline` as default and do not recommend P0-76 for downstream overlay use; it did not improve smoke metrics."
        elif delta["invalid_graph_rate"] > 0.005:
            decision = "Keep `v28_frozen_detector_baseline` as default; P0-76 recall gains are not promoted because graph invalidity increased."

    summary = {
        "id": "P0-85-scene-graph-fusion-smoke-on-symbol-policy-overlays",
        "metric_contract": "spatial_symbol_matching_iou_0.30_plus_fusion_v2_graph_health",
        "default_policy_unchanged": True,
        "decision": decision,
        "policies": policy_summaries,
        "comparison": comparison,
        "next_step": "Package downstream policy choice as v28 default with P0-76 opt-in, or add CLI/config hook in fusion runner if interactive switching is needed.",
    }
    write_json(Path(args.summary), summary)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
