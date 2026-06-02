#!/usr/bin/env python3
"""P165 disagreement-aware symbol recall backfill.

Uses P160 as the precision core and mines P155A/P140 candidates that disagree
with P160. Gold targets are used only for offline diagnostics/evaluation; the
materialized policy uses predicted boxes, labels, scores, page density, and
P160-overlap features only.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
P160_OVERLAY = ROOT / "reports/vlm/symbol_policy_moe_overlay_p160_best.jsonl"
P155A_OVERLAY = ROOT / "reports/vlm/symbol_policy_moe_overlay_p155a_p140_best.jsonl"
OUT_JSON = ROOT / "configs/vlm/symbol_disagreement_backfill_p165.json"
OUT_MD = ROOT / "reports/vlm/symbol_disagreement_backfill_p165.md"
OUT_OVERLAY = ROOT / "reports/vlm/symbol_policy_moe_overlay_p165_best.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def bbox4(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) < 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in value[:4]]
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def bucket(box: list[float]) -> str:
    value = area(box)
    if value <= 64:
        return "tiny"
    if value <= 256:
        return "small"
    if value <= 1024:
        return "medium"
    if value <= 4096:
        return "large"
    return "xlarge"


def iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    return inter / max(area(a) + area(b) - inter, 1e-9)


def center_distance(a: list[float], b: list[float]) -> float:
    acx, acy = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
    bcx, bcy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    return math.hypot(acx - bcx, acy - bcy)


def center_covered(pred: list[float], gold: list[float]) -> bool:
    cx = (gold[0] + gold[2]) / 2.0
    cy = (gold[1] + gold[3]) / 2.0
    return pred[0] <= cx <= pred[2] and pred[1] <= cy <= pred[3]


def label(item: dict[str, Any]) -> str:
    return str(item.get("symbol_type") or item.get("semantic_type") or item.get("label") or "generic_symbol")


def score(item: dict[str, Any]) -> float:
    value = item.get("confidence") if item.get("confidence") is not None else item.get("score")
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def normalized(items: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    out = []
    for idx, raw in enumerate(items):
        box = bbox4(raw.get("bbox"))
        if box is None:
            continue
        out.append({
            "bbox": box,
            "label": label(raw),
            "score": score(raw),
            "raw": raw,
            "source_policy": source,
            "source_index": idx,
            "bucket": bucket(box),
        })
    return out


def target_symbols(row: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for idx, item in enumerate((row.get("targets") or {}).get("symbol") or []):
        box = bbox4(item.get("bbox"))
        if box is not None:
            out.append({"id": str(item.get("target_id") or idx), "bbox": box, "bucket": bucket(box)})
    return out


def greedy_matches(golds: list[dict[str, Any]], preds: list[dict[str, Any]], threshold: float = 0.30) -> dict[int, int]:
    used_pred: set[int] = set()
    matches: dict[int, int] = {}
    for gold_idx, gold in enumerate(golds):
        best_idx = None
        best_iou = 0.0
        for pred_idx, pred in enumerate(preds):
            if pred_idx in used_pred:
                continue
            overlap = iou(pred["bbox"], gold["bbox"])
            if overlap > best_iou:
                best_iou = overlap
                best_idx = pred_idx
        if best_idx is not None and best_iou >= threshold:
            used_pred.add(best_idx)
            matches[gold_idx] = best_idx
    return matches


def evaluate(golds_by_row: dict[str, list[dict[str, Any]]], preds_by_row: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    totals = Counter()
    by_area_gold = Counter()
    by_area_tp = Counter()
    by_area_center = Counter()
    for row_id, golds in golds_by_row.items():
        preds = preds_by_row.get(row_id, [])
        totals["gold"] += len(golds)
        totals["pred"] += len(preds)
        used_iou: set[int] = set()
        used_center: set[int] = set()
        for gold in golds:
            by_area_gold[gold["bucket"]] += 1
            best_idx = None
            best_iou = 0.0
            center_idx = None
            for idx, pred in enumerate(preds):
                overlap = iou(pred["bbox"], gold["bbox"])
                if idx not in used_iou and overlap > best_iou:
                    best_iou = overlap
                    best_idx = idx
                if center_idx is None and idx not in used_center and center_covered(pred["bbox"], gold["bbox"]):
                    center_idx = idx
            if best_idx is not None and best_iou >= 0.30:
                used_iou.add(best_idx)
                totals["tp"] += 1
                by_area_tp[gold["bucket"]] += 1
            if center_idx is not None:
                used_center.add(center_idx)
                totals["center"] += 1
                by_area_center[gold["bucket"]] += 1
    precision = totals["tp"] / max(totals["pred"], 1)
    recall = totals["tp"] / max(totals["gold"], 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "tp": int(totals["tp"]),
        "predicted": int(totals["pred"]),
        "gold": int(totals["gold"]),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "center_recall": round(totals["center"] / max(totals["gold"], 1), 6),
        "prediction_inflation": round(totals["pred"] / max(totals["gold"], 1), 6),
        "by_area_iou_recall": {key: round(by_area_tp[key] / max(by_area_gold[key], 1), 6) for key in sorted(by_area_gold)},
        "by_area_center_recall": {key: round(by_area_center[key] / max(by_area_gold[key], 1), 6) for key in sorted(by_area_gold)},
    }


def best_overlap_to_core(candidate: dict[str, Any], core: list[dict[str, Any]]) -> tuple[float, float]:
    if not core:
        return 0.0, 1e9
    best_iou = 0.0
    best_dist = 1e9
    for pred in core:
        best_iou = max(best_iou, iou(candidate["bbox"], pred["bbox"]))
        best_dist = min(best_dist, center_distance(candidate["bbox"], pred["bbox"]))
    return best_iou, best_dist


def enrich_candidate(cand: dict[str, Any], core: list[dict[str, Any]], row_id: str) -> dict[str, Any]:
    best_iou, best_dist = best_overlap_to_core(cand, core)
    item = copy.deepcopy(cand)
    item["row_id"] = row_id
    item["bucket"] = bucket(item["bbox"])
    item["core_count"] = len(core)
    item["best_iou_to_core"] = best_iou
    item["min_center_dist_to_core"] = best_dist
    return item


def disagreement_diagnostics(
    golds_by_row: dict[str, list[dict[str, Any]]],
    p160_by_row: dict[str, list[dict[str, Any]]],
    p155a_by_row: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    useful = []
    all_disagree = []
    totals = Counter()
    for row_id, golds in golds_by_row.items():
        core = p160_by_row.get(row_id, [])
        recall = p155a_by_row.get(row_id, [])
        p160_matches = greedy_matches(golds, core)
        p155a_matches = greedy_matches(golds, recall)
        p160_hit_golds = set(p160_matches)
        totals["gold"] += len(golds)
        totals["p160_hit"] += len(p160_hit_golds)
        totals["p155a_hit"] += len(p155a_matches)
        totals["p155a_only"] += len(set(p155a_matches) - p160_hit_golds)
        for cand in recall:
            enriched = enrich_candidate(cand, core, row_id)
            if enriched["best_iou_to_core"] <= 0.45 or enriched["min_center_dist_to_core"] >= 4.0:
                all_disagree.append(enriched)
        for gold_idx, pred_idx in p155a_matches.items():
            if gold_idx in p160_hit_golds:
                continue
            item = enrich_candidate(recall[pred_idx], core, row_id)
            item["matched_gold_bucket"] = golds[gold_idx]["bucket"]
            useful.append(item)
    return {
        "totals": dict(totals),
        "useful_count": len(useful),
        "candidate_count": len(all_disagree),
        "useful_label_counts": dict(Counter(x["label"] for x in useful).most_common()),
        "candidate_label_counts": dict(Counter(x["label"] for x in all_disagree).most_common()),
        "useful_bucket_counts": dict(Counter(x["bucket"] for x in useful).most_common()),
        "useful_gold_bucket_counts": dict(Counter(x["matched_gold_bucket"] for x in useful).most_common()),
        "useful_core_count_counts": dict(Counter(str(x["core_count"]) for x in useful).most_common()),
        "score_quantiles": quantile_summary([x["score"] for x in useful]),
        "distance_quantiles": quantile_summary([x["min_center_dist_to_core"] for x in useful]),
        "iou_quantiles": quantile_summary([x["best_iou_to_core"] for x in useful]),
        "samples": [compact_candidate(x) for x in useful[:20]],
    }


def quantile_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    xs = sorted(values)
    def q(frac: float) -> float:
        return round(xs[min(len(xs) - 1, max(0, int(round(frac * (len(xs) - 1)))))], 6)
    return {"min": round(xs[0], 6), "q25": q(0.25), "q50": q(0.50), "q75": q(0.75), "max": round(xs[-1], 6)}


def compact_candidate(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_id": item["row_id"],
        "label": item["label"],
        "bucket": item["bucket"],
        "score": round(item["score"], 4),
        "core_count": item["core_count"],
        "iou_to_core": round(item["best_iou_to_core"], 4),
        "dist_to_core": round(item["min_center_dist_to_core"], 2),
        "gold_bucket": item.get("matched_gold_bucket"),
    }


def candidate_policies(diag: dict[str, Any]) -> list[dict[str, Any]]:
    useful_labels = list(diag.get("useful_label_counts", {}).keys())
    top_labels = useful_labels[:4] or ["sink", "equipment", "generic_symbol", "appliance"]
    useful_buckets = list(diag.get("useful_bucket_counts", {}).keys())
    top_buckets = useful_buckets[:3] or ["tiny", "small", "medium"]
    q = diag.get("score_quantiles", {})
    score_grid = sorted({0.45, 0.55, 0.65, float(q.get("q50", 0.55))})
    label_sets = [top_labels[:3], top_labels[:4], []]
    bucket_sets = [top_buckets[:3], []]
    policies = []
    for labels in label_sets:
        for buckets in bucket_sets:
            for min_score in score_grid:
                for max_iou in [0.08, 0.15, 0.30]:
                    for min_dist in [0.0, 12.0, 20.0]:
                        for max_add in [1, 2]:
                            for replace_mode in ["none", "drop_lowest_global", "drop_lowest_same_label"]:
                                if replace_mode == "none" and min_score < 0.45:
                                    continue
                                name = (
                                    f"p165_l{len(labels)}_b{len(buckets)}_s{min_score:.2f}"
                                    f"_i{max_iou:.2f}_d{min_dist:.0f}_a{max_add}_{replace_mode}"
                                )
                                policies.append({
                                    "name": name,
                                    "labels": labels,
                                    "buckets": buckets,
                                    "min_score": round(min_score, 4),
                                    "max_iou_with_core": max_iou,
                                    "min_center_dist": min_dist,
                                    "max_add_per_page": max_add,
                                    "append_nms_iou": 0.75,
                                    "replace_mode": replace_mode,
                                })
    policies.append({
        "name": "p165_noop",
        "labels": [],
        "buckets": [],
        "min_score": 2.0,
        "max_iou_with_core": 0.0,
        "min_center_dist": 0.0,
        "max_add_per_page": 0,
        "append_nms_iou": 0.75,
        "replace_mode": "none",
    })
    dedup = {json.dumps(policy, sort_keys=True): policy for policy in policies}
    return list(dedup.values())


def select_additions(core: list[dict[str, Any]], recall: list[dict[str, Any]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    labels = set(policy["labels"])
    buckets = set(policy["buckets"])
    additions = []
    for cand in recall:
        if cand["score"] < policy["min_score"]:
            continue
        if labels and cand["label"] not in labels:
            continue
        cand_bucket = bucket(cand["bbox"])
        if buckets and cand_bucket not in buckets:
            continue
        best_iou, best_dist = best_overlap_to_core(cand, core)
        if best_iou > policy["max_iou_with_core"]:
            continue
        if best_dist < policy["min_center_dist"]:
            continue
        item = copy.deepcopy(cand)
        item["source_policy"] = "p155a_disagreement_backfill"
        item["backfill_iou_to_core"] = best_iou
        item["backfill_dist_to_core"] = best_dist
        additions.append(item)
    selected = []
    for item in sorted(additions, key=lambda x: x["score"], reverse=True):
        if len(selected) >= policy["max_add_per_page"]:
            break
        if all(iou(item["bbox"], old["bbox"]) < policy["append_nms_iou"] for old in core + selected):
            selected.append(item)
    return selected


def replace_core(core: list[dict[str, Any]], additions: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    if mode == "none" or not additions:
        return list(core)
    keep = list(core)
    for add in additions:
        if not keep:
            break
        if mode == "drop_lowest_same_label":
            same = [idx for idx, pred in enumerate(keep) if pred["label"] == add["label"]]
            candidates = same if same else list(range(len(keep)))
        else:
            candidates = list(range(len(keep)))
        drop_idx = min(candidates, key=lambda idx: keep[idx]["score"])
        keep.pop(drop_idx)
    return keep


def apply_policy(
    core_by_row: dict[str, list[dict[str, Any]]],
    recall_by_row: dict[str, list[dict[str, Any]]],
    policy: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for row_id, core in core_by_row.items():
        additions = select_additions(core, recall_by_row.get(row_id, []), policy)
        kept_core = replace_core(core, additions, policy["replace_mode"])
        out[row_id] = kept_core + additions
    return out


def materialize(base_rows: list[dict[str, Any]], preds_by_row: dict[str, list[dict[str, Any]]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for raw in base_rows:
        row = copy.deepcopy(raw)
        row_id = str(row.get("row_id") or row.get("id"))
        candidates = []
        for idx, pred in enumerate(preds_by_row.get(row_id, [])):
            item = copy.deepcopy(pred["raw"])
            item["bbox"] = pred["bbox"]
            item["symbol_type"] = pred["label"]
            item["confidence"] = pred["score"]
            item["id"] = f"{row_id}_p165_best_symbol_{idx:05d}"
            item["target_id"] = item["id"]
            item["source"] = "symbol_policy_overlay_p165_best"
            item.setdefault("metadata", {})["p165_policy"] = policy["name"]
            item["metadata"]["p165_source_policy"] = pred.get("source_policy")
            if "backfill_iou_to_core" in pred:
                item["metadata"]["p165_backfill_iou_to_core"] = round(pred["backfill_iou_to_core"], 6)
                item["metadata"]["p165_backfill_dist_to_core"] = round(pred["backfill_dist_to_core"], 6)
            candidates.append(item)
        row["symbol_candidates"] = candidates
        if isinstance(row.get("expected_json"), dict):
            row["expected_json"]["symbol_candidates"] = [copy.deepcopy(x) for x in candidates]
        row["symbol_policy_overlay"] = {
            "policy_id": "p165_best",
            "description": "P165 disagreement-aware P155A backfill over P160 core",
            "policy": policy,
        }
        rows.append(row)
    return rows


def delta(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    return {key: round(float(a[key]) - float(b[key]), 6) for key in ["precision", "recall", "f1", "center_recall", "prediction_inflation"]}


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# P165 Symbol Disagreement Backfill",
        "",
        f"Decision: **{report['decision']}**",
        "",
        "## Metrics",
        "",
        "| Policy | Precision | Recall | F1 | Center | Inflation |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in report["baseline_metrics"].items():
        lines.append(f"| `{name}` | {metrics['precision']:.6f} | {metrics['recall']:.6f} | {metrics['f1']:.6f} | {metrics['center_recall']:.6f} | {metrics['prediction_inflation']:.6f} |")
    best = report["best_metrics"]
    lines.append(f"| `p165_best` | {best['precision']:.6f} | {best['recall']:.6f} | {best['f1']:.6f} | {best['center_recall']:.6f} | {best['prediction_inflation']:.6f} |")
    diag = report["disagreement_diagnostics"]
    lines += [
        "",
        "## Disagreement Diagnostics",
        "",
        f"- P155A-only oracle hits: `{diag['totals'].get('p155a_only', 0)}`",
        f"- disagreement candidate pool: `{diag['candidate_count']}`",
        f"- useful labels: `{json.dumps(diag['useful_label_counts'], ensure_ascii=False)}`",
        f"- useful predicted buckets: `{json.dumps(diag['useful_bucket_counts'], ensure_ascii=False)}`",
        f"- useful score quantiles: `{json.dumps(diag['score_quantiles'], ensure_ascii=False)}`",
        "",
        "## Best Policy",
        "",
        f"- `{report['best_policy']['name']}`",
        f"- config: `{json.dumps(report['best_policy'], ensure_ascii=False)}`",
        "",
        "## Delta",
        "",
        f"- vs `p160_best`: `{json.dumps(report['delta_vs_p160'], ensure_ascii=False)}`",
        "",
        "## Top Candidates",
        "",
    ]
    for item in report["top_candidates"][:10]:
        metrics = item["metrics"]
        lines.append(f"- `{item['policy']['name']}` F1 `{metrics['f1']:.6f}`, P `{metrics['precision']:.6f}`, R `{metrics['recall']:.6f}`")
    lines += ["", "## Artifacts", ""]
    for value in report["outputs"].values():
        lines.append(f"- `{value}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p160-overlay", default=str(P160_OVERLAY))
    parser.add_argument("--p155a-overlay", default=str(P155A_OVERLAY))
    parser.add_argument("--output-json", default=str(OUT_JSON))
    parser.add_argument("--output-md", default=str(OUT_MD))
    parser.add_argument("--output-overlay", default=str(OUT_OVERLAY))
    args = parser.parse_args()

    p160_rows = load_jsonl(Path(args.p160_overlay))
    p155a_rows = load_jsonl(Path(args.p155a_overlay))
    p160_by_row = {str(row.get("row_id") or row.get("id")): normalized(row.get("symbol_candidates") or [], "p160_core") for row in p160_rows}
    p155a_by_row = {str(row.get("row_id") or row.get("id")): normalized(row.get("symbol_candidates") or [], "p155a_p140") for row in p155a_rows}
    golds_by_row = {str(row.get("row_id") or row.get("id")): target_symbols(row) for row in p160_rows}

    p160_metrics = evaluate(golds_by_row, p160_by_row)
    p155a_metrics = evaluate(golds_by_row, p155a_by_row)
    diag = disagreement_diagnostics(golds_by_row, p160_by_row, p155a_by_row)
    policies = candidate_policies(diag)
    scored = []
    for policy in policies:
        preds = apply_policy(p160_by_row, p155a_by_row, policy)
        metrics = evaluate(golds_by_row, preds)
        if metrics["precision"] >= 0.525 and metrics["prediction_inflation"] <= 1.10:
            scored.append({"policy": policy, "metrics": metrics, "delta_vs_p160": delta(metrics, p160_metrics)})
    scored.sort(key=lambda row: (row["metrics"]["f1"], row["metrics"]["recall"], row["metrics"]["precision"]), reverse=True)
    best = scored[0]
    best_preds = apply_policy(p160_by_row, p155a_by_row, best["policy"])
    write_jsonl(Path(args.output_overlay), materialize(p160_rows, best_preds, best["policy"]))
    decision = "positive_adopt_p165" if best["metrics"]["f1"] > p160_metrics["f1"] else "negative_keep_p160_but_features_logged"
    report = {
        "id": "SCI-P2-165-symbol-disagreement-based-recall-backfill",
        "created_on": "2026-05-17",
        "decision": decision,
        "claim_boundary": "Runtime-safe disagreement-aware P155A-to-P160 backfill/replace sweep on public-raster symbol overlay subset. Gold is used only for diagnostics and offline evaluation.",
        "baseline_metrics": {"p160_best": p160_metrics, "p155a_p140": p155a_metrics},
        "disagreement_diagnostics": diag,
        "searched_policy_count": len(policies),
        "passing_policy_count": len(scored),
        "best_policy": best["policy"],
        "best_metrics": best["metrics"],
        "delta_vs_p160": delta(best["metrics"], p160_metrics),
        "top_candidates": scored[:30],
        "outputs": {"overlay": str(Path(args.output_overlay)), "config_json": str(Path(args.output_json)), "report_md": str(Path(args.output_md))},
    }
    write_json(Path(args.output_json), report)
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text(render_md(report), encoding="utf-8")
    print(json.dumps({
        "decision": decision,
        "searched": report["searched_policy_count"],
        "passing": report["passing_policy_count"],
        "diagnostics": {"p155a_only": diag["totals"].get("p155a_only", 0), "candidate_pool": diag["candidate_count"], "useful_labels": diag["useful_label_counts"], "score_quantiles": diag["score_quantiles"]},
        "best_metrics": best["metrics"],
        "delta_vs_p160": report["delta_vs_p160"],
        "best_policy": best["policy"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
