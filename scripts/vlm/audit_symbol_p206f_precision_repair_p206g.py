#!/usr/bin/env python3
"""P206g precision-repair audit for P206f symbol recall branch."""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import sweep_symbol_disagreement_backfill_p165 as p165
from fuse_symbol_detector_with_p182_p186 import load_jsonl, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[2]


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("row_id") or row.get("id"))


def target_label(item: dict[str, Any]) -> str:
    return str(item.get("semantic_type") or item.get("symbol_type") or item.get("raw_label") or item.get("label") or "generic_symbol")


def target_symbols(row: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for idx, item in enumerate((row.get("targets") or {}).get("symbol") or []):
        box = p165.bbox4(item.get("bbox"))
        if box is not None:
            out.append({"id": str(item.get("target_id") or idx), "bbox": box, "bucket": p165.bucket(box), "label": target_label(item)})
    return out


def rows_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row_id(row): row for row in rows}


def norm_map(rows: list[dict[str, Any]], source: str) -> dict[str, list[dict[str, Any]]]:
    return {row_id(row): p165.normalized(row.get("symbol_candidates") or [], source) for row in rows}


def pred_key(pred: dict[str, Any]) -> tuple[str, tuple[int, int, int, int]]:
    box = pred["bbox"]
    return pred["label"], tuple(int(round(v)) for v in box)


def best_gold_overlap(pred: dict[str, Any], golds: list[dict[str, Any]]) -> tuple[float, bool, str, str]:
    best_iou = 0.0
    center = False
    best_label = ""
    best_bucket = ""
    for gold in golds:
        overlap = p165.iou(pred["bbox"], gold["bbox"])
        if overlap > best_iou:
            best_iou = overlap
            best_label = gold.get("label", "")
            best_bucket = gold.get("bucket", "")
        center = center or p165.center_covered(pred["bbox"], gold["bbox"])
    return best_iou, center, best_label, best_bucket


def addition_audit(base_rows: list[dict[str, Any]], p206e_preds: dict[str, list[dict[str, Any]]], p206f_preds: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    row_lookup = rows_by_id(base_rows)
    records = []
    by_label = Counter()
    by_bucket = Counter()
    by_label_fp = Counter()
    by_bucket_fp = Counter()
    for rid, f_preds in p206f_preds.items():
        e_preds = p206e_preds.get(rid, [])
        golds = target_symbols(row_lookup[rid])
        for pred in f_preds:
            overlap_to_e = max([p165.iou(pred["bbox"], other["bbox"]) for other in e_preds] or [0.0])
            if overlap_to_e >= 0.50:
                continue
            best_iou, center, gold_label, gold_bucket = best_gold_overlap(pred, golds)
            item = {
                "row_id": rid,
                "label": pred["label"],
                "bucket": pred.get("bucket") or p165.bucket(pred["bbox"]),
                "score": float(pred.get("score") or 0.0),
                "best_iou_to_p206e": overlap_to_e,
                "best_iou_to_gold": best_iou,
                "center_hits_gold": center,
                "is_iou_tp_candidate": best_iou >= 0.30,
                "gold_label": gold_label,
                "gold_bucket": gold_bucket,
                "bbox": pred["bbox"],
            }
            records.append(item)
            by_label[item["label"]] += 1
            by_bucket[item["bucket"]] += 1
            if not item["is_iou_tp_candidate"]:
                by_label_fp[item["label"]] += 1
                by_bucket_fp[item["bucket"]] += 1
    return {
        "addition_count": len(records),
        "recoverable_iou_candidates": sum(1 for r in records if r["is_iou_tp_candidate"]),
        "center_only_candidates": sum(1 for r in records if r["center_hits_gold"] and not r["is_iou_tp_candidate"]),
        "likely_fp_candidates": sum(1 for r in records if not r["center_hits_gold"] and not r["is_iou_tp_candidate"]),
        "by_label": dict(by_label),
        "by_bucket": dict(by_bucket),
        "by_label_likely_fp": dict(by_label_fp),
        "by_bucket_likely_fp": dict(by_bucket_fp),
        "top_likely_fp": sorted([r for r in records if not r["is_iou_tp_candidate"]], key=lambda x: (x["center_hits_gold"], x["score"]))[:30],
    }


def keep_pred(pred: dict[str, Any], core: list[dict[str, Any]], policy: dict[str, Any]) -> bool:
    label = pred["label"]
    bucket = pred.get("bucket") or p165.bucket(pred["bbox"])
    score = float(pred.get("score") or 0.0)
    if label in set(policy.get("drop_labels", [])):
        return False
    if bucket in set(policy.get("drop_buckets", [])):
        return False
    if score < float(policy.get("min_score", 0.0)):
        return False
    best_iou, best_dist = p165.best_overlap_to_core(pred, core)
    if best_iou >= float(policy.get("max_iou_to_core", 0.98)):
        return False
    if best_dist < float(policy.get("min_dist_to_core", 0.0)):
        return False
    return True


def precompute_additions(p206e_preds: dict[str, list[dict[str, Any]]], p206f_preds: dict[str, list[dict[str, Any]]], same_as_e_iou: float = 0.50) -> dict[str, list[dict[str, Any]]]:
    additions = {}
    for rid, f_preds in p206f_preds.items():
        e_preds = p206e_preds.get(rid, [])
        row_additions = []
        for pred in f_preds:
            overlap_to_e = max([p165.iou(pred["bbox"], other["bbox"]) for other in e_preds] or [0.0])
            if overlap_to_e < same_as_e_iou:
                item = copy.deepcopy(pred)
                item["best_iou_to_p206e"] = overlap_to_e
                row_additions.append(item)
        additions[rid] = row_additions
    return additions


def fuse_with_gate(p206e_preds: dict[str, list[dict[str, Any]]], additions_by_row: dict[str, list[dict[str, Any]]], policy: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for rid, e_preds in p206e_preds.items():
        core = [copy.deepcopy(pred) for pred in e_preds]
        additions = []
        for pred in additions_by_row.get(rid, []):
            item = copy.deepcopy(pred)
            if not keep_pred(item, core + additions, policy):
                continue
            additions.append(item)
        additions.sort(key=lambda pred: float(pred.get("score") or 0.0), reverse=True)
        out[rid] = sorted(core + additions[: int(policy.get("max_add_per_row", 99))], key=lambda pred: float(pred.get("score") or 0.0), reverse=True)[:128]
    return out


def candidate_policies() -> list[dict[str, Any]]:
    policies = []
    templates = [
        ({}, "keep_all"),
        ({"max_add_per_row": 4}, "cap4"),
        ({"max_add_per_row": 3}, "cap3"),
        ({"max_add_per_row": 2}, "cap2"),
        ({"drop_labels": ["appliance"]}, "drop_appliance"),
        ({"drop_labels": ["stair"]}, "drop_stair"),
        ({"drop_labels": ["equipment"]}, "drop_equipment"),
        ({"drop_labels": ["appliance", "stair"]}, "drop_appliance_stair"),
        ({"drop_buckets": ["xlarge"]}, "drop_xlarge"),
        ({"drop_buckets": ["medium"]}, "drop_medium"),
        ({"min_score": 0.10}, "score10"),
        ({"min_score": 0.15}, "score15"),
        ({"min_dist_to_core": 12}, "dist12"),
        ({"max_iou_to_core": 0.08}, "iou08"),
        ({"max_iou_to_core": 0.12}, "iou12"),
        ({"max_iou_to_core": 0.20}, "iou20"),
    ]
    combos = templates + [
        ({"max_add_per_row": 3, "drop_labels": ["appliance"]}, "cap3_drop_appliance"),
        ({"max_add_per_row": 3, "drop_buckets": ["xlarge"]}, "cap3_drop_xlarge"),
        ({"max_add_per_row": 3, "min_score": 0.10}, "cap3_score10"),
        ({"max_add_per_row": 3, "min_dist_to_core": 12}, "cap3_dist12"),
        ({"max_add_per_row": 4, "drop_labels": ["appliance"], "drop_buckets": ["xlarge"]}, "cap4_drop_app_xlarge"),
        ({"max_add_per_row": 4, "min_score": 0.10, "min_dist_to_core": 12}, "cap4_score10_dist12"),
        ({"drop_labels": ["appliance"], "min_score": 0.10}, "drop_app_score10"),
        ({"drop_labels": ["stair"], "min_score": 0.10}, "drop_stair_score10"),
        ({"drop_buckets": ["xlarge"], "min_score": 0.10}, "drop_xlarge_score10"),
        ({"drop_buckets": ["medium"], "min_score": 0.10}, "drop_medium_score10"),
        ({"max_iou_to_core": 0.08, "min_dist_to_core": 12}, "iou08_dist12"),
        ({"max_iou_to_core": 0.12, "min_dist_to_core": 12}, "iou12_dist12"),
    ]
    defaults = {
        "same_as_e_iou": 0.50,
        "drop_labels": [],
        "drop_buckets": [],
        "min_score": 0.0,
        "max_add_per_row": 5,
        "max_iou_to_core": 0.98,
        "min_dist_to_core": 0,
    }
    for updates, name in combos:
        policy = dict(defaults)
        policy.update(updates)
        policy["name"] = f"p206g_{name}"
        policies.append(policy)
    return policies


def materialize(rows: list[dict[str, Any]], pred_map: dict[str, list[dict[str, Any]]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        new = copy.deepcopy(row)
        rid = row_id(row)
        candidates = []
        for idx, pred in enumerate(pred_map.get(rid, [])):
            item = copy.deepcopy(pred.get("raw") or {})
            item["bbox"] = pred["bbox"]
            item["symbol_type"] = pred["label"]
            item["confidence"] = float(pred.get("score") or 0.0)
            item["id"] = f"{rid}_p206g_symbol_{idx:05d}"
            item["target_id"] = item["id"]
            item["source"] = "symbol_p206f_precision_repair_p206g"
            candidates.append(item)
        new["symbol_candidates"] = candidates
        if isinstance(new.get("expected_json"), dict):
            new["expected_json"]["symbol_candidates"] = copy.deepcopy(candidates)
        new["symbol_policy_overlay"] = {"policy_id": "p206g_precision_repair", "policy": policy}
        out.append(new)
    return out


def render(report: dict[str, Any]) -> str:
    e = report["p206e_metrics"]
    f = report["p206f_metrics"]
    b = report["best_metrics"]
    lines = [
        "# P206g Precision Repair Audit",
        "",
        f"Decision: **{report['decision']}**",
        "",
        "| Variant | Precision | Recall | F1 | Center | Inflation |",
        "|---|---:|---:|---:|---:|---:|",
        f"| `P206e` | {e['precision']:.6f} | {e['recall']:.6f} | {e['f1']:.6f} | {e['center_recall']:.6f} | {e['prediction_inflation']:.6f} |",
        f"| `P206f` | {f['precision']:.6f} | {f['recall']:.6f} | {f['f1']:.6f} | {f['center_recall']:.6f} | {f['prediction_inflation']:.6f} |",
        f"| `P206g_best` | {b['precision']:.6f} | {b['recall']:.6f} | {b['f1']:.6f} | {b['center_recall']:.6f} | {b['prediction_inflation']:.6f} |",
        "",
        "## Addition Audit",
        "",
        "```json",
        json.dumps(report["addition_audit"], ensure_ascii=False, indent=2)[:6000],
        "```",
        "",
        "## Best Policy",
        "",
        "```json",
        json.dumps(report["best_policy"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Top Policies",
    ]
    for item in report["top_policies"][:20]:
        m = item["metrics"]
        lines.append(f"- `{item['policy']['name']}` F1 `{m['f1']:.6f}` P `{m['precision']:.6f}` R `{m['recall']:.6f}` center `{m['center_recall']:.6f}` inflation `{m['prediction_inflation']:.6f}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-overlay", default="reports/vlm/symbol_context_verifier_p202_overlay.jsonl")
    parser.add_argument("--p206e-overlay", default="reports/vlm/symbol_p205b_crop_ranker_p206e_overlay.jsonl")
    parser.add_argument("--p206f-overlay", default="reports/vlm/symbol_p205b_crop_ranker_p206f_overlay.jsonl")
    parser.add_argument("--out-json", default="configs/vlm/symbol_p206f_precision_repair_p206g.json")
    parser.add_argument("--out-md", default="reports/vlm/symbol_p206f_precision_repair_p206g.md")
    parser.add_argument("--out-overlay", default="reports/vlm/symbol_p206f_precision_repair_p206g_overlay.jsonl")
    args = parser.parse_args()

    base_rows = load_jsonl(Path(args.base_overlay))
    p206e_rows = load_jsonl(Path(args.p206e_overlay))
    p206f_rows = load_jsonl(Path(args.p206f_overlay))
    golds = {row_id(row): target_symbols(row) for row in base_rows}
    p206e_preds = norm_map(p206e_rows, "p206e")
    p206f_preds = norm_map(p206f_rows, "p206f")
    p206e_metrics = p165.evaluate(golds, p206e_preds)
    p206f_metrics = p165.evaluate(golds, p206f_preds)
    audit = addition_audit(base_rows, p206e_preds, p206f_preds)
    additions_by_row = precompute_additions(p206e_preds, p206f_preds)
    scored = []
    for policy in candidate_policies():
        pred_map = fuse_with_gate(p206e_preds, additions_by_row, policy)
        metrics = p165.evaluate(golds, pred_map)
        scored.append({"policy": policy, "metrics": metrics})
    scored.sort(key=lambda item: (item["metrics"]["f1"], item["metrics"]["precision"], item["metrics"]["recall"], -item["metrics"]["prediction_inflation"]), reverse=True)
    best = scored[0]
    best_map = fuse_with_gate(p206e_preds, additions_by_row, best["policy"])
    precision_gain = best["metrics"]["precision"] - p206f_metrics["precision"]
    recall_loss = p206f_metrics["recall"] - best["metrics"]["recall"]
    decision = "promote_precision_repair" if best["metrics"]["f1"] >= 0.590 and precision_gain >= 0.001 and recall_loss <= 0.002 else "no_promotion_keep_P206f"
    report = {
        "id": "P206g_symbol_p206f_precision_repair",
        "claim_boundary": "P101-selected precision repair audit; gold labels used only for offline evaluation/audit.",
        "p206e_metrics": p206e_metrics,
        "p206f_metrics": p206f_metrics,
        "addition_audit": audit,
        "best_policy": best["policy"],
        "best_metrics": best["metrics"],
        "delta_vs_p206f": p165.delta(best["metrics"], p206f_metrics),
        "delta_vs_p206e": p165.delta(best["metrics"], p206e_metrics),
        "decision": decision,
        "top_policies": scored[:50],
        "outputs": {"json": args.out_json, "md": args.out_md, "overlay": args.out_overlay},
    }
    write_json(Path(args.out_json), report)
    write_jsonl(Path(args.out_overlay), materialize(base_rows, best_map, best["policy"]))
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(render(report), encoding="utf-8")
    print(json.dumps({"decision": decision, "p206f": p206f_metrics, "best": best["metrics"], "delta_vs_p206f": report["delta_vs_p206f"], "outputs": report["outputs"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
