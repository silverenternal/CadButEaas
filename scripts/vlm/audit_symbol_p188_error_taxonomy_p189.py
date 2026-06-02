#!/usr/bin/env python3
"""P189 symbol error taxonomy and hard-case manifest for P188.

Offline analysis only: uses P101 gold targets to identify failure modes and
produce a training/evaluation hard-case manifest. It does not define runtime
features for inference.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
P182 = ROOT / "reports/vlm/symbol_policy_moe_overlay_p182_best.jsonl"
P188 = ROOT / "reports/vlm/symbol_pro5090_yolov8m_seg_rect_p185_p188_recall_expanded_overlay.jsonl"
OUT_JSON = ROOT / "reports/vlm/symbol_p188_error_taxonomy_p189.json"
OUT_MD = ROOT / "reports/vlm/symbol_p188_error_taxonomy_p189.md"
OUT_MANIFEST = ROOT / "datasets/symbol_hardcase_rescue_p189/manifest.jsonl"
OUT_CONFIG = ROOT / "configs/vlm/symbol_hardcase_rescue_p189.json"


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


def center(box: list[float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def center_covered(pred: list[float], gold: list[float]) -> bool:
    cx, cy = center(gold)
    return pred[0] <= cx <= pred[2] and pred[1] <= cy <= pred[3]


def center_distance(a: list[float], b: list[float]) -> float:
    ax, ay = center(a)
    bx, by = center(b)
    return math.hypot(ax - bx, ay - by)


def nwd_similarity(a: list[float], b: list[float]) -> float:
    acx, acy = center(a)
    bcx, bcy = center(b)
    aw, ah = max(1e-6, a[2] - a[0]), max(1e-6, a[3] - a[1])
    bw, bh = max(1e-6, b[2] - b[0]), max(1e-6, b[3] - b[1])
    dist2 = (acx - bcx) ** 2 + (acy - bcy) ** 2 + ((aw - bw) / 2.0) ** 2 + ((ah - bh) / 2.0) ** 2
    scale = max(math.sqrt(area(a)) + math.sqrt(area(b)), 1e-6)
    return math.exp(-math.sqrt(dist2) / scale)


def label_gold(item: dict[str, Any]) -> str:
    return str(item.get("semantic_type") or item.get("symbol_type") or item.get("label") or item.get("raw_label") or "generic_symbol").lower()


def label_pred(item: dict[str, Any]) -> str:
    return str(item.get("symbol_type") or item.get("semantic_type") or item.get("label") or "generic_symbol").lower()


def score_pred(item: dict[str, Any]) -> float:
    try:
        return float(item.get("confidence") if item.get("confidence") is not None else item.get("score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def golds(row: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for idx, item in enumerate((row.get("targets") or {}).get("symbol") or []):
        box = bbox4(item.get("bbox"))
        if box is None:
            continue
        out.append({"index": idx, "id": str(item.get("target_id") or idx), "bbox": box, "label": label_gold(item), "bucket": bucket(box), "raw": item})
    return out


def preds(row: dict[str, Any], source: str) -> list[dict[str, Any]]:
    out = []
    for idx, item in enumerate(row.get("symbol_candidates") or []):
        box = bbox4(item.get("bbox"))
        if box is None:
            continue
        out.append({"index": idx, "id": str(item.get("id") or idx), "bbox": box, "label": label_pred(item), "score": score_pred(item), "bucket": bucket(box), "source": source, "raw": item})
    return out


def greedy_iou_matches(gold_items: list[dict[str, Any]], pred_items: list[dict[str, Any]], threshold: float = 0.30) -> tuple[dict[int, int], set[int]]:
    pairs = []
    for gi, gold in enumerate(gold_items):
        for pi, pred in enumerate(pred_items):
            overlap = iou(gold["bbox"], pred["bbox"])
            if overlap >= threshold:
                pairs.append((overlap, gi, pi))
    pairs.sort(reverse=True)
    gold_used: set[int] = set()
    pred_used: set[int] = set()
    matches: dict[int, int] = {}
    for _overlap, gi, pi in pairs:
        if gi in gold_used or pi in pred_used:
            continue
        gold_used.add(gi)
        pred_used.add(pi)
        matches[gi] = pi
    return matches, pred_used


def nearest(gold: dict[str, Any], pred_items: list[dict[str, Any]]) -> dict[str, Any]:
    if not pred_items:
        return {"best_iou": 0.0, "best_nwd": 0.0, "best_center_distance": None, "center_covered": False, "best_pred_label": None, "best_pred_score": None}
    best_iou = max(iou(gold["bbox"], pred["bbox"]) for pred in pred_items)
    best_nwd = max(nwd_similarity(gold["bbox"], pred["bbox"]) for pred in pred_items)
    best_dist_pred = min(pred_items, key=lambda pred: center_distance(gold["bbox"], pred["bbox"]))
    return {
        "best_iou": round(best_iou, 6),
        "best_nwd": round(best_nwd, 6),
        "best_center_distance": round(center_distance(gold["bbox"], best_dist_pred["bbox"]), 3),
        "center_covered": any(center_covered(pred["bbox"], gold["bbox"]) for pred in pred_items),
        "best_pred_label": best_dist_pred["label"],
        "best_pred_score": round(float(best_dist_pred["score"]), 6),
    }


def classify_fn(near: dict[str, Any]) -> str:
    if near["best_iou"] >= 0.20:
        return "near_iou_0_20_0_30_box_refine"
    if near["center_covered"]:
        return "center_hit_iou_fail_box_refine"
    if near["best_nwd"] >= 0.70:
        return "nwd_hit_iou_fail_box_refine"
    if near["best_center_distance"] is not None and near["best_center_distance"] <= 24:
        return "close_center_missing_or_small_box"
    return "true_miss_or_low_confidence"


def page_density(count: int) -> str:
    if count <= 16:
        return "sparse"
    if count <= 32:
        return "medium"
    return "dense"


def compare_rows(row: dict[str, Any], p182_row: dict[str, Any], p188_row: dict[str, Any]) -> dict[str, Any]:
    row_id = str(row.get("row_id") or row.get("id"))
    gold_items = golds(row)
    p182_preds = preds(p182_row, "p182")
    p188_preds = preds(p188_row, "p188")
    p182_matches, p182_used = greedy_iou_matches(gold_items, p182_preds)
    p188_matches, p188_used = greedy_iou_matches(gold_items, p188_preds)
    false_negatives = []
    rescued = []
    regressed = []
    for gi, gold in enumerate(gold_items):
        in182 = gi in p182_matches
        in188 = gi in p188_matches
        near188 = nearest(gold, p188_preds)
        near182 = nearest(gold, p182_preds)
        base = {
            "row_id": row_id,
            "image": row.get("image") or row.get("image_path"),
            "target_id": gold["id"],
            "label": gold["label"],
            "bucket": gold["bucket"],
            "bbox": gold["bbox"],
            "page_gold_count": len(gold_items),
            "page_density": page_density(len(gold_items)),
            "p188_nearest": near188,
            "p182_nearest": near182,
        }
        if not in188:
            item = dict(base)
            item["failure_mode"] = classify_fn(near188)
            item["p182_matched"] = in182
            false_negatives.append(item)
        if in188 and not in182:
            rescued.append(base)
        if in182 and not in188:
            regressed.append(base)
    false_positives = []
    for pi, pred in enumerate(p188_preds):
        if pi in p188_used:
            continue
        best_gold = max((iou(pred["bbox"], gold["bbox"]) for gold in gold_items), default=0.0)
        closest = min((center_distance(pred["bbox"], gold["bbox"]) for gold in gold_items), default=None)
        false_positives.append({
            "row_id": row_id,
            "image": row.get("image") or row.get("image_path"),
            "pred_id": pred["id"],
            "label": pred["label"],
            "bucket": pred["bucket"],
            "bbox": pred["bbox"],
            "score": round(float(pred["score"]), 6),
            "page_gold_count": len(gold_items),
            "page_density": page_density(len(gold_items)),
            "best_gold_iou": round(best_gold, 6),
            "closest_gold_center_distance": None if closest is None else round(closest, 3),
            "source_policy": ((pred["raw"].get("metadata") or {}).get("p186_source_policy") if isinstance(pred.get("raw"), dict) else None),
        })
    return {"row_id": row_id, "gold_count": len(gold_items), "p182_pred_count": len(p182_preds), "p188_pred_count": len(p188_preds), "false_negatives": false_negatives, "false_positives": false_positives, "rescued": rescued, "regressed": regressed}


def top_counter(counter: Counter, limit: int = 20) -> list[dict[str, Any]]:
    return [{"key": key, "count": int(value)} for key, value in counter.most_common(limit)]


def summarize(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fn_items = [item for row in rows for item in row["false_negatives"]]
    fp_items = [item for row in rows for item in row["false_positives"]]
    rescued = [item for row in rows for item in row["rescued"]]
    regressed = [item for row in rows for item in row["regressed"]]
    fn_label_bucket = Counter(f"{x['label']}|{x['bucket']}" for x in fn_items)
    fp_label_bucket = Counter(f"{x['label']}|{x['bucket']}" for x in fp_items)
    fn_mode = Counter(x["failure_mode"] for x in fn_items)
    fn_density = Counter(x["page_density"] for x in fn_items)
    fp_density = Counter(x["page_density"] for x in fp_items)
    page_loss = []
    for row in rows:
        page_loss.append({
            "row_id": row["row_id"],
            "gold": row["gold_count"],
            "fn": len(row["false_negatives"]),
            "fp": len(row["false_positives"]),
            "rescued_vs_p182": len(row["rescued"]),
            "regressed_vs_p182": len(row["regressed"]),
            "net_tp_delta_vs_p182": len(row["rescued"]) - len(row["regressed"]),
        })
    page_loss.sort(key=lambda x: (x["fn"] + min(x["fp"], 20), x["fn"], x["fp"]), reverse=True)
    summary = {
        "totals": {
            "rows": len(rows),
            "false_negatives": len(fn_items),
            "false_positives": len(fp_items),
            "rescued_vs_p182": len(rescued),
            "regressed_vs_p182": len(regressed),
            "net_tp_delta_vs_p182": len(rescued) - len(regressed),
        },
        "false_negative_by_label_bucket": top_counter(fn_label_bucket, 30),
        "false_positive_by_label_bucket": top_counter(fp_label_bucket, 30),
        "false_negative_by_mode": top_counter(fn_mode, 20),
        "false_negative_by_density": top_counter(fn_density, 10),
        "false_positive_by_density": top_counter(fp_density, 10),
        "worst_pages": page_loss[:25],
    }
    manifest = build_manifest(fn_items, fp_items, page_loss)
    return summary, manifest


def build_manifest(fn_items: list[dict[str, Any]], fp_items: list[dict[str, Any]], page_loss: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_row: dict[str, dict[str, Any]] = {}
    for item in fn_items:
        rec = by_row.setdefault(item["row_id"], {"row_id": item["row_id"], "image": item["image"], "hard_targets": [], "false_positive_pressure": [], "priority_score": 0})
        mode = item["failure_mode"]
        priority = 3
        if item["bucket"] in {"tiny", "small"}:
            priority += 3
        if item["label"] in {"shower", "sink", "equipment", "column"}:
            priority += 2
        if mode.endswith("box_refine") or "center" in mode or "nwd" in mode:
            priority += 1
        rec["priority_score"] += priority
        rec["hard_targets"].append({
            "target_id": item["target_id"],
            "label": item["label"],
            "bucket": item["bucket"],
            "bbox": item["bbox"],
            "failure_mode": mode,
            "p188_nearest": item["p188_nearest"],
            "recommended_action": recommend_action(item),
        })
    for item in fp_items:
        if item["score"] < 0.45 and item["best_gold_iou"] < 0.05:
            continue
        rec = by_row.setdefault(item["row_id"], {"row_id": item["row_id"], "image": item["image"], "hard_targets": [], "false_positive_pressure": [], "priority_score": 0})
        rec["priority_score"] += 1
        rec["false_positive_pressure"].append({"label": item["label"], "bucket": item["bucket"], "bbox": item["bbox"], "score": item["score"], "best_gold_iou": item["best_gold_iou"]})
    worst_rank = {row["row_id"]: idx for idx, row in enumerate(page_loss[:50])}
    for row_id, rec in by_row.items():
        rec["page_rank"] = worst_rank.get(row_id, 999)
        rec["hard_target_count"] = len(rec["hard_targets"])
        rec["false_positive_count"] = len(rec["false_positive_pressure"])
        rec["recommended_sampling"] = "oversample_4x" if rec["priority_score"] >= 20 else "oversample_2x" if rec["priority_score"] >= 10 else "include_1x"
    out = sorted(by_row.values(), key=lambda x: (x["priority_score"], x["hard_target_count"], -x["page_rank"]), reverse=True)
    return out


def recommend_action(item: dict[str, Any]) -> str:
    if item["failure_mode"] in {"near_iou_0_20_0_30_box_refine", "center_hit_iou_fail_box_refine", "nwd_hit_iou_fail_box_refine"}:
        return "box_refiner_or_size_calibration"
    if item["bucket"] in {"tiny", "small"}:
        return "tiny_small_specialist_high_overlap_crop"
    if item["label"] in {"shower", "sink", "equipment", "column"}:
        return "class_balanced_hardcase_oversampling"
    return "hard_negative_and_recall_oversampling"


def render_md(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = ["# P189 Symbol Error Taxonomy", "", "## Decision", "", "- Stop generic capacity/fusion sweeps; current P188 gain is too small.", "- Next highest-value route is hard-case data + tiny/small specialist + box refinement.", "", "## Totals", "", "| FN | FP | Rescued vs P182 | Regressed vs P182 | Net TP Δ |", "|---:|---:|---:|---:|---:|", f"| {s['totals']['false_negatives']} | {s['totals']['false_positives']} | {s['totals']['rescued_vs_p182']} | {s['totals']['regressed_vs_p182']} | {s['totals']['net_tp_delta_vs_p182']} |", "", "## False Negative Modes", ""]
    for row in s["false_negative_by_mode"]:
        lines.append(f"- `{row['key']}`: `{row['count']}`")
    lines += ["", "## Top FN Label/Buckets", ""]
    for row in s["false_negative_by_label_bucket"][:15]:
        lines.append(f"- `{row['key']}`: `{row['count']}`")
    lines += ["", "## Top FP Label/Buckets", ""]
    for row in s["false_positive_by_label_bucket"][:15]:
        lines.append(f"- `{row['key']}`: `{row['count']}`")
    lines += ["", "## Worst Pages", ""]
    for row in s["worst_pages"][:12]:
        lines.append(f"- `{row['row_id']}` FN `{row['fn']}`, FP `{row['fp']}`, rescued `{row['rescued_vs_p182']}`, regressed `{row['regressed_vs_p182']}`")
    lines += ["", "## Planned Artifacts", ""]
    for key, value in report["outputs"].items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p182", default=str(P182))
    parser.add_argument("--p188", default=str(P188))
    parser.add_argument("--out-json", default=str(OUT_JSON))
    parser.add_argument("--out-md", default=str(OUT_MD))
    parser.add_argument("--out-manifest", default=str(OUT_MANIFEST))
    parser.add_argument("--out-config", default=str(OUT_CONFIG))
    args = parser.parse_args()

    p182_rows = {str(row.get("row_id") or row.get("id")): row for row in load_jsonl(Path(args.p182))}
    p188_rows = {str(row.get("row_id") or row.get("id")): row for row in load_jsonl(Path(args.p188))}
    common_ids = sorted(set(p182_rows) & set(p188_rows))
    analyses = [compare_rows(p188_rows[row_id], p182_rows[row_id], p188_rows[row_id]) for row_id in common_ids]
    summary, manifest = summarize(analyses)
    config = {
        "id": "P189_symbol_hardcase_rescue_set",
        "source_overlays": {"p182": args.p182, "p188": args.p188},
        "claim_boundary": "Offline error taxonomy only; gold targets used for analysis and training-set construction, never as runtime input.",
        "recommended_next": ["P190 hard-case oversampling detector retrain", "P191 tiny/small specialist", "P192 box refinement for near misses"],
        "sampling_rules": {
            "oversample_4x": "priority_score >= 20",
            "oversample_2x": "priority_score >= 10",
            "include_1x": "remaining hard-case pages",
        },
    }
    report = {
        "id": "SCI-P2-189-symbol-p188-error-taxonomy",
        "source_integrity": config["claim_boundary"],
        "inputs": {"p182": args.p182, "p188": args.p188},
        "summary": summary,
        "sample_false_negatives": [item for row in analyses for item in row["false_negatives"]][:80],
        "sample_false_positives": [item for row in analyses for item in row["false_positives"]][:80],
        "outputs": {"json": args.out_json, "md": args.out_md, "manifest": args.out_manifest, "config": args.out_config},
    }
    write_json(Path(args.out_json), report)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(render_md(report), encoding="utf-8")
    write_jsonl(Path(args.out_manifest), manifest)
    write_json(Path(args.out_config), config)
    print(json.dumps({"summary": summary["totals"], "top_fn_modes": summary["false_negative_by_mode"][:5], "top_fn_label_bucket": summary["false_negative_by_label_bucket"][:8], "manifest_rows": len(manifest), "outputs": report["outputs"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
