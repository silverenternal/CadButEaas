#!/usr/bin/env python3
"""Fuse P206g precision overlay with P211 recall proposals for P212 rescue."""
from __future__ import annotations

import argparse
import json
import resource
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
P206G = ROOT / "reports/vlm/symbol_p206f_precision_repair_p206g_overlay.jsonl"
P211 = ROOT / "reports/vlm/symbol_recall_detector_p211_20k_yolov8s_page_predictions.jsonl"
REPORT = ROOT / "reports/vlm/symbol_p206g_p211_fusion_p212.md"
CONFIG = ROOT / "configs/vlm/symbol_p206g_p211_fusion_p212.json"
OVERLAY = ROOT / "reports/vlm/symbol_p206g_p211_fusion_p212_overlay.jsonl"


LABELS = ["appliance", "bathtub", "column", "equipment", "generic_symbol", "shower", "sink", "stair", "table"]
LABEL_TO_ID = {label: index + 1 for index, label in enumerate(LABELS)}
FORBIDDEN_RUNTIME_FIELDS = ["raw_label", "semantic_type", "expected_json", "annotation_path", "svg_geometry"]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def bbox_iou(left: list[float], right: list[float]) -> float:
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    la = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    ra = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return inter / max(la + ra - inter, 1e-9)


def center_covered(pred: list[float], gold: list[float], margin: float = 2.0) -> bool:
    cx = (gold[0] + gold[2]) / 2.0
    cy = (gold[1] + gold[3]) / 2.0
    return pred[0] - margin <= cx <= pred[2] + margin and pred[1] - margin <= cy <= pred[3] + margin


def area_bucket(box: list[float]) -> str:
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    if area <= 64:
        return "tiny_le_64"
    if area <= 256:
        return "small_le_256"
    if area <= 1024:
        return "medium_le_1024"
    if area <= 4096:
        return "large_le_4096"
    return "xlarge_gt_4096"


def nwd_similarity(left: list[float], right: list[float], normalizer: float = 32.0) -> float:
    import math
    lcx = (left[0] + left[2]) / 2.0; lcy = (left[1] + left[3]) / 2.0
    rcx = (right[0] + right[2]) / 2.0; rcy = (right[1] + right[3]) / 2.0
    lw = max(0.0, left[2] - left[0]); lh = max(0.0, left[3] - left[1])
    rw = max(0.0, right[2] - right[0]); rh = max(0.0, right[3] - right[1])
    distance = ((lcx - rcx) ** 2 + (lcy - rcy) ** 2 + ((lw - rw) ** 2 + (lh - rh) ** 2) / 4.0) ** 0.5
    return math.exp(-distance / max(normalizer, 1e-6))



def simple_nms(preds: list[dict[str, Any]], iou_threshold: float) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for pred in sorted(preds, key=lambda item: float(item.get("score", 0.0)), reverse=True):
        label = pred.get("label")
        box = [float(v) for v in pred["bbox"]]
        if any(other.get("label") == label and bbox_iou(box, [float(v) for v in other["bbox"]]) >= iou_threshold for other in kept):
            continue
        kept.append(pred)
    return kept


def score_predictions(
    page_preds: dict[str, list[dict[str, Any]]],
    page_golds: dict[str, dict[str, dict[str, Any]]],
    score_threshold: float,
    nms_threshold: float,
    max_per_page: int,
    tile_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    totals = Counter()
    by_label = Counter(); by_label_center = Counter(); by_label_iou = Counter()
    by_area = Counter(); by_area_center = Counter(); by_area_iou = Counter(); by_area_nwd_070 = Counter()
    typed_correct = 0
    for row_id, gold_map in page_golds.items():
        filtered = [pred for pred in page_preds.get(row_id, []) if float(pred.get("score", 0.0)) >= score_threshold]
        merged = simple_nms(filtered, nms_threshold)[:max_per_page]
        used_iou: set[int] = set(); used_center: set[int] = set()
        for gold in gold_map.values():
            gold_box = [float(v) for v in gold["bbox"]]
            label = str(gold["label"])
            bucket = area_bucket(gold_box)
            by_label[label] += 1; by_area[bucket] += 1
            best_iou = 0.0; best_iou_index: int | None = None; best_nwd = 0.0; center_index: int | None = None
            for pred_index, pred in enumerate(merged):
                pred_box = [float(v) for v in pred["bbox"]]
                iou = bbox_iou(pred_box, gold_box)
                if iou > best_iou:
                    best_iou = iou; best_iou_index = pred_index
                best_nwd = max(best_nwd, nwd_similarity(pred_box, gold_box))
                if center_index is None and pred_index not in used_center and center_covered(pred_box, gold_box):
                    center_index = pred_index
            if best_nwd >= 0.70:
                by_area_nwd_070[bucket] += 1
            if best_iou_index is not None and best_iou >= 0.30 and best_iou_index not in used_iou:
                used_iou.add(best_iou_index); totals["matched_iou"] += 1; by_label_iou[label] += 1; by_area_iou[bucket] += 1
                if merged[best_iou_index].get("label") == label:
                    typed_correct += 1
            if center_index is not None:
                used_center.add(center_index); totals["matched_center"] += 1; by_label_center[label] += 1; by_area_center[bucket] += 1
        totals["gold"] += len(gold_map); totals["predicted"] += len(merged)
        predictions.append({"row_id": row_id, "predicted_symbols": merged, "gold_symbol_count": len(gold_map)})
    precision = totals["matched_iou"] / max(totals["predicted"], 1)
    recall = totals["matched_iou"] / max(totals["gold"], 1)
    return {
        "rows": len(page_golds), "tiles": tile_count,
        "symbol_bbox_iou_0_30": {"matched": int(totals["matched_iou"]), "predicted": int(totals["predicted"]), "gold": int(totals["gold"]), "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall), 6)},
        "symbol_bbox_center_recall": round(totals["matched_center"] / max(totals["gold"], 1), 6),
        "candidate_inflation": round(totals["predicted"] / max(totals["gold"], 1), 6),
        "typed_accuracy_on_iou_matches": round(typed_correct / max(totals["matched_iou"], 1), 6),
        "type_center_recall": {k: round(by_label_center[k] / max(v, 1), 6) for k, v in sorted(by_label.items())},
        "type_iou_recall": {k: round(by_label_iou[k] / max(v, 1), 6) for k, v in sorted(by_label.items())},
        "area_center_recall": {k: round(by_area_center[k] / max(v, 1), 6) for k, v in sorted(by_area.items())},
        "area_iou_recall": {k: round(by_area_iou[k] / max(v, 1), 6) for k, v in sorted(by_area.items())},
        "nwd_tiny_box_audit": {"area_recall_at_0_70": {k: round(by_area_nwd_070[k] / max(v, 1), 6) for k, v in sorted(by_area.items())}},
    }, predictions

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_p206g(path: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, dict[str, dict[str, Any]]]]:
    rows = load_jsonl(path)
    preds: dict[str, list[dict[str, Any]]] = {}
    golds: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        row_id = str(row.get("id") or row.get("row_id"))
        row_preds: list[dict[str, Any]] = []
        for index, candidate in enumerate(row.get("symbol_candidates") or []):
            label = str(candidate.get("symbol_type") or candidate.get("label") or "generic_symbol")
            row_preds.append({
                "bbox": [float(v) for v in candidate.get("bbox")],
                "label": label,
                "label_id": LABEL_TO_ID.get(label, LABEL_TO_ID["generic_symbol"]),
                "score": float(candidate.get("confidence") or candidate.get("score") or 1.0),
                "source": "p206g",
                "tile_id": ((candidate.get("metadata") or {}).get("tile_id")),
            })
        preds[row_id] = row_preds
        gold_map: dict[str, dict[str, Any]] = {}
        for target in (row.get("targets") or {}).get("symbol") or []:
            label = str(target.get("semantic_type") or "generic_symbol")
            target_id = str(target.get("target_id") or f"{row_id}_{len(gold_map)}")
            gold_map[target_id] = {"target_id": target_id, "bbox": [float(v) for v in target.get("bbox")], "label": label}
        golds[row_id] = gold_map
    return rows, preds, golds


def load_p211(path: Path, allowed_rows: set[str]) -> dict[str, list[dict[str, Any]]]:
    by_row: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(path):
        row_id = str(row.get("row_id"))
        if row_id not in allowed_rows:
            continue
        for pred in row.get("predicted_symbols") or []:
            label = str(pred.get("label") or "generic_symbol")
            by_row[row_id].append({
                "bbox": [float(v) for v in pred.get("bbox")],
                "label": label,
                "label_id": LABEL_TO_ID.get(label, LABEL_TO_ID["generic_symbol"]),
                "score": float(pred.get("score") or 0.0),
                "source": "p211",
                "tile_id": pred.get("tile_id"),
            })
    return by_row


def conflicts_with_core(candidate: dict[str, Any], core: list[dict[str, Any]], max_iou: float, min_center_dist: float, same_label_only: bool) -> bool:
    box = [float(v) for v in candidate["bbox"]]
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    label = str(candidate.get("label"))
    for pred in core:
        if same_label_only and str(pred.get("label")) != label:
            continue
        other = [float(v) for v in pred["bbox"]]
        if bbox_iou(box, other) >= max_iou:
            return True
        ox = (other[0] + other[2]) / 2.0
        oy = (other[1] + other[3]) / 2.0
        if ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5 <= min_center_dist:
            return True
    return False


def fuse(core_by_row: dict[str, list[dict[str, Any]]], p211_by_row: dict[str, list[dict[str, Any]]], policy: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    thresholds = {str(k): float(v) for k, v in (policy.get("label_thresholds") or {}).items()}
    default_threshold = float(policy.get("default_threshold", 0.9))
    allowed_labels = set(policy.get("allowed_labels") or LABELS)
    max_add_per_row = int(policy.get("max_add_per_row", 3))
    max_iou_to_core = float(policy.get("max_iou_to_core", 0.35))
    min_dist_to_core = float(policy.get("min_dist_to_core", 6.0))
    same_label_only = bool(policy.get("same_label_only", False))
    out: dict[str, list[dict[str, Any]]] = {}
    for row_id, core in core_by_row.items():
        merged = [dict(pred) for pred in core]
        additions: list[dict[str, Any]] = []
        candidates = sorted(p211_by_row.get(row_id, []), key=lambda pred: float(pred.get("score", 0.0)), reverse=True)
        for pred in candidates:
            label = str(pred.get("label") or "generic_symbol")
            if label not in allowed_labels:
                continue
            if float(pred.get("score", 0.0)) < thresholds.get(label, default_threshold):
                continue
            if conflicts_with_core(pred, merged + additions, max_iou_to_core, min_dist_to_core, same_label_only):
                continue
            addition = dict(pred)
            addition["source"] = "p211_added_by_p212_fusion"
            additions.append(addition)
            if len(additions) >= max_add_per_row:
                break
        out[row_id] = merged + additions
    return out


def metric_key(metrics: dict[str, Any], baseline_f1: float) -> tuple[float, ...]:
    iou = metrics["symbol_bbox_iou_0_30"]
    f1 = float(iou["f1"])
    precision = float(iou["precision"])
    recall = float(iou["recall"])
    center = float(metrics["symbol_bbox_center_recall"])
    return (f1 - baseline_f1, f1, precision, recall, center, -float(metrics["candidate_inflation"]))


def build_overlay(rows: list[dict[str, Any]], fused: dict[str, list[dict[str, Any]]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    out=[]
    for row in rows:
        row_id=str(row.get("id") or row.get("row_id"))
        new_row=dict(row)
        candidates=[]
        for index,pred in enumerate(fused.get(row_id, [])):
            candidates.append({
                "id": f"{row_id}_p212_symbol_{index:05d}",
                "target_id": f"{row_id}_p212_symbol_{index:05d}",
                "symbol_type": pred.get("label"),
                "bbox": pred.get("bbox"),
                "confidence": pred.get("score"),
                "source": pred.get("source"),
                "metadata": {"tile_id": pred.get("tile_id"), "fusion_policy": policy.get("name")},
            })
        new_row["symbol_candidates"]=candidates
        new_row["symbol_policy_overlay"]={"policy_id":"p212_p206g_p211_fusion", "policy":policy}
        out.append(new_row)
    return out


def memory_audit() -> dict[str, Any]:
    return {"max_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--p206g", default=str(P206G))
    parser.add_argument("--p211", default=str(P211))
    parser.add_argument("--report", default=str(REPORT))
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--overlay", default=str(OVERLAY))
    parser.add_argument("--max-per-page", type=int, default=700)
    args=parser.parse_args()

    print(json.dumps({"phase":"load_p206g_start"}), flush=True)
    rows, core, golds = load_p206g(Path(args.p206g))
    print(json.dumps({"phase":"load_p211_start", "rows": len(rows), "core_rows": len(core)}), flush=True)
    p211 = load_p211(Path(args.p211), set(core))
    print(json.dumps({"phase":"baseline_score_start", "p211_rows": len(p211)}), flush=True)
    baseline_metrics, _ = score_predictions(core, golds, 0.0, 0.98, args.max_per_page, 0)
    print(json.dumps({"phase":"baseline_score_done", "baseline": baseline_metrics["symbol_bbox_iou_0_30"]}), flush=True)
    baseline_f1 = float(baseline_metrics["symbol_bbox_iou_0_30"]["f1"])
    policies=[]
    label_sets=[
        ["stair","equipment","sink","shower"],
        ["stair","equipment","sink"],
        ["stair","equipment"],
        ["sink","shower"],
        ["shower"],
    ]
    thresholds=[0.65,0.75,0.85,0.9]
    for labels in label_sets:
        for threshold in thresholds:
            for max_add in [1,2,3]:
                for dist in [8,12]:
                    policies.append({
                        "name": f"p212_labels{'-'.join(labels)}_t{threshold}_add{max_add}_d{dist}",
                        "allowed_labels": labels,
                        "default_threshold": threshold,
                        "max_add_per_row": max_add,
                        "max_iou_to_core": 0.25,
                        "min_dist_to_core": dist,
                        "same_label_only": False,
                    })
    reports=[]
    print(json.dumps({"policies": len(policies), "phase": "fusion_grid_start"}), flush=True)
    for policy_index, policy in enumerate(policies, start=1):
        fused=fuse(core,p211,policy)
        metrics,_=score_predictions(fused,golds,0.0,0.98,args.max_per_page,0)
        additions=sum(max(0,len(fused[row_id])-len(core.get(row_id,[]))) for row_id in fused)
        reports.append({"policy":policy,"metrics":metrics,"additions":additions})
        if policy_index % 20 == 0:
            best_so_far = max(reports, key=lambda row: metric_key(row["metrics"], baseline_f1))
            print(json.dumps({"done": policy_index, "total": len(policies), "best_f1": best_so_far["metrics"]["symbol_bbox_iou_0_30"]["f1"], "best_policy": best_so_far["policy"]["name"]}), flush=True)
    reports.sort(key=lambda row: metric_key(row["metrics"], baseline_f1), reverse=True)
    best=reports[0]
    best_fused=fuse(core,p211,best["policy"])
    best_metrics,prediction_rows=score_predictions(best_fused,golds,0.0,0.98,args.max_per_page,0)
    overlay_rows=build_overlay(rows,best_fused,best["policy"])
    write_jsonl(Path(args.overlay), overlay_rows)
    result={
        "id":"P212_p206g_p211_fusion_grid",
        "claim_boundary":"Internal P101/P206g overlay fusion with P211 raster proposals; selected by grid on this split, requires bootstrap/independent validation before paper claim.",
        "source_integrity":{
            "model_input":"raster predictions from P206g and P211 only",
            "raw_label_or_semantic_type_used_as_runtime_feature":False,
            "forbidden_runtime_features":FORBIDDEN_RUNTIME_FIELDS,
            "gold_use":"evaluation_and_policy_selection_only_in_this_audit",
        },
        "inputs":{"p206g":rel(Path(args.p206g)),"p211":rel(Path(args.p211))},
        "baseline_p206g":baseline_metrics,
        "selected":best,
        "top10":reports[:10],
        "outputs":{"overlay":rel(Path(args.overlay)),"report":rel(Path(args.report)),"config":rel(Path(args.config))},
        "memory_audit":memory_audit(),
    }
    write_json(Path(args.config), result)
    lines=[
        "# P212 P206g + P211 Fusion Grid",
        "",
        f"- Baseline P206g F1: {baseline_f1:.6f}",
        f"- Selected policy: `{best['policy']['name']}`",
        f"- Selected F1: {best_metrics['symbol_bbox_iou_0_30']['f1']:.6f}",
        f"- Selected precision: {best_metrics['symbol_bbox_iou_0_30']['precision']:.6f}",
        f"- Selected recall: {best_metrics['symbol_bbox_iou_0_30']['recall']:.6f}",
        f"- Selected center recall: {best_metrics['symbol_bbox_center_recall']:.6f}",
        f"- Candidate inflation: {best_metrics['candidate_inflation']:.6f}",
        f"- Additions: {best['additions']}",
        "",
        "## Claim Boundary",
        result["claim_boundary"],
        "",
        "## Top Policies",
    ]
    for row in reports[:10]:
        m=row['metrics']['symbol_bbox_iou_0_30']
        lines.append(f"- `{row['policy']['name']}`: F1={m['f1']:.6f}, P={m['precision']:.6f}, R={m['recall']:.6f}, add={row['additions']}")
    Path(args.report).parent.mkdir(parents=True,exist_ok=True)
    Path(args.report).write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"baseline":baseline_metrics["symbol_bbox_iou_0_30"],"selected":best_metrics["symbol_bbox_iou_0_30"],"policy":best["policy"],"additions":best["additions"]},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
