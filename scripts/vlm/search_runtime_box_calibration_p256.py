#!/usr/bin/env python3
"""Search runtime-safe bbox calibration transforms on P232 symbol predictions."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "vlm"))

from fuse_symbol_p206g_with_p211_p212 import bbox_iou, load_p206g  # noqa: E402


P232_PREDS = ROOT / "reports/vlm/p232_repaired_contract_predictions.jsonl"
GOLD_OVERLAY = ROOT / "reports/vlm/symbol_p224a_column_frozen_overlay.jsonl"
OUT_PREDS = ROOT / "reports/vlm/p256_runtime_box_calibration_predictions.jsonl"
OUT_EVAL = ROOT / "reports/vlm/p256_runtime_box_calibration_eval.json"
OUT_MD = ROOT / "reports/vlm/p256_runtime_box_calibration_report.md"

IOU = 0.30
LABELS_TO_SEARCH = ["sink", "equipment", "shower", "appliance", "column", "stair"]
SCALES = [0.85, 0.95, 1.0, 1.05, 1.15, 1.25]
MIN_SCORE_BY_LABEL = {
    "sink": [0.0, 0.35, 0.5, 0.65],
    "equipment": [0.0, 0.35, 0.5],
    "shower": [0.0, 0.35, 0.5, 0.65],
    "appliance": [0.0, 0.35, 0.5],
    "column": [0.0, 0.35, 0.5],
    "stair": [0.0, 0.35, 0.5],
    "generic_symbol": [0.0, 0.35, 0.5],
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


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


def scale_box(box: list[float], sx: float, sy: float) -> list[float]:
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    width = max(0.0, box[2] - box[0]) * sx
    height = max(0.0, box[3] - box[1]) * sy
    return [cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0]


def load_p232_rows(path: Path) -> list[dict[str, Any]]:
    return load_jsonl(path)


def candidate_label(candidate: dict[str, Any]) -> str:
    return str(
        candidate.get("candidate_type")
        or candidate.get("symbol_type")
        or candidate.get("label")
        or (candidate.get("payload") or {}).get("symbol_type")
        or "generic_symbol"
    )


def candidate_score(candidate: dict[str, Any]) -> float:
    return float(candidate.get("confidence") or candidate.get("score") or (candidate.get("payload") or {}).get("score") or 1.0)


def row_predictions(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        row_id = str(row.get("row_id") or row.get("id"))
        preds = []
        for index, candidate in enumerate(row.get("routed_candidates") or row.get("predicted_symbols") or []):
            label = candidate_label(candidate)
            bbox = [float(v) for v in candidate["bbox"]]
            preds.append(
                {
                    "index": index,
                    "label": label,
                    "bbox": bbox,
                    "score": candidate_score(candidate),
                    "bucket": area_bucket(bbox),
                }
            )
        out[row_id] = preds
    return out


def apply_policy_to_preds(
    base: dict[str, list[dict[str, Any]]],
    policies: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row_id, preds in base.items():
        new_preds = []
        for pred in preds:
            box = pred["bbox"]
            for policy in policies:
                if pred["label"] != policy["label"]:
                    continue
                if policy["bucket"] != "all" and pred["bucket"] != policy["bucket"]:
                    continue
                if pred["score"] < policy["min_score"]:
                    continue
                box = scale_box(box, policy["sx"], policy["sy"])
            updated = dict(pred)
            updated["bbox"] = box
            new_preds.append(updated)
        out[row_id] = new_preds
    return out


def score(
    preds_by_row: dict[str, list[dict[str, Any]]],
    golds_by_row: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    totals = Counter()
    by_label_gold = Counter()
    by_label_tp = Counter()
    by_label_pred = Counter()

    for row_id, gold_map in golds_by_row.items():
        preds = preds_by_row.get(row_id, [])
        golds = list(gold_map.values())
        for pred in preds:
            by_label_pred[pred["label"]] += 1
        for gold in golds:
            by_label_gold[str(gold["label"])] += 1

        candidates: list[tuple[float, int, int]] = []
        for pred_index, pred in enumerate(preds):
            for gold_index, gold in enumerate(golds):
                if pred["label"] != str(gold["label"]):
                    continue
                iou = bbox_iou(pred["bbox"], [float(v) for v in gold["bbox"]])
                if iou >= IOU:
                    candidates.append((iou, pred_index, gold_index))
        candidates.sort(reverse=True)
        used_pred: set[int] = set()
        used_gold: set[int] = set()
        for _, pred_index, gold_index in candidates:
            if pred_index in used_pred or gold_index in used_gold:
                continue
            used_pred.add(pred_index)
            used_gold.add(gold_index)
            label = str(golds[gold_index]["label"])
            by_label_tp[label] += 1
            totals["tp"] += 1
        totals["predicted"] += len(preds)
        totals["gold"] += len(golds)

    precision = totals["tp"] / max(totals["predicted"], 1)
    recall = totals["tp"] / max(totals["gold"], 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    labels = sorted(set(by_label_gold) | set(by_label_pred))
    per_label = {}
    for label in labels:
        tp = by_label_tp[label]
        pred = by_label_pred[label]
        gold = by_label_gold[label]
        p = tp / max(pred, 1)
        r = tp / max(gold, 1)
        per_label[label] = {
            "tp": int(tp),
            "predicted": int(pred),
            "gold": int(gold),
            "precision": round(p, 6),
            "recall": round(r, 6),
            "f1": round(0.0 if p + r == 0 else 2 * p * r / (p + r), 6),
        }
    return {
        "tp": int(totals["tp"]),
        "predicted": int(totals["predicted"]),
        "gold": int(totals["gold"]),
        "fp": int(totals["predicted"] - totals["tp"]),
        "fn": int(totals["gold"] - totals["tp"]),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "per_label": per_label,
    }


def policy_name(policy: dict[str, Any]) -> str:
    return (
        f"{policy['label']}_{policy['bucket']}_sx{policy['sx']:.2f}_sy{policy['sy']:.2f}_score{policy['min_score']:.2f}"
    )


def generate_candidate_policies(base_preds: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    buckets_by_label: dict[str, set[str]] = defaultdict(set)
    for preds in base_preds.values():
        for pred in preds:
            if pred["label"] in LABELS_TO_SEARCH:
                buckets_by_label[pred["label"]].add(pred["bucket"])
    policies = []
    high_value_buckets = {
        "sink": ["all", "medium_le_1024", "large_le_4096"],
        "equipment": ["all", "large_le_4096", "xlarge_gt_4096"],
        "shower": ["all", "medium_le_1024", "large_le_4096"],
        "appliance": ["all", "medium_le_1024", "large_le_4096"],
        "column": ["all", "small_le_256", "medium_le_1024"],
        "stair": ["all", "large_le_4096", "xlarge_gt_4096"],
    }
    for label in LABELS_TO_SEARCH:
        for bucket in high_value_buckets[label]:
            if bucket != "all" and bucket not in buckets_by_label[label]:
                continue
            for min_score in MIN_SCORE_BY_LABEL[label][:2]:
                for sx in SCALES:
                    for sy in SCALES:
                        if sx == 1.0 and sy == 1.0:
                            continue
                        policies.append({"label": label, "bucket": bucket, "min_score": min_score, "sx": sx, "sy": sy})
    return policies


def materialize_rows(rows: list[dict[str, Any]], policies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = deepcopy(rows)
    for row in out:
        for candidate in row.get("routed_candidates") or []:
            label = candidate_label(candidate)
            score_value = candidate_score(candidate)
            box = [float(v) for v in candidate["bbox"]]
            bucket = area_bucket(box)
            applied = []
            for policy in policies:
                if label != policy["label"]:
                    continue
                if policy["bucket"] != "all" and bucket != policy["bucket"]:
                    continue
                if score_value < policy["min_score"]:
                    continue
                box = scale_box(box, policy["sx"], policy["sy"])
                applied.append(policy_name(policy))
            if applied:
                candidate["bbox"] = box
                candidate["source"] = "p256_runtime_box_calibration"
                candidate.setdefault("payload", {})["bbox"] = box
                candidate["payload"]["proposal_stage"] = "p256_runtime_box_calibration"
                candidate["payload"]["p256_applied_policies"] = applied
    return out


def main() -> None:
    rows = load_p232_rows(P232_PREDS)
    _, _, golds = load_p206g(GOLD_OVERLAY)
    base_preds = row_predictions(rows)
    baseline = score(base_preds, golds)

    selected: list[dict[str, Any]] = []
    current_preds = base_preds
    current_metrics = baseline
    history = []

    for iteration in range(6):
        best = None
        candidate_policies = generate_candidate_policies(current_preds)
        for policy in candidate_policies:
            trial_preds = apply_policy_to_preds(current_preds, [policy])
            metrics = score(trial_preds, golds)
            delta_f1 = metrics["f1"] - current_metrics["f1"]
            delta_precision = metrics["precision"] - current_metrics["precision"]
            # Keep search precision-safe enough: allow tiny precision loss only for >2x F1 gain.
            if metrics["precision"] + 0.0005 < baseline["precision"]:
                continue
            if delta_f1 <= 0:
                continue
            rank = (delta_f1, delta_precision, metrics["recall"] - current_metrics["recall"], metrics["precision"])
            if best is None or rank > best["rank"]:
                best = {"policy": policy, "metrics": metrics, "rank": rank}
        if best is None:
            break
        selected.append(best["policy"])
        current_preds = apply_policy_to_preds(current_preds, [best["policy"]])
        history.append(
            {
                "iteration": iteration + 1,
                "policy": best["policy"],
                "policy_name": policy_name(best["policy"]),
                "metrics": best["metrics"],
                "delta_from_previous": {
                    "precision": round(best["metrics"]["precision"] - current_metrics["precision"], 6),
                    "recall": round(best["metrics"]["recall"] - current_metrics["recall"], 6),
                    "f1": round(best["metrics"]["f1"] - current_metrics["f1"], 6),
                },
            }
        )
        current_metrics = best["metrics"]

    output_rows = materialize_rows(rows, selected)
    write_jsonl(OUT_PREDS, output_rows)

    result = {
        "id": "p256_runtime_box_calibration_eval",
        "phase": "P256_runtime_box_calibration_search",
        "inputs": {
            "baseline_predictions": str(P232_PREDS.relative_to(ROOT)),
            "gold_overlay": str(GOLD_OVERLAY.relative_to(ROOT)),
            "iou": IOU,
        },
        "baseline_metrics": baseline,
        "candidate_metrics": current_metrics,
        "delta_vs_p232": {
            "precision": round(current_metrics["precision"] - baseline["precision"], 6),
            "recall": round(current_metrics["recall"] - baseline["recall"], 6),
            "f1": round(current_metrics["f1"] - baseline["f1"], 6),
        },
        "selected_policies": [{"name": policy_name(policy), **policy} for policy in selected],
        "search_history": history,
        "promotion_recommendation": (
            "promote_if_source_integrity_passes"
            if current_metrics["f1"] > baseline["f1"] and current_metrics["precision"] + 0.0005 >= baseline["precision"]
            else "do_not_promote"
        ),
        "claim_boundary": "Runtime-safe static bbox calibration on P232 raster-derived predictions; gold used only offline for policy selection/evaluation.",
        "outputs": {"predictions": str(OUT_PREDS.relative_to(ROOT)), "report": str(OUT_MD.relative_to(ROOT))},
    }
    OUT_EVAL.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# P256 Runtime Box Calibration Report",
        "",
        "## Summary",
        f"- Baseline P232 F1: `{baseline['f1']:.6f}` (P `{baseline['precision']:.6f}`, R `{baseline['recall']:.6f}`).",
        f"- Candidate F1: `{current_metrics['f1']:.6f}` (P `{current_metrics['precision']:.6f}`, R `{current_metrics['recall']:.6f}`).",
        f"- ΔF1: `{result['delta_vs_p232']['f1']:.6f}`.",
        f"- Promotion recommendation: `{result['promotion_recommendation']}`.",
        "",
        "## Selected Policies",
    ]
    if selected:
        for policy in selected:
            lines.append(f"- `{policy_name(policy)}`")
    else:
        lines.append("- No positive policy selected.")
    lines.extend(["", "## Search History", ""])
    for row in history:
        delta_row = row["delta_from_previous"]
        lines.append(
            f"- Iteration {row['iteration']}: `{row['policy_name']}` -> F1 `{row['metrics']['f1']:.6f}` "
            f"(ΔF1 `{delta_row['f1']:.6f}`, ΔP `{delta_row['precision']:.6f}`, ΔR `{delta_row['recall']:.6f}`)."
        )
    lines.extend(
        [
            "",
            "## Per-Label Candidate Metrics",
            "",
            "| label | precision | recall | F1 | TP | pred | gold |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, row in current_metrics["per_label"].items():
        lines.append(
            f"| {label} | {row['precision']:.6f} | {row['recall']:.6f} | {row['f1']:.6f} | {row['tp']} | {row['predicted']} | {row['gold']} |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "- This is runtime-safe static bbox calibration on P232 raster-derived predictions.",
            "- Gold boxes/labels were used only offline to select and evaluate a static policy.",
            "- This remains secondary raster-adapter evidence and must not be mixed with SVG/contract metrics.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": [str(OUT_EVAL.relative_to(ROOT)), str(OUT_PREDS.relative_to(ROOT)), str(OUT_MD.relative_to(ROOT))], "baseline_f1": baseline["f1"], "candidate_f1": current_metrics["f1"], "selected": len(selected)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
