#!/usr/bin/env python3
"""Evaluate non-learning support-aware symbol candidate compressors."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from analyze_symbol_subset_failure_p117 import bbox_iou, center_covered, reconstruct_page_golds


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL = ROOT / "reports/vlm/full_public_raster_symbol_eval_subset_p119_expanded_grid.json"
DEFAULT_DATASET = ROOT / "datasets/symbol_selector_subset_p122/all.jsonl"
DEFAULT_JSON = ROOT / "configs/vlm/symbol_support_rule_compressor_p123.json"
DEFAULT_REPORT = ROOT / "reports/vlm/symbol_support_rule_compressor_p123.md"

CENTER_GATE = 0.851394
TINY_IOU_GATE = 0.393013
INFLATION_GATE = 7.919152


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def group_by_page(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["row_id"])].append(row)
    for page_rows in grouped.values():
        page_rows.sort(key=lambda item: float(item["runtime_features"]["score"]), reverse=True)
    return dict(grouped)


def keep_by_rule(row: dict[str, Any], policy: dict[str, Any]) -> bool:
    f = row["runtime_features"]
    score = float(f["score"])
    label_rank = float(f["score_rank_label"])
    page_rank = float(f["score_rank_page"])
    same_higher = float(f["max_iou_higher_score_same_label"])
    any_higher = float(f["max_iou_higher_score_any"])
    same_overlap_count = float(f["higher_score_overlap_same_label_count"])
    if score < float(policy["min_score"]):
        return False
    if page_rank > float(policy["max_page_rank"]):
        return False
    if label_rank > float(policy["max_label_rank"]):
        return False
    if same_higher >= float(policy["drop_same_label_iou_ge"]):
        return False
    if policy.get("drop_any_label_iou_ge") is not None and any_higher >= float(policy["drop_any_label_iou_ge"]):
        return False
    if same_overlap_count >= float(policy["drop_same_label_higher_overlap_count_ge"]):
        return False
    return True


def score_policy(page_rows: dict[str, list[dict[str, Any]]], page_golds: dict[str, dict[str, dict[str, Any]]], policy: dict[str, Any]) -> dict[str, Any]:
    totals = Counter()
    by_area = Counter()
    by_area_iou = Counter()
    typed_correct = 0
    offline_kept = Counter()
    selected_label_counts = Counter()
    for row_id, gold_map in page_golds.items():
        selected = [row for row in page_rows.get(row_id, []) if keep_by_rule(row, policy)]
        used_iou: set[int] = set()
        used_center: set[int] = set()
        for item in selected:
            selected_label_counts[item["predicted"]["label"]] += 1
            if item["offline_label"]["keep_any"]:
                offline_kept["keep_any"] += 1
            else:
                offline_kept["negative"] += 1
        for gold in gold_map.values():
            gold_box = [float(v) for v in gold["bbox"]]
            bucket = area_bucket(gold_box)
            by_area[bucket] += 1
            best_iou = 0.0
            best_iou_index = None
            center_index = None
            for idx, row in enumerate(selected):
                pred_box = [float(v) for v in row["predicted"]["bbox"]]
                iou = bbox_iou(pred_box, gold_box)
                if iou > best_iou:
                    best_iou = iou
                    best_iou_index = idx
                if center_index is None and idx not in used_center and center_covered(pred_box, gold_box):
                    center_index = idx
            if best_iou_index is not None and best_iou >= 0.30 and best_iou_index not in used_iou:
                used_iou.add(best_iou_index)
                totals["matched_iou"] += 1
                by_area_iou[bucket] += 1
                if str(selected[best_iou_index]["predicted"]["label"]) == str(gold["label"]):
                    typed_correct += 1
            if center_index is not None:
                used_center.add(center_index)
                totals["matched_center"] += 1
        totals["gold"] += len(gold_map)
        totals["predicted"] += len(selected)
    precision = totals["matched_iou"] / max(totals["predicted"], 1)
    recall = totals["matched_iou"] / max(totals["gold"], 1)
    tiny_iou = by_area_iou["tiny_le_64"] / max(by_area["tiny_le_64"], 1)
    center = totals["matched_center"] / max(totals["gold"], 1)
    inflation = totals["predicted"] / max(totals["gold"], 1)
    return {
        "policy": policy,
        "matched": int(totals["matched_iou"]),
        "predicted": int(totals["predicted"]),
        "gold": int(totals["gold"]),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall), 6),
        "center_recall": round(center, 6),
        "tiny_iou_recall": round(tiny_iou, 6),
        "candidate_inflation": round(inflation, 6),
        "typed_accuracy_on_iou_matches": round(typed_correct / max(totals["matched_iou"], 1), 6),
        "area_iou_recall": {bucket: round(by_area_iou[bucket] / max(by_area[bucket], 1), 6) for bucket in sorted(by_area)},
        "selected_offline_label_counts": dict(offline_kept),
        "selected_label_counts": dict(selected_label_counts.most_common()),
        "passes_center": center > CENTER_GATE,
        "passes_tiny_iou": tiny_iou > TINY_IOU_GATE,
        "passes_inflation": inflation <= INFLATION_GATE,
        "passes_all_gates": center > CENTER_GATE and tiny_iou > TINY_IOU_GATE and inflation <= INFLATION_GATE,
    }


def policy_grid() -> list[dict[str, Any]]:
    policies: list[dict[str, Any]] = []
    for min_score in [0.004, 0.005, 0.006, 0.007, 0.008]:
        for max_page_rank in [90, 120, 160]:
            for max_label_rank in [20, 30, 45]:
                for same_iou in [0.30, 0.45, 0.60, 0.75]:
                    policies.append({
                        "name": f"s{min_score}_p{max_page_rank}_l{max_label_rank}_same{same_iou}",
                        "min_score": min_score,
                        "max_page_rank": max_page_rank,
                        "max_label_rank": max_label_rank,
                        "drop_same_label_iou_ge": same_iou,
                        "drop_any_label_iou_ge": None,
                        "drop_same_label_higher_overlap_count_ge": 8,
                    })
    # A few stricter cross-label duplicate filters.
    for min_score in [0.004, 0.005, 0.006]:
        for any_iou in [0.75, 0.85, 0.95]:
            policies.append({
                "name": f"s{min_score}_p160_l45_same075_any{any_iou}",
                "min_score": min_score,
                "max_page_rank": 160,
                "max_label_rank": 45,
                "drop_same_label_iou_ge": 0.75,
                "drop_any_label_iou_ge": any_iou,
                "drop_same_label_higher_overlap_count_ge": 8,
            })
    return policies


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default=str(DEFAULT_EVAL))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output", default=str(DEFAULT_JSON))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    args = parser.parse_args()

    eval_path = Path(args.eval)
    page_golds = reconstruct_page_golds(json.loads(eval_path.read_text(encoding="utf-8")))
    rows = load_jsonl(Path(args.dataset))
    page_rows = group_by_page(rows)
    results = [score_policy(page_rows, page_golds, policy) for policy in policy_grid()]
    results.sort(
        key=lambda row: (
            row["passes_all_gates"],
            row["center_recall"] > CENTER_GATE,
            row["tiny_iou_recall"] > TINY_IOU_GATE,
            row["candidate_inflation"] <= INFLATION_GATE,
            row["f1"],
            row["center_recall"],
            row["tiny_iou_recall"],
            -row["candidate_inflation"],
        ),
        reverse=True,
    )
    passing = [row for row in results if row["passes_all_gates"]]
    best = results[0] if results else None
    summary = {
        "id": "SCI-P2-123-symbol-support-rule-compressor-baseline",
        "claim_boundary": "Rule-compressor subset baseline only; no runtime adoption or full locked claim.",
        "inputs": {"eval": rel(eval_path), "dataset": rel(Path(args.dataset))},
        "baseline_gates": {
            "center_recall_gt": CENTER_GATE,
            "tiny_iou_recall_gt": TINY_IOU_GATE,
            "candidate_inflation_lte": INFLATION_GATE,
        },
        "policies_evaluated": len(results),
        "passing_rows": passing[:20],
        "best_row": best,
        "top_rows": results[:20],
        "decision": "rule_compressor_passes_subset_gates" if passing else "rule_compressor_does_not_pass_subset_gates",
        "recommendation": (
            "Validate passing policy with a proper prediction export and larger split before adoption."
            if passing
            else "Rule baseline is insufficient; train/evaluate a lightweight selector on P122 train/dev."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# P2-123 Symbol Support Rule Compressor Baseline",
        "",
        "## Decision",
        "",
        f"- Decision: `{summary['decision']}`",
        f"- Policies evaluated: `{len(results)}`",
        f"- Passing policies: `{len(passing)}`",
        f"- Recommendation: {summary['recommendation']}",
        "",
        "## Best Policy",
        "",
        "```json",
        json.dumps(best, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Top Policies",
        "",
        "| Policy | Center | Tiny IoU | Inflation | F1 | Precision | Recall | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results[:10]:
        lines.append(
            f"| `{row['policy']['name']}` | {row['center_recall']:.6f} | {row['tiny_iou_recall']:.6f} | "
            f"{row['candidate_inflation']:.6f} | {row['f1']:.6f} | {row['precision']:.6f} | {row['recall']:.6f} | "
            f"{str(row['passes_all_gates']).lower()} |"
        )
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": rel(output), "report": rel(report), "decision": summary["decision"], "passing": len(passing), "best": best}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
