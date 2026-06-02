#!/usr/bin/env python3
"""Audit oracle-hit gold targets missed by the recall-preserving selector."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from typing import Any

from apply_symbol_detector_recall_preserving_policy_v47 import feature_score, group_pages, page_gold_targets, safe_float, select_rows
from train_symbol_support_suppression_v35 import load_jsonl, source_path
from train_symbol_tile_detector_v20 import rel, write_json, write_jsonl


def target_hits(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    hits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        labels = row.get("labels") or {}
        target = str(labels.get("best_iou_target_id") or "")
        if target and safe_float(labels.get("best_iou")) >= 0.30:
            hits[target].append(row)
    for items in hits.values():
        items.sort(key=feature_score, reverse=True)
    return hits


def rank_in(items: list[dict[str, Any]], candidate_id: str, key_func: Any = feature_score) -> int | None:
    for idx, row in enumerate(sorted(items, key=key_func, reverse=True), start=1):
        if str(row.get("candidate_id") or "") == candidate_id:
            return idx
    return None


def classify_reason(
    best_row: dict[str, Any],
    page_rows: list[dict[str, Any]],
    selected_ids: set[str],
    policy: dict[str, Any],
) -> dict[str, Any]:
    score_threshold = float(policy["score_threshold"])
    cluster_topk = int(policy["cluster_topk"])
    label_topk = int(policy["label_topk"])
    max_per_page = int(policy["max_per_page"])
    score = safe_float(best_row.get("score"))
    label = str(best_row.get("label") or "")
    cluster_key = str(best_row.get("cluster_key") or best_row.get("cluster_id") or "")
    candidates_above = [row for row in page_rows if safe_float(row.get("score")) >= score_threshold]
    by_cluster = [row for row in candidates_above if str(row.get("cluster_key") or row.get("cluster_id") or "") == cluster_key]
    by_label = [row for row in candidates_above if str(row.get("label") or "") == label]
    cid = str(best_row.get("candidate_id") or "")
    cluster_rank = rank_in(by_cluster, cid)
    label_rank = rank_in(by_label, cid)
    page_rank = rank_in(candidates_above, cid)
    selected_same_target = []
    target = str((best_row.get("labels") or {}).get("best_iou_target_id") or "")
    for row in page_rows:
        if str(row.get("candidate_id") or "") not in selected_ids:
            continue
        labels = row.get("labels") or {}
        if str(labels.get("best_iou_target_id") or "") == target and safe_float(labels.get("best_iou")) >= 0.30:
            selected_same_target.append(row)
    if score < score_threshold:
        reason = "below_score_threshold"
    elif selected_same_target:
        reason = "selected_wrong_duplicate_or_type"
    elif cluster_rank is not None and cluster_rank > cluster_topk:
        reason = "cluster_rank_gt_topk"
    elif label_rank is not None and label_rank > label_topk:
        reason = "label_rank_gt_topk"
    elif page_rank is not None and page_rank > max_per_page:
        reason = "page_rank_gt_cap"
    else:
        reason = "ordering_or_unknown"
    return {
        "reason": reason,
        "score": round(score, 6),
        "feature_score": round(feature_score(best_row), 6),
        "cluster_rank": cluster_rank,
        "label_rank": label_rank,
        "page_rank": page_rank,
        "selected_same_target_count": len(selected_same_target),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="datasets/symbol_detector_listwise_recovery_v47_p2_transfer_locked_full_v2/manifest.json")
    parser.add_argument("--split", default="smoke_eval", choices=["train", "dev", "smoke_eval", "all"])
    parser.add_argument("--score-threshold", type=float, default=0.02)
    parser.add_argument("--cluster-topk", type=int, default=1)
    parser.add_argument("--label-topk", type=int, default=4)
    parser.add_argument("--max-per-page", type=int, default=200)
    parser.add_argument("--output", default="reports/vlm/symbol_recall_preserving_missed_oracle_v62_audit.json")
    parser.add_argument("--cases-output", default="reports/vlm/symbol_recall_preserving_missed_oracle_v62_cases.jsonl")
    args = parser.parse_args()

    manifest = json.loads(source_path(args.data).read_text(encoding="utf-8"))
    rows = load_jsonl(source_path(manifest["outputs"]["rows"]))
    pages = group_pages(rows, args.split)
    policy = {
        "score_threshold": args.score_threshold,
        "cluster_topk": args.cluster_topk,
        "label_topk": args.label_topk,
        "max_per_page": args.max_per_page,
    }
    totals = Counter()
    by_label = Counter()
    by_area = Counter()
    cases: list[dict[str, Any]] = []
    for page_id, page_rows in pages.items():
        gold = page_gold_targets(page_rows)
        oracle_hits = target_hits(page_rows)
        selected = select_rows(page_rows, args.score_threshold, args.cluster_topk, args.label_topk, args.max_per_page)
        selected_ids = {str(row.get("candidate_id") or "") for row in selected}
        selected_hits = set(target_hits(selected))
        for target_id, hit_rows in oracle_hits.items():
            if target_id in selected_hits:
                continue
            best_row = hit_rows[0]
            meta = gold.get(target_id, {})
            reason = classify_reason(best_row, page_rows, selected_ids, policy)
            label = meta.get("label", "unknown")
            area = meta.get("area_bucket", "unknown")
            totals["missed_oracle_hit"] += 1
            totals[f"reason:{reason['reason']}"] += 1
            by_label[f"{reason['reason']}:{label}"] += 1
            by_area[f"{reason['reason']}:{area}"] += 1
            cases.append(
                {
                    "page_id": page_id,
                    "target_id": target_id,
                    "target_label": label,
                    "target_area": area,
                    "best_candidate_id": best_row.get("candidate_id"),
                    "best_candidate_label": best_row.get("label"),
                    "best_iou": (best_row.get("labels") or {}).get("best_iou"),
                    **reason,
                    "source_integrity": {"gold_used_for_inference": False, "runtime_uses_svg_or_cad_geometry": False},
                }
            )
    report = {
        "version": "symbol_recall_preserving_missed_oracle_v62",
        "data": rel(source_path(args.data)),
        "split": args.split,
        "policy": policy,
        "source_integrity": {
            "purpose": "offline audit only; not used by selector inference",
            "gold_used_for_inference": False,
            "runtime_uses_svg_or_cad_geometry": False,
        },
        "summary": dict(totals.most_common()),
        "reason_by_label": dict(by_label.most_common()),
        "reason_by_area": dict(by_area.most_common()),
        "top_recoverable_reasons": [
            {"reason": key.removeprefix("reason:"), "count": value}
            for key, value in totals.most_common()
            if key.startswith("reason:")
        ],
    }
    write_json(source_path(args.output), report)
    write_jsonl(source_path(args.cases_output), cases)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
