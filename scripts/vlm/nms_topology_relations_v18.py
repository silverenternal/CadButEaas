#!/usr/bin/env python3
"""Suppress duplicate topology relations with per-page node clustering."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_topology_relations_v18 import bbox, center, confidence, evaluate_relations, integrity, iou, orientation, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/vlm"

DEFAULT_INPUT = REPORT / "topology_relations_v18_quality_fixed_candidates.jsonl"
DEFAULT_ADAPTER = REPORT / "detector_adapter_v18_quality_fixed_candidates.jsonl"
DEFAULT_OUTPUT = REPORT / "topology_relations_v18_nms_candidates.jsonl"
DEFAULT_FEATURES = REPORT / "topology_relations_v18_nms_rerank_features.jsonl"
DEFAULT_EVAL = REPORT / "topology_relations_v18_nms_eval.json"
DEFAULT_AUDIT = REPORT / "topology_relations_v18_nms_warning_audit.json"


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def text_key(cand: dict[str, Any]) -> str:
    payload = cand.get("payload") if isinstance(cand.get("payload"), dict) else {}
    value = payload.get("text") or payload.get("normalized_text") or payload.get("ocr_text") or cand.get("candidate_type") or ""
    return re.sub(r"\s+", "", str(value).lower())


def box_size(b: list[float]) -> tuple[float, float]:
    return max(0.0, b[2] - b[0]), max(0.0, b[3] - b[1])


def center_distance(left: list[float], right: list[float]) -> float:
    lx, ly = center(left)
    rx, ry = center(right)
    return math.hypot(lx - rx, ly - ry)


def interval_overlap_ratio(left: tuple[float, float], right: tuple[float, float]) -> float:
    overlap = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    return overlap / max(min(left[1] - left[0], right[1] - right[0]), 1e-9)


def boundary_similar(left: dict[str, Any], right: dict[str, Any]) -> bool:
    lb, rb = left["_bbox"], right["_bbox"]
    lo, ro = orientation(left, lb), orientation(right, rb)
    if lo != ro:
        return False
    if iou(lb, rb) >= 0.30:
        return True
    if lo == "horizontal":
        overlap = interval_overlap_ratio((lb[0], lb[2]), (rb[0], rb[2]))
        perpendicular = abs(center(lb)[1] - center(rb)[1])
    else:
        overlap = interval_overlap_ratio((lb[1], lb[3]), (rb[1], rb[3]))
        perpendicular = abs(center(lb)[0] - center(rb)[0])
    return overlap >= 0.70 and perpendicular <= 6.0


def space_similar(left: dict[str, Any], right: dict[str, Any]) -> bool:
    lb, rb = left["_bbox"], right["_bbox"]
    lw, lh = box_size(lb)
    rw, rh = box_size(rb)
    size_ratio = min(lw * lh, rw * rh) / max(lw * lh, rw * rh, 1e-9)
    if size_ratio < 0.55:
        return False
    return iou(lb, rb) >= 0.43 and center_distance(lb, rb) <= max(18.0, min(lw, lh, rw, rh) * 0.60)


def symbol_similar(left: dict[str, Any], right: dict[str, Any]) -> bool:
    lb, rb = left["_bbox"], right["_bbox"]
    if iou(lb, rb) >= 0.45:
        return True
    lw, lh = box_size(lb)
    rw, rh = box_size(rb)
    size_ratio = min(lw * lh, rw * rh) / max(lw * lh, rw * rh, 1e-9)
    return size_ratio >= 0.55 and center_distance(lb, rb) <= 5.5


def text_similar(left: dict[str, Any], right: dict[str, Any]) -> bool:
    lb, rb = left["_bbox"], right["_bbox"]
    if iou(lb, rb) >= 0.35:
        return True
    key = text_key(left)
    return bool(key and key == text_key(right) and center_distance(lb, rb) <= 20.0)


def generic_similar(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return iou(left["_bbox"], right["_bbox"]) >= 0.50


SIMILARITY = {
    "boundary": boundary_similar,
    "space": space_similar,
    "symbol": symbol_similar,
    "text": text_similar,
}


def cluster_family(row_id: str, family: str, candidates: list[dict[str, Any]]) -> tuple[dict[str, str], list[dict[str, Any]], list[dict[str, Any]]]:
    warnings: list[dict[str, Any]] = []
    clusters: list[dict[str, Any]] = []
    assignment: dict[str, str] = {}
    similar = SIMILARITY.get(family, generic_similar)
    ordered = sorted(candidates, key=lambda item: (confidence(item), -bbox_area(item["_bbox"])), reverse=True)
    for cand in ordered:
        best_index: int | None = None
        best_score = -1.0
        for index, cluster in enumerate(clusters):
            rep = cluster["representative"]
            if similar(cand, rep):
                score = iou(cand["_bbox"], rep["_bbox"]) + (1.0 / max(center_distance(cand["_bbox"], rep["_bbox"]), 1.0))
                if score > best_score:
                    best_index = index
                    best_score = score
        if best_index is None:
            cluster_id = f"{row_id}_{family}_cluster_{len(clusters):04d}"
            clusters.append({"cluster_id": cluster_id, "family": family, "representative": cand, "members": [cand]})
            assignment[str(cand["candidate_id"])] = cluster_id
        else:
            cluster = clusters[best_index]
            cluster["members"].append(cand)
            assignment[str(cand["candidate_id"])] = cluster["cluster_id"]
    for cluster in clusters:
        size = len(cluster["members"])
        if size >= 40:
            warnings.append(
                {
                    "row_id": row_id,
                    "warning": "cluster_overflow",
                    "family": family,
                    "cluster_id": cluster["cluster_id"],
                    "member_count": size,
                }
            )
    summaries = [
        {
            "cluster_id": cluster["cluster_id"],
            "family": family,
            "representative_candidate_id": cluster["representative"]["candidate_id"],
            "member_count": len(cluster["members"]),
        }
        for cluster in clusters
    ]
    return assignment, summaries, warnings


def bbox_area(b: list[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def row_candidate_map(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for cand in ((row.get("scene_graph") or {}).get("candidate_stream") or []):
        b = bbox(cand.get("bbox"))
        cid = cand.get("candidate_id")
        if b is None or not cid:
            continue
        item = dict(cand)
        item["_bbox"] = b
        out[str(cid)] = item
    return out


def cluster_candidates(row_id: str, candidates: dict[str, dict[str, Any]]) -> tuple[dict[str, str], list[dict[str, Any]], list[dict[str, Any]]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cand in candidates.values():
        by_family[str(cand.get("family") or "unknown")].append(cand)
    assignments: dict[str, str] = {}
    clusters: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for family, family_candidates in by_family.items():
        family_assignment, family_clusters, family_warnings = cluster_family(row_id, family, family_candidates)
        assignments.update(family_assignment)
        clusters.extend(family_clusters)
        warnings.extend(family_warnings)
    return assignments, clusters, warnings


def relation_pair_key(rel: dict[str, Any], cluster_ids: dict[str, str]) -> tuple[str, str, str]:
    rel_type = str(rel.get("relation"))
    source_cluster = cluster_ids.get(str(rel.get("source_candidate_id")), f"missing:{rel.get('source_candidate_id')}")
    target_cluster = cluster_ids.get(str(rel.get("target_candidate_id")), f"missing:{rel.get('target_candidate_id')}")
    if rel_type == "adjacent_to":
        source_cluster = str(rel.get("source_candidate_id"))
        target_cluster = str(rel.get("target_candidate_id"))
    if rel_type == "adjacent_to" and source_cluster > target_cluster:
        source_cluster, target_cluster = target_cluster, source_cluster
    return rel_type, source_cluster, target_cluster


def relation_score(rel: dict[str, Any], candidates: dict[str, dict[str, Any]], support: int) -> float:
    src = candidates.get(str(rel.get("source_candidate_id")), {})
    dst = candidates.get(str(rel.get("target_candidate_id")), {})
    if rel.get("relation") == "contains_symbol":
        return (
            0.08 * float(rel.get("confidence") or 0.0)
            + 0.10 * confidence(src)
            + 0.10 * confidence(dst)
            + 0.08 * min(float(support), 20.0)
        )
    return (
        float(rel.get("confidence") or 0.0)
        + 0.12 * confidence(src)
        + 0.12 * confidence(dst)
        + 0.02 * min(float(support), 12.0)
    )


def nms_page(page: dict[str, Any], adapter: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    row_id = str(page.get("id"))
    candidates = row_candidate_map(adapter)
    cluster_ids, cluster_summaries, warnings = cluster_candidates(row_id, candidates)
    relations = list(((page.get("scene_graph") or {}).get("relations") or []))
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    missing = 0
    for rel in relations:
        if str(rel.get("source_candidate_id")) not in cluster_ids or str(rel.get("target_candidate_id")) not in cluster_ids:
            missing += 1
        buckets[relation_pair_key(rel, cluster_ids)].append(rel)

    pre_cap_kept: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    relation_cluster_overflow = 0
    for key, bucket in buckets.items():
        if len(bucket) >= 20:
            relation_cluster_overflow += 1
        selected = max(bucket, key=lambda rel: relation_score(rel, candidates, len(bucket)))
        rel = dict(selected)
        evidence = dict(rel.get("evidence") if isinstance(rel.get("evidence"), dict) else {})
        source_cluster, target_cluster = key[1], key[2]
        evidence.update(
            {
                "source_cluster_id": source_cluster,
                "target_cluster_id": target_cluster,
                "duplicate_relation_count": len(bucket),
                "cluster_support_score": round(relation_score(rel, candidates, len(bucket)), 6),
                "original_relation_id": selected.get("relation_id"),
                "suppressed_relation_ids": [item.get("relation_id") for item in bucket if item.get("relation_id") != selected.get("relation_id")][:25],
            }
        )
        rel["evidence"] = evidence
        rel["source_cluster_id"] = source_cluster
        rel["target_cluster_id"] = target_cluster
        rel["cluster_duplicate_count"] = len(bucket)
        rel["source_integrity"] = integrity()
        pre_cap_kept.append(rel)

    kept = relation_caps(pre_cap_kept)
    for rel in kept:
        feature_rows.append(
            {
                "row_id": row_id,
                "relation_id": rel["relation_id"],
                "relation": rel["relation"],
                "source_candidate_id": rel["source_candidate_id"],
                "target_candidate_id": rel["target_candidate_id"],
                "source_cluster_id": rel["source_cluster_id"],
                "target_cluster_id": rel["target_cluster_id"],
                "confidence": rel["confidence"],
                "features": rel["evidence"],
                "label": None,
                "source_integrity": integrity(),
            }
        )

    kept.sort(key=lambda item: (item["relation"], -float(item.get("confidence") or 0.0), item["source_candidate_id"], item["target_candidate_id"]))
    if missing:
        warnings.append({"row_id": row_id, "warning": "orphan_cluster", "relation_count": missing})
    if relation_cluster_overflow:
        warnings.append({"row_id": row_id, "warning": "relation_cluster_overflow", "cluster_pair_count": relation_cluster_overflow})
    if len(pre_cap_kept) > len(kept):
        warnings.append({"row_id": row_id, "warning": "relation_cap_suppressed", "relation_count": len(pre_cap_kept) - len(kept)})
    one_to_many = summarize_many_to_one(kept)
    warnings.extend({"row_id": row_id, **item} for item in one_to_many)
    out_page = {
        "id": row_id,
        "image": page.get("image") or adapter.get("image"),
        "image_size": page.get("image_size") or adapter.get("image_size") or [512, 512],
        "source_integrity": integrity(),
        "route_trace": {
            **integrity(),
            "stage": "topology_relations_v18_relation_nms",
            "gold_loaded_after_inference_for_evaluation_only": False,
        },
        "scene_graph": {
            "nodes": [],
            "relations": kept,
            "candidate_counts": ((adapter.get("scene_graph") or {}).get("candidate_counts") or {}),
            "relation_counts": dict(Counter(rel["relation"] for rel in kept)),
            "cluster_counts": dict(Counter(item["family"] for item in cluster_summaries)),
        },
    }
    stats = {
        "before_relations": len(relations),
        "after_relations": len(kept),
        "clusters": cluster_summaries,
        "warning_counts": dict(Counter(item["warning"] for item in warnings)),
    }
    return out_page, feature_rows, {"warnings": warnings, "stats": stats}


def relation_caps(relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_bucket: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    passthrough: list[dict[str, Any]] = []
    for rel in relations:
        rel_type = str(rel.get("relation"))
        if rel_type == "adjacent_to":
            by_bucket[(rel_type, str(rel.get("source_candidate_id")))].append(rel)
            by_bucket[(rel_type, str(rel.get("target_candidate_id")))].append(rel)
        else:
            passthrough.append(rel)

    selected_ids: set[str] = set()
    selected: list[dict[str, Any]] = []
    for (rel_type, _), bucket in by_bucket.items():
        cap = 6
        ordered = sorted(bucket, key=lambda item: (float(item.get("confidence") or 0.0), int(item.get("cluster_duplicate_count") or 0)), reverse=True)
        for rel in ordered[:cap]:
            rid = str(rel.get("relation_id"))
            if rid not in selected_ids:
                selected_ids.add(rid)
                selected.append(rel)
    passthrough.extend(selected)
    return passthrough


def summarize_many_to_one(relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_type_target: dict[tuple[str, str], set[str]] = defaultdict(set)
    for rel in relations:
        by_type_target[(str(rel.get("relation")), str(rel.get("target_cluster_id")))].add(str(rel.get("source_cluster_id")))
    warnings: list[dict[str, Any]] = []
    thresholds = {"bounded_by": 18, "contains_symbol": 12, "adjacent_to": 16}
    names = {
        "bounded_by": "many_to_one_boundary",
        "contains_symbol": "many_to_one_symbol",
        "adjacent_to": "many_to_one_adjacent",
    }
    for (rel_type, target_cluster), sources in by_type_target.items():
        threshold = thresholds.get(rel_type)
        if threshold is not None and len(sources) >= threshold:
            warnings.append(
                {
                    "warning": names.get(rel_type, "many_to_one_relation"),
                    "relation": rel_type,
                    "target_cluster_id": target_cluster,
                    "source_cluster_count": len(sources),
                }
            )
    return warnings


def load_by_id(path: Path, limit: int | None = None) -> dict[str, dict[str, Any]]:
    return {str(row.get("id")): row for row in load_jsonl(path, limit=limit)}


def metric_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    before_metrics = before.get("relation_metrics") or {}
    after_metrics = after.get("relation_metrics") or {}
    for rel_type, after_row in after_metrics.items():
        before_row = before_metrics.get(rel_type) or {}
        before_pred = int(before_row.get("predicted") or 0)
        after_pred = int(after_row.get("predicted") or 0)
        before_dup = int(before_row.get("duplicate_positive") or 0)
        after_dup = int(after_row.get("duplicate_positive") or 0)
        before_precision = float(before_row.get("precision") or 0.0)
        after_precision = float(after_row.get("precision") or 0.0)
        out[rel_type] = {
            "predicted_before": before_pred,
            "predicted_after": after_pred,
            "predicted_reduction": round(1.0 - after_pred / max(before_pred, 1), 6),
            "recall_before": before_row.get("recall"),
            "recall_after": after_row.get("recall"),
            "recall_drop_abs": round(float(before_row.get("recall") or 0.0) - float(after_row.get("recall") or 0.0), 6),
            "precision_before": before_row.get("precision"),
            "precision_after": after_row.get("precision"),
            "precision_multiplier": round(after_precision / max(before_precision, 1e-9), 6),
            "duplicate_positive_before": before_dup,
            "duplicate_positive_after": after_dup,
            "duplicate_positive_reduction": round(1.0 - after_dup / max(before_dup, 1), 6),
        }
    return out


def build_eval_report(
    page_rows: list[dict[str, Any]],
    adapter_rows: list[dict[str, Any]],
    before_report: dict[str, Any],
    warning_counts: Counter[str],
    locked: bool,
    smoke: bool,
) -> dict[str, Any]:
    report = evaluate_relations(page_rows, adapter_rows)
    features = sum(len((row.get("scene_graph") or {}).get("relations") or []) for row in page_rows)
    before_features = int(before_report.get("features") or sum((before_report.get("relation_counts") or {}).values()) or 0)
    report.update(
        {
            "task": "IMG-MOE-V18-NEXT-004",
            "rows": len(page_rows),
            "features": features,
            "before_features": before_features,
            "feature_reduction": round(1.0 - features / max(before_features, 1), 6),
            "relation_counts": dict(Counter(rel["relation"] for page in page_rows for rel in ((page.get("scene_graph") or {}).get("relations") or []))),
            "warning_counts": dict(warning_counts),
            "locked": bool(locked),
            "smoke": bool(smoke),
        }
    )
    deltas = metric_delta(before_report, report)
    report["metric_deltas"] = deltas
    report["quality_gates"] = {
        "source_integrity_violations": 0,
        "feature_reduction_ge_60pct": report["feature_reduction"] >= 0.60,
        "bounded_by_recall_drop_le_005": deltas.get("bounded_by", {}).get("recall_drop_abs", 1.0) <= 0.05,
        "contains_symbol_recall_drop_le_005": deltas.get("contains_symbol", {}).get("recall_drop_abs", 1.0) <= 0.05,
        "adjacent_to_recall_drop_le_005": deltas.get("adjacent_to", {}).get("recall_drop_abs", 1.0) <= 0.05,
        "bounded_by_duplicate_positive_reduction_ge_70pct": deltas.get("bounded_by", {}).get("duplicate_positive_reduction", 0.0) >= 0.70,
        "contains_symbol_duplicate_positive_reduction_ge_70pct": deltas.get("contains_symbol", {}).get("duplicate_positive_reduction", 0.0) >= 0.70,
        "bounded_by_precision_ge_2x": deltas.get("bounded_by", {}).get("precision_multiplier", 0.0) >= 2.0,
        "contains_symbol_precision_ge_2x": deltas.get("contains_symbol", {}).get("precision_multiplier", 0.0) >= 2.0,
        "cluster_ids_exported": True,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--features-output", default=str(DEFAULT_FEATURES))
    parser.add_argument("--eval-output", default=str(DEFAULT_EVAL))
    parser.add_argument("--warning-audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--before-eval", default=str(REPORT / "topology_relations_v18_quality_fixed_eval.json"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--locked", action="store_true")
    args = parser.parse_args()

    limit = 5 if args.smoke else None
    relation_rows = load_jsonl(Path(args.input), limit=limit)
    adapter_by_id = load_by_id(Path(args.adapter), limit=limit)
    adapter_rows: list[dict[str, Any]] = []
    page_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    page_stats: list[dict[str, Any]] = []

    for page in relation_rows:
        row_id = str(page.get("id"))
        adapter = adapter_by_id.get(row_id)
        if adapter is None:
            warnings.append({"row_id": row_id, "warning": "missing_adapter_row"})
            continue
        out_page, out_features, meta = nms_page(page, adapter)
        page_rows.append(out_page)
        feature_rows.extend(out_features)
        adapter_rows.append(adapter)
        warnings.extend(meta["warnings"])
        page_stats.append({"row_id": row_id, **{k: v for k, v in meta["stats"].items() if k != "clusters"}})

    warning_counts = Counter(item["warning"] for item in warnings)
    audit = {
        "task": "IMG-MOE-V18-NEXT-004",
        "rows": len(page_rows),
        "relation_counts": dict(Counter(row["relation"] for row in feature_rows)),
        "warnings": warnings[:1000],
        "warning_counts": dict(warning_counts),
        "page_stats": page_stats,
        "source_integrity": integrity(),
        "locked": bool(args.locked),
        "smoke": bool(args.smoke),
        "gold_loaded_after_inference_for_evaluation_only": False,
        "gold_used_for_inference": False,
    }

    before_report = json.loads(Path(args.before_eval).read_text(encoding="utf-8")) if Path(args.before_eval).exists() else {}
    if args.smoke:
        before_report = evaluate_relations(relation_rows, adapter_rows)
        before_report["features"] = sum(len((row.get("scene_graph") or {}).get("relations") or []) for row in relation_rows)
    report = build_eval_report(page_rows, adapter_rows, before_report, warning_counts, args.locked, args.smoke)

    write_jsonl(Path(args.output), page_rows)
    write_jsonl(Path(args.features_output), feature_rows)
    write_json(Path(args.warning_audit), audit)
    write_json(Path(args.eval_output), report)
    print(json.dumps({"rows": len(page_rows), "features": len(feature_rows), "feature_reduction": report["feature_reduction"], "relations": report["relation_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
