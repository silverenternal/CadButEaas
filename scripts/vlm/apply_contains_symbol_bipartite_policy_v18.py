#!/usr/bin/env python3
"""Apply the v18 contains_symbol bipartite policy to topology candidates."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_contains_symbol_bipartite_dataset_v18 import build_page_edges
from build_topology_relations_v18 import evaluate_relations, integrity, write_json, write_jsonl
from nms_topology_relations_v18 import load_by_id, load_jsonl
from train_contains_symbol_bipartite_policy_v18 import score_row, select_policy

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/vlm"
CHECKPOINT = ROOT / "checkpoints/contains_symbol_bipartite_policy_v18/model.json"

DEFAULT_INPUT = REPORT / "topology_relations_v18_symbol_boundary_fixed_candidates.jsonl"
DEFAULT_ADAPTER = REPORT / "detector_adapter_v18_symbol_boundary_fixed_candidates.jsonl"
DEFAULT_POLICY = CHECKPOINT
DEFAULT_OUTPUT = REPORT / "topology_relations_v18_bipartite_candidates.jsonl"
DEFAULT_EVAL = REPORT / "topology_relations_v18_bipartite_eval.json"
DEFAULT_AUDIT = REPORT / "topology_relations_v18_bipartite_audit.json"


def load_model(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def with_scores(rows: list[dict[str, Any]], model: dict[str, Any]) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["bipartite_edge_score"] = round(score_row(row, model), 8)
        scored.append(item)
    return scored


def apply_page(page: dict[str, Any], adapter: dict[str, Any], model: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    row_id = str(page.get("id"))
    edge_rows, edge_audit = build_page_edges(page, adapter, gold=None)
    scored_edges = with_scores(edge_rows, model)
    selected_edges = select_policy(scored_edges, model.get("policy") or {})
    selected_ids = {str(row.get("relation_id")) for row in selected_edges}

    relations: list[dict[str, Any]] = []
    suppressed_contains_symbol = 0
    score_by_id = {str(row.get("relation_id")): row for row in scored_edges}
    for rel in ((page.get("scene_graph") or {}).get("relations") or []):
        rel_type = str(rel.get("relation"))
        if rel_type != "contains_symbol":
            relations.append(rel)
            continue
        rid = str(rel.get("relation_id"))
        if rid not in selected_ids:
            suppressed_contains_symbol += 1
            continue
        edge = score_by_id.get(rid) or {}
        out = dict(rel)
        evidence = dict(out.get("evidence") if isinstance(out.get("evidence"), dict) else {})
        evidence.update(
            {
                "contains_symbol_bipartite_policy": "contains_symbol_bipartite_policy_v18",
                "bipartite_edge_score": edge.get("bipartite_edge_score"),
                "component_id": edge.get("component_id"),
                "room_instance_cluster_id": edge.get("room_instance_cluster_id"),
                "symbol_instance_cluster_id": edge.get("symbol_instance_cluster_id"),
                "relation_pair_key": edge.get("relation_pair_key"),
            }
        )
        out["evidence"] = evidence
        out["bipartite_edge_score"] = edge.get("bipartite_edge_score")
        out["component_id"] = edge.get("component_id")
        out["room_instance_cluster_id"] = edge.get("room_instance_cluster_id")
        out["symbol_instance_cluster_id"] = edge.get("symbol_instance_cluster_id")
        out["source_integrity"] = integrity()
        relations.append(out)

    relations.sort(key=lambda item: (str(item.get("relation")), -float(item.get("bipartite_edge_score") or item.get("confidence") or 0.0), str(item.get("source_candidate_id")), str(item.get("target_candidate_id"))))
    out_page = dict(page)
    scene_graph = dict(out_page.get("scene_graph") or {})
    scene_graph["relations"] = relations
    scene_graph["relation_counts"] = dict(Counter(str(rel.get("relation")) for rel in relations))
    out_page["scene_graph"] = scene_graph
    out_page["source_integrity"] = integrity()
    out_page["route_trace"] = {
        **integrity(),
        "stage": "contains_symbol_bipartite_policy_v18",
        "policy": model.get("policy"),
        "gold_loaded_after_inference_for_evaluation_only": False,
        "gold_used_for_inference": False,
    }
    audit = {
        "row_id": row_id,
        "edge_rows": len(edge_rows),
        "selected_contains_symbol": len(selected_ids),
        "suppressed_contains_symbol": suppressed_contains_symbol,
        "edge_audit": edge_audit,
    }
    return out_page, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--eval-output", default=str(DEFAULT_EVAL))
    parser.add_argument("--audit-output", default=str(DEFAULT_AUDIT))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    limit = 5 if args.smoke else args.limit
    pages = load_jsonl(Path(args.input), limit=limit)
    adapter_by_id = load_by_id(Path(args.adapter), limit=limit)
    model = load_model(Path(args.policy))

    out_pages: list[dict[str, Any]] = []
    page_audits: list[dict[str, Any]] = []
    missing_adapter = 0
    for page in pages:
        row_id = str(page.get("id"))
        adapter = adapter_by_id.get(row_id)
        if not adapter:
            missing_adapter += 1
            continue
        out_page, audit = apply_page(page, adapter, model)
        out_pages.append(out_page)
        page_audits.append(audit)

    write_jsonl(Path(args.output), out_pages)
    eval_report = evaluate_relations(out_pages, [adapter_by_id[str(page.get("id"))] for page in out_pages if str(page.get("id")) in adapter_by_id])
    eval_report["rows"] = len(out_pages)
    eval_report["policy"] = model.get("policy")
    eval_report["missing_adapter_rows"] = missing_adapter
    eval_report["source_integrity"] = integrity()
    write_json(Path(args.eval_output), eval_report)

    audit = {
        "task": "IMG-MOE-V18-REBUILD-001.step_c_apply_contains_symbol_bipartite_policy",
        "rows": len(out_pages),
        "missing_adapter_rows": missing_adapter,
        "policy": model.get("policy"),
        "contains_symbol_edges_before": sum(item["edge_rows"] for item in page_audits),
        "contains_symbol_edges_after": sum(item["selected_contains_symbol"] for item in page_audits),
        "contains_symbol_reduction": round(1.0 - sum(item["selected_contains_symbol"] for item in page_audits) / max(sum(item["edge_rows"] for item in page_audits), 1), 6),
        "page_audits_sample": page_audits[:100],
        "source_integrity": integrity(),
        "gold_used_for_inference": False,
    }
    write_json(Path(args.audit_output), audit)
    print(json.dumps({"rows": len(out_pages), "contains_symbol_reduction": audit["contains_symbol_reduction"], "policy": model.get("policy")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
