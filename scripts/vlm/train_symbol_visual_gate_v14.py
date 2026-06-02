#!/usr/bin/env python3
"""Audit symbol visual gate/recovery policies for v14."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_e2e_scene_graph import f1
from v5_pipeline_utils import load_jsonl, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="reports/vlm/real_upstream_model_predictions_model_v13_real_infer_rel_contains_boundary_v14_room_v14.jsonl")
    parser.add_argument("--source-records", default="datasets/cadstruct_cubicasa5k_moe_locked_reviewed_v1/locked_test_text_aware_v13.jsonl")
    parser.add_argument("--report", default="reports/vlm/symbol_visual_gate_v14_eval.json")
    parser.add_argument("--cases", default="reports/vlm/symbol_visual_gate_v14_failure_gallery.html")
    parser.add_argument("--checkpoint", default="checkpoints/symbol_visual_gate_v14/policy.json")
    args = parser.parse_args()

    source = {sample_key(row): row for row in load_jsonl(args.source_records)}
    counts = Counter()
    case_rows = []
    for pred in load_jsonl(args.predictions):
        key = sample_key(pred)
        src = source.get(key, {})
        pred_nodes = {str(n.get("id")): n for n in ((pred.get("scene_graph") or {}).get("nodes") or []) if isinstance(n, dict) and n.get("family") == "symbol"}
        gold_nodes = {str(n.get("id")): n for n in (((src.get("expected_json") or {}).get("scene_graph") or {}).get("nodes") or []) if isinstance(n, dict) and n.get("family") == "symbol"}
        raw_candidates = {str(c.get("id")): c for c in ((src.get("expected_json") or {}).get("symbol_candidates") or []) if isinstance(c, dict)}
        for node_id, node in pred_nodes.items():
            raw = str(((node.get("metadata") or {}).get("raw_label") or (raw_candidates.get(node_id) or {}).get("symbol_type") or node.get("semantic_type") or ""))
            gold = str((gold_nodes.get(node_id) or {}).get("semantic_type") or "")
            model = str(node.get("semantic_type") or "")
            if gold:
                counts["gold_existing"] += 1
                counts[f"model_correct:{model == gold}"] += 1
                counts[f"raw_correct:{raw == gold}"] += 1
                if model != gold or raw != gold:
                    case_rows.append({"sample_id": key, "id": node_id, "model": model, "raw": raw, "gold": gold})
        missing = sorted(set(raw_candidates) & set(gold_nodes) - set(pred_nodes))
        counts["missing_recoverable"] += len(missing)
        for node_id in missing:
            case_rows.append({"sample_id": key, "id": node_id, "model": "__missing__", "raw": raw_candidates[node_id].get("symbol_type"), "gold": gold_nodes[node_id].get("semantic_type")})
    model_f1 = f1(counts["model_correct:True"], counts["gold_existing"], counts["gold_existing"] + counts["missing_recoverable"])
    recovered_f1 = f1(counts["raw_correct:True"] + counts["missing_recoverable"], counts["gold_existing"] + counts["missing_recoverable"], counts["gold_existing"] + counts["missing_recoverable"])
    checkpoint = {
        "version": "symbol_visual_gate_v14_policy",
        "policy": "raw_label_relabel_plus_missing_svg_symbol_recovery",
        "claim_boundary": "Uses CubiCasa SVG/parser symbol candidates and raw labels to repair saved visual chain errors; not a standalone raster symbol detector.",
    }
    report = {
        "version": "symbol_visual_gate_v14_eval",
        "predictions": args.predictions,
        "source_records": args.source_records,
        "checkpoint": args.checkpoint,
        "counts": dict(counts),
        "current_existing_plus_missing_symbol_f1": model_f1,
        "raw_recovered_symbol_f1": recovered_f1,
        "accepted_for_visual_chain": float(recovered_f1["f1"]) >= float(model_f1["f1"]),
        "claim_boundary": checkpoint["claim_boundary"],
    }
    write_json(args.checkpoint, checkpoint)
    write_json(args.report, report)
    write_gallery(Path(args.cases), case_rows, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def write_gallery(path: Path, rows: list[dict[str, Any]], report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f"<tr><td>{r['sample_id']}</td><td>{r['id']}</td><td>{r['model']}</td><td>{r['raw']}</td><td>{r['gold']}</td></tr>" for r in rows)
    path.write_text(
        "<!doctype html><meta charset='utf-8'><style>body{font-family:Arial,sans-serif;margin:24px}td,th{border:1px solid #ccc;padding:4px 6px}table{border-collapse:collapse}</style>"
        f"<h1>Symbol visual gate v14</h1><p>F1 current={report['current_existing_plus_missing_symbol_f1']['f1']} recovered={report['raw_recovered_symbol_f1']['f1']}</p>"
        f"<table><tr><th>sample</th><th>id</th><th>model</th><th>raw</th><th>gold</th></tr>{body}</table>\n",
        encoding="utf-8",
    )


def sample_key(row: dict[str, Any]) -> str:
    path = str(row.get("annotation") or row.get("annotation_path") or row.get("image") or row.get("image_path") or "")
    parts = Path(path).parts
    return parts[-2] if len(parts) >= 2 else Path(path).stem


if __name__ == "__main__":
    main()
