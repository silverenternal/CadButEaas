#!/usr/bin/env python3
"""P205 high-recall proposal fusion probe for symbol MoE.

This script is an offline ablation: it merges already-materialized proposal
branches into the current P202 best overlay, scores merged candidates with the
P202 context verifier, and sweeps recall-preserving gates.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch

import sweep_symbol_disagreement_backfill_p165 as p165
import train_symbol_context_verifier_p202 as p202

ROOT = Path(__file__).resolve().parents[2]
WET_LABELS = {"sink", "shower", "bathtub"}
TARGET_LABELS = {"sink", "shower", "equipment", "stair", "appliance", "bathtub"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("row_id") or row.get("id"))


def target_symbols(row: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for idx, item in enumerate((row.get("targets") or {}).get("symbol") or []):
        box = p165.bbox4(item.get("bbox"))
        if box is not None:
            out.append({"id": str(item.get("target_id") or idx), "bbox": box, "bucket": p165.bucket(box)})
    return out


def candidate_from_pred(raw: dict[str, Any], source: str, idx: int) -> dict[str, Any] | None:
    box = p165.bbox4(raw.get("bbox"))
    if box is None:
        return None
    label = str(raw.get("label") or raw.get("symbol_type") or raw.get("semantic_type") or "generic_symbol")
    score = raw.get("score") if raw.get("score") is not None else raw.get("confidence")
    try:
        score_value = float(score)
    except (TypeError, ValueError):
        score_value = 0.0
    item = copy.deepcopy(raw)
    item["bbox"] = box
    item["symbol_type"] = label
    item["confidence"] = score_value
    item.setdefault("metadata", {})["p205_source"] = source
    item.setdefault("metadata", {})["p205_source_index"] = idx
    return item


def load_prediction_map(path: Path, source: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    if not path.exists():
        return out
    for row in load_jsonl(path):
        rid = str(row.get("row_id") or row.get("id"))
        items = row.get("predicted_symbols") or row.get("symbol_candidates") or []
        converted = []
        for idx, raw in enumerate(items):
            item = candidate_from_pred(raw, source, idx)
            if item is not None:
                converted.append(item)
        out[rid] = converted
    return out


def max_iou(item: dict[str, Any], others: list[dict[str, Any]]) -> float:
    box = p165.bbox4(item.get("bbox"))
    if box is None:
        return 0.0
    best = 0.0
    for other in others:
        other_box = p165.bbox4(other.get("bbox"))
        if other_box is not None:
            best = max(best, p165.iou(box, other_box))
    return best


def build_merged_rows(base_rows: list[dict[str, Any]], proposal_maps: list[dict[str, list[dict[str, Any]]]], min_score: float, max_add: int, labels: set[str], novelty_iou: float) -> list[dict[str, Any]]:
    merged = []
    for raw_row in base_rows:
        row = copy.deepcopy(raw_row)
        rid = row_id(row)
        base = copy.deepcopy(row.get("symbol_candidates") or [])
        additions = []
        for proposal_map in proposal_maps:
            for cand in proposal_map.get(rid, []):
                label = str(cand.get("symbol_type") or cand.get("label") or "generic_symbol")
                score = float(cand.get("confidence") or cand.get("score") or 0.0)
                if label not in labels or score < min_score:
                    continue
                if max_iou(cand, base + additions) >= novelty_iou:
                    continue
                additions.append(cand)
        additions.sort(key=lambda item: float(item.get("confidence") or item.get("score") or 0.0), reverse=True)
        selected = additions[:max_add]
        for idx, item in enumerate(selected):
            item["id"] = f"{rid}_p205_add_{idx:05d}"
            item["target_id"] = item["id"]
            item["source"] = "symbol_high_recall_proposals_p205"
        row["symbol_candidates"] = base + selected
        if isinstance(row.get("expected_json"), dict):
            row["expected_json"]["symbol_candidates"] = copy.deepcopy(row["symbol_candidates"])
        merged.append(row)
    return merged


def score_with_p202(rows: list[dict[str, Any]], checkpoint: Path) -> dict[tuple[str, int], float]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    model = p202.MLP(int(ckpt["dim"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    examples, _dim = p202.build_examples(rows)
    scored = p202.score_examples(model, examples)
    return {(ex["row_id"], int(ex["candidate_index"])): float(ex["context_score"]) for ex in scored}


def apply_policy(rows: list[dict[str, Any]], score_map: dict[tuple[str, int], float], threshold: float, protected: set[str], rescue_threshold: float) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for row in rows:
        rid = row_id(row)
        preds = p165.normalized(row.get("symbol_candidates") or [], "p205")
        kept = []
        for idx, pred in enumerate(preds):
            context_score = float(score_map.get((rid, idx), 0.0))
            source = str((pred.get("raw") or {}).get("source") or "")
            is_p205 = source == "symbol_high_recall_proposals_p205"
            required = rescue_threshold if is_p205 and pred["label"] in protected else threshold
            if pred["label"] not in protected and context_score < threshold:
                continue
            if pred["label"] in protected and context_score < required:
                continue
            item = copy.deepcopy(pred)
            raw = copy.deepcopy(item.get("raw") or {})
            raw.setdefault("metadata", {})["p205_context_score"] = round(context_score, 6)
            item["raw"] = raw
            kept.append(item)
        out[rid] = kept
    return out


def materialize(rows: list[dict[str, Any]], pred_map: dict[str, list[dict[str, Any]]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for raw_row in rows:
        row = copy.deepcopy(raw_row)
        rid = row_id(row)
        candidates = []
        for idx, pred in enumerate(pred_map.get(rid, [])):
            item = copy.deepcopy(pred.get("raw") or {})
            item["bbox"] = pred["bbox"]
            item["symbol_type"] = pred["label"]
            item["confidence"] = float(pred["score"])
            item["id"] = f"{rid}_p205_symbol_{idx:05d}"
            item["target_id"] = item["id"]
            item.setdefault("metadata", {})["p205_policy"] = policy
            candidates.append(item)
        row["symbol_candidates"] = candidates
        if isinstance(row.get("expected_json"), dict):
            row["expected_json"]["symbol_candidates"] = copy.deepcopy(candidates)
        row["symbol_policy_overlay"] = {"policy_id": "p205_high_recall_proposals", "policy": policy}
        out.append(row)
    return out


def render(report: dict[str, Any]) -> str:
    lines = [
        "# P205 High-Recall Proposal Rescue Probe",
        "",
        "## Decision",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Baseline F1: {report['baseline_metrics']['f1']:.6f}",
        f"- Best F1: {report['best_metrics']['f1']:.6f}",
        "",
        "## Best Policy",
        "",
        "```json",
        json.dumps(report["best_policy"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Top Policies",
        "",
        "| Rank | F1 | Precision | Recall | Center | Inflation | Added | Policy |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for idx, item in enumerate(report["top_policies"], 1):
        m = item["metrics"]
        lines.append(f"| {idx} | {m['f1']:.6f} | {m['precision']:.6f} | {m['recall']:.6f} | {m['center_recall']:.6f} | {m['prediction_inflation']:.6f} | {item['added_candidates']} | `{item['policy']['name']}` |")
    lines += ["", "## Interpretation", ""]
    for note in report["interpretation"]:
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-overlay", default="reports/vlm/symbol_context_verifier_p202_overlay.jsonl")
    parser.add_argument("--proposal", action="append", default=["reports/vlm/symbol_center_size_p201_crop_verified_predictions.jsonl", "reports/vlm/symbol_sink_shower_specialist_p198_p101_predictions.jsonl"])
    parser.add_argument("--checkpoint", default="checkpoints/symbol_context_verifier_p202/model.pt")
    parser.add_argument("--out-json", default="configs/vlm/symbol_high_recall_proposals_p205.json")
    parser.add_argument("--out-md", default="reports/vlm/symbol_high_recall_proposals_p205_eval.md")
    parser.add_argument("--out-overlay", default="reports/vlm/symbol_high_recall_proposals_p205_overlay.jsonl")
    parser.add_argument("--quick", action="store_true", help="Run a compact policy sweep for fast iteration.")
    args = parser.parse_args()

    base_rows = load_jsonl(ROOT / args.base_overlay)
    golds = {row_id(row): target_symbols(row) for row in base_rows}
    baseline = p165.evaluate(golds, {row_id(row): p165.normalized(row.get("symbol_candidates") or [], "p202") for row in base_rows})
    proposal_maps = [load_prediction_map(ROOT / path, Path(path).stem) for path in args.proposal]
    policies = []
    min_scores = [0.25, 0.35] if args.quick else [0.15, 0.25, 0.35, 0.45, 0.55]
    max_adds = [4, 8] if args.quick else [2, 4, 8, 12, 20]
    label_sets = [WET_LABELS, TARGET_LABELS]
    thresholds = [0.20, 0.25] if args.quick else [0.15, 0.20, 0.25, 0.30]
    rescue_thresholds = [0.10, 0.15] if args.quick else [0.05, 0.10, 0.15, 0.20]
    for min_score in min_scores:
        for max_add in max_adds:
            for labels in label_sets:
                for threshold in thresholds:
                    for rescue_threshold in rescue_thresholds:
                        policies.append({
                            "name": f"p205_s{min_score}_a{max_add}_l{len(labels)}_t{threshold}_r{rescue_threshold}",
                            "min_score": min_score,
                            "max_add_per_row": max_add,
                            "labels": sorted(labels),
                            "context_threshold": threshold,
                            "rescue_threshold": rescue_threshold,
                            "novelty_iou": 0.20,
                        })
    scored = []
    cache: dict[tuple[float, int, tuple[str, ...], float], tuple[list[dict[str, Any]], dict[tuple[str, int], float], int]] = {}
    for policy in policies:
        merge_key = (policy["min_score"], policy["max_add_per_row"], tuple(policy["labels"]), policy["novelty_iou"])
        if merge_key not in cache:
            merged = build_merged_rows(base_rows, proposal_maps, policy["min_score"], policy["max_add_per_row"], set(policy["labels"]), policy["novelty_iou"])
            added = sum(max(0, len(row.get("symbol_candidates") or []) - len(base_rows[idx].get("symbol_candidates") or [])) for idx, row in enumerate(merged))
            score_map = score_with_p202(merged, ROOT / args.checkpoint)
            cache[merge_key] = (merged, score_map, added)
        merged, score_map, added = cache[merge_key]
        pred_map = apply_policy(merged, score_map, policy["context_threshold"], set(policy["labels"]), policy["rescue_threshold"])
        metrics = p165.evaluate(golds, pred_map)
        scored.append({"policy": policy, "metrics": metrics, "added_candidates": added})
    scored.sort(key=lambda item: (item["metrics"]["f1"], item["metrics"]["recall"], item["metrics"]["center_recall"]), reverse=True)
    best = scored[0]
    best_key = (best["policy"]["min_score"], best["policy"]["max_add_per_row"], tuple(best["policy"]["labels"]), best["policy"]["novelty_iou"])
    merged, score_map, _added = cache[best_key]
    best_pred = apply_policy(merged, score_map, best["policy"]["context_threshold"], set(best["policy"]["labels"]), best["policy"]["rescue_threshold"])
    write_jsonl(ROOT / args.out_overlay, materialize(merged, best_pred, best["policy"]))
    decision = "promote" if best["metrics"]["f1"] > baseline["f1"] else "no_promotion_keep_P202"
    report = {
        "id": "P205_high_recall_proposal_rescue_probe",
        "claim_boundary": "Offline P101 proposal-fusion probe. Uses raster-only materialized proposal artifacts at runtime, with gold only for evaluation/policy selection.",
        "inputs": {"base_overlay": args.base_overlay, "proposals": args.proposal, "checkpoint": args.checkpoint},
        "baseline_metrics": baseline,
        "best_policy": best["policy"],
        "best_metrics": best["metrics"],
        "delta_vs_baseline": p165.delta(best["metrics"], baseline),
        "decision": decision,
        "top_policies": scored[:30],
        "interpretation": [
            "This tests whether existing high-recall branches can recover P202 false negatives without retraining.",
            "If no policy promotes, P205 should move to true tiled/SAHI retraining rather than adding more low-quality candidates.",
        ],
        "outputs": {"json": args.out_json, "md": args.out_md, "overlay": args.out_overlay},
    }
    write_json(ROOT / args.out_json, report)
    (ROOT / args.out_md).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / args.out_md).write_text(render(report), encoding="utf-8")
    print(json.dumps({"decision": decision, "baseline": baseline, "best_policy": best["policy"], "best_metrics": best["metrics"], "delta": report["delta_vs_baseline"], "outputs": report["outputs"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
