#!/usr/bin/env python3
"""Apply P200 crop verifier scores to gate/relabel symbol candidates."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import sweep_symbol_disagreement_backfill_p165 as p165


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")


def target_symbols(row: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for idx, item in enumerate((row.get("targets") or {}).get("symbol") or []):
        box = p165.bbox4(item.get("bbox"))
        if box is not None:
            out.append({"id": str(item.get("target_id") or idx), "bbox": box, "bucket": p165.bucket(box)})
    return out


def verifier_map(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    out = {}
    for row in load_jsonl(path):
        out[(str(row["row_id"]), int(row["candidate_index"]))] = row
    return out


def apply_policy(rows: list[dict[str, Any]], vmap: dict[tuple[str, int], dict[str, Any]], policy: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out = {}
    fp_threshold = float(policy["fp_threshold"])
    relabel_threshold = float(policy.get("relabel_threshold", 1.1))
    protected_labels = set(policy.get("protected_labels") or [])
    gate_labels = set(policy.get("gate_labels") or [])
    for row in rows:
        row_id = str(row.get("row_id") or row.get("id"))
        preds = p165.normalized(row.get("symbol_candidates") or [], "p200_base")
        kept = []
        for idx, pred in enumerate(preds):
            item = copy.deepcopy(pred)
            v = vmap.get((row_id, idx), {})
            fp_prob = float(v.get("verifier_false_positive_prob") or 0.0)
            pred_label = str(v.get("verifier_pred_label") or "")
            pred_prob = float((v.get("verifier_probs") or {}).get(pred_label) or 0.0)
            label = str(item["label"])
            should_gate = (not gate_labels) or label in gate_labels
            if should_gate and label not in protected_labels and fp_prob >= fp_threshold:
                continue
            if pred_label and pred_label != "false_positive" and pred_label != label and pred_prob >= relabel_threshold:
                item["label"] = pred_label
                raw = copy.deepcopy(item.get("raw") or {})
                raw["symbol_type"] = pred_label
                raw.setdefault("metadata", {})["p200_relabel"] = {"from": label, "to": pred_label, "prob": round(pred_prob, 6)}
                item["raw"] = raw
            raw = copy.deepcopy(item.get("raw") or {})
            raw.setdefault("metadata", {})["p200_verifier"] = {"fp_prob": round(fp_prob, 6), "pred_label": pred_label, "policy": policy["name"]}
            item["raw"] = raw
            kept.append(item)
        out[row_id] = kept
    return out


def materialize(rows: list[dict[str, Any]], preds_by_row: dict[str, list[dict[str, Any]]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for raw_row in rows:
        row = copy.deepcopy(raw_row)
        row_id = str(row.get("row_id") or row.get("id"))
        candidates = []
        for idx, pred in enumerate(preds_by_row.get(row_id, [])):
            item = copy.deepcopy(pred.get("raw") or {})
            item["bbox"] = pred["bbox"]
            item["symbol_type"] = pred["label"]
            item["confidence"] = float(pred["score"])
            item["id"] = f"{row_id}_p200_symbol_{idx:05d}"
            item["target_id"] = item["id"]
            item["source"] = "symbol_crop_verifier_p200"
            item.setdefault("metadata", {})["p200_policy"] = policy
            candidates.append(item)
        row["symbol_candidates"] = candidates
        if isinstance(row.get("expected_json"), dict):
            row["expected_json"]["symbol_candidates"] = [copy.deepcopy(item) for item in candidates]
        row["symbol_policy_overlay"] = {"policy_id": "p200_crop_verifier", "description": "P200 crop verifier gated overlay", "policy": policy}
        out.append(row)
    return out


def render(report: dict[str, Any]) -> str:
    lines = ["# P200 Crop Verifier Fusion", "", "Decision: **" + report["decision"] + "**", "", "| Variant | Precision | Recall | F1 | Center | Inflation |", "|---|---:|---:|---:|---:|---:|"]
    for name, metrics in report["metrics"].items():
        lines.append(f"| `{name}` | {metrics['precision']:.6f} | {metrics['recall']:.6f} | {metrics['f1']:.6f} | {metrics['center_recall']:.6f} | {metrics['prediction_inflation']:.6f} |")
    lines += ["", "## Best Policy", "", "```json", json.dumps(report["best_policy"], ensure_ascii=False, indent=2), "```", "", "## Top Candidates", ""]
    for item in report["top_candidates"][:20]:
        m = item["metrics"]
        lines.append(f"- `{item['policy']['name']}` F1 `{m['f1']:.6f}`, P `{m['precision']:.6f}`, R `{m['recall']:.6f}`, inflation `{m['prediction_inflation']:.6f}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-overlay", default="reports/vlm/symbol_box_refiner_p197b_over_p196c_best_overlay.jsonl")
    parser.add_argument("--verifier-predictions", default="reports/vlm/symbol_crop_verifier_p200_predictions.jsonl")
    parser.add_argument("--out-json", default="configs/vlm/symbol_crop_verifier_p200_fusion.json")
    parser.add_argument("--out-md", default="reports/vlm/symbol_crop_verifier_p200_fusion.md")
    parser.add_argument("--out-overlay", default="reports/vlm/symbol_crop_verifier_p200_overlay.jsonl")
    args = parser.parse_args()
    rows = load_jsonl(Path(args.base_overlay))
    vmap = verifier_map(Path(args.verifier_predictions))
    golds = {str(row.get("row_id") or row.get("id")): target_symbols(row) for row in rows}
    baseline_preds = {str(row.get("row_id") or row.get("id")): p165.normalized(row.get("symbol_candidates") or [], "baseline") for row in rows}
    baseline = p165.evaluate(golds, baseline_preds)
    policies = []
    gate_sets = [[], ["generic_symbol", "appliance", "equipment"], ["generic_symbol", "appliance", "equipment", "sink", "shower"]]
    protected_sets = [[], ["sink", "shower"], ["sink", "shower", "stair"]]
    for threshold in [0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]:
        for gate_labels in gate_sets:
            for protected in protected_sets:
                policies.append({
                    "name": f"p200_fp{threshold}_gate{len(gate_labels)}_protect{len(protected)}",
                    "fp_threshold": threshold,
                    "gate_labels": gate_labels,
                    "protected_labels": protected,
                    "relabel_threshold": 1.1,
                })
    scored = []
    for policy in policies:
        preds = apply_policy(rows, vmap, policy)
        metrics = p165.evaluate(golds, preds)
        scored.append({"policy": policy, "metrics": metrics})
    scored.sort(key=lambda item: (item["metrics"]["f1"], item["metrics"]["precision"], item["metrics"]["recall"], -item["metrics"]["prediction_inflation"]), reverse=True)
    best = scored[0]
    best_preds = apply_policy(rows, vmap, best["policy"])
    write_jsonl(Path(args.out_overlay), materialize(rows, best_preds, best["policy"]))
    report = {
        "id": "P200_symbol_crop_verifier_fusion",
        "decision": "promote_candidate" if best["metrics"]["f1"] > baseline["f1"] else "no_promotion_keep_baseline",
        "claim_boundary": "Verifier trained/evaluated from candidate crops with gold-derived labels; fusion is P101-selected and requires held-out validation before paper claim.",
        "inputs": {"base_overlay": args.base_overlay, "verifier_predictions": args.verifier_predictions},
        "metrics": {"baseline": baseline, "p200_best": best["metrics"]},
        "delta_vs_baseline": p165.delta(best["metrics"], baseline),
        "best_policy": best["policy"],
        "top_candidates": scored[:40],
        "outputs": {"json": args.out_json, "md": args.out_md, "overlay": args.out_overlay},
    }
    write_json(Path(args.out_json), report)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(render(report))
    print(json.dumps({"decision": report["decision"], "metrics": report["metrics"], "delta": report["delta_vs_baseline"], "best_policy": report["best_policy"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
