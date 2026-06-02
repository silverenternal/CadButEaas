#!/usr/bin/env python3
"""Audit existing symbol/raster artifacts for reviewer-driven metric rescue.

This is an offline inventory: it reads prior eval/report JSON files and ranks
which artifacts are worth retrying or citing. It does not create runtime
predictions.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports" / "vlm"
OUT_JSON = REPORT_DIR / "p240_symbol_artifact_rescue_matrix.json"
OUT_MD = REPORT_DIR / "p240_symbol_artifact_rescue_matrix.md"
OUT_DECISION = REPORT_DIR / "p240_next_rescue_decision.md"
BASELINE = {"precision": 0.688326, "recall": 0.768740, "f1": 0.726314, "source": "reports/vlm/p232_repaired_contract_eval.json"}


def read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def find_metric_dict(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        if all(key in value for key in ["precision", "recall", "f1"]):
            return value
        preferred = [
            "candidate_metrics_iou_0_30",
            "overall_metrics_iou_0_30",
            "candidate_metrics",
            "metrics",
            "symbol_bbox_iou_0_30",
        ]
        for key in preferred:
            found = find_metric_dict(value.get(key))
            if found:
                return found
        for child in value.values():
            found = find_metric_dict(child)
            if found:
                return found
    if isinstance(value, list):
        for child in value:
            found = find_metric_dict(child)
            if found:
                return found
    return None


def per_label(value: dict[str, Any]) -> dict[str, Any]:
    for key in ["per_label_metrics_iou_0_30", "per_label_metrics", "type_metrics"]:
        if isinstance(value.get(key), dict):
            return value[key]
    return {}


def bootstrap_summary(value: dict[str, Any]) -> dict[str, Any] | None:
    for key in ["bootstrap_vs_p229b", "bootstrap_vs_p232", "bootstrap"]:
        if isinstance(value.get(key), dict):
            return value[key]
    return None


def source_integrity_for(eval_path: Path) -> str:
    stem = eval_path.name.replace("_eval.json", "")
    candidates = [
        eval_path.with_name(stem + "_source_integrity.json"),
        eval_path.with_name(eval_path.name.replace("_eval.json", "_source_integrity.json")),
    ]
    for path in candidates:
        data = read_json(path)
        if isinstance(data, dict) and data.get("pass_integrity") is True:
            return "pass"
        if isinstance(data, dict) and data.get("pass_integrity") is False:
            return "fail"
    return "unknown"


def decision_for(item: dict[str, Any]) -> str:
    if item["source_integrity"] == "fail":
        return "reject_source_integrity"
    if item["f1"] is None:
        return "inspect_manually"
    if item["f1_delta_vs_p232"] >= 0.01 and item["precision_delta_vs_p232"] >= -0.002:
        return "high_priority_retry_or_promote_candidate"
    if item["f1_delta_vs_p232"] > 0 and item["precision_delta_vs_p232"] >= 0:
        return "small_positive_candidate"
    if item["recall_delta_vs_p232"] > 0.01 and item["precision_delta_vs_p232"] < 0:
        return "retry_with_precision_gate"
    if item["f1_delta_vs_p232"] < 0:
        return "freeze_negative"
    return "appendix_or_context"


def main() -> None:
    rows = []
    for path in sorted(REPORT_DIR.glob("**/*eval.json")):
        if "symbol" not in path.name and "p22" not in path.name and "p23" not in path.name:
            continue
        data = read_json(path)
        if not isinstance(data, dict):
            continue
        metric = find_metric_dict(data)
        if not metric:
            continue
        precision = metric.get("precision")
        recall = metric.get("recall")
        f1 = metric.get("f1")
        if precision is None or recall is None or f1 is None:
            continue
        item = {
            "path": str(path.relative_to(ROOT)),
            "id": data.get("id") or data.get("version") or path.stem,
            "precision": round(float(precision), 6),
            "recall": round(float(recall), 6),
            "f1": round(float(f1), 6),
            "precision_delta_vs_p232": round(float(precision) - BASELINE["precision"], 6),
            "recall_delta_vs_p232": round(float(recall) - BASELINE["recall"], 6),
            "f1_delta_vs_p232": round(float(f1) - BASELINE["f1"], 6),
            "promotion_recommendation": data.get("promotion_recommendation") or data.get("promotion") or data.get("decision") or "unknown",
            "source_integrity": source_integrity_for(path),
            "bootstrap": bootstrap_summary(data),
            "weak_labels": {},
        }
        labels = per_label(data)
        if labels:
            weak = {}
            for label, values in labels.items():
                if isinstance(values, dict) and values.get("f1") is not None and float(values["f1"]) < 0.75:
                    weak[label] = {k: values.get(k) for k in ["precision", "recall", "f1", "gold", "tp", "predicted"] if k in values}
            item["weak_labels"] = weak
        item["rescue_decision"] = decision_for(item)
        rows.append(item)
    rows = sorted(rows, key=lambda item: (item["f1"], item["precision"]), reverse=True)
    matrix = {
        "id": "p240_symbol_artifact_rescue_matrix",
        "baseline": BASELINE,
        "artifact_count": len(rows),
        "top_by_f1": rows[:25],
        "retry_with_precision_gate": [item for item in rows if item["rescue_decision"] == "retry_with_precision_gate"][:25],
        "small_positive_candidates": [item for item in rows if item["rescue_decision"] == "small_positive_candidate"][:25],
        "negative_or_rejected": [item for item in rows if item["rescue_decision"] in {"freeze_negative", "reject_source_integrity"}][:25],
        "all_rows": rows,
        "claim_boundary": "Offline artifact inventory. Promotion requires rerun/source-integrity/locked bootstrap under current P232 baseline.",
    }
    OUT_JSON.write_text(json.dumps(matrix, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = ["# P240 Symbol Artifact Rescue Matrix", "", f"Baseline P232: P `{BASELINE['precision']:.6f}` / R `{BASELINE['recall']:.6f}` / F1 `{BASELINE['f1']:.6f}`", "", "## Top Artifacts by F1", "| Artifact | P | R | F1 | ΔF1 | Integrity | Decision |", "|---|---:|---:|---:|---:|---|---|"]
    for item in rows[:20]:
        md.append(f"| `{item['path']}` | {item['precision']:.6f} | {item['recall']:.6f} | {item['f1']:.6f} | {item['f1_delta_vs_p232']:+.6f} | {item['source_integrity']} | {item['rescue_decision']} |")
    md += ["", "## Reviewer-Relevant Interpretation", "- Artifacts above P232 must be checked for source integrity and precision loss before reuse.", "- Recall-only gains with precision regression should become gated retry candidates, not paper claims.", "- If no material runtime-safe artifact beats P232, paper framing must move away from raster recognition strength.", ""]
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    retry = matrix["retry_with_precision_gate"][:5]
    best = rows[0] if rows else None
    decision = ["# P240 Next Rescue Decision", ""]
    if best:
        decision += [f"Best raw F1 artifact: `{best['path']}` with F1 `{best['f1']:.6f}` (ΔF1 `{best['f1_delta_vs_p232']:+.6f}`), source-integrity `{best['source_integrity']}`, decision `{best['rescue_decision']}`.", ""]
    if retry:
        decision += ["## Recommended Next Action", "Run a precision-gated retry using the best recall-positive artifacts below:", ""]
        for item in retry:
            decision.append(f"- `{item['path']}`: P `{item['precision']:.6f}`, R `{item['recall']:.6f}`, F1 `{item['f1']:.6f}`, ΔF1 `{item['f1_delta_vs_p232']:+.6f}`")
    else:
        decision += ["## Recommended Next Action", "No obvious recall-positive precision-gate candidate was found. Shift to MoE ablation/utility evidence and manuscript reframing unless manual inspection identifies a missed artifact."]
    decision.append("")
    OUT_DECISION.write_text("\n".join(decision), encoding="utf-8")
    print(json.dumps({"out": str(OUT_JSON), "artifacts": len(rows), "best": rows[0] if rows else None, "retry_count": len(matrix['retry_with_precision_gate'])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
