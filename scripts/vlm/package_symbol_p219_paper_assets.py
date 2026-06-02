#!/usr/bin/env python3
"""Package paper-safe P219 assets for the P217/P218 symbol result."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from fuse_symbol_p206g_with_p211_p212 import area_bucket, bbox_iou, load_p206g

ROOT = Path(__file__).resolve().parents[2]
OUT_ABL = ROOT / "reports/vlm/symbol_p219_paper_bounded_ablation.md"
OUT_RES = ROOT / "reports/vlm/symbol_p219_residual_error_table.md"
OUT_JSON = ROOT / "reports/vlm/symbol_p219_residual_error_table.json"
P217 = ROOT / "reports/vlm/symbol_p218_p217_frozen_overlay.jsonl"

METRICS = [
    ("P206g precision repair", "reports/vlm/symbol_p206f_precision_repair_p206g.md", 0.591737, 0.668004, 0.531100, "P101 selected/bootstrap-bounded earlier scorer"),
    ("P212 FN specialist precision repair", "reports/vlm/symbol_p212_precision_repair_summary.md", 0.682464, 0.676056, 0.688995, "P101/bootstrap-bounded; specialist page fusion"),
    ("P215 narrow gate", "reports/vlm/symbol_p215_narrow_gate_summary.md", 0.698910, 0.671318, 0.728868, "P101/bootstrap-bounded; precision CI slightly negative"),
    ("P216 oracle row-label subset", "reports/vlm/symbol_p216_oracle_rowlabel_subset_summary.md", 0.700523, 0.681630, 0.720494, "Oracle proof only; not runtime deployable"),
    ("P217/P218 runtime-safe verifier", "reports/vlm/symbol_p218_p217_frozen_validation.md", 0.704383, 0.682890, 0.727273, "Runtime-safe verifier; P101/bootstrap-bounded, frozen in P218"),
]


def matched_indices(preds, golds):
    matched = set()
    used = set()
    for gold_index, gold in enumerate(golds):
        gold_box = [float(v) for v in gold["bbox"]]
        best_iou = 0.0
        best_pred = None
        for pred_index, pred in enumerate(preds):
            if pred_index in used:
                continue
            if str(pred.get("label", "unknown")) != str(gold.get("label", "unknown")):
                continue
            iou = bbox_iou([float(v) for v in pred["bbox"]], gold_box)
            if iou > best_iou:
                best_iou = iou
                best_pred = pred_index
        if best_pred is not None and best_iou >= 0.30:
            used.add(best_pred)
            matched.add(gold_index)
    return matched


def main() -> None:
    OUT_ABL.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Symbol MoE Bounded Ablation (P219)",
        "",
        "| Stage | F1 | Precision | Recall | Evidence boundary | Artifact |",
        "|---|---:|---:|---:|---|---|",
    ]
    for name, artifact, f1, precision, recall, boundary in METRICS:
        lines.append(f"| {name} | {f1:.6f} | {precision:.6f} | {recall:.6f} | {boundary} | `{artifact}` |")
    lines += [
        "",
        "## Paper-Safe Wording",
        "- The P217/P218 result is a frozen, runtime-safe verifier result on the P101 overlay evaluation rows.",
        "- It does not use row IDs, gold labels, annotation paths, expected_json, SVG geometry, or parser geometry at runtime.",
        "- The result should be described as P101/bootstrap-bounded unless an independent held-out validation is added.",
        "- Do not claim >0.90 symbol F1 or broad cross-dataset generalization from this table.",
    ]
    OUT_ABL.write_text("\n".join(lines) + "\n", encoding="utf-8")

    rows, preds, golds = load_p206g(P217)
    by_label = Counter()
    by_bucket = Counter()
    by_row = Counter()
    examples = []
    for row in rows:
        row_id = str(row.get("id") or row.get("row_id"))
        gold_list = list(golds[row_id].values())
        matched = matched_indices(preds[row_id], gold_list)
        for index, gold in enumerate(gold_list):
            if index in matched:
                continue
            label = str(gold["label"])
            bucket = area_bucket([float(v) for v in gold["bbox"]])
            by_label[label] += 1
            by_bucket[bucket] += 1
            by_row[row_id] += 1
            if len(examples) < 80:
                examples.append({"row_id": row_id, "target_id": gold.get("target_id"), "label": label, "bucket": bucket, "bbox": gold["bbox"]})
    residual = {
        "id": "P219_P217_residual_errors",
        "total_fn": sum(by_label.values()),
        "by_label": dict(by_label),
        "by_bucket": dict(by_bucket),
        "worst_rows": dict(by_row.most_common(20)),
        "examples": examples,
        "claim_boundary": "Residual FN audit after frozen P217/P218 overlay; gold used for evaluation only.",
    }
    OUT_JSON.write_text(json.dumps(residual, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    res_lines = [
        "# P217/P218 Residual Error Table",
        "",
        f"- Total FN: {residual['total_fn']}",
        f"- By label: `{json.dumps(dict(by_label), ensure_ascii=False)}`",
        f"- By bucket: `{json.dumps(dict(by_bucket), ensure_ascii=False)}`",
        f"- Worst rows: `{json.dumps(dict(by_row.most_common(10)), ensure_ascii=False)}`",
        "",
        "## Claim Boundary",
        residual["claim_boundary"],
    ]
    OUT_RES.write_text("\n".join(res_lines) + "\n", encoding="utf-8")
    print(json.dumps({"ablation": str(OUT_ABL), "residual": str(OUT_RES), "total_fn": residual["total_fn"], "by_label": dict(by_label)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
