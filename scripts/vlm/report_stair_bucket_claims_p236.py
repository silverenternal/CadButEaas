#!/usr/bin/env python3
"""Generate paper-ready bucket-aware stair claim tables for P236."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT = ROOT / "reports" / "vlm" / "p235_stair_representation_audit.json"
DEFAULT_BUCKETS = ROOT / "reports" / "vlm" / "p235_stair_bucket_metrics.json"
DEFAULT_MAIN = ROOT / "reports" / "vlm" / "p232_repaired_contract_eval.json"
DEFAULT_JSON = ROOT / "reports" / "vlm" / "p236_stair_bucket_claim_table.json"
DEFAULT_MD = ROOT / "reports" / "vlm" / "p236_stair_bucket_claim_table.md"
DEFAULT_CLAIM = ROOT / "reports" / "vlm" / "p236_paper_claim_language.md"


DISPLAY = {
    "medium_large_stair_object": "Object-like stair boxes",
    "xlarge_grouped_stair": "Grouped stair structures",
    "sentinel_placeholder": "Sentinel placeholders",
    "ultra_thin_tread": "Ultra-thin tread/line boxes",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def combined_visual_metrics(bucket_metrics: dict[str, dict[str, Any]], buckets: list[str]) -> dict[str, Any]:
    tp = sum(bucket_metrics[item]["tp"] for item in buckets)
    pred = sum(bucket_metrics[item]["pred_or_near"] for item in buckets)
    gold = sum(bucket_metrics[item]["gold"] for item in buckets)
    fp = max(pred - tp, 0)
    fn = max(gold - tp, 0)
    precision = tp / max(pred, 1)
    recall = tp / max(gold, 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "tp": int(tp),
        "pred_or_near": int(pred),
        "gold": int(gold),
        "fp_est": int(fp),
        "fn": int(fn),
        "precision_or_near_precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1_or_near_f1": round(f1, 6),
    }


def table_rows(bucket_metrics: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    order = ["medium_large_stair_object", "xlarge_grouped_stair", "sentinel_placeholder", "ultra_thin_tread"]
    rows = []
    for bucket in order:
        metric = bucket_metrics[bucket]
        rows.append({
            "bucket": bucket,
            "display_name": DISPLAY[bucket],
            "gold": metric["gold"],
            "tp": metric["tp"],
            "pred_or_near": metric["pred_or_near"],
            "recall": metric["recall"],
            "precision_or_near_precision": metric["precision_or_near_precision"],
            "f1_or_near_f1": metric["f1_or_near_f1"],
            "metric_role": "main_visual_detector_bucket" if bucket in {"medium_large_stair_object", "xlarge_grouped_stair"} else "auxiliary_or_non_visual_representation_bucket",
        })
    return rows


def render_table(report: dict[str, Any]) -> str:
    lines = [
        "# P236 Bucket-Aware Stair Claim Table",
        "",
        "## Main Symbol Baseline",
        f"- Promoted baseline: `P232`",
        f"- Overall symbol Precision/Recall/F1: `{report['main_symbol_metrics']['precision']:.6f}` / `{report['main_symbol_metrics']['recall']:.6f}` / `{report['main_symbol_metrics']['f1']:.6f}`",
        f"- Aggregate locked stair F1: `{report['aggregate_stair_metric']['f1']:.6f}`",
        "",
        "## Stair Buckets Under P232",
        "| Bucket | Role | Gold | TP | Pred/Near | Precision/Near-P | Recall | F1/Near-F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["p232_bucket_rows"]:
        lines.append(
            f"| {row['display_name']} | {row['metric_role']} | {row['gold']} | {row['tp']} | {row['pred_or_near']} | {row['precision_or_near_precision']:.6f} | {row['recall']:.6f} | {row['f1_or_near_f1']:.6f} |"
        )
    visual = report["p232_visual_stair_combined"]
    lines.extend([
        "",
        "## Visual Stair Combined",
        f"- Buckets: `medium_large_stair_object + xlarge_grouped_stair`",
        f"- Gold/TP/Pred-near: `{visual['gold']}` / `{visual['tp']}` / `{visual['pred_or_near']}`",
        f"- Precision/Recall/F1: `{visual['precision_or_near_precision']:.6f}` / `{visual['recall']:.6f}` / `{visual['f1_or_near_f1']:.6f}`",
        "",
        "## Interpretation",
        "- The aggregate stair metric mixes visual object/group stair targets with sentinel placeholders and ultra-thin tread-line boxes.",
        "- P232 performs substantially better on visual stair buckets than the aggregate stair F1 suggests.",
        "- Placeholder and tread-line buckets should be reported as auxiliary/non-visual representation failures, not as ordinary object-detector failures.",
        "",
    ])
    return "\n".join(lines)


def render_claim_language(report: dict[str, Any]) -> str:
    visual = report["p232_visual_stair_combined"]
    aggregate = report["aggregate_stair_metric"]
    return "\n".join([
        "# P236 Paper Claim Language",
        "",
        "## Recommended Wording",
        "The proposed raster-to-contract CadStruct pipeline reports the locked benchmark metrics for comparability, but stair symbols require bucket-aware interpretation because the public target stream mixes visually distinct representations. In the locked stair subset, object-like and grouped stair targets are separable from sentinel placeholder boxes and ultra-thin tread-line boxes. Under the promoted P232 symbol contract baseline, the visual stair buckets achieve a combined near-F1 of "
        f"`{visual['f1_or_near_f1']:.6f}` with precision/recall `{visual['precision_or_near_precision']:.6f}`/`{visual['recall']:.6f}`, while the aggregate locked stair F1 is `{aggregate['f1']:.6f}`. We therefore report aggregate stair results for benchmark comparability and additionally disclose representation-aware bucket metrics to avoid conflating object detection errors with non-visual placeholder or line-structure targets.",
        "",
        "## Reviewer-Safe Claim Boundary",
        "- It is acceptable to claim that object-like/grouped stair recognition is materially stronger than the aggregate stair F1 indicates.",
        "- It is not acceptable to claim that all stair representations are solved.",
        "- Sentinel placeholders should be treated as annotation/metadata artifacts, not visual object targets.",
        "- Ultra-thin tread boxes should be routed to line/structure extraction or auxiliary evaluation, not ordinary bbox detection.",
        "",
        "## Table Reference",
        "Use `reports/vlm/p236_stair_bucket_claim_table.md` for the paper-ready table and `reports/vlm/p236_stair_bucket_claim_table.json` for exact values.",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--buckets", type=Path, default=DEFAULT_BUCKETS)
    parser.add_argument("--main", type=Path, default=DEFAULT_MAIN)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD)
    parser.add_argument("--claim-out", type=Path, default=DEFAULT_CLAIM)
    args = parser.parse_args()

    audit = read_json(args.audit)
    buckets = read_json(args.buckets)
    main = read_json(args.main)
    p232_bucket = buckets["p232_bucket_metrics"]
    p234_bucket = buckets["p234_bucket_metrics"]
    visual_buckets = ["medium_large_stair_object", "xlarge_grouped_stair"]
    aggregate_stair = main["per_label_metrics_iou_0_30"]["stair"]
    report = {
        "id": "p236_stair_bucket_claim_table",
        "phase": "P236_bucket_aware_stair_eval_and_paper_claim_wrapper",
        "sources": {"audit": str(args.audit), "bucket_metrics": str(args.buckets), "main_eval": str(args.main)},
        "main_symbol_metrics": main["candidate_metrics_iou_0_30"],
        "aggregate_stair_metric": aggregate_stair,
        "taxonomy_counts": audit["taxonomy_counts"],
        "p232_bucket_rows": table_rows(p232_bucket),
        "p234_bucket_rows": table_rows(p234_bucket),
        "p232_visual_stair_combined": combined_visual_metrics(p232_bucket, visual_buckets),
        "p234_visual_stair_combined": combined_visual_metrics(p234_bucket, visual_buckets),
        "recommended_metric_roles": {
            "main_visual_detector_stair": visual_buckets,
            "auxiliary_structure_or_metadata": ["sentinel_placeholder", "ultra_thin_tread"],
        },
        "claim_boundary": "Offline paper-reporting wrapper. It does not replace the locked aggregate benchmark metric; it discloses representation buckets for interpretation.",
    }
    write_json(args.json_out, report)
    write_text(args.md_out, render_table(report))
    write_text(args.claim_out, render_claim_language(report))
    print(json.dumps({"json": str(args.json_out), "md": str(args.md_out), "claim": str(args.claim_out), "visual_stair": report["p232_visual_stair_combined"], "aggregate_stair": aggregate_stair}, ensure_ascii=False))


if __name__ == "__main__":
    main()
