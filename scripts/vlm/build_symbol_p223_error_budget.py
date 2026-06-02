#!/usr/bin/env python3
"""P223 reviewer-grade symbol error budget and oracle ceilings."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from fuse_symbol_p206g_with_p211_p212 import area_bucket, bbox_iou, load_p206g, write_json

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports/vlm/symbol_p222_p221a_frozen_overlay.jsonl"
PROPOSALS = [
    ROOT / "reports/vlm/symbol_p212_p213b_residual_fusion_overlay.jsonl",
    ROOT / "reports/vlm/symbol_p221c_candidate_gate_dataset.jsonl",
    ROOT / "reports/vlm/symbol_p221b_stair_specialist_page_predictions.jsonl",
]
OUT_JSON = ROOT / "reports/vlm/symbol_p223_error_budget.json"
OUT_MD = ROOT / "reports/vlm/symbol_p223_error_budget.md"
LABELS = ["appliance", "bathtub", "column", "equipment", "generic_symbol", "shower", "sink", "stair", "table"]


def rel_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def pred_label(pred: dict[str, Any]) -> str:
    return str(pred.get("label") or pred.get("symbol_type") or "generic_symbol")


def pred_score(pred: dict[str, Any]) -> float:
    return float(pred.get("score") or pred.get("confidence") or 1.0)


def metrics_from_counts(tp: int, pred: int, gold: int) -> dict[str, Any]:
    precision = tp / max(pred, 1)
    recall = tp / max(gold, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"tp": tp, "predicted": pred, "gold": gold, "fp": pred - tp, "fn": gold - tp, "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6)}


def greedy_match(preds: list[dict[str, Any]], golds: list[dict[str, Any]], typed: bool = True, iou_threshold: float = 0.30) -> tuple[set[int], set[int], list[tuple[int, int, float]]]:
    pairs = []
    for pi, pred in enumerate(preds):
        pbox = [float(v) for v in pred["bbox"]]
        plabel = pred_label(pred)
        for gi, gold in enumerate(golds):
            if typed and plabel != str(gold["label"]):
                continue
            iou = bbox_iou(pbox, [float(v) for v in gold["bbox"]])
            if iou >= iou_threshold:
                pairs.append((iou, pi, gi))
    used_p, used_g, matched = set(), set(), []
    for iou, pi, gi in sorted(pairs, reverse=True):
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi); used_g.add(gi); matched.append((pi, gi, iou))
    return used_p, used_g, matched


def load_overlay_predictions(path: Path) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for row in read_jsonl(path):
        rid = str(row.get("id") or row.get("row_id"))
        preds = []
        for cand in row.get("symbol_candidates") or row.get("predicted_symbols") or []:
            if "bbox" not in cand:
                continue
            preds.append({"bbox": [float(v) for v in cand["bbox"]], "label": pred_label(cand), "score": pred_score(cand), "source": cand.get("source") or path.stem})
        out[rid] = preds
    return out


def load_gate_dataset(path: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(path):
        cand = row.get("candidate") or {}
        if "bbox" not in cand:
            continue
        out[str(row["row_id"])].append({"bbox": [float(v) for v in cand["bbox"]], "label": str(row.get("label") or pred_label(cand)), "score": float(row.get("score") or cand.get("score") or 0.0), "source": path.stem})
    return out


def merge_proposals(paths: list[Path], row_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    merged = {rid: [] for rid in row_ids}
    seen = {rid: set() for rid in row_ids}
    for path in paths:
        if not path.exists():
            continue
        by_row = load_gate_dataset(path) if path.name.endswith("candidate_gate_dataset.jsonl") else load_overlay_predictions(path)
        for rid in row_ids:
            for pred in by_row.get(rid, []):
                key = (pred_label(pred), tuple(round(float(v), 2) for v in pred["bbox"]))
                if key in seen[rid]:
                    continue
                seen[rid].add(key)
                merged[rid].append(pred)
    return merged


def evaluate_by_buckets(preds_by_row: dict[str, list[dict[str, Any]]], golds_by_row: dict[str, dict[str, dict[str, Any]]], ids: list[str]) -> dict[str, Any]:
    total = Counter()
    by_label = {label: Counter() for label in LABELS}
    by_size = defaultdict(Counter)
    confusion = defaultdict(Counter)
    hard_rows = []
    fp_by_label = Counter()
    fn_by_label = Counter()
    for rid in ids:
        preds = preds_by_row.get(rid, [])
        golds = list(golds_by_row[rid].values())
        used_p, used_g, matched = greedy_match(preds, golds, typed=True)
        total.update({"tp": len(used_g), "pred": len(preds), "gold": len(golds), "fp": len(preds) - len(used_p), "fn": len(golds) - len(used_g)})
        for pi, gi, _ in matched:
            label = str(golds[gi]["label"])
            bucket = area_bucket([float(v) for v in golds[gi]["bbox"]])
            by_label[label].update({"tp": 1, "pred": 1, "gold": 1})
            by_size[bucket].update({"tp": 1, "pred": 1, "gold": 1})
            confusion[label][pred_label(preds[pi])] += 1
        for pi, pred in enumerate(preds):
            if pi not in used_p:
                label = pred_label(pred)
                fp_by_label[label] += 1
                by_label.setdefault(label, Counter()).update({"pred": 1, "fp": 1})
                # FP size bucket by prediction area for debugging.
                by_size[area_bucket([float(v) for v in pred["bbox"]])].update({"pred": 1, "fp": 1})
        for gi, gold in enumerate(golds):
            if gi not in used_g:
                label = str(gold["label"])
                bucket = area_bucket([float(v) for v in gold["bbox"]])
                fn_by_label[label] += 1
                by_label.setdefault(label, Counter()).update({"gold": 1, "fn": 1})
                by_size[bucket].update({"gold": 1, "fn": 1})
        row_f1 = metrics_from_counts(len(used_g), len(preds), len(golds))["f1"]
        hard_rows.append({"row_id": rid, "f1": row_f1, "tp": len(used_g), "pred": len(preds), "gold": len(golds), "fp": len(preds) - len(used_p), "fn": len(golds) - len(used_g)})
    def pack(counter: Counter) -> dict[str, Any]:
        return metrics_from_counts(counter["tp"], counter["pred"], counter["gold"])
    return {
        "overall": pack(total),
        "by_label": sorted([{"label": k, **pack(v)} for k, v in by_label.items() if v["gold"] or v["pred"]], key=lambda x: (-x["fn"], x["label"])),
        "by_size": sorted([{"bucket": k, **pack(v)} for k, v in by_size.items()], key=lambda x: x["bucket"]),
        "fp_by_label": dict(fp_by_label.most_common()),
        "fn_by_label": dict(fn_by_label.most_common()),
        "confusion": {k: dict(v.most_common()) for k, v in confusion.items()},
        "hard_rows": sorted(hard_rows, key=lambda x: (x["f1"], -x["fn"], -x["fp"]))[:20],
    }


def oracle_ceiling(base: dict[str, list[dict[str, Any]]], proposals: dict[str, list[dict[str, Any]]], golds_by_row: dict[str, dict[str, dict[str, Any]]], ids: list[str]) -> dict[str, Any]:
    totals = {"base_typed": Counter(), "box_only_relabel": Counter(), "candidate_add_typed": Counter(), "candidate_add_relabel": Counter(), "perfect_label_existing_boxes": Counter()}
    coverage_by_label = defaultdict(Counter)
    coverage_by_size = defaultdict(Counter)
    for rid in ids:
        golds = list(golds_by_row[rid].values())
        base_preds = base.get(rid, [])
        union = base_preds + proposals.get(rid, [])
        variants = {
            "base_typed": (base_preds, True),
            "box_only_relabel": (base_preds, False),
            "candidate_add_typed": (union, True),
            "candidate_add_relabel": (union, False),
            "perfect_label_existing_boxes": (base_preds, False),
        }
        for name, (preds, typed) in variants.items():
            used_p, used_g, _ = greedy_match(preds, golds, typed=typed)
            totals[name].update({"tp": len(used_g), "pred": len(preds) if name not in {"box_only_relabel", "perfect_label_existing_boxes"} else len(used_g), "gold": len(golds)})
        # proposal recall coverage by gold: any union box with typed/agnostic coverage.
        for gold in golds:
            gbox = [float(v) for v in gold["bbox"]]
            label = str(gold["label"])
            bucket = area_bucket(gbox)
            any_cover = any(bbox_iou([float(v) for v in pred["bbox"]], gbox) >= 0.30 for pred in union)
            typed_cover = any(pred_label(pred) == label and bbox_iou([float(v) for v in pred["bbox"]], gbox) >= 0.30 for pred in union)
            coverage_by_label[label].update({"gold": 1, "any_cover": int(any_cover), "typed_cover": int(typed_cover)})
            coverage_by_size[bucket].update({"gold": 1, "any_cover": int(any_cover), "typed_cover": int(typed_cover)})
    out = {name: metrics_from_counts(c["tp"], c["pred"], c["gold"]) for name, c in totals.items()}
    out["coverage_by_label"] = sorted([{"label": k, "gold": v["gold"], "any_cover_recall": round(v["any_cover"] / max(v["gold"], 1), 6), "typed_cover_recall": round(v["typed_cover"] / max(v["gold"], 1), 6)} for k, v in coverage_by_label.items()], key=lambda x: x["typed_cover_recall"])
    out["coverage_by_size"] = sorted([{"bucket": k, "gold": v["gold"], "any_cover_recall": round(v["any_cover"] / max(v["gold"], 1), 6), "typed_cover_recall": round(v["typed_cover"] / max(v["gold"], 1), 6)} for k, v in coverage_by_size.items()], key=lambda x: x["bucket"])
    return out


def render(report: dict[str, Any]) -> str:
    m = report["p222_metrics"]["overall"]
    lines = [
        "# P223 Symbol Error Budget",
        "",
        "## Reviewer Baseline",
        f"- P222 F1/P/R: `{m['f1']:.6f}` / `{m['precision']:.6f}` / `{m['recall']:.6f}`",
        f"- Counts TP/Pred/Gold/FP/FN: `{m['tp']}` / `{m['predicted']}` / `{m['gold']}` / `{m['fp']}` / `{m['fn']}`",
        "",
        "## Oracle Ceilings",
        "| Variant | F1 | P | R | TP | Pred | Gold |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ["base_typed", "box_only_relabel", "candidate_add_typed", "candidate_add_relabel", "perfect_label_existing_boxes"]:
        x = report["oracle_ceiling"][name]
        lines.append(f"| {name} | {x['f1']:.6f} | {x['precision']:.6f} | {x['recall']:.6f} | {x['tp']} | {x['predicted']} | {x['gold']} |")
    lines += ["", "## Worst Labels", "| Label | F1 | P | R | TP | FP | FN |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["p222_metrics"]["by_label"][:12]:
        lines.append(f"| {row['label']} | {row['f1']:.6f} | {row['precision']:.6f} | {row['recall']:.6f} | {row['tp']} | {row['fp']} | {row['fn']} |")
    lines += ["", "## Size Buckets", "| Bucket | F1 | P | R | TP | FP | FN |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["p222_metrics"]["by_size"]:
        lines.append(f"| {row['bucket']} | {row['f1']:.6f} | {row['precision']:.6f} | {row['recall']:.6f} | {row['tp']} | {row['fp']} | {row['fn']} |")
    lines += ["", "## Interpretation", report["interpretation"], ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=str(BASE))
    parser.add_argument("--out-json", default=str(OUT_JSON))
    parser.add_argument("--out-md", default=str(OUT_MD))
    parser.add_argument("--proposal", action="append", default=[], help="Additional proposal jsonl paths to include in oracle union")
    args = parser.parse_args()
    rows, base, golds = load_p206g(Path(args.base))
    ids = [str(row.get("id") or row.get("row_id")) for row in rows]
    proposal_paths = PROPOSALS + [Path(item) if Path(item).is_absolute() else ROOT / item for item in args.proposal]
    proposals = merge_proposals(proposal_paths, ids)
    p222_metrics = evaluate_by_buckets(base, golds, ids)
    oracle = oracle_ceiling(base, proposals, golds, ids)
    base_f1 = p222_metrics["overall"]["f1"]
    candidate_add_relabel_f1 = oracle["candidate_add_relabel"]["f1"]
    if oracle["candidate_add_relabel"]["recall"] >= 0.90 and candidate_add_relabel_f1 > base_f1 + 0.10:
        recommendation = "High union-box oracle: prioritize candidate classifier/relabeler and NMS/precision gate; boxes likely exist but labels/selection are weak."
    elif oracle["candidate_add_typed"]["recall"] < 0.85:
        recommendation = "Typed proposal recall is low: prioritize stronger multi-class detector/proposal generation before more gating."
    elif oracle["box_only_relabel"]["recall"] > p222_metrics["overall"]["recall"] + 0.08:
        recommendation = "Existing boxes have label/oracle headroom: train crop classifier/relabeler."
    else:
        recommendation = "Mixed bottleneck: train stronger detector with classifier/refiner heads; threshold-only tuning is insufficient."
    report = {
        "id": "P223_symbol_error_budget",
        "base_overlay": rel_path(Path(args.base)),
        "proposal_sources": [rel_path(p) for p in proposal_paths if p.exists()],
        "p222_metrics": p222_metrics,
        "oracle_ceiling": oracle,
        "recommendation": recommendation,
        "interpretation": recommendation + " Reviewer target F1>=0.85 requires a material architecture/data branch, not another +0.002 policy tweak.",
        "claim_boundary": "Gold labels are used only for offline error-budget/oracle analysis; no runtime features are introduced.",
    }
    write_json(Path(args.out_json), report)
    Path(args.out_md).write_text(render(report), encoding="utf-8")
    print(json.dumps({"baseline": p222_metrics["overall"], "oracle": {k: oracle[k] for k in ["box_only_relabel", "candidate_add_typed", "candidate_add_relabel"]}, "recommendation": recommendation, "outputs": [rel_path(Path(args.out_json)), rel_path(Path(args.out_md))]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
