#!/usr/bin/env python3
"""P192 lightweight symbol box calibration sweep over P188 overlay."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import sweep_symbol_disagreement_backfill_p165 as p165

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports/vlm/symbol_pro5090_yolov8m_seg_rect_p185_p188_recall_expanded_overlay.jsonl"
OUT_JSON = ROOT / "configs/vlm/symbol_box_refiner_p192.json"
OUT_MD = ROOT / "reports/vlm/symbol_box_refiner_p192_eval.md"
OUT_OVERLAY = ROOT / "reports/vlm/symbol_box_refiner_p192_overlay.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def clamp(box: list[float], image_size: Any) -> list[float]:
    width, height = 4096.0, 4096.0
    if isinstance(image_size, list) and len(image_size) >= 2:
        width, height = float(image_size[0]), float(image_size[1])
    elif isinstance(image_size, dict):
        width = float(image_size.get("width") or image_size.get("w") or width)
        height = float(image_size.get("height") or image_size.get("h") or height)
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(width - 1.0, x1))
    y1 = max(0.0, min(height - 1.0, y1))
    x2 = max(x1 + 1.0, min(width, x2))
    y2 = max(y1 + 1.0, min(height, y2))
    return [x1, y1, x2, y2]


def transform(box: list[float], policy: dict[str, Any], image_size: Any) -> list[float]:
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    w = max(1.0, box[2] - box[0]) * float(policy.get("sx", 1.0)) + float(policy.get("px", 0.0)) * 2.0
    h = max(1.0, box[3] - box[1]) * float(policy.get("sy", 1.0)) + float(policy.get("py", 0.0)) * 2.0
    dx = float(policy.get("dx", 0.0)) * max(1.0, box[2] - box[0])
    dy = float(policy.get("dy", 0.0)) * max(1.0, box[3] - box[1])
    return clamp([cx + dx - w / 2.0, cy + dy - h / 2.0, cx + dx + w / 2.0, cy + dy + h / 2.0], image_size)


def applies(pred: dict[str, Any], policy: dict[str, Any]) -> bool:
    labels = set(policy.get("labels") or [])
    buckets = set(policy.get("buckets") or [])
    label_buckets = set(policy.get("label_buckets") or [])
    label = str(pred.get("label"))
    bucket = str(pred.get("bucket"))
    if label_buckets and f"{label}|{bucket}" not in label_buckets:
        return False
    if labels and label not in labels:
        return False
    if buckets and bucket not in buckets:
        return False
    score_min = policy.get("score_min")
    score_max = policy.get("score_max")
    score = float(pred.get("score") or 0.0)
    if score_min is not None and score < float(score_min):
        return False
    if score_max is not None and score > float(score_max):
        return False
    return True


def apply_policy(rows: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    by_row = {}
    for row in rows:
        row_id = str(row.get("row_id") or row.get("id"))
        preds = p165.normalized(row.get("symbol_candidates") or [], "p188")
        refined = []
        for pred in preds:
            item = copy.deepcopy(pred)
            if applies(item, policy):
                item["bbox"] = transform(item["bbox"], policy, row.get("image_size"))
            refined.append(item)
        by_row[row_id] = refined
    return by_row


def candidate_policies() -> list[dict[str, Any]]:
    groups = [
        {"name": "all", "labels": [], "buckets": [], "label_buckets": []},
        {"name": "tiny_small", "labels": [], "buckets": ["tiny", "small"], "label_buckets": []},
        {"name": "boxrefine_top_fn", "labels": [], "buckets": [], "label_buckets": ["sink|tiny", "shower|tiny", "shower|small", "stair|small", "equipment|small", "appliance|small"]},
        {"name": "large_xlarge_equipment", "labels": [], "buckets": [], "label_buckets": ["equipment|large", "equipment|xlarge", "stair|xlarge"]},
        {"name": "sink_shower", "labels": ["sink", "shower"], "buckets": [], "label_buckets": []},
    ]
    scales = [(1.0, 1.0), (1.05, 1.05), (1.10, 1.10), (1.15, 1.15), (1.20, 1.20), (1.10, 1.25), (1.25, 1.10), (0.95, 0.95)]
    pads = [(0, 0), (1, 1), (2, 2), (4, 4), (0, 2), (2, 0), (6, 6)]
    shifts = [(0, 0), (-0.05, 0), (0.05, 0), (0, -0.05), (0, 0.05)]
    policies = [{"name": "p192_noop", "labels": [], "buckets": [], "label_buckets": [], "sx": 1.0, "sy": 1.0, "px": 0, "py": 0, "dx": 0, "dy": 0}]
    for group in groups:
        for sx, sy in scales:
            for px, py in pads:
                for dx, dy in shifts:
                    if (sx, sy, px, py, dx, dy) == (1.0, 1.0, 0, 0, 0, 0):
                        continue
                    policy = dict(group)
                    policy.update({"name": f"p192_{group['name']}_sx{sx}_sy{sy}_px{px}_py{py}_dx{dx}_dy{dy}", "sx": sx, "sy": sy, "px": px, "py": py, "dx": dx, "dy": dy})
                    policies.append(policy)
    return policies


def materialize(rows: list[dict[str, Any]], preds_by_row: dict[str, list[dict[str, Any]]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for raw in rows:
        row = copy.deepcopy(raw)
        row_id = str(row.get("row_id") or row.get("id"))
        candidates = []
        for idx, pred in enumerate(preds_by_row.get(row_id, [])):
            item = copy.deepcopy(pred["raw"])
            item["bbox"] = pred["bbox"]
            item["symbol_type"] = pred["label"]
            item["confidence"] = pred["score"]
            item["id"] = f"{row_id}_p192_symbol_{idx:05d}"
            item["target_id"] = item["id"]
            item["source"] = "symbol_box_refiner_p192"
            item.setdefault("metadata", {})["p192_policy"] = policy["name"]
            candidates.append(item)
        row["symbol_candidates"] = candidates
        if isinstance(row.get("expected_json"), dict):
            row["expected_json"]["symbol_candidates"] = [copy.deepcopy(x) for x in candidates]
        row["symbol_policy_overlay"] = {"policy_id": "p192_best", "description": "P192 lightweight box calibration over P188", "policy": policy}
        out.append(row)
    return out


def render(report: dict[str, Any]) -> str:
    lines = ["# P192 Symbol Box Refiner", "", f"Decision: **{report['decision']}**", "", "| Variant | Precision | Recall | F1 | Center | Inflation |", "|---|---:|---:|---:|---:|---:|"]
    for name, metrics in report["baseline_metrics"].items():
        lines.append(f"| `{name}` | {metrics['precision']:.6f} | {metrics['recall']:.6f} | {metrics['f1']:.6f} | {metrics['center_recall']:.6f} | {metrics['prediction_inflation']:.6f} |")
    b = report["best_metrics"]
    lines.append(f"| `p192_best` | {b['precision']:.6f} | {b['recall']:.6f} | {b['f1']:.6f} | {b['center_recall']:.6f} | {b['prediction_inflation']:.6f} |")
    lines += ["", "## Best Policy", "", "```json", json.dumps(report["best_policy"], ensure_ascii=False, indent=2), "```", "", "## Top Candidates", ""]
    for item in report["top_candidates"][:20]:
        m = item["metrics"]
        lines.append(f"- `{item['policy']['name']}` F1 `{m['f1']:.6f}`, P `{m['precision']:.6f}`, R `{m['recall']:.6f}`, center `{m['center_recall']:.6f}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-overlay", default=str(BASE))
    parser.add_argument("--out-json", default=str(OUT_JSON))
    parser.add_argument("--out-md", default=str(OUT_MD))
    parser.add_argument("--out-overlay", default=str(OUT_OVERLAY))
    args = parser.parse_args()
    rows = load_jsonl(Path(args.base_overlay))
    golds_by_row = {str(row.get("row_id") or row.get("id")): p165.target_symbols(row) for row in rows}
    baseline_preds = {str(row.get("row_id") or row.get("id")): p165.normalized(row.get("symbol_candidates") or [], "p188") for row in rows}
    baseline = p165.evaluate(golds_by_row, baseline_preds)
    scored = []
    for policy in candidate_policies():
        pred_map = apply_policy(rows, policy)
        metrics = p165.evaluate(golds_by_row, pred_map)
        scored.append({"policy": policy, "metrics": metrics})
    scored.sort(key=lambda item: (item["metrics"]["f1"], item["metrics"]["recall"], item["metrics"]["precision"], item["metrics"]["center_recall"]), reverse=True)
    best = scored[0]
    best_preds = apply_policy(rows, best["policy"])
    write_jsonl(Path(args.out_overlay), materialize(rows, best_preds, best["policy"]))
    report = {
        "id": "SCI-P2-192-symbol-box-refiner",
        "decision": "promote_candidate" if best["metrics"]["f1"] > baseline["f1"] else "no_promotion_keep_p188",
        "claim_boundary": "P101 offline box calibration sweep over P188; gold used for analysis only. Requires non-leaky train/dev calibration before final claim if used.",
        "baseline_metrics": {"p188": baseline},
        "searched_policy_count": len(scored),
        "best_policy": best["policy"],
        "best_metrics": best["metrics"],
        "delta_vs_p188": p165.delta(best["metrics"], baseline),
        "top_candidates": scored[:50],
        "outputs": {"json": args.out_json, "md": args.out_md, "overlay": args.out_overlay},
    }
    write_json(Path(args.out_json), report)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(render(report), encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "best_metrics": best["metrics"], "delta_vs_p188": report["delta_vs_p188"], "best_policy": best["policy"], "searched": len(scored)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
