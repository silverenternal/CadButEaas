#!/usr/bin/env python3
"""Bootstrap top P221c policies and pick a precision-safe candidate if one exists."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/vlm"))

from freeze_symbol_p222_p221a_sink_tiny import bootstrap, metrics, score_rows  # noqa: E402
from fuse_symbol_p206g_with_p211_p212 import load_p206g, write_json, write_jsonl  # noqa: E402
from train_symbol_p221c_candidate_gate import fuse, build_overlay, row_id, read_jsonl  # noqa: E402

BASE = ROOT / "reports/vlm/symbol_p222_p221a_frozen_overlay.jsonl"
REPORT = ROOT / "reports/vlm/symbol_p221c_candidate_gate_eval.json"
DATASET = ROOT / "reports/vlm/symbol_p221c_candidate_gate_dataset.jsonl"
OUT = ROOT / "reports/vlm/symbol_p221c_precision_safe_policy_selection.json"
MD = ROOT / "reports/vlm/symbol_p221c_precision_safe_policy_selection.md"
OVERLAY = ROOT / "reports/vlm/symbol_p221c_precision_safe_overlay.jsonl"


def main() -> None:
    report = json.loads(REPORT.read_text())
    rows, core, golds = load_p206g(BASE)
    ids = [row_id(row) for row in rows]
    dataset = read_jsonl(DATASET)
    base_per = score_rows(core, golds, ids)
    bm = metrics(base_per)
    candidates = []
    seen = set()
    for item in report["top_grid"][:40]:
        policy = item["policy"]
        key = json.dumps(policy, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        fused = fuse(core, dataset, policy)
        per = score_rows(fused, golds, ids)
        m = metrics(per)
        b = bootstrap(base_per, per, seed=224)
        candidates.append({"policy": policy, "metrics": m, "added_predictions": item["added_predictions"], "bootstrap_vs_p222": b})
    safe = [c for c in candidates if c["bootstrap_vs_p222"]["f1_delta"]["ci95"][0] > 0 and c["bootstrap_vs_p222"]["precision_delta"]["ci95"][0] >= 0]
    weak_safe = [c for c in candidates if c["bootstrap_vs_p222"]["f1_delta"]["ci95"][0] >= 0 and c["bootstrap_vs_p222"]["precision_delta"]["ci95"][0] >= 0]
    if safe:
        selected = sorted(safe, key=lambda c: (c["metrics"]["f1"], c["metrics"]["recall"], c["metrics"]["precision"]), reverse=True)[0]
        decision = "precision_safe_promotable_candidate"
    elif weak_safe:
        selected = sorted(weak_safe, key=lambda c: (c["metrics"]["f1"], c["metrics"]["recall"], c["metrics"]["precision"]), reverse=True)[0]
        decision = "weak_safe_candidate_ci_touches_zero"
    else:
        selected = sorted(candidates, key=lambda c: (c["bootstrap_vs_p222"]["precision_delta"]["ci95"][0], c["metrics"]["f1"]), reverse=True)[0]
        decision = "no_precision_safe_candidate"
    fused = fuse(core, dataset, selected["policy"])
    write_jsonl(OVERLAY, build_overlay(rows, fused, selected["policy"]))
    out = {"id": "P221c_precision_safe_policy_selection", "baseline_metrics": bm, "decision": decision, "selected": selected, "safe_count": len(safe), "weak_safe_count": len(weak_safe), "bootstrapped_candidates": candidates, "outputs": {"overlay": str(OVERLAY.relative_to(ROOT)), "markdown": str(MD.relative_to(ROOT))}}
    write_json(OUT, out)
    lines = ["# P221c Precision-Safe Policy Selection", "", f"- Decision: `{decision}`", f"- Safe count: `{len(safe)}`", f"- Weak-safe count: `{len(weak_safe)}`", "", "## Selected", f"- Policy: `{selected['policy']['name']}`", f"- Metrics: F1 `{selected['metrics']['f1']:.6f}`, P `{selected['metrics']['precision']:.6f}`, R `{selected['metrics']['recall']:.6f}`", f"- Added: `{selected['added_predictions']}`", f"- ΔF1 CI: `{selected['bootstrap_vs_p222']['f1_delta']['ci95']}`", f"- ΔPrecision CI: `{selected['bootstrap_vs_p222']['precision_delta']['ci95']}`", f"- ΔRecall CI: `{selected['bootstrap_vs_p222']['recall_delta']['ci95']}`", "", "## Top Bootstrapped", "| Policy | Added | F1 | P | R | ΔF1 CI | ΔP CI |", "|---|---:|---:|---:|---:|---|---|"]
    for c in candidates[:20]:
        m = c["metrics"]; b = c["bootstrap_vs_p222"]
        lines.append(f"| {c['policy']['name']} | {c['added_predictions']} | {m['f1']:.6f} | {m['precision']:.6f} | {m['recall']:.6f} | `{b['f1_delta']['ci95']}` | `{b['precision_delta']['ci95']}` |")
    MD.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "selected": selected, "safe_count": len(safe), "weak_safe_count": len(weak_safe)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
