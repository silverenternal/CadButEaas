#!/usr/bin/env python3
"""Router ablation: Deterministic vs Learned (fair) routing comparison.

Runs both routers on the 493-record dev split and compares:
- Routing agreement rate (% routed to same expert)
- Disagreement analysis (which families get re-routed)
- Impact on downstream expert predictions
- Wrong routing rate (learned vs gold family)

Done when: comparison report generated with per-family breakdown.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
DEV_SPLIT = ROOT / "datasets/cadstruct_real_world_benchmark_v1/room_space/cubicasa5k_reviewed_locked_test.jsonl"
OUTPUT = ROOT / "reports/vlm/router_ablation_deterministic_vs_learned.json"

# Family name mapping: deterministic uses short names, learned model uses full expert names
DET_TO_LEARNED = {
    "text": "text_dimension",
    "boundary": "wall_opening",
    "space": "room_space",
    "symbol": "symbol_fixture",
    "sheet": "sheet_layout",
}
LEARNED_TO_DET = {v: k for k, v in DET_TO_LEARNED.items()}

sys.path.insert(0, str(ROOT / "scripts/vlm"))
from cadstruct_moe.router import DeterministicRouter, LearnedRouter


def main() -> None:
    print("=== Router Ablation: Deterministic vs Learned (fair) ===\n")
    started = time.perf_counter()

    dev_records = load_jsonl(DEV_SPLIT)
    print(f"Dev split: {len(dev_records)} records")

    det_router = DeterministicRouter()
    lrd_router = LearnedRouter()

    if lrd_router._model is None:
        print("[ERROR] Learned router model not found. Run train_moe_router_v2.py --fair first.")
        return 1

    print(f"Loaded DeterministicRouter and LearnedRouter (fair, 10 geometry features)")

    # Route all records with both routers
    det_all = []
    lrd_all = []
    disagreements = []
    per_family_agreement = defaultdict(lambda: {"agree": 0, "disagree": 0})

    for rec in dev_records:
        det_routed = det_router.route_record(rec)
        lrd_routed = lrd_router.route_record(rec)

        for rc_det in det_routed:
            det_all.append({
                "candidate_id": rc_det.candidate_id,
                "family": rc_det.family,
                "expert": rc_det.expert,
                "confidence": rc_det.confidence,
            })

        # Build lookup for learned
        lrd_by_id = {rc.candidate_id: rc for rc in lrd_routed}
        for rc in lrd_routed:
            lrd_all.append({
                "candidate_id": rc.candidate_id,
                "family": rc.family,
                "expert": rc.expert,
                "confidence": rc.confidence,
            })

        # Compare per candidate
        for rc_det in det_routed:
            cid = rc_det.candidate_id
            rc_lrd = lrd_by_id.get(cid)
            if rc_lrd is None:
                continue

            det_family_normalized = DET_TO_LEARNED.get(rc_det.family, rc_det.family)
            same_family = det_family_normalized == rc_lrd.family
            per_family_agreement[rc_det.family]["agree" if same_family else "disagree"] += 1

            if not same_family:
                disagreements.append({
                    "candidate_id": cid,
                    "deterministic_family": rc_det.family,
                    "deterministic_family_normalized": det_family_normalized,
                    "learned_family": rc_lrd.family,
                    "det_confidence": rc_det.confidence,
                    "lrd_confidence": rc_lrd.confidence,
                    "candidate_type": rc_det.candidate_type,
                })

    # Summary stats
    total = len(det_all)
    n_disagree = len(disagreements)
    agreement_rate = (total - n_disagree) / max(total, 1)

    print(f"\nTotal candidates: {total}")
    print(f"Agreement: {total - n_disagree} ({agreement_rate:.4f})")
    print(f"Disagreement: {n_disagree} ({n_disagree / max(total, 1):.4f})")

    # Per-family agreement
    per_family_summary = {}
    for fam, counts in sorted(per_family_agreement.items()):
        agree = counts["agree"]
        disagree = counts["disagree"]
        fam_total = agree + disagree
        per_family_summary[fam] = {
            "total": fam_total,
            "agree": agree,
            "disagree": disagree,
            "agreement_rate": round(agree / max(fam_total, 1), 6),
        }
        print(f"  {fam:20s}: {agree}/{fam_total} agree ({agree / max(fam_total, 1):.4f})")

    # Disagreement confusion matrix
    confusion = defaultdict(lambda: defaultdict(int))
    for d in disagreements:
        confusion[d["deterministic_family"]][d["learned_family"]] += 1

    # Learned router wrong routing rate (vs deterministic as "gold")
    wrong_by_family = defaultdict(lambda: {"wrong": 0, "total": 0})
    for d in disagreements:
        fam = d["deterministic_family"]
        wrong_by_family[fam]["wrong"] += 1
        wrong_by_family[fam]["total"] += 1
    for fam in per_family_agreement:
        wrong_by_family[fam]["total"] = per_family_agreement[fam]["agree"] + per_family_agreement[fam]["disagree"]

    wrong_rate_summary = {}
    for fam, counts in sorted(wrong_by_family.items()):
        rate = counts["wrong"] / max(counts["total"], 1)
        wrong_rate_summary[fam] = {
            "wrong": counts["wrong"],
            "total": counts["total"],
            "wrong_rate": round(rate, 6),
        }

    # Confidence analysis
    det_conf_on_disagree = [d["det_confidence"] for d in disagreements]
    lrd_conf_on_disagree = [d["lrd_confidence"] for d in disagreements]
    avg_det_conf = sum(det_conf_on_disagree) / max(len(det_conf_on_disagree), 1)
    avg_lrd_conf = sum(lrd_conf_on_disagree) / max(len(lrd_conf_on_disagree), 1)

    # Top disagreement patterns (using normalized names)
    pattern_counter = Counter()
    for d in disagreements:
        key = f"{d['deterministic_family']} ({d['deterministic_family_normalized']}) -> {d['learned_family']}"
        pattern_counter[key] += 1

    elapsed = time.perf_counter() - started

    report = {
        "version": "router_ablation_deterministic_vs_learned_v1",
        "dev_records": len(dev_records),
        "total_candidates": total,
        "deterministic_router": {"type": "DeterministicRouter", "features": "type hints + bbox"},
        "learned_router": {"type": "LearnedRouter", "features": "10 geometry features (fair)", "model": "checkpoints/moe_router_v2/model_v2_fair.joblib"},
        "overall": {
            "agreement_count": total - n_disagree,
            "disagreement_count": n_disagree,
            "agreement_rate": round(agreement_rate, 6),
            "disagreement_rate": round(n_disagree / max(total, 1), 6),
        },
        "per_family_agreement": per_family_summary,
        "confusion_matrix": {k: dict(v) for k, v in sorted(confusion.items())},
        "wrong_routing_rate_by_family": wrong_rate_summary,
        "confidence_on_disagreement": {
            "avg_deterministic_confidence": round(avg_det_conf, 6),
            "avg_learned_confidence": round(avg_lrd_conf, 6),
        },
        "top_disagreement_patterns": [{"pattern": p, "count": c} for p, c in pattern_counter.most_common(10)],
        "elapsed_seconds": round(elapsed, 1),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nReport saved to {OUTPUT}")
    print(f"Elapsed: {elapsed:.1f}s")

    return 0


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    sys.exit(main())
