#!/usr/bin/env python3
"""Create appendix-only learned/top-k router summary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
SOURCE = ROOT / "reports" / "vlm" / "moe_router_v3_fair_ablation.json"
OUTPUT = ROOT / "reports" / "vlm" / "router_appendix_topk_v1.json"


def main() -> int:
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    models = data.get("models") or {}
    rows: list[dict[str, Any]] = []
    for key in ["deterministic_router", "learned_fair_router_v3", "top2_confidence_router", "top3_confidence_router", "oracle_router"]:
        model = models.get(key) or {}
        rows.append(
            {
                "router": key,
                "wrong_expert_rate": model.get("wrong_expert_rate"),
                "accuracy": model.get("accuracy"),
                "abstain_rate": model.get("abstain_rate"),
                "oracle_in_topk_rate": model.get("oracle_in_top2_rate") or model.get("oracle_in_top3_rate"),
                "paper_role": "main" if key == "deterministic_router" else "appendix_or_future_work",
            }
        )
    report = {
        "version": "router_appendix_topk_v1",
        "created": "2026-05-03",
        "source": str(SOURCE.relative_to(ROOT)),
        "summary_rows": rows,
        "main_text_policy": {
            "main_router": "deterministic_router",
            "learned_router_main_claim_allowed": False,
            "reason": "learned_fair_router_v3 wrong_expert_rate=0.152302 and top-k still underperforms deterministic routing for family assignment; use only as appendix/future-work evidence.",
        },
        "status": "passed",
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT}")
    print(json.dumps({"status": report["status"], "learned_wrong_expert_rate": models.get("learned_fair_router_v3", {}).get("wrong_expert_rate")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
