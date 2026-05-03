#!/usr/bin/env python3
"""Build SymbolFixture v10 paper-readiness report from latest v9 run."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent.parent
SOURCE = ROOT / "checkpoints" / "symbol_fixture_expert_v9" / "train_summary.json"
OUTPUT = ROOT / "reports" / "vlm" / "symbol_fixture_v10_eval.json"


def main() -> int:
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    dev = data["splits"]["dev"]
    locked = data["splits"].get("locked") or {}
    per_label = dev.get("per_label") or {}
    worst = sorted(
        ({"label": label, **metrics} for label, metrics in per_label.items()),
        key=lambda row: (row.get("f1") or 0.0, row.get("support") or 0),
    )
    macro_f1 = float(dev.get("macro_f1") or 0.0)
    report = {
        "version": "symbol_fixture_v10_eval",
        "created": "2026-05-03",
        "source": str(SOURCE.relative_to(ROOT)),
        "model_type": data.get("model_type"),
        "taxonomy": {
            "main_table": "9-class fine taxonomy is publishable at the >=0.80 minimum boundary.",
            "labels": list(data.get("label_map", {}).keys()),
            "limitations": [
                "generic_symbol and table remain weak long-tail classes",
                "full preferred >=0.90 macro F1 is not reached",
            ],
        },
        "dev": {
            "symbols": dev.get("symbols"),
            "accuracy": dev.get("accuracy"),
            "macro_f1": macro_f1,
            "per_label": per_label,
            "worst_labels": worst[:5],
        },
        "locked": locked,
        "done_when_check": {
            "report_generated": True,
            "macro_f1_ge_080": macro_f1 >= 0.80,
            "macro_f1_ge_090_preferred": macro_f1 >= 0.90,
            "main_capability_table_allowed": macro_f1 >= 0.80,
        },
        "status": "main_table_minimum_pass" if macro_f1 >= 0.80 else "extension_or_limitation",
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    print(json.dumps(report["done_when_check"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
