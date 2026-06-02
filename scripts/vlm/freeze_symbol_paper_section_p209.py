#!/usr/bin/env python3
"""Freeze bounded manuscript-ready symbol MoE section assets."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper_assets"


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p208 = load("configs/vlm/symbol_locked_validation_p208.json")
    p207 = load("configs/vlm/symbol_paper_assets_p207.json")
    best = p207["current_best"]
    p208_boot = p208["bootstrap"]["P206g"]
    p202 = next(stage for stage in p208["stages"] if stage["id"] == "P202")["metrics"]

    section = f"""# Bounded Manuscript Section: Symbol Structural MoE

## Suggested Results Paragraph

For raster symbol parsing, we evaluate a staged structural mixture-of-experts (MoE) branch on the 74-row P101 public-raster overlay subset. The precision-oriented P202 context verifier is augmented by high-recall tiled proposals, learned candidate ranking, crop-visual box refinement, hard-mined crop training, and a conservative precision-repair gate. With all policies frozen and no re-selection in P208, the final P206g branch reaches F1 `{best['f1']:.6f}`, precision `{best['precision']:.6f}`, recall `{best['recall']:.6f}`, and center recall `{best['center_recall']:.6f}`. Relative to the frozen P202 context core, P206g improves F1 by `{best['f1'] - p202['f1']:.6f}` with paired-bootstrap 95% CI `[{p208_boot['delta_f1_vs_baseline_ci95'][0]:.6f}, {p208_boot['delta_f1_vs_baseline_ci95'][1]:.6f}]` and P(Δ>0)=`{p208_boot['prob_delta_positive']:.3f}`.

## Required Claim Boundary Sentence

These symbol results are P101/bootstrap-bounded internal ablation evidence on 74 public-raster overlay rows. They are not independent held-out, full-raster, cross-dataset, or >0.95 symbol-F1 claims; broad benchmark claims require replacing the P208 manifest with an independent locked split and rerunning the frozen overlays without policy re-selection.

## Suggested Table Caption

Ablation of the raster symbol structural MoE on the P101 public-raster overlay subset. Each row uses frozen materialized overlays and the P208 script performs no policy re-selection. Confidence intervals are paired bootstrap estimates over the declared 74-row manifest. Because the manifest is not independent from development rows, the table supports bounded internal evidence and reproducibility, not broad cross-dataset claims.

## Suggested Figure Caption

The symbol branch starts from a precision-oriented context verifier, adds high-recall tiled symbol proposals, ranks proposals with tabular and crop-visual experts, hard-mines candidate crops, and finally applies a conservative predicted-feature precision repair. Runtime inputs remain raster pixels, model weights, and configuration only; offline gold labels and IoU targets are used only for training/evaluation.

## Do Not Say

- Do not say the current symbol branch achieves full-raster symbol recognition performance.
- Do not say the current result generalizes across datasets.
- Do not say symbol F1 exceeds 0.95.
- Do not imply expected JSON, SVG geometry, annotation paths, or gold labels are runtime inputs.
"""
    (OUT / "symbol_moe_manuscript_section_p209.md").write_text(section, encoding="utf-8")

    checklist = {
        "id": "P209_symbol_section_freeze_checklist",
        "current_best_stage": "P206g",
        "current_best_metrics": {
            "precision": best["precision"],
            "recall": best["recall"],
            "f1": best["f1"],
            "center_recall": best["center_recall"],
            "prediction_inflation": best["prediction_inflation"],
        },
        "p208_frozen_delta_f1_vs_p202_ci95": p208_boot["delta_f1_vs_baseline_ci95"],
        "paper_claim_eligible": p208.get("paper_claim_eligible", False),
        "required_claim_boundary": "P101/bootstrap-bounded internal ablation only; not independent held-out/full-raster/cross-dataset/>0.95 evidence.",
        "must_cite_artifacts": [
            "paper_assets/symbol_moe_ablation_table_p207.md",
            "paper_assets/symbol_moe_pipeline_p203.md",
            "paper_assets/symbol_claim_boundary_p207.md",
            "paper_assets/symbol_moe_manuscript_section_p209.md",
            "configs/vlm/symbol_locked_validation_manifest_p208.json",
            "configs/vlm/symbol_locked_validation_p208.json",
            "reports/vlm/symbol_locked_validation_p208.md",
            "reports/vlm/symbol_p206f_precision_repair_p206g.md",
        ],
        "freeze_status": "ready_for_bounded_manuscript_use",
    }
    (ROOT / "configs/vlm/symbol_section_freeze_p209.json").write_text(json.dumps(checklist, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    index = [
        "# Symbol MoE Final Asset Index",
        "",
        "## Manuscript-facing Assets",
        "",
        "- `paper_assets/symbol_moe_manuscript_section_p209.md` — ready-to-paste bounded results paragraph, captions, and forbidden-claim list.",
        "- `paper_assets/symbol_moe_ablation_table_p207.md` — ablation table including P202/P206d/P206e/P206f/P206g.",
        "- `paper_assets/symbol_moe_pipeline_p203.md` — staged raster-only MoE pipeline description.",
        "- `paper_assets/symbol_claim_boundary_p207.md` — allowed/forbidden claims and reviewer-safe limitation text.",
        "",
        "## Reproducibility Artifacts",
        "",
        "- `configs/vlm/symbol_locked_validation_manifest_p208.json` — declared 74-row frozen manifest.",
        "- `scripts/vlm/validate_symbol_locked_manifest_p208.py` — no-reselection frozen validation script.",
        "- `reports/vlm/symbol_locked_validation_p208.md` — frozen metrics and paired bootstrap report.",
        "- `configs/vlm/symbol_section_freeze_p209.json` — freeze checklist and required citations.",
        "",
        "## Current Best",
        "",
        f"- Stage: `P206g`",
        f"- Metrics: F1 `{best['f1']:.6f}`, precision `{best['precision']:.6f}`, recall `{best['recall']:.6f}`, center recall `{best['center_recall']:.6f}`.",
        f"- Frozen ΔF1 vs P202 CI: `[{p208_boot['delta_f1_vs_baseline_ci95'][0]:.6f}, {p208_boot['delta_f1_vs_baseline_ci95'][1]:.6f}]`.",
        "- Claim status: bounded internal evidence only; independent validation still required for broad claims.",
        "",
    ]
    (OUT / "symbol_final_asset_index_p209.md").write_text("\n".join(index), encoding="utf-8")
    print(json.dumps(checklist, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
