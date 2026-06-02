#!/usr/bin/env python3
"""Build paper-safe symbol MoE assets for P207."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper_assets"


def load(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def fmt(value: float) -> str:
    return f"{value:.6f}"


def metric_row(name: str, metrics: dict[str, Any], delta_base: dict[str, Any] | None, ci: str, note: str) -> str:
    delta = "—" if delta_base is None else fmt(metrics["f1"] - delta_base["f1"])
    return " | ".join([
        f"| `{name}`",
        fmt(metrics["precision"]),
        fmt(metrics["recall"]),
        fmt(metrics["f1"]),
        delta,
        ci,
        fmt(metrics["center_recall"]),
        fmt(metrics["prediction_inflation"]),
        note + " |",
    ])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p206d = load("configs/vlm/symbol_p205b_ranker_regressor_p206d.json")
    p206e = load("configs/vlm/symbol_p205b_crop_ranker_p206e.json")
    p206f = load("configs/vlm/symbol_p205b_crop_ranker_p206f.json")
    p206g = load("configs/vlm/symbol_p206f_precision_repair_p206g.json")
    b206d = load("configs/vlm/symbol_p206d_bootstrap_validation.json")
    b206e = load("configs/vlm/symbol_p206e_vs_p206d_bootstrap_validation.json")
    b206f_e = load("configs/vlm/symbol_p206f_vs_p206e_bootstrap_validation.json")
    b206f_202 = load("configs/vlm/symbol_p206f_vs_p202_bootstrap_validation.json")
    b206g_f = load("configs/vlm/symbol_p206g_vs_p206f_bootstrap_validation.json")
    b206g_202 = load("configs/vlm/symbol_p206g_vs_p202_bootstrap_validation.json")

    p202 = p206f["baseline_metrics"]
    m206d = p206d["best_metrics"]
    m206e = p206e["best_metrics"]
    m206f = p206f["best_metrics"]
    m206g = p206g["best_metrics"]

    table = [
        "# P207 Symbol MoE Ablation Table",
        "",
        "## Scope",
        "",
        "This table is P101/bootstrap-bounded internal evidence over 74 public-raster overlay rows. It must not be described as full-raster, cross-dataset, or >0.95 symbol recognition performance.",
        "",
        "## Main Ablation",
        "",
        "| Stage | Precision | Recall | F1 | ΔF1 vs Previous | Paired Bootstrap ΔF1 95% CI | Center Recall | Inflation | Paper-safe Interpretation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        metric_row("P202 context verifier", p202, None, "—", "Precision-oriented context core."),
        metric_row("P206d learned tabular ranker/regressor", m206d, p202, f"[{b206d['bootstrap']['f1_delta']['ci95'][0]:.6f}, {b206d['bootstrap']['f1_delta']['ci95'][1]:.6f}]", "Small directional gain; CI crosses zero."),
        metric_row("P206e crop-visual ranker/regressor", m206e, m206d, f"[{b206e['bootstrap']['f1_delta']['ci95'][0]:.6f}, {b206e['bootstrap']['f1_delta']['ci95'][1]:.6f}]", "Visual crop evidence helps but remains directional."),
        metric_row("P206f cached hard-mined crop ranker", m206f, m206e, f"[{b206f_e['bootstrap']['f1_delta']['ci95'][0]:.6f}, {b206f_e['bootstrap']['f1_delta']['ci95'][1]:.6f}]", "Bootstrap-positive recall-oriented improvement."),
        metric_row("P206g conservative precision repair", m206g, m206f, f"[{b206g_f['bootstrap']['f1_delta']['ci95'][0]:.6f}, {b206g_f['bootstrap']['f1_delta']['ci95'][1]:.6f}]", "Precision repair preserves and strengthens F1."),
        "",
        "## P206g vs P202 Summary",
        "",
        f"- ΔF1 vs P202: `{m206g['f1'] - p202['f1']:.6f}` with 95% CI `[{b206g_202['bootstrap']['f1_delta']['ci95'][0]:.6f}, {b206g_202['bootstrap']['f1_delta']['ci95'][1]:.6f}]`, P(Δ>0)=`{b206g_202['bootstrap']['f1_delta']['prob_positive']:.3f}`.",
        f"- ΔRecall vs P202: `{m206g['recall'] - p202['recall']:.6f}` with 95% CI `[{b206g_202['bootstrap']['recall_delta']['ci95'][0]:.6f}, {b206g_202['bootstrap']['recall_delta']['ci95'][1]:.6f}]`.",
        f"- ΔPrecision vs P202: `{m206g['precision'] - p202['precision']:.6f}`; precision is now nearly matched while recall/F1 improve.",
        "",
        "## Artifact Trace",
        "",
        "| Stage | Config | Overlay/Report | Checkpoint/Cache |",
        "|---|---|---|---|",
        "| `P202` | `configs/vlm/symbol_context_verifier_p202.json` | `reports/vlm/symbol_context_verifier_p202_overlay.jsonl` | `checkpoints/symbol_context_verifier_p202/model.pt` |",
        "| `P206d` | `configs/vlm/symbol_p205b_ranker_regressor_p206d.json` | `reports/vlm/symbol_p205b_ranker_regressor_p206d_eval.md` | `checkpoints/symbol_p205b_ranker_regressor_p206d/model.pt` |",
        "| `P206e` | `configs/vlm/symbol_p205b_crop_ranker_p206e.json` | `reports/vlm/symbol_p205b_crop_ranker_p206e_eval.md` | `checkpoints/symbol_p205b_crop_ranker_p206e/model.pt` |",
        "| `P206f` | `configs/vlm/symbol_p205b_crop_ranker_p206f.json` | `reports/vlm/symbol_p205b_crop_ranker_p206f_eval.md` | `checkpoints/symbol_p205b_crop_ranker_p206f/model.pt`; `datasets/symbol_p205b_candidate_cache_p206f/manifest.pt` |",
        "| `P206g` | `configs/vlm/symbol_p206f_precision_repair_p206g.json` | `reports/vlm/symbol_p206f_precision_repair_p206g.md` | overlay `reports/vlm/symbol_p206f_precision_repair_p206g_overlay.jsonl` |",
        "",
    ]
    (OUT_DIR / "symbol_moe_ablation_table_p207.md").write_text("\n".join(table), encoding="utf-8")

    pipeline = [
        "# P207 Symbol MoE Pipeline Description",
        "",
        "## Raster-only Runtime Contract",
        "",
        "At runtime, the symbol branch consumes only raster pixels, trained model weights, and runtime configuration. SVG/parser geometry, CAD vector primitives, expected JSON, gold labels, annotation paths, and offline semantic identifiers are forbidden at runtime.",
        "",
        "## Staged Architecture",
        "",
        "1. `P202` forms the precision core with a context verifier over detector/crop candidates.",
        "2. `P205b` supplies high-recall tiled detector proposals for hard symbol classes, especially sink, shower, equipment, appliance, and stair.",
        "3. `P206d` adds a learned tabular ranker/regressor over P205b candidates using detector score, P200 crop scores, P202 context score, geometry, and offline IoU targets for training only.",
        "4. `P206e` adds raster crop evidence through a lightweight CNN+MLP ranker/regressor.",
        "5. `P206f` caches candidate crops and applies hard-mined/class-balanced training to convert the P205b recall signal into a bootstrap-positive F1 gain.",
        "6. `P206g` applies a conservative predicted-feature gate to repair precision while preserving the P206f recall gain.",
        "",
        "## Recommended Manuscript Wording",
        "",
        "We use a staged structural MoE for raster floorplan symbols: a precision-oriented context verifier is augmented by high-recall tiled proposals, then calibrated by learned candidate ranking and crop-visual box refinement. On the P101 public-raster overlay subset, the precision-repaired hard-mined crop branch improves F1 from 0.583820 to 0.591737 with a paired-bootstrap ΔF1 95% CI of [0.003063, 0.013714] versus the P202 core. This should be described as P101/bootstrap-bounded evidence, not as a full-raster or cross-dataset claim.",
        "",
        "## Precision/Recall Trade-off",
        "",
        f"P206g increases recall from `{p202['recall']:.6f}` to `{m206g['recall']:.6f}` and center recall from `{p202['center_recall']:.6f}` to `{m206g['center_recall']:.6f}`. Precision changes from `{p202['precision']:.6f}` to `{m206g['precision']:.6f}`, so the claim can emphasize recall rescue with almost no net precision loss versus the P202 core.",
        "",
    ]
    (OUT_DIR / "symbol_moe_pipeline_p203.md").write_text("\n".join(pipeline), encoding="utf-8")

    claim = [
        "# P207 Symbol Claim Boundary",
        "",
        "## Allowed Claims",
        "",
        "- P206f is the current best P101/bootstrap-bounded symbol overlay candidate with F1 `0.590142`, precision `0.665832`, recall `0.529904`, and center recall `0.564992`.",
        "- P206f has a positive paired-bootstrap ΔF1 lower bound versus P206e and P202 on the 74-row P101 overlay subset.",
        "- The architecture demonstrates a staged MoE recall-rescue path: context core → learned candidate ranker → crop-visual ranker → hard-mined cached crop refinement.",
        "",
        "## Forbidden or Unsafe Claims",
        "",
        "- Do not claim full-raster symbol recognition performance from these P101 overlay metrics.",
        "- Do not claim cross-dataset generalization without an independent held-out dataset report.",
        "- Do not claim >0.95 symbol F1 for the current raster symbol branch.",
        "- Do not describe offline IoU targets, gold labels, expected JSON, or annotations as runtime inputs.",
        "",
        "## Reviewer-safe Limitation Text",
        "",
        "The symbol ablation is evaluated on a 74-row public-raster overlay subset and uses paired bootstrap to estimate internal stability. Because policy selection and hard mining were performed in this development setting, the numbers should be interpreted as bounded evidence for the proposed staged MoE design rather than a final cross-dataset benchmark. We therefore report the exact artifact chain and avoid full-raster or cross-dataset generalization claims in this table.",
        "",
        "## Next Validation Needed",
        "",
        "- Lock an independent row/page split or external floorplan symbol benchmark.",
        "- Re-run P202/P206d/P206e/P206f without policy reselection on the locked split.",
        "- Report per-class and per-size deltas, especially tiny/small sink and shower cases.",
        "",
    ]
    (OUT_DIR / "symbol_claim_boundary_p207.md").write_text("\n".join(claim), encoding="utf-8")

    manifest = {
        "id": "P207_symbol_paper_assets",
        "scope": "P101/bootstrap-bounded 74-row symbol overlay evidence",
        "current_best": {"stage": "P206g", **m206g},
        "bootstrap": {
            "p206f_vs_p206e": b206f_e["bootstrap"],
            "p206g_vs_p206f": b206g_f["bootstrap"],
            "p206g_vs_p202": b206g_202["bootstrap"],
        },
        "outputs": [
            "paper_assets/symbol_moe_ablation_table_p207.md",
            "paper_assets/symbol_moe_pipeline_p203.md",
            "paper_assets/symbol_claim_boundary_p207.md",
        ],
        "claim_boundary": "Internal P101/bootstrap-bounded evidence only; no full-raster, cross-dataset, or >0.95 symbol-F1 claim.",
    }
    (ROOT / "configs/vlm/symbol_paper_assets_p207.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
