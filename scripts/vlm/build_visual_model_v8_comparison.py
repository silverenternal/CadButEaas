#!/usr/bin/env python3
"""Render v8 advisor comparison pages with source-mode boundaries."""

from __future__ import annotations

import html
import json
from pathlib import Path

from v8_raster_e2e_utils import ROOT, load_json, update_todo_remove, write_json


def main() -> None:
    raster_decision = load_json("reports/vlm/raster_e2e_model_v8_adoption_decisions.json", {})
    hybrid_decision = load_json("reports/vlm/hybrid_visual_model_v8_adoption_decisions.json", {})
    make_raster_rejection_pack(raster_decision)
    make_comparison(raster_decision, hybrid_decision)
    update_todo_remove(["RASTER-V8-T7"])
    print(ROOT / "reports/vlm/visual_demo_v8_comparison/index.html")


def make_raster_rejection_pack(decision: dict) -> None:
    out = ROOT / "reports/vlm/visual_demo_raster_e2e_v8_model/review_pack_v2"
    out.mkdir(parents=True, exist_ok=True)
    html_text = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Raster E2E v8 Rejected</title>
<style>body{{font-family:Arial,Helvetica,sans-serif;margin:24px;color:#222}}code{{background:#f4f4f4;padding:2px 4px}}.badge{{display:inline-block;background:#8b1a1a;color:white;padding:3px 8px;border-radius:4px}}</style></head>
<body>
<h1>model_v8_raster_e2e</h1>
<p><span class="badge">rejected / no-adoption</span></p>
<p>This panel is intentionally not a visual success gallery. The image-only raster detector failed the locked adoption guard, so no pure raster scene-graph stream is claimed.</p>
<p>Decision report: <code>reports/vlm/raster_e2e_model_v8_adoption_decisions.json</code></p>
<p>Detector metric report: <code>reports/vlm/raster_candidate_detector_v8_eval.json</code></p>
<pre>{html.escape(json.dumps(decision.get("no_adoption") or {}, ensure_ascii=False, indent=2))}</pre>
</body></html>
"""
    (out / "index.html").write_text(html_text, encoding="utf-8")
    (out / "README.md").write_text(
        "# Raster E2E v8 Review Pack\n\nRejected/no-adoption because the image-only detector failed locked metrics. This is failure evidence, not a model-quality figure.\n",
        encoding="utf-8",
    )
    write_json("reports/vlm/visual_demo_raster_e2e_v8_model/sample_manifest_v1.json", {"samples": [], "status": "rejected_no_adoption"})


def make_comparison(raster_decision: dict, hybrid_decision: dict) -> None:
    out = ROOT / "reports/vlm/visual_demo_v8_comparison"
    out.mkdir(parents=True, exist_ok=True)
    html_text = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CadStruct-MoE v8 Source-Mode Comparison</title>
  <style>
    body{font-family:Arial,Helvetica,sans-serif;margin:24px;color:#222}
    h1{font-size:22px;margin:0 0 12px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;align-items:start}
    iframe{width:100%;height:650px;border:1px solid #ccc}.panel h2{font-size:16px;margin:8px 0}
    .badge{display:inline-block;padding:3px 7px;border-radius:4px;background:#eee;margin-left:6px}.bad{background:#8b1a1a;color:white}.hybrid{background:#76520e;color:white}.base{background:#315b7d;color:white}
    p{max-width:1100px;line-height:1.45}
  </style>
</head>
<body>
  <h1>CadStruct-MoE v8 Source-Mode Comparison</h1>
  <p>v8 separates claim boundaries: v7 remains SVG/parser candidate geometry plus model refiners; raster_e2e is rejected because the image-only detector failed locked metrics; hybrid_v8 keeps SVG candidate geometry but adds adopted raster crop visual-evidence review flags. Postprocess remains separate.</p>
  <div class="grid">
    <section class="panel"><h2>v7_svg_candidate <span class="badge base">baseline</span></h2><iframe src="../visual_demo_model_v7_model/review_pack_v2/index.html"></iframe></section>
    <section class="panel"><h2>v8_raster_e2e <span class="badge bad">rejected</span></h2><iframe src="../visual_demo_raster_e2e_v8_model/review_pack_v2/index.html"></iframe></section>
    <section class="panel"><h2>v8_hybrid <span class="badge hybrid">hybrid</span></h2><iframe src="../visual_demo_hybrid_v8_model/review_pack_v2/index.html"></iframe></section>
    <section class="panel"><h2>postprocess_v7 <span class="badge">postprocess</span></h2><iframe src="../visual_demo_model_v7_postprocessed/review_pack_v2/index.html"></iframe></section>
  </div>
</body>
</html>
"""
    (out / "index.html").write_text(html_text, encoding="utf-8")
    write_json(
        out / "comparison_manifest.json",
        {
            "version": "visual_demo_v8_comparison",
            "raster_e2e": {"adopted": raster_decision.get("adopted"), "report": "reports/vlm/raster_e2e_model_v8_adoption_decisions.json"},
            "hybrid_v8": {"adopted_components": hybrid_decision.get("adopted_components"), "report": "reports/vlm/hybrid_visual_model_v8_adoption_decisions.json"},
            "claim_boundary": "The comparison page must not call hybrid or v7 outputs pure raster E2E.",
        },
    )


if __name__ == "__main__":
    main()
