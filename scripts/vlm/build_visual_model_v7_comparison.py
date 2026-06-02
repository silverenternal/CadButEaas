#!/usr/bin/env python3
"""Build a compact HTML comparison page for v7 visual outputs."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    output = ROOT / "reports/vlm/visual_demo_model_v7_comparison"
    output.mkdir(parents=True, exist_ok=True)
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CadStruct-MoE v7 Visual Comparison</title>
  <style>
    body{font-family:Arial,Helvetica,sans-serif;margin:24px;color:#222}
    h1{font-size:22px;margin:0 0 12px}
    p{max-width:980px;line-height:1.45}
    .grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;align-items:start}
    iframe{width:100%;height:720px;border:1px solid #ccc}
    .panel h2{font-size:16px;margin:8px 0}
  </style>
</head>
<body>
  <h1>CadStruct-MoE v7 Visual Comparison</h1>
  <p>Left: model_v7 with only adopted retrained model components. Right: postprocess_v7, reported separately. Boundary v7 is model-side; appliance/equipment cleanup is postprocess-side because SymbolFixture v13 was rejected by locked evaluation guards.</p>
  <div class="grid">
    <section class="panel">
      <h2>model_v7</h2>
      <iframe src="../visual_demo_model_v7_model/review_pack_v2/index.html"></iframe>
    </section>
    <section class="panel">
      <h2>postprocess_v7</h2>
      <iframe src="../visual_demo_model_v7_postprocessed/review_pack_v2/index.html"></iframe>
    </section>
  </div>
</body>
</html>
"""
    (output / "index.html").write_text(html, encoding="utf-8")
    print(output / "index.html")


if __name__ == "__main__":
    main()
