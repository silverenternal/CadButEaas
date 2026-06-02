#!/usr/bin/env python3
"""Build a simple side-by-side index linking raw/model/postprocess v5 packs."""

from __future__ import annotations

import argparse
import html
from pathlib import Path

from v5_pipeline_utils import load_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="reports/vlm/visual_demo_model_v5_raw")
    parser.add_argument("--model-dir", default="reports/vlm/visual_demo_model_v5_model")
    parser.add_argument("--post-dir", default="reports/vlm/visual_demo_model_v5_postprocessed")
    parser.add_argument("--output-dir", default="reports/vlm/visual_demo_model_v5_side_by_side")
    args = parser.parse_args()

    raw = load_json(Path(args.raw_dir) / "sample_manifest_v1.json", {})
    model = load_json(Path(args.model_dir) / "sample_manifest_v1.json", {})
    post = load_json(Path(args.post_dir) / "sample_manifest_v1.json", {})
    samples = raw.get("samples") or model.get("samples") or post.get("samples") or []
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cards = []
    for sample in samples:
        sid = str(sample.get("sample_id") or "")
        cards.append(
            f"""
      <section class="sample">
        <h2>{html.escape(sid)}</h2>
        <div class="grid">
          {frame('Raw upstream', Path('..') / Path(args.raw_dir).name / sid / 'side_by_side.svg')}
          {frame('Model v5', Path('..') / Path(args.model_dir).name / sid / 'side_by_side.svg')}
          {frame('Postprocess v5', Path('..') / Path(args.post_dir).name / sid / 'side_by_side.svg')}
        </div>
      </section>"""
        )
    content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CadStruct-MoE v5 raw/model/postprocess comparison</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #222; }}
    h1 {{ font-size: 24px; margin-bottom: 6px; }}
    .note {{ color: #555; margin-bottom: 22px; }}
    .sample {{ margin: 28px 0 42px; }}
    h2 {{ font-size: 18px; margin: 0 0 10px; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, minmax(280px, 1fr)); gap: 14px; align-items: start; }}
    .panel {{ border: 1px solid #bbb; padding: 8px; background: #fff; }}
    .panel h3 {{ font-size: 14px; margin: 0 0 8px; }}
    iframe {{ width: 100%; aspect-ratio: 2 / 1; border: 0; background: #fafafa; }}
    @media (max-width: 1100px) {{ .grid {{ grid-template-columns: 1fr; }} iframe {{ aspect-ratio: 1.4 / 1; }} }}
  </style>
</head>
<body>
  <h1>CadStruct-MoE v5 CubiCasa Visual Comparison</h1>
  <p class="note">Raw upstream, model_v5, and postprocess_v5 are separated. Postprocess cleanup is not reported as model retraining gain.</p>
  {''.join(cards)}
</body>
</html>
"""
    (out_dir / "index.html").write_text(content, encoding="utf-8")
    print({"output": str(out_dir / "index.html"), "samples": len(samples)})


def frame(title: str, src: Path) -> str:
    return f'<div class="panel"><h3>{html.escape(title)}</h3><iframe src="{html.escape(src.as_posix())}"></iframe></div>'


if __name__ == "__main__":
    main()
