#!/usr/bin/env python3
"""Render a static HTML helper for room ambiguity review queues."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-csv", default="reports/vlm/room_ambiguity_review_pack_v1/review_queue.csv")
    parser.add_argument("--output", default="reports/vlm/room_ambiguity_review_pack_v1/review.html")
    args = parser.parse_args()

    review_csv = Path(args.review_csv)
    output = Path(args.output)
    rows = load_csv(review_csv)
    output.write_text(render_html(rows, output.parent), encoding="utf-8")
    print(json.dumps({"output": str(output), "items": len(rows)}, ensure_ascii=False, indent=2))


def render_html(rows: list[dict[str, str]], base_dir: Path) -> str:
    cards = "\n".join(render_card(row, base_dir) for row in rows)
    counts = {}
    for row in rows:
        counts[row["prediction"]] = counts.get(row["prediction"], 0) + 1
    counts_text = ", ".join(f"{label}: {count}" for label, count in sorted(counts.items()))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Room Ambiguity Review Pack</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.4;
      background: #f6f7f9;
      color: #17202a;
    }}
    body {{
      margin: 0;
      padding: 24px;
    }}
    header {{
      max-width: 1280px;
      margin: 0 auto 18px;
    }}
    h1 {{
      font-size: 24px;
      margin: 0 0 8px;
    }}
    .meta {{
      color: #5c6670;
      font-size: 14px;
    }}
    .toolbar {{
      display: flex;
      gap: 12px;
      align-items: center;
      margin-top: 16px;
      flex-wrap: wrap;
    }}
    input, select {{
      height: 34px;
      border: 1px solid #c9d1d9;
      border-radius: 6px;
      padding: 0 10px;
      background: white;
    }}
    main {{
      max-width: 1280px;
      margin: 0 auto;
      display: grid;
      gap: 14px;
    }}
    article {{
      background: white;
      border: 1px solid #d8dee4;
      border-radius: 8px;
      padding: 14px;
      display: grid;
      grid-template-columns: minmax(280px, 460px) 1fr;
      gap: 16px;
    }}
    .preview {{
      display: grid;
      gap: 8px;
      align-content: start;
    }}
    iframe {{
      width: 100%;
      height: 360px;
      border: 1px solid #d8dee4;
      border-radius: 6px;
      background: #fff;
    }}
    .fields {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px 14px;
      font-size: 14px;
    }}
    .field strong {{
      display: block;
      color: #5c6670;
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
    }}
    code {{
      background: #f0f3f6;
      padding: 2px 5px;
      border-radius: 4px;
      word-break: break-all;
    }}
    .actions {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 12px;
    }}
    .pill {{
      border: 1px solid #c9d1d9;
      border-radius: 999px;
      padding: 4px 9px;
      font-size: 12px;
      background: #f6f8fa;
    }}
    .path {{
      grid-column: 1 / -1;
    }}
    a {{
      color: #0969da;
      text-decoration: none;
    }}
    a:hover {{
      text-decoration: underline;
    }}
    @media (max-width: 900px) {{
      article {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Room Ambiguity Review Pack</h1>
    <div class="meta">{len(rows)} candidates. Prediction counts: {html.escape(counts_text)}</div>
    <div class="meta">Use this page to inspect candidates, then fill <code>review_queue.csv</code>. Valid labels: <code>accept_typed</code>, <code>keep_room</code>, <code>unclear</code>, <code>exclude</code>.</div>
    <div class="toolbar">
      <input id="search" type="search" placeholder="Search text/path/id">
      <select id="prediction">
        <option value="">All predictions</option>
        {''.join(f'<option value="{html.escape(label)}">{html.escape(label)}</option>' for label in sorted(counts))}
      </select>
    </div>
  </header>
  <main>
    {cards}
  </main>
  <script>
    const search = document.getElementById('search');
    const prediction = document.getElementById('prediction');
    function applyFilter() {{
      const q = search.value.toLowerCase();
      const p = prediction.value;
      document.querySelectorAll('article').forEach(card => {{
        const matchText = card.dataset.search.includes(q);
        const matchPred = !p || card.dataset.prediction === p;
        card.style.display = matchText && matchPred ? '' : 'none';
      }});
    }}
    search.addEventListener('input', applyFilter);
    prediction.addEventListener('change', applyFilter);
  </script>
</body>
</html>
"""


def render_card(row: dict[str, str], base_dir: Path) -> str:
    annotation = row["annotation"]
    svg_href = relative_href(annotation, base_dir)
    image_href = relative_href(image_path_from_annotation(annotation), base_dir)
    search_text = " ".join(
        [
            row.get("review_id", ""),
            row.get("annotation", ""),
            row.get("room_id", ""),
            row.get("prediction", ""),
            row.get("texts", ""),
            row.get("source_bucket", ""),
        ]
    ).lower()
    shape = parse_json(row.get("shape"))
    bbox = parse_json(row.get("bbox"))
    shape_text = compact_json(shape)
    bbox_text = compact_json(bbox)
    return f"""<article data-prediction="{html.escape(row.get('prediction', ''))}" data-search="{html.escape(search_text)}">
  <div class="preview">
    <iframe src="{html.escape(svg_href)}" loading="lazy" title="{html.escape(row.get('review_id', ''))}"></iframe>
    <div><a href="{html.escape(svg_href)}" target="_blank">Open SVG</a> · <a href="{html.escape(image_href)}" target="_blank">Open PNG</a></div>
  </div>
  <section>
    <div class="actions">
      <span class="pill">{html.escape(row.get('review_id', ''))}</span>
      <span class="pill">gold={html.escape(row.get('gold', ''))}</span>
      <span class="pill">pred={html.escape(row.get('prediction', ''))}</span>
      <span class="pill">source={html.escape(row.get('source_bucket', ''))}</span>
    </div>
    <div class="fields">
      {field('Room ID', row.get('room_id', ''))}
      {field('Confidence', row.get('confidence', ''))}
      {field('Texts', row.get('texts', ''))}
      {field('Shape Bucket', row.get('shape_bucket', ''))}
      {field('BBox', bbox_text)}
      {field('Shape', shape_text)}
      {field('Recommended', row.get('recommended_action', ''))}
      {field('Review Labels', 'accept_typed / keep_room / unclear / exclude')}
      {field('Annotation', row.get('annotation', ''), class_name='path')}
    </div>
  </section>
</article>"""


def field(label: str, value: str, class_name: str = "") -> str:
    cls = f"field {class_name}".strip()
    return f'<div class="{cls}"><strong>{html.escape(label)}</strong><code>{html.escape(str(value))}</code></div>'


def relative_href(path: str, base_dir: Path) -> str:
    if not path:
        return ""
    return os.path.relpath(Path(path), base_dir).replace(os.sep, "/")


def image_path_from_annotation(annotation: str) -> str:
    directory = Path(annotation).parent
    original = directory / "F1_original.png"
    scaled = directory / "F1_scaled.png"
    if original.exists():
        return str(original)
    if scaled.exists():
        return str(scaled)
    return ""


def parse_json(value: str | None) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    main()
