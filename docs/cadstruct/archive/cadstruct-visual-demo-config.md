# CadStruct-MoE Visual Demo Config

The CubiCasa visual review pack is driven by:

`configs/vlm/cubicasa_visual_demo.json`

Use that file for visualization policy changes instead of editing `scripts/vlm/render_scene_graph_visual_demo.py`.

Config-owned items:

- input/output defaults: prediction JSONL, output directory, sample limit
- dataset policy: selected `source_dataset`, sample id prefix, manifest text
- rendering styles: family colors, legend labels, family ordering, opacity, stroke width
- label policy: label priority, spacing, max label length
- bbox audit policy: suspicious boundary semantic types and thresholds
- source canvas policy: whether to prefer SVG `viewBox`, how to fit SVG coordinates to `F1_scaled.png`, outside-canvas tolerance
- report text: review pack title, coverage text, notes, boundary-overlay audit explanation
- output naming: review pack directory and paper-candidate figure stem

The renderer keeps fallback defaults only so the script can fail gracefully if a config key is missing. The paper/demo behavior should be changed in config.

For CubiCasa5K review figures, `canvas_policy.svg_image_fit` is set to `stretch`. The `F1_scaled.png` raster can have a different aspect ratio from the SVG root viewBox, and using an SVG `meet` fit adds artificial horizontal or vertical padding.

Some CubiCasa samples also pair a scanned/photographic `F1_scaled.png` with a clean `model.svg` annotation that is not pixel-aligned to the scan. For advisor-facing overlay figures, `background_policy.mode` is therefore `annotation_render`: the overlay is drawn on an `rsvg-convert` rendering of `model.svg`, while the original CubiCasa raster is preserved as `input_reference.png`. This keeps the qualitative recognition overlay in the same coordinate frame as the scene graph and avoids claiming pixel-perfect alignment to noisy scans.
