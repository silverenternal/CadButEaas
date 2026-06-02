# CadStruct Real Image Model Architecture v3

Date: 2026-05-07

## Decision

Use a hybrid plan next: keep the current parser-candidate MoE for auditable scene-graph output, but add a minimal raster detector/segmentation prototype for the failure modes that parser/SVG candidate geometry cannot answer.

Do not replace the whole MoE stack immediately. The current expert models already provide typed labels, relations, and audit traces. The missing capability is pixel-level proposal generation and rejection, especially for wall/room masks, text candidates, and symbol false positives.

## Options

| Option | Benefit | Risk | Decision |
|---|---|---|---|
| Continue candidate-classification MoE only | Fastest, auditable, uses existing reports | Cannot claim direct scan-image detection; candidate geometry errors leak into outputs | Keep as current production/research baseline |
| Add raster detector/segmentation heads | Can answer advisor's input-image recognition question more directly | Needs masks/boxes, training environment, and new eval protocol | Start minimal prototype |
| Replace all experts with one end-to-end model | Simpler story if it works | High data cost, less interpretability, likely weaker long-tail relations | Do not do now |

## Minimal Prototype

1. Wall/room segmentation head: predict wall/opening/room masks from CubiCasa rasterized SVG supervision.
2. Symbol/text detector head: predict compact symbol/text boxes with a reject/no-object class.
3. Bridge to MoE: use raster detections as proposal candidates, then keep expert classifiers and graph fusion for labels/relations.
4. Evaluation: report raw detector proposal recall/precision separately from MoE classification and final scene-graph metrics.

## Required Data

- CubiCasa raster images and SVG-derived masks.
- Hard cases from `datasets/cadstruct_hard_cases_v3/`.
- Split policy that keeps visual-demo review samples out of locked-test claims.
- Human review for external/wild scans before claiming real-world generalization.

## Current Constraint

The local system Python environment does not expose `torch`, `sklearn`, or `PIL`. Visual validation was run through `uv --with pillow`. Model retraining requires a pinned training environment before checkpoints can be regenerated.

<!-- CADSTRUCT_V8_RASTER_E2E_START -->
## CadStruct v8 Raster E2E Claim Boundary

| Stream | Source mode | Status | Claim |
| --- | --- | --- | --- |
| v7_svg_candidate | SVG/parser candidates + model refiners | baseline | Visualizes current saved-model output over parser geometry. |
| v8_raster_e2e | image-only raster detector | rejected | Detector locked macro-F1=0.007207; no pure raster E2E success claim. |
| v8_hybrid | SVG candidates + raster crop evidence | available | Uses adopted components: symbol_visual_evidence_v8. |
| postprocess_v7 | postprocess over model stream | separate | Cleanup events are not model-recognition credit. |

- `raster_candidate_detector_v8`: adopted=False, locked macro-F1=0.007207. Pure raster E2E is therefore rejected/exploratory.
- `symbol_visual_evidence_v8`: adopted=True, locked reject precision=0.974989, recall=1.0. This is crop evidence for review flags, not geometry detection.
- Boundary v7 remains a model-side geometry-output refiner over SVG/parser candidate geometry.
- `empty_symbol` cleanup from v7 postprocess remains postprocess-only; v8 hybrid additionally marks low visual-evidence symbol nodes for review when the crop model fires.
- Visual outputs: `reports/vlm/visual_demo_v8_comparison/index.html`; metrics: `reports/vlm/real_model_locked_eval_v8.json`, `reports/vlm/raster_e2e_defect_audit_v8.json`.
<!-- CADSTRUCT_V8_RASTER_E2E_END -->
