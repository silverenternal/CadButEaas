# Symbol Frontend Current State

Updated: 2026-05-12

## Current Best

The best current page-level locked result is still `v45_quality_policy`:

- report: `reports/vlm/symbol_visual_box_refiner_v45_quality_policy_page_locked_eval.json`
- matched: `7543`
- recall: `0.703966`
- precision: `0.023400`
- tiny IoU recall: `0.331199`
- small IoU recall: `0.771812`

This is a real improvement over v44, mainly because quality-score routing stopped hard-denying tiny/small/shower candidates.

For detector-side baselines, the best reusable compact page result currently remains `symbol_yolov8n_seg_rect_v27`:

- report: `reports/vlm/symbol_yolov8n_seg_rect_v27_page_eval.json`
- matched: `1789`
- recall: `0.735609`
- precision: `0.093095`
- candidate inflation: `7.901727`

This is materially better than the page-level refiner in precision and candidate inflation, so it is the right detector-side comparison point for P0-22.

## Negative Result

`v46_enhanced_features` improves crop-row diagnostics but fails the page gate:

- crop policy recall: `0.59275 -> 0.63575`
- page matched: `7543 -> 7534`
- page recall: `0.703966 -> 0.703126`

Do not continue hand-crafted ExtraTrees feature tweaking.

`symbol_detector_frontier_yolo_v47` smoke is also a negative result:

- smoke matched: `0`
- smoke precision: `0.0`
- smoke recall: `0.0`
- smoke candidate inflation: `0.038388`

The filtered frontier detector view did not produce any usable signal after a smoke train/eval pass. Do not treat it as a viable main path yet.

## Code Review Findings

- `apply_symbol_visual_box_refiner_v45_quality_page.py` still has hardcoded v44 baselines in stage gates.
- `audit_symbol_visual_refiner_v43_page_failures.py` reports any-candidate-hit attribution, not official one-to-one page matched.
- Full page prediction files are about `107MB` per run and are duplicated across v41-v46.
- Quality labels are trained from the same refiner's train-set predictions, so calibration may be optimistic.
- Precision remains around `0.0234`, so bbox refinement alone is not the system bottleneck.
- `symbol_detector_frontier_yolo_v47` currently fails to produce nonzero smoke matches, so detector-side training still needs data/label/training-formula review before suppression work can matter.

## Next Order

1. P0-20: harden eval/audit/artifact pipeline.
2. P0-21: audit and register dedicated symbol spotting datasets.
3. P0-22: build a stronger detector/localization/suppression baseline.

`P1-19-lightweight-cnn-crop-regressor-v47` is paused until P0-20 and P0-21 are complete.

## Dataset Leads

- FloorPlanCAD: local assets exist and Hugging Face/FiftyOne lists rasterizations and 30 categories.
- CADSpotting / LS-CAD: recent panoptic symbol spotting benchmark; verify data availability.
- ArchCAD-400K: large architectural CAD symbol spotting dataset; verify availability and license.
- SESYD: synthetic floor-plan symbol spotting data; useful for small pretraining or sanity checks.
- ArchNetv2 data is not shared, but its multiscale/attention detector design is relevant.

## P0-22 Snapshot

- `reports/vlm/symbol_detector_baseline_comparison_v47.json` records the current reusable detector baseline as `symbol_yolov8n_seg_rect_v27`.
- `reports/vlm/symbol_detector_frontier_yolo_v47_smoke_train3_eval.json` shows the new frontier detector smoke is still all zero on official IoU match metrics.
- `reports/vlm/symbol_detector_external_data_readiness_v47.json` already marks FloorPlanCAD as trainable and CubiCasa/few others as research-only for now.

## Direction

The 0.98 target requires detector, localization, duplicate/support suppression, and type experts to improve together. The current evidence does not support spending the next iteration on another isolated crop refiner.
