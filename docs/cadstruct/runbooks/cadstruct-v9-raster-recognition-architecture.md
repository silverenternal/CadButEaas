# CadStruct v9 Raster Recognition Architecture

v9 changes the raster path from connected-component proposal boxes to semantic segmentation, heatmaps, and mask vectorization.
SVG/parser geometry is allowed only as offline CubiCasa gold labels and locked evaluation gold; it is not used as inference-time candidate geometry.

## Components

- T1 raster labels: `datasets/raster_segmentation_v9/*.jsonl`
- T2 CubiCasa-style segmentation baseline: `reports/vlm/raster_segmentation_baseline_v9_eval.json`
- T3 MuraNet-lite multitask branch: `reports/vlm/muranet_lite_v9_eval.json`
- T4 room mask-to-polygon vectorization: `reports/vlm/room_polygon_vectorization_v9_eval.json`
- T5 wall/opening/window vectorization: `reports/vlm/wall_opening_vectorization_v9_eval.json`
- T6 raster text detection and OCR audit: `reports/vlm/text_detection_ocr_v9_eval.json`
- T7 model stream decision: `reports/vlm/model_v9_raster_adoption_decisions.json`

## Locked Result

- adopted: `False`
- locked mean IoU: `0.056503`
- locked mean F1: `0.084395`

External research basis: CubiCasa5K official segmentation labels, MuraNet multitask segmentation+detection, Raster-to-Graph/Raster2Seq graph or polygon sequence prediction, and SAM-style exploratory segmentation. The current repo keeps sequence/foundation-model work exploratory until dependencies, weights, and locked metrics are pinned.
