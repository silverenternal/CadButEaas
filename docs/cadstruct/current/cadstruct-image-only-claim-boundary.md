# CadStruct Image-Only Claim Boundary

The deployable CadStruct-MoE inference input is a non-SVG raster floor-plan image, such as PNG, JPG, or a rasterized PDF page. The model must not receive CubiCasa SVG geometry, parser candidates, expected JSON candidates, parser raw labels, or annotation-derived recovery candidates at inference time.

## Valid For Model-Credit Metrics

- Raster image tensor input.
- Model-predicted masks, heatmaps, boxes, polygons, OCR boxes, and relation proposals.
- MoE expert decisions computed from raster proposals, crops, mask statistics, and image-only topology.
- Source rows passing `configs/vlm/image_only_moe_contract_v1.json`.

## Valid Only For Training Or Evaluation

- CubiCasa SVG annotations rasterized into offline masks, boxes, polygons, or locked gold.
- SVG-derived gold used after inference for matching and scoring.
- Parser-assisted v13/v14 outputs used as oracle/debug upper bound.

## Not Valid As Image-Only Model Output

- `source_mode=real_upstream_saved_model_predictions`.
- `proposal_source=svg_candidate_geometry`.
- `raw_label` or `base_raw_label` copied from parser/SVG candidates.
- Missing room/symbol/text recovery from SVG annotation files.
- Text-aware gold adapters reported as normal model output.

The current high visual scores from the v13/v14 stream must therefore be described as parser-assisted upper-bound/debug evidence, not as true raster end-to-end recognition.
