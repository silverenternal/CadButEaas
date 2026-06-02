# CadStruct v16 Frontier Structured Baseline Decision

v15 is a valid raster-only negative baseline, but its mask plus connected-component vectorizer is the wrong output form for floorplans.

Decision:
- Reimplement a HEAT-style light boundary graph expert locally, because it directly targets junctions and edges.
- Use Raster-to-Graph as the target research direction and compatibility reference, but do not block local progress on importing its full training stack.
- Use Floor-SP style constraints after boundary prediction to stabilize room polygons.
- Use FloorPlanCAD for symbol detector pretraining where its raster/detection export is locally available.

Claim boundary: CubiCasa/FloorPlanCAD labels are offline supervision only. v16 model-credit inference must consume raster images only.
