# CadStruct Image-Only MoE Roadmap v15

v15 enforces the real project contract: the MoE front end receives only a raster floorplan image. CubiCasa SVG/parser geometry is used only as offline supervision and locked evaluation gold.

Current status:
- Image-only adopted: `False`
- Proposal mean F1: `0.080493`
- Parser-assisted v13/v14 remains oracle/debug only, not model-credit evidence.

The main remaining bottleneck is raster proposal generation. Expert models can only improve labels after credible wall/room/symbol/text proposals exist.
