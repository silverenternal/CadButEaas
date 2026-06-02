# P305c External Generalization Claim Decision

Status: `external_generalization_claim_blocked`.

## Decision

Do not make external/source-heldout generalization claims for P301 or for RoomSpace/SymbolFixture/TextDimension.

Keep P301 claims scoped to reviewed SVG/contract normalized-candidate metrics and internal locked fine-threshold relation audits.

## Basis

- P305a preflight found the existing external-generalization decision still blocked and the human-gold manifest pending.
- P305b true source-heldout batch evaluated only WallOpening and left RoomSpace, SymbolFixture, and TextDimension blocked.
- WallOpening source transfer metrics are weak and do not support zero-shot source transfer.
- `relation_no_repair_heldout_scorer_v1` is internal record-heldout within locked dev, not external source validation.

## Allowed Language

- P301 is a reviewed SVG/contract experiment-line candidate under normalized-candidate evaluation.
- P301 relation F1 is an internal locked fine-threshold audit, not external validation.
- Relation scorer has supporting internal record-heldout robustness within the locked benchmark.
- WallOpening source-transfer evidence is weak and does not support a zero-shot source-transfer claim.

## Forbidden Language

- Do not call P301 externally validated.
- Do not claim source-heldout generalization for RoomSpace, SymbolFixture, or TextDimension.
- Do not use `relation_no_repair_heldout_scorer_v1` as external source validation.
- Do not claim runtime raster generalization from SVG/contract metrics.

## Unblockers

- Collect or unlock non-CubiCasa room-space labels for source-heldout RoomSpace validation.
- Collect FloorPlanCAD/internal symbol fixture locked labels for SymbolFixture validation.
- Collect FloorPlanCAD/internal OCR/text locked labels for TextDimension validation.
- Refresh external human-gold annotations from `external_human_gold_manifest_v3`.
