# P301 Conservative Relation-Confidence-Preserved Rescue

## Decision
- Status: `passed_conservative_relation_confidence_preserved_residual_rescue_candidate`.
- Selected dev policy: `et_800_leaf2_s20260532` threshold `0.75` margin `0.0`.
- Locked symbol macro-F1 delta vs P297: `+0.0730 pp`.
- Locked end-to-end node macro-F1 delta vs P297: `+0.0227 pp`.
- Locked relation F1 delta vs P297: `+0.0000 pp`.
- Changed locked symbols: `77`.

## Why This Is Safer Than P300
- Selection uses smallest dev change count within `0.03` pp of the best dev macro-F1 gain.
- Dev selection requires non-regression for every symbol label.
- Locked audit keeps all tracked symbol F1 labels non-regressing while preserving P297 relation F1.

## Claim Boundary
- SVG/contract normalized-candidate symbol classification plus internal locked fine-relation audit.
- Not raster detector performance and not external validation.
