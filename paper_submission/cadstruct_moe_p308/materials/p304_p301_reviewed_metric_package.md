# P304 P301 Reviewed Metric Package

Status: `packaged_reviewed_svg_contract_candidate`.

## Main Candidate

P301 is the reviewed SVG/contract experiment-line candidate after P303 model-code review.

| Metric | Value | Role |
| --- | ---: | --- |
| Node macro-F1 | 0.962907 | SVG/contract normalized-candidate audit |
| Node accuracy | 0.982498 | SVG/contract normalized-candidate audit |
| Relation precision | 0.994900 | Internal locked fine-threshold audit |
| Relation recall | 0.872665 | Internal locked fine-threshold audit |
| Relation F1 | 0.929782 | Internal locked fine-threshold audit |
| Invalid graph rate | 0.000000 | Internal locked fine-threshold audit |

P301 changes 77 locked symbols on top of P297: 54 `appliance->equipment`, 20 `stair->equipment`, and 3 `column->equipment`. It improves node macro-F1 by +0.0227 pp over P297 while preserving P297 relation F1 by keeping relation confidence unchanged when symbol labels are relabeled.

## Evidence Chain

| Step | Role | Status | Boundary |
| --- | --- | --- | --- |
| P291 | Clean train/dev-selected node-symbol baseline | passed | SVG/contract normalized-candidate |
| P292 | Relation fine-threshold audit | passed | Internal locked threshold audit |
| P294 | Risk-clamped column rescue | passed | SVG/contract plus internal relation audit |
| P295 | Risk-clamped shower rescue | passed | SVG/contract plus internal relation audit |
| P296 | Bathtub refresh rescue | passed | SVG/contract plus internal relation audit |
| P297 | Strong pre-residual sink baseline | passed | SVG/contract plus internal relation audit |
| P298 | Tiny overlay rejection | rejected | Internal diagnostic |
| P299 | Confidence-changing residual relabeler | rejected | Internal diagnostic |
| P300 | Confidence-preservation diagnostic | diagnostic | Less conservative than P301 |
| P301 | Reviewed promoted candidate | passed | SVG/contract plus internal relation audit |
| P302 | Pairwise micro-rescue | rejected | Locked-negative diagnostic |
| P303 | Model-code review | passed with restrictions | Packaging gate |

## P303 Restrictions

- Do not present P301 relation F1 as external validation.
- Do not present P301 or P291-P302 SVG/contract metrics as runtime raster detector performance.
- State that fine relation thresholds were selected on locked records and therefore are internal locked threshold audits.
- Keep `expected_json` and gold-compatible ID reconstruction offline-only.
- Do not change prediction sources without `candidate_id`/`record_index` alignment assertions.

## Reviewer-Safe Paragraph

P301 is a reviewed SVG/contract experiment-line candidate, not a runtime raster detector result. It applies a conservative residual relabeler on top of P297 and preserves relation confidence when labels change. On the locked SVG/contract audit it raises node macro-F1 to 0.962907 and keeps relation F1 at 0.929782, where the relation value is an internal locked fine-threshold audit rather than external validation. P302 shows that further pairwise micro-rescue is not stable enough to promote.
