# P308 Submission Static Check

- Source: `reports/vlm/p308_generic_submission_manuscript.tex`
- Static pass: `true`
- Decision: `ready_for_template_insertion_pending_latex_engine`
- Compile status: `not_attempted_no_latex_engine`
- Legacy P268 check: `not_applicable_to_p308` because it is hard-coded for old P265/P266 metrics.

## Required Metrics
- `0.962907`: `present`
- `0.982498`: `present`
- `0.994900`: `present`
- `0.872665`: `present`
- `0.929782`: `present`
- `0.000000`: `present`
- `0.729861`: `present`
- `0.729524`: `present`
- `0.485651`: `present`
- `0.159075`: `present`

## Required Boundary Phrases
- `SVG/contract normalized-candidate`: `present`
- `internal locked fine-threshold audit`: `present`
- `not external validation`: `present`
- `external/source-heldout generalization remains blocked`: `present`
- `not source-heldout or pure-raster evidence`: `present`
- `not as the headline model result`: `present`
- `not adopted`: `present`

## Structural Checks
- `has_documentclass`: `true`
- `has_abstract`: `true`
- `has_main_results_table`: `true`
- `has_symbol_table`: `true`
- `has_e2e_inventory_table`: `true`
- `has_external_generalization_table`: `true`
- `has_figure_plan`: `true`
- `uses_old_p265_headline_node_macro_f1`: `false`
- `uses_old_p265_headline_relation_f1`: `false`

## Forbidden Claim Hits
- None.

## Next Step
- Insert `reports/vlm/p308_generic_submission_manuscript.tex` into a target venue template or install/use a LaTeX engine, then rerun compile validation.
