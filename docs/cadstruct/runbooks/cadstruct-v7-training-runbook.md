# CadStruct-MoE v7 Training Runbook

## Scope

v7 separates model retraining from postprocess cleanup:

- `boundary_geometry_refiner_v7` is a model-side geometry-output refiner.
- `symbol_fixture_expert_v13` is a trained candidate model, but it is adopted only if locked guards pass.
- `postprocess_v7` remains a separate traceable cleanup stream.

## Commands

```bash
uv run python scripts/vlm/train_boundary_geometry_refiner_v7.py
uv run python scripts/vlm/build_symbol_fixture_hard_cases_v13.py
uv run python scripts/vlm/train_symbol_fixture_expert_v13.py
uv run python scripts/vlm/build_real_upstream_model_predictions_v7.py
uv run python scripts/vlm/apply_visual_postprocess_v7.py
uv run python scripts/vlm/render_scene_graph_visual_demo.py --predictions reports/vlm/real_upstream_model_predictions_model_v7.jsonl --output-dir reports/vlm/visual_demo_model_v7_model --limit 5
uv run python scripts/vlm/render_scene_graph_visual_demo.py --predictions reports/vlm/real_upstream_model_postprocessed_predictions_v7.jsonl --output-dir reports/vlm/visual_demo_model_v7_postprocessed --limit 5
uv run python scripts/vlm/build_visual_defect_ablation_v7.py
```

Use `--smoke` only for fast integration checks:

```bash
uv run python scripts/vlm/train_boundary_geometry_refiner_v7.py --smoke
```

Smoke results must not be reported as full model evidence.

## Current Result

- Boundary v7 full locked: accuracy `1.0`, macro-F1 `1.0`, train/locked overlap `0`.
- SymbolFixture v13 locked macro-F1: `0.759348`; rejected because it does not beat v11 guards.
- Visual defects: model_v7 `unsupported_wall=0`, `empty_symbol=1`; postprocess_v7 defect counts are empty.

## Claim Boundary

The v7 model stream adopts only full locked accepted model components. The appliance/equipment fix is postprocess evidence, not model recognition evidence.
