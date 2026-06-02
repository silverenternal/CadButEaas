# CadStruct v8 Raster E2E Runbook

This runbook reproduces the v8 source-mode separation work.

## Commands

```bash
uv run python scripts/vlm/audit_raster_e2e_assets_v8.py
```

```bash
uv run python scripts/vlm/build_raster_detection_dataset_v8.py
```

```bash
uv run python scripts/vlm/train_raster_candidate_detector_v8.py
```

```bash
uv run python scripts/vlm/build_symbol_visual_evidence_dataset_v8.py
```

```bash
uv run python scripts/vlm/train_symbol_visual_evidence_v8.py
```

```bash
uv run python scripts/vlm/build_raster_e2e_predictions_v8.py
```

```bash
uv run python scripts/vlm/build_hybrid_visual_model_v8.py
```

```bash
uv run python scripts/vlm/render_scene_graph_visual_demo.py --predictions reports/vlm/hybrid_visual_model_v8_predictions.jsonl --output-dir reports/vlm/visual_demo_hybrid_v8_model --source-dataset cubicasa5k --limit 5
```

```bash
uv run python scripts/vlm/build_visual_model_v8_comparison.py
```

```bash
uv run python scripts/vlm/audit_raster_e2e_defects_v8.py
```

## Claim Boundary

Pure raster E2E is not adopted because `raster_candidate_detector_v8` failed locked metrics.
Hybrid v8 uses SVG/parser candidate geometry plus adopted raster crop visual-evidence review flags.
Postprocess cleanup remains separate from model recognition credit.
