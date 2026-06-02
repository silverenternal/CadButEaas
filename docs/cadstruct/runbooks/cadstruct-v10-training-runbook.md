# CadStruct v10 Training Runbook

```bash
uv run python scripts/vlm/v10_raster_pipeline.py run-all --epochs 1 --max-train 128 --max-eval 56
uv run python -m json.tool reports/vlm/model_v10_raster_locked_eval.json
```

Outputs are under `datasets/raster_supervision_v10/`, `checkpoints/*_v10/`, and `reports/vlm/*v10*`.
