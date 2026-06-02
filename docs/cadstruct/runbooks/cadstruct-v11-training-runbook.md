# CadStruct v11 Training Runbook

```bash
uv run python scripts/vlm/v11_frontier_pipeline.py run-all --limit 520 --epochs 1 --max-train 96 --max-eval 40 --train-size 128 --batch-size 4
uv run python -m json.tool reports/vlm/model_v11_adoption_decisions.json
```

The bounded run is intended to validate the complete pipeline and produce honest locked evidence. Full adoption requires official CubiCasa reproduction or stronger edge/polygon/text branches.
