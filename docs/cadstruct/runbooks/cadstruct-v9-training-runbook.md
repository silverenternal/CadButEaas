# CadStruct v9 Training Runbook

Run the bounded local pipeline:

```bash
uv run python scripts/vlm/build_cubicasa_raster_label_tensors_v9.py --limit 0
uv run python scripts/vlm/train_raster_segmentation_baseline_v9.py --epochs 2 --max-train 320 --max-eval 80
uv run python scripts/vlm/train_muranet_lite_v9.py --epochs 2 --max-train 320 --max-eval 80
uv run python scripts/vlm/vectorize_room_polygons_v9.py
uv run python scripts/vlm/vectorize_wall_opening_v9.py
uv run python scripts/vlm/train_text_detection_v9.py
uv run python scripts/vlm/build_model_v9_raster_scene_graph.py
uv run python scripts/vlm/evaluate_model_v9_raster_locked.py
```

Visual review:

```bash
uv run python scripts/vlm/render_raster_model_v9_review_pack.py
```

The generated visual pages are under `reports/vlm/visual_demo_model_v9_raster/`, `reports/vlm/visual_demo_v9_comparison/`, and `reports/vlm/visual_demo_v9_failure_gallery/`.
