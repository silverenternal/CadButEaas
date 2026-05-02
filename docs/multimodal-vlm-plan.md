# Multimodal Raster VLM Plan

The raster pipeline now has a replaceable semantic backend:

- `disabled`: no model candidates.
- `heuristic`: default local OCR/dimension/rule candidates.
- `http`: POSTs schema-compatible input to a Python sidecar and falls back to `heuristic` on failure.

Rust contract:

- Request: `RasterVlmInput` in `crates/vectorize/src/semantic/schema.rs`.
- Response: `RasterVlmOutput` with `dimension_candidates`, `symbol_candidates`, `semantic_candidates`, `warnings`, and `model_info`.
- Report metadata: `vlm_backend`, `vlm_model_name`, `vlm_latency_ms`, `vlm_fallback_reason`, `vlm_warnings`.

Rules:

- Model output is candidate-only and never overwrites geometry.
- Every candidate keeps `source` and `confidence`.
- Real model tests should be ignored by default; CI should use the mock HTTP backend or the heuristic backend.
- Python model dependencies stay in `scripts/vlm` and are not part of the Rust build.

Model plan:

1. Run the mock sidecar and verify `/analyze_raster`.
2. Generate synthetic data with `scripts/vlm/generate_dataset.py`.
3. Use `Qwen/Qwen3-VL-8B-Instruct` for the first real smoke run.
4. Use `Qwen/Qwen3-VL-32B-Instruct` as the paper-grade primary local model on the 96GB GPU.
5. Compare against InternVL3.5, GLM-4.6V, and Kimi-VL-A3B-Thinking when their model loaders are compatible with the local Transformers release.
6. Use `configs/vlm/lora_smoke.json` for a QLoRA smoke run after the Qwen3-VL inference path is stable.

Config matrix:

| Role | Config |
|------|--------|
| Mock CI | `configs/vlm/default.json` |
| Qwen3-VL smoke | `configs/vlm/qwen3_vl_8b_smoke.json` |
| Qwen3-VL paper model | `configs/vlm/qwen3_vl_32b_paper.json` |
| InternVL baseline | `configs/vlm/internvl3_5_baseline.json` |
| GLM baseline | `configs/vlm/glm4_6v_baseline.json` |
| Kimi efficiency baseline | `configs/vlm/kimi_vl_efficiency_baseline.json` |

Known limitations:

- The mock backend is deterministic and does not inspect image pixels semantically.
- QLoRA training is currently a guarded smoke entry point until Qwen3-VL inference and evaluation are stable.
- The HTTP client supports plain `http://` sidecar endpoints; deploy TLS at a proxy layer if needed.
- Baseline configs may require model-specific trust-remote-code or processor adjustments depending on upstream model cards.

Current smoke result:

- `Qwen/Qwen3-VL-8B-Instruct` downloaded and loaded on GPU0.
- Observed VRAM after load: about 17.6 GiB.
- Synthetic v2 data now generates 200 train, 30 dev, and 8 smoke samples with partition wall, door, window, and centerline variants.
- Mock 8-sample smoke: JSON success 1.0, dimension hit 1.0, semantic hit 0.5.
- Qwen3-VL-8B 4-sample smoke: JSON success 1.0, dimension hit 1.0, semantic hit 0.25, median latency about 4.2s after model load.
- Current finding: Qwen3-VL-8B reliably follows numeric dimension candidates, but door/window/centerline subclass extraction needs stronger prompts, better visual labels, or rule/VLM fusion.
- InternVL3.5-14B local weights downloaded to `models/vlm/internvl3_5_14b_hf` and 1-sample smoke passed JSON/dimension extraction. It did not emit semantic candidates on the first sample, which confirms the need for CadStruct-VL SFT and graph supervision.

External datasets:

- `datasets/external/floorplancad`: complete FloorPlanCAD snapshot from Hugging Face; about 465 MiB, 5,308 PNG files plus metadata.
- `datasets/external/cvc_fp_figshare`: complete CVC-FP Figshare package; about 1.1 GiB after unpacking, 5,408 PNG files and 74,592 shapefile-sidecar annotation files.
- `datasets/external/cubicasa5k_hf`: small Hugging Face instruction mirror; kept only as metadata/provenance, not as the primary image dataset.
- `datasets/external/cubicasa5k_zenodo/cubicasa5k.zip`: partial official CubiCasa5K Zenodo download, about 96 MiB of 5.09 GiB. Resume with `curl -L -C - -o datasets/external/cubicasa5k_zenodo/cubicasa5k.zip 'https://zenodo.org/records/2613548/files/cubicasa5k.zip?download=1'`.
