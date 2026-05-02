#!/usr/bin/env python3
"""HTTP sidecar for raster VLM semantic extraction.

Backends:
- mock: deterministic CI/smoke backend.
- qwen3_vl: Qwen3-VL local inference through Hugging Face Transformers.
- transformers_image_text: generic image-text-to-text Transformers backend.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image
import uvicorn

from output_contract import normalize_output, parse_model_json


SCHEMA_VERSION = "raster-vlm-1.0"


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_app(config: dict[str, Any]) -> FastAPI:
    app = FastAPI(title="Raster VLM Backend", version="0.2.0")
    max_request_bytes = int(config.get("max_request_bytes", 8 * 1024 * 1024))
    engine = RasterVlmEngine(config)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "ok": True,
            "backend": engine.backend,
            "model_name": engine.model_name,
            "loaded": engine.loaded,
        }

    @app.post("/analyze_raster")
    async def analyze_raster(request: Request) -> JSONResponse:
        body = await request.body()
        if len(body) > max_request_bytes:
            raise HTTPException(status_code=413, detail="request too large")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"invalid JSON: {exc}") from exc

        try:
            response = engine.analyze(payload)
        except ModelLoadError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ModelInferenceError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return JSONResponse(response)

    return app


class ModelLoadError(RuntimeError):
    pass


class ModelInferenceError(RuntimeError):
    pass


@dataclass
class LoadedModel:
    model: Any
    processor: Any
    torch: Any


class RasterVlmEngine:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.backend = str(config.get("backend", "mock"))
        self.model_name = str(config.get("model_name") or "mock-raster-vlm")
        self.model_path = str(config.get("model_path") or self.model_name)
        self.prompt_template = str(config.get("prompt_template") or default_prompt())
        self.max_new_tokens = int(config.get("max_new_tokens", 768))
        self.temperature = float(config.get("temperature", 0.0))
        self.loaded_model: LoadedModel | None = None
        if devices := config.get("cuda_visible_devices"):
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(devices))
        if config.get("eager_load", False) and self.backend != "mock":
            self._load_model()

    @property
    def loaded(self) -> bool:
        return self.loaded_model is not None

    def analyze(self, payload: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        if self.backend == "mock":
            return self._mock_analyze(payload, started)
        if self.backend in {"qwen3_vl", "transformers_image_text"}:
            return self._transformers_analyze(payload, started)
        raise ModelInferenceError(f"unsupported backend: {self.backend}")

    def _mock_analyze(self, payload: dict[str, Any], started: float) -> dict[str, Any]:
        text_candidates = payload.get("text_candidates") or []
        polylines = payload.get("polylines") or []
        dimensions = []
        for candidate in text_candidates[:16]:
            text = str(candidate.get("content", "")).strip()
            if not any(ch.isdigit() for ch in text):
                continue
            dimensions.append(
                {
                    "raw_text": text,
                    "nominal_value": first_number(text),
                    "tolerance_type": None,
                    "upper_deviation": None,
                    "lower_deviation": None,
                    "geometric_type": None,
                    "datums": [],
                    "roughness": None,
                    "bbox": candidate.get("bbox", [0.0, 0.0, 0.0, 0.0]),
                    "confidence": min(float(candidate.get("confidence", 0.5)) + 0.05, 0.95),
                    "source": "vlm_http_mock",
                }
            )

        semantics = []
        for idx, polyline in enumerate(polylines[:128]):
            length = polyline_length(polyline)
            semantic_type = "hard_wall" if length > 80 else "opening" if length > 25 else "detail_line"
            semantics.append(
                {
                    "target_id": idx,
                    "semantic_type": semantic_type,
                    "confidence": 0.68 if semantic_type == "hard_wall" else 0.58,
                    "source": "vlm_http_mock",
                }
            )

        return self._response(
            started,
            dimensions,
            [],
            semantics,
            mock_scene_graph(semantics, payload.get("primitive_graph")),
            ["mock_backend: replace with a real VLM model for production inference"],
            None,
        )

    def _transformers_analyze(self, payload: dict[str, Any], started: float) -> dict[str, Any]:
        loaded = self._load_model()
        image = decode_thumbnail(payload)
        prompt = self._build_prompt(payload)
        raw_text = self._generate_text(loaded, image, prompt)
        parsed, warnings = parse_model_json(raw_text)
        normalized = normalize_output(parsed, self.backend)
        warnings.extend(normalized.pop("warnings", []))
        return self._response(
            started,
            normalized["dimension_candidates"],
            normalized["symbol_candidates"],
            normalized["semantic_candidates"],
            normalized["scene_graph"],
            warnings,
            summarize_raw(raw_text),
        )

    def _load_model(self) -> LoadedModel:
        if self.loaded_model is not None:
            return self.loaded_model
        try:
            import torch
            from transformers import AutoProcessor, BitsAndBytesConfig
        except Exception as exc:  # pragma: no cover - environment-specific.
            raise ModelLoadError(f"missing model dependencies: {exc}") from exc

        try:
            if self.backend == "qwen3_vl":
                from transformers import Qwen3VLForConditionalGeneration

                model_cls = Qwen3VLForConditionalGeneration
            else:
                from transformers import AutoModelForImageTextToText

                model_cls = AutoModelForImageTextToText

            dtype = getattr(torch, str(self.config.get("torch_dtype", "bfloat16")))
            load_kwargs: dict[str, Any] = {
                "torch_dtype": dtype,
                "device_map": self.config.get("device_map", "auto"),
            }
            if self.config.get("attn_implementation"):
                load_kwargs["attn_implementation"] = self.config["attn_implementation"]
            if self.config.get("load_in_4bit", False):
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=dtype,
                )

            model = model_cls.from_pretrained(self.model_path, **load_kwargs)
            if adapter_path := self.config.get("peft_adapter_path"):
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, str(adapter_path))
            processor = AutoProcessor.from_pretrained(self.model_path)
        except Exception as exc:  # pragma: no cover - requires model weights.
            raise ModelLoadError(f"failed to load {self.model_path}: {exc}") from exc

        self.loaded_model = LoadedModel(model=model, processor=processor, torch=torch)
        return self.loaded_model

    def _generate_text(self, loaded: LoadedModel, image: Image.Image, prompt: str) -> str:
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        processor = loaded.processor
        torch = loaded.torch
        try:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            try:
                from qwen_vl_utils import process_vision_info

                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
            except Exception:
                inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")

            inputs = inputs.to(loaded.model.device)
            generate_kwargs: dict[str, Any] = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": self.temperature > 0,
            }
            if self.temperature > 0:
                generate_kwargs["temperature"] = self.temperature
            with torch.inference_mode():
                generated_ids = loaded.model.generate(**inputs, **generate_kwargs)
            input_len = inputs["input_ids"].shape[-1]
            generated_trimmed = generated_ids[:, input_len:]
            return processor.batch_decode(
                generated_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        except Exception as exc:  # pragma: no cover - requires model weights.
            raise ModelInferenceError(f"model inference failed: {exc}") from exc

    def _build_prompt(self, payload: dict[str, Any]) -> str:
        primitive_graph = summarize_primitive_graph(payload.get("primitive_graph"))
        compact = {
            "image": payload.get("image", {}),
            "text_candidates": (payload.get("text_candidates") or [])[:32],
            "symbol_candidates": (payload.get("symbol_candidates") or [])[:32],
            "polyline_count": len(payload.get("polylines") or []),
            "polyline_samples": (payload.get("polylines") or [])[:24],
            "primitive_graph": primitive_graph,
            "output_requirements": {
                "priority": ["semantic_candidates", "scene_graph", "symbol_candidates", "dimension_candidates"],
                "min_semantic_candidates": 1 if primitive_graph.get("nodes") else 0,
                "semantic_target_id_source": "primitive_graph.nodes[].id",
            },
        }
        return self.prompt_template.replace("{context}", json.dumps(compact, ensure_ascii=False))

    def _response(
        self,
        started: float,
        dimension_candidates: list[dict[str, Any]],
        symbol_candidates: list[dict[str, Any]],
        semantic_candidates: list[dict[str, Any]],
        scene_graph: dict[str, Any] | None,
        warnings: list[str],
        raw_output_summary: str | None,
    ) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "model_info": {
                "backend": "http",
                "model_name": self.model_name,
                "latency_ms": int((time.perf_counter() - started) * 1000),
            },
            "dimension_candidates": dimension_candidates,
            "symbol_candidates": symbol_candidates,
            "semantic_candidates": semantic_candidates,
            "scene_graph": scene_graph,
            "warnings": warnings,
            "raw_output_summary": raw_output_summary,
        }


def default_prompt() -> str:
    return (
        "You are analyzing raster CAD and engineering drawings for a candidate-only extraction pipeline. "
        "Return one strict JSON object only, with no markdown. Do not infer accepted truth; emit candidates. "
        "Schema, preserving this field order: {\"semantic_candidates\":[],\"scene_graph\":{\"nodes\":[],\"edges\":[]},\"symbol_candidates\":[],\"dimension_candidates\":[],\"warnings\":[]}. "
        "Dimension fields: raw_text, nominal_value, tolerance_type, upper_deviation, lower_deviation, geometric_type, datums, roughness, bbox, confidence, source. "
        "Symbol fields: symbol_type, confidence, bbox, rotation. "
        "Semantic fields: target_id, semantic_type, confidence, source. "
        "Scene graph node fields: id, semantic_type, primitive_id. Scene graph edge fields: source, target, relation. "
        "Allowed semantic_type examples: hard_wall, partition_wall, opening, door, window, dimension_line, centerline, datum, detail_line. "
        "If primitive_graph.nodes is non-empty, semantic_candidates must not be empty; classify visible graph nodes and use target_id values from primitive_graph node ids. "
        "Mirror semantic candidates into scene_graph.nodes. Emit semantic_candidates and scene_graph before any empty dimension or symbol arrays. "
        "Use source=\"vlm_http\" and confidence in [0,1]. Preserve uncertainty in warnings. "
        "Context JSON: {context}"
    )


def decode_thumbnail(payload: dict[str, Any]) -> Image.Image:
    encoded = payload.get("thumbnail_png_base64")
    if not encoded:
        width = int((payload.get("image") or {}).get("width", 32) or 32)
        height = int((payload.get("image") or {}).get("height", 32) or 32)
        return Image.new("L", (min(width, 1024), min(height, 1024)), color=255).convert("RGB")
    raw = base64.b64decode(encoded)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def summarize_primitive_graph(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"nodes": [], "edges": []}
    nodes = value.get("nodes") if isinstance(value.get("nodes"), list) else []
    edges = value.get("edges") if isinstance(value.get("edges"), list) else []
    return {"nodes": nodes[:64], "edges": edges[:128], "truncated": len(nodes) > 64 or len(edges) > 128}


def mock_scene_graph(semantics: list[dict[str, Any]], primitive_graph: Any) -> dict[str, Any] | None:
    if not isinstance(primitive_graph, dict):
        return None
    semantic_by_id = {
        int(item["target_id"]): str(item["semantic_type"])
        for item in semantics
        if "target_id" in item and "semantic_type" in item
    }
    nodes = [
        {"id": target_id, "semantic_type": semantic_type, "primitive_id": target_id}
        for target_id, semantic_type in semantic_by_id.items()
    ]
    edges = []
    for edge in primitive_graph.get("edges", []):
        if not isinstance(edge, dict):
            continue
        try:
            source = int(edge.get("source"))
            target = int(edge.get("target"))
        except (TypeError, ValueError):
            continue
        if source not in semantic_by_id or target not in semantic_by_id:
            continue
        edges.append({"source": source, "target": target, "relation": str(edge.get("relation", "related_to"))})
    return {"nodes": nodes, "edges": edges}


def summarize_raw(raw_text: str) -> str:
    return raw_text.replace("\n", " ")[:512]


def first_number(text: str) -> float | None:
    token = ""
    for ch in text:
        if ch.isdigit() or ch in ".-":
            token += ch
        elif token:
            break
    try:
        return float(token) if token else None
    except ValueError:
        return None


def polyline_length(polyline: list[list[float]]) -> float:
    total = 0.0
    for left, right in zip(polyline, polyline[1:]):
        dx = float(right[0]) - float(left[0])
        dy = float(right[1]) - float(left[1])
        total += (dx * dx + dy * dy) ** 0.5
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vlm/default.json")
    args = parser.parse_args()
    config = load_config(args.config)
    app = build_app(config)
    uvicorn.run(app, host=config.get("host", "127.0.0.1"), port=int(config.get("port", 8765)))


if __name__ == "__main__":
    main()
