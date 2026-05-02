#!/usr/bin/env python3
"""Shared CadStruct SFT utilities.

Keep encoding and budget accounting out of the trainer so profiling, training,
and future evaluators use the same rules.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class EncodedSampleStats:
    input_tokens: int
    supervised_tokens: int
    vision_tiles: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class SftBudget:
    max_length: int
    max_vision_tiles: int = 0
    skip_at_max_length: bool = False

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SftBudget":
        return cls(
            max_length=int(config.get("max_length", 12288)),
            max_vision_tiles=int(config.get("max_vision_tiles", 0) or 0),
            skip_at_max_length=bool(config.get("skip_at_max_length", False)),
        )

    def skip_reason(self, stats: EncodedSampleStats) -> str | None:
        if self.max_vision_tiles > 0 and stats.vision_tiles > self.max_vision_tiles:
            return "vision_tiles"
        if self.skip_at_max_length and stats.input_tokens >= self.max_length:
            return "max_length"
        return None

    def to_dict(self) -> dict[str, int | bool]:
        return asdict(self)


class SftJsonlDataset:
    def __init__(self, path: Path, limit: int | None = None) -> None:
        self.rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                self.rows.append(json.loads(line))
                if limit is not None and len(self.rows) >= limit:
                    break

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def encode_sft_row(
    processor: Any, row: dict[str, Any], max_length: int, max_image_side: int = 0
) -> dict[str, Any]:
    messages = row["messages"]
    image = Image.open(row["image"]).convert("RGB")
    if max_image_side > 0 and max(image.size) > max_image_side:
        image.thumbnail((max_image_side, max_image_side), Image.Resampling.LANCZOS)
    prompt_text = processor.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
    full_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_inputs = processor(
        text=[prompt_text],
        images=[image],
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    inputs = processor(
        text=[full_text],
        images=[image],
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    labels = inputs["input_ids"].clone()
    prompt_len = min(prompt_inputs["input_ids"].shape[-1], labels.shape[-1])
    labels[:, :prompt_len] = -100
    pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        labels[inputs["input_ids"] == pad_token_id] = -100
    inputs["labels"] = labels
    return inputs


def encoded_sample_stats(encoded: dict[str, Any]) -> EncodedSampleStats:
    return EncodedSampleStats(
        input_tokens=int(encoded["input_ids"].shape[-1]),
        supervised_tokens=int((encoded["labels"] != -100).sum().item()),
        vision_tiles=int(encoded["pixel_values"].shape[0]) if "pixel_values" in encoded else 0,
    )


def move_batch_to_device(encoded: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in encoded.items()}


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())
