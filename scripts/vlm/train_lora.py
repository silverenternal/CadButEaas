#!/usr/bin/env python3
"""QLoRA smoke training entry point.

This script validates configuration and dependency availability before running
heavy training. It intentionally keeps the production training loop out of the
Rust build and behind an explicit Python environment.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from sft_utils import (
    SftBudget,
    SftJsonlDataset,
    count_jsonl,
    encode_sft_row,
    encoded_sample_stats,
    move_batch_to_device,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vlm/lora_smoke.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--audit-output")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.get("cuda_visible_devices", "0"))
    dataset = Path(config["dataset_jsonl"])
    if not dataset.exists():
        raise SystemExit(f"dataset not found: {dataset}. Run scripts/vlm/generate_dataset.py first.")
    dev_dataset = Path(config["dev_jsonl"])
    if not dev_dataset.exists():
        raise SystemExit(f"dev dataset not found: {dev_dataset}.")
    sft_dataset = Path(config.get("sft_dataset_jsonl", ""))
    sft_dev_dataset = Path(config.get("sft_dev_jsonl", ""))
    if config.get("sft_dataset_jsonl") and not sft_dataset.exists():
        raise SystemExit(
            f"SFT dataset not found: {sft_dataset}. Run scripts/vlm/prepare_sft_dataset.py first."
        )
    if config.get("sft_dev_jsonl") and not sft_dev_dataset.exists():
        raise SystemExit(f"SFT dev dataset not found: {sft_dev_dataset}.")

    if args.dry_run:
        print(
            json.dumps(
                {
                    "ok": True,
                    "config": config,
                    "datasets": {
                        "train_rows": count_jsonl(dataset),
                        "dev_rows": count_jsonl(dev_dataset),
                        "sft_train_rows": count_jsonl(sft_dataset) if sft_dataset.exists() else None,
                        "sft_dev_rows": count_jsonl(sft_dev_dataset) if sft_dev_dataset.exists() else None,
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from torch.utils.data import DataLoader
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    except ImportError as exc:
        raise SystemExit(
            "missing training dependency. Install the environment from scripts/vlm/README.md first."
        ) from exc

    if int(config.get("per_device_train_batch_size", 1)) != 1:
        raise SystemExit("current multimodal SFT loop supports per_device_train_batch_size=1 only.")

    base_model = str(config["base_model"])
    max_steps = args.max_steps or int(config.get("max_steps", 20))
    learning_rate = args.learning_rate or float(config.get("learning_rate", 1e-4))
    max_length = int(config.get("max_length", 12288))
    max_image_side = int(config.get("max_image_side", 0) or 0)
    budget = SftBudget.from_config(config)
    output_dir = Path(config["output_dir"])
    audit_output = Path(args.audit_output) if args.audit_output else output_dir / "train_audit.jsonl"

    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    quantization_config = None
    if config.get("load_in_4bit", False):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForImageTextToText.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    if config.get("load_in_4bit", False):
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=int(config.get("lora_r", 16)),
        lora_alpha=int(config.get("lora_alpha", 32)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(config.get("target_modules", ["q_proj", "v_proj"])),
    )
    model = get_peft_model(model, lora_config)
    model.train()
    model.print_trainable_parameters()

    train_dataset = SftJsonlDataset(sft_dataset, limit=args.limit)
    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=lambda rows: rows[0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    grad_accum = int(config.get("gradient_accumulation_steps", 1))
    started = time.perf_counter()
    losses = []
    skipped_oom = 0
    skipped_no_labels = 0
    skipped_nonfinite = 0
    skipped_budget = 0
    peak_memory_mib = 0
    optimizer.zero_grad(set_to_none=True)
    audit_events = []

    for sample_index, row in enumerate(dataloader, start=1):
        encoded = encode_sft_row(processor, row, max_length=max_length, max_image_side=max_image_side)
        stats = encoded_sample_stats(encoded)
        skip_reason = budget.skip_reason(stats)
        if skip_reason is not None:
            skipped_budget += 1
            event = training_event(sample_index, row, stats, skipped=f"budget_{skip_reason}")
            audit_events.append(event)
            print(json.dumps(event, ensure_ascii=False), flush=True)
            continue
        if stats.supervised_tokens == 0:
            skipped_no_labels += 1
            event = training_event(sample_index, row, stats, skipped="no_supervised_tokens")
            audit_events.append(event)
            print(json.dumps(event, ensure_ascii=False), flush=True)
            continue
        try:
            encoded = move_batch_to_device(encoded, model.device)
            outputs = model(**encoded)
            loss = outputs.loss / grad_accum
            if not torch.isfinite(loss):
                skipped_nonfinite += 1
                optimizer.zero_grad(set_to_none=True)
                event = training_event(sample_index, row, stats, skipped="nonfinite_loss")
                audit_events.append(event)
                print(json.dumps(event, ensure_ascii=False), flush=True)
                continue
            loss.backward()
        except torch.OutOfMemoryError:
            if not config.get("skip_oom_samples", False):
                raise
            skipped_oom += 1
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            event = training_event(sample_index, row, stats, skipped="oom")
            audit_events.append(event)
            print(json.dumps(event, ensure_ascii=False), flush=True)
            continue
        losses.append(float(loss.detach().cpu()) * grad_accum)
        train_step = len(losses)
        if train_step % grad_accum == 0 or train_step == max_steps:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        event = training_event(
            sample_index,
            row,
            stats,
            step=train_step,
            loss=round(losses[-1], 6),
            cuda_peak_mib=current_peak_memory_mib(torch),
        )
        audit_events.append(event)
        print(json.dumps(event, ensure_ascii=False), flush=True)
        peak_memory_mib = max(peak_memory_mib, current_peak_memory_mib(torch))
        if train_step >= max_steps:
            break

    output = {
        "ok": True,
        "steps": len(losses),
        "mean_loss": round(sum(losses) / len(losses), 6) if losses else None,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "output_dir": str(output_dir),
        "skipped_oom": skipped_oom,
        "skipped_no_labels": skipped_no_labels,
        "skipped_nonfinite": skipped_nonfinite,
        "skipped_budget": skipped_budget,
        "peak_memory_mib": peak_memory_mib,
        "budget": budget.to_dict(),
        "audit_path": str(audit_output),
    }
    if not args.no_save:
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        (output_dir / "train_summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    if not args.no_save or args.audit_output:
        audit_output.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(audit_output, audit_events)
    print(json.dumps(output, ensure_ascii=False, indent=2))


def current_peak_memory_mib(torch) -> int:
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.max_memory_allocated() / 1024 / 1024)


def training_event(
    sample_index,
    row,
    stats,
    *,
    step=None,
    loss=None,
    skipped=None,
    cuda_peak_mib=None,
):
    event = {
        "sample": sample_index,
        "image": row.get("image"),
        "source_dataset": row.get("source_dataset"),
        **stats.to_dict(),
    }
    if step is not None:
        event["step"] = step
    if loss is not None:
        event["loss"] = loss
    if skipped is not None:
        event["skipped"] = skipped
    if cuda_peak_mib is not None:
        event["cuda_peak_mib"] = cuda_peak_mib
    return event


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
