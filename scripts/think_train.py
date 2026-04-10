#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Training entry point for the formatted speech/tool dataset."""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import random
import time
from typing import Any, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from think_dataset import (
    FormatAgentDataset,
    S2SDataCollator,
    SpeechTokenizer,
    load_conversations_from_json,
    StreamingFormatAgentDataset,
)

LOGGER = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FormatSFTTrainer(Trainer):
    """Trainer that optionally logs forward pass token ids for inspection."""

    def __init__(self, *args, forward_log_dir: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward_log_dir = forward_log_dir

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        labels = inputs.get("labels")
        if labels is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        labels = labels.to(logits.device)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        if self.forward_log_dir and self.is_world_process_zero():
            self._log_forward_step(
                step=getattr(self.state, "global_step", 0),
                logits=logits,
                labels=labels,
                input_ids=inputs.get("input_ids"),
                attention_mask=inputs.get("attention_mask"),
            )

        return (loss, outputs) if return_outputs else loss

    def _log_forward_step(
        self,
        step: int,
        logits: torch.Tensor,
        labels: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> None:
        try:
            os.makedirs(self.forward_log_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            path = os.path.join(self.forward_log_dir, f"forward_{timestamp}_{step}.json")
            with torch.no_grad():
                shift_logits = logits[..., :-1, :]
                shift_labels = labels[..., 1:]
                shift_preds = torch.argmax(shift_logits, dim=-1)
                mask = shift_labels != -100
                batch_records = []
                for idx in range(shift_labels.size(0)):
                    sample_labels = labels[idx].detach().cpu().tolist()
                    label_tokens = shift_labels[idx][mask[idx]].detach().cpu().tolist()
                    output_tokens = shift_preds[idx][mask[idx]].detach().cpu().tolist()
                    raw_inputs = input_ids[idx].detach().cpu().tolist() if input_ids is not None else None
                    raw_attention = attention_mask[idx].detach().cpu().tolist() if attention_mask is not None else None
                    batch_records.append(
                        {
                            "sample_index": idx,
                            "raw_labels": sample_labels,
                            "label_tokens": label_tokens,
                            "output_tokens": output_tokens,
                            "raw_input_ids": raw_inputs,
                            "raw_attention_mask": raw_attention,
                        }
                    )
            record = {"step": step, "batch": batch_records}
            with open(path, "w", encoding="utf-8") as file:
                json.dump(record, file, ensure_ascii=False)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to write forward log '%s': %s", step, exc)


def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as file:
        return file.read().strip()


def _load_tool_schema(path: str) -> Optional[Any]:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, dict):
        if "tools" in data and isinstance(data["tools"], list):
            return data["tools"]
        if "functions" in data and isinstance(data["functions"], list):
            return data["functions"]
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formatter-aware Step-Audio-2 SFT trainer.")
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--audio-root", type=str, required=True, help="Directory that stores wav files referenced by the dataset.")
    parser.add_argument("--token2wav-path", type=str, required=True, help="Path to Step-Audio-2-mini/token2wav for speech tokenizer weights.")
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=20960)

    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--num-epochs", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    # Scheduler settings (use cosine by default)
    parser.add_argument(
        "--lr-scheduler-type",
        type=str,
        default="cosine",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
        help="学习率调度类型，默认 cosine。",
    )
    parser.add_argument(
        "--num-cycles",
        type=int,
        default=1,
        help="cosine_with_restarts 的重启次数（仅在该调度下生效）",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Warmup 的绝对步数（>0 时优先于 warmup_ratio）",
    )

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--no-speech-transcript", action="store_true", help="Skip inserting assistant transcript before speech tokens.")
    parser.add_argument("--freeze-audio", action="store_true", help="Freeze audio encoder/adapter so only the LLM trains.")

    parser.add_argument("--report-to", type=str, default="none")
    parser.add_argument("--logging-dir", type=str, default=None)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--save-steps", type=int, default=500, help="Save checkpoint every X update steps.")

    # Prompt/tool configuration (formatter dataset uses built-in defaults; keep minimal args)
    parser.add_argument("--forward-log-dir", type=str, default=None, help="Optional directory to save per-step forward pass logs.")
    parser.add_argument("--max-target-new-tokens", type=int, default=4096, help="Max number of trainable target tokens per sample (caps assistant new tokens).")
    parser.add_argument(
        "--speech-tokenizer-device",
        type=str,
        default="auto",
        help="Device for speech tokenizer (cpu, cuda, cuda:<index>, or auto to follow local_rank).",
    )
    parser.add_argument("--debug-audio", action="store_true", help="Enable lightweight logging for audio placeholder/mel alignment.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    LOGGER.info(f"[Rank {local_rank}] Script started. Arguments parsed.")

    seed_everything(args.seed)
    LOGGER.info(f"[Rank {local_rank}] Seeding complete for seed {args.seed}.")

    # Detect streaming mode by file extension; prefer .jsonl for on-the-fly loading
    use_streaming = str(args.dataset_path).lower().endswith(".jsonl")
    if use_streaming:
        LOGGER.info(f"[Rank {local_rank}] Using streaming dataset from JSONL: {args.dataset_path}")
    else:
        train_data = load_conversations_from_json(args.dataset_path)
        LOGGER.info(f"[Rank {local_rank}] JSON dataset loaded from {args.dataset_path}. Found {len(train_data)} records.")
        random.shuffle(train_data)
        LOGGER.info(f"[Rank {local_rank}] Shuffled training records for better mixing.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        padding_side="right",
    )
    LOGGER.info(f"[Rank {local_rank}] LLM Tokenizer loaded from {args.model_name_or_path}.")

    eot_token = "<|EOT|>"
    eot_id = tokenizer.convert_tokens_to_ids(eot_token)
    if eot_id is not None and eot_id >= 0:
        tokenizer.eos_token = eot_token
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif eot_id is not None and eot_id >= 0:
            tokenizer.pad_token = eot_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id
    tts_end_id = tokenizer.convert_tokens_to_ids("<tts_end>")
    LOGGER.info(f"[Rank {local_rank}] Tokenizer special tokens configured.")

    requested_speech_device = (args.speech_tokenizer_device or "cpu").strip()
    normalized_request = requested_speech_device.lower()
    if normalized_request == "auto":
        speech_device = f"cuda:{local_rank}" if torch.cuda.is_available() and local_rank != -1 else "cpu"
    elif normalized_request == "cuda":
        if torch.cuda.is_available():
            speech_device = f"cuda:{local_rank if local_rank != -1 else 0}"
        else:
            LOGGER.warning(
                f"[Rank {local_rank}] CUDA requested for speech tokenizer but no GPU detected; falling back to CPU."
            )
            speech_device = "cpu"
    elif normalized_request.startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning(
            f"[Rank {local_rank}] Speech tokenizer device '{requested_speech_device}' requires CUDA but none is available; using CPU."
        )
        speech_device = "cpu"
    else:
        speech_device = requested_speech_device or "cpu"

    speech_tokenizer = SpeechTokenizer(args.token2wav_path, device=speech_device)
    try:
        if hasattr(speech_tokenizer, "audio_tokenizer"):
            import torch as _t
            speech_tokenizer.audio_tokenizer = speech_tokenizer.audio_tokenizer.to(_t.device(speech_device))
    except Exception as _exc:
        LOGGER.warning(f"[Rank {local_rank}] Failed to move speech tokenizer to {speech_device}: {_exc}")
    LOGGER.info(
        f"[Rank {local_rank}] Speech Tokenizer loaded from {args.token2wav_path} onto {speech_device}."
    )

    if args.debug_audio:
        LOGGER.info(f"[Rank {local_rank}] Audio debug logging enabled (limited sample/batch reports).")

    if use_streaming:
        LOGGER.info(f"[Rank {local_rank}] Initializing StreamingFormatAgentDataset...")
        train_dataset = StreamingFormatAgentDataset(
            dataset_path=args.dataset_path,
            tokenizer=tokenizer,
            audio_root=args.audio_root,
            speech_tokenizer=speech_tokenizer,
            include_transcript_for_speech=not args.no_speech_transcript,
            max_length=args.max_length,
            max_target_new_tokens=args.max_target_new_tokens,
            debug_audio=args.debug_audio,
        )
        LOGGER.info(f"[Rank {local_rank}] StreamingFormatAgentDataset initialized (IterableDataset).")
    else:
        LOGGER.info(f"[Rank {local_rank}] Initializing FormatAgentDataset...")
        train_dataset = FormatAgentDataset(
            train_data,
            tokenizer=tokenizer,
            audio_root=args.audio_root,
            speech_tokenizer=speech_tokenizer,
            include_transcript_for_speech=not args.no_speech_transcript,
            max_length=args.max_length,
            max_target_new_tokens=args.max_target_new_tokens,
            debug_audio=args.debug_audio,
        )
        LOGGER.info(f"[Rank {local_rank}] FormatAgentDataset initialized. Total training samples: {len(train_dataset)}")

    LOGGER.info(f"[Rank {local_rank}] Loading model from {args.model_name_or_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    LOGGER.info(f"[Rank {local_rank}] Model loaded successfully.")

    try:
        if args.max_length and getattr(model.config, "max_position_embeddings", None):
            if model.config.max_position_embeddings < args.max_length:
                model.config.max_position_embeddings = args.max_length
                LOGGER.info(f"[Rank {local_rank}] Model max_position_embeddings updated to {args.max_length}.")
    except Exception:
        LOGGER.warning(f"[Rank {local_rank}] Failed to update model max_position_embeddings; continuing with default.")

    eos_ids = []
    if eot_id is not None and eot_id >= 0:
        eos_ids.append(eot_id)
    if tts_end_id is not None and tts_end_id >= 0:
        eos_ids.append(tts_end_id)
    if eos_ids:
        unique_eos = list(dict.fromkeys(eos_ids))
        model.config.eos_token_id = unique_eos[0] if len(unique_eos) == 1 else unique_eos
    if pad_id is not None:
        model.config.pad_token_id = pad_id
    LOGGER.info(f"[Rank {local_rank}] Model config updated for EOS and PAD tokens.")

    if args.freeze_audio:
        frozen_params = 0
        target_modules = []
        if hasattr(model, "encoder") and isinstance(model.encoder, torch.nn.Module):
            target_modules.append(model.encoder)
        if hasattr(model, "adapter") and isinstance(model.adapter, torch.nn.Module):
            target_modules.append(model.adapter)
        for mod in target_modules:
            mod.eval()
            for p in mod.parameters():
                if p.requires_grad:
                    p.requires_grad = False
                    frozen_params += p.numel()
        LOGGER.info(f"[Rank {local_rank}] Frozen {frozen_params} parameters in audio encoder/adapter via module-level freeze.")

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        LOGGER.info(f"[Rank {local_rank}] Gradient checkpointing enabled.")

    from torch.utils.data import IterableDataset as _IterableDataset

    if args.max_steps and args.max_steps > 0:
        total_steps = args.max_steps
    else:
        if use_streaming and isinstance(train_dataset, _IterableDataset):
            # Estimate total steps from line count for JSONL when max_steps not provided.
            try:
                line_count = 0
                with open(args.dataset_path, "r", encoding="utf-8") as _f:
                    for _line in _f:
                        if _line.strip():
                            line_count += 1
                total_steps = int(max(1, line_count) * args.num_epochs)
                LOGGER.warning(
                    f"[Rank {local_rank}] IterableDataset without --max-steps. Estimated total_steps={total_steps} from {line_count} lines."
                )
            except Exception:
                total_steps = int(1000 * args.num_epochs)
                LOGGER.warning(
                    f"[Rank {local_rank}] Failed to estimate steps from JSONL. Defaulting total_steps={total_steps}. Consider setting --max-steps."
                )
        else:
            steps_per_epoch = math.ceil(len(train_dataset) / (args.batch_size * args.gradient_accumulation_steps))
            total_steps = int(steps_per_epoch * args.num_epochs)
    LOGGER.info(f"[Rank {local_rank}] Planned total steps: {total_steps}")

    # Build TrainingArguments with optional Accelerate knobs for streaming
    from transformers import TrainingArguments as _TA
    _fields = getattr(_TA, "__dataclass_fields__", {})
    _ta_kwargs = dict(
        remove_unused_columns=False,
        output_dir=args.output_dir,
        # Keep per-device batch size to match DeepSpeed micro-batch config
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        # warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        logging_dir=args.logging_dir,
        # save_total_limit=2,
        save_only_model=True,
        save_safetensors=True,
        save_on_each_node=False,
        save_strategy="steps",
        save_steps=args.save_steps,
        bf16=args.bf16,
        num_train_epochs=(args.num_epochs if (args.max_steps <= 0 and not use_streaming) else 1.0),
        max_steps=total_steps,
        report_to=args.report_to,
        run_name=args.run_name,
        evaluation_strategy="no",
        dataloader_num_workers=0,
        dataloader_drop_last=True,
        gradient_checkpointing=args.gradient_checkpointing,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        deepspeed=args.deepspeed,
        optim='adamw_torch'
    )

    # Disable split_batches to avoid batch_size division issues with small per-device batch
    if "split_batches" in _fields:
        _ta_kwargs["split_batches"] = False
    # Disable dispatch_batches when streaming to avoid per-process batch fetching
    if "dispatch_batches" in _fields and use_streaming:
        _ta_kwargs["dispatch_batches"] = False

    # Prefer new accelerator_config if available (transformers>=4.41)
    if "accelerator_config" in _fields and use_streaming:
        _acc_conf = dict(split_batches=False, dispatch_batches=False)
        _ta_kwargs["accelerator_config"] = _acc_conf

    training_args = TrainingArguments(**_ta_kwargs)

    # Build collator with backward-compatible debug flag handling
    try:
        data_collator = S2SDataCollator(tokenizer, debug_audio=args.debug_audio)
    except TypeError:
        LOGGER.warning("S2SDataCollator does not support 'debug_audio'; continuing without it.")
        data_collator = S2SDataCollator(tokenizer)

    trainer = FormatSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        forward_log_dir=args.forward_log_dir,
    )
    LOGGER.info(f"[Rank {local_rank}] Trainer initialized. Starting training process...")

    trainer.train()
    LOGGER.info(f"[Rank {local_rank}] Training finished.")

    final_dir = os.path.join(args.output_dir, "final")
    if trainer.is_world_process_zero():
        os.makedirs(final_dir, exist_ok=True)

    LOGGER.info(f"[Rank {local_rank}] Saving final model...")
    trainer.save_model(final_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(final_dir)

    LOGGER.info(f"[Rank {local_rank}] Script finished successfully.")


if __name__ == "__main__":
    main()
