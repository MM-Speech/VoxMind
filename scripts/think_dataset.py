#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Speech dataset variant that emits audio+tool responses with <tts_start> in inputs only."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence
import json
import torch

from think_dataset_s2s import (
    S2SDataCollator,
    SpeechToSpeechDataset,
    SpeechTokenizer,
    _append_text_segment,
    _to_text,
    _interleave_token_ids,
    load_conversations_from_json,
)
from utils import compute_token_num, load_audio, log_mel_spectrogram

LOGGER = logging.getLogger(__name__)

# --- MODIFIED: Updated DEFAULT_PROMPT_TEMPLATE ---
DEFAULT_PROMPT_TEMPLATE = """You are an end-to-end speech large language model.
You understand speech and text and can decide whether to answer normally or call tools.

## Reasoning Requirement

Before every assistant response, you MUST generate an internal reasoning block:

<|THINK_START|>
... your chain-of-thought reasoning ...
<|THINK_END|> 

## Tool Decision Rules

- If the user request can be answered directly → respond normally (no <tool_call>).
- If the request requires external capabilities → call tools.
- If there is no available tool that meets the requirements, and this task cannot be solved by itself, a tool must be used → You must first call:

<tool_call>
[{{"name":"searchTools","arguments":{{}}}}]
</tool_call>

Then select a suitable tool if found.

## Tool Call Format
When calling tools, output ONLY the following structure:

<tool_call>
[
  {{"name":"TOOL_NAME", "arguments":{{...}}}},
  {{"name":"TOOL_NAME_2", "arguments":{{...}}}}
]
</tool_call>

Rules:
- Inside <tool_call></tool_call> must be a JSON array only.
- Each item: {{"name": "...", "arguments": {{...}}}}
- Even single tool calls must be wrapped in a JSON array.
- Ensure valid JSON (no trailing commas).

Normal language responses must NOT include <tool_call>.

## Available Tools
{tool_section}

"""

OBSERVATION_PREFIX = "Below is the return content of the function you used:"


def format_tool_descriptions(tool_spec: Optional[Sequence[Dict[str, Any]]] | str) -> str:
    """Render tool metadata for the prompt. Prefer raw JSON to match user spec."""
    if isinstance(tool_spec, str):
        text = tool_spec.strip()
        return text or "[]"
    if not tool_spec:
        return "[]"
    try:
        return json.dumps(tool_spec, ensure_ascii=False)
    except Exception:
        return "[]"

# --- MODIFIED: Function signature and implementation updated to include user_id ---
def build_default_system_prompt(
    tool_spec: Optional[Sequence[Dict[str, Any]]] | str = None,
    current_time: Optional[str] = None,
) -> str:
    tool_section = format_tool_descriptions(tool_spec)
    base = DEFAULT_PROMPT_TEMPLATE.format(tool_section=tool_section)
    ct = str(current_time).strip() if current_time is not None else ""
    return base + (f"\nCurrent time: {ct}" if ct else "")


def _clone_content(value: Any) -> Any:
    if isinstance(value, list):
        return [_clone_content(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_content(val) for key, val in value.items()}
    return value


def _append_id_segment(segments: List[Dict], token_ids: Sequence[int], trainable: bool) -> None:
    if not token_ids:
        return
    segments.append({"type": "ids", "value": list(token_ids), "trainable": trainable})


class FormatAgentDataset(SpeechToSpeechDataset):
    """Dataset that keeps <tts_start> in inputs only and outputs <tts_end> + tool_call without speech preamble."""

    def __init__(
        self,
        data: Sequence[Dict],
        tokenizer,
        audio_root: str,
        speech_tokenizer: Optional[SpeechTokenizer],
        include_transcript_for_speech: bool = True,
        max_length: Optional[int] = None,
        audio_chunk_size: int = 25 * 16000,
        base_system_prompt: Optional[str] = None,
        append_system_prompt: bool = True,
        include_default_prompt: bool = True,
        fallback_tool_spec: Optional[Sequence[Dict[str, Any]]] = None,
        max_target_new_tokens: Optional[int] = None,
        
        debug_audio: bool = False,
    ):
        self.base_system_prompt = base_system_prompt.strip() if base_system_prompt else None
        self.append_system_prompt = append_system_prompt
        self.include_default_prompt = include_default_prompt
        self.fallback_tool_spec = fallback_tool_spec
        self.max_target_new_tokens = max_target_new_tokens if (max_target_new_tokens and max_target_new_tokens > 0) else None
        self.debug_audio = bool(debug_audio)
        self._debug_audio_emitted = 0
        self._debug_audio_limit = 10

        self.audio_start_token_id = tokenizer.convert_tokens_to_ids("<audio_start>")
        self.audio_end_token_id = tokenizer.convert_tokens_to_ids("<audio_end>")
        self.audio_patch_token_id = tokenizer.convert_tokens_to_ids("<audio_patch>")
        self.tts_start_token_id = tokenizer.convert_tokens_to_ids("<tts_start>")
        self.tts_end_token_id = tokenizer.convert_tokens_to_ids("<tts_end>")
        self.think_start_token_id = tokenizer.convert_tokens_to_ids("<|THINK_START|>")
        self.think_end_token_id = tokenizer.convert_tokens_to_ids("<|THINK_END|>")
        if (
            self.audio_start_token_id is None
            or self.audio_end_token_id is None
            or self.audio_patch_token_id is None
            or self.audio_start_token_id < 0
            or self.audio_end_token_id < 0
            or self.audio_patch_token_id < 0
        ):
            raise ValueError("Tokenizer must provide <audio_start>, <audio_end>, and <audio_patch> tokens.")
        if (
            self.tts_start_token_id is None
            or self.tts_end_token_id is None
            or self.tts_start_token_id < 0
            or self.tts_end_token_id < 0
        ):
            raise ValueError("Tokenizer must provide <tts_start> and <tts_end> tokens.")
        if (
            self.think_start_token_id is None
            or self.think_end_token_id is None
            or self.think_start_token_id < 0
            or self.think_end_token_id < 0
        ):
            LOGGER.warning("Tokenizer lacks THINK tokens; think will be treated as plain text.")

        processed_data = self._prepare_data(data)

        super().__init__(
            processed_data,
            tokenizer=tokenizer,
            audio_root=audio_root,
            speech_tokenizer=speech_tokenizer,
            include_transcript_for_speech=include_transcript_for_speech,
            max_length=max_length,
            audio_chunk_size=audio_chunk_size,
        )

    def _build_single_sample(
        self,
        turns: Sequence[Dict],
        user_index: int,
        assistant_index: int,
        conversation_index: int,
    ) -> Optional[Dict]:
        segments: List[Dict] = []
        user_audio_segments: List[torch.Tensor] = []

        for ctx_idx, ctx in enumerate(turns[:user_index]):
            try:
                self._append_context_turn(segments, ctx)
            except ValueError as exc:
                LOGGER.warning(
                    "Skipping context turn %s in conversation %s: %s",
                    ctx_idx,
                    conversation_index,
                    exc,
                )

        user_turn = turns[user_index]
        self._append_user_turn(segments, user_turn, audio_buffer=user_audio_segments)

        assistant_turn = turns[assistant_index]
        prev_user_type = str(user_turn.get("type") or "").lower()
        self._append_assistant_target(segments, assistant_turn, prev_user_type)

        tokenized = self._segments_to_tensors(segments)
        if tokenized is None:
            return None

        input_ids, labels = tokenized
        if self.max_length is not None and input_ids.size(0) > self.max_length:
            input_ids = input_ids[-self.max_length:]
            labels = labels[-self.max_length:]

        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        sample: Dict[str, Any] = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
        if user_audio_segments:
            sample["wavs"] = user_audio_segments
            if self.debug_audio and self._debug_audio_emitted < self._debug_audio_limit:
                placeholder_count = int((input_ids == self.audio_start_token_id).sum().item())
                chunk_frames = [int(mel.shape[1]) for mel in user_audio_segments]
                LOGGER.info(
                    "[AudioDebug] convo=%s user_turn=%s chunks=%s placeholders=%s frames=%s",
                    conversation_index,
                    user_index,
                    len(user_audio_segments),
                    placeholder_count,
                    chunk_frames,
                )
                self._debug_audio_emitted += 1

        if self.max_target_new_tokens is None:
            return sample

        labels_tensor = sample.get("labels")
        if labels_tensor is None or not hasattr(labels_tensor, "numel") or labels_tensor.numel() == 0:
            return sample

        trainable_positions = (labels_tensor != -100).nonzero(as_tuple=False).view(-1)
        if trainable_positions.numel() <= int(self.max_target_new_tokens):
            return sample

        cutoff_pos = int(trainable_positions[int(self.max_target_new_tokens) - 1].item() + 1)
        for key in ("input_ids", "labels", "attention_mask"):
            tensor = sample.get(key)
            if hasattr(tensor, "dim") and tensor.dim() == 1 and tensor.size(0) >= cutoff_pos:
                sample[key] = tensor[:cutoff_pos]
        return sample

    def _prepare_data(self, data: Sequence[Dict]) -> List[Dict]:
        records: List[Dict] = []
        for record in data:
            cloned = dict(record)
            conversations = record.get("conversations") or []
            cloned_conversations = [self._clone_turn(turn) for turn in conversations]
            prompt = self._compose_system_prompt(record)
            cloned["conversations"] = self._inject_system_prompt(cloned_conversations, prompt)
            records.append(cloned)
        return records

    # --- MODIFIED: Extracts userId and passes it to the prompt builder ---
    def _compose_system_prompt(self, record: Dict) -> Optional[str]:
        segments: List[str] = []
        tool_spec = record.get("tools")
        current_time = record.get("time")
        if not tool_spec and self.fallback_tool_spec:
            tool_spec = self.fallback_tool_spec
        if self.include_default_prompt:
            segments.append(build_default_system_prompt(tool_spec, current_time=current_time))
        elif tool_spec:
            segments.append("Available tools:\n" + format_tool_descriptions(tool_spec))
        if self.base_system_prompt:
            segments.append(self.base_system_prompt)
        prompt = "\n\n".join(seg.strip() for seg in segments if isinstance(seg, str) and seg.strip())
        return prompt or None

    def _clone_turn(self, turn: Dict) -> Dict:
        cloned = dict(turn)
        role = cloned.get("role")
        if isinstance(role, str) and role.lower() == "assistant":
            cloned["role"] = "assistant"
        if "content" in cloned:
            cloned["content"] = _clone_content(cloned["content"])
        return cloned

    def _inject_system_prompt(self, conversations: List[Dict], prompt: Optional[str]) -> List[Dict]:
        if not prompt:
            return conversations
        if not conversations:
            return [{"role": "system", "content": prompt}]

        first_role = str(conversations[0].get("role") or "").lower()
        if first_role == "system":
            if self.append_system_prompt:
                existing = _to_text(conversations[0].get("content"))
                if prompt not in existing:
                    merged = f"{existing}\n\n{prompt}" if existing else prompt
                    conversations[0] = {**conversations[0], "content": merged}
            return conversations

        conversations.insert(0, {"role": "system", "content": prompt})
        return conversations

    def _render_observation_text(self, turn: Dict) -> str:
        content = self._collect_text(turn.get("content"))
        if content:
            return f"{OBSERVATION_PREFIX}\n{content}"
        return OBSERVATION_PREFIX

    # --- REVIEWED: This logic correctly handles tool calls in context ---
    def _render_context_text(self, turn: Dict) -> str:
        turn_type = str(turn.get("type") or "").lower()
        if turn_type == "observation":
            return self._render_observation_text(turn)
        if turn_type == "tool":
            return self._format_tool_output(turn)
        return self._collect_text(turn.get("content"))

    def _append_context_turn(self, segments: List[Dict], turn: Dict) -> None:
        turn_type = str(turn.get("type") or "").lower()
        role = "observation" if turn_type == "observation" else self._normalize_role(turn.get("role"))
        _append_text_segment(segments, f"<|BOT|>{role}\n", trainable=False)
        context_text = self._render_context_text(turn)
        _append_text_segment(segments, context_text, trainable=False)
        if turn.get("eot", True):
            _append_text_segment(segments, "<|EOT|>", trainable=False)

    def _append_user_turn(
        self,
        segments: List[Dict],
        arg2,
        turn: Optional[Dict] = None,
        audio_buffer: Optional[List[torch.Tensor]] = None,
    ) -> None:
        if turn is None:
            turn = arg2

        turn_type = str(turn.get("type") or "").lower()
        eot = turn.get("eot", True)

        if turn_type == "observation":
            _append_text_segment(segments, "<|BOT|>observation\n", trainable=False)
            _append_text_segment(segments, self._render_observation_text(turn), trainable=False)
            if eot:
                _append_text_segment(segments, "<|EOT|>", trainable=False)
            return

        _append_text_segment(segments, "<|BOT|>human\n", trainable=False)

        if turn_type == "audio" and turn.get("audio-path"):
            self._append_user_audio_segments(segments, turn, audio_buffer)
        else:
            if turn_type == "audio" and not turn.get("audio-path"):
                LOGGER.warning("User turn marked as audio but missing 'audio-path'; using transcript instead.")
            _append_text_segment(segments, self._collect_text(turn.get("content")), trainable=False)

        if eot:
            _append_text_segment(segments, "<|EOT|>", trainable=False)

    def _append_user_audio_segments(
        self,
        segments: List[Dict],
        turn: Dict,
        audio_buffer: Optional[List[torch.Tensor]],
    ) -> None:
        if audio_buffer is None:
            LOGGER.debug("Audio buffer unavailable; degrading audio turn to transcript text.")
            _append_text_segment(segments, self._collect_text(turn.get("content")), trainable=False)
            return

        audio_path = turn.get("audio-path")
        resolved_path = self._resolve_audio_path(audio_path)
        try:
            waveform = load_audio(resolved_path)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to load user audio %s (%s); falling back to transcript.", resolved_path, exc)
            _append_text_segment(segments, self._collect_text(turn.get("content")), trainable=False)
            return

        chunk_size = max(int(self.audio_chunk_size), 1)
        has_chunk = False
        for start in range(0, waveform.shape[0], chunk_size):
            chunk = waveform[start : start + chunk_size]
            if chunk.numel() == 0:
                continue
            mel = log_mel_spectrogram(chunk, n_mels=128, padding=479).float()
            audio_buffer.append(mel)
            patch_count = compute_token_num(mel.shape[1])
            placeholder = "<audio_start>" + ("<audio_patch>" * patch_count) + "<audio_end>"
            _append_text_segment(segments, placeholder, trainable=False)
            has_chunk = True

        if not has_chunk:
            LOGGER.warning("User audio %s produced no valid chunks; using transcript instead.", resolved_path)
            _append_text_segment(segments, self._collect_text(turn.get("content")), trainable=False)
            return


    def _append_assistant_target(self, segments: List[Dict], turn: Dict, prev_user_type: str) -> None:
        _append_text_segment(segments, "<|BOT|>assistant\n", trainable=False)
        turn_type = str(turn.get("type") or "").lower()
        eot_flag = turn.get("eot", True)

        think_text = _to_text(turn.get("think"))
        self._append_think_segments(segments, think_text)

        if turn_type == "audio" and turn.get("audio-path"):
            self._append_audio_response(segments, turn)
        elif turn_type == "tool":
            content_text = self._collect_text(turn.get("content"))
            if prev_user_type == "audio":
                self._append_tts_markers(segments)
                if content_text:
                    _append_text_segment(segments, content_text, trainable=True)
            else:
                if content_text:
                    _append_text_segment(segments, content_text, trainable=True)
        else:
            content_text = self._collect_text(turn.get("content"))
            if content_text:
                _append_text_segment(segments, content_text, trainable=True)

        if eot_flag:
            _append_text_segment(segments, "<|EOT|>", trainable=True)

    def _append_audio_response(self, segments: List[Dict], turn: Dict) -> None:
        if self.speech_tokenizer is None:
            raise ValueError("Speech tokenizer is required for assistant speech targets.")
        resolved_path = self._resolve_audio_path(turn.get("audio-path"))
        speech_tokens = self.speech_tokenizer.encode(resolved_path)

        transcript_text = _to_text(turn.get("content")) if self.include_transcript_for_speech else ""
        transcript_tokens = self._tokenize_text(transcript_text)

        audio_token_ids = [self.audio_token_offset + tid for tid in speech_tokens]
        self._append_speech_segments(segments, transcript_tokens, audio_token_ids)

    def _append_tts_markers(self, segments: List[Dict]) -> None:
        if self.tts_start_token_id is not None and self.tts_start_token_id >= 0:
            _append_id_segment(segments, [self.tts_start_token_id], trainable=True)
        else:
            _append_text_segment(segments, "<tts_start>", trainable=True)
        if self.tts_end_token_id is not None and self.tts_end_token_id >= 0:
            _append_id_segment(segments, [self.tts_end_token_id], trainable=True)
        else:
            _append_text_segment(segments, "<tts_end>", trainable=True)

    def _append_speech_segments(
        self,
        segments: List[Dict],
        transcript_tokens: Sequence[int],
        audio_token_ids: Sequence[int],
    ) -> None:
        if self.tts_start_token_id is not None and self.tts_start_token_id >= 0:
            _append_id_segment(segments, [self.tts_start_token_id], trainable=True)
        else:
            _append_text_segment(segments, "<tts_start>", trainable=True)

        if transcript_tokens and audio_token_ids:
            payload = _interleave_token_ids(transcript_tokens, audio_token_ids, self.tts_pad_id)
            _append_id_segment(segments, payload, trainable=True)
        elif audio_token_ids:
            _append_id_segment(segments, audio_token_ids, trainable=True)
        elif transcript_tokens:
            _append_id_segment(segments, transcript_tokens, trainable=True)

        if self.tts_end_token_id is not None and self.tts_end_token_id >= 0:
            _append_id_segment(segments, [self.tts_end_token_id], trainable=True)
        else:
            _append_text_segment(segments, "<tts_end>", trainable=True)

    def _append_think_segments(self, segments: List[Dict], think_text: str) -> None:
        inner = self._extract_think_inner(think_text)
        if self.think_start_token_id is not None and self.think_start_token_id >= 0:
            _append_id_segment(segments, [self.think_start_token_id], trainable=True)
        else:
            if think_text:
                _append_text_segment(segments, "<|THINK_START|>", trainable=True)
        if inner:
            ids = self._tokenize_text(inner)
            if ids:
                _append_id_segment(segments, ids, trainable=True)
        if self.think_end_token_id is not None and self.think_end_token_id >= 0:
            _append_id_segment(segments, [self.think_end_token_id], trainable=True)
        else:
            if think_text:
                _append_text_segment(segments, "<|THINK_END|>", trainable=True)

    def _extract_think_inner(self, text: str) -> str:
        if not text:
            return ""
        s = text.strip()
        start = "<|THINK_START|>"
        end = "<|THINK_END|>"
        if s.startswith(start) and s.endswith(end):
            return s[len(start): -len(end)].strip()
        return s

    def _tokenize_text(self, text: str) -> List[int]:
        if not text:
            return []
        return self.tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_tensors=None,
        )["input_ids"]

    def _contains_tool_marker(self, value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, dict):
            return any(self._contains_tool_marker(v) for v in value.values())
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
            return any(self._contains_tool_marker(item) for item in value)
        text = str(value).strip()
        if not text:
            return False
        return text.startswith("<tool_call>")

    def _is_tool_call(self, turn: Dict) -> bool:
        turn_type = str(turn.get("type") or "").lower()
        if turn_type == "tool":
            return True
        content = turn.get("content")
        return self._contains_tool_marker(content)

    def _collect_text(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            parts = [self._collect_text(value) for value in content.values()]
            return "\n".join(part for part in parts if part)
        if isinstance(content, Sequence) and not isinstance(content, (bytes, bytearray)):
            parts = [self._collect_text(item) for item in content]
            return "\n".join(part for part in parts if part)
        return str(content)

    def _format_tool_output(self, turn: Dict) -> str:
        raw = self._collect_text(turn.get("content")).strip()
        if not raw:
            return "<tool_call>[]</tool_call>"

        if raw.startswith("<tool_call>"):
            text = raw
        else:
            payload = raw
            if not payload.startswith("["):
                payload = f"[{payload}]"
            text = f"<tool_call>{payload}</tool_call>"

        if not text.endswith("</tool_call>"):
            text = text.rstrip() + "</tool_call>"
        return text.strip()

    def _tool_text_from_turn(self, turn: Dict) -> str:
        return self._format_tool_output(turn)


__all__ = [
    "FormatAgentDataset",
    "SpeechTokenizer",
    "S2SDataCollator",
    "build_default_system_prompt",
    "format_tool_descriptions",
    "load_conversations_from_json",
    # New export for streaming training
    "StreamingFormatAgentDataset",
]

# -----------------------------------------------------------------------------
# Streaming IterableDataset wrapper
# -----------------------------------------------------------------------------
import io
import os as _os
import json as _json
from typing import Iterator as _Iterator
import torch as _torch
from torch.utils.data import IterableDataset as _IterableDataset


class StreamingFormatAgentDataset(_IterableDataset):
    """IterableDataset that reads JSONL (or JSON) progressively and yields samples.

    This wrapper avoids loading the entire dataset into memory. It parses each
    record, constructs a temporary FormatAgentDataset with that single record,
    then yields its built training samples on the fly.

    Notes:
    - Prefer `.jsonl` files for true streaming. `.json` files are supported but
      will be read once and iterated over without materializing all samples.
    - Shuffling is not applied here (PyTorch DataLoader cannot shuffle IterableDatasets).
      If you need some randomness, consider running with multiple workers or
      pre-shuffling your `.jsonl` on disk.
    """

    def __init__(
        self,
        dataset_path: str,
        tokenizer,
        audio_root: str,
        speech_tokenizer: SpeechTokenizer,
        include_transcript_for_speech: bool = True,
        max_length: Optional[int] = None,
        audio_chunk_size: int = 25 * 16000,
        base_system_prompt: Optional[str] = None,
        append_system_prompt: bool = True,
        include_default_prompt: bool = True,
        fallback_tool_spec: Optional[Sequence[Dict[str, Any]]] = None,
        max_target_new_tokens: Optional[int] = None,
        debug_audio: bool = False,
    ) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.audio_root = audio_root
        self.speech_tokenizer = speech_tokenizer
        self.include_transcript_for_speech = include_transcript_for_speech
        self.max_length = max_length
        self.audio_chunk_size = audio_chunk_size
        self.base_system_prompt = base_system_prompt
        self.append_system_prompt = append_system_prompt
        self.include_default_prompt = include_default_prompt
        self.fallback_tool_spec = fallback_tool_spec
        self.max_target_new_tokens = max_target_new_tokens
        self.debug_audio = debug_audio

    def _is_jsonl(self) -> bool:
        return str(self.dataset_path).lower().endswith(".jsonl")

    def _iter_jsonl(self) -> _Iterator[Dict]:
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = _json.loads(line)
                except Exception:
                    continue
                if isinstance(record, dict):
                    yield record

    def _iter_json(self) -> _Iterator[Dict]:
        # Stream-friendly JSON reader: loads the top-level but yields records one by one.
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            try:
                data = _json.load(f)
            except Exception:
                data = []
        if isinstance(data, dict):
            # If dict, try common keys
            seq = []
            for key in ("data", "records", "items", "conversations"):
                val = data.get(key)
                if isinstance(val, list):
                    seq = val
                    break
            if not seq:
                return iter(())
            for item in seq:
                if isinstance(item, dict):
                    yield item
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    yield item

    def __iter__(self):
        iterator = self._iter_jsonl() if self._is_jsonl() else self._iter_json()
        for record in iterator:
            try:
                # Build a temporary per-record dataset and yield its samples
                tmp_ds = FormatAgentDataset(
                    data=[record],
                    tokenizer=self.tokenizer,
                    audio_root=self.audio_root,
                    speech_tokenizer=self.speech_tokenizer,
                    include_transcript_for_speech=self.include_transcript_for_speech,
                    max_length=self.max_length,
                    audio_chunk_size=self.audio_chunk_size,
                    base_system_prompt=self.base_system_prompt,
                    append_system_prompt=self.append_system_prompt,
                    include_default_prompt=self.include_default_prompt,
                    fallback_tool_spec=self.fallback_tool_spec,
                    max_target_new_tokens=self.max_target_new_tokens,
                    debug_audio=self.debug_audio,
                )
                for i in range(len(tmp_ds)):
                    yield tmp_ds[i]
            except Exception as _exc:
                try:
                    LOGGER.warning("[StreamingDataset] Failed to build sample for one record: %s", _exc)
                except Exception:
                    pass
