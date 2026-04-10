#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Speech-to-speech SFT dataset that trains on every user/assistant pair."""
from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from utils import compute_token_num, load_audio, log_mel_spectrogram, padding_mels

LOGGER = logging.getLogger(__name__)

ROLE_MAP = {"system": "system", "user": "human", "human": "human", "assistant": "assistant", "observation": "observation"}
DEFAULT_AUDIO_WINDOW = 25 * 16000  # 25 seconds @ 16 kHz
AUDIO_PADDING = 479
AUDIO_GROUP_SIZE = 4


def _to_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    raise TypeError("Conversation content must be a string when used as text.")


class SpeechTokenizer:
    """ONNX speech tokenizer wrapper."""

    def __init__(self, tokenizer_dir: str, device: Optional[str] = None):
        import s3tokenizer

        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model_path = os.path.join(tokenizer_dir, "speech_tokenizer_v2_25hz.onnx")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"speech_tokenizer_v2_25hz.onnx is missing under {tokenizer_dir}. "
                "Make sure Step-Audio-2-mini/token2wav is available locally."
            )
        self.audio_tokenizer = s3tokenizer.load_model(model_path)
        if self._device.type == "cuda":
            self.audio_tokenizer = self.audio_tokenizer.cuda()
        self.audio_tokenizer.eval()
        self._cache: Dict[str, List[int]] = {}

    def encode(self, wav_path: str) -> List[int]:
        import s3tokenizer

        path = os.path.abspath(wav_path)
        if path in self._cache:
            return self._cache[path]
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Assistant speech file not found: {path}")
        audio = s3tokenizer.load_audio(path, sr=16000)
        mels = s3tokenizer.log_mel_spectrogram(audio)
        mels, mels_lens = s3tokenizer.padding([mels])
        tokens, token_lens = self.audio_tokenizer.quantize(
            mels.to(self._device), mels_lens.to(self._device)
        )
        speech_tokens = tokens[0, : token_lens[0].item()].tolist()
        self._cache[path] = speech_tokens
        return speech_tokens


def _append_text_segment(segments: List[Dict], text: str, trainable: bool) -> None:
    if not text:
        return
    segments.append({"type": "text", "value": text, "trainable": trainable})


def _append_token_segment(segments: List[Dict], token_ids: Sequence[int], trainable: bool) -> None:
    if not token_ids:
        return
    segments.append({"type": "ids", "value": list(token_ids), "trainable": trainable})


def _interleave_token_ids(
    text_token_ids: Sequence[int],
    audio_token_ids: Sequence[int],
    pad_token_id: Optional[int],
) -> List[int]:
    if not text_token_ids:
        return list(audio_token_ids)
    if not audio_token_ids:
        return list(text_token_ids)

    interleaved: List[int] = []
    audio_index = 0
    total_audio = len(audio_token_ids)

    for text_id in text_token_ids:
        interleaved.append(text_id)
        next_audio_index = min(total_audio, audio_index + AUDIO_GROUP_SIZE)
        if next_audio_index > audio_index:
            interleaved.extend(audio_token_ids[audio_index:next_audio_index])
            audio_index = next_audio_index

    while audio_index < total_audio:
        if pad_token_id is not None:
            interleaved.append(pad_token_id)
        next_audio_index = min(total_audio, audio_index + AUDIO_GROUP_SIZE)
        interleaved.extend(audio_token_ids[audio_index:next_audio_index])
        audio_index = next_audio_index
    return interleaved


def _chunk_audio_to_mels(wav_path: str, chunk_size: int) -> List[torch.Tensor]:
    audio = load_audio(wav_path, target_rate=16000)
    if audio.ndim > 1:
        audio = audio.squeeze(0)
    chunks: List[torch.Tensor] = []
    step = max(chunk_size, 1)
    for start in range(0, audio.shape[0], step):
        piece = audio[start : start + step]
        if piece.numel() == 0:
            continue
        mel = log_mel_spectrogram(piece, n_mels=128, padding=AUDIO_PADDING)
        chunks.append(mel)
    return chunks


class SpeechToSpeechDataset(Dataset):
    """Multi-sample dataset that yields one training example per assistant reply."""

    def __init__(
        self,
        data: Sequence[Dict],
        tokenizer,
        audio_root: str,
        speech_tokenizer: Optional[SpeechTokenizer],
        include_transcript_for_speech: bool = True,
        max_length: Optional[int] = None,
        audio_chunk_size: int = DEFAULT_AUDIO_WINDOW,
    ):
        self.data = list(data)
        self.tokenizer = tokenizer
        self.audio_root = audio_root
        self.speech_tokenizer = speech_tokenizer
        self.include_transcript_for_speech = include_transcript_for_speech
        self.max_length = max_length
        self.audio_chunk_size = int(audio_chunk_size or DEFAULT_AUDIO_WINDOW)

        self.tts_pad_id = tokenizer.convert_tokens_to_ids("<tts_pad>")
        if self.tts_pad_id is None:
            LOGGER.warning("Tokenizer lacks <tts_pad>; trailing speech bundles will not be padded.")
        audio_token_id = tokenizer.convert_tokens_to_ids("<audio_0>")
        if audio_token_id is None or audio_token_id < 0:
            raise ValueError("Tokenizer must provide the <audio_0> token to encode speech ids.")
        self.audio_token_offset = audio_token_id

        self.tts_start_id = tokenizer.convert_tokens_to_ids("<tts_start>")
        self.tts_end_id = tokenizer.convert_tokens_to_ids("<tts_end>")
        self.think_start_id = tokenizer.convert_tokens_to_ids("<|THINK_START|>")
        self.think_end_id = tokenizer.convert_tokens_to_ids("<|THINK_END|>")

        self.examples: List[Dict] = []

        self._build_examples()

    def _build_examples(self) -> None:
        for convo_idx, record in enumerate(self.data):
            turns = record.get("conversations")
            if not isinstance(turns, Sequence) or len(turns) < 2:
                continue

            for turn_idx, turn in enumerate(turns):
                try:
                    role = self._normalize_role(turn.get("role"))
                except ValueError as exc:
                    LOGGER.warning(
                        "Conversation %s turn %s has unsupported role: %s",
                        convo_idx,
                        turn_idx,
                        exc,
                    )
                    continue
                if role != "assistant":
                    continue
                if turn_idx == 0:
                    LOGGER.warning(
                        "Conversation %s assistant turn %s lacks preceding context; skipping.",
                        convo_idx,
                        turn_idx,
                    )
                    continue

                user_turn = turns[turn_idx - 1]
                try:
                    if self._normalize_role(user_turn.get("role")) != "human":
                        LOGGER.warning(
                            "Conversation %s assistant turn %s is not preceded by a human turn; skipping.",
                            convo_idx,
                            turn_idx,
                        )
                        continue
                except ValueError as exc:
                    LOGGER.warning(
                        "Conversation %s user turn before assistant %s has unsupported role: %s",
                        convo_idx,
                        turn_idx,
                        exc,
                    )
                    continue

                sample = self._build_single_sample(
                    turns=turns,
                    user_index=turn_idx - 1,
                    assistant_index=turn_idx,
                    conversation_index=convo_idx,
                )
                if sample is not None:
                    self.examples.append(sample)

    def _build_single_sample(
        self,
        turns: Sequence[Dict],
        user_index: int,
        assistant_index: int,
        conversation_index: int,
    ) -> Optional[Dict]:
        segments: List[Dict] = []
        audio_mels: List[torch.Tensor] = []

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
        self._append_user_turn(segments, audio_mels, user_turn)

        assistant_turn = turns[assistant_index]
        prev_user_type = str(user_turn.get("type") or "text").lower()
        self._append_assistant_target(segments, assistant_turn, prev_user_type)

        tokenized = self._segments_to_tensors(segments)
        if tokenized is None:
            return None

        input_ids, labels = tokenized
        if self.max_length is not None and input_ids.size(0) > self.max_length:
            input_ids = input_ids[-self.max_length:]
            labels = labels[-self.max_length:]

        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        sample = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "audio_mels": audio_mels,
        }
        return sample

    def _append_context_turn(self, segments: List[Dict], turn: Dict) -> None:
        role = self._normalize_role(turn.get("role"))
        text = _to_text(turn.get("content"))
        _append_text_segment(segments, f"<|BOT|>{role}\n", trainable=False)
        _append_text_segment(segments, text, trainable=False)
        _append_text_segment(segments, "<|EOT|>", trainable=False)

    def _append_user_turn(
        self,
        segments: List[Dict],
        audio_mels: List[torch.Tensor],
        turn: Dict,
    ) -> None:
        _append_text_segment(segments, "<|BOT|>human\n", trainable=False)
        turn_type = str(turn.get("type") or "text").lower()
        wav_path = turn.get("audio-path")
        if turn_type == "audio" and wav_path:
            resolved = self._resolve_audio_path(wav_path)
            mel_chunks = _chunk_audio_to_mels(resolved, self.audio_chunk_size)
            if not mel_chunks:
                LOGGER.warning("User audio %s produced no mel slices; falling back to transcript.", wav_path)
                _append_text_segment(segments, _to_text(turn.get("content")), trainable=False)
            else:
                for mel in mel_chunks:
                    audio_mels.append(mel)
                    patch_count = compute_token_num(mel.shape[1])
                    payload = "<audio_start>" + ("<audio_patch>" * patch_count) + "<audio_end>"
                    _append_text_segment(segments, payload, trainable=False)
        else:
            if turn_type == "audio" and not wav_path:
                LOGGER.warning("User turn marked as audio but missing 'audio-path'; using transcript instead.")
            _append_text_segment(segments, _to_text(turn.get("content")), trainable=False)
        _append_text_segment(segments, "<|EOT|>", trainable=False)

    def _append_assistant_target(
        self,
        segments: List[Dict],
        turn: Dict,
        prev_user_type: str,
    ) -> None:
        _append_text_segment(segments, "<|BOT|>assistant\n", trainable=False)
        turn_type = str(turn.get("type") or "text").lower()
        text_value = _to_text(turn.get("content"))
        wav_path = turn.get("audio-path")

        self._append_think(segments, _to_text(turn.get("think")))

        if turn_type == "audio" and wav_path:
            if self.speech_tokenizer is None:
                raise ValueError("Speech tokenizer is required for assistant speech targets.")
            resolved_path = self._resolve_audio_path(wav_path)
            speech_tokens = self.speech_tokenizer.encode(resolved_path)
            if self.tts_start_id is None or self.tts_start_id < 0:
                raise ValueError("Tokenizer must provide the <tts_start> token for speech responses.")
            _append_token_segment(segments, [self.tts_start_id], trainable=True)
            if self.include_transcript_for_speech and text_value:
                transcript_tokens = self.tokenizer(
                    text_value,
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_tensors=None,
                )["input_ids"]
            else:
                transcript_tokens = self.tokenizer(
                    text_value or "",
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_tensors=None,
                )["input_ids"] if text_value else []
            audio_token_ids = [self.audio_token_offset + tid for tid in speech_tokens]
            interleaved = _interleave_token_ids(transcript_tokens, audio_token_ids, self.tts_pad_id)
            if interleaved:
                _append_token_segment(segments, interleaved, trainable=True)
            else:
                _append_token_segment(segments, audio_token_ids, trainable=True)
            if self.tts_end_id is not None and self.tts_end_id >= 0:
                _append_token_segment(segments, [self.tts_end_id], trainable=True)
            else:
                _append_text_segment(segments, "<tts_end>", trainable=True)
        elif turn_type == "tool":
            if prev_user_type == "audio":
                if self.tts_start_id is None or self.tts_start_id < 0:
                    raise ValueError("Tokenizer must provide the <tts_start> token for tool responses.")
                _append_token_segment(segments, [self.tts_start_id], trainable=True)
                if self.tts_end_id is not None and self.tts_end_id >= 0:
                    _append_token_segment(segments, [self.tts_end_id], trainable=True)
                else:
                    _append_text_segment(segments, "<tts_end>", trainable=True)
                if text_value:
                    _append_text_segment(segments, text_value, trainable=True)
            else:
                if text_value:
                    _append_text_segment(segments, text_value, trainable=True)
        else:
            if turn_type == "audio" and not wav_path:
                LOGGER.warning("Assistant turn marked as audio but missing 'audio-path'; falling back to text target.")
            if text_value:
                _append_text_segment(segments, text_value, trainable=True)

    def _segments_to_tensors(
        self,
        segments: List[Dict],
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        input_ids: List[int] = []
        labels: List[int] = []

        for segment in segments:
            if segment["type"] == "text":
                ids = self.tokenizer(
                    segment["value"],
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_tensors=None,
                )["input_ids"]
            elif segment["type"] == "ids":
                ids = segment["value"]
            else:
                raise ValueError(f"Unknown segment type: {segment['type']}")
            input_ids.extend(ids)
            labels.extend(ids if segment["trainable"] else [-100] * len(ids))

        if not input_ids or all(l == -100 for l in labels):
            return None
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    def _tokenize_text(self, text: str) -> List[int]:
        if not text:
            return []
        return self.tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_tensors=None,
        )["input_ids"]

    def _append_think(self, segments: List[Dict], think_text: str) -> None:
        inner = self._extract_think_inner(think_text)
        if self.think_start_id is not None and self.think_start_id >= 0:
            _append_token_segment(segments, [self.think_start_id], trainable=True)
        else:
            if think_text:
                _append_text_segment(segments, "<|THINK_START|>", trainable=True)
        if inner:
            ids = self._tokenize_text(inner)
            if ids:
                _append_token_segment(segments, ids, trainable=True)
        if self.think_end_id is not None and self.think_end_id >= 0:
            _append_token_segment(segments, [self.think_end_id], trainable=True)
        else:
            if think_text:
                _append_text_segment(segments, "<|THINK_END|>", trainable=True)

    @staticmethod
    def _extract_think_inner(text: str) -> str:
        if not text:
            return ""
        s = text.strip()
        start = "<|THINK_START|>"
        end = "<|THINK_END|>"
        if s.startswith(start) and s.endswith(end):
            return s[len(start): -len(end)].strip()
        return s

    def _resolve_audio_path(self, relative_path: str) -> str:
        if os.path.isabs(relative_path):
            return relative_path
        return os.path.abspath(os.path.join(self.audio_root, relative_path))

    @staticmethod
    def _normalize_role(raw_role: Optional[str]) -> str:
        if raw_role is None:
            raise ValueError("Turn is missing 'role'.")
        key = str(raw_role).lower()
        if key not in ROLE_MAP:
            raise ValueError(f"Unsupported role: {raw_role}")
        return ROLE_MAP[key]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        return self.examples[idx]


class S2SDataCollator:
    """Pads text tensors and packs audio mels -> (wavs, wav_lens) for the encoder."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer must expose pad_token_id for batching.")

        batch_input_ids = torch.nn.utils.rnn.pad_sequence(
            [f["input_ids"] for f in features],
            batch_first=True,
            padding_value=pad_token_id,
        )
        batch_labels = torch.nn.utils.rnn.pad_sequence(
            [f.get("labels", torch.full_like(f["input_ids"], fill_value=-100)) for f in features],
            batch_first=True,
            padding_value=-100,
        )
        batch_attention = torch.nn.utils.rnn.pad_sequence(
            [f["attention_mask"] for f in features],
            batch_first=True,
            padding_value=0,
        )

        all_mels: List[torch.Tensor] = []
        for feature in features:
            all_mels.extend(feature.get("audio_mels", []))

        batch = {
            "input_ids": batch_input_ids,
            "labels": batch_labels,
            "attention_mask": batch_attention,
        }
        if all_mels:
            wavs, wav_lens = padding_mels(all_mels)
            batch["wavs"] = wavs
            batch["wav_lens"] = wav_lens
        return batch


def load_conversations_from_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)
