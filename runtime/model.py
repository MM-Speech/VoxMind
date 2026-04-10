import json
import re
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

try:
    from ..utils import compute_token_num, load_audio, log_mel_spectrogram, padding_mels
except ImportError:
    from utils import compute_token_num, load_audio, log_mel_spectrogram, padding_mels
from .response import VoxMindResponse


class StepAudio2Base:

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = device
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="right",
            use_fast=True,
        )
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        ).to(device).eval()
        self.eos_token_id = self.llm_tokenizer.eos_token_id

    def __call__(self, messages: list, **kwargs):
        messages, mels = self.apply_chat_template(messages)
        prompt_ids = []
        for msg in messages:
            if isinstance(msg, str):
                prompt_ids.append(self.llm_tokenizer(text=msg, return_tensors="pt", padding=True)["input_ids"])
            elif isinstance(msg, list):
                prompt_ids.append(torch.tensor([msg], dtype=torch.int32))
            else:
                raise ValueError(f"Unsupported content type: {type(msg)}")
        prompt_ids = torch.cat(prompt_ids, dim=-1).to(self.device)
        attention_mask = torch.ones_like(prompt_ids)

        if len(mels) == 0:
            mels = None
            mel_lengths = None
        else:
            mels, mel_lengths = padding_mels(mels)
            mels = mels.to(self.device)
            mel_lengths = mel_lengths.to(self.device)

        generate_inputs = {
            "input_ids": prompt_ids,
            "wavs": mels,
            "wav_lens": mel_lengths,
            "attention_mask": attention_mask,
        }

        generation_config = dict(
            max_new_tokens=2048,
            pad_token_id=self.llm_tokenizer.pad_token_id,
            eos_token_id=self.eos_token_id,
        )
        generation_config.update(kwargs)
        generation_config = GenerationConfig(**generation_config)

        outputs = self.llm.generate(
            **generate_inputs,
            generation_config=generation_config,
            tokenizer=self.llm_tokenizer,
        )
        output_token_ids = outputs[0, prompt_ids.shape[-1] : -1].tolist()
        output_text_tokens = [i for i in output_token_ids if i < 151688]
        output_audio_tokens = [i - 151696 for i in output_token_ids if i > 151695]
        output_text = self.llm_tokenizer.decode(output_text_tokens)
        return output_token_ids, output_text, output_audio_tokens

    def apply_chat_template(self, messages: list):
        results = []
        mels = []
        for msg in messages:
            content = msg
            if isinstance(content, str):
                results.append(content)
            elif isinstance(content, dict):
                if content["type"] == "text":
                    results.append(f"{content['text']}")
                elif content["type"] == "audio":
                    audio = load_audio(content["audio"])
                    for i in range(0, audio.shape[0], 16000 * 25):
                        mel = log_mel_spectrogram(audio[i : i + 16000 * 25], n_mels=128, padding=479)
                        mels.append(mel)
                        audio_tokens = "<audio_patch>" * compute_token_num(mel.shape[1])
                        results.append(f"<audio_start>{audio_tokens}<audio_end>")
                elif content["type"] == "token":
                    results.append(content["token"])
                else:
                    raise ValueError(f"Unsupported content type: {content['type']}")
            else:
                raise ValueError(f"Unsupported content type: {type(content)}")
        return results, mels


class StepAudio2(StepAudio2Base):

    def __init__(self, model_path: str, device: str = "cuda", torch_dtype: torch.dtype = torch.bfloat16):
        super().__init__(model_path, device=device, torch_dtype=torch_dtype)
        self.llm_tokenizer.eos_token = "<|EOT|>"
        self.llm.config.eos_token_id = self.llm_tokenizer.convert_tokens_to_ids("<|EOT|>")
        self.eos_token_id = self.llm_tokenizer.convert_tokens_to_ids("<|EOT|>")

    def apply_chat_template(self, messages: list):
        results = []
        mels = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                role = "human"
            if isinstance(content, str):
                text_with_audio = f"<|BOT|>{role}\n{content}"
                text_with_audio += "<|EOT|>" if msg.get("eot", True) else ""
                results.append(text_with_audio)
            elif isinstance(content, list):
                results.append(f"<|BOT|>{role}\n")
                for item in content:
                    if item["type"] == "text":
                        results.append(f"{item['text']}")
                    elif item["type"] == "audio":
                        audio = load_audio(item["audio"])
                        for i in range(0, audio.shape[0], 16000 * 25):
                            mel = log_mel_spectrogram(audio[i : i + 16000 * 25], n_mels=128, padding=479)
                            mels.append(mel)
                            audio_tokens = "<audio_patch>" * compute_token_num(mel.shape[1])
                            results.append(f"<audio_start>{audio_tokens}<audio_end>")
                    elif item["type"] == "token":
                        results.append(item["token"])
                    else:
                        raise ValueError(f"Unsupported content type: {item['type']}")
                if msg.get("eot", True):
                    results.append("<|EOT|>")
            elif content is None:
                results.append(f"<|BOT|>{role}\n")
            else:
                raise ValueError(f"Unsupported content type: {type(content)}")
        return results, mels


class VoxMind(StepAudio2):
    THINK_START = "<|THINK_START|>"
    THINK_END = "<|THINK_END|>"
    TOOL_CALL_START = "<tool_call>"
    TOOL_CALL_END = "</tool_call>"
    TTS_START = "<tts_start>"
    OBSERVATION_PREFIX = "Below is the return content of the function you used:"

    def _build_think_prompt(self) -> str:
        return f"{self.THINK_START}\n"

    def _build_response_prefix(
        self,
        response_mode: Literal["text", "speech"],
        post_think_prefix: str,
    ) -> str:
        if response_mode == "speech":
            return self.TTS_START
        return post_think_prefix

    def _normalize_messages(
        self,
        messages: List[Dict[str, Any]],
        assistant_content: str,
    ) -> List[Dict[str, Any]]:
        normalized = [dict(message) for message in messages]
        if not normalized:
            raise ValueError("messages must not be empty")

        if normalized[-1].get("role") == "assistant":
            normalized[-1]["content"] = assistant_content
            normalized[-1]["eot"] = False
            return normalized

        normalized.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "eot": False,
            }
        )
        return normalized

    def _strip_think_markers(self, text: str) -> str:
        return text.replace(self.THINK_START, "").replace(self.THINK_END, "").strip()

    def _extract_tool_calls(self, text: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        raw_blocks = [
            match.strip()
            for match in re.findall(
                rf"{re.escape(self.TOOL_CALL_START)}\s*(.*?)\s*{re.escape(self.TOOL_CALL_END)}",
                text,
                re.DOTALL,
            )
        ]

        parsed_tool_calls: List[Dict[str, Any]] = []
        for block in raw_blocks:
            try:
                parsed = json.loads(block)
            except json.JSONDecodeError:
                continue

            if isinstance(parsed, list):
                parsed_tool_calls.extend(item for item in parsed if isinstance(item, dict))
            elif isinstance(parsed, dict):
                parsed_tool_calls.append(parsed)

        return parsed_tool_calls, raw_blocks

    def _strip_think_block(self, text: str) -> str:
        stripped = re.sub(
            rf"{re.escape(self.THINK_START)}[\s\S]*?{re.escape(self.THINK_END)}",
            "",
            text,
        )
        return stripped.strip()

    def build_keys_section(self, extra_context: Optional[Dict[str, Any]] = None) -> str:
        now = datetime.now().astimezone()
        timezone_name = now.tzname() or str(now.tzinfo) or "unknown"
        context: Dict[str, Any] = {
            "current_time": now.isoformat(timespec="seconds"),
            "current_date": now.date().isoformat(),
            "current_timezone": timezone_name,
        }
        if extra_context:
            context.update(extra_context)
        return json.dumps(context, ensure_ascii=False, indent=2)

    def build_system_prompt(
        self,
        system_prompt_template: str,
        tools: List[Dict[str, Any]],
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        keys_section = self.build_keys_section(extra_context=extra_context)
        tool_section = json.dumps(tools, ensure_ascii=False, indent=2)
        return system_prompt_template.format(keys_section=keys_section, tool_section=tool_section)

    def build_observation_message(self, observation: Any) -> Dict[str, str]:
        if isinstance(observation, str):
            observation_text = observation
        else:
            observation_text = json.dumps(observation, ensure_ascii=False)
        return {
            "role": "observation",
            "content": f"{self.OBSERVATION_PREFIX}\n{observation_text}",
        }

    def parse_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        parsed_tool_calls, _ = self._extract_tool_calls(text)
        return parsed_tool_calls

    def generate(
        self,
        messages: List[Dict[str, Any]],
        response_mode: Literal["text", "speech"] = "text",
        post_think_prefix: str = "",
        **generation_kwargs,
    ) -> VoxMindResponse:
        think_prompt = self._build_think_prompt()
        think_messages = self._normalize_messages(messages, assistant_content=think_prompt)

        think_generation_kwargs = dict(generation_kwargs)
        think_generation_kwargs["stop_strings"] = [self.THINK_END]
        think_token_ids, think_text, _ = self(think_messages, **think_generation_kwargs)
        think = self._strip_think_markers(think_text)

        response_prefix = self._build_response_prefix(response_mode, post_think_prefix)
        answer_prompt = f"{think_prompt}{think}{self.THINK_END}{response_prefix}"
        answer_messages = self._normalize_messages(messages, assistant_content=answer_prompt)
        answer_token_ids, answer_text, audio_tokens = self(answer_messages, **generation_kwargs)

        answer = self._strip_think_block(answer_text)
        raw_text = f"{think_prompt}{think}{self.THINK_END}{response_prefix}{answer_text}"

        return VoxMindResponse(
            think_token_ids=think_token_ids,
            answer_token_ids=answer_token_ids,
            raw_text=raw_text,
            text=raw_text,
            think=think,
            answer=answer,
            response_prefix=response_prefix,
            audio_tokens=audio_tokens,
        )
