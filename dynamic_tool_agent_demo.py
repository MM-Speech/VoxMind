"""Dynamic tool retrieval demo for VoxMind."""

from __future__ import annotations

import json
import os
import threading
import time
from collections import OrderedDict
from http import HTTPStatus
from pathlib import Path
from typing import Any, Dict, List

from runtime import DEFAULT_SYSTEM_PROMPT, VoxMind

try:
    import dashscope
    from dashscope import Generation
except ImportError:
    dashscope = None
    Generation = None

MODEL_PATH = "/root/autodl-tmp/models/VoxMind"
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
INITIAL_TOOLS_PATH = Path(__file__).resolve().parent / "15tools.json"
GLOBAL_TOOLS_PATH = Path(__file__).resolve().parent / "100tools.json"
QWEN_MODEL_NAME = os.getenv("QWEN_MODEL_NAME", "qwen-plus")
LOCAL_TOOL_LIMIT = 15
TOP_K_RETRIEVAL = 5
INPUT_MODE = "audio"

EXTRA_CONTEXT = {"current_city": "Beijing", "user_language": "en"}

TEST_CASE = {
    "id": 100,
    "title": "Dynamic retrieval for music playback",
    "user": "I want to listen to the song 'Hotel California' by the Eagles right now.",
    "audio": "I want to listen to the song 'Hotel California' by the Eagles right now.wav",
}


class ToolCache:
    def __init__(self, tools: List[Dict[str, Any]], limit: int = LOCAL_TOOL_LIMIT):
        self.limit = limit
        self.tools: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        for tool in tools:
            self.add(tool)

    def add(self, tool: Dict[str, Any]) -> None:
        name = tool["name"]
        if name in self.tools:
            self.tools.move_to_end(name)
        self.tools[name] = tool
        while len(self.tools) > self.limit:
            self.tools.popitem(last=False)

    def merge_topk(self, tools: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        injected = []
        for tool in tools[:top_k]:
            name = tool.get("name")
            if not name:
                continue
            existed = name in self.tools
            self.add(tool)
            if not existed:
                injected.append(tool)
        return injected

    def mark_used(self, names: List[str]) -> None:
        for name in names:
            if name in self.tools:
                self.tools.move_to_end(name)

    def to_list(self) -> List[Dict[str, Any]]:
        return list(self.tools.values())


def load_initial_local_tools() -> List[Dict[str, Any]]:
    with INITIAL_TOOLS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_global_tools() -> List[Dict[str, Any]]:
    with GLOBAL_TOOLS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_user_message(case: Dict[str, Any]) -> Dict[str, Any]:
    if INPUT_MODE == "audio":
        return {"role": "user", "content": [{"type": "audio", "audio": str(ASSETS_DIR / case["audio"])}]}
    return {"role": "user", "content": case["user"]}


def build_messages(model: VoxMind, case: Dict[str, Any], tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": model.build_system_prompt(DEFAULT_SYSTEM_PROMPT, tools, extra_context=EXTRA_CONTEXT)},
        build_user_message(case),
    ]


def parse_json_array(text: str) -> List[str]:
    text = text.strip().strip("`")
    l, r = text.find("["), text.rfind("]")
    if l == -1 or r == -1 or r < l:
        return []
    try:
        arr = json.loads(text[l : r + 1])
    except json.JSONDecodeError:
        return []
    return [x for x in arr if isinstance(x, str)] if isinstance(arr, list) else []


def build_qwen_prompt(capability_trace: str, global_tools: List[Dict[str, Any]]) -> str:
    lite = [{"name": t.get("name"), "description": t.get("description"), "parameters": t.get("parameters")} for t in global_tools]
    return (
        "You are the auxiliary retrieval agent in a dynamic tool-management pipeline.\n"
        "Use reasoning-based semantic selection.\n\n"
        f"Capability trace c_t:\n{capability_trace}\n\n"
        f"Global toolset T_all (JSON):\n{json.dumps(lite, ensure_ascii=False)}\n\n"
        "Rules:\n"
        f"- Return ONLY a strict JSON array with at most {TOP_K_RETRIEVAL} exact tool names.\n"
        "- Rank most useful tool at Top-1.\n"
        "- No explanation, no markdown.\n"
    )


def call_qwen_select_tools(capability_trace: str, global_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if Generation is None or dashscope is None:
        raise RuntimeError("dashscope is not installed. Run: pip install dashscope")
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is not set")
    dashscope.api_key = api_key
    resp = Generation.call(model=QWEN_MODEL_NAME, prompt=build_qwen_prompt(capability_trace, global_tools), temperature=0.1, top_p=0.8)
    if resp.status_code != HTTPStatus.OK:
        raise RuntimeError(f"Qwen retrieval failed: {resp}")
    text = getattr(getattr(resp, "output", None), "text", "") or getattr(resp, "output_text", "") or str(resp)
    selected = parse_json_array(text)
    name_to_tool = {t.get("name"): t for t in global_tools}
    return [name_to_tool[name] for name in selected if name in name_to_tool]


def retrieval_worker(capability_trace: str, global_tools: List[Dict[str, Any]], box: Dict[str, Any]) -> None:
    box["start"] = time.time()
    try:
        box["tools"] = call_qwen_select_tools(capability_trace, global_tools)
        box["error"] = None
    except Exception as exc:
        box["tools"] = []
        box["error"] = str(exc)
    box["end"] = time.time()
    box["duration"] = box["end"] - box["start"]


def run_stage1_with_parallel_retrieval(
    model: VoxMind,
    case: Dict[str, Any],
    local_tools: List[Dict[str, Any]],
    global_tools: List[Dict[str, Any]],
) -> Dict[str, Any]:
    messages = build_messages(model, case, local_tools)

    think_prompt = model._build_think_prompt()
    think_messages = model._normalize_messages(messages, assistant_content=think_prompt)
    think_kwargs = {
        "max_new_tokens": 4096,
        "temperature": 0.5,
        "repetition_penalty": 1.05,
        "top_p": 0.9,
        "do_sample": True,
        "stop_strings": [model.THINK_END],
    }
    think_token_ids, think_text, _ = model(think_messages, **think_kwargs)
    think = model._strip_think_markers(think_text)

    capability_trace = think.strip()
    retrieval_box: Dict[str, Any] = {}
    retrieval_thread = threading.Thread(
        target=retrieval_worker,
        args=(capability_trace, global_tools, retrieval_box),
        daemon=True,
    )
    retrieval_thread.start()

    response_prefix = model._build_response_prefix("speech", "After careful reasoning, here is my detailed answer:\n")
    answer_prompt = f"{think_prompt}{think}{model.THINK_END}{response_prefix}"
    answer_messages = model._normalize_messages(messages, assistant_content=answer_prompt)
    answer_kwargs = {
        "max_new_tokens": 4096,
        "temperature": 0.5,
        "repetition_penalty": 1.05,
        "top_p": 0.9,
        "do_sample": True,
    }
    answer_token_ids, answer_text, audio_tokens = model(answer_messages, **answer_kwargs)
    answer = model._strip_think_block(answer_text)
    raw_text = f"{think_prompt}{think}{model.THINK_END}{response_prefix}{answer_text}"

    answer_finished_at = time.time()
    retrieval_thread.join()
    waiting_overhead = max(0.0, retrieval_box.get("end", answer_finished_at) - answer_finished_at)

    stage1_response = {
        "think_token_ids": think_token_ids,
        "answer_token_ids": answer_token_ids,
        "raw_text": raw_text,
        "think": think,
        "answer": answer,
        "audio_tokens": audio_tokens,
    }
    return {
        "messages": messages,
        "response": stage1_response,
        "retrieval": retrieval_box,
        "waiting_overhead": waiting_overhead,
    }


def run_standard_stage(
    model: VoxMind,
    case: Dict[str, Any],
    local_tools: List[Dict[str, Any]],
):
    messages = build_messages(model, case, local_tools)
    return model.generate(
        messages,
        post_think_prefix="After careful reasoning, here is my detailed answer:\n",
        response_mode="speech",
        max_new_tokens=4096,
        temperature=0.6,
        repetition_penalty=1.05,
        top_p=0.9,
        do_sample=True,
    )


def execute_tool_calls(model: VoxMind, response_text: str, cache: ToolCache) -> None:
    calls = model.parse_tool_calls(response_text)
    print("parsed tool calls:", json.dumps(calls, ensure_ascii=False, indent=2))
    used = []
    for call in calls:
        name = call.get("name", "")
        args = call.get("arguments", {})
        used.append(name)
        print(json.dumps({"tool": name, "arguments": args}, ensure_ascii=False, indent=2))
    cache.mark_used(used)


def main() -> None:
    print(f"Loading model from: {MODEL_PATH}")
    model = VoxMind(MODEL_PATH)
    initial_local_tools = load_initial_local_tools()
    global_tools = load_global_tools()
    cache = ToolCache(initial_local_tools)

    print("\n" + "=" * 100)
    print(f"CASE {TEST_CASE['id']}: {TEST_CASE['title']}")
    print("user_audio:", ASSETS_DIR / TEST_CASE["audio"])
    print("initial local tools:", [t["name"] for t in cache.to_list()])
    print("global tool pool size:", len(global_tools))
    print("-" * 100)

    stage1 = run_stage1_with_parallel_retrieval(model, TEST_CASE, cache.to_list(), global_tools)
    stage1_response = stage1["response"]
    retrieval = stage1["retrieval"]

    print("stage 1 think:\n", stage1_response["think"])
    print("stage 1 answer:\n", stage1_response["answer"])
    print("aux retrieval duration:", round(retrieval.get("duration", 0.0), 4), "s")
    print("waiting overhead:", round(stage1["waiting_overhead"], 4), "s")
    if retrieval.get("error"):
        print("retrieval error:", retrieval["error"])

    if "searchTools" not in stage1_response["answer"]:
        print("stage 1 did not request searchTools; stop.")
        return

    selected_tools = retrieval.get("tools", [])
    print("retrieved candidate tools:")
    print(json.dumps(selected_tools, ensure_ascii=False, indent=2))
    injected = cache.merge_topk(selected_tools, TOP_K_RETRIEVAL)
    print("injected tools:", [t["name"] for t in injected])
    print("updated local tools:", [t["name"] for t in cache.to_list()])

    stage2 = run_standard_stage(model, TEST_CASE, cache.to_list())
    print("stage 2 think:\n", stage2.think)
    print("stage 2 answer:\n", stage2.answer)
    execute_tool_calls(model, stage2.answer, cache)


if __name__ == "__main__":
    main()
