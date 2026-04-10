"""Demo agent loop for natural multi-step tool-use conversations."""

import json
from pathlib import Path
from typing import Any, Dict, List

from runtime import DEFAULT_SYSTEM_PROMPT, VoxMind
from token2wav import Token2wav

MODEL_PATH = "/root/autodl-tmp/models/VoxMind"
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
TOKEN2WAV_PATH = f"{MODEL_PATH}/token2wav"
PROMPT_WAV_PATH = ASSETS_DIR / "default_female.wav"
INPUT_MODE = "audio"  # "audio" or "text"

# Extra context passed into system prompt
EXTRA_CONTEXT = {
    "current_city": "Beijing",
    "user_language": "en",
    "current_time": "2026-04-08T10:00:00+08:00",
    "current_date": "2026-04-08",
    "current_timezone": "Asia/Shanghai",
}

# =========================
# Natural user cases
# =========================

SINGLE_TOOL_CASE = {
    "id": 1,
    "title": "Daily weather question",
    "user": (
        "I'm heading out later and want to dress appropriately. "
        "What's the weather like in Beijing today?"
    ),
    "audio": "Daily weather question.wav",
}

DUAL_TOOL_CASE = {
    "id": 2,
    "title": "Trip planning request",
    "user": (
        "I'm planning to go to Shanghai tomorrow and spend three days there starting the day after. "
        "However, I haven't booked anything yet. Could you help me book a flight from Beijing to Shanghai and find me a place to stay near the Bund?"
    ),
    "audio": "Trip planning request.wav",
}

PARALLEL_SAME_TOOL_CASE = {
    "id": 3,
    "title": "City comparison question",
    "user": (
        "I'm deciding between Beijing, Shanghai, and Shenzhen for a short trip. "
        "How does today's weather compare across those three cities?"
    ),
    "audio": "City comparison question.wav",
}

MISSING_TOOL_CASE = {
    "id": 4,
    "title": "Request outside current tool set",
    "user": (
        "Please turn on the living room lights. " ),
    "audio": "Request outside current tool set.wav",
}

SUMMARY_CASE = {
    "id": 5,
    "title": "Missing-tool question with follow-up reasoning",
    "user": (
        "Please turn on the living room lights."
    ),
    "audio": "Request outside current tool set.wav",
}

WEATHER_AGENT_CASE = {
    "id": 6,
    "title": "Weather tool execution with final spoken answer",
    "user": (
        "Could you check today's weather in Beijing and then tell me whether I should bring a jacket?"
    ),
    "audio": "Daily weather question.wav",
}

SINGLE_STEP_CASES = [
    SINGLE_TOOL_CASE,
    DUAL_TOOL_CASE,
    PARALLEL_SAME_TOOL_CASE,
    MISSING_TOOL_CASE,
]

# =========================
# Mock tool implementations
# =========================

TOOL_IMPLEMENTATIONS = {
    "Get Weather": lambda arguments: {
        "city": arguments.get("city", "Unknown"),
        "temperature": {
            "Beijing": 21,
            "Shanghai": 24,
            "Shenzhen": 27,
        }.get(arguments.get("city", "Unknown"), 22),
        "condition": {
            "Beijing": "Sunny",
            "Shanghai": "Cloudy",
            "Shenzhen": "Light Rain",
        }.get(arguments.get("city", "Unknown"), "Sunny"),
        "advice": {
            "Beijing": "Good for outdoor activities.",
            "Shanghai": "Carry a light jacket.",
            "Shenzhen": "Bring a compact umbrella.",
        }.get(arguments.get("city", "Unknown"), "Weather is stable."),
    },
    "Search Flights": lambda arguments: {
        "origin": arguments.get("origin", ""),
        "destination": arguments.get("destination", ""),
        "date": arguments.get("date", ""),
        "options": [
            {"flight_no": "MU5101", "depart": "08:30", "arrive": "10:45", "price": 920},
            {"flight_no": "CA1885", "depart": "13:10", "arrive": "15:25", "price": 980},
        ],
    },
    "Search Hotels": lambda arguments: {
        "city": arguments.get("city", ""),
        "district": arguments.get("district", ""),
        "check_in": arguments.get("check_in", ""),
        "check_out": arguments.get("check_out", ""),
        "options": [
            {"name": "Bund View Hotel", "price_per_night": 680, "rating": 4.6},
            {"name": "Riverside Boutique Hotel", "price_per_night": 520, "rating": 4.4},
        ],
    },
    "searchTools": lambda arguments: {
        "status": "missing-capability",
        "requested_need": "turn on the living room lights",
        "suggested_tools": [
            {
                "name": "Turn On Light",
                "description": "Turn on lights for a target room or device. Use this tool when the user asks to switch on lights, set brightness, or choose a color temperature for indoor lighting. The tool returns the target area, power state, brightness, color temperature, and a short execution summary.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "room": {
                            "type": "string",
                            "description": "Target room or zone for the light, for example living room or bedroom."
                        },
                        "device": {
                            "type": "string",
                            "description": "Optional light name or device identifier if the user refers to a specific lamp or fixture."
                        },
                        "brightness": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 100,
                            "description": "Brightness level as a percentage from 1 to 100."
                        },
                        "color_temperature": {
                            "type": "string",
                            "enum": ["warm", "neutral", "cool"],
                            "description": "Preferred white light temperature. Use warm for cozy lighting, neutral for balanced lighting, and cool for crisp lighting."
                        }
                    },
                    "required": ["room"]
                }
            }
        ],
        "note": "The current tool set does not include a light-control tool, so this response proposes the closest missing tool schema.",
    },
}

# =========================
# Load tools.json
# =========================

def load_tools() -> list[dict]:
    """Load tool schema definitions used by the demo."""
    tool_path = Path(__file__).resolve().parent / "tools.json"
    with tool_path.open("r", encoding="utf-8") as f:
        return json.load(f)

# =========================
# Execute tool (mock)
# =========================

def mock_execute_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a local mock tool and return a structured result."""
    executor = TOOL_IMPLEMENTATIONS.get(name)
    if executor is None:
        return {
            "error": f"Unknown tool: {name}",
            "arguments": arguments,
        }
    return executor(arguments)

# =========================
# Utils
# =========================

def build_user_message(case: Dict[str, Any]) -> Dict[str, Any]:
    if INPUT_MODE == "audio":
        audio_filename = case.get("audio")
        if audio_filename:
            audio_path = ASSETS_DIR / audio_filename
            return {
                "role": "user",
                "content": [{"type": "audio", "audio": str(audio_path)}],
            }
    return {"role": "user", "content": case["user"]}


def print_case_header(case: Dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print(f"CASE {case['id']}: {case['title']}")
    if INPUT_MODE == "audio" and case.get("audio"):
        print("user_audio:", ASSETS_DIR / case["audio"])
    else:
        print("user_text:", case["user"])
    print("-" * 80)


def print_case_footer() -> None:
    print("-" * 80)


def append_assistant_message(messages: List[Dict[str, Any]], model: VoxMind, think: str, answer: str) -> None:
    """Append the assistant turn so later rounds can consume tool observations."""
    messages.append(
        {
            "role": "assistant",
            "content": f"{model.THINK_START}\n{think}{model.THINK_END}{answer}",
        }
    )


def save_audio_response(token2wav: Token2wav, audio_tokens: List[int], output_path: Path) -> None:
    """Convert generated audio tokens into wav bytes and save them."""
    if not audio_tokens:
        print("save audio note:\n no audio tokens generated; skipping wav export.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wav_bytes = token2wav(audio_tokens, prompt_wav=str(PROMPT_WAV_PATH))
    with output_path.open("wb") as f:
        f.write(wav_bytes)
    print("saved_audio:", output_path)

# =========================
# Single-step inference
# =========================

def run_single_inference_case(model: VoxMind, system_prompt: str, case: Dict[str, Any]) -> None:
    print_case_header(case)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        build_user_message(case),
    ]

    response = model.generate(
        messages,
        post_think_prefix="After careful reasoning, here is my detailed answer:\n",
        response_mode="speech",
        max_new_tokens=4096,
        temperature=0.6,
        repetition_penalty=1.05,
        top_p=0.9,
        do_sample=True,
    )

    print("think:\n", response.think)
    print("answer:\n", response.answer)
    print("audio_tokens:\n", response.audio_tokens)
    print_case_footer()

# =========================
# Multi-step (tool + summary)
# =========================

def run_summary_case(model: VoxMind, base_tools: List[Dict[str, Any]]) -> None:
    case = SUMMARY_CASE
    print_case_header(case)

    initial_system_prompt = model.build_system_prompt(
        DEFAULT_SYSTEM_PROMPT,
        base_tools,
        extra_context=EXTRA_CONTEXT,
    )
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": initial_system_prompt},
        build_user_message(case),
    ]

    # Step 1: model decides tool calls
    first_response = model.generate(
        messages,
        post_think_prefix="After careful reasoning, here is my detailed answer:\n",
        response_mode="speech",
        max_new_tokens=4096,
        temperature=0.6,
        repetition_penalty=1.05,
        top_p=0.9,
        do_sample=True,
    )

    print("step 1 think:\n", first_response.think)
    print("step 1 answer:\n", first_response.answer)

    tool_calls = model.parse_tool_calls(first_response.answer)

    injected_tools: List[Dict[str, Any]] = []

    # Execute tools and show tool return content for demo purposes
    for index, tool_call in enumerate(tool_calls, start=1):
        observation = {
            "tool": tool_call.get("name", ""),
            "arguments": tool_call.get("arguments", {}),
            "result": mock_execute_tool(tool_call.get("name", ""), tool_call.get("arguments", {})),
        }
        print(f"tool result {index}:\n", json.dumps(observation, ensure_ascii=False, indent=2))

        if tool_call.get("name") == "searchTools":
            injected_tools.extend(observation["result"].get("suggested_tools", []))

    if not injected_tools:
        print("step 2 note:\n no new tools returned from searchTools; skipping follow-up reasoning.")
        print_case_footer()
        return

    print("injected tools:\n", json.dumps(injected_tools, ensure_ascii=False, indent=2))

    # Step 2: rebuild context with newly returned tools and reason again
    updated_tools = base_tools + injected_tools
    updated_system_prompt = model.build_system_prompt(
        DEFAULT_SYSTEM_PROMPT,
        updated_tools,
        extra_context=EXTRA_CONTEXT,
    )
    followup_messages: List[Dict[str, Any]] = [
        {"role": "system", "content": updated_system_prompt},
        build_user_message(case),
    ]

    final_response = model.generate(
        followup_messages,
        post_think_prefix="After careful reasoning, here is my detailed answer:\n",
        response_mode="speech",
        max_new_tokens=4096,
        temperature=0.6,
        repetition_penalty=1.05,
        top_p=0.9,
        do_sample=True,
    )

    print("step 2 think:\n", final_response.think)
    print("step 2 answer:\n", final_response.answer)
    print_case_footer()


def run_weather_tool_voice_case(model: VoxMind, base_tools: List[Dict[str, Any]], token2wav: Token2wav) -> None:
    case = WEATHER_AGENT_CASE
    print_case_header(case)

    system_prompt = model.build_system_prompt(
        DEFAULT_SYSTEM_PROMPT,
        base_tools,
        extra_context=EXTRA_CONTEXT,
    )
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        build_user_message(case),
    ]

    first_response = model.generate(
        messages,
        post_think_prefix="After careful reasoning, here is my detailed answer:\n",
        response_mode="text",
        max_new_tokens=2048,
        temperature=0.6,
        repetition_penalty=1.05,
        top_p=0.9,
        do_sample=True,
    )

    print("step 1 think:\n", first_response.think)
    print("step 1 answer:\n", first_response.answer)

    tool_calls = model.parse_tool_calls(first_response.answer)
    if not tool_calls:
        print("weather tool note:\n no tool call parsed; skipping follow-up answer generation.")
        print_case_footer()
        return

    append_assistant_message(messages, model, first_response.think or "", first_response.answer)

    for index, tool_call in enumerate(tool_calls, start=1):
        observation = {
            "tool": tool_call.get("name", ""),
            "arguments": tool_call.get("arguments", {}),
            "result": mock_execute_tool(tool_call.get("name", ""), tool_call.get("arguments", {})),
        }
        print(f"weather tool result {index}:\n", json.dumps(observation, ensure_ascii=False, indent=2))
        messages.append(model.build_observation_message(observation))

    final_response = model.generate(
        messages,
        post_think_prefix="After careful reasoning, here is my detailed answer:\n",
        response_mode="speech",
        max_new_tokens=4096,
        temperature=0.6,
        repetition_penalty=1.05,
        top_p=0.9,
        do_sample=True,
    )

    print("final think:\n", final_response.think)
    print("final answer:\n", final_response.answer)
    print("final audio_tokens:\n", final_response.audio_tokens)

    output_path = OUTPUT_DIR / "weather_agent_answer.wav"
    save_audio_response(token2wav, final_response.audio_tokens, output_path)
    print_case_footer()

# =========================
# Main
# =========================

def main() -> None:
    print(f"Loading model from: {MODEL_PATH}")
    model = VoxMind(MODEL_PATH)
    token2wav = Token2wav(TOKEN2WAV_PATH)

    print("Loading tools...")
    tools = load_tools()

    system_prompt = model.build_system_prompt(
        DEFAULT_SYSTEM_PROMPT,
        tools,
        extra_context=EXTRA_CONTEXT,
    )

    for case in SINGLE_STEP_CASES:
        run_single_inference_case(model, system_prompt, case)
    run_summary_case(model, tools)
    run_weather_tool_voice_case(model, tools, token2wav)


if __name__ == "__main__":
    main()
