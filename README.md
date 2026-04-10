# 🎙️ VoxMind
<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
  <img alt="Speech Agent" src="https://img.shields.io/badge/Speech-Agent-6A5ACD?style=flat-square" />
  <img alt="Tool Calling" src="https://img.shields.io/badge/Tool-Calling-0F9D58?style=flat-square" />
  <img alt="Dynamic Retrieval" src="https://img.shields.io/badge/Dynamic-Retrieval-F57C00?style=flat-square" />
</p>

---

## ✨ Overview

Recent end-to-end spoken dialogue models have made natural voice interaction increasingly practical. However, as user requests become more complex and task-oriented, conversational ability alone is often not enough. To address real-world spoken tasks, these models must be equipped with agentic capabilities such as structured reasoning, tool use, and dynamic access to external functions.

VoxMind is an integrated framework designed to equip end-to-end spoken dialogue models with comprehensive agentic abilities. Built around a **Think-before-Speak** paradigm, VoxMind enables the model to internalize structured reasoning before response generation, which improves planning, tool selection, and spoken answer quality. In addition, to alleviate the latency bottleneck introduced by large-scale tool integration, VoxMind includes a **Multi-Agent Dynamic Tool Management** architecture that asynchronously delegates tool retrieval to an auxiliary agent aligned with the main model’s reasoning trajectory.
---

## 🔥 Highlights

- 🎧 Unified `audio` / `text` input workflow
- 🧠 Built-in reasoning structure with `<|THINK_START|> ... <|THINK_END|>`
- 🛠️ Structured parsing from `<tool_call>...</tool_call>` blocks
- 🔁 Multi-round observation feedback and follow-up reasoning
- 🔎 Missing-tool discovery and dynamic tool injection
- ⚡ Parallel retrieval design to reduce dynamic-agent latency
- 🏋️ Includes training-related scripts and launcher template

---

## 🗂️ Project Structure

```text
voxmind/
├── assets/                      # demo audio files
├── runtime/                     # runtime implementation
│   ├── model.py                 # VoxMind model wrapper
│   ├── response.py              # VoxMindResponse definition
│   └── prompts.py               # default system prompt
├── scripts/                     # training-related scripts
│   ├── think_train.py           # training entry
│   ├── think_dataset.py         # training dataset processing
│   └── think_dataset_s2s.py     # seq2seq dataset processing
├── think.sh                     # example multi-GPU training launcher
├── tools.json                   # base tool definitions for agent_demo.py
├── 15tools.json                 # initial local tool cache for dynamic demo
├── 100tools.json                # global tool pool for retrieval demo
├── agent_demo.py                # fixed toolset multi-case demo
├── dynamic_tool_agent_demo.py   # dynamic retrieval + cache injection demo
└── README.md
```

---

## 🤗 Model & Datasets

### Model

- 🤗 Hugging Face: [`leungtianle/VoxMind`](https://huggingface.co/leungtianle/VoxMind)

### Training data

- 🤗 JSONL annotations: [`leungtianle/VoxMind-jsonl`](https://huggingface.co/datasets/leungtianle/VoxMind-jsonl)
- 🔷 ModelScope speech data: [`BEISHUI/AgentChat`](https://www.modelscope.cn/datasets/BEISHUI/AgentChat)

These resources correspond to the released model weights, structured JSONL data, and speech-side training assets used in the VoxMind pipeline.

---

## 🚀 Installation

Before running the demos or training scripts, prepare your environment first.

### Step 1: Create environment

```bash
conda create -n voxmind python=3.10 -y
conda activate voxmind
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

If some runtime components or local model wrappers have extra dependencies in your environment, install them separately as needed.

### Step 3: Prepare local model files

Both demos rely on a local VoxMind model directory. Please update the model path in the scripts according to your own machine.

---

## 🎯 Inference

VoxMind currently provides two main demo scripts.

### Part 1: Fixed-tool agent reasoning

`agent_demo.py` demonstrates a standard fixed-tool workflow for speech / text reasoning agents.

#### Covered scenarios

- single tool call
- multi-tool decomposition in one request
- repeated use of the same tool with different arguments
- request outside the current toolset
- missing-tool suggestion followed by second-round reasoning

#### Built-in mock tools

The demo contains several local mock tools, including:

- `Get Weather`
- `Search Flights`
- `Search Hotels`
- `searchTools`

Among them, `searchTools` is used to simulate the case where the current tool inventory is insufficient and the model needs a missing capability to continue.

#### Run

```bash
python agent_demo.py
```

---

### Part 2: Dynamic tool retrieval

`dynamic_tool_agent_demo.py` demonstrates a more realistic dynamic tool-management pipeline.

Instead of exposing a full tool universe to the model at once, the script uses:

- a **small local cache** (`15tools.json`)
- a **large global tool pool** (`100tools.json`)
- an **auxiliary retrieval model** to recall the most relevant missing tools

#### Core idea

1. the model reasons using a limited local tool window
2. the reasoning trace becomes a signal
3. an auxiliary model retrieves top candidate tools from the global pool
4. the retrieved tools are injected into the local cache
5. the model performs a second reasoning stage with the updated tool context

#### Key component: `ToolCache`

`ToolCache` is implemented with `OrderedDict` and is responsible for:

- maintaining local tool-window size
- refreshing recency after use
- injecting only newly retrieved tools
- evicting older tools when capacity is exceeded

#### Parallel retrieval

A notable feature is that retrieval runs in parallel with first-stage answer generation:

1. produce think trace
2. start retrieval thread using the capability trace
3. continue answer generation
4. wait for retrieval completion
5. inject top-k tools if necessary

This makes the script a useful reference for latency-aware dynamic agents.

#### Run

```bash
python dynamic_tool_agent_demo.py
```

---

## ⚙️ Configuration

### Input mode

Both demos default to audio input:

```python
INPUT_MODE = "audio"
```

If you want plain-text testing, change it to:

```python
INPUT_MODE = "text"
```

### Extra context

Both scripts inject runtime metadata through `extra_context`, for example:

```python
EXTRA_CONTEXT = {
    "current_city": "Beijing",
    "user_language": "en",
}
```

This is useful when building system prompts with current context information.

---

## 🔑 DashScope API Key

Please configure your own key locally.

### Linux / macOS

```bash
export DASHSCOPE_API_KEY="your_api_key_here"
python dynamic_tool_agent_demo.py
```

### Windows PowerShell

```powershell
$env:DASHSCOPE_API_KEY="your_api_key_here"
python dynamic_tool_agent_demo.py
```

### Optional model override

```bash
export QWEN_MODEL_NAME="qwen-plus"
```

Recommended usage in code:

```python
api_key = os.getenv("DASHSCOPE_API_KEY")
```

---

## 🏋️ Training

This repository also contains training-related scripts and a launcher template.

### Training files

- `scripts/think_train.py` — training entry script
- `scripts/think_dataset.py` — dataset loading / preprocessing
- `scripts/think_dataset_s2s.py` — seq2seq-style dataset handling
- `think.sh` — example distributed training launcher

### Step 1: Prepare your dataset

Prepare your own training JSONL file and corresponding audio directory according to your local project layout.

### Step 2: Modify `think.sh`

Before training, open `think.sh` and manually fill in all required paths for your own environment:

```bash
ROOT_DIR=""
MODEL_DIR=""
TOKEN2WAV_DIR=""
DATASET_PATH=""
AUDIO_ROOT=""
OUTPUT_DIR=""
LOG_DIR=""
DEEPSPEED_CONFIG=""
```

### Step 3: Start training

After completing the path configuration, run:

```bash
bash think.sh
```

### Notes

- `think.sh` no longer contains private hardcoded absolute paths.
- Please manually configure all local paths according to your machine.
- If you are not using multi-GPU training, you can simplify the launcher further.

---

## 🧪 Minimal Usage Example

```python
from runtime import DEFAULT_SYSTEM_PROMPT, VoxMind

model = VoxMind("/path/to/VoxMind")

tools = []
system_prompt = model.build_system_prompt(
    DEFAULT_SYSTEM_PROMPT,
    tools,
    extra_context={"current_city": "Beijing", "user_language": "en"},
)

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What's the weather like in Beijing today?"},
]

response = model.generate(
    messages,
    post_think_prefix="After careful reasoning, here is my detailed answer:\n",
    max_new_tokens=512,
    temperature=0.6,
    top_p=0.9,
    do_sample=True,
)

print(response.think)
print(response.answer)
print(model.parse_tool_calls(response.answer))
```

---

## 📚 Citation

If this repository or its workflow design is helpful to your research, please cite or reference it appropriately.

<!-- ```bibtex

``` -->

