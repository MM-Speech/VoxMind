#!/usr/bin/env bash
# run_sft_zero3_format.sh
# Step-Audio-2 formatter SFT (tool-call emits <tts_end> then <tool_call> without speech preamble) with DeepSpeed ZeRO-3
set -euo pipefail

# Please set these paths to match your own local environment before running.
ROOT_DIR=""
MODEL_DIR=""
TOKEN2WAV_DIR=""
DATASET_PATH=""
AUDIO_ROOT=""
OUTPUT_DIR=""
LOG_DIR=""
DEEPSPEED_CONFIG=""

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

export PYTHONPATH="${ROOT_DIR}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_IB_DISABLE=1
export PYTHONDONTWRITEBYTECODE=1
find "${ROOT_DIR}/scripts" -name "__pycache__" -type d -exec rm -rf {} +


BATCH_SIZE=1
GRAD_ACCUM=8
NUM_EPOCHS=2
LEARNING_RATE=1e-5
MAX_LENGTH=27648
LOGGING_STEPS=10
SAVE_STEPS=500
MAX_STEPS=4900
RUN_NAME="tool-think-two2one"
MAX_TARGET_NEW_TOKENS=5120


CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.run --nproc_per_node=2 \
  "${ROOT_DIR}/scripts/think_train.py" \
  --model-name-or-path "${MODEL_DIR}" \
  --dataset-path "${DATASET_PATH}" \
  --audio-root "${AUDIO_ROOT}" \
  --token2wav-path "${TOKEN2WAV_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size ${BATCH_SIZE} \
  --gradient-accumulation-steps ${GRAD_ACCUM} \
  --max-steps ${MAX_STEPS} \
  --freeze-audio \
  --run-name "${RUN_NAME}" \
  --learning-rate ${LEARNING_RATE} \
  --max-length ${MAX_LENGTH} \
  --logging-steps ${LOGGING_STEPS} \
  --lr-scheduler-type cosine \
  --save-steps ${SAVE_STEPS} \
  --report-to "wandb" \
  --logging-dir "${LOG_DIR}" \
  --bf16 \
  --gradient-checkpointing \
  --max_grad_norm 1.0 \
  --weight_decay 0.01 \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --max-target-new-tokens ${MAX_TARGET_NEW_TOKENS} \
  --forward-log-dir "${LOG_DIR}/forward" \
  ${EXTRA_FLAGS:-}
