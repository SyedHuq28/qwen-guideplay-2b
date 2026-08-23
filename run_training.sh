#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/workspace
MODEL_PATH=/dev/shm/models/Qwen3.5-2B

# Stage 1 success SFT adapter
SFT_OUT=/workspace/qwen35-2b-playpen-sft-success

# Original Stage 2 weighted-turn data
STAGE2_DATA=/workspace/playpen_success_turns.jsonl

# LLM-generated files
LLM_JUDGED_DATA=/workspace/llm_judged_stage2_turns.jsonl
LLM_REPAIR_DATA=/workspace/llm_format_repair.jsonl

# Final mixed dataset
MIXED_DATA=/workspace/llm_faipd_rr_lite_30000_1000_100.jsonl

# Output model dirs
OUT_DIR=/workspace/qwen35-2b-playpen-llm-faipd-rr-lite-30000-1000-100
MERGED_OUT=/workspace/qwen35-2b-playpen-llm-faipd-rr-lite-30000-1000-100-merged

cd "${PROJECT_DIR}"
mkdir -p outputs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "Disk check:"
df -h / /workspace /dev/shm || true

echo "Checking required files..."
test -d "${MODEL_PATH}"
test -d "${SFT_OUT}"
test -f "${STAGE2_DATA}"
test -f "${LLM_JUDGED_DATA}"
test -f "${LLM_REPAIR_DATA}"
test -f scripts/build_llm_faipd_rr_lite_dataset.py
test -f scripts/train_weighted_turn_sft.py
test -f scripts/merge_lora_fix_qwen_generation.py

echo "Checking GPU visibility..."
GPU_COUNT=$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)

echo "CUDA GPU count: ${GPU_COUNT}"

if [ "${GPU_COUNT}" -lt 1 ]; then
  echo "No CUDA GPU visible. Stop."
  exit 1
fi

# Use at most 2 GPUs, but do not request 2 if only 1 is visible.
if [ "${GPU_COUNT}" -ge 2 ]; then
  NUM_PROCESSES=2
else
  NUM_PROCESSES=1
fi

echo "Using accelerate num_processes=${NUM_PROCESSES}"

echo "Building LLM-FAIPD-RR-lite mixed dataset..."

python scripts/build_llm_faipd_rr_lite_dataset.py \
  --stage2_jsonl "${STAGE2_DATA}" \
  --llm_judged_jsonl "${LLM_JUDGED_DATA}" \
  --llm_repair_jsonl "${LLM_REPAIR_DATA}" \
  --out_jsonl "${MIXED_DATA}" \
  --num_stage2 30000 \
  --num_llm_judged 1000 \
  --num_llm_repair 100 \
  --min_llm_score 0.70 \
  --exclude_stage2_rows_used_by_llm \
  --seed 28

echo "Checking mixed dataset..."
test -f "${MIXED_DATA}"
wc -l "${MIXED_DATA}"

echo "Removing failed output dir, if any..."
rm -rf "${OUT_DIR}"

echo "Starting LLM-FAIPD-RR-lite weighted SFT..."

accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  scripts/train_weighted_turn_sft.py \
  --model_path "${MODEL_PATH}" \
  --adapter_path "${SFT_OUT}" \
  --train_jsonl "${MIXED_DATA}" \
  --output_dir "${OUT_DIR}" \
  --max_length 1024 \
  --num_train_epochs 0.25 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-5 \
  --bf16

echo "Training complete."

echo "Merging LLM-FAIPD-RR-lite model..."

rm -rf "${MERGED_OUT}"

python scripts/merge_lora_fix_qwen_generation.py \
  --base_model "${MODEL_PATH}" \
  --adapter "${OUT_DIR}" \
  --out "${MERGED_OUT}"

echo "Done."
echo "Final merged model:"
echo "${MERGED_OUT}"

echo "Final disk check:"
df -h / /workspace /dev/shm || true