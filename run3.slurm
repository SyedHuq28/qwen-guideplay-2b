#!/bin/bash
#SBATCH --job-name=train
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=syed-mahbubul.huq.2@city.ac.uk
#SBATCH --partition=preemptgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100_80g:2
#SBATCH -o outputs/%x_%j.o
#SBATCH -e outputs/%x_%j.e

set -euo pipefail

PROJECT_DIR=/users/adhp263/dev/play/lm_playschool_training
MODEL_PATH=/mnt/scratch/users/adhp263/models/Qwen3.5-2B

SFT_OUT=/mnt/scratch/users/adhp263/runs/qwen35-2b-playpen-sft-success

WEIGHTED_OUT=/mnt/scratch/users/adhp263/runs/qwen35-2b-playpen-weighted-turn-sft-050
WEIGHTED_MERGED=/mnt/scratch/users/adhp263/runs/qwen35-2b-playpen-weighted-turn-sft-050-merged

TURN_DATA=/mnt/scratch/users/adhp263/data/playpen_success_turns.jsonl

mkdir -p outputs

source /users/adhp263/sharedscratch/miniconda3/etc/profile.d/conda.sh
conda activate diplomacy_env

cd "${PROJECT_DIR}"

if [ -f "${PROJECT_DIR}/.venv/bin/activate" ]; then
  source "${PROJECT_DIR}/.venv/bin/activate"
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/mnt/scratch/users/adhp263/hf_cache
export HF_DATASETS_CACHE=/mnt/scratch/users/adhp263/hf_cache/datasets
export TRANSFORMERS_CACHE=/mnt/scratch/users/adhp263/hf_cache/transformers
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "=========================================="
echo "Weighted turn SFT 0.50 epoch"
echo "=========================================="
echo "PROJECT_DIR: ${PROJECT_DIR}"
echo "MODEL_PATH: ${MODEL_PATH}"
echo "SFT_OUT: ${SFT_OUT}"
echo "WEIGHTED_OUT: ${WEIGHTED_OUT}"
echo "WEIGHTED_MERGED: ${WEIGHTED_MERGED}"
echo "TURN_DATA: ${TURN_DATA}"
echo "=========================================="

echo "Checking required files..."
test -d "${MODEL_PATH}"
test -d "${SFT_OUT}"
test -f "${TURN_DATA}"
test -f scripts/train_weighted_turn_sft.py
test -f scripts/merge_lora_fix_qwen_generation.py

echo "Safety check: this should NOT point to old 0.25 output."
if [ "${WEIGHTED_OUT}" = "/mnt/scratch/users/adhp263/runs/qwen35-2b-playpen-weighted-turn-sft" ]; then
  echo "ERROR: WEIGHTED_OUT points to old 0.25 model. Refusing to overwrite."
  exit 1
fi

if [ "${WEIGHTED_MERGED}" = "/mnt/scratch/users/adhp263/runs/qwen35-2b-playpen-weighted-turn-sft-merged" ]; then
  echo "ERROR: WEIGHTED_MERGED points to old 0.25 merged model. Refusing to overwrite."
  exit 1
fi

echo "Removing previous 0.50 output dir, if any..."
rm -rf "${WEIGHTED_OUT}"
rm -rf "${WEIGHTED_MERGED}"

echo "Starting weighted turn SFT 0.50 epoch..."

accelerate launch \
  --num_processes 2 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  scripts/train_weighted_turn_sft.py \
  --model_path "${MODEL_PATH}" \
  --adapter_path "${SFT_OUT}" \
  --train_jsonl "${TURN_DATA}" \
  --output_dir "${WEIGHTED_OUT}" \
  --max_length 1024 \
  --num_train_epochs 0.50 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-5 \
  --bf16

echo "Weighted turn SFT 0.50 complete."

echo "Merging weighted 0.50 model..."

python scripts/merge_lora_fix_qwen_generation.py \
  --base_model "${MODEL_PATH}" \
  --adapter "${WEIGHTED_OUT}" \
  --out "${WEIGHTED_MERGED}"

echo "Patching merged generation_config.json..."

python - <<PY
import json
from pathlib import Path

p = Path("${WEIGHTED_MERGED}") / "generation_config.json"
cfg = json.loads(p.read_text()) if p.exists() else {}

cfg["eos_token_id"] = [248046, 248044]
cfg["pad_token_id"] = 248044

p.write_text(json.dumps(cfg, indent=2))
print(json.dumps(cfg, indent=2))
PY

echo "Done."
echo "Final weighted 0.50 merged model:"
echo "${WEIGHTED_MERGED}"


python fix_qwen_playpen_config.py