#!/bin/bash
#SBATCH --job-name=playpen_weighted
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
WEIGHTED_OUT=/mnt/scratch/users/adhp263/runs/qwen35-2b-playpen-weighted-turn-sft
WEIGHTED_MERGED=/mnt/scratch/users/adhp263/runs/qwen35-2b-playpen-weighted-turn-sft-merged
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

echo "Checking required files..."
test -d "${MODEL_PATH}"
test -d "${SFT_OUT}"
test -f "${TURN_DATA}"
test -f scripts/train_weighted_turn_sft.py
test -f scripts/merge_lora_fix_qwen_generation.py

echo "Removing failed weighted output dir, if any..."
rm -rf "${WEIGHTED_OUT}"

echo "Starting weighted turn SFT..."

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
  --num_train_epochs 0.25 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-5 \
  --bf16

echo "Weighted turn SFT complete."

echo "Merging weighted model..."

python scripts/merge_lora_fix_qwen_generation.py \
  --base_model "${MODEL_PATH}" \
  --adapter "${WEIGHTED_OUT}" \
  --out "${WEIGHTED_MERGED}"

echo "Done."
echo "Final weighted merged model:"
echo "${WEIGHTED_MERGED}"
