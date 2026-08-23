#!/bin/bash
#SBATCH --job-name=tr
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=syed-mahbubul.huq.2@city.ac.uk
#SBATCH --partition=preemptgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=18:00:00
#SBATCH --gres=gpu:a100_80g:2
#SBATCH -o outputs/%x_%j.o
#SBATCH -e outputs/%x_%j.e

set -euo pipefail

PROJECT_DIR=/users/adhp263/dev/play/lm_playschool_training

# Start from the proven 0.25 weighted-turn model.
BASE_MODEL=/mnt/scratch/users/adhp263/runs/qwen35-2b-playpen-weighted-turn-sft-merged

# New Stage 3 output paths. These do not overwrite 0.25.
RR_OUT=/mnt/scratch/users/adhp263/runs/qwen35-2b-playpen-faipd-rr-025
RR_MERGED=/mnt/scratch/users/adhp263/runs/qwen35-2b-playpen-faipd-rr-025-merged

SUCCESS_JSONL=/mnt/scratch/users/adhp263/data/playpen_interactions_jsonl/train_success.jsonl
TURN_DATA=/mnt/scratch/users/adhp263/data/playpen_success_turns.jsonl
RR_DATA=/mnt/scratch/users/adhp263/data/faipd_rr_025_mix.jsonl

mkdir -p "${PROJECT_DIR}/outputs"

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
export HF_HUB_OFFLINE=1

echo "=========================================="
echo "FAIPD-RR Stage 3 from weighted-turn 0.25"
echo "=========================================="
echo "PROJECT_DIR: ${PROJECT_DIR}"
echo "BASE_MODEL: ${BASE_MODEL}"
echo "RR_OUT: ${RR_OUT}"
echo "RR_MERGED: ${RR_MERGED}"
echo "SUCCESS_JSONL: ${SUCCESS_JSONL}"
echo "TURN_DATA: ${TURN_DATA}"
echo "RR_DATA: ${RR_DATA}"
echo "=========================================="

echo "Checking required files..."
test -d "${BASE_MODEL}"
test -f "${SUCCESS_JSONL}"
test -f "${TURN_DATA}"
test -f scripts/build_faipd_rr_dataset.py
test -f scripts/train_faipd_rr_sft.py
test -f scripts/merge_lora_fix_qwen_generation.py

echo "Safety check: do not overwrite old best 0.25 model."
if [ "${RR_MERGED}" = "/mnt/scratch/users/adhp263/runs/qwen35-2b-playpen-weighted-turn-sft-merged" ]; then
  echo "ERROR: RR_MERGED points to old 0.25 model. Refusing."
  exit 1
fi

echo "Removing previous FAIPD-RR output dirs, if any..."
rm -rf "${RR_OUT}"
rm -rf "${RR_MERGED}"

echo "Building FAIPD-RR replay + repair dataset..."

python scripts/build_faipd_rr_dataset.py \
  --success_jsonl "${SUCCESS_JSONL}" \
  --turn_jsonl "${TURN_DATA}" \
  --out_jsonl "${RR_DATA}" \
  --num_replay 8000 \
  --num_turn 20000 \
  --num_repair 6000 \
  --seed 28

echo "Dataset built:"
wc -l "${RR_DATA}"
head -n 1 "${RR_DATA}"

echo "Starting FAIPD-RR training..."

accelerate launch \
  --num_processes 2 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  scripts/train_faipd_rr_sft.py \
  --model_path "${BASE_MODEL}" \
  --train_jsonl "${RR_DATA}" \
  --output_dir "${RR_OUT}" \
  --max_length 1024 \
  --num_train_epochs 0.20 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 3e-5 \
  --bf16

echo "FAIPD-RR training complete."

echo "Merging FAIPD-RR adapter into base 0.25 model..."

python scripts/merge_lora_fix_qwen_generation.py \
  --base_model "${BASE_MODEL}" \
  --adapter "${RR_OUT}" \
  --out "${RR_MERGED}"

echo "Patching generation_config.json..."

python - <<PY
import json
from pathlib import Path

p = Path("${RR_MERGED}") / "generation_config.json"
cfg = json.loads(p.read_text()) if p.exists() else {}

cfg["eos_token_id"] = [248046, 248044]
cfg["pad_token_id"] = 248044

p.write_text(json.dumps(cfg, indent=2))
print(json.dumps(cfg, indent=2))
PY

echo "=========================================="
echo "Done."
echo "Final FAIPD-RR merged model:"
echo "${RR_MERGED}"
echo "=========================================="
