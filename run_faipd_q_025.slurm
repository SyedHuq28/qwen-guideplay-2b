#!/bin/bash
#SBATCH --job-name=fa2
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

# Base Qwen and Stage 1 success-SFT adapter.
MODEL_PATH=/mnt/scratch/users/adhp263/models/Qwen3.5-2B
SFT_OUT=/mnt/scratch/users/adhp263/runs/qwen35-2b-playpen-sft-success

# New FAIPD-Q outputs. These do NOT overwrite existing best models.
FAIPD_Q_OUT=/mnt/scratch/users/adhp263/runs/qwen35-2b-playpen-faipd-q-025
FAIPD_Q_MERGED=/mnt/scratch/users/adhp263/runs/qwen35-2b-playpen-faipd-q-025-merged

# TRAIN split only. Do not use validation/test/eval data here.
TRAIN_JSONL=/mnt/scratch/users/adhp263/data/playpen_interactions_jsonl/train.jsonl
FAIPD_Q_DATA=/mnt/scratch/users/adhp263/data/faipd_q_025_mix.jsonl

# Playpen registry patching after merge.
PLAYPEN_DIR=/users/adhp263/dev/play/playpen
PLAYPEN_FIX_SCRIPT=${PLAYPEN_DIR}/fix_qwen_playpen_config.py
PLAYPEN_PYTHON=${PLAYPEN_DIR}/eval_venv/bin/python
PLAYPEN_MODEL_NAME=qwen35-2b-playpen-faipd-q-025

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
echo "FAIPD-Q 0.25"
echo "Outcome-grounded weighted turn SFT"
echo "=========================================="
echo "PROJECT_DIR: ${PROJECT_DIR}"
echo "MODEL_PATH: ${MODEL_PATH}"
echo "SFT_OUT: ${SFT_OUT}"
echo "FAIPD_Q_OUT: ${FAIPD_Q_OUT}"
echo "FAIPD_Q_MERGED: ${FAIPD_Q_MERGED}"
echo "TRAIN_JSONL: ${TRAIN_JSONL}"
echo "FAIPD_Q_DATA: ${FAIPD_Q_DATA}"
echo "=========================================="

echo "Checking required files..."
test -d "${MODEL_PATH}"
test -d "${SFT_OUT}"
test -f "${TRAIN_JSONL}"
test -f scripts/build_faipd_q_dataset.py
test -f scripts/train_weighted_turn_sft.py
test -f scripts/merge_lora_fix_qwen_generation.py

echo "Safety checks..."

if echo "${TRAIN_JSONL}" | grep -Eiq "validation|val\.|test|playpen-eval"; then
  echo "ERROR: TRAIN_JSONL looks like validation/test/eval data:"
  echo "${TRAIN_JSONL}"
  exit 1
fi

if [ "${FAIPD_Q_MERGED}" = "/mnt/scratch/users/adhp263/runs/qwen35-2b-playpen-faipd-rr-025-merged" ]; then
  echo "ERROR: refusing to overwrite current FAIPD-RR best model."
  exit 1
fi

if [ "${FAIPD_Q_MERGED}" = "/mnt/scratch/users/adhp263/runs/qwen35-2b-playpen-weighted-turn-sft-merged" ]; then
  echo "ERROR: refusing to overwrite weighted-turn 0.25 model."
  exit 1
fi

if [ "${FAIPD_Q_MERGED}" = "/mnt/scratch/users/adhp263/runs/qwen35-2b-playpen-weighted-turn-sft-050-merged" ]; then
  echo "ERROR: refusing to overwrite weighted-turn 0.50 model."
  exit 1
fi

echo "Removing previous FAIPD-Q output dirs, if any..."
rm -rf "${FAIPD_Q_OUT}"
rm -rf "${FAIPD_Q_MERGED}"

echo "Building FAIPD-Q dataset from TRAIN split only..."

python scripts/build_faipd_q_dataset.py \
  --train_jsonl "${TRAIN_JSONL}" \
  --out_jsonl "${FAIPD_Q_DATA}" \
  --num_turn 25000 \
  --num_replay 6000 \
  --num_repair 3000 \
  --seed 28

echo "Dataset built:"
wc -l "${FAIPD_Q_DATA}"
head -n 1 "${FAIPD_Q_DATA}"

echo "Checking value fields in first 100 examples..."

python - <<PY
import json
from pathlib import Path

p = Path("${FAIPD_Q_DATA}")
rows = []
with p.open() as f:
    for i, line in enumerate(f):
        if i >= 100:
            break
        rows.append(json.loads(line))

assert rows, "No rows found"
assert all("prompt" in r for r in rows), "Some rows missing prompt"
assert all("completion" in r for r in rows), "Some rows missing completion"
assert all("messages" in r for r in rows), "Some rows missing messages"
assert all(("value" in r or "weight" in r) for r in rows), "Some rows missing value/weight"

print("Preflight OK.")
print("Example keys:", rows[0].keys())
print("First example value:", rows[0].get("value"), "weight:", rows[0].get("weight"))
print("First completion:", rows[0].get("completion"))
PY

echo "Starting FAIPD-Q training..."

accelerate launch \
  --num_processes 2 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  scripts/train_weighted_turn_sft.py \
  --model_path "${MODEL_PATH}" \
  --adapter_path "${SFT_OUT}" \
  --train_jsonl "${FAIPD_Q_DATA}" \
  --output_dir "${FAIPD_Q_OUT}" \
  --max_length 1024 \
  --num_train_epochs 0.25 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-5 \
  --bf16

echo "FAIPD-Q training complete."

echo "Merging FAIPD-Q adapter..."

python scripts/merge_lora_fix_qwen_generation.py \
  --base_model "${MODEL_PATH}" \
  --adapter "${FAIPD_Q_OUT}" \
  --out "${FAIPD_Q_MERGED}"

echo "Patching generation_config.json..."

python - <<PY
import json
from pathlib import Path

p = Path("${FAIPD_Q_MERGED}") / "generation_config.json"
cfg = json.loads(p.read_text()) if p.exists() else {}

cfg["eos_token_id"] = [248046, 248044]
cfg["pad_token_id"] = 248044

p.write_text(json.dumps(cfg, indent=2))
print(json.dumps(cfg, indent=2))
PY

echo "Registering and patching FAIPD-Q model in Playpen..."

test -d "${PLAYPEN_DIR}"
test -f "${PLAYPEN_FIX_SCRIPT}"
test -x "${PLAYPEN_PYTHON}"

cd "${PLAYPEN_DIR}"

"${PLAYPEN_PYTHON}" "${PLAYPEN_FIX_SCRIPT}" \
  --model "${PLAYPEN_MODEL_NAME}=${FAIPD_Q_MERGED}"

echo "Verifying Playpen registry entry..."

"${PLAYPEN_PYTHON}" - <<PYVERIFY
import json
from pathlib import Path

model_name = "${PLAYPEN_MODEL_NAME}"
model_path = "${FAIPD_Q_MERGED}"
registry = Path("${PLAYPEN_DIR}") / "model_registry.json"

data = json.loads(registry.read_text())
items = data.values() if isinstance(data, dict) else data

matches = [m for m in items if m.get("model_name") == model_name]
if not matches:
    raise SystemExit(f"Registry entry not found: {model_name}")

entry = matches[-1]
print(json.dumps(entry, indent=2))

cfg = entry.get("model_config", {})
assert entry.get("huggingface_id") == model_path, entry.get("huggingface_id")
assert cfg.get("premade_chat_template") is True, cfg
assert cfg.get("eos_to_cull") == r"<\|im_end\|>", cfg
assert cfg.get("padding_side") == "left", cfg

print("Playpen registry patch OK.")
PYVERIFY

cd "${PROJECT_DIR}"


echo "=========================================="
echo "Done."
echo "Final FAIPD-Q merged model:"
echo "${FAIPD_Q_MERGED}"
echo "=========================================="
