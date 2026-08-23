#!/bin/bash
#SBATCH --job-name=faipd_qr_025
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

# Start from the proven Stage 2 weighted-turn 0.25 model.
BASE_MODEL=/mnt/scratch/users/adhp263/runs/qwen35-2b-playpen-weighted-turn-sft-merged

# New QR outputs. These do not overwrite current best models.
QR_OUT=/mnt/scratch/users/adhp263/runs/qwen35-2b-playpen-faipd-qr-025
QR_MERGED=/mnt/scratch/users/adhp263/runs/qwen35-2b-playpen-faipd-qr-025-merged

# Train split only.
TRAIN_JSONL=/mnt/scratch/users/adhp263/data/playpen_interactions_jsonl/train.jsonl
QR_DATA=/mnt/scratch/users/adhp263/data/faipd_qr_025_mix.jsonl

# Playpen patch/register after merge.
PLAYPEN_DIR=/users/adhp263/dev/play/playpen
PLAYPEN_FIX_SCRIPT=${PLAYPEN_DIR}/fix_qwen_playpen_config.py
PLAYPEN_PYTHON=${PLAYPEN_DIR}/eval_venv/bin/python
PLAYPEN_MODEL_NAME=qwen35-2b-playpen-faipd-qr-025

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
echo "FAIPD-QR 0.25"
echo "Success-only Q-like turns + replay + repair"
echo "=========================================="
echo "BASE_MODEL: ${BASE_MODEL}"
echo "QR_OUT: ${QR_OUT}"
echo "QR_MERGED: ${QR_MERGED}"
echo "TRAIN_JSONL: ${TRAIN_JSONL}"
echo "QR_DATA: ${QR_DATA}"
echo "PLAYPEN_MODEL_NAME: ${PLAYPEN_MODEL_NAME}"
echo "=========================================="

echo "Checking required files..."
test -d "${BASE_MODEL}"
test -f "${TRAIN_JSONL}"
test -f scripts/build_faipd_qr_dataset.py
test -f scripts/train_weighted_turn_sft.py
test -f scripts/merge_lora_fix_qwen_generation.py

echo "Safety checks..."

if echo "${TRAIN_JSONL}" | grep -Eiq "validation|val\.|test|playpen-eval"; then
  echo "ERROR: TRAIN_JSONL looks like validation/test/eval data:"
  echo "${TRAIN_JSONL}"
  exit 1
fi

if [ "${QR_MERGED}" = "/mnt/scratch/users/adhp263/runs/qwen35-2b-playpen-faipd-rr-025-merged" ]; then
  echo "ERROR: refusing to overwrite FAIPD-RR best model."
  exit 1
fi

if [ "${QR_MERGED}" = "/mnt/scratch/users/adhp263/runs/qwen35-2b-playpen-weighted-turn-sft-merged" ]; then
  echo "ERROR: refusing to overwrite weighted-turn 0.25 model."
  exit 1
fi

echo "Removing previous QR output dirs, if any..."
rm -rf "${QR_OUT}"
rm -rf "${QR_MERGED}"

echo "Building FAIPD-QR dataset..."

python scripts/build_faipd_qr_dataset.py \
  --train_jsonl "${TRAIN_JSONL}" \
  --out_jsonl "${QR_DATA}" \
  --num_turn 22000 \
  --num_replay 10000 \
  --num_repair 4000 \
  --seed 28

echo "Dataset built:"
wc -l "${QR_DATA}"
head -n 1 "${QR_DATA}"

echo "Preflight schema check..."

python - <<PY
import json
from pathlib import Path

p = Path("${QR_DATA}")
rows = []
with p.open() as f:
    for i, line in enumerate(f):
        if i >= 100:
            break
        rows.append(json.loads(line))

assert rows, "No rows found"
assert all("prompt" in r for r in rows), "Some rows missing prompt"
assert all("completion" in r for r in rows), "Some rows missing completion"
assert all("value" in r for r in rows), "Some rows missing value"
assert all(isinstance(r["prompt"], list) for r in rows), "Some prompts are not lists"
assert all(isinstance(r["completion"], list) for r in rows), "Some completions are not lists"

print("Preflight OK.")
print("Example keys:", rows[0].keys())
print("First value:", rows[0].get("value"))
print("First completion:", rows[0].get("completion"))
PY

echo "Starting FAIPD-QR training..."

accelerate launch \
  --num_processes 2 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  scripts/train_weighted_turn_sft.py \
  --model_path "${BASE_MODEL}" \
  --train_jsonl "${QR_DATA}" \
  --output_dir "${QR_OUT}" \
  --max_length 1024 \
  --num_train_epochs 0.15 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 3e-5 \
  --bf16

echo "FAIPD-QR training complete."

echo "Merging FAIPD-QR adapter..."

python scripts/merge_lora_fix_qwen_generation.py \
  --base_model "${BASE_MODEL}" \
  --adapter "${QR_OUT}" \
  --out "${QR_MERGED}"

echo "Patching generation_config.json..."

python - <<PY
import json
from pathlib import Path

p = Path("${QR_MERGED}") / "generation_config.json"
cfg = json.loads(p.read_text()) if p.exists() else {}
cfg["eos_token_id"] = [248046, 248044]
cfg["pad_token_id"] = 248044
p.write_text(json.dumps(cfg, indent=2))
print(json.dumps(cfg, indent=2))
PY

echo "Registering and patching FAIPD-QR in Playpen..."

test -d "${PLAYPEN_DIR}"
test -f "${PLAYPEN_FIX_SCRIPT}"
test -x "${PLAYPEN_PYTHON}"

cd "${PLAYPEN_DIR}"

"${PLAYPEN_PYTHON}" "${PLAYPEN_FIX_SCRIPT}" \
  --model "${PLAYPEN_MODEL_NAME}=${QR_MERGED}"

echo "Verifying Playpen registry entry..."

"${PLAYPEN_PYTHON}" - <<PYVERIFY
import json
from pathlib import Path

model_name = "${PLAYPEN_MODEL_NAME}"
model_path = "${QR_MERGED}"
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
echo "Final FAIPD-QR merged model:"
echo "${QR_MERGED}"
echo "Registered as:"
echo "${PLAYPEN_MODEL_NAME}"
echo "=========================================="
