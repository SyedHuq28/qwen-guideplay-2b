#!/bin/bash
#SBATCH --job-name=hem_mix_008
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

# Training anchor: base Qwen + Stage 2 LoRA adapter.
MODEL_PATH=/mnt/scratch/users/adhp263/models/Qwen3.5-2B
STAGE2_ADAPTER=/mnt/scratch/users/adhp263/runs/qwen35-2b-playpen-weighted-turn-sft

# Scoring model: Stage 2 merged model, used only to compute per-example loss.
STAGE2_MERGED=/mnt/scratch/users/adhp263/runs/qwen35-2b-playpen-weighted-turn-sft-merged

# Existing Stage 2 weighted-turn data.
STAGE2_JSONL=/mnt/scratch/users/adhp263/data/playpen_success_turns.jsonl

# HEM-Mix outputs.
HEM_DATA=/mnt/scratch/users/adhp263/data/hem_mix_008.jsonl
HEM_SCORE_CACHE=/mnt/scratch/users/adhp263/data/hem_mix_008_scores.jsonl
HEM_OUT=/mnt/scratch/users/adhp263/runs/qwen35-2b-playpen-hem-mix-008
HEM_MERGED=/mnt/scratch/users/adhp263/runs/qwen35-2b-playpen-hem-mix-008-merged

# Playpen patch/register.
PLAYPEN_DIR=/users/adhp263/dev/play/playpen
PLAYPEN_FIX_SCRIPT=${PLAYPEN_DIR}/fix_qwen_playpen_config.py
PLAYPEN_PYTHON=${PLAYPEN_DIR}/eval_venv/bin/python
PLAYPEN_MODEL_NAME=qwen35-2b-playpen-hem-mix-008

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
echo "HEM-Mix 0.08"
echo "Hard Example Mining + Stage 2 replay"
echo "=========================================="
echo "MODEL_PATH: ${MODEL_PATH}"
echo "STAGE2_ADAPTER: ${STAGE2_ADAPTER}"
echo "STAGE2_MERGED: ${STAGE2_MERGED}"
echo "STAGE2_JSONL: ${STAGE2_JSONL}"
echo "HEM_DATA: ${HEM_DATA}"
echo "HEM_SCORE_CACHE: ${HEM_SCORE_CACHE}"
echo "HEM_OUT: ${HEM_OUT}"
echo "HEM_MERGED: ${HEM_MERGED}"
echo "PLAYPEN_MODEL_NAME: ${PLAYPEN_MODEL_NAME}"
echo "=========================================="

echo "Checking required files..."
test -d "${MODEL_PATH}"
test -d "${STAGE2_ADAPTER}"
test -d "${STAGE2_MERGED}"
test -f "${STAGE2_JSONL}"
test -f scripts/build_hem_mix_dataset.py
test -f scripts/train_weighted_turn_sft.py
test -f scripts/merge_lora_fix_qwen_generation.py

echo "Safety checks..."

if echo "${STAGE2_JSONL}" | grep -Eiq "validation|val\.|test|playpen-eval|eval_results|leaderboard|public"; then
  echo "ERROR: STAGE2_JSONL looks unsafe:"
  echo "${STAGE2_JSONL}"
  exit 1
fi

if [ "${HEM_MERGED}" = "/mnt/scratch/users/adhp263/runs/qwen35-2b-playpen-faipd-rr-025-merged" ]; then
  echo "ERROR: refusing to overwrite FAIPD-RR best model."
  exit 1
fi

if [ "${HEM_MERGED}" = "/mnt/scratch/users/adhp263/runs/qwen35-2b-playpen-weighted-turn-sft-merged" ]; then
  echo "ERROR: refusing to overwrite Stage 2 merged model."
  exit 1
fi

echo "Removing previous HEM output dirs, if any..."
rm -rf "${HEM_OUT}"
rm -rf "${HEM_MERGED}"

echo "Building HEM-Mix dataset..."

python -u scripts/build_hem_mix_dataset.py \
  --stage2_jsonl "${STAGE2_JSONL}" \
  --scoring_model "${STAGE2_MERGED}" \
  --out_jsonl "${HEM_DATA}" \
  --score_cache_jsonl "${HEM_SCORE_CACHE}" \
  --max_length 1024 \
  --total_examples 12000 \
  --hard_ratio 0.70 \
  --min_value 0.70 \
  --min_loss 0.25 \
  --max_loss 2.50 \
  --max_rows_to_score 20000 \
  --seed 28

echo "Dataset built:"
wc -l "${HEM_DATA}"
head -n 1 "${HEM_DATA}"

echo "Preflight schema check..."

python - <<PY
import json
from pathlib import Path
from collections import Counter

p = Path("${HEM_DATA}")
rows = []
with p.open() as f:
    for i, line in enumerate(f):
        if i >= 1000:
            break
        rows.append(json.loads(line))

assert rows, "No rows found"
assert all("prompt" in r for r in rows), "Some rows missing prompt"
assert all("completion" in r for r in rows), "Some rows missing completion"
assert all("value" in r for r in rows), "Some rows missing value"
assert all(isinstance(r["prompt"], list) for r in rows), "Some prompts are not lists"
assert all(isinstance(r["completion"], list) for r in rows), "Some completions are not lists"

print("Preflight OK.")
print("First keys:", rows[0].keys())
print("First value:", rows[0].get("value"))
print("First source:", rows[0].get("source"))
print("Source counts first 1000:")
print(Counter(r.get("source", "unknown") for r in rows))

losses = []
for r in rows:
    meta = r.get("meta", {})
    if isinstance(meta, dict) and meta.get("stage2_loss") is not None:
        losses.append(float(meta["stage2_loss"]))

if losses:
    print("Loss first 1000 min/avg/max:", min(losses), sum(losses)/len(losses), max(losses))
PY

echo "Starting HEM-Mix training..."

accelerate launch \
  --num_processes 2 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  scripts/train_weighted_turn_sft.py \
  --model_path "${MODEL_PATH}" \
  --adapter_path "${STAGE2_ADAPTER}" \
  --train_jsonl "${HEM_DATA}" \
  --output_dir "${HEM_OUT}" \
  --max_length 1024 \
  --num_train_epochs 0.08 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --bf16

echo "HEM-Mix training complete."

echo "Checking HEM adapter output..."

test -f "${HEM_OUT}/adapter_config.json"
test -f "${HEM_OUT}/adapter_model.safetensors"

python - <<PY
import json
from pathlib import Path
p = Path("${HEM_OUT}") / "adapter_config.json"
cfg = json.loads(p.read_text())
print(json.dumps({
    "base_model_name_or_path": cfg.get("base_model_name_or_path"),
    "peft_type": cfg.get("peft_type"),
    "r": cfg.get("r"),
    "lora_alpha": cfg.get("lora_alpha"),
}, indent=2))
PY

echo "Merging HEM-Mix adapter..."

python scripts/merge_lora_fix_qwen_generation.py \
  --base_model "${MODEL_PATH}" \
  --adapter "${HEM_OUT}" \
  --out "${HEM_MERGED}"

echo "Patching generation_config.json..."

python - <<PY
import json
from pathlib import Path

p = Path("${HEM_MERGED}") / "generation_config.json"
cfg = json.loads(p.read_text()) if p.exists() else {}
cfg["eos_token_id"] = [248046, 248044]
cfg["pad_token_id"] = 248044
p.write_text(json.dumps(cfg, indent=2))
print(json.dumps(cfg, indent=2))
PY

echo "Registering and patching HEM-Mix in Playpen..."

test -d "${PLAYPEN_DIR}"
test -f "${PLAYPEN_FIX_SCRIPT}"
test -x "${PLAYPEN_PYTHON}"

cd "${PLAYPEN_DIR}"

"${PLAYPEN_PYTHON}" "${PLAYPEN_FIX_SCRIPT}" \
  --model "${PLAYPEN_MODEL_NAME}=${HEM_MERGED}"

echo "Verifying Playpen registry entry..."

"${PLAYPEN_PYTHON}" - <<PYVERIFY
import json
from pathlib import Path

model_name = "${PLAYPEN_MODEL_NAME}"
model_path = "${HEM_MERGED}"
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
echo "Final HEM-Mix merged model:"
echo "${HEM_MERGED}"
echo "Registered as:"
echo "${PLAYPEN_MODEL_NAME}"
echo "=========================================="
