#!/bin/bash
#SBATCH --job-name=tr
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=syed-mahbubul.huq.2@city.ac.uk
#SBATCH --partition=preemptgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100_80g:1
#SBATCH -o outputs/%x_%j.o
#SBATCH -e outputs/%x_%j.e

set -eo pipefail

echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "start=$(date)"

mkdir -p /mnt/scratch/users/adhp263/logs

# -------------------------
# Environment setup
# -------------------------
set +u
source ~/.bashrc || true

conda activate diplomacy_env || {
  source ~/miniconda3/etc/profile.d/conda.sh || true
  conda activate diplomacy_env
}
set -u

cd /users/adhp263/dev/play/lm_playschool_training

export HF_HOME=/mnt/scratch/users/adhp263/hf_cache
export HF_DATASETS_CACHE=/mnt/scratch/users/adhp263/hf_cache/datasets
export TRANSFORMERS_CACHE=/mnt/scratch/users/adhp263/hf_cache/transformers
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# -------------------------
# Paths
# -------------------------
BASE_MODEL=/mnt/scratch/users/adhp263/models/Qwen3.5-2B
STAGE2_ADAPTER=/mnt/scratch/users/adhp263/runs/qwen35-2b-playpen-weighted-turn-sft
JUDGE_MODEL=/mnt/scratch/users/adhp263/models/Qwen3.5-9B

INPUT_DATA=/mnt/scratch/users/adhp263/data/playpen_success_turns.jsonl
JUDGE_DATA=/mnt/scratch/users/adhp263/data/playpen_success_turns_qwen35_9b_judged.jsonl

OUT_DIR=/mnt/scratch/users/adhp263/runs/qwen35-2b-playpen-llmjudge-stage2-refine
MERGED_DIR=/mnt/scratch/users/adhp263/runs/qwen35-2b-playpen-llmjudge-stage2-refine-merged

# Optional smoke test:
# LIMIT=2000 sbatch run_llmjudge_stage2_refine.slurm
LIMIT=${LIMIT:-}

echo "BASE_MODEL=${BASE_MODEL}"
echo "STAGE2_ADAPTER=${STAGE2_ADAPTER}"
echo "JUDGE_MODEL=${JUDGE_MODEL}"
echo "INPUT_DATA=${INPUT_DATA}"
echo "JUDGE_DATA=${JUDGE_DATA}"
echo "OUT_DIR=${OUT_DIR}"
echo "MERGED_DIR=${MERGED_DIR}"
echo "LIMIT=${LIMIT:-FULL}"

# -------------------------
# Sanity checks
# -------------------------
test -d "${BASE_MODEL}" || { echo "Missing BASE_MODEL: ${BASE_MODEL}"; exit 1; }
test -d "${STAGE2_ADAPTER}" || { echo "Missing STAGE2_ADAPTER: ${STAGE2_ADAPTER}"; exit 1; }
test -d "${JUDGE_MODEL}" || { echo "Missing JUDGE_MODEL: ${JUDGE_MODEL}"; exit 1; }
test -f "${INPUT_DATA}" || { echo "Missing INPUT_DATA: ${INPUT_DATA}"; exit 1; }
test -f scripts/score_with_qwen35_9b_judge.py || { echo "Missing scripts/score_with_qwen35_9b_judge.py"; exit 1; }
test -f scripts/train_weighted_turn_sft.py || { echo "Missing scripts/train_weighted_turn_sft.py"; exit 1; }
test -f scripts/merge_lora_and_patch_qwen.py || { echo "Missing scripts/merge_lora_and_patch_qwen.py"; exit 1; }

# -------------------------
# 1. Score Stage 2 data with Qwen3.5-9B judge
# -------------------------
echo "===== STEP 1: scoring with Qwen3.5-9B judge ====="

if [ -n "${LIMIT}" ]; then
  LIMIT_ARGS="--limit ${LIMIT}"
else
  LIMIT_ARGS=""
fi

python scripts/score_with_qwen35_9b_judge.py \
  --input "${INPUT_DATA}" \
  --output "${JUDGE_DATA}" \
  --judge_model "${JUDGE_MODEL}" \
  --resume \
  --fallback_old_value \
  --dtype bf16 \
  --max_new_tokens 96 \
  --max_context_chars 8000 \
  ${LIMIT_ARGS}

echo "===== judge scoring done ====="

# -------------------------
# 2. Inspect judged value distribution
# -------------------------
echo "===== judged score distribution ====="

python - <<PY
import json, statistics, os

p = "${JUDGE_DATA}"
vals = []

with open(p, encoding="utf-8") as f:
    for i, line in enumerate(f):
        if not line.strip():
            continue
        r = json.loads(line)
        vals.append(float(r["value"]))

        if i < 3:
            print("ROW", i)
            print("old_value:", r.get("old_value"))
            print("judge_value:", r.get("judge_value"))
            print("judge_reason:", r.get("judge_reason"))
            print("judge_raw:", r.get("judge_raw", "")[:300].replace("\n", " "))
            print()

if not vals:
    raise SystemExit("No judged values found.")

print("n", len(vals))
print("min", min(vals), "max", max(vals))
print("mean", statistics.mean(vals))
print("median", statistics.median(vals))

for lo in [0, .2, .4, .6, .8]:
    hi = lo + .2
    print(f"bucket {lo:.1f}-{hi:.1f}", sum(lo <= v < hi for v in vals))
PY

# -------------------------
# 3. Continue training from Stage 2 adapter
# -------------------------
echo "===== STEP 2: continue training Stage 2 adapter with judged values ====="

rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"

python scripts/train_weighted_turn_sft.py \
  --model_path "${BASE_MODEL}" \
  --adapter_path "${STAGE2_ADAPTER}" \
  --train_jsonl "${JUDGE_DATA}" \
  --output_dir "${OUT_DIR}" \
  --max_length 1024 \
  --num_train_epochs 0.25 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-5 \
  --warmup_ratio 0.03 \
  --save_steps 500 \
  --logging_steps 10 \
  --bf16 \
  --seed 42

echo "===== training done ====="

# -------------------------
# 4. Merge refined LoRA adapter
# -------------------------
echo "===== STEP 3: merge adapter ====="

rm -rf "${MERGED_DIR}"

python scripts/merge_lora_and_patch_qwen.py \
  --base_model "${BASE_MODEL}" \
  --adapter "${OUT_DIR}" \
  --output "${MERGED_DIR}"

echo "===== merge done ====="

# -------------------------
# 5. Patch generation config for Qwen / Playpen
# -------------------------
echo "===== STEP 4: patch generation_config.json ====="

cat > "${MERGED_DIR}/generation_config.json" <<'JSON'
{
  "eos_token_id": [248046, 248044],
  "pad_token_id": 248044
}
JSON

cat "${MERGED_DIR}/generation_config.json"

# -------------------------
# 6. Patch Playpen registry/model config
# -------------------------
echo "===== STEP 5: patch Playpen config ====="

cd /users/adhp263/dev/play/playpen

/users/adhp263/dev/play/playpen/eval_venv/bin/python fix_qwen_playpen_config.py \
  --model "llmjudge_stage2_refine=${MERGED_DIR}" || true

echo "===== done ====="
echo "end=$(date)"
echo "merged_model=${MERGED_DIR}"
