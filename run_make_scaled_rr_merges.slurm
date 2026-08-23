#!/bin/bash
#SBATCH --job-name=rrscale
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=syed-mahbubul.huq.2@city.ac.uk
#SBATCH --partition=preemptgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a100_80g:1
#SBATCH -o outputs/%x_%j.o
#SBATCH -e outputs/%x_%j.e

set -eo pipefail

cd /users/adhp263/dev/play/lm_playschool_training
mkdir -p outputs

set +u
# Use eval_venv because diplomacy_env cannot load model_type qwen3_5_text.
source /users/adhp263/dev/play/playpen/eval_venv/bin/activate
set -u
pip install peft
export HF_HOME=/mnt/scratch/users/adhp263/hf_cache
export TRANSFORMERS_CACHE=/mnt/scratch/users/adhp263/hf_cache/transformers
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

STAGE2=/mnt/scratch/users/adhp263/runs/qwen35-2b-playpen-weighted-turn-sft-merged
RR_ADAPTER=/mnt/scratch/users/adhp263/runs/qwen35-2b-playpen-faipd-rr-025
OUT_ROOT=/mnt/scratch/users/adhp263/runs/qwen35-2b-playpen-rrscale

python -u scripts/make_scaled_rr_merges.py \
  --stage2_merged "${STAGE2}" \
  --rr_adapter "${RR_ADAPTER}" \
  --out_root "${OUT_ROOT}" \
  --alphas 0.25 0.50 0.75

echo "DONE scaled RR merges."
