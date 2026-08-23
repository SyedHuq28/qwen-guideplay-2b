#!/bin/bash
#SBATCH --job-name=play
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

# -----------------------------
# Project/code directory
# -----------------------------
PROJECT_DIR=/users/adhp263/dev/play/lm_playschool_training

# Folder where you manually placed:
# validation-00000-of-00001.parquet / .paraquet
# train-00000-of-00001 (4).parquet / .paraquet
PARQUET_SRC=${PROJECT_DIR}

# -----------------------------
# Model and output locations
# -----------------------------
MODEL_PATH=/mnt/scratch/users/adhp263/models/Qwen3.5-2B

RUN_DIR=/mnt/scratch/users/adhp263/runs
DATA_DIR=/mnt/scratch/users/adhp263/data
CACHE_DIR=/mnt/scratch/users/adhp263/hf_cache

LOCAL_PARQUET_DIR=${DATA_DIR}/playpen_interactions_parquet
LOCAL_JSONL_DIR=${DATA_DIR}/playpen_interactions_jsonl

TRAIN_JSONL=${LOCAL_JSONL_DIR}/train.jsonl
VALID_JSONL=${LOCAL_JSONL_DIR}/validation.jsonl
SUCCESS_JSONL=${LOCAL_JSONL_DIR}/train_success.jsonl
TURN_DATA=${DATA_DIR}/playpen_success_turns.jsonl

SFT_OUT=${RUN_DIR}/qwen35-2b-playpen-sft-success
SFT_MERGED=${RUN_DIR}/qwen35-2b-playpen-sft-success-merged

WEIGHTED_OUT=${RUN_DIR}/qwen35-2b-playpen-weighted-turn-sft
WEIGHTED_MERGED=${RUN_DIR}/qwen35-2b-playpen-weighted-turn-sft-merged

DPO_DATA=${DATA_DIR}/corrections_dpo.jsonl
DPO_OUT=${RUN_DIR}/qwen35-2b-playpen-dpo
DPO_MERGED=${RUN_DIR}/qwen35-2b-playpen-dpo-merged

# -----------------------------
# Make dirs
# -----------------------------
mkdir -p outputs
mkdir -p "${RUN_DIR}"
mkdir -p "${DATA_DIR}"
mkdir -p "${CACHE_DIR}"
mkdir -p "${CACHE_DIR}/datasets"
mkdir -p "${CACHE_DIR}/transformers"
mkdir -p "${LOCAL_PARQUET_DIR}"
mkdir -p "${LOCAL_JSONL_DIR}"

# -----------------------------
# Environment
# -----------------------------
source /users/adhp263/sharedscratch/miniconda3/etc/profile.d/conda.sh
conda activate diplomacy_env

cd "${PROJECT_DIR}"

# If you use your local venv, activate it.
if [ -f "${PROJECT_DIR}/.venv/bin/activate" ]; then
  source "${PROJECT_DIR}/.venv/bin/activate"
fi

# -----------------------------
# Memory/cache/offline settings
# -----------------------------
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

export HF_HOME="${CACHE_DIR}"
export HF_DATASETS_CACHE="${CACHE_DIR}/datasets"
export TRANSFORMERS_CACHE="${CACHE_DIR}/transformers"

# Fully offline: model local + dataset local.
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "=========================================="
echo "LM Playschool full offline training pipeline"
echo "Project dir: ${PROJECT_DIR}"
echo "Parquet src: ${PARQUET_SRC}"
echo "Model path: ${MODEL_PATH}"
echo "Run dir: ${RUN_DIR}"
echo "Data dir: ${DATA_DIR}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "=========================================="

# -----------------------------
# Sanity checks
# -----------------------------
echo "Checking project directory..."
test -d "${PROJECT_DIR}"

echo "Checking model directory..."
test -d "${MODEL_PATH}"

echo "Checking scripts..."
test -f scripts/merge_lora_fix_qwen_generation.py
test -f scripts/train_weighted_turn_sft.py
test -f scripts/train_dpo.py

echo "Checking parquet source directory..."
test -d "${PARQUET_SRC}"

# Find local parquet files.
# Handles both correct .parquet and mistaken .paraquet extension.
TRAIN_PARQUET_SRC=$(find "${PARQUET_SRC}" -type f \( -iname "*train*.parquet" -o -iname "*train*.paraquet" \) | head -n 1 || true)
VALID_PARQUET_SRC=$(find "${PARQUET_SRC}" -type f \( -iname "*validation*.parquet" -o -iname "*validation*.paraquet" -o -iname "*valid*.parquet" -o -iname "*valid*.paraquet" \) | head -n 1 || true)

if [ -z "${TRAIN_PARQUET_SRC}" ]; then
  echo "ERROR: Could not find train parquet file in ${PARQUET_SRC}"
  echo "Files found:"
  find "${PARQUET_SRC}" -maxdepth 2 -type f
  exit 1
fi

if [ -z "${VALID_PARQUET_SRC}" ]; then
  echo "WARNING: Could not find validation parquet file. Continuing with train only."
else
  echo "Found validation parquet: ${VALID_PARQUET_SRC}"
  cp "${VALID_PARQUET_SRC}" "${LOCAL_PARQUET_DIR}/validation.parquet"
fi

echo "Found train parquet: ${TRAIN_PARQUET_SRC}"
cp "${TRAIN_PARQUET_SRC}" "${LOCAL_PARQUET_DIR}/train.parquet"

echo "Local parquet directory:"
ls -lh "${LOCAL_PARQUET_DIR}"

# -----------------------------
# Convert local parquet to JSONL
# -----------------------------
echo ""
echo "=========================================="
echo "Converting local parquet to JSONL"
echo "=========================================="

python - <<PY
import json
import os
import pandas as pd

local_parquet_dir = "${LOCAL_PARQUET_DIR}"
local_jsonl_dir = "${LOCAL_JSONL_DIR}"
os.makedirs(local_jsonl_dir, exist_ok=True)

def write_jsonl(parquet_path, jsonl_path):
    print(f"Reading {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in df.to_dict(orient="records"):
            def clean(x):
                # pandas/numpy objects sometimes need conversion
                if hasattr(x, "tolist"):
                    return x.tolist()
                return x
            rec = {k: clean(v) for k, v in rec.items()}
            f.write(json.dumps(rec, ensure_ascii=False) + "\\n")

    print(f"Wrote {jsonl_path}")

train_parquet = os.path.join(local_parquet_dir, "train.parquet")
valid_parquet = os.path.join(local_parquet_dir, "validation.parquet")

write_jsonl(train_parquet, "${TRAIN_JSONL}")

if os.path.exists(valid_parquet):
    write_jsonl(valid_parquet, "${VALID_JSONL}")
else:
    print("No validation parquet found; skipping validation JSONL.")
PY

echo "JSONL files:"
ls -lh "${LOCAL_JSONL_DIR}"

# -----------------------------
# Build success-filtered JSONL
# -----------------------------
echo ""
echo "=========================================="
echo "Creating success-filtered train JSONL"
echo "=========================================="

python - <<PY
import json

train_jsonl = "${TRAIN_JSONL}"
success_jsonl = "${SUCCESS_JSONL}"

def maybe_json(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return x
    return x

n_total = 0
n_success = 0

with open(train_jsonl, "r", encoding="utf-8") as fin, open(success_jsonl, "w", encoding="utf-8") as fout:
    for line in fin:
        n_total += 1
        ex = json.loads(line)
        meta = maybe_json(ex.get("meta", {}))
        outcome = None

        if isinstance(meta, dict):
            outcome = meta.get("outcome")

        # Keep successful interaction episodes only.
        if outcome == "success":
            fout.write(json.dumps(ex, ensure_ascii=False) + "\\n")
            n_success += 1

print(f"Total rows: {n_total}")
print(f"Success rows: {n_success}")
print(f"Wrote: {success_jsonl}")

if n_success == 0:
    raise SystemExit("ERROR: Found 0 successful rows. Check whether meta.outcome exists in your parquet.")
PY

# -----------------------------
# Create local SFT training script
# -----------------------------
echo ""
echo "=========================================="
echo "Writing local JSONL SFT script"
echo "=========================================="

cat > scripts/train_sft_success_from_jsonl.py <<'PY'
import argparse
import json
from typing import Any, Dict, List

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model


def maybe_json(x: Any) -> Any:
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return x
    return x


def normalize_messages(messages: Any) -> List[Dict[str, str]]:
    messages = maybe_json(messages)
    if hasattr(messages, "tolist"):
        messages = messages.tolist()

    out = []
    for m in messages:
        m = maybe_json(m)
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role is None or content is None:
            continue
        if role not in {"system", "user", "assistant"}:
            continue
        out.append({"role": str(role), "content": str(content)})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map=None,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",
        modules_to_save=[],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    rows = []
    with open(args.train_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            messages = normalize_messages(ex.get("messages", []))
            if not messages:
                continue
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            rows.append({"text": text})

    if not rows:
        raise RuntimeError("No usable rows found. Check messages column.")

    print(f"Loaded {len(rows)} success SFT rows from {args.train_jsonl}")
    ds = Dataset.from_list(rows)

    def tok(batch):
        enc = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding=False,
        )
        return enc

    ds = ds.map(tok, batched=True, remove_columns=["text"])

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        bf16=args.bf16,
        logging_steps=10,
        save_steps=1000,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
PY

# -----------------------------
# Stage 1: Success-filtered SFT from local JSONL
# -----------------------------
echo ""
echo "=========================================="
echo "Stage 1: success-filtered LoRA SFT from local JSONL"
echo "Input: ${SUCCESS_JSONL}"
echo "Output: ${SFT_OUT}"
echo "=========================================="

accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  scripts/train_sft_success_from_jsonl.py \
  --model_path "${MODEL_PATH}" \
  --train_jsonl "${SUCCESS_JSONL}" \
  --output_dir "${SFT_OUT}" \
  --max_length 1024 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --bf16

echo "Stage 1 complete."

# -----------------------------
# Merge Stage 1
# -----------------------------
echo ""
echo "=========================================="
echo "Merging Stage 1 adapter and fixing Qwen generation"
echo "Merged output: ${SFT_MERGED}"
echo "=========================================="

python scripts/merge_lora_fix_qwen_generation.py \
  --base_model "${MODEL_PATH}" \
  --adapter "${SFT_OUT}" \
  --out "${SFT_MERGED}"

echo "Stage 1 merge complete."

# -----------------------------
# Build turn-level value dataset from local success JSONL
# -----------------------------
echo ""
echo "=========================================="
echo "Creating turn-level value dataset from local success JSONL"
echo "Output: ${TURN_DATA}"
echo "=========================================="

python - <<PY
import json
import os

success_jsonl = "${SUCCESS_JSONL}"
turn_data = "${TURN_DATA}"
os.makedirs(os.path.dirname(turn_data), exist_ok=True)

def maybe_json(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return x
    return x

def normalize_messages(messages):
    messages = maybe_json(messages)
    if hasattr(messages, "tolist"):
        messages = messages.tolist()

    out = []
    for m in messages:
        m = maybe_json(m)
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role in {"system", "user", "assistant"} and content is not None:
            out.append({"role": str(role), "content": str(content)})
    return out

def simple_value(content):
    c = content.strip()
    value = 0.70

    # Reward short, parseable, turn-bounded replies.
    if 1 <= len(c) <= 400:
        value += 0.15
    if "<|im_start|>" not in c and "<|im_end|>" not in c:
        value += 0.05
    if "\\nuser" not in c.lower() and "\\nassistant" not in c.lower():
        value += 0.05
    if len(c) > 0:
        value += 0.05

    return min(1.0, max(0.0, value))

n_rows = 0
n_turns = 0

with open(success_jsonl, "r", encoding="utf-8") as fin, open(turn_data, "w", encoding="utf-8") as fout:
    for line in fin:
        n_rows += 1
        ex = json.loads(line)
        messages = normalize_messages(ex.get("messages", []))
        meta = maybe_json(ex.get("meta", {}))
        if not isinstance(meta, dict):
            meta = {}

        for i, msg in enumerate(messages):
            if msg["role"] != "assistant":
                continue
            prompt = messages[:i]
            completion = [msg]
            if not prompt:
                continue

            row = {
                "prompt": prompt,
                "completion": completion,
                "value": simple_value(msg["content"]),
                "meta": {
                    **meta,
                    "turn_index": i,
                    "function_label": "auto_turn_success"
                }
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\\n")
            n_turns += 1

print(f"Success episodes read: {n_rows}")
print(f"Turn-level rows written: {n_turns}")
print(f"Wrote: {turn_data}")

if n_turns == 0:
    raise SystemExit("ERROR: No assistant turns found for turn-level data.")
PY

echo "Turn-level data complete:"
ls -lh "${TURN_DATA}"

