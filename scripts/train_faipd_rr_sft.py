

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def qwen_segment(role: str, content: str) -> str:
    return f"{IM_START}{role}\n{content}{IM_END}\n"


class ChatDataset(Dataset):
    def __init__(self, rows: List[Dict], tokenizer, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        for row in rows:
            messages = row.get("messages")
            if not isinstance(messages, list):
                continue
            ex = self.encode(messages)
            if ex is not None:
                self.examples.append(ex)

        print(f"encoded examples: {len(self.examples)} / {len(rows)}")

    def encode(self, messages: List[Dict]):
        input_ids = []
        labels = []

        for m in messages:
            role = m.get("role")
            content = m.get("content", "")

            if role not in {"system", "user", "assistant"}:
                continue

            seg = qwen_segment(role, content)
            ids = self.tokenizer(seg, add_special_tokens=False)["input_ids"]

            input_ids.extend(ids)

            if role == "assistant":
                labels.extend(ids)
            else:
                labels.extend([-100] * len(ids))

        if not input_ids:
            return None

        # Truncate from the left to keep the final/current turn.
        if len(input_ids) > self.max_length:
            input_ids = input_ids[-self.max_length:]
            labels = labels[-self.max_length:]

        if all(x == -100 for x in labels):
            return None

        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)

        batch_input_ids = []
        batch_attention = []
        batch_labels = []

        for f in features:
            pad_len = max_len - len(f["input_ids"])

            batch_input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            batch_attention.append(f["attention_mask"] + [0] * pad_len)
            batch_labels.append(f["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--num_train_epochs", type=float, default=0.20)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=3e-5)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 248044

    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
    )

    model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    rows = read_jsonl(Path(args.train_jsonl))
    train_ds = ChatDataset(rows, tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        logging_steps=10,
        save_strategy="no",
        bf16=args.bf16,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="none",
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=DataCollator(tokenizer),
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Saved FAIPD-RR LoRA adapter to:", args.output_dir)


if __name__ == "__main__":
    main()
