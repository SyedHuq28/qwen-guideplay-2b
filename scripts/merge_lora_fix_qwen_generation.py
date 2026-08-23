
from __future__ import annotations

import argparse
import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    eos_ids = []
    if isinstance(tok.eos_token_id, int):
        eos_ids.append(tok.eos_token_id)
    if isinstance(im_end_id, int) and im_end_id >= 0:
        eos_ids.append(im_end_id)
    eos_ids = sorted(set(eos_ids))

    model.generation_config = GenerationConfig.from_model_config(model.config)
    model.generation_config.eos_token_id = eos_ids if len(eos_ids) > 1 else eos_ids[0]
    model.generation_config.pad_token_id = tok.pad_token_id or tok.eos_token_id

    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    model.generation_config.save_pretrained(args.out)

    with open(os.path.join(args.out, "generation_fix_note.json"), "w") as f:
        json.dump({"eos_token_id": model.generation_config.eos_token_id, "pad_token_id": model.generation_config.pad_token_id}, f, indent=2)
    print(f"Saved merged model to {args.out}")
    print(f"eos_token_id={model.generation_config.eos_token_id}, pad_token_id={model.generation_config.pad_token_id}")


if __name__ == "__main__":
    main()
