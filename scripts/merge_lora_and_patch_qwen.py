#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    base_model = args.base_model
    adapter = args.adapter
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    print(f"[merge] base_model={base_model}", flush=True)
    print(f"[merge] adapter={adapter}", flush=True)
    print(f"[merge] output={output}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        local_files_only=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )

    model = PeftModel.from_pretrained(
        model,
        adapter,
        local_files_only=True,
    )

    print("[merge] merging adapter into base model", flush=True)
    merged = model.merge_and_unload()

    print("[merge] saving merged model", flush=True)
    merged.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)

    # Patch generation config for Playpen/Qwen chat stop behavior.
    gen_cfg = {
        "eos_token_id": [248046, 248044],
        "pad_token_id": 248044
    }
    with (output / "generation_config.json").open("w", encoding="utf-8") as f:
        json.dump(gen_cfg, f, indent=2)

    print("[merge] wrote generation_config.json:", gen_cfg, flush=True)
    print("[merge] done", flush=True)


if __name__ == "__main__":
    main()