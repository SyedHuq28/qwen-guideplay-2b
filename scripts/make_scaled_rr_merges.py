import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def patch_generation_config(out_dir: str):
    p = Path(out_dir) / "generation_config.json"
    cfg = {}
    if p.exists():
        try:
            cfg = json.loads(p.read_text())
        except Exception:
            cfg = {}

    cfg["eos_token_id"] = [248046, 248044]
    cfg["pad_token_id"] = 248044

    p.write_text(json.dumps(cfg, indent=2))


def scale_lora_delta(peft_model, alpha: float):
    """
    Scale LoRA delta by multiplying lora_B weights.
    LoRA delta is proportional to B @ A, so scaling B scales the whole adapter delta.
    """
    n = 0
    for module_name, module in peft_model.named_modules():
        if hasattr(module, "lora_B"):
            for adapter_name, layer in module.lora_B.items():
                layer.weight.data.mul_(alpha)
                n += 1
    print(f"Scaled {n} LoRA B matrices by alpha={alpha}", flush=True)
    if n == 0:
        raise RuntimeError("No LoRA B matrices found. Is this a PEFT LoRA adapter?")


def make_one(stage2_merged: str, rr_adapter: str, out_dir: str, alpha: float):
    out = Path(out_dir)
    if out.exists():
        print("Removing existing:", out, flush=True)
        shutil.rmtree(out)

    print("=" * 80, flush=True)
    print("Creating scaled RR merge", flush=True)
    print("Stage2 merged:", stage2_merged, flush=True)
    print("RR adapter:", rr_adapter, flush=True)
    print("alpha:", alpha, flush=True)
    print("out:", out_dir, flush=True)
    print("=" * 80, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        stage2_merged,
        trust_remote_code=True,
        local_files_only=True,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        stage2_merged,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )

    model = PeftModel.from_pretrained(
        base,
        rr_adapter,
        local_files_only=True,
        is_trainable=False,
    )

    scale_lora_delta(model, alpha)

    merged = model.merge_and_unload()

    merged.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    patch_generation_config(out_dir)

    meta = {
        "method": "scaled_faipd_rr_adapter_merge",
        "stage2_merged": stage2_merged,
        "rr_adapter": rr_adapter,
        "alpha": alpha,
        "interpretation": "output model = Stage2 merged + alpha * FAIPD-RR LoRA delta",
    }
    Path(out_dir, "scaled_merge_metadata.json").write_text(json.dumps(meta, indent=2))

    print("Saved:", out_dir, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2_merged", required=True)
    ap.add_argument("--rr_adapter", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.25, 0.50, 0.75])
    args = ap.parse_args()

    for alpha in args.alphas:
        suffix = f"{int(round(alpha * 100)):03d}"
        out_dir = f"{args.out_root}-{suffix}-merged"
        make_one(
            stage2_merged=args.stage2_merged,
            rr_adapter=args.rr_adapter,
            out_dir=out_dir,
            alpha=alpha,
        )


if __name__ == "__main__":
    main()
