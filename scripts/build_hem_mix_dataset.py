

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


FORBIDDEN_PATH_TOKENS = [
    "validation",
    "/val",
    "val.",
    ".val.",
    "test",
    "playpen-eval",
    "eval_results",
    "leaderboard",
    "public",
]

BAD_OUTPUT_PATTERNS = [
    "<|im_start|>",
    "<|im_end|>",
    "```",
]

GO_RE = re.compile(r"^\s*GO:\s*(north|south|east|west)\s*$", re.I)
DONE_RE = re.compile(r"^\s*DONE\s*$", re.I)
JSON_ACTION_RE = re.compile(r'^\s*\{\s*"action"\s*:', re.I)
GUESS_RE = re.compile(r"^\s*(guess|answer)\s*:", re.I)
EXPLANATION_GUESS_RE = re.compile(r"explanation\s*:.*guess\s*:", re.I | re.S)


def safety_check_path(path: Path, name: str, must_exist: bool = True) -> None:
    s = str(path).lower()
    for token in FORBIDDEN_PATH_TOKENS:
        if token in s:
            raise ValueError(f"Refusing unsafe validation/test/eval/public path for {name}: {path}")

    if must_exist and not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except Exception as e:
                print(f"WARNING: bad JSON on line {line_no}: {e}")
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def is_msg(m: Any) -> bool:
    return (
        isinstance(m, dict)
        and m.get("role") in {"system", "user", "assistant"}
        and isinstance(m.get("content"), str)
    )


def valid_messages(messages: Any) -> bool:
    return (
        isinstance(messages, list)
        and len(messages) >= 1
        and all(is_msg(m) for m in messages)
    )


def normalize_completion(completion: Any) -> Optional[List[Dict[str, str]]]:
    if isinstance(completion, list):
        if valid_messages(completion):
            return [{"role": m["role"], "content": m["content"]} for m in completion]
        return None

    if isinstance(completion, str):
        return [{"role": "assistant", "content": completion}]

    return None


def normalize_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Normalize an existing Stage 2 row to prompt/completion/value schema.
    """
    if "prompt" in row and "completion" in row:
        prompt = row.get("prompt")
        completion = normalize_completion(row.get("completion"))

        if valid_messages(prompt) and completion:
            value = float(row.get("value", row.get("weight", 1.0)))
            out = dict(row)
            out["prompt"] = [{"role": m["role"], "content": m["content"]} for m in prompt]
            out["completion"] = completion
            out["value"] = value
            out["weight"] = value
            out["source"] = row.get("source", "stage2_weighted_turn")
            return out

    messages = row.get("messages")
    if valid_messages(messages):
        last_assistant_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                last_assistant_idx = i
                break

        if last_assistant_idx is None or last_assistant_idx == 0:
            return None

        prompt = messages[:last_assistant_idx]
        response = messages[last_assistant_idx]["content"]

        value = float(row.get("value", row.get("weight", 1.0)))

        return {
            "prompt": [{"role": m["role"], "content": m["content"]} for m in prompt],
            "completion": [{"role": "assistant", "content": response}],
            "response": response,
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
            "value": value,
            "weight": value,
            "source": row.get("source", "stage2_weighted_turn"),
            "meta": row.get("meta", {}),
        }

    return None


def completion_text(row: Dict[str, Any]) -> str:
    comp = row.get("completion", [])
    if isinstance(comp, list) and comp:
        parts = []
        for m in comp:
            if isinstance(m, dict) and isinstance(m.get("content"), str):
                parts.append(m["content"])
        return "\n".join(parts).strip()
    return str(row.get("response", "")).strip()


def response_format_score(text: str) -> float:
    t = text.strip()

    if not t:
        return 0.05

    if GO_RE.match(t):
        return 1.10

    if DONE_RE.match(t):
        return 1.10

    if JSON_ACTION_RE.match(t) and t.endswith("}"):
        return 1.05

    if GUESS_RE.match(t) or EXPLANATION_GUESS_RE.search(t):
        return 0.95

    if any(bad in t for bad in BAD_OUTPUT_PATTERNS):
        return 0.25

    if len(t) <= 120:
        return 0.80

    if len(t) <= 400:
        return 0.55

    return 0.25


def render_chat(tokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    except Exception:
        # Fallback if template fails.
        chunks = []
        for m in messages:
            chunks.append(f"{m['role']}: {m['content']}")
        if add_generation_prompt:
            chunks.append("assistant:")
        return "\n".join(chunks)


def compute_example_loss(
    model,
    tokenizer,
    row: Dict[str, Any],
    max_length: int,
    device: torch.device,
) -> Optional[Tuple[float, int]]:
    """
    Compute mean CE loss on completion tokens only.
    """
    prompt = row["prompt"]
    completion = row["completion"]
    full_messages = prompt + completion

    prompt_text = render_chat(tokenizer, prompt, add_generation_prompt=True)
    full_text = render_chat(tokenizer, full_messages, add_generation_prompt=False)

    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]

    enc = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    seq_len = input_ids.shape[1]
    prompt_len = min(len(prompt_ids), seq_len)

    labels = input_ids.clone()
    labels[:, :prompt_len] = -100

    target_tokens = int((labels != -100).sum().item())
    if target_tokens < 1:
        return None

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss_flat = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        )

        valid = shift_labels.view(-1) != -100
        if valid.sum().item() < 1:
            return None

        loss = loss_flat[valid].mean().item()
        n_tokens = int(valid.sum().item())

    if not math.isfinite(loss):
        return None

    return float(loss), n_tokens


def percentile(xs: List[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    idx = int(round((len(xs) - 1) * q))
    idx = max(0, min(len(xs) - 1, idx))
    return xs[idx]


def audit_loss(scored: List[Dict[str, Any]]) -> None:
    losses = [float(r["stage2_loss"]) for r in scored]
    values = [float(r.get("value", 1.0)) for r in scored]

    print("scored rows:", len(scored))
    print("loss min/p10/p25/p50/p75/p90/max:",
          min(losses),
          percentile(losses, 0.10),
          percentile(losses, 0.25),
          percentile(losses, 0.50),
          percentile(losses, 0.75),
          percentile(losses, 0.90),
          max(losses))
    print("value min/p50/max:",
          min(values),
          percentile(values, 0.50),
          max(values))


def sample_rows(rows: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    rows = list(rows)

    if len(rows) <= n:
        rng.shuffle(rows)
        return rows

    return rng.sample(rows, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2_jsonl", type=Path, required=True)
    ap.add_argument("--scoring_model", type=Path, required=True)
    ap.add_argument("--out_jsonl", type=Path, required=True)
    ap.add_argument("--score_cache_jsonl", type=Path, default=None)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--total_examples", type=int, default=20000)
    ap.add_argument("--hard_ratio", type=float, default=0.70)
    ap.add_argument("--min_value", type=float, default=0.70)
    ap.add_argument("--min_loss", type=float, default=0.25)
    ap.add_argument("--max_loss", type=float, default=2.50)
    ap.add_argument("--max_rows_to_score", type=int, default=0)
    ap.add_argument("--seed", type=int, default=28)
    args = ap.parse_args()

    safety_check_path(args.stage2_jsonl, "stage2_jsonl")
    safety_check_path(args.scoring_model, "scoring_model")
    safety_check_path(args.out_jsonl.parent, "out_jsonl_parent", must_exist=True)

    if args.hard_ratio <= 0 or args.hard_ratio >= 1:
        raise ValueError("--hard_ratio must be between 0 and 1")

    print("Loading Stage 2 data:", args.stage2_jsonl, flush=True)
    raw_rows = read_jsonl(args.stage2_jsonl)

    rows = []
    for r in raw_rows:
        nr = normalize_row(r)
        if nr is not None:
            txt = completion_text(nr)
            fmt = response_format_score(txt)

            # Do not let obviously malformed rows become hard examples.
            if fmt < 0.55:
                continue

            nr["format_score"] = fmt
            rows.append(nr)

    if args.max_rows_to_score and args.max_rows_to_score > 0:
        rows = rows[: args.max_rows_to_score]

    print("raw rows:", len(raw_rows), flush=True)
    print("usable normalized rows:", len(rows), flush=True)

    if len(rows) < 1000:
        raise SystemExit("Too few usable rows. Stop.")

    scored = []

    if args.score_cache_jsonl and args.score_cache_jsonl.exists():
        print("Loading cached scores:", args.score_cache_jsonl, flush=True)
        scored = read_jsonl(args.score_cache_jsonl)
    else:
        print("Loading scoring model:", args.scoring_model, flush=True)

        tokenizer = AutoTokenizer.from_pretrained(
            str(args.scoring_model),
            trust_remote_code=True,
            local_files_only=True,
            padding_side="left",
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            str(args.scoring_model),
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=dtype,
            device_map="auto",
        )
        model.eval()

        device = next(model.parameters()).device

        iterator = enumerate(rows, start=1)
        if tqdm is not None:
            iterator = tqdm(
                iterator,
                total=len(rows),
                desc="Scoring Stage2 examples",
                dynamic_ncols=True,
                mininterval=5,
            )

        for i, r in iterator:
            result = compute_example_loss(
                model=model,
                tokenizer=tokenizer,
                row=r,
                max_length=args.max_length,
                device=device,
            )

            if result is None:
                continue

            loss, n_tokens = result

            out = dict(r)
            out["stage2_loss"] = loss
            out["target_tokens"] = n_tokens
            out["source_original"] = out.get("source", "stage2_weighted_turn")
            scored.append(out)

            if tqdm is not None:
                iterator.set_postfix({
                    "usable": len(scored),
                    "loss": f"{loss:.4f}",
                })

            if i % 100 == 0:
                print(
                    f"scored {i}/{len(rows)} usable={len(scored)} latest_loss={loss:.4f}",
                    flush=True,
                )

            # Incremental cache so progress survives preemption/cancel.
            if args.score_cache_jsonl and i % 500 == 0:
                write_jsonl(args.score_cache_jsonl, scored)
                print(
                    f"incremental cache written: {args.score_cache_jsonl} rows={len(scored)}",
                    flush=True,
                )

        if args.score_cache_jsonl:
            print("Writing score cache:", args.score_cache_jsonl, flush=True)
            write_jsonl(args.score_cache_jsonl, scored)

    if len(scored) < 1000:
        raise SystemExit("Too few scored rows. Stop.")

    audit_loss(scored)

    hard_candidates = []
    replay_candidates = []

    for r in scored:
        loss = float(r["stage2_loss"])
        value = float(r.get("value", 1.0))

        # Replay pool: all clean Stage 2 rows.
        replay = dict(r)
        replay["source"] = "hem_stage2_replay"
        replay_candidates.append(replay)

        # Hard pool: high value, uncertain, but not extreme/outlier.
        if value >= args.min_value and args.min_loss <= loss <= args.max_loss:
            hard = dict(r)
            hard["source"] = "hem_hard_high_value"
            # Slightly boost value for useful hard examples while keeping original value visible.
            hard["value_original"] = value
            hard["value"] = min(1.25, max(value, 0.90))
            hard["weight"] = hard["value"]
            hard_candidates.append(hard)

    print("hard candidates:", len(hard_candidates))
    print("replay candidates:", len(replay_candidates))

    if len(hard_candidates) < 500:
        print("WARNING: very few hard candidates. Consider lowering --min_loss or --min_value.")

    n_hard = int(args.total_examples * args.hard_ratio)
    n_replay = args.total_examples - n_hard

    hard_sample = sample_rows(hard_candidates, n_hard, args.seed + 1)
    replay_sample = sample_rows(replay_candidates, n_replay, args.seed + 2)

    final_rows = hard_sample + replay_sample
    random.Random(args.seed).shuffle(final_rows)

    print("sampled hard:", len(hard_sample))
    print("sampled replay:", len(replay_sample))
    print("final rows:", len(final_rows))

    # Remove scoring-only keys that trainer does not need, but keep useful audit metadata in meta.
    cleaned = []
    for r in final_rows:
        meta = r.get("meta", {})
        if not isinstance(meta, dict):
            meta = {}

        meta.update({
            "stage2_loss": r.get("stage2_loss"),
            "target_tokens": r.get("target_tokens"),
            "format_score": r.get("format_score"),
            "source_original": r.get("source_original"),
            "value_original": r.get("value_original", r.get("value")),
        })

        out = {
            "prompt": r["prompt"],
            "completion": r["completion"],
            "response": completion_text(r),
            "messages": r.get("messages", r["prompt"] + r["completion"]),
            "value": float(r.get("value", 1.0)),
            "weight": float(r.get("weight", r.get("value", 1.0))),
            "source": r.get("source", "hem_mix"),
            "meta": meta,
        }
        cleaned.append(out)

    counts = {}
    vals = []
    losses = []
    for r in cleaned:
        counts[r["source"]] = counts.get(r["source"], 0) + 1
        vals.append(float(r["value"]))
        if r.get("meta", {}).get("stage2_loss") is not None:
            losses.append(float(r["meta"]["stage2_loss"]))

    print("final source counts:")
    print(json.dumps(counts, indent=2))
    print("final value min/avg/max:", min(vals), sum(vals) / len(vals), max(vals))
    print("final loss min/avg/max:", min(losses), sum(losses) / len(losses), max(losses))

    write_jsonl(args.out_jsonl, cleaned)
    print("wrote:", args.out_jsonl)


if __name__ == "__main__":
    main()
