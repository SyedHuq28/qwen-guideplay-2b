

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

from datasets import load_dataset
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out_jsonl", required=True)
    p.add_argument("--dataset_name", default="colab-potsdam/playpen-data")
    p.add_argument("--dataset_config", default="interactions")
    p.add_argument("--split", default="train")
    p.add_argument("--success_only", action="store_true", default=True)
    p.add_argument("--min_completion_chars", type=int, default=1)
    return p.parse_args()


def outcome(ex: Dict[str, Any]) -> str:
    return str((ex.get("meta") or {}).get("outcome", "")).lower()


def simple_action_value(messages: List[Dict[str, str]], assistant_idx: int, ex: Dict[str, Any]) -> float:
    """Cheap value proxy. Replace with game-specific scoring/checkers later.

    The point is not to be perfect: it gives weighted SFT a notion of safe, compact,
    parseable turn behaviour before DPO/RL.
    """
    msg = messages[assistant_idx]["content"].strip()
    val = 0.50
    if outcome(ex) == "success":
        val += 0.30
    # Reward concise actions; many clembench games punish verbose invalid responses.
    n_words = len(msg.split())
    if n_words <= 30:
        val += 0.10
    if n_words <= 10:
        val += 0.05
    # Penalize obvious fake dialogue continuation.
    bad_markers = ["<|im_start|>", "user:", "assistant:", "game master:"]
    if any(m in msg.lower() for m in bad_markers):
        val -= 0.30
    # Reward parseable-looking command prefixes if present, without making this game-specific.
    good_prefixes = ["guess", "answer", "clue", "move", "action", "yes", "no"]
    if any(msg.lower().startswith(p) for p in good_prefixes):
        val += 0.05
    return max(0.0, min(1.0, val))


def main() -> None:
    args = parse_args()
    ds = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
    if args.success_only:
        ds = ds.filter(lambda ex: outcome(ex) == "success")

    n = 0
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for ex in tqdm(ds, desc="Writing turn-level samples"):
            messages = ex["messages"]
            meta = ex.get("meta") or {}
            for i, m in enumerate(messages):
                if m.get("role") != "assistant":
                    continue
                content = (m.get("content") or "").strip()
                if len(content) < args.min_completion_chars:
                    continue
                prompt = messages[:i]
                if not prompt:
                    continue
                row = {
                    "prompt": prompt,
                    "completion": [{"role": "assistant", "content": content}],
                    "value": simple_action_value(messages, i, ex),
                    "meta": {
                        **meta,
                        "turn_index": i,
                        "function_label": "unknown_auto_turn",
                    },
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
    print(f"Wrote {n} turn-level rows to {args.out_jsonl}")


if __name__ == "__main__":
    main()
