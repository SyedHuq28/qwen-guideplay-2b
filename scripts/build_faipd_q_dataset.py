

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


FORBIDDEN_PATH_TOKENS = [
    "validation",
    "/val",
    "val.",
    "test",
    "playpen-eval",
    ".val.json",
]

GO_RE = re.compile(r"^\s*GO:\s*(north|south|east|west)\s*$", re.I)
DONE_RE = re.compile(r"^\s*DONE\s*$", re.I)
JSON_ACTION_RE = re.compile(r'^\s*\{\s*"action"\s*:', re.I)
GUESS_RE = re.compile(r"^\s*(guess|answer)\s*:", re.I)
EXPLANATION_GUESS_RE = re.compile(r"explanation\s*:.*guess\s*:", re.I | re.S)


def safety_check_path(path: Path, name: str) -> None:
    s = str(path).lower()
    for token in FORBIDDEN_PATH_TOKENS:
        if token in s:
            raise ValueError(
                f"Refusing to use possible validation/test/eval path for {name}: {path}"
            )


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def to_weighted_sft_row(
    messages: List[Dict[str, str]],
    value: float,
    source: str,
    meta: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Convert chat messages into the schema expected by train_weighted_turn_sft.py.

    Existing trainer expects:
      row["prompt"]   = list of messages before assistant answer
      row["response"] = assistant answer string
      row["value"]    = float weight

    We also keep row["messages"] for debugging.
    """
    if not messages:
        return None

    # Find the last assistant message as the supervised target.
    last_assistant_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            last_assistant_idx = i
            break

    if last_assistant_idx is None:
        return None

    response = messages[last_assistant_idx].get("content", "").strip()
    if not response:
        return None

    prompt = messages[:last_assistant_idx]
    if not prompt:
        return None

    completion = [{"role": "assistant", "content": response}]

    return {
        "prompt": prompt,
        "completion": completion,
        "response": response,
        "messages": messages,
        "value": float(value),
        "weight": float(value),
        "source": source,
        "meta": meta or {},
    }




def valid_messages(messages: Any) -> bool:
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    for m in messages:
        if not isinstance(m, dict):
            return False
        if m.get("role") not in {"system", "user", "assistant"}:
            return False
        if not isinstance(m.get("content"), str):
            return False
    return any(m.get("role") == "assistant" for m in messages)


def normalize_messages(row: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    """
    Accept either:
    - {"messages": [{"role": ..., "content": ...}, ...]}
    - {"prompt": ..., "response"/"answer"/"target"/"action"/"completion": ...}
    """
    messages = row.get("messages")
    if valid_messages(messages):
        return [{"role": m["role"], "content": m["content"]} for m in messages]

    prompt = row.get("prompt") or row.get("input") or row.get("state")
    answer = (
        row.get("response")
        or row.get("answer")
        or row.get("target")
        or row.get("action")
        or row.get("completion")
    )

    if isinstance(prompt, str) and isinstance(answer, str):
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]

    return None


def get_meta(row: Dict[str, Any]) -> Dict[str, Any]:
    meta = row.get("meta")
    if isinstance(meta, dict):
        return meta
    return {}


def get_outcome(row: Dict[str, Any]) -> str:
    """
    Try several common locations for outcome metadata.
    Unknown rows receive medium value, but actual success/lose/aborted rows are preferred.
    """
    meta = get_meta(row)

    candidates = [
        meta.get("outcome"),
        row.get("outcome"),
        row.get("episode_outcome"),
        row.get("status"),
    ]

    for c in candidates:
        if isinstance(c, str) and c.strip():
            return c.strip().lower()

    scores = row.get("scores")
    if isinstance(scores, dict):
        if scores.get("Success") == 1 or scores.get("success") == 1:
            return "success"
        if scores.get("Aborted") == 1 or scores.get("aborted") == 1:
            return "aborted"
        if scores.get("Lose") == 1 or scores.get("lose") == 1:
            return "lose"

    return "unknown"


def episode_base_value(outcome: str) -> float:
    """
    Outcome-grounded base return proxy.

    success: strong positive signal
    lose/failure: lower but not zero
    aborted: small but non-zero, because early turns may be valid
    unknown: neutral medium value
    """
    outcome = outcome.lower()

    if outcome in {"success", "successful", "win", "won"}:
        return 1.0

    if outcome in {"lose", "lost", "failure", "failed", "unsuccessful"}:
        return 0.35

    if outcome in {"aborted", "abort", "invalid"}:
        return 0.18

    return 0.50


def response_format_score(text: str) -> float:
    """
    Interactional-validity proxy.

    This does not decide strategic correctness. It only scores whether the answer
    looks like a safe Playpen action: short, parseable, no stop-token leaks,
    no markdown, no role leakage.
    """
    t = text.strip()

    if not t:
        return 0.05

    if GO_RE.match(t):
        return 1.10

    if DONE_RE.match(t):
        return 1.05

    if JSON_ACTION_RE.match(t) and t.endswith("}"):
        return 1.05

    if GUESS_RE.match(t) or EXPLANATION_GUESS_RE.search(t):
        return 0.95

    if any(bad in t for bad in ["<|im_start|>", "<|im_end|>", "||", "```"]):
        return 0.35

    if len(t) <= 120 and not any(bad in t for bad in ["user\n", "assistant\n"]):
        return 0.80

    if len(t) <= 400:
        return 0.55

    return 0.25


def assistant_turn_indices(messages: List[Dict[str, str]]) -> List[int]:
    return [i for i, m in enumerate(messages) if m.get("role") == "assistant"]


def make_turn_example(
    messages: List[Dict[str, str]],
    assistant_index: int,
    outcome: str,
    turn_number: int,
    total_turns: int,
) -> Optional[Dict[str, Any]]:
    answer = messages[assistant_index]["content"].strip()
    if not answer:
        return None

    context_plus_answer = messages[: assistant_index + 1]

    base = episode_base_value(outcome)
    fmt = response_format_score(answer)

    # Smooth exponential turn discount.
    # Earlier decisions condition more future behaviour, but late-game moves still matter.
    if total_turns <= 1:
        turn_discount = 1.0
    else:
        progress = turn_number / max(total_turns - 1, 1)
        lambda_decay = 0.50
        turn_discount = math.exp(-lambda_decay * progress)

    stop_bonus = 1.0
    if DONE_RE.match(answer) and outcome in {"success", "successful", "win", "won"}:
        stop_bonus = 1.15

    value = base * fmt * turn_discount * stop_bonus

    # Clamp for stability.
    value = max(0.05, min(1.25, value))

    # Drop extremely weak examples.
    if value < 0.12:
        return None

    return to_weighted_sft_row(
        messages=context_plus_answer,
        value=value,
        source="faipd_q_outcome_turn",
        meta={
            "outcome": outcome,
            "turn_number": turn_number,
            "total_turns": total_turns,
            "format_score": fmt,
            "episode_base_value": base,
            "turn_discount": turn_discount,
            "stop_bonus": stop_bonus,
        },
    )


def build_outcome_turns(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    examples = []

    for row in rows:
        messages = normalize_messages(row)
        if not messages:
            continue

        outcome = get_outcome(row)
        indices = assistant_turn_indices(messages)
        total = len(indices)

        if total == 0:
            continue

        for turn_number, idx in enumerate(indices):
            ex = make_turn_example(
                messages=messages,
                assistant_index=idx,
                outcome=outcome,
                turn_number=turn_number,
                total_turns=total,
            )
            if ex is not None:
                examples.append(ex)

    return examples


def build_success_replay(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    examples = []

    for row in rows:
        messages = normalize_messages(row)
        if not messages:
            continue

        outcome = get_outcome(row)
        if outcome not in {"success", "successful", "win", "won"}:
            continue

        ex = to_weighted_sft_row(
            messages=messages,
            value=0.85,
            source="faipd_q_success_replay",
            meta={"outcome": outcome},
        )
        if ex is not None:
            examples.append(ex)

    return examples


def get_first_user_assistant_pair(messages: List[Dict[str, str]]) -> Optional[Tuple[str, str]]:
    last_user = None
    for m in messages:
        if m["role"] == "user":
            last_user = m["content"]
        elif m["role"] == "assistant" and last_user:
            ans = m["content"].strip()
            if ans:
                return last_user, ans
    return None


def corrupt_answer(good: str) -> List[str]:
    good = good.strip()
    bads = []

    if GO_RE.match(good):
        bads.extend([
            f"Sure, I will answer with {good}.",
            f"{good}<|im_end|>",
            f"{good}||",
            f"```text\n{good}\n```",
            f"The best move is {good}.",
        ])

    elif DONE_RE.match(good):
        bads.extend([
            "I think we are done now. DONE",
            "DONE<|im_end|>",
            "DONE||",
            "```text\nDONE\n```",
        ])

    elif good.startswith("{") and good.endswith("}"):
        bads.extend([
            f"Here is the JSON:\n{good}",
            f"```json\n{good}\n```",
            f"{good}<|im_end|>",
            f"{good}||",
        ])

    elif len(good) <= 200:
        bads.extend([
            f"Sure. {good}",
            f"{good}<|im_end|>",
            f"{good}||",
            f"```text\n{good}\n```",
        ])

    return bads


def build_repair_examples(rows: List[Dict[str, Any]], max_pool: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    pool = []

    for row in rows:
        messages = normalize_messages(row)
        if not messages:
            continue

        pair = get_first_user_assistant_pair(messages)
        if not pair:
            continue

        user_prompt, good = pair
        good = good.strip()

        if not good or len(good) > 500:
            continue

        for bad in corrupt_answer(good):
            repair_prompt = (
                user_prompt.strip()
                + "\n\nThe previous assistant answer was invalid because it contained extra text, "
                  "stop tokens, markdown, or the wrong format:\n"
                + bad
                + "\n\nRewrite the answer correctly. Output only the final valid answer and nothing else."
            )
            ex = to_weighted_sft_row(
                messages=[
                    {"role": "user", "content": repair_prompt},
                    {"role": "assistant", "content": good},
                ],
                value=0.75,
                source="faipd_q_repair_train_derived",
                meta={"repair_type": "strict_format"},
            )
            if ex is not None:
                pool.append(ex)

    # Generic movement format examples. These do not use validation instances.
    directions = ["north", "south", "east", "west"]
    rooms = ["Kitchen", "Library", "Lobby", "Garden", "Bedroom", "Office", "Hallway", "Cellar"]

    for _ in range(1000):
        room = rng.choice(rooms)
        dirs = rng.sample(directions, k=rng.randint(1, 3))
        chosen = rng.choice(dirs)

        prompt = (
            "Please help me with the following task. To move, answer exactly with GO: DIRECTION, "
            "where DIRECTION is one of [north, south, east, west]. To stop, answer DONE. "
            "Omit any other text.\n"
            f"You are in the {room}. Currently available directions: {', '.join(dirs)}. "
            "What is your next instruction?"
        )

        ex = to_weighted_sft_row(
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": f"GO: {chosen}"},
            ],
            value=0.80,
            source="faipd_q_generic_format",
            meta={"repair_type": "generic_move_format"},
        )
        if ex is not None:
            pool.append(ex)

    if len(pool) <= max_pool:
        rng.shuffle(pool)
        return pool

    return rng.sample(pool, max_pool)


def weighted_sample(rows: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)

    if len(rows) <= n:
        rng.shuffle(rows)
        return rows

    # Prefer higher-value rows but preserve diversity by sampling from a larger high-value pool.
    rows = sorted(rows, key=lambda x: x.get("value", 0.5), reverse=True)
    candidate_pool = rows[: max(n * 3, n)]
    return rng.sample(candidate_pool, n)


def print_value_audit(final_rows: List[Dict[str, Any]]) -> None:
    counts = {}
    values = []

    for r in final_rows:
        counts[r["source"]] = counts.get(r["source"], 0) + 1
        values.append(float(r.get("value", 1.0)))

    print("final source counts:")
    print(json.dumps(counts, indent=2))
    print("total:", len(final_rows))

    if not values:
        raise ValueError("No values found; empty dataset?")

    avg_value = sum(values) / len(values)
    print("value min/max/avg:", min(values), max(values), avg_value)

    buckets = {
        "very_low_0.00_0.20": 0,
        "low_0.20_0.40": 0,
        "mid_0.40_0.70": 0,
        "high_0.70_1.00": 0,
        "very_high_1.00_plus": 0,
    }

    for v in values:
        if v < 0.20:
            buckets["very_low_0.00_0.20"] += 1
        elif v < 0.40:
            buckets["low_0.20_0.40"] += 1
        elif v < 0.70:
            buckets["mid_0.40_0.70"] += 1
        elif v < 1.00:
            buckets["high_0.70_1.00"] += 1
        else:
            buckets["very_high_1.00_plus"] += 1

    print("value buckets:")
    print(json.dumps(buckets, indent=2))

    source_value_sum = {}
    source_value_count = {}

    for r in final_rows:
        src = r.get("source", "unknown")
        v = float(r.get("value", 1.0))
        source_value_sum[src] = source_value_sum.get(src, 0.0) + v
        source_value_count[src] = source_value_count.get(src, 0) + 1

    print("avg value by source:")
    for src in sorted(source_value_count):
        print(src, source_value_sum[src] / source_value_count[src])

    if avg_value < 0.35:
        print("[warn] average value is quite low; dataset may be dominated by weak examples.")

    if buckets["very_low_0.00_0.20"] > len(values) * 0.35:
        print("[warn] more than 35% of examples are very low value.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", type=Path, required=True)
    ap.add_argument("--out_jsonl", type=Path, required=True)
    ap.add_argument("--num_turn", type=int, default=25000)
    ap.add_argument("--num_replay", type=int, default=6000)
    ap.add_argument("--num_repair", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=28)
    args = ap.parse_args()

    safety_check_path(args.train_jsonl, "train_jsonl")
    safety_check_path(args.out_jsonl, "out_jsonl")

    print("Loading TRAIN episodes:", args.train_jsonl)
    rows = read_jsonl(args.train_jsonl)
    print("train rows:", len(rows))

    print("Building outcome-grounded turn examples...")
    turns_all = build_outcome_turns(rows)
    turns = weighted_sample(turns_all, args.num_turn, args.seed + 1)
    print("outcome turns all:", len(turns_all))
    print("outcome turns sampled:", len(turns))

    print("Building success replay examples...")
    replay_all = build_success_replay(rows)
    replay = weighted_sample(replay_all, args.num_replay, args.seed + 2)
    print("success replay all:", len(replay_all))
    print("success replay sampled:", len(replay))

    print("Building repair examples...")
    repairs = build_repair_examples(rows, args.num_repair, args.seed + 3)
    print("repairs sampled:", len(repairs))

    final_rows = turns + replay + repairs
    rng = random.Random(args.seed)
    rng.shuffle(final_rows)

    print_value_audit(final_rows)

    write_jsonl(args.out_jsonl, final_rows)
    print("wrote:", args.out_jsonl)


if __name__ == "__main__":
    main()
