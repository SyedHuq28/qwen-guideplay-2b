

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


GO_RE = re.compile(r"^\s*GO:\s*(north|south|east|west)\s*$", re.I)
DONE_RE = re.compile(r"^\s*DONE\s*$", re.I)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        print(f"[warn] missing file: {path}")
        return rows

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


def valid_messages(messages: Any) -> bool:
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    for m in messages:
        if not isinstance(m, dict):
            return False
        if m.get("role") not in {"user", "assistant", "system"}:
            return False
        if not isinstance(m.get("content"), str):
            return False
    return any(m.get("role") == "assistant" for m in messages)


def normalize_messages(row: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    if valid_messages(row.get("messages")):
        return [{"role": m["role"], "content": m["content"]} for m in row["messages"]]

    # Flexible turn-level fallbacks.
    prompt = row.get("prompt") or row.get("input") or row.get("state")
    answer = (
        row.get("response")
        or row.get("answer")
        or row.get("target")
        or row.get("action")
        or row.get("completion")
    )

    if isinstance(prompt, list):
        # Sometimes prompt may already be messages.
        if valid_messages(prompt):
            msgs = [{"role": m["role"], "content": m["content"]} for m in prompt]
            if answer and (not msgs or msgs[-1]["role"] != "assistant"):
                msgs.append({"role": "assistant", "content": str(answer)})
            return msgs

    if isinstance(prompt, str) and isinstance(answer, str):
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]

    return None


def get_last_user_and_answer(messages: List[Dict[str, str]]) -> Optional[tuple[str, str]]:
    last_user = None
    for m in messages:
        if m["role"] == "user":
            last_user = m["content"]
        elif m["role"] == "assistant" and last_user is not None:
            ans = m["content"].strip()
            if ans:
                return last_user, ans
    return None


def get_weight(row: Dict[str, Any]) -> float:
    for key in ["weight", "score", "value", "sample_weight", "quality"]:
        if key in row:
            try:
                return float(row[key])
            except Exception:
                pass
    return 1.0


def sample_rows(rows: List[Any], n: int, seed: int) -> List[Any]:
    rng = random.Random(seed)
    if len(rows) <= n:
        rng.shuffle(rows)
        return rows
    return rng.sample(rows, n)


def make_replay_examples(success_rows: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    examples = []
    for row in success_rows:
        msgs = normalize_messages(row)
        if not msgs:
            continue
        # Keep full successful dialogues. These preserve global game flow.
        examples.append({
            "messages": msgs,
            "source": "full_dialogue_replay",
            "weight": 1.0,
        })

    return sample_rows(examples, n, seed)


def make_turn_examples(turn_rows: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    examples = []
    for row in turn_rows:
        w = get_weight(row)
        if w < 0.55:
            continue
        msgs = normalize_messages(row)
        if not msgs:
            continue
        examples.append({
            "messages": msgs,
            "source": "weighted_turn_replay",
            "weight": w,
        })

    # Prefer high-value examples but still shuffle.
    examples.sort(key=lambda x: x.get("weight", 1.0), reverse=True)
    examples = examples[: max(n * 3, n)]
    return sample_rows(examples, n, seed)


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

    else:
        # Generic strict-answer corruption.
        if len(good) <= 200:
            bads.extend([
                f"Sure. {good}",
                f"{good}<|im_end|>",
                f"{good}||",
                f"```text\n{good}\n```",
            ])

    return bads


def make_repair_examples(success_rows: List[Dict[str, Any]], turn_rows: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    pool = []

    for row in turn_rows + success_rows:
        msgs = normalize_messages(row)
        if not msgs:
            continue
        pair = get_last_user_and_answer(msgs)
        if not pair:
            continue

        user_prompt, good = pair
        good = good.strip()
        if not good or len(good) > 500:
            continue

        bads = corrupt_answer(good)
        for bad in bads:
            repair_prompt = (
                user_prompt.strip()
                + "\n\nThe previous assistant answer was invalid because it contained extra text, stop tokens, "
                  "markdown, or the wrong format:\n"
                + bad
                + "\n\nRewrite the answer correctly. Output only the final valid answer and nothing else."
            )
            pool.append({
                "messages": [
                    {"role": "user", "content": repair_prompt},
                    {"role": "assistant", "content": good},
                ],
                "source": "repair_from_train_trace",
                "weight": 1.0,
            })

    # Add generic strict movement examples, not from validation.
    directions = ["north", "south", "east", "west"]
    rooms = ["Kitchen", "Library", "Lobby", "Garden", "Bedroom", "Office", "Hallway", "Cellar"]
    for _ in range(1000):
        room = rng.choice(rooms)
        dirs = rng.sample(directions, k=rng.randint(1, 3))
        chosen = rng.choice(dirs)
        prompt = (
            "Please help me with the following task. The goal is to visit all rooms. "
            "To move, answer exactly with GO: DIRECTION, where DIRECTION is one of "
            "[north, south, east, west]. To stop, answer DONE. Omit any other text.\n"
            f"You are in the {room}. Currently available directions: {', '.join(dirs)}. "
            "What is your next instruction?"
        )
        pool.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": f"GO: {chosen}"},
            ],
            "source": "generic_strict_format_move",
            "weight": 1.0,
        })

    if len(pool) <= n:
        rng.shuffle(pool)
        return pool

    return rng.sample(pool, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--success_jsonl", type=Path, required=True)
    ap.add_argument("--turn_jsonl", type=Path, required=True)
    ap.add_argument("--out_jsonl", type=Path, required=True)
    ap.add_argument("--num_replay", type=int, default=8000)
    ap.add_argument("--num_turn", type=int, default=20000)
    ap.add_argument("--num_repair", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=28)
    args = ap.parse_args()

    print("Loading success rows:", args.success_jsonl)
    success_rows = read_jsonl(args.success_jsonl)
    print("success rows:", len(success_rows))

    print("Loading turn rows:", args.turn_jsonl)
    turn_rows = read_jsonl(args.turn_jsonl)
    print("turn rows:", len(turn_rows))

    replay = make_replay_examples(success_rows, args.num_replay, args.seed + 1)
    turns = make_turn_examples(turn_rows, args.num_turn, args.seed + 2)
    repairs = make_repair_examples(success_rows, turn_rows, args.num_repair, args.seed + 3)

    rows = replay + turns + repairs
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    print("final counts:")
    counts = {}
    for r in rows:
        counts[r["source"]] = counts.get(r["source"], 0) + 1
    print(json.dumps(counts, indent=2))
    print("total:", len(rows))

    write_jsonl(args.out_jsonl, rows)
    print("wrote:", args.out_jsonl)


if __name__ == "__main__":
    main()
