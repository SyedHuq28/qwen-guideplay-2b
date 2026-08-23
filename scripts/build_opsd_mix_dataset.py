
import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


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

GO_RE = re.compile(r"^\s*GO:\s*(north|south|east|west)\s*$", re.I)
DONE_RE = re.compile(r"^\s*DONE\s*$", re.I)
JSON_ACTION_RE = re.compile(r'^\s*\{\s*"action"\s*:', re.I)
GUESS_RE = re.compile(r"^\s*(guess|answer)\s*:", re.I)
EXPLANATION_GUESS_RE = re.compile(r"explanation\s*:.*guess\s*:", re.I | re.S)


def safety_check_path(path: Path, name: str, allow_dir: bool = False) -> None:
    s = str(path).lower()
    for token in FORBIDDEN_PATH_TOKENS:
        if token in s:
            raise ValueError(
                f"Refusing possible validation/test/eval/public path for {name}: {path}"
            )

    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")

    if path.is_dir() and not allow_dir:
        raise ValueError(f"{name} is a directory but a file was expected: {path}")


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
                print(f"WARNING: skipped bad JSONL line {line_no} in {path}: {e}")
    return rows


def read_json_file(path: Path) -> List[Dict[str, Any]]:
    try:
        obj = json.loads(path.read_text())
    except Exception as e:
        print(f"WARNING: skipped unreadable JSON file {path}: {e}")
        return []

    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]

    if isinstance(obj, dict):
        return [obj]

    return []


def iter_raw_objects(path: Path) -> Iterable[Dict[str, Any]]:
    if path.is_file():
        if path.suffix.lower() == ".jsonl":
            yield from read_jsonl(path)
        elif path.suffix.lower() == ".json":
            yield from read_json_file(path)
        else:
            raise ValueError(f"Unsupported rollout input file type: {path}")
        return

    for p in sorted(path.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() == ".jsonl":
            yield from read_jsonl(p)
        elif p.suffix.lower() == ".json":
            yield from read_json_file(p)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def is_message(m: Any) -> bool:
    return (
        isinstance(m, dict)
        and m.get("role") in {"system", "user", "assistant"}
        and isinstance(m.get("content"), str)
    )


def valid_messages(messages: Any) -> bool:
    return (
        isinstance(messages, list)
        and len(messages) >= 2
        and all(is_message(m) for m in messages)
        and any(m.get("role") == "assistant" for m in messages)
    )


def normalize_role(role: Any) -> Optional[str]:
    if role is None:
        return None

    r = str(role).strip().lower()

    if r in {"assistant", "model", "agent", "player", "bot"}:
        return "assistant"

    if r in {"user", "human", "environment", "env", "game", "gm", "system_user"}:
        return "user"

    if r == "system":
        return "system"

    return None


def normalize_messages_from_list(items: Any) -> Optional[List[Dict[str, str]]]:
    if not isinstance(items, list):
        return None

    messages = []

    for item in items:
        if isinstance(item, dict):
            role = normalize_role(
                item.get("role")
                or item.get("speaker")
                or item.get("from")
                or item.get("author")
                or item.get("name")
            )

            content = (
                item.get("content")
                or item.get("text")
                or item.get("utterance")
                or item.get("message")
                or item.get("response")
                or item.get("action")
            )

            if role and isinstance(content, str) and content.strip():
                messages.append({"role": role, "content": content.strip()})

        elif isinstance(item, str):
            # Last-resort parsing of simple alternating logs is intentionally avoided,
            # because role ambiguity can corrupt training data.
            continue

    if valid_messages(messages):
        return messages

    return None


def find_messages_deep(obj: Any, depth: int = 0, max_depth: int = 5) -> Optional[List[Dict[str, str]]]:
    if depth > max_depth:
        return None

    if valid_messages(obj):
        return [{"role": m["role"], "content": m["content"]} for m in obj]

    if isinstance(obj, dict):
        # Direct common keys.
        for key in [
            "messages",
            "conversation",
            "dialogue",
            "dialog",
            "turns",
            "interactions",
            "transcript",
            "history",
            "records",
        ]:
            if key in obj:
                msgs = normalize_messages_from_list(obj[key])
                if msgs:
                    return msgs

                deeper = find_messages_deep(obj[key], depth + 1, max_depth)
                if deeper:
                    return deeper

        # Prompt/response fallback.
        prompt = obj.get("prompt") or obj.get("input") or obj.get("state")
        response = (
            obj.get("response")
            or obj.get("completion")
            or obj.get("answer")
            or obj.get("target")
            or obj.get("action")
        )

        if isinstance(prompt, list) and isinstance(response, str):
            prompt_msgs = normalize_messages_from_list(prompt)
            if prompt_msgs:
                messages = prompt_msgs + [{"role": "assistant", "content": response.strip()}]
                if valid_messages(messages):
                    return messages

        if isinstance(prompt, str) and isinstance(response, str):
            messages = [
                {"role": "user", "content": prompt.strip()},
                {"role": "assistant", "content": response.strip()},
            ]
            if valid_messages(messages):
                return messages

        # Recursive search.
        for v in obj.values():
            deeper = find_messages_deep(v, depth + 1, max_depth)
            if deeper:
                return deeper

    if isinstance(obj, list):
        msgs = normalize_messages_from_list(obj)
        if msgs:
            return msgs

        for item in obj:
            deeper = find_messages_deep(item, depth + 1, max_depth)
            if deeper:
                return deeper

    return None


def get_outcome(obj: Dict[str, Any]) -> str:
    candidates = []

    for key in [
        "outcome",
        "episode_outcome",
        "status",
        "result",
        "game_outcome",
        "final_status",
    ]:
        candidates.append(obj.get(key))

    meta = obj.get("meta")
    if isinstance(meta, dict):
        for key in [
            "outcome",
            "episode_outcome",
            "status",
            "result",
            "game_outcome",
            "final_status",
        ]:
            candidates.append(meta.get(key))

    scores = obj.get("scores")
    if isinstance(scores, dict):
        if scores.get("Success") == 1 or scores.get("success") == 1:
            return "success"
        if scores.get("Aborted") == 1 or scores.get("aborted") == 1:
            return "aborted"
        if scores.get("Lose") == 1 or scores.get("lose") == 1:
            return "failure"

    for c in candidates:
        if isinstance(c, str) and c.strip():
            return c.strip().lower()

        if isinstance(c, bool):
            return "success" if c else "failure"

    # Last-resort score-style fields.
    for key in ["success", "won", "win"]:
        v = obj.get(key)
        if v is True or v == 1:
            return "success"

    for key in ["aborted", "failed", "failure"]:
        v = obj.get(key)
        if v is True or v == 1:
            return "aborted" if key == "aborted" else "failure"

    return "unknown"


def is_success(outcome: str) -> bool:
    return outcome.lower() in {
        "success",
        "successful",
        "win",
        "won",
        "complete",
        "completed",
        "solved",
        "passed",
    }


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

    if any(bad in t for bad in ["<|im_start|>", "<|im_end|>", "||", "```"]):
        return 0.25

    if len(t) <= 120 and not any(bad in t.lower() for bad in ["user\n", "assistant\n"]):
        return 0.80

    if len(t) <= 400:
        return 0.55

    return 0.25


def to_weighted_sft_row(
    messages: List[Dict[str, str]],
    value: float,
    source: str,
    meta: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
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

    return {
        "prompt": prompt,
        "completion": [{"role": "assistant", "content": response}],
        "response": response,
        "messages": messages,
        "value": float(value),
        "weight": float(value),
        "source": source,
        "meta": meta or {},
    }


def normalize_existing_stage2_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Accepts either already weighted-turn rows or raw message rows.
    Preserves Stage 2 values when available.
    """
    if "prompt" in row and "completion" in row:
        prompt = row.get("prompt")
        completion = row.get("completion")

        if isinstance(prompt, list) and isinstance(completion, list):
            value = float(row.get("value", row.get("weight", 1.0)))
            out = dict(row)
            out["value"] = value
            out["weight"] = value
            out["source"] = row.get("source", "stage2_weighted_turn_replay")
            return out

    messages = find_messages_deep(row)
    if not messages:
        return None

    value = float(row.get("value", row.get("weight", 1.0)))
    return to_weighted_sft_row(
        messages=messages,
        value=value,
        source="stage2_weighted_turn_replay",
        meta={"original_source": row.get("source", "unknown")},
    )


def assistant_turn_indices(messages: List[Dict[str, str]]) -> List[int]:
    return [i for i, m in enumerate(messages) if m.get("role") == "assistant"]


def build_onpolicy_success_turns(raw_objects: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    examples = []
    outcome_counts = {}

    total_objects = 0
    success_objects = 0
    parsed_success_objects = 0

    for obj in raw_objects:
        total_objects += 1

        outcome = get_outcome(obj)
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1

        if not is_success(outcome):
            continue

        success_objects += 1

        messages = find_messages_deep(obj)
        if not messages:
            continue

        parsed_success_objects += 1

        indices = assistant_turn_indices(messages)
        total_turns = len(indices)
        if total_turns == 0:
            continue

        for turn_number, idx in enumerate(indices):
            answer = messages[idx]["content"].strip()
            if not answer:
                continue

            fmt = response_format_score(answer)

            # On-policy successes are trusted, but malformed outputs are still risky.
            if fmt < 0.55:
                continue

            if total_turns <= 1:
                turn_discount = 1.0
            else:
                progress = turn_number / max(total_turns - 1, 1)
                turn_discount = math.exp(-0.35 * progress)

            stop_bonus = 1.10 if DONE_RE.match(answer) else 1.0

            value = 1.0 * fmt * turn_discount * stop_bonus
            value = max(0.50, min(1.20, value))

            ex = to_weighted_sft_row(
                messages=messages[:idx + 1],
                value=value,
                source="opsd_onpolicy_faipd_rr_success_turn",
                meta={
                    "outcome": outcome,
                    "turn_number": turn_number,
                    "total_turns": total_turns,
                    "format_score": fmt,
                    "turn_discount": turn_discount,
                    "stop_bonus": stop_bonus,
                },
            )

            if ex is not None:
                examples.append(ex)

    print("raw rollout objects:", total_objects)
    print("rollout outcome counts:")
    print(json.dumps(outcome_counts, indent=2))
    print("successful rollout objects:", success_objects)
    print("parsed successful rollout objects:", parsed_success_objects)
    print("on-policy success turn examples:", len(examples))

    return examples


def sample_rows(rows: List[Dict[str, Any]], n: int, seed: int, prefer_high_value: bool = False) -> List[Dict[str, Any]]:
    rng = random.Random(seed)

    if len(rows) <= n:
        rows = list(rows)
        rng.shuffle(rows)
        return rows

    if prefer_high_value:
        sorted_rows = sorted(rows, key=lambda r: float(r.get("value", 1.0)), reverse=True)
        pool = sorted_rows[: max(n * 3, n)]
        return rng.sample(pool, n)

    return rng.sample(rows, n)


def audit(rows: List[Dict[str, Any]]) -> None:
    counts = {}
    values = []

    for r in rows:
        counts[r.get("source", "unknown")] = counts.get(r.get("source", "unknown"), 0) + 1
        values.append(float(r.get("value", 1.0)))

    print("final source counts:")
    print(json.dumps(counts, indent=2))
    print("total:", len(rows))

    if values:
        print("value min/max/avg:", min(values), max(values), sum(values) / len(values))

    bad_schema = 0
    for r in rows[:1000]:
        if not isinstance(r.get("prompt"), list):
            bad_schema += 1
        if not isinstance(r.get("completion"), list):
            bad_schema += 1
        if "value" not in r:
            bad_schema += 1

    print("schema problems in first 1000 checks:", bad_schema)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2_jsonl", type=Path, required=True)
    ap.add_argument("--onpolicy_input", type=Path, required=True)
    ap.add_argument("--out_jsonl", type=Path, required=True)
    ap.add_argument("--total_examples", type=int, default=30000)
    ap.add_argument("--onpolicy_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=28)
    args = ap.parse_args()

    if args.onpolicy_ratio <= 0 or args.onpolicy_ratio >= 0.5:
        raise ValueError("For safety, onpolicy_ratio must be >0 and <0.5")

    safety_check_path(args.stage2_jsonl, "stage2_jsonl")
    safety_check_path(args.onpolicy_input, "onpolicy_input", allow_dir=True)
    safety_check_path(args.out_jsonl.parent, "out_jsonl parent", allow_dir=True)

    print("Loading Stage 2 weighted-turn data:", args.stage2_jsonl)
    stage2_raw = read_jsonl(args.stage2_jsonl)

    stage2_rows = []
    for row in stage2_raw:
        ex = normalize_existing_stage2_row(row)
        if ex is not None:
            stage2_rows.append(ex)

    print("stage2 raw rows:", len(stage2_raw))
    print("stage2 usable rows:", len(stage2_rows))

    if len(stage2_rows) < 1000:
        raise SystemExit("Too few usable Stage 2 rows. Stop.")

    print("Loading on-policy FAIPD-RR train rollouts:", args.onpolicy_input)
    onpolicy_rows = build_onpolicy_success_turns(iter_raw_objects(args.onpolicy_input))

    if len(onpolicy_rows) < 100:
        raise SystemExit(
            "Too few on-policy success rows. Stop. "
            "Check that the rollout file is train-only, has outcome labels, and contains transcripts."
        )

    n_onpolicy = int(args.total_examples * args.onpolicy_ratio)
    n_stage2 = args.total_examples - n_onpolicy

    stage2_sample = sample_rows(
        stage2_rows,
        n_stage2,
        seed=args.seed + 1,
        prefer_high_value=False,
    )

    onpolicy_sample = sample_rows(
        onpolicy_rows,
        n_onpolicy,
        seed=args.seed + 2,
        prefer_high_value=True,
    )

    final_rows = stage2_sample + onpolicy_sample
    random.Random(args.seed).shuffle(final_rows)

    print("sampled stage2 rows:", len(stage2_sample))
    print("sampled on-policy rows:", len(onpolicy_sample))

    audit(final_rows)

    write_jsonl(args.out_jsonl, final_rows)
    print("wrote:", args.out_jsonl)


if __name__ == "__main__":
    main()
