
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI


GO_RE = re.compile(r"^\s*GO:\s*(north|south|east|west)\s*$", re.I)
DONE_RE = re.compile(r"^\s*DONE\s*$", re.I)

ALLOWED_ROLES = {"system", "user", "assistant"}

DEFAULT_SYSTEM = (
    "You are a strict data-quality judge and synthetic-error generator for a dialogue-game agent. "
    "You must return valid JSON only. Do not use markdown."
)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except Exception as e:
                print(f"[warn] skipped bad JSON line {line_no}: {e}", file=sys.stderr)
    return rows


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def load_done_indices(path: Path) -> set[int]:
    done = set()
    if not path.exists():
        return done

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                idx = obj.get("meta", {}).get("llm_input_index")
                if isinstance(idx, int):
                    done.add(idx)
            except Exception:
                continue

    return done


def is_message(m: Any) -> bool:
    return (
        isinstance(m, dict)
        and m.get("role") in ALLOWED_ROLES
        and isinstance(m.get("content"), str)
    )


def normalize_messages(messages: Any) -> List[Dict[str, str]]:
    if not isinstance(messages, list):
        return []

    out = []
    for m in messages:
        if is_message(m):
            out.append(
                {
                    "role": str(m["role"]),
                    "content": str(m["content"]),
                }
            )
    return out


def get_target_text(row: Dict[str, Any]) -> Optional[str]:
    completion = row.get("completion")

    if isinstance(completion, list):
        parts = []
        for m in completion:
            if isinstance(m, dict) and isinstance(m.get("content"), str):
                parts.append(m["content"].strip())
        text = "\n".join([p for p in parts if p]).strip()
        if text:
            return text

    for key in ["response", "answer", "target", "action"]:
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()

    return None


def get_prompt_messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    prompt = normalize_messages(row.get("prompt"))
    if prompt:
        return prompt

    messages = normalize_messages(row.get("messages"))
    if messages:
        # Remove final assistant answer if this is a full row.
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "assistant":
                return messages[:i]

    return []


def compact_messages_for_llm(messages: List[Dict[str, str]], max_chars: int) -> str:
    """
    Serialize messages for the judge. If too long, keep the latest context,
    because the last game state usually matters most for the next action.
    """
    text = json.dumps(messages, ensure_ascii=False, indent=2)

    if len(text) <= max_chars:
        return text

    return text[-max_chars:]


def valid_target_for_repair(target: str) -> bool:
    """
    We mainly want strict action-format repair. Keep generic support,
    but prioritize GO/DONE-style rows.
    """
    t = target.strip()
    if GO_RE.match(t):
        return True
    if DONE_RE.match(t):
        return True
    if len(t) <= 200:
        return True
    return False


def safe_float(x: Any, default: float) -> float:
    try:
        v = float(x)
        if v != v:
            return default
        return v
    except Exception:
        return default


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def extract_json_object(text: str) -> Dict[str, Any]:
    """
    Robust JSON extraction in case the API returns text around JSON.
    """
    text = text.strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start : end + 1]
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj

    raise ValueError("No valid JSON object found")


def build_llm_user_prompt(
    row_index: int,
    prompt_messages: List[Dict[str, str]],
    target: str,
    original_weight: float,
    max_context_chars: int,
    max_bad_outputs: int,
) -> str:
    context = compact_messages_for_llm(prompt_messages, max_context_chars)

    return f"""
You are judging one training example for a dialogue-game agent.

The agent must obey a strict output contract:
- Output exactly one final answer.
- For movement tasks, valid movement answers look like exactly:
  GO: north
  GO: south
  GO: east
  GO: west
- If the task is complete, the answer should be exactly:
  DONE
- No explanation.
- No markdown.
- No role prefix.
- No punctuation around the answer.
- No stop tokens.
- No multiple actions.

You are given:
1. The dialogue state/history BEFORE the assistant answer.
2. The original gold assistant answer from a successful or high-quality training trace.
3. The old Stage 2 training weight.

You must NOT invent a new gold answer.
The clean target must remain the provided gold answer.

Your tasks:
A. Score how good this training row is for teaching strict, correct dialogue-game behaviour.
B. Generate realistic bad assistant outputs that a smaller model might produce for this exact example.
   These bad outputs should be close to the gold answer but invalid or less strict.
   Examples of badness:
   - extra explanation
   - missing GO: prefix
   - wrong casing
   - punctuation
   - markdown
   - role leakage
   - stop-token leakage
   - multiple actions
   - premature DONE
   - verbose DONE

Return JSON only with this schema:
{{
  "format_score": 0.0,
  "action_score": 0.0,
  "task_progress_score": 0.0,
  "done_score": 0.0,
  "overall_score": 0.0,
  "keep": true,
  "reason": "short reason",
  "bad_outputs": ["bad output 1", "bad output 2"]
}}

Scoring rules:
- Scores must be floats from 0.0 to 1.0.
- "overall_score" should summarize whether the row is useful for training.
- "keep" should be false only if the row looks malformed, useless, contradictory, or too ambiguous.
- bad_outputs must contain at most {max_bad_outputs} strings.
- bad_outputs must NOT include the exact clean target.
- bad_outputs must NOT be empty unless keep is false.

Row index:
{row_index}

Old Stage 2 weight:
{original_weight}

Dialogue state/history before answer:
{context}

Gold assistant answer:
{target}
""".strip()


async def call_deepinfra_judge(
    client: AsyncOpenAI,
    model: str,
    row_index: int,
    prompt_messages: List[Dict[str, str]],
    target: str,
    original_weight: float,
    max_context_chars: int,
    max_bad_outputs: int,
    temperature: float,
    max_tokens: int,
    retries: int,
) -> Dict[str, Any]:
    user_prompt = build_llm_user_prompt(
        row_index=row_index,
        prompt_messages=prompt_messages,
        target=target,
        original_weight=original_weight,
        max_context_chars=max_context_chars,
        max_bad_outputs=max_bad_outputs,
    )

    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]

    last_err = None

    for attempt in range(retries + 1):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )

            content = resp.choices[0].message.content or ""
            obj = extract_json_object(content)

            usage = getattr(resp, "usage", None)
            if usage is not None:
                obj["_usage"] = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }

            return obj

        except Exception as e:
            last_err = e
            sleep_s = min(30.0, 2.0 ** attempt + random.random())
            print(
                f"[warn] row={row_index} attempt={attempt + 1}/{retries + 1} failed: {e}; sleep {sleep_s:.1f}s",
                file=sys.stderr,
            )
            await asyncio.sleep(sleep_s)

    raise RuntimeError(f"DeepInfra call failed after retries for row {row_index}: {last_err}")


def normalize_judge_output(
    judge: Dict[str, Any],
    target: str,
    max_bad_outputs: int,
) -> Dict[str, Any]:
    format_score = clamp(safe_float(judge.get("format_score"), 0.5), 0.0, 1.0)
    action_score = clamp(safe_float(judge.get("action_score"), 0.5), 0.0, 1.0)
    task_progress_score = clamp(safe_float(judge.get("task_progress_score"), 0.5), 0.0, 1.0)
    done_score = clamp(safe_float(judge.get("done_score"), 0.5), 0.0, 1.0)
    overall_score = clamp(safe_float(judge.get("overall_score"), 0.5), 0.0, 1.0)

    keep = judge.get("keep", True)
    keep = bool(keep)

    reason = judge.get("reason", "")
    if not isinstance(reason, str):
        reason = str(reason)

    bad_outputs = judge.get("bad_outputs", [])
    if not isinstance(bad_outputs, list):
        bad_outputs = []

    clean_target = target.strip()
    cleaned_bad = []
    seen = set()

    for x in bad_outputs:
        if not isinstance(x, str):
            continue
        b = x.strip()
        if not b:
            continue
        if b == clean_target:
            continue
        if len(b) > 300:
            continue
        if b in seen:
            continue
        seen.add(b)
        cleaned_bad.append(b)

    cleaned_bad = cleaned_bad[:max_bad_outputs]

    return {
        "format_score": format_score,
        "action_score": action_score,
        "task_progress_score": task_progress_score,
        "done_score": done_score,
        "overall_score": overall_score,
        "keep": keep,
        "reason": reason[:1000],
        "bad_outputs": cleaned_bad,
        "usage": judge.get("_usage", {}),
        "raw": judge,
    }


def make_completion(target: str) -> List[Dict[str, str]]:
    return [{"role": "assistant", "content": target.strip()}]


def make_judged_turn_row(
    row: Dict[str, Any],
    row_index: int,
    target: str,
    prompt_messages: List[Dict[str, str]],
    judge: Dict[str, Any],
    min_weight: float,
    max_weight: float,
    replace_weight: bool,
) -> Optional[Dict[str, Any]]:
    old_value = safe_float(row.get("value", row.get("weight", 1.0)), 1.0)
    judge_score = float(judge["overall_score"])

    if replace_weight:
        new_value = judge_score
    else:
        # Mild modifier: preserve original Stage 2 weighting signal.
        modifier = 0.75 + 0.50 * judge_score
        new_value = old_value * modifier

    new_value = clamp(new_value, min_weight, max_weight)

    if not judge["keep"]:
        return None

    out = dict(row)
    out["prompt"] = prompt_messages
    out["completion"] = make_completion(target)
    out["response"] = target.strip()
    out["value"] = float(new_value)
    out["weight"] = float(new_value)
    out["source"] = "llm_judged_stage2_turn"

    meta = out.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}

    meta.update(
        {
            "llm_input_index": row_index,
            "llm_judge_model": "Qwen/Qwen3.6-35B-A3B",
            "llm_original_value": old_value,
            "llm_new_value": new_value,
            "llm_overall_score": judge["overall_score"],
            "llm_format_score": judge["format_score"],
            "llm_action_score": judge["action_score"],
            "llm_task_progress_score": judge["task_progress_score"],
            "llm_done_score": judge["done_score"],
            "llm_keep": judge["keep"],
            "llm_reason": judge["reason"],
            "llm_usage": judge.get("usage", {}),
        }
    )

    out["meta"] = meta
    return out


def make_repair_rows(
    row_index: int,
    prompt_messages: List[Dict[str, str]],
    target: str,
    bad_outputs: List[str],
    judge: Dict[str, Any],
    repair_weight: float,
    max_context_chars: int,
) -> List[Dict[str, Any]]:
    if not valid_target_for_repair(target):
        return []

    context = compact_messages_for_llm(prompt_messages, max_context_chars)
    rows = []

    for bad in bad_outputs:
        repair_prompt = (
            "You are playing a dialogue game. The previous assistant answer was invalid because "
            "it did not follow the exact required output format or it contained extra text.\n\n"
            "Game state/history before the assistant answer:\n"
            f"{context}\n\n"
            "Invalid assistant answer:\n"
            f"{bad}\n\n"
            "Rewrite the assistant answer correctly. Output only one final valid answer and nothing else."
        )

        ex = {
            "prompt": [{"role": "user", "content": repair_prompt}],
            "completion": make_completion(target),
            "response": target.strip(),
            "messages": [
                {"role": "user", "content": repair_prompt},
                {"role": "assistant", "content": target.strip()},
            ],
            "value": float(repair_weight),
            "weight": float(repair_weight),
            "source": "llm_generated_format_repair",
            "meta": {
                "llm_input_index": row_index,
                "repair_type": "llm_strict_format",
                "bad_output": bad,
                "llm_overall_score": judge["overall_score"],
                "llm_reason": judge["reason"],
            },
        }
        rows.append(ex)

    return rows


def make_dpo_rows(
    row_index: int,
    prompt_messages: List[Dict[str, str]],
    target: str,
    bad_outputs: List[str],
    judge: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows = []

    for bad in bad_outputs:
        rows.append(
            {
                "prompt": prompt_messages,
                "chosen": make_completion(target),
                "rejected": [{"role": "assistant", "content": bad}],
                "source": "llm_generated_format_dpo",
                "meta": {
                    "llm_input_index": row_index,
                    "failure_type": "strict_format",
                    "bad_output": bad,
                    "llm_overall_score": judge["overall_score"],
                    "llm_reason": judge["reason"],
                },
            }
        )

    return rows


async def process_one(
    row_index: int,
    row: Dict[str, Any],
    client: AsyncOpenAI,
    args: argparse.Namespace,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Optional[str]]:
    target = get_target_text(row)
    if not target:
        return None, [], [], "missing target"

    prompt_messages = get_prompt_messages(row)
    if not prompt_messages:
        return None, [], [], "missing prompt messages"

    original_weight = safe_float(row.get("value", row.get("weight", 1.0)), 1.0)

    judge_raw = await call_deepinfra_judge(
        client=client,
        model=args.model,
        row_index=row_index,
        prompt_messages=prompt_messages,
        target=target,
        original_weight=original_weight,
        max_context_chars=args.max_context_chars,
        max_bad_outputs=args.max_bad_outputs,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        retries=args.retries,
    )

    judge = normalize_judge_output(
        judge=judge_raw,
        target=target,
        max_bad_outputs=args.max_bad_outputs,
    )

    judged_row = make_judged_turn_row(
        row=row,
        row_index=row_index,
        target=target,
        prompt_messages=prompt_messages,
        judge=judge,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
        replace_weight=args.replace_weight,
    )

    if judged_row is None:
        return None, [], [], "judge keep=false"

    repair_rows = make_repair_rows(
        row_index=row_index,
        prompt_messages=prompt_messages,
        target=target,
        bad_outputs=judge["bad_outputs"],
        judge=judge,
        repair_weight=args.repair_weight,
        max_context_chars=args.max_context_chars,
    )

    dpo_rows = make_dpo_rows(
        row_index=row_index,
        prompt_messages=prompt_messages,
        target=target,
        bad_outputs=judge["bad_outputs"],
        judge=judge,
    )

    return judged_row, repair_rows, dpo_rows, None


async def main_async(args: argparse.Namespace) -> None:
    token = os.environ.get("DEEPINFRA_TOKEN")
    if not token:
        raise RuntimeError("DEEPINFRA_TOKEN is not set.")

    args.out_judged_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.out_repair_jsonl:
        args.out_repair_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.out_dpo_jsonl:
        args.out_dpo_jsonl.parent.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(args.in_jsonl)
    print(f"loaded rows: {len(rows)}")

    if args.shuffle:
        rng = random.Random(args.seed)
        indexed = list(enumerate(rows))
        rng.shuffle(indexed)
    else:
        indexed = list(enumerate(rows))

    if args.max_rows is not None and args.max_rows > 0:
        indexed = indexed[: args.max_rows]

    done = load_done_indices(args.out_judged_jsonl) if args.resume else set()
    indexed = [(i, r) for i, r in indexed if i not in done]

    print(f"already done: {len(done)}")
    print(f"to process:   {len(indexed)}")
    print(f"concurrency:  {args.concurrency}")

    if args.dry_run:
        for i, r in indexed[:3]:
            target = get_target_text(r)
            prompt_messages = get_prompt_messages(r)
            old_weight = safe_float(r.get("value", r.get("weight", 1.0)), 1.0)
            print("=" * 80)
            print(
                build_llm_user_prompt(
                    row_index=i,
                    prompt_messages=prompt_messages,
                    target=target or "",
                    original_weight=old_weight,
                    max_context_chars=args.max_context_chars,
                    max_bad_outputs=args.max_bad_outputs,
                )
            )
        return

    client = AsyncOpenAI(
        api_key=token,
        base_url=args.base_url,
    )

    queue: asyncio.Queue = asyncio.Queue()
    write_lock = asyncio.Lock()

    for item in indexed:
        await queue.put(item)

    for _ in range(args.concurrency):
        await queue.put(None)

    stats = {
        "processed": 0,
        "judged_written": 0,
        "repair_written": 0,
        "dpo_written": 0,
        "skipped": 0,
        "failed": 0,
        "start_time": time.time(),
    }

    async def worker(worker_id: int) -> None:
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                return

            row_index, row = item

            try:
                judged_row, repair_rows, dpo_rows, skip_reason = await process_one(
                    row_index=row_index,
                    row=row,
                    client=client,
                    args=args,
                )

                async with write_lock:
                    stats["processed"] += 1

                    if judged_row is None:
                        stats["skipped"] += 1
                        print(f"[skip] row={row_index} reason={skip_reason}")
                    else:
                        append_jsonl(args.out_judged_jsonl, judged_row)
                        stats["judged_written"] += 1

                        if args.out_repair_jsonl:
                            for rr in repair_rows[: args.max_repairs_per_row]:
                                append_jsonl(args.out_repair_jsonl, rr)
                                stats["repair_written"] += 1

                        if args.out_dpo_jsonl:
                            for dr in dpo_rows[: args.max_repairs_per_row]:
                                append_jsonl(args.out_dpo_jsonl, dr)
                                stats["dpo_written"] += 1

                    if stats["processed"] % args.log_every == 0:
                        elapsed = max(1.0, time.time() - stats["start_time"])
                        rate = stats["processed"] / elapsed
                        print(
                            f"[progress] processed={stats['processed']} "
                            f"judged={stats['judged_written']} "
                            f"repair={stats['repair_written']} "
                            f"dpo={stats['dpo_written']} "
                            f"skipped={stats['skipped']} "
                            f"failed={stats['failed']} "
                            f"rate={rate:.2f}/s"
                        )

            except Exception as e:
                async with write_lock:
                    stats["failed"] += 1
                    print(f"[fail] row={row_index}: {e}", file=sys.stderr)

            finally:
                queue.task_done()

    workers = [asyncio.create_task(worker(i)) for i in range(args.concurrency)]
    await queue.join()

    for w in workers:
        await w

    elapsed = max(1.0, time.time() - stats["start_time"])
    print("done")
    print(json.dumps(stats, indent=2))
    print(f"elapsed seconds: {elapsed:.1f}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("--in_jsonl", type=Path, required=True)
    ap.add_argument("--out_judged_jsonl", type=Path, required=True)
    ap.add_argument("--out_repair_jsonl", type=Path, default=None)
    ap.add_argument("--out_dpo_jsonl", type=Path, default=None)

    ap.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--base_url", default="https://api.deepinfra.com/v1/openai")

    ap.add_argument("--concurrency", type=int, default=15)
    ap.add_argument("--retries", type=int, default=5)

    ap.add_argument("--temperature", type=float, default=0.35)
    ap.add_argument("--max_tokens", type=int, default=700)
    ap.add_argument("--max_context_chars", type=int, default=7000)
    ap.add_argument("--max_bad_outputs", type=int, default=4)
    ap.add_argument("--max_repairs_per_row", type=int, default=2)

    ap.add_argument("--repair_weight", type=float, default=0.75)

    ap.add_argument("--min_weight", type=float, default=0.40)
    ap.add_argument("--max_weight", type=float, default=1.20)
    ap.add_argument(
        "--replace_weight",
        action="store_true",
        help="If set, replace original weight with judge score. Default: original_weight × judge_modifier.",
    )

    ap.add_argument("--max_rows", type=int, default=None)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=28)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--log_every", type=int, default=100)

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()