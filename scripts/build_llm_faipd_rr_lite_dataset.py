

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional


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


def safety_check_path(path: Path, name: str, must_exist: bool = True) -> None:
    s = str(path).lower()

    for token in FORBIDDEN_PATH_TOKENS:
        if token in s:
            raise ValueError(f"Refusing possible validation/test/eval path for {name}: {path}")

    if must_exist and not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[warn] bad JSON line {line_no} in {path}: {e}")
                continue

            if isinstance(obj, dict):
                rows.append(obj)

    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def is_message(m: Any) -> bool:
    return (
        isinstance(m, dict)
        and m.get("role") in {"system", "user", "assistant"}
        and isinstance(m.get("content"), str)
    )


def valid_messages(messages: Any) -> bool:
    return isinstance(messages, list) and len(messages) >= 1 and all(is_message(m) for m in messages)


def normalize_completion(completion: Any) -> Optional[List[Dict[str, str]]]:
    if isinstance(completion, list) and valid_messages(completion):
        out = []

        for m in completion:
            if m["role"] == "assistant":
                out.append({"role": "assistant", "content": m["content"]})

        if out:
            return out

    if isinstance(completion, str) and completion.strip():
        return [{"role": "assistant", "content": completion.strip()}]

    return None


def completion_text(row: Dict[str, Any]) -> str:
    comp = row.get("completion", [])

    if isinstance(comp, list):
        parts = []

        for m in comp:
            if isinstance(m, dict) and isinstance(m.get("content"), str):
                text = m["content"].strip()
                if text:
                    parts.append(text)

        if parts:
            return "\n".join(parts).strip()

    response = row.get("response")
    if isinstance(response, str):
        return response.strip()

    return ""


def normalize_sft_row(row: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
    prompt = row.get("prompt")
    completion = normalize_completion(row.get("completion"))

    if not valid_messages(prompt) or not completion:
        return None

    response = completion_text({"completion": completion, "response": row.get("response", "")})
    if not response:
        return None

    try:
        value = float(row.get("value", row.get("weight", 1.0)))
    except Exception:
        value = 1.0

    out = dict(row)
    out["prompt"] = [{"role": m["role"], "content": m["content"]} for m in prompt]
    out["completion"] = completion
    out["response"] = response
    out["value"] = value
    out["weight"] = value
    out["source"] = source

    meta = out.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    meta["mixed_source_original"] = row.get("source", "unknown")
    out["meta"] = meta

    return out


def get_llm_input_index(row: Dict[str, Any]) -> Optional[int]:
    meta = row.get("meta", {})
    if not isinstance(meta, dict):
        return None

    idx = meta.get("llm_input_index")
    return idx if isinstance(idx, int) else None


def get_llm_score(row: Dict[str, Any]) -> float:
    meta = row.get("meta", {})
    if isinstance(meta, dict):
        for key in ["llm_overall_score", "overall_score"]:
            try:
                return float(meta.get(key))
            except Exception:
                pass

    try:
        return float(row.get("value", row.get("weight", 0.5)))
    except Exception:
        return 0.5


def get_weight(row: Dict[str, Any]) -> float:
    try:
        return float(row.get("value", row.get("weight", 1.0)))
    except Exception:
        return 1.0


def sample_random(rows: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)

    rows = list(rows)
    if len(rows) <= n:
        rng.shuffle(rows)
        return rows

    return rng.sample(rows, n)


def sample_high_quality(
    rows: List[Dict[str, Any]],
    n: int,
    seed: int,
    pool_multiplier: int = 3,
) -> List[Dict[str, Any]]:
    """
    Sort rows by a quality score, then sample from a larger high-quality pool.
    This avoids taking only near-duplicate top rows.
    """
    rng = random.Random(seed)

    if len(rows) <= n:
        rng.shuffle(rows)
        return rows

    rows = sorted(
        rows,
        key=lambda r: (get_llm_score(r), get_weight(r)),
        reverse=True,
    )

    pool_size = min(len(rows), max(n, n * pool_multiplier))
    pool = rows[:pool_size]

    return rng.sample(pool, n)


def audit(rows: List[Dict[str, Any]]) -> None:
    counts = {}
    values = []

    for r in rows:
        src = r.get("source", "unknown")
        counts[src] = counts.get(src, 0) + 1

        try:
            values.append(float(r.get("value", r.get("weight", 1.0))))
        except Exception:
            pass

    print("final source counts:")
    print(json.dumps(counts, indent=2))
    print("total:", len(rows))

    if values:
        print("value min/avg/max:", min(values), sum(values) / len(values), max(values))


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--stage2_jsonl", type=Path, required=True)
    ap.add_argument("--llm_judged_jsonl", type=Path, required=True)
    ap.add_argument("--llm_repair_jsonl", type=Path, required=True)
    ap.add_argument("--out_jsonl", type=Path, required=True)

    ap.add_argument("--num_stage2", type=int, default=27000)
    ap.add_argument("--num_llm_judged", type=int, default=3000)
    ap.add_argument("--num_llm_repair", type=int, default=750)

    ap.add_argument("--min_llm_score", type=float, default=0.70)
    ap.add_argument("--exclude_stage2_rows_used_by_llm", action="store_true")
    ap.add_argument("--seed", type=int, default=28)

    args = ap.parse_args()

    safety_check_path(args.stage2_jsonl, "stage2_jsonl")
    safety_check_path(args.llm_judged_jsonl, "llm_judged_jsonl")
    safety_check_path(args.llm_repair_jsonl, "llm_repair_jsonl")
    safety_check_path(args.out_jsonl, "out_jsonl", must_exist=False)

    print("Loading Stage 2 rows:", args.stage2_jsonl)
    raw_stage2 = read_jsonl(args.stage2_jsonl)
    print("raw Stage 2 rows:", len(raw_stage2))

    print("Loading LLM-judged rows:", args.llm_judged_jsonl)
    raw_llm_judged = read_jsonl(args.llm_judged_jsonl)
    print("raw LLM-judged rows:", len(raw_llm_judged))

    print("Loading LLM-repair rows:", args.llm_repair_jsonl)
    raw_llm_repair = read_jsonl(args.llm_repair_jsonl)
    print("raw LLM-repair rows:", len(raw_llm_repair))

    llm_used_indices = set()
    for r in raw_llm_judged:
        idx = get_llm_input_index(r)
        if idx is not None:
            llm_used_indices.add(idx)

    stage2_norm = []
    for i, r in enumerate(raw_stage2):
        if args.exclude_stage2_rows_used_by_llm and i in llm_used_indices:
            continue

        nr = normalize_sft_row(r, "llm_faipd_rr_lite_stage2_original")
        if nr is not None:
            stage2_norm.append(nr)

    llm_judged_norm = []
    for r in raw_llm_judged:
        if get_llm_score(r) < args.min_llm_score:
            continue

        nr = normalize_sft_row(r, "llm_faipd_rr_lite_llm_judged_turn")
        if nr is not None:
            llm_judged_norm.append(nr)

    llm_repair_norm = []
    for r in raw_llm_repair:
        nr = normalize_sft_row(r, "llm_faipd_rr_lite_llm_repair")
        if nr is not None:
            llm_repair_norm.append(nr)

    print("usable Stage 2 rows:", len(stage2_norm))
    print("usable LLM-judged rows:", len(llm_judged_norm))
    print("usable LLM-repair rows:", len(llm_repair_norm))

    if len(stage2_norm) < 1000:
        raise SystemExit("Too few usable Stage 2 rows.")

    if len(llm_judged_norm) < min(100, args.num_llm_judged):
        print("[warn] few usable LLM-judged rows; consider lowering --min_llm_score")

    if len(llm_repair_norm) < min(100, args.num_llm_repair):
        print("[warn] few usable LLM-repair rows.")

    stage2_sample = sample_high_quality(
        stage2_norm,
        args.num_stage2,
        args.seed + 1,
        pool_multiplier=4,
    )

    llm_judged_sample = sample_high_quality(
        llm_judged_norm,
        args.num_llm_judged,
        args.seed + 2,
        pool_multiplier=3,
    )

    llm_repair_sample = sample_random(
        llm_repair_norm,
        args.num_llm_repair,
        args.seed + 3,
    )

    final_rows = stage2_sample + llm_judged_sample + llm_repair_sample

    rng = random.Random(args.seed)
    rng.shuffle(final_rows)

    audit(final_rows)

    write_jsonl(args.out_jsonl, final_rows)
    print("wrote:", args.out_jsonl)


if __name__ == "__main__":
    main()