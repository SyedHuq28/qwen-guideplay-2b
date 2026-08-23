
import argparse
import json
from pathlib import Path
from typing import Any, Dict


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


def valid_stage2_row(row: Dict[str, Any]) -> bool:
    """
    Basic check for weighted-turn SFT rows.
    Expected by train_weighted_turn_sft.py:
      prompt: list of messages
      completion: list containing assistant message
      value/weight: numeric, overwritten here
    """
    if not isinstance(row, dict):
        return False

    prompt = row.get("prompt")
    completion = row.get("completion")

    if not isinstance(prompt, list) or len(prompt) < 1:
        return False

    if not isinstance(completion, list) or len(completion) < 1:
        return False

    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", type=Path, required=True)
    ap.add_argument("--out_jsonl", type=Path, required=True)
    ap.add_argument("--uniform_value", type=float, default=1.0)
    args = ap.parse_args()

    safety_check_path(args.in_jsonl, "in_jsonl", must_exist=True)
    safety_check_path(args.out_jsonl, "out_jsonl", must_exist=False)

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0
    skipped = 0

    original_values = []

    with args.in_jsonl.open("r", encoding="utf-8") as f_in, args.out_jsonl.open(
        "w", encoding="utf-8"
    ) as f_out:
        for line_no, line in enumerate(f_in, start=1):
            line = line.strip()
            if not line:
                continue

            total += 1

            try:
                row = json.loads(line)
            except Exception as e:
                skipped += 1
                print(f"[warn] skipped bad JSON on line {line_no}: {e}")
                continue

            if not valid_stage2_row(row):
                skipped += 1
                print(f"[warn] skipped invalid Stage 2 row on line {line_no}")
                continue

            old_value = row.get("value", row.get("weight", None))
            old_weight = row.get("weight", row.get("value", None))

            try:
                if old_value is not None:
                    original_values.append(float(old_value))
            except Exception:
                pass

            meta = row.get("meta", {})
            if not isinstance(meta, dict):
                meta = {}

            meta["uniform_weight_ablation"] = True
            meta["original_value"] = old_value
            meta["original_weight"] = old_weight

            row["meta"] = meta
            row["value"] = float(args.uniform_value)
            row["weight"] = float(args.uniform_value)

            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

    print("Uniform Stage 2 dataset created.")
    print(f"input:  {args.in_jsonl}")
    print(f"output: {args.out_jsonl}")
    print(f"total rows read: {total}")
    print(f"rows kept:       {kept}")
    print(f"rows skipped:    {skipped}")
    print(f"uniform value:   {args.uniform_value}")

    if original_values:
        avg = sum(original_values) / len(original_values)
        print("original value audit:")
        print(f"  min: {min(original_values)}")
        print(f"  avg: {avg}")
        print(f"  max: {max(original_values)}")


if __name__ == "__main__":
    main()