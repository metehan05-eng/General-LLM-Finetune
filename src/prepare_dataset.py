"""
prepare_dataset.py

Builds a general-purpose, English-first instruction-following dataset:
1) Loads a well-established general instruction dataset from Hugging Face
   (Alpaca-style: broad topic coverage, not domain-specific)
2) Adds your own examples from data/custom_examples.jsonl on top
3) Formats everything into the "text" field SFTTrainer expects
"""

import json
import random
from pathlib import Path

try:
    from datasets import load_dataset, concatenate_datasets, Dataset
except ImportError:  # pragma: no cover
    Dataset = None
    load_dataset = None
    concatenate_datasets = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HF_DATASET_NAME = "tatsu-lab/alpaca"
CUSTOM_EXAMPLES_PATH = PROJECT_ROOT / "data" / "custom_examples.jsonl"

PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

MAX_OUTPUT_CHARS = 15000
REQUIRED_FIELDS = ("instruction", "input", "output")


def format_example(instruction: str, input_text: str, output: str, eos_token: str) -> str:
    return PROMPT_TEMPLATE.format(instruction, input_text, output) + eos_token


def validate_rows(rows):
    valid_rows = []
    invalid_rows = []
    seen_keys = set()

    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            invalid_rows.append({
                "index": idx,
                "code": "invalid_row_type",
                "message": "Row is not a JSON object",
            })
            continue

        raw_row = {key: row.get(key) for key in REQUIRED_FIELDS}
        issues = []

        for field_name in ("instruction", "output"):
            value = raw_row.get(field_name)
            if value is None or str(value).strip() == "":
                issues.append({
                    "index": idx,
                    "field": field_name,
                    "code": "missing_output" if field_name == "output" else "missing_instruction",
                    "message": f"Missing required field: {field_name}",
                })

        output_value = raw_row.get("output")
        if isinstance(output_value, str) and len(output_value) > MAX_OUTPUT_CHARS:
            issues.append({
                "index": idx,
                "field": "output",
                "code": "too_long",
                "message": f"Output exceeds {MAX_OUTPUT_CHARS} characters",
            })

        dedupe_key = (
            str(raw_row.get("instruction", "")).strip(),
            str(raw_row.get("input", "")).strip(),
            str(raw_row.get("output", "")).strip(),
        )
        if dedupe_key in seen_keys:
            issues.append({
                "index": idx,
                "field": "output",
                "code": "duplicate_row",
                "message": "Duplicate instruction/input/output sample detected",
            })
        else:
            seen_keys.add(dedupe_key)

        if issues:
            invalid_rows.extend(issues)
            continue

        valid_rows.append(row)

    return valid_rows, invalid_rows


def validate_jsonl_file(file_path: str | Path):
    path = Path(file_path)
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return validate_rows(rows)


def split_train_eval_rows(rows, eval_fraction: float = 0.1, seed: int = 42):
    if not rows:
        return [], []

    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    split_index = max(1, int(len(shuffled) * (1.0 - eval_fraction)))
    train_rows = shuffled[:split_index]
    eval_rows = shuffled[split_index:]
    return train_rows, eval_rows


def load_custom_examples() -> "Dataset":
    if not CUSTOM_EXAMPLES_PATH.exists():
        raise FileNotFoundError(f"Custom examples file not found: {CUSTOM_EXAMPLES_PATH}")

    rows = []
    with open(CUSTOM_EXAMPLES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    valid_rows, invalid_rows = validate_rows(rows)
    if invalid_rows:
        print(f"Warning: {len(invalid_rows)} invalid custom examples were dropped before training.")
    return Dataset.from_list(valid_rows)


def build_dataset(tokenizer, use_hf_dataset: bool = True, hf_sample_size=5000) -> "Dataset":
    """
    hf_sample_size: on a free Colab T4, training on the full 52k Alpaca set
    takes a while. Set this to e.g. 5000 for a faster first run, or None to
    use the full dataset once you're confident the pipeline works end to end.
    """
    custom_ds = load_custom_examples()

    if use_hf_dataset:
        try:
            if load_dataset is None:
                raise ImportError("datasets package not available")
            hf_ds = load_dataset(HF_DATASET_NAME, split="train")
            if hf_sample_size is not None and hf_sample_size < len(hf_ds):
                hf_ds = hf_ds.shuffle(seed=42).select(range(hf_sample_size))
            combined = concatenate_datasets([hf_ds, custom_ds])
        except Exception as e:
            print(f"Warning: could not load HF dataset ({e}). Falling back to custom examples only.")
            combined = custom_ds
    else:
        combined = custom_ds

    def _map_fn(examples):
        texts = [
            format_example(instr, inp or "", out, tokenizer.eos_token)
            for instr, inp, out in zip(
                examples["instruction"], examples["input"], examples["output"]
            )
        ]
        return {"text": texts}

    combined = combined.map(_map_fn, batched=True)
    return combined


if __name__ == "__main__":
    ds = load_custom_examples()
    print(f"Custom example count: {len(ds)}")
