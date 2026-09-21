"""
prepare_dataset.py

Builds a general-purpose, English-first instruction-following dataset:
1) Loads a well-established general instruction dataset from Hugging Face
   (Alpaca-style: broad topic coverage, not domain-specific)
2) Adds your own examples from data/custom_examples.jsonl on top
3) Formats everything into the "text" field SFTTrainer expects

NOTE: "tatsu-lab/alpaca" is a long-standing, widely used general instruction
dataset (52k examples, broad coverage: writing, reasoning, coding, general
knowledge). If you want a different mix, swap HF_DATASET_NAME for another
general-purpose dataset (e.g. "databricks/databricks-dolly-15k"), but always
check the dataset's column names first — they aren't all identical.
"""

import json
from pathlib import Path

from datasets import load_dataset, concatenate_datasets, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HF_DATASET_NAME = "tatsu-lab/alpaca"
CUSTOM_EXAMPLES_PATH = PROJECT_ROOT / "data" / "custom_examples.jsonl"

# Standard Alpaca-style prompt template, used widely in Unsloth/TRL examples.
PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


def format_example(instruction: str, input_text: str, output: str, eos_token: str) -> str:
    return PROMPT_TEMPLATE.format(instruction, input_text, output) + eos_token


def load_custom_examples() -> Dataset:
    if not CUSTOM_EXAMPLES_PATH.exists():
        raise FileNotFoundError(f"Custom examples file not found: {CUSTOM_EXAMPLES_PATH}")

    rows = []
    with open(CUSTOM_EXAMPLES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return Dataset.from_list(rows)


def build_dataset(tokenizer, use_hf_dataset: bool = True, hf_sample_size=5000) -> Dataset:
    """
    hf_sample_size: on a free Colab T4, training on the full 52k Alpaca set
    takes a while. Set this to e.g. 5000 for a faster first run, or None to
    use the full dataset once you're confident the pipeline works end to end.
    """
    custom_ds = load_custom_examples()

    if use_hf_dataset:
        try:
            hf_ds = load_dataset(HF_DATASET_NAME, split="train")
            if hf_sample_size is not None and hf_sample_size < len(hf_ds):
                hf_ds = hf_ds.shuffle(seed=42).select(range(hf_sample_size))
            # Alpaca's native columns are already instruction/input/output,
            # matching custom_examples.jsonl, so no remapping needed here.
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
    # Quick local sanity check (no tokenizer needed, just counts custom rows)
    ds = load_custom_examples()
    print(f"Custom example count: {len(ds)}")
