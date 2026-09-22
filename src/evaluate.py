"""
evaluate.py

Loads the base model and the fine-tuned (LoRA) model side by side, and runs
the same set of test prompts against both so you can see, in plain text,
whether fine-tuning actually changed anything for the better.

This step is not optional: fine-tuning without evaluation is just expensive
prompt engineering with extra steps. Paste the printed output into your
README as your before/after evidence.
"""

import argparse
from pathlib import Path

import torch

if not torch.cuda.is_available():
    raise RuntimeError(
        "GPU required for evaluation. In Google Colab, use a GPU runtime (T4/A100) before running this script. "
        "CPU-only execution is not supported by Unsloth."
    )

from unsloth import FastLanguageModel

try:
    from train import MAX_SEQ_LENGTH, LOAD_IN_4BIT
    from prepare_dataset import PROMPT_TEMPLATE
except ImportError:
    from src.train import MAX_SEQ_LENGTH, LOAD_IN_4BIT
    from src.prepare_dataset import PROMPT_TEMPLATE

from model_config import resolve_model_name
from prepare_dataset import build_prompt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = str(PROJECT_ROOT / "outputs")

TEST_PROMPTS = [
    ("Explain what overfitting means in machine learning.", ""),
    ("Suggest a short daily plan.", "I'm preparing for a busy work day."),
    ("Who are you and how should you behave as an assistant?", ""),
    ("Write a two-sentence summary of the water cycle.", ""),
    ("Make this email subject line more professional.", "subject: about the meeting"),
]


def render_block(title: str, message: str):
    print("\n" + "=" * 80)
    print(f"{title}")
    print("-" * 80)
    print(message.strip())


def generate(model, tokenizer, instruction: str, input_text: str, max_new_tokens: int = 200, system_prompt: str | None = None) -> str:
    prompt = build_prompt(instruction, input_text, "", system_prompt=system_prompt)
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        temperature=0.7,
        do_sample=True,
    )
    decoded = tokenizer.batch_decode(outputs)[0]
    if "Response:" in decoded:
        response = decoded.split("Response:")[-1].strip()
    else:
        response = decoded.strip()
    return response


def parse_args():
    parser = argparse.ArgumentParser(description="Compare a base model vs. a fine-tuned LoRA adapter.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model key or Hugging Face model id to evaluate. Examples: llama-3.1-8b, mistral-7b, qwen2.5-7b.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Optional system prompt to show before the evaluation instructions.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Maximum number of tokens to generate per prompt.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = resolve_model_name(args.model)
    max_new_tokens = args.max_new_tokens

    print(f"Loading base model: {model_name}")
    base_model, base_tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
    )
    FastLanguageModel.for_inference(base_model)

    adapter_path = Path(OUTPUT_DIR) / "lora_adapter"
    if not adapter_path.exists():
        raise FileNotFoundError(
            f"Fine-tuned adapter not found at {adapter_path}. "
            "Train the model first with `python src/train.py` or `cd src && python train.py`."
        )

    print(f"Loading fine-tuned model (LoRA adapter) from {adapter_path}...")
    ft_model, ft_tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
    )
    FastLanguageModel.for_inference(ft_model)

    for instruction, input_text in TEST_PROMPTS:
        print("\n" + "=" * 80)
        print(f"PROMPT: {instruction}")
        if input_text:
            print(f"INPUT: {input_text}")
        print("-" * 80)

        base_output = generate(
            base_model,
            base_tokenizer,
            instruction,
            input_text,
            max_new_tokens=max_new_tokens,
            system_prompt=args.system_prompt,
        )
        render_block("BASE MODEL", base_output)

        ft_output = generate(
            ft_model,
            ft_tokenizer,
            instruction,
            input_text,
            max_new_tokens=max_new_tokens,
            system_prompt=args.system_prompt,
        )
        render_block("FINE-TUNED MODEL", ft_output)

    print("\n" + "=" * 80)
    print("Done. Copy the outputs above into your README as before/after evidence.")


if __name__ == "__main__":
    main()
