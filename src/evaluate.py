"""
evaluate.py

Loads the base model and the fine-tuned (LoRA) model side by side, and runs
the same set of test prompts against both so you can see, in plain text,
whether fine-tuning actually changed anything for the better.

This step is not optional: fine-tuning without evaluation is just expensive
prompt engineering with extra steps. Paste the printed output into your
README as your before/after evidence.
"""

from pathlib import Path

from unsloth import FastLanguageModel

try:
    from train import MODEL_NAME, MAX_SEQ_LENGTH, LOAD_IN_4BIT
    from prepare_dataset import PROMPT_TEMPLATE
except ImportError:
    from src.train import MODEL_NAME, MAX_SEQ_LENGTH, LOAD_IN_4BIT
    from src.prepare_dataset import PROMPT_TEMPLATE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = str(PROJECT_ROOT / "outputs")

TEST_PROMPTS = [
    ("Explain what overfitting means in machine learning.", ""),
    ("Suggest a short daily plan.", "I'm preparing for a busy work day."),
    ("Who are you and how should you behave as an assistant?", ""),
    ("Write a two-sentence summary of the water cycle.", ""),
    ("Make this email subject line more professional.", "subject: about the meeting"),
]


def generate(model, tokenizer, instruction: str, input_text: str, max_new_tokens: int = 200) -> str:
    prompt = PROMPT_TEMPLATE.format(instruction, input_text, "")
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        temperature=0.7,
        do_sample=True,
    )
    decoded = tokenizer.batch_decode(outputs)[0]
    # Strip the prompt echo, keep only the generated response section
    response = decoded.split("### Response:")[-1].strip()
    return response


def main():
    print("Loading base model...")
    base_model, base_tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
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

        base_output = generate(base_model, base_tokenizer, instruction, input_text)
        print(f"BASE MODEL:\n{base_output}\n")

        ft_output = generate(ft_model, ft_tokenizer, instruction, input_text)
        print(f"FINE-TUNED MODEL:\n{ft_output}")

    print("\n" + "=" * 80)
    print("Done. Copy the outputs above into your README as before/after evidence.")


if __name__ == "__main__":
    main()
