"""
train.py

General-purpose instruction fine-tuning with Unsloth + QLoRA.
Designed to run on a free Google Colab T4 GPU (16 GB).
"""

import argparse
from pathlib import Path

import torch

if not torch.cuda.is_available():
    raise RuntimeError(
        "GPU required for training. In Google Colab, go to Runtime > Change runtime type > Hardware accelerator = T4 (or A100), "
        "then rerun this script. CPU-only environments are not supported by Unsloth."
    )

from unsloth import FastLanguageModel

try:
    from unsloth import is_bfloat16_supported
except ImportError:
    is_bfloat16_supported = FastLanguageModel.is_bfloat16_supported

from trl import SFTTrainer, SFTConfig

try:
    from prepare_dataset import build_dataset
except ImportError:
    from src.prepare_dataset import build_dataset

from model_config import generate_run_name, prompt_for_model_choice, resolve_model_name, resolve_model_spec

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL_SPEC = resolve_model_spec()
MODEL_NAME = DEFAULT_MODEL_SPEC.hf_id
MAX_SEQ_LENGTH = DEFAULT_MODEL_SPEC.default_max_seq_length
LOAD_IN_4BIT = True
OUTPUT_DIR = str(PROJECT_ROOT / "outputs")
LORA_R = 16
LORA_ALPHA = 16
NUM_TRAIN_EPOCHS = 1
LEARNING_RATE = 2e-4
PER_DEVICE_BATCH_SIZE = 2
GRAD_ACCUMULATION_STEPS = 4


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a general-purpose LLM with Unsloth + QLoRA.")
    parser.add_argument("--model", type=str, default=None, help="Model key or Hugging Face model id.")
    parser.add_argument("--family", type=str, default=None, help="Optional model family filter: llama, mistral, qwen, gemma.")
    parser.add_argument("--run-name", type=str, default=None, help="Friendly name for this fine-tuning run.")
    parser.add_argument("--system-prompt", type=str, default=None, help="System prompt that describes the model's role or behavior.")
    parser.add_argument("--dataset", type=str, action="append", default=None, help="JSONL dataset path to include in training. Can be passed multiple times.")
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH, help="Maximum sequence length for tokenization.")
    parser.add_argument("--epochs", type=int, default=NUM_TRAIN_EPOCHS, help="Number of train epochs.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE, help="Learning rate for the optimizer.")
    parser.add_argument("--lora-r", type=int, default=LORA_R, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=LORA_ALPHA, help="LoRA alpha scaling.")
    parser.add_argument("--batch-size", type=int, default=PER_DEVICE_BATCH_SIZE, help="Per-device batch size.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=GRAD_ACCUMULATION_STEPS, help="Gradient accumulation steps.")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Directory to save outputs and adapters.")
    parser.add_argument("--quantization", choices=["4bit", "8bit"], default="4bit", help="Quantization mode for model loading.")
    return parser.parse_args()


def main():
    args = parse_args()
    global MODEL_NAME, MAX_SEQ_LENGTH, LOAD_IN_4BIT, OUTPUT_DIR, LORA_R, LORA_ALPHA
    global NUM_TRAIN_EPOCHS, LEARNING_RATE, PER_DEVICE_BATCH_SIZE, GRAD_ACCUMULATION_STEPS

    if args.model is None:
        selected_model, auto_run_name, system_prompt = prompt_for_model_choice(
            include_system_prompt=True,
            enable_family_selection=True,
        )
        args.model = selected_model
        if args.run_name is None:
            args.run_name = auto_run_name
        if args.system_prompt is None:
            args.system_prompt = system_prompt

    model_spec = resolve_model_spec(args.model)
    MODEL_NAME = model_spec.hf_id
    MAX_SEQ_LENGTH = args.max_seq_length if args.max_seq_length else model_spec.default_max_seq_length
    LOAD_IN_4BIT = args.quantization == "4bit"

    run_name = generate_run_name(model_spec.key, args.run_name)
    if args.output_dir == OUTPUT_DIR:
        args.output_dir = str(PROJECT_ROOT / "outputs" / run_name)
    OUTPUT_DIR = args.output_dir

    LORA_R = args.lora_r
    LORA_ALPHA = args.lora_alpha
    NUM_TRAIN_EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    PER_DEVICE_BATCH_SIZE = args.batch_size
    GRAD_ACCUMULATION_STEPS = args.gradient_accumulation_steps

    print(f"Selected model: {MODEL_NAME} ({model_spec.family})")
    print(f"Training run name: {run_name}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Loading base model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    print("Building dataset...")
    dataset = build_dataset(
        tokenizer,
        use_hf_dataset=True,
        hf_sample_size=5000,
        dataset_paths=args.dataset,
        system_prompt=args.system_prompt,
    )
    print(f"Total training examples: {len(dataset)}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=SFTConfig(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
            num_train_epochs=NUM_TRAIN_EPOCHS,
            learning_rate=LEARNING_RATE,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            save_strategy="epoch",
            report_to="none",
        ),
    )

    print("Starting training...")
    trainer.train()

    print("Saving LoRA adapter...")
    adapter_path = Path(OUTPUT_DIR) / "lora_adapter"
    adapter_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))

    print(f"Done. Adapter saved to {adapter_path}")
    print("Next: run src/evaluate.py to compare base vs fine-tuned outputs.")


if __name__ == "__main__":
    main()
