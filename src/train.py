"""
train.py

General-purpose instruction fine-tuning with Unsloth + QLoRA.
Designed to run on a free Google Colab T4 GPU (16 GB).

Usage (from repo root, inside Colab or a local GPU environment):
    python src/train.py

Before running: check Unsloth's model list at
https://huggingface.co/unsloth for the current recommended 4-bit model id —
model names get updated as new base models (Llama, Gemma, Qwen, ...) are
released. MODEL_NAME below is a safe, long-standing default.
"""

from pathlib import Path

from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

try:
    from prepare_dataset import build_dataset
except ImportError:
    from src.prepare_dataset import build_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Config -----------------------------------------------------------
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"  # verify current model id on huggingface.co/unsloth
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True

OUTPUT_DIR = str(PROJECT_ROOT / "outputs")
LORA_R = 16
LORA_ALPHA = 16
NUM_TRAIN_EPOCHS = 1  # start small; raise to 2-3 only if eval shows underfitting
LEARNING_RATE = 2e-4
PER_DEVICE_BATCH_SIZE = 2
GRAD_ACCUMULATION_STEPS = 4
# ------------------------------------------------------------------------


def main():
    print(f"Loading base model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # auto-detect (bf16/fp16 depending on GPU)
        load_in_4bit=LOAD_IN_4BIT,
    )

    # Apply LoRA to all transformer layers (not just attention) —
    # this consistently outperforms attention-only LoRA configs.
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
    dataset = build_dataset(tokenizer, use_hf_dataset=True, hf_sample_size=5000)
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
            fp16=not FastLanguageModel.is_bfloat16_supported(),
            bf16=FastLanguageModel.is_bfloat16_supported(),
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
