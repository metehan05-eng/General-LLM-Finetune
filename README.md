# General-LLM-Finetune

A lightweight project for fine-tuning a general-purpose instruction-following LLM using [Unsloth](https://github.com/unslothai/unsloth) and QLoRA.

This repository is designed to work well in a free Google Colab environment with a T4 GPU and is structured so you can train, evaluate, and iterate quickly on a custom instruction dataset.

## What this project does

- Loads a pretrained base model from Hugging Face
- Applies LoRA/QLoRA fine-tuning instead of retraining the full model
- Combines a public general instruction dataset with your own examples from `data/custom_examples.jsonl`
- Formats the data into ChatML/Alpaca-style prompt-response training examples
- Compares the base model vs. the fine-tuned model using the same prompts

## Project structure

```text
general-llm-finetune/
├── README.md
├── LICENSE
├── requirements.txt
├── data/
│   └── custom_examples.jsonl
├── notebooks/
│   └── finetune_colab.ipynb
├── src/
│   ├── prepare_dataset.py
│   ├── train.py
│   └── evaluate.py
└── outputs/                  # generated during training
```

## Dataset format

Each line in `data/custom_examples.jsonl` should follow this JSON structure:

```json
{"instruction": "Explain what overfitting means in machine learning.", "input": "", "output": "Overfitting happens when ..."}
```

You can add as many rows as you want. The project will mix these examples with the public instruction dataset before training.

## Google Colab quickstart

For this project to work, you need a GPU runtime. Unsloth will not run on CPU-only environments.

### 1) Open a GPU-enabled Colab session

1. Open Google Colab.
2. Go to Runtime > Change runtime type.
3. Set Hardware accelerator to T4 or A100.
4. Make sure the session shows a GPU is available.

Test it with:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
```

If this prints `False`, switch to a GPU runtime and rerun the cell.

### 2) Clean Colab install from scratch

Do not rely on a relative folder name like `%cd General-LLM-Finetune` unless you are certain the current directory is correct. In Colab, the most reliable pattern is to clone directly into an exact path and then enter that path explicitly.

```bash
!git clone https://github.com/metehan05-eng/General-LLM-Finetune.git /content/General-LLM-Finetune
%cd /content/General-LLM-Finetune
!pwd
!ls

# Check GPU first (must print a "+cu..." torch build, not "+cpu")
import torch
print(torch.__version__, torch.cuda.is_available())

# Install dependencies (PyPI wheels; no git clone, no setuptools upgrade)
!pip install --no-cache-dir -r requirements.txt
```

If you still see a version conflict like `gcsfs 2025.12.0 requires fsspec==2025.12.0`, run this next and then restart the runtime:

```bash
!pip install --no-cache-dir --force-reinstall "fsspec==2025.12.0" "gcsfs==2025.12.0"
```

After installing packages, go to Runtime > Restart runtime and rerun the notebook cells.

### 3) Run training

```bash
!python src/train.py --model llama-3.1-8b
```

### 3A) Single-cell Colab runner (recommended)

This is the cleanest method for Google Colab. Everything is done in one cell with a fixed repo path, so there is no dependence on the current directory name.

```python
# Final one-cell Colab runner
REPO_PATH = "/content/General-LLM-Finetune"
MODEL_KEY = "llama-3.1-8b"
RUN_NAME = "colab-demo-run"
SYSTEM_PROMPT = "You are a helpful AI assistant for coding, writing, and practical problem solving."

# Export environment variables for shell commands
%env REPO_PATH={REPO_PATH}
%env MODEL_KEY={MODEL_KEY}
%env RUN_NAME={RUN_NAME}
%env SYSTEM_PROMPT={SYSTEM_PROMPT}

# 0) Verify GPU runtime FIRST (torch is preinstalled in Colab)
#    torch.__version__ must end in "+cu..." (e.g. 2.7.0+cu126).
#    If it ends in "+cpu" you are on a CPU runtime -> switch to GPU below.
import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("No GPU! Runtime > Change runtime type > T4 or A100, then Runtime > Restart runtime.")
print("GPU:", torch.cuda.get_device_name(0))

# 1) Clone repo safely without deleting the current working directory
!mkdir -p /content
!cd /content && if [ ! -d "$REPO_PATH" ]; then git clone https://github.com/metehan05-eng/General-LLM-Finetune.git "$REPO_PATH"; fi
%cd "$REPO_PATH"
!pwd
!ls

# 2) Install dependencies (PyPI wheels only - no slow/fragile git clone).
#    Do NOT upgrade setuptools to >= 82 or you will break the preinstalled torch.
!pip install -q --no-cache-dir -r requirements.txt

# 3) Train the model in one go
!python src/train.py --model "$MODEL_KEY" --run-name "$RUN_NAME" --system-prompt "$SYSTEM_PROMPT"

# 4) Evaluate the trained model
!python src/evaluate.py --model "$MODEL_KEY" --system-prompt "$SYSTEM_PROMPT"
```

> Important: if you see `CUDA available: False`, stop here and switch Colab to a T4/A100 GPU runtime. Training will not work on CPU-only mode.

If the install step still warns about stale imports, click Runtime > Restart runtime and run the cell again once.

### 4) Run evaluation

```bash
!python src/evaluate.py --model llama-3.1-8b
```

### 5) Optional: install compatible versions for Colab package conflicts

If you see a warning like `gcsfs requires fsspec==2025.12.0`, you can fix it with:

```bash
!pip install -q "fsspec==2025.12.0" "gcsfs==2025.12.0"
```

If you do not use `gcsfs`, removing it is also fine.

### 6) Model selection examples

```bash
!python src/train.py --model mistral-7b
!python src/train.py --model qwen2.5-7b
!python src/train.py --model gemma-2-9b
```

> Important: This repository is designed for GPU-enabled training. CPU-only Colab sessions will fail with the Unsloth accelerator error.

## Local usage

```bash
git clone https://github.com/metehan05-eng/General-LLM-Finetune.git
cd General-LLM-Finetune
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python src/train.py
python src/evaluate.py
```

> A CUDA-capable NVIDIA GPU is strongly recommended. For a quick starter setup, Google Colab with a T4 GPU is the easiest path.

## Training flow

1. Model is loaded from the Hugging Face Hub.
2. LoRA adapters are attached to the transformer layers.
3. Instruction examples are converted into text training samples.
4. The trainer fine-tunes the adapter while keeping the base model frozen.
5. The adapter is saved to `outputs/lora_adapter`.
6. Evaluation compares the base model and the fine-tuned adapter on the same prompts.

## Customization

- Change the base model by passing `--model` when running training and evaluation
- Supported keys include: `llama-3.1-8b`, `mistral-7b`, `qwen2.5-7b`, `gemma-2-9b`, and more
- Update the public dataset source in `src/prepare_dataset.py`
- Add your own examples inside `data/custom_examples.jsonl`
- Adjust training parameters like `NUM_TRAIN_EPOCHS`, `LEARNING_RATE`, and `hf_sample_size`

> Bigger models need more VRAM. A 7B/8B model can usually fit on a T4 or single 16GB GPU with QLoRA, while 70B-class models require much more memory and are best run on larger GPUs or in hosted environments.

## Roadmap

- [ ] Add a cleaner evaluation report with before/after examples
- [ ] Add automatic model export to Hugging Face Hub
- [ ] Expand the custom dataset for a specific language or domain
- [ ] Create a more robust benchmark set for testing improvements

## License

This project is licensed under the [MIT License](LICENSE).
