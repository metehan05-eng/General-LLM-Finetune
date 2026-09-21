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

1. Open Google Colab.
2. Create a new notebook.
3. Run the following commands in a cell:

```bash
!git clone https://github.com/metehan05-eng/General-LLM-Finetune.git
%cd General-LLM-Finetune
!pip install -r requirements.txt
!python src/train.py
!python src/evaluate.py
```

4. Change runtime to GPU: Runtime > Change runtime type > T4 GPU.
5. Run cells top to bottom.

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

- Change the base language model in `src/train.py`
- Update the public dataset source in `src/prepare_dataset.py`
- Add your own examples inside `data/custom_examples.jsonl`
- Adjust training parameters like `NUM_TRAIN_EPOCHS`, `LEARNING_RATE`, and `hf_sample_size`

## Roadmap

- [ ] Add a cleaner evaluation report with before/after examples
- [ ] Add automatic model export to Hugging Face Hub
- [ ] Expand the custom dataset for a specific language or domain
- [ ] Create a more robust benchmark set for testing improvements

## License

This project is licensed under the [MIT License](LICENSE).
