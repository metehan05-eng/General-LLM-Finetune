import os

SUPPORTED_MODELS = {
    "llama-3.1-8b": "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    "llama-3.2-3b": "unsloth/Llama-3.2-3B-bnb-4bit",
    "llama-3.1-70b": "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "mistral-7b": "unsloth/Mistral-7B-v0.3-bnb-4bit",
    "mistral-7b-instruct": "unsloth/Mistral-7B-Instruct-v0.3-bnb-4bit",
    "gemma-2-9b": "unsloth/gemma-2-9b-bnb-4bit",
    "qwen2.5-7b": "unsloth/Qwen2.5-7B-bnb-4bit",
    "qwen2.5-14b": "unsloth/Qwen2.5-14B-bnb-4bit",
}

DEFAULT_MODEL_KEY = "llama-3.1-8b"


def resolve_model_name(model_name: str | None = None) -> str:
    raw = model_name or os.environ.get("MODEL_NAME") or DEFAULT_MODEL_KEY
    candidate = raw.strip()

    if candidate.lower() in SUPPORTED_MODELS:
        return SUPPORTED_MODELS[candidate.lower()]

    if "/" in candidate:
        return candidate

    return SUPPORTED_MODELS.get(candidate.lower(), SUPPORTED_MODELS[DEFAULT_MODEL_KEY])
