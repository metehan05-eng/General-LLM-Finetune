import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class ModelSpec:
    key: str
    family: str
    hf_id: str
    default_max_seq_length: int = 2048
    quantization: str = "4bit"
    supported_quantizations: tuple[str, ...] = ("4bit", "8bit")


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
MODEL_SELECTION_ORDER = [
    "llama-3.1-8b",
    "mistral-7b",
    "qwen2.5-7b",
    "gemma-2-9b",
    "llama-3.2-3b",
    "mistral-7b-instruct",
    "qwen2.5-14b",
    "llama-3.1-70b",
]


def _guess_family(model_key: str) -> str:
    key = model_key.lower()
    if "llama" in key:
        return "llama"
    if "mistral" in key:
        return "mistral"
    if "gemma" in key:
        return "gemma"
    if "qwen" in key:
        return "qwen"
    return "custom"


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    key: ModelSpec(
        key=key,
        family=_guess_family(key),
        hf_id=value,
        default_max_seq_length=2048,
        quantization="4bit",
        supported_quantizations=("4bit", "8bit"),
    )
    for key, value in SUPPORTED_MODELS.items()
}


def resolve_model_spec(model_name: str | None = None) -> ModelSpec:
    raw = model_name or os.environ.get("MODEL_NAME") or DEFAULT_MODEL_KEY
    candidate = raw.strip()
    lowered = candidate.lower()

    if lowered in MODEL_REGISTRY:
        return MODEL_REGISTRY[lowered]

    if "/" in candidate:
        family = _guess_family(candidate)
        return ModelSpec(
            key="custom-model",
            family=family,
            hf_id=candidate,
            default_max_seq_length=2048,
            quantization="4bit",
            supported_quantizations=("4bit", "8bit"),
        )

    return MODEL_REGISTRY.get(lowered, MODEL_REGISTRY[DEFAULT_MODEL_KEY])


def resolve_model_name(model_name: str | None = None) -> str:
    return resolve_model_spec(model_name).hf_id


def sanitize_run_name(value: str | None) -> str:
    if value is None:
        return ""
    cleaned = re.sub(r"[^a-zA-Z0-9.]+", "-", value.strip().lower())
    cleaned = cleaned.strip("-")
    return cleaned


def generate_run_name(model_key: str, custom_name: str | None = None) -> str:
    model_slug = sanitize_run_name(model_key)
    custom_slug = sanitize_run_name(custom_name)
    if not custom_slug:
        return model_slug
    return f"{model_slug}-{custom_slug}"


def get_model_family_options() -> List[str]:
    families = []
    seen = set()
    for model_key in MODEL_SELECTION_ORDER:
        family = MODEL_REGISTRY[model_key].family if model_key in MODEL_REGISTRY else _guess_family(model_key)
        if family not in seen:
            families.append(family)
            seen.add(family)
    for key in list_supported_models():
        family = MODEL_REGISTRY[key].family
        if family not in seen:
            families.append(family)
            seen.add(family)
    return families


def get_models_for_family(family_name: str) -> List[str]:
    family = (family_name or "").strip().lower()
    matches = []
    for key in MODEL_SELECTION_ORDER:
        if key in MODEL_REGISTRY and MODEL_REGISTRY[key].family.lower() == family:
            matches.append(key)
    for key in list_supported_models():
        if key not in matches and MODEL_REGISTRY[key].family.lower() == family:
            matches.append(key)
    return matches


def get_model_selection_options() -> List[str]:
    ordered = []
    seen = set()
    for key in MODEL_SELECTION_ORDER:
        if key in SUPPORTED_MODELS and key not in seen:
            ordered.append(key)
            seen.add(key)
    for key in list_supported_models():
        if key not in seen:
            ordered.append(key)
            seen.add(key)
    return ordered


def prompt_for_model_choice(
    input_fn=input,
    print_fn=print,
    include_system_prompt: bool = False,
    enable_family_selection: bool = False,
):
    if enable_family_selection:
        families = get_model_family_options()
        print_fn("Choose a model family:")
        for index, family in enumerate(families, start=1):
            print_fn(f"  {index}. {family}")

        while True:
            raw_family = input_fn("Enter family number or family name: ").strip()
            family_choice = None
            if raw_family.isdigit():
                choice = int(raw_family)
                if 1 <= choice <= len(families):
                    family_choice = families[choice - 1]
            else:
                candidate = raw_family.lower().strip()
                if candidate in {family.lower() for family in families}:
                    family_choice = candidate

            if family_choice is not None:
                break
            print_fn("Invalid family selection.")

        options = get_models_for_family(family_choice)
        print_fn(f"Available models for family '{family_choice}':")
    else:
        options = get_model_selection_options()
        print_fn("Select a base model to fine-tune:")
        family_choice = None

    for index, key in enumerate(options, start=1):
        print_fn(f"  {index}. {key}")

    while True:
        raw_model = input_fn("Enter model number or model key: ").strip()
        selected = None
        if raw_model.isdigit():
            choice = int(raw_model)
            if 1 <= choice <= len(options):
                selected = options[choice - 1]
        else:
            candidate = raw_model.lower().strip()
            if candidate in SUPPORTED_MODELS:
                selected = candidate

        if selected is not None:
            break
        print_fn("Invalid model choice. Please select one of the listed models.")

    custom_name = input_fn("Name this training run (optional): ").strip()
    run_name = generate_run_name(selected, custom_name)

    if not include_system_prompt:
        return selected, run_name

    system_prompt = input_fn(
        "Describe the system prompt for this model (optional, for example: 'You are a helpful coding assistant.'): "
    ).strip()
    return selected, run_name, system_prompt


def list_supported_models() -> List[str]:
    return sorted(SUPPORTED_MODELS.keys())


def list_model_families() -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for key in list_supported_models():
        family = MODEL_REGISTRY[key].family
        grouped.setdefault(family, []).append(key)
    return grouped
