"""Simple Streamlit demo for model selection and prompt testing.

Run:
    streamlit run src/demo_app.py
"""

from __future__ import annotations

import streamlit as st

from model_config import get_model_family_options, get_models_for_family, resolve_model_name


st.set_page_config(page_title="General LLM Finetune Demo", layout="wide")

st.title("General LLM Finetune Demo")
st.caption("Choose a model family, select a base model, set a system prompt, and test a prompt quickly.")

family = st.sidebar.selectbox("Model family", get_model_family_options())
models = get_models_for_family(family)
selected_model = st.sidebar.selectbox("Model", models)

system_prompt = st.text_area(
    "System prompt",
    value="You are a helpful AI assistant for writing, coding, and summarizing tasks.",
    height=120,
)

user_prompt = st.text_area("User prompt", value="Explain the difference between a transformer and an LSTM in simple terms.")

if st.button("Generate demo output"):
    st.info(f"Selected model: {selected_model}")
    st.info(f"Resolved Hugging Face model: {resolve_model_name(selected_model)}")
    st.info(f"System prompt: {system_prompt}")
    st.markdown("### Prompt preview")
    st.code(f"System: {system_prompt}\n\nUser: {user_prompt}")
    st.success("This demo is ready for a GPU-enabled generation step. Connect it to your training/eval pipeline when you want to run real model inference.")

st.markdown("---")
st.markdown("### Recommended workflow")
st.markdown(
    "1. Pick the model family and model.\n"
    "2. Write the system prompt the model should follow.\n"
    "3. Create a custom dataset with `instruction`, `input`, and `output`.\n"
    "4. Train with `python src/train.py --model <key> --system-prompt "
    "\"<your system prompt>\"`\n"
    "5. Evaluate the result with `python src/evaluate.py --system-prompt "
    "\"<your system prompt>\"`"
)
