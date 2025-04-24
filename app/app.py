import streamlit as st
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model options
MODELS = {
    "GPT-Neo": "EleutherAI/gpt-neo-1.3B",
    "Qwen2": "Qwen/Qwen2-0.5B-Instruct"
}

TRAINED_TOKENS = 128
CONFIDENCE = 0.75


@st.cache_resource
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16
    )
    tokenizer.pad_token = tokenizer.eos_token
    lora_path = f"../notebooks/models/{model_name}-{TRAINED_TOKENS}"
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval().to(device=device)
    return tokenizer, model


def generate_response(prompt, tokenizer, model, max_new_tokens=128):
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device=device)
    input_len = inputs.input_ids.shape[1]
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id,
            output_scores=True
        )
    response_ids = outputs.sequences[0][input_len:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    # probs = F.softmax(outputs.scores[0], dim=-1)
    max_probs = [F.softmax(score, dim=-1).max().item() for score in outputs.scores]
    avg_confidence = sum(max_probs) / len(max_probs)
    return response.strip(), avg_confidence


# UI
st.set_page_config(page_title="Ubuntu Support Chatbot", layout="centered")
st.title("🤖 LW Streamlined Chatbot")

model_choice = st.selectbox("Choose a model:", list(MODELS.keys()))
tokenizer, model = load_model_and_tokenizer(MODELS[model_choice])

st.markdown("""
            If confidence is low, the query is escalated to a human. 
            Low-confidence logs are saved for retraining and model improvement.
            """)

if "messages" not in st.session_state:
    st.session_state.messages = []

import csv
import os

log_path = "logs/low_confidence_log.csv"

user_input = st.chat_input("Type your question here...")

if user_input:

    st.session_state.messages.append(("user", user_input))
    prompt = f"User: {user_input}\nAssistant:"
    response, confidence = generate_response(prompt, tokenizer, model)

    if confidence < CONFIDENCE:
        fallback = "⚠️ I'm not confident in this answer. Forwarding to a human support agent."
        st.session_state.messages.append(("bot", fallback))

        # Log low-confidence prompt and response for retraining
        log_entry = {
            "prompt": user_input.replace('\n', ' '),
            "response": response.replace('\n', ' '),
            "confidence": confidence
        }

        file_exists = os.path.isfile(log_path)

        with open(log_path, "a", newline="", encoding='utf-8') as csvfile:
            fieldnames = ["prompt", "response", "confidence"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(log_entry)
    else:
        st.session_state.messages.append(("bot", response))

for role, msg in st.session_state.messages:
    align = "right" if role == "user" else "left"
    color = "#2e7d32" if role == "user" else "#424242"
    bubble_style = f"""
        <div style='background-color: {color}; padding: 10px; border-radius: 10px; margin: 5px; max-width: 75%; float: {align}; clear: both;'>
        {msg}
        </div>
    """
    st.markdown(bubble_style, unsafe_allow_html=True)
