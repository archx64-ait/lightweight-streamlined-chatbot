import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# load model and tokenizer
@st.cache_resource
def load_model():
    adapter_path = '../notebooks/models/gpt-neo-lora-checkpoint-final'
    config = PeftConfig.from_pretrained(adapter_path)
    base = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base, adapter_path).eval().to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    return model, tokenizer

model, tokenizer = load_model()

def chat(prompt, temperature):
    input_text = f'### Prompt:\n{prompt}\n### Response:\n'
    inputs = tokenizer(input_text, return_tensors='pt').to('cuda')
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=temperature,
            top_p=0.9,
            top_k=64,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split('### Response:')[-1].strip()

# === Streamlit UI ===
st.title('LW Streamlined Chatbot 🤖')
st.markdown('Ask me anything about Ubuntu!')

# initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# temperature slider
temperature = st.slider('Creativity (temperature)', 0.2, 1.5, 0.7, step=0.1)

# user input
user_input = st.text_input('You:', key='input')

# user submits input
if user_input:
    try:
        response = chat(user_input, temperature)
        st.session_state.chat_history.append(('user', user_input))
        st.session_state.chat_history.append(('bot', response))
        st.experimental_rerun()  # Rerun to display new messages
    except Exception as e:
        print(str(e))

# display chat history
for role, msg in st.session_state.chat_history:
    if role == 'user':
        st.markdown(f'🧑 **You:** {msg}')
    else:
        st.markdown(f'🤖 **Bot:** {msg}')
