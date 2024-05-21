import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    return tokenizer, model


tokenizer, model = load_model()

# Function to generate chatbot responses using GPT-2
def generate_response(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Streamlit applicationst.title("AI Chatbot")
if 'history' not in st.session_state:
    st.session_state.history = []



# User input
user_input = st.text_input("You: ", key="input")

if st.button("Send"):
    if user_input:
        st.session_state.history.append(f"You: {user_input}")
        response = generate_response(user_input)
        st.session_state.history.append(f"Bot: {response}")
        st.session_state.input = ""


# Display conversation history
for message in st.session_state.history:
    st.write(message)

