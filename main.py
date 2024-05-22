import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load pre-trained model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    return tokenizer, model

tokenizer, model = load_model()


# Function to generate chatbot responses using distilGPT-2
def generate_response(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Callback function to handle user input and generate response
def send_message():
    user_input = st.session_state.input
    if user_input:
        st.session_state.history.append(f"You: {user_input}")
        response = generate_response(user_input)
        st.session_state.history.append(f"Bot: {response}")
        st.session_state.input = ""  # Clear the input after sending

# Streamlit application
st.title("AI Chatbot")


# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

# User input
st.text_input("You: ", key="input", on_change=send_message)

# Display conversation history
for message in st.session_state.history:
    st.write(message)

