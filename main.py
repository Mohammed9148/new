import streamlit as st

# Function to generate chatbot responses
def generate_response(message):
    if "hello" in message.lower():
        return "Hi there! How can I help you?"
    elif "bye" in message.lower():
        return "Goodbye! Have a nice day!"
    elif "help" in message.lower():
        return "Sure, I am here to help you. What do you need assistance with?"
    else:
        return "I'm not sure how to respond to that. Can you please rephrase?"
# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'input' not in st.session_state:
    st.session_state.input = ""


# Streamlit application
st.title("Simple Chatbot")

def send_message():
    user_input = st.session_state.input
    if user_input:
        st.session_state.history.append(f"You: {user_input}")
        response = generate_response(user_input)
        st.session_state.history.append(f"Bot: {response}")
        st.session_state.input = "" # Clear the input after sending


# User input
st.text_input("You: ", key="input", on_change=send_message)

# Display conversation history
for message in st.session_state.history:
    st.write(message)
