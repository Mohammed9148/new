import streamlit as st
from sentence_transformers import SentenceTransformer
import requests
import os
import pickle
import openai

# Set your OpenAI API key
openai.api_key = 'your_openai_api_key_here'  # Replace with your OpenAI API key

# Function to download preprocessed data
@st.cache_data
def download_preprocessed_data(url):
    local_filename = 'preprocessed_data.pkl'
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Ensure we got a valid response

    with open(local_filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    # Check the downloaded file size
    file_size = os.path.getsize(local_filename)
    st.write(f"Downloaded file size: {file_size} bytes")

    # Optional: Print first few bytes to check content
    with open(local_filename, 'rb') as f:
        file_start = f.read(10)
        st.write(f"File starts with: {file_start}")

    return local_filename

# Function to load preprocessed data
@st.cache_data
def load_preprocessed_data(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

# Ensure the download and loading functions work correctly
try:
    preprocessed_data_url = 'https://raw.githubusercontent.com/Mohammed9148/new/main/preprocessed_data.pkl'
    data_file = download_preprocessed_data(preprocessed_data_url)
    data = load_preprocessed_data(data_file)

    # Check if data is in the expected format
    if 'text' in data and 'metadata' in data:
        chunks = data['text']
        st.write("Data loaded successfully.")
        st.write(f"Number of chunks: {len(chunks)}")
    else:
        st.error("Data is not in the expected format.")
        st.stop()

except Exception as e:
    st.error(f"Error loading preprocessed data: {e}")
    st.stop()

# Load Sentence Transformer model
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

model = load_model()

# Function to perform similarity search and get the most relevant chunk
def get_relevant_chunk(question):
    question_embedding = model.encode([question])[0]  # Flatten the list of list
    similarities = model.encode(chunks)
    distances = model.similarity(question_embedding, similarities)
    most_relevant_chunk = chunks[distances.argmax()]
    return most_relevant_chunk

# Function to handle question submission
def handle_question():
    if st.session_state.user_question:
        relevant_chunk = get_relevant_chunk(st.session_state.user_question)
        prompt = f"Answer the following question based on this text: {relevant_chunk}\n\nQuestion: {st.session_state.user_question}\nAnswer:"
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        
        st.session_state.response = response['choices'][0]['message']['content']

# Streamlit app interface
st.title("PDF Chatbot with OpenAI")

st.text_input("Type your question here:", key="user_question", on_change=handle_question)

if "response" in st.session_state:
    st.write("Response:", st.session_state.response)
