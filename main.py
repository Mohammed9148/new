import streamlit as st
from langchain_openai import AzureChatOpenAI
from sentence_transformers import SentenceTransformer
import requests
import os
import pickle

# URL to the preprocessed data file
preprocessed_data_url = 'https://raw.githubusercontent.com/Mohammed9148/new/main/processed_data.pkl'

# Function to download preprocessed data
@st.cache
def download_preprocessed_data(url):
    local_filename = 'preprocessed_data.pkl'
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Ensure we got a valid response

    with open(local_filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return local_filename

# Function to load preprocessed data
@st.cache
def load_preprocessed_data(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

# Ensure the download and loading functions work correctly
try:
    data_file = download_preprocessed_data(preprocessed_data_url)
    data = load_preprocessed_data(data_file)
    if isinstance(data, list):
        st.write("Data loaded successfully.")
    else:
        st.error("Data is not in the expected format.")
        st.stop()

except Exception as e:
    st.error(f"Error loading preprocessed data: {e}")
    st.stop()

# Load Sentence Transformer model
@st.cache(allow_output_mutation=True)
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

model = load_model()

# Initialize Azure OpenAI model
llm = AzureChatOpenAI(
    model="gpt-35-turbo-16k",
    deployment_name="VISAGenAI",
    api_key="dbb69df9354846529d8994cb313275e0",
    azure_endpoint="https://visagenai.openai.azure.com/",
    api_version="2024-02-01",
)

# Function to perform similarity search and get the most relevant chunk from data
def get_relevant_chunk(question, data):
    question_embedding = model.encode([question])[0]
    max_similarity = -1
    relevant_chunk = None

    for chunk in data:
        chunk_embedding = chunk['embedding']
        similarity = model.cosine_similarity([question_embedding], [chunk_embedding])[0][0]
        
        if similarity > max_similarity:
            max_similarity = similarity
            relevant_chunk = chunk['text']

    return relevant_chunk if relevant_chunk else "No relevant chunk found."

# Function to handle question submission
def handle_question():
    if st.session_state.user_question:
        relevant_chunk = get_relevant_chunk(st.session_state.user_question, data)
        prompt = f"Answer the following question based on this text: {relevant_chunk}\n\nQuestion: {st.session_state.user_question}\nAnswer:"
        response = llm.invoke(prompt)
        st.session_state.response = response.content

# Streamlit app interface
st.title("PDF Chatbot with Azure OpenAI")

st.text_input("Type your question here:", key="user_question", on_change=handle_question)

if "response" in st.session_state:
    st.write("Response:", st.session_state.response)
