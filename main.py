import streamlit as st
import PyPDF2
from langchain_openai import AzureChatOpenAI
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import requests
import os

# URL to the preprocessed data file on GitHub
preprocessed_data_url = 'https://github.com/Mohammed9148/new/blob/main/preprocessed_data.pkl'

# Download preprocessed data
@st.cache_data
def download_preprocessed_data(url):
    local_filename = 'preprocessed_data.pkl'
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

@st.cache_data
def load_preprocessed_data(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

# Load preprocessed data
data_file = download_preprocessed_data(preprocessed_data_url)
chunks, embeddings = load_preprocessed_data(data_file)

# Load Sentence Transformer model
@st.cache_resource
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

# Function to perform similarity search and get the most relevant chunk
def get_relevant_chunk(question, chunks, embeddings):
    question_embedding = model.encode([question])
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    _, I = index.search(question_embedding, 1)
    return chunks[I[0][0]]

# Function to handle question submission
def handle_question():
    if st.session_state.user_question:
        relevant_chunk = get_relevant_chunk(st.session_state.user_question, chunks, embeddings)
        prompt = f"Answer the following question based on this text: {relevant_chunk}\n\nQuestion: {st.session_state.user_question}\nAnswer:"
        response = llm.invoke(prompt)
        st.session_state.response = response.content

# Streamlit app interface
st.title("PDF Chatbot with Azure OpenAI")

st.text_input("Type your question here:", key="user_question", on_change=handle_question)

if "response" in st.session_state:
    st.write("Response:", st.session_state.response)
