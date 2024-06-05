import streamlit as st
import PyPDF2
from langchain_openai import AzureChatOpenAI
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle


llm = AzureChatOpenAI(
    model="gpt-35-turbo-16k",
    deployment_name="VISAGenAI",
    api_key="dbb69df9354846529d8994cb313275e0",
    azure_endpoint="https://visagenai.openai.azure.com/",
    api_version="2024-02-01",
)


# Load preprocessed data
with open('preprocessed_data.pkl', 'rb') as f:
    chunks, embeddings = pickle.load(f)

# Function to perform similarity search and get the most relevant chunk
def get_relevant_chunk(question, chunks, embeddings):
    model = SentenceTransformer('all-MiniLM-L6-v2')
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

