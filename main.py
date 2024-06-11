import streamlit as st
from langchain_openai import AzureChatOpenAI
from sentence_transformers import SentenceTransformer
import weaviate
import requests
import os
import pickle

# URL to the preprocessed data file on GitHub
preprocessed_data_url = 'https://raw.githubusercontent.com/Mohammed9148/new/main/preprocessed_data.pkl'

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
    data_file = download_preprocessed_data(preprocessed_data_url)
    data = load_preprocessed_data(data_file)
    # Check if data is a tuple containing chunks and embeddings
    if isinstance(data, tuple) and len(data) == 2:
        chunks, embeddings = data
        st.write("Data loaded successfully.")
        st.write(f"Number of chunks: {len(chunks)}")
        st.write(f"Shape of embeddings: {embeddings.shape}")
    else:
        st.error("Data is not in the expected format.")
        st.stop()

except Exception as e:
    st.error(f"Error loading preprocessed data: {e}")
    st.stop()

# Initialize Weaviate client
client = weaviate.Client(
    url="https://test-6k1b5vrp.weaviate.network",  # Replace with your Weaviate instance URL
    auth_client_secret=weaviate.AuthApiKey(api_key="viv4g4LcZpE7DDNQx6Fc9Yj3oK7n6DwIeZWF")  # Replace with your API key
)

# Function to upload data to Weaviate
def upload_data_to_weaviate(chunks, embeddings):
    for chunk, embedding in zip(chunks, embeddings):
        data_object = {
            "text": chunk,
            "embedding": embedding.tolist()
        }
        client.data_object.create(data_object, "DocumentChunk")

# Check if data is already uploaded to Weaviate
if st.button('Upload Data to Weaviate'):
    upload_data_to_weaviate(chunks, embeddings)
    st.write("Data uploaded to Weaviate successfully.")

# Function to verify data upload to Weaviate
def verify_data_upload():
    try:
        query = client.query.get("DocumentChunk", ["text", "embedding"]).do()
        st.write("Weaviate data:", query)
    except Exception as e:
        st.write(f"Error verifying data upload: {e}")

# Button to verify data upload
if st.button('Verify Data Upload'):
    verify_data_upload()

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

# Function to perform similarity search and get the most relevant chunk from Weaviate
def get_relevant_chunk(question):
    question_embedding = model.encode([question])[0]  # Flatten the list of list
    near_vector = {
        "vector": question_embedding,
        "certainty": 0.7  # Adjust based on your requirement
    }
    try:
        result = client.query.get("DocumentChunk", ["text"]).with_near_vector(near_vector).do()
        
        # Print the entire response to understand its structure
        st.write("Weaviate response:", result)
        
        # Handle the response safely
        document_chunks = result.get('data', {}).get('Get', {}).get('DocumentChunk', [])
        if document_chunks:
            return document_chunks[0].get('text', "No text found.")
        else:
            return "No relevant chunk found."
    except Exception as e:
        st.error(f"Error during Weaviate query: {e}")
        return "Error during Weaviate query."

# Function to handle question submission
def handle_question():
    if st.session_state.user_question:
        relevant_chunk = get_relevant_chunk(st.session_state.user_question)
        prompt = f"Answer the following question based on this text: {relevant_chunk}\n\nQuestion: {st.session_state.user_question}\nAnswer:"
        response = llm.invoke(prompt)
        st.session_state.response = response.content

# Streamlit app interface
st.title("PDF Chatbot with Azure OpenAI")

st.text_input("Type your question here:", key="user_question", on_change=handle_question)

if "response" in st.session_state:
    st.write("Response:", st.session_state.response)
