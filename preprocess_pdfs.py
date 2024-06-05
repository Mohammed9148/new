import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Load Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract text from multiple PDFs
def extract_text_from_pdfs(pdf_files):
    text = ""
    for pdf_file in pdf_files:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to chunk text into smaller pieces
def chunk_text(text, chunk_size=1000, overlap=200):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Function to compute embeddings for text chunks
def compute_embeddings(chunks):
    embeddings = model.encode(chunks)
    return embeddings

# Specify the paths to your PDF files
pdf_files = ['path/to/pdf1.pdf', 'path/to/pdf2.pdf', 'path/to/pdf3.pdf', 'path/to/pdf4.pdf', 'path/to/pdf5.pdf', 'path/to/pdf6.pdf']

# Extract and preprocess text from PDFs
pdf_text = extract_text_from_pdfs(pdf_files)
chunks = chunk_text(pdf_text)
embeddings = compute_embeddings(chunks)

# Save the preprocessed data
with open('preprocessed_data.pkl', 'wb') as f:
    pickle.dump((chunks, embeddings), f)
