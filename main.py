import streamlit as st
import PyPDF2
from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    model="gpt-35-turbo-16k",
    deployment_name="VISAGenAI",
    api_key="dbb69df9354846529d8994cb313275e0",
    azure_endpoint="https://visagenai.openai.azure.com/",
    api_version="2024-02-01",
)


# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to handle question submission
def handle_question():
    if st.session_state.user_question:
        prompt = f"Answer the following question based on this text: {st.session_state.pdf_text}\n\nQuestion: {st.session_state.user_question}\nAnswer:"
        response = llm.invoke(prompt)
        st.session_state.response = response.content

# Streamlit app interface
st.title("PDF Chatbot with Azure OpenAI")
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    pdf_text = extract_text_from_pdf(uploaded_file)
    st.session_state.pdf_text = pdf_text
    st.write("PDF content successfully extracted. You can now ask questions based on this content.")

    st.text_input("Type your question here:", key="user_question", on_change=handle_question)

    if "response" in st.session_state:
        st.write("Response:", st.session_state.response)
