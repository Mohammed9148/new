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
    pdf_reader = PyPDF2.PdfFileReader(pdf_file)
    text = ""
    for page_num in range(pdf_reader.getNumPages()):
        page = pdf_reader.getPage(page_num)
        text += page.extract_text()
    return text

# Streamlit app interface
st.title("Chat with Azure OpenAI")
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    pdf_text = extract_text_from_pdf(uploaded_file)
    st.write("Pdf successfully extracted")
    
    st.text_input("Type your question here:", key ="user_question", on_change=handle_question)

    if "response" in st.session_state:
        st.write("Response:", st.session_state.content)
