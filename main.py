import streamlit as st
from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    model="gpt-35-turbo-16k",
    deployment_name="VISAGenAI",
    api_key="dbb69df9354846529d8994cb313275e0",
    azure_endpoint="https://visagenai.openai.azure.com/",
    api_version="2024-02-01",
)

st.title("Chat with Azure OpenAI")
user_input = st.text_input("Type your message here:")

if st.button("Send"):
    response = llm.invoke(user_input)
    st.write("Response:", response)
