#Building simple Q&A chat bot

import os
from config import OPENAI_API_TYPE,OPENAI_API_VERSION,AZURE_OPENAI_API_KEY,AZURE_OPENAI_ENDPOINT
from langchain_openai import AzureChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

import streamlit as st

os.environ["AZURE_OPENAI_API_KEY"]=AZURE_OPENAI_API_KEY
os.environ["AZURE_OPENAI_ENDPOINT"]=AZURE_OPENAI_ENDPOINT
os.environ["OPENAI_API_TYPE"]=OPENAI_API_TYPE
os.environ["OPENAI_API_VERSION"]=OPENAI_API_VERSION

# Initializing streamlit framework
st.title('Demo Langchain')
input_text =st.text_input("Capital of country search")


# Openai LLMs
llm = AzureChatOpenAI(temperature=0.8,azure_deployment='gpt4')



if input_text:
    st.write(llm.invoke(input_text))