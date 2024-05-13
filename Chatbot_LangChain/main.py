import os
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain,SequentialChain
from langchain.memory import ConversationBufferMemory
from config import OPENAI_API_TYPE,OPENAI_API_VERSION,AZURE_OPENAI_API_KEY,AZURE_OPENAI_ENDPOINT

import streamlit as st

# Setting up env variables
os.environ["AZURE_OPENAI_API_KEY"]=AZURE_OPENAI_API_KEY
os.environ["AZURE_OPENAI_ENDPOINT"]=AZURE_OPENAI_ENDPOINT
os.environ["OPENAI_API_TYPE"]=OPENAI_API_TYPE
os.environ["OPENAI_API_VERSION"]=OPENAI_API_VERSION

# Initializing Streamlit
st.title("Know capital of your State/Coiuntry")
input_text=st.text_input("Demo")

# Memory Initiazlization
capital_memory = ConversationBufferMemory(input_key='State/Country',memory_key='chat_history')
place_memory = ConversationBufferMemory(input_key='capital',memory_key='chat_history')
list_of_places = ConversationBufferMemory(input_key='Places',memory_key='chat_history')

# First Prompt 
first_input_prompt = PromptTemplate(
    input_variables=['State/Country'],
    template="What is capital of {State/Country}?"
)

# Model Initialization
llm = AzureChatOpenAI(temperature=0.8,azure_deployment='gpt4')
chain1 = LLMChain(llm=llm,prompt=first_input_prompt,verbose=True,output_key='capital',memory=capital_memory)


# Second Prompt
second_prompt_template = PromptTemplate(
    input_variables=['capital'],
    template="Give me population for {capital}"
)

# Model Initialization
llm = AzureChatOpenAI(temperature=0.7,azure_deployment='gpt4')
chain2 = LLMChain(llm=llm,prompt=second_prompt_template,verbose=True,output_key='Places',memory=place_memory)


# Third Template
third_prompt_template = PromptTemplate(
    input_variables=['Places'],
    template="Suggest me top 5 {Places} in this {capital} "
)

llm = AzureChatOpenAI(temperature=0.7,azure_deployment='gpt4')
chain3 = LLMChain(llm=llm,prompt=third_prompt_template,verbose=True,output_key='list of places',memory=list_of_places)


parent_chain = SequentialChain(chains=[chain1,chain2,chain3],input_variables=['State/Country'],output_variables=['capital','Places','list of places'],verbose=True)

# Printing output on streamlit
if input_text:
    st.write(parent_chain({'State/Country':input_text}))
