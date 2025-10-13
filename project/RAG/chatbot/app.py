from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser #can be made custom as well
from langchain_community.llms import Ollama


import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()
# Langsmith tracking
os.environ['LANGCHAIN_TRACING_V2'] = 'TRUE'
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')


# PROMPT TEMPLATE
prompt = ChatPromptTemplate.from_messages(
    [ 
    ('system','You are a helpful assistant. Please respond to the user queries'),
    ('user','Question:{question}')
]
)

# streamlit framework
st.title('Langchain Demo with Ollama api')
input_text = st.text_input('Search the topic u want')

# Ollma llm
llm = Ollama(model = 'gemma3:1b')
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({'question':input_text}))
