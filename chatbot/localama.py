from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama 

import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="../.env", override=True)

# Langchain monitoring, tracing 
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# Langsmith tracking
os.environ["LANGCHAIN_TRACKING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to user queries"),
        ("user", "Question:{question}")
    ]
)

#OpenAI LLMs
llm = Ollama(model="gemma3:1b")
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

#Streamlit template
st.title('Langchain demo with Gemma API')
input_text = st.text_input("Search the topic you want")


if input_text:
    st.write(chain.invoke({'question':input_text}))


