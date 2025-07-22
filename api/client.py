import requests
import streamlit as st

def get_llama_response(input_text):
    response = requests.get("http://localhost:8000/essay/invoke",
                            json={'input':{'topic': input_text}})
    return response.json()['output']['content']

def get_gemma_response(input_text):
    response = requests.get("http://localhost:8000/poem/invoke",
                            json={'input':{'topic': input_text}})
    return response.json()['output']

#streamlit framework
st.title('Langchain demo with Gemma3 API')
input_text1 = st.text_input("Write an essay on")
input_text2 = st.text_input("Write an poem on")

if input_text1:
    st.write(get_llama_response(input_text1))
    st.write(get_gemma_response(input_text2))

