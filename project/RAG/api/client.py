import requests
import streamlit as st 

# def get_llama2_response(input_text):
#     json = {'input':{'topic':input_text}}
#     response = requests.post("http://localhost:8000/essay/invoke", json = json)

    # data = response.json()
    # st.write(data)
    # return data.get('output','No output found')

def get_gemma_response(input_text):
    
    json = {'input':{'topic':input_text}}
    response = requests.post("http://localhost:8000/poem/invoke", json = json)

    data = response.json()
    st.write(data)
    return data.get('output','No output found')

#streamlit framework
st.title('Langchain demo with llama2 api')
# input_text=st.text_input('Write an essay')
input_text1 = st.text_input('Write a poem on')

# if input_text:
#     st.write(get_llama2_response(input_text))

if input_text1:
    st.write(get_gemma_response(input_text1))
    