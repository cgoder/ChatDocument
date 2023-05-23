import streamlit as st
from dotenv import load_dotenv, find_dotenv
import os

def load_config():
    # 加载环境变量
    load_dotenv(find_dotenv())

    # 从环境变量中读取配置项，并存储到 session_state 中
    if 'API_TYPE' in os.environ:
        st.session_state['openai_api_type'] = os.environ['API_TYPE']
    else:
        st.session_state['openai_api_type'] = 'OpenAI'

    if 'API_BASE' in os.environ:
        st.session_state['openai_api_base'] = os.environ['API_BASE']
    else:
        st.session_state['openai_api_base'] = 'https://api.openai.com/'

    if 'API_KEY' in os.environ:
        st.session_state['openai_api_key'] = os.environ['API_KEY']
    else:
        st.session_state['openai_api_key'] = 'sk-'

    if 'DEPLOYMENT_NAME_CHAT' in os.environ:
        st.session_state['openai_model_chat'] = os.environ['DEPLOYMENT_NAME_CHAT']
    else:
        st.session_state['openai_model_chat'] = 'gpt-3.5-turbo'

    if 'DEPLOYMENT_NAME_EMBEDDING' in os.environ:
        st.session_state['openai_model_embedding'] = os.environ['DEPLOYMENT_NAME_EMBEDDING']
    else:
        st.session_state['openai_model_embedding'] = 'text-davinci-003'


    if 'TEMPERATURE' in os.environ:
        st.session_state['temperature'] = os.environ['TEMPERATURE']
    else:
        st.session_state['temperature'] = 0.1

    if 'MAX_TOKENS' in os.environ:
        st.session_state['max_tokens'] = os.environ['MAX_TOKENS']
    else:
        st.session_state['max_tokens'] = 1000

    if 'chunk_size' in os.environ:
        st.session_state['chunk_size'] = os.environ['chunk_size']
    else:
        st.session_state['chunk_size'] = 500

    if 'chunk_overlap' in os.environ:
        st.session_state['chunk_overlap'] = os.environ['chunk_overlap']
    else:
        st.session_state['chunk_overlap'] = 0

    # print("\n\n--------- ",st.session_state)

def get_config(key):
    # 读取 session_state 的配置项
    if key not in st.session_state:
        load_config()

    if key in st.session_state:
        return st.session_state[key]
    else:
        return 0
