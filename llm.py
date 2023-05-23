
import streamlit as st

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from config import get_config


def load_openAILLM():
    openai_api_base = get_config('openai_api_base')
    openai_api_key = get_config('openai_api_key')
    temperature = get_config('temperature')

    chat = ChatOpenAI(
        openai_api_base=openai_api_base,
        openai_api_key=openai_api_key,
        temperature=temperature)

    embedding = OpenAIEmbeddings()

    return chat, embedding


def load_azureLLM():
    openai_api_type = get_config('openai_api_type')
    openai_api_base = get_config('openai_api_base')
    openai_api_key = get_config('openai_api_key')
    temperature = get_config('temperature')
    maxToken = get_config('max_tokens')
    openai_model_chat = get_config('openai_model_chat')
    openai_model_embedding = get_config('openai_model_embedding')

    chat = AzureChatOpenAI(
        openai_api_type= "azure",
        openai_api_base=openai_api_base,
        openai_api_key=openai_api_key,
        openai_api_version="2023-03-15-preview",
        deployment_name=openai_model_chat,
        temperature=temperature,
        max_tokens=maxToken,
        )

    # 初始化向量模型
    embedding = OpenAIEmbeddings(
        openai_api_type= "azure",
        openai_api_base=openai_api_base,
        openai_api_key=openai_api_key,
        openai_api_version="2023-03-15-preview",
        deployment = openai_model_embedding,chunk_size=1)


    return chat, embedding

chatLLM = None
embeddingLLM = None

def get_LLM():
    if chatLLM is not None:
        chat = chatLLM
        embedding = embeddingLLM
    else:
        chat, embedding = load_LLM()

    return chat, embedding

def load_LLM():
    if 'openai_api_type' in st.session_state:
        if st.session_state['openai_api_type'] == "OpenAI":
            chatLLM, embeddingLLM = load_openAILLM()
        else:
            chatLLM, embeddingLLM = load_azureLLM()
    else:
        st.session_state['openai_api_type'] = "Azure"
        chatLLM, embeddingLLM = load_azureLLM()

    return chatLLM, embeddingLLM