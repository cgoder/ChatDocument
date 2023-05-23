
import streamlit as st

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from llm import get_LLM
from vectordb import getVectorDBstore

from streamlit.logger import get_logger
logger = get_logger(__name__)

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)


def count_tokens(question, chatLLM):
    count = f'Words: {len(question.split())}'
    return count


def chat_with_doc():
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    question = st.text_area("与LittaPower交流")

    columns = st.columns(3)
    with columns[0]:
        button = st.button("Ask")
    with columns[1]:
        count_button = st.button("Count Tokens", type='secondary')
    with columns[2]:
        clear_history = st.button("Clear History", type='secondary')
    
    
    if clear_history:
        # Clear memory in Langchain
        memory.clear()
        st.session_state['chat_history'] = []
        st.experimental_rerun()

    if button:
        qa = None
        chatLLM,embeddingLLM = get_LLM()
        vectorDB_documents, vectorDB_summaries = getVectorDBstore()
        
        qa = ConversationalRetrievalChain.from_llm(chatLLM, vectorDB_documents.as_retriever(), memory=memory, verbose=True)
    
        st.session_state['chat_history'].append(("You", question))

        # Generate model's response and add it to chat history
        model_response = qa({"question": question})
        logger.info('Result: %s', model_response)

        st.session_state['chat_history'].append(("LittaPower", model_response["answer"]))

        # Display chat history
        st.empty()
        for speaker, text in st.session_state['chat_history']:
            st.markdown(f"**{speaker}:** {text}")
        
    if count_button:
        st.write(count_tokens(question, chatLLM))
