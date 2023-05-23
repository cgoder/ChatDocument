# main.py
import streamlit as st

from file import file_processors
from chat import chat_with_doc
from vectordb import getVectorDBClient,file_filter
from explorer import view_document
from config import load_config



models = ["OpenAI", "Azure"]


def main():

    # Set the theme
    st.set_page_config(
        page_title="LITTA Power",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("LITTA Power 🧠")
    st.markdown("#### 您的专属运动健康专家")
    st.markdown("---\n\n")

    # Initialize session state variables
    load_config()

    # Create a radio button for user to choose between adding knowledge or asking a question
    user_choice = st.radio(
        "Choose an action", ("💬与Power交流", '⚡为Power充电', 'Forget', "Explore"))

    st.markdown("---\n\n")


    if user_choice == '⚡为Power充电':
        # Display chunk size and overlap selection only when adding knowledge
        st.sidebar.title("Configuration")
        st.sidebar.markdown(
            "配置模型参数")
        st.session_state['chunk_size'] = st.sidebar.slider(
            "Select Chunk Size", 100, 1000, st.session_state['chunk_size'], 50)
        st.session_state['chunk_overlap'] = st.sidebar.slider(
            "Select Chunk Overlap", 0, 100, st.session_state['chunk_overlap'], 10)

        file_uploader()
    elif user_choice == '💬与Power交流':
        # Display model and temperature selection only when asking questions
        st.sidebar.title("Configuration")
        st.sidebar.markdown("配置交谈的参数")

        # st.session_state['openai_api_type'] = st.sidebar.selectbox(
        #     "Select Model", models, index=(models).index(st.session_state['openai_api_type']))
        st.session_state['temperature'] = st.sidebar.slider(
            "Select Temperature", 0.0, 1.0, st.session_state['temperature'], 0.1)
        st.session_state['max_tokens'] = st.sidebar.slider(
            "Select Max Tokens", 256, 2048, st.session_state['max_tokens'], 2048)

        chat_with_doc()
    elif user_choice == 'Forget':
        st.sidebar.title("Configuration")

        # brain(supabase)
    elif user_choice == 'Explore':
        st.sidebar.title("Configuration")

        vectorDB_client = getVectorDBClient()
        view_document(vectorDB_client)

    st.markdown("---\n\n")


def file_uploader():
    accepted_file_extensions = list(file_processors.keys())

    file = st.file_uploader("**上传文件**", type=accepted_file_extensions)

    if st.button("⚡充电"):
        msg = file_filter(file, False, file_processors)
        file.close()
        st.empty()
        st.write(msg.get('message'))


if __name__ == "__main__":
    main()