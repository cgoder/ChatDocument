from dotenv import load_dotenv
import streamlit as st

from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader,
    UnstructuredEPubLoader, UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
import tempfile
import chromadb


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

AI_TEMPERATURE = 0
AI_MAX_TOKENS = 500

DB_VECTOR_NAME = 'langchain_store'


def extract_pdf_content(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()


def extract_word_content(file_path):
    loader = UnstructuredWordDocumentLoader(file_path, mode="elements")
    return loader.load_and_split()


def extract_csv_content(file_path):
    loader = CSVLoader(file_path)
    return loader.load_and_split()


def extract_epub_content(file_path):
    loader = UnstructuredEPubLoader(file_path, mode="elements")
    return loader.load_and_split()


def extract_md_content(file_path):
    loader = UnstructuredMarkdownLoader(file_path, mode="elements")
    return loader.load_and_split()


def extract_txt_content(file_path):
    loader = TextLoader(file_path, encoding="utf8")
    return loader.load_and_split()


def extract_file_content(file):
    file_extension = os.path.splitext(file.name)[1]
    documents = []

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(file.read())
        temp_file_name = temp_file.name

    loaders = {
        ".pdf": extract_pdf_content,
        ".xls": extract_csv_content,
        ".xlsx": extract_csv_content,
        ".csv": extract_csv_content,
        ".docx": extract_word_content,
        ".epub": extract_epub_content,
        ".md": extract_md_content,
        ".txt": extract_txt_content
    }
    if file_extension in loaders:
        documents = loaders[file_extension](temp_file_name)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)

    temp_file.close()
    os.remove(temp_file_name)

    return chunks

def upload(embedding):
    st.markdown("# 上传")

    file = st.file_uploader("上传文档", type=["pdf", "xlsx", "csv", "docx", "epub", "md", "txt"])

    if file is not None:
        chunks = extract_file_content(file)

        vectorstore = load_vectorDB(embedding)

        st.write(f"## 已上传 {len(chunks)} 个文档块")

        # for chunk in chunks:
        #     chunk_id = vectorstore.add_document(chunk)
        #     vectorstore = save_vectorDB(vectorstore, [chunk], embedding)
        docList = vectorstore.add_documents(documents=chunks, embedding=embedding)
        st.write("当前文件列表：",docList)
        vectorstore.persist()

        # st.write(f"## 上传文档完成")

def load_vectorDB(embedding):
    ABS_PATH = os.path.dirname(os.path.abspath(__file__))
    DB_DIR = os.path.join(ABS_PATH, "db")

    if not os.path.exists(DB_DIR):
        os.mkdir(DB_DIR)

    print(embedding)

    vectorstore = Chroma(DB_VECTOR_NAME, embedding)


    # client_settings = chromadb.config.Settings(
    #     chroma_db_impl="duckdb+parquet",
    #     persist_directory=DB_DIR,
    #     anonymized_telemetry=False,
    # )
    # vectorstore = Chroma(
    #     collection_name=DB_VECTOR_NAME,
    #     embedding_function=embedding,
    #     client_settings=client_settings,
    #     persist_directory=DB_DIR,
    # )

    return vectorstore


def save_vectorDB(vectorstore, docs, embedding):
    ABS_PATH = os.path.dirname(os.path.abspath(__file__))
    DB_DIR = os.path.join(ABS_PATH, "db")

    if not os.path.exists(DB_DIR):
        os.mkdir(DB_DIR)

    vectorstore.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory=DB_DIR)

    return vectorstore


def load_openAILLM():
    os.environ["OPENAI_API_KEY"] = os.environ['API_KEY']
    os.environ["OPENAI_API_BASE"] = os.environ['API_BASE']

    chat = ChatOpenAI(
        openai_api_base=os.environ['API_BASE'],
        openai_api_key=os.environ['API_KEY'],
        temperature=AI_TEMPERATURE)

    embedding = OpenAIEmbeddings()

    return chat, embedding


def load_azureLLM():
    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_BASE"] = os.environ['API_BASE']
    os.environ["OPENAI_API_KEY"] = os.environ['API_KEY']
    AZURE_DEPLOYMENT_NAME = os.environ['DEPLOYMENT_NAME_CHAT']
    AZURE_DEPLOYMENT_NAME_EMBEDDING = os.environ['DEPLOYMENT_NAME_EMBEDDING']

    chat = AzureChatOpenAI(
        openai_api_type= "azure",
        openai_api_base=os.environ['API_BASE'],
        openai_api_key=os.environ['API_KEY'],
        openai_api_version="2023-03-15-preview",
        deployment_name=AZURE_DEPLOYMENT_NAME,
        temperature=AI_TEMPERATURE,
        max_tokens=AI_MAX_TOKENS,
        )

    # 初始化向量模型
    embedding = OpenAIEmbeddings(deployment = AZURE_DEPLOYMENT_NAME_EMBEDDING,chunk_size=1)
    

    return chat, embedding


def load_QA(chat):
    qa_chain = load_qa_chain(chat, chain_type="stuff",verbose=True)

    # ABS_PATH = os.path.dirname(os.path.abspath(__file__))
    # QA_DIR = os.path.join(ABS_PATH, "question_answering")
    # QA_FILE = os.path.join(QA_DIR, "models", "en", "roberta-base-squad2-uncased")

    # qa_chain = load_qa_chain(QA_FILE)

    return qa_chain


def get_answer(question, context, qa_chain):
    result = qa_chain.predict(question, context=context)

    if result.score > 0.5:
        return result.answer

    return None


def chatbot(chat,embedding):
    st.markdown("# 交流")

    vectorstore = load_vectorDB(embedding)
    # if vectorstore.vector_count() == 0:
    #     st.error("请先上传资料")
    #     if st.button("跳转到上传页面"):
    #         st.experimental_rerun()

    # qa_chain = load_QA(chat)
    qa_chain = load_qa_chain(chat, chain_type="stuff")

    user_input = st.text_input("让我们来聊一聊","")
    if st.button("提问") or user_input:
        st.write("正在查找答案...")

        # 在向量数据库中查找相似度最高的TopN结果
        docs = vectorstore.similarity_search(user_input)
        # with get_openai_callback() as cb:
        #     response = qa_chain.run(input_documents=docs, question=user_input)
        #     # print(cb)
        
        # ## 回显
        # st.write(response)
        # st.write(cb)

        st.write(docs)

def model():
    st.markdown("# 模型")
    chat_mode = st.selectbox("请选择一个语言模型", ["OpenAI", "Azure"], index=["OpenAI", "Azure"].index(st.session_state["chat_mode"]))
    
    if st.button("保存"):
        st.session_state["chat_mode"] = chat_mode
        st.success("保存成功！")

    if st.session_state["chat_mode"] == "OpenAI":
        chat,embedding = load_openAILLM()
    else:
        chat,embedding = load_azureLLM()
    
    return chat,embedding

def main():
    load_dotenv()
    st.set_page_config(page_title="ChatDocument", page_icon=":earth_asia:", layout="wide")

    st.sidebar.title("ChatDocument")
    option = st.sidebar.radio("选择一个功能", ["模型", "交流", "上传"])

    chat = None
    embedding = None
    chat_mode = None

    chat_mode = "Azure"
    st.session_state["chat_mode"] = "Azure"
    chat,embedding = load_azureLLM()

    if option == "模型":
        chat, embedding = model()

    elif option == "交流":
        # st.subheader("交流")
        if chat_mode is None or chat is None or embedding is None:
            st.write(chat_mode,chat,embedding)
            st.warning("请先选择语言模型！")
        else:
            chatbot(chat,embedding)

    else:
        # st.subheader("上传")
        if chat_mode is None or embedding is None:
            st.warning("请先选择语言模型！")
        else:
            upload(embedding)

if __name__ == "__main__":
    main()