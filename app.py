from dotenv import load_dotenv
import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredEPubLoader
from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.document_loaders.csv_loader import CSVLoader
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

    if file_extension == ".pdf":
        documents = extract_pdf_content(temp_file_name)
    elif file_extension in (".xls", ".xlsx", ".csv"):
        documents = extract_csv_content(temp_file_name)
    elif file_extension in (".docx"):
        documents = extract_word_content(temp_file_name)
    elif file_extension == ".epub":
        documents = extract_epub_content(temp_file_name)
    elif file_extension == ".epub":
        documents = extract_md_content(temp_file_name)
    elif file_extension == ".txt":
        documents = extract_txt_content(temp_file_name)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap  = CHUNK_OVERLAP,
        length_function = len,
    )
    chunks = text_splitter.split_documents(documents)

    temp_file.close()
    os.remove(temp_file_name)

    return chunks

def load_vectorDB(embedding):
    ABS_PATH = os.path.dirname(os.path.abspath(__file__))
    DB_DIR = os.path.join(ABS_PATH, "db")

    if not os.path.exists(DB_DIR):
        os.mkdir(DB_DIR)

    # 向量数据库的配置
    client_settings = chromadb.config.Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=DB_DIR,
        anonymized_telemetry=False,
    )
    vectorstore = Chroma(
        collection_name=DB_VECTOR_NAME,
        embedding_function=embedding,
        client_settings=client_settings,
        persist_directory=DB_DIR,
    )

    return vectorstore

def save_vectorDB(vectorstore,docs,embedding):
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
        openai_api_base = os.environ['API_BASE'],
        openai_api_key = os.environ['API_KEY'],
        temperature=AI_TEMPERATURE)
    
    embedding = OpenAIEmbeddings()
    
    return chat,embedding

def load_azureLLM():
    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_BASE"] = os.environ['API_BASE']
    os.environ["OPENAI_API_KEY"] = os.environ['API_KEY']
    os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"

    AZURE_DEPLOYMENT_NAME = os.environ['DEPLOYMENT_NAME_CHAT']
    AZURE_DEPLOYMENT_NAME_EMBEDDING = os.environ['DEPLOYMENT_NAME_EMBEDDING']

    # 初始化大语言模型
    chat = AzureChatOpenAI(
        openai_api_type= "azure",
        openai_api_base=os.environ['API_BASE'],
        openai_api_key=os.environ['API_KEY'],
        openai_api_version="2023-03-15-preview",
        deployment_name=AZURE_DEPLOYMENT_NAME,
        temperature=AI_TEMPERATURE,
        max_tokens=AI_MAX_TOKENS,
        streaming=True,
        )

    # 初始化向量模型
    embedding = OpenAIEmbeddings(deployment = AZURE_DEPLOYMENT_NAME_EMBEDDING,chunk_size=1)
    
    return chat,embedding

def load_LLM():
    ai_type = os.environ["API_TYPE"]
    if ai_type == 'azure':
        chat,embedding = load_azureLLM()
    else:
        chat,embedding = load_openAILLM()

    return chat,embedding

def ask(chat,vectorstore):
    user_question = st.text_input("请向有什么可以帮助您的？")
    if user_question:
      # 在向量数据库中查找相似度最高的TopN结果
      docs = vectorstore.similarity_search(user_question)
      
      # 聚合topN相似度的embddings，向llm提问
      chain = load_qa_chain(chat, chain_type="stuff")
      with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=user_question)
        # print(cb)
      
      ## 回显
      st.write(response)
      st.write(cb)

def upload(vectorstore,embedding):
    # upload file
    uploaded_file = st.file_uploader("上传文档", type=["pdf", "epub", "md", "txt", "docx", ".xls", ".xlsx", ".csv"])

    if uploaded_file is not None:
      st.write("正在提取内容，请稍等...")
      docs = extract_file_content(uploaded_file)
    #   st.write("提取的内容如下：")
    #   st.write(docs)

      st.write("正在向量化存储内容......")
        # vectorstore.from_documents(
        #     documents=docs, 
        #     embedding=embedding)
      docList = vectorstore.add_documents(documents=docs, embedding=embedding)
      st.write("增加后的文件列表：",docList)
      vectorstore.persist()

def main():
    load_dotenv()
    st.set_page_config(page_title="ChatDocument")
    st.header("ChatDocument 与文档交流 💬")

    # 加载模型
    chat,embedding = load_LLM()

    # 加载向量数据库
    vectorstore = load_vectorDB(embedding)

    # 上传文件
    upload(vectorstore,embedding)

    # 用户交互提问
    ask(chat,vectorstore)



if __name__ == '__main__':
    main()
