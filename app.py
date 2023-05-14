from dotenv import load_dotenv
import streamlit as st
from langchain.llms import OpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
import chromadb
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredEPubLoader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
import re
import tempfile
# from PyPDF2 import PdfReader
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

import chardet


def replace_newlines_and_spaces(text):
    # Replace all newline characters with spaces
    text = text.replace("\n", " ")

    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)

    return text

def extract_file_content(file):
    file_extension = os.path.splitext(file.name)[1]
    documents = []

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(file.read())
        temp_file_name = temp_file.name

    if file_extension == ".pdf":
        # 提取 pdf 文件内容
        # loader = UnstructuredPDFLoader(temp_file_name, mode="elements")
        # documents = loader.load()
        # text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200,length_function=len)
        # chunks = text_splitter.split_documents(documents)

        loader = PyPDFLoader(temp_file_name)
        documents = loader.load_and_split()

    elif file_extension == ".epub":
        # 提取 epub 文件内容
        loader = UnstructuredEPubLoader(temp_file_name, mode="elements")
        documents = loader.load()
        # text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200,length_function=len)
        # chunks = text_splitter.split_text(documents)

        
        # book = epub.read_epub(temp_file_name)
        # for item in book.get_items():
        #     if item.get_type() == ebooklib.ITEM_DOCUMENT:
        #         soup = BeautifulSoup(item.get_content(), "html.parser")
        #         text = soup.get_text()
        #         documents.append(text)
        
        # # 将文本内容连接成一个字符串，并通过text-splitting算法进行分块
        # text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200,length_function=len)
        # chunks = text_splitter.split_documents(documents)

    elif file_extension == ".txt":
        # 提取 txt 文件内容
        raw_content = file.read()    
        # candidate_encodings = ['utf-8', 'gbk', 'gb18030', 'big5']
        # detected_encoding = None
        # for encoding in candidate_encodings:
        #     try:
        #         detected_encoding = encoding
        #         content = raw_content.decode(encoding).splitlines()
        #         if content:
        #             break
        #     except UnicodeDecodeError:
        #         continue

        result = chardet.detect(raw_content)
        # loader = TextLoader(temp_file_name,'utf-8')
        st.write(result)
        loader = TextLoader(temp_file_name,result['encoding'])
        documents = loader.load()

    else:
        st.warning("Unsupported file type. Please upload a PDF, EPUB or TXT file.")
        return ""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap  = 200,
        length_function = len,
    )
    chunks = text_splitter.split_documents(documents)

    temp_file.close()
    os.remove(temp_file_name)

    return chunks


def embedding(embeddings,contents):
    # create embeddings
    knowledge_db = FAISS.from_documents(contents, embeddings)
    return knowledge_db

def embedding_2_vectorDB(embeddings,contents):
    ABS_PATH = os.path.dirname(os.path.abspath(__file__))
    DB_DIR = os.path.join(ABS_PATH, "db")

    # 初始化向量数据库
    if not os.path.exists(DB_DIR):
        os.mkdir(DB_DIR)

    client_settings = chromadb.config.Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=DB_DIR,
        anonymized_telemetry=False,
    )

    vectorstore = Chroma(
        collection_name="langchain_store",
        embedding_function=embeddings,
        client_settings=client_settings,
        persist_directory=DB_DIR,
    )

    # 存储向量至数据库
    vectorstore.add_documents(documents=contents, embedding=embeddings)
    vectorstore.persist()
    # print("vectorstore: \n",vectorstore)

    return vectorstore


def load_openAILLM():
    llm = OpenAI()
    return llm

def load_azureLLM():
    AZURE_DEPLOYMENT_NAME = os.environ['AZURE_DEPLOYMENT_NAME']
    AZURE_DEPLOYMENT_NAME_EMBEDDING = os.environ['AZURE_DEPLOYMENT_NAME_EMBEDDING']
    # OPENAI_NAME_GPT = os.environ['OPENAI_NAME_GPT']
    # OPENAI_NAME_EMBEDDING = os.environ['OPENAI_NAME_EMBEDDING']
    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_KEY"] = os.environ['AZURE_OPEN_API_KEY']
    os.environ["OPENAI_API_BASE"] = os.environ['AZURE_API_BASE']
    os.environ["OPENAI_VERSION"] = "2023-03-15-preview"

    # 初始化大语言模型
    llm = AzureChatOpenAI(
        openai_api_type= "azure",
        openai_api_base=os.environ['AZURE_API_BASE'],
        openai_api_key=os.environ['AZURE_OPEN_API_KEY'],
        openai_api_version="2023-03-15-preview",
        deployment_name=AZURE_DEPLOYMENT_NAME,
        temperature=0,
        client=None,
        request_timeout=180)

    # print("url: ",os.environ["OPENAI_API_BASE"])
    # print("url: ",os.environ["OPENAI_API_KEY"])
    # 初始化向量模型
    embeddings = OpenAIEmbeddings(deployment=AZURE_DEPLOYMENT_NAME_EMBEDDING,chunk_size=1)
    # embeddings = OpenAIEmbeddings(
    #     api_type= "azure",
    #     api_base=os.environ['AZURE_API_BASE'],
    #     api_key=os.environ['AZURE_OPEN_API_KEY'],
    #     api_version="2023-03-15-preview",
    #     model=AZURE_DEPLOYMENT_NAME_EMBEDDING,
    # )
    # text = "This is a test query."
    # query_result = embeddings.embed_query(text)
    # st.write(query_result)
    
    return llm,embeddings
    
def ask(llm,knowledge_db):
    user_question = st.text_input("请向有什么可以帮助您的？")
    if user_question:
      # 在向量数据库中查找相似度最高的TopN结果
      docs = knowledge_db.similarity_search(user_question)
      
      # 聚合topN相似度的embddings，向llm提问
      chain = load_qa_chain(llm, chain_type="stuff")
      with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=user_question)
        # print(cb)
      
      ## 回显
      st.write(response)
      st.write(cb)

def main():
    load_dotenv()
    st.set_page_config(page_title="ChatDocument")
    st.header("ChatDocument 与文档交流 💬")

    # upload file
    uploaded_file = st.file_uploader("上传文档", type=["pdf", "epub", "txt"])

    if uploaded_file is not None:
      # 根据文件类型调用不同的解析函数
      st.write("正在提取内容，请稍等...")
      chunks = extract_file_content(uploaded_file)

    #   st.write("提取的内容如下：")
    #   st.write(chunks)

      # 加载模型
      llm,embeddings = load_azureLLM()

      st.write("正在向量化存储内容......")
      knowledge_db = embedding_2_vectorDB(embeddings,chunks)

      # 用户交互提问
      ask(llm,knowledge_db)




if __name__ == '__main__':
    main()
