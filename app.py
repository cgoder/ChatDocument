from dotenv import load_dotenv
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

import os
import tempfile
from PyPDF2 import PdfReader
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import chardet

def extract_file_content(file):
    file_extension = os.path.splitext(file.name)[1]
    content = []

    if file_extension == ".pdf":
        # 提取 pdf 文件内容
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            content.append(page.extract_text())

    elif file_extension == ".epub":
        # 提取 epub 文件内容
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.read())
            temp_file_name = temp_file.name
        
        book = epub.read_epub(temp_file_name)
        os.remove(temp_file_name)
        
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), "html.parser")
                text = soup.get_text()
                content.append(text)

    elif file_extension == ".txt":
        # 提取 txt 文件内容
        raw_content = file.read()
        
        candidate_encodings = ['utf-8', 'gbk', 'gb18030', 'big5']
        detected_encoding = None
        for encoding in candidate_encodings:
            try:
                detected_encoding = encoding
                content = raw_content.decode(encoding).splitlines()
                break
            except UnicodeDecodeError:
                continue
        
        if detected_encoding is None:
            raise ValueError("Could not detect encoding for the text file.")
            

    else:
        st.warning("Unsupported file type. Please upload a PDF, EPUB or TXT file.")
        return ""

    # 将文本内容连接成一个字符串，并通过text-splitting算法进行分块
    content_str = "\n".join(content)
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200,length_function=len)
    chunks = text_splitter.split_text(content_str)

    return chunks
      

def embedding(contents):
    # create embeddings
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(contents, embeddings)
    return knowledge_base
    
def ask(knowledge_base):
    # show user input
    user_question = st.text_input("请向您的文档提问:")
    if user_question:
      docs = knowledge_base.similarity_search(user_question)
      
      llm = OpenAI()
      chain = load_qa_chain(llm, chain_type="stuff")
      with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=user_question)
        print(cb)
          
      st.write(response)


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

      st.write("提取的内容如下：")
      st.write(chunks)

      st.write("向量化存储内容......")
      knowledge_base = embedding(chunks)

      # 用户交互提问
      ask(knowledge_base)




if __name__ == '__main__':
    main()
