
import os
import time
import tempfile


from langchain.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader,
    UnstructuredEPubLoader, Docx2txtLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader
)
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from vectordb import file_summary_create,getVectorDBstore
from uitl import compute_sha1_from_file



def process_pdf(file, enable_summarization):
    return process_file(file, PyPDFLoader, enable_summarization)

def process_txt(file, enable_summarization):
    return process_file(file, TextLoader, enable_summarization)

def process_epub(file, enable_summarization):
    return process_file(file, UnstructuredEPubLoader, enable_summarization)

def process_csv(file, enable_summarization):
    return process_file(file, CSVLoader, enable_summarization)

def process_docx(file, enable_summarization):
    return process_file(file, Docx2txtLoader, enable_summarization)

def process_markdown(file, enable_summarization):
    return process_file(file, UnstructuredMarkdownLoader, enable_summarization)

def process_powerpoint(file, enable_summarization):
    return process_file(file, UnstructuredPowerPointLoader, enable_summarization)


file_processors = {
    ".txt": process_txt,
    ".csv": process_csv,
    ".md": process_markdown,
    ".md": process_markdown,
    ".markdown": process_markdown,
    ".pdf": process_pdf,
    ".pptx": process_powerpoint,
    ".docx": process_docx,
    ".epub": process_epub,
}



def process_file(file, loader_class, enable_summarization):
    documents = []
    file_name = file.name
    file_size = file.size
    dateshort = time.strftime("%Y%m%d")

    # Here, we're writing the uploaded file to a temporary file, so we can use it with your existing code.
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.name) as tmp_file:
        file.seek(0)
        content = file.read()
        tmp_file.write(content)
        tmp_file.flush()

        loader = loader_class(tmp_file.name)
        documents = loader.load()
        # Ensure this function works with FastAPI
        file_sha1 = compute_sha1_from_file(tmp_file.name)

    os.remove(tmp_file.name)
    chunk_size = 500
    chunk_overlap = 0

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    documents = text_splitter.split_documents(documents)

    vectorDB_documents, vectorDB_summaries = getVectorDBstore()
    for doc in documents:
        metadata = {
            "file_sha1": file_sha1,
            "file_size": file_size,
            "file_name": file_name,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "date": dateshort,
            "summarization": "true" if enable_summarization else "false"
        }
        doc_with_metadata = Document(
            page_content=doc.page_content, metadata=metadata)
        ids = vectorDB_documents.add_documents([doc_with_metadata])
        if enable_summarization and ids and len(ids) > 0:
            file_summary_create(ids[0], doc.page_content, metadata, vectorDB_summaries)



