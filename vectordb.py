import os

from langchain.vectorstores import SupabaseVectorStore
from supabase import create_client
# from summarization import llm_summerize
from langchain.schema import Document
from llm import get_LLM
from uitl import compute_sha1_from_content

from streamlit.logger import get_logger
logger = get_logger(__name__)


vectorDB_client = None
vectorDB_documents = None
vectorDB_summaries = None




def file_exists_check(file):
    file_content = file.read()
    file_sha1 = compute_sha1_from_content(file_content)
    client = getVectorDBClient()
    response = client.table("documents").select("id").eq(
        "metadata->>file_sha1", file_sha1).execute()
    return len(response.data) > 0

def file_summary_update(document_id, sids):
    if sids and len(sids) > 0:
        client = getVectorDBClient()
        client.table("summaries").update(
            {"document_id": document_id}).match({"id": sids[0]}).execute()


def file_summary_create(document_id, content, metadata):
    logger.info(f"Summarizing document {content[:100]}")
    # summary = llm_summerize(content)
    summary = content
    logger.info(f"Summary: {summary}")

    metadata['document_id'] = document_id

    summary_doc_with_metadata = Document(page_content=summary, metadata=metadata)
    
    sids = vectorDB_summaries.add_documents([summary_doc_with_metadata])
    file_summary_update(document_id,sids)

def file_filter(file, enable_summarization: bool, file_processors):
    if file_exists_check(file):
        return {"message": f"🤔 {file.name} already exists.", "type": "warning"}
    elif file.size < 1:
        return {"message": f"❌ {file.name} is empty.", "type": "error"}
    else:
        file_extension = os.path.splitext(file.name)[-1]
        if file_extension in file_processors:
            file_processors[file_extension](file, enable_summarization)
            return {"message": f"✅ {file.name} has been uploaded.", "type": "success"}
        else:
            return {"message": f"❌ {file.name} is not supported.", "type": "error"}

def getVectorDBClient():
    if vectorDB_client is None:
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")
        client = create_client(supabase_url, supabase_key)
    else:
        client = vectorDB_client
    
    return client

def getVectorDBstore():
    if vectorDB_documents is not None:
        documents = vectorDB_documents
        summaries = vectorDB_summaries
    else:
        chat, embedding = get_LLM()
        documents, summaries = initVectorDB(embedding)

    return documents, summaries


def initVectorDB(embedding):
    client = getVectorDBClient()

    vectorDB_documents = SupabaseVectorStore(
        client, embedding, table_name="documents")

    vectorDB_summaries = SupabaseVectorStore(
        client, embedding, table_name="summaries")
    
    return vectorDB_documents, vectorDB_summaries