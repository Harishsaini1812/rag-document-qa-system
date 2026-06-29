# ===================== IMPORTS =====================
import os
import tempfile
import sqlite3
from typing import Dict, Any, Optional, List, TypedDict, Annotated

from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# ===================== ENV =====================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ===================== LLM + EMBEDDINGS =====================
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    streaming=True
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# ===================== FASTAPI APP =====================
app = FastAPI(title="RAG Document QA System")

# ===================== MEMORY STORAGE =====================
# thread-wise vector stores
THREAD_VECTORS: Dict[str, Any] = {}

# thread metadata
THREAD_META: Dict[str, dict] = {}

# ===================== REQUEST MODELS =====================

class ChatRequest(BaseModel):
    thread_id: str
    question: str


class ChatResponse(BaseModel):
    answer: str
    thread_id: str


# ===================== PDF INGESTION =====================

def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None):
    """
    1. Save PDF temporarily
    2. Load PDF
    3. Split into chunks
    4. Create embeddings
    5. Store FAISS vector DB per thread
    """

    if not file_bytes:
        raise ValueError("Empty PDF file")

    # save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        temp_path = tmp.name

    try:
        # load PDF
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        # split text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(docs)

        # create vector store
        vector_store = FAISS.from_documents(chunks, embeddings)

        # store per thread
        THREAD_VECTORS[thread_id] = vector_store

        THREAD_META[thread_id] = {
            "filename": filename or os.path.basename(temp_path),
            "pages": len(docs),
            "chunks": len(chunks)
        }

        return THREAD_META[thread_id]

    finally:
        try:
            os.remove(temp_path)
        except:
            pass

#################################################################################################################################################

# ===================== RETRIEVAL FUNCTION =====================

def retrieve_context(thread_id: str, query: str, k: int = 4) -> str:
    """
    Fetch relevant chunks from FAISS vector DB
    """
    vector_store = THREAD_VECTORS.get(thread_id)

    if not vector_store:
        return ""

    docs = vector_store.similarity_search(query, k=k)

    context = "\n\n".join([doc.page_content for doc in docs])
    return context


# ===================== LANGGRAPH STATE =====================

class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    thread_id: str


# ===================== LLM RESPONSE NODE =====================

def call_llm(state: ChatState):
    """
    Main AI brain:
    - Takes question
    - Retrieves context from FAISS
    - Sends to LLM
    """

    messages = state["messages"]
    thread_id = state["thread_id"]

    # last user message
    user_question = messages[-1].content

    # get context from vector DB
    context = retrieve_context(thread_id, user_question)

    system_prompt = f"""
You are a helpful AI assistant.
Use the given context to answer the user.

CONTEXT:
{context}

If context is empty, answer normally but say you don't have document reference.
"""
    
# ============================ Strict RAG Mode ====================================    
#     system_prompt = f"""
# You are a strict document-based assistant.

# RULES:
# - Only use the given context below.
# - If answer is not in context, say:
#   "Answer is not available in the provided document."

# CONTEXT:
# {context}
# """
# ============================ Strict RAG Mode ====================================   

    full_prompt = [
        {"role": "system", "content": system_prompt},
        *[{"role": m.type, "content": m.content} for m in messages]
    ]

    response = llm.invoke(full_prompt)

    return {
        "messages": [response]
    }


# ===================== SIMPLE GRAPH SETUP =====================

from langgraph.graph import StateGraph, START, END

graph = StateGraph(ChatState)

graph.add_node("llm", call_llm)

graph.add_edge(START, "llm")
graph.add_edge("llm", END)

rag_app = graph.compile()


# ===================== CHAT FUNCTION =====================

def chat_with_rag_stream(thread_id: str, question: str):

    context = retrieve_context(thread_id, question)

    system_prompt = f"""
You are a helpful AI assistant.
Answer ONLY using the context below.
If answer is not found in context, say:
'I could not find this information in the uploaded document.'

Context:
{context}
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]

    for chunk in llm.stream(messages):

        if chunk.content:
            yield chunk.content

#################################################################################################################################################

# ===================== ROOT =====================

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "RAG system running"}


# ===================== PDF UPLOAD =====================

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    thread_id: str = "default"
):
    """
    Upload PDF and create vector store
    """

    file_bytes = await file.read()

    meta = ingest_pdf(
        file_bytes=file_bytes,
        thread_id=thread_id,
        filename=file.filename
    )

    return {
        "message": "PDF uploaded successfully",
        "thread_id": thread_id,
        "metadata": meta
    }


# ===================== CHAT ENDPOINT =====================

class ChatRequest(BaseModel):
    thread_id: str
    question: str


@app.post("/chat")
def chat(req: ChatRequest):

    return StreamingResponse(
        chat_with_rag_stream(
            req.thread_id,
            req.question
        ),
        media_type="text/plain"
    )


# ===================== LIST THREADS =====================

@app.get("/threads")
def list_threads():
    """
    List all active threads
    """

    return {
        "threads": list(THREAD_VECTORS.keys())
    }


# ===================== THREAD DETAILS =====================

@app.get("/thread/{thread_id}")
def get_thread(thread_id: str):
    """
    Get metadata of a thread
    """

    return {
        "thread_id": thread_id,
        "metadata": THREAD_META.get(thread_id, {}),
        "has_data": thread_id in THREAD_VECTORS
    }

#################################################################################################################################################

# ===================== ERROR HANDLING WRAPPER =====================

from fastapi import HTTPException

def safe_execute(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===================== WRAPPED ENDPOINT FIX (OPTIONAL SAFETY) =====================
# (keeping simple, but ensures stability)

@app.post("/upload_safe")
async def upload_pdf_safe(file: UploadFile = File(...), thread_id: str = "default"):
    file_bytes = await file.read()
    return safe_execute(
        ingest_pdf,
        file_bytes,
        thread_id,
        file.filename
    )


@app.post("/chat_safe")
def chat_safe(req: ChatRequest):
    return {
        "thread_id": req.thread_id,
        "answer": safe_execute(
            chat_with_rag_stream,
            req.thread_id,
            req.question
        )
    }


# ===================== START SERVER =====================

if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting RAG Backend Server...")

    uvicorn.run(
        "backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )