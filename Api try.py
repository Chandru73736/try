import os
import io
import json
import time
import uuid
import asyncio
import tempfile
import threading
import logging
import requests
import pdfplumber
import pytesseract
from PIL import Image
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import camelot
from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import streamlit as st
from streamlit_chat import message
from pdf2image import convert_from_path
from langchain_core.globals import set_debug, set_verbose
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.document_loaders import PyPDFLoader

# === Logging Setup ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === Storage Directory Setup ===
STORAGE_PATH = os.environ.get("QPDF_STORAGE_PATH", r"C:\Users\AIPC\Desktop\qpdf_data")
os.makedirs(STORAGE_PATH, exist_ok=True)
logger.info(f"Using storage directory: {STORAGE_PATH}")

# === Create Static Directory and Index File ===
STATIC_DIR = "static"
INDEX_HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head><title>QPDF API</title></head>
<body>
    <h1>Welcome to QPDF API</h1>
    <p>Use the <a href="/docs">/docs</a> endpoint to explore the API.</p>
</body>
</html>
"""
os.makedirs(STATIC_DIR, exist_ok=True)
index_path = os.path.join(STATIC_DIR, "index.html")
if not os.path.exists(index_path):
    with open(index_path, "w") as f:
        f.write(INDEX_HTML_CONTENT)
    logger.info(f"Created {index_path}")

# === Optimized OCR and Table Extraction ===
def preprocess_image(image):
    """Preprocess the image for faster OCR."""
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return Image.fromarray(thresh)

async def extract_text_from_scanned_pdf(file_path: str) -> List[Document]:
    """Extract text from scanned PDFs with no page limit."""
    try:
        images = convert_from_path(file_path, dpi=100)
        preprocessed_images = [preprocess_image(img) for img in images]
        return [Document(page_content=pytesseract.image_to_string(img), metadata={"page": i + 1})
                for i, img in enumerate(preprocessed_images)]
    except Exception as e:
        logger.error(f"Error in OCR extraction: {str(e)}")
        return []

async def extract_text_from_tabular_pdf(file_path: str) -> List[Document]:
    """Extract tables from PDFs using camelot with no page limit."""
    try:
        tables = camelot.read_pdf(file_path, pages="all", flavor="stream")
        documents = []
        for i, table in enumerate(tables):
            content = table.df.to_string(index=False)
            documents.append(Document(page_content=content, metadata={"page": i + 1}))
        return documents
    except Exception as e:
        logger.error(f"Error in table extraction: {str(e)}")
        return []

async def ingest_pdf(file_path: str, file_name: str, source_id: str) -> List[Document]:
    """Ingest PDF file with optimized processing and no page limit."""
    docs = []
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["file_name"] = file_name
            doc.metadata["source_id"] = source_id
    except Exception as e:
        logger.error(f"Error loading {file_name} as text PDF: {str(e)}")

    if not docs or all(len(doc.page_content.strip()) == 0 for doc in docs):
        docs = await extract_text_from_tabular_pdf(file_path)
        for doc in docs:
            doc.metadata["file_name"] = file_name
            doc.metadata["source_id"] = source_id

    if not docs or all(len(doc.page_content.strip()) == 0 for doc in docs):
        docs = await extract_text_from_scanned_pdf(file_path)
        for doc in docs:
            doc.metadata["file_name"] = file_name
            doc.metadata["source_id"] = source_id

    if not docs:
        logger.error(f"No content extracted from {file_name}")
    return docs

# === QPDF Class ===
class QPDF:
    def __init__(self, llm_model: str = "llama3.1"):
        set_debug(True)
        set_verbose(True)
        try:
            self.model = ChatOllama(model=llm_model)
        except Exception as e:
            logger.warning(f"Failed to load model {llm_model}: {str(e)}. Falling back to qwen2.5")
            self.model = ChatOllama(model="qwen2.5")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that answers questions about uploaded PDF documents."),
            ("human", "Document content: {context}\nQuestion: {message}"),
        ])
        chroma_db_path = os.path.join(STORAGE_PATH, "chroma_db")
        logger.info(f"Initializing Chroma vector store at {chroma_db_path}")
        try:
            self.vector_store = Chroma(
                embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
                persist_directory=chroma_db_path
            )
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise

    async def ingest(self, pdf_file_path: str, file_name: str, source_id: str, progress_bar=None, progress_text=None):
        """Ingest a new PDF and update the vector store."""
        try:
            if progress_text:
                progress_text.text(f"Extracting content from {file_name}...")
            docs = await ingest_pdf(pdf_file_path, file_name, source_id)
            if progress_bar:
                progress_bar.progress(0.3)

            if not docs:
                raise Exception(f"No content extracted from {file_name}")

            if progress_text:
                progress_text.text(f"Splitting documents for {file_name}...")
            chunks = self.text_splitter.split_documents(docs)
            chunks = filter_complex_metadata(chunks)
            if progress_bar:
                progress_bar.progress(0.6)

            if progress_text:
                progress_text.text(f"Updating vector store for {file_name}...")
            existing_ids = self.vector_store.get(where={"file_name": file_name})["ids"]
            if existing_ids:
                self.vector_store.delete(ids=existing_ids)

            self.vector_store.add_documents(chunks)
            if progress_bar:
                progress_bar.progress(1.0)
            if progress_text:
                progress_text.text(f"Completed ingestion of {file_name}.")
            logger.info(f"Successfully ingested {file_name} (source_id: {source_id})")
        except Exception as e:
            logger.error(f"Error ingesting {file_name}: {str(e)}")
            if progress_text:
                st.error(f"Error ingesting {file_name}: {str(e)}")
            raise

    def remove_document(self, source_id: str):
        """Remove all documents associated with a specific source_id."""
        try:
            existing_ids = self.vector_store.get(where={"source_id": source_id})["ids"]
            if existing_ids:
                self.vector_store.delete(ids=existing_ids)
                logger.info(f"Removed documents with source_id {source_id}")
            else:
                logger.warning(f"No documents found for source_id {source_id}")
        except Exception as e:
            logger.error(f"Error removing source_id {source_id}: {str(e)}")
            raise

    def clear_cache(self):
        """Clear the entire vector store."""
        try:
            all_ids = self.vector_store.get()["ids"]
            if all_ids:
                self.vector_store.delete(ids=all_ids)
                logger.info("Vector store cleared")
            history_file = os.path.join(STORAGE_PATH, "chat_history.json")
            if os.path.exists(history_file):
                os.remove(history_file)
                logger.info(f"Chat history file removed: {history_file}")
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            raise

    def ask(self, message: str, source_id: Optional[str] = None):
        """Process a message and generate a response."""
        try:
            retriever = self.vector_store.as_retriever(search_kwargs={"filter": {"source_id": source_id}} if source_id else {})
            retrieved = retriever.invoke(message)
            context = "\n\n".join([doc.page_content for doc in retrieved])
            response = self.model.invoke(self.prompt.format_prompt(context=context, message=message).to_messages()).content
            sources = [{"file_name": doc.metadata.get("file_name"), "page": doc.metadata.get("page"), "source_id": doc.metadata.get("source_id")} for doc in retrieved]
            logger.info(f"Message processed: {message}")
            return response, sources
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            raise

    def refine_vector_store(self, message: str, response: str, feedback: str = None):
        """Update vector store with high-quality responses."""
        if feedback == "Good":
            content = f"Question: {message}\nAnswer: {response}"
            new_doc = Document(
                page_content=content,
                metadata={"source": "user_feedback", "timestamp": time.time()}
            )
            chunks = self.text_splitter.split_documents([new_doc])
            chunks = filter_complex_metadata(chunks)
            self.vector_store.add_documents(chunks)
            logger.info("Vector store refined with feedback")

# === FastAPI Setup ===
app = FastAPI(title="QPDF API", version="1.0", description="API for PDF processing, inspired by ChatPDF")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
qpdf_instance = QPDF()
API_KEY_FILE = os.path.join(STORAGE_PATH, "api_keys.json")
CHAT_SESSIONS = {}  # {chatId: {sourceId: str, messages: List[Dict]}}

def load_api_keys() -> dict:
    """Load API keys from JSON file."""
    try:
        if os.path.exists(API_KEY_FILE):
            with open(API_KEY_FILE, "r") as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.error(f"Error loading API keys: {str(e)}")
        return {}

def save_api_keys(api_keys: dict):
    """Save API keys to JSON file."""
    try:
        with open(API_KEY_FILE, "w") as f:
            json.dump(api_keys, f)
    except Exception as e:
        logger.error(f"Error saving API keys: {str(e)}")
        raise

def verify_api_key(x_api_key: str = Header(None)):
    """Verify if the provided API key is valid (optional for local use)."""
    if x_api_key:
        try:
            api_keys = load_api_keys()
            valid_keys = [entry["key"] for entry in api_keys.values() if isinstance(entry, dict) and "key" in entry]
            if x_api_key not in valid_keys:
                raise HTTPException(status_code=401, detail="Invalid API key")
            return x_api_key
        except Exception as e:
            logger.error(f"Error verifying API key: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error verifying API key: {str(e)}")
    return None

class CreateAPIKeyRequest(BaseModel):
    user_id: str
    description: Optional[str] = None

class MessageRequest(BaseModel):
    content: str

class SourceUrlRequest(BaseModel):
    url: str

class ChatCreateRequest(BaseModel):
    sourceId: str

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page."""
    with open(index_path, "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/api/v1/api-key/create", summary="Create a new API key")
async def create_api_key(request: CreateAPIKeyRequest):
    """Generate and store a new API key."""
    try:
        api_keys = load_api_keys()
        new_key = f"qpdf-{uuid.uuid4().hex}"
        api_keys[request.user_id] = {
            "key": new_key,
            "description": request.description or "Generated API key",
            "created_at": time.time()
        }
        save_api_keys(api_keys)
        logger.info(f"API key created for user {request.user_id}")
        return {"user_id": request.user_id, "api_key": new_key, "description": request.description}
    except Exception as e:
        logger.error(f"Error creating API key: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating API key: {str(e)}")

@app.get("/api/v1/health", summary="Check API health")
async def health_check(x_api_key: str = Header(None)):
    """Check if the API is running."""
    verify_api_key(x_api_key)
    return {"status": "healthy"}

@app.post("/api/v1/sources/add-file", summary="Upload and ingest a PDF file")
async def upload_pdf(file: UploadFile = File(...), x_api_key: str = Header(None)):
    """Upload a PDF file and return a sourceId."""
    verify_api_key(x_api_key)
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    source_id = str(uuid.uuid4())  # Dynamic sourceId
    file_path = os.path.join(STORAGE_PATH, f"{source_id}.pdf")
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
        await qpdf_instance.ingest(file_path, file.filename, source_id)
        return {"sourceId": source_id}
    except Exception as e:
        logger.error(f"Error ingesting {file.filename} (source_id: {source_id}): {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error ingesting file: {str(e)}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/api/v1/sources/add-url", summary="Upload and ingest a PDF from a URL")
async def upload_pdf_url(request: SourceUrlRequest, x_api_key: str = Header(None)):
    """Upload a PDF from a URL and return a sourceId."""
    verify_api_key(x_api_key)
    source_id = str(uuid.uuid4())  # Dynamic sourceId
    file_path = os.path.join(STORAGE_PATH, f"{source_id}.pdf")
    try:
        response = requests.get(request.url, timeout=10)
        response.raise_for_status()
        with open(file_path, "wb") as f:
            f.write(response.content)
        file_name = request.url.split("/")[-1] or f"downloaded_{source_id}.pdf"
        await qpdf_instance.ingest(file_path, file_name, source_id)
        return {"sourceId": source_id}
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading PDF from {request.url}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error downloading PDF: {str(e)}")
    except Exception as e:
        logger.error(f"Error ingesting PDF from {request.url} (source_id: {source_id}): {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error ingesting file: {str(e)}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/api/v1/chats", summary="Create a chat session")
async def create_chat(request: ChatCreateRequest, x_api_key: str = Header(None)):
    """Create a chat session for a sourceId."""
    verify_api_key(x_api_key)
    chat_id = str(uuid.uuid4())  # Dynamic chatId
    CHAT_SESSIONS[chat_id] = {"sourceId": request.sourceId, "messages": []}
    logger.info(f"Chat session created: {chat_id} for sourceId: {request.sourceId}")
    return {"chatId": chat_id}

@app.post("/api/v1/chats/{chatId}/messages", summary="Send a message to a chat session")
async def message_pdf(chatId: str, request: MessageRequest, x_api_key: str = Header(None)):
    """Send a message to a chat session."""
    verify_api_key(x_api_key)
    if chatId not in CHAT_SESSIONS:
        raise HTTPException(status_code=404, detail="Chat session not found")
    try:
        source_id = CHAT_SESSIONS[chatId]["sourceId"]
        answer, sources = qpdf_instance.ask(request.content, source_id)
        CHAT_SESSIONS[chatId]["messages"].append({
            "content": request.content,
            "response": answer,
            "sources": sources,
            "timestamp": time.time()
        })
        return {"message": request.content, "response": answer, "sources": sources}
    except Exception as e:
        logger.error(f"Error processing message in chat {chatId}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@app.delete("/api/v1/sources/{sourceId}", summary="Remove a document")
async def remove_document(sourceId: str, x_api_key: str = Header(None)):
    """Remove documents associated with a sourceId."""
    verify_api_key(x_api_key)
    try:
        qpdf_instance.remove_document(sourceId)
        for chat_id in list(CHAT_SESSIONS.keys()):
            if CHAT_SESSIONS[chat_id]["sourceId"] == sourceId:
                del CHAT_SESSIONS[chat_id]
        return {"message": f"Removed documents with sourceId {sourceId}"}
    except Exception as e:
        logger.error(f"Error removing sourceId {sourceId}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error removing file: {str(e)}")

@app.delete("/api/v1/cache", summary="Clear cache")
async def clear_cache(x_api_key: str = Header(None)):
    """Clear vector store and chat sessions."""
    verify_api_key(x_api_key)
    try:
        qpdf_instance.clear_cache()
        CHAT_SESSIONS.clear()
        return {"message": "Cache and chat sessions cleared"}
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}")

# === Chat History Management ===
def load_chat_history():
    """Load chat history from JSON file."""
    history_file = os.path.join(STORAGE_PATH, "chat_history.json")
    try:
        if os.path.exists(history_file):
            with open(history_file, "r") as f:
                return json.load(f)
        return []
    except Exception as e:
        logger.error(f"Error loading chat history: {str(e)}")
        return []

def save_chat_history(history):
    """Save chat history to JSON file."""
    history_file = os.path.join(STORAGE_PATH, "chat_history.json")
    try:
        with open(history_file, "w") as f:
            json.dump(history, f)
    except Exception as e:
        logger.error(f"Error saving chat history: {str(e)}")

def clear_chat_history():
    """Clear chat history and session state."""
    history_file = os.path.join(STORAGE_PATH, "chat_history.json")
    st.session_state["messages"] = []
    st.session_state["chat_history"] = []
    if os.path.exists(history_file):
        os.remove(history_file)
        logger.info(f"Chat history file removed: {history_file}")

# === Streamlit UI ===
def streamlit_app():
    st.set_page_config(page_title="QPDF", layout="wide")
    if "initialized" not in st.session_state:
        st.session_state.clear()
        st.session_state["messages"] = []
        st.session_state["assistant"] = QPDF()
        st.session_state["user_input"] = ""
        st.session_state["chat_history"] = load_chat_history()
        st.session_state["uploaded_files"] = []
        st.session_state["source_ids"] = {}
        st.session_state["chat_ids"] = {}
        st.session_state["initialized"] = True
        st.session_state["api_key"] = None  # Optional API key
        logger.info("Session state initialized")

    async def read_and_save_file():
        """Process uploaded files."""
        for file in st.session_state["file_uploader"]:
            file_name = file.name
            if file_name not in st.session_state["uploaded_files"]:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
                    tf.write(file.getbuffer())
                    path = tf.name
                source_id = str(uuid.uuid4())  # Dynamic sourceId
                progress_bar = st.progress(0)
                progress_text = st.empty()
                try:
                    await st.session_state["assistant"].ingest(path, file_name, source_id, progress_bar, progress_text)
                    st.session_state["messages"].append((f"Ingested {file_name} (sourceId: {source_id})", False))
                    st.session_state["chat_history"].append({"file": file_name, "source_id": source_id, "action": "ingested", "timestamp": time.time()})
                    st.session_state["uploaded_files"].append(file_name)
                    st.session_state["source_ids"][file_name] = source_id
                    st.session_state["chat_ids"][file_name] = None
                    save_chat_history(st.session_state["chat_history"])
                finally:
                    os.remove(path)

    def create_chat_session(file_name: str):
        """Create a chat session for a file."""
        source_id = st.session_state["source_ids"].get(file_name)
        if source_id:
            chat_id = str(uuid.uuid4())  # Dynamic chatId
            CHAT_SESSIONS[chat_id] = {"sourceId": source_id, "messages": []}
            st.session_state["chat_ids"][file_name] = chat_id
            st.session_state["chat_history"].append({
                "file": file_name,
                "source_id": source_id,
                "chat_id": chat_id,
                "action": "chat_created",
                "timestamp": time.time()
            })
            save_chat_history(st.session_state["chat_history"])
            st.success(f"Chat session created for {file_name}: {chat_id}")

    def remove_uploaded_file(file_name: str):
        """Remove an uploaded file."""
        source_id = st.session_state["source_ids"].get(file_name)
        if source_id:
            st.session_state["assistant"].remove_document(source_id)
            st.session_state["uploaded_files"] = [f for f in st.session_state["uploaded_files"] if f != file_name]
            st.session_state["messages"] = [(msg, is_user) for msg, is_user in st.session_state["messages"] if f"Ingested {file_name}" not in msg]
            st.session_state["chat_history"] = [entry for entry in st.session_state["chat_history"] if not ("file" in entry and entry["file"] == file_name)]
            st.session_state["source_ids"].pop(file_name, None)
            st.session_state["chat_ids"].pop(file_name, None)
            save_chat_history(st.session_state["chat_history"])
            st.rerun()

    def process_input():
        """Process user input."""
        user_text = st.session_state["user_input"].strip()
        if user_text:
            selected_file = st.session_state.get("selected_file")
            chat_id = st.session_state["chat_ids"].get(selected_file)
            if not chat_id:
                st.error("Please create a chat session for the selected file")
                return
            source_id = st.session_state["source_ids"].get(selected_file)
            answer, sources = st.session_state["assistant"].ask(user_text, source_id)
            st.session_state["messages"].append((user_text, True))
            st.session_state["messages"].append((answer, False))
            st.session_state["chat_history"].append({
                "user_message": user_text,
                "assistant_response": answer,
                "source_id": source_id,
                "chat_id": chat_id,
                "timestamp": time.time(),
                "feedback": None
            })
            save_chat_history(st.session_state["chat_history"])
            st.session_state["user_input"] = ""

    def handle_feedback(index: int, feedback: str):
        """Handle user feedback."""
        history_index = (index - 1) // 2
        if history_index < len(st.session_state["chat_history"]) and "user_message" in st.session_state["chat_history"][history_index]:
            st.session_state["chat_history"][history_index]["feedback"] = feedback
            save_chat_history(st.session_state["chat_history"])
            if feedback == "Good":
                st.session_state["assistant"].refine_vector_store(
                    message=st.session_state["chat_history"][history_index]["user_message"],
                    response=st.session_state["chat_history"][history_index]["assistant_response"],
                    feedback=feedback
                )

    def create_api_key():
        """Create an API key (optional feature)."""
        user_id = st.session_state.get("api_user_id", "").strip()
        description = st.session_state.get("api_description", "").strip()
        if not user_id:
            st.error("Please provide a User ID")
            return
        try:
            response = requests.post(
                "http://localhost:8000/api/v1/api-key/create",
                json={"user_id": user_id, "description": description or None},
                timeout=5
            )
            response.raise_for_status()
            data = response.json()
            st.session_state["api_key"] = data["api_key"]
            st.success(f"API Key created: {data['api_key']}")
            logger.info(f"API key created for user {user_id}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error creating API key: {str(e)}. Ensure FastAPI server is running on port 8000")
            logger.error(f"Error creating API key: {str(e)}")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.header("QPDF: Chat with PDFs")
        st.file_uploader("Upload PDF", type=["pdf"], key="file_uploader", on_change=lambda: asyncio.run(read_and_save_file()), accept_multiple_files=True)
        st.subheader("API Key Management (Optional)")
        if st.button("Generate API Key"):
            st.session_state["show_api_form"] = True
        if st.session_state.get("show_api_form", False):
            st.text_input("User ID", key="api_user_id")
            st.text_input("Description (Optional)", key="api_description")
            if st.button("Create API Key"):
                create_api_key()
        if st.session_state["uploaded_files"]:
            st.subheader("Uploaded Files")
            for file_name in st.session_state["uploaded_files"]:
                col_file = st.columns([3, 1, 1])
                with col_file[0]:
                    st.write(f"{file_name} (sourceId: {st.session_state['source_ids'].get(file_name)})")
                with col_file[1]:
                    if st.button("Remove", key=f"remove_{file_name}"):
                        remove_uploaded_file(file_name)
                with col_file[2]:
                    if st.button("Start Chat", key=f"chat_{file_name}"):
                        create_chat_session(file_name)
            st.selectbox("Select a file to message", options=["Select a file"] + st.session_state["uploaded_files"], key="selected_file")
        for i, (msg, is_user) in enumerate(st.session_state["messages"]):
            message(msg, is_user=is_user, key=str(i))
            if not is_user and "Ingested" not in msg:
                col_feedback = st.columns([1, 1])
                with col_feedback[0]:
                    if st.button(" Good", key=f"good_{i}"):
                        handle_feedback(i, "Good")
                with col_feedback[1]:
                    if st.button(" Bad", key=f"bad_{i}"):
                        handle_feedback(i, "Bad")
        st.text_input("Send a message", key="user_input", on_change=process_input)

    with col2:
        st.header("Chat History")
        if st.button("Clear History"):
            clear_chat_history()
            st.rerun()
        if st.button("Clear Cache"):
            st.session_state["assistant"].clear_cache()
            CHAT_SESSIONS.clear()
            st.rerun()
        for entry in st.session_state["chat_history"]:
            with st.expander(f"{'File' if 'file' in entry else 'Message'} at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(entry['timestamp']))}"):
                if "file" in entry:
                    st.write(f"**File**: {entry['file']} (sourceId: {entry.get('source_id')})")
                    if "chat_id" in entry:
                        st.write(f"**Chat ID**: {entry['chat_id']}")
                else:
                    st.write(f"**User**: {entry.get('user_message')}")
                    st.write(f"**Assistant**: {entry['assistant_response']}")
                    if entry.get("source_id"):
                        st.write(f"**Source ID**: {entry['source_id']}")
                    if entry.get("chat_id"):
                        st.write(f"**Chat ID**: {entry['chat_id']}")
                    if entry.get("feedback"):
                        st.write(f"**Feedback**: {entry['feedback']}")

# === Run FastAPI ===
def run_fastapi():
    """Run the FastAPI server."""
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    except Exception as e:
        logger.error(f"Error starting FastAPI server: {str(e)}")

# === Entry Point ===
if __name__ == "__main__":
    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()
    time.sleep(2)
    streamlit_app()
