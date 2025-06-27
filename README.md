QPDF: Chat with PDFs
QPDF is a FastAPI and Streamlit-based application designed to process and interact with PDF documents. It allows users to upload PDFs, extract text (including from scanned and tabular PDFs), and ask questions about the content using a conversational AI interface powered by LangChain and a local LLM (e.g., LLaMA 3.1 or Qwen2.5). The application supports OCR, table extraction, and vector-based document retrieval, with a user-friendly web interface and a RESTful API for programmatic access.
Features

PDF Ingestion: Upload PDFs via file upload or URL, with support for text, scanned, and tabular PDFs.
Text Extraction: Extracts text using PyPDFLoader, with fallback to OCR (via Tesseract) for scanned PDFs and table extraction (via Camelot) for tabular data.
Conversational AI: Ask questions about PDFs using a local LLM (e.g., LLaMA 3.1 or Qwen2.5) with LangChain for context-aware responses.
Vector Store: Uses Chroma for efficient document retrieval and storage of extracted content.
API Support: FastAPI-based RESTful API for uploading PDFs, creating chat sessions, and querying documents.
Streamlit UI: Interactive web interface for uploading files, managing chat sessions, and providing feedback.
Feedback Mechanism: Users can rate responses to refine the vector store with high-quality answers.
API Key Management: Optional API key generation for secure access to the API.

Prerequisites

Python 3.8+
Required Python packages (listed in requirements.txt):
fastapi
uvicorn
streamlit
pdfplumber
pytesseract
pdf2image
camelot-py
langchain
langchain-community
chromadb
huggingface_hub
requests
opencv-python
pillow
numpy


Tesseract OCR installed on your system (see Tesseract installation guide).
Poppler-utils for PDF-to-image conversion (required by pdf2image).
Ghostscript (optional, for some PDF processing tasks with Camelot).

Installation

Clone the Repository:
git clone <repository-url>
cd qpdf


Create a Virtual Environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt


Install Tesseract and Poppler:

For Tesseract, follow the installation instructions for your OS (e.g., apt-get install tesseract-ocr on Ubuntu, or download from here).
For Poppler, install poppler-utils (e.g., apt-get install poppler-utils on Ubuntu, or download from here).
Ensure Tesseract and Poppler binaries are in your system PATH.


Set Environment Variables (optional):

Set QPDF_STORAGE_PATH to specify the storage directory for PDFs and vector store (defaults to C:\Users\AIPC\Desktop\qpdf_data on Windows).

export QPDF_STORAGE_PATH="/path/to/storage"



Usage

Run the Application:
python app.py

This starts both the FastAPI server (on http://localhost:8000) and the Streamlit UI (on http://localhost:8501).

Access the Streamlit UI:

Open http://localhost:8501 in your browser.
Upload a PDF file or provide a URL to a PDF.
Create a chat session for the uploaded file.
Ask questions about the PDF content and provide feedback on responses.


Access the API:

Explore the API at http://localhost:8000/docs.
Generate an API key using the /api/v1/api-key/create endpoint.
Use endpoints like /api/v1/sources/add-file, /api/v1/sources/add-url, and /api/v1/chats/{chatId}/messages to interact programmatically.


Example API Usage:
# Generate API key
curl -X POST "http://localhost:8000/api/v1/api-key/create" -H "Content-Type: application/json" -d '{"user_id": "test_user", "description": "Test API key"}'

# Upload a PDF
curl -X POST "http://localhost:8000/api/v1/sources/add-file" -H "X-API-Key: <your-api-key>" -F "file=@/path/to/your/file.pdf"

# Create a chat session
curl -X POST "http://localhost:8000/api/v1/chats" -H "X-API-Key: <your-api-key>" -H "Content-Type: application/json" -d '{"sourceId": "<source-id>"}'

# Send a message
curl -X POST "http://localhost:8000/api/v1/chats/<chat-id>/messages" -H "X-API-Key: <your-api-key>" -H "Content-Type: application/json" -d '{"content": "What is the main topic of the PDF?"}'



Directory Structure

app.py: Main application script containing FastAPI and Streamlit logic.
static/: Directory for serving static files (e.g., index.html).
qpdf_data/: Storage directory for PDFs, vector store (chroma_db), API keys (api_keys.json), and chat history (chat_history.json).

Configuration

Storage Path: Set via QPDF_STORAGE_PATH environment variable or defaults to C:\Users\AIPC\Desktop\qpdf_data on Windows.
LLM Model: Defaults to llama3.1, with fallback to qwen2.5 if unavailable. Configure via the QPDF class initialization.
Vector Store: Uses Chroma with all-MiniLM-L6-v2 embeddings from HuggingFace.
Logging: Configured to log at INFO level to console and stored in the storage directory.

Notes

The application uses a local LLM (via ChatOllama). Ensure Ollama is installed and the desired model is pulled (e.g., ollama pull llama3.1).
Scanned PDFs require Tesseract and Poppler for OCR and image conversion.
Table extraction uses Camelot with the stream flavor for better performance on complex tables.
The API is optional and can be secured with API keys. For local use, API key verification can be skipped.
Feedback on responses ("Good" or "Bad") refines the vector store for better future responses.

Troubleshooting

Tesseract/Poppler Not Found: Ensure Tesseract and Poppler binaries are installed and added to your system PATH.
PDF Extraction Fails: Check if the PDF is valid and not encrypted. For scanned PDFs, ensure sufficient disk space for image conversion.
FastAPI/Streamlit Not Starting: Verify port 8000 (FastAPI) and 8501 (Streamlit) are free.
LLM Model Issues: Ensure Ollama is running and the specified model is available (ollama list).

Contributing
Contributions are welcome! Please submit a pull request or open an issue on the repository for bugs, feature requests, or improvements.
License
This project is licensed under the MIT License. See the LICENSE file for details.
