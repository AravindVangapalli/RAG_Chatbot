# RAG_Chatbot
A local Retrieval-Augmented Generation (RAG) chatbot built with PDF parsing, OCR (via Tesseract), semantic search using FAISS, and a custom LLM backend. This project extracts information from PDF documents (including scanned images) and answers questions based on that content.

**Features**
Supports both text-based and image-based PDFs (with OCR)
Embeds document content using SentenceTransformer
Performs semantic search with FAISS
Interfaces with a local LLM API for grounded answers
Maintains processed document tracking to avoid reprocessing
Answers in bullet-point format when possible

**Requirements**
Python 3.10+
Tesseract OCR installed
Local LLM API running (compatible with OpenAI-style format)

**Dependencies Install with:**
pip install -r requirements.txt

**Configuration**
Edit these values in main.py: It depends on youe model and your local file paths
DOCS_DIR = r"C:\\Users\\HP\\OneDrive\\Desktop\\Chatbot\\Documents"
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
LM_API_URL = "http://127.0.0.1:1234/v1/chat/completions"
EMBED_MODEL = "intfloat/e5-base-v2"
LLM_MODEL = "deepseek-r1-distill-qwen-7b"

**Working**
1. Document Embedding
  Extracts text and images from each PDF.
  OCR is applied on image regions.
  Text is chunked, embedded, and indexed using FAISS.
2. User Query
  Accepts a natural language question.
  Finds top-matching chunks from the indexed document embeddings.
  Constructs a prompt with relevant context and sends it to the LLM.
  Displays the LLM’s final response after parsing.

**Note**
OCR quality depends on Tesseract setup and image clarity.
The chatbot only answers based on document content. If the answer isn't found, it replies with:"I don't know"




















