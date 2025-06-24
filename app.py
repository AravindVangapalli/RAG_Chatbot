# === RAG Chatbot with Fixes Applied ===
import os
import re
import fitz
import json
import faiss
import pickle
import pytesseract
import requests
import hashlib
import numpy as np
from PIL import Image
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

# === CONFIGURATION ===

DOCS_DIR = r"C:\\Users\\HP\\OneDrive\\Desktop\\Chatbot\\Documents"

pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

EMBED_DIR = "embeddings"
INDEX_PATH = os.path.join(EMBED_DIR, "vector.index")
META_PATH = os.path.join(EMBED_DIR, "metadata.pkl")
TRACK_PATH = os.path.join(EMBED_DIR, "processed.json")
TEMP_IMG_DIR = "temp_images"

LM_API_URL = "http://127.0.0.1:1234/v1/chat/completions"
EMBED_MODEL = "intfloat/e5-base-v2"
LLM_MODEL = "deepseek-r1-distill-qwen-7b"
TOP_K = 3

os.makedirs(EMBED_DIR, exist_ok=True)
os.makedirs(TEMP_IMG_DIR, exist_ok=True)

embedder = SentenceTransformer(EMBED_MODEL)

# === UTILITIES ===

def hash_file(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def extract_text_and_images(pdf_path):
    text_data = []
    image_texts = []

    reader = PdfReader(pdf_path)
    for page in reader.pages:
        if page.extract_text():
            text_data.append(page.extract_text())

    doc = fitz.open(pdf_path)
    for i in range(len(doc)):
        images = doc[i].get_images(full=True)
        for j, img in enumerate(images):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.n > 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            img_path = os.path.join(TEMP_IMG_DIR, f"{os.path.basename(pdf_path)}_page{i}_img{j}.png")
            pix.save(img_path)
            try:
                ocr_text = pytesseract.image_to_string(Image.open(img_path))
                if ocr_text.strip():
                    image_texts.append(ocr_text)
            except Exception as e:
                print(f"[!] OCR failed for {img_path}: {e}")

    return "\n".join(text_data + image_texts)

def chunk_text(text, max_chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_chunk_size - overlap):
        chunk = " ".join(words[i:i + max_chunk_size])
        chunks.append(chunk)
    return chunks

def load_index():
    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "rb") as f:
            metadata = pickle.load(f)
    else:
        dim = embedder.get_sentence_embedding_dimension()
        index = faiss.IndexFlatL2(dim)
        metadata = []
    return index, metadata

def save_index(index, metadata):
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)

def load_processed():
    if os.path.exists(TRACK_PATH):
        with open(TRACK_PATH, "r") as f:
            return json.load(f)
    return {}

def save_processed(data):
    with open(TRACK_PATH, "w") as f:
        json.dump(data, f)

def extract_final_answer(ai_response: str) -> str:
    answer_match = re.search(r'Answer:\s*(.+)', ai_response, re.IGNORECASE | re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    response_without_think = re.sub(r'<think>.*?</think>', '', ai_response, flags=re.DOTALL | re.IGNORECASE)
    return response_without_think.strip()

# === STEP 1: EMBEDDING ===

def embed_new_documents():
    index, metadata = load_index()
    processed = load_processed()

    for filename in os.listdir(DOCS_DIR):
        if not filename.lower().endswith(".pdf"):
            continue

        path = os.path.join(DOCS_DIR, filename)
        file_hash = hash_file(path)

        if file_hash in processed:
            continue

        print(f"[+] Embedding new document: {filename}")
        content = extract_text_and_images(path)
        if not content.strip():
            continue

        chunks = chunk_text(content)
        if not chunks:
            continue

        print(f"Total Chunks: {len(chunks)}")
        embeddings = embedder.encode(["passage: " + c for c in chunks])
        index.add(np.array(embeddings))
        metadata.extend(chunks)
        processed[file_hash] = filename

    save_index(index, metadata)
    save_processed(processed)
    print("[✓] Embedding complete.\n")

# === STEP 2: QUERY INTERFACE ===

def search_chunks(query, index, metadata, top_k=TOP_K):
    query_emb = embedder.encode(["query: " + query])
    D, I = index.search(np.array(query_emb), top_k)
    return [metadata[i] for i in I[0] if i < len(metadata)]

def build_prompt(context, question):
    return f"""
You are a helpful assistant. Answer the question strictly using ONLY the context below.
If the context does not contain the answer, reply exactly with: "I don't know".

Answer in well-structured, bullet-point format if possible.
Context:
{context}

Question:
{question}

Answer:
"""

def ask_llm(prompt):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {
            "role": "system", "content": "You are a strict assistant. Only answer based on the given context. If the context does not contain the answer, reply with: 'I don't know'. Never use external knowledge."
            },
            {
            "role": "user", "content": prompt
            }
        ],
        "temperature": 0
    }
    try:
        response = requests.post(LM_API_URL, json=payload, headers=headers)
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[LLM Error] {e}"

def start_chat():
    print("\nHii I am an AI Assistant. Ask a question or type 'exit'.\n")
    index, metadata = load_index()

    while True:
        user_input = input("Ask question : ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        context_chunks = search_chunks(user_input, index, metadata)
        context = "\n".join(context_chunks)
        #print("\nRetrieved Context:\n", context)
        prompt = build_prompt(context, user_input)
        #print("\nPrompt Sent to LLM:\n", prompt)
        raw_response = ask_llm(prompt)
        final_answer = extract_final_answer(raw_response)
        print("\nAnswer:\n", final_answer, "\n")

# === MAIN ===

if __name__ == "__main__":
    embed_new_documents()
    start_chat()
