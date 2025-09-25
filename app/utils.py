import os
import uuid
import pdfplumber
from fastapi import UploadFile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .embeddings import embed_text
from .qdrant_client import qdrant, COLLECTION_NAME
from qdrant_client.http import models

import os
import uuid
import pdfplumber

# ---------------- PDF extraction ----------------
def extract_text_to_file(pdf_path, doc_id):
    extracted_file = os.path.join(os.path.dirname(pdf_path), f"{doc_id}_extracted.txt")
    with pdfplumber.open(pdf_path) as pdf, open(extracted_file, "w", encoding="utf-8") as f:
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if text:
                f.write(f"[PAGE_{i}]\n{text}\n")
    return extracted_file

# ---------------- Chunking ----------------
def chunk_text_and_save(extracted_path, doc_id, chunk_size=500):
    chunks_file = os.path.join(os.path.dirname(extracted_path), f"{doc_id}_chunks.txt")
    with open(extracted_path, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    with open(chunks_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk + "\n")
    return chunks_file, len(chunks)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "../uploads")
TEXT_DIR = os.path.join(os.path.dirname(__file__), "../texts")
CHUNK_DIR = os.path.join(os.path.dirname(__file__), "../chunks")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)
os.makedirs(CHUNK_DIR, exist_ok=True)


async def save_pdf_file(file: UploadFile):
    """
    Save uploaded PDF to disk.
    Returns: (uid, filename, saved_path)
    """
    uid = str(uuid.uuid4())
    filename = file.filename
    saved_path = os.path.join(UPLOAD_DIR, f"{uid}_{filename}")

    with open(saved_path, "wb") as f:
        f.write(await file.read())

    return uid, filename, saved_path


def extract_text_to_file(pdf_path: str, uid: str) -> str:
    """
    Extract text from PDF and save as a .txt file.
    """
    text_output_path = os.path.join(TEXT_DIR, f"{uid}.txt")
    text_content = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_content += page_text + "\n"

    with open(text_output_path, "w", encoding="utf-8") as f:
        f.write(text_content)

    return text_output_path


def chunk_text_to_file(text_path: str, uid: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Split text into chunks and save each chunk to a file.
    Returns: (chunks_path, num_chunks)
    """
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(text)

    chunks_path = os.path.join(CHUNK_DIR, f"{uid}_chunks.txt")
    with open(chunks_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(f"{chunk}\n")

    return chunks_path, len(chunks)


def generate_embeddings_and_store(chunks_path: str, doc_id: str):
    """
    Generate embeddings for each chunk and store in Qdrant using UUIDs for point IDs.
    """
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = [line.strip() for line in f if line.strip()]

    embeddings = embed_text(chunks)

    points = []
    for chunk, emb in zip(chunks, embeddings):
        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),  # Use valid UUID for Qdrant
                vector=emb,
                payload={"text": chunk, "doc_id": doc_id}
            )
        )

    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )


# Alias for backward compatibility
chunk_text_and_save = chunk_text_to_file
