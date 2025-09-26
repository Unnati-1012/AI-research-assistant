# app/utils.py

import os
import uuid
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client.http import models

# Import your embeddings function and Qdrant client
from .embeddings import embed_text
from .qdrant_client import qdrant, COLLECTION_NAME

# --- Define Directories Relative to this file ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEXT_DIR = os.path.join(BASE_DIR, "text")
AI_DIR = os.path.join(BASE_DIR, "AI")

# Ensure directories exist
os.makedirs(TEXT_DIR, exist_ok=True)
os.makedirs(AI_DIR, exist_ok=True)


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from PDF and add a page marker for each page.
    """
    all_text = ""
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            # Add page marker at the start of each page
            all_text += f"--- Page {i} ---\n{page_text}\n"
    return all_text


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> list[str]:
    """
    Split text into manageable chunks for embeddings.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_text(text)
    return chunks


def generate_embeddings_and_store(chunks: list[str], doc_id: str):
    """
    Generate embeddings for each chunk and store in Qdrant.
    Keeps correct page numbers from the PDF.
    """
    # Get embeddings for the chunks
    embeddings = embed_text(chunks)

    points = []
    for chunk, emb in zip(chunks, embeddings):
        page_number = None

        # Look for the last seen "--- Page X ---" marker inside the chunk
        if "--- Page " in chunk:
            try:
                marker = chunk.split("--- Page ")[-1].split("---")[0].strip()
                page_number = int(marker)
            except (ValueError, IndexError):
                page_number = None

        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=emb,
                payload={
                    "text": chunk,
                    "doc_id": doc_id,
                    "page": page_number,
                },
            )
        )

    if not points:
        return

    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True,
    )


def save_text_to_file(text: str, filename: str):
    """
    Save text to a file in the TEXT_DIR.
    """
    path = os.path.join(TEXT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path
