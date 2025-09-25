# app/utils.py
import os
import uuid
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client.http import models

# Assuming these are in other files and imported correctly
from .embeddings import embed_text
from .qdrant_client import qdrant, COLLECTION_NAME

# --- Define Directories Relative to this file ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEXT_DIR = os.path.join(BASE_DIR, "../texts") # Assuming 'texts' is one level up
CHUNK_DIR = os.path.join(BASE_DIR, "../chunks") # No longer needed, but kept for context

os.makedirs(TEXT_DIR, exist_ok=True)
# os.makedirs(CHUNK_DIR, exist_ok=True) # We won't be saving chunks to a file anymore


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts raw text content from a PDF file.
    Returns: A single string with all the text.
    """
    text_content = ""
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            page_text = page.extract_text()
            if page_text:
                # Add a page marker for context
                text_content += f"\n\n--- Page {i} ---\n\n{page_text}"
    return text_content


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> list[str]:
    """
    Splits a long text into smaller chunks.
    Returns: A list of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_text(text)
    return chunks


def generate_embeddings_and_store(chunks: list[str], doc_id: str):
    """
    Generate embeddings for each chunk and store in Qdrant.
    This function now takes a list of chunks directly.
    """
    # Get embeddings for the chunks
    embeddings = embed_text(chunks)

    # Prepare points for Qdrant
    points = []
    for chunk, emb in zip(chunks, embeddings):
        # You could parse the '--- Page X ---' marker here to get page number
        page_number = None
        if "--- Page " in chunk:
            try:
                # Simple parsing, can be made more robust
                page_number = int(chunk.split('---')[1].split('Page')[1].strip())
            except (ValueError, IndexError):
                page_number = None
        
        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=emb,
                payload={"text": chunk, "doc_id": doc_id, "page": page_number}
            )
        )

    if not points:
        return # Nothing to upsert

    # Upsert points to Qdrant
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True # Ensures the operation is completed before returning
    )