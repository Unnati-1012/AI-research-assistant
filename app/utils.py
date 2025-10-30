# app/utils.py

import os
import uuid
import base64
from io import BytesIO
import pdfplumber
from groq import Groq
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
    # Default threshold: if extracted characters < threshold -> use VLM (image OCR)
    return extract_text_from_pdf_with_vlm(file_path)


def _call_groq_vlm_with_image_bytes(image_bytes: bytes, model: str = "meta-llama/llama-4-scout-17b-16e-instruct") -> str:
    """Send image bytes to Groq VLM and return text result.

    Expects GROQ_API_KEY in environment.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set in environment")

    client = Groq(api_key=api_key)

    # encode bytes to base64 data URI
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{b64}"

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the contents of this page and extract any readable text."},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ],
        model=model,
    )

    # The Groq response content can be a string or a list of message pieces
    content = chat_completion.choices[0].message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def extract_text_from_pdf_with_vlm(file_path: str, threshold: int = 200, vlm_model: str = "meta-llama/llama-4-scout-17b-16e-instruct") -> str:
    """
    Extract text from PDF page-by-page. If the page's extracted text length is below
    `threshold`, treat it as scanned/poor text and send the page image to Groq VLM.

    Returns a single string with page markers: '--- Page N ---\n<page text>\n'.
    """
    all_text = ""
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            if len((page_text or "").strip()) >= threshold:
                chosen_text = page_text
            else:
                # Render page to image and send to VLM
                try:
                    pil_img = page.to_image(resolution=150).original.convert("RGB")
                    buf = BytesIO()
                    pil_img.save(buf, format="JPEG", quality=85)
                    image_bytes = buf.getvalue()
                    vlm_text = _call_groq_vlm_with_image_bytes(image_bytes, model=vlm_model)
                    chosen_text = vlm_text
                except Exception as e:
                    # If image->VLM fails, fall back to whatever text we have (maybe empty)
                    chosen_text = page_text or f"[unreadable page {i}: error {e}]"

            # Add page marker at the start of each page
            all_text += f"--- Page {i} ---\n{chosen_text}\n"

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
