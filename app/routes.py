from fastapi import APIRouter, Request, UploadFile, Body
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import os
import asyncio
import uuid
import json
import traceback

from app.utils import extract_text_to_file, chunk_text_and_save, generate_embeddings_and_store
from app.qdrant_client import search_qdrant_for_doc
from app.genai_client import answer_with_groq_async

router = APIRouter()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Templates
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "../templates"))

# Directories
UPLOAD_DIR = os.path.join(BASE_DIR, "../uploaded_pdfs")
TEXT_DIR = os.path.join(BASE_DIR, "../texts")
CHUNK_DIR = os.path.join(BASE_DIR, "../chunks")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)
os.makedirs(CHUNK_DIR, exist_ok=True)

# Memory storage
uploaded_docs = {}
chat_history = {}

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "uploaded_docs": uploaded_docs}
    )

@router.post("/upload/")
async def upload_pdf(file: UploadFile = None):
    if not file:
        return JSONResponse({"error": "No file uploaded"}, status_code=400)
    try:
        filename = file.filename
        uid = str(uuid.uuid4())
        saved_path = os.path.join(UPLOAD_DIR, f"{uid}_{filename}")

        with open(saved_path, "wb") as f:
            f.write(await file.read())

        extracted_path = extract_text_to_file(saved_path, uid)
        chunks_path, num_chunks = chunk_text_and_save(extracted_path, uid)
        generate_embeddings_and_store(chunks_path, uid)

        uploaded_docs[uid] = {"filename": filename, "path": saved_path}
        chat_history[uid] = []

        return JSONResponse({"id": uid, "filename": filename, "chunks": num_chunks})
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": f"Error processing file: {e}"}, status_code=500)

@router.post("/ask/")
async def ask_question(payload: dict = Body(...)):
    query = payload.get("question")
    doc_id = payload.get("doc_id")
    if not query or not doc_id:
        return JSONResponse({"error": "Missing question or doc_id"}, status_code=400)
    if doc_id not in uploaded_docs:
        return JSONResponse({"error": "Selected document not found"}, status_code=404)

    try:
        results = search_qdrant_for_doc(query, doc_id, top_k=10) or []
        context_chunks = [r.payload.get("text", "") for r in results if r.payload.get("text")]

        prompt = (
            f"Use the following context from a PDF to answer the question.\n\n"
            f"Context:\n{'\n\n'.join(context_chunks)}\n\n"
            f"Question: {query}\n"
            f"Answer based only on the context provided. If the answer is not present, respond 'Not available in the document.'"
        ) if context_chunks else query

        answer = await answer_with_groq_async(prompt)
        chat_history.setdefault(doc_id, []).append({"question": query, "answer": answer})

        return JSONResponse({
            "query": query,
            "answer": answer,
            "doc_id": doc_id,
            "history": chat_history.get(doc_id, []),
            "context": context_chunks,
        })
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": f"Error generating answer: {e}"}, status_code=500)

@router.post("/ask/stream/")
async def ask_question_stream(payload: dict = Body(...)):
    query = payload.get("question")
    doc_id = payload.get("doc_id")
    if not query or not doc_id:
        return JSONResponse({"error": "Missing question or doc_id"}, status_code=400)
    if doc_id not in uploaded_docs:
        return JSONResponse({"error": "Selected document not found"}, status_code=404)

    results = search_qdrant_for_doc(query, doc_id, top_k=10) or []
    context_chunks = [r.payload.get("text", "") for r in results if r.payload.get("text")]

    prompt = (
        f"Use the following context from a PDF to answer the question.\n\n"
        f"Context:\n{'\n\n'.join(context_chunks)}\n\n"
        f"Question: {query}\n"
        f"Answer based only on the context provided. If the answer is not present, respond 'Not available in the document.'"
    ) if context_chunks else query

    async def answer_generator():
        try:
            answer = await answer_with_groq_async(prompt)
            if not answer or not answer.strip():
                answer = "⚠️ No relevant content found." if not context_chunks else "⚠️ Unable to generate answer from the context."

            for i in range(0, len(answer), 20):
                yield answer[i:i+20]
                await asyncio.sleep(0.02)

            chat_history.setdefault(doc_id, []).append({
                "question": query,
                "answer": answer,
                "sources": results
            })

            meta = {
                "filename": uploaded_docs[doc_id]["filename"],
                "pages": list({r.payload.get("page_number") for r in results if r.payload.get("page_number") is not None})
            }
            yield f"[META]{json.dumps(meta)}"

        except Exception as e:
            yield f"⚠️ Error: {e}"

    return StreamingResponse(answer_generator(), media_type="text/plain")
