# main.py
import os
import uuid
import asyncio
import json
from fastapi import FastAPI, Request, UploadFile, Body
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Make sure these imports are correct and the files exist
from app.utils import extract_text_to_file, chunk_text_and_save, generate_embeddings_and_store
from app.qdrant_client import search_qdrant_for_doc
from app.genai_client import answer_with_groq_async
from app.routes import router as routes

# -------------------------------
# FastAPI app
# -------------------------------
app = FastAPI(
    title="PDF Research Assistant",
    description="Upload PDFs and ask questions based on PDF content",
    version="1.0.0"
)

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Directories
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- FIX START ---

# Define the static directory relative to the current file
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Create the directory if it doesn't exist to prevent errors
os.makedirs(STATIC_DIR, exist_ok=True)

# Mount the static folder using the relative path
app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static"
)

# Templates directory, also using a relative path
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# PDF uploads folder
UPLOAD_DIR = os.path.join(BASE_DIR, "uploaded_pdfs")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- FIX END ---

# In-memory storage
uploaded_docs = {}
chat_history = {}

# Include routes
app.include_router(routes)

# -------------------------------
# ROUTES
# -------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve main page"""
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "uploaded_docs": uploaded_docs}
    )


@app.post("/upload/")
async def upload_pdf(file: UploadFile = None):
    if not file:
        return JSONResponse({"error": "No file uploaded"}, status_code=400)
    try:
        filename = file.filename
        doc_id = str(uuid.uuid4())
        saved_path = os.path.join(UPLOAD_DIR, f"{doc_id}_{filename}")
        with open(saved_path, "wb") as f:
            f.write(await file.read())

        extracted_path = extract_text_to_file(saved_path, doc_id)
        chunks_path, num_chunks = chunk_text_and_save(extracted_path, doc_id)
        generate_embeddings_and_store(chunks_path, doc_id)

        uploaded_docs[doc_id] = {"filename": filename, "path": saved_path}
        chat_history[doc_id] = []

        return JSONResponse({"id": doc_id, "filename": filename, "chunks": num_chunks})
    except Exception as e:
        return JSONResponse({"error": f"Error processing file: {e}"}, status_code=500)


@app.post("/ask/")
async def ask_question(payload: dict = Body(...)):
    query = payload.get("question")
    doc_id = payload.get("doc_id")

    if not query or not doc_id:
        return JSONResponse({"error": "Missing question or doc_id"}, status_code=400)
    if doc_id not in uploaded_docs:
        return JSONResponse({"error": "Selected document not found"}, status_code=404)

    try:
        results = search_qdrant_for_doc(query, doc_id, top_k=10) or []
        context_chunks = [
            {"text": r.payload.get("text", ""), "page": r.payload.get("page", None)}
            for r in results if r.payload.get("text")
        ]
        prompt_chunks = [c["text"] for c in context_chunks]

        prompt = (
            f"Use the following context from a PDF to answer the question.\n\n"
            f"Context:\n{'\n\n'.join(prompt_chunks)}\n\n"
            f"Question: {query}\n"
            f"Answer based only on the context provided. "
            f"If the answer is not present, respond 'Not available in the document.'"
        ) if prompt_chunks else query

        answer = await answer_with_groq_async(prompt)
        chat_history.setdefault(doc_id, []).append({"question": query, "answer": answer})

        # PDF page metadata — include ALL pages
        import pdfplumber
        pdf_path = uploaded_docs[doc_id]["path"]
        try:
            with pdfplumber.open(pdf_path) as pdf:
                all_pages = list(range(1, len(pdf.pages) + 1))
        except Exception:
            all_pages = []

        metadata = {
            "filename": uploaded_docs[doc_id]["filename"],
            "pages": all_pages,
            "used_pages": sorted(list(set([c["page"] for c in context_chunks if c["page"] is not None])))
        }

        return JSONResponse({
            "query": query,
            "answer": answer,
            "doc_id": doc_id,
            "history": chat_history.get(doc_id, []),
            "context": context_chunks,
            "metadata": metadata
        })

    except Exception as e:
        return JSONResponse({"error": f"Error generating answer: {e}"}, status_code=500)


@app.post("/ask/stream/")
async def ask_question_stream(payload: dict = Body(...)):
    query = payload.get("question")
    doc_id = payload.get("doc_id")

    if not query or not doc_id:
        return JSONResponse({"error": "Missing question or doc_id"}, status_code=400)
    if doc_id not in uploaded_docs:
        return JSONResponse({"error": "Selected document not found"}, status_code=404)

    results = search_qdrant_for_doc(query, doc_id, top_k=10) or []
    context_chunks = [
        {"text": r.payload.get("text", ""), "page": r.payload.get("page", None)}
        for r in results if r.payload.get("text")
    ]
    prompt_chunks = [c["text"] for c in context_chunks]
    prompt = (
        f"Use the following context from a PDF to answer the question.\n\n"
        f"Context:\n{'\n\n'.join(prompt_chunks)}\n\n"
        f"Question: {query}\n"
        f"Answer based only on the context provided. "
        f"If the answer is not present, respond 'Not available in the document.'"
    ) if prompt_chunks else query

    async def answer_generator():
        try:
            pdf_path = uploaded_docs[doc_id]["path"]
            import pdfplumber
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    all_pages = list(range(1, len(pdf.pages) + 1))
            except Exception:
                all_pages = []

            metadata = {
                "filename": uploaded_docs[doc_id]["filename"],
                "pages": all_pages,
                "used_pages": sorted(list(set([c["page"] for c in context_chunks if c["page"] is not None])))
            }

            full_answer = ""
            async for chunk in await answer_with_groq_async(prompt, stream=True): # Assuming your client supports streaming
                full_answer += chunk
                yield chunk
                await asyncio.sleep(0.01)

            if not full_answer.strip():
                final_text = "⚠️ No relevant content found." if not context_chunks else "⚠️ Unable to generate answer from the context."
                yield final_text
                full_answer = final_text


            yield f"[META]{json.dumps(metadata)}"

            chat_history.setdefault(doc_id, []).append({
                "question": query,
                "answer": full_answer,
                "sources": context_chunks # It's better to save the processed context
            })

        except Exception as e:
            yield f"⚠️ Error: {e}"

    return StreamingResponse(answer_generator(), media_type="text/event-stream")