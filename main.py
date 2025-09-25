import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.routes import router as api_router

# -------------------------------
# FastAPI app Setup
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
# Directories and Static Files
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# Mount the static folder
app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static"
)

# -------------------------------
# Include API Routes
# -------------------------------
# All of the application's routes (/upload, /ask, etc.) are now handled
# by the router defined in app/routes.py.
app.include_router(api_router)

# Note: No more route definitions (@app.get, @app.post) in this file.
# Note: No more in-memory storage (uploaded_docs, chat_history) in this file.