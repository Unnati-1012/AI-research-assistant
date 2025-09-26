import os
import logging
from logging.config import dictConfig
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.routes import router as api_router

# -------------------------------
# Logging Configuration (File Only)
# -------------------------------
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.log")

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        },
    },
    "handlers": {
        "file": {
            "class": "logging.FileHandler",
            "formatter": "default",
            "level": "DEBUG",
            "filename": LOG_FILE,
            "encoding": "utf-8",
        },
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["file"],
    },
    "loggers": {
        "uvicorn": {"handlers": ["file"], "level": "INFO", "propagate": False},
        "uvicorn.error": {"handlers": ["file"], "level": "INFO", "propagate": False},
        "uvicorn.access": {"handlers": ["file"], "level": "INFO", "propagate": False},
        "fastapi": {"handlers": ["file"], "level": "DEBUG", "propagate": False},
    },
}

dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# -------------------------------
# FastAPI App Setup
# -------------------------------
app = FastAPI(
    title="PDF Research Assistant",
    description="Upload PDFs and ask questions based on PDF content",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 FastAPI app started successfully")

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
app.include_router(api_router)
