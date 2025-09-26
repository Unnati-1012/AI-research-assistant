# logging_config.py
import logging
from logging.config import dictConfig

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,  # <- important
    "formatters": {
        "default": {
            "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "DEBUG",
        },
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console"],
    },
    "loggers": {
        "uvicorn": {"handlers": ["console"], "level": "INFO", "propagate": False},
        "uvicorn.error": {"handlers": ["console"], "level": "INFO", "propagate": False},
        "uvicorn.access": {"handlers": ["console"], "level": "INFO", "propagate": False},
        "fastapi": {"handlers": ["console"], "level": "DEBUG", "propagate": False},
    },
}

def setup_logging():
    dictConfig(LOGGING_CONFIG)
    logging.getLogger(__name__).info("✅ Custom logging configured")
