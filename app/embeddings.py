from fastembed import TextEmbedding
import os

# Load embedding model from environment variable, fallback to default
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
embedder = TextEmbedding(EMBED_MODEL)


def embed_text(texts):
    """
    Embed text using FastEmbed.

    Args:
        texts (str or list[str]): Single string or list of strings to embed.

    Returns:
        list: List of embedding vectors.
    """
    if isinstance(texts, str):
        texts = [texts]

    embeddings = embedder.embed(texts)
    return list(embeddings)
from fastembed import TextEmbedding
import os

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
embedder = TextEmbedding(EMBED_MODEL)

def embed_text(texts):
    if isinstance(texts, str):
        texts = [texts]
    return list(embedder.embed(texts))
