import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from .embeddings import embed_text
import uuid

# Qdrant configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "pdf_chunks")

# Initialize Qdrant client
qdrant = QdrantClient(url=QDRANT_URL)

# Ensure collection exists
collections_info = qdrant.get_collections()
existing_collections = [c.name for c in collections_info.collections] if collections_info.collections else []

if COLLECTION_NAME not in existing_collections:
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
    )

def search_qdrant_for_doc(query: str, doc_id: str, top_k: int = 3):
    """
    Searches Qdrant for the top_k most similar vectors to the query within a specific document.
    """
    if not query or not doc_id:
        return []

    query_emb = embed_text([query])[0]

    flt = models.Filter(
        must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
    )

    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_emb,
        limit=top_k,
        query_filter=flt
    )
    return results
import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from .embeddings import embed_text

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "pdf_chunks")

qdrant = QdrantClient(url=QDRANT_URL)

# Ensure collection exists
collections_info = qdrant.get_collections()
existing_collections = [c.name for c in collections_info.collections] if collections_info.collections else []
if COLLECTION_NAME not in existing_collections:
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
    )

def generate_embeddings_and_store(chunks_path: str, doc_id: str):
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = [line.strip() for line in f.readlines() if line.strip()]

    points = []
    for i, chunk in enumerate(chunks):
        embedding_vector = embed_text(chunk)[0]
        points.append({
            "id": f"{doc_id}_{i}",
            "vector": embedding_vector,
            "payload": {"text": chunk, "doc_id": doc_id}
        })

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

def search_qdrant_for_doc(query: str, doc_id: str, top_k: int = 10):
    query_emb = embed_text([query])[0]
    flt = models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])
    results = qdrant.search(collection_name=COLLECTION_NAME, query_vector=query_emb, limit=top_k, query_filter=flt)
    return results
