import asyncio
import os
from app.utils import extract_text_to_file, chunk_text_and_save, generate_embeddings_and_store
from app.qdrant_client import search_qdrant_for_doc
from app.genai_client import answer_with_groq_async

# Set paths for a test PDF
TEST_PDF_PATH = r"C:\Users\unnat\OneDrive\Desktop\DatasmithAI\AI powered research assisstant\uploads\Unit V-Risk Assessment in-vivo testing of biomaterials (1).pdf"
DOC_ID = "test_doc"

async def main():
    print("1. Extracting text from PDF...")
    text_path = extract_text_to_file(TEST_PDF_PATH, DOC_ID)
    print(f"Text saved to: {text_path}")

    print("2. Chunking text...")
    chunks_path, num_chunks = chunk_text_and_save(text_path, DOC_ID)
    print(f"{num_chunks} chunks saved to: {chunks_path}")

    print("3. Generating embeddings and storing in Qdrant...")
    generate_embeddings_and_store(chunks_path, DOC_ID)
    print("Chunks stored in Qdrant.")

    print("4. Searching Qdrant for test query...")
    query = "Your question about the PDF"
    results = search_qdrant_for_doc(query, DOC_ID, top_k=5)
    context_chunks = [r.payload.get("text", "") for r in results if r.payload.get("text")]
    print(f"Found {len(context_chunks)} chunks from search.")

    if context_chunks:
        context_text = "\n\n".join(context_chunks)
        prompt = (
            f"Use the following context from a PDF to answer the question.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {query}\n"
            f"Answer based only on the context provided. If the answer is not present, respond 'Not available in the document.'"
        )
    else:
        prompt = query

    print("5. Generating answer using Groq...")
    answer = await answer_with_groq_async(prompt)
    print("\n=== Answer ===")
    print(answer)

if __name__ == "__main__":
    asyncio.run(main())
