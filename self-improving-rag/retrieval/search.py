"""Retrieval: embed query → Chroma search → return ranked chunks."""

from embeddings import EmbeddingClient
from vector_store import ChromaStore


def search(collection_name: str, query: str, k: int = 5) -> list[dict]:
    """Embed query, search Chroma, return top-k chunks with scores.

    Args:
        collection_name: Chroma collection to search (e.g. "rag_v1_fixed_256")
        query: The user's question
        k: Number of chunks to retrieve

    Returns:
        List of dicts: [{chunk_id, text, score, metadata}, ...]
        Sorted by score descending (highest similarity first).
        Score = 1 - distance (cosine distance → similarity).
    """
    client = EmbeddingClient()
    store = ChromaStore(collection_name)

    # Embed the query (single string in a list for batch API)
    qvec = client.embed([query])[0]

    # Search Chroma
    results = store.query(qvec, k=k)

    # Normalize: Chroma returns nested lists, flatten to list of dicts
    chunks = []
    for i, doc_id in enumerate(results["ids"][0]):
        chunks.append({
            "chunk_id": doc_id,
            "text": results["documents"][0][i],
            "score": round(1.0 - results["distances"][0][i], 4),  # distance → similarity
            "metadata": results["metadatas"][0][i],
        })

    return chunks
