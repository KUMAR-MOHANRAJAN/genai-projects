"""ChromaDB vector store wrapper — one collection per config version."""

import chromadb
from config import CHROMA_PERSIST_DIR


class ChromaStore:
    """Wraps a ChromaDB collection: upsert, query, count, delete."""

    def __init__(self, collection_name: str):
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, ids: list[str], embeddings: list[list[float]],
               metadatas: list[dict], documents: list[str]):
        self.collection.upsert(
            ids=ids, embeddings=embeddings,
            metadatas=metadatas, documents=documents,
        )

    def query(self, query_embedding: list[float], k: int = 5,
              where: dict | None = None) -> dict:
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k, where=where,
        )

    def count(self) -> int:
        return self.collection.count()

    def delete(self):
        """Delete the entire collection (used for cleanup between runs)."""
        self.client.delete_collection(name=self.collection.name)

    @staticmethod
    def list_collections() -> list[dict]:
        """List all ChromaDB collections with their chunk counts.

        Returns list of {"name": str, "count": int} dicts, sorted by name.
        """
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        collections = []
        for col in client.list_collections():
            name = col.name if hasattr(col, "name") else str(col)
            count = client.get_collection(name).count()
            collections.append({"name": name, "count": count})
        return sorted(collections, key=lambda c: c["name"])
