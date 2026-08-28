"""Embedding generation — supports EURI (OpenAI-compatible) and Google Gemini."""

from config import (
    LLM_PROVIDER, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS,
    EURI_API_KEY, EURI_BASE_URL, GOOGLE_API_KEY,
)


class EmbeddingClient:
    """Generates embeddings via the active provider.

    Usage:
        client = EmbeddingClient()
        vecs = client.embed(["sentence 1", "sentence 2"])
    """

    def __init__(self, model: str = "", dimensions: int = 0):
        self.model = model or EMBEDDING_MODEL
        self.dimensions = dimensions or EMBEDDING_DIMENSIONS
        self._provider = LLM_PROVIDER

        if self._provider == "google":
            from google import genai
            self._google_client = genai.Client(api_key=GOOGLE_API_KEY)
        else:
            from openai import OpenAI
            self._openai_client = OpenAI(
                api_key=EURI_API_KEY,
                base_url=EURI_BASE_URL,
            )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._provider == "google":
            return self._embed_google(texts)
        return self._embed_openai(texts)

    def _embed_google(self, texts: list[str]) -> list[list[float]]:
        """Embed via Google's native genai SDK (supports output_dimensionality)."""
        response = self._google_client.models.embed_content(
            model=self.model,
            contents=texts,
            config={"output_dimensionality": self.dimensions},
        )
        return [e.values for e in response.embeddings]

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        """Embed via EURI's OpenAI-compatible API."""
        resp = self._openai_client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self.dimensions,
        )
        return [item.embedding for item in resp.data]
