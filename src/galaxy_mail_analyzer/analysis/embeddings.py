"""Embedding generation using sentence-transformers (local, free)."""

import logging
from typing import Optional

from sentence_transformers import SentenceTransformer

from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings using sentence-transformers (local, no API key needed)."""

    # Default model - good balance of quality and speed
    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str | None = None,
    ):
        """Initialize the embedding generator.

        Args:
            model_name: Sentence-transformers model name. Defaults to all-MiniLM-L6-v2.
        """
        self._model_name = model_name or self.DEFAULT_MODEL

        logger.info(f"Loading embedding model: {self._model_name}")
        self._model = SentenceTransformer(self._model_name)
        self._dimensions = self._model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimensions: {self._dimensions}")

    @property
    def dimensions(self) -> int:
        """Get the embedding dimensions."""
        return self._dimensions

    def embed_documents(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> list[list[float]]:
        """Generate embeddings for documents (knowledge base articles, emails).

        Args:
            texts: List of text documents to embed.
            batch_size: Number of documents to embed per batch.
            show_progress: Whether to show a progress bar.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        logger.debug(f"Embedding {len(texts)} documents")

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        return [emb.tolist() for emb in embeddings]

    def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a query (for searching).

        Args:
            query: Query text.

        Returns:
            Embedding vector.
        """
        embedding = self._model.encode(query, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed.
            batch_size: Batch size for encoding.

        Returns:
            List of embedding vectors.
        """
        return self.embed_documents(texts, batch_size=batch_size)
