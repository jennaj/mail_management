"""Embedding generation using Voyage AI."""

import logging
from typing import List

import voyageai

from config.settings import get_settings

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings using Voyage AI."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
    ):
        """Initialize the embedding generator.

        Args:
            api_key: Voyage AI API key.
            model: Model to use for embeddings.
            dimensions: Output embedding dimensions.
        """
        settings = get_settings()
        self._api_key = api_key or settings.voyage_api_key
        self._model = model or settings.voyage_model
        self._dimensions = dimensions or settings.embedding_dimensions

        self._client = voyageai.Client(api_key=self._api_key)

    def embed_documents(
        self,
        texts: list[str],
        batch_size: int = 128,
    ) -> list[list[float]]:
        """Generate embeddings for documents (knowledge base articles, emails).

        Args:
            texts: List of text documents to embed.
            batch_size: Number of documents to embed per API call.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.debug(f"Embedding batch {i // batch_size + 1}")

            result = self._client.embed(
                batch,
                model=self._model,
                input_type="document",
                output_dimension=self._dimensions,
            )
            all_embeddings.extend(result.embeddings)

        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a query (for searching).

        Args:
            query: Query text.

        Returns:
            Embedding vector.
        """
        result = self._client.embed(
            [query],
            model=self._model,
            input_type="query",
            output_dimension=self._dimensions,
        )
        return result.embeddings[0]

    def embed_batch(
        self,
        texts: list[str],
        input_type: str = "document",
    ) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed.
            input_type: Either "document" or "query".

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        result = self._client.embed(
            texts,
            model=self._model,
            input_type=input_type,
            output_dimension=self._dimensions,
        )
        return result.embeddings
