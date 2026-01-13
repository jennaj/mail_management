"""ChromaDB vector store for semantic search."""

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from ..config.settings import get_settings


class VectorStore:
    """ChromaDB vector store for embeddings."""

    # Collection names
    KB_COLLECTION = "galaxy_knowledge_base"
    EMAIL_COLLECTION = "galaxy_emails"

    def __init__(self, persist_dir: str | None = None):
        """Initialize ChromaDB client.

        Args:
            persist_dir: Directory for ChromaDB persistence.
        """
        self._persist_dir = persist_dir or get_settings().chroma_persist_dir

        # Ensure directory exists
        Path(self._persist_dir).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=self._persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # Initialize collections
        self._kb_collection = self._client.get_or_create_collection(
            name=self.KB_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._email_collection = self._client.get_or_create_collection(
            name=self.EMAIL_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def kb_collection(self) -> chromadb.Collection:
        """Get the knowledge base collection."""
        return self._kb_collection

    @property
    def email_collection(self) -> chromadb.Collection:
        """Get the email collection."""
        return self._email_collection

    def add_kb_articles(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Add knowledge base articles to the vector store.

        Args:
            ids: Unique identifiers for each article.
            embeddings: Embedding vectors.
            documents: Article text content.
            metadatas: Metadata for each article.
        """
        self._kb_collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def add_emails(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Add emails to the vector store.

        Args:
            ids: Unique identifiers for each email.
            embeddings: Embedding vectors.
            documents: Email text content.
            metadatas: Metadata for each email.
        """
        self._email_collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def search_kb(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Search knowledge base for similar articles.

        Args:
            query_embedding: Query embedding vector.
            n_results: Number of results to return.
            where: Optional filter conditions.

        Returns:
            Query results with documents, metadatas, and distances.
        """
        return self._kb_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

    def search_emails(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Search emails for similar messages.

        Args:
            query_embedding: Query embedding vector.
            n_results: Number of results to return.
            where: Optional filter conditions.

        Returns:
            Query results with documents, metadatas, and distances.
        """
        return self._email_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

    def get_kb_article(self, article_id: str) -> dict[str, Any] | None:
        """Get a specific knowledge base article by ID.

        Args:
            article_id: The article's unique identifier.

        Returns:
            Article data or None if not found.
        """
        result = self._kb_collection.get(
            ids=[article_id],
            include=["documents", "metadatas", "embeddings"],
        )
        if result["ids"]:
            return {
                "id": result["ids"][0],
                "document": result["documents"][0] if result["documents"] else None,
                "metadata": result["metadatas"][0] if result["metadatas"] else None,
                "embedding": result["embeddings"][0] if result["embeddings"] else None,
            }
        return None

    def delete_kb_articles(self, ids: list[str]) -> None:
        """Delete knowledge base articles by ID.

        Args:
            ids: List of article IDs to delete.
        """
        self._kb_collection.delete(ids=ids)

    def delete_emails(self, ids: list[str]) -> None:
        """Delete emails by ID.

        Args:
            ids: List of email IDs to delete.
        """
        self._email_collection.delete(ids=ids)

    def get_kb_count(self) -> int:
        """Get the number of articles in the knowledge base."""
        return self._kb_collection.count()

    def get_email_count(self) -> int:
        """Get the number of emails in the store."""
        return self._email_collection.count()

    def reset(self) -> None:
        """Delete all data from both collections. Use with caution!"""
        self._client.delete_collection(self.KB_COLLECTION)
        self._client.delete_collection(self.EMAIL_COLLECTION)
        # Recreate collections
        self._kb_collection = self._client.get_or_create_collection(
            name=self.KB_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._email_collection = self._client.get_or_create_collection(
            name=self.EMAIL_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )


# Global vector store instance
_vector_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """Get or create the global vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
