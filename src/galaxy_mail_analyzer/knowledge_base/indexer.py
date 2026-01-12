"""Knowledge base indexer for building searchable embeddings."""

import asyncio
import logging
from datetime import datetime

from sqlalchemy.orm import Session

from config.settings import get_settings
from ..storage.database import get_db
from ..storage.models import KnowledgeBaseArticle
from ..storage.vector_store import get_vector_store
from ..analysis.embeddings import EmbeddingGenerator
from .discourse import DiscourseClient, DiscourseArticle

logger = logging.getLogger(__name__)


class KnowledgeBaseIndexer:
    """Index knowledge base articles with embeddings for semantic search."""

    def __init__(
        self,
        discourse_client: DiscourseClient | None = None,
        embedding_generator: EmbeddingGenerator | None = None,
    ):
        """Initialize the indexer.

        Args:
            discourse_client: Optional Discourse client instance.
            embedding_generator: Optional embedding generator instance.
        """
        self._discourse = discourse_client or DiscourseClient()
        self._embeddings = embedding_generator or EmbeddingGenerator()
        self._db = get_db()
        self._vector_store = get_vector_store()

    async def sync_from_discourse(
        self,
        categories: list[str] | None = None,
        max_topics: int = 10000,
        batch_size: int = 50,
    ) -> int:
        """Sync knowledge base from Discourse.

        Args:
            categories: Optional list of category slugs to sync.
            max_topics: Maximum number of topics to sync.
            batch_size: Number of articles to process per batch.

        Returns:
            Number of articles synced.
        """
        logger.info("Starting knowledge base sync from Discourse")

        articles_batch: list[DiscourseArticle] = []
        total_synced = 0

        async for article in self._discourse.fetch_all_topics(
            categories=categories,
            max_topics=max_topics,
        ):
            articles_batch.append(article)

            if len(articles_batch) >= batch_size:
                synced = await self._process_batch(articles_batch)
                total_synced += synced
                articles_batch = []

        # Process remaining articles
        if articles_batch:
            synced = await self._process_batch(articles_batch)
            total_synced += synced

        logger.info(f"Knowledge base sync complete: {total_synced} articles")
        return total_synced

    async def _process_batch(
        self,
        articles: list[DiscourseArticle],
    ) -> int:
        """Process a batch of articles.

        Args:
            articles: List of articles to process.

        Returns:
            Number of articles successfully processed.
        """
        if not articles:
            return 0

        # Generate embeddings
        texts = [f"{a.title}\n\n{a.content}" for a in articles]
        embeddings = self._embeddings.embed_documents(texts)

        processed = 0
        with self._db.session() as session:
            for article, embedding in zip(articles, embeddings):
                try:
                    # Store or update in database
                    db_article = self._upsert_article(session, article)

                    # Store embedding in vector store
                    embedding_id = f"kb_{article.topic_id}"
                    self._vector_store.add_kb_articles(
                        ids=[embedding_id],
                        embeddings=[embedding],
                        documents=[f"{article.title}\n\n{article.content}"],
                        metadatas=[{
                            "topic_id": article.topic_id,
                            "title": article.title,
                            "category": article.category,
                            "url": article.url,
                            "is_solved": article.is_solved,
                        }],
                    )

                    db_article.embedding_id = embedding_id
                    processed += 1
                except Exception as e:
                    logger.error(f"Error processing article {article.topic_id}: {e}")
                    continue

            session.commit()

        logger.debug(f"Processed batch of {processed} articles")
        return processed

    def _upsert_article(
        self,
        session: Session,
        article: DiscourseArticle,
    ) -> KnowledgeBaseArticle:
        """Insert or update an article in the database.

        Args:
            session: Database session.
            article: Article to upsert.

        Returns:
            Database article instance.
        """
        # Check if article exists
        db_article = session.query(KnowledgeBaseArticle).filter(
            KnowledgeBaseArticle.discourse_topic_id == article.topic_id
        ).first()

        if db_article:
            # Update existing
            db_article.title = article.title
            db_article.slug = article.slug
            db_article.category = article.category
            db_article.content = article.content
            db_article.url = article.url
            db_article.view_count = article.view_count
            db_article.like_count = article.like_count
            db_article.reply_count = article.reply_count
            db_article.is_solved = article.is_solved
            db_article.updated_at = article.updated_at
            db_article.last_synced = datetime.utcnow()
        else:
            # Create new
            db_article = KnowledgeBaseArticle(
                discourse_topic_id=article.topic_id,
                discourse_post_id=article.post_id,
                title=article.title,
                slug=article.slug,
                category=article.category,
                content=article.content,
                url=article.url,
                view_count=article.view_count,
                like_count=article.like_count,
                reply_count=article.reply_count,
                is_solved=article.is_solved,
                created_at=article.created_at,
                updated_at=article.updated_at,
                last_synced=datetime.utcnow(),
            )
            session.add(db_article)

        return db_article

    def search(
        self,
        query: str,
        n_results: int = 5,
        solved_only: bool = False,
    ) -> list[dict]:
        """Search the knowledge base for relevant articles.

        Args:
            query: Search query text.
            n_results: Number of results to return.
            solved_only: Only return solved topics.

        Returns:
            List of matching articles with scores.
        """
        # Generate query embedding
        query_embedding = self._embeddings.embed_query(query)

        # Build filter
        where = {"is_solved": True} if solved_only else None

        # Search vector store
        results = self._vector_store.search_kb(
            query_embedding=query_embedding,
            n_results=n_results,
            where=where,
        )

        # Format results
        articles = []
        if results["ids"] and results["ids"][0]:
            for i, id_ in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0
                document = results["documents"][0][i] if results["documents"] else ""

                articles.append({
                    "id": id_,
                    "topic_id": metadata.get("topic_id"),
                    "title": metadata.get("title"),
                    "category": metadata.get("category"),
                    "url": metadata.get("url"),
                    "is_solved": metadata.get("is_solved", False),
                    "similarity": 1 - distance,  # Convert distance to similarity
                    "excerpt": document[:500] if document else "",
                })

        return articles

    def get_article_by_id(self, topic_id: int) -> KnowledgeBaseArticle | None:
        """Get an article by its Discourse topic ID.

        Args:
            topic_id: Discourse topic ID.

        Returns:
            Article or None if not found.
        """
        with self._db.session() as session:
            return session.query(KnowledgeBaseArticle).filter(
                KnowledgeBaseArticle.discourse_topic_id == topic_id
            ).first()

    def get_stats(self) -> dict:
        """Get knowledge base statistics.

        Returns:
            Dictionary with statistics.
        """
        with self._db.session() as session:
            total = session.query(KnowledgeBaseArticle).count()
            solved = session.query(KnowledgeBaseArticle).filter(
                KnowledgeBaseArticle.is_solved == True
            ).count()
            categories = session.query(
                KnowledgeBaseArticle.category
            ).distinct().count()

        vector_count = self._vector_store.get_kb_count()

        return {
            "total_articles": total,
            "solved_articles": solved,
            "categories": categories,
            "indexed_embeddings": vector_count,
        }
