"""Email clustering using BERTopic."""

import json
import logging
from datetime import datetime
from typing import Any

import numpy as np
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP

from config.settings import get_settings
from ..storage.database import get_db
from ..storage.models import Email, TopicCluster, ClusterAssignment
from .embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)


class EmailClusterer:
    """Cluster emails by topic using BERTopic."""

    def __init__(
        self,
        embedding_generator: EmbeddingGenerator | None = None,
        min_cluster_size: int | None = None,
    ):
        """Initialize the clusterer.

        Args:
            embedding_generator: Optional embedding generator.
            min_cluster_size: Minimum cluster size for HDBSCAN.
        """
        settings = get_settings()
        self._embeddings = embedding_generator or EmbeddingGenerator()
        self._min_cluster_size = min_cluster_size or settings.min_cluster_size
        self._db = get_db()

        # Initialize BERTopic components
        self._umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        )
        self._hdbscan_model = HDBSCAN(
            min_cluster_size=self._min_cluster_size,
            min_samples=5,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )

        self._topic_model: BERTopic | None = None

    def cluster_emails(
        self,
        email_ids: list[int] | None = None,
        refit: bool = False,
    ) -> dict[str, Any]:
        """Cluster emails by topic.

        Args:
            email_ids: Optional list of email IDs to cluster. If None, clusters all.
            refit: If True, refit the model even if it exists.

        Returns:
            Clustering results with topic info.
        """
        # Get emails
        with self._db.session() as session:
            query = session.query(Email)
            if email_ids:
                query = query.filter(Email.id.in_(email_ids))

            emails = query.all()
            if len(emails) < self._min_cluster_size:
                logger.warning(
                    f"Not enough emails for clustering: {len(emails)} < {self._min_cluster_size}"
                )
                return {"error": "Not enough emails for clustering"}

            # Prepare documents
            docs = []
            ids = []
            for email in emails:
                text = f"{email.subject}\n\n{email.body_text}"
                docs.append(text)
                ids.append(email.id)

        logger.info(f"Clustering {len(docs)} emails")

        # Generate embeddings
        embeddings = self._embeddings.embed_documents(docs)
        embeddings_array = np.array(embeddings)

        # Fit or transform
        if self._topic_model is None or refit:
            self._topic_model = BERTopic(
                umap_model=self._umap_model,
                hdbscan_model=self._hdbscan_model,
                calculate_probabilities=True,
                verbose=True,
            )
            topics, probs = self._topic_model.fit_transform(docs, embeddings_array)
        else:
            topics, probs = self._topic_model.transform(docs, embeddings_array)

        # Get topic info
        topic_info = self._topic_model.get_topic_info()

        # Save clusters to database
        self._save_clusters(ids, topics, probs, topic_info)

        return {
            "n_emails": len(docs),
            "n_topics": len(topic_info) - 1,  # Exclude -1 (outliers)
            "topic_info": topic_info.to_dict(orient="records"),
        }

    def _save_clusters(
        self,
        email_ids: list[int],
        topics: list[int],
        probs: np.ndarray,
        topic_info: Any,
    ) -> None:
        """Save clustering results to database.

        Args:
            email_ids: List of email IDs.
            topics: Topic assignments.
            probs: Assignment probabilities.
            topic_info: DataFrame with topic information.
        """
        with self._db.session() as session:
            # Clear existing cluster assignments for these emails
            session.query(ClusterAssignment).filter(
                ClusterAssignment.email_id.in_(email_ids)
            ).delete(synchronize_session=False)

            # Create or update topic clusters
            for _, row in topic_info.iterrows():
                cluster_id = row["Topic"]
                if cluster_id == -1:
                    continue  # Skip outliers

                # Get keywords
                topic_words = self._topic_model.get_topic(cluster_id)
                keywords = [word for word, _ in topic_words[:10]] if topic_words else []

                # Get representative docs
                rep_docs = self._topic_model.get_representative_docs(cluster_id)
                rep_email_ids = []
                if rep_docs:
                    for doc in rep_docs[:5]:
                        # Find matching email ID
                        for i, email_id in enumerate(email_ids):
                            text = self._get_email_text(session, email_id)
                            if text and doc[:100] in text:
                                rep_email_ids.append(email_id)
                                break

                # Upsert cluster
                cluster = session.query(TopicCluster).filter(
                    TopicCluster.cluster_id == cluster_id
                ).first()

                if cluster:
                    cluster.name = row.get("Name", f"Topic {cluster_id}")
                    cluster.keywords = json.dumps(keywords)
                    cluster.email_count = row.get("Count", 0)
                    cluster.representative_docs = json.dumps(rep_email_ids)
                    cluster.updated_at = datetime.utcnow()
                else:
                    cluster = TopicCluster(
                        cluster_id=cluster_id,
                        name=row.get("Name", f"Topic {cluster_id}"),
                        keywords=json.dumps(keywords),
                        email_count=row.get("Count", 0),
                        representative_docs=json.dumps(rep_email_ids),
                    )
                    session.add(cluster)
                    session.flush()

            # Create cluster assignments
            for i, (email_id, topic, prob_row) in enumerate(zip(email_ids, topics, probs)):
                if topic == -1:
                    continue  # Skip outliers

                cluster = session.query(TopicCluster).filter(
                    TopicCluster.cluster_id == topic
                ).first()

                if cluster:
                    probability = float(prob_row[topic]) if hasattr(prob_row, "__getitem__") else float(prob_row)
                    assignment = ClusterAssignment(
                        email_id=email_id,
                        cluster_id=cluster.id,
                        probability=probability,
                    )
                    session.add(assignment)

            session.commit()
            logger.info("Saved clustering results to database")

    def _get_email_text(self, session, email_id: int) -> str | None:
        """Get email text by ID."""
        email = session.query(Email).filter(Email.id == email_id).first()
        if email:
            return f"{email.subject}\n\n{email.body_text}"
        return None

    def get_similar_emails(
        self,
        email_id: int,
        n_results: int = 5,
    ) -> list[dict]:
        """Find emails similar to a given email.

        Args:
            email_id: Email ID to find similar emails for.
            n_results: Number of results to return.

        Returns:
            List of similar emails with scores.
        """
        with self._db.session() as session:
            email = session.query(Email).filter(Email.id == email_id).first()
            if not email:
                return []

            # Get cluster assignment
            assignment = session.query(ClusterAssignment).filter(
                ClusterAssignment.email_id == email_id
            ).first()

            if not assignment:
                return []

            # Find other emails in same cluster
            similar = session.query(ClusterAssignment).filter(
                ClusterAssignment.cluster_id == assignment.cluster_id,
                ClusterAssignment.email_id != email_id,
            ).order_by(ClusterAssignment.probability.desc()).limit(n_results).all()

            results = []
            for s in similar:
                other_email = session.query(Email).filter(Email.id == s.email_id).first()
                if other_email:
                    results.append({
                        "email_id": other_email.id,
                        "subject": other_email.subject,
                        "sender": other_email.sender_email,
                        "date": other_email.date_sent.isoformat(),
                        "probability": s.probability,
                    })

            return results

    def get_cluster_summary(self) -> list[dict]:
        """Get summary of all clusters.

        Returns:
            List of cluster summaries.
        """
        with self._db.session() as session:
            clusters = session.query(TopicCluster).order_by(
                TopicCluster.email_count.desc()
            ).all()

            return [
                {
                    "id": c.id,
                    "cluster_id": c.cluster_id,
                    "name": c.name,
                    "keywords": json.loads(c.keywords) if c.keywords else [],
                    "email_count": c.email_count,
                }
                for c in clusters
            ]
