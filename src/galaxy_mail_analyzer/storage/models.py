"""SQLAlchemy ORM models for the Galaxy Mail Analyzer."""

import enum
from datetime import datetime
from typing import List, Optional

from sqlalchemy import DateTime, Enum as SQLEnum, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass


class EmailStatus(enum.Enum):
    """Processing status for emails."""

    PENDING = "pending"
    ANALYZED = "analyzed"
    AUTO_ANSWERED = "auto_answered"
    NEEDS_HUMAN = "needs_human"
    RESOLVED = "resolved"


class IssuePriority(enum.Enum):
    """Priority levels for issues."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IssueType(enum.Enum):
    """Types of issues detected in emails."""

    BUG = "bug"
    QUESTION = "question"
    FEATURE_REQUEST = "feature_request"
    SECURITY = "security"
    DOCUMENTATION = "documentation"
    INSTALLATION = "installation"
    PERFORMANCE = "performance"
    OTHER = "other"


class Email(Base):
    """Email message from the mailing list."""

    __tablename__ = "emails"

    id: Mapped[int] = mapped_column(primary_key=True)
    message_id: Mapped[str] = mapped_column(String(500), unique=True, index=True)
    thread_id: Mapped[Optional[str]] = mapped_column(String(500), index=True)
    in_reply_to: Mapped[Optional[str]] = mapped_column(String(500))
    references: Mapped[Optional[str]] = mapped_column(Text)

    # Sender info
    sender_email: Mapped[str] = mapped_column(String(255), index=True)
    sender_name: Mapped[Optional[str]] = mapped_column(String(255))

    # Content
    subject: Mapped[str] = mapped_column(String(1000))
    body_text: Mapped[str] = mapped_column(Text)
    body_html: Mapped[Optional[str]] = mapped_column(Text)

    # Timestamps
    date_sent: Mapped[datetime] = mapped_column(DateTime, index=True)
    date_imported: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    date_analyzed: Mapped[Optional[datetime]] = mapped_column(DateTime)

    # Processing status
    status: Mapped[EmailStatus] = mapped_column(
        SQLEnum(EmailStatus), default=EmailStatus.PENDING, index=True
    )
    priority: Mapped[Optional[IssuePriority]] = mapped_column(SQLEnum(IssuePriority))

    # Embedding reference (stored in ChromaDB)
    embedding_id: Mapped[Optional[str]] = mapped_column(String(100))

    # Relationships
    analysis: Mapped[Optional["EmailAnalysis"]] = relationship(
        back_populates="email", cascade="all, delete-orphan"
    )
    cluster_assignments: Mapped[List["ClusterAssignment"]] = relationship(
        back_populates="email", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_email_date_status", "date_sent", "status"),)

    def __repr__(self) -> str:
        return f"<Email {self.id}: {self.subject[:50]}...>"


class EmailAnalysis(Base):
    """AI analysis results for an email."""

    __tablename__ = "email_analyses"

    id: Mapped[int] = mapped_column(primary_key=True)
    email_id: Mapped[int] = mapped_column(ForeignKey("emails.id"), unique=True)

    # Classification results
    summary: Mapped[str] = mapped_column(Text)
    issue_type: Mapped[IssueType] = mapped_column(SQLEnum(IssueType), index=True)
    is_security_related: Mapped[bool] = mapped_column(default=False, index=True)
    requires_human_attention: Mapped[bool] = mapped_column(default=False, index=True)
    confidence_score: Mapped[float] = mapped_column(Float)
    reasoning: Mapped[Optional[str]] = mapped_column(Text)

    # Suggested response
    suggested_response: Mapped[Optional[str]] = mapped_column(Text)
    matched_kb_article_ids: Mapped[Optional[str]] = mapped_column(Text)  # JSON list

    # Metadata
    model_version: Mapped[str] = mapped_column(String(50))
    analyzed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    email: Mapped["Email"] = relationship(back_populates="analysis")

    def __repr__(self) -> str:
        return f"<EmailAnalysis {self.id}: {self.issue_type.value}>"


class TopicCluster(Base):
    """Topic clusters from BERTopic analysis."""

    __tablename__ = "topic_clusters"

    id: Mapped[int] = mapped_column(primary_key=True)
    cluster_id: Mapped[int] = mapped_column(Integer, index=True)  # BERTopic topic ID

    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[Optional[str]] = mapped_column(Text)
    keywords: Mapped[str] = mapped_column(Text)  # JSON list
    representative_docs: Mapped[Optional[str]] = mapped_column(Text)  # JSON list of email IDs

    email_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, onupdate=datetime.utcnow)

    # Relationships
    assignments: Mapped[List["ClusterAssignment"]] = relationship(
        back_populates="cluster", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<TopicCluster {self.cluster_id}: {self.name}>"


class ClusterAssignment(Base):
    """Many-to-many relationship between emails and topic clusters."""

    __tablename__ = "cluster_assignments"

    id: Mapped[int] = mapped_column(primary_key=True)
    email_id: Mapped[int] = mapped_column(ForeignKey("emails.id"), index=True)
    cluster_id: Mapped[int] = mapped_column(ForeignKey("topic_clusters.id"), index=True)
    probability: Mapped[float] = mapped_column(Float)  # Assignment confidence

    # Relationships
    email: Mapped["Email"] = relationship(back_populates="cluster_assignments")
    cluster: Mapped["TopicCluster"] = relationship(back_populates="assignments")

    __table_args__ = (Index("ix_cluster_email", "cluster_id", "email_id"),)

    def __repr__(self) -> str:
        return f"<ClusterAssignment email={self.email_id} cluster={self.cluster_id}>"


class KnowledgeBaseArticle(Base):
    """Cached Discourse articles from help.galaxyproject.org."""

    __tablename__ = "kb_articles"

    id: Mapped[int] = mapped_column(primary_key=True)
    discourse_topic_id: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    discourse_post_id: Mapped[int] = mapped_column(Integer)

    title: Mapped[str] = mapped_column(String(500))
    slug: Mapped[str] = mapped_column(String(500))
    category: Mapped[str] = mapped_column(String(100), index=True)
    content: Mapped[str] = mapped_column(Text)
    url: Mapped[str] = mapped_column(String(500))

    # Metadata
    view_count: Mapped[int] = mapped_column(Integer, default=0)
    like_count: Mapped[int] = mapped_column(Integer, default=0)
    reply_count: Mapped[int] = mapped_column(Integer, default=0)
    is_solved: Mapped[bool] = mapped_column(default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime)
    last_synced: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Embedding reference (stored in ChromaDB)
    embedding_id: Mapped[Optional[str]] = mapped_column(String(100))

    def __repr__(self) -> str:
        return f"<KnowledgeBaseArticle {self.discourse_topic_id}: {self.title[:50]}...>"


class AnalysisReport(Base):
    """Generated analysis reports."""

    __tablename__ = "analysis_reports"

    id: Mapped[int] = mapped_column(primary_key=True)
    report_type: Mapped[str] = mapped_column(String(50), index=True)  # daily, weekly, monthly, yearly
    period_start: Mapped[datetime] = mapped_column(DateTime, index=True)
    period_end: Mapped[datetime] = mapped_column(DateTime)

    # Summary metrics
    total_emails: Mapped[int] = mapped_column(Integer)
    auto_answered: Mapped[int] = mapped_column(Integer)
    needs_human: Mapped[int] = mapped_column(Integer)
    security_issues: Mapped[int] = mapped_column(Integer, default=0)

    # JSON data
    by_issue_type: Mapped[str] = mapped_column(Text)  # JSON dict
    top_clusters: Mapped[Optional[str]] = mapped_column(Text)  # JSON list
    trending_topics: Mapped[Optional[str]] = mapped_column(Text)  # JSON list

    # Full report content
    report_markdown: Mapped[str] = mapped_column(Text)

    generated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("ix_report_type_period", "report_type", "period_start"),)

    def __repr__(self) -> str:
        return f"<AnalysisReport {self.report_type}: {self.period_start.date()}>"
