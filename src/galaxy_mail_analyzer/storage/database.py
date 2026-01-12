"""Database connection and session management."""

from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from config.settings import get_settings
from .models import Base


class Database:
    """Database connection manager."""

    def __init__(self, database_url: str | None = None):
        """Initialize database connection.

        Args:
            database_url: Optional database URL. If not provided, uses settings.
        """
        self._url = database_url or get_settings().database_url

        # Ensure data directory exists for SQLite
        if self._url.startswith("sqlite"):
            db_path = self._url.replace("sqlite:///", "")
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._engine = create_engine(
            self._url,
            echo=False,  # Set to True for SQL debugging
            pool_pre_ping=True,
        )
        self._session_factory = sessionmaker(
            bind=self._engine,
            autocommit=False,
            autoflush=False,
        )

    def create_tables(self) -> None:
        """Create all database tables."""
        Base.metadata.create_all(self._engine)

    def drop_tables(self) -> None:
        """Drop all database tables. Use with caution!"""
        Base.metadata.drop_all(self._engine)

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Get a database session context manager.

        Usage:
            with db.session() as session:
                session.add(email)
                session.commit()
        """
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_session(self) -> Session:
        """Get a new database session.

        Note: Caller is responsible for closing the session.
        Prefer using the session() context manager instead.
        """
        return self._session_factory()


# Global database instance
_db: Database | None = None


def get_db() -> Database:
    """Get or create the global database instance."""
    global _db
    if _db is None:
        _db = Database()
        _db.create_tables()
    return _db


def init_db(database_url: str | None = None) -> Database:
    """Initialize a new database instance.

    Args:
        database_url: Optional database URL override.

    Returns:
        Database instance.
    """
    global _db
    _db = Database(database_url)
    _db.create_tables()
    return _db
