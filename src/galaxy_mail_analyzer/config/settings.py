"""Application settings using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # API Keys
    anthropic_api_key: str = Field(description="Anthropic API key for Claude")

    # Optional - only needed if using Voyage AI instead of local embeddings
    voyage_api_key: Optional[str] = Field(
        default=None,
        description="Optional Voyage AI API key (not needed - using local embeddings)",
    )

    # Database
    database_url: str = Field(
        default="sqlite:///./data/galaxy_mail.db",
        description="Database connection URL",
    )

    # Vector Store
    chroma_persist_dir: str = Field(
        default="./data/chroma",
        description="Directory for ChromaDB persistence",
    )

    # Mailing List
    hyperkitty_base_url: str = Field(
        default="https://lists.galaxyproject.org",
        description="HyperKitty base URL",
    )
    mailing_list_address: str = Field(
        default="galaxy-bugs@lists.galaxyproject.org",
        description="Mailing list address",
    )

    # Discourse (Knowledge Base)
    discourse_host: str = Field(
        default="https://help.galaxyproject.org",
        description="Discourse instance URL",
    )
    discourse_api_key: Optional[str] = Field(
        default=None,
        description="Optional Discourse API key for higher rate limits",
    )
    discourse_api_username: Optional[str] = Field(
        default=None,
        description="Optional Discourse API username",
    )

    # Analysis Settings
    claude_model: str = Field(
        default="claude-sonnet-4-5-20250929",
        description="Claude model for analysis",
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformers model for embeddings (local, free)",
    )
    min_cluster_size: int = Field(
        default=50,
        description="Minimum emails required for clustering",
    )

    # Paths
    @property
    def data_dir(self) -> Path:
        """Get the data directory path."""
        return Path("./data")

    @property
    def mbox_dir(self) -> Path:
        """Get the mbox directory path."""
        return self.data_dir / "mbox"

    @property
    def reports_dir(self) -> Path:
        """Get the reports directory path."""
        return self.data_dir / "reports"

    @property
    def hyperkitty_export_url(self) -> str:
        """Construct the HyperKitty mbox export URL."""
        list_id = self.mailing_list_address.replace("@", ".")
        return (
            f"{self.hyperkitty_base_url}/archives/list/{list_id}/"
            f"export/{list_id}.mbox.gz"
        )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
