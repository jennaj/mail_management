"""Discourse API client for help.galaxyproject.org."""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncIterator

import httpx
from bs4 import BeautifulSoup

from ..config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class DiscourseArticle:
    """Representation of a Discourse topic/article."""

    topic_id: int
    post_id: int
    title: str
    slug: str
    category: str
    content: str  # Plain text content
    content_html: str  # Original HTML
    url: str
    created_at: datetime
    updated_at: datetime
    view_count: int
    like_count: int
    reply_count: int
    is_solved: bool


class DiscourseClient:
    """Client for Discourse API (public read-only access)."""

    # Rate limiting: 60 requests per minute for anonymous users
    RATE_LIMIT_REQUESTS = 60
    RATE_LIMIT_WINDOW = 60  # seconds

    def __init__(
        self,
        host: str | None = None,
        api_key: str | None = None,
        api_username: str | None = None,
    ):
        """Initialize the Discourse client.

        Args:
            host: Discourse instance URL.
            api_key: Optional API key for authenticated access.
            api_username: Optional API username for authenticated access.
        """
        settings = get_settings()
        self._host = (host or settings.discourse_host).rstrip("/")
        self._api_key = api_key or settings.discourse_api_key
        self._api_username = api_username or settings.discourse_api_username

        self._request_count = 0
        self._window_start = datetime.now()

    def _get_headers(self) -> dict[str, str]:
        """Get request headers, including auth if configured."""
        headers = {
            "Accept": "application/json",
            "User-Agent": "GalaxyMailAnalyzer/0.1.0",
        }
        if self._api_key and self._api_username:
            headers["Api-Key"] = self._api_key
            headers["Api-Username"] = self._api_username
        return headers

    async def _rate_limit(self) -> None:
        """Apply rate limiting."""
        now = datetime.now()
        elapsed = (now - self._window_start).total_seconds()

        if elapsed > self.RATE_LIMIT_WINDOW:
            # Reset window
            self._request_count = 0
            self._window_start = now
        elif self._request_count >= self.RATE_LIMIT_REQUESTS:
            # Wait for window to reset
            wait_time = self.RATE_LIMIT_WINDOW - elapsed
            logger.info(f"Rate limit reached, waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)
            self._request_count = 0
            self._window_start = datetime.now()

        self._request_count += 1

    def _html_to_text(self, html: str) -> str:
        """Convert HTML content to plain text.

        Args:
            html: HTML content.

        Returns:
            Plain text content.
        """
        soup = BeautifulSoup(html, "html.parser")

        # Remove script and style elements
        for element in soup(["script", "style"]):
            element.decompose()

        # Get text
        text = soup.get_text(separator="\n")

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        text = "\n".join(line for line in lines if line)

        return text

    async def fetch_categories(self) -> list[dict]:
        """Fetch all categories from the Discourse instance.

        Returns:
            List of category dictionaries.
        """
        await self._rate_limit()

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self._host}/categories.json",
                headers=self._get_headers(),
            )
            response.raise_for_status()
            data = response.json()
            return data.get("category_list", {}).get("categories", [])

    async def fetch_topics_in_category(
        self,
        category_slug: str,
        category_id: int,
        page: int = 0,
    ) -> dict:
        """Fetch topics in a category with pagination.

        Args:
            category_slug: Category slug.
            category_id: Category ID.
            page: Page number (0-indexed).

        Returns:
            Topic list data.
        """
        await self._rate_limit()

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(
                f"{self._host}/c/{category_slug}/{category_id}.json",
                params={"page": page},
                headers=self._get_headers(),
            )
            response.raise_for_status()
            return response.json()

    async def fetch_topic(self, topic_id: int) -> dict:
        """Fetch a topic with its posts.

        Args:
            topic_id: Topic ID.

        Returns:
            Topic data including posts.
        """
        await self._rate_limit()

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self._host}/t/{topic_id}.json",
                headers=self._get_headers(),
            )
            response.raise_for_status()
            return response.json()

    async def search(
        self,
        query: str,
        category: str | None = None,
        solved: bool | None = None,
    ) -> list[dict]:
        """Search for topics.

        Args:
            query: Search query.
            category: Optional category filter.
            solved: Optional solved status filter.

        Returns:
            List of matching topics.
        """
        await self._rate_limit()

        search_query = query
        if category:
            search_query += f" category:{category}"
        if solved is not None:
            search_query += " status:solved" if solved else " -status:solved"

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self._host}/search.json",
                params={"q": search_query},
                headers=self._get_headers(),
            )
            response.raise_for_status()
            return response.json().get("topics", [])

    async def fetch_all_topics(
        self,
        categories: list[str] | None = None,
        max_topics: int = 10000,
    ) -> AsyncIterator[DiscourseArticle]:
        """Bulk fetch all topics for building knowledge base.

        Args:
            categories: Optional list of category slugs to fetch.
            max_topics: Maximum number of topics to fetch.

        Yields:
            DiscourseArticle objects.
        """
        fetched = 0

        # Get all categories
        all_categories = await self.fetch_categories()

        # Filter to specified categories or use all
        if categories:
            target_categories = [
                c for c in all_categories
                if c.get("slug") in categories
            ]
        else:
            target_categories = all_categories

        logger.info(f"Fetching topics from {len(target_categories)} categories")

        for category in target_categories:
            if fetched >= max_topics:
                break

            category_slug = category.get("slug", "")
            category_id = category.get("id", 0)
            category_name = category.get("name", "Unknown")
            logger.info(f"Fetching category: {category_name}")

            page = 0
            while fetched < max_topics:
                try:
                    topics_data = await self.fetch_topics_in_category(
                        category_slug, category_id, page
                    )
                except httpx.HTTPStatusError as e:
                    logger.warning(f"Error fetching category {category_slug}: {e}")
                    break

                topics = topics_data.get("topic_list", {}).get("topics", [])
                if not topics:
                    break

                for topic_summary in topics:
                    if fetched >= max_topics:
                        break

                    topic_id = topic_summary.get("id")
                    if not topic_id:
                        continue

                    try:
                        article = await self._fetch_article(
                            topic_id, topic_summary, category_name
                        )
                        if article:
                            fetched += 1
                            yield article

                            if fetched % 100 == 0:
                                logger.info(f"Fetched {fetched} topics")
                    except httpx.HTTPStatusError as e:
                        logger.warning(f"Error fetching topic {topic_id}: {e}")
                        continue

                page += 1

        logger.info(f"Finished fetching {fetched} topics")

    async def _fetch_article(
        self,
        topic_id: int,
        topic_summary: dict,
        category_name: str,
    ) -> DiscourseArticle | None:
        """Fetch and parse a single article.

        Args:
            topic_id: Topic ID.
            topic_summary: Topic summary from listing.
            category_name: Category name.

        Returns:
            DiscourseArticle or None if parsing fails.
        """
        full_topic = await self.fetch_topic(topic_id)

        # Get first post content
        posts = full_topic.get("post_stream", {}).get("posts", [])
        if not posts:
            return None

        first_post = posts[0]
        html_content = first_post.get("cooked", "")
        text_content = self._html_to_text(html_content)

        if not text_content:
            return None

        # Parse dates
        created_at = self._parse_date(topic_summary.get("created_at", ""))
        updated_at = self._parse_date(
            topic_summary.get("last_posted_at")
            or topic_summary.get("created_at", "")
        )

        slug = topic_summary.get("slug", "")
        url = f"{self._host}/t/{slug}/{topic_id}"

        return DiscourseArticle(
            topic_id=topic_id,
            post_id=first_post.get("id", 0),
            title=topic_summary.get("title", ""),
            slug=slug,
            category=category_name,
            content=text_content,
            content_html=html_content,
            url=url,
            created_at=created_at,
            updated_at=updated_at,
            view_count=topic_summary.get("views", 0),
            like_count=topic_summary.get("like_count", 0),
            reply_count=topic_summary.get("reply_count", 0) or topic_summary.get("posts_count", 1) - 1,
            is_solved=topic_summary.get("has_accepted_answer", False),
        )

    def _parse_date(self, date_str: str) -> datetime:
        """Parse ISO format date string.

        Args:
            date_str: ISO format date string.

        Returns:
            datetime object.
        """
        if not date_str:
            return datetime.now()

        try:
            # Handle ISO format with Z suffix
            date_str = date_str.replace("Z", "+00:00")
            return datetime.fromisoformat(date_str)
        except ValueError:
            return datetime.now()
