"""HyperKitty mbox export downloader."""

import gzip
import logging
from datetime import date
from pathlib import Path

import httpx

from config.settings import get_settings

logger = logging.getLogger(__name__)


class HyperKittyExporter:
    """Download mbox archives from HyperKitty."""

    def __init__(
        self,
        base_url: str | None = None,
        list_address: str | None = None,
    ):
        """Initialize the HyperKitty exporter.

        Args:
            base_url: HyperKitty base URL (e.g., https://lists.galaxyproject.org)
            list_address: Mailing list address (e.g., galaxy-bugs@lists.galaxyproject.org)
        """
        settings = get_settings()
        self._base_url = (base_url or settings.hyperkitty_base_url).rstrip("/")
        self._list_address = list_address or settings.mailing_list_address
        self._list_id = self._list_address.replace("@", ".")

    def get_export_url(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> str:
        """Construct the mbox export URL with optional date range.

        Args:
            start_date: Start date for the export (inclusive).
            end_date: End date for the export (inclusive).

        Returns:
            Full URL for mbox export.
        """
        url = (
            f"{self._base_url}/archives/list/{self._list_id}/"
            f"export/{self._list_id}.mbox.gz"
        )

        params = []
        if start_date:
            params.append(f"start={start_date.isoformat()}")
        if end_date:
            params.append(f"end={end_date.isoformat()}")

        if params:
            url += "?" + "&".join(params)
        return url

    async def download_archive(
        self,
        output_path: Path,
        start_date: date | None = None,
        end_date: date | None = None,
        timeout: float = 300.0,
    ) -> Path:
        """Download and decompress mbox archive.

        Args:
            output_path: Path to save the decompressed mbox file.
            start_date: Start date for the export (inclusive).
            end_date: End date for the export (inclusive).
            timeout: Request timeout in seconds (default 5 minutes).

        Returns:
            Path to the downloaded mbox file.

        Raises:
            httpx.HTTPStatusError: If the download fails.
        """
        url = self.get_export_url(start_date, end_date)
        logger.info(f"Downloading mbox from: {url}")

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()

            # Check if response is gzipped
            content_type = response.headers.get("content-type", "")
            if "gzip" in content_type or url.endswith(".gz"):
                logger.info("Decompressing gzipped content...")
                try:
                    decompressed = gzip.decompress(response.content)
                    output_path.write_bytes(decompressed)
                except gzip.BadGzipFile:
                    # Not actually gzipped, save as-is
                    logger.warning("Content not gzipped despite extension, saving raw")
                    output_path.write_bytes(response.content)
            else:
                output_path.write_bytes(response.content)

            logger.info(f"Saved mbox to: {output_path}")
            return output_path

    def download_archive_sync(
        self,
        output_path: Path,
        start_date: date | None = None,
        end_date: date | None = None,
        timeout: float = 300.0,
    ) -> Path:
        """Synchronous version of download_archive.

        Args:
            output_path: Path to save the decompressed mbox file.
            start_date: Start date for the export (inclusive).
            end_date: End date for the export (inclusive).
            timeout: Request timeout in seconds (default 5 minutes).

        Returns:
            Path to the downloaded mbox file.
        """
        url = self.get_export_url(start_date, end_date)
        logger.info(f"Downloading mbox from: {url}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            if "gzip" in content_type or url.endswith(".gz"):
                logger.info("Decompressing gzipped content...")
                try:
                    decompressed = gzip.decompress(response.content)
                    output_path.write_bytes(decompressed)
                except gzip.BadGzipFile:
                    logger.warning("Content not gzipped despite extension, saving raw")
                    output_path.write_bytes(response.content)
            else:
                output_path.write_bytes(response.content)

            logger.info(f"Saved mbox to: {output_path}")
            return output_path

    def get_archive_path(self, start_date: date, end_date: date) -> Path:
        """Generate a standard archive file path based on date range.

        Args:
            start_date: Start date.
            end_date: End date.

        Returns:
            Path for the mbox file.
        """
        settings = get_settings()
        filename = f"{self._list_id}_{start_date.isoformat()}_{end_date.isoformat()}.mbox"
        return settings.mbox_dir / filename
