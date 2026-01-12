"""Mbox file parser using Python's mailbox module."""

import logging
import mailbox
import re
from dataclasses import dataclass
from datetime import datetime
from email.header import decode_header
from email.utils import parseaddr, parsedate_to_datetime
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


@dataclass
class ParsedEmail:
    """Structured representation of a parsed email."""

    message_id: str
    subject: str
    sender_email: str
    sender_name: str | None
    date_sent: datetime
    body_text: str
    body_html: str | None
    in_reply_to: str | None
    references: list[str]
    thread_id: str | None


class MboxParser:
    """Parse mbox files into structured email data."""

    def __init__(self, mbox_path: Path | str):
        """Initialize the parser.

        Args:
            mbox_path: Path to the mbox file.
        """
        self._mbox_path = Path(mbox_path)
        if not self._mbox_path.exists():
            raise FileNotFoundError(f"Mbox file not found: {self._mbox_path}")

    def parse_emails(self) -> Iterator[ParsedEmail]:
        """Iterate through all emails in the mbox file.

        Yields:
            ParsedEmail objects for each successfully parsed message.
        """
        mbox = mailbox.mbox(str(self._mbox_path))
        total = len(mbox)
        parsed = 0
        errors = 0

        logger.info(f"Parsing {total} messages from {self._mbox_path}")

        for i, message in enumerate(mbox):
            try:
                email = self._parse_message(message)
                if email:
                    parsed += 1
                    yield email
            except Exception as e:
                errors += 1
                logger.warning(f"Failed to parse message {i}: {e}")
                continue

        logger.info(f"Parsed {parsed} messages, {errors} errors")
        mbox.close()

    def _parse_message(self, message: mailbox.mboxMessage) -> ParsedEmail | None:
        """Parse a single email message.

        Args:
            message: The mbox message to parse.

        Returns:
            ParsedEmail object or None if parsing fails.
        """
        # Get message ID
        message_id = message.get("Message-ID", "")
        if not message_id:
            logger.debug("Skipping message without Message-ID")
            return None

        # Clean up message ID (remove angle brackets if present)
        message_id = message_id.strip().strip("<>")

        # Get sender info
        from_header = message.get("From", "")
        sender_name, sender_email = self._parse_address(from_header)
        if not sender_email:
            logger.debug(f"Skipping message without sender: {message_id}")
            return None

        # Get subject
        subject = self._decode_header(message.get("Subject", "(No Subject)"))

        # Get date
        date_sent = self._parse_date(message.get("Date", ""))
        if not date_sent:
            logger.debug(f"Skipping message without valid date: {message_id}")
            return None

        # Get body content
        body_text, body_html = self._extract_body(message)
        if not body_text:
            body_text = "(Empty body)"

        # Get threading info
        in_reply_to = message.get("In-Reply-To")
        if in_reply_to:
            in_reply_to = in_reply_to.strip().strip("<>")

        references_str = message.get("References", "")
        references = self._parse_references(references_str)

        # Compute thread ID from References or In-Reply-To
        # The first reference is typically the original message in the thread
        if references:
            thread_id = references[0]
        elif in_reply_to:
            thread_id = in_reply_to
        else:
            thread_id = message_id  # This is a new thread

        return ParsedEmail(
            message_id=message_id,
            subject=subject,
            sender_email=sender_email,
            sender_name=sender_name,
            date_sent=date_sent,
            body_text=body_text,
            body_html=body_html,
            in_reply_to=in_reply_to,
            references=references,
            thread_id=thread_id,
        )

    def _decode_header(self, header: str | None) -> str:
        """Decode an email header that may be encoded.

        Args:
            header: The header value to decode.

        Returns:
            Decoded string.
        """
        if not header:
            return ""

        try:
            decoded_parts = decode_header(header)
            result = []
            for data, charset in decoded_parts:
                if isinstance(data, bytes):
                    charset = charset or "utf-8"
                    try:
                        result.append(data.decode(charset, errors="replace"))
                    except (LookupError, UnicodeDecodeError):
                        result.append(data.decode("utf-8", errors="replace"))
                else:
                    result.append(data)
            return "".join(result)
        except Exception:
            return str(header)

    def _parse_address(self, address: str) -> tuple[str | None, str]:
        """Parse an email address header.

        Args:
            address: The address header value.

        Returns:
            Tuple of (name, email).
        """
        decoded = self._decode_header(address)
        name, email = parseaddr(decoded)
        return (name if name else None, email.lower())

    def _parse_date(self, date_str: str) -> datetime | None:
        """Parse an email date header.

        Args:
            date_str: The date header value.

        Returns:
            datetime object or None if parsing fails.
        """
        if not date_str:
            return None

        try:
            return parsedate_to_datetime(date_str)
        except (ValueError, TypeError):
            # Try some common alternative formats
            for fmt in [
                "%Y-%m-%d %H:%M:%S",
                "%d %b %Y %H:%M:%S",
                "%a, %d %b %Y %H:%M:%S",
            ]:
                try:
                    # Strip timezone info for simple parsing
                    clean_date = re.sub(r"\s*[+-]\d{4}.*$", "", date_str)
                    return datetime.strptime(clean_date.strip(), fmt)
                except ValueError:
                    continue
            logger.debug(f"Could not parse date: {date_str}")
            return None

    def _parse_references(self, references_str: str) -> list[str]:
        """Parse the References header into a list of message IDs.

        Args:
            references_str: The References header value.

        Returns:
            List of message IDs (without angle brackets).
        """
        if not references_str:
            return []

        # Split on whitespace and clean up
        refs = []
        for ref in references_str.split():
            ref = ref.strip().strip("<>")
            if ref and "@" in ref:  # Basic validation
                refs.append(ref)
        return refs

    def _extract_body(self, message: mailbox.mboxMessage) -> tuple[str, str | None]:
        """Extract plain text and HTML body from a message.

        Args:
            message: The email message.

        Returns:
            Tuple of (text_body, html_body).
        """
        body_text = ""
        body_html = None

        if message.is_multipart():
            for part in message.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition", ""))

                # Skip attachments
                if "attachment" in content_disposition:
                    continue

                try:
                    payload = part.get_payload(decode=True)
                    if payload is None:
                        continue

                    charset = part.get_content_charset() or "utf-8"
                    try:
                        decoded = payload.decode(charset, errors="replace")
                    except (LookupError, UnicodeDecodeError):
                        decoded = payload.decode("utf-8", errors="replace")

                    if content_type == "text/plain" and not body_text:
                        body_text = decoded
                    elif content_type == "text/html" and not body_html:
                        body_html = decoded
                except Exception as e:
                    logger.debug(f"Error extracting part: {e}")
                    continue
        else:
            try:
                payload = message.get_payload(decode=True)
                if payload:
                    charset = message.get_content_charset() or "utf-8"
                    try:
                        body_text = payload.decode(charset, errors="replace")
                    except (LookupError, UnicodeDecodeError):
                        body_text = payload.decode("utf-8", errors="replace")
            except Exception as e:
                logger.debug(f"Error extracting payload: {e}")

        # Clean up the text body
        body_text = self._clean_text(body_text)

        return body_text, body_html

    def _clean_text(self, text: str) -> str:
        """Clean up extracted text content.

        Args:
            text: Raw text content.

        Returns:
            Cleaned text.
        """
        if not text:
            return ""

        # Remove excessive whitespace
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

        return text
