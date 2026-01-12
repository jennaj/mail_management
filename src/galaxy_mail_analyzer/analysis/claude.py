"""Claude AI analyzer for email classification."""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import anthropic

from config.settings import get_settings
from ..storage.database import get_db
from ..storage.models import Email, EmailAnalysis, EmailStatus, IssueType

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Result of email analysis."""

    summary: str
    issue_type: IssueType
    is_security_related: bool
    requires_human_attention: bool
    confidence_score: float
    reasoning: str
    suggested_response: str | None = None
    matched_kb_article_ids: list[int] | None = None


class ClaudeAnalyzer:
    """Analyze emails using Claude API."""

    SYSTEM_PROMPT = """You are an expert at analyzing technical support emails for the Galaxy Project bioinformatics platform. Your task is to classify emails from the galaxy-bugs mailing list.

For each email, you must:

1. **Classify the issue type** - Choose ONE of:
   - bug: Software defect, error, or unexpected behavior
   - question: User asking how to do something
   - feature_request: Request for new functionality
   - security: Security vulnerability, data exposure, authentication issues
   - documentation: Issues with docs, tutorials, or guides
   - installation: Installation, deployment, or configuration issues
   - performance: Slow performance, memory issues, timeouts
   - other: Doesn't fit other categories

2. **Detect security issues** - Flag as security-related if it mentions:
   - Data exposure or leaks
   - Authentication/authorization problems
   - Credential issues
   - Unauthorized access
   - Privacy concerns
   BE CONSERVATIVE: When in doubt, flag as security-related.

3. **Determine if human attention is needed** - Flag for human review if:
   - It's security-related
   - It involves data loss or corruption
   - The issue is complex or unclear
   - It's from a known contributor or maintainer
   - It requires access to internal systems
   - You're not confident in your classification

4. **Provide a brief summary** - 1-2 sentences describing the issue.

5. **Suggest a response** (optional) - If the issue can be answered with information from the knowledge base, suggest a helpful response.

Respond ONLY with valid JSON in this exact format:
{
    "summary": "Brief summary of the issue",
    "issue_type": "bug|question|feature_request|security|documentation|installation|performance|other",
    "is_security_related": true/false,
    "requires_human_attention": true/false,
    "confidence_score": 0.0-1.0,
    "suggested_response": "Suggested response text or null",
    "matched_kb_article_ids": [list of relevant article IDs or empty array],
    "reasoning": "Brief explanation of your classification"
}"""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        """Initialize the analyzer.

        Args:
            api_key: Anthropic API key.
            model: Claude model to use.
        """
        settings = get_settings()
        self._api_key = api_key or settings.anthropic_api_key
        self._model = model or settings.claude_model

        self._client = anthropic.Anthropic(api_key=self._api_key)
        self._db = get_db()

    def analyze_email(
        self,
        email_id: int,
        kb_context: list[dict] | None = None,
        thread_context: list[str] | None = None,
    ) -> AnalysisResult:
        """Analyze a single email.

        Args:
            email_id: Email ID to analyze.
            kb_context: Optional knowledge base context.
            thread_context: Optional previous messages in thread.

        Returns:
            Analysis result.
        """
        with self._db.session() as session:
            email = session.query(Email).filter(Email.id == email_id).first()
            if not email:
                raise ValueError(f"Email not found: {email_id}")

            # Build user message
            user_message = self._build_prompt(
                subject=email.subject,
                body=email.body_text,
                sender=email.sender_email,
                kb_context=kb_context,
                thread_context=thread_context,
            )

            # Call Claude
            response = self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )

            # Parse response
            response_text = response.content[0].text
            result = self._parse_response(response_text)

            # Save to database
            self._save_analysis(session, email, result)

            return result

    def analyze_batch(
        self,
        email_ids: list[int],
        kb_context_fn: Any | None = None,
    ) -> list[tuple[int, AnalysisResult | Exception]]:
        """Analyze a batch of emails.

        Args:
            email_ids: List of email IDs to analyze.
            kb_context_fn: Optional function to get KB context for each email.

        Returns:
            List of (email_id, result or exception) tuples.
        """
        results = []

        for email_id in email_ids:
            try:
                # Get KB context if function provided
                kb_context = None
                if kb_context_fn:
                    with self._db.session() as session:
                        email = session.query(Email).filter(Email.id == email_id).first()
                        if email:
                            kb_context = kb_context_fn(email.body_text)

                result = self.analyze_email(email_id, kb_context=kb_context)
                results.append((email_id, result))
                logger.info(f"Analyzed email {email_id}: {result.issue_type.value}")
            except Exception as e:
                logger.error(f"Error analyzing email {email_id}: {e}")
                results.append((email_id, e))

        return results

    def _build_prompt(
        self,
        subject: str,
        body: str,
        sender: str,
        kb_context: list[dict] | None = None,
        thread_context: list[str] | None = None,
    ) -> str:
        """Build the analysis prompt.

        Args:
            subject: Email subject.
            body: Email body.
            sender: Sender email address.
            kb_context: Optional KB context.
            thread_context: Optional thread context.

        Returns:
            Formatted prompt string.
        """
        parts = [
            "Analyze this email from the galaxy-bugs mailing list:",
            "",
            f"Subject: {subject}",
            f"From: {sender}",
            "",
            "Body:",
            body[:5000],  # Limit body length
        ]

        if thread_context:
            parts.extend([
                "",
                "Previous messages in thread:",
                "---",
                "\n---\n".join(thread_context[-3:]),  # Last 3 messages
            ])

        if kb_context:
            kb_text = "\n".join([
                f"- [{a.get('title', 'Untitled')}]({a.get('url', '')}): {a.get('excerpt', '')[:200]}"
                for a in kb_context[:5]
            ])
            parts.extend([
                "",
                "Potentially relevant knowledge base articles:",
                kb_text,
            ])

        return "\n".join(parts)

    def _parse_response(self, response_text: str) -> AnalysisResult:
        """Parse Claude's JSON response.

        Args:
            response_text: Raw response text.

        Returns:
            Parsed AnalysisResult.
        """
        try:
            # Extract JSON from response
            response_text = response_text.strip()
            if response_text.startswith("```"):
                # Handle code block wrapped response
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1])

            data = json.loads(response_text)

            # Map issue type
            issue_type_str = data.get("issue_type", "other").lower()
            try:
                issue_type = IssueType(issue_type_str)
            except ValueError:
                issue_type = IssueType.OTHER

            return AnalysisResult(
                summary=data.get("summary", ""),
                issue_type=issue_type,
                is_security_related=data.get("is_security_related", False),
                requires_human_attention=data.get("requires_human_attention", True),
                confidence_score=float(data.get("confidence_score", 0.5)),
                reasoning=data.get("reasoning", ""),
                suggested_response=data.get("suggested_response"),
                matched_kb_article_ids=data.get("matched_kb_article_ids", []),
            )
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Claude response: {e}")
            # Return conservative default
            return AnalysisResult(
                summary="Failed to parse AI analysis",
                issue_type=IssueType.OTHER,
                is_security_related=False,
                requires_human_attention=True,
                confidence_score=0.0,
                reasoning=f"Parse error: {e}",
            )

    def _save_analysis(
        self,
        session: Any,
        email: Email,
        result: AnalysisResult,
    ) -> None:
        """Save analysis result to database.

        Args:
            session: Database session.
            email: Email object.
            result: Analysis result.
        """
        # Check if analysis exists
        analysis = session.query(EmailAnalysis).filter(
            EmailAnalysis.email_id == email.id
        ).first()

        kb_ids = json.dumps(result.matched_kb_article_ids or [])

        if analysis:
            # Update existing
            analysis.summary = result.summary
            analysis.issue_type = result.issue_type
            analysis.is_security_related = result.is_security_related
            analysis.requires_human_attention = result.requires_human_attention
            analysis.confidence_score = result.confidence_score
            analysis.reasoning = result.reasoning
            analysis.suggested_response = result.suggested_response
            analysis.matched_kb_article_ids = kb_ids
            analysis.model_version = self._model
            analysis.analyzed_at = datetime.utcnow()
        else:
            # Create new
            analysis = EmailAnalysis(
                email_id=email.id,
                summary=result.summary,
                issue_type=result.issue_type,
                is_security_related=result.is_security_related,
                requires_human_attention=result.requires_human_attention,
                confidence_score=result.confidence_score,
                reasoning=result.reasoning,
                suggested_response=result.suggested_response,
                matched_kb_article_ids=kb_ids,
                model_version=self._model,
            )
            session.add(analysis)

        # Update email status
        if result.requires_human_attention:
            email.status = EmailStatus.NEEDS_HUMAN
        else:
            email.status = EmailStatus.ANALYZED

        email.date_analyzed = datetime.utcnow()

        session.commit()
