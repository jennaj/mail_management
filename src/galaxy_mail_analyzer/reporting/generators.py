"""Report generation for email analysis summaries."""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from sqlalchemy import func

from config.settings import get_settings
from ..storage.database import get_db
from ..storage.models import (
    AnalysisReport,
    Email,
    EmailAnalysis,
    EmailStatus,
    IssueType,
    TopicCluster,
)

logger = logging.getLogger(__name__)

ReportType = Literal["daily", "weekly", "monthly", "yearly"]


class ReportGenerator:
    """Generate analysis reports in Markdown format."""

    def __init__(self):
        """Initialize the report generator."""
        self._db = get_db()
        self._settings = get_settings()

    def generate_report(
        self,
        report_type: ReportType,
        target_date: datetime | None = None,
    ) -> tuple[str, Path]:
        """Generate a report for the specified period.

        Args:
            report_type: Type of report (daily, weekly, monthly, yearly).
            target_date: Date within the report period. Defaults to today.

        Returns:
            Tuple of (markdown content, file path).
        """
        if target_date is None:
            target_date = datetime.now()

        # Calculate period boundaries
        start_date, end_date = self._get_period_bounds(report_type, target_date)

        logger.info(f"Generating {report_type} report for {start_date.date()} to {end_date.date()}")

        # Gather statistics
        stats = self._gather_stats(start_date, end_date)

        # Generate markdown
        markdown = self._generate_markdown(report_type, start_date, end_date, stats)

        # Save report
        file_path = self._save_report(report_type, start_date, end_date, stats, markdown)

        return markdown, file_path

    def _get_period_bounds(
        self,
        report_type: ReportType,
        target_date: datetime,
    ) -> tuple[datetime, datetime]:
        """Calculate the start and end dates for a report period.

        Args:
            report_type: Type of report.
            target_date: Date within the period.

        Returns:
            Tuple of (start_date, end_date).
        """
        if report_type == "daily":
            start = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1) - timedelta(microseconds=1)
        elif report_type == "weekly":
            # Week starts on Monday
            start = target_date - timedelta(days=target_date.weekday())
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=7) - timedelta(microseconds=1)
        elif report_type == "monthly":
            start = target_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            # Get last day of month
            if target_date.month == 12:
                end = target_date.replace(year=target_date.year + 1, month=1, day=1)
            else:
                end = target_date.replace(month=target_date.month + 1, day=1)
            end = end - timedelta(microseconds=1)
        elif report_type == "yearly":
            start = target_date.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            end = target_date.replace(month=12, day=31, hour=23, minute=59, second=59)
        else:
            raise ValueError(f"Unknown report type: {report_type}")

        return start, end

    def _gather_stats(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> dict:
        """Gather statistics for the report period.

        Args:
            start_date: Period start.
            end_date: Period end.

        Returns:
            Dictionary of statistics.
        """
        with self._db.session() as session:
            # Base query for emails in period
            base_query = session.query(Email).filter(
                Email.date_sent >= start_date,
                Email.date_sent <= end_date,
            )

            # Total emails
            total_emails = base_query.count()

            # By status
            status_counts = {}
            for status in EmailStatus:
                count = base_query.filter(Email.status == status).count()
                status_counts[status.value] = count

            # By issue type (from analyses)
            issue_type_counts = {}
            for issue_type in IssueType:
                count = session.query(EmailAnalysis).join(Email).filter(
                    Email.date_sent >= start_date,
                    Email.date_sent <= end_date,
                    EmailAnalysis.issue_type == issue_type,
                ).count()
                issue_type_counts[issue_type.value] = count

            # Security issues
            security_count = session.query(EmailAnalysis).join(Email).filter(
                Email.date_sent >= start_date,
                Email.date_sent <= end_date,
                EmailAnalysis.is_security_related == True,
            ).count()

            # Needs human attention
            needs_human = session.query(EmailAnalysis).join(Email).filter(
                Email.date_sent >= start_date,
                Email.date_sent <= end_date,
                EmailAnalysis.requires_human_attention == True,
            ).count()

            # Auto-answered (could be answered by AI)
            auto_answerable = session.query(EmailAnalysis).join(Email).filter(
                Email.date_sent >= start_date,
                Email.date_sent <= end_date,
                EmailAnalysis.requires_human_attention == False,
                EmailAnalysis.suggested_response.isnot(None),
            ).count()

            # Top senders
            top_senders = session.query(
                Email.sender_email,
                func.count(Email.id).label("count"),
            ).filter(
                Email.date_sent >= start_date,
                Email.date_sent <= end_date,
            ).group_by(Email.sender_email).order_by(
                func.count(Email.id).desc()
            ).limit(10).all()

            # Top clusters
            top_clusters = session.query(TopicCluster).order_by(
                TopicCluster.email_count.desc()
            ).limit(10).all()

            # Emails needing attention
            attention_emails = session.query(Email).join(EmailAnalysis).filter(
                Email.date_sent >= start_date,
                Email.date_sent <= end_date,
                EmailAnalysis.requires_human_attention == True,
            ).order_by(Email.date_sent.desc()).limit(20).all()

            return {
                "total_emails": total_emails,
                "by_status": status_counts,
                "by_issue_type": issue_type_counts,
                "security_issues": security_count,
                "needs_human": needs_human,
                "auto_answerable": auto_answerable,
                "top_senders": [(s.sender_email, s.count) for s in top_senders],
                "top_clusters": [
                    {"name": c.name, "count": c.email_count, "keywords": json.loads(c.keywords) if c.keywords else []}
                    for c in top_clusters
                ],
                "attention_emails": [
                    {
                        "id": e.id,
                        "subject": e.subject,
                        "sender": e.sender_email,
                        "date": e.date_sent.isoformat(),
                        "analysis": {
                            "summary": e.analysis.summary if e.analysis else "",
                            "issue_type": e.analysis.issue_type.value if e.analysis else "",
                            "is_security": e.analysis.is_security_related if e.analysis else False,
                        } if e.analysis else None,
                    }
                    for e in attention_emails
                ],
            }

    def _generate_markdown(
        self,
        report_type: ReportType,
        start_date: datetime,
        end_date: datetime,
        stats: dict,
    ) -> str:
        """Generate Markdown report content.

        Args:
            report_type: Type of report.
            start_date: Period start.
            end_date: Period end.
            stats: Statistics dictionary.

        Returns:
            Markdown formatted report.
        """
        title_map = {
            "daily": "Daily",
            "weekly": "Weekly",
            "monthly": "Monthly",
            "yearly": "Yearly",
        }

        lines = [
            f"# Galaxy Bugs {title_map[report_type]} Report",
            "",
            f"**Period:** {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            "",
            "## Summary",
            "",
            f"- **Total Emails:** {stats['total_emails']}",
            f"- **Security Issues:** {stats['security_issues']}",
            f"- **Needs Human Attention:** {stats['needs_human']}",
            f"- **Auto-Answerable:** {stats['auto_answerable']}",
            "",
        ]

        # Calculate auto-answer rate
        if stats['total_emails'] > 0:
            rate = (stats['auto_answerable'] / stats['total_emails']) * 100
            lines.append(f"**Auto-Answer Rate:** {rate:.1f}%")
            lines.append("")

        # Issue Type Breakdown
        lines.extend([
            "## Issue Type Breakdown",
            "",
            "| Type | Count |",
            "|------|-------|",
        ])
        for issue_type, count in sorted(stats['by_issue_type'].items(), key=lambda x: -x[1]):
            if count > 0:
                lines.append(f"| {issue_type.replace('_', ' ').title()} | {count} |")
        lines.append("")

        # Status Breakdown
        lines.extend([
            "## Status Breakdown",
            "",
            "| Status | Count |",
            "|--------|-------|",
        ])
        for status, count in sorted(stats['by_status'].items(), key=lambda x: -x[1]):
            if count > 0:
                lines.append(f"| {status.replace('_', ' ').title()} | {count} |")
        lines.append("")

        # Top Clusters
        if stats['top_clusters']:
            lines.extend([
                "## Top Topic Clusters",
                "",
            ])
            for i, cluster in enumerate(stats['top_clusters'][:5], 1):
                keywords = ", ".join(cluster['keywords'][:5]) if cluster['keywords'] else "N/A"
                lines.append(f"{i}. **{cluster['name']}** ({cluster['count']} emails)")
                lines.append(f"   - Keywords: {keywords}")
            lines.append("")

        # Top Senders
        if stats['top_senders']:
            lines.extend([
                "## Top Senders",
                "",
                "| Sender | Count |",
                "|--------|-------|",
            ])
            for sender, count in stats['top_senders'][:10]:
                lines.append(f"| {sender} | {count} |")
            lines.append("")

        # Emails Needing Attention
        if stats['attention_emails']:
            lines.extend([
                "## Emails Requiring Human Attention",
                "",
            ])
            for email in stats['attention_emails'][:10]:
                analysis = email.get('analysis', {})
                security_badge = " [SECURITY]" if analysis and analysis.get('is_security') else ""
                lines.extend([
                    f"### {email['subject'][:80]}{security_badge}",
                    "",
                    f"- **From:** {email['sender']}",
                    f"- **Date:** {email['date'][:10]}",
                    f"- **Type:** {analysis.get('issue_type', 'Unknown') if analysis else 'Not analyzed'}",
                    f"- **Summary:** {analysis.get('summary', 'N/A') if analysis else 'N/A'}",
                    "",
                ])

        # Footer
        lines.extend([
            "---",
            "",
            "*Report generated by Galaxy Mail Analyzer*",
        ])

        return "\n".join(lines)

    def _save_report(
        self,
        report_type: ReportType,
        start_date: datetime,
        end_date: datetime,
        stats: dict,
        markdown: str,
    ) -> Path:
        """Save report to file and database.

        Args:
            report_type: Type of report.
            start_date: Period start.
            end_date: Period end.
            stats: Statistics dictionary.
            markdown: Markdown content.

        Returns:
            Path to saved file.
        """
        # Save to file
        reports_dir = self._settings.reports_dir
        reports_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{report_type}-{start_date.strftime('%Y-%m-%d')}.md"
        file_path = reports_dir / filename
        file_path.write_text(markdown)

        logger.info(f"Saved report to: {file_path}")

        # Save to database
        with self._db.session() as session:
            # Check if report exists
            existing = session.query(AnalysisReport).filter(
                AnalysisReport.report_type == report_type,
                AnalysisReport.period_start == start_date,
            ).first()

            if existing:
                # Update
                existing.period_end = end_date
                existing.total_emails = stats['total_emails']
                existing.auto_answered = stats['auto_answerable']
                existing.needs_human = stats['needs_human']
                existing.security_issues = stats['security_issues']
                existing.by_issue_type = json.dumps(stats['by_issue_type'])
                existing.top_clusters = json.dumps(stats['top_clusters'])
                existing.report_markdown = markdown
                existing.generated_at = datetime.utcnow()
            else:
                # Create new
                report = AnalysisReport(
                    report_type=report_type,
                    period_start=start_date,
                    period_end=end_date,
                    total_emails=stats['total_emails'],
                    auto_answered=stats['auto_answerable'],
                    needs_human=stats['needs_human'],
                    security_issues=stats['security_issues'],
                    by_issue_type=json.dumps(stats['by_issue_type']),
                    top_clusters=json.dumps(stats['top_clusters']),
                    report_markdown=markdown,
                )
                session.add(report)

            session.commit()

        return file_path

    def list_reports(self, report_type: ReportType | None = None) -> list[dict]:
        """List generated reports.

        Args:
            report_type: Optional filter by report type.

        Returns:
            List of report summaries.
        """
        with self._db.session() as session:
            query = session.query(AnalysisReport)
            if report_type:
                query = query.filter(AnalysisReport.report_type == report_type)

            reports = query.order_by(AnalysisReport.period_start.desc()).all()

            return [
                {
                    "id": r.id,
                    "type": r.report_type,
                    "period_start": r.period_start.isoformat(),
                    "period_end": r.period_end.isoformat(),
                    "total_emails": r.total_emails,
                    "generated_at": r.generated_at.isoformat(),
                }
                for r in reports
            ]
