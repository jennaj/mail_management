"""Command-line interface for Galaxy Mail Analyzer."""

import asyncio
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

app = typer.Typer(
    name="galaxy-mail",
    help="Analyze galaxy-bugs mailing list emails with AI-powered classification.",
)
console = Console()


@app.command()
def sync(
    start_date: Optional[str] = typer.Option(
        None,
        "--start",
        "-s",
        help="Start date (YYYY-MM-DD). Defaults to yesterday.",
    ),
    end_date: Optional[str] = typer.Option(
        None,
        "--end",
        "-e",
        help="End date (YYYY-MM-DD). Defaults to today.",
    ),
    mbox_file: Optional[Path] = typer.Option(
        None,
        "--mbox",
        "-m",
        help="Path to mbox file instead of downloading.",
    ),
):
    """Sync emails from HyperKitty archive or mbox file."""
    from .ingestion.hyperkitty import HyperKittyExporter
    from .ingestion.mbox_parser import MboxParser
    from .storage.database import get_db
    from .storage.models import Email
    from .config.settings import get_settings

    settings = get_settings()

    # Parse dates
    if start_date:
        start = date.fromisoformat(start_date)
    else:
        start = date.today() - timedelta(days=1)

    if end_date:
        end = date.fromisoformat(end_date)
    else:
        end = date.today()

    console.print(f"[bold]Syncing emails from {start} to {end}[/bold]")

    if mbox_file:
        # Use provided mbox file
        if not mbox_file.exists():
            console.print(f"[red]Error: File not found: {mbox_file}[/red]")
            raise typer.Exit(1)
        mbox_path = mbox_file
        console.print(f"Using mbox file: {mbox_path}")
    else:
        # Download from HyperKitty
        exporter = HyperKittyExporter()
        mbox_path = exporter.get_archive_path(start, end)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Downloading mbox archive...", total=None)
            mbox_path = exporter.download_archive_sync(mbox_path, start, end)

        console.print(f"[green]Downloaded to: {mbox_path}[/green]")

    # Parse emails
    parser = MboxParser(mbox_path)
    db = get_db()

    imported = 0
    skipped = 0
    errors = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Importing emails...", total=None)

        with db.session() as session:
            for parsed in parser.parse_emails():
                try:
                    # Check if email already exists
                    existing = session.query(Email).filter(
                        Email.message_id == parsed.message_id
                    ).first()

                    if existing:
                        skipped += 1
                        continue

                    # Create new email
                    email = Email(
                        message_id=parsed.message_id,
                        thread_id=parsed.thread_id,
                        in_reply_to=parsed.in_reply_to,
                        references=" ".join(parsed.references) if parsed.references else None,
                        sender_email=parsed.sender_email,
                        sender_name=parsed.sender_name,
                        subject=parsed.subject,
                        body_text=parsed.body_text,
                        body_html=parsed.body_html,
                        date_sent=parsed.date_sent,
                    )
                    session.add(email)
                    imported += 1

                    if imported % 100 == 0:
                        progress.update(task, description=f"Imported {imported} emails...")
                        session.commit()
                except Exception as e:
                    errors += 1
                    console.print(f"[yellow]Warning: {e}[/yellow]")

            session.commit()

    console.print(f"[green]Import complete![/green]")
    console.print(f"  Imported: {imported}")
    console.print(f"  Skipped (duplicates): {skipped}")
    console.print(f"  Errors: {errors}")


@app.command()
def analyze(
    date_str: Optional[str] = typer.Option(
        None,
        "--date",
        "-d",
        help="Analyze emails from this date (YYYY-MM-DD). Defaults to today.",
    ),
    email_id: Optional[int] = typer.Option(
        None,
        "--email",
        "-e",
        help="Analyze a specific email by ID.",
    ),
    limit: int = typer.Option(
        100,
        "--limit",
        "-l",
        help="Maximum number of emails to analyze.",
    ),
    use_kb: bool = typer.Option(
        True,
        "--kb/--no-kb",
        help="Use knowledge base for context.",
    ),
):
    """Analyze emails using Claude AI."""
    from .analysis.claude import ClaudeAnalyzer
    from .knowledge_base.indexer import KnowledgeBaseIndexer
    from .storage.database import get_db
    from .storage.models import Email, EmailStatus

    db = get_db()
    analyzer = ClaudeAnalyzer()
    kb_indexer = KnowledgeBaseIndexer() if use_kb else None

    def get_kb_context(text: str) -> list[dict]:
        if kb_indexer:
            return kb_indexer.search(text, n_results=3)
        return []

    with db.session() as session:
        if email_id:
            # Analyze specific email
            email_ids = [email_id]
        else:
            # Get pending emails
            query = session.query(Email).filter(
                Email.status == EmailStatus.PENDING
            )

            if date_str:
                target_date = date.fromisoformat(date_str)
                query = query.filter(
                    Email.date_sent >= datetime.combine(target_date, datetime.min.time()),
                    Email.date_sent < datetime.combine(target_date + timedelta(days=1), datetime.min.time()),
                )

            emails = query.order_by(Email.date_sent.desc()).limit(limit).all()
            email_ids = [e.id for e in emails]

    if not email_ids:
        console.print("[yellow]No emails to analyze.[/yellow]")
        return

    console.print(f"[bold]Analyzing {len(email_ids)} emails...[/bold]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing...", total=len(email_ids))

        results = analyzer.analyze_batch(
            email_ids,
            kb_context_fn=get_kb_context if use_kb else None,
        )

        for i, (eid, result) in enumerate(results):
            progress.update(task, completed=i + 1)
            if isinstance(result, Exception):
                console.print(f"[red]Error analyzing {eid}: {result}[/red]")

    # Print summary
    successful = [r for _, r in results if not isinstance(r, Exception)]
    console.print(f"\n[green]Analyzed {len(successful)} emails successfully.[/green]")

    # Show breakdown
    if successful:
        table = Table(title="Analysis Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="magenta")

        needs_human = sum(1 for r in successful if r.requires_human_attention)
        security = sum(1 for r in successful if r.is_security_related)
        auto = len(successful) - needs_human

        table.add_row("Total Analyzed", str(len(successful)))
        table.add_row("Auto-Answerable", str(auto))
        table.add_row("Needs Human", str(needs_human))
        table.add_row("Security Related", str(security))

        console.print(table)


@app.command()
def report(
    report_type: str = typer.Argument(
        ...,
        help="Report type: daily, weekly, monthly, or yearly.",
    ),
    date_str: Optional[str] = typer.Option(
        None,
        "--date",
        "-d",
        help="Date within the report period (YYYY-MM-DD). Defaults to today.",
    ),
):
    """Generate analysis report."""
    from .reporting.generators import ReportGenerator

    if report_type not in ("daily", "weekly", "monthly", "yearly"):
        console.print(f"[red]Invalid report type: {report_type}[/red]")
        console.print("Valid types: daily, weekly, monthly, yearly")
        raise typer.Exit(1)

    target_date = None
    if date_str:
        target_date = datetime.fromisoformat(date_str)

    console.print(f"[bold]Generating {report_type} report...[/bold]")

    generator = ReportGenerator()
    markdown, file_path = generator.generate_report(report_type, target_date)

    console.print(f"[green]Report saved to: {file_path}[/green]")
    console.print("\n" + "=" * 60 + "\n")
    console.print(markdown)


@app.command("kb-sync")
def kb_sync(
    categories: Optional[str] = typer.Option(
        None,
        "--categories",
        "-c",
        help="Comma-separated list of category slugs to sync.",
    ),
    max_topics: int = typer.Option(
        10000,
        "--max",
        "-m",
        help="Maximum number of topics to sync.",
    ),
):
    """Sync knowledge base from Discourse."""
    from .knowledge_base.indexer import KnowledgeBaseIndexer

    category_list = None
    if categories:
        category_list = [c.strip() for c in categories.split(",")]

    console.print("[bold]Syncing knowledge base from Discourse...[/bold]")

    indexer = KnowledgeBaseIndexer()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Fetching topics...", total=None)
        count = asyncio.run(indexer.sync_from_discourse(
            categories=category_list,
            max_topics=max_topics,
        ))

    console.print(f"[green]Synced {count} articles.[/green]")

    # Show stats
    stats = indexer.get_stats()
    table = Table(title="Knowledge Base Stats")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Total Articles", str(stats["total_articles"]))
    table.add_row("Solved Articles", str(stats["solved_articles"]))
    table.add_row("Categories", str(stats["categories"]))
    table.add_row("Indexed Embeddings", str(stats["indexed_embeddings"]))

    console.print(table)


@app.command("kb-search")
def kb_search(
    query: str = typer.Argument(..., help="Search query."),
    n_results: int = typer.Option(5, "--n", "-n", help="Number of results."),
    solved_only: bool = typer.Option(False, "--solved", help="Only show solved topics."),
):
    """Search the knowledge base."""
    from .knowledge_base.indexer import KnowledgeBaseIndexer

    indexer = KnowledgeBaseIndexer()
    results = indexer.search(query, n_results=n_results, solved_only=solved_only)

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    console.print(f"[bold]Found {len(results)} results:[/bold]\n")

    for i, result in enumerate(results, 1):
        solved = "[green][SOLVED][/green]" if result["is_solved"] else ""
        console.print(f"{i}. [bold]{result['title']}[/bold] {solved}")
        console.print(f"   Category: {result['category']}")
        console.print(f"   Similarity: {result['similarity']:.2%}")
        console.print(f"   URL: {result['url']}")
        console.print()


@app.command()
def cluster(
    refit: bool = typer.Option(
        False,
        "--refit",
        help="Refit the clustering model.",
    ),
):
    """Cluster emails by topic."""
    from .analysis.clustering import EmailClusterer

    console.print("[bold]Clustering emails...[/bold]")

    clusterer = EmailClusterer()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Running clustering...", total=None)
        result = clusterer.cluster_emails(refit=refit)

    if "error" in result:
        console.print(f"[red]Error: {result['error']}[/red]")
        return

    console.print(f"[green]Clustering complete![/green]")
    console.print(f"  Emails clustered: {result['n_emails']}")
    console.print(f"  Topics found: {result['n_topics']}")

    # Show clusters
    clusters = clusterer.get_cluster_summary()
    if clusters:
        table = Table(title="Top Topic Clusters")
        table.add_column("Cluster", style="cyan")
        table.add_column("Emails", style="magenta")
        table.add_column("Keywords", style="green")

        for cluster in clusters[:10]:
            keywords = ", ".join(cluster["keywords"][:5])
            table.add_row(cluster["name"], str(cluster["email_count"]), keywords)

        console.print(table)


@app.command()
def stats():
    """Show database statistics."""
    from .storage.database import get_db
    from .storage.models import Email, EmailAnalysis, EmailStatus, KnowledgeBaseArticle, TopicCluster
    from .storage.vector_store import get_vector_store

    db = get_db()
    vector_store = get_vector_store()

    with db.session() as session:
        total_emails = session.query(Email).count()
        analyzed = session.query(Email).filter(Email.status != EmailStatus.PENDING).count()
        needs_human = session.query(Email).filter(Email.status == EmailStatus.NEEDS_HUMAN).count()
        kb_articles = session.query(KnowledgeBaseArticle).count()
        clusters = session.query(TopicCluster).count()

    table = Table(title="Galaxy Mail Analyzer Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Total Emails", str(total_emails))
    table.add_row("Analyzed", str(analyzed))
    table.add_row("Needs Human Attention", str(needs_human))
    table.add_row("Knowledge Base Articles", str(kb_articles))
    table.add_row("Topic Clusters", str(clusters))
    table.add_row("Email Embeddings", str(vector_store.get_email_count()))
    table.add_row("KB Embeddings", str(vector_store.get_kb_count()))

    console.print(table)


@app.command("import-history")
def import_history(
    start_date: str = typer.Option(
        ...,
        "--start",
        "-s",
        help="Start date (YYYY-MM-DD).",
    ),
    end_date: Optional[str] = typer.Option(
        None,
        "--end",
        "-e",
        help="End date (YYYY-MM-DD). Defaults to today.",
    ),
):
    """Import historical emails from HyperKitty."""
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date) if end_date else date.today()

    console.print(f"[bold]Importing historical emails from {start} to {end}[/bold]")
    console.print("[yellow]This may take a while for large date ranges...[/yellow]")

    # Use sync command
    sync(start_date=start_date, end_date=end_date or end.isoformat(), mbox_file=None)


if __name__ == "__main__":
    app()
