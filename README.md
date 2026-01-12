# Galaxy Mail Analyzer

AI-powered analysis of the galaxy-bugs mailing list to classify issues, identify auto-answerable questions, and flag items requiring human attention.

## Features

- **Email Ingestion**: Fetch emails from HyperKitty (Mailman 3) archives or parse local mbox files
- **Knowledge Base Integration**: Sync and search help.galaxyproject.org Discourse forum
- **AI Classification**: Use Claude to classify issues (bug, question, security, etc.)
- **Topic Clustering**: Group similar emails using BERTopic
- **Smart Routing**: Identify which emails can be auto-answered vs need human attention
- **Reporting**: Generate daily, weekly, monthly, and yearly summaries

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/mail_management.git
cd mail_management

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"
```

## Configuration

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your API keys:
   ```bash
   ANTHROPIC_API_KEY=sk-ant-...    # Required: Claude API key
   VOYAGE_API_KEY=pa-...           # Required: Voyage AI key for embeddings
   ```

## Usage

### Sync Emails

Download and import emails from the mailing list:

```bash
# Sync yesterday's emails
galaxy-mail sync

# Sync specific date range
galaxy-mail sync --start 2025-01-01 --end 2025-01-10

# Import from local mbox file
galaxy-mail sync --mbox /path/to/archive.mbox
```

### Sync Knowledge Base

Import articles from help.galaxyproject.org:

```bash
# Sync all topics
galaxy-mail kb-sync

# Sync specific categories
galaxy-mail kb-sync --categories "support,tutorials"

# Limit number of topics
galaxy-mail kb-sync --max 1000
```

### Analyze Emails

Classify emails using AI:

```bash
# Analyze pending emails from today
galaxy-mail analyze

# Analyze specific date
galaxy-mail analyze --date 2025-01-10

# Analyze without knowledge base context
galaxy-mail analyze --no-kb
```

### Generate Reports

```bash
# Daily report
galaxy-mail report daily

# Weekly report for specific week
galaxy-mail report weekly --date 2025-01-10

# Monthly report
galaxy-mail report monthly

# Yearly summary
galaxy-mail report yearly
```

### Other Commands

```bash
# Search knowledge base
galaxy-mail kb-search "tool installation error"

# Cluster emails by topic
galaxy-mail cluster

# View statistics
galaxy-mail stats

# Import historical data
galaxy-mail import-history --start 2024-01-01
```

## Project Structure

```
mail_management/
├── config/
│   └── settings.py          # Configuration management
├── src/galaxy_mail_analyzer/
│   ├── cli.py               # Command-line interface
│   ├── ingestion/           # Email fetching and parsing
│   ├── knowledge_base/      # Discourse integration
│   ├── analysis/            # AI classification and clustering
│   ├── storage/             # Database and vector store
│   └── reporting/           # Report generation
├── data/                    # Local data (gitignored)
│   ├── mbox/               # Downloaded mbox files
│   ├── chroma/             # Vector database
│   └── reports/            # Generated reports
└── tests/                   # Test suite
```

## Programmatic Email Access

### Method 1: HyperKitty Export (Recommended)

The tool automatically downloads mbox archives from HyperKitty:

```
https://lists.galaxyproject.org/archives/list/galaxy-bugs.lists.galaxyproject.org/export/galaxy-bugs.lists.galaxyproject.org.mbox.gz?start=2025-01-01&end=2025-01-10
```

### Method 2: Direct Mbox File

Ask your sysadmin for mbox exports from:
```
/var/lib/mailman3/archives/
```

Then import:
```bash
galaxy-mail sync --mbox /path/to/galaxy-bugs.mbox
```

## API Costs

Estimated costs per 1000 emails:

| Service | Cost |
|---------|------|
| Claude (claude-sonnet-4-5-20250929) | ~$1.50 |
| Voyage AI (voyage-3.5) | ~$0.06 |
| **Total** | ~$1.56 |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy src/

# Linting
ruff check src/
```

## License

MIT
