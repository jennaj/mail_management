# Galaxy Mail Analyzer

AI-powered analysis of the galaxy-bugs mailing list to classify issues, identify auto-answerable questions, and flag items requiring human attention.

## Features

- **Email Ingestion**: Fetch emails from HyperKitty (Mailman 3) archives or parse local mbox files
- **Knowledge Base Integration**: Sync and search help.galaxyproject.org Discourse forum
- **GTN FAQ Integration**: Compare with Galaxy Training Network FAQs
- **AI Classification**: Use Claude to classify issues (bug, question, security, etc.)
- **Topic Clustering**: Group similar emails using BERTopic
- **Smart Routing**: Identify which emails can be auto-answered vs need human attention
- **Coverage Analysis**: Identify documentation gaps between bugs and existing help resources
- **Reporting**: Generate daily, weekly, monthly, and yearly summaries
- **Interactive Dashboards**: HTML dashboards for visualizing trends and gaps

## Quick Start

```bash
# Clone the repository
git clone https://github.com/jennaj/mail_management.git
cd mail_management

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .

# Copy and configure environment
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY
```

## Data Storage

### Local Data (Not in Git)

Data files are stored locally and excluded from version control:

```
~/bugs-mbox/YYYYMMDD_download/
├── galaxy-bugs-2021.mbox      # Raw mbox by year
├── galaxy-bugs-2022.mbox
├── galaxy-bugs-2023.mbox
├── galaxy-bugs-2024.mbox
├── galaxy-bugs-2025.mbox
├── bugs_analysis_YYYYMMDD.json        # Categorized bug topics
├── comparison_analysis_YYYYMMDD.json  # Cross-source gap analysis
├── gtn-faqs/
│   └── gtn_faqs_YYYYMMDD.json         # GTN FAQ content
└── discourse-help/
    └── discourse_topics_YYYYMMDD.json # Help forum topics
```

### Reports (In Git)

Generated HTML dashboards are tracked in version control:

```
reports/
├── mbox-dashboard.html        # Tool analysis from bug emails
├── coverage-comparison.html   # Gap analysis across sources
├── dashboard.html             # System status
└── insights.html              # Data insights
```

## Dashboards

### Mbox Analysis Dashboard (`reports/mbox-dashboard.html`)

Analyzes bug email content:
- **24,351 emails** from 2021-2025
- Top 25 tools generating issues
- Tools grouped by category (RNA-seq, Assembly, QC, etc.)
- Issue type distribution

### Coverage Comparison Dashboard (`reports/coverage-comparison.html`)

Compares three knowledge sources:
- **Bug emails** (galaxy-bugs mailing list)
- **Help forum** (help.galaxyproject.org Discourse)
- **GTN FAQs** (training.galaxyproject.org)

Identifies:
- Documentation gaps (topics in bugs but not in help/FAQs)
- Tools with no help coverage
- Priority areas for new FAQ development

## Key Findings

### Documentation Gaps (from Coverage Analysis)

| Category | Bug Reports | Help Resources | Gap |
|----------|-------------|----------------|-----|
| File Formats | 1,738 | 110 | 16x under-documented |
| History & Datasets | 1,040 | 81 | 13x under-documented |
| Storage & Quota | 617 | 16 | 39x under-documented |

### Top Tools Needing FAQs

1. **DESeq2** - 489 bug mentions, 0 FAQs
2. **Trinity** - 451 bug mentions, 0 FAQs
3. **Bowtie2** - 375 bug mentions, 0 FAQs
4. **HISAT2** - 368 bug mentions, 0 FAQs
5. **Trimmomatic** - 348 bug mentions, 0 FAQs

### Tools with No Help Forum Coverage

SPAdes (197), Flye (108), Prokka (78), edgeR (65), minimap2 (65), Circos (63), Roary (57)

## Usage

### Download Mbox Archives

The HyperKitty export requires authentication:

```python
# Downloads are saved to ~/bugs-mbox/YYYYMMDD_download/
# Use date-range exports to avoid timeouts:
# ?start=2025-01-01&end=2026-01-01
```

### Sync Knowledge Base

```bash
# Import from help.galaxyproject.org
galaxy-mail kb-sync --max 200

# Search knowledge base
galaxy-mail kb-search "tool installation error"
```

### Generate Analysis

```bash
# View statistics
galaxy-mail stats

# Generate reports
galaxy-mail report daily
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `galaxy-mail sync` | Download and import emails |
| `galaxy-mail kb-sync` | Sync Discourse help topics |
| `galaxy-mail kb-search` | Search knowledge base |
| `galaxy-mail analyze` | Classify emails with AI |
| `galaxy-mail cluster` | Group emails by topic |
| `galaxy-mail report` | Generate summaries |
| `galaxy-mail stats` | View statistics |

## Project Structure

```
mail_management/
├── pyproject.toml              # Dependencies and project config
├── .env.example                # Environment template
├── README.md                   # This file
├── reports/                    # Generated HTML dashboards
│   ├── mbox-dashboard.html
│   ├── coverage-comparison.html
│   ├── dashboard.html
│   └── insights.html
├── src/galaxy_mail_analyzer/
│   ├── cli.py                  # Typer CLI commands
│   ├── config/
│   │   └── settings.py         # Pydantic settings
│   ├── ingestion/
│   │   ├── hyperkitty.py       # HyperKitty mbox downloader
│   │   └── mbox_parser.py      # Email parsing
│   ├── knowledge_base/
│   │   ├── discourse.py        # Discourse API client
│   │   └── indexer.py          # Embedding indexer
│   ├── analysis/
│   │   ├── embeddings.py       # sentence-transformers embeddings
│   │   ├── clustering.py       # BERTopic clustering
│   │   └── claude.py           # Claude AI analysis
│   ├── storage/
│   │   ├── models.py           # SQLAlchemy models
│   │   ├── database.py         # DB connection
│   │   └── vector_store.py     # ChromaDB
│   └── reporting/
│       └── generators.py       # Report generation
├── data/                       # Local data (gitignored)
└── tests/                      # Test suite
```

## Configuration

### Environment Variables

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...    # Claude API key

# Optional (defaults provided)
DATABASE_URL=sqlite:///./data/galaxy_mail.db
CHROMA_PERSIST_DIR=./data/chroma
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### Embedding Model

Uses **sentence-transformers** with `all-MiniLM-L6-v2` for free, local embeddings (no API costs).

## Incremental Updates

Data files are date-stamped for incremental updates:

```
bugs_analysis_20260113.json
comparison_analysis_20260113.json
gtn_faqs_20260113.json
discourse_topics_20260113.json
```

To update, run the analysis again - new files will be created with the current date, preserving history.

## Use Cases

### 1. Drafting Email Replies

Use the coverage analysis to find relevant help resources:
- Search Discourse topics for matching issues
- Link to GTN FAQs for common questions
- Identify when no help exists (flag for human attention)

### 2. Building FAQs

Priority areas identified from gap analysis:
1. File format troubleshooting (FASTQ, BAM, BED, VCF)
2. Storage/quota management
3. Tool-specific guides (DESeq2, Trinity, assembly tools)
4. Job queue explanations

### 3. Trend Analysis

Track issue patterns over time:
- Which tools generate most support requests
- Seasonal patterns in user questions
- Emerging issues with new tools

## API Costs

| Service | Usage | Cost |
|---------|-------|------|
| Claude (analysis) | ~500K tokens/1000 emails | ~$1.50 |
| Embeddings | Local (sentence-transformers) | Free |

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

## Links

- [Galaxy Help Forum](https://help.galaxyproject.org)
- [Galaxy Training Network](https://training.galaxyproject.org)
- [Galaxy Project](https://galaxyproject.org)
