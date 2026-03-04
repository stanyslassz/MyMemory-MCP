# memory-ai

Personal persistent memory system for LLMs. Local-first, Markdown-based.

## Quick Start

```bash
# Install dependencies
uv sync --extra dev

# Check everything works
uv run memory --help
uv run pytest tests/ -v

# View memory stats
uv run memory stats

# Process pending chats
uv run memory run

# Start MCP server (stdio)
uv run memory serve
```

## Architecture

```
src/
├── core/           # Config, LLM abstraction, Pydantic models
├── memory/         # MD file CRUD, graph, scoring, context generation
├── pipeline/       # Extraction → Resolution → Arbitration → Enrichment → FAISS
├── mcp/            # MCP server (3 tools: get_context, save_chat, search_rag)
└── cli.py          # Click CLI

memory/             # Source of truth — Markdown files
prompts/            # LLM prompts — editable .md files (never hardcoded in Python)
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `memory run` | Process pending chats through the full pipeline |
| `memory rebuild-graph` | Rebuild `_graph.json` from MD files |
| `memory rebuild-faiss` | Full FAISS index rebuild |
| `memory rebuild-all` | Rebuild graph + context + FAISS |
| `memory validate` | Check graph consistency |
| `memory stats` | Display memory metrics |
| `memory inbox` | Process files in `_inbox/` |
| `memory serve` | Start the MCP server |

## MCP Tools

- **`get_context()`** — Returns pre-compiled memory context (`_context.md`)
- **`save_chat(messages)`** — Saves a conversation for later processing
- **`search_rag(query)`** — Semantic search across memory with relation enrichment

## Configuration

### LLM Providers

Edit `config.yaml` to use your preferred LLM:

```yaml
# Ollama (local, free)
llm:
  extraction:
    model: "ollama/llama3.1:8b"

# OpenAI
llm:
  extraction:
    model: "openai/gpt-4o-mini"  # requires OPENAI_API_KEY in .env

# Anthropic
llm:
  extraction:
    model: "anthropic/claude-sonnet-4-5-20250929"  # requires ANTHROPIC_API_KEY in .env
```

### Embeddings

```yaml
# Local (no external service needed)
embeddings:
  provider: "sentence-transformers"
  model: "all-MiniLM-L6-v2"

# Ollama
embeddings:
  provider: "ollama"
  model: "nomic-embed-text"
```

### API Keys

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
# Edit .env with your API keys
```

## Claude Desktop / Claude Code Integration

Add to your MCP settings:

```json
{
  "mcpServers": {
    "memory-ai": {
      "command": "uv",
      "args": ["--directory", "/path/to/memory-ai", "run", "memory", "serve"]
    }
  }
}
```

## Development

```bash
# Run tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_graph.py -v
```
