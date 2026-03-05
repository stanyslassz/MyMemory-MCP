# memory-ai

**A personal adaptive memory system for LLMs — like a human brain, but for your AI.**

memory-ai saves your conversations, extracts structured knowledge in the background, scores it with cognitive science algorithms, and builds a living context file that gets injected into future chats. The result: your AI remembers what matters, forgets what doesn't, and can recall dormant memories when they become relevant again.

Open-source. Local-first. Local LLMs. Not a product — a brain.

---

## Architecture

> [View the full interactive architecture diagram on Excalidraw](https://excalidraw.com/#json=HhxHYjjxa--HWXq3YavWR,fYAW4qfWUDF5rQvwvM53-A)

```
Chat text
  |
  v
[spaCy pre-filter] -- dates, NER hints, dedup signals
  |
  v
Extractor (LLM) --> RawExtraction (entities, relations, observations)
  |
  v
Resolver (slug + alias + FAISS similarity) --> resolved / ambiguous / new
  |
  v
Arbitrator (LLM, ambiguous only) --> EntityResolution
  |
  v
Enricher
  |-- Write/update Markdown entity files
  |-- Update _graph.json (enriched relations: strength, dates, context)
  |-- Recalculate ACT-R scores + spreading activation
  |-- Update mention_dates (windowed) + monthly_buckets
  |
  v
Context Builder (deterministic template, zero LLM)
  |-- Top entities by ACT-R score with summaries
  |-- Vigilances section
  |-- Available in memory (entities below threshold)
  |-- Weighted tags cloud
  |
  v
FAISS Indexer (incremental)
  |
  v
MCP Server
  |-- get_context()  --> _context.md
  |-- save_chat()    --> store + queue for pipeline
  |-- search_rag()   --> FAISS + re-ranking + L2->L1 promotion
```

---

## How It Works

### 1. Save a conversation
When you chat with your AI, the conversation is saved via MCP's `save_chat()` tool and queued for background processing.

### 2. Extract knowledge
The pipeline runs each chat through an LLM extractor that identifies **entities** (people, projects, health topics...), **relations** between them, and **observations** (facts, preferences, decisions...).

### 3. Resolve & enrich
A deterministic resolver matches extracted entities against known ones using slug matching, aliases, and FAISS semantic similarity. Only truly ambiguous cases go to the LLM arbitrator. The enricher then updates Markdown files and the knowledge graph.

### 4. Score with cognitive science
Every entity gets a score using **ACT-R Base-Level Activation** — the same mathematical model cognitive scientists use to model human memory:

```
B = ln(sum of t_j^(-d))    # power-law decay over all mention dates
S = sum(w_ij * A_j)         # spreading activation from connected entities
score = sigmoid(B + wS)     # normalized to [0, 1]
```

**Key properties:**
- 5 mentions in 2 days (burst) scores higher than 5 mentions over 5 months
- Entities connected to active ones get a boost (spreading activation)
- Unused memories naturally decay — just like human forgetting
- But they're never truly lost (see: re-emergence below)

### 5. Build living context
A deterministic template assembles the context file — no LLM involved, fully reproducible:

```markdown
# Memory Context -- 2026-03-05

## Identity
{summary of "self" entity}

## Top of mind
{top N entities by score, with summaries}

## Vigilances
{health diagnoses, treatments to watch}

## Work & Projects
{active work and projects}

## Close ones
{people and pets}

## Available in memory (not detailed above)
alice (person, 0.42) | renovation (project, 0.38) | python (interest, 0.35)

## Memory tags
#health(0.8) #work(0.7) #family(0.9) #renovation(0.4) ...
```

### 6. Re-emergence: memories come back
When `search_rag()` retrieves an entity from FAISS, it **bumps its mention_dates** — just like recalling a memory strengthens it. The entity's ACT-R score naturally rises, and it can re-enter the context file in future builds. No special logic needed — pure mnemonic reinforcement.

---

## 3-Level Memory

| Level | Storage | Purpose |
|-------|---------|---------|
| **L1** — Context | `_context.md` | Active memories injected into every conversation |
| **L2** — FAISS | Vector index | Searchable via RAG, can promote back to L1 |
| **L3** — Archive | Markdown files | Full knowledge base, source of truth |

The flow **L2 -> L1** (re-emergence) is what makes this adaptive: a dormant memory retrieved by RAG gets reinforced and naturally bubbles back into active context.

---

## Enriched Knowledge Graph

Entities and relations are stored in `_graph.json` with rich metadata:

**Entities** carry:
- `mention_dates` — last 50 precise dates (powers ACT-R)
- `monthly_buckets` — older mentions aggregated by month (`{"2025-06": 10}`)
- `importance` — LLM-assessed significance (0-1)
- `retention` — `permanent` | `long_term` | `short_term`
- `summary` — pre-computed LLM summary (regenerated only when facts change)

**Relations** carry:
- `strength` — 0.0 to 1.0, grows logarithmically with co-mentions
- `last_reinforced` — decays with 6-month half-life if not reinforced
- `mention_count` — number of co-occurrences
- `context` — preserved from original extraction

---

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

## CLI Commands

| Command | Description |
|---------|-------------|
| `memory run` | Process pending chats through the full pipeline |
| `memory replay <file>` | Re-process a specific chat file |
| `memory rebuild-graph` | Rebuild `_graph.json` from Markdown files |
| `memory rebuild-faiss` | Full FAISS index rebuild |
| `memory rebuild-all` | Rebuild graph + context + FAISS |
| `memory validate` | Check graph consistency |
| `memory stats` | Display memory metrics |
| `memory inbox` | Process files in `_inbox/` |
| `memory serve` | Start the MCP server |
| `memory clean` | Flush memory data (with backup + dry-run) |
| `memory consolidate` | Detect and merge duplicate entities |
| `memory retry-ledger` | Show/retry failed pipeline jobs |

## MCP Tools

Exposed via [Model Context Protocol](https://modelcontextprotocol.io/) for any MCP-compatible client:

| Tool | Description |
|------|-------------|
| `get_context()` | Returns pre-compiled memory context (`_context.md`) |
| `save_chat(messages)` | Saves a conversation for background processing |
| `search_rag(query)` | Semantic search with re-ranking and L2->L1 promotion |

## Claude Desktop / Claude Code Integration

Add to your MCP settings (`claude_desktop_config.json` or `.claude/settings.json`):

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

---

## Configuration

All configuration lives in `config.yaml` at the project root.

### LLM Providers

Different pipeline steps can use different models:

```yaml
llm:
  extraction:
    model: "ollama/llama3.1:8b"     # Local via Ollama
    # model: "openai/gpt-4o-mini"   # OpenAI (needs OPENAI_API_KEY in .env)
  arbitration:
    model: "ollama/llama3.1:8b"
  context:
    model: "ollama/llama3.1:8b"
  consolidation:
    model: "ollama/llama3.1:8b"
```

### Scoring (ACT-R)

```yaml
scoring:
  model: "act_r"
  decay_factor: 0.5              # Standard ACT-R decay
  decay_factor_short_term: 0.8   # Faster decay for short-term memories
  importance_weight: 0.3         # Weight of LLM-assessed importance
  spreading_weight: 0.2          # Weight of spreading activation
  permanent_min_score: 0.5       # Floor for permanent entities
  relation_strength_base: 0.5    # Initial relation strength
  relation_decay_halflife: 180   # Days for unreinforced relations to halve
  window_size: 50                # Recent mention dates to keep
```

### NLP Pre-filter (optional)

```yaml
nlp:
  enabled: true
  model: "fr_core_news_sm"       # ~15 MB, install with: python -m spacy download fr_core_news_sm
  dedup_threshold: 0.85          # Similarity threshold for observation dedup
  date_extraction: true          # Extract dates from French text
  pre_ner: true                  # Detect proper nouns as resolver hints
```

### Embeddings

```yaml
embeddings:
  provider: "sentence-transformers"  # Local, no API needed
  model: "all-MiniLM-L6-v2"
```

### API Keys

```bash
cp .env.example .env
# Add your keys: OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.
```

---

## Project Structure

```
src/
  core/
    config.py          # Dataclass config from config.yaml + .env
    models.py          # Pydantic v2 models (entities, relations, observations)
    llm.py             # LiteLLM + Instructor, stall-aware streaming
  memory/
    store.py           # Markdown CRUD with YAML frontmatter
    graph.py           # _graph.json management, relation reinforcement
    scoring.py         # ACT-R base activation + spreading activation
    mentions.py        # Windowed mention_dates + monthly bucket consolidation
    context.py         # Deterministic context template builder
  pipeline/
    extractor.py       # LLM extraction -> RawExtraction
    resolver.py        # Deterministic slug/alias/FAISS matching
    arbitrator.py      # LLM resolution for ambiguous entities only
    enricher.py        # Write Markdown + update graph
    indexer.py         # FAISS vector index (incremental)
    nlp_prefilter.py   # Optional spaCy pre-filter
  mcp/
    server.py          # FastMCP server (get_context, save_chat, search_rag)
  cli.py               # Click CLI entrypoint

memory/                # Source of truth (gitignored)
  self/                # Personal entities
  close_ones/          # People, pets
  work/                # Work-related entities
  projects/            # Project entities
  interests/           # Interests, hobbies
  chats/               # Raw conversations
  _inbox/              # Pending files for processing

prompts/               # LLM prompt templates (.md files, never hardcoded)
```

---

## Data Model

### Entity Types
`person` | `health` | `work` | `project` | `interest` | `place` | `animal` | `organization`

### Observation Categories
`fact` | `preference` | `diagnosis` | `treatment` | `progression` | `technique` | `vigilance` | `decision` | `emotion` | `interpersonal` | `skill` | `project` | `context` | `rule`

### Relation Types
`affects` | `improves` | `worsens` | `requires` | `linked_to` | `lives_with` | `works_at` | `parent_of` | `friend_of` | `uses` | `part_of` | `contrasts_with` | `precedes`

### Retention Policies

| Policy | Behavior |
|--------|----------|
| `permanent` | Always in context, minimum score = 0.5 |
| `long_term` | Normal ACT-R decay (d = 0.5) |
| `short_term` | Accelerated decay (d = 0.8), fades faster |

---

## Development

```bash
# Install with dev dependencies
uv sync --extra dev

# Run all tests
uv run pytest tests/ -v

# Run specific test
uv run pytest tests/test_scoring.py::test_actr_burst_beats_spread -v

# Check memory stats
uv run memory stats
```

### Tech Stack
- **Python 3.11+** managed with `uv` and built with `hatchling`
- **Pydantic v2** for data models
- **LiteLLM + Instructor** for LLM calls (supports Ollama, OpenAI, Anthropic, LM Studio)
- **FAISS** for vector search (IndexFlatIP)
- **sentence-transformers** for local embeddings
- **FastMCP** for Model Context Protocol server
- **Click** for CLI
- **spaCy** (optional) for NLP pre-filtering

---

## Philosophy

This is not a product. It's a personal brain for your AI assistant.

Every conversation you have is a memory. Most memory systems just dump everything into a vector database and call it a day. memory-ai takes a different approach: it models memory the way cognitive science says human brains work.

- **Adaptive scoring** — ACT-R power-law decay means frequently and recently used memories surface naturally, while unused ones fade
- **Associative recall** — Spreading activation through the knowledge graph means related concepts strengthen each other
- **Natural forgetting** — Not a bug, it's a feature. Your context window is limited, so only what matters gets in
- **Re-emergence** — A forgotten memory can come back when retrieved via RAG search, just like a human suddenly remembering something
- **Deterministic output** — The context file is built from a template, not generated by an LLM. Reproducible, fast, token-efficient

The goal is a context file that makes your AI feel like it truly knows you — your health, your projects, your preferences, your relationships — without overwhelming it with irrelevant details.

---

## License

MIT
