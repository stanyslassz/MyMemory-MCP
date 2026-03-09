<div align="center">

# 🧠 memory-ai

**A cognitive memory system for LLMs — modeled after the human brain.**

*Your AI doesn't just store conversations. It learns, forgets, and remembers — like you do.*

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![MCP Compatible](https://img.shields.io/badge/MCP-compatible-purple.svg)](https://modelcontextprotocol.io/)

</div>

---

## Why memory-ai?

Most AI memory systems dump everything into a vector database and call it a day. memory-ai takes a radically different approach: it models memory **the way cognitive science says human brains actually work**.

The result? Your AI remembers what matters, naturally forgets what doesn't, and can recall dormant memories when they become relevant again. Not a product — a brain.

> **Local-first. Markdown-based. Works with any LLM.**

---

## How the Brain Works

### The Pipeline — From Chat to Cognition

```
                        ┌─────────────────────────────────────────┐
                        │              Your Chat                  │
                        └───────────────┬─────────────────────────┘
                                        │
                                        ▼
                        ┌───────────────────────────────┐
                        │   1. Extractor (LLM)          │
                        │   Entities, relations, facts   │
                        └───────────────┬───────────────┘
                                        │
                                        ▼
                        ┌───────────────────────────────┐
                        │   2. Resolver (deterministic)  │
                        │   Slug + alias + FAISS match   │
                        └───────────────┬───────────────┘
                                        │
                          ┌─────────────┴─────────────┐
                          │ ambiguous?                 │
                          ▼                           ▼
                ┌──────────────────┐        ┌──────────────┐
                │ 3. Arbitrator    │        │   Resolved   │
                │    (LLM)        │        │              │
                └────────┬────────┘        └──────┬───────┘
                          │                        │
                          └──────────┬─────────────┘
                                     ▼
                        ┌───────────────────────────────┐
                        │   4. Enricher                  │
                        │   Write MD + Graph + ACT-R     │
                        └───────────────┬───────────────┘
                                        │
                          ┌─────────────┴─────────────┐
                          ▼                           ▼
                ┌──────────────────┐        ┌──────────────────┐
                │ 5. Context       │        │ 6. FAISS Index   │
                │    Builder       │        │    (incremental)  │
                │  (zero LLM)     │        │                   │
                └────────┬────────┘        └──────┬────────────┘
                          │                        │
                          └──────────┬─────────────┘
                                     ▼
                        ┌───────────────────────────────┐
                        │       MCP Server              │
                        │  get_context · save_chat      │
                        │       search_rag              │
                        └───────────────────────────────┘
```

Each step is designed to minimize LLM calls. Only extraction and ambiguous arbitration touch the LLM — everything else is deterministic, fast, and reproducible.

---

## The Science Behind It

### ACT-R: How Human Memory Works

memory-ai uses **ACT-R** (Adaptive Control of Thought—Rational), the gold standard cognitive architecture from psychology research. The same math that models how humans remember and forget.

#### Base-Level Activation — The Forgetting Curve

```
B = ln( Σ tⱼ⁻ᵈ )
```

| Symbol | Meaning |
|--------|---------|
| `tⱼ` | Days since each mention (minimum 0.5) |
| `d` | Decay factor: `0.5` for long-term, `0.8` for short-term |
| `B` | Base activation — how accessible the memory is |

**What this means in practice:**
- 5 mentions in 2 days → high score (burst = importance)
- 5 mentions over 5 months → lower score (spread out = less urgent)
- A memory mentioned yesterday and 6 months ago → yesterday dominates
- Unused memories decay following a power law — just like human forgetting

#### Spreading Activation — Associative Recall

```
S = Σ( strengthᵢ × base_scoreᵢ ) / total_strength
```

When you think about **Paris**, **France** lights up too. That's spreading activation. In memory-ai, entities connected by relations boost each other's scores:

- Your project "Kitchen Renovation" is active → **contractor**, **budget**, **timeline** all get a boost
- You mention your **dog** → **veterinarian** and **dog park** become more accessible
- Unused connections fade over time (180-day half-life)

#### Hebbian Learning — "Neurons That Fire Together Wire Together"

```
strength = min(1.0, strength + 0.05)    # Each co-occurrence
effective = strength × e^(-days / 180)   # Time decay
```

When two entities appear together in conversation, their connection strengthens. Stop mentioning them together, and the connection fades. Exactly like synaptic plasticity in the brain.

#### The Final Score

```
score = σ(B + importance × w₁ + spreading × w₂)
```

A sigmoid function normalizes everything to `[0, 1]`. The highest-scoring entities make it into your context — the rest remain in long-term storage, searchable but not actively loaded.

---

## Three-Level Memory — Like the Human Brain

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   L1: Working Memory      _context.md                      │
│   ━━━━━━━━━━━━━━━━━━━     Injected into every conversation │
│   Top entities by ACT-R   Token-budgeted sections          │
│   score. Always present.  Deterministic, reproducible.     │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   L2: Semantic Memory      FAISS Vector Index              │
│   ━━━━━━━━━━━━━━━━━━━     Searchable via RAG              │
│   All indexed entities.   Retrieval bumps mention_dates    │
│   Dormant but findable.   → promotes back to L1 ↑         │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   L3: Episodic Memory      Markdown Files                  │
│   ━━━━━━━━━━━━━━━━━━━     Full knowledge base             │
│   Source of truth.        Every fact, date, relation.      │
│   Never deleted.          Human-readable & editable.       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Re-Emergence: Memories Come Back

The killer feature. When `search_rag()` retrieves a dormant entity from L2, it **bumps its mention_dates** — just like recalling a memory strengthens it in the brain. The entity's ACT-R score rises, and it naturally re-enters L1 context. No special logic needed — pure mnemonic reinforcement.

> You haven't thought about your **swimming routine** in months. It decayed out of L1. Then you ask about back pain — RAG retrieves swimming as related. Its score bumps. Next context rebuild: swimming is back in your active memory.

---

## Dream Mode — Sleep Consolidation for AI

```bash
uv run memory dream
```

Like the brain during deep sleep, dream mode reorganizes memories without new input. An LLM coordinator analyzes memory state and plans which steps to run:

```
┌──────────────────────────────────────────────────────────┐
│                    Dream Pipeline                        │
│                                                          │
│  ○ 1. Load          Load graph + entity files            │
│  ○ 2. Extract docs  RAG documents → structured entities  │
│  ○ 3. Consolidate   Merge redundant facts (8+ per entity)│
│  ○ 4. Merge         Detect & merge duplicate entities    │
│  ○ 5. Relations     FAISS similarity → LLM validation    │
│  ○ 6. Prune         Archive dead entities → _archive/    │
│  ○ 7. Summaries     Generate entity summaries via LLM    │
│  ○ 8. Rescore       Recalculate all ACT-R scores         │
│  ○ 9. Rebuild       Rebuild context + FAISS index        │
│                                                          │
│  Dashboard: Rich Live terminal UI (real-time progress)   │
│  Coordinator: LLM plans steps + validates results        │
└──────────────────────────────────────────────────────────┘
```

**Brain-like behaviors:**
- **Consolidation** — merging fragmented memories into coherent knowledge (like slow-wave sleep)
- **Pruning** — removing weak, isolated memories (like synaptic pruning during REM)
- **Relation discovery** — finding connections between unrelated memories (like dream associations)
- **Re-scoring** — updating salience based on the new state (like memory reconsolidation)

---

## Knowledge Graph

Every entity and relation is stored in a rich knowledge graph with metadata:

```
         ┌──────────┐     affects      ┌──────────────┐
         │  Sciatica ├────────────────►│Daily Routine  │
         │  health   │                 │  interest     │
         │  s: 0.82  │                 │  s: 0.65      │
         └─────┬─────┘                 └───────────────┘
               │ improves
               ▼
         ┌──────────┐     uses         ┌──────────────┐
         │ Swimming  ├────────────────►│ Local Pool    │
         │ interest  │                 │  place        │
         │ s: 0.71   │                 │  s: 0.45      │
         └──────────┘                  └───────────────┘
```

**Entities** carry: `mention_dates` (last 50), `monthly_buckets` (older), `importance` (0-1), `retention` policy, `summary`, `aliases`, `tags`

**Relations** carry: `strength` (0-1, Hebbian), `last_reinforced`, `mention_count`, `context`

Visualize your graph interactively:
```bash
uv run memory graph    # Opens HTML visualization in browser
```

---

## Import & Ingest

### Chat Exports (Claude, ChatGPT)

Drop your JSON export into `_inbox/` and memory-ai splits it automatically:

```bash
cp claude_conversations.json memory/_inbox/
uv run memory inbox
# → "JSON import: 342 conversation(s) from claude_conversations.json"
uv run memory run      # Extract entities from all conversations
```

Supports **Claude.ai exports**, **ChatGPT exports**, and generic `[{role, content}]` arrays.

### Documents

Any `.md` or `.txt` file dropped in `_inbox/` gets automatically routed:
- **Conversations** → saved as chats for the extraction pipeline
- **Documents** → chunked and indexed in FAISS (RAG-searchable)
- **Dream mode** later promotes document knowledge to structured entities

---

## Quick Start

```bash
# 1. Install
git clone https://github.com/stanyslassz/MyMemory.git
cd MyMemory
uv sync --extra dev

# 2. Configure
cp config.yaml.example config.yaml
cp .env.example .env
# Edit config.yaml: set your LLM model
# Edit .env: add API keys

# 3. Run
uv run memory stats        # Check setup
uv run memory run           # Process pending chats
uv run memory dream         # Consolidate memories
uv run memory serve         # Start MCP server
```

### LLM Providers

Works with any LLM. Mix and match per pipeline step:

```yaml
llm:
  extraction:
    model: openrouter/google/gemini-3.1-flash-lite-preview
  arbitration:
    model: ollama/llama3.1:8b        # Local, free
  context:
    model: openai/gpt-4o-mini        # Fast, cheap
  dream:
    model: ollama/qwen3:14b          # Bigger model for overnight
```

| Provider | Prefix | Notes |
|----------|--------|-------|
| Ollama | `ollama/` | Local, free, private |
| OpenRouter | `openrouter/` | 200+ models, pay-per-token |
| OpenAI | `openai/` | GPT-4o, GPT-4o-mini |
| Anthropic | `anthropic/` | Claude models |
| LM Studio | `openai/` + `api_base` | Local GUI server |

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `memory run` | Full pipeline: extract + resolve + enrich + context + FAISS |
| `memory run-light` | Same but skip auto-consolidation (no extra LLM calls) |
| `memory context` | Rebuild `_context.md` on demand (no extraction) |
| `memory dream` | Brain-like memory reorganization (9-step pipeline) |
| `memory inbox` | Process files in `_inbox/` (JSON exports, docs) |
| `memory replay` | Retry failed extractions (`--list` to preview) |
| `memory consolidate` | Detect duplicate entities (`--facts` for fact merging) |
| `memory graph` | Interactive graph visualization in browser |
| `memory stats` | Display memory metrics |
| `memory rebuild-all` | Rebuild graph + scores + context + FAISS |
| `memory validate` | Check graph consistency |
| `memory clean` | Remove generated artifacts (with backup) |
| `memory serve` | Start MCP server (stdio or SSE) |

---

## MCP Integration

Plug memory-ai into any MCP-compatible client (Claude Desktop, Claude Code, etc.):

```json
{
  "mcpServers": {
    "memory-ai": {
      "command": "uv",
      "args": ["--directory", "/path/to/MyMemory", "run", "memory", "serve"]
    }
  }
}
```

| Tool | What it does |
|------|-------------|
| `get_context()` | Returns the living context file — your AI's "working memory" |
| `save_chat(messages)` | Saves a conversation for background processing |
| `search_rag(query)` | Semantic search + ACT-R re-ranking + L2→L1 memory promotion |

---

## The Context File

What your AI actually sees — a token-budgeted, deterministic snapshot of your most important memories:

```markdown
# Memory Context — 2026-03-09

## AI Personality
Tu es un assistant personnel qui connait bien l'utilisateur...

## Identity
- Alexis, developer, lives in Lyon with partner and dog Max

## Top of Mind
- Kitchen renovation: contractor selected, starting April
- Job interview at TechCorp next Tuesday
- Back pain improving with swimming (3x/week)

## Vigilances
- [diagnosis] Chronic sciatica — avoid prolonged sitting
- [treatment] Daily stretching routine — check compliance

## Available in memory (not detailed above)
python (0.42) | react (0.38) | alice (0.35) | ...
```

Each section has a token budget. Only what fits gets in. Everything else stays in L2/L3, searchable via RAG.

---

## Architecture

```
src/
  core/
    config.py           # Config from YAML + .env
    models.py           # Pydantic v2 models (20+ types)
    llm.py              # LiteLLM + Instructor, stall-aware streaming, JSON repair
    utils.py            # slugify, token estimation, frontmatter parsing
  memory/
    store.py            # Markdown CRUD with YAML frontmatter
    graph.py            # Knowledge graph (atomic writes, lockfile, backup)
    scoring.py          # ACT-R + spreading activation (two-pass)
    mentions.py         # Windowed mention_dates + monthly buckets
    context.py          # Deterministic context builder (zero LLM)
  pipeline/
    extractor.py        # LLM extraction with stall detection
    resolver.py         # Deterministic entity resolution (zero LLM)
    arbitrator.py       # LLM arbitration (ambiguous only)
    enricher.py         # Write entities + update graph + score
    indexer.py          # FAISS vector index (incremental)
    dream.py            # 9-step dream pipeline with coordinator
    dream_dashboard.py  # Rich Live terminal dashboard
    chat_splitter.py    # Claude/ChatGPT JSON export splitter
    doc_ingest.py       # Document → FAISS (fallback path)
    orchestrator.py     # Pipeline orchestration (extracted from CLI)
    visualize.py        # Interactive graph HTML visualization
  mcp/
    server.py           # FastMCP server (3 tools)

prompts/                # All LLM prompts as .md files (never hardcoded)
memory/                 # Source of truth — Markdown files (gitignored)
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ (`uv` + `hatchling`) |
| Models | Pydantic v2 |
| LLM | LiteLLM + Instructor (any provider) |
| Vectors | FAISS (IndexFlatIP, cosine similarity) |
| Embeddings | sentence-transformers or API-based |
| MCP | FastMCP |
| CLI | Click + Rich |
| NLP | spaCy (optional) |
| Resilience | json-repair (small model JSON fixing) |

---

## Philosophy

> *"The art of memory is the art of attention."* — Samuel Johnson

This isn't a RAG database. It's a cognitive system.

- **Adaptive** — ACT-R power-law decay means memories surface when they matter
- **Associative** — Spreading activation through the graph, like neural priming
- **Forgetful by design** — Your context window is limited. Only what matters gets in
- **Resilient** — Dormant memories re-emerge when recalled, just like human memory
- **Transparent** — Everything is Markdown files you can read and edit
- **Local-first** — Your memories stay on your machine

---

<div align="center">

**Open source. Local first. Not a product — a brain.**

MIT License

</div>
