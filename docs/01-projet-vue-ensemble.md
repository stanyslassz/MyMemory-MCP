# MyMemory (memory-ai) — Vue d'ensemble du projet

> Système de mémoire persistante personnelle pour LLMs.
> Local-first, base Markdown, inspiré des sciences cognitives.
> Références au format `fichier:fonction()` — stables, pas de numéros de ligne.

---

## Table des matières

1. [Architecture et philosophie](#1-architecture-et-philosophie)
2. [Installation et configuration](#2-installation-et-configuration)
3. [Utilisation quotidienne](#3-utilisation-quotidienne-cli)
4. [Serveur MCP](#4-serveur-mcp)
5. [Format des fichiers](#5-format-des-fichiers)
6. [Fichiers de données générés](#6-fichiers-de-données-générés)

---

## 1. Architecture et philosophie

### 1.1 Principe fondamental

MyMemory résout le problème de l'amnésie des LLMs entre sessions. Il extrait des connaissances structurées (entités, relations, observations) à partir de conversations, les stocke en fichiers Markdown, les score avec un modèle cognitif ACT-R, et les sert via MCP (Model Context Protocol).

### 1.2 Pipeline en 6 étapes

```
Chat text (chats/ avec processed: false)
  → Étape 1 : Extractor (LLM, streaming avec détection de stall)
  → Étape 2 : Resolver (déterministe : slug/alias/FAISS, 0 token LLM)
  → Étape 3 : Arbitrator (LLM, uniquement entités ambiguës)
  → Étape 4 : Enricher (écriture MD + graphe + scoring ACT-R)
  → Étape 5 : Context builder (template déterministe, 0 token LLM)
  → Étape 6 : FAISS indexer (incrémental)
```

Point d'entrée : `cli.py:run()` → `orchestrator.py:run_pipeline()`.

### 1.3 Mémoire à 3 niveaux

| Niveau | Stockage | Rôle |
|--------|----------|------|
| **L1** | `_context.md` | Mémoires actives injectées dans chaque conversation |
| **L2** | Index FAISS | Recherchable via RAG, remontée vers L1 via bump de mentions |
| **L3** | Fichiers Markdown | Base de connaissances complète, source de vérité |

**Ré-émergence L2→L1** : `server.py:search_rag()` bumpe `mention_dates` des entités retrouvées → le score ACT-R monte → l'entité réintègre naturellement le contexte au prochain rebuild.

### 1.4 Carte des modules (42 fichiers Python)

```
src/
  cli.py                           # 17 commandes Click (wrappers minces)
  core/
    config.py                      # 12 dataclasses de configuration + load_config()
    models.py                      # 24 modèles Pydantic (données pures)
    llm.py                         # Abstraction LLM : 9 fonctions d'appel, stall detection
    utils.py                       # slugify(), estimate_tokens(), parse_frontmatter()
    action_log.py                  # Log centralisé d'actions (JSONL)
    metrics.py                     # MetricsCollector pour SLO tracking
  memory/
    graph.py                       # CRUD graphe, BFS, persistance atomique avec lockfile
    store.py                       # I/O entités Markdown, chats, consolidation facts
    scoring.py                     # ACT-R + spreading activation + LTD + emotional boost
    mentions.py                    # Fenêtrage des mentions + consolidation mensuelle
    event_log.py                   # Log d'événements append-only (JSONL)
    insights.py                    # Métriques cognitives pour CLI
    context/
      __init__.py                  # Re-exports : build_context, write_context, write_index...
      builder.py                   # Génération contexte (déterministe + LLM + natural)
      formatter.py                 # Enrichissement entités, filtrage facts, vigilances
      utilities.py                 # Dédup trigram, tokenisation, clustering
  pipeline/
    extractor.py                   # Étape 1 : extraction chat avec segment merging
    resolver.py                    # Étape 2 : résolution entité (exact/alias/FAISS)
    arbitrator.py                  # Étape 3 : arbitrage ambiguïtés (LLM)
    enricher.py                    # Étape 4 : écritures MD + graphe + scoring
    orchestrator.py                # Coordinateur pipeline (logique métier)
    indexer.py                     # Étape 6 : indexation FAISS + recherche
    doc_ingest.py                  # Pipeline fallback (chunk + FAISS sans entités)
    inbox.py                       # Traitement fichiers _inbox/
    router.py                      # Classification déterministe doc/conversation
    chat_splitter.py               # Split JSON multi-conversation (Claude/ChatGPT/generic)
    ingest_state.py                # Machine à états ingestion + retry ledger
    keyword_index.py               # Index SQLite FTS5 pour recherche keyword
    visualize.py                   # Génération HTML graphe interactif (D3.js)
    dream.py                       # Coordinateur dream (10 étapes) + checkpoint/resume
    dream_dashboard.py             # UI Rich Live pour progression dream
    dream/
      consolidator.py              # Steps 2,3,8 : extraction docs, consolidation facts, summaries
      discovery.py                 # Steps 5,6 : découverte relations, inférence transitive
      merger.py                    # Step 4 : dédup FAISS + merge entités
  mcp/
    server.py                      # 7 outils FastMCP + recherche hybride RRF
mcp_stdio.py                       # Launcher standalone pour Claude Desktop
```

### 1.5 Graphe de dépendances (chemins critiques)

```
cli:run() → orchestrator:run_pipeline()
  → extractor:extract_from_chat() → llm:call_extraction()
  → resolver:resolve_all()
  → arbitrator:arbitrate_entity() [si ambiguë]
  → enricher:enrich_memory() → store:create_entity()/update_entity()
  → graph:add_relation()
  → scoring:recalculate_all_scores() → scoring:spreading_activation()
  → context/builder:build_context() → context/formatter:_enrich_entity()
  → indexer:incremental_update()
  [si consolidate] orchestrator:auto_consolidate() → llm:call_fact_consolidation()
```

---

## 2. Installation et configuration

### 2.1 Prérequis

- Python 3.11+
- `uv` (gestionnaire de packages)
- Un provider LLM (Ollama local, LM Studio, ou API OpenAI/compatible)

### 2.2 Installation

```bash
uv sync --extra dev    # Installe toutes les dépendances (dev inclus pour pytest)
```

Point d'entrée défini dans `pyproject.toml` : `memory = "src.cli:cli"`.
Tous les imports utilisent le préfixe `src.` (ex: `from src.core.config import load_config`).

### 2.3 Configuration

Le fichier `config.yaml` (gitignored) est copié depuis `config.yaml.example`. Authentification via `.env` (ex: `OPENAI_API_KEY=lm-studio`).

Chargement : `config.py:load_config()` → charge `.env` d'abord, puis YAML, résout les chemins relatifs au `project_root`.

#### Hiérarchie des dataclasses de configuration

| Classe | Rôle | Champs clés |
|--------|------|-------------|
| `Config` | Maître | `user_language`, `memory_path`, `context_max_tokens`, `context_format`, `mcp_transport` |
| `LLMStepConfig` | Par étape LLM | `model`, `temperature`, `max_retries`, `timeout`, `api_base`, `context_window` |
| `ScoringConfig` | Paramètres ACT-R | `decay_factor` (0.5), `importance_weight` (0.3), `spreading_weight` (0.2), `retrieval_threshold` (0.05) |
| `EmbeddingsConfig` | Embeddings | `provider` ("sentence-transformers"), `model`, `chunk_size` (400), `chunk_overlap` (80) |
| `FAISSConfig` | Index FAISS | `index_path`, `mapping_path`, `manifest_path`, `top_k` (5) |
| `CategoriesConfig` | Types fermés | `observations`, `entity_types`, `relation_types`, `folders` |
| `SearchConfig` | Recherche hybride | `hybrid_enabled`, `fts_db_path`, `rrf_k`, `weight_semantic/keyword/actr` |
| `ContextConfig` | Budget contexte | `top_entities_count` (50), `reserved_tokens_*`, `fact_dedup_threshold` (0.35) |
| `FactTTLConfig` | TTL des faits | Durées d'expiration par catégorie |
| `FeaturesConfig` | Feature flags | `doc_pipeline` (bool) |
| `IngestConfig` | Ingestion docs | `recovery_threshold_seconds`, `max_retries`, `jobs_path` |
| `NLPConfig` | NLP local | `enabled`, `model`, `dedup_threshold` |

#### Configs LLM disponibles

5 slots LLM indépendants, chacun configurable séparément :
- `llm_extraction` — Étape 1 (extraction)
- `llm_arbitration` — Étape 3 (arbitrage)
- `llm_consolidation` — Consolidation de faits
- `llm_context` — Génération contexte + résumés entités
- `llm_dream` — Mode dream (optionnel, fallback vers `llm_context` via `config.llm_dream_effective`)

---

## 3. Utilisation quotidienne (CLI)

Toutes les commandes : `uv run memory [commande] [options]`
Flags globaux : `-v`/`--verbose` (debug), `-c`/`--config <path>` (config.yaml alternatif).

### 3.1 Commandes principales

#### `run` — Pipeline complet
```bash
uv run memory run
```
Traite les chats non traités (capped à `job_max_chats_per_run`), exécute le pipeline 6 étapes, puis auto-consolidation + contexte LLM si activé.
**Chemin** : `cli.py:run()` → `orchestrator.py:run_pipeline(config, console, consolidate=True)`

#### `run-light` — Pipeline léger
```bash
uv run memory run-light
```
Identique à `run` sauf :
- **Saute** l'auto-consolidation LLM (étape 5b)
- **Contexte** toujours déterministe (jamais `build_context_with_llm()`)

**Chemin** : `cli.py:run_light()` → `orchestrator.py:run_pipeline(config, console, consolidate=False)`

#### `dream` — Réorganisation cérébrale
```bash
uv run memory dream              # Exécution complète (10 étapes)
uv run memory dream --dry-run    # Planification sans exécution
uv run memory dream --step 5     # Exécuter uniquement l'étape N
uv run memory dream --resume     # Reprendre depuis checkpoint
uv run memory dream --reset      # Effacer checkpoint et redémarrer
```
Pipeline de 10 étapes : load → extract docs → consolidate facts → merge entities → discover relations → transitive relations → prune dead → generate summaries → rescore → rebuild.

Le coordinateur déterministe `dream.py:decide_dream_steps()` planifie quelles étapes exécuter selon les statistiques mémoire. Validation déterministe après chaque étape via `dream.py:validate_dream_step()`.

**Chemin** : `cli.py:dream()` → `dream.py:run_dream()`

### 3.2 Commandes de construction

#### `context` — Reconstruire le contexte
```bash
uv run memory context
```
Recharge le graphe, rescore, génère `_context.md` + `_index.md`. Pas d'extraction, pas de LLM (sauf si `context_format: natural`).
**Chemin** : `cli.py:context()` → `scoring.py:recalculate_all_scores()` → `builder.py:build_context()`

#### `rebuild-graph` — Reconstruire le graphe
```bash
uv run memory rebuild-graph
```
Scanne tous les MD d'entités et reconstruit `_graph.json` from scratch.
**Chemin** : `cli.py:rebuild_graph()` → `graph.py:rebuild_from_md()`

#### `rebuild-faiss` — Reconstruire l'index FAISS
```bash
uv run memory rebuild-faiss
```
Rebuild complet : scan MD → chunks → embeddings → `_memory.faiss` + `_memory.pkl`.
**Chemin** : `cli.py:rebuild_faiss()` → `indexer.py:build_index()`

#### `rebuild-all` — Tout reconstruire
```bash
uv run memory rebuild-all
```
Graphe + scores + contexte + FAISS en une seule commande.
**Chemin** : `cli.py:rebuild_all()` → `graph.py:rebuild_from_md()` → `scoring.py:recalculate_all_scores()` → `builder.py:build_context()` → `indexer.py:build_index()`

### 3.3 Commandes d'analyse

#### `stats` — Métriques mémoire
```bash
uv run memory stats
```
Affiche : entités (graphe/fichiers), relations, chats en attente, présence contexte/index/FAISS.

#### `validate` — Vérifier la cohérence
```bash
uv run memory validate
```
Vérifie : fichiers d'entités existent, endpoints de relations existent dans le graphe.
**Chemin** : `cli.py:validate()` → `graph.py:validate_graph()`

#### `insights` — Insights cognitifs ACT-R
```bash
uv run memory insights              # Texte formaté
uv run memory insights --format json # JSON
```
Distribution des scores, courbe d'oubli, hotspots émotionnels, relations faibles, hubs réseau.
**Chemin** : `cli.py:insights()` → `insights.py:compute_insights()`

#### `actions` — Historique d'actions
```bash
uv run memory actions                    # 20 dernières
uv run memory actions --last 50          # 50 dernières
uv run memory actions --entity alice     # Filtrer par entité
uv run memory actions --action created   # Filtrer par type
```
**Chemin** : `cli.py:actions()` → `action_log.py:read_actions()`

#### `graph` — Visualisation interactive
```bash
uv run memory graph
```
Génère `_graph.html` (D3.js) et ouvre dans le navigateur.
**Chemin** : `cli.py:graph()` → `visualize.py:open_graph()`

### 3.4 Commandes de maintenance

#### `consolidate` — Détecter les doublons
```bash
uv run memory consolidate --dry-run           # Prévisualiser les doublons
uv run memory consolidate --facts --dry-run   # Prévisualiser la consolidation de faits
uv run memory consolidate --facts --min-facts 6  # Consolider (LLM)
```
Mode par défaut : détection de doublons d'entités (titre/alias).
Mode `--facts` : consolidation LLM des observations redondantes.
**Chemin** : `cli.py:consolidate()` → `orchestrator.py:consolidate_facts()`

#### `relations` — Découvrir des relations
```bash
uv run memory relations --dry-run          # Prévisualiser
uv run memory relations --entity alice     # Pour une entité spécifique
```
FAISS similarity + tag overlap + co-occurrence. Zéro LLM.
**Chemin** : `cli.py:relations()` → `orchestrator.py:discover_relations_deterministic()`

#### `replay` — Réessayer les extractions échouées
```bash
uv run memory replay --list    # Lister les échecs
uv run memory replay           # Réessayer (LLM)
```
**Chemin** : `cli.py:replay()` → `ingest_state.py:list_retriable()` → pipeline complet

#### `inbox` — Traiter la boîte de réception
```bash
uv run memory inbox
```
Traite les fichiers déposés dans `memory/_inbox/` : conversations → chats non traités, documents → FAISS.
Supporte le JSON multi-conversation (Claude.ai, ChatGPT, generic).
**Chemin** : `cli.py:inbox()` → `inbox.py:process_inbox()` → `router.py:classify()`

#### `clean` — Nettoyer les artefacts
```bash
uv run memory clean --artifacts --dry-run  # Prévisualiser
uv run memory clean --all                  # Tout sauf les MD sources
uv run memory clean --full                 # Tout + reset chats
uv run memory clean --chats                # Reset chats uniquement
```
Crée une sauvegarde tar.gz horodatée avant suppression.

#### `serve` — Démarrer le serveur MCP
```bash
uv run memory serve                # stdio (défaut)
uv run memory serve -t sse         # SSE
```
**Chemin** : `cli.py:serve()` → `server.py:run_server()`

---

## 4. Serveur MCP

7 outils exposés via FastMCP (`server.py`). Config chargée au module level via `server.py:_get_config()`.

### 4.1 Outils de lecture

#### `get_context()` → str
Retourne `_context.md` (ou `_index.md` en fallback). Aucun paramètre.

#### `search_rag(query: str)` → dict
Recherche sémantique + keyword dans la mémoire.

**Flux** :
1. `indexer.py:search()` — FAISS vectoriel
2. Si `hybrid_enabled` et FTS5 existe : `keyword_index.py:search_keyword()` → `server.py:_rrf_fusion()` (RRF : semantic + keyword + ACT-R)
3. Sinon : re-ranking linéaire (60% vector + 40% ACT-R)
4. Enrichissement avec relations du graphe (adjacence pré-calculée)
5. **L2→L1 bump** : `mentions.py:add_mention()` sur les entités retrouvées

**Retour** : `{query, total, results: [{entity_id, file, score, title, type, relations}]}`

### 4.2 Outils d'écriture

#### `save_chat(messages: list[dict])` → dict
Sauvegarde la conversation dans `chats/` avec `processed: false`.
Validation : chaque message doit avoir `role` (str) et `content` (str).

### 4.3 Outils CRUD

#### `delete_fact(entity_name, fact_content)` → str
Trouve l'entité → match le fait par contenu → supprime du MD → entrée History.

#### `delete_relation(from_entity, to_entity, relation_type)` → str
Résout les entités → supprime du graphe + MD → sauvegarde graphe.

#### `modify_fact(entity_name, old_content, new_content)` → str
Trouve le fait → remplace le contenu en préservant les métadonnées → entrée History.

#### `correct_entity(entity_name, field, new_value)` → str
Met à jour les métadonnées d'entité (title, type, aliases, retention). Déplace le fichier si le type change (nouveau dossier).

---

## 5. Format des fichiers

### 5.1 Entités Markdown

Structure gérée par `store.py:read_entity()` / `store.py:write_entity()` :

```markdown
---
title: Nom de l'entité
type: health
retention: long_term
score: 0.72
importance: 0.85
frequency: 12
last_mentioned: "2026-03-07"
created: "2025-09-15"
aliases: [mal de dos, sciatique]
tags: [santé, chronique]
mention_dates: ["2026-03-01", "2026-03-07"]
monthly_buckets: {"2025-06": 3, "2025-09": 5}
summary: "Problème chronique de dos avec sciatique."
---
## Facts
- [diagnosis] (2024-03) Sciatique chronique [-]
- [treatment] (2025-11) Début de physiothérapie [+]
- [fact] Suivi régulier nécessaire

## Relations
- affects [[Routine quotidienne]]
- improves [[Natation]]

## History
- 2025-09-15: Created
```

#### Format d'observation
```
- [catégorie] (date) contenu [valence] #tag1 #tag2
```
- **Date** : optionnelle, YYYY-MM ou YYYY-MM-DD
- **Valence** : `[+]` positive, `[-]` négative, `[~]` neutre (optionnel)
- **Tags** : optionnels, préfixés `#`, max 3 par fait
- **Longueur** : max 150 caractères post-consolidation

#### Types fermés (doivent être synchronisés avec `categories` dans `config.yaml`)

| Type | Valeurs |
|------|---------|
| `ObservationCategory` | fact, preference, diagnosis, treatment, progression, technique, vigilance, decision, emotion, interpersonal, skill, project, context, rule, ai_style, user_reaction, interaction_rule |
| `EntityType` | person, health, work, project, interest, place, animal, organization, ai_self |
| `RelationType` | affects, improves, worsens, requires, linked_to, lives_with, works_at, parent_of, friend_of, uses, part_of, contrasts_with, precedes |

#### Mapping type → dossier

Configuré dans `categories.folders` (ex: person → close_ones, health → self, ai_self → self, interest → interests). Méthode : `config.py:Config.get_folder_for_type()`.

### 5.2 Graphe JSON (`_graph.json`)

```json
{
  "generated": "2026-03-12T10:00:00",
  "entities": {
    "entity-slug": {
      "file": "self/entity-slug.md",
      "type": "health",
      "title": "Nom",
      "score": 0.72,
      "importance": 0.85,
      "frequency": 12,
      "last_mentioned": "2026-03-07",
      "retention": "long_term",
      "aliases": [],
      "tags": [],
      "mention_dates": [],
      "monthly_buckets": {},
      "created": "2025-09-15",
      "summary": "",
      "negative_valence_ratio": 0.1
    }
  },
  "relations": [
    {
      "from": "entity-a",
      "to": "entity-b",
      "type": "affects",
      "strength": 0.5,
      "created": "2025-09-15",
      "last_reinforced": "2026-03-07",
      "mention_count": 3,
      "context": ""
    }
  ]
}
```

Persistance atomique : `graph.py:save_graph()` utilise temp file + `os.replace()` + lockfile (`_graph.lock`, timeout 5min) + backup `.bak`.

### 5.3 Chats

```markdown
---
processed: false
date: "2026-03-12"
source: manual
---
Human: ...
Assistant: ...
```

Après traitement : `processed: true`, `processed_at`, `entities_created`, `entities_updated`.

---

## 6. Fichiers de données générés

Tous gitignored. Résidence dans `memory/` (configurable via `memory_path`).

| Fichier | Créé par | Lu par | Rôle |
|---------|----------|--------|------|
| `_graph.json` | `graph:save_graph()` | `graph:load_graph()` | Index entités/relations |
| `_graph.json.bak` | `graph:save_graph()` | `graph:load_graph()` | Backup (recovery si corruption) |
| `_graph.lock` | `graph:_acquire_lock()` | `graph:_acquire_lock()` | Lock fichier atomicité |
| `_context.md` | `builder:write_context()` | `server:get_context()` | Contexte L1 pour LLM |
| `_index.md` | `builder:write_index()` | `server:get_context()` | Fallback si pas de contexte |
| `_memory.faiss` | `indexer:build_index()` | `indexer:search()` | Index vectoriel FAISS |
| `_memory.pkl` | `indexer:_save_index()` | `indexer:search()` | Mapping chunks (pickle) |
| `_faiss_manifest.json` | `indexer:_save_index()` | `indexer:load_manifest()` | Manifeste (hashes, modèle) |
| `_memory_fts.db` | `keyword_index:build_keyword_index()` | `keyword_index:search_keyword()` | Index FTS5 SQLite |
| `_retry_ledger.json` | `ingest_state:_save_ledger()` | `ingest_state:list_retriable()` | Échecs extraction |
| `_ingest_jobs.json` | `ingest_state:_save_jobs()` | `ingest_state:_load_jobs()` | Machine à états ingestion |
| `_actions.jsonl` | `action_log:log_action()` | `action_log:read_actions()` | Historique d'actions |
| `_event_log.jsonl` | `event_log:append_event()` | `event_log:read_events()` | Log événements (thread-safe) |
| `_dream_checkpoint.json` | `dream:_save_checkpoint()` | `dream:_load_checkpoint()` | État dream (resume) |
| `_graph.html` | `visualize:generate_graph_html()` | Navigateur | Visualisation D3.js |

### Sous-dossiers de `memory/`

| Dossier | Contenu |
|---------|---------|
| `self/` | Entités type health, ai_self |
| `close_ones/` | Entités type person, animal |
| `projects/` | Entités type project |
| `work/` | Entités type work, organization |
| `interests/` | Entités type interest, place (et défaut) |
| `chats/` | Transcriptions de conversations |
| `_inbox/` | Zone de dépôt pour ingestion |
| `_archive/` | Entités élaguées (dream mode, réversible) |

Initialisation automatique à chaque démarrage CLI : `store.py:init_memory_structure()`.
