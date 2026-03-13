# 03 — Interactions entre modules et pipelines

> Documentation technique des flux de donnees, interactions entre modules et pipelines de MyMemory.
> Source de verite : code source uniquement. References au format `fichier:fonction()`.
> Mise a jour : 2026-03-12

---

## Table des matieres

1. [Carte des modules](#1-carte-des-modules)
2. [Pipeline principal (memory run)](#2-pipeline-principal-memory-run)
3. [Serveur MCP (7 outils)](#3-serveur-mcp-7-outils)
4. [Pipeline Dream (10 etapes)](#4-pipeline-dream-10-etapes)
5. [Generation de contexte](#5-generation-de-contexte)
6. [Pipeline Inbox / Ingest](#6-pipeline-inbox--ingest)
7. [Gestion des erreurs](#7-gestion-des-erreurs)
8. [Concurrence et securite](#8-concurrence-et-securite)
9. [Mecanismes de retry et recovery](#9-mecanismes-de-retry-et-recovery)
10. [Concerns transversaux](#10-concerns-transversaux)

---

## 1. Carte des modules

### 1.1. Graphe de dependances

```
                          cli.py
                            │
                   ┌────────┼────────┐
                   ▼        ▼        ▼
            orchestrator  dream    mcp/server
                   │        │        │
         ┌────┬───┼───┬────┤        │
         ▼    ▼   ▼   ▼    ▼        ▼
      extract resolve arbit enrich  indexer
         │              │    │        │
         ▼              ▼    ▼        ▼
       llm.py        llm.py graph  store
                             │        │
                             ▼        ▼
                          scoring  mentions
                             │
                             ▼
                        context/
                      ┌────┼────┐
                      ▼    ▼    ▼
                   builder format utilities
```

### 1.2. Couches architecturales

| Couche | Modules | Responsabilite |
|--------|---------|----------------|
| **CLI** | `cli.py` | Wrappers Click minces, delegation a `orchestrator.py` |
| **Core** | `core/config.py`, `core/llm.py`, `core/models.py`, `core/utils.py`, `core/action_log.py`, `core/event_log.py`, `core/metrics.py` | Configuration, abstraction LLM, modeles Pydantic, utilitaires |
| **Pipeline** | `pipeline/orchestrator.py`, `pipeline/extractor.py`, `pipeline/resolver.py`, `pipeline/arbitrator.py`, `pipeline/enricher.py`, `pipeline/indexer.py` | Chaine d'extraction a 6 etapes |
| **Pipeline (ingest)** | `pipeline/inbox.py`, `pipeline/router.py`, `pipeline/chat_splitter.py`, `pipeline/doc_ingest.py`, `pipeline/ingest_state.py` | Ingestion fichiers et documents |
| **Pipeline (dream)** | `pipeline/dream.py`, `pipeline/dream/consolidator.py`, `pipeline/dream/discovery.py`, `pipeline/dream/merger.py`, `pipeline/dream_dashboard.py` | Reorganisation cognitive |
| **Memory** | `memory/graph.py`, `memory/store.py`, `memory/scoring.py`, `memory/mentions.py`, `memory/insights.py` | Graphe, scoring ACT-R, stockage MD |
| **Context** | `memory/context/builder.py`, `memory/context/formatter.py`, `memory/context/utilities.py` | Generation du contexte (_context.md) |
| **MCP** | `mcp/server.py` | 7 outils FastMCP (stdio) |

### 1.3. Flux de donnees entre couches

```
Utilisateur → CLI → Orchestrator → Pipeline → Memory → Context → _context.md
                                        ↕           ↕
                                      LLM.py     FAISS (indexer.py)
                                        ↕
                                   Prompts/*.md
```

Chaque couche ne communique qu'avec la couche adjacente. Pas de dependance circulaire (verifie par Agent 1 de l'audit).

---

## 2. Pipeline principal (memory run)

### 2.1. Sequence complete

**Point d'entree** : `cli.py:run()` → `orchestrator.py:run_pipeline(config, console, consolidate=True)`

```
orchestrator.py:run_pipeline()
│
├─ [Pre-flight] ingest_state.py:recover_stale_jobs(config)
│
├─ [Boucle par chat] store.py:list_unprocessed_chats(memory_path)
│   │
│   ├─ ETAPE 1 — Extraction (LLM, streaming stall-aware)
│   │   └─ extractor.py:extract_from_chat(content, config)
│   │       └─ llm.py:call_extraction(step_config, prompt, RawExtraction)
│   │           └─ llm.py:_call_with_stall_detection(...)
│   │   └─ extractor.py:sanitize_extraction(extraction)
│   │   [Retourne: RawExtraction(entities, relations, summary)]
│   │
│   ├─ ETAPE 2 — Resolution (zero LLM, deterministe)
│   │   └─ resolver.py:resolve_all(extraction, graph, faiss_search_fn)
│   │       └─ resolver.py:resolve_entity(name, graph, faiss_search_fn)
│   │           ├─ [1] Match exact par slug
│   │           ├─ [2] Containment d'alias
│   │           ├─ [3] Similarite FAISS (seuil 0.75, requete context-enriched)
│   │           └─ [4] Nouvelle entite
│   │   [Retourne: ResolvedExtraction]
│   │
│   ├─ ETAPE 3 — Arbitration (LLM, seulement si ambiguous)
│   │   └─ [pour chaque entite status="ambiguous"]
│   │       └─ arbitrator.py:arbitrate_entity(name, context, candidates, graph, config)
│   │           └─ llm.py:call_arbitration(...) → EntityResolution
│   │
│   ├─ ETAPE 4 — Enrichissement (MD + graphe + scoring)
│   │   └─ enricher.py:enrich_memory(resolved, config, today)
│   │       ├─ [Entites existantes] store.py:update_entity(...)
│   │       │   └─ mentions.py:add_mention(today, mention_dates, monthly_buckets)
│   │       │   └─ store.py:_is_duplicate_observation(line, existing_facts)
│   │       ├─ [Nouvelles entites] store.py:create_entity(...)
│   │       ├─ [Relations] graph.py:add_relation(...) (Hebbian: strength += 0.05)
│   │       │   └─ enricher.py:_check_relation_conflicts(graph, from, to, type)
│   │       ├─ scoring.py:recalculate_all_scores(graph, config)
│   │       └─ graph.py:save_graph(memory_path, graph) (atomique + lock)
│   │   [Retourne: EnrichmentReport]
│   │
│   └─ store.py:mark_chat_processed(chat_path, entities_updated, entities_created)
│
├─ ETAPE 5a — Relations batch (zero LLM, FAISS + tag overlap)
│   └─ orchestrator.py:discover_batch_relations(touched_ids, graph, config, memory_path)
│       └─ [FAISS top-3 + tag overlap >= 2 AND score >= 0.8]
│           └─ graph.py:add_relation(graph, GraphRelation("linked_to"))
│
├─ ETAPE 5b — Auto-consolidation (LLM, si consolidate=True)
│   └─ orchestrator.py:auto_consolidate(memory_path, config, console)
│       └─ [entites avec 8+ facts]
│           └─ llm.py:call_fact_consolidation(title, type, facts_text, config)
│
├─ ETAPE 6 — Generation de contexte
│   └─ [3 modes possibles, voir section 5]
│
└─ ETAPE 7 — Indexation FAISS (incrementale)
    └─ indexer.py:incremental_update(memory_path, config)
        └─ [compare hash manifest → chunk → embed → IndexFlatIP.add()]
```

### 2.2. Donnees echangees entre etapes

| De → Vers | Type | Contenu |
|------------|------|---------|
| Extraction → Sanitization | `RawExtraction` | `entities: list[RawEntity]`, `relations: list[RawRelation]`, `summary: str` |
| Sanitization → Resolution | `RawExtraction` | Types corriges, importance clampee [0,1], refs vides supprimees |
| Resolution → Arbitration | `ResolvedExtraction` | Entites avec `status: resolved\|new\|ambiguous`, `candidates` si ambiguous |
| Arbitration → Enrichment | `ResolvedExtraction` | `status` mis a jour → `resolved` ou `new` |
| Enrichment → Scoring | `GraphData` | Graphe avec entites/relations mises a jour |
| Scoring → Context | `GraphData` | Scores ACT-R recalcules pour toutes les entites |
| Context → _context.md | `str` | Markdown token-budgete pret a servir |

### 2.3. Differences `run` vs `run-light`

| Aspect | `run` | `run-light` |
|--------|-------|-------------|
| Appel | `run_pipeline(config, console, consolidate=True)` | `run_pipeline(config, console, consolidate=False)` |
| Etape 5b (auto-consolidation) | Active (LLM) | Sautee |
| Contexte | `build_context_with_llm()` si `context_llm_sections: true` | Toujours `build_context()` deterministe |

---

## 3. Serveur MCP (7 outils)

**Point d'entree** : `cli.py:serve()` → `mcp/server.py:run_server(config)`

### 3.1. get_context()

```
server.py:get_context()
  ├─ Lit _context.md
  └─ [Fallback] Lit _index.md
  └─ [Fallback] Message "No context available"
```

Aucune interaction avec d'autres modules. Lecture fichier simple.

### 3.2. save_chat(messages)

```
server.py:save_chat(messages)
  ├─ Validation structure messages (role + content requis)
  └─ store.py:save_chat(messages, memory_path)
      └─ store.py:_atomic_write_text(filepath, content)
          └─ Frontmatter: processed: false
```

### 3.3. search_rag(query) — le plus complexe

```
server.py:search_rag(query)
  │
  ├─ [Recherche semantique]
  │   └─ indexer.py:search(query, config, memory_path, top_k)
  │       └─ faiss.IndexFlatIP.search(embedding, top_k) → list[SearchResult]
  │
  ├─ [Chargement graphe]
  │   └─ graph.py:load_graph(memory_path) → GraphData
  │
  ├─ [Re-ranking hybride]
  │   ├─ [Si FTS5 disponible] RRF fusion :
  │   │   ├─ keyword_index.py:search_keyword(query, fts_db_path, top_k*2)
  │   │   └─ server.py:_rrf_fusion(faiss_results, keyword_results, graph)
  │   │       └─ score = w_sem/(k+rank_sem) + w_kw/(k+rank_kw) + w_actr/(k+rank_actr)
  │   │
  │   └─ [Sinon] Re-ranking lineaire :
  │       └─ score = faiss_score * 0.6 + graph_score * 0.4
  │
  ├─ [Re-emergence L2→L1]
  │   ├─ [Pour chaque resultat]
  │   │   └─ mentions.py:add_mention(today, entity.mention_dates, entity.monthly_buckets)
  │   └─ graph.py:save_graph(memory_path, graph) (atomique + lock)
  │
  └─ [Enrichissement relations]
      └─ [Construction adjacence bidirectionnelle → relations par resultat]
```

**Mecanisme L2→L1** : Les entites retrouvees via RAG recoivent un bump `mention_date` → le score ACT-R augmente → l'entite re-entre naturellement dans `_context.md` au prochain rebuild.

### 3.4. delete_fact(entity_name, fact_content)

```
server.py:_delete_fact_impl(entity_name, fact_content, config)
  ├─ graph.py:load_graph(memory_path)
  ├─ server.py:_resolve_entity_by_name(entity_name, graph)
  │   ├─ [1] Match slug exact
  │   ├─ [2] Match titre (case-insensitive)
  │   └─ [3] Match alias
  ├─ store.py:read_entity(filepath)
  ├─ [Match fact par contenu substring]
  ├─ facts.pop(matched_idx)
  ├─ [Ajout entree History]
  └─ store.py:write_entity(filepath, frontmatter, sections)
```

### 3.5. delete_relation(from_entity, to_entity, relation_type)

```
server.py:_delete_relation_impl(...)
  ├─ graph.py:load_graph(memory_path)
  ├─ server.py:_resolve_entity_by_name(...) [×2]
  ├─ graph.py:remove_relation(graph, from_id, to_id, relation_type)
  ├─ store.py:remove_relation_line(from_path, type, to_title)
  └─ graph.py:save_graph(memory_path, graph)
```

### 3.6. modify_fact(entity_name, old_content, new_content)

```
server.py:_modify_fact_impl(...)
  ├─ store.py:read_entity(filepath)
  ├─ [Match fact par old_content substring]
  ├─ store.py:parse_observation(old_line) → obs dict
  ├─ obs["content"] = new_content
  ├─ store.py:format_observation(obs) → new_line
  │   [Preserve category, date, valence, tags]
  └─ store.py:write_entity(filepath, frontmatter, sections)
```

### 3.7. correct_entity(entity_name, field, new_value)

```
server.py:_correct_entity_impl(...)
  ├─ [Validation: field in {title, type, aliases, retention}]
  ├─ store.py:read_entity(filepath)
  ├─ [Si field == "type"]
  │   ├─ config.get_folder_for_type(new_value) → nouveau dossier
  │   └─ shutil.move(filepath, new_filepath)
  ├─ [Mise a jour frontmatter + graph entity]
  ├─ store.py:write_entity(filepath, frontmatter, sections)
  └─ graph.py:save_graph(memory_path, graph)
```

---

## 4. Pipeline Dream (10 etapes)

**Point d'entree** : `cli.py:dream()` → `dream.py:run_dream(config, console, dry_run, step, resume, reset)`

### 4.1. Coordinateur deterministe

```
dream.py:decide_dream_steps(stats)
  ├─ Step 1 (load) : toujours
  ├─ Step 2 (extract docs) : si unextracted_docs > 0
  ├─ Step 3 (consolidate facts) : si candidates >= 3
  ├─ Step 4 (merge entities) : si candidates >= 2
  ├─ Step 5 (discover relations) : si candidates >= 5
  ├─ Step 6 (transitive) : si candidates >= 3
  ├─ Step 7 (prune dead) : si candidates >= 1
  ├─ Step 8 (summaries) : si candidates >= 3
  └─ Steps 9-10 (rescore, rebuild) : si d'autres steps ont tourne
```

Validation post-step via `dream.py:validate_dream_step()` (deterministe).

### 4.2. Interactions inter-modules

| Step | Module | Appelle | LLM |
|------|--------|---------|-----|
| 1. Load | `dream.py:_step_load()` | `graph.py:load_graph()`, scan entity MDs | Non |
| 2. Extract docs | `consolidator.py` | `extractor.py`, `resolver.py`, `enricher.py`, `indexer.py` | Oui |
| 3. Consolidate facts | `consolidator.py` | `store.py:read_entity()`, `llm.py:call_fact_consolidation()` | Oui |
| 4. Merge entities | `merger.py` | `indexer.py:search()`, `llm.py:call_dedup_check()`, `store.py`, `graph.py` | Oui |
| 5. Discover relations | `discovery.py` | `indexer.py:search()`, `llm.py:call_relation_discovery()`, `graph.py:add_relation()` | Oui |
| 6. Transitive relations | `discovery.py` | `graph.py:add_relation()` | Non |
| 7. Prune dead | `merger.py` | `shutil.move()` → `_archive/`, `graph.py:remove_entity()` | Non |
| 8. Summaries | `consolidator.py` | `llm.py:call_entity_summary()`, `store.py:update_entity()` | Oui |
| 9. Rescore | `dream.py` | `scoring.py:recalculate_all_scores()`, `graph.py:save_graph()` | Non |
| 10. Rebuild | `dream.py` | `context/builder.py:build_context()`, `indexer.py:incremental_update()` | Non* |

\* Sauf si `context_llm_sections: true` ou `context_format: natural`.

### 4.3. Checkpoint et reprise

```
dream.py:_save_checkpoint(memory_path, dream_id, step, steps_planned)
  └─ Ecrit _dream_checkpoint.json apres chaque step reussi

dream.py:_load_checkpoint(memory_path)
  └─ Filtre steps_to_run → seulement ceux > last_completed_step
  └─ Reutilise le dream_id original

dream.py:_clear_checkpoint(memory_path)
  └─ Supprime _dream_checkpoint.json en fin de succes
```

### 4.4. Dashboard temps reel

`dream_dashboard.py` : Affichage Rich Live dans le terminal.

```
dashboard.start_step(step_num, description)
  → Etat: "running" (spinner)

dashboard.complete_step(step_num, summary)
  → Etat: "done" (checkmark vert)

dashboard.fail_step(step_num, error)
  → Etat: "failed" (croix rouge)
```

---

## 5. Generation de contexte

### 5.1. Trois modes

| Mode | Declencheur config | Fonction | LLM |
|------|-------------------|----------|-----|
| **Structured** (defaut) | `context_format: structured` | `builder.py:build_context()` | Non |
| **LLM per-section** | `context_llm_sections: true` | `builder.py:build_context_with_llm()` | Oui |
| **Natural** | `context_format: natural` | `builder.py:build_natural_context()` | Optionnel |

### 5.2. Mode structured (deterministe)

```
builder.py:build_context(graph, memory_path, config)
  │
  ├─ scoring.py:get_top_entities(graph, n=50, min_score=0.3)
  │
  ├─ [Categorisation en sections]
  │   ├─ AI Personality : type=ai_self
  │   ├─ Identity : entites dans self/
  │   ├─ Work : type in (work, organization)
  │   ├─ Personal : type in (person, animal, place)
  │   └─ Top of Mind : restant a score eleve
  │
  ├─ [Pour chaque entite selectionnee]
  │   └─ formatter.py:_enrich_entity(eid, entity, graph, memory_path)
  │       ├─ formatter.py:_read_entity_facts(eid, entity, memory_path)
  │       │   └─ store.py:read_entity(entity_path)
  │       ├─ formatter.py:_filter_expired_facts(facts, config, today)
  │       │   └─ formatter.py:_is_fact_expired(obs_dict, config, today)
  │       ├─ utilities.py:_deduplicate_facts_for_context(facts, threshold=0.35)
  │       │   └─ utilities.py:_content_similarity(text_a, text_b)
  │       │       └─ 50% word Jaccard (stopwords filtres) + 50% trigram Jaccard
  │       ├─ utilities.py:_sort_facts_by_date(facts) [chronologique]
  │       └─ graph.py:get_related(graph, eid, depth=1) [BFS voisins directs]
  │
  ├─ formatter.py:_extract_vigilances(graph, selected_entities)
  │   └─ [Scan facts: [vigilance], [diagnosis], [treatment]]
  │
  ├─ [Application template prompts/context_template.md]
  │   └─ Variables: {date}, {user_language_name}, {ai_personality}, {sections},
  │     {available_entities}, {custom_instructions}
  │
  └─ builder.py:write_context(memory_path, context_text)
      └─ store.py:_atomic_write_text(filepath, content)
```

### 5.3. Mode LLM per-section

```
builder.py:build_context_with_llm(graph, memory_path, config)
  │
  ├─ [Memes etapes de selection et categorisation]
  │
  ├─ [Pour chaque section: ai_personality, identity, work, personal, top_of_mind]
  │   └─ formatter.py:_build_section_llm(section_label, entities, graph, memory_path, config, budget)
  │       ├─ [RAG pre-fetch: enrichissement par entite]
  │       └─ llm.py:call_context_section(section_label, entities_dossier, config, budget_tokens)
  │
  ├─ formatter.py:_extract_vigilances(...) [toujours deterministe]
  │
  └─ builder.py:write_context(memory_path, context_text)
```

### 5.4. Mode natural

```
builder.py:build_natural_context(graph, memory_path, config, use_llm)
  │
  ├─ formatter.py:_select_entities_for_natural(all_top, graph)
  │   └─ [Filtre: person, health, animal, ai_self + work/projects pertinents]
  │
  ├─ formatter.py:_classify_temporal(entity, today)
  │   └─ Retourne "long_term"|"medium_term"|"short_term"
  │
  ├─ [Pour chaque bucket temporel avec budget tokens]
  │   ├─ [Si use_llm] llm.py:call_natural_context_section(...)
  │   └─ [Sinon] formatter.py:_build_natural_bullet(eid, entity, graph, memory_path)
  │
  └─ builder.py:write_context(memory_path, context_text)
```

---

## 6. Pipeline Inbox / Ingest

**Point d'entree** : `cli.py:inbox()` → `inbox.py:process_inbox(memory_path, config)`

### 6.1. Flux principal

```
inbox.py:process_inbox(memory_path, config) → list[str]
  │
  └─ [Pour chaque fichier dans _inbox/ (*.md, *.txt, *.json)]
      │
      ├─ [Si .json]
      │   └─ chat_splitter.py:split_export_json(filepath, memory_path)
      │       ├─ chat_splitter.py:_parse_claude_export(data)
      │       ├─ chat_splitter.py:_parse_chatgpt_export(data)
      │       └─ chat_splitter.py:_parse_generic_json_array(data)
      │       → store.py:save_chat(messages, memory_path) [par conversation]
      │
      ├─ [Si .md/.txt ET doc_pipeline active]
      │   └─ inbox.py:_routed_ingest(filename, content, memory_path, config)
      │       ├─ ingest_state.py:compute_ingest_key(filename, content) → sha256
      │       ├─ ingest_state.py:has_been_ingested(key, config) [garde idempotence]
      │       ├─ router.py:classify(content, source_filename)
      │       │   └─ [Heuristiques deterministes: patterns de speakers, marqueurs JSON]
      │       │   └─ Retourne RouteDecision(route, confidence, reasons)
      │       ├─ [Si "conversation"] → inbox.py:_legacy_ingest(content, memory_path)
      │       │   └─ [Convertit en pseudo-chat → store.py:save_chat()]
      │       └─ [Si "document"|"uncertain"]
      │           ├─ ingest_state.py:create_job(key, config, route)
      │           ├─ ingest_state.py:transition_job(job_id, "running", config)
      │           ├─ doc_ingest.py:ingest_document(source_id, content, key, memory_path, config)
      │           │   ├─ doc_ingest.py:_normalize_text(content)
      │           │   ├─ indexer.py:chunk_text(normalized, chunk_size=400, overlap=80)
      │           │   ├─ indexer.py:get_embedding_fn(config)
      │           │   ├─ embed_fn(chunks) → np.ndarray
      │           │   ├─ faiss.IndexFlatIP.add(embeddings)
      │           │   └─ indexer.py:save_manifest(manifest_path, manifest)
      │           └─ ingest_state.py:transition_job(job_id, "succeeded", config)
      │
      └─ [Si .md/.txt ET doc_pipeline desactive]
          └─ inbox.py:_legacy_ingest(content, memory_path)
              └─ store.py:save_chat([{"role": "user", "content": content}], memory_path)
```

### 6.2. Machine a etats Ingest

```
ingest_state.py:transition_job() — Transitions valides :

  pending ──→ running
  running ──→ succeeded (terminal)
  running ──→ failed (terminal)
  running ──→ retriable
  retriable ──→ running

Fichier: _ingest_jobs.json
```

Promotion automatique `retriable → failed` si retries >= `config.ingest.max_retries`.

### 6.3. Fallback extraction → doc_ingest

```
orchestrator.py:run_pipeline() — Sur echec extraction :
  │
  ├─ orchestrator.py:is_timeout_error(e)
  │   └─ Reconnait: StallError, TimeoutError, "ReadTimeout", "ConnectTimeout"
  │
  ├─ [retries++ dans frontmatter chat]
  │
  └─ [Si timeout OU retries >= 2]
      └─ orchestrator.py:fallback_to_doc_ingest(chat_path, content, config)
          ├─ ingest_state.py:compute_ingest_key(source_id, content)
          ├─ ingest_state.py:has_been_ingested(key, config) [idempotence]
          ├─ ingest_state.py:create_job(key, config, route="fallback_doc_ingest")
          └─ doc_ingest.py:ingest_document(source_id, content, key, memory_path, config)
      └─ ingest_state.py:record_failure(source_id, error) → _retry_ledger.json
```

---

## 7. Gestion des erreurs

### 7.1. Strategie globale

Le projet applique une strategie **fail-soft** : chaque entite/relation est traitee dans son propre `try/except`. Une erreur sur un element n'arrete pas le pipeline — elle est loguee et le traitement continue.

### 7.2. Par module

#### orchestrator.py:run_pipeline()

| Etape | Exception | Action |
|-------|-----------|--------|
| Extraction | `Exception` | Incremente retry counter. Si timeout ou retries >= 2 → `fallback_to_doc_ingest()` + `record_failure()`. Sinon, log et continue. |
| Arbitration (par entite) | `Exception` | Log warning, cree nouvelle entite au lieu d'arbitrer |
| Enrichment | `Exception` | Log erreur, continue au chat suivant |
| Batch relations | `Exception` | Log warning, continue |
| Auto-consolidation (par entite) | `Exception` | Log warning jaune, continue |
| Context | `Exception` | Log warning, continue (pas de contexte cette fois) |
| FAISS | `Exception` | Log warning, continue |

#### core/llm.py

| Fonction | Mecanisme |
|----------|-----------|
| `_call_with_stall_detection()` | Worker thread + watchdog. Lock (`threading.Lock`) protege `last_activity`. Leve `StallError` si aucun token pendant `stall_timeout` secondes. Timeout double pour le premier token. |
| `_repaired_json()` | Context manager. Attrape `json.JSONDecodeError`, tente `json_repair.repair_json()`. Guard recursion via `threading.local()`. |

#### memory/graph.py:load_graph()

Recovery en 3 tiers :

```
Tier 1: Lire _graph.json
  └─ [json.JSONDecodeError|ValueError|KeyError] → Tier 2

Tier 2: Lire _graph.json.bak
  └─ [Memes exceptions] → Tier 3

Tier 3: rebuild_from_md(memory_path) sous lock
  └─ Scan tous les MDs → reconstruit le graphe
  └─ save_graph() atomiquement
```

#### pipeline/enricher.py:enrich_memory()

```
[Pour chaque entite] try/except Exception
  └─ Appends to report.errors, continue

[Pour chaque relation] try/except Exception
  └─ Appends to report.errors, continue

[Pre-consolidation gate] try/except Exception
  └─ Log warning, continue
```

#### pipeline/resolver.py:resolve_entity()

```
[Recherche FAISS] try/except Exception
  └─ Passe silencieusement → retourne "new" (fallback sans FAISS)
```

#### mcp/server.py

| Outil | Erreur | Action |
|-------|--------|--------|
| `search_rag()` | FAISS failure | Log warning, retourne resultats vides |
| `search_rag()` | Graph load failure | Log warning, retourne resultats sans enrichissement |
| `search_rag()` | L2→L1 save failure (`RuntimeError` lockfile) | Log warning, skip le bump |
| CRUD tools | Entite introuvable | Retourne JSON `{"status": "error", "message": ...}` |

#### pipeline/dream.py:run_dream()

```
[Pour chaque step] try/except Exception
  └─ dashboard.fail_step(step, error)
  └─ report.errors.append(...)
  └─ Continue au step suivant
```

#### memory/event_log.py

```
append_event(): try/except OSError → log warning, silently fails
read_events(): try/except json.JSONDecodeError per line → skip malformed
```

#### pipeline/ingest_state.py

```
_load_ledger(): try/except (JSONDecodeError|OSError) → return empty list
_load_jobs(): try/except (JSONDecodeError|OSError) → return empty dict
recover_stale_jobs(): try/except (ValueError|TypeError) per job → continue
```

### 7.3. Arbre de propagation des erreurs

```
Erreur LLM (timeout, format invalide, stall)
  │
  ├─ [Si extraction] → StallError/TimeoutError
  │   └─ orchestrator.py attrape → retry ou fallback_to_doc_ingest
  │
  ├─ [Si arbitration] → Exception quelconque
  │   └─ orchestrator.py attrape → cree nouvelle entite
  │
  ├─ [Si consolidation] → Exception quelconque
  │   └─ enricher.py ou orchestrator.py attrape → log + continue
  │
  └─ [Si dream step] → Exception quelconque
      └─ dream.py attrape → dashboard.fail_step + continue

Erreur fichier (corruption JSON, fichier manquant)
  │
  ├─ [_graph.json] → graph.py recovery 3 tiers
  ├─ [entity MD] → enricher.py/dream.py log + continue
  └─ [_ingest_jobs.json] → ingest_state.py → return empty

Erreur FAISS (index manquant, embedding failure)
  │
  ├─ [resolution] → resolver.py passe → "new" entity
  ├─ [search_rag] → server.py → resultats vides
  └─ [incremental_update] → orchestrator.py → log warning
```

---

## 8. Concurrence et securite

### 8.1. Streaming LLM avec watchdog

`llm.py:_call_with_stall_detection()` :

```
Thread principal                    Worker thread (daemon)
     │                                    │
     ├─ Cree Lock, Event, variables       │
     ├─ Lance worker ──────────────────→  ├─ _do_call()
     │                                    ├─ Streaming: chunk par chunk
     │  ┌─ Boucle watchdog (2s)           │  ├─ Lock.acquire()
     │  │  ├─ done.wait(2s)               │  ├─ last_activity = time()
     │  │  ├─ Si done → break             │  └─ Lock.release()
     │  │  ├─ Lock.acquire()              │
     │  │  ├─ idle = now - last_activity  │
     │  │  ├─ Lock.release()              ├─ [Termine]
     │  │  └─ Si idle > stall_timeout     ├─ done.set()
     │  │      → raise StallError         └─
     │  └─ Boucle
     │
     ├─ Si error set → raise error
     └─ Retourne result
```

Protection : `threading.Lock()` sur `last_activity` et `first_token_received`.
Thread-local JSON repair : `threading.local()` empeche l'interference entre threads concurrents.

### 8.2. Verrouillage fichier graphe

`graph.py:_acquire_lock()` / `_release_lock()` :

```
Lockfile: _graph.lock
Creation: os.open(..., os.O_CREAT | os.O_EXCL | os.O_WRONLY)
  → Atomique au niveau OS (race-condition safe entre processus)

Nettoyage stale: Si lock > 300s (LOCK_TIMEOUT_SECONDS)
  → Supprime le lock, retente

Utilise par:
  ├─ graph.py:save_graph() — ecriture atomique
  ├─ graph.py:load_graph() — pendant rebuild_from_md()
  └─ mcp/server.py:search_rag() — pendant L2→L1 bump save
```

### 8.3. Ecritures atomiques

Deux implementations identiques :

| Module | Fonction | Usage |
|--------|----------|-------|
| `graph.py` | `_atomic_write()` | `_graph.json` |
| `store.py` | `_atomic_write_text()` | Fichiers entite MD, chats |

Pattern commun :
```python
fd, tmp = tempfile.mkstemp(dir=filepath.parent, suffix=".tmp")
os.write(fd, content.encode("utf-8"))
os.close(fd)
os.replace(tmp, filepath)  # Atomique au niveau OS
```

`os.replace()` est atomique sur les systemes POSIX — le fichier destination est mis a jour en une seule operation, eliminant les lectures partielles.

### 8.4. Event log thread-safe

`event_log.py:append_event()` : Protege par `threading.Lock()` au niveau module.
`event_log.py:read_events()` : Pas de lock (lecture seule, pas de race d'ecriture).

### 8.5. Securite des chemins

`context/builder.py:_enrich_entity()` : Valide que les chemins d'entite restent dans `memory_path` via `Path.is_relative_to()`. Empeche les traversees de repertoire.

---

## 9. Mecanismes de retry et recovery

### 9.1. Retry extraction

```
orchestrator.py:run_pipeline()
  │
  ├─ Extraction echoue → Exception
  ├─ orchestrator.py:is_timeout_error(e)
  │   └─ Reconnait: StallError, TimeoutError, "ReadTimeout", "ConnectTimeout", "stall"
  ├─ retries = increment_extraction_retries(chat_path)
  │   └─ Compteur dans le frontmatter du chat
  │
  └─ [Si timeout OU retries >= EXTRACTION_MAX_RETRIES (2)]
      ├─ fallback_to_doc_ingest() → chunk + FAISS (pas d'entites)
      └─ record_failure() → _retry_ledger.json (status="pending", attempts=1)
```

Replay manuel via `memory replay` :
- `ingest_state.py:list_retriable()` → liste les entrees pending
- `ingest_state.py:mark_replayed()` → incremente attempts, transition vers "succeeded"/"exhausted" apres 3 tentatives

### 9.2. Recovery jobs ingestion stale

```
orchestrator.py:run_pipeline() — Pre-flight
  └─ ingest_state.py:recover_stale_jobs(config)
      └─ [Jobs bloques en "running" > recovery_threshold_seconds]
          └─ Transition vers "retriable", incremente retries
```

### 9.3. Checkpoint Dream

```
dream.py — Apres chaque step reussi :
  └─ _save_checkpoint(memory_path, dream_id, last_step, steps_planned)
      └─ Ecrit _dream_checkpoint.json

Resume: memory dream --resume
  └─ _load_checkpoint(memory_path)
      └─ Filtre steps_to_run > last_completed_step

Clear: Apres succes complet ou --reset
  └─ _clear_checkpoint(memory_path)
```

### 9.4. Recovery graphe corrompu

```
graph.py:load_graph()
  ├─ Tier 1: _graph.json [json.JSONDecodeError] → Tier 2
  ├─ Tier 2: _graph.json.bak [memes exceptions] → Tier 3
  └─ Tier 3: rebuild_from_md(memory_path) → scan tous les MDs → save_graph()
```

Le graphe est toujours reconstituable a partir des fichiers MD (source de verite L3).

---

## 10. Concerns transversaux

### 10.1. Action log

```
core/action_log.py:
  ├─ log_action(memory_path, action, entity_id, details, source)
  │   └─ Append JSON line → _actions.jsonl
  └─ read_actions(memory_path, entity_id, action, last_n)
      └─ Filtre et retourne entrees

Usage: cli.py:actions() affiche l'historique
Note: Non appele depuis le pipeline actuellement — usage futur potentiel.
```

### 10.2. Event log

```
core/event_log.py:
  ├─ append_event(memory_path, event_type, data)
  │   └─ Thread-safe (Lock), append-only JSONL
  │   └─ try/except OSError → log warning, fail silently
  └─ read_events(memory_path, event_type, since)
      └─ JSON parse par ligne, skip malformed
```

### 10.3. Metriques

```
core/metrics.py — MetricsCollector (singleton module-level)
  ├─ Compteurs: ingest_success, ingest_failure, ingest_duplicate_skip
  ├─ Histogrammes: ingest_latency, readiness_latency, router_latency
  ├─ Route counts: dict[route → count]
  └─ Timers: start_timer(name) / stop_timer(name)

Usage:
  └─ inbox.py:_routed_ingest()
      ├─ metrics.record_route(decision.route, latency)
      ├─ metrics.record_readiness(source_id) [succes]
      ├─ metrics.record_ingest_failure(source_id, error) [echec]
      └─ metrics.record_duplicate_skip(source_id) [doublon]
```

### 10.4. Mention windowing (integration ACT-R)

```
memory/mentions.py:
  ├─ add_mention(date_iso, mention_dates, monthly_buckets, window_size=50)
  │   └─ Append date → mention_dates
  │   └─ [Si len > window_size] → consolidate_window()
  │       └─ Deplace les plus anciennes dates vers monthly_buckets (YYYY-MM → count)
  │
  Appele par:
  ├─ enricher.py:_update_existing_entity() — a chaque mise a jour d'entite
  └─ mcp/server.py:search_rag() — bump L2→L1 re-emergence
  │
  Consomme par:
  └─ scoring.py:calculate_actr_base()
      └─ Utilise mention_dates (haute resolution) + monthly_buckets (agrege)
```

### 10.5. Insights cognitifs

```
memory/insights.py:
  └─ generate_insights(graph, config)
      └─ Analyse ACT-R : entites a risque d'oubli, relations faiblissantes,
         patterns de renforcement, suggestions de dream

Usage: cli.py:insights() — affiche en text ou JSON
```

### 10.6. Visualisation graphe

```
pipeline/visualize.py:generate_graph_html(graph, memory_path)
  └─ Genere _graph.html (vis.js interactif)
  └─ Noeuds colores par type, aretes par type relation
  └─ Taille noeud proportionnelle au score

Usage: cli.py:graph() — ouvre dans le navigateur
```

### 10.7. Tableau recapitulatif des fichiers de donnees

| Fichier | Ecrit par | Lu par | Thread-safe |
|---------|-----------|--------|-------------|
| `_graph.json` | `graph.py:save_graph()` | `graph.py:load_graph()` | Oui (lockfile + atomic write) |
| `_graph.json.bak` | `graph.py:save_graph()` | `graph.py:load_graph()` (fallback) | Oui |
| `_graph.lock` | `graph.py:_acquire_lock()` | `graph.py:_acquire_lock()` | Oui (O_CREAT\|O_EXCL) |
| `memory/*/[slug].md` | `store.py:create_entity()`, `write_entity()` | `store.py:read_entity()` | Oui (atomic write) |
| `memory/chats/[date].md` | `store.py:save_chat()` | `store.py:list_unprocessed_chats()` | Oui (atomic write) |
| `_context.md` | `builder.py:write_context()` | `mcp/server.py:get_context()` | Oui (atomic write) |
| `_index.md` | `builder.py:write_index()` | CLI, fallback contexte | Oui (atomic write) |
| `_memory.faiss` | `indexer.py:build_index()` | `indexer.py:search()` | Non (single-writer) |
| `_memory.pkl` | `indexer.py:_save_index()` | `indexer.py:search()` | Non (single-writer) |
| `_faiss_manifest.json` | `indexer.py:_save_index()` | `indexer.py:load_manifest()` | Non (single-writer) |
| `_memory_fts.db` | `keyword_index.py:build_keyword_index()` | `keyword_index.py:search_keyword()` | Non (SQLite single-writer) |
| `_ingest_jobs.json` | `ingest_state.py:_save_jobs()` | `ingest_state.py:_load_jobs()` | Non (single-writer) |
| `_retry_ledger.json` | `ingest_state.py:_save_ledger()` | `ingest_state.py:list_retriable()` | Non (single-writer) |
| `_actions.jsonl` | `action_log.py:log_action()` | `action_log.py:read_actions()` | Non |
| `_event_log.jsonl` | `event_log.py:append_event()` | `event_log.py:read_events()` | Oui (Lock module) |
| `_dream_checkpoint.json` | `dream.py:_save_checkpoint()` | `dream.py:_load_checkpoint()` | Non (single-writer) |
| `_graph.html` | `visualize.py:generate_graph_html()` | Navigateur | N/A |
