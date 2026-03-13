# Audit du code source — Résultats consolidés

> Date : 2026-03-12
> Source de vérité : code source uniquement (6 agents d'exploration parallèles)
> Aucun appel LLM effectué (dry-run uniquement)

---

## Résumé exécutif

| Métrique | Valeur |
|----------|--------|
| Modules Python (src/) | 42 |
| Fonctions/Classes | ~284 |
| Lignes de code (src/) | ~10 269 |
| Modules de test | 44 |
| Tests découverts | 426 |
| Types de fichiers générés | 16 |
| Commandes CLI | 17 |
| Outils MCP exposés | 7 |
| Dépendances circulaires | 0 |
| Fonctions orphelines | 0 |
| Imports inutilisés | 0 |

---

## Anomalies par sévérité

### CRITICAL — Crash ou résultats faux

#### C1. ~~Attribut `config.dream` inexistant~~ FIXED (Agent 4)

**Fichier** : `src/pipeline/dream/merger.py`
**Accès invalides** :
- `config.dream.faiss_merge_threshold`
- `config.dream.faiss_merge_max_candidates`
- `config.dream.dossier_max_facts`

**Impact** : `AttributeError` au runtime quand le dream mode step 4 (merge entities) est exécuté. Les steps 1-3 et 5-10 ne sont pas affectés.

**Correction requise** : Créer une dataclass `DreamConfig` avec ces 3 champs, l'ajouter à `Config`, et mettre à jour `load_config()`.

---

### SIGNIFICANT — Fonctionne mais de manière dégradée ou inattendue

#### S1. ~~Port MCP : valeur par défaut incohérente~~ FIXED (Agent 4)

| Source | Valeur |
|--------|--------|
| `config.yaml.example` | `port: 8765` |
| `config.py` (défaut) | `mcp_port: int = 8000` |

**Impact** : Confusion utilisateur. Le code utilise 8000 si rien n'est configuré, mais l'exemple montre 8765.

#### S2. Variable `{user_language_name}` non injectée dans context_template.md (Agent 3)

**Fichier** : `src/memory/context/builder.py:build_context()`
**Prompt** : `prompts/context_template.md` utilise `{user_language_name}`

**Impact** : Si le template est lu depuis le fichier (pas le fallback inline), le placeholder `{user_language_name}` reste non-résolu dans le contexte généré.

#### S3. ~~Champ `context_llm_sections` absent de config.yaml.example~~ FIXED (Agent 4)

**Impact** : Fonctionnalité non documentée pour l'utilisateur. Le champ existe dans `Config` mais pas dans l'exemple.

#### S4. ~~Champs ContextConfig absents de config.yaml.example~~ FIXED (Agent 4)

Champs manquants dans l'exemple :
- `reserved_tokens_natural` (défaut: 500)
- `reserved_tokens_structured` (défaut: 500)
- `min_budget_tokens` (défaut: 500)
- `rag_chunk_preview_len` (défaut: 200)
- `max_rag_results` (défaut: 15)
- `max_vigilance_items` (défaut: 15)
- `fact_dedup_threshold` (défaut: 0.35)

---

### MINOR — Cosmétique, performance, maintenabilité

#### M1. ~~Champ mort `context_narrative`~~ FIXED (Agent 4)

**Fichier** : `config.py` → `context_narrative: bool = False`
**Usage** : Jamais référencé dans le code. Remplacé par `context_format` ("structured" | "natural").

#### M2. ~~Prompts morts dans le répertoire `prompts/`~~ FIXED (Agent 3)

| Fichier | Raison |
|---------|--------|
| `consolidate.md` | Remplacé par `consolidate_facts.md` |
| `extract_relations.md` | Réservé v2 — les relations sont extraites dans step 1 |
| `dream_plan.md` | Remplacé par `decide_dream_steps()` déterministe |
| `dream_validate.md` | Remplacé par `validate_dream_step()` déterministe |

**Impact** : Aucun fonctionnel, mais confusion potentielle pour la maintenance.

#### M3. Constantes hardcodées non configurables (Agent 4)

**Seuils FAISS/Search** :
- `resolver.py` : `threshold=0.75` (résolution entité)
- `server.py` : `score * 0.6 + graph_score * 0.4` (pondération RAG linéaire)

**Seuils Dream** (aucune config n'existe) :
- `dream.py` : `score < 0.1` (seuil entité morte)
- `dream.py` : `similarity_threshold=0.80` (dédup FAISS)
- `dream.py` : `confidence < 0.7` (seuil verdict dédup)
- `dream/discovery.py` : `min_strength=0.4`, `max_new=20`

**Seuils Scoring/Sélection** :
- `formatter.py` : `frequency >= 2/3`, `score >= 0.6`
- `scoring.py` : `days > 90` (déclencheur LTD), `max(0.1, ...)` (force minimale relation)

**Seuils temporels** :
- `formatter.py` : `days > 30` (seuil d'âge)
- `scoring.py` : `frequency >= 3/2/10`, `days > 30` (règles de rétention)

---

### INFO — Observations utiles

#### I1. Architecture bien structurée (Agent 1)

- Zéro dépendance circulaire
- Toutes les fonctions "privées" (`_prefixées`) sont appelées dans leur module
- Séparation nette : core (config/models/llm) → memory (graph/store/scoring) → pipeline (extractor/resolver/enricher) → mcp (server)

#### I2. Couverture complète des prompts (Agent 3)

- 9/16 prompts activement chargés via `load_prompt()`
- 4/16 templates lus directement (context_template, context_natural, context_instructions, context_natural_section)
- 4/16 morts (voir M2)
- 0 prompt inline (toutes les interactions LLM passent par des fichiers)
- Correspondance variables template ↔ code : 100% vérifiée

#### I3. 17 commandes CLI opérationnelles (Agent 5, 6)

Toutes les commandes retournent exit code 0 en dry-run. 426 tests découverts sans erreur de collection.

#### I4. Appels LLM — 9 fonctions distinctes (Agent 3)

| Fonction | Config utilisée | Streaming | Modèle réponse |
|----------|----------------|-----------|----------------|
| `call_extraction()` | `llm_extraction` | Oui (stall detection) | `RawExtraction` |
| `call_arbitration()` | `llm_arbitration` | Non | `EntityResolution` |
| `call_fact_consolidation()` | `llm_consolidation` | Non | `FactConsolidation` |
| `call_entity_summary()` | `llm_context` | Non | `EntitySummary` |
| `call_relation_discovery()` | `llm_dream_effective` | Non | `RelationProposal` |
| `call_dedup_check()` | `llm_dream_effective` | Non | `DedupVerdict` |
| `call_context_generation()` | `llm_context` | Non | str |
| `call_context_section()` | `llm_context` | Non | str |
| `call_natural_context_section()` | `llm_context` | Non | str |

#### I5. 16 fichiers de données générés au runtime (Agent 1)

| Fichier | Créé par | Lu par | Format |
|---------|----------|--------|--------|
| `_graph.json` | `graph:save_graph()` | `graph:load_graph()` | JSON |
| `_graph.json.bak` | `graph:save_graph()` | `graph:load_graph()` (fallback) | JSON |
| `_graph.lock` | `graph:_acquire_lock()` | Polling lock | Lockfile |
| `memory/*/[slug].md` | `store:create_entity()` | `store:read_entity()` | MD + YAML |
| `memory/chats/[date].md` | `store:save_chat()` | `store:list_unprocessed_chats()` | MD + YAML |
| `_context.md` | `context/builder:write_context()` | `mcp:get_context()` | Markdown |
| `_index.md` | `context/builder:write_index()` | CLI, fallback | Markdown |
| `_memory.faiss` | `indexer:build_index()` | `indexer:search()` | FAISS binaire |
| `_memory.pkl` | `indexer:_save_index()` | `indexer:search()` | Pickle |
| `_faiss_manifest.json` | `indexer:_save_index()` | `indexer:load_manifest()` | JSON |
| `_memory_fts.db` | `keyword_index:build_keyword_index()` | `keyword_index:search_keyword()` | SQLite FTS5 |
| `_ingest_jobs.json` | `ingest_state:_save_jobs()` | `ingest_state:_load_jobs()` | JSON |
| `_retry_ledger.json` | `ingest_state:_save_ledger()` | `ingest_state:list_retriable()` | JSON |
| `_actions.jsonl` | `action_log:log_action()` | `action_log:read_actions()` | JSONL |
| `_event_log.jsonl` | `event_log:append_event()` | `event_log:read_events()` | JSONL |
| `_dream_checkpoint.json` | `dream:_save_checkpoint()` | `dream:_load_checkpoint()` | JSON |
| `_graph.html` | `visualize:generate_graph_html()` | Navigateur | HTML |

#### I6. Différences exactes `run` vs `run-light` (Agent 5)

`run` appelle `run_pipeline(config, console, consolidate=True)` :
- Étape 5b : auto-consolidation LLM activée
- Contexte : peut utiliser `build_context_with_llm()` si `context_llm_sections: true`

`run-light` appelle `run_pipeline(config, console, consolidate=False)` :
- Étape 5b : sautée (pas de consolidation LLM)
- Contexte : toujours `build_context()` déterministe
