# MyMemory — Algorithmes et méthodes

> Chaque algorithme tel qu'implémenté dans le code.
> Références au format `fichier:fonction()`.
> Paramètres avec vraies valeurs par défaut extraites de `config.py`.

---

## Table des matières

1. [Scoring ACT-R](#1-scoring-act-r)
2. [Spreading activation](#2-spreading-activation)
3. [Hebbian learning + LTD](#3-hebbian-learning--ltd)
4. [Modulation émotionnelle](#4-modulation-émotionnelle)
5. [Résolution d'entités](#5-résolution-dentités)
6. [Indexation FAISS + recherche hybride](#6-indexation-faiss--recherche-hybride)
7. [Génération de contexte](#7-génération-de-contexte)
8. [Pipeline Dream](#8-pipeline-dream)
9. [Extraction et sanitization](#9-extraction-et-sanitization)
10. [Consolidation de faits](#10-consolidation-de-faits)

---

## 1. Scoring ACT-R

Implémenté dans `scoring.py:calculate_score()`, appelé par `scoring.py:recalculate_all_scores()`.

### 1.1 Score final

```
score = sigmoid(B + β + spreading_weight × S + emotional_boost)
```

Où :
- `B` = activation de base ACT-R
- `β` = `importance × importance_weight`
- `S` = bonus de spreading activation
- `emotional_boost` = `negative_valence_ratio × emotional_boost_weight`

### 1.2 Activation de base ACT-R

Calculée dans `scoring.py:calculate_actr_base()` :

```
B = ln(Σ t_j^(-d))
```

- `t_j` = jours depuis chaque mention (minimum 0.5 pour éviter division par zéro)
- `d` = facteur de décroissance :
  - `decay_factor_short_term` = **0.8** (entités short_term)
  - `decay_factor` = **0.5** (entités long_term/permanent)

**Sources de données** :
1. `mention_dates` (list[str]) — dates récentes, haute résolution
2. `monthly_buckets` (dict[YYYY-MM → count]) — données anciennes agrégées, date représentative = 15 du mois

Si aucune somme : retourne **-5.0** (activation très basse).

### 1.3 Paramètres de scoring

| Paramètre | Valeur défaut | Rôle |
|-----------|--------------|------|
| `decay_factor` | 0.5 | Décroissance long_term/permanent |
| `decay_factor_short_term` | 0.8 | Décroissance short_term (plus rapide) |
| `importance_weight` | 0.3 | Poids de l'importance |
| `spreading_weight` | 0.2 | Poids du spreading activation |
| `permanent_min_score` | 0.5 | Score plancher pour entités permanentes |
| `retrieval_threshold` | 0.05 | Seuil d'oubli vrai (score → 0.0 si en dessous) |
| `emotional_boost_weight` | 0.15 | Poids de la modulation émotionnelle |

### 1.4 Seuils de scoring

- **Entités permanentes** : `score >= permanent_min_score` (plancher 0.5)
- **Seuil de récupération** : si `score < retrieval_threshold` (0.05) et pas permanent → `score = 0.0` (oubli vrai, échec de récupération ACT-R)

### 1.5 Exemple de calcul

```
Entité: "Alice" (person, long_term, importance=0.7)
mention_dates: ["2026-03-01", "2026-03-05", "2026-03-10"]
monthly_buckets: {"2025-12": 5, "2026-01": 3}
negative_valence_ratio: 0.1
Relation: Alice → affects → Health (strength 0.8, reinforced 2026-03-08)
Today: 2026-03-12, decay_factor: 0.5

B (activation de base):
  - 2026-03-01: 11j → 11.5^(-0.5) = 0.294
  - 2026-03-05: 7j  → 7.5^(-0.5)  = 0.365
  - 2026-03-10: 2j  → 2.5^(-0.5)  = 0.632
  - "2025-12": 88j  → 88.5^(-0.5) × 5 = 0.830
  - "2026-01": 56j  → 56.5^(-0.5) × 3 = 0.699
  summation ≈ 2.82 → B = ln(2.82) ≈ 1.037

β = 0.7 × 0.3 = 0.21
emotional_boost = 0.1 × 0.15 = 0.015

spreading_bonus ≈ 0.6 (voir section Spreading)

activation = 1.037 + 0.21 + 0.2 × 0.6 + 0.015 = 1.382
score = sigmoid(1.382) ≈ 0.799
```

---

## 2. Spreading activation

Implémenté dans `scoring.py:spreading_activation()`. Processus en deux passes appelé par `scoring.py:recalculate_all_scores()`.

### 2.1 Passe 1 : scores de base

Pour chaque entité :
```
base[e] = sigmoid(B + importance × importance_weight)
```

### 2.2 Passe 2 : propagation par les relations

Pour chaque relation, calcul de la force effective avec décroissance en loi de puissance :

```
eff_strength = rel.strength × (days_since_reinforced + 0.5)^(-relation_decay_power)
```

- `relation_decay_power` = **0.3** (complémentaire à la LTD exponentielle)

Construction d'un graphe d'adjacence bidirectionnel. Pour chaque entité :

```
spreading_bonus = Σ(eff_strength_i × base_score_neighbor_i) / total_strength
```

Plafonnement du nombre de voisins pour éviter la dilution des hubs (`max_spreading_neighbors`).

### 2.3 Paramètres

| Paramètre | Valeur défaut | Rôle |
|-----------|--------------|------|
| `spreading_weight` | 0.2 | Poids dans le score final |
| `relation_decay_power` | 0.3 | Décroissance en loi de puissance |
| `relation_strength_base` | 0.5 | Force initiale d'une relation |

---

## 3. Hebbian learning + LTD

### 3.1 Renforcement hebbien (LTP)

Implémenté dans `graph.py:add_relation()`. Quand deux entités co-apparaissent :

```
mention_count += 1
strength = min(1.0, strength + relation_strength_growth)
last_reinforced = now
```

- `relation_strength_growth` = **0.05** (par co-occurrence, plafonné à 1.0)
- Le contexte est mis à jour si la nouvelle relation a un contexte et l'existant est vide

### 3.2 Dépression à long terme (LTD)

Implémentée dans `scoring.py:_apply_ltd()`, appelée lors de `scoring.py:recalculate_all_scores()` :

```python
if days_since_reinforced > 90:
    strength = max(0.1, strength × exp(-days / relation_ltd_halflife))
```

- `relation_ltd_halflife` = **360** jours
- Force minimale : **0.1** (les connexions inutilisées ne disparaissent jamais complètement)

### 3.3 Paramètres

| Paramètre | Valeur défaut | Rôle |
|-----------|--------------|------|
| `relation_strength_growth` | 0.05 | Incrément par renforcement |
| `relation_strength_base` | 0.5 | Force initiale |
| `relation_ltd_halflife` | 360 | Demi-vie LTD (jours) |
| `relation_decay_halflife` | 180 | Demi-vie de décroissance relation |

### 3.4 Résumé du cycle

"Les neurones qui s'activent ensemble se connectent" :
1. **Co-occurrence** → `graph.py:add_relation()` → strength ↑ (LTP)
2. **Temps sans renforcement > 90j** → `scoring.py:_apply_ltd()` → strength ↓ (LTD exponentielle)
3. **Scoring** → `scoring.py:spreading_activation()` → force effective ↓ (loi de puissance sur le temps)

---

## 4. Modulation émotionnelle

### 4.1 Calcul du ratio de valence négative

Calculé dans `graph.py:rebuild_from_md()` lors de la reconstruction du graphe :

- Faits avec valence `[-]`, ou catégories `vigilance`/`diagnosis`/`treatment` comptent comme "émotionnels"
- `negative_valence_ratio = emotional_facts / total_facts`

### 4.2 Impact sur le score

Dans `scoring.py:calculate_score()` :

```
emotional_boost = negative_valence_ratio × emotional_boost_weight
```

- `emotional_boost_weight` = **0.15**

Modélise la consolidation amygdale-hippocampe : les souvenirs émotionnels persistent plus longtemps.

---

## 5. Résolution d'entités

Implémentée dans `resolver.py:resolve_all()` → `resolver.py:resolve_entity()`. Quatre étapes séquentielles, zéro token LLM.

### 5.1 Les 4 étapes

1. **Match slug exact** : `utils.py:slugify(name)` → recherche dans `graph.entities`
2. **Containment alias** : recherche case-insensitive parmi les `aliases` de chaque entité
3. **Similarité FAISS** : `indexer.py:search()` avec seuil **0.75**, requête enrichie avec catégorie + contenu de la première observation
4. **Nouvelle entité** : si aucun match, crée une nouvelle entrée

### 5.2 Résultats possibles

- `status="resolved"` → entité existante trouvée (étapes 1-3)
- `status="new"` → aucun match (étape 4)
- `status="ambiguous"` → FAISS a trouvé **plusieurs** candidats → envoyé à l'arbitre LLM (étape 3 du pipeline)

### 5.3 Arbitrage (entités ambiguës uniquement)

`arbitrator.py:arbitrate_entity()` → `llm.py:call_arbitration()` :
- Prompt : `prompts/arbitrate_entity.md`
- Config : `config.llm_arbitration`
- Modèle réponse : `EntityResolution(action: "existing"|"new", existing_id?, new_type?)`
- En cas d'échec : fallback à "new" avec slug suggéré

### 5.4 Requête FAISS context-aware

La requête FAISS est enrichie avec la catégorie et le début du contenu de la première observation, pour améliorer la désambiguïsation. Implémenté dans `resolver.py:resolve_entity()`.

---

## 6. Indexation FAISS + recherche hybride

### 6.1 Construction de l'index

`indexer.py:build_index()` :

1. Scan des fichiers MD d'entités (`indexer.py:_get_entity_files()` — exclut `_*` et `chats/`)
2. Découpage en chunks : `indexer.py:chunk_text()` — **400 tokens**, overlap **80 tokens**
3. Embeddings : `indexer.py:get_embedding_fn()` — supporte sentence-transformers, ollama, openai
4. Normalisation L2 : `indexer.py:_normalize_l2()`
5. Index : `faiss.IndexFlatIP` (similarité cosinus sur vecteurs normalisés)
6. Sauvegarde : `_memory.faiss` + `_memory.pkl` (mapping chunks) + `_faiss_manifest.json`

### 6.2 Mise à jour incrémentale

`indexer.py:incremental_update()` :
- Charge le manifeste, vérifie si le modèle d'embedding a changé → rebuild complet si oui
- Hash les fichiers MD, détecte les changements
- V1 : rebuild complet dès qu'un fichier change

### 6.3 Recherche hybride RRF

Implémentée dans `server.py:search_rag()` → `server.py:_rrf_fusion()` :

**Quand FTS5 est disponible et `hybrid_enabled`** :

1. Recherche FAISS (sémantique) : `indexer.py:search()`
2. Recherche keyword FTS5 : `keyword_index.py:search_keyword()`
3. Fusion RRF (Reciprocal Rank Fusion) :

```
score_rrf = w_sem / (k + rank_sem) + w_kw / (k + rank_kw) + w_actr / (k + rank_actr)
```

| Paramètre | Valeur défaut | Config |
|-----------|--------------|--------|
| `w_sem` | 0.5 | `search.weight_semantic` |
| `w_kw` | 0.3 | `search.weight_keyword` |
| `w_actr` | 0.2 | `search.weight_actr` |
| `k` | 60 | `search.rrf_k` |

**Sans FTS5 ou hybrid désactivé** : re-ranking linéaire :
```
score = faiss_score × 0.6 + graph_score × 0.4
```

### 6.4 Index FTS5

`keyword_index.py:build_keyword_index()` construit un index SQLite FTS5 à partir des fichiers MD d'entités. `keyword_index.py:search_keyword()` effectue la recherche textuelle.

---

## 7. Génération de contexte

Trois modes disponibles, sélectionnés par configuration.

### 7.1 Mode structuré (déterministe, défaut)

`builder.py:build_context()` — zéro token LLM.

**Phase 1 : Sélection d'entités**
- `scoring.py:get_top_entities()` : top N (`ctx.top_entities_count` = **50**) par score, seuil `min_score_for_context` (**0.3**)
- Entités permanentes toujours incluses

**Phase 2 : Catégorisation en sections**
1. **AI Personality** : `type == "ai_self"`
2. **Identity** : `file.startswith("self/")`
3. **Work** : `type in ("work", "organization")`
4. **Personal** : `type in ("person", "animal", "place")`
5. **Top of Mind** : entités restantes, groupées par cluster (connectivité BFS via `utilities.py:_sort_by_cluster()`)

**Phase 3 : Enrichissement des entités**
`formatter.py:_enrich_entity()` par entité :
- Lecture du MD : `store.py:read_entity()`
- Collecte des faits, relations (BFS depth-1 via `graph.py:get_related()`), tags
- Dédup des faits : `utilities.py:_deduplicate_facts_for_context()` (similarité trigram, seuil `fact_dedup_threshold` = **0.35**)
- Tri chronologique : `utilities.py:_sort_facts_by_date()`
- Filtrage TTL : `formatter.py:_filter_expired_facts()` (expiration par catégorie)
- Guard path traversal : `is_relative_to()` vérifie que les chemins restent dans `memory_path`

**Phase 4 : Sections spéciales**
- **Vigilances** : `formatter.py:_extract_vigilances()` scanne les entités affichées pour faits `[vigilance]`/`[diagnosis]`/`[treatment]`
- **Brief History** : tri par récence (30j / 1an / reste)

**Phase 5 : Budget tokens**
Allocation de `context_max_tokens` (défaut **3000**) via pourcentages par section (`config.context_budget`).

**Phase 6 : Template**
`prompts/context_template.md` avec substitution : `{date}`, `{user_language_name}`, `{ai_personality}`, `{sections}`, `{available_entities}`, `{custom_instructions}`.
Instructions custom depuis `prompts/context_instructions.md`.

### 7.2 Mode LLM par section

`builder.py:build_context_with_llm()` — activé si `context_llm_sections: true`.

Pour chaque section (sauf vigilances qui restent déterministes) :
1. `builder.py:_rag_prefetch()` — pré-fetch FAISS pour contexte additionnel
2. `formatter.py:_build_section_llm()` → `llm.py:call_context_section()` avec prompt `context_section.md`
3. Config : `llm_context`

### 7.3 Mode natural

`builder.py:build_natural_context()` — activé si `context_format: "natural"`.

1. Sélection d'entités : `formatter.py:_select_entities_for_natural()` (critères différents : score ≥ 0.5, frequency ≥ 2/3, etc.)
2. Enrichissement : `formatter.py:_enrich_entity_natural()` avec bullets narratifs
3. Classification temporelle : `formatter.py:_classify_temporal()` (long_term/medium_term/short_term)
4. Template : `prompts/context_natural.md`

### 7.4 Écriture

`builder.py:write_context()` → `store.py:_atomic_write_text()` — écrit `_context.md`.
`builder.py:write_index()` → `builder.py:generate_index()` — écrit `_index.md`.
Événement loggé : `event_log.py:append_event()`.

---

## 8. Pipeline Dream

10 étapes coordonnées par `dream.py:run_dream()`. Coordinateur déterministe `dream.py:decide_dream_steps()`.

### 8.1 Planification

`dream.py:decide_dream_steps()` analyse les statistiques mémoire et retourne un `DreamPlan(steps, reasoning)`. Pas de LLM — logique conditionnelle pure.

`dream.py:_collect_dream_stats()` collecte : nombre d'entités, relations, faits, documents non extraits, entités avec beaucoup de faits, entités sans résumé, etc.

### 8.2 Les 10 étapes

| # | Nom | Type | Implémentation | Rôle |
|---|-----|------|----------------|------|
| 1 | Load | déterministe | `dream.py:_step_load()` | Charge graphe + scan MD |
| 2 | Extract docs | LLM | `dream/consolidator.py:_step_extract_documents()` | Entités depuis docs RAG non traités |
| 3 | Consolidate facts | LLM | `dream/consolidator.py:_step_consolidate_facts()` | Merge faits redondants (8+) |
| 4 | Merge entities | déterministe + LLM | `dream/merger.py:_step_merge_entities()` | Dédup slug/alias + FAISS + LLM validation |
| 5 | Discover relations | FAISS + LLM | `dream/discovery.py:_step_discover_relations()` | Nouvelles relations par similarité |
| 6 | Transitive relations | déterministe | `dream/discovery.py:_step_transitive_relations()` | Inférence A→B→C donc A→C |
| 7 | Prune dead | déterministe | `dream.py:_step_prune_dead()` | Archive entités mortes |
| 8 | Generate summaries | LLM | `dream/consolidator.py:_step_generate_summaries()` | Résumés entités sans summary |
| 9 | Rescore | déterministe | `scoring.py:recalculate_all_scores()` | ACT-R + spreading complet |
| 10 | Rebuild | déterministe | `builder.py:build_context()` + `indexer.py:build_index()` | Contexte + FAISS |

### 8.3 Détail des étapes clés

**Étape 4 — Merge entities** :
- Détection : slug/alias overlap + `dream/merger.py:_find_faiss_dedup_candidates()` (seuil **0.80**, max **20** candidats)
- Validation LLM : `llm.py:call_dedup_check()` (seuil confiance **0.7**)
- Merge : `dream/merger.py:_do_merge()` — fusionne observations, aliases, relations, conserve l'entité avec le meilleur score

**Étape 5 — Discover relations** :
- Pour chaque entité : FAISS top-5 similaires
- Filtre paires déjà liées
- Validation LLM : `llm.py:call_relation_discovery()` → `RelationProposal(action, relation_type, context)`
- Validation type : doit être dans `_VALID_RELATION_TYPES`

**Étape 6 — Transitive relations** :
| (rel_A, rel_B) | → inféré |
|-----------------|----------|
| (affects, affects) | affects |
| (part_of, part_of) | part_of |
| (requires, requires) | requires |
| (improves, affects) | improves |
| (worsens, affects) | worsens |
| (uses, part_of) | uses |

Force inférée : `min(rel_ab.strength, rel_bc.strength) × 0.5`. Seuil min : **0.4**. Cap : **20** nouvelles.

**Étape 7 — Prune dead** :
Condition : `score < 0.1 AND frequency ≤ 1 AND age > 90j AND no relations AND retention != "permanent"`.
Action : déplace vers `memory/_archive/`.

**Étape 9 — Rescore** :
1. `scoring.py:_upgrade_retention()` — promotion (jamais downgrade) :
   - ai_self → permanent
   - person/animal + frequency ≥ 3 → long_term
   - health + frequency ≥ 2 → long_term
   - any + frequency ≥ 10 + age > 30j → long_term
2. `scoring.py:_apply_ltd()` — décroissance relations inutilisées
3. `scoring.py:spreading_activation()` — two-pass
4. `scoring.py:calculate_score()` — par entité

### 8.4 Checkpoint/Resume

Sauvegarde après chaque étape : `dream.py:_save_checkpoint()` → `_dream_checkpoint.json`.
Reprise : `--resume`. Reset : `--reset`.

### 8.5 Dashboard

`dream_dashboard.py:DreamDashboard` — UI Rich Live (pending/running/done/failed/skipped).

---

## 9. Extraction et sanitization

### 9.1 Extraction

`extractor.py:extract_from_chat()` :

1. Estimation tokens : `utils.py:estimate_tokens()`
2. Si contenu > 70% de `context_window` : `extractor.py:_split_text()` découpe en segments avec overlap
3. Par segment : `llm.py:call_extraction()` avec streaming stall-aware
4. Merge : `extractor.py:_merge_extractions()` — dédup par slug/relation tuple
5. Résultat : `RawExtraction(entities, relations, summary)`

### 9.2 Streaming avec détection de stall

`llm.py:_call_with_stall_detection()` :
- Thread worker pour le streaming Instructor
- Thread watchdog vérifiant toutes les **2s**
- Grâce premier token : `stall_timeout × 2` avant le premier token
- Timeout connexion : `config.timeout × 3` pour réponses lentes mais progressantes
- `StallError` si aucun progrès pendant `stall_timeout` secondes

### 9.3 Sanitization

`extractor.py:sanitize_extraction()` — appelée dans `orchestrator.py:_run_pipeline()` et `orchestrator.py:replay()` :
- Fuzzy-map des types de relation inventés (ex: `prescrit_par` → `linked_to`)
- Fallback types d'entités invalides → `interest`
- Fallback catégories d'observations invalides → `fact`
- Clamp importance à [0, 1]
- Suppression des refs vides
- Coercion `None` → defaults

### 9.4 Fallback doc_ingest

Si extraction échoue ou retries ≥ 2 :
- `orchestrator.py:fallback_to_doc_ingest()` → `doc_ingest.py:ingest_document()`
- Normalise texte → chunks → embeddings → FAISS (pas de création d'entités)
- Chat marqué : `store.py:mark_chat_fallback()`
- Échec enregistré : `ingest_state.py:record_failure()` dans `_retry_ledger.json`

### 9.5 Réparation JSON

`llm.py:_repaired_json()` — context manager patchant `json.loads` pour auto-réparer le JSON malformé via `json-repair`. Restaure `json.loads` original pendant la réparation pour éviter la récursion.

### 9.6 Support modèles pensants

`llm.py:strip_thinking()` — supprime `<think>...</think>` des modèles raisonneurs (Qwen3, DeepSeek-R1).

---

## 10. Consolidation de faits

### 10.1 Consolidation par entité

`store.py:consolidate_entity_facts()` :

**Phase 1 — Dédup déterministe** :
`store.py:_dedup_facts_deterministic()` — similarité Levenshtein, seuil **0.85**.

**Phase 2 — Consolidation LLM** (si toujours au-dessus de `max_facts`) :
`llm.py:call_fact_consolidation()` avec prompt `consolidate_facts.md`.
- Modèle : `FactConsolidation(consolidated: list[ConsolidatedFact])`
- `ConsolidatedFact` : `category`, `content`, `date`, `importance`, `valence`, `tags`, `replaces_indices`

**Garde-fous** :
- Longueur max : **150** caractères par fait post-consolidation
- Tags max : **3** par fait
- Safety net : si > `max_facts × 2`, garde uniquement les derniers (hard cap)

### 10.2 Max facts par type

Configurable dans `config.max_facts` (dict[str, int]) :
- Défaut : **50**
- `ai_self` : **20**

Méthode : `config.py:Config.get_max_facts(entity_type)`.

### 10.3 Auto-consolidation

`orchestrator.py:auto_consolidate()` — appelée par `memory run` (pas `run-light`) :
- Pour chaque entité avec `live_facts > max_facts`
- Appelle `store.py:consolidate_entity_facts()`
- `min_facts` par défaut : **8**

### 10.4 Supersession d'observations

`store.py:mark_observation_superseded()` — marque une observation existante comme remplacée quand une nouvelle observation de même catégorie la remplace. Entrée History ajoutée.

### 10.5 Fenêtrage des mentions

`mentions.py:add_mention()` :
- Ajoute la date à `mention_dates`
- Si la liste dépasse `window_size` (**50**) : `mentions.py:consolidate_window()` déplace les dates les plus anciennes dans `monthly_buckets` (YYYY-MM → count)
