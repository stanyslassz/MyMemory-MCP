# Design: MyMemory — Amélioration Globale

## Contexte

Le projet MyMemory est un système de mémoire persistante pour LLMs locaux. Un audit exhaustif (10 fichiers, 6243 lignes, `docs/audit/`) a révélé 10 bugs confirmés, 7 gaps architecturaux, et des problèmes de prompts pour petits LLMs.

**Problème principal rapporté par l'utilisateur** : les relations entre entités ne se corrigent jamais. Exemple : "Louise est ma fille, Anaïs est ma femme" — le système garde les anciennes relations inverses car il n'existe aucun mécanisme de suppression/supersession pour les relations.

---

## Phase 1 — Correctifs Critiques

### 1.1 Mécanisme de suppression/supersession des relations

**Problème** : Relations append-only. `add_relation()` dans `graph.py` ne fait que renforcer ou créer. Pas de `remove_relation()`. Le modèle `RawRelation` n'a pas de champ `supersedes` (contrairement à `RawObservation`).

**Design** :

#### 1.1.1 Nouveau champ `supersedes` sur `RawRelation` (models.py)

Ajouter un champ optionnel `supersedes` au modèle `RawRelation` pour permettre au LLM de signaler qu'une nouvelle relation remplace une ancienne.

```python
class RawRelation(BaseModel):
    from_name: str
    to_name: str
    type: RelationType
    context: str = ""
    supersedes: str = ""  # Format: "from_slug:to_slug:relation_type" — marks old relation for removal
```

#### 1.1.2 Nouvelle fonction `remove_relation()` dans graph.py

Fonction déterministe de suppression par tuple `(from, to, type)`.

```python
def remove_relation(graph: GraphData, from_entity: str, to_entity: str, rel_type: str) -> bool:
    """Remove a specific relation by (from, to, type) tuple. Returns True if found and removed."""
    before = len(graph.relations)
    graph.relations = [
        r for r in graph.relations
        if not (r.from_entity == from_entity and r.to_entity == to_entity and r.type == rel_type)
    ]
    return len(graph.relations) < before
```

#### 1.1.3 Relation supersession dans enricher.py

Dans `_process_relations()`, avant l'appel à `add_relation()` :

- Vérifier si `relation.supersedes` est défini
- Si oui, parser le format `"from_slug:to_slug:type"` et appeler `remove_relation()`
- Supprimer également la ligne de relation dans la section `## Relations` du fichier MD de l'entité source
- Puis ajouter la nouvelle relation normalement

#### 1.1.4 Mise à jour du prompt d'extraction (extract_facts.md)

Ajouter une section après les instructions `supersedes` pour les observations :

```markdown
### Relation corrections
If the user corrects a relationship (e.g., "Louise is not my wife, she's my daughter"):
- Extract the NEW correct relation normally
- Set `supersedes` to the old relation in format "from_slug:to_slug:old_type"
- Example: user says "Louise est ma fille, pas ma femme"
  → new relation: {from_name: "Alexis", to_name: "Louise", type: "parent_of", supersedes: "alexis:louise:linked_to"}
```

#### 1.1.5 Suppression de lignes relation dans store.py

Ajouter une fonction `remove_relation_line()` pour nettoyer les fichiers MD côté disque :

```python
def remove_relation_line(entity_path: Path, relation_type: str, target_title: str) -> bool:
    """Remove a specific relation line from ## Relations section."""
```

---

### 1.2 Correction du bug supersedes observations

**Problème** : Le champ `supersedes` est extrait par le LLM mais `mark_observation_superseded()` n'est jamais appelé dans le pipeline normal. Uniquement appelé dans les tests.

**Fix** : Dans `enricher.py`, `_update_existing_entity()`, après le check de déduplication et avant l'ajout de nouvelles observations :

```python
for obs in new_observations:
    if obs.get("supersedes"):
        mark_observation_superseded(entity_path, obs["supersedes"])
```

Le code existe déjà à `enricher.py:131-140` mais nécessite une vérification qu'il est bien exécuté dans le flow normal de `enrich_memory()` — l'audit a identifié qu'il pourrait se trouver dans une branche morte.

---

### 1.3 Filtrage des relations dans le contexte (context.py)

**Problème** : `_enrich_entity()` dans `context.py` inclut TOUTES les relations sans filtrage par force, âge ou validité.

**Fix** : Dans `_enrich_entity()`, filtrer les relations avant inclusion :

```python
# Filter weak/stale relations
min_strength = 0.3
max_age_days = 365
filtered_relations = [
    r for r in entity_relations
    if r.strength >= min_strength
    and (today - parse_date(r.last_reinforced)).days <= max_age_days
]
```

---

### 1.4 Correction dream_plan.md step numbering

**Problème** : Le prompt liste 9 steps, le code en a 10. Le step 6 (transitive relations) est manquant.

**Fix** : Mettre à jour `dream_plan.md` pour correspondre à la structure à 10 steps du code. Ajouter la description du step 6.

---

### 1.5 Fix call_fact_consolidation LLM config

**Problème** : Utilise `config.llm_context` au lieu de `config.llm_dream_effective`.

**Fix** : Dans `llm.py:365`, changer pour utiliser le bon chemin de config.

---

### 1.6 Fix prompt overhead constant

**Problème** : Constante hardcodée à 500 tokens alors que le prompt fait ~1200+.

**Fix** : Dans `extractor.py:201`, augmenter à 1500 ou calculer dynamiquement depuis le template du prompt.

---

### 1.7 Fix vector normalization pour Ollama/OpenAI

**Problème** : Seul `sentence-transformers` normalise les vecteurs. `IndexFlatIP` suppose des vecteurs normalisés.

**Fix** : Dans `indexer.py`, `_get_embedding_fn()`, ajouter une normalisation L2 pour tous les providers :

```python
import numpy as np
vec = np.array(vec, dtype=np.float32)
norm = np.linalg.norm(vec)
if norm > 0:
    vec = vec / norm
```

---

### 1.8 Fix recalculate_all_scores on every search_rag

**Problème** : Calcul O(E+R) à chaque requête.

**Fix** : Ne bumper que les `mention_dates` et recalculer le score uniquement pour les entités récupérées, pas le graphe entier. Alternativement, utiliser un dirty flag et un recalcul batch.

---

### 1.9 Fix tags perdus lors de la création d'entité

**Problème** : `enricher.py:222-225` supprime les tags des observations.

**Fix** : Passer les tags à l'appel `create_entity()`.

---

### 1.10 Fix recover_stale_jobs jamais appelé

**Fix** : Appeler `recover_stale_jobs()` au début de `run_pipeline()` dans `orchestrator.py`.

---

## Phase 2 — MCP CRUD & Correction Interactive

### 2.1 Nouveaux outils MCP

Ajouter 4 nouveaux outils à `src/mcp/server.py` :

#### `delete_fact(entity_name, fact_content)`

- Résout l'entité par nom/slug/alias
- Trouve le fait correspondant dans le MD (fuzzy match sur le contenu)
- Le marque comme `[superseded]` ou le supprime
- Met à jour le graphe (recalcule l'importance à partir des faits restants)
- Retourne une confirmation avec le fait supprimé

#### `delete_relation(from_entity, to_entity, relation_type)`

- Résout les deux entités
- Appelle `remove_relation()` sur le graphe
- Supprime la ligne de relation des deux fichiers MD
- Sauvegarde le graphe
- Retourne une confirmation

#### `modify_fact(entity_name, old_content, new_content)`

- Résout l'entité
- Trouve le fait correspondant (fuzzy)
- Remplace le contenu, préserve category/date/valence
- Met à jour le MD de l'entité
- Retourne ancien vs nouveau

#### `correct_entity(entity_name, corrections)`

- Prend un dict de corrections : `{field: new_value}`
- Supporte : `title`, `type`, `aliases`, `retention`
- Met à jour le frontmatter MD et le graphe
- Si le type change, déplace le fichier vers le bon dossier

---

### 2.2 Modèle de permission

Tous les outils de modification nécessitent un pattern de confirmation :

- L'outil retourne un aperçu des changements
- Le LLM doit confirmer avec l'utilisateur avant d'appliquer
- Approche retenue pour MCP : rendre les outils idempotents et faire confiance au LLM pour confirmer avec l'utilisateur dans la conversation. Le protocole MCP ne supporte pas nativement la confirmation multi-étapes.

---

### 2.3 Rate limiting

Ajouter du rate limiting pour prévenir les abus :

- Max 10 modifications par session
- Max 5 suppressions d'entités par session
- Cooldown : 1 seconde entre les appels de modification

---

## Phase 3 — Recherche Hybride RRF

### 3.1 Index Keyword SQLite FTS5

**Fichiers** : Nouveau fichier `src/pipeline/keyword_index.py` + modifications à `indexer.py`

#### Architecture

```
Entity MD files
  → build_keyword_index()
    → SQLite DB with FTS5 virtual table
    → Columns: entity_id, chunk_idx, content (FTS5 indexed)
  → search_keyword(query, top_k)
    → FTS5 BM25 ranking
    → Returns list[KeywordResult(entity_id, chunk_idx, bm25_score)]
```

#### Schéma FTS5

```sql
CREATE VIRTUAL TABLE memory_fts USING fts5(
    entity_id,
    chunk_idx,
    content,
    tokenize='unicode61 remove_diacritics 2'
);
```

- `remove_diacritics 2` gère les accents français (é→e, ç→c pour le matching)
- Stocké aux côtés des fichiers FAISS : `_memory_fts.db`

#### Flow de construction

- Appelé durant `incremental_update()` aux côtés du rebuild FAISS
- Même chunking que FAISS (400 tokens, 80 overlap) pour l'alignement
- Détection de changements basée sur le manifest (partagé avec FAISS)

---

### 3.2 Reciprocal Rank Fusion dans search_rag

**Fichier** : `src/mcp/server.py`

Fusion de 3 signaux : sémantique (FAISS cosine similarity), keyword (FTS5 BM25), et cognitif (score ACT-R).

```python
def _rrf_fusion(faiss_results, keyword_results, graph, k=60,
                weight_semantic=0.5, weight_keyword=0.3, weight_actr=0.2):
    """
    Reciprocal Rank Fusion combining 3 signals:
    - Semantic (FAISS cosine similarity)
    - Keyword (FTS5 BM25)
    - Cognitive (ACT-R score)

    rrf_score = w_sem/(k + rank_sem) + w_kw/(k + rank_kw) + w_actr/(k + rank_actr)
    """
    # Rank each result set
    semantic_ranks = {r.entity_id: i+1 for i, r in enumerate(faiss_results)}
    keyword_ranks = {r.entity_id: i+1 for i, r in enumerate(keyword_results)}

    # Collect all unique entity_ids
    all_ids = set(semantic_ranks) | set(keyword_ranks)

    scored = []
    for eid in all_ids:
        sem_rank = semantic_ranks.get(eid, len(faiss_results) + 1)
        kw_rank = keyword_ranks.get(eid, len(keyword_results) + 1)
        actr_score = graph.entities.get(eid, GraphEntity()).score
        # ACT-R rank: higher score = lower rank
        actr_rank = ...  # computed from sorted scores

        score = (weight_semantic / (k + sem_rank) +
                 weight_keyword / (k + kw_rank) +
                 weight_actr / (k + actr_rank))
        scored.append((eid, score))

    return sorted(scored, key=lambda x: x[1], reverse=True)
```

---

### 3.3 Configuration

```yaml
search:
  hybrid_enabled: true
  rrf_k: 60
  weight_semantic: 0.5
  weight_keyword: 0.3
  weight_actr: 0.2
  fts_db_path: "_memory_fts.db"
```

Fallback : si `hybrid_enabled: false`, utiliser le mode actuel FAISS-only + re-ranking linéaire ACT-R.

---

## Phase 4 — Prompts & Petits LLMs

### 4.1 Simplification extract_facts.md

**Stratégie** : Réduire les tâches cognitives simultanées de 6+ à 3.

Changements :

1. **Ajouter des exemples par catégorie** — les petits LLMs ont besoin d'exemples concrets, pas juste des noms de catégories
2. **Retirer `supersedes` du prompt** — gérer les corrections en post-processing (détection déterministe via similarité de contenu entre anciens et nouveaux faits)
3. **Ajouter les listes de catégories en clair** — actuellement utilise `{categories_observations}` qui est une liste brute. Formater en tableau avec descriptions.
4. **Simplifier le JSON** — réduire les champs d'observation de 7 à 5 (retirer `supersedes`, rendre `tags` optionnel avec défaut `[]`)
5. **Ajouter une date ancre** — injecter `{today}` pour la résolution de dates relatives ("yesterday" → date réelle)

---

### 4.2 Dream coordinator — Règles déterministes

**Remplacer** `call_dream_plan()` par une fonction déterministe :

```python
def decide_dream_steps(stats: dict) -> list[int]:
    steps = [1]  # Load always
    if stats.get("unextracted_docs", 0) > 0: steps.append(2)
    if stats.get("consolidation_candidates", 0) >= 3: steps.append(3)
    if stats.get("merge_candidates", 0) >= 2: steps.append(4)
    if stats.get("relation_candidates", 0) >= 5: steps.append(5)
    if stats.get("transitive_candidates", 0) >= 3: steps.append(6)
    if stats.get("prune_candidates", 0) >= 1: steps.append(7)
    if stats.get("summary_candidates", 0) >= 3: steps.append(8)
    if any(s in steps for s in [2, 3, 4, 5, 6, 7, 8]):
        steps.extend([9, 10])  # Rescore + Rebuild
    return sorted(set(steps))
```

Avantages : zéro token LLM, instantané, 100% reproductible, pas de bugs de numérotation de steps.

---

### 4.3 Dream validate — Validation déterministe

Remplacer `call_dream_validate()` par des vérifications déterministes :

```python
def validate_dream_step(step: int, before_state: dict, after_state: dict) -> tuple[bool, list[str]]:
    issues = []
    if step == 3:  # Consolidation
        if after_state["total_facts"] > before_state["total_facts"]:
            issues.append("Consolidation increased fact count")
    if step == 4:  # Merge
        if after_state["total_entities"] > before_state["total_entities"]:
            issues.append("Merge increased entity count")
    # etc.
    return len(issues) == 0, issues
```

---

### 4.4 Simplification consolidate_facts.md

Découper en 2 phases :

1. **Phase déterministe** : Dedup Levenshtein (seuil 0.85) — supprimer les doublons exacts/quasi sans LLM
2. **Phase LLM** (si toujours au-dessus de `max_facts`) : prompt simplifié demandant UNIQUEMENT de fusionner les faits sémantiquement similaires, avec max 3 instructions au lieu de 8+

---

### 4.5 Simplification discover_relations.md

Réduire le contexte par entité à : title + type + top 3 facts (par importance) + tags. Max ~200 tokens par entité au lieu du dossier complet (500+).

---

### 4.6 Enrichir arbitrate_entity.md

Ajouter les faits et le résumé des entités candidates pour que le LLM ait assez de contexte pour désambiguïser. Utiliser le format `[EXISTING_N]` au lieu du `entity_id` brut.

---

### 4.7 Structured output pour summarize_entity.md

```python
class EntitySummary(BaseModel):
    summary: str = Field(max_length=150)
```

Remplacer le free-text par un output validé par Instructor.

---

## Phase 5 — Nice-to-Have

### 5.1 Historique centralisé (_actions.jsonl)

**Fichier** : `_actions.jsonl` (append-only, une ligne JSON par action)

```python
# src/core/action_log.py
import json
from datetime import datetime
from pathlib import Path

def log_action(memory_path: Path, action: str, entity_id: str = "",
               details: dict = None, source: str = "pipeline"):
    """Append action to centralized log."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "action": action,  # create, update, delete, merge, consolidate, archive, add_relation, remove_relation
        "entity_id": entity_id,
        "source": source,  # pipeline, dream, mcp, manual
        "details": details or {}
    }
    log_path = memory_path / "_actions.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
```

#### Hook points

| Module | Actions loguées |
|--------|----------------|
| `enricher.py` | `create_entity`, `update_entity`, `add_relation` |
| `dream.py` | `merge`, `prune`, `consolidate`, `discover_relations` |
| `mcp/server.py` | `delete_fact`, `delete_relation`, `modify_fact` (Phase 2) |
| `store.py` | `mark_observation_superseded`, `consolidate_entity_facts` |

#### Commande CLI

```
memory actions [--last N] [--entity NAME] [--action TYPE]
```

Pour interroger le log.

---

### 5.2 Insights ACT-R

**Fichier** : Nouvelle commande dans `cli.py` + fonctions d'analyse dans `scoring.py`

```
memory insights [--format json|text]
```

Sections de sortie :

| Section | Description |
|---------|-------------|
| **Forgetting curve** | Entités approchant le `retrieval_threshold` (0.05) |
| **L1 movements** | Entités entrées/sorties de `_context.md` depuis le dernier run |
| **Emotional hotspots** | Entités avec un `negative_valence_ratio` élevé |
| **Relation health** | Relations approchant le seuil LTD (strength < 0.2, non renforcées > 90j) |
| **Network hubs** | Entités avec le plus de connexions (degree centrality) |
| **Scoring distribution** | Histogramme des scores (buckets : 0-0.1, 0.1-0.3, 0.3-0.5, 0.5-0.7, 0.7-1.0) |

Toutes les données viennent de `_graph.json` — zéro LLM, calcul instantané.

---

### 5.3 Dream Step 4 LLM Deduplication

**Enhancement du step 4 dans `dream.py`** :

Après les merges déterministes par slug/alias, ajouter une expansion de candidats via FAISS :

```python
# After deterministic merges
for entity_id, entity in graph.entities.items():
    faiss_results = search(entity.title, config, memory_path, top_k=5)
    for result in faiss_results:
        if (result.entity_id != entity_id
            and result.score > 0.8
            and graph.entities[result.entity_id].type == entity.type
            and not already_merged(entity_id, result.entity_id)):
            # LLM confirmation
            verdict = call_dedup_check(entity, graph.entities[result.entity_id], config)
            if verdict.is_duplicate:
                merge_candidates.append((entity_id, result.entity_id))
```

#### Nouvelle fonction LLM

```python
class DedupVerdict(BaseModel):
    is_duplicate: bool
    confidence: float = Field(ge=0, le=1)
    reason: str = ""

def call_dedup_check(entity_a: GraphEntity, entity_b: GraphEntity, config) -> DedupVerdict:
    """Ask LLM if two FAISS-similar entities are duplicates."""
```

#### Nouveau prompt : `prompts/dedup_check.md`

Prompt minimal (~100 tokens) :

```markdown
Are these two entities the same thing?

Entity A: {title_a} ({type_a}) — {summary_a}
Entity B: {title_b} ({type_b}) — {summary_b}

Answer as JSON: {"is_duplicate": true/false, "confidence": 0.0-1.0, "reason": "..."}
```

---

## Dépendances entre phases

```
Phase 1 (bugs) ──→ Phase 2 (MCP CRUD) ──→ Phase 5.1 (action log hooks MCP)
     │
     ├──→ Phase 3 (RRF) ── indépendant
     │
     ├──→ Phase 4 (prompts) ── indépendant
     │
     └──→ Phase 5.2-5.3 (insights, dedup) ── indépendant après Phase 1
```

- **Phase 1** est le prérequis pour tout le reste.
- **Phases 3, 4, 5** peuvent être développées en parallèle après Phase 1.
- **Phase 2** dépend de Phase 1 (utilise `remove_relation()`).
- **Phase 5.1** peut être faite à tout moment mais ses hooks MCP dépendent de Phase 2.

---

## Fichiers impactés (résumé)

| Fichier | Phases |
|---------|--------|
| `src/core/models.py` | 1, 2, 5.3 |
| `src/core/llm.py` | 1, 4, 5.3 |
| `src/core/config.py` | 3 |
| `src/memory/graph.py` | 1 |
| `src/memory/scoring.py` | 1, 5.2 |
| `src/memory/store.py` | 1, 2 |
| `src/memory/context.py` | 1 |
| `src/pipeline/enricher.py` | 1 |
| `src/pipeline/extractor.py` | 1, 4 |
| `src/pipeline/indexer.py` | 1, 3 |
| `src/pipeline/dream.py` | 1, 4, 5.3 |
| `src/pipeline/orchestrator.py` | 1 |
| `src/mcp/server.py` | 2, 3 |
| `src/cli.py` | 5.1, 5.2 |
| `prompts/extract_facts.md` | 1, 4 |
| `prompts/dream_plan.md` | 1, 4 |
| `prompts/consolidate_facts.md` | 4 |
| `prompts/discover_relations.md` | 4 |
| `prompts/arbitrate_entity.md` | 4 |
| `prompts/summarize_entity.md` | 4 |
| **NEW:** `src/pipeline/keyword_index.py` | 3 |
| **NEW:** `src/core/action_log.py` | 5.1 |
| **NEW:** `prompts/dedup_check.md` | 5.3 |
