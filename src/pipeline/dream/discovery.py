"""Dream steps 5-6: Relation discovery and transitive inference."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import get_args

from rich.console import Console

from src.core.config import Config
from src.core.models import GraphData, GraphRelation, RelationType

logger = logging.getLogger(__name__)

_VALID_RELATION_TYPES: set[str] = set(get_args(RelationType))

# Transitive inference rules: (rel_type_A, rel_type_B) -> inferred_type
_TRANSITIVE_RULES: dict[tuple[str, str], str] = {
    ("affects", "affects"): "affects",
    ("part_of", "part_of"): "part_of",
    ("requires", "requires"): "requires",
    ("improves", "affects"): "improves",
    ("worsens", "affects"): "worsens",
    ("uses", "part_of"): "uses",
}


def _step_discover_relations(
    graph: GraphData,
    memory_path: Path,
    config: Config,
    console: Console,
    report,
    dry_run: bool,
) -> None:
    """Step 5: Use FAISS similarity + LLM to discover new relations."""
    from src.memory.rag import search as rag_search, SearchOptions
    from src.memory.graph import add_relation, save_graph
    from src.pipeline.dream.merger import _build_dossier

    # Build existing relation set for fast lookup
    existing_rels: set[tuple[str, str]] = set()
    for rel in graph.relations:
        existing_rels.add((rel.from_entity, rel.to_entity))
        existing_rels.add((rel.to_entity, rel.from_entity))

    candidates: list[tuple[str, str]] = []
    entity_ids = list(graph.entities.keys())

    for eid in entity_ids:
        entity = graph.entities[eid]
        try:
            results = rag_search(entity.title, config, memory_path, SearchOptions(
                top_k=5, bump_mentions=False, use_fts5=False, rerank_actr=False,
            ))
        except Exception:
            continue

        for result in results:
            other_id = result.entity_id
            if other_id == eid:
                continue
            if other_id not in graph.entities:
                continue
            pair = tuple(sorted([eid, other_id]))
            if pair in existing_rels:
                continue
            if pair not in {tuple(sorted(c)) for c in candidates}:
                candidates.append((eid, other_id))

    if not candidates:
        console.print("  No new relation candidates found")
        return

    console.print(f"  Found {len(candidates)} candidate pair(s) to evaluate")

    from src.core.llm import call_relation_discovery

    discovered = 0
    for eid_a, eid_b in candidates:
        entity_a = graph.entities.get(eid_a)
        entity_b = graph.entities.get(eid_b)
        if not entity_a or not entity_b:
            continue

        dossier_a = _build_dossier(eid_a, entity_a, memory_path, config)
        dossier_b = _build_dossier(eid_b, entity_b, memory_path, config)

        if dry_run:
            console.print(f"  [dim]Would evaluate: {entity_a.title} <-> {entity_b.title}[/dim]")
            continue

        try:
            proposal = call_relation_discovery(
                entity_a.title, entity_a.type, dossier_a,
                entity_b.title, entity_b.type, dossier_b,
                config,
            )

            if proposal.action == "relate" and proposal.relation_type:
                rel_type = proposal.relation_type
                if rel_type not in _VALID_RELATION_TYPES:
                    logger.warning("Dream discovered invalid relation type '%s', skipping", rel_type)
                    continue

                new_rel = GraphRelation(
                    from_entity=eid_a,
                    to_entity=eid_b,
                    type=rel_type,
                    context=proposal.context,
                )
                add_relation(graph, new_rel, strength_growth=config.scoring.relation_strength_growth)
                discovered += 1
                console.print(f"    [green]{entity_a.title} -> {rel_type} -> {entity_b.title}[/green]")
        except Exception as e:
            report.errors.append(f"Relation discovery failed for {eid_a}/{eid_b}: {e}")

    report.relations_discovered = discovered
    if discovered and not dry_run:
        save_graph(memory_path, graph)
        console.print(f"  [green]Discovered {discovered} new relation(s)[/green]")


def _step_transitive_relations(
    graph: GraphData,
    memory_path: Path,
    config: Config,
    console: Console,
    report,
    dry_run: bool,
    min_strength: float | None = None,
    max_new: int | None = None,
) -> None:
    """Step 6: Infer transitive relations (deterministic, no LLM).

    For each triple (A->rel1->B, B->rel2->C) where A and C have no direct relation,
    apply transitive rules to create inferred relations with reduced strength.
    """
    min_strength = min_strength if min_strength is not None else config.dream.transitive_min_strength
    max_new = max_new if max_new is not None else config.dream.transitive_max_new
    from src.memory.graph import add_relation, save_graph

    # Build adjacency: entity -> list of (target, relation)
    adjacency: dict[str, list[tuple[str, GraphRelation]]] = defaultdict(list)
    for rel in graph.relations:
        if rel.strength >= min_strength:
            adjacency[rel.from_entity].append((rel.to_entity, rel))

    # Build existing relation set
    existing: set[tuple[str, str]] = set()
    for rel in graph.relations:
        existing.add((rel.from_entity, rel.to_entity))
        existing.add((rel.to_entity, rel.from_entity))

    discovered = 0
    for entity_a, neighbors_a in adjacency.items():
        if discovered >= max_new:
            break
        for entity_b, rel_ab in neighbors_a:
            if discovered >= max_new:
                break
            for entity_c, rel_bc in adjacency.get(entity_b, []):
                if discovered >= max_new:
                    break
                if entity_c == entity_a:
                    continue
                if (entity_a, entity_c) in existing:
                    continue
                # Check transitive rule
                rule_key = (rel_ab.type, rel_bc.type)
                inferred_type = _TRANSITIVE_RULES.get(rule_key)
                if not inferred_type:
                    continue
                # Inferred strength = min of both * 0.5
                inferred_strength = min(rel_ab.strength, rel_bc.strength) * 0.5

                title_a = graph.entities.get(entity_a)
                title_b = graph.entities.get(entity_b)
                title_c = graph.entities.get(entity_c)
                if not title_a or not title_c:
                    continue
                context = f"transitive: {title_a.title} →{rel_ab.type}→ {title_b.title if title_b else entity_b} →{rel_bc.type}→ {title_c.title}"

                if dry_run:
                    console.print(f"  [dim]Would infer: {title_a.title} →{inferred_type}→ {title_c.title}[/dim]")
                    discovered += 1
                    existing.add((entity_a, entity_c))
                    existing.add((entity_c, entity_a))
                    continue

                new_rel = GraphRelation(
                    from_entity=entity_a, to_entity=entity_c,
                    type=inferred_type,
                    strength=inferred_strength,
                    context=context,
                )
                add_relation(graph, new_rel, strength_growth=0.0)
                existing.add((entity_a, entity_c))
                existing.add((entity_c, entity_a))
                discovered += 1
                console.print(f"    [green]{title_a.title} →{inferred_type}→ {title_c.title} (transitive)[/green]")

    report.transitive_relations = discovered
    if discovered and not dry_run:
        save_graph(memory_path, graph)
        console.print(f"  [green]Inferred {discovered} transitive relation(s)[/green]")
    elif not discovered:
        console.print("  No transitive relations to infer")
