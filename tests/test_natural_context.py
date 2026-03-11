"""Tests for natural Claude Chat-like context generation."""

from __future__ import annotations

import tempfile
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

from src.core.config import Config, ScoringConfig
from src.core.models import GraphData, GraphEntity, GraphRelation, EntityFrontmatter
from src.memory.context import (
    _select_entities_for_natural,
    _classify_temporal,
    _build_natural_bullet,
    _enrich_entity_natural,
    _build_section_llm,
    _extract_vigilances,
    _read_entity_facts,
    build_natural_context,
)
from src.memory.graph import add_relation
from src.memory.store import write_entity


def _today():
    return date(2026, 3, 11)


def _make_entity(**kwargs) -> GraphEntity:
    defaults = dict(
        file="self/test.md",
        type="person",
        title="Test",
        score=0.5,
        importance=0.5,
        frequency=3,
        last_mentioned=_today().isoformat(),
        retention="long_term",
        created="2025-01-01",
    )
    defaults.update(kwargs)
    return GraphEntity(**defaults)


def _make_graph() -> GraphData:
    from datetime import datetime
    return GraphData(generated=datetime.now().isoformat())


def _write_entity_file(memory_path: Path, entity: GraphEntity, facts: list[str] | None = None):
    """Write an entity MD file."""
    filepath = memory_path / entity.file
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fm = EntityFrontmatter(
        title=entity.title,
        type=entity.type,
        retention=entity.retention,
        score=entity.score,
        importance=entity.importance,
        frequency=entity.frequency,
        last_mentioned=entity.last_mentioned,
        created=entity.created or "2025-01-01",
    )
    sections = {
        "Facts": facts or [],
        "Relations": [],
        "History": [],
    }
    write_entity(filepath, fm, sections)


# --- Test 1: Entity selection ---

def test_select_entities_filters_transient():
    """person/health always included, project with freq=1 excluded."""
    graph = _make_graph()

    person = _make_entity(type="person", title="Alice", frequency=1)
    health = _make_entity(type="health", title="Sciatique", frequency=1)
    project_low = _make_entity(type="project", title="SideProject", frequency=1, score=0.3)
    project_high = _make_entity(type="project", title="MainProject", frequency=5, score=0.6)

    all_top = [
        ("alice", person),
        ("sciatique", health),
        ("sideproject", project_low),
        ("mainproject", project_high),
    ]

    selected = _select_entities_for_natural(all_top, graph)
    selected_ids = {eid for eid, _ in selected}

    assert "alice" in selected_ids
    assert "sciatique" in selected_ids
    assert "sideproject" not in selected_ids  # freq=1, score < 0.5
    assert "mainproject" in selected_ids


# --- Test 2: Temporal classification (stable person) ---

def test_classify_temporal_stable_person():
    """Stable person (long_term, freq>=5) stays long_terme even if mentioned yesterday."""
    entity = _make_entity(
        type="person",
        retention="long_term",
        frequency=10,
        last_mentioned=(_today() - timedelta(days=1)).isoformat(),
    )
    assert _classify_temporal(entity, _today()) == "long_terme"


# --- Test 3: Temporal classification (recent project) ---

def test_classify_temporal_recent_project():
    """Project mentioned 3 days ago → court_terme."""
    entity = _make_entity(
        type="project",
        retention="short_term",
        frequency=2,
        last_mentioned=(_today() - timedelta(days=3)).isoformat(),
    )
    assert _classify_temporal(entity, _today()) == "court_terme"


# --- Test 4: Bullet with summary ---

def test_build_natural_bullet_with_summary():
    """Uses entity.summary when available."""
    graph = _make_graph()
    entity = _make_entity(title="Sciatique", summary="Hernie discale L5-S1 en amélioration")
    graph.entities["sciatique"] = entity

    with tempfile.TemporaryDirectory() as tmpdir:
        memory_path = Path(tmpdir)
        _write_entity_file(memory_path, entity)
        bullet = _build_natural_bullet("sciatique", entity, graph, memory_path)
        assert "Hernie discale L5-S1 en amélioration" in bullet
        assert bullet.startswith("- ")


# --- Test 5: Bullet without summary (fallback to fact) ---

def test_build_natural_bullet_without_summary():
    """Falls back to most recent fact when no summary."""
    graph = _make_graph()
    entity = _make_entity(title="Sciatique", summary="", file="self/sciatique.md")
    graph.entities["sciatique"] = entity

    with tempfile.TemporaryDirectory() as tmpdir:
        memory_path = Path(tmpdir)
        facts = [
            "- [diagnosis] (2026-03) Hernie discale L5-S1 [-]",
            "- [treatment] (2026-03) Lyrica et Seresta [+]",
        ]
        _write_entity_file(memory_path, entity, facts=facts)
        bullet = _build_natural_bullet("sciatique", entity, graph, memory_path)
        # Should use the last fact's content
        assert "Lyrica et Seresta" in bullet


# --- Test 6: Bullet with relations ---

def test_build_natural_bullet_with_relations():
    """Relations integrated as natural text in bullet."""
    graph = _make_graph()
    entity_a = _make_entity(title="Sciatique", file="self/sciatique.md", summary="Hernie discale")
    entity_b = _make_entity(title="Natation", file="interests/natation.md")
    graph.entities["sciatique"] = entity_a
    graph.entities["natation"] = entity_b

    rel = GraphRelation(
        from_entity="natation", to_entity="sciatique",
        type="improves", strength=0.6,
    )
    add_relation(graph, rel)

    with tempfile.TemporaryDirectory() as tmpdir:
        memory_path = Path(tmpdir)
        _write_entity_file(memory_path, entity_a)
        _write_entity_file(memory_path, entity_b)
        bullet = _build_natural_bullet("sciatique", entity_a, graph, memory_path)
        assert "amélioré par" in bullet or "Natation" in bullet


# --- Test 7: Extract vigilances ---

def test_extract_vigilances():
    """Extracts vigilance facts, deduplicates, caps at 15."""
    graph = _make_graph()
    entity = _make_entity(title="Lysanxia", file="self/lysanxia.md")
    graph.entities["lysanxia"] = entity

    with tempfile.TemporaryDirectory() as tmpdir:
        memory_path = Path(tmpdir)
        facts = [
            "- [vigilance] Interaction dangereuse avec alcool [-]",
            "- [vigilance] Interaction dangereuse avec alcool [-]",  # duplicate
            "- [treatment] (2026-03) Posologie réduite [+]",
            "- [fact] Note sans importance",
        ]
        _write_entity_file(memory_path, entity, facts=facts)

        vigilances = _extract_vigilances([("lysanxia", entity)], graph, memory_path)
        # Only 1 unique vigilance (treatment with + valence not included, fact excluded)
        assert len(vigilances) == 1
        assert "Interaction dangereuse" in vigilances[0]


# --- Test 8: Full end-to-end ---

def test_build_natural_context_full():
    """End-to-end: build_natural_context produces expected structure."""
    graph = _make_graph()

    # Long-term person
    person = _make_entity(
        title="Anaïs", type="person", file="close_ones/anais.md",
        retention="long_term", frequency=10, score=0.8,
        last_mentioned=(_today() - timedelta(days=2)).isoformat(),
        summary="Épouse, 38 ans",
    )
    graph.entities["anais"] = person

    # Recent health
    health = _make_entity(
        title="Sciatique", type="health", file="self/sciatique.md",
        retention="short_term", frequency=3, score=0.6,
        last_mentioned=(_today() - timedelta(days=1)).isoformat(),
        summary="Hernie discale L5-S1, jour 29",
    )
    graph.entities["sciatique"] = health

    with tempfile.TemporaryDirectory() as tmpdir:
        memory_path = Path(tmpdir)
        prompts_path = Path(tmpdir) / "prompts"
        prompts_path.mkdir()
        # Write template
        (prompts_path / "context_natural.md").write_text(
            "# Mémoire — {date}\n\n**Langue : {user_language_name}**\n\n"
            "{ai_personality}\n\n{sections}\n\n{available_entities}\n\n"
            "{extended_memory}\n\n{custom_instructions}",
            encoding="utf-8",
        )

        _write_entity_file(memory_path, person)
        _write_entity_file(memory_path, health, facts=[
            "- [vigilance] Éviter les mouvements brusques [-]",
        ])

        config = Config(
            memory_path=memory_path,
            prompts_path=prompts_path,
            context_max_tokens=5000,
            context_format="natural",
            scoring=ScoringConfig(min_score_for_context=0.1),
        )

        result = build_natural_context(graph, memory_path, config)

        # Structure checks
        assert "Mémoire" in result
        assert "French" in result  # user_language_name
        assert "Identité & long terme" in result or "Cette semaine" in result or "En ce moment" in result
        assert "Épouse" in result  # Anaïs via her summary
        assert "Hernie" in result or "Sciatique" in result
        assert "search_rag" in result


# --- Test 9: Token budget respected ---

def test_token_budget_respected():
    """Output should not exceed context_max_tokens."""
    graph = _make_graph()

    # Add many entities to potentially exceed budget
    for i in range(30):
        eid = f"entity_{i}"
        entity = _make_entity(
            title=f"Entity {i}",
            file=f"self/{eid}.md",
            type="health",
            frequency=5,
            score=0.7,
            summary=f"A description for entity {i} with some content to take up tokens",
            last_mentioned=(_today() - timedelta(days=i % 40)).isoformat(),
        )
        graph.entities[eid] = entity

    with tempfile.TemporaryDirectory() as tmpdir:
        memory_path = Path(tmpdir)
        prompts_path = Path(tmpdir) / "prompts"
        prompts_path.mkdir()
        (prompts_path / "context_natural.md").write_text(
            "{date}\n{user_language_name}\n{ai_personality}\n{sections}\n"
            "{available_entities}\n{extended_memory}\n{custom_instructions}",
            encoding="utf-8",
        )

        for eid, entity in graph.entities.items():
            _write_entity_file(memory_path, entity)

        config = Config(
            memory_path=memory_path,
            prompts_path=prompts_path,
            context_max_tokens=800,  # Very tight budget
            context_format="natural",
            scoring=ScoringConfig(min_score_for_context=0.1),
        )

        result = build_natural_context(graph, memory_path, config)
        # Rough token estimate: words * 1.3
        estimated_tokens = int(len(result.split()) * 1.3)
        # Should be within reasonable bounds of the budget
        assert estimated_tokens < 800 * 2, f"Output too large: ~{estimated_tokens} tokens"


# --- Test 10: Routing config ---

def test_routing_config():
    """context_format=natural should route to build_natural_context."""
    config = Config(context_format="natural")
    assert config.context_format == "natural"

    config_default = Config()
    assert config_default.context_format == "structured"


# --- Test 11: _enrich_entity_natural caps relations at 3 ---

def test_enrich_entity_natural_caps_relations():
    """_enrich_entity_natural should include max 3 relations."""
    graph = _make_graph()
    entity = _make_entity(title="Alexis", file="close_ones/alexis.md", summary="Fils aîné")
    graph.entities["alexis"] = entity

    # Add 6 related entities + relations
    for i in range(6):
        eid = f"rel_{i}"
        rel_entity = _make_entity(title=f"Related {i}", file=f"interests/{eid}.md")
        graph.entities[eid] = rel_entity
        rel = GraphRelation(
            from_entity="alexis", to_entity=eid,
            type="linked_to", strength=0.5 + i * 0.05,
        )
        add_relation(graph, rel)

    with tempfile.TemporaryDirectory() as tmpdir:
        memory_path = Path(tmpdir)
        _write_entity_file(memory_path, entity)
        for eid, e in graph.entities.items():
            if eid != "alexis":
                _write_entity_file(memory_path, e)

        dossier = _enrich_entity_natural("alexis", entity, graph, memory_path)
        # Count relation lines (→ or ←)
        rel_lines = [l for l in dossier.split("\n") if "→" in l or "←" in l]
        assert len(rel_lines) <= 3, f"Expected max 3 relations, got {len(rel_lines)}"


# --- Test 12: _enrich_entity_natural has no score header ---

def test_enrich_entity_natural_no_score_header():
    """_enrich_entity_natural should not include score or retention metadata."""
    graph = _make_graph()
    entity = _make_entity(title="Test", file="self/test.md", score=0.72, summary="A test entity")
    graph.entities["test"] = entity

    with tempfile.TemporaryDirectory() as tmpdir:
        memory_path = Path(tmpdir)
        _write_entity_file(memory_path, entity)
        dossier = _enrich_entity_natural("test", entity, graph, memory_path)
        assert "(score:" not in dossier
        assert "retention:" not in dossier.lower()


# --- Test 13: _build_section_llm fallback on LLM failure ---

def test_build_section_llm_fallback():
    """If LLM call raises, _build_section_llm falls back to deterministic bullets."""
    graph = _make_graph()
    entity = _make_entity(title="Sciatique", file="self/sciatique.md", summary="Hernie discale")
    graph.entities["sciatique"] = entity

    with tempfile.TemporaryDirectory() as tmpdir:
        memory_path = Path(tmpdir)
        _write_entity_file(memory_path, entity)

        config = Config(
            memory_path=memory_path,
            prompts_path=Path(tmpdir) / "prompts",
            context_max_tokens=3000,
            scoring=ScoringConfig(min_score_for_context=0.1),
        )

        # Mock LLM to raise
        with patch("src.core.llm.call_natural_context_section", side_effect=RuntimeError("LLM down")):
            result = _build_section_llm(
                "Test Section",
                [("sciatique", entity)],
                graph, memory_path, config, 500,
            )
        assert "Hernie discale" in result
        assert result.startswith("- ")


# --- Test 14: run_pipeline with no chats still rebuilds context ---

def test_run_pipeline_no_chats_still_rebuilds_context():
    """run_pipeline should rebuild context even when no chats are pending."""
    from unittest.mock import MagicMock, call

    with tempfile.TemporaryDirectory() as tmpdir:
        memory_path = Path(tmpdir)
        prompts_path = Path(tmpdir) / "prompts"
        prompts_path.mkdir()
        (prompts_path / "context_natural.md").write_text(
            "{date}\n{user_language_name}\n{ai_personality}\n{sections}\n"
            "{available_entities}\n{extended_memory}\n{custom_instructions}",
            encoding="utf-8",
        )

        config = Config(
            memory_path=memory_path,
            prompts_path=prompts_path,
            context_max_tokens=3000,
            context_format="natural",
            scoring=ScoringConfig(min_score_for_context=0.1),
        )

        # Create empty chats dir so list_unprocessed_chats returns []
        (memory_path / "chats").mkdir(parents=True, exist_ok=True)

        # Create a minimal graph
        from src.memory.graph import save_graph
        graph = _make_graph()
        entity = _make_entity(title="Test", file="self/test.md", score=0.6, type="health")
        graph.entities["test"] = entity
        save_graph(memory_path, graph)
        _write_entity_file(memory_path, entity)

        console = MagicMock()

        # Patch out stale job recovery and FAISS
        with patch("src.pipeline.ingest_state.recover_stale_jobs", return_value=[]), \
             patch("src.pipeline.indexer.incremental_update"):
            from src.pipeline.orchestrator import run_pipeline
            run_pipeline(config, console, consolidate=False)

        # Check that context was generated (write_context was reached)
        output_calls = [str(c) for c in console.print.call_args_list]
        joined = " ".join(output_calls)
        assert "No pending chats" in joined, "Should report no pending chats"
        # Should still reach context generation step
        assert "context" in joined.lower(), f"Should generate context, got: {joined}"
