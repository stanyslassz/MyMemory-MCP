"""Tests for fact supersession, context dedup, and observation consolidation."""

from pathlib import Path

from src.core.models import (
    RawObservation, RawEntity, RawExtraction, RawRelation,
    ResolvedEntity, ResolvedExtraction, Resolution,
    GraphData, GraphEntity, EntityFrontmatter,
    ConsolidatedFact, FactConsolidation,
)
from src.memory.store import (
    format_observation as _format_observation, parse_observation as _parse_observation,
    _is_duplicate_observation,
    mark_observation_superseded, write_entity, read_entity,
)
from src.memory.context import _deduplicate_facts_for_context, _content_similarity


# ── Phase 1: Supersession ──────────────────────────────────────


def test_raw_observation_supersedes_field():
    """RawObservation should accept the supersedes field."""
    obs = RawObservation(
        category="fact",
        content="Va skier à La Rosière",
        importance=0.6,
        supersedes="ski à Toulouse",
    )
    assert obs.supersedes == "ski à Toulouse"


def test_raw_observation_supersedes_default_empty():
    """supersedes defaults to empty string."""
    obs = RawObservation(category="fact", content="Test", importance=0.5)
    assert obs.supersedes == ""


def test_format_observation_superseded():
    """Superseded observations should be formatted with ~~ and [superseded]."""
    obs = {"category": "fact", "content": "Ski à Toulouse", "superseded": True}
    line = _format_observation(obs)
    assert "~~Ski à Toulouse~~" in line
    assert "[superseded]" in line
    assert line.startswith("- [fact]")


def test_format_observation_normal():
    """Normal observations should not have superseded markers."""
    obs = {"category": "fact", "content": "Goes skiing"}
    line = _format_observation(obs)
    assert "[superseded]" not in line
    assert "~~" not in line


def test_parse_observation_superseded():
    """Parser should detect [superseded] marker and strip ~~."""
    line = "- [fact] (2026-02) ~~Ski à Toulouse~~ [-] [superseded]"
    parsed = _parse_observation(line)
    assert parsed is not None
    assert parsed["superseded"] is True
    assert parsed["content"] == "Ski à Toulouse"
    assert parsed["category"] == "fact"
    assert parsed["valence"] == "negative"


def test_parse_observation_normal():
    """Normal observations should not have superseded key."""
    line = "- [fact] (2026-02) Goes skiing [+]"
    parsed = _parse_observation(line)
    assert parsed is not None
    assert "superseded" not in parsed


def test_is_duplicate_skips_superseded():
    """Duplicate check should skip superseded lines."""
    existing = [
        "- [fact] (2026-01) ~~Ski à Toulouse~~ [superseded]",
        "- [fact] (2026-02) Va skier à La Rosière [+]",
    ]
    # A new fact about Toulouse should not be considered duplicate (old one is superseded)
    new_line = "- [fact] Ski à Toulouse"
    assert not _is_duplicate_observation(new_line, existing)


def test_mark_observation_superseded():
    """mark_observation_superseded should mark matching facts."""
    facts = [
        "- [fact] (2026-01) Ski à Toulouse [+]",
        "- [fact] (2026-02) Aime le yoga",
        "- [preference] Préfère le thé",
    ]
    result = mark_observation_superseded(facts, "fact", "ski à toulouse")
    # First line should be superseded
    assert "[superseded]" in result[0]
    assert "~~" in result[0]
    # Others unchanged
    assert result[1] == facts[1]
    assert result[2] == facts[2]


def test_mark_observation_superseded_no_match():
    """mark_observation_superseded should not modify when no match."""
    facts = [
        "- [fact] (2026-01) Ski à Toulouse [+]",
        "- [fact] Aime le yoga",
    ]
    result = mark_observation_superseded(facts, "fact", "natation")
    assert result == facts


def test_mark_observation_superseded_category_must_match():
    """Supersession only applies within same category."""
    facts = [
        "- [preference] Ski à Toulouse",
    ]
    result = mark_observation_superseded(facts, "fact", "ski à toulouse")
    assert "[superseded]" not in result[0]


def test_enricher_supersession(tmp_path):
    """Enricher should mark old facts as superseded when new observation has supersedes."""
    from src.pipeline.enricher import _update_existing_entity
    from src.core.models import EnrichmentReport, GraphEntity

    # Create an entity with a fact
    entity_dir = tmp_path / "self"
    entity_dir.mkdir()
    filepath = entity_dir / "vacances.md"
    fm = EntityFrontmatter(
        title="Vacances", type="interest", created="2026-01-01",
        last_mentioned="2026-01-01", frequency=1,
    )
    write_entity(filepath, fm, {
        "Facts": ["- [fact] (2026-01) Ski à Toulouse [+]"],
        "Relations": [],
        "History": [],
    })

    # Build graph with entity
    graph = GraphData(generated="2026-03-07")
    graph.entities["vacances"] = GraphEntity(
        file="self/vacances.md", type="interest", title="Vacances",
        score=0.5, importance=0.5, frequency=1, last_mentioned="2026-01-01",
        mention_dates=["2026-01-01"],
    )

    # Create raw entity with superseding observation
    raw = RawEntity(
        name="Vacances", type="interest",
        observations=[
            RawObservation(
                category="fact",
                content="Va skier à La Rosière",
                importance=0.6,
                date="2026-02",
                valence="positive",
                supersedes="ski à Toulouse",
            ),
        ],
    )

    report = EnrichmentReport()
    _update_existing_entity("vacances", raw, graph, tmp_path, "2026-03-07", report)

    # Read back and verify
    _, sections = read_entity(filepath)
    facts = sections["Facts"]

    # Old fact should be superseded
    superseded = [f for f in facts if "[superseded]" in f]
    assert len(superseded) == 1
    assert "Toulouse" in superseded[0]

    # New fact should be present
    new_facts = [f for f in facts if "La Rosière" in f]
    assert len(new_facts) == 1
    assert "[superseded]" not in new_facts[0]


# ── Phase 2: Context dedup ─────────────────────────────────────


def test_deduplicate_facts_identical():
    """Near-identical facts in same category should be deduped."""
    facts = [
        "- [ai_style] User prefers direct concise answers",
        "- [ai_style] User prefers direct and concise answers",
        "- [fact] Something different",
    ]
    result = _deduplicate_facts_for_context(facts)
    ai_style = [f for f in result if "[ai_style]" in f]
    assert len(ai_style) == 1
    assert len(result) == 2


def test_deduplicate_french_semantic_duplicates():
    """French semantic duplicates with synonym/morphological variation should be caught."""
    facts = [
        "- [ai_style] Fournir des réponses structurées avec des exemples concrets",
        "- [ai_style] Donner des réponses structurés avec des exemples concrets",
    ]
    result = _deduplicate_facts_for_context(facts)
    assert len(result) == 1


def test_deduplicate_french_morphological_variants():
    """Morphological variants (structuré/structurées) should be caught via trigrams."""
    sim = _content_similarity(
        "réponses structurées avec exemples concrets",
        "réponses structurés avec exemples concrets",
    )
    assert sim > 0.35


def test_deduplicate_short_facts_not_falsely_matched():
    """Short distinct facts should NOT be deduped."""
    facts = [
        "- [fact] Alexis a 40 ans",
        "- [fact] Alexis travaille chez Airbus",
    ]
    result = _deduplicate_facts_for_context(facts)
    assert len(result) == 2


def test_deduplicate_cross_category_preserved():
    """Similar text in different categories should both be kept."""
    facts = [
        "- [fact] Préfère les réponses concises et directes",
        "- [ai_style] Préfère les réponses concises et directes",
    ]
    result = _deduplicate_facts_for_context(facts)
    assert len(result) == 2


def test_deduplicate_category_cap():
    """Per-category cap should limit to max_per_category (default 5)."""
    facts = [
        "- [ai_style] Préfère les réponses concises",
        "- [ai_style] Utilise le markdown pour structurer",
        "- [ai_style] Toujours inclure des exemples pratiques",
        "- [ai_style] Éviter le jargon technique inutile",
        "- [ai_style] Répondre dans la langue de la question",
        "- [ai_style] Proposer des alternatives quand possible",
        "- [ai_style] Commencer par un résumé avant les détails",
    ]
    result = _deduplicate_facts_for_context(facts)
    ai_style = [f for f in result if "[ai_style]" in f]
    assert len(ai_style) == 5


def test_deduplicate_category_cap_custom():
    """Custom max_per_category should be respected."""
    facts = [
        "- [fact] Travaille chez Airbus depuis cinq ans",
        "- [fact] Habite à Toulouse centre-ville",
        "- [fact] Pratique la natation chaque semaine",
        "- [fact] Joue de la guitare acoustique",
        "- [fact] Collectionne les vinyles de jazz",
    ]
    result = _deduplicate_facts_for_context(facts, max_per_category=3)
    assert len(result) == 3


def test_deduplicate_facts_different_categories():
    """Facts in different categories should not be deduped even if similar."""
    facts = [
        "- [fact] User prefers direct answers",
        "- [ai_style] User prefers direct answers",
    ]
    result = _deduplicate_facts_for_context(facts)
    assert len(result) == 2


def test_deduplicate_facts_distinct():
    """Clearly distinct facts should all be kept."""
    facts = [
        "- [fact] Works at Airbus",
        "- [fact] Has two cats",
        "- [fact] Lives in Toulouse",
    ]
    result = _deduplicate_facts_for_context(facts)
    assert len(result) == 3


def test_deduplicate_preserves_order():
    """Dedup should preserve the order of kept facts."""
    facts = [
        "- [ai_style] Be concise and direct",
        "- [fact] (2024-01) First event",
        "- [ai_style] Be very concise and very direct",  # dup of first
        "- [fact] (2025-06) Second event",
    ]
    result = _deduplicate_facts_for_context(facts)
    assert len(result) == 3
    assert result[0] == facts[0]
    assert result[1] == facts[1]
    assert result[2] == facts[3]


def test_context_filters_superseded(tmp_path):
    """Context builder should exclude superseded facts."""
    from src.memory.context import _enrich_entity

    entity_dir = tmp_path / "self"
    entity_dir.mkdir()
    filepath = entity_dir / "test-ent.md"

    fm = EntityFrontmatter(title="Test", type="interest")
    write_entity(filepath, fm, {
        "Facts": [
            "- [fact] ~~Old fact~~ [superseded]",
            "- [fact] New correct fact",
        ],
        "Relations": [],
        "History": [],
    })

    graph = GraphData(generated="2026-03-07")
    entity = GraphEntity(
        file="self/test-ent.md", type="interest", title="Test",
        score=0.5, importance=0.5,
    )
    graph.entities["test-ent"] = entity

    dossier = _enrich_entity("test-ent", entity, graph, tmp_path)
    assert "New correct fact" in dossier
    assert "Old fact" not in dossier
    assert "[superseded]" not in dossier


# ── Phase 3: Models ────────────────────────────────────────────


def test_consolidated_fact_model():
    """ConsolidatedFact model should validate correctly."""
    cf = ConsolidatedFact(
        category="ai_style",
        content="User prefers direct, concise answers",
        date="2026-03",
        tags=["communication"],
        replaces_indices=[0, 2, 5],
    )
    assert cf.category == "ai_style"
    assert cf.replaces_indices == [0, 2, 5]


def test_fact_consolidation_model():
    """FactConsolidation model should validate correctly."""
    fc = FactConsolidation(consolidated=[
        ConsolidatedFact(
            category="ai_style",
            content="Merged text",
            replaces_indices=[0, 1],
        ),
    ])
    assert len(fc.consolidated) == 1


def test_roundtrip_superseded_observation():
    """Format then parse a superseded observation should roundtrip."""
    obs = {
        "category": "fact",
        "content": "Ski à Toulouse",
        "date": "2026-01",
        "valence": "positive",
        "superseded": True,
    }
    line = _format_observation(obs)
    parsed = _parse_observation(line)
    assert parsed is not None
    assert parsed["superseded"] is True
    assert parsed["content"] == "Ski à Toulouse"
    assert parsed["category"] == "fact"
    assert parsed["date"] == "2026-01"
    assert parsed["valence"] == "positive"
