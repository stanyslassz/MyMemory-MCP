"""Tests for dream pipeline persistence (save_graph calls)."""

from datetime import datetime
from unittest.mock import MagicMock, patch

from src.core.models import GraphData, GraphEntity, GraphRelation
from src.pipeline.dream import DreamReport, _step_merge_entities, _step_discover_relations, _step_generate_summaries


def test_merge_saves_graph_per_iteration(tmp_path):
    """save_graph should be called after each merge, not just at the end."""
    entity_a = GraphEntity(
        file="close_ones/alice.md", type="person", title="Alice",
        score=0.8, importance=0.7, frequency=5,
        last_mentioned="2026-03-15", retention="long_term",
        aliases=["alice dupont"], tags=["family"],
        mention_dates=["2026-03-15"], monthly_buckets={},
        created="2025-01-01", summary="Alice.",
        negative_valence_ratio=0.0,
    )
    entity_b = GraphEntity(
        file="close_ones/alice-dupont.md", type="person", title="Alice Dupont",
        score=0.3, importance=0.4, frequency=2,
        last_mentioned="2026-01-01", retention="short_term",
        aliases=["alice"], tags=[],
        mention_dates=["2026-01-01"], monthly_buckets={},
        created="2025-06-01", summary="",
        negative_valence_ratio=0.0,
    )
    graph = GraphData(
        generated=datetime.now().isoformat(),
        entities={"alice": entity_a, "alice-dupont": entity_b},
        relations=[],
    )

    memory_path = tmp_path / "memory"
    memory_path.mkdir()
    for eid, entity in graph.entities.items():
        p = memory_path / entity.file
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            f"---\ntitle: {entity.title}\ntype: {entity.type}\nretention: long_term\n"
            f"score: 0.5\nimportance: 0.5\nfrequency: 1\nlast_mentioned: '2026-03-15'\n"
            f"created: '2025-01-01'\naliases: [{', '.join(entity.aliases)}]\ntags: []\n"
            f"mention_dates: ['2026-03-15']\nmonthly_buckets: {{}}\nsummary: ''\n---\n"
            f"## Facts\n- [fact] test\n\n## Relations\n\n## History\n"
        )

    config = MagicMock()
    config.memory_path = memory_path
    config.dream = MagicMock()
    config.dream.faiss_merge_threshold = 0.80
    config.dream.faiss_merge_max_candidates = 20
    console = MagicMock()
    report = DreamReport()

    with patch("src.memory.graph.save_graph") as mock_save, \
         patch("src.pipeline.dream._find_faiss_dedup_candidates", return_value=[]):
        _step_merge_entities(graph, memory_path, config, console, report, dry_run=False, entity_paths={
            "alice": memory_path / "close_ones/alice.md",
            "alice-dupont": memory_path / "close_ones/alice-dupont.md",
        })

    assert mock_save.call_count >= 1
    assert report.entities_merged >= 1


def test_discover_relations_saves_graph_per_relation(tmp_path):
    """save_graph should be called after each new relation, not just at the end."""
    entity_a = GraphEntity(
        file="interests/cooking.md", type="interest", title="Cooking",
        score=0.7, importance=0.6, frequency=3,
        last_mentioned="2026-03-15", retention="long_term",
        aliases=[], tags=["hobby"], mention_dates=["2026-03-15"],
        monthly_buckets={}, created="2025-01-01", summary="Cooking hobby.",
        negative_valence_ratio=0.0,
    )
    entity_b = GraphEntity(
        file="interests/nutrition.md", type="interest", title="Nutrition",
        score=0.6, importance=0.5, frequency=2,
        last_mentioned="2026-03-10", retention="long_term",
        aliases=[], tags=["health"], mention_dates=["2026-03-10"],
        monthly_buckets={}, created="2025-01-01", summary="Nutrition interest.",
        negative_valence_ratio=0.0,
    )
    graph = GraphData(
        generated=datetime.now().isoformat(),
        entities={"cooking": entity_a, "nutrition": entity_b},
        relations=[],
    )

    memory_path = tmp_path / "memory"
    memory_path.mkdir()
    for eid, entity in graph.entities.items():
        p = memory_path / entity.file
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"---\ntitle: {entity.title}\ntype: {entity.type}\nretention: long_term\n---\n## Facts\n- [fact] test\n")

    config = MagicMock()
    config.memory_path = memory_path
    config.scoring = MagicMock()
    config.scoring.relation_strength_growth = 0.05
    console = MagicMock()
    report = DreamReport()

    mock_result = MagicMock()
    mock_result.entity_id = "nutrition"
    mock_result.score = 0.9

    mock_proposal = MagicMock()
    mock_proposal.action = "relate"
    mock_proposal.relation_type = "linked_to"
    mock_proposal.context = "test context"

    with patch("src.memory.rag.search", return_value=[mock_result]), \
         patch("src.core.llm.call_relation_discovery", return_value=mock_proposal), \
         patch("src.memory.graph.save_graph") as mock_save:
        _step_discover_relations(graph, memory_path, config, console, report, dry_run=False)

    assert mock_save.call_count >= 1
    assert report.relations_discovered >= 1


def test_summaries_saves_graph_per_entity(tmp_path):
    """save_graph should be called after each summary is generated."""
    entity = GraphEntity(
        file="interests/cooking.md", type="interest", title="Cooking",
        score=0.7, importance=0.6, frequency=3,
        last_mentioned="2026-03-15", retention="long_term",
        aliases=[], tags=[], mention_dates=["2026-03-15"],
        monthly_buckets={}, created="2025-01-01", summary="",
        negative_valence_ratio=0.0,
    )
    graph = GraphData(
        generated=datetime.now().isoformat(),
        entities={"cooking": entity},
        relations=[],
    )

    memory_path = tmp_path / "memory"
    memory_path.mkdir()
    p = memory_path / "interests/cooking.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("---\ntitle: Cooking\ntype: interest\nretention: long_term\n---\n## Facts\n- [fact] likes cooking\n")

    config = MagicMock()
    config.memory_path = memory_path
    console = MagicMock()
    report = DreamReport()

    with patch("src.core.llm.call_entity_summary", return_value="A cooking enthusiast."), \
         patch("src.memory.graph.save_graph") as mock_save:
        _step_generate_summaries(graph, {"cooking": p}, config, console, report, dry_run=False)

    assert mock_save.call_count >= 1
    assert report.summaries_generated >= 1
    assert graph.entities["cooking"].summary == "A cooking enthusiast."
