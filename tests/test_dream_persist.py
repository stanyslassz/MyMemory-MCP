"""Tests for dream per-iteration persistence."""

from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.core.models import GraphData, GraphEntity
from src.pipeline.dream import _step_merge_entities, DreamReport


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
        p.write_text(f"---\ntitle: {entity.title}\ntype: {entity.type}\nretention: long_term\nscore: 0.5\nimportance: 0.5\nfrequency: 1\nlast_mentioned: '2026-03-15'\ncreated: '2025-01-01'\naliases: [{', '.join(entity.aliases)}]\ntags: []\nmention_dates: ['2026-03-15']\nmonthly_buckets: {{}}\nsummary: ''\n---\n## Facts\n- [fact] test\n\n## Relations\n\n## History\n")

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
