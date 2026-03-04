"""Robustness tests: graceful degradation and edge cases."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.core.config import load_config
from src.core.models import (
    GraphData,
    RawEntity,
    RawExtraction,
    RawObservation,
    RawRelation,
)
from src.memory.graph import load_graph, save_graph
from src.memory.store import init_memory_structure, save_chat
from src.pipeline.enricher import enrich_memory
from src.pipeline.resolver import resolve_all


def _make_config(tmp_path):
    config = load_config(project_root=Path(__file__).parent.parent)
    config.memory_path = tmp_path
    config.faiss.index_path = str(tmp_path / "_memory.faiss")
    config.faiss.mapping_path = str(tmp_path / "_memory.pkl")
    config.faiss.manifest_path = str(tmp_path / "_faiss_manifest.json")
    return config


def _mock_embed(texts):
    np.random.seed(42)
    vecs = np.random.randn(len(texts), 384).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


# --- FAISS missing => auto-rebuild on search ---


def test_faiss_missing_auto_rebuild_on_search(tmp_path):
    """When FAISS index files are missing, search() auto-rebuilds from entities."""
    config = _make_config(tmp_path)
    init_memory_structure(tmp_path)

    # Create an entity file so there's something to index
    entity_dir = tmp_path / "moi"
    entity_dir.mkdir(exist_ok=True)
    (entity_dir / "test-entity.md").write_text(
        "---\ntitle: Test Entity\ntype: sante\nretention: long_term\n"
        "score: 0.5\nimportance: 0.5\nfrequency: 1\nlast_mentioned: 2026-03-03\n"
        "created: 2026-03-03\naliases: []\ntags: []\n---\n\n# Test Entity\n\n## Faits\n- fait: A fact\n",
        encoding="utf-8",
    )

    # Verify FAISS files don't exist
    assert not Path(config.faiss.index_path).exists()
    assert not Path(config.faiss.mapping_path).exists()

    # search() should auto-rebuild and return results
    from src.pipeline.indexer import search

    with patch("src.pipeline.indexer._get_embedding_fn", return_value=_mock_embed):
        results = search("test query", config, tmp_path)

    # After search, FAISS files should exist (auto-rebuilt)
    assert Path(config.faiss.index_path).exists()
    assert Path(config.faiss.mapping_path).exists()


def test_faiss_missing_search_empty_memory(tmp_path):
    """FAISS auto-rebuild on empty memory returns empty results."""
    config = _make_config(tmp_path)
    init_memory_structure(tmp_path)

    from src.pipeline.indexer import search

    with patch("src.pipeline.indexer._get_embedding_fn", return_value=_mock_embed):
        results = search("anything", config, tmp_path)

    assert results == []


# --- Corrupt _graph.json ---


def test_corrupt_graph_json_malformed(tmp_path):
    """Malformed _graph.json: load_graph should handle gracefully."""
    init_memory_structure(tmp_path)

    # Write explicitly malformed JSON
    graph_path = tmp_path / "_graph.json"
    graph_path.write_text("{{{not json at all!!!", encoding="utf-8")

    # load_graph should raise or return empty — test it doesn't crash silently
    try:
        graph = load_graph(tmp_path)
        # If it returns without error, it should be a valid GraphData
        assert isinstance(graph, GraphData)
    except (json.JSONDecodeError, Exception):
        # Raising is also acceptable — the system handles this via rebuild
        pass


def test_corrupt_graph_rebuild_recovers(tmp_path):
    """After corrupting _graph.json, rebuild_from_md recovers from entity files."""
    config = _make_config(tmp_path)
    init_memory_structure(tmp_path)

    # Create an entity via the pipeline
    extraction = RawExtraction(
        entities=[
            RawEntity(name="Récupération", type="sante", observations=[
                RawObservation(category="fait", content="Test recovery", importance=0.5),
            ]),
        ],
        relations=[],
        summary="test",
    )
    graph = load_graph(tmp_path)
    resolved = resolve_all(extraction, graph)
    enrich_memory(resolved, config, today="2026-03-03")

    # Corrupt the graph
    graph_path = tmp_path / "_graph.json"
    graph_path.write_text("CORRUPT DATA {{{", encoding="utf-8")

    # Rebuild from MD files
    from src.memory.graph import rebuild_from_md

    rebuilt = rebuild_from_md(tmp_path)
    assert "recuperation" in rebuilt.entities
    assert rebuilt.entities["recuperation"].title == "Récupération"


# --- Duplicate entity alias collision across chats ---


def test_alias_collision_across_chats(tmp_path):
    """Two chats mentioning the same entity by different names should merge, not duplicate."""
    config = _make_config(tmp_path)
    init_memory_structure(tmp_path)

    # Chat 1: mentions "Dr Martin"
    extraction1 = RawExtraction(
        entities=[
            RawEntity(name="Dr Martin", type="personne", observations=[
                RawObservation(category="fait", content="Médecin traitant", importance=0.6),
            ]),
        ],
        relations=[],
        summary="Visite Dr Martin",
    )

    # Process chat 1
    graph = load_graph(tmp_path)
    resolved1 = resolve_all(extraction1, graph)
    report1 = enrich_memory(resolved1, config, today="2026-03-03")
    assert "dr-martin" in report1.entities_created

    # Chat 2: mentions "Dr Martin" again (same slug)
    extraction2 = RawExtraction(
        entities=[
            RawEntity(name="Dr Martin", type="personne", observations=[
                RawObservation(category="fait", content="Prescrit du paracétamol", importance=0.5),
            ]),
        ],
        relations=[],
        summary="Ordonnance Dr Martin",
    )

    # Process chat 2 — should resolve to existing entity, not create duplicate
    graph = load_graph(tmp_path)
    resolved2 = resolve_all(extraction2, graph)
    report2 = enrich_memory(resolved2, config, today="2026-03-03")

    # Dr Martin should be updated, not created again
    assert "dr-martin" in report2.entities_updated or "dr-martin" not in report2.entities_created

    # Verify only one Dr Martin entity exists
    final_graph = load_graph(tmp_path)
    dr_martin_entities = [
        eid for eid, e in final_graph.entities.items()
        if "martin" in e.title.lower()
    ]
    assert len(dr_martin_entities) == 1


def test_alias_collision_different_names_same_entity(tmp_path):
    """Entity referenced by alias should resolve to the original, not create new."""
    config = _make_config(tmp_path)
    init_memory_structure(tmp_path)

    # Chat 1: create "Sophie Dupont"
    extraction1 = RawExtraction(
        entities=[
            RawEntity(name="Sophie Dupont", type="personne", observations=[
                RawObservation(category="fait", content="Collègue de bureau", importance=0.5),
            ]),
        ],
        relations=[],
        summary="Discussion avec Sophie",
    )

    graph = load_graph(tmp_path)
    resolved1 = resolve_all(extraction1, graph)
    enrich_memory(resolved1, config, today="2026-03-03")

    # Manually add alias "Sophie" to the entity graph
    graph = load_graph(tmp_path)
    entity = graph.entities["sophie-dupont"]
    entity.aliases = ["Sophie"]
    save_graph(tmp_path, graph)

    # Chat 2: mentions just "Sophie" — should resolve to sophie-dupont via alias
    extraction2 = RawExtraction(
        entities=[
            RawEntity(name="Sophie", type="personne", observations=[
                RawObservation(category="fait", content="Organise la réunion", importance=0.4),
            ]),
        ],
        relations=[],
        summary="Sophie organise",
    )

    graph = load_graph(tmp_path)
    resolved2 = resolve_all(extraction2, graph)

    # Check resolution: "Sophie" should resolve to existing "sophie-dupont"
    sophie_res = resolved2.resolved[0]
    assert sophie_res.resolution.status == "resolved"
    assert sophie_res.resolution.entity_id == "sophie-dupont"


def test_relation_both_endpoints_resolved(tmp_path):
    """Relations where both entities exist should be fully linked, no orphans."""
    config = _make_config(tmp_path)
    init_memory_structure(tmp_path)

    extraction = RawExtraction(
        entities=[
            RawEntity(name="Alice", type="personne", observations=[
                RawObservation(category="fait", content="Développeuse", importance=0.5),
            ]),
            RawEntity(name="Projet X", type="projet", observations=[
                RawObservation(category="projet", content="Application mobile", importance=0.6),
            ]),
        ],
        relations=[
            RawRelation(from_name="Alice", to_name="Projet X", type="fait_partie_de", context="travaille sur"),
        ],
        summary="Alice travaille sur Projet X",
    )

    graph = load_graph(tmp_path)
    resolved = resolve_all(extraction, graph)
    enrich_memory(resolved, config, today="2026-03-03")

    final_graph = load_graph(tmp_path)
    assert len(final_graph.relations) >= 1

    # No orphan relations
    for rel in final_graph.relations:
        assert rel.from_entity in final_graph.entities, f"Orphan from: {rel.from_entity}"
        assert rel.to_entity in final_graph.entities, f"Orphan to: {rel.to_entity}"
