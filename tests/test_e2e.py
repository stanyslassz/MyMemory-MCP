"""End-to-end tests: simulate full pipeline from empty memory to populated state."""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.core.config import load_config
from src.core.models import (
    RawEntity,
    RawExtraction,
    RawObservation,
    RawRelation,
    Resolution,
    ResolvedEntity,
    ResolvedExtraction,
)
from src.memory.graph import load_graph, rebuild_from_md
from src.memory.store import (
    get_chat_content,
    init_memory_structure,
    list_unprocessed_chats,
    mark_chat_processed,
    read_entity,
    save_chat,
)
from src.pipeline.enricher import enrich_memory
from src.pipeline.resolver import resolve_all, slugify


def _mock_extract_fixture():
    """Return a mock extraction matching sample_chat.md content."""
    return RawExtraction(
        entities=[
            RawEntity(
                name="Mal de dos",
                type="sante",
                observations=[
                    RawObservation(category="diagnostic", content="Sciatique qui revient", importance=0.8),
                    RawObservation(category="vigilance", content="JAMAIS ibuprofène — allergie sévère", importance=0.95, tags=["permanent"]),
                    RawObservation(category="traitement", content="Paracétamol prescrit par Dr Martin", importance=0.6),
                ],
            ),
            RawEntity(
                name="Dr Martin",
                type="personne",
                observations=[
                    RawObservation(category="fait", content="Médecin traitant", importance=0.5),
                ],
            ),
            RawEntity(
                name="TechCorp",
                type="organisation",
                observations=[
                    RawObservation(category="fait", content="Employeur actuel", importance=0.6),
                ],
            ),
            RawEntity(
                name="Projet Phoenix",
                type="projet",
                observations=[
                    RawObservation(category="projet", content="Migration cloud, lancement la semaine prochaine", importance=0.7),
                ],
            ),
            RawEntity(
                name="Louise",
                type="personne",
                observations=[
                    RawObservation(category="fait", content="Collègue, co-gère le projet Phoenix", importance=0.5),
                ],
            ),
            RawEntity(
                name="Sophie",
                type="personne",
                observations=[
                    RawObservation(category="relation_interpersonnelle", content="Femme de l'utilisateur", importance=0.7),
                ],
            ),
            RawEntity(
                name="Natation",
                type="interet",
                observations=[
                    RawObservation(category="fait", content="Pratique pour soulager le dos", importance=0.5),
                ],
            ),
        ],
        relations=[
            RawRelation(from_name="Natation", to_name="Mal de dos", type="ameliore", context="soulage la douleur"),
            RawRelation(from_name="Dr Martin", to_name="Mal de dos", type="ameliore", context="prescrit traitement"),
            RawRelation(from_name="Louise", to_name="Projet Phoenix", type="fait_partie_de", context="co-gère"),
            RawRelation(from_name="Sophie", to_name="Natation", type="lie_a", context="accompagne"),
        ],
        summary="L'utilisateur souffre de sciatique, suivi par Dr Martin. Travaille chez TechCorp, lance le projet Phoenix avec Louise. Fait de la natation avec Sophie.",
    )


def _make_config(tmp_path):
    """Create config pointing to tmp_path as memory root."""
    config = load_config(project_root=Path(__file__).parent.parent)
    config.memory_path = tmp_path
    config.faiss.index_path = str(tmp_path / "_memory.faiss")
    config.faiss.mapping_path = str(tmp_path / "_memory.pkl")
    config.faiss.manifest_path = str(tmp_path / "_faiss_manifest.json")
    return config


def _mock_embed(texts):
    """Mock embedding function."""
    import numpy as np
    np.random.seed(42)
    vecs = np.random.randn(len(texts), 384).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def test_e2e_full_pipeline(tmp_path):
    """Full E2E: empty memory → save chats → run pipeline → verify state."""
    config = _make_config(tmp_path)
    init_memory_structure(tmp_path)

    # 1. Save fixture chats
    fixture = Path(__file__).parent / "fixtures" / "sample_chat.md"
    chat_content = fixture.read_text(encoding="utf-8")

    messages1 = [
        {"role": "user", "content": "J'ai mal au dos, ma sciatique revient. Le Dr Martin m'a prescrit du paracétamol. JAMAIS d'ibuprofène, allergie sévère."},
        {"role": "assistant", "content": "Je comprends. Suivez les recommandations de votre médecin."},
    ]
    messages2 = [
        {"role": "user", "content": "Chez TechCorp on lance le projet Phoenix. Louise co-gère avec moi."},
        {"role": "assistant", "content": "Un beau projet !"},
    ]
    messages3 = [
        {"role": "user", "content": "Ce week-end natation avec Sophie, ma femme. Ça soulage mon dos."},
        {"role": "assistant", "content": "Bonne idée !"},
    ]

    chat1 = save_chat(messages1, tmp_path)
    chat2 = save_chat(messages2, tmp_path)
    chat3 = save_chat(messages3, tmp_path)

    # 2. Verify chats are unprocessed
    unprocessed = list_unprocessed_chats(tmp_path)
    assert len(unprocessed) == 3

    # 3. Run pipeline (with mocked extraction)
    mock_extraction = _mock_extract_fixture()

    for chat_path in unprocessed:
        content = get_chat_content(chat_path)

        # Step 2: Resolve (no FAISS yet)
        graph = load_graph(tmp_path)
        resolved = resolve_all(mock_extraction, graph)

        # Step 4: Enrich
        report = enrich_memory(resolved, config, today="2026-03-03")

        # Mark processed
        mark_chat_processed(chat_path, report.entities_updated, report.entities_created)

    # 4. Verify entities were created
    graph = load_graph(tmp_path)
    assert len(graph.entities) > 0

    # Check key entities exist
    entity_titles = {e.title for e in graph.entities.values()}
    assert "Mal de dos" in entity_titles
    assert "Dr Martin" in entity_titles
    assert "TechCorp" in entity_titles
    assert "Projet Phoenix" in entity_titles
    assert "Louise" in entity_titles
    assert "Sophie" in entity_titles
    assert "Natation" in entity_titles

    # 5. Verify relations
    assert len(graph.relations) > 0

    # 6. Verify entity files exist in correct folders
    assert (tmp_path / "moi").exists()
    assert (tmp_path / "proches").exists()

    # 7. Verify chats are now processed
    unprocessed = list_unprocessed_chats(tmp_path)
    assert len(unprocessed) == 0

    # 8. Verify _index.md was generated
    assert (tmp_path / "_index.md").exists()

    # 9. Build FAISS index
    from src.pipeline.indexer import build_index
    with patch("src.pipeline.indexer._get_embedding_fn", return_value=_mock_embed):
        manifest = build_index(tmp_path, config)
    assert len(manifest["indexed_files"]) > 0


def test_e2e_rebuild_from_md(tmp_path):
    """Test that graph can be rebuilt entirely from MD files."""
    config = _make_config(tmp_path)
    init_memory_structure(tmp_path)

    # Run a mini pipeline to create entities
    mock_extraction = RawExtraction(
        entities=[
            RawEntity(name="Test Entity", type="sante", observations=[
                RawObservation(category="fait", content="A test fact", importance=0.5),
            ]),
        ],
        relations=[],
        summary="test",
    )
    graph = load_graph(tmp_path)
    resolved = resolve_all(mock_extraction, graph)
    enrich_memory(resolved, config, today="2026-03-03")

    # Delete graph
    graph_path = tmp_path / "_graph.json"
    assert graph_path.exists()
    graph_path.unlink()

    # Rebuild
    graph = rebuild_from_md(tmp_path)
    assert "test-entity" in graph.entities
    assert graph.entities["test-entity"].title == "Test Entity"


def test_e2e_empty_chat_graceful(tmp_path):
    """Empty chat should be processed gracefully."""
    config = _make_config(tmp_path)
    init_memory_structure(tmp_path)

    # Test the extractor directly with truly empty content
    from src.pipeline.extractor import extract_from_chat
    result = extract_from_chat("", config)
    assert len(result.entities) == 0

    # Test with whitespace-only content
    result2 = extract_from_chat("   \n  \n  ", config)
    assert len(result2.entities) == 0


def test_e2e_inbox_processing(tmp_path):
    """Test inbox → chat → pipeline flow."""
    config = _make_config(tmp_path)
    init_memory_structure(tmp_path)

    # Drop a file in inbox
    inbox = tmp_path / "_inbox"
    (inbox / "test_note.md").write_text("J'ai commencé à apprendre le piano cette semaine.", encoding="utf-8")

    from src.pipeline.inbox import process_inbox
    processed = process_inbox(tmp_path, config)
    assert len(processed) == 1

    # Should have created a chat
    unprocessed = list_unprocessed_chats(tmp_path)
    assert len(unprocessed) == 1

    # Original file should be in _processed
    assert not (inbox / "test_note.md").exists()
    processed_files = list((inbox / "_processed").iterdir())
    assert len(processed_files) == 1


def test_e2e_stale_lockfile(tmp_path):
    """Stale lockfile (>5min old) is automatically removed."""
    import time
    from src.memory.graph import save_graph, load_graph
    from src.core.models import GraphData

    # Create a stale lockfile
    lock_path = tmp_path / "_graph.lock"
    lock_path.write_text("pid=99999\n")

    # Make it appear old (we'll just test the logic)
    import os
    # Set mtime to 10 minutes ago
    old_time = time.time() - 600
    os.utime(lock_path, (old_time, old_time))

    # save_graph should handle the stale lock
    graph = GraphData()
    save_graph(tmp_path, graph)
    assert (tmp_path / "_graph.json").exists()
    assert not lock_path.exists()  # lock was released
