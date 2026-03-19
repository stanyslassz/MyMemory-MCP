"""Integration test: extraction → resolution → enrichment → context.

Mocks only the LLM extraction (returns a fixed RawExtraction).
All downstream steps (resolve, enrich, context build) run for real.
"""

import json
from pathlib import Path

from src.core.config import Config
from src.core.models import (
    GraphData,
    RawEntity,
    RawExtraction,
    RawObservation,
    RawRelation,
)
from src.memory.graph import load_graph, save_graph
from src.memory.store import read_entity
from src.pipeline.enricher import enrich_memory
from src.pipeline.resolver import resolve_all


def _make_config(tmp_path: Path) -> Config:
    config = Config()
    config.memory_path = tmp_path
    config.project_root = tmp_path
    config.faiss.index_path = str(tmp_path / "_memory.faiss")
    config.faiss.mapping_path = str(tmp_path / "_memory.pkl")
    config.faiss.manifest_path = str(tmp_path / "_faiss_manifest.json")
    config.prompts_path = tmp_path / "prompts"
    config.categories.folders = {
        "person": "close_ones",
        "health": "self",
        "interest": "interests",
        "work": "work",
        "ai_self": "self",
        "project": "projects",
        "place": "interests",
        "animal": "close_ones",
        "organization": "work",
    }
    return config


def _make_extraction() -> RawExtraction:
    """Fixed extraction simulating what the LLM would return."""
    return RawExtraction(
        entities=[
            RawEntity(
                name="Dr. Martin",
                type="person",
                observations=[
                    RawObservation(
                        category="fact",
                        content="Family doctor, general practitioner",
                        importance=0.7,
                        tags=["health"],
                    ),
                    RawObservation(
                        category="fact",
                        content="Office located downtown",
                        importance=0.4,
                    ),
                ],
            ),
            RawEntity(
                name="Back Pain",
                type="health",
                observations=[
                    RawObservation(
                        category="diagnosis",
                        content="Chronic lumbar pain",
                        importance=0.8,
                        valence="negative",
                    ),
                    RawObservation(
                        category="treatment",
                        content="Physical therapy 3x/week",
                        importance=0.7,
                        valence="positive",
                    ),
                ],
            ),
        ],
        relations=[
            RawRelation(
                from_name="Dr. Martin",
                to_name="Back Pain",
                type="linked_to",
                context="Dr. Martin manages back pain treatment",
            ),
        ],
        summary="Discussion about back pain treatment with Dr. Martin.",
    )


def _init_dirs(tmp_path: Path):
    """Create required subdirectories."""
    for d in ["self", "close_ones", "interests", "work", "projects", "chats"]:
        (tmp_path / d).mkdir(parents=True, exist_ok=True)


def test_full_pipeline_creates_entities(tmp_path):
    """Extraction → resolution → enrichment should create entity files."""
    config = _make_config(tmp_path)
    _init_dirs(tmp_path)

    extraction = _make_extraction()
    graph = GraphData()
    save_graph(tmp_path, graph)

    # Step 2: Resolution (no existing entities, so all are "new")
    resolved = resolve_all(extraction, graph)

    # Step 4: Enrichment
    report = enrich_memory(resolved, config)

    # Verify entities were created
    assert len(report.entities_created) >= 2
    assert report.relations_added >= 1

    # Verify graph has the entities
    graph = load_graph(tmp_path)
    entity_ids = list(graph.entities.keys())
    assert any("martin" in eid for eid in entity_ids)
    assert any("back-pain" in eid or "back_pain" in eid for eid in entity_ids)


def test_full_pipeline_md_files_exist(tmp_path):
    """Entity MD files should be created in correct folders."""
    config = _make_config(tmp_path)
    _init_dirs(tmp_path)

    extraction = _make_extraction()
    graph = GraphData()
    save_graph(tmp_path, graph)

    resolved = resolve_all(extraction, graph)
    enrich_memory(resolved, config)

    # Dr. Martin is a person → should be in close_ones/
    person_files = list((tmp_path / "close_ones").glob("*.md"))
    assert len(person_files) >= 1

    # Back Pain is health → should be in self/
    health_files = list((tmp_path / "self").glob("*.md"))
    assert len(health_files) >= 1


def test_full_pipeline_facts_in_md(tmp_path):
    """Entity MD files should contain the extracted facts."""
    config = _make_config(tmp_path)
    _init_dirs(tmp_path)

    extraction = _make_extraction()
    graph = GraphData()
    save_graph(tmp_path, graph)

    resolved = resolve_all(extraction, graph)
    enrich_memory(resolved, config)

    graph = load_graph(tmp_path)

    # Find the back pain entity
    back_pain_id = None
    for eid in graph.entities:
        if "back" in eid and "pain" in eid:
            back_pain_id = eid
            break

    assert back_pain_id is not None, f"Back pain entity not found in {list(graph.entities.keys())}"

    entity = graph.entities[back_pain_id]
    md_path = tmp_path / entity.file
    assert md_path.exists()

    fm, sections = read_entity(md_path)
    facts = sections.get("Facts", [])
    assert len(facts) >= 2

    # Check content is present
    facts_text = "\n".join(facts)
    assert "lumbar" in facts_text.lower() or "chronic" in facts_text.lower()
    assert "therapy" in facts_text.lower() or "physical" in facts_text.lower()


def test_full_pipeline_relations_in_graph(tmp_path):
    """Relations should be persisted in the graph."""
    config = _make_config(tmp_path)
    _init_dirs(tmp_path)

    extraction = _make_extraction()
    graph = GraphData()
    save_graph(tmp_path, graph)

    resolved = resolve_all(extraction, graph)
    enrich_memory(resolved, config)

    graph = load_graph(tmp_path)
    assert len(graph.relations) >= 1

    # Should have a linked_to relation
    rel = graph.relations[0]
    assert rel.type == "linked_to"


def test_full_pipeline_graph_consistent_with_md(tmp_path):
    """Graph entity count should match MD file count."""
    config = _make_config(tmp_path)
    _init_dirs(tmp_path)

    extraction = _make_extraction()
    graph = GraphData()
    save_graph(tmp_path, graph)

    resolved = resolve_all(extraction, graph)
    enrich_memory(resolved, config)

    graph = load_graph(tmp_path)

    # Count all entity MD files
    md_count = 0
    for folder in ["self", "close_ones", "interests", "work", "projects"]:
        md_count += len(list((tmp_path / folder).glob("*.md")))

    # Graph entity count should equal MD file count
    assert len(graph.entities) == md_count, (
        f"Graph has {len(graph.entities)} entities but found {md_count} MD files"
    )

    # Each graph entity should reference an existing file
    for eid, entity in graph.entities.items():
        assert (tmp_path / entity.file).exists(), (
            f"Entity {eid} references missing file {entity.file}"
        )


def test_full_pipeline_context_generation(tmp_path):
    """After enrichment, context generation should produce valid _context.md."""
    config = _make_config(tmp_path)
    _init_dirs(tmp_path)
    config.scoring.min_score_for_context = 0.0  # Include everything

    # Create prompts directory with context template
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir(exist_ok=True)
    (prompts_dir / "context_template.md").write_text(
        "# Memory Context — {date}\n\n"
        "Language: {user_language_name}\n\n"
        "{ai_personality}\n\n"
        "{sections}\n\n"
        "## Available Entities\n{available_entities}\n\n"
        "{custom_instructions}\n",
        encoding="utf-8",
    )
    (prompts_dir / "context_instructions.md").write_text("", encoding="utf-8")

    extraction = _make_extraction()
    graph = GraphData()
    save_graph(tmp_path, graph)

    resolved = resolve_all(extraction, graph)
    enrich_memory(resolved, config)

    # Build context
    graph = load_graph(tmp_path)
    from src.memory.context import build_context_for_config, write_context

    context_text = build_context_for_config(graph, tmp_path, config, use_llm=False)
    assert len(context_text) > 0

    write_context(tmp_path, context_text)
    context_file = tmp_path / "_context.md"
    assert context_file.exists()

    content = context_file.read_text(encoding="utf-8")
    assert "Memory Context" in content


def test_pipeline_second_run_updates_existing(tmp_path):
    """Running the pipeline twice should update (not duplicate) entities."""
    config = _make_config(tmp_path)
    _init_dirs(tmp_path)

    extraction = _make_extraction()
    graph = GraphData()
    save_graph(tmp_path, graph)

    # First run
    resolved = resolve_all(extraction, graph)
    enrich_memory(resolved, config)

    graph = load_graph(tmp_path)
    entity_count_1 = len(graph.entities)

    # Second run with same extraction
    resolved2 = resolve_all(extraction, graph)
    enrich_memory(resolved2, config)

    graph2 = load_graph(tmp_path)
    entity_count_2 = len(graph2.entities)

    # Should not create duplicate entities
    assert entity_count_2 == entity_count_1, (
        f"Second run created duplicates: {entity_count_1} → {entity_count_2}"
    )
