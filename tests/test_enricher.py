"""Tests for pipeline/enricher.py."""

from pathlib import Path

from src.core.config import load_config
from src.core.models import (
    RawEntity,
    RawObservation,
    RawRelation,
    Resolution,
    ResolvedEntity,
    ResolvedExtraction,
    EntityFrontmatter,
    GraphData,
    GraphEntity,
)
from src.memory.graph import add_entity, save_graph, load_graph
from src.memory.store import init_memory_structure, write_entity
from src.pipeline.enricher import enrich_memory


def _setup_memory(tmp_path):
    """Create a memory structure with a config pointing to tmp_path."""
    init_memory_structure(tmp_path)
    config = load_config(project_root=Path(__file__).parent.parent)
    # Override memory_path to use tmp
    config.memory_path = tmp_path
    return config


def test_enrich_new_entity(tmp_path):
    config = _setup_memory(tmp_path)

    resolved = ResolvedExtraction(
        resolved=[
            ResolvedEntity(
                raw=RawEntity(
                    name="Yoga",
                    type="interest",
                    observations=[
                        RawObservation(category="fact", content="Pratique le yoga", importance=0.5),
                    ],
                ),
                resolution=Resolution(status="new", suggested_slug="yoga"),
            ),
        ],
        relations=[],
        summary="Test",
    )

    report = enrich_memory(resolved, config, today="2026-03-03")
    assert "yoga" in report.entities_created
    assert len(report.errors) == 0

    # Verify MD file was created
    assert (tmp_path / "interests" / "yoga.md").exists()

    # Verify graph was updated
    graph = load_graph(tmp_path)
    assert "yoga" in graph.entities


def test_enrich_existing_entity(tmp_path):
    config = _setup_memory(tmp_path)

    # Pre-create an entity
    fm = EntityFrontmatter(
        title="Natation", type="interest", frequency=5,
        importance=0.6, last_mentioned="2026-01-01", created="2025-01-01",
    )
    write_entity(
        tmp_path / "interests" / "natation.md", fm,
        {"Facts": ["- [fact] Aime nager"], "Relations": [], "History": []},
    )

    # Pre-create graph
    graph = GraphData()
    graph = add_entity(graph, "natation", GraphEntity(
        file="interests/natation.md", type="interest", title="Natation",
        frequency=5, importance=0.6, last_mentioned="2026-01-01",
    ))
    save_graph(tmp_path, graph)

    resolved = ResolvedExtraction(
        resolved=[
            ResolvedEntity(
                raw=RawEntity(
                    name="Natation",
                    type="interest",
                    observations=[
                        RawObservation(category="fact", content="Nage 3 fois par semaine", importance=0.7),
                    ],
                ),
                resolution=Resolution(status="resolved", entity_id="natation"),
            ),
        ],
        relations=[],
        summary="Test update",
    )

    report = enrich_memory(resolved, config, today="2026-03-03")
    assert "natation" in report.entities_updated

    # Verify file was updated
    from src.memory.store import read_entity
    fm2, sections = read_entity(tmp_path / "interests" / "natation.md")
    assert fm2.frequency == 6  # incremented
    assert fm2.last_mentioned == "2026-03-03"
    assert len(sections["Facts"]) == 2  # original + new


def test_enrich_with_relations(tmp_path):
    config = _setup_memory(tmp_path)

    # Pre-create source entity
    fm = EntityFrontmatter(
        title="Mal de dos", type="health", frequency=3,
        last_mentioned="2026-01-01", created="2025-01-01",
    )
    write_entity(
        tmp_path / "self" / "mal-de-dos.md", fm,
        {"Facts": ["- [diagnosis] Sciatique"], "Relations": [], "History": []},
    )
    graph = GraphData()
    graph = add_entity(graph, "mal-de-dos", GraphEntity(
        file="self/mal-de-dos.md", type="health", title="Mal de dos",
        frequency=3, last_mentioned="2026-01-01",
    ))
    save_graph(tmp_path, graph)

    resolved = ResolvedExtraction(
        resolved=[
            ResolvedEntity(
                raw=RawEntity(name="Mal de dos", type="health", observations=[]),
                resolution=Resolution(status="resolved", entity_id="mal-de-dos"),
            ),
        ],
        relations=[
            RawRelation(
                from_name="Mal de dos", to_name="Natation",
                type="improves", context="soulage la douleur",
            ),
        ],
        summary="Test relations",
    )

    report = enrich_memory(resolved, config, today="2026-03-03")
    assert report.relations_added >= 1

    # Natation should have been created as a stub (forward reference)
    graph = load_graph(tmp_path)
    assert "natation" in graph.entities


def test_create_new_entity_has_mention_dates(tmp_path):
    """New entities must have mention_dates=[today] in frontmatter."""
    import re
    import yaml
    from src.pipeline.enricher import enrich_memory
    from src.core.config import Config, CategoriesConfig
    from src.core.models import (
        ResolvedExtraction, ResolvedEntity, Resolution,
        RawEntity, RawObservation,
    )

    memory_path = tmp_path / "memory"
    (memory_path / "close_ones").mkdir(parents=True)
    (memory_path / "_graph.json").write_text('{"entities":{}, "relations":[]}')

    config = Config(
        memory_path=memory_path,
        categories=CategoriesConfig(folders={"person": "close_ones"}),
    )

    resolved = ResolvedExtraction(
        resolved=[
            ResolvedEntity(
                raw=RawEntity(
                    name="TestPerson",
                    type="person",
                    observations=[
                        RawObservation(category="fact", content="Test fact", importance=0.5, tags=["test"]),
                    ],
                ),
                resolution=Resolution(status="new", suggested_slug="testperson"),
            ),
        ],
        relations=[],
    )

    enrich_memory(resolved, config, today="2026-03-06")

    entity_file = memory_path / "close_ones" / "testperson.md"
    assert entity_file.exists()
    text = entity_file.read_text()
    match = re.match(r"^---\n(.*?\n)---\n", text, re.DOTALL)
    assert match
    fm = yaml.safe_load(match.group(1))
    assert fm["mention_dates"] == ["2026-03-06"]
