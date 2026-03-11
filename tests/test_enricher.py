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


def test_relation_supersession(tmp_path):
    """When a relation has supersedes set, the old relation is removed from graph and MD."""
    from src.core.models import GraphRelation

    config = _setup_memory(tmp_path)

    # Pre-create Alice with a parent_of relation to Bob in MD
    alice_fm = EntityFrontmatter(
        title="Alice", type="person", frequency=3,
        importance=0.5, last_mentioned="2026-03-10", created="2026-01-01",
    )
    write_entity(
        tmp_path / "close_ones" / "alice.md", alice_fm,
        {"Facts": ["- [fact] Test person"], "Relations": ["- parent_of [[Bob]]"], "History": []},
    )

    bob_fm = EntityFrontmatter(
        title="Bob", type="person", frequency=3,
        importance=0.5, last_mentioned="2026-03-10", created="2026-01-01",
    )
    write_entity(
        tmp_path / "close_ones" / "bob.md", bob_fm,
        {"Facts": ["- [fact] Test person"], "Relations": [], "History": []},
    )

    # Pre-create graph with the old (wrong) relation
    graph = GraphData()
    graph = add_entity(graph, "alice", GraphEntity(
        file="close_ones/alice.md", type="person", title="Alice",
        frequency=3, importance=0.5, last_mentioned="2026-03-10",
    ))
    graph = add_entity(graph, "bob", GraphEntity(
        file="close_ones/bob.md", type="person", title="Bob",
        frequency=3, importance=0.5, last_mentioned="2026-03-10",
    ))
    from src.memory.graph import add_relation as graph_add_relation
    graph = graph_add_relation(graph, GraphRelation(
        from_entity="alice", to_entity="bob", type="parent_of",
    ))
    save_graph(tmp_path, graph)

    # Now enrich with a new relation that supersedes the old one
    resolved = ResolvedExtraction(
        resolved=[
            ResolvedEntity(
                raw=RawEntity(name="Alice", type="person", observations=[]),
                resolution=Resolution(status="resolved", entity_id="alice"),
            ),
        ],
        relations=[
            RawRelation(
                from_name="Alice", to_name="Bob",
                type="friend_of", context="actually friends",
                supersedes="alice:bob:parent_of",
            ),
        ],
        summary="Fix relation",
    )

    report = enrich_memory(resolved, config, today="2026-03-11")
    assert report.relations_added >= 1
    assert len(report.errors) == 0

    # Verify old relation removed from graph, new one added
    graph = load_graph(tmp_path)
    parent_rels = [r for r in graph.relations if r.type == "parent_of"
                   and r.from_entity == "alice" and r.to_entity == "bob"]
    assert len(parent_rels) == 0, "Old parent_of relation should be removed"

    friend_rels = [r for r in graph.relations if r.type == "friend_of"
                   and r.from_entity == "alice" and r.to_entity == "bob"]
    assert len(friend_rels) == 1, "New friend_of relation should exist"

    # Verify old relation removed from MD file
    alice_text = (tmp_path / "close_ones" / "alice.md").read_text()
    assert "parent_of [[Bob]]" not in alice_text, "Old relation line should be removed from MD"
    assert "friend_of [[Bob]]" in alice_text, "New relation line should be in MD"
