"""Graph operations on _graph.json. CRUD, BFS traversal, rebuild from MDs."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path

from src.core.models import GraphData, GraphEntity, GraphRelation
from src.core.utils import parse_frontmatter, slugify


LOCK_TIMEOUT_SECONDS = 300  # 5 minutes
logger = logging.getLogger(__name__)


def load_graph(memory_path: Path) -> GraphData:
    """Load _graph.json with graceful degradation.

    On corruption: try .bak restore, then rebuild from MDs.
    """
    graph_path = memory_path / "_graph.json"
    bak_path = memory_path / "_graph.json.bak"

    if not graph_path.exists():
        return GraphData(generated=datetime.now().isoformat())

    # Try primary file
    try:
        text = graph_path.read_text(encoding="utf-8")
        data = json.loads(text)
        return GraphData.model_validate(data)
    except (json.JSONDecodeError, ValueError, KeyError) as exc:
        logger.warning("Corrupt _graph.json: %s", exc)

    # Try .bak restore
    if bak_path.exists():
        try:
            text = bak_path.read_text(encoding="utf-8")
            data = json.loads(text)
            graph = GraphData.model_validate(data)
            logger.info("Restored graph from .bak")
            # Replace corrupt primary with good backup (atomic)
            _atomic_write(graph_path, json.dumps(data, indent=2, ensure_ascii=False))
            return graph
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            logger.warning("Corrupt _graph.json.bak: %s", exc)

    # Last resort: rebuild from MDs
    logger.warning("Rebuilding graph from markdown files")
    graph = rebuild_from_md(memory_path)
    save_graph(memory_path, graph)
    return graph


def save_graph(memory_path: Path, graph: GraphData) -> None:
    """Save _graph.json with .bak backup, atomic writes, and lockfile protection."""
    graph_path = memory_path / "_graph.json"
    lock_path = memory_path / "_graph.lock"

    _acquire_lock(lock_path)
    try:
        # Backup existing
        if graph_path.exists():
            shutil.copy2(graph_path, memory_path / "_graph.json.bak")

        graph.generated = datetime.now().isoformat()
        data = graph.model_dump(by_alias=True)
        _atomic_write(graph_path, json.dumps(data, indent=2, ensure_ascii=False))
    finally:
        _release_lock(lock_path)


def add_entity(graph: GraphData, entity_id: str, entity: GraphEntity) -> GraphData:
    """Add or replace an entity in the graph."""
    graph.entities[entity_id] = entity
    return graph


def update_entity(graph: GraphData, entity_id: str, **updates) -> GraphData:
    """Update specific fields of an entity."""
    if entity_id in graph.entities:
        entity = graph.entities[entity_id]
        for key, value in updates.items():
            if hasattr(entity, key):
                setattr(entity, key, value)
    return graph


def add_relation(graph: GraphData, relation: GraphRelation, *, strength_growth: float = 0.05) -> GraphData:
    """Add a relation, or reinforce existing duplicate (same from+to+type)."""
    for existing in graph.relations:
        if (existing.from_entity == relation.from_entity
                and existing.to_entity == relation.to_entity
                and existing.type == relation.type):
            # Reinforce existing relation
            existing.mention_count += 1
            existing.last_reinforced = datetime.now().isoformat()
            # Hebbian: strength grows with co-activation, capped at 1.0
            existing.strength = min(1.0, existing.strength + strength_growth)
            if relation.context and not existing.context:
                existing.context = relation.context
            return graph
    # New relation
    if not relation.created:
        relation.created = datetime.now().isoformat()
    if not relation.last_reinforced:
        relation.last_reinforced = relation.created
    graph.relations.append(relation)
    return graph


def remove_orphan_relations(graph: GraphData) -> GraphData:
    """Remove relations that reference non-existent entities."""
    graph.relations = [
        r for r in graph.relations
        if r.from_entity in graph.entities and r.to_entity in graph.entities
    ]
    return graph


def get_related(graph: GraphData, entity_id: str, depth: int = 1) -> list[str]:
    """BFS traversal: get related entity IDs up to given depth."""
    if entity_id not in graph.entities:
        return []

    visited = set()
    queue = [(entity_id, 0)]
    result = []

    while queue:
        current, d = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)

        if current != entity_id:
            result.append(current)

        if d < depth:
            for rel in graph.relations:
                if rel.from_entity == current and rel.to_entity not in visited:
                    queue.append((rel.to_entity, d + 1))
                if rel.to_entity == current and rel.from_entity not in visited:
                    queue.append((rel.from_entity, d + 1))

    return result


def get_aliases_lookup(graph: GraphData) -> dict[str, str]:
    """Build a lowercase alias → entity_id lookup dict."""
    lookup: dict[str, str] = {}
    for entity_id, entity in graph.entities.items():
        lookup[entity_id.lower()] = entity_id
        lookup[entity.title.lower()] = entity_id
        for alias in entity.aliases:
            lookup[alias.lower()] = entity_id
    return lookup


def validate_graph(graph: GraphData, memory_path: Path) -> list[str]:
    """Validate graph consistency. Returns list of warnings."""
    warnings = []

    # Check entity files exist
    for entity_id, entity in graph.entities.items():
        filepath = memory_path / entity.file
        if not filepath.exists():
            warnings.append(f"Entity '{entity_id}' references missing file: {entity.file}")

    # Check relations reference existing entities
    for rel in graph.relations:
        if rel.from_entity not in graph.entities:
            warnings.append(f"Relation from unknown entity: {rel.from_entity}")
        if rel.to_entity not in graph.entities:
            warnings.append(f"Relation to unknown entity: {rel.to_entity}")

    return warnings


def rebuild_from_md(memory_path: Path) -> GraphData:
    """Rebuild _graph.json entirely from markdown files."""
    graph = GraphData(generated=datetime.now().isoformat())

    for md_file in sorted(memory_path.rglob("*.md")):
        rel = md_file.relative_to(memory_path)
        parts = rel.parts
        # Skip _ prefixed, chats/, etc.
        if any(p.startswith("_") for p in parts) or (parts and parts[0] == "chats"):
            continue

        try:
            text = md_file.read_text(encoding="utf-8")
            fm_data, body = _parse_frontmatter_raw(text)
            if "title" not in fm_data or "type" not in fm_data:
                continue

            slug = md_file.stem
            entity = GraphEntity(
                file=str(rel),
                type=fm_data["type"],
                title=fm_data["title"],
                score=fm_data.get("score", 0.0),
                importance=fm_data.get("importance", 0.0),
                frequency=fm_data.get("frequency", 0),
                last_mentioned=str(fm_data.get("last_mentioned", "")),
                retention=fm_data.get("retention", "short_term"),
                aliases=fm_data.get("aliases", []),
                tags=fm_data.get("tags", []),
                mention_dates=fm_data.get("mention_dates", []),
                monthly_buckets=fm_data.get("monthly_buckets", {}),
                created=str(fm_data.get("created", "")),
                summary=fm_data.get("summary", ""),
                negative_valence_ratio=_compute_negative_valence_ratio(body),
            )
            graph.entities[slug] = entity

            # Parse relations from ## Relations section
            _parse_relations_from_body(body, slug, graph)

        except Exception:
            continue

    return graph


# ── Private helpers ──────────────────────────────────────────

def _atomic_write(filepath: Path, content: str) -> None:
    """Write content to file atomically via temp file + os.replace."""
    fd, tmp = tempfile.mkstemp(dir=filepath.parent, suffix=".tmp")
    try:
        os.write(fd, content.encode("utf-8"))
        os.close(fd)
        os.replace(tmp, filepath)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _acquire_lock(lock_path: Path) -> None:
    """Acquire lockfile atomically. Delete stale locks (>5 min)."""
    if lock_path.exists():
        try:
            age = time.time() - lock_path.stat().st_mtime
            if age > LOCK_TIMEOUT_SECONDS:
                lock_path.unlink()
            else:
                raise RuntimeError(f"Graph is locked by another process. Lock file: {lock_path}")
        except FileNotFoundError:
            pass  # Race condition, lock was already removed

    # Atomic lock creation: O_CREAT|O_EXCL fails if file already exists
    content = f"pid={os.getpid()}\ntime={datetime.now().isoformat()}\n".encode("utf-8")
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, content)
        os.close(fd)
    except FileExistsError:
        raise RuntimeError(f"Graph is locked by another process. Lock file: {lock_path}")


def _release_lock(lock_path: Path) -> None:
    """Release lockfile."""
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def _parse_frontmatter_raw(text: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from markdown text. Delegates to shared util."""
    return parse_frontmatter(text)


def _parse_relations_from_body(body: str, entity_slug: str, graph: GraphData) -> None:
    """Parse [[Target]] relations from the Relations section of a markdown body."""
    in_relations = False
    for line in body.split("\n"):
        if line.startswith("## Relations"):
            in_relations = True
            continue
        if line.startswith("## ") and in_relations:
            break
        if in_relations and line.strip().startswith("- "):
            # Pattern: "- relation_type [[Target]]"
            match = re.match(r"- (\w+) \[\[(.+?)\]\]", line.strip())
            if match:
                rel_type = match.group(1)
                target_title = match.group(2)
                target_slug = _slugify(target_title)
                try:
                    rel = GraphRelation(from_entity=entity_slug, to_entity=target_slug, type=rel_type)
                    add_relation(graph, rel)
                except Exception:
                    pass  # Invalid relation type, skip


def _compute_negative_valence_ratio(body: str) -> float:
    """Compute ratio of negative/vigilance/diagnosis facts in the Facts section.

    Scans for [-] valence markers and [vigilance]/[diagnosis] categories.
    Returns 0.0 if no facts found.
    """
    _EMOTIONAL_CATEGORIES = {"vigilance", "diagnosis", "treatment"}
    in_facts = False
    total = 0
    negative_count = 0
    for line in body.split("\n"):
        if line.startswith("## Facts"):
            in_facts = True
            continue
        if line.startswith("## ") and in_facts:
            break
        if in_facts and line.strip().startswith("- ["):
            total += 1
            # Check for negative valence marker
            if "[-]" in line:
                negative_count += 1
            else:
                # Check for emotional categories
                cat_match = re.match(r"- \[(\w+)\]", line.strip())
                if cat_match and cat_match.group(1) in _EMOTIONAL_CATEGORIES:
                    negative_count += 1
    if total == 0:
        return 0.0
    return round(negative_count / total, 4)


def _slugify(text: str) -> str:
    """Convert a title to a slug (lowercase, hyphens). Delegates to shared util."""
    return slugify(text)
