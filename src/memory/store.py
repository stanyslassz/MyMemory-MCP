"""Markdown file CRUD operations for memory entities and chats."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from src.core.models import EntityFrontmatter


def init_memory_structure(memory_path: Path) -> None:
    """Create memory folder structure if missing."""
    folders = ["moi", "proches", "projets", "travail", "interets", "chats", "_inbox/_processed"]
    for folder in folders:
        (memory_path / folder).mkdir(parents=True, exist_ok=True)


def read_entity(filepath: Path) -> tuple[EntityFrontmatter, dict[str, list[str]]]:
    """Read a markdown entity file. Returns (frontmatter, sections).

    Sections is a dict like {"Faits": [...lines], "Relations": [...lines], "Historique": [...lines]}.
    """
    text = filepath.read_text(encoding="utf-8")
    fm_data, body = _parse_frontmatter(text)
    frontmatter = EntityFrontmatter.model_validate(fm_data)
    sections = _parse_sections(body)
    return frontmatter, sections


def write_entity(filepath: Path, frontmatter: EntityFrontmatter, sections: dict[str, list[str]]) -> None:
    """Write an entity to a markdown file with YAML frontmatter + sections."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    fm_dict = frontmatter.model_dump()
    # Ensure clean YAML output
    fm_yaml = yaml.safe_dump(fm_dict, default_flow_style=False, allow_unicode=True, sort_keys=False)

    lines = ["---\n", fm_yaml, "---\n", "\n"]
    lines.append(f"# {frontmatter.title}\n\n")

    for section_name in ["Faits", "Relations", "Historique"]:
        lines.append(f"## {section_name}\n")
        items = sections.get(section_name, [])
        for item in items:
            if item.strip():
                lines.append(f"{item}\n")
        lines.append("\n")

    filepath.write_text("".join(lines), encoding="utf-8")


def update_entity(
    filepath: Path,
    new_observations: list[dict[str, str]] | None = None,
    new_relations: list[str] | None = None,
    frequency_increment: int = 1,
    last_mentioned: str | None = None,
) -> EntityFrontmatter:
    """Update an existing entity: add observations, relations, bump frequency."""
    frontmatter, sections = read_entity(filepath)

    # Add new observations (avoid duplicates: same category + similar content)
    if new_observations:
        existing_facts = sections.get("Faits", [])
        for obs in new_observations:
            line = f"- [{obs['category']}] {obs['content']}"
            if obs.get("tags"):
                for tag in obs["tags"]:
                    if tag.startswith("#"):
                        line += f" {tag}"
                    else:
                        line += f" #{tag}"
            if not _is_duplicate_observation(line, existing_facts):
                existing_facts.append(line)
        sections["Faits"] = existing_facts

    # Add new relations (avoid duplicates)
    if new_relations:
        existing_rels = sections.get("Relations", [])
        for rel in new_relations:
            if rel not in existing_rels:
                existing_rels.append(rel)
        sections["Relations"] = existing_rels

    # Update frontmatter
    frontmatter.frequency += frequency_increment
    if last_mentioned:
        frontmatter.last_mentioned = last_mentioned

    write_entity(filepath, frontmatter, sections)
    return frontmatter


def create_entity(
    memory_path: Path,
    folder: str,
    slug: str,
    frontmatter: EntityFrontmatter,
    observations: list[dict[str, str]] | None = None,
    relations: list[str] | None = None,
) -> Path:
    """Create a new entity MD file in the appropriate folder."""
    filepath = memory_path / folder / f"{slug}.md"
    sections: dict[str, list[str]] = {
        "Faits": [],
        "Relations": [],
        "Historique": [f"- {frontmatter.created}: Créé"],
    }

    if observations:
        for obs in observations:
            line = f"- [{obs['category']}] {obs['content']}"
            sections["Faits"].append(line)

    if relations:
        sections["Relations"] = relations

    write_entity(filepath, frontmatter, sections)
    return filepath


def create_stub_entity(
    memory_path: Path,
    folder: str,
    slug: str,
    title: str,
    entity_type: str,
    today: str,
) -> Path:
    """Create a stub entity for forward references."""
    fm = EntityFrontmatter(
        title=title,
        type=entity_type,
        retention="short_term",
        score=0.0,
        importance=0.3,
        frequency=1,
        last_mentioned=today,
        created=today,
        aliases=[],
        tags=[],
    )
    sections = {
        "Faits": [],
        "Relations": [],
        "Historique": [f"- {today}: Created by forward reference"],
    }
    filepath = memory_path / folder / f"{slug}.md"
    write_entity(filepath, fm, sections)
    return filepath


def list_entities(base_path: Path) -> list[dict[str, Any]]:
    """List all entity MD files under base_path (recursive, exclude _* and chats/)."""
    results = []
    for md_file in sorted(base_path.rglob("*.md")):
        rel = md_file.relative_to(base_path)
        parts = rel.parts
        # Skip _ prefixed dirs/files and chats/
        if any(p.startswith("_") for p in parts) or (parts and parts[0] == "chats"):
            continue
        try:
            fm, _ = read_entity(md_file)
            results.append({"path": md_file, "frontmatter": fm})
        except Exception:
            continue
    return results


def save_chat(messages: list[dict], memory_path: Path) -> Path:
    """Save a conversation to chats/ with processed: false."""
    chats_dir = memory_path / "chats"
    chats_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    filename = f"{now.strftime('%Y-%m-%d_%Hh%M')}.md"
    filepath = chats_dir / filename

    # Avoid collision
    counter = 1
    while filepath.exists():
        filename = f"{now.strftime('%Y-%m-%d_%Hh%M')}_{counter}.md"
        filepath = chats_dir / filename
        counter += 1

    fm = {
        "date": now.strftime("%Y-%m-%d"),
        "participants": ["user", "assistant"],
        "processed": False,
    }
    fm_yaml = yaml.safe_dump(fm, default_flow_style=False, allow_unicode=True)

    lines = ["---\n", fm_yaml, "---\n", "\n", f"# Chat {now.strftime('%Y-%m-%d')}\n\n"]
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        lines.append(f"**{role}**: {content}\n\n")

    filepath.write_text("".join(lines), encoding="utf-8")
    return filepath


def list_unprocessed_chats(memory_path: Path) -> list[Path]:
    """List chat files that haven't been processed yet."""
    chats_dir = memory_path / "chats"
    if not chats_dir.exists():
        return []

    unprocessed = []
    for md_file in sorted(chats_dir.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        fm_data, _ = _parse_frontmatter(text)
        if not fm_data.get("processed", False):
            unprocessed.append(md_file)
    return unprocessed


def mark_chat_processed(
    filepath: Path,
    entities_updated: list[str],
    entities_created: list[str],
) -> None:
    """Mark a chat file as processed by updating its frontmatter."""
    text = filepath.read_text(encoding="utf-8")
    fm_data, body = _parse_frontmatter(text)

    fm_data["processed"] = True
    fm_data["processed_at"] = datetime.now().isoformat()
    fm_data["entities_updated"] = entities_updated
    fm_data["entities_created"] = entities_created

    fm_yaml = yaml.safe_dump(fm_data, default_flow_style=False, allow_unicode=True)
    filepath.write_text(f"---\n{fm_yaml}---\n{body}", encoding="utf-8")


def get_chat_content(filepath: Path) -> str:
    """Read a chat file and return only the body content (no frontmatter)."""
    text = filepath.read_text(encoding="utf-8")
    _, body = _parse_frontmatter(text)
    return body


# ── Private helpers ──────────────────────────────────────────

def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from a markdown file. Returns (fm_dict, body)."""
    match = re.match(r"^---\n(.*?\n)---\n(.*)", text, re.DOTALL)
    if not match:
        return {}, text
    fm_text = match.group(1)
    body = match.group(2)
    fm_data = yaml.safe_load(fm_text) or {}
    return fm_data, body


def _parse_sections(body: str) -> dict[str, list[str]]:
    """Parse markdown sections (## Header) from body text."""
    sections: dict[str, list[str]] = {}
    current_section = None

    for line in body.split("\n"):
        if line.startswith("## "):
            current_section = line[3:].strip()
            sections[current_section] = []
        elif current_section is not None and line.strip():
            # Skip the # Title line
            if not line.startswith("# "):
                sections[current_section].append(line.strip())

    return sections


def _is_duplicate_observation(new_line: str, existing_lines: list[str]) -> bool:
    """Check if an observation is a duplicate (same category + similar content)."""
    # Extract category from line like "- [category] content"
    new_match = re.match(r"- \[(\w+)\] (.+)", new_line)
    if not new_match:
        return False
    new_cat, new_content = new_match.group(1), new_match.group(2).lower()

    for existing in existing_lines:
        ex_match = re.match(r"- \[(\w+)\] (.+)", existing)
        if not ex_match:
            continue
        ex_cat, ex_content = ex_match.group(1), ex_match.group(2).lower()
        if new_cat == ex_cat and (new_content in ex_content or ex_content in new_content):
            return True
    return False
