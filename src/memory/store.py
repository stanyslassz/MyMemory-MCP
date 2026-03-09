"""Markdown file CRUD operations for memory entities and chats."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from src.core.models import EntityFrontmatter
from src.core.utils import parse_frontmatter as _shared_parse_frontmatter


def init_memory_structure(memory_path: Path) -> None:
    """Create memory folder structure if missing."""
    folders = ["self", "close_ones", "projects", "work", "interests", "chats", "_inbox/_processed", "_archive"]
    for folder in folders:
        (memory_path / folder).mkdir(parents=True, exist_ok=True)


def read_entity(filepath: Path) -> tuple[EntityFrontmatter, dict[str, list[str]]]:
    """Read a markdown entity file. Returns (frontmatter, sections).

    Sections is a dict like {"Facts": [...lines], "Relations": [...lines], "History": [...lines]}.
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

    for section_name in ["Facts", "Relations", "History"]:
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
        existing_facts = sections.get("Facts", [])
        for obs in new_observations:
            line = _format_observation(obs)
            if not _is_duplicate_observation(line, existing_facts):
                existing_facts.append(line)
        sections["Facts"] = existing_facts

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
        "Facts": [],
        "Relations": [],
        "History": [f"- {frontmatter.created}: Created"],
    }

    if observations:
        for obs in observations:
            line = _format_observation(obs)
            sections["Facts"].append(line)

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
        "Facts": [],
        "Relations": [],
        "History": [f"- {today}: Created by forward reference"],
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


def consolidate_entity_facts(
    filepath: Path,
    config,
) -> dict:
    """Consolidate redundant observations in an entity via LLM.

    Returns a dict with 'original_count', 'consolidated_count', 'changes'.
    """
    from src.core.llm import call_fact_consolidation

    frontmatter, sections = read_entity(filepath)
    facts = sections.get("Facts", [])
    if not facts:
        return {"original_count": 0, "consolidated_count": 0, "changes": []}

    # Filter out already superseded facts — only consolidate live ones
    live_facts = [f for f in facts if "[superseded]" not in f]
    superseded_facts = [f for f in facts if "[superseded]" in f]

    if len(live_facts) < 3:
        return {"original_count": len(live_facts), "consolidated_count": len(live_facts), "changes": []}

    # Build indexed text for LLM
    indexed_text = "\n".join(f"{i}: {f}" for i, f in enumerate(live_facts))

    result = call_fact_consolidation(
        frontmatter.title, frontmatter.type, indexed_text, config,
    )

    # Build new facts list from consolidated result (with length guard)
    new_facts = []
    for cf in result.consolidated:
        content = cf.content
        # Guard: if LLM produced a mega-line, truncate to 150 chars
        if len(content) > 150:
            content = content[:147] + "..."
        obs = {
            "category": cf.category,
            "content": content,
            "date": cf.date,
            "valence": cf.valence,
            "tags": cf.tags[:3],  # Max 3 tags per fact
        }
        new_facts.append(_format_observation(obs))

    # Preserve superseded facts
    new_facts.extend(superseded_facts)

    changes = []
    if len(new_facts) != len(facts):
        changes.append(f"{len(live_facts)} → {len(result.consolidated)} live facts")

    sections["Facts"] = new_facts

    # Log in History
    from datetime import date
    history = sections.get("History", [])
    history.append(f"- {date.today().isoformat()}: Facts consolidated ({len(live_facts)} → {len(result.consolidated)})")
    sections["History"] = history

    write_entity(filepath, frontmatter, sections)

    return {
        "original_count": len(live_facts),
        "consolidated_count": len(result.consolidated),
        "changes": changes,
    }


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


def mark_chat_fallback(
    filepath: Path,
    fallback: str,
    error: str = "",
) -> None:
    """Mark a chat as processed via fallback (e.g. doc_ingest after extraction timeout).

    Sets processed=True plus fallback metadata so it is never retried for extraction.
    """
    text = filepath.read_text(encoding="utf-8")
    fm_data, body = _parse_frontmatter(text)

    fm_data["processed"] = True
    fm_data["processed_at"] = datetime.now().isoformat()
    fm_data["fallback"] = fallback
    if error:
        fm_data["fallback_reason"] = error
    fm_data["entities_updated"] = []
    fm_data["entities_created"] = []

    fm_yaml = yaml.safe_dump(fm_data, default_flow_style=False, allow_unicode=True)
    filepath.write_text(f"---\n{fm_yaml}---\n{body}", encoding="utf-8")


def increment_extraction_retries(filepath: Path) -> int:
    """Increment and return the extraction_retries counter in chat frontmatter."""
    text = filepath.read_text(encoding="utf-8")
    fm_data, body = _parse_frontmatter(text)

    retries = fm_data.get("extraction_retries", 0) + 1
    fm_data["extraction_retries"] = retries

    fm_yaml = yaml.safe_dump(fm_data, default_flow_style=False, allow_unicode=True)
    filepath.write_text(f"---\n{fm_yaml}---\n{body}", encoding="utf-8")
    return retries


def get_chat_content(filepath: Path) -> str:
    """Read a chat file and return only the body content (no frontmatter)."""
    text = filepath.read_text(encoding="utf-8")
    _, body = _parse_frontmatter(text)
    return body


# ── Private helpers ──────────────────────────────────────────

def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from a markdown file. Delegates to shared util."""
    return _shared_parse_frontmatter(text)


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


_VALENCE_MARKERS = {"positive": "[+]", "negative": "[-]", "neutral": "[~]"}
_VALENCE_REVERSE = {v: k for k, v in _VALENCE_MARKERS.items()}


def _format_observation(obs: dict) -> str:
    """Format an observation dict to markdown line.

    Format: - [category] (YYYY-MM) content [+] #tags
    Date and valence are optional.
    Superseded observations are wrapped in ~~ and suffixed with [superseded].
    """
    is_superseded = obs.get("superseded", False)
    content = obs['content']
    if is_superseded:
        content = f"~~{content}~~"
    line = f"- [{obs['category']}]"
    date = obs.get("date", "")
    if date:
        line += f" ({date})"
    line += f" {content}"
    valence = obs.get("valence", "")
    if valence and valence in _VALENCE_MARKERS:
        line += f" {_VALENCE_MARKERS[valence]}"
    if obs.get("tags"):
        for tag in obs["tags"]:
            if tag.startswith("#"):
                line += f" {tag}"
            else:
                line += f" #{tag}"
    if is_superseded:
        line += " [superseded]"
    return line


def _parse_observation(line: str) -> dict | None:
    """Parse a markdown fact line back into a dict.

    Handles: - [category] (date) content [+/-/~] #tags [superseded]
    """
    m = re.match(r"- \[(\w+)\]\s*(?:\(([^)]+)\)\s*)?(.+)", line)
    if not m:
        return None
    category = m.group(1)
    date = m.group(2) or ""
    rest = m.group(3).strip()

    # Detect superseded marker
    superseded = "[superseded]" in rest
    if superseded:
        rest = rest.replace("[superseded]", "").strip()

    # Extract valence marker
    valence = ""
    for marker, val in _VALENCE_REVERSE.items():
        if f" {marker}" in rest:
            valence = val
            rest = rest.replace(f" {marker}", "", 1).strip()
            break

    # Extract tags
    tags = re.findall(r"#(\S+)", rest)
    content = re.sub(r"\s*#\S+", "", rest).strip()

    # Strip strikethrough markers from content
    if superseded:
        content = content.strip("~")

    result = {"category": category, "date": date, "content": content,
              "valence": valence, "tags": tags}
    if superseded:
        result["superseded"] = True
    return result


def _is_duplicate_observation(new_line: str, existing_lines: list[str]) -> bool:
    """Check if an observation is a duplicate (same category + similar content).

    Ignores date/valence/tags in comparison — only category + content matter.
    Skips superseded lines when checking for duplicates.
    """
    new_obs = _parse_observation(new_line)
    if not new_obs:
        return False
    new_cat, new_content = new_obs["category"], new_obs["content"].lower()

    for existing in existing_lines:
        ex_obs = _parse_observation(existing)
        if not ex_obs:
            continue
        if ex_obs.get("superseded"):
            continue
        ex_cat, ex_content = ex_obs["category"], ex_obs["content"].lower()
        if new_cat == ex_cat and (new_content in ex_content or ex_content in new_content):
            return True
    return False


def mark_observation_superseded(
    existing_facts: list[str],
    category: str,
    supersedes_text: str,
) -> list[str]:
    """Find and mark a matching observation as superseded.

    Searches for a fact with the same category whose content contains
    the supersedes_text (case-insensitive). Returns updated list.
    """
    supersedes_lower = supersedes_text.lower()
    result = []
    for line in existing_facts:
        obs = _parse_observation(line)
        if (obs and not obs.get("superseded")
                and obs["category"] == category
                and supersedes_lower in obs["content"].lower()):
            obs["superseded"] = True
            result.append(_format_observation(obs))
        else:
            result.append(line)
    return result
