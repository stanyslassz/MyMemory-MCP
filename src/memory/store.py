"""Markdown file CRUD operations for memory entities and chats."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from src.core.models import EntityFrontmatter
from src.core.utils import atomic_write_text as _atomic_write_text, filter_live_facts, is_entity_file, parse_frontmatter as _shared_parse_frontmatter

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


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
    fm_data, body = _shared_parse_frontmatter(text)
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

    _atomic_write_text(filepath, "".join(lines))


def update_entity(
    filepath: Path,
    new_observations: list[dict[str, str]] | None = None,
    new_relations: list[str] | None = None,
    frequency_increment: int = 1,
    last_mentioned: str | None = None,
    max_facts: int | None = None,
    mention_dates: list[str] | None = None,
    monthly_buckets: dict[str, int] | None = None,
) -> EntityFrontmatter:
    """Update an existing entity: add observations, relations, bump frequency."""

    frontmatter, sections = read_entity(filepath)

    # Add new observations (avoid duplicates: same category + similar content)
    if new_observations:
        existing_facts = sections.get("Facts", [])
        for obs in new_observations:
            line = format_observation(obs)
            if not _is_duplicate_observation(line, existing_facts):
                existing_facts.append(line)
        # Hard cap safety net: if way over limit, keep most recent
        if max_facts:
            live_facts = filter_live_facts(existing_facts)
            if len(live_facts) > max_facts * 2:
                logger.warning(
                    "Entity %s has %d facts (cap %d), truncating to %d most recent",
                    frontmatter.title, len(live_facts), max_facts, max_facts * 2,
                )
                superseded = [f for f in existing_facts if "[superseded]" in f]
                # Keep the last max_facts*2 live facts (most recently added)
                existing_facts = live_facts[-(max_facts * 2):] + superseded
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
    if mention_dates is not None:
        frontmatter.mention_dates = mention_dates
    if monthly_buckets is not None:
        frontmatter.monthly_buckets = monthly_buckets

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
            line = format_observation(obs)
            sections["Facts"].append(line)

    if relations:
        sections["Relations"] = relations

    write_entity(filepath, frontmatter, sections)
    return filepath


def _stub_retention(entity_type: str) -> str:
    """Determine initial retention for stub entities."""
    if entity_type == "ai_self":
        return "permanent"
    if entity_type in ("person", "animal", "health"):
        return "long_term"
    return "short_term"


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
        retention=_stub_retention(entity_type),
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
        if not is_entity_file(rel.parts):
            continue
        try:
            fm, _ = read_entity(md_file)
            results.append({"path": md_file, "frontmatter": fm})
        except Exception as e:
            logger.debug("Could not read entity %s: %s", md_file.name, e)
            continue
    return results


def _dedup_facts_deterministic(facts: list[str], threshold: float = 0.85) -> list[str]:
    """Remove near-duplicate facts using sequence similarity.

    Delegates to nlp.dedup_facts_deterministic (rapidfuzz with difflib fallback).
    Threshold is 0..1 here (legacy interface) but nlp uses 0..100, so we convert.
    """
    from src.pipeline.nlp import dedup_facts_deterministic
    return dedup_facts_deterministic(facts, threshold=threshold * 100.0)


def consolidate_entity_facts(
    filepath: Path,
    config,
    max_facts: int | None = None,
) -> dict:
    """Consolidate redundant observations in an entity via LLM.

    First applies deterministic Levenshtein dedup. If that reduces facts
    below max_facts, skips the LLM call entirely.

    Returns a dict with 'original_count', 'consolidated_count', 'changes'.
    """
    from src.core.llm import call_fact_consolidation

    frontmatter, sections = read_entity(filepath)
    facts = sections.get("Facts", [])
    if not facts:
        return {"original_count": 0, "consolidated_count": 0, "changes": []}

    # Filter out already superseded facts — only consolidate live ones
    live_facts = filter_live_facts(facts)
    superseded_facts = [f for f in facts if "[superseded]" in f]

    if len(live_facts) < 3:
        return {"original_count": len(live_facts), "consolidated_count": len(live_facts), "changes": []}

    effective_max = max_facts if max_facts else 50

    # Phase 1: Deterministic dedup (no LLM)
    deduped_facts = _dedup_facts_deterministic(live_facts)

    # If deterministic dedup brought us under max_facts, skip the LLM call
    if len(deduped_facts) <= effective_max:
        logger.info(
            "Deterministic dedup sufficient for %s: %d → %d facts (max %d)",
            frontmatter.title, len(live_facts), len(deduped_facts), effective_max,
        )
        changes = []
        if len(deduped_facts) != len(live_facts):
            changes.append(f"{len(live_facts)} → {len(deduped_facts)} live facts (deterministic dedup)")
            sections["Facts"] = deduped_facts + superseded_facts

            # Log in History
            from datetime import date
            history = sections.get("History", [])
            history.append(f"- {date.today().isoformat()}: Facts deduped ({len(live_facts)} → {len(deduped_facts)}, deterministic)")
            sections["History"] = history

            write_entity(filepath, frontmatter, sections)

        return {
            "original_count": len(live_facts),
            "consolidated_count": len(deduped_facts),
            "changes": changes,
        }

    # Phase 2: Use deduped facts as input to LLM consolidation
    live_facts = deduped_facts

    # Build indexed text for LLM
    indexed_text = "\n".join(f"{i}: {f}" for i, f in enumerate(live_facts))

    result = call_fact_consolidation(
        frontmatter.title, frontmatter.type, indexed_text, config,
        max_facts=effective_max,
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
        new_facts.append(format_observation(obs))

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

    _atomic_write_text(filepath, "".join(lines))
    return filepath


def list_unprocessed_chats(memory_path: Path) -> list[Path]:
    """List chat files that haven't been processed yet."""
    chats_dir = memory_path / "chats"
    if not chats_dir.exists():
        return []

    unprocessed = []
    for md_file in sorted(chats_dir.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        fm_data, _ = _shared_parse_frontmatter(text)
        if not fm_data.get("processed", False):
            unprocessed.append(md_file)
    return unprocessed


def _update_chat_frontmatter(filepath: Path, updates: dict) -> dict:
    """Read chat frontmatter, apply updates, write back. Returns updated fm_data."""
    text = filepath.read_text(encoding="utf-8")
    fm_data, body = _shared_parse_frontmatter(text)
    fm_data.update(updates)
    fm_yaml = yaml.safe_dump(fm_data, default_flow_style=False, allow_unicode=True)
    _atomic_write_text(filepath, f"---\n{fm_yaml}---\n{body}")
    return fm_data


def mark_chat_processed(
    filepath: Path,
    entities_updated: list[str],
    entities_created: list[str],
) -> None:
    """Mark a chat file as processed by updating its frontmatter."""
    _update_chat_frontmatter(filepath, {
        "processed": True,
        "processed_at": datetime.now().isoformat(),
        "entities_updated": entities_updated,
        "entities_created": entities_created,
    })


def mark_chat_fallback(
    filepath: Path,
    fallback: str,
    error: str = "",
) -> None:
    """Mark a chat as processed via fallback (e.g. doc_ingest after extraction timeout).

    Sets processed=True plus fallback metadata so it is never retried for extraction.
    """
    updates = {
        "processed": True,
        "processed_at": datetime.now().isoformat(),
        "fallback": fallback,
        "entities_updated": [],
        "entities_created": [],
    }
    if error:
        updates["fallback_reason"] = error
    _update_chat_frontmatter(filepath, updates)


def increment_extraction_retries(filepath: Path) -> int:
    """Increment and return the extraction_retries counter in chat frontmatter."""
    text = filepath.read_text(encoding="utf-8")
    fm_data, _ = _shared_parse_frontmatter(text)
    retries = fm_data.get("extraction_retries", 0) + 1
    _update_chat_frontmatter(filepath, {"extraction_retries": retries})
    return retries


def get_chat_content(filepath: Path) -> str:
    """Read a chat file and return only the body content (no frontmatter)."""
    text = filepath.read_text(encoding="utf-8")
    _, body = _shared_parse_frontmatter(text)
    return body


# ── Private helpers ──────────────────────────────────────────

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


def format_observation(obs: dict) -> str:
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


def parse_observation(line: str) -> dict | None:
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

    # Extract valence marker (only at end of content, before optional #tags)
    valence = ""
    for marker, val in _VALENCE_REVERSE.items():
        pattern = r' ' + re.escape(marker) + r'(?=\s*(#|$))'
        if re.search(pattern, rest):
            valence = val
            rest = re.sub(pattern, '', rest, count=1).strip()
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
    new_obs = parse_observation(new_line)
    if not new_obs:
        return False
    new_cat, new_content = new_obs["category"], new_obs["content"].lower()

    for existing in existing_lines:
        ex_obs = parse_observation(existing)
        if not ex_obs:
            continue
        if ex_obs.get("superseded"):
            continue
        ex_cat, ex_content = ex_obs["category"], ex_obs["content"].lower()
        if new_cat == ex_cat and (new_content in ex_content or ex_content in new_content):
            return True
    return False


def remove_relation_line(entity_path: Path, relation_type: str, target_title: str) -> bool:
    """Remove a specific relation line from ## Relations section.

    Looks for lines matching '- {relation_type} [[{target_title}]]'.
    Returns True if a line was removed, False otherwise.
    """
    frontmatter, sections = read_entity(entity_path)
    relations = sections.get("Relations", [])
    if not relations:
        return False

    target_lower = target_title.lower()
    new_relations = []
    removed = False
    for line in relations:
        stripped = line.strip()
        if (stripped.startswith(f"- {relation_type} [[")
                and target_lower in stripped.lower()):
            removed = True
        else:
            new_relations.append(line)

    if removed:
        sections["Relations"] = new_relations
        write_entity(entity_path, frontmatter, sections)
    return removed


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
        obs = parse_observation(line)
        if (obs and not obs.get("superseded")
                and obs["category"] == category
                and supersedes_lower in obs["content"].lower()):
            obs["superseded"] = True
            result.append(format_observation(obs))
        else:
            result.append(line)
    return result
