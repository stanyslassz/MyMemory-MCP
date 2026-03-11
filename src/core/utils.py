"""Shared utility functions for memory-ai."""

from __future__ import annotations

import re
import unicodedata

import yaml


def slugify(text: str) -> str:
    """Convert a title to a slug (lowercase, hyphens, ASCII)."""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s-]", "", text.lower())
    text = re.sub(r"[-\s]+", "-", text).strip("-")
    return text


def estimate_tokens(text: str) -> int:
    """Rough token estimate: words * 1.3."""
    return int(len(text.split()) * 1.3)


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from a markdown file. Returns (fm_dict, body)."""
    text = text.replace("\r\n", "\n")
    match = re.match(r"^---\n(.*?\n)---\n(.*)", text, re.DOTALL)
    if not match:
        return {}, text
    fm_text = match.group(1)
    body = match.group(2)
    fm_data = yaml.safe_load(fm_text) or {}
    return fm_data, body
