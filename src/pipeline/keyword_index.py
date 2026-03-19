"""SQLite FTS5 keyword index for hybrid search."""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from src.core.utils import is_entity_file

logger = logging.getLogger(__name__)


@dataclass
class KeywordResult:
    entity_id: str
    chunk_idx: int
    bm25_score: float


def build_keyword_index(memory_path: Path, db_path: Path, *, chunk_size: int = 300, chunk_overlap: int = 60) -> int:
    """Build SQLite FTS5 index from entity MD files. Returns chunk count."""
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE VIRTUAL TABLE memory_fts USING fts5(
            entity_id,
            chunk_idx,
            content,
            tokenize='unicode61 remove_diacritics 2'
        )
    """)

    count = 0
    for md_file in sorted(memory_path.rglob("*.md")):
        rel = md_file.relative_to(memory_path)
        if not is_entity_file(rel.parts):
            continue

        entity_id = md_file.stem
        text = md_file.read_text(encoding="utf-8")

        # Extract title and facts from MD structure
        lines = text.split("\n")
        title = entity_id
        in_frontmatter = False
        facts_lines: list[str] = []
        in_facts = False
        for line in lines:
            if line.strip() == "---":
                in_frontmatter = not in_frontmatter
                continue
            if in_frontmatter and line.startswith("title:"):
                title = line.split(":", 1)[1].strip().strip('"')
            if line.startswith("## Facts"):
                in_facts = True
                continue
            if line.startswith("## ") and in_facts:
                in_facts = False
            if in_facts and line.strip():
                facts_lines.append(line.strip())

        full_text = f"{title}\n" + "\n".join(facts_lines)

        words = full_text.split()
        overlap = chunk_overlap
        i = 0
        chunk_idx = 0
        while i < len(words):
            chunk = " ".join(words[i : i + chunk_size])
            conn.execute(
                "INSERT INTO memory_fts (entity_id, chunk_idx, content) VALUES (?, ?, ?)",
                (entity_id, chunk_idx, chunk),
            )
            count += 1
            chunk_idx += 1
            i += chunk_size - overlap

        if not words:
            # Index at least the title for empty entities
            conn.execute(
                "INSERT INTO memory_fts (entity_id, chunk_idx, content) VALUES (?, ?, ?)",
                (entity_id, 0, title),
            )
            count += 1

    conn.commit()
    conn.close()
    logger.info("Built FTS5 keyword index: %d chunks in %s", count, db_path)
    return count


def search_keyword(query: str, db_path: Path, top_k: int = 10) -> list[KeywordResult]:
    """Search FTS5 index with BM25 ranking."""
    if not db_path.exists():
        return []

    conn = sqlite3.connect(str(db_path))
    try:
        # Try phrase match first (best for proper names like "Dr. Martin")
        safe_query = query.replace('"', '""')
        rows = conn.execute(
            """
            SELECT entity_id, chunk_idx, bm25(memory_fts) as score
            FROM memory_fts
            WHERE memory_fts MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (f'"{safe_query}"', top_k),
        ).fetchall()
    except sqlite3.OperationalError:
        # Fallback: try individual terms with OR
        try:
            terms = query.split()
            if terms:
                match_expr = " OR ".join(f'"{t.replace(chr(34), "")}"' for t in terms)
                rows = conn.execute(
                    """
                    SELECT entity_id, chunk_idx, bm25(memory_fts) as score
                    FROM memory_fts
                    WHERE memory_fts MATCH ?
                    ORDER BY score
                    LIMIT ?
                    """,
                    (match_expr, top_k),
                ).fetchall()
            else:
                rows = []
        except sqlite3.OperationalError:
            rows = []
    finally:
        conn.close()

    # bm25() returns negative scores (lower = better match), so negate
    return [
        KeywordResult(entity_id=row[0], chunk_idx=row[1], bm25_score=-row[2])
        for row in rows
    ]
