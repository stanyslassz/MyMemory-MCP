"""Process files dropped into _inbox/ folder."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

from src.core.config import Config
from src.memory.store import save_chat


def process_inbox(memory_path: Path, config: Config) -> list[str]:
    """Process files in _inbox/ → convert to chats → archive to _processed/.

    Supports .md and .txt files.
    Returns list of processed file names.
    """
    inbox_path = memory_path / "_inbox"
    processed_path = inbox_path / "_processed"
    processed_path.mkdir(parents=True, exist_ok=True)

    if not inbox_path.exists():
        return []

    processed_files = []

    for filepath in sorted(inbox_path.iterdir()):
        if filepath.is_dir():
            continue
        if filepath.suffix not in (".md", ".txt"):
            continue

        try:
            content = filepath.read_text(encoding="utf-8")
            if not content.strip():
                continue

            # Convert to pseudo-chat format
            messages = [
                {"role": "user", "content": content},
            ]

            # Save as a chat for pipeline processing
            save_chat(messages, memory_path)

            # Move to _processed/
            dest = processed_path / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filepath.name}"
            shutil.move(str(filepath), str(dest))
            processed_files.append(filepath.name)

        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Error processing inbox file {filepath}: {e}")

    return processed_files
