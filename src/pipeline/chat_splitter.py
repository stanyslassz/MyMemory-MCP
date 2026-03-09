"""Split multi-conversation JSON exports (Claude, ChatGPT) into individual chats.

Supports:
- Claude.ai official export (conversations array with chat_messages)
- ChatGPT export (array of conversations with mapping/messages)
- Generic JSON arrays with role/content messages
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def split_export_json(filepath: Path, memory_path: Path) -> list[Path]:
    """Detect JSON export format, split into individual chat files.

    Returns list of saved chat file paths.
    """
    data = json.loads(filepath.read_text(encoding="utf-8"))

    # Try each format parser in order
    conversations = (
        _parse_claude_export(data)
        or _parse_chatgpt_export(data)
        or _parse_generic_json_array(data)
    )

    if not conversations:
        logger.warning("Could not parse JSON export: %s", filepath.name)
        return []

    from src.memory.store import save_chat

    saved = []
    for conv in conversations:
        messages = conv.get("messages", [])
        if not messages:
            continue

        path = save_chat(messages, memory_path)

        # Overwrite frontmatter with export metadata if available
        if conv.get("date") or conv.get("title"):
            _patch_chat_frontmatter(path, conv)

        saved.append(path)

    logger.info("Split %s into %d conversation(s)", filepath.name, len(saved))
    return saved


def _parse_claude_export(data) -> list[dict] | None:
    """Parse Claude.ai official export format.

    Claude exports conversations with chat_messages containing sender/text fields.
    Format variants:
    - Top-level array of conversations
    - Object with conversations key
    """
    conversations_list = None

    if isinstance(data, list) and data:
        # Check if first item looks like a Claude conversation
        first = data[0]
        if isinstance(first, dict) and ("chat_messages" in first or "uuid" in first):
            conversations_list = data

    if isinstance(data, dict):
        for key in ("conversations", "chats", "data"):
            if key in data and isinstance(data[key], list):
                if data[key] and isinstance(data[key][0], dict):
                    if "chat_messages" in data[key][0] or "uuid" in data[key][0]:
                        conversations_list = data[key]
                        break

    if not conversations_list:
        return None

    result = []
    for conv in conversations_list:
        if not isinstance(conv, dict):
            continue

        messages = []
        chat_messages = conv.get("chat_messages", conv.get("messages", []))

        for msg in chat_messages:
            if not isinstance(msg, dict):
                continue

            # Claude format: sender="human"/"assistant", text="..."
            sender = msg.get("sender", msg.get("role", ""))
            text = msg.get("text", msg.get("content", ""))

            if not text or not sender:
                continue

            # Normalize role
            role = "user" if sender in ("human", "user", "User") else "assistant"
            messages.append({"role": role, "content": text})

        if messages:
            title = conv.get("name", conv.get("title", ""))
            created = conv.get("created_at", conv.get("created", ""))
            date_str = ""
            if created:
                try:
                    dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                    date_str = dt.strftime("%Y-%m-%d")
                except (ValueError, TypeError):
                    pass

            result.append({
                "messages": messages,
                "title": title,
                "date": date_str,
            })

    return result if result else None


def _parse_chatgpt_export(data) -> list[dict] | None:
    """Parse ChatGPT export format.

    ChatGPT uses a mapping dict with message nodes forming a tree.
    """
    conversations_list = None

    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict) and "mapping" in first:
            conversations_list = data

    if not conversations_list:
        return None

    result = []
    for conv in conversations_list:
        if not isinstance(conv, dict):
            continue

        mapping = conv.get("mapping", {})
        if not mapping:
            continue

        # Flatten the message tree: follow current_node chain
        messages_by_create = []
        for node_id, node in mapping.items():
            msg = node.get("message")
            if not msg or not isinstance(msg, dict):
                continue

            author = msg.get("author", {})
            role = author.get("role", "") if isinstance(author, dict) else ""
            if role not in ("user", "assistant"):
                continue

            parts = msg.get("content", {}).get("parts", [])
            text = "\n".join(str(p) for p in parts if isinstance(p, str))
            if not text.strip():
                continue

            create_time = msg.get("create_time", 0) or 0
            messages_by_create.append((create_time, {"role": role, "content": text}))

        # Sort by creation time
        messages_by_create.sort(key=lambda x: x[0])
        messages = [m for _, m in messages_by_create]

        if messages:
            title = conv.get("title", "")
            create_time = conv.get("create_time", 0)
            date_str = ""
            if create_time:
                try:
                    dt = datetime.fromtimestamp(create_time)
                    date_str = dt.strftime("%Y-%m-%d")
                except (ValueError, TypeError, OSError):
                    pass

            result.append({
                "messages": messages,
                "title": title,
                "date": date_str,
            })

    return result if result else None


def _parse_generic_json_array(data) -> list[dict] | None:
    """Parse a simple JSON array of role/content messages as a single conversation."""
    if not isinstance(data, list) or not data:
        return None

    first = data[0]
    if isinstance(first, dict) and "role" in first and "content" in first:
        messages = [
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in data
            if isinstance(m, dict) and m.get("content")
        ]
        if messages:
            return [{"messages": messages, "title": "", "date": ""}]

    return None


def _patch_chat_frontmatter(path: Path, conv: dict) -> None:
    """Update chat file frontmatter with export metadata."""
    import yaml

    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return

    end = text.find("---", 3)
    if end < 0:
        return

    fm_text = text[3:end]
    fm = yaml.safe_load(fm_text) or {}

    if conv.get("date"):
        fm["date"] = conv["date"]
    if conv.get("title"):
        fm["source_title"] = conv["title"]
    fm["source"] = "import"

    new_fm = yaml.safe_dump(fm, default_flow_style=False, allow_unicode=True)
    new_text = "---\n" + new_fm + "---" + text[end + 3:]
    path.write_text(new_text, encoding="utf-8")
