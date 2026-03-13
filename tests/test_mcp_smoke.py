"""MCP smoke tests: verify the 3 tools return expected shapes."""

from pathlib import Path
from unittest.mock import patch

from src.core.config import load_config
from src.core.models import SearchResult
from src.memory.store import init_memory_structure, save_chat


def _make_config(tmp_path):
    config = load_config(project_root=Path(__file__).parent.parent)
    config.memory_path = tmp_path
    config.faiss.index_path = str(tmp_path / "_memory.faiss")
    config.faiss.mapping_path = str(tmp_path / "_memory.pkl")
    config.faiss.manifest_path = str(tmp_path / "_faiss_manifest.json")
    return config


def test_get_context_fallback_no_files(tmp_path):
    """get_context returns fallback string when no context/index files exist."""
    config = _make_config(tmp_path)
    init_memory_structure(tmp_path)

    with patch("src.mcp.server._get_config", return_value=config):
        from src.mcp.server import get_context
        result = get_context()
    assert isinstance(result, str)
    assert "No memory context" in result or len(result) > 0


def test_get_context_reads_context_md(tmp_path):
    """get_context returns _context.md content when present."""
    config = _make_config(tmp_path)
    init_memory_structure(tmp_path)
    (tmp_path / "_context.md").write_text("# Test Context\nHello", encoding="utf-8")

    with patch("src.mcp.server._get_config", return_value=config):
        from src.mcp.server import get_context
        result = get_context()
    assert "Test Context" in result


def test_get_context_falls_back_to_index(tmp_path):
    """get_context falls back to _index.md when _context.md absent."""
    config = _make_config(tmp_path)
    init_memory_structure(tmp_path)
    (tmp_path / "_index.md").write_text("# Index\nEntities here", encoding="utf-8")

    with patch("src.mcp.server._get_config", return_value=config):
        from src.mcp.server import get_context
        result = get_context()
    assert "Index" in result


def test_save_chat_returns_status(tmp_path):
    """save_chat returns dict with status and file keys."""
    config = _make_config(tmp_path)
    init_memory_structure(tmp_path)

    messages = [
        {"role": "user", "content": "Bonjour"},
        {"role": "assistant", "content": "Salut"},
    ]

    with patch("src.mcp.server._get_config", return_value=config):
        from src.mcp.server import save_chat as mcp_save_chat
        result = mcp_save_chat(messages)

    assert result["status"] == "saved"
    assert "file" in result
    assert result["file"].startswith("chats/")


def test_search_rag_returns_structure(tmp_path):
    """search_rag returns dict with query, results, total."""
    config = _make_config(tmp_path)
    init_memory_structure(tmp_path)

    mock_results = [
        SearchResult(entity_id="test-entity", file="moi/test-entity.md", chunk="[chunk 0]", score=0.9),
    ]

    with patch("src.mcp.server._get_config", return_value=config), \
         patch("src.memory.rag.search", return_value=mock_results):
        from src.mcp.server import search_rag
        result = search_rag("test query")

    assert result["query"] == "test query"
    assert result["total"] == 1
    assert isinstance(result["results"], list)
    assert result["results"][0]["entity_id"] == "test-entity"


def test_search_rag_empty_index(tmp_path):
    """search_rag with no results returns empty list."""
    config = _make_config(tmp_path)
    init_memory_structure(tmp_path)

    with patch("src.mcp.server._get_config", return_value=config), \
         patch("src.memory.rag.search", return_value=[]):
        from src.mcp.server import search_rag
        result = search_rag("nonexistent")

    assert result["total"] == 0
    assert result["results"] == []
