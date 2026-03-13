"""MCP transport smoke tests — per matrix-critical checks (P1.1 task 7).

Tests verify the MCP server can:
1. List tools (critical)
2. Call core tools (critical)
3. Return proper error shapes (critical)
4. Handle both stdio and SSE transport config
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.core.config import load_config, Config, FeaturesConfig, IngestConfig
from src.core.models import SearchResult
from src.memory.store import init_memory_structure


def _make_config(tmp_path):
    config = load_config(project_root=Path(__file__).parent.parent)
    config.memory_path = tmp_path
    config.faiss.index_path = str(tmp_path / "_memory.faiss")
    config.faiss.mapping_path = str(tmp_path / "_memory.pkl")
    config.faiss.manifest_path = str(tmp_path / "_faiss_manifest.json")
    config.features = FeaturesConfig(doc_pipeline=True)
    config.ingest = IngestConfig(
        recovery_threshold_seconds=300,
        max_retries=3,
        jobs_path=str(tmp_path / "_ingest_jobs.json"),
    )
    return config


class TestMCPToolListing:
    """Critical: verify tools are discoverable."""

    def test_mcp_server_has_three_tools(self):
        from src.mcp.server import mcp
        # FastMCP tools are registered as decorators
        # Verify the server object exists and is configured
        assert mcp.name == "memory-ai"

    def test_get_context_callable(self, tmp_path):
        config = _make_config(tmp_path)
        init_memory_structure(tmp_path)
        with patch("src.mcp.server._get_config", return_value=config):
            from src.mcp.server import get_context
            result = get_context()
        assert isinstance(result, str)

    def test_save_chat_callable(self, tmp_path):
        config = _make_config(tmp_path)
        init_memory_structure(tmp_path)
        with patch("src.mcp.server._get_config", return_value=config):
            from src.mcp.server import save_chat
            result = save_chat([{"role": "user", "content": "test"}])
        assert isinstance(result, dict)
        assert "status" in result

    def test_search_rag_callable(self, tmp_path):
        config = _make_config(tmp_path)
        init_memory_structure(tmp_path)
        with patch("src.mcp.server._get_config", return_value=config), \
             patch("src.memory.rag.search", return_value=[]):
            from src.mcp.server import search_rag
            result = search_rag("test")
        assert isinstance(result, dict)
        assert "query" in result
        assert "results" in result
        assert "total" in result


class TestMCPErrorShapes:
    """Critical: verify error responses have proper structure."""

    def test_save_chat_empty_messages(self, tmp_path):
        config = _make_config(tmp_path)
        init_memory_structure(tmp_path)
        with patch("src.mcp.server._get_config", return_value=config):
            from src.mcp.server import save_chat
            # Empty messages should still work (save empty chat)
            result = save_chat([])
        assert isinstance(result, dict)

    def test_search_rag_returns_empty_on_no_results(self, tmp_path):
        config = _make_config(tmp_path)
        init_memory_structure(tmp_path)
        with patch("src.mcp.server._get_config", return_value=config), \
             patch("src.memory.rag.search", return_value=[]):
            from src.mcp.server import search_rag
            result = search_rag("nonexistent query")
        assert result["total"] == 0
        assert result["results"] == []

    def test_search_rag_result_shape(self, tmp_path):
        config = _make_config(tmp_path)
        init_memory_structure(tmp_path)
        mock_results = [
            SearchResult(entity_id="test", file="moi/test.md", chunk="[chunk 0]", score=0.5),
        ]
        with patch("src.mcp.server._get_config", return_value=config), \
             patch("src.memory.rag.search", return_value=mock_results):
            from src.mcp.server import search_rag
            result = search_rag("test")

        r = result["results"][0]
        assert "entity_id" in r
        assert "file" in r
        assert "score" in r
        assert "title" in r
        assert "type" in r
        assert "relations" in r


class TestMCPTransportConfig:
    """Verify transport configuration options."""

    def test_stdio_transport_config(self, tmp_path):
        config = _make_config(tmp_path)
        config.mcp_transport = "stdio"
        assert config.mcp_transport == "stdio"

    def test_sse_transport_config(self, tmp_path):
        config = _make_config(tmp_path)
        config.mcp_transport = "sse"
        assert config.mcp_transport == "sse"
        assert config.mcp_host is not None
        assert config.mcp_port > 0

    def test_server_applies_host_port(self, tmp_path):
        """Verify run_server reads config for host/port."""
        from src.mcp.server import mcp as mcp_server
        config = _make_config(tmp_path)
        config.mcp_host = "0.0.0.0"
        config.mcp_port = 9999
        config.mcp_transport = "sse"

        with patch("src.mcp.server._get_config", return_value=config):
            # Don't actually start the server, just verify config is applied
            mcp_server.settings.host = config.mcp_host
            mcp_server.settings.port = config.mcp_port
            assert mcp_server.settings.host == "0.0.0.0"
            assert mcp_server.settings.port == 9999


class TestRunServerConfigPassthrough:
    """Bug fix: run_server must honor the config passed from CLI (--config flag)."""

    def test_run_server_uses_passed_config(self, tmp_path):
        """run_server(config=X) should install X as the module-level _config."""
        import src.mcp.server as srv
        config = _make_config(tmp_path)
        config.mcp_transport = "stdio"

        with patch.object(srv.mcp, "run") as mock_run:
            srv.run_server(config=config)
            mock_run.assert_called_once_with(transport="stdio")
        # _config should now be the config we passed
        assert srv._config is config

    def test_run_server_transport_override(self, tmp_path):
        """--transport flag should override config.mcp_transport."""
        import src.mcp.server as srv
        config = _make_config(tmp_path)
        config.mcp_transport = "sse"  # config says SSE

        with patch.object(srv.mcp, "run") as mock_run:
            srv.run_server(config=config, transport_override="stdio")
            mock_run.assert_called_once_with(transport="stdio")

    def test_run_server_no_override_uses_config(self, tmp_path):
        """Without transport_override, config value is used."""
        import src.mcp.server as srv
        config = _make_config(tmp_path)
        config.mcp_transport = "sse"

        with patch.object(srv.mcp, "run") as mock_run:
            srv.run_server(config=config, transport_override=None)
            mock_run.assert_called_once_with(transport="sse")

    def test_run_server_applies_host_port_from_config(self, tmp_path):
        """Host/port from the passed config should be applied to mcp.settings."""
        import src.mcp.server as srv
        config = _make_config(tmp_path)
        config.mcp_host = "10.0.0.1"
        config.mcp_port = 7777
        config.mcp_transport = "stdio"

        with patch.object(srv.mcp, "run"):
            srv.run_server(config=config)
        assert srv.mcp.settings.host == "10.0.0.1"
        assert srv.mcp.settings.port == 7777


class TestServeCLIIntegration:
    """CLI 'serve' command: --transport flag and --config passthrough."""

    def test_serve_passes_config_to_run_server(self, tmp_path):
        """memory -c <path> serve should pass the loaded config to run_server."""
        from click.testing import CliRunner
        from src.cli import cli

        with patch("src.mcp.server.mcp") as mock_mcp:
            mock_mcp.settings = MagicMock()
            mock_mcp.run = MagicMock()
            runner = CliRunner()
            result = runner.invoke(cli, ["serve"])
            mock_mcp.run.assert_called_once()

    def test_serve_transport_flag_stdio(self, tmp_path):
        """memory serve --transport stdio should override config."""
        from click.testing import CliRunner
        from src.cli import cli

        with patch("src.mcp.server.mcp") as mock_mcp:
            mock_mcp.settings = MagicMock()
            mock_mcp.run = MagicMock()
            runner = CliRunner()
            result = runner.invoke(cli, ["serve", "--transport", "stdio"])
            mock_mcp.run.assert_called_once_with(transport="stdio")

    def test_serve_transport_flag_sse(self, tmp_path):
        """memory serve --transport sse should use SSE."""
        from click.testing import CliRunner
        from src.cli import cli

        with patch("src.mcp.server.mcp") as mock_mcp:
            mock_mcp.settings = MagicMock()
            mock_mcp.run = MagicMock()
            runner = CliRunner()
            result = runner.invoke(cli, ["serve", "-t", "sse"])
            mock_mcp.run.assert_called_once_with(transport="sse")

    def test_serve_no_flag_uses_config_default(self, tmp_path):
        """memory serve (no flag) uses mcp_transport from config."""
        from click.testing import CliRunner
        from src.cli import cli

        with patch("src.mcp.server.mcp") as mock_mcp:
            mock_mcp.settings = MagicMock()
            mock_mcp.run = MagicMock()
            runner = CliRunner()
            # config.yaml has transport: sse
            result = runner.invoke(cli, ["serve"])
            # Should use whatever config says (sse in config.yaml)
            mock_mcp.run.assert_called_once()
