"""CLI tests using Click's CliRunner."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
from click.testing import CliRunner

from src.cli import cli
from src.core.config import load_config
from src.core.models import (
    RawEntity,
    RawExtraction,
    RawObservation,
)
from src.memory.store import init_memory_structure, save_chat


def _make_config(tmp_path):
    config = load_config(project_root=Path(__file__).parent.parent)
    config.memory_path = tmp_path
    config.faiss.index_path = str(tmp_path / "_memory.faiss")
    config.faiss.mapping_path = str(tmp_path / "_memory.pkl")
    config.faiss.manifest_path = str(tmp_path / "_faiss_manifest.json")
    return config


def _mock_embed(texts):
    np.random.seed(42)
    vecs = np.random.randn(len(texts), 384).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def _invoke(tmp_path, args, config=None):
    """Invoke CLI with config pointed to tmp_path."""
    if config is None:
        config = _make_config(tmp_path)
    runner = CliRunner()
    with patch("src.cli.load_config", return_value=config):
        return runner.invoke(cli, args, catch_exceptions=False)


def test_cli_run_no_pending(tmp_path):
    """'run' with no pending chats prints a message and exits 0."""
    init_memory_structure(tmp_path)
    result = _invoke(tmp_path, ["run"])
    assert result.exit_code == 0
    assert "No pending" in result.output


def test_cli_run_with_chat(tmp_path):
    """'run' processes a pending chat through the pipeline."""
    config = _make_config(tmp_path)
    init_memory_structure(tmp_path)
    save_chat([{"role": "user", "content": "J'aime le vélo"}], tmp_path)

    mock_extraction = RawExtraction(
        entities=[
            RawEntity(name="Vélo", type="interet", observations=[
                RawObservation(category="fait", content="Aime le vélo", importance=0.5),
            ]),
        ],
        relations=[],
        summary="Aime le vélo",
    )

    with patch("src.pipeline.extractor.call_extraction", return_value=mock_extraction), \
         patch("src.pipeline.indexer._get_embedding_fn", return_value=_mock_embed):
        result = _invoke(tmp_path, ["run"], config)

    assert result.exit_code == 0
    assert "Processing" in result.output
    assert "Pipeline complete" in result.output


def test_cli_rebuild_graph(tmp_path):
    """'rebuild-graph' runs without error on empty memory."""
    init_memory_structure(tmp_path)
    result = _invoke(tmp_path, ["rebuild-graph"])
    assert result.exit_code == 0
    assert "rebuilt" in result.output.lower()


def test_cli_rebuild_faiss(tmp_path):
    """'rebuild-faiss' builds an empty index on empty memory."""
    init_memory_structure(tmp_path)
    with patch("src.pipeline.indexer._get_embedding_fn", return_value=_mock_embed):
        result = _invoke(tmp_path, ["rebuild-faiss"])
    assert result.exit_code == 0
    assert "rebuilt" in result.output.lower()


def test_cli_rebuild_all(tmp_path):
    """'rebuild-all' rebuilds graph + context + FAISS."""
    init_memory_structure(tmp_path)
    with patch("src.pipeline.indexer._get_embedding_fn", return_value=_mock_embed):
        result = _invoke(tmp_path, ["rebuild-all"])
    assert result.exit_code == 0
    assert "rebuild complete" in result.output.lower()


def test_cli_validate_clean(tmp_path):
    """'validate' reports no warnings on fresh memory."""
    init_memory_structure(tmp_path)
    result = _invoke(tmp_path, ["validate"])
    assert result.exit_code == 0
    assert "consistent" in result.output.lower()


def test_cli_stats(tmp_path):
    """'stats' displays a table without error."""
    init_memory_structure(tmp_path)
    result = _invoke(tmp_path, ["stats"])
    assert result.exit_code == 0
    assert "Total entities" in result.output or "Pending chats" in result.output


def test_cli_inbox_empty(tmp_path):
    """'inbox' with no inbox files."""
    init_memory_structure(tmp_path)
    result = _invoke(tmp_path, ["inbox"])
    assert result.exit_code == 0
    assert "No files" in result.output


def test_cli_inbox_with_file(tmp_path):
    """'inbox' processes a dropped file."""
    init_memory_structure(tmp_path)
    inbox = tmp_path / "_inbox"
    inbox.mkdir(exist_ok=True)
    (inbox / "note.md").write_text("Une note importante.", encoding="utf-8")

    result = _invoke(tmp_path, ["inbox"])
    assert result.exit_code == 0
    assert "Processed" in result.output


def test_cli_serve_help(tmp_path):
    """'serve --help' shows help text without starting the server."""
    init_memory_structure(tmp_path)
    result = _invoke(tmp_path, ["serve", "--help"])
    assert result.exit_code == 0
    assert "MCP server" in result.output
