"""Tests for sentence-aware chunking with spaCy fallback."""

from unittest.mock import patch, MagicMock
from src.pipeline.indexer import chunk_text, _get_spacy_nlp, _spacy_nlp_cache, SPACY_MODELS


def setup_function():
    """Clear the spaCy cache before each test."""
    _spacy_nlp_cache.clear()


def test_chunk_text_with_language_param():
    """chunk_text should accept a language parameter."""
    import src.pipeline.indexer as idx_mod
    _spacy_nlp_cache.clear()
    with patch.object(idx_mod, "_get_spacy_nlp", return_value=None):
        result = chunk_text("Hello world. This is a test.", language="en")
        assert len(result) >= 1


def test_chunk_text_fallback_regex_when_no_spacy():
    """When spaCy is not installed, chunk_text should use regex fallback."""
    import src.pipeline.indexer as idx_mod

    _spacy_nlp_cache.clear()
    with patch.object(idx_mod, "_get_spacy_nlp", return_value=None):
        text = "First sentence. Second sentence. Third sentence."
        result = chunk_text(text, chunk_size=400, language="fr")
        assert len(result) >= 1
        assert "First sentence" in result[0]


def test_spacy_model_mapping():
    """SPACY_MODELS should map language codes to model names."""
    assert SPACY_MODELS["fr"] == "fr_core_news_sm"
    assert SPACY_MODELS["en"] == "en_core_web_sm"
    assert SPACY_MODELS["zh"] == "zh_core_web_sm"


def test_get_spacy_nlp_caches_model():
    """Model should be loaded once and cached."""
    _spacy_nlp_cache.clear()
    mock_nlp = MagicMock()
    mock_spacy = MagicMock()
    mock_spacy.load.return_value = mock_nlp

    with patch.dict("sys.modules", {"spacy": mock_spacy, "spacy.cli": MagicMock()}):
        result1 = _get_spacy_nlp("en")
        result2 = _get_spacy_nlp("en")

    assert mock_spacy.load.call_count == 1
    assert result1 is result2


def test_get_spacy_nlp_auto_downloads_on_oserror():
    """If model not found, should auto-download then retry load."""
    _spacy_nlp_cache.clear()
    mock_nlp = MagicMock()
    mock_spacy = MagicMock()
    mock_cli = MagicMock()
    mock_spacy.cli = mock_cli

    mock_spacy.load.side_effect = [OSError("Model not found"), mock_nlp]

    with patch.dict("sys.modules", {"spacy": mock_spacy, "spacy.cli": mock_cli}):
        result = _get_spacy_nlp("fr")

    mock_cli.download.assert_called_once_with("fr_core_news_sm")
    assert result is mock_nlp


def test_get_spacy_nlp_returns_none_on_network_failure():
    """If auto-download fails, should return None (fallback to regex)."""
    _spacy_nlp_cache.clear()
    mock_spacy = MagicMock()
    mock_cli = MagicMock()
    mock_spacy.cli = mock_cli
    mock_spacy.load.side_effect = OSError("Model not found")
    mock_cli.download.side_effect = Exception("Network error")

    with patch.dict("sys.modules", {"spacy": mock_spacy, "spacy.cli": mock_cli}):
        result = _get_spacy_nlp("fr")

    assert result is None


def test_chunk_text_sentence_boundaries():
    """With spaCy available, chunks should break on sentence boundaries."""
    import src.pipeline.indexer as idx_mod
    _spacy_nlp_cache.clear()

    class MockSpan:
        def __init__(self, text):
            self.text = text

    class MockDoc:
        def __init__(self, text):
            parts = text.replace(". ", ".\n").split("\n")
            self._sents = [MockSpan(p.strip()) for p in parts if p.strip()]

        @property
        def sents(self):
            return iter(self._sents)

    mock_nlp = MagicMock(side_effect=MockDoc)

    with patch.object(idx_mod, "_get_spacy_nlp", return_value=mock_nlp):
        text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence is longer to fill chunk."
        result = chunk_text(text, chunk_size=20, overlap=5, language="en")
        assert len(result) >= 2
        for chunk in result:
            assert chunk.endswith(".") or chunk == result[-1]
