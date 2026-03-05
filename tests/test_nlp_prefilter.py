"""Tests for NLP pre-filter (works whether or not spaCy is installed)."""

from src.pipeline.nlp_prefilter import is_available, _try_parse_french_date


def test_is_available_returns_bool():
    """Should return True or False, never raise."""
    result = is_available()
    assert isinstance(result, bool)


def test_parse_french_date_hier():
    result = _try_parse_french_date("hier", "2026-03-05")
    assert result == "2026-03-04"


def test_parse_french_date_il_y_a_jours():
    result = _try_parse_french_date("il y a 3 jours", "2026-03-05")
    assert result == "2026-03-02"


def test_parse_french_date_il_y_a_semaines():
    result = _try_parse_french_date("il y a 2 semaines", "2026-03-05")
    assert result == "2026-02-19"


def test_parse_french_date_iso():
    result = _try_parse_french_date("2026-01-15", "2026-03-05")
    assert result == "2026-01-15"


def test_parse_french_date_unknown():
    result = _try_parse_french_date("mardi prochain", "2026-03-05")
    assert result is None
