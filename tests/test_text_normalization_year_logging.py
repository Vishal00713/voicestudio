"""Unit tests verifying exception logging and fallbacks in year-to-words normalization."""
import logging
import sys
import types
import pytest


@pytest.mark.parametrize(
    "exc_type, raw_path, forbidden_prefix",
    [
        (NotImplementedError, "/Users/author/data/locale.json", "/Users/author"),
        (ValueError, r"C:\Users\author\data\locale.json", r"C:\Users\author"),
        (KeyError, "C:/Users/author/data/locale.json", "C:/Users/author"),
        (TypeError, "/Users/author/data/locale.json", "/Users/author"),
        (RuntimeError, "/Users/author/data/locale.json", "/Users/author"),
    ],
)
def test_year_conversion_exception_logs_debug_and_falls_back_to_cardinal(
    monkeypatch, caplog, exc_type, raw_path, forbidden_prefix
):
    orig_num2words = sys.modules.get("num2words")

    mock_mod = types.ModuleType("num2words")

    def _mock_num2words(n, lang="en", to=None):
        if to == "year":
            raise exc_type(f"year unsupported at {raw_path}")
        return f"cardinal_{n}"

    mock_mod.num2words = _mock_num2words
    monkeypatch.setitem(sys.modules, "num2words", mock_mod)

    try:
        from services import text_normalization

        with caplog.at_level(logging.DEBUG, logger="omnivoice.text_normalization"):
            res = text_normalization._numbers_to_words("In 1984 George", lang="sv")

        assert res == "In cardinal_1984 George"

        assert any(
            "Year conversion unavailable for lang=sv; falling back to cardinal" in r.message
            for r in caplog.records
        )
        assert forbidden_prefix not in caplog.text
        assert "~" in caplog.text
    finally:
        if orig_num2words is not None:
            sys.modules["num2words"] = orig_num2words
        else:
            sys.modules.pop("num2words", None)


@pytest.mark.parametrize(
    "raw_path, forbidden_prefix",
    [
        ("/Users/author/data/faulty.json", "/Users/author"),
        (r"C:\Users\author\data\faulty.json", r"C:\Users\author"),
        ("C:/Users/author/data/faulty.json", "C:/Users/author"),
    ],
)
def test_verbalizer_safe_outer_catch_logs_debug_and_preserves_token(
    monkeypatch, caplog, raw_path, forbidden_prefix
):
    orig_num2words = sys.modules.get("num2words")

    mock_mod = types.ModuleType("num2words")

    def _always_fails(n, lang="en", to=None):
        raise RuntimeError(f"catastrophic failure at {raw_path}")

    mock_mod.num2words = _always_fails
    monkeypatch.setitem(sys.modules, "num2words", mock_mod)

    try:
        from services import text_normalization

        with caplog.at_level(logging.DEBUG, logger="omnivoice.text_normalization"):
            # When both year and cardinal fail, token must be preserved untouched
            res = text_normalization._numbers_to_words("In 1984 George", lang="en")

        assert res == "In 1984 George"
        assert any(
            "num2words verbalizer exception for '1984' (leaving untouched)" in r.message
            for r in caplog.records
        )
        assert forbidden_prefix not in caplog.text
        assert "~" in caplog.text
    finally:
        if orig_num2words is not None:
            sys.modules["num2words"] = orig_num2words
        else:
            sys.modules.pop("num2words", None)

