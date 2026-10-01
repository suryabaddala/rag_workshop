"""
Unit tests for TextCleaner module.
Validates text normalization, hyphenation resolution, ligature fixing,
control character removal, and question cleaning.
"""

import pytest
from core.text_cleaner import TextCleaner


class TestTextCleaner:
    def test_clean_empty_and_none(self):
        assert TextCleaner.clean(None) == ""
        assert TextCleaner.clean("") == ""
        assert TextCleaner.clean("   \n\t  ") == ""

    def test_excess_whitespace_and_newlines(self):
        raw = "Line 1   with    extra    spaces.\n\n\n\n\nLine 2 after massive gaps."
        cleaned = TextCleaner.clean(raw)
        assert "    " not in cleaned
        assert "\n\n\n" not in cleaned
        assert "Line 1 with extra spaces.\n\nLine 2 after massive gaps." == cleaned

    def test_hyphenated_line_breaks(self):
        raw = "This is a demon-\nstration of hyphen-\n  ated word recon-\nstruction."
        cleaned = TextCleaner.clean(raw)
        assert "demonstration" in cleaned
        assert "hyphenated" in cleaned
        assert "reconstruction" in cleaned

    def test_non_printable_control_characters(self):
        # Including null bytes \x00 and backspace \x08
        raw = "Valid text with \x00 null and \x08 backspace."
        cleaned = TextCleaner.clean(raw)
        assert "\x00" not in cleaned
        assert "\x08" not in cleaned
        assert "Valid text with null and backspace." in cleaned

    def test_ligatures_and_smart_quotes(self):
        raw = "“Artificial Intelligence” with ﬁle and ﬂow."
        cleaned = TextCleaner.clean(raw)
        assert '"Artificial Intelligence"' in cleaned
        assert "file" in cleaned
        assert "flow" in cleaned

    def test_preserves_structure_and_punctuation(self):
        raw = "Key topics:\n1. Cosine similarity: (u · v) / (||u|| * ||v||)\n2. Top-K retrieval >= 0.30"
        cleaned = TextCleaner.clean(raw)
        assert "Cosine similarity: (u · v) / (||u|| * ||v||)" in cleaned
        assert "Top-K retrieval >= 0.30" in cleaned

    def test_clean_query(self):
        query = "   What   is the pass   percentage?\x00   "
        assert TextCleaner.clean_query(query) == "What is the pass percentage?"
        assert TextCleaner.clean_query("") == ""
        assert TextCleaner.clean_query(None) == ""
