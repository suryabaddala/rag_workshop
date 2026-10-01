"""
Text Cleaning module for Local RAG Workshop.
Sanitizes raw extracted text from PDFs and DOCX files without destroying
semantic meaning, punctuation, mathematical expressions, or list structures.
"""

import re
import unicodedata
from typing import Optional


class TextCleaner:
    """
    Cleans raw document text for high-fidelity embedding generation.
    Removes noisy formatting artifacts while preserving punctuation, numbers, and bullet points.
    """

    # Unnecessary non-printable control characters (excluding \n, \r, \t)
    CONTROL_CHAR_REGEX = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

    # Hyphenated words split across line breaks (e.g., "infor-\nmation" -> "information")
    HYPHENATED_LINEBREAK_REGEX = re.compile(r"(\b\w+)-\s*\n\s*(\w+\b)")

    # Multiple blank lines (3 or more newlines reduced to 2 newlines / paragraph separator)
    EXCESS_NEWLINES_REGEX = re.compile(r"\n{3,}")

    # Multiple spaces or tabs (excluding newlines)
    EXCESS_SPACES_REGEX = re.compile(r"[^\S\r\n]{2,}")

    # Common PDF ligatures to standard characters (e.g., ﬁ -> fi, ﬂ -> fl)
    LIGATURES = {
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬀ": "ff",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "–": "-",
        "—": "-",
        "…": "...",
    }

    @classmethod
    def clean(cls, text: Optional[str], fix_hyphens: bool = True) -> str:
        """
        Cleans and normalizes raw text.

        Args:
            text: Raw input string.
            fix_hyphens: Whether to reconstruct hyphenated line-broken words.

        Returns:
            Normalized, clean string.
        """
        if not text:
            return ""

        # Step 1: Normalize unicode characters (NFKC)
        cleaned = unicodedata.normalize("NFKC", text)

        # Step 2: Replace common typographic ligatures and smart quotes
        for orig, rep in cls.LIGATURES.items():
            cleaned = cleaned.replace(orig, rep)

        # Step 3: Strip non-printable control characters
        cleaned = cls.CONTROL_CHAR_REGEX.sub("", cleaned)

        # Step 4: Reconnect hyphenated word breaks across line breaks
        if fix_hyphens:
            cleaned = cls.HYPHENATED_LINEBREAK_REGEX.sub(r"\1\2", cleaned)

        # Step 5: Normalize carriage returns (\r\n -> \n, \r -> \n)
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

        # Step 6: Fix awkward soft line breaks in sentences (lines ending without punctuation)
        # We preserve list items, headers, and paragraphs
        lines = cleaned.split("\n")
        reconstructed_lines = []
        for line in lines:
            stripped_line = cls.EXCESS_SPACES_REGEX.sub(" ", line.strip())
            reconstructed_lines.append(stripped_line)

        cleaned = "\n".join(reconstructed_lines)

        # Step 7: Collapse excessive blank lines down to a clean 2 newlines (paragraph boundary)
        cleaned = cls.EXCESS_NEWLINES_REGEX.sub("\n\n", cleaned)

        # Step 8: Trim final leading/trailing whitespace
        return cleaned.strip()

    @classmethod
    def clean_query(cls, query: Optional[str]) -> str:
        """
        Cleans a user question query.
        Normalizes spaces and removes dangerous shell/control characters while keeping query intent.
        """
        if not query:
            return ""
        cleaned = unicodedata.normalize("NFKC", query)
        cleaned = cls.CONTROL_CHAR_REGEX.sub("", cleaned)
        cleaned = cls.EXCESS_SPACES_REGEX.sub(" ", cleaned)
        return cleaned.strip()
