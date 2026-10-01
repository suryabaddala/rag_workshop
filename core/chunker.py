"""
Chunking module for Local RAG Workshop.
Splits extracted document text into semantically meaningful chunks with
configurable size and overlap, maintaining complete source and page metadata.
"""

from dataclasses import dataclass
import re
from typing import List, Optional
from .document_loader import ExtractedDocument, DocumentPage
from .text_cleaner import TextCleaner


@dataclass
class DocumentChunk:
    """Represents an atomic, searchable piece of a document with rich metadata."""
    chunk_id: str
    source_file: str
    page_number: int
    chunk_index: int
    text: str
    char_count: int
    word_count: int

    def __repr__(self) -> str:
        snippet = self.text[:60] + "..." if len(self.text) > 60 else self.text
        return f"<DocumentChunk id={self.chunk_id} page={self.page_number} chars={self.char_count} text='{snippet}'>"


class DocumentChunker:
    """
    Splits text into meaningful, overlapping chunks for semantic retrieval.
    Balances chunk coherence, context retention, and laptop memory footprint.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        min_chunk_size: int = 40,
    ):
        """
        Initialize chunker parameters.

        Args:
            chunk_size: Target maximum character length for each chunk (default: 500).
            chunk_overlap: Character overlap between consecutive chunks (default: 50).
            min_chunk_size: Minimum characters required to retain a chunk (filters noise).
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_document(self, doc: ExtractedDocument) -> List[DocumentChunk]:
        """
        Chunks an entire ExtractedDocument, preserving page numbers for every chunk.

        Args:
            doc: ExtractedDocument instance.

        Returns:
            List of DocumentChunk objects with unique IDs and source metadata.
        """
        all_chunks: List[DocumentChunk] = []
        global_chunk_idx = 1
        safe_filename = re.sub(r"[^\w\-.]", "_", doc.filename)

        for page in doc.pages:
            if page.is_empty or not page.raw_text:
                continue

            cleaned_page_text = TextCleaner.clean(page.raw_text)
            if len(cleaned_page_text) < self.min_chunk_size:
                continue

            page_chunks = self._split_text_into_chunks(cleaned_page_text)

            for p_chunk in page_chunks:
                chunk_id = f"{safe_filename}_p{page.page_number}_c{global_chunk_idx:03d}"
                all_chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        source_file=doc.filename,
                        page_number=page.page_number,
                        chunk_index=global_chunk_idx,
                        text=p_chunk,
                        char_count=len(p_chunk),
                        word_count=len(p_chunk.split()),
                    )
                )
                global_chunk_idx += 1

        return all_chunks

    def chunk_raw_text(
        self,
        text: str,
        source_name: str = "document.txt",
        page_number: int = 1,
    ) -> List[DocumentChunk]:
        """
        Convenience method to chunk raw text directly. Useful for tests or single text blocks.
        """
        cleaned = TextCleaner.clean(text)
        if len(cleaned) < self.min_chunk_size:
            if cleaned:
                return [
                    DocumentChunk(
                        chunk_id=f"{source_name}_p1_c001",
                        source_file=source_name,
                        page_number=page_number,
                        chunk_index=1,
                        text=cleaned,
                        char_count=len(cleaned),
                        word_count=len(cleaned.split()),
                    )
                ]
            return []

        raw_chunks = self._split_text_into_chunks(cleaned)
        chunks = []
        for idx, text_block in enumerate(raw_chunks, start=1):
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{source_name}_p{page_number}_c{idx:03d}",
                    source_file=source_name,
                    page_number=page_number,
                    chunk_index=idx,
                    text=text_block,
                    char_count=len(text_block),
                    word_count=len(text_block.split()),
                )
            )
        return chunks

    def _split_text_into_chunks(self, text: str) -> List[str]:
        """
        Recursively splits text using natural boundaries (paragraphs -> sentences -> words)
        with sliding-window character overlap.
        """
        if len(text) <= self.chunk_size:
            return [text.strip()]

        chunks: List[str] = []
        start_idx = 0
        text_len = len(text)

        while start_idx < text_len:
            # End index candidate
            end_idx = min(start_idx + self.chunk_size, text_len)

            if end_idx < text_len:
                # Look for natural breaking points near the end candidate to avoid splitting words
                # 1. Paragraph break
                para_break = text.rfind("\n\n", start_idx, end_idx)
                if para_break != -1 and para_break > start_idx + (self.chunk_size // 2):
                    end_idx = para_break
                else:
                    # 2. Sentence end (. ! ?)
                    match = None
                    for pattern in (r"\.\s", r"\?\s", r"!\s", r"\n"):
                        matches = list(re.finditer(pattern, text[start_idx:end_idx]))
                        if matches:
                            candidate = start_idx + matches[-1].end()
                            if candidate > start_idx + (self.chunk_size // 2):
                                end_idx = candidate
                                break
                    else:
                        # 3. Space / word break
                        space_break = text.rfind(" ", start_idx, end_idx)
                        if space_break != -1 and space_break > start_idx + (self.chunk_size // 3):
                            end_idx = space_break

            chunk_text = text[start_idx:end_idx].strip()
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(chunk_text)

            # Move forward with overlap
            step = end_idx - start_idx
            if step <= self.chunk_overlap:
                start_idx += max(1, step)
            else:
                start_idx = end_idx - self.chunk_overlap

            # Guard against infinite loops
            if start_idx >= text_len or end_idx >= text_len:
                break

        return chunks
