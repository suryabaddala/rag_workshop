"""
Unit tests for DocumentChunker module.
Validates chunk sizing, sliding window overlap, metadata retention,
and boundary conditions.
"""

import pytest
from core.chunker import DocumentChunker, DocumentChunk
from core.document_loader import ExtractedDocument, DocumentPage


class TestDocumentChunker:
    def test_chunker_initialization_validation(self):
        with pytest.raises(ValueError):
            DocumentChunker(chunk_size=0)
        with pytest.raises(ValueError):
            DocumentChunker(chunk_size=-10)
        with pytest.raises(ValueError):
            DocumentChunker(chunk_size=500, chunk_overlap=500)
        with pytest.raises(ValueError):
            DocumentChunker(chunk_size=500, chunk_overlap=600)
        with pytest.raises(ValueError):
            DocumentChunker(chunk_size=500, chunk_overlap=-1)

    def test_chunk_raw_text(self):
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20, min_chunk_size=20)
        sample_text = (
            "Retrieval-Augmented Generation enhances large language models by integrating external knowledge bases. "
            "Instead of relying purely on parametric memory, the system fetches relevant document chunks using cosine similarity. "
            "This methodology dramatically mitigates hallucination and enables verifiable source attribution."
        )
        chunks = chunker.chunk_raw_text(sample_text, source_name="test_guide.txt", page_number=1)

        assert len(chunks) >= 2
        for idx, chunk in enumerate(chunks, start=1):
            assert chunk.source_file == "test_guide.txt"
            assert chunk.page_number == 1
            assert chunk.chunk_index == idx
            assert chunk.char_count > 0
            assert chunk.word_count > 0
            assert len(chunk.text) <= 150  # Allows slight boundary buffer

    def test_chunk_document_with_pages(self):
        chunker = DocumentChunker(chunk_size=150, chunk_overlap=30, min_chunk_size=20)
        doc = ExtractedDocument(
            filename="syllabus.pdf",
            file_type="pdf",
            total_pages=2,
            pages=[
                DocumentPage(
                    page_number=1,
                    raw_text="Page 1 covers the course overview and eligibility criteria for students.",
                    char_count=72,
                ),
                DocumentPage(
                    page_number=2,
                    raw_text="Page 2 outlines the grading scheme, quizzes, and project evaluation policies.",
                    char_count=79,
                ),
            ],
            total_chars=151,
        )

        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 2
        assert chunks[0].page_number == 1
        assert chunks[0].source_file == "syllabus.pdf"
        assert "Page 1 covers" in chunks[0].text
        assert chunks[1].page_number == 2
        assert "Page 2 outlines" in chunks[1].text

    def test_empty_or_tiny_text_chunking(self):
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=50, min_chunk_size=30)
        assert chunker.chunk_raw_text("") == []
        assert chunker.chunk_raw_text("   ") == []
