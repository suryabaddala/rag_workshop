"""
Unit tests for CosineRetriever module.
Validates mathematical cosine similarity, Top-K ranking,
threshold filtering, and deduplication.
"""

import numpy as np
import pytest
from core.retriever import CosineRetriever, RetrievedChunk
from core.chunker import DocumentChunk


class TestCosineRetriever:
    def test_cosine_similarity_mathematics(self):
        # 1. Identical unit vectors should yield 1.0
        u = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        sim = CosineRetriever.calculate_cosine_similarity(u, v)
        assert np.isclose(sim[0], 1.0, atol=1e-5)

        # 2. Orthogonal vectors should yield 0.0
        v_ortho = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        sim_ortho = CosineRetriever.calculate_cosine_similarity(u, v_ortho)
        assert np.isclose(sim_ortho[0], 0.0, atol=1e-5)

        # 3. Opposite vectors should yield -1.0
        v_opp = np.array([[-1.0, 0.0, 0.0]], dtype=np.float32)
        sim_opp = CosineRetriever.calculate_cosine_similarity(u, v_opp)
        assert np.isclose(sim_opp[0], -1.0, atol=1e-5)

        # 4. Unnormalized vectors (e.g. [3, 4, 0] with norm 5, and [6, 8, 0] with norm 10)
        u_unnorm = np.array([3.0, 4.0, 0.0], dtype=np.float32)
        v_unnorm = np.array([[6.0, 8.0, 0.0]], dtype=np.float32)
        sim_unnorm = CosineRetriever.calculate_cosine_similarity(u_unnorm, v_unnorm)
        assert np.isclose(sim_unnorm[0], 1.0, atol=1e-5)

    def test_top_k_ranking_and_threshold(self):
        retriever = CosineRetriever()

        # Create 3 synthetic chunks
        chunks = [
            DocumentChunk(
                chunk_id="chunk_high",
                source_file="guide.pdf",
                page_number=1,
                chunk_index=1,
                text="High similarity content regarding RAG architecture.",
                char_count=52,
                word_count=7,
            ),
            DocumentChunk(
                chunk_id="chunk_med",
                source_file="guide.pdf",
                page_number=1,
                chunk_index=2,
                text="Moderate similarity content regarding general python.",
                char_count=54,
                word_count=7,
            ),
            DocumentChunk(
                chunk_id="chunk_low",
                source_file="guide.pdf",
                page_number=2,
                chunk_index=3,
                text="Irrelevant content about astrophysics.",
                char_count=39,
                word_count=4,
            ),
        ]

        # Normalized 3D synthetic embeddings
        # Query will be [1.0, 0.0, 0.0]
        # chunk 1 is [0.95, 0.31, 0.0] -> sim approx 0.95
        # chunk 2 is [0.50, 0.86, 0.0] -> sim approx 0.50
        # chunk 3 is [0.05, 0.99, 0.0] -> sim approx 0.05
        embeddings = np.array([
            [0.9512, 0.3085, 0.0],
            [0.5000, 0.8660, 0.0],
            [0.0500, 0.9987, 0.0],
        ], dtype=np.float32)

        retriever.index_chunks(chunks, embeddings)
        query_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # Retrieve with top_k=2 and threshold=0.30
        results = retriever.retrieve(query_vec, top_k=2, similarity_threshold=0.30)
        assert len(results) == 2
        assert results[0].chunk_id == "chunk_high"
        assert results[0].similarity_score > 0.90
        assert results[1].chunk_id == "chunk_med"
        assert results[1].similarity_score > 0.45

        # With higher threshold (0.60), only chunk_high should pass
        results_strict = retriever.retrieve(query_vec, top_k=3, similarity_threshold=0.60)
        assert len(results_strict) == 1
        assert results_strict[0].chunk_id == "chunk_high"

        # With very high threshold (0.98), empty list
        results_none = retriever.retrieve(query_vec, top_k=3, similarity_threshold=0.98)
        assert len(results_none) == 0

    def test_empty_retriever(self):
        retriever = CosineRetriever()
        q = np.array([1.0, 0.0], dtype=np.float32)
        assert retriever.retrieve(q, top_k=3) == []

    def test_reindex_replaces_only_matching_source(self):
        retriever = CosineRetriever()
        original_chunks = [
            DocumentChunk("old", "guide.pdf", 1, 1, "Old guide content", 17, 3),
            DocumentChunk("other", "notes.docx", 1, 1, "Other document content", 22, 3),
        ]
        retriever.index_chunks(
            original_chunks,
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )

        retriever.index_chunks(
            [DocumentChunk("new", "guide.pdf", 1, 1, "Updated guide content", 21, 3)],
            np.array([[0.8, 0.6]], dtype=np.float32),
            replace_sources=True,
        )

        assert [chunk.chunk_id for chunk in retriever.chunks] == ["other", "new"]
        assert retriever.embeddings.shape == (2, 2)
