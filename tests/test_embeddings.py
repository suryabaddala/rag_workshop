"""
Unit tests for EmbeddingManager module.
Validates Sentence Transformers embedding generation, L2-normalization,
vector dimensionality, and query encoding.
"""

import numpy as np
import pytest
from core.embeddings import EmbeddingManager


class TestEmbeddingManager:
    @pytest.fixture(scope="class")
    def embed_mgr(self):
        """Shared instance for tests to avoid re-loading weights."""
        return EmbeddingManager.get_instance(model_name="all-MiniLM-L6-v2")

    def test_embedding_dimension(self, embed_mgr):
        assert embed_mgr.dimension == 384

    def test_embed_texts_normalization(self, embed_mgr):
        texts = [
            "Retrieval-Augmented Generation connects LLMs to verified documents.",
            "Sentence Transformers map sentences to a 384-dimensional vector space.",
        ]
        vectors = embed_mgr.embed_texts(texts)

        assert vectors.shape == (2, 384)
        assert vectors.dtype == np.float32

        # Check L2 norm for each vector: ||v||_2 should be 1.0 (approx)
        norms = np.linalg.norm(vectors, axis=1)
        for n in norms:
            assert np.isclose(n, 1.0, atol=1e-3)

    def test_embed_query_normalization(self, embed_mgr):
        query = "What is the passing score for the course?"
        q_vec = embed_mgr.embed_query(query)

        assert q_vec.shape == (384,)
        norm = np.linalg.norm(q_vec)
        assert np.isclose(norm, 1.0, atol=1e-3)

    def test_embed_query_empty_error(self, embed_mgr):
        with pytest.raises(ValueError):
            embed_mgr.embed_query("")
        with pytest.raises(ValueError):
            embed_mgr.embed_query("   ")
