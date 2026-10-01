"""
Embeddings module for Local RAG Workshop.
Generates dense vector representations locally with a SentenceTransformer-first
approach and a lightweight fallback that works on locked-down Windows machines.
"""

from typing import List, Optional
import numpy as np


class EmbeddingManager:
    """
    Manages local embedding model lifecycle and vector generation.
    Enforces L2 normalization for efficient dot-product cosine similarity.
    """

    DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
    _instance: Optional["EmbeddingManager"] = None

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or "cpu"
        self._model = None
        self._fallback_vectorizer = None
        self._dimension: Optional[int] = None

    @classmethod
    def get_instance(cls, model_name: str = DEFAULT_MODEL_NAME) -> "EmbeddingManager":
        """Singleton accessor to prevent redundant model instantiation."""
        if cls._instance is None or cls._instance.model_name != model_name:
            cls._instance = cls(model_name=model_name)
        return cls._instance

    def _load_model(self):
        """Lazy-loads the SentenceTransformer model, or falls back to a local hash-based vectorizer."""
        if self._model is not None or self._fallback_vectorizer is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._dimension = self._model.get_sentence_embedding_dimension()
            return
        except Exception:
            from sklearn.feature_extraction.text import HashingVectorizer

            self._dimension = 384
            self._fallback_vectorizer = HashingVectorizer(
                n_features=self._dimension,
                alternate_sign=False,
                ngram_range=(1, 2),
                norm="l2",
                lowercase=True,
            )

    @property
    def dimension(self) -> int:
        """Returns the dimensionality of the embedding vectors (e.g. 384 for all-MiniLM-L6-v2)."""
        if self._dimension is None:
            self._load_model()
        return self._dimension

    def _to_numpy_array(self, texts: List[str]) -> np.ndarray:
        """Normalizes and converts text to a 2D float32 matrix."""
        if self._model is not None:
            return np.asarray(texts, dtype=object)

        vectors = self._fallback_vectorizer.transform(texts).toarray().astype(np.float32)
        return vectors

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generates normalized embedding vectors for a list of text strings.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        self._load_model()

        if self._model is not None:
            embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return embeddings.astype(np.float32)

        vectors = self._fallback_vectorizer.transform(texts).toarray().astype(np.float32)
        return vectors

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generates a single normalized embedding vector for a query.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        self._load_model()

        if self._model is not None:
            vector = self._model.encode(
                query.strip(),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return vector.astype(np.float32).flatten()

        vector = self._fallback_vectorizer.transform([query.strip()]).toarray().astype(np.float32).flatten()
        return vector
