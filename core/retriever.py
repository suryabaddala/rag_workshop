"""
Retriever and Cosine Similarity module for Local RAG Workshop.
Implements mathematically rigorous vector similarity search, Top-K ranking,
threshold filtering, deduplication, and educational score transparency.
"""

from dataclasses import dataclass
from typing import List, Optional, Set
import numpy as np

from .chunker import DocumentChunk


@dataclass
class RetrievedChunk:
    """Represents a chunk retrieved during semantic search with score and rank."""
    chunk: DocumentChunk
    similarity_score: float
    rank: int
    retrieval_explanation: str

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def source_file(self) -> str:
        return self.chunk.source_file

    @property
    def page_number(self) -> int:
        return self.chunk.page_number

    @property
    def text(self) -> str:
        return self.chunk.text


class CosineRetriever:
    """
    In-memory vector store and cosine similarity search engine.
    Demonstrates exact vector mathematics for workshop students without black-box abstraction.
    """

    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None  # shape: (N, D)

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    @property
    def indexed_documents(self) -> Set[str]:
        return {c.source_file for c in self.chunks}

    def index_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: np.ndarray,
        replace_sources: bool = False,
    ):
        """
        Populate the retriever with document chunks and their corresponding embedding vectors.

        Args:
            chunks: List of DocumentChunk instances.
            embeddings: 2D numpy array of shape (len(chunks), embedding_dim).
            replace_sources: Remove existing chunks from the same source files before indexing.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch between chunks count ({len(chunks)}) and embeddings count ({len(embeddings)})."
            )

        if len(chunks) == 0:
            return

        if replace_sources and self.embeddings is not None:
            replacement_names = {chunk.source_file for chunk in chunks}
            keep_indices = [
                index for index, chunk in enumerate(self.chunks)
                if chunk.source_file not in replacement_names
            ]
            self.chunks = [self.chunks[index] for index in keep_indices]
            self.embeddings = self.embeddings[keep_indices]
            if not self.chunks:
                self.embeddings = None

        if self.embeddings is None or len(self.chunks) == 0:
            self.chunks = list(chunks)
            self.embeddings = embeddings.astype(np.float32)
        else:
            self.chunks.extend(chunks)
            self.embeddings = np.vstack([self.embeddings, embeddings.astype(np.float32)])

    def clear(self):
        """Resets the knowledge base index."""
        self.chunks = []
        self.embeddings = None

    @staticmethod
    def calculate_cosine_similarity(query_vector: np.ndarray, chunk_vectors: np.ndarray) -> np.ndarray:
        """
        Calculates cosine similarity between a 1D query vector and a 2D matrix of chunk vectors.

        Mathematical Formula:
            Cosine Similarity = (u · v) / (||u||_2 * ||v||_2)

        When vectors are already L2-normalized (||u||_2 = 1 and ||v||_2 = 1):
            Cosine Similarity = u · v (dot product)

        This method computes full norms to guarantee mathematical correctness regardless of input normalization.

        Args:
            query_vector: 1D array of shape (D,)
            chunk_vectors: 2D array of shape (N, D)

        Returns:
            1D array of similarity scores in [-1.0, 1.0] of shape (N,)
        """
        if query_vector.ndim == 1:
            q = query_vector.reshape(1, -1)
        else:
            q = query_vector

        # Dot product of query with all chunks: shape (1, N)
        dot_product = np.dot(q, chunk_vectors.T).flatten()

        # Compute L2 norms
        q_norm = np.linalg.norm(q, axis=1, keepdims=True)  # shape (1, 1)
        chunk_norms = np.linalg.norm(chunk_vectors, axis=1)  # shape (N,)

        # Avoid division by zero with epsilon
        epsilon = 1e-10
        norm_product = np.maximum(q_norm * chunk_norms, epsilon).flatten()

        cosine_sim = dot_product / norm_product
        # Clip numerical artifacts within [-1.0, 1.0]
        return np.clip(cosine_sim, -1.0, 1.0)

    def retrieve(
        self,
        query_vector: np.ndarray,
        top_k: int = 3,
        similarity_threshold: float = 0.30,
        deduplicate: bool = True,
    ) -> List[RetrievedChunk]:
        """
        Performs Top-K retrieval against stored chunks using cosine similarity.

        Args:
            query_vector: 1D query embedding of shape (D,).
            top_k: Number of highest-similarity chunks to return.
            similarity_threshold: Minimum cosine similarity required (filters irrelevant chunks).
            deduplicate: If True, suppresses duplicate or highly overlapping text chunks.

        Returns:
            Ranked list of RetrievedChunk objects.
        """
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        if top_k <= 0:
            return []

        # Step 1: Compute similarity scores across all indexed chunks
        scores = self.calculate_cosine_similarity(query_vector, self.embeddings)

        # Step 2: Sort indices by descending score
        ranked_indices = np.argsort(scores)[::-1]

        retrieved: List[RetrievedChunk] = []
        seen_texts: Set[str] = set()
        rank = 1

        for idx in ranked_indices:
            score = float(scores[idx])

            # Step 3: Apply minimum similarity threshold
            if score < similarity_threshold:
                continue

            chunk = self.chunks[idx]

            # Step 4: Deduplicate redundant chunks
            if deduplicate:
                normalized_text_snippet = " ".join(chunk.text.lower().split()[:25])
                if normalized_text_snippet in seen_texts:
                    continue
                seen_texts.add(normalized_text_snippet)

            # Step 5: Educational explanation of the score
            explanation = (
                f"Cosine similarity = {score:.4f} "
                f"({'High Relevance' if score >= 0.55 else 'Moderate Relevance' if score >= 0.35 else 'Borderline Relevance'})."
            )

            retrieved.append(
                RetrievedChunk(
                    chunk=chunk,
                    similarity_score=round(score, 4),
                    rank=rank,
                    retrieval_explanation=explanation,
                )
            )
            rank += 1

            if len(retrieved) >= top_k:
                break

        return retrieved
