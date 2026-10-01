"""
Core components for Local RAG Workshop pipeline.
Includes Document Loader, Text Cleaner, Chunker, Embeddings,
Retriever, Ollama Client, Model Selector, and RAG Pipeline.
"""

from .document_loader import DocumentLoader, ExtractedDocument, DocumentPage
from .text_cleaner import TextCleaner
from .chunker import DocumentChunker, DocumentChunk
from .embeddings import EmbeddingManager
from .retriever import CosineRetriever, RetrievedChunk
from .ollama_client import OllamaClient
from .model_selector import ModelSelector, ModelStatus
from .rag_pipeline import RAGPipeline, RAGResult

__all__ = [
    "DocumentLoader",
    "ExtractedDocument",
    "DocumentPage",
    "TextCleaner",
    "DocumentChunker",
    "DocumentChunk",
    "EmbeddingManager",
    "CosineRetriever",
    "RetrievedChunk",
    "OllamaClient",
    "ModelSelector",
    "ModelStatus",
    "RAGPipeline",
    "RAGResult",
]
