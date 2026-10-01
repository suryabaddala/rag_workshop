"""
UI module for Local RAG Workshop.
Includes styling definitions and reusable interface components.
"""

from .styles import get_custom_css
from .components import (
    render_header,
    render_model_alert,
    render_document_metrics,
    render_retrieval_panel,
    render_educational_flow,
)

__all__ = [
    "get_custom_css",
    "render_header",
    "render_model_alert",
    "render_document_metrics",
    "render_retrieval_panel",
    "render_educational_flow",
]
