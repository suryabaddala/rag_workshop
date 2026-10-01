"""
Reusable UI Components for Local RAG Workshop Streamlit Interface.
Includes header badges, document statistics, retrieval transparency panels,
and educational walkthroughs.
"""

from typing import List, Optional
import streamlit as st

from core.model_selector import ModelStatus
from core.retriever import RetrievedChunk
from core.document_loader import ExtractedDocument


def render_header(model_status: ModelStatus, ollama_connected: bool):
    """Renders the top application header with live status badges."""
    col_title, col_badges = st.columns([3, 2])

    with col_title:
        st.markdown("### 🎓 Local RAG Workshop: Chat with Documents")
        st.caption("100% Local Retrieval-Augmented Generation • Sentence Transformers • Ollama (Qwen/Llama)")

    with col_badges:
        # Ollama connection badge
        if ollama_connected:
            conn_html = '<span class="status-badge badge-primary-qwen">🟢 Ollama Connected</span>'
        else:
            conn_html = '<span class="status-badge badge-danger">🔴 Ollama Offline</span>'

        # Model badge
        if model_status.status == "active":
            if model_status.family == "Qwen":
                model_html = f'<span class="status-badge badge-primary-qwen">⚡ Primary: {model_status.selected_model}</span>'
            elif model_status.family == "Llama":
                model_html = f'<span class="status-badge badge-fallback-llama">🔄 Fallback: {model_status.selected_model}</span>'
            else:
                model_html = f'<span class="status-badge badge-warning">📦 Active: {model_status.selected_model}</span>'
        elif model_status.status == "offline":
            model_html = '<span class="status-badge badge-danger">⚠️ Daemon Unreachable</span>'
        else:
            model_html = '<span class="status-badge badge-warning">⚠️ No Model Installed</span>'

        st.markdown(
            f"""
            <div style="display: flex; gap: 0.5rem; justify-content: flex-end; align-items: center; height: 100%; flex-wrap: wrap;">
                {conn_html}
                {model_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()


def render_model_alert(model_status: ModelStatus):
    """Displays prominent banner if Ollama is offline or missing required models."""
    if model_status.status == "offline":
        st.error(
            f"**Ollama Offline**: {model_status.message}\n\n"
            "Please ensure Ollama is installed and running locally on your computer."
        )
        with st.expander("🛠️ View Ollama Startup Instructions", expanded=True):
            st.code(model_status.setup_instructions, language="bash")

    elif model_status.status == "no_models":
        st.warning(
            f"**Missing Model**: {model_status.message}\n\n"
            "Please open your terminal and install Qwen (primary) or Llama (fallback)."
        )
        with st.expander("📥 View Model Installation Commands", expanded=True):
            st.code(model_status.setup_instructions, language="bash")

    elif model_status.status == "other_only":
        st.warning(f"**No supported workshop model**: {model_status.message}")
        with st.expander("View model installation commands", expanded=True):
            st.code(model_status.setup_instructions, language="bash")


def render_document_metrics(doc: ExtractedDocument, chunk_count: int):
    """Renders visual cards displaying document extraction statistics."""
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-card-title">File Type</div>
                <div class="stat-card-value">{doc.file_type.upper()}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        unit = "Pages" if doc.file_type == "pdf" else "Sections"
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-card-title">{unit}</div>
                <div class="stat-card-value">{doc.total_pages}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-card-title">Total Characters</div>
                <div class="stat-card-value">{doc.total_chars:,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-card-title">Chunks Generated</div>
                <div class="stat-card-value">{chunk_count}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_retrieval_panel(retrieved_chunks: List[RetrievedChunk]):
    """Renders the educational retrieval transparency panel showing chunks and cosine scores."""
    if not retrieved_chunks:
        st.info("No chunks were retrieved for this query.")
        return

    st.markdown("#### 🔬 Retrieval Transparency Panel (Evidence & Similarity Scores)")
    st.caption("Here is the exact grounded context retrieved from your document that was supplied to the LLM:")

    for idx, item in enumerate(retrieved_chunks, start=1):
        score = item.similarity_score
        if score >= 0.55:
            score_class = "badge-score-high"
            relevance_label = "High Relevance"
        elif score >= 0.35:
            score_class = "badge-score-med"
            relevance_label = "Moderate Relevance"
        else:
            score_class = "badge-score-low"
            relevance_label = "Borderline Relevance"

        page_str = f"Page {item.page_number}" if item.page_number else "Section"

        with st.expander(
            f"Chunk #{idx} | {item.source_file} ({page_str}) | Similarity: {score:.4f} [{relevance_label}]",
            expanded=(idx == 1),
        ):
            st.markdown(
                f"""
                <div style="margin-bottom: 0.5rem; display: flex; gap: 0.5rem; align-items: center;">
                    <span class="status-badge {score_class}">Cosine Similarity: {score:.4f}</span>
                    <span class="status-badge" style="background:#F1F5F9; color:#475569;">Chunk ID: {item.chunk_id}</span>
                    <span class="status-badge" style="background:#F1F5F9; color:#475569;">Words: {item.chunk.word_count}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(f"```text\n{item.text}\n```")
            st.caption(
                f"📐 **Calculation note:** Vector dot product = {score:.4f}. "
                "Normalized Sentence Transformer vectors (384 dimensions) compared via cosine similarity."
            )


def render_educational_flow():
    """Renders an interactive educational guide explaining the 8-stage RAG pipeline."""
    st.markdown("### 📚 How Local RAG Works: Step-by-Step Workshop Guide")
    st.markdown(
        """
        Retrieval-Augmented Generation (RAG) connects an LLM to external documents without retraining or fine-tuning.
        Instead of asking the LLM to 'know' your document, we search your document for relevant passages and paste them into the prompt.
        """
    )

    steps = [
        ("1. Document Ingestion", "PyMuPDF (`fitz`) or `python-docx` extracts raw text and tracks source page numbers."),
        ("2. Text Cleaning", "Removes PDF ligatures, hyphenated line-breaks, extra newlines, and non-printable control characters."),
        ("3. Semantic Chunking", "Splits long text into overlapping windows (e.g. 500 chars with 50 char overlap) on sentence boundaries."),
        ("4. Local Embeddings", "A lightweight local model (`all-MiniLM-L6-v2`) converts each text chunk into a 384-dimensional dense vector."),
        ("5. Cosine Similarity Search", "When you ask a question, its vector is compared against all chunk vectors using $\\cos(\\theta) = \\frac{\\mathbf{u} \\cdot \\mathbf{v}}{\\|\\mathbf{u}\\| \\|\\mathbf{v}\\|}$."),
        ("6. Top-K Threshold Filtering", "Ranks chunks descending by score and filters out any chunk scoring below the minimum threshold (e.g. 0.30)."),
        ("7. Grounded Context Construction", "Wraps the top retrieved chunks inside `<retrieved_context>` with strict rules: 'Do not invent. Only use this context.'"),
        ("8. Local LLM Generation", "Sends prompt to locally running Ollama (Qwen or Llama). The model produces an exact, cited answer!"),
    ]

    for title, desc in steps:
        st.markdown(
            f"""
            <div class="edu-callout">
                <div class="edu-title">{title}</div>
                <div class="edu-body">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
