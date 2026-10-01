"""
CSS and UI Styling module for Local RAG Workshop.
Provides responsive, high-contrast, clean styling with visual status badges,
educational callouts, and structured retrieval inspection cards.
"""

def get_custom_css() -> str:
    """Returns CSS rules for Streamlit application styling."""
    return """
    <style>
    /* Global Container Adjustments */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Badges & Status Indicators */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .badge-primary-qwen {
        background-color: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
    }
    .badge-fallback-llama {
        background-color: #EEF2FF;
        color: #3730A3;
        border: 1px solid #C7D2FE;
    }
    .badge-warning {
        background-color: #FFFBEB;
        color: #92400E;
        border: 1px solid #FDE68A;
    }
    .badge-danger {
        background-color: #FEF2F2;
        color: #991B1B;
        border: 1px solid #FECACA;
    }
    .badge-score-high {
        background-color: #D1FAE5;
        color: #065F46;
        font-weight: 700;
    }
    .badge-score-med {
        background-color: #DBEAFE;
        color: #1E40AF;
        font-weight: 700;
    }
    .badge-score-low {
        background-color: #FEF3C7;
        color: #92400E;
        font-weight: 700;
    }

    /* Metric & Stat Cards */
    .stat-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
    }
    .stat-card-title {
        font-size: 0.78rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
    }
    .stat-card-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0F172A;
    }

    /* Retrieval Transparency Card */
    .retrieval-chunk-card {
        background-color: #FFFFFF;
        border-left: 4px solid #3B82F6;
        border-right: 1px solid #E2E8F0;
        border-top: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
        border-radius: 0 8px 8px 0;
        padding: 0.85rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .retrieval-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.4rem;
        font-size: 0.85rem;
    }
    .retrieval-content {
        font-size: 0.9rem;
        color: #334155;
        line-height: 1.5;
        white-space: pre-wrap;
        background-color: #F8FAFC;
        padding: 0.6rem;
        border-radius: 6px;
        border: 1px solid #F1F5F9;
    }

    /* Educational Callout Box */
    .edu-callout {
        background: linear-gradient(135deg, #F0FDF4 0%, #EFF6FF 100%);
        border: 1px solid #BFDBFE;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 0.75rem;
        margin-bottom: 1rem;
    }
    .edu-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.3rem;
    }
    .edu-body {
        font-size: 0.88rem;
        color: #1E293B;
        line-height: 1.5;
    }

    /* Code & Terminal block */
    .terminal-box {
        background-color: #1E293B;
        color: #F8FAFC;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 0.85rem;
        padding: 0.75rem;
        border-radius: 6px;
        overflow-x: auto;
    }
    </style>
    """
