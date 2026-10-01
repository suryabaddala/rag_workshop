"""
Unit tests for DocumentLoader module.
Validates PDF extraction, DOCX extraction, metadata retention,
empty document handling, and error conditions.
"""

from pathlib import Path
import pytest
from core.document_loader import DocumentLoader, DocumentLoaderError

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_docs"


class TestDocumentLoader:
    def test_load_sample_pdf(self):
        pdf_path = SAMPLE_DIR / "college_workshop_syllabus.pdf"
        assert pdf_path.exists(), "Sample syllabus PDF must exist."

        doc = DocumentLoader.load_document(pdf_path)
        assert doc.filename == "college_workshop_syllabus.pdf"
        assert doc.file_type == "pdf"
        assert doc.total_pages == 2
        assert len(doc.pages) == 2
        assert doc.total_chars > 500
        assert not doc.is_image_only
        assert "ELIGIBILITY REQUIREMENTS" in doc.get_full_text()

    def test_load_sample_docx(self):
        docx_path = SAMPLE_DIR / "ai_engineering_handbook.docx"
        assert docx_path.exists(), "Sample handbook DOCX must exist."

        doc = DocumentLoader.load_document(docx_path)
        assert doc.filename == "ai_engineering_handbook.docx"
        assert doc.file_type == "docx"
        assert doc.total_pages >= 1
        assert doc.total_chars > 300
        assert "Retrieval-Augmented Generation" in doc.get_full_text()

    def test_unsupported_file_extension(self):
        dummy_path = Path("fake_document.xlsx")
        with pytest.raises(DocumentLoaderError) as exc_info:
            DocumentLoader.load_document("some content".encode("utf-8"), filename="data.xlsx")
        assert "Unsupported file format '.xlsx'" in str(exc_info.value)

    def test_empty_bytes_error(self):
        with pytest.raises(DocumentLoaderError) as exc_info:
            DocumentLoader.load_document(b"", filename="empty.pdf")
        assert "is empty (0 bytes)" in str(exc_info.value)

    def test_nonexistent_file_error(self):
        with pytest.raises(DocumentLoaderError) as exc_info:
            DocumentLoader.load_document("non_existent_file_987654.pdf")
        assert "File not found" in str(exc_info.value)
