"""
Document Loader module for Local RAG Workshop.
Supports PDF (via PyMuPDF/fitz) and DOCX (via python-docx) extraction.
Preserves page/section metadata and handles scanned or empty documents gracefully.
"""

from dataclasses import dataclass, field
import io
import os
from pathlib import Path
from typing import List, Optional, Union, BinaryIO

import pymupdf
import docx  # python-docx


@dataclass
class DocumentPage:
    """Represents a single extracted page or section of a document."""
    page_number: int
    raw_text: str
    char_count: int
    is_empty: bool = False
    warning: Optional[str] = None


@dataclass
class ExtractedDocument:
    """Represents the complete extracted document with metadata."""
    filename: str
    file_type: str  # 'pdf' or 'docx'
    total_pages: int
    pages: List[DocumentPage] = field(default_factory=list)
    total_chars: int = 0
    is_image_only: bool = False
    error: Optional[str] = None

    def get_full_text(self) -> str:
        """Returns the concatenated text from all non-empty pages."""
        return "\n\n".join(page.raw_text for page in self.pages if not page.is_empty)


class DocumentLoaderError(Exception):
    """Custom exception raised when document extraction fails."""
    pass


class DocumentLoader:
    """
    Extracts text and page metadata from PDF and DOCX documents.
    Optimized for workshop laptops: lightweight, memory-efficient, and fail-safe.
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".docx"}

    @classmethod
    def load_document(
        cls,
        file_source: Union[str, Path, BinaryIO, bytes],
        filename: Optional[str] = None,
    ) -> ExtractedDocument:
        """
        Loads and extracts text from a PDF or DOCX file.

        Args:
            file_source: File path, Path object, byte stream, or bytes.
            filename: Original filename (required if file_source is stream/bytes).

        Returns:
            ExtractedDocument containing page-level text and metadata.
        """
        # Determine filename and extension
        if isinstance(file_source, (str, Path)):
            path = Path(file_source)
            if not path.exists():
                raise DocumentLoaderError(f"File not found: {path}")
            inferred_filename = path.name
            ext = path.suffix.lower()
            with open(path, "rb") as f:
                file_bytes = f.read()
        elif isinstance(file_source, (io.BytesIO, BinaryIO)) or hasattr(file_source, "read"):
            if not filename:
                raise DocumentLoaderError("Filename must be provided when passing a file stream.")
            inferred_filename = filename
            ext = Path(filename).suffix.lower()
            file_bytes = file_source.read() if hasattr(file_source, "read") else bytes(file_source)
        elif isinstance(file_source, bytes):
            if not filename:
                raise DocumentLoaderError("Filename must be provided when passing raw bytes.")
            inferred_filename = filename
            ext = Path(filename).suffix.lower()
            file_bytes = file_source
        else:
            raise DocumentLoaderError(f"Unsupported file source type: {type(file_source)}")

        if ext not in cls.SUPPORTED_EXTENSIONS:
            raise DocumentLoaderError(
                f"Unsupported file format '{ext}'. Supported formats: {', '.join(cls.SUPPORTED_EXTENSIONS)}"
            )

        if len(file_bytes) == 0:
            raise DocumentLoaderError(f"File '{inferred_filename}' is empty (0 bytes).")

        try:
            if ext == ".pdf":
                return cls._extract_pdf(file_bytes, inferred_filename)
            elif ext == ".docx":
                return cls._extract_docx(file_bytes, inferred_filename)
        except Exception as e:
            if isinstance(e, DocumentLoaderError):
                raise
            raise DocumentLoaderError(f"Failed to extract '{inferred_filename}': {str(e)}") from e

        raise DocumentLoaderError(f"Unhandled file extension: {ext}")

    @classmethod
    def _extract_pdf(cls, file_bytes: bytes, filename: str) -> ExtractedDocument:
        """Extracts text page-by-page from a PDF using PyMuPDF (fitz)."""
        try:
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        except Exception as e:
            raise DocumentLoaderError(f"Corrupted or invalid PDF file: {str(e)}")

        pages: List[DocumentPage] = []
        total_chars = 0
        empty_page_count = 0

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_text = page.get_text("text") or ""
            stripped = page_text.strip()
            char_count = len(stripped)

            is_empty = char_count == 0
            warning = None
            if is_empty:
                empty_page_count += 1
                # Check if page contains images
                image_list = page.get_images(full=True)
                if image_list:
                    warning = "Page appears to be scanned or image-only; no extractable text found."
                else:
                    warning = "Empty page."

            pages.append(
                DocumentPage(
                    page_number=page_idx + 1,
                    raw_text=stripped,
                    char_count=char_count,
                    is_empty=is_empty,
                    warning=warning,
                )
            )
            total_chars += char_count

        doc.close()

        is_image_only = (total_chars == 0 and len(pages) > 0)

        return ExtractedDocument(
            filename=filename,
            file_type="pdf",
            total_pages=len(pages),
            pages=pages,
            total_chars=total_chars,
            is_image_only=is_image_only,
            error="Warning: Document has no extractable text. It may be scanned or image-based." if is_image_only else None
        )

    @classmethod
    def _extract_docx(cls, file_bytes: bytes, filename: str) -> ExtractedDocument:
        """Extracts text from a DOCX file using python-docx."""
        try:
            stream = io.BytesIO(file_bytes)
            doc = docx.Document(stream)
        except Exception as e:
            raise DocumentLoaderError(f"Corrupted or invalid DOCX file: {str(e)}")

        # Collect paragraphs
        paragraph_texts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

        # Collect text from tables
        table_texts = []
        for table in doc.tables:
            for row in table.rows:
                row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_cells:
                    table_texts.append(" | ".join(row_cells))

        all_blocks = paragraph_texts + table_texts

        # DOCX files do not have explicit page breaks unless calculated by a layout engine.
        # For educational clarity and workshop utility, we segment paragraphs into logical sections
        # (approx. 4-6 paragraphs per logical section) or preserve full text if small.
        pages: List[DocumentPage] = []
        total_chars = 0

        if not all_blocks:
            pages.append(
                DocumentPage(
                    page_number=1,
                    raw_text="",
                    char_count=0,
                    is_empty=True,
                    warning="No text or table content found in DOCX file.",
                )
            )
        else:
            # Group into logical sections of roughly 500-1000 chars or 5 paragraphs
            current_section: List[str] = []
            current_chars = 0
            section_number = 1

            for block in all_blocks:
                current_section.append(block)
                current_chars += len(block)
                if current_chars >= 800 or len(current_section) >= 6:
                    section_text = "\n\n".join(current_section)
                    pages.append(
                        DocumentPage(
                            page_number=section_number,
                            raw_text=section_text,
                            char_count=len(section_text),
                            is_empty=False,
                        )
                    )
                    total_chars += len(section_text)
                    current_section = []
                    current_chars = 0
                    section_number += 1

            if current_section:
                section_text = "\n\n".join(current_section)
                pages.append(
                    DocumentPage(
                        page_number=section_number,
                        raw_text=section_text,
                        char_count=len(section_text),
                        is_empty=False,
                    )
                )
                total_chars += len(section_text)

        return ExtractedDocument(
            filename=filename,
            file_type="docx",
            total_pages=len(pages),
            pages=pages,
            total_chars=total_chars,
            is_image_only=False,
            error="Document contains no text content." if total_chars == 0 else None,
        )
