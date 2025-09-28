"""
PDF text extraction utilities for the RAG system.
Supports both PyMuPDF and pdfplumber for comparison.
"""

import fitz  # type: ignore # PyMuPDF
import pdfplumber
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    PDF text extraction with multiple backend support.
    """

    def __init__(self, method: str = "pymupdf"):
        """
        Initialize PDF processor.

        Args:
            method: Extraction method ("pymupdf" or "pdfplumber")
        """
        if method not in ["pymupdf", "pdfplumber"]:
            raise ValueError("Method must be 'pymupdf' or 'pdfplumber'")
        self.method = method

    def extract_text(self, pdf_path: str) -> Dict[str, any]:
        """
        Extract text from PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary containing extracted text and metadata
        """
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if self.method == "pymupdf":
            return self._extract_with_pymupdf(pdf_path)
        else:
            return self._extract_with_pdfplumber(pdf_path)

    def _extract_with_pymupdf(self, pdf_path: str) -> Dict[str, any]:
        """Extract text using PyMuPDF."""
        doc = fitz.open(pdf_path)

        text_data = {
            "full_text": "",
            "pages": [],
            "metadata": {
                "total_pages": len(doc),
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "creator": doc.metadata.get("creator", ""),
                "producer": doc.metadata.get("producer", ""),
                "creation_date": doc.metadata.get("creationDate", ""),
                "modification_date": doc.metadata.get("modDate", ""),
                "method": "pymupdf"
            }
        }

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text()

            # Clean and normalize text
            page_text = self._clean_text(page_text)

            text_data["pages"].append({
                "page_number": page_num + 1,
                "text": page_text,
                "char_count": len(page_text)
            })

            text_data["full_text"] += page_text + "\n\n"

        doc.close()

        # Clean full text
        text_data["full_text"] = self._clean_text(text_data["full_text"])
        text_data["metadata"]["total_chars"] = len(text_data["full_text"])

        return text_data

    def _extract_with_pdfplumber(self, pdf_path: str) -> Dict[str, any]:
        """Extract text using pdfplumber."""
        text_data = {
            "full_text": "",
            "pages": [],
            "metadata": {
                "method": "pdfplumber"
            }
        }

        with pdfplumber.open(pdf_path) as pdf:
            text_data["metadata"]["total_pages"] = len(pdf.pages)

            # Extract metadata if available
            if hasattr(pdf, 'metadata') and pdf.metadata:
                metadata = pdf.metadata
                text_data["metadata"].update({
                    "title": metadata.get("Title", ""),
                    "author": metadata.get("Author", ""),
                    "subject": metadata.get("Subject", ""),
                    "creator": metadata.get("Creator", ""),
                    "producer": metadata.get("Producer", ""),
                    "creation_date": str(metadata.get("CreationDate", "")),
                    "modification_date": str(metadata.get("ModDate", ""))
                })

            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""

                # Clean and normalize text
                page_text = self._clean_text(page_text)

                text_data["pages"].append({
                    "page_number": page_num + 1,
                    "text": page_text,
                    "char_count": len(page_text)
                })

                text_data["full_text"] += page_text + "\n\n"

        # Clean full text
        text_data["full_text"] = self._clean_text(text_data["full_text"])
        text_data["metadata"]["total_chars"] = len(text_data["full_text"])

        return text_data

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Remove excessive whitespace
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            if line:  # Skip empty lines
                cleaned_lines.append(line)

        # Join with single newlines
        cleaned_text = '\n'.join(cleaned_lines)

        # Replace multiple spaces with single space
        import re
        cleaned_text = re.sub(r' +', ' ', cleaned_text)

        return cleaned_text

    def get_text_stats(self, text_data: Dict[str, any]) -> Dict[str, any]:
        """
        Get statistics about extracted text.

        Args:
            text_data: Text data from extract_text()

        Returns:
            Dictionary with text statistics
        """
        full_text = text_data["full_text"]
        pages = text_data["pages"]

        stats = {
            "total_characters": len(full_text),
            "total_words": len(full_text.split()),
            "total_pages": len(pages),
            "avg_chars_per_page": len(full_text) / len(pages) if pages else 0,
            "avg_words_per_page": len(full_text.split()) / len(pages) if pages else 0,
            "page_stats": []
        }

        for page in pages:
            page_words = len(page["text"].split())
            stats["page_stats"].append({
                "page_number": page["page_number"],
                "characters": page["char_count"],
                "words": page_words
            })

        return stats


def extract_pdf_text(pdf_path: str, method: str = "pymupdf") -> Dict[str, any]:
    """
    Convenience function to extract text from PDF.

    Args:
        pdf_path: Path to PDF file
        method: Extraction method ("pymupdf" or "pdfplumber")

    Returns:
        Dictionary containing extracted text and metadata
    """
    processor = PDFProcessor(method=method)
    return processor.extract_text(pdf_path)


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) != 2:
        print("Usage: python pdf_processor.py <pdf_path>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    # Test both methods
    for method in ["pymupdf", "pdfplumber"]:
        print(f"\n--- Testing {method} ---")
        try:
            processor = PDFProcessor(method=method)
            data = processor.extract_text(pdf_path)
            stats = processor.get_text_stats(data)

            print(f"Title: {data['metadata'].get('title', 'N/A')}")
            print(f"Author: {data['metadata'].get('author', 'N/A')}")
            print(f"Pages: {stats['total_pages']}")
            print(f"Characters: {stats['total_characters']:,}")
            print(f"Words: {stats['total_words']:,}")
            print(f"Preview: {data['full_text'][:200]}...")

        except Exception as e:
            print(f"Error with {method}: {e}")