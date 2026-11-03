from io import BytesIO
from typing import Dict

from pypdf import PdfReader


def extract_page_texts(pdf_bytes: bytes) -> Dict[int, str]:
    """Return {page_number (1-based): text}."""
    reader = PdfReader(BytesIO(pdf_bytes))
    return {i + 1: (page.extract_text() or "").strip() for i, page in enumerate(reader.pages)}
