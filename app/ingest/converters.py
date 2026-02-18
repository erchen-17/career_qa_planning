"""
File converters: PDF → images, DOCX → text, image passthrough.
"""

from __future__ import annotations

import io
import logging
from typing import List

logger = logging.getLogger(__name__)


def pdf_to_images(file_bytes: bytes, dpi: int = 200) -> list[bytes]:
    """
    Render each page of a PDF as a PNG image.
    Returns a list of PNG bytes (one per page).
    """
    import fitz  # PyMuPDF

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    images: list[bytes] = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=dpi)
        images.append(pix.tobytes("png"))
        logger.debug("PDF page %d rendered to PNG (%d bytes)", page_num + 1, len(images[-1]))
    doc.close()
    return images


def docx_to_text(file_bytes: bytes) -> str:
    """
    Extract plain text from a DOCX file using python-docx.
    Preserves paragraph structure with double newlines.
    """
    from docx import Document

    doc = Document(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    text = "\n\n".join(paragraphs)
    logger.debug("DOCX extracted %d paragraphs, %d chars", len(paragraphs), len(text))
    return text


def image_to_bytes(file_bytes: bytes) -> list[bytes]:
    """
    Passthrough for PNG/JPG files.
    Returns a single-element list for consistent interface with pdf_to_images.
    """
    return [file_bytes]


def convert_file(file_bytes: bytes, file_ext: str, dpi: int = 200) -> dict:
    """
    Unified conversion dispatcher.

    Returns:
        {
            "mode": "images" | "text",
            "images": list[bytes]  (if mode == "images"),
            "text": str            (if mode == "text"),
            "pages": int,
        }
    """
    ext = file_ext.lower().lstrip(".")

    if ext in ("png", "jpg", "jpeg"):
        imgs = image_to_bytes(file_bytes)
        return {"mode": "images", "images": imgs, "pages": 1}

    if ext == "pdf":
        imgs = pdf_to_images(file_bytes, dpi=dpi)
        return {"mode": "images", "images": imgs, "pages": len(imgs)}

    if ext in ("docx", "doc"):
        text = docx_to_text(file_bytes)
        return {"mode": "text", "text": text, "pages": 1}

    raise ValueError(f"Unsupported file extension: .{ext}")
