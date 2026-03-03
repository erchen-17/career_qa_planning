"""
File converters: PDF → images, DOCX → text, image passthrough.
"""

from __future__ import annotations

import io
import logging

logger = logging.getLogger(__name__)


def pdf_extract_text(file_bytes: bytes, min_chars_per_page: int = 50) -> str | None:
    """
    尝试直接从 PDF 提取文字层文本。
    如果每页平均字符数 < min_chars_per_page，视为扫描件，返回 None。
    """
    import fitz  # PyMuPDF

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages_text: list[str] = []
    for page in doc:
        pages_text.append(page.get_text())
    doc.close()

    full_text = "\n\n".join(t.strip() for t in pages_text if t.strip())
    total_chars = sum(len(t.strip()) for t in pages_text)
    avg_chars = total_chars / max(len(pages_text), 1)

    if avg_chars < min_chars_per_page:
        logger.debug("PDF 文字层内容不足（平均 %.0f 字/页），回退到 OCR", avg_chars)
        return None

    logger.debug("PDF 直接提取文字：%d 页，共 %d 字", len(pages_text), total_chars)
    return full_text


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
        # 优先尝试直接提取文字层，失败再回退到渲染图片走 OCR
        text = pdf_extract_text(file_bytes)
        if text:
            return {"mode": "text", "text": text, "pages": 1}
        imgs = pdf_to_images(file_bytes, dpi=dpi)
        return {"mode": "images", "images": imgs, "pages": len(imgs)}

    if ext in ("docx", "doc"):
        text = docx_to_text(file_bytes)
        return {"mode": "text", "text": text, "pages": 1}

    raise ValueError(f"Unsupported file extension: .{ext}")
