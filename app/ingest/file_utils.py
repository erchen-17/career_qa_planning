"""
文件类型检测、验证、标签解析工具函数。
"""

from __future__ import annotations

from fastapi import HTTPException

SUPPORTED_EXTS = {"pdf", "docx", "doc", "png", "jpg", "jpeg"}

_CONTENT_TYPE_MAP = {
    "application/pdf": "pdf",
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/msword": "doc",
}


def ext_from_name(name: str) -> str:
    """从文件名提取扩展名"""
    return name.rsplit(".", 1)[-1].lower() if "." in name else ""


def ext_from_url(url: str) -> str:
    """从 URL 路径提取扩展名"""
    path = url.split("?")[0].split("#")[0]
    last_segment = path.rsplit("/", 1)[-1]
    return last_segment.rsplit(".", 1)[-1].lower() if "." in last_segment else ""


def ext_from_content_type(content_type: str) -> str:
    """从 HTTP Content-Type 推断扩展名"""
    ct = content_type.split(";")[0].strip().lower()
    return _CONTENT_TYPE_MAP.get(ct, "")


def ext_from_magic_bytes(data: bytes) -> str:
    """从文件头 magic bytes 推断扩展名"""
    if data[:5] == b"%PDF-":
        return "pdf"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:2] == b"\xff\xd8":
        return "jpg"
    if data[:4] == b"PK\x03\x04":  # ZIP-based (DOCX)
        return "docx"
    return ""


def detect_file_ext(
    file_name: str,
    file_bytes: bytes | None = None,
    url: str | None = None,
    content_type: str | None = None,
) -> str:
    """
    多级文件类型检测链：文件名 → URL 路径 → Content-Type → magic bytes。
    返回检测到的扩展名（小写，不带点），未识别时返回空字符串。
    """
    ext = ext_from_name(file_name)
    if ext in SUPPORTED_EXTS:
        return ext

    if url:
        ext = ext_from_url(url)
        if ext in SUPPORTED_EXTS:
            return ext

    if content_type:
        ext = ext_from_content_type(content_type)
        if ext in SUPPORTED_EXTS:
            return ext

    if file_bytes:
        ext = ext_from_magic_bytes(file_bytes)
        if ext in SUPPORTED_EXTS:
            return ext

    return ext


def validate_file_ext(ext: str, file_name: str = "") -> None:
    """验证扩展名是否受支持，不支持则抛出 HTTPException 400。"""
    if ext not in SUPPORTED_EXTS:
        detail = "Unsupported file type"
        if file_name:
            detail += f" (file_name={file_name})"
        detail += ". Supported: pdf, docx, png, jpg"
        raise HTTPException(status_code=400, detail=detail)


def parse_tags(tags: str | None) -> list[str] | None:
    """解析逗号分隔的标签字符串，返回列表或 None。"""
    if not tags:
        return None
    result = [t.strip() for t in tags.split(",") if t.strip()]
    return result if result else None
