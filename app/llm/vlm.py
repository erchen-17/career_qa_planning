"""
VLM-based OCR: extract text from images using multimodal LLMs.
"""

from __future__ import annotations

import base64
import logging
from langchain_core.messages import HumanMessage

from app.llm.router import Provider, get_vlm_model

logger = logging.getLogger(__name__)

# OCR instruction (强约束 prompt)
OCR_INSTRUCTION = (
    "请从图片中尽可能完整、忠实地提取所有可见文字。"
    "保持原有段落与列表结构。"
    "仅输出提取出的文字，不要添加任何解释或总结。"
    "表格请按行输出或用 Markdown 表格表示。"
    "无法辨认处用 [UNK] 标注。"
)


def _encode_image(image_bytes: bytes) -> str:
    """Base64-encode an image for multimodal API calls."""
    return base64.b64encode(image_bytes).decode("utf-8")


async def extract_text_from_image(
    image_bytes: bytes,
    provider: Provider,
    model: str,
) -> str:
    """
    Send a single image to a VLM and extract text via OCR prompt.
    Returns the extracted text string.
    """
    llm = get_vlm_model(provider=provider, model=model)
    b64 = _encode_image(image_bytes)

    message = HumanMessage(
        content=[
            {"type": "text", "text": OCR_INSTRUCTION},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            },
        ]
    )

    response = await llm.ainvoke([message])
    text = response.content
    logger.debug("VLM OCR extracted %d chars from image", len(text))
    return text


async def extract_text_from_images(
    images: list[bytes],
    provider: Provider,
    model: str,
) -> list[str]:
    """
    Process multiple page images sequentially.
    Returns a list of extracted text strings (one per page).
    """
    results: list[str] = []
    for i, img_bytes in enumerate(images):
        logger.info("OCR processing page %d/%d ...", i + 1, len(images))
        text = await extract_text_from_image(img_bytes, provider, model)
        results.append(text)
    return results
