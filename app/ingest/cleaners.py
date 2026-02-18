"""
Text cleaning utilities for post-OCR processing.
"""

from __future__ import annotations

import re


def clean_text(raw: str) -> str:
    """
    Clean OCR-extracted or document text:
    - Remove excessive blank lines
    - Merge broken lines (Chinese and English)
    - Strip leading/trailing whitespace per line
    """
    lines = raw.splitlines()

    # Strip each line
    lines = [line.strip() for line in lines]

    # Remove page numbers (standalone digits, optionally with dashes)
    lines = [line for line in lines if not re.match(r"^[-–—]?\s*\d+\s*[-–—]?$", line)]

    # Merge broken Chinese lines:
    # If a line ends with a Chinese character and the next starts with one,
    # join without space.
    merged: list[str] = []
    for line in lines:
        if not line:
            merged.append("")
            continue

        if merged and merged[-1] and not merged[-1].endswith("\n"):
            prev = merged[-1]
            # Chinese character at end of prev + start of current → merge directly
            if (
                prev
                and re.search(r"[\u4e00-\u9fff]$", prev)
                and re.match(r"[\u4e00-\u9fff]", line)
            ):
                merged[-1] = prev + line
                continue
            # English word continuation (prev doesn't end with punctuation)
            if prev and re.search(r"[a-zA-Z]$", prev) and re.match(r"[a-zA-Z]", line):
                merged[-1] = prev + " " + line
                continue

        merged.append(line)

    # Collapse multiple blank lines into at most two newlines (paragraph break)
    text = "\n".join(merged)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
