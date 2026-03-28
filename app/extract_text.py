from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from .types import BBox, DocumentPage, TextSpan


def normalize_text(text: str) -> str:
    cleaned = unicodedata.normalize("NFKC", text)
    cleaned = cleaned.replace("\u00a0", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def uppercase_normalized(text: str) -> str:
    return normalize_text(text).upper()


def words_to_text_spans(words: list[tuple[float, float, float, float, str, int, int, int]]) -> list[TextSpan]:
    spans: list[TextSpan] = []
    for x0, y0, x1, y1, text, block_no, line_no, word_no in words:
        spans.append(
            TextSpan(
                text=str(text),
                bbox=BBox(x0=float(x0), y0=float(y0), x1=float(x1), y1=float(y1)),
                block_no=int(block_no),
                line_no=int(line_no),
                word_no=int(word_no),
            )
        )
    return spans


def join_page_text(page: DocumentPage) -> str:
    return "\n".join(span.text for span in page.text_spans)


def find_spans(page: DocumentPage, pattern: str) -> list[TextSpan]:
    regex = re.compile(pattern, re.IGNORECASE)
    return [span for span in page.text_spans if regex.search(span.text)]


def save_optional_ocr_text(render_path: Path) -> str:
    try:
        import pytesseract  # type: ignore
        from PIL import Image
    except ImportError:
        return ""

    try:
        image = Image.open(render_path)
        return pytesseract.image_to_string(image, lang="fin+eng")
    except Exception:
        return ""

