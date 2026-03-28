from __future__ import annotations

from collections import Counter
import re

from .extract_text import uppercase_normalized
from .types import BBox, ClassificationResult, DocumentPage


PLAN_HINTS = ("POHJA", "PARVEN POHJA", "FLOOR PLAN")
SECTION_HINTS = ("LEIKKAUS", "SECTION")
ELEVATION_HINTS = ("JULKISIVU", "JULKISIVUMATERIAALIT", "ELEVATION")


def _contains_hint(text: str, hint: str) -> bool:
    pattern = r"(?<![A-ZÅÄÖ0-9])" + re.escape(hint) + r"(?![A-ZÅÄÖ0-9])"
    return re.search(pattern, text) is not None


def _label_clusters(page: DocumentPage, needle: str) -> list[BBox]:
    matches = [span.bbox for span in page.text_spans if needle in uppercase_normalized(span.text)]
    return matches


def classify_page(page: DocumentPage) -> ClassificationResult:
    text = uppercase_normalized(page.raw_text)
    reasons: list[str] = []
    scores = Counter(plan=0.0, section=0.0, elevations=0.0, unknown=0.0)

    for hint in PLAN_HINTS:
        if _contains_hint(text, hint):
            scores["plan"] += 2.0
            reasons.append(f"text contains {hint}")
    for hint in SECTION_HINTS:
        if _contains_hint(text, hint):
            scores["section"] += 2.0
            reasons.append(f"text contains {hint}")
    for hint in ELEVATION_HINTS:
        if _contains_hint(text, hint):
            scores["elevations"] += 2.0
            reasons.append(f"text contains {hint}")

    name_text = uppercase_normalized(page.file_name)
    if "POHJA" in name_text:
        scores["plan"] += 1.0
        reasons.append("filename suggests plan")
    if "LEIKKAUS" in name_text:
        scores["section"] += 1.0
        reasons.append("filename suggests section")
    if "JULKISIVU" in name_text:
        scores["elevations"] += 1.0
        reasons.append("filename suggests elevations")

    facade_labels = [
        span
        for span in page.text_spans
        if uppercase_normalized(span.text) == "JULKISIVU"
    ]
    if len(facade_labels) >= 3:
        scores["elevations"] += 3.0
        reasons.append("multiple facade labels detected")

    plan_labels = [
        span
        for span in page.text_spans
        if uppercase_normalized(span.text) == "POHJA"
    ]
    parvi_labels = [
        span
        for span in page.text_spans
        if uppercase_normalized(span.text) == "PARVEN"
    ]
    if plan_labels:
        scores["plan"] += 1.0
    if parvi_labels:
        scores["plan"] += 0.5
        reasons.append("loft plan text detected")

    role, score = max(scores.items(), key=lambda item: item[1])
    if score <= 0:
        role = "unknown"
        score = 0.0

    subviews: dict[str, BBox] = {}
    if role == "plan":
        if plan_labels:
            subviews["plan_anchor"] = plan_labels[0].bbox
        if parvi_labels:
            subviews["loft_anchor"] = parvi_labels[0].bbox
    elif role == "elevations":
        for index, label in enumerate(sorted(facade_labels, key=lambda item: (item.bbox.cy, item.bbox.cx)), start=1):
            subviews[f"facade_label_{index}"] = label.bbox

    return ClassificationResult(
        page_id=page.page_id,
        file_name=page.file_name,
        role=role,  # type: ignore[arg-type]
        score=min(1.0, score / 7.0),
        reasons=reasons,
        subview_boxes=subviews,
    )


def classify_pages(pages: list[DocumentPage]) -> list[ClassificationResult]:
    return [classify_page(page) for page in pages]
