from pathlib import Path

from app.classify import classify_page
from app.types import BBox, DocumentPage, TextSpan


def make_page(name: str, raw_text: str) -> DocumentPage:
    return DocumentPage(
        page_id=name,
        file_name=name,
        file_path=Path(name),
        page_number=1,
        width_pt=1000,
        height_pt=700,
        render_path=Path(f"{name}.png"),
        raw_text=raw_text,
        text_spans=[
            TextSpan(text=token, bbox=BBox(x0=0, y0=0, x1=10, y1=10))
            for token in raw_text.split()
        ],
        vector_primitives=[],
    )


def test_classify_plan_page():
    result = classify_page(make_page("plan.pdf", "POHJA 1 : 50 PARVEN POHJA"))
    assert result.role == "plan"


def test_classify_section_page():
    result = classify_page(make_page("section.pdf", "LEIKKAUS B - B"))
    assert result.role == "section"


def test_classify_elevation_page():
    result = classify_page(make_page("elevations.pdf", "JULKISIVUMATERIAALIT JULKISIVU JULKISIVU JULKISIVU"))
    assert result.role == "elevations"

