from pathlib import Path

from app.materials import parse_material_specs
from app.types import DocumentPage


def test_parse_material_specs_from_legend():
    page = DocumentPage(
        page_id="elev",
        file_name="ARK 03 Julkisivut.pdf",
        file_path=Path("ARK 03 Julkisivut.pdf"),
        page_number=1,
        width_pt=100,
        height_pt=100,
        render_path=Path("elev.png"),
        raw_text="\n".join(
            [
                "JULKISIVUMATERIAALIT:",
                "3a. VAAKAULKOVERHOUSPANEELI, 28x170, tumma harmaa",
                "3b. ULKOVERHOUSPANEELI, 21x95, kuultokasitelty ruskea",
            ]
        ),
        text_spans=[],
        vector_primitives=[],
    )
    specs = parse_material_specs([page])
    assert specs["3a"].nominal_cover_mm == 170
    assert specs["3b"].nominal_cover_mm == 95

