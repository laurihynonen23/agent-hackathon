from __future__ import annotations

import json
from pathlib import Path

from .extract_text import save_optional_ocr_text, words_to_text_spans
from .types import BBox, DocumentPage, VectorPrimitive


def _require_fitz():
    try:
        import fitz  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency issue
        raise RuntimeError(
            "PyMuPDF is required. Install dependencies from requirements.txt before running the CLI."
        ) from exc
    return fitz


def discover_pdf_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.rglob("*.pdf") if path.is_file())


def summarize_drawings(drawings: list[dict]) -> list[VectorPrimitive]:
    primitives: list[VectorPrimitive] = []
    for drawing in drawings:
        rect = drawing.get("rect")
        if rect is None:
            continue
        items = drawing.get("items", [])
        primitive_type = "path"
        if items:
            primitive_type = str(items[0][0])
        primitives.append(
            VectorPrimitive(
                primitive_type=primitive_type,
                bbox=BBox(x0=float(rect.x0), y0=float(rect.y0), x1=float(rect.x1), y1=float(rect.y1)),
                stroke_width=float(drawing.get("width") or 0.0),
                fill=bool(drawing.get("fill")),
                item_count=len(items),
            )
        )
    return primitives


def ingest_documents(
    input_dir: Path,
    output_dir: Path,
    render_dpi: int = 200,
    ocr_mode: str = "auto",
) -> list[DocumentPage]:
    fitz = _require_fitz()
    pdf_files = discover_pdf_files(input_dir)
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found under {input_dir}")

    debug_dir = output_dir / "debug"
    render_dir = debug_dir / "rendered_pages"
    render_dir.mkdir(parents=True, exist_ok=True)

    pages: list[DocumentPage] = []
    ingest_debug: list[dict] = []
    scale = render_dpi / 72.0

    for pdf_path in pdf_files:
        document = fitz.open(pdf_path)
        for page_index in range(document.page_count):
            page = document[page_index]
            page_id = f"{pdf_path.stem.lower().replace(' ', '_')}_p{page_index + 1}"
            render_path = render_dir / f"{page_id}.png"
            pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            pixmap.save(render_path)

            raw_text = page.get_text("text")
            words = page.get_text("words")
            if not raw_text.strip() and ocr_mode in {"auto", "require"}:
                raw_text = save_optional_ocr_text(render_path)
                if ocr_mode == "require" and not raw_text.strip():
                    raise RuntimeError(f"OCR was required but no text could be extracted from {pdf_path.name}")

            drawings = page.get_drawings()
            document_page = DocumentPage(
                page_id=page_id,
                file_name=pdf_path.name,
                file_path=pdf_path,
                page_number=page_index + 1,
                width_pt=float(page.rect.width),
                height_pt=float(page.rect.height),
                render_path=render_path,
                raw_text=raw_text,
                text_spans=words_to_text_spans(words),
                vector_primitives=summarize_drawings(drawings),
            )
            pages.append(document_page)
            ingest_debug.append(
                {
                    "page_id": page_id,
                    "file_name": pdf_path.name,
                    "page_number": page_index + 1,
                    "size_pt": {"width": page.rect.width, "height": page.rect.height},
                    "text_spans": [span.model_dump(mode="json") for span in document_page.text_spans],
                    "vector_primitives": [primitive.model_dump(mode="json") for primitive in document_page.vector_primitives],
                }
            )

    (debug_dir / "ingest.json").write_text(
        json.dumps(ingest_debug, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return pages

