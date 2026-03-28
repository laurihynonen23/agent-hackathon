from __future__ import annotations

import re
from collections import defaultdict

from .classify import classify_pages
from .extract_text import uppercase_normalized
from .types import BBox, ClassificationResult, DocumentPage, Measurement, MeasurementStore


SCALE_RE = re.compile(r"(\d+)\s*:\s*(\d+)")
LEVEL_RE = re.compile(r"^[+-]\d+(?:[.,]\d+)$")
DIM_RE = re.compile(r"^\d{4,5}$")
SIZE_RE = re.compile(r"(?P<thickness>\d+)\s*[xX]\s*(?P<cover>\d+)")


def infer_orientation(bbox: BBox) -> str:
    if bbox.height > bbox.width * 1.2:
        return "vertical"
    if bbox.width > bbox.height * 1.2:
        return "horizontal"
    return "unknown"


def parse_dimension_text(raw_text: str) -> tuple[str, float | None, dict[str, float | str]]:
    cleaned = uppercase_normalized(raw_text).replace(" ", "")
    meta: dict[str, float | str] = {}
    scale_match = SCALE_RE.search(cleaned)
    if scale_match:
        numerator = float(scale_match.group(1))
        denominator = float(scale_match.group(2))
        meta["scale_numerator"] = numerator
        meta["scale_denominator"] = denominator
        return "drawing_scale", denominator / numerator, meta

    if LEVEL_RE.fullmatch(cleaned):
        value = float(cleaned.replace(",", "."))
        meta["raw_value"] = value
        return "level_marker", value, meta

    if SIZE_RE.search(cleaned):
        size_match = SIZE_RE.search(cleaned)
        assert size_match is not None
        thickness = float(size_match.group("thickness"))
        cover = float(size_match.group("cover"))
        meta["thickness_mm"] = thickness
        meta["cover_mm"] = cover
        return "material_size", cover / 1000.0, meta

    if DIM_RE.fullmatch(cleaned):
        value_mm = float(cleaned)
        meta["raw_value_mm"] = value_mm
        return "dimension_mm", value_mm / 1000.0, meta

    if re.fullmatch(r"\d+(?:[.,]\d+)", cleaned):
        value = float(cleaned.replace(",", "."))
        meta["raw_value"] = value
        return "numeric_value", value, meta

    return "unknown", None, meta


def extract_measurements(
    pages: list[DocumentPage],
    classifications: list[ClassificationResult] | None = None,
) -> MeasurementStore:
    if classifications is None:
        classifications = classify_pages(pages)
    page_roles = {result.page_id: result.role for result in classifications}

    measurements: list[Measurement] = []
    for page in pages:
        role = page_roles.get(page.page_id, "unknown")
        for span in page.text_spans:
            kind, value_m, meta = parse_dimension_text(span.text)
            if value_m is None:
                continue

            confidence = 0.5
            tags = [role]
            orientation = infer_orientation(span.bbox)

            if kind == "drawing_scale":
                confidence = 0.95
                tags.append("scale")
            elif kind == "level_marker":
                confidence = 0.85
                tags.append("height")
            elif kind == "dimension_mm":
                confidence = 0.8
                tags.append("dimension")
            elif kind == "material_size":
                confidence = 0.9
                tags.extend(["material", "cover"])

            if role == "plan" and kind == "dimension_mm":
                if value_m > 4.0:
                    tags.append("overall_candidate")
            if role in {"section", "elevations"} and kind == "level_marker":
                tags.append("level_candidate")

            measurements.append(
                Measurement(
                    kind=kind,
                    value_m=value_m,
                    raw_text=span.text,
                    source_page_id=page.page_id,
                    bbox=span.bbox,
                    confidence=confidence,
                    orientation=orientation,  # type: ignore[arg-type]
                    tags=tags,
                    meta=meta,
                )
            )

    return MeasurementStore(measurements=measurements)


def strongest_measurements_by_tag(store: MeasurementStore, tag: str) -> list[Measurement]:
    return sorted(
        [measurement for measurement in store.measurements if tag in measurement.tags],
        key=lambda measurement: (measurement.confidence, measurement.value_m),
        reverse=True,
    )


def measurements_for_page(store: MeasurementStore, page_id: str) -> list[Measurement]:
    return [measurement for measurement in store.measurements if measurement.source_page_id == page_id]


def measurement_summary(store: MeasurementStore) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for measurement in store.measurements:
        counts[measurement.kind] += 1
    return dict(counts)
