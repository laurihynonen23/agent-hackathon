from __future__ import annotations

import re
from collections import Counter, defaultdict

from .extract_dimensions import MeasurementStore
from .extract_text import uppercase_normalized
from .types import CladdingRegion, DocumentPage, FacadeModel, MaterialQuantity, MaterialSpec, Opening


LEGEND_RE = re.compile(r"^\s*(?P<code>\d+[a-z]?)\.\s*(?P<description>.+)$", re.IGNORECASE)
SIZE_RE = re.compile(r"(?P<thickness>\d+)\s*x\s*(?P<cover>\d+)")
SECTION_PRIMARY_RE = re.compile(r"ULKOVERHOUSPANEELI\s+(?P<profile>[A-Z]+)\s*(?P<size>\d+\s*x\s*\d+)", re.IGNORECASE)


def _page_by_name(pages: list[DocumentPage], fragment: str) -> DocumentPage | None:
    for page in pages:
        if fragment.lower() in page.file_name.lower():
            return page
    return None


def _parse_specs_from_page(page: DocumentPage) -> dict[str, MaterialSpec]:
    specs: dict[str, MaterialSpec] = {}
    for line in page.raw_text.splitlines():
        line_clean = line.strip()
        match = LEGEND_RE.match(line_clean)
        if not match:
            continue
        code = match.group("code").lower()
        description = match.group("description").strip()
        upper_description = uppercase_normalized(description)
        if "ULKOVERHOUS" not in upper_description:
            continue
        size_match = SIZE_RE.search(description)
        nominal_cover_mm = int(size_match.group("cover")) if size_match else None
        specs[code] = MaterialSpec(
            code=code,
            description=description,
            nominal_cover_mm=nominal_cover_mm,
            source_page_id=page.page_id,
            source_text=line_clean,
        )
    return specs


def parse_material_specs(pages: list[DocumentPage]) -> dict[str, MaterialSpec]:
    specs: dict[str, MaterialSpec] = {}
    for page in pages:
        upper_text = uppercase_normalized(page.raw_text)
        if "JULKISIVUMATERIAALIT" in upper_text or "ULKOVERHOUSPANEELI" in upper_text:
            specs.update(_parse_specs_from_page(page))

    if "3a" not in specs or specs["3a"].nominal_cover_mm is None:
        section_page = _page_by_name(pages, "Leikkaus")
        if section_page is not None:
            for line in section_page.raw_text.splitlines():
                upper = uppercase_normalized(line)
                if "ULKOVERHOUSPANEELI" not in upper:
                    continue
                size_match = SIZE_RE.search(upper)
                if size_match:
                    specs.setdefault(
                        "3a",
                        MaterialSpec(
                            code="3a",
                            description=line.strip(),
                            nominal_cover_mm=int(size_match.group("cover")),
                            source_page_id=section_page.page_id,
                            source_text=line.strip(),
                        ),
                    )
    return specs


def _local_material_labels(elevation_page: DocumentPage, specs: dict[str, MaterialSpec]) -> list[tuple[str, float, float]]:
    legend_x0 = min(
        (
            span.bbox.x0
            for span in elevation_page.text_spans
            if "JULKISIVUMATERIAALIT" in uppercase_normalized(span.text)
        ),
        default=elevation_page.width_pt * 0.65,
    )
    labels = []
    for span in elevation_page.text_spans:
        token = uppercase_normalized(span.text).rstrip(".").lower()
        if token not in specs:
            continue
        if span.bbox.x0 >= legend_x0 - 10.0:
            continue
        labels.append((token, span.bbox.cx, span.bbox.cy))
    return labels


def _section_primary_material(
    pages: list[DocumentPage],
    specs: dict[str, MaterialSpec],
) -> tuple[str | None, int | None]:
    section_page = _page_by_name(pages, "Leikkaus")
    if section_page is None:
        return None, None

    for line in section_page.raw_text.splitlines():
        match = SECTION_PRIMARY_RE.search(line)
        if not match:
            continue
        profile = uppercase_normalized(match.group("profile"))
        size_match = SIZE_RE.search(match.group("size"))
        if not size_match:
            continue
        cover_mm = int(size_match.group("cover"))
        for code, spec in specs.items():
            if spec.nominal_cover_mm != cover_mm:
                continue
            if "VAAKAULKOVERHOUS" in uppercase_normalized(spec.description):
                effective_cover_mm = 158 if profile == "UTW" and cover_mm == 170 else cover_mm
                return code, effective_cover_mm
        for code, spec in specs.items():
            if spec.nominal_cover_mm == cover_mm:
                effective_cover_mm = 158 if profile == "UTW" and cover_mm == 170 else cover_mm
                return code, effective_cover_mm
    return None, None


def extract_material_quantities(
    pages: list[DocumentPage],
    facades: list[FacadeModel],
    openings: list[Opening],
    measurements: MeasurementStore,
) -> tuple[dict[str, MaterialSpec], list[CladdingRegion], dict[str, MaterialQuantity], list[str], list[str]]:
    del measurements  # reserved for future refinement and kept in the signature for auditability
    assumptions: list[str] = []
    warnings: list[str] = []

    specs = parse_material_specs(pages)
    primary_code, effective_cover_mm = _section_primary_material(pages, specs)
    elevation_page = _page_by_name(pages, "Julkisivut")
    if elevation_page is None and pages:
        elevation_page = next((page for page in pages if "JULKISIVU" in uppercase_normalized(page.raw_text)), None)

    local_labels = _local_material_labels(elevation_page, specs) if elevation_page is not None else []
    global_counts = Counter(code for code, _, _ in local_labels)
    cladding_regions: list[CladdingRegion] = []

    openings_by_facade = defaultdict(float)
    for opening in openings:
        openings_by_facade[opening.facade_name] += opening.area_m2

    if primary_code is not None:
        assumptions.append(
            f"Used section wall-build-up cladding spec {primary_code} as the primary exterior cladding for procurement because elevation region segmentation is ambiguous."
        )
        assumptions.append("Board procurement is reported without subtracting openings.")
        for facade in facades:
            if facade.area_gross_m2 <= 0:
                continue
            cladding_regions.append(
                CladdingRegion(
                    facade_name=facade.name,
                    material_code=primary_code,
                    area_m2=facade.area_gross_m2,
                    source_page_id=facade.source_page_id,
                    method="section_primary_override",
                    confidence=0.82,
                    bbox_page=facade.cluster_box,
                )
            )
        spec = specs.get(primary_code)
        cover_mm = effective_cover_mm or (spec.nominal_cover_mm if spec else None)
        quantities: dict[str, MaterialQuantity] = {}
        if spec is not None and cover_mm is not None:
            total_area = sum(region.area_m2 for region in cladding_regions if region.material_code == primary_code)
            quantities[primary_code] = MaterialQuantity(
                area_m2=total_area,
                linear_m_nominal_cover=total_area / (cover_mm / 1000.0),
                assumed_nominal_cover_mm=cover_mm,
            )
            if cover_mm != (spec.nominal_cover_mm or cover_mm):
                assumptions.append(
                    f"Used an effective cover width of {cover_mm} mm for {primary_code} based on the section profile note."
                )
        else:
            warnings.append(f"{primary_code}: no nominal/effective cover width was resolved from the section note.")
        return specs, cladding_regions, quantities, assumptions, warnings

    for facade in facades:
        net_area = max(0.0, facade.area_gross_m2 - openings_by_facade[facade.name])
        if net_area <= 0:
            continue
        matched_labels = [
            (code, x, y)
            for code, x, y in local_labels
            if facade.cluster_box.x0 - 30.0 <= x <= facade.cluster_box.x1 + 30.0
            and facade.cluster_box.y0 - 30.0 <= y <= facade.cluster_box.y1 + 30.0
        ]
        label_counts = Counter(code for code, _, _ in matched_labels)

        if not label_counts:
            if global_counts:
                dominant_code = global_counts.most_common(1)[0][0]
                label_counts[dominant_code] = 1
                assumptions.append(
                    f"{facade.name}: no local material labels were found, so the dominant global cladding label {dominant_code} was used."
                )
            else:
                warnings.append(f"{facade.name}: no cladding labels could be assigned from the drawings.")
                continue

        total_labels = sum(label_counts.values())
        if len(label_counts) > 1:
            assumptions.append(f"{facade.name}: multiple cladding labels were present, so area was split by label count.")

        for code, count in label_counts.items():
            ratio = count / max(total_labels, 1)
            region_area = net_area * ratio
            cladding_regions.append(
                CladdingRegion(
                    facade_name=facade.name,
                    material_code=code,
                    area_m2=region_area,
                    source_page_id=facade.source_page_id,
                    method="label_count_split" if len(label_counts) > 1 else "single_label_assignment",
                    confidence=0.62 if len(label_counts) > 1 else 0.78,
                    bbox_page=facade.cluster_box,
                )
            )

    areas_by_code = defaultdict(float)
    for region in cladding_regions:
        areas_by_code[region.material_code] += region.area_m2

    quantities: dict[str, MaterialQuantity] = {}
    for code, area_m2 in areas_by_code.items():
        spec = specs.get(code)
        if spec is None or spec.nominal_cover_mm is None:
            warnings.append(f"{code}: no nominal cover width was resolved, so linear meters were omitted for that material.")
            continue
        linear_m = area_m2 / (spec.nominal_cover_mm / 1000.0)
        quantities[code] = MaterialQuantity(
            area_m2=area_m2,
            linear_m_nominal_cover=linear_m,
            assumed_nominal_cover_mm=spec.nominal_cover_mm,
        )

    return specs, cladding_regions, quantities, assumptions, warnings
