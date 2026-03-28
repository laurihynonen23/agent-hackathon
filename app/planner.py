from __future__ import annotations

from pathlib import Path

from .classify import classify_pages
from .extract_dimensions import extract_measurements
from .geometry import reconstruct_geometry
from .ingest import ingest_documents
from .materials import extract_material_quantities
from .openings import extract_openings
from .report import write_json, write_overlays, write_report
from .types import ConfidenceScores, PipelineArtifacts, Results
from .validate import validate_takeoff


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    render_dpi: int = 200,
    ocr_mode: str = "auto",
    report_format: str = "md",
) -> PipelineArtifacts:
    if report_format != "md":
        raise ValueError("Only Markdown reporting is implemented in this version.")

    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    pages = ingest_documents(input_dir=input_dir, output_dir=output_dir, render_dpi=render_dpi, ocr_mode=ocr_mode)
    classifications = classify_pages(pages)
    measurements = extract_measurements(pages, classifications)
    footprint, facades, geometry_assumptions, geometry_warnings = reconstruct_geometry(pages, classifications, measurements)
    openings, opening_warnings = extract_openings(pages, classifications, facades)
    material_specs, cladding_regions, cladding_by_type, material_assumptions, material_warnings = extract_material_quantities(
        pages=pages,
        facades=facades,
        openings=openings,
        measurements=measurements,
    )
    validation = validate_takeoff(
        footprint=footprint,
        facades=facades,
        openings=openings,
        cladding_regions=cladding_regions,
        measurements=measurements,
        classifications=classifications,
    )

    perimeter_exterior_m = footprint.perimeter_m
    gross_outer_wall_area_m2 = sum(facade.area_gross_m2 for facade in facades)
    openings_area_m2 = sum(opening.area_m2 for opening in openings)
    net_cladding_area_m2 = max(0.0, gross_outer_wall_area_m2 - openings_area_m2)

    assumptions = geometry_assumptions + material_assumptions + validation.assumptions
    warnings = geometry_warnings + opening_warnings + material_warnings + validation.warnings

    results = Results(
        perimeter_exterior_m=perimeter_exterior_m,
        gross_outer_wall_area_m2=gross_outer_wall_area_m2,
        openings_area_m2=openings_area_m2,
        net_cladding_area_m2=net_cladding_area_m2,
        cladding_by_type=cladding_by_type,
        assumptions=assumptions,
        warnings=warnings,
        confidence=ConfidenceScores(
            overall=validation.confidence.overall,
            geometry=validation.confidence.geometry,
            openings=validation.confidence.openings,
            materials=validation.confidence.materials,
        ),
    )

    artifacts = PipelineArtifacts(
        pages=pages,
        classifications=classifications,
        measurements=measurements,
        footprint=footprint,
        facades=facades,
        openings=openings,
        cladding_regions=cladding_regions,
        material_specs=material_specs,
        validation=validation,
        results=results,
    )

    write_json(debug_dir / "classification.json", [item.model_dump(mode="json") for item in classifications])
    write_json(debug_dir / "measurements.json", measurements.model_dump(mode="json"))
    write_json(
        debug_dir / "geometry.json",
        {
            "footprint": footprint.model_dump(mode="json"),
            "facades": [facade.model_dump(mode="json") for facade in facades],
            "assumptions": geometry_assumptions,
            "warnings": geometry_warnings,
        },
    )
    write_json(debug_dir / "openings.json", [item.model_dump(mode="json") for item in openings])
    write_json(
        debug_dir / "materials.json",
        {
            "specs": {code: spec.model_dump(mode="json") for code, spec in material_specs.items()},
            "regions": [region.model_dump(mode="json") for region in cladding_regions],
            "quantities": {code: quantity.model_dump(mode="json") for code, quantity in cladding_by_type.items()},
            "warnings": material_warnings,
            "assumptions": material_assumptions,
        },
    )
    write_json(debug_dir / "validation.json", validation.model_dump(mode="json"))
    write_json(output_dir / "results.json", results.model_dump(mode="json"))

    write_report(output_dir / "report.md", artifacts)
    write_overlays(output_dir / "overlays", artifacts)
    return artifacts

