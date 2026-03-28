from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .classify import classify_pages
from .extract_dimensions import extract_measurements, measurement_summary
from .geometry import reconstruct_geometry
from .ingest import ingest_documents
from .materials import extract_material_quantities
from .openings import extract_openings
from .report import write_json, write_overlays, write_report
from .types import ConfidenceScores, PipelineArtifacts, Results
from .validate import validate_takeoff


ProgressCallback = Callable[[dict[str, Any]], None]


def _emit(
    progress_callback: ProgressCallback | None,
    stage: str,
    status: str,
    summary: str,
    details: dict[str, Any] | None = None,
) -> None:
    if progress_callback is None:
        return
    payload = {
        "stage": stage,
        "status": status,
        "summary": summary,
        "details": details or {},
    }
    progress_callback(payload)


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    render_dpi: int = 200,
    ocr_mode: str = "auto",
    report_format: str = "md",
    progress_callback: ProgressCallback | None = None,
) -> PipelineArtifacts:
    if report_format != "md":
        raise ValueError("Only Markdown reporting is implemented in this version.")

    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    _emit(progress_callback, "ingest", "running", "Loading PDFs and extracting text/vector data.")
    pages = ingest_documents(input_dir=input_dir, output_dir=output_dir, render_dpi=render_dpi, ocr_mode=ocr_mode)
    _emit(
        progress_callback,
        "ingest",
        "completed",
        f"Ingested {len(pages)} page(s).",
        {
            "pdf_count": len({page.file_name for page in pages}),
            "page_count": len(pages),
            "text_span_count": sum(len(page.text_spans) for page in pages),
            "vector_primitive_count": sum(len(page.vector_primitives) for page in pages),
        },
    )

    _emit(progress_callback, "classify", "running", "Classifying sheets into plan, section, elevations, or unknown.")
    classifications = classify_pages(pages)
    _emit(
        progress_callback,
        "classify",
        "completed",
        "Sheet roles assigned.",
        {
            "pages": [
                {
                    "file_name": result.file_name,
                    "page_id": result.page_id,
                    "role": result.role,
                    "score": result.score,
                    "reasons": result.reasons[:4],
                }
                for result in classifications
            ]
        },
    )

    _emit(progress_callback, "extract", "running", "Extracting dimensions, level markers, and material sizes.")
    measurements = extract_measurements(pages, classifications)
    _emit(
        progress_callback,
        "extract",
        "completed",
        f"Captured {len(measurements.measurements)} normalized measurements.",
        {"measurement_counts": measurement_summary(measurements)},
    )

    _emit(progress_callback, "geometry", "running", "Reconstructing footprint and facade geometry.")
    footprint, facades, geometry_assumptions, geometry_warnings = reconstruct_geometry(pages, classifications, measurements)
    _emit(
        progress_callback,
        "geometry",
        "completed",
        "Footprint and facade geometry reconstructed.",
        {
            "perimeter_m": footprint.perimeter_m,
            "width_m": footprint.width_m,
            "depth_m": footprint.depth_m,
            "facades": [
                {
                    "name": facade.name,
                    "width_m": facade.width_m,
                    "height_m": facade.total_height_m,
                    "gross_area_m2": facade.area_gross_m2,
                    "is_gable": facade.is_gable,
                }
                for facade in facades
            ],
            "warnings": geometry_warnings,
        },
    )

    _emit(progress_callback, "openings", "running", "Detecting openings from facade geometry.")
    openings, opening_warnings = extract_openings(pages, classifications, facades)
    _emit(
        progress_callback,
        "openings",
        "completed",
        f"Detected {len(openings)} opening(s).",
        {
            "opening_count": len(openings),
            "openings_area_m2": sum(opening.area_m2 for opening in openings),
            "warnings": opening_warnings,
        },
    )

    _emit(progress_callback, "materials", "running", "Assigning cladding labels and material quantities.")
    material_specs, cladding_regions, cladding_by_type, material_assumptions, material_warnings = extract_material_quantities(
        pages=pages,
        facades=facades,
        openings=openings,
        measurements=measurements,
    )
    _emit(
        progress_callback,
        "materials",
        "completed",
        "Material split computed.",
        {
            "specs": {code: spec.description for code, spec in material_specs.items()},
            "quantities": {code: quantity.model_dump(mode="json") for code, quantity in cladding_by_type.items()},
            "warnings": material_warnings,
        },
    )

    _emit(progress_callback, "validate", "running", "Cross-checking geometry, openings, and material coverage.")
    validation = validate_takeoff(
        footprint=footprint,
        facades=facades,
        openings=openings,
        cladding_regions=cladding_regions,
        measurements=measurements,
        classifications=classifications,
    )
    _emit(
        progress_callback,
        "validate",
        "completed",
        "Confidence and warnings finalized.",
        {
            "confidence": validation.confidence.model_dump(mode="json"),
            "warnings": validation.warnings,
            "cross_checks": validation.cross_checks,
        },
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

    _emit(progress_callback, "report", "running", "Rendering report and annotated overlays.")
    write_report(output_dir / "report.md", artifacts)
    write_overlays(output_dir / "overlays", artifacts)
    _emit(
        progress_callback,
        "report",
        "completed",
        "Artifacts written to disk.",
        {
            "results_path": str(output_dir / "results.json"),
            "report_path": str(output_dir / "report.md"),
            "overlay_dir": str(output_dir / "overlays"),
        },
    )
    return artifacts
