from __future__ import annotations

import math

import numpy as np

from .extract_dimensions import MeasurementStore, measurements_for_page
from .types import ClassificationResult, CladdingRegion, ConfidenceScores, FacadeModel, FootprintModel, Opening, ValidationSummary


def _best_page_id(classifications: list[ClassificationResult], role: str) -> str | None:
    best = None
    best_score = -1.0
    for classification in classifications:
        if classification.role != role:
            continue
        if classification.score > best_score:
            best = classification.page_id
            best_score = classification.score
    return best


def validate_takeoff(
    footprint: FootprintModel,
    facades: list[FacadeModel],
    openings: list[Opening],
    cladding_regions: list[CladdingRegion],
    measurements: MeasurementStore,
    classifications: list[ClassificationResult],
) -> ValidationSummary:
    assumptions: list[str] = []
    warnings: list[str] = []
    cross_checks: dict[str, float] = {}

    section_page_id = _best_page_id(classifications, "section")
    if section_page_id is not None:
        section_levels = sorted(
            measurement.value_m
            for measurement in measurements_for_page(measurements, section_page_id)
            if measurement.kind == "level_marker" and measurement.value_m > 0
        )
    else:
        section_levels = []

    long_facades = [facade for facade in facades if not facade.is_gable]
    gable_facades = [facade for facade in facades if facade.is_gable]
    mean_eave_height = float(np.mean([facade.total_height_m for facade in long_facades])) if long_facades else 0.0
    mean_ridge_height = float(np.mean([facade.total_height_m for facade in gable_facades])) if gable_facades else mean_eave_height

    if section_levels and mean_eave_height > 0:
        nearest_eave = min(section_levels, key=lambda value: abs(value - mean_eave_height))
        diff = abs(nearest_eave - mean_eave_height)
        cross_checks["section_eave_height_delta_m"] = diff
        if diff > max(0.15, mean_eave_height * 0.05):
            warnings.append(
                f"Section/elevation eave-height disagreement is {diff:.2f} m, which exceeds the tolerance."
            )
    if section_levels and mean_ridge_height > 0:
        nearest_ridge = min(section_levels, key=lambda value: abs(value - mean_ridge_height))
        diff = abs(nearest_ridge - mean_ridge_height)
        cross_checks["section_ridge_height_delta_m"] = diff
        if diff > max(0.15, mean_ridge_height * 0.05):
            warnings.append(
                f"Section/elevation ridge-height disagreement is {diff:.2f} m, which exceeds the tolerance."
            )

    gross_area = sum(facade.area_gross_m2 for facade in facades)
    openings_area = sum(opening.area_m2 for opening in openings)
    opening_ratio = openings_area / gross_area if gross_area > 0 else 0.0
    cross_checks["opening_area_ratio"] = opening_ratio
    if opening_ratio < 0.02:
        warnings.append("Detected openings cover less than 2% of facade area; opening subtraction may be incomplete.")
    if opening_ratio > 0.45:
        warnings.append("Detected openings cover more than 45% of facade area; review opening detection.")

    cladding_area = sum(region.area_m2 for region in cladding_regions)
    net_area = max(0.0, gross_area - openings_area)
    coverage_ratio = cladding_area / net_area if net_area > 0 else 0.0
    cross_checks["material_area_coverage_ratio"] = coverage_ratio
    if coverage_ratio < 0.85:
        warnings.append("Cladding labels cover less than 85% of the net facade area; material split is approximate.")

    geometry_conf = max(0.2, min(1.0, 0.55 * footprint.confidence + 0.45 * np.mean([facade.confidence for facade in facades])))
    openings_conf = max(0.15, min(1.0, 0.35 + np.mean([opening.confidence for opening in openings]) * 0.5 if openings else 0.25))
    materials_conf = max(0.15, min(1.0, 0.35 + coverage_ratio * 0.45))
    overall_conf = min(1.0, 0.5 * geometry_conf + 0.2 * openings_conf + 0.3 * materials_conf)

    confidence = ConfidenceScores(
        overall=float(overall_conf),
        geometry=float(geometry_conf),
        openings=float(openings_conf),
        materials=float(materials_conf),
    )
    return ValidationSummary(
        assumptions=assumptions,
        warnings=warnings,
        cross_checks=cross_checks,
        confidence=confidence,
    )

