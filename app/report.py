from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from .extract_dimensions import measurements_for_page
from .types import BBox, PipelineArtifacts


def write_json(path: Path, payload: object) -> None:
    def _default(value: object) -> object:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")  # type: ignore[no-any-return]
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Object of type {type(value)} is not JSON serializable")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_default), encoding="utf-8")


def _page_lookup(artifacts: PipelineArtifacts) -> dict[str, object]:
    return {page.page_id: page for page in artifacts.pages}


def _load_font() -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("Arial.ttf", 20)
    except Exception:
        return ImageFont.load_default()


def _scaled_box(box: BBox, image: Image.Image, width_pt: float, height_pt: float) -> tuple[float, float, float, float]:
    scale_x = image.width / width_pt
    scale_y = image.height / height_pt
    return (box.x0 * scale_x, box.y0 * scale_y, box.x1 * scale_x, box.y1 * scale_y)


def _draw_label(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, fill: str) -> None:
    font = _load_font()
    x, y = xy
    draw.rectangle((x - 3, y - 3, x + 8 + len(text) * 7, y + 18), fill="white")
    draw.text((x, y), text, fill=fill, font=font)


def _render_plan_overlay(output_dir: Path, artifacts: PipelineArtifacts) -> None:
    page = next(page for page in artifacts.pages if page.page_id == artifacts.footprint.source_page_id)
    image = Image.open(page.render_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    if artifacts.footprint.plan_bbox is not None:
        draw.rectangle(_scaled_box(artifacts.footprint.plan_bbox, image, page.width_pt, page.height_pt), outline="#cc3300", width=4)
    _draw_label(draw, (20, 20), f"Perimeter: {artifacts.results.perimeter_exterior_m:.2f} m", "#cc3300")
    _draw_label(draw, (20, 48), f"Footprint: {artifacts.footprint.width_m:.3f} x {artifacts.footprint.depth_m:.3f} m", "#cc3300")
    for measurement in measurements_for_page(artifacts.measurements, page.page_id):
        if measurement.kind != "dimension_mm" or measurement.value_m < 4.0:
            continue
        color = "#0b7285" if measurement.orientation == "horizontal" else "#5c940d"
        draw.rectangle(_scaled_box(measurement.bbox, image, page.width_pt, page.height_pt), outline=color, width=2)
    image.save(output_dir / "plan_overlay.png")


def _render_section_overlay(output_dir: Path, artifacts: PipelineArtifacts) -> None:
    section_pages = [page for page in artifacts.pages if any(result.page_id == page.page_id and result.role == "section" for result in artifacts.classifications)]
    if not section_pages:
        return
    page = section_pages[0]
    image = Image.open(page.render_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for measurement in measurements_for_page(artifacts.measurements, page.page_id):
        if measurement.kind != "level_marker":
            continue
        draw.rectangle(_scaled_box(measurement.bbox, image, page.width_pt, page.height_pt), outline="#6741d9", width=3)
        left, top, _, _ = _scaled_box(measurement.bbox, image, page.width_pt, page.height_pt)
        _draw_label(draw, (left + 4, top - 2), measurement.raw_text, "#6741d9")
    image.save(output_dir / "section_overlay.png")


def _render_elevation_overlay(output_dir: Path, artifacts: PipelineArtifacts) -> None:
    elevation_pages = [page for page in artifacts.pages if any(result.page_id == page.page_id and result.role == "elevations" for result in artifacts.classifications)]
    if not elevation_pages:
        return
    page = elevation_pages[0]
    image = Image.open(page.render_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for facade in artifacts.facades:
        if facade.source_page_id != page.page_id:
            continue
        box = _scaled_box(facade.cluster_box, image, page.width_pt, page.height_pt)
        draw.rectangle(box, outline="#e67700", width=4)
        _draw_label(
            draw,
            (box[0] + 6, max(6, box[1] + 6)),
            f"{facade.name}: {facade.width_m:.2f}m x {facade.total_height_m:.2f}m",
            "#e67700",
        )
    for opening in artifacts.openings:
        if opening.source_page_id != page.page_id:
            continue
        box = _scaled_box(opening.bbox_page, image, page.width_pt, page.height_pt)
        draw.rectangle(box, outline="#1971c2", width=3)
    material_colors = {"3a": "#2b8a3e", "3b": "#862e9c"}
    for region in artifacts.cladding_regions:
        matching_facade = next((facade for facade in artifacts.facades if facade.name == region.facade_name), None)
        if matching_facade is None:
            continue
        box = _scaled_box(matching_facade.cluster_box, image, page.width_pt, page.height_pt)
        color = material_colors.get(region.material_code, "#495057")
        _draw_label(draw, (box[0] + 6, box[1] + 26), f"{region.material_code}: {region.area_m2:.1f} m2", color)
    image.save(output_dir / "elevations_overlay.png")


def write_overlays(output_dir: Path, artifacts: PipelineArtifacts) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _render_plan_overlay(output_dir, artifacts)
    _render_section_overlay(output_dir, artifacts)
    _render_elevation_overlay(output_dir, artifacts)


def write_report(path: Path, artifacts: PipelineArtifacts) -> None:
    sheet_rows = []
    for classification in artifacts.classifications:
        sheet_rows.append(
            f"| {classification.file_name} | {classification.page_id} | {classification.role} | {classification.score:.2f} | {'; '.join(classification.reasons[:4])} |"
        )

    facade_rows = []
    for facade in artifacts.facades:
        facade_rows.append(
            f"| {facade.name} | {facade.width_m:.2f} | {facade.total_height_m:.2f} | {facade.area_gross_m2:.2f} | {'yes' if facade.is_gable else 'no'} |"
        )

    material_rows = []
    for code, quantity in sorted(artifacts.results.cladding_by_type.items()):
        material_rows.append(
            f"| {code} | {quantity.area_m2:.2f} | {quantity.linear_m_nominal_cover:.2f} | {quantity.assumed_nominal_cover_mm} |"
        )

    assumptions = "\n".join(f"- {item}" for item in artifacts.results.assumptions) or "- None"
    warnings = "\n".join(f"- {item}" for item in artifacts.results.warnings) or "- None"
    report = f"""# First Mate Estimator Report

## Final Quantities

- Exterior perimeter: **{artifacts.results.perimeter_exterior_m:.2f} m**
- Gross outer wall area: **{artifacts.results.gross_outer_wall_area_m2:.2f} m2**
- Openings area: **{artifacts.results.openings_area_m2:.2f} m2**
- Net cladding area: **{artifacts.results.net_cladding_area_m2:.2f} m2**

## Confidence

- Overall: {artifacts.results.confidence.overall:.2f}
- Geometry: {artifacts.results.confidence.geometry:.2f}
- Openings: {artifacts.results.confidence.openings:.2f}
- Materials: {artifacts.results.confidence.materials:.2f}

## Formulas

- `perimeter_exterior_m = footprint perimeter`
- `gross_outer_wall_area_m2 = sum(facade gross silhouette areas)`
- `openings_area_m2 = sum(detected windows and doors from elevations)`
- `net_cladding_area_m2 = gross_outer_wall_area_m2 - openings_area_m2`
- `linear_m_nominal_cover = area_m2 / (nominal_cover_mm / 1000)`

## Sheet Classification

| File | Page ID | Role | Score | Reasons |
| --- | --- | --- | --- | --- |
{chr(10).join(sheet_rows)}

## Facade Geometry

| Facade | Width (m) | Height (m) | Gross Area (m2) | Gable |
| --- | --- | --- | --- | --- |
{chr(10).join(facade_rows)}

## Cladding By Type

| Type | Area (m2) | Linear m | Nominal cover (mm) |
| --- | --- | --- | --- |
{chr(10).join(material_rows) if material_rows else '| None | 0.00 | 0.00 | 0 |'}

## Assumptions

{assumptions}

## Warnings

{warnings}

## Generated Files

- `results.json`
- `debug/*.json`
- `overlays/plan_overlay.png`
- `overlays/section_overlay.png`
- `overlays/elevations_overlay.png`
"""
    path.write_text(report, encoding="utf-8")

