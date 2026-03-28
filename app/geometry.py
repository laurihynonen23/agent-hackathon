from __future__ import annotations

from collections import defaultdict, Counter
import re

import numpy as np
from shapely.geometry import Polygon
from PIL import Image

from .extract_dimensions import measurements_for_page
from .extract_text import uppercase_normalized
from .types import BBox, ClassificationResult, DocumentPage, FacadeModel, FootprintModel, MeasurementStore, Point2D


def _page_for_role(
    pages: list[DocumentPage],
    classifications: list[ClassificationResult],
    role: str,
) -> DocumentPage | None:
    best = None
    best_score = -1.0
    pages_by_id = {page.page_id: page for page in pages}
    for classification in classifications:
        if classification.role != role:
            continue
        if classification.score > best_score:
            best = pages_by_id.get(classification.page_id)
            best_score = classification.score
    return best


def _detect_split_boundary(page: DocumentPage) -> float | None:
    centers = sorted(
        primitive.bbox.cx
        for primitive in page.vector_primitives
        if primitive.bbox.area > 100.0
        and primitive.bbox.width < page.width_pt * 0.9
        and primitive.bbox.height < page.height_pt * 0.9
    )
    if len(centers) < 10:
        return None
    gaps = []
    for left, right in zip(centers, centers[1:]):
        gap = right - left
        if gap > page.width_pt * 0.06:
            gaps.append((gap, (left + right) / 2.0))
    if not gaps:
        return None
    gaps.sort(reverse=True)
    return gaps[0][1]


def _detect_split_boundary_from_raster(page: DocumentPage) -> float | None:
    try:
        import cv2  # type: ignore
    except ImportError:
        cv2 = None  # type: ignore[assignment]

    image = Image.open(page.render_path).convert("L")
    if cv2 is not None:
        array = np.array(image)
        _, binary = cv2.threshold(array, 240, 255, cv2.THRESH_BINARY_INV)
        projection = binary.sum(axis=0)
    else:
        array = np.array(image)
        binary = (array < 240).astype(np.uint8)
        projection = binary.sum(axis=0)

    quiet_columns = np.where(projection < np.percentile(projection, 15))[0]
    if len(quiet_columns) == 0:
        return None
    gaps = np.split(quiet_columns, np.where(np.diff(quiet_columns) != 1)[0] + 1)
    if not gaps:
        return None
    widest = max(gaps, key=len)
    midpoint_px = float(widest[len(widest) // 2])
    return midpoint_px * page.width_pt / image.width


def detect_primary_plan_region(page: DocumentPage) -> BBox:
    poja_spans = [span for span in page.text_spans if uppercase_normalized(span.text) == "POHJA"]
    parvi_spans = [span for span in page.text_spans if uppercase_normalized(span.text) == "PARVEN"]
    split_boundary = _detect_split_boundary(page)

    if poja_spans and split_boundary is not None:
        main_anchor = min(poja_spans, key=lambda span: span.bbox.x0)
        if main_anchor.bbox.cx < split_boundary:
            return BBox(x0=0.0, y0=0.0, x1=split_boundary, y1=page.height_pt)
        return BBox(x0=split_boundary, y0=0.0, x1=page.width_pt, y1=page.height_pt)

    if poja_spans:
        raster_boundary = _detect_split_boundary_from_raster(page)
        if raster_boundary is not None:
            main_anchor = min(poja_spans, key=lambda span: span.bbox.x0)
            if main_anchor.bbox.cx < raster_boundary:
                return BBox(x0=0.0, y0=0.0, x1=raster_boundary, y1=page.height_pt)
            return BBox(x0=raster_boundary, y0=0.0, x1=page.width_pt, y1=page.height_pt)

    if poja_spans and parvi_spans:
        main_anchor = min(poja_spans, key=lambda span: span.bbox.x0)
        loft_anchor = max(parvi_spans, key=lambda span: span.bbox.x0)
        boundary = max(main_anchor.bbox.x1 + 200.0, loft_anchor.bbox.x0 - 100.0)
        return BBox(x0=0.0, y0=0.0, x1=min(boundary, page.width_pt), y1=page.height_pt)

    return BBox(x0=0.0, y0=0.0, x1=page.width_pt, y1=page.height_pt)


def _largest_dimension_by_orientation(
    store: MeasurementStore,
    page_id: str,
    plan_region: BBox,
    orientation: str,
) -> float | None:
    candidates = []
    for measurement in measurements_for_page(store, page_id):
        if measurement.kind != "dimension_mm":
            continue
        if measurement.orientation != orientation:
            continue
        if measurement.value_m < 4.0:
            continue
        if not measurement.bbox.intersects(plan_region.expand(220.0)):
            continue
        candidates.append(measurement.value_m)
    if not candidates:
        return None
    return max(candidates)


def _chain_dimension_sum(
    store: MeasurementStore,
    page_id: str,
    plan_region: BBox,
    orientation: str,
) -> float | None:
    bucket_size = 15.0
    groups: dict[int, list[float]] = defaultdict(list)
    for measurement in measurements_for_page(store, page_id):
        if measurement.kind != "dimension_mm":
            continue
        if measurement.orientation != orientation:
            continue
        if measurement.value_m < 0.9:
            continue
        if not measurement.bbox.intersects(plan_region.expand(260.0)):
            continue
        anchor = measurement.bbox.y0 if orientation == "horizontal" else measurement.bbox.x0
        bucket = int(round(anchor / bucket_size))
        groups[bucket].append(measurement.value_m)

    chain_sums = [
        round(sum(values), 3)
        for values in groups.values()
        if len(values) >= 2
    ]
    if not chain_sums:
        return None
    return max(chain_sums)


def _fallback_dimensions_from_any(store: MeasurementStore, page_id: str) -> tuple[float | None, float | None]:
    candidates = sorted(
        {
            round(measurement.value_m, 3)
            for measurement in measurements_for_page(store, page_id)
            if measurement.kind == "dimension_mm" and measurement.value_m >= 4.0
        },
        reverse=True,
    )
    if len(candidates) >= 2:
        return min(candidates[:2]), max(candidates[:2])
    if len(candidates) == 1:
        return candidates[0], candidates[0]
    return None, None


def build_rectangular_footprint(width_m: float, depth_m: float) -> Polygon:
    return Polygon([(0.0, 0.0), (width_m, 0.0), (width_m, depth_m), (0.0, depth_m)])


def gable_wall_area(width_m: float, eave_height_m: float, total_height_m: float) -> float:
    ridge_extra = max(0.0, total_height_m - eave_height_m)
    return width_m * eave_height_m + 0.5 * width_m * ridge_extra


def _preferred_wall_height_m(
    measurements: MeasurementStore,
    candidate_page_ids: list[str],
) -> float | None:
    values: list[float] = []
    for page_id in candidate_page_ids:
        for measurement in measurements_for_page(measurements, page_id):
            if measurement.kind != "level_marker":
                continue
            if measurement.value_m < 2.5 or measurement.value_m > 5.5:
                continue
            values.append(round(measurement.value_m, 3))
    if not values:
        return None

    counts = Counter(values)
    best_value, best_count = max(counts.items(), key=lambda item: (item[1], -abs(item[0] - 4.0)))
    if best_count >= 2:
        return best_value
    return float(np.median(values))


def _line_text(page: DocumentPage, block_no: int, line_no: int) -> str:
    line_spans = [span for span in page.text_spans if span.block_no == block_no and span.line_no == line_no]
    line_spans.sort(key=lambda span: span.word_no)
    return " ".join(span.text for span in line_spans)


def _detect_facade_labels(page: DocumentPage) -> list[tuple[str, BBox]]:
    labels: list[tuple[str, BBox]] = []
    for span in page.text_spans:
        if uppercase_normalized(span.text) != "JULKISIVU":
            continue
        line_text = uppercase_normalized(_line_text(page, span.block_no, span.line_no))
        if "MATERIAALIT" in line_text:
            continue
        labels.append((line_text or f"FACADE {len(labels) + 1}", span.bbox))
    labels.sort(key=lambda item: (item[1].cy, item[1].cx))
    return labels


def _clip_box(box: BBox, page: DocumentPage) -> BBox:
    return BBox(
        x0=max(0.0, box.x0),
        y0=max(0.0, box.y0),
        x1=min(page.width_pt, box.x1),
        y1=min(page.height_pt, box.y1),
    )


def _facade_cluster_box(page: DocumentPage, label_box: BBox) -> BBox:
    candidate_boxes = []
    for primitive in page.vector_primitives:
        box = primitive.bbox
        if box.area < 80.0:
            continue
        if box.width > page.width_pt * 0.85 or box.height > page.height_pt * 0.85:
            continue
        if abs(box.cx - label_box.cx) > 420.0:
            continue
        if box.cy >= label_box.cy:
            continue
        if label_box.cy - box.cy > 380.0:
            continue
        candidate_boxes.append(box)

    union_box = BBox.union(candidate_boxes)
    if union_box is not None:
        return _clip_box(union_box.expand(12.0), page)

    fallback = BBox(
        x0=label_box.cx - 240.0,
        y0=max(0.0, label_box.y0 - 320.0),
        x1=label_box.cx + 240.0,
        y1=max(label_box.y0 - 15.0, 0.0),
    )
    return _clip_box(fallback, page)


def _section_level_candidates(store: MeasurementStore, page_id: str) -> list[float]:
    values = []
    for measurement in measurements_for_page(store, page_id):
        if measurement.kind != "level_marker":
            continue
        if measurement.value_m <= 0.0:
            continue
        values.append(measurement.value_m)
    return sorted(values)


def reconstruct_geometry(
    pages: list[DocumentPage],
    classifications: list[ClassificationResult],
    measurements: MeasurementStore,
) -> tuple[FootprintModel, list[FacadeModel], list[str], list[str]]:
    assumptions: list[str] = []
    warnings: list[str] = []

    plan_page = _page_for_role(pages, classifications, "plan")
    if plan_page is None:
        raise RuntimeError("No plan sheet was classified. Geometry reconstruction requires a plan.")

    plan_region = detect_primary_plan_region(plan_page)
    width_m = _chain_dimension_sum(measurements, plan_page.page_id, plan_region, "horizontal")
    depth_m = _chain_dimension_sum(measurements, plan_page.page_id, plan_region, "vertical")
    if width_m is None:
        width_m = _largest_dimension_by_orientation(measurements, plan_page.page_id, plan_region, "horizontal")
    if depth_m is None:
        depth_m = _largest_dimension_by_orientation(measurements, plan_page.page_id, plan_region, "vertical")
    if width_m is None or depth_m is None:
        fallback_width, fallback_depth = _fallback_dimensions_from_any(measurements, plan_page.page_id)
        width_m = width_m or fallback_width
        depth_m = depth_m or fallback_depth
        assumptions.append("Fell back to the strongest overall plan dimensions because orientation filtering was incomplete.")

    if width_m is None or depth_m is None:
        raise RuntimeError("Could not detect overall building dimensions from the plan.")

    footprint_polygon = build_rectangular_footprint(width_m, depth_m)
    section_page = _page_for_role(pages, classifications, "section")
    elevation_page = _page_for_role(pages, classifications, "elevations")
    wall_height_m = _preferred_wall_height_m(
        measurements,
        [page_id for page_id in [section_page.page_id if section_page else None, elevation_page.page_id if elevation_page else None] if page_id],
    )
    if wall_height_m is None:
        wall_height_m = 4.0
        warnings.append("No repeated exterior wall height marker was found; using a 4.0 m fallback wall height.")
    else:
        assumptions.append(
            f"Exterior wall surface uses the repeated section/elevation wall-height marker {wall_height_m:.3f} m."
        )

    footprint = FootprintModel(
        width_m=width_m,
        depth_m=depth_m,
        polygon=[Point2D(x=float(x), y=float(y)) for x, y in footprint_polygon.exterior.coords[:-1]],
        perimeter_m=float(footprint_polygon.length),
        source_page_id=plan_page.page_id,
        source_strategy="plan_outer_dimension_chain_rectangle",
        confidence=0.8 if plan_region.width < plan_page.width_pt else 0.7,
        plan_bbox=plan_region,
    )

    facades: list[FacadeModel] = []

    if elevation_page is not None:
        labels = _detect_facade_labels(elevation_page)
        cluster_candidates = [(name, _facade_cluster_box(elevation_page, label_box)) for name, label_box in labels]
        if len(cluster_candidates) < 4:
            warnings.append("Elevation sheet yielded fewer than four facade clusters; geometry confidence reduced.")

        sorted_clusters = sorted(cluster_candidates, key=lambda item: item[1].width, reverse=True)
        long_dimension = max(width_m, depth_m)
        short_dimension = min(width_m, depth_m)
        long_cluster_names = {name for name, _ in sorted_clusters[:2]}
        assigned_meta: list[tuple[str, BBox, float]] = []
        for name, cluster_box in cluster_candidates:
            assigned_width_m = long_dimension if name in long_cluster_names else short_dimension
            assigned_meta.append((name, cluster_box, assigned_width_m))

        for name, cluster_box, assigned_width_m in assigned_meta:
            gross_area = assigned_width_m * wall_height_m

            facades.append(
                FacadeModel(
                    name=name,
                    width_m=assigned_width_m,
                    total_height_m=wall_height_m,
                    eave_height_m=wall_height_m,
                    ridge_height_m=None,
                    area_gross_m2=gross_area,
                    cluster_box=cluster_box,
                    source_page_id=elevation_page.page_id,
                    confidence=0.84,
                    is_gable=False,
                    assigned_dimension_kind="long" if assigned_width_m == long_dimension else "short",
                )
            )
    else:
        warnings.append("No elevation page was classified; using section-based rectangular facade fallback.")

    if not facades:
        facades = [
            FacadeModel(
                name="FACADE A",
                width_m=max(width_m, depth_m),
                total_height_m=wall_height_m,
                eave_height_m=wall_height_m,
                ridge_height_m=None,
                area_gross_m2=max(width_m, depth_m) * wall_height_m,
                cluster_box=BBox(x0=0, y0=0, x1=0, y1=0),
                source_page_id=section_page.page_id if section_page else plan_page.page_id,
                confidence=0.55,
                is_gable=False,
                assigned_dimension_kind="long",
            ),
            FacadeModel(
                name="FACADE B",
                width_m=max(width_m, depth_m),
                total_height_m=wall_height_m,
                eave_height_m=wall_height_m,
                ridge_height_m=None,
                area_gross_m2=max(width_m, depth_m) * wall_height_m,
                cluster_box=BBox(x0=0, y0=0, x1=0, y1=0),
                source_page_id=section_page.page_id if section_page else plan_page.page_id,
                confidence=0.55,
                is_gable=False,
                assigned_dimension_kind="long",
            ),
            FacadeModel(
                name="FACADE C",
                width_m=min(width_m, depth_m),
                total_height_m=wall_height_m,
                eave_height_m=wall_height_m,
                ridge_height_m=None,
                area_gross_m2=min(width_m, depth_m) * wall_height_m,
                cluster_box=BBox(x0=0, y0=0, x1=0, y1=0),
                source_page_id=section_page.page_id if section_page else plan_page.page_id,
                confidence=0.55,
                is_gable=False,
                assigned_dimension_kind="short",
            ),
            FacadeModel(
                name="FACADE D",
                width_m=min(width_m, depth_m),
                total_height_m=wall_height_m,
                eave_height_m=wall_height_m,
                ridge_height_m=None,
                area_gross_m2=min(width_m, depth_m) * wall_height_m,
                cluster_box=BBox(x0=0, y0=0, x1=0, y1=0),
                source_page_id=section_page.page_id if section_page else plan_page.page_id,
                confidence=0.55,
                is_gable=False,
                assigned_dimension_kind="short",
            ),
        ]
        assumptions.append("Facade widths use the outer dimension-chain rectangle and wall height uses repeated section/elevation markers.")

    return footprint, facades, assumptions, warnings
